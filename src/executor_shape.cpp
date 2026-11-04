#include "backend_internal.hpp"

#include <cstring>
#include <map>

// RESHAPE (C order) and BROADCAST (numpy.broadcast_to): each source chare sends
// result chares the elements they need. A result chare finishes once they cover
// its tile (BROADCAST expands the received source block into the tile).
// GPU builds pack/unpack on device; same-process messages pass a device buffer, others
// send their elements as device messages (CUDA IPC or pinned staging).

namespace {

/// 16-byte element (complex128); elements only move here.
struct ShapeElem16 {
    uint64_t w[2];
};

/// f(T{}) with T an element type of dt's size.
template <typename F>
void with_elem_type(DType dt, F&& f) {
    switch (dtype_size(dt)) {
    case 1: f(uint8_t{}); break;
    case 2: f(uint16_t{}); break;
    case 4: f(uint32_t{}); break;
    case 8: f(uint64_t{}); break;
    default: f(ShapeElem16{}); break;
    }
}

struct ShapeSpec {
    Opcode op = Opcode::RESHAPE;
    int src_name = -1;
    int src_nd = 1;
    Region* src_region = nullptr;
    int out_name = -1;
    int out_nd = 1;
    DType dt = DType::FLOAT64;
};

/// Header of a receive_shape message. RESHAPE: `nruns` run groups (RunGroups words) into
/// the result tile, then `count` elements. BROADCAST: one block [start, stop) in source view
/// coordinates, `count` elements row-major.
struct ShapeMsgHeader {
    int nd;
    int src_node; // sender's process (device buffers are only valid there)
    int start[3];
    int stop[3];
    int from[3];  // sending chare
    int dev;      // the elements come as a device message (SHAPE_DEV_INPUT) from `from`
    int64_t nruns;
    int64_t count;
    int64_t dbuf; // device buffer holding the elements (0: they follow inline)
};

/// Runs in groups: group g is cnt[g] runs of len[g] elements, run r at x[g] + r * xr[g] on
/// one side and y[g] + r * yr[g] on the other; consecutive runs are packed back to back.
struct RunGroups {
    std::vector<int64_t> x, xr, y, yr, len, cnt;
    int64_t total = 0;

    /// c runs of n elements at xo + r * xs and yo + r * ys; elements of a run are xstep / ystep
    /// apart. Merged into the last group when they continue it.
    void add(int64_t xo, int64_t xs, int64_t yo, int64_t ys, int64_t n, int64_t c, int64_t xstep,
             int64_t ystep) {
        total += n * c;
        if (!len.empty()) {
            size_t k = len.size() - 1;
            if (cnt[k] == 1 && c == 1 && x[k] + len[k] * xstep == xo &&
                y[k] + len[k] * ystep == yo) {
                len[k] += n;
                return;
            }
            if (len[k] == n) {
                int64_t dx = cnt[k] == 1 ? xo - x[k] : xr[k], dy = cnt[k] == 1 ? yo - y[k] : yr[k];
                if (xo == x[k] + cnt[k] * dx && yo == y[k] + cnt[k] * dy &&
                    (c == 1 || (xs == dx && ys == dy))) {
                    xr[k] = dx;
                    yr[k] = dy;
                    cnt[k] += c;
                    return;
                }
            }
        }
        x.push_back(xo);
        xr.push_back(c > 1 ? xs : 0);
        y.push_back(yo);
        yr.push_back(c > 1 ? ys : 0);
        len.push_back(n);
        cnt.push_back(c);
    }
    size_t size() const { return len.size(); }
    /// Message words of the x side: (x, xr, len, cnt) per group.
    std::vector<int64_t> words() const {
        std::vector<int64_t> w;
        w.reserve(4 * len.size());
        for (size_t k = 0; k < len.size(); ++k)
            w.insert(w.end(), {x[k], xr[k], len[k], cnt[k]});
        return w;
    }
};

/// Result decomposition, padded to 3 dims.
struct OutGeom {
    int nd = 1;
    int shape[3] = {1, 1, 1};
    int off[3] = {0, 0, 0};
    int tile = 1;

    int chare_start(int k, int c) const { return std::max(c * tile, off[k]); }
    int chare_stop(int k, int c) const { return std::min((c + 1) * tile, off[k] + shape[k]); }
};

ShapeSpec parse_shape(ASTNode* root) {
    ShapeSpec s;
    s.op = static_cast<Opcode>(root->opcode);
    ASTNode* src = root->operands[0];
    if (src->is_scalar || src->is_broadcast)
        CkAbort("reshape/broadcast_to: the source must be a regular array operand");
    s.src_name = src->result_name;
    s.src_nd = src->ndims;
    s.src_region = root->get_operand_region(0);
    s.out_name = root->result_name;
    s.out_nd = root->ndims;
    s.dt = root->dtype;
    return s;
}

OutGeom out_geom(ArrayDAGGroup* g, const ShapeSpec& s) {
    const auto& meta = g->array_meta.at(s.out_name);
    OutGeom o;
    o.nd = s.out_nd;
    o.tile = meta.tile;
    for (int k = 0; k < o.nd; ++k) {
        o.shape[k] = meta.global_shape[k];
        o.off[k] = meta.offset[k];
    }
    return o;
}

template <int M>
ArrayDecomp<M> out_decomp(const OutGeom& o) {
    std::array<int, M> gs{}, off{};
    for (int k = 0; k < M; ++k) {
        gs[k] = o.shape[k];
        off[k] = o.off[k];
    }
    return ArrayDecomp<M>::offset_decomp(gs, off, o.tile);
}

/// Source dim d of a BROADCAST is repeated along result dim d + delta.
bool bcast_dim(const int* vsize, const OutGeom& o, int delta, int d) {
    return vsize[d] == 1 && o.shape[d + delta] != 1;
}

/// Sends a receive_shape message to chare c of the out_nd-D partition.
void send_shape(ArrayDAGGroup* g, int out_nd, const std::array<int, 3>& c, int node_id,
                std::vector<char>& buf) {
    int64_t bytes = static_cast<int64_t>(buf.size());
    if (out_nd == 1)
        ct_send_bytes(proxy_at<1>(g->partition_proxy_1, ChareIndex<1>{{c[0]}}), MSG_SHAPE,
                      node_id, bytes, buf.data());
    else if (out_nd == 2)
        ct_send_bytes(proxy_at<2>(g->partition_proxy_2, ChareIndex<2>{{c[0], c[1]}}), MSG_SHAPE,
                      node_id, bytes, buf.data());
    else
        ct_send_bytes(proxy_at<3>(g->partition_proxy_3, ChareIndex<3>{{c[0], c[1], c[2]}}),
                      MSG_SHAPE, node_id, bytes, buf.data());
}

/// This chare's part of the source view: view indices [lo, hi) per dim, and
/// the local storage offset of view index v (offset(v) = base + sum v*vstride).
template <int N>
struct SrcBlock {
    int vsize[3] = {1, 1, 1};
    int lo[3] = {0, 0, 0};
    int hi[3] = {0, 0, 0};
    int64_t base = 0;
    int64_t vstride[3] = {0, 0, 0};
};

template <int N>
bool src_block(PartitionImpl<N>* partition, CTArrayBase<N>* arr, Region* view_base,
               SrcBlock<N>& b) {
    const auto& dec = arr->decomp;
    auto tile = dec.chare_region_global(partition->nd_index());
    std::array<int64_t, N> lstride{};
    lstride[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        lstride[d] = lstride[d + 1] * arr->region.size(d + 1);
    for (int d = 0; d < N; ++d) {
        int vstart = 0, vstep = 1, vsize = arr->global_shape[d];
        if (view_base != nullptr && !view_base->is_global) {
            auto* r = static_cast<ArrayRegion<N>*>(view_base);
            vstart = r->start[d];
            vstep = r->step[d];
            vsize = r->size(d);
        }
        int a = tile.start[d] - dec.offset[d] - vstart;
        int e = tile.stop[d] - dec.offset[d] - vstart;
        b.vsize[d] = vsize;
        b.lo[d] = a <= 0 ? 0 : (a + vstep - 1) / vstep;
        b.hi[d] = e <= 0 ? 0 : std::min(vsize, (e + vstep - 1) / vstep);
        if (b.lo[d] >= b.hi[d])
            return false;
        b.base += (int64_t)(vstart + dec.offset[d] - tile.start[d]) * lstride[d];
        b.vstride[d] = (int64_t)vstep * lstride[d];
    }
    return true;
}

/// Iterate the rows (all dims but the last) of the box [lo, hi).
template <typename Fn>
void for_each_row(int nd, const int* lo, const int* hi, Fn&& fn) {
    int v[3] = {lo[0], lo[1], lo[2]};
    while (true) {
        fn(v);
        int d = nd - 2;
        for (; d >= 0; --d) {
            if (++v[d] < hi[d])
                break;
            v[d] = lo[d];
        }
        if (d < 0)
            return;
    }
}

/// A message for result chare c: `head` (header and runs, then the elements when they travel
/// inline); in GPU builds the elements may instead be in device buffer `dbuf`.
struct OutMsg {
    std::array<int, 3> c;
    std::vector<char> head;
    char* dbuf = nullptr;
    int64_t bytes = 0;
};
using Outbox = std::vector<OutMsg>;

template <typename T>
struct Bucket {
    RunGroups runs; // x: result tile, y: source storage
    std::vector<T> data;
};

template <typename T>
std::vector<char> pack_runs(const Bucket<T>& b) {
    ShapeMsgHeader h = {};
    std::vector<int64_t> w = b.runs.words();
    h.nruns = static_cast<int64_t>(b.runs.size());
    h.count = static_cast<int64_t>(b.data.size());
    size_t run_bytes = w.size() * sizeof(int64_t);
    std::vector<char> buf(sizeof(h) + run_bytes + b.data.size() * sizeof(T));
    memcpy(buf.data(), &h, sizeof(h));
    memcpy(buf.data() + sizeof(h), w.data(), run_bytes);
    memcpy(buf.data() + sizeof(h) + run_bytes, b.data.data(), b.data.size() * sizeof(T));
    return buf;
}

/// RESHAPE: split source rows into runs inside one result row and tile:
/// fn(result chare, offset in its tile, tile step, length, source storage offset, source step,
/// count) stands for `count` runs, run r at the offsets plus r times the steps. Whole result
/// rows within one tile row come as one group per tile along the last result dim.
template <int N, typename Fn>
void for_each_reshape_run(const SrcBlock<N>& b, const OutGeom& o, Fn&& fn) {
    int64_t vfs[3] = {1, 1, 1};
    for (int d = N - 2; d >= 0; --d)
        vfs[d] = vfs[d + 1] * b.vsize[d + 1];
    int tn = o.nd;
    int last = tn - 1;
    int64_t W = o.shape[last];
    struct TileGeom {
        int cstart[3] = {0, 0, 0};
        int64_t lstride[3] = {0, 0, 0};
    };
    std::map<std::array<int, 3>, TileGeom> tiles;
    // result chare of result index t and t's offset in its tile
    auto place = [&](const int* t, std::array<int, 3>& c) -> std::pair<int64_t, const TileGeom*> {
        c = {0, 0, 0};
        for (int k = 0; k < tn; ++k)
            c[k] = (t[k] + o.off[k]) / o.tile;
        auto it = tiles.find(c);
        if (it == tiles.end()) {
            TileGeom g;
            int64_t st = 1;
            for (int k = tn - 1; k >= 0; --k) {
                g.cstart[k] = o.chare_start(k, c[k]);
                g.lstride[k] = st;
                st *= o.chare_stop(k, c[k]) - g.cstart[k];
            }
            it = tiles.emplace(c, g).first;
        }
        int64_t lo = 0;
        for (int k = 0; k < tn; ++k)
            lo += (int64_t)(t[k] + o.off[k] - it->second.cstart[k]) * it->second.lstride[k];
        return {lo, &it->second};
    };

    for_each_row(N, b.lo, b.hi, [&](const int* v) {
        int64_t f0 = 0, s0 = b.base;
        for (int d = 0; d < N; ++d) {
            f0 += v[d] * vfs[d];
            s0 += v[d] * b.vstride[d];
        }
        int64_t sstep = b.vstride[N - 1];
        int64_t len = b.hi[N - 1] - b.lo[N - 1];
        for (int64_t pos = 0; pos < len;) {
            int t[3] = {0, 0, 0};
            int64_t f = f0 + pos;
            for (int k = tn - 1; k >= 0; --k) {
                t[k] = static_cast<int>(f % o.shape[k]);
                f /= o.shape[k];
            }
            std::array<int, 3> c;
            if (tn > 1 && t[last] == 0) {
                int r = last - 1;
                int64_t rows = std::min<int64_t>((len - pos) / W, o.shape[r] - t[r]);
                rows = std::min<int64_t>(rows, o.tile - (t[r] + o.off[r]) % o.tile);
                if (rows > 1) {
                    for (int64_t a = 0; a < W;) {
                        t[last] = static_cast<int>(a);
                        int64_t n = std::min<int64_t>(W - a, o.tile - (a + o.off[last]) % o.tile);
                        auto [lo, tg] = place(t, c);
                        fn(c, lo, tg->lstride[r], n, s0 + (pos + a) * sstep, W * sstep, rows);
                        a += n;
                    }
                    pos += rows * W;
                    continue;
                }
            }
            int64_t n = std::min<int64_t>(len - pos, W - t[last]);
            n = std::min<int64_t>(n, o.tile - (t[last] + o.off[last]) % o.tile);
            auto [lo, tg] = place(t, c);
            (void)tg;
            fn(c, lo, int64_t(0), n, s0 + pos * sstep, int64_t(0), int64_t(1));
            pos += n;
        }
    });
}

template <int N, typename T>
void reshape_send(const T* src, const SrcBlock<N>& b, const OutGeom& o, Outbox& out) {
    std::map<std::array<int, 3>, Bucket<T>> buckets;
    int64_t sstep = b.vstride[N - 1];
    for_each_reshape_run<N>(b, o, [&](const std::array<int, 3>& c, int64_t lo, int64_t lstep,
                                      int64_t n, int64_t soff, int64_t sr, int64_t cnt) {
        Bucket<T>& bk = buckets[c];
        bk.runs.add(lo, lstep, soff, sr, n, cnt, 1, sstep);
        for (int64_t r = 0; r < cnt; ++r) {
            const T* p = src + soff + r * sr;
            if (sstep == 1) {
                bk.data.insert(bk.data.end(), p, p + n);
            } else {
                for (int64_t i = 0; i < n; ++i)
                    bk.data.push_back(p[i * sstep]);
            }
        }
    });
    for (auto& [c, bk] : buckets)
        out.push_back({c, pack_runs<T>(bk)});
}

/// BROADCAST: the part of this block each result tile reads (all of it along
/// repeated dims): fn(result chare, header with box [start, stop) and count).
template <int N, typename Fn>
void for_each_broadcast_box(const SrcBlock<N>& b, const OutGeom& o, Fn&& fn) {
    int tn = o.nd;
    int delta = tn - N;
    int cf[3] = {0, 0, 0}, cl[3] = {0, 0, 0};
    for (int k = 0; k < tn; ++k) {
        int d = k - delta;
        if (d < 0 || bcast_dim(b.vsize, o, delta, d)) {
            cf[k] = o.off[k] / o.tile;
            cl[k] = (o.off[k] + o.shape[k] - 1) / o.tile;
        } else {
            cf[k] = (b.lo[d] + o.off[k]) / o.tile;
            cl[k] = (b.hi[d] - 1 + o.off[k]) / o.tile;
        }
    }
    std::array<int, 3> c = {cf[0], cf[1], cf[2]};
    while (true) {
        ShapeMsgHeader h = {};
        h.nd = N;
        h.count = 1;
        bool empty = false;
        for (int d = 0; d < N; ++d) {
            int k = d + delta;
            int lo = b.lo[d], hi = b.hi[d];
            if (!bcast_dim(b.vsize, o, delta, d)) {
                lo = std::max(lo, o.chare_start(k, c[k]) - o.off[k]);
                hi = std::min(hi, o.chare_stop(k, c[k]) - o.off[k]);
            }
            h.start[d] = lo;
            h.stop[d] = hi;
            if (lo >= hi)
                empty = true;
            h.count *= hi - lo;
        }
        if (!empty)
            fn(c, h);
        int k = tn - 1;
        for (; k >= 0; --k) {
            if (++c[k] <= cl[k])
                break;
            c[k] = cf[k];
        }
        if (k < 0)
            break;
    }
}

template <int N, typename T>
void broadcast_send(const T* src, const SrcBlock<N>& b, const OutGeom& o, Outbox& out) {
    for_each_broadcast_box<N>(b, o, [&](const std::array<int, 3>& c, const ShapeMsgHeader& h) {
        std::vector<char> buf(sizeof(h) + h.count * sizeof(T));
        memcpy(buf.data(), &h, sizeof(h));
        T* dst = reinterpret_cast<T*>(buf.data() + sizeof(h));
        int64_t row = h.stop[N - 1] - h.start[N - 1];
        for_each_row(N, h.start, h.stop, [&](const int* v) {
            int64_t s = b.base;
            for (int d = 0; d < N; ++d)
                s += v[d] * b.vstride[d];
            int64_t step = b.vstride[N - 1];
            for (int64_t i = 0; i < row; ++i)
                *dst++ = src[s + i * step];
        });
        out.push_back({c, std::move(buf)});
    });
}

#ifdef USE_KOKKOS
/// Element q of the packed buffer, in group g = the one with pre[g] <= q < pre[g + 1], run r,
/// position j, is at x[g] + r * xr[g] + j * xstep on the strided side.
template <typename T>
struct GroupCopyFn {
    T* packed;
    T* strided;
    int64_t xstep;
    const int64_t* tab; // x | xr | len | pre (ng + 1)
    int64_t ng;
    bool to_packed;

    KOKKOS_INLINE_FUNCTION void operator()(const int64_t q) const {
        const int64_t* pre = tab + 3 * ng;
        int64_t lo = 0, hi = ng;
        while (hi - lo > 1) {
            int64_t m = (lo + hi) / 2;
            if (pre[m] <= q)
                lo = m;
            else
                hi = m;
        }
        int64_t k = q - pre[lo], n = tab[2 * ng + lo];
        int64_t i = tab[lo] + (k / n) * tab[ng + lo] + (k % n) * xstep;
        if (to_packed)
            packed[q] = strided[i];
        else
            strided[i] = packed[q];
    }
};

/// Packs the x side of `r` (to_packed) or unpacks into it, on the compute stream.
template <int N, typename T>
void copy_groups(PartitionImpl<N>* p, T* packed, T* strided, int64_t xstep, const RunGroups& r,
                 bool to_packed) {
    int64_t ng = static_cast<int64_t>(r.size());
    if (ng == 0)
        return;
    std::vector<int64_t> h(4 * ng + 1);
    for (int64_t k = 0; k < ng; ++k) {
        h[k] = r.x[k];
        h[ng + k] = r.xr[k];
        h[2 * ng + k] = r.len[k];
        h[3 * ng + k + 1] = h[3 * ng + k] + r.len[k] * r.cnt[k];
    }
    int64_t* tab = upload_words(p, h);
    Kokkos::parallel_for("ct_shape_groups", CT_COMPUTE_POLICY(p, r.total),
                         GroupCopyFn<T>{packed, strided, xstep, tab, ng, to_packed});
    free_after_compute(p, tab);
}

/// Box copy over sz[0..nd): out[obase + sum c_k os_k] = in[ibase + sum c_k is_k].
template <int N, typename T>
void copy_box(PartitionImpl<N>* p, T* out, int64_t obase, const int64_t* ostr, const T* in,
              int64_t ibase, const int64_t* istr, const int* size, int nd) {
    int64_t os[3] = {0, 0, 0}, is[3] = {0, 0, 0};
    int sz[3] = {1, 1, 1};
    int64_t total = 1;
    for (int k = 0; k < nd; ++k) {
        os[k] = ostr[k];
        is[k] = istr[k];
        sz[k] = size[k];
        total *= size[k];
    }
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, total), KOKKOS_LAMBDA(int64_t f) {
            int64_t rem = f, o = obase, i = ibase;
            for (int k = nd - 1; k >= 0; --k) {
                int64_t c = rem % sz[k];
                rem /= sz[k];
                o += c * os[k];
                i += c * is[k];
            }
            out[o] = in[i];
        });
}

/// Process (node) currently holding result chare c.
int chare_node(ArrayDAGGroup* g, int nd, const std::array<int, 3>& c) {
    int pe;
    if (nd == 1)
        pe = g->partition_proxy_1.ckLocalBranch()->lastKnown(CkArrayIndex1D(c[0]));
    else if (nd == 2)
        pe = g->partition_proxy_2.ckLocalBranch()->lastKnown(CkArrayIndex2D(c[0], c[1]));
    else
        pe = g->partition_proxy_3.ckLocalBranch()->lastKnown(CkArrayIndex3D(c[0], c[1], c[2]));
    return CkNodeOf(pe);
}

/// Packs this chare's part of every result tile into a device buffer (compute stream); the
/// caller posts the messages once the stream is ordered for them (post_shape_msg).
template <int N, typename T>
void shape_send_device(PartitionImpl<N>* p, const T* src, const SrcBlock<N>& b,
                       const ShapeSpec& s, const OutGeom& o, Outbox& out) {
    int64_t sstep = b.vstride[N - 1];
    auto me = p->nd_index();
    auto add = [&](const std::array<int, 3>& c, ShapeMsgHeader h, const std::vector<int64_t>& w,
                   char* d) {
        h.src_node = CkMyNode();
        for (int k = 0; k < N; ++k)
            h.from[k] = me[k];
        h.dbuf = reinterpret_cast<int64_t>(d);
        OutMsg m{c, std::vector<char>(sizeof(h) + w.size() * sizeof(int64_t)), d,
                 h.count * (int64_t)sizeof(T)};
        memcpy(m.head.data(), &h, sizeof(h));
        memcpy(m.head.data() + sizeof(h), w.data(), w.size() * sizeof(int64_t));
        out.push_back(std::move(m));
    };
    if (s.op == Opcode::RESHAPE) {
        std::map<std::array<int, 3>, RunGroups> plans; // x: result tile, y: source storage
        for_each_reshape_run<N>(b, o, [&](const std::array<int, 3>& c, int64_t lo, int64_t lstep,
                                          int64_t n, int64_t soff, int64_t sr, int64_t cnt) {
            plans[c].add(lo, lstep, soff, sr, n, cnt, 1, sstep);
        });
        for (auto& [c, r] : plans) {
            char* d = static_cast<char*>(comm_alloc(std::max<int64_t>(r.total * sizeof(T), 1)));
            RunGroups srcside = r;
            std::swap(srcside.x, srcside.y);
            std::swap(srcside.xr, srcside.yr);
            copy_groups<N, T>(p, reinterpret_cast<T*>(d), const_cast<T*>(src), sstep, srcside, true);
            ShapeMsgHeader h = {};
            h.nruns = static_cast<int64_t>(r.size());
            h.count = r.total;
            add(c, h, r.words(), d);
        }
        return;
    }
    for_each_broadcast_box<N>(b, o, [&](const std::array<int, 3>& c, const ShapeMsgHeader& h) {
        char* d = static_cast<char*>(comm_alloc(std::max<int64_t>(h.count * sizeof(T), 1)));
        int size[3];
        int64_t os[3], ibase = b.base;
        for (int d0 = 0; d0 < N; ++d0) {
            size[d0] = h.stop[d0] - h.start[d0];
            ibase += h.start[d0] * b.vstride[d0];
        }
        os[N - 1] = 1;
        for (int d0 = N - 2; d0 >= 0; --d0)
            os[d0] = os[d0 + 1] * size[d0 + 1];
        copy_box<N, T>(p, reinterpret_cast<T*>(d), 0, os, src, ibase, b.vstride, size, N);
        add(c, h, {}, d);
    });
}

/// Sends message m of the sender's packs to its result chare, whose dims are out_nd: by
/// pointer within this process once the packs have run, else the header now and the elements
/// as a device message (CUDA IPC or pinned staging). The comm stream is ordered after the packs.
template <int N>
void post_shape_msg(PartitionImpl<N>* p, ArrayDAGGroup* g, int out_nd, int node_id, OutMsg& m) {
    std::array<int, 3> c = m.c;
    if (chare_node(g, out_nd, c) == CkMyNode()) {
        std::vector<char> buf = std::move(m.head);
#ifdef USE_NVIDIA
        ct_after_stream(p->compute_stream_raw, [g, out_nd, c, node_id, buf]() mutable {
            send_shape(g, out_nd, c, node_id, buf);
        });
#else
        CT_KOKKOS_FENCE();
        send_shape(g, out_nd, c, node_id, buf);
#endif
        return;
    }
    ShapeMsgHeader h;
    memcpy(&h, m.head.data(), sizeof(h));
    h.dbuf = 0;
#ifdef USE_NVIDIA
    h.dev = 1;
    memcpy(m.head.data(), &h, sizeof(h));
    send_shape(g, out_nd, c, node_id, m.head);
    int region_data[9] = {h.from[0], h.from[1], h.from[2], 0, 0, 0, 0, 0, 0};
    switch (out_nd) {
    case 1:
        device_pack_send<N, 1>(p, g->partition_proxy_1, ChareIndex<1>{{c[0]}}, node_id,
                               SHAPE_DEV_INPUT, 0, region_data, m.bytes, m.dbuf);
        break;
    case 2:
        device_pack_send<N, 2>(p, g->partition_proxy_2, ChareIndex<2>{{c[0], c[1]}}, node_id,
                               SHAPE_DEV_INPUT, 0, region_data, m.bytes, m.dbuf);
        break;
    default:
        device_pack_send<N, 3>(p, g->partition_proxy_3, ChareIndex<3>{{c[0], c[1], c[2]}},
                               node_id, SHAPE_DEV_INPUT, 0, region_data, m.bytes, m.dbuf);
        break;
    }
#else
    size_t head = m.head.size();
    m.head.resize(head + m.bytes);
    memcpy(m.head.data(), &h, sizeof(h));
    Kokkos::fence("ct_shape_stage");
    Kokkos::deep_copy(Kokkos::View<char*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                          m.head.data() + head, m.bytes),
                      Kokkos::View<char*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                          m.dbuf, m.bytes));
    comm_free(m.dbuf);
    send_shape(g, out_nd, c, node_id, m.head);
#endif
}
#endif

template <int N, typename T>
void shape_send_typed(PartitionImpl<N>* partition, ArrayDAGGroup* g, CTArrayBase<N>* arr,
                      const ShapeSpec& s, const OutGeom& o, Outbox& out) {
    SrcBlock<N> b;
    if (!src_block<N>(partition, arr, s.src_region, b))
        return;
#ifdef USE_KOKKOS
    (void)g;
    shape_send_device<N, T>(partition, static_cast<const T*>(arr->device_data_ptr()), b, s, o,
                            out);
#else
    (void)g;
    const T* src = static_cast<const T*>(arr->data_ptr());
    if (s.op == Opcode::RESHAPE)
        reshape_send<N, T>(src, b, o, out);
    else
        broadcast_send<N, T>(src, b, o, out);
#endif
}

/// Source block a result chare's tile `mine` reads (BROADCAST).
template <int M>
void bcast_block_box(const ShapeSpec& s, const OutGeom& o, const ArrayRegion<M>& mine,
                     const int* vsize, int* pstart, int* psize) {
    int delta = s.out_nd - s.src_nd;
    for (int d = 0; d < s.src_nd; ++d) {
        int k = d + delta;
        if (bcast_dim(vsize, o, delta, d)) {
            pstart[d] = 0;
            psize[d] = 1;
        } else {
            pstart[d] = mine.start[k] - o.off[k];
            psize[d] = mine.stop[k] - mine.start[k];
        }
    }
}

/// Source view extents (from the operand region or the source metadata).
void src_view_size(ArrayDAGGroup* g, const ShapeSpec& s, int* vsize) {
    const auto& meta = g->array_meta.at(s.src_name);
    for (int d = 0; d < 3; ++d)
        vsize[d] = d < s.src_nd ? meta.global_shape[d] : 1;
    Region* r = s.src_region;
    if (r == nullptr || r->is_global)
        return;
    for (int d = 0; d < s.src_nd; ++d) {
        switch (s.src_nd) {
        case 1:
            vsize[d] = static_cast<ArrayRegion<1>*>(r)->size(d);
            break;
        case 2:
            vsize[d] = static_cast<ArrayRegion<2>*>(r)->size(d);
            break;
        default:
            vsize[d] = static_cast<ArrayRegion<3>*>(r)->size(d);
            break;
        }
    }
}

#ifdef USE_KOKKOS
template <int M, typename T>
void expand_device(PartitionImpl<M>* partition, T* out, const T* blk, const int* osize,
                   const int64_t* pstride) {
    int sz[3] = {1, 1, 1};
    int64_t ps[3] = {0, 0, 0};
    int64_t total = 1;
    for (int k = 0; k < M; ++k) {
        sz[k] = osize[k];
        ps[k] = pstride[k];
        total *= osize[k];
    }
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, total), KOKKOS_LAMBDA(int64_t flat) {
            int64_t rem = flat, p = 0;
            for (int k = M - 1; k >= 0; --k) {
                p += (rem % sz[k]) * ps[k];
                rem /= sz[k];
            }
            out[flat] = blk[p];
        });
}

/// Apply one received message to the result tile (RESHAPE) or the device block (BROADCAST);
/// `dev`: its elements when they came as a device message. Returns the device buffer to free
/// once the copies have run.
template <int M, typename T>
char* apply_msg_device(PartitionImpl<M>* partition, ShapeState<M>& st, const ShapeSpec& s,
                       CTArrayBase<M>* out, const std::vector<char>& msg, char* dev,
                       const int* pstart, const int* psize) {
    ShapeMsgHeader h;
    memcpy(&h, msg.data(), sizeof(h));
    const auto* runs = reinterpret_cast<const int64_t*>(msg.data() + sizeof(h));
    size_t head = sizeof(h) + h.nruns * 4 * sizeof(int64_t);
    char* data = dev;
    if (data == nullptr && h.dbuf != 0) {
        if (h.src_node != CkMyNode())
            CkAbort("reshape/broadcast_to: device buffer from another process");
        data = reinterpret_cast<char*>(h.dbuf);
    } else if (data == nullptr) {
        data = static_cast<char*>(comm_alloc(std::max<int64_t>(h.count * sizeof(T), 1)));
        Kokkos::deep_copy(
            Kokkos::View<char*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                data, h.count * sizeof(T)),
            Kokkos::View<const char*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                msg.data() + head, h.count * sizeof(T)));
    }
    if (s.op == Opcode::RESHAPE) {
        RunGroups r;
        for (int64_t k = 0; k < h.nruns; ++k)
            r.add(runs[4 * k], runs[4 * k + 1], 0, 0, runs[4 * k + 2], runs[4 * k + 3], 1, 1);
        copy_groups<M, T>(partition, reinterpret_cast<T*>(data),
                          static_cast<T*>(out->device_data_ptr()), 1, r, false);
    } else {
        int nd = s.src_nd;
        int size[3];
        int64_t bst[3] = {1, 1, 1}, is[3] = {1, 1, 1}, obase = 0;
        for (int d = nd - 2; d >= 0; --d)
            bst[d] = bst[d + 1] * psize[d + 1];
        for (int d = 0; d < nd; ++d) {
            size[d] = h.stop[d] - h.start[d];
            obase += (h.start[d] - pstart[d]) * bst[d];
        }
        for (int d = nd - 2; d >= 0; --d)
            is[d] = is[d + 1] * size[d + 1];
        copy_box<M, T>(partition, reinterpret_cast<T*>(st.d_block), obase, bst,
                       reinterpret_cast<T*>(data), 0, is, size, nd);
    }
    return data;
}
#endif

/// BROADCAST: out[i] = block[sum_k i_k * pstride[k]] over the tile.
template <int M, typename T>
void expand_block(PartitionImpl<M>* partition, CTArrayBase<M>* out_arr, const ShapeState<M>& st,
                  const int* osize, const int64_t* pstride) {
#ifdef USE_KOKKOS
    expand_device<M, T>(partition, static_cast<T*>(out_arr->device_data_ptr()),
                        reinterpret_cast<const T*>(st.d_block), osize, pstride);
#else
    (void)partition;
    const T* block = reinterpret_cast<const T*>(st.block.data());
    T* out = static_cast<T*>(out_arr->data_ptr());
    int lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
    for (int k = 0; k < M; ++k)
        hi[k] = osize[k];
    int64_t row = osize[M - 1];
    for_each_row(M, lo, hi, [&](const int* i) {
        int64_t p = 0, o = 0, ostride = 1;
        for (int k = M - 1; k >= 0; --k) {
            p += i[k] * pstride[k];
            o += i[k] * ostride;
            ostride *= osize[k];
        }
        if (pstride[M - 1] == 1)
            memcpy(out + o, block + p, row * sizeof(T));
        else
            std::fill(out + o, out + o + row, block[p]);
    });
#endif
}

template <int M, typename T>
void finish_typed(PartitionImpl<M>* partition, ShapeState<M>& st, const ShapeSpec& s,
                  const OutGeom& o, CTArrayBase<M>* out, const ArrayRegion<M>& mine,
                  const int* vsize) {
    if (s.op == Opcode::RESHAPE)
        return;
    int pstart[3], psize[3];
    bcast_block_box<M>(s, o, mine, vsize, pstart, psize);
    int delta = s.out_nd - s.src_nd;
    int64_t bst[3] = {1, 1, 1};
    for (int d = s.src_nd - 2; d >= 0; --d)
        bst[d] = bst[d + 1] * psize[d + 1];
    int osize[3] = {1, 1, 1};
    int64_t pstride[3] = {0, 0, 0};
    for (int k = 0; k < M; ++k) {
        osize[k] = mine.stop[k] - mine.start[k];
        int d = k - delta;
        pstride[k] = (d < 0 || bcast_dim(vsize, o, delta, d)) ? 0 : bst[d];
    }
    expand_block<M, T>(partition, out, st, osize, pstride);
}

} // namespace

/// Apply stashed messages; finish once the expected elements have arrived.
template <int M>
static bool shape_try_finish(PartitionImpl<M>* partition, ArrayDAGGroup* g, ShapeState<M>& st,
                             const ShapeSpec& s) {
    OutGeom o = out_geom(g, s);
    auto mine = out_decomp<M>(o).chare_region_global(partition->nd_index());
    CTArrayBase<M>* out = partition->arrays.at(s.out_name);
    int vsize[3];
    src_view_size(g, s, vsize);
    int pstart[3] = {0, 0, 0}, psize[3] = {1, 1, 1};
    if (s.op == Opcode::BROADCAST)
        bcast_block_box<M>(s, o, mine, vsize, pstart, psize);

#ifdef USE_KOKKOS
    std::vector<std::vector<char>> waiting;
    for (auto& msg : st.stash) {
        ShapeMsgHeader h;
        memcpy(&h, msg.data(), sizeof(h));
        char* dev = nullptr;
        if (h.dev) {
            auto it = st.payloads.find({h.from[0], h.from[1], h.from[2]});
            if (it == st.payloads.end()) {
                waiting.push_back(std::move(msg));
                continue;
            }
            dev = it->second;
            st.payloads.erase(it);
        }
        with_elem_type(s.dt, [&](auto tag) {
            free_after_compute(partition, apply_msg_device<M, decltype(tag)>(
                                              partition, st, s, out, msg, dev, pstart, psize));
        });
        st.covered += h.count;
    }
    st.stash.swap(waiting);
#else
    int es = dtype_size(s.dt);
    for (auto& msg : st.stash) {
        ShapeMsgHeader h;
        memcpy(&h, msg.data(), sizeof(h));
        if (s.op == Opcode::RESHAPE) {
            const auto* runs = reinterpret_cast<const int64_t*>(msg.data() + sizeof(h));
            const char* data = msg.data() + sizeof(h) + h.nruns * 4 * sizeof(int64_t);
            char* dst = static_cast<char*>(out->data_ptr());
            for (int64_t g = 0; g < h.nruns; ++g) {
                const int64_t* w = runs + 4 * g;
                for (int64_t r = 0; r < w[3]; ++r) {
                    memcpy(dst + (w[0] + r * w[1]) * es, data, w[2] * es);
                    data += w[2] * es;
                }
            }
        } else {
            int nd = s.src_nd;
            int64_t bst[3] = {1, 1, 1};
            for (int d = nd - 2; d >= 0; --d)
                bst[d] = bst[d + 1] * psize[d + 1];
            const char* data = msg.data() + sizeof(h);
            int64_t row = (h.stop[nd - 1] - h.start[nd - 1]) * (int64_t)es;
            for_each_row(nd, h.start, h.stop, [&](const int* v) {
                int64_t p = 0;
                for (int d = 0; d < nd; ++d)
                    p += (v[d] - pstart[d]) * bst[d];
                memcpy(st.block.data() + p * es, data, row);
                data += row;
            });
        }
        st.covered += h.count;
    }
    st.stash.clear();
#endif
    if (st.covered < st.expected)
        return false;

    with_elem_type(s.dt, [&](auto tag) {
        finish_typed<M, decltype(tag)>(partition, st, s, o, out, mine, vsize);
    });
#ifdef USE_KOKKOS
    if (st.d_block) {
        free_after_compute(partition, st.d_block);
        st.d_block = nullptr;
    }
#endif
    return true;
}

static ShapeSpec shape_spec_of(DAGNode* node) {
    for (ASTNode* root : node->ast->roots)
        if (is_shape_op(static_cast<Opcode>(root->opcode)))
            return parse_shape(root);
    CkAbort("execute_shape_node: node %d has no reshape/broadcast root", node->id);
    return ShapeSpec{};
}

template <int N>
void ArrayDAGExecutorND<N>::execute_shape_node(DAGNode* node) {
    auto* g = static_cast<ArrayDAGGroup*>(group);
    ShapeSpec s = shape_spec_of(node);
    OutGeom o = out_geom(g, s);
    auto nd_idx = partition->nd_index();

    // Send side first: once this node finishes, later nodes may overwrite the source.
    if (N == s.src_nd) {
        auto it = partition->arrays.find(s.src_name);
        if (it != partition->arrays.end() && it->second->local_size() > 0) {
            if (it->second->dtype != s.dt)
                CkAbort("reshape/broadcast_to: source dtype differs from the result dtype");
            Outbox outbox;
            with_elem_type(s.dt, [&](auto tag) {
                shape_send_typed<N, decltype(tag)>(partition, g, it->second, s, o, outbox);
            });
#ifdef USE_KOKKOS
            comm_after_compute(partition);
#endif
            for (auto& m : outbox) {
                bool self = s.out_nd == N;
                for (int k = 0; k < N && self; ++k)
                    self = m.c[k] == nd_idx[k];
                if (self) {
                    shape_state[node->id].stash.push_back(std::move(m.head));
                    continue;
                }
#ifdef USE_KOKKOS
                post_shape_msg<N>(partition, g, s.out_nd, node->id, m);
#else
                send_shape(g, s.out_nd, m.c, node->id, m.head);
#endif
            }
        }
    }

    if (N == s.out_nd) {
        auto decomp = out_decomp<N>(o);
        auto mine = decomp.chare_region_global(nd_idx);
        bool receiver = true;
        for (int k = 0; k < N; ++k)
            if (mine.stop[k] <= mine.start[k])
                receiver = false;
        if (receiver) {
            std::array<int, N> zeros{}, sizes{}, ones{};
            for (int k = 0; k < N; ++k) {
                sizes[k] = mine.stop[k] - mine.start[k];
                ones[k] = 1;
            }
            ArrayRegion<N> local(zeros, sizes, ones);
            partition->ensure_array(local, decomp.global_shape, s.out_name, s.dt, decomp);

            ShapeState<N>& st = shape_state[node->id];
            st.node = node;
            st.started = true;
            if (s.op == Opcode::RESHAPE) {
                st.expected = mine.size();
            } else {
                int vsize[3], pstart[3], psize[3];
                src_view_size(g, s, vsize);
                bcast_block_box<N>(s, o, mine, vsize, pstart, psize);
                st.expected = 1;
                for (int d = 0; d < s.src_nd; ++d)
                    st.expected *= psize[d];
#ifdef USE_KOKKOS
                st.d_block = static_cast<char*>(comm_alloc(st.expected * dtype_size(s.dt)));
#else
                st.block.assign(st.expected * dtype_size(s.dt), 0);
#endif
            }
            if (shape_try_finish<N>(partition, g, st, s)) {
                shape_state.erase(node->id);
                node_finished(node->id);
            }
            return;
        }
    }
    shape_state.erase(node->id);
    node_finished(node->id);
}

template <int N>
void ArrayDAGExecutorND<N>::on_shape_msg(int node_id, char* data, int64_t size) {
    ShapeState<N>& st = shape_state[node_id];
    st.stash.emplace_back(data, data + size);
    if (!st.started)
        return; // this chare has not reached the node yet
    auto* g = static_cast<ArrayDAGGroup*>(group);
    ShapeSpec s = shape_spec_of(st.node);
    if (shape_try_finish<N>(partition, g, st, s)) {
        shape_state.erase(node_id);
        node_finished(node_id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::on_shape_payload(int node_id, const int* from, char* buf) {
    ShapeState<N>& st = shape_state[node_id];
    st.payloads[{from[0], from[1], from[2]}] = buf;
    if (!st.started)
        return;
    auto* g = static_cast<ArrayDAGGroup*>(group);
    ShapeSpec s = shape_spec_of(st.node);
    if (shape_try_finish<N>(partition, g, st, s)) {
        shape_state.erase(node_id);
        node_finished(node_id);
    }
}

template <int N>
void PartitionImpl<N>::receive_shape(int node_id, int64_t size, char* data) {
    executor->on_shape_msg(node_id, data, size);
}

template void ArrayDAGExecutorND<1>::execute_shape_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_shape_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_shape_node(DAGNode*);

template void ArrayDAGExecutorND<1>::on_shape_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<2>::on_shape_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<3>::on_shape_msg(int, char*, int64_t);

template void ArrayDAGExecutorND<1>::on_shape_payload(int, const int*, char*);
template void ArrayDAGExecutorND<2>::on_shape_payload(int, const int*, char*);
template void ArrayDAGExecutorND<3>::on_shape_payload(int, const int*, char*);

template void PartitionImpl<1>::receive_shape(int, int64_t, char*);
template void PartitionImpl<2>::receive_shape(int, int64_t, char*);
template void PartitionImpl<3>::receive_shape(int, int64_t, char*);
