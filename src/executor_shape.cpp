#include "backend_internal.hpp"

#include <cstring>
#include <map>

// RESHAPE (C order) and BROADCAST (numpy.broadcast_to): each source chare sends
// result chares the elements they need. A result chare finishes once they cover
// its tile (BROADCAST expands the received source block into the tile).
// GPU builds pack/unpack on device; same-process messages pass a device buffer.

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

/// Header of a receive_shape message. RESHAPE: `nruns` (local offset, length)
/// int64 pairs into the result tile, then `count` elements. BROADCAST: one
/// block [start, stop) in source view coordinates, `count` elements row-major.
struct ShapeMsgHeader {
    int nd;
    int src_node; // sender's process (device buffers are only valid there)
    int start[3];
    int stop[3];
    int64_t nruns;
    int64_t count;
    int64_t dbuf; // device buffer holding the elements (0: they follow inline)
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

template <int M>
void send_to(ArrayDAGGroup* g, const std::array<int, 3>& c, int node_id, std::vector<char>& buf) {
    ChareIndex<M> ci;
    for (int k = 0; k < M; ++k)
        ci.idx[k] = c[k];
    int64_t bytes = static_cast<int64_t>(buf.size());
    if constexpr (M == 1)
        proxy_at<1>(g->partition_proxy_1, ci).receive_shape(node_id, bytes, buf.data());
    else if constexpr (M == 2)
        proxy_at<2>(g->partition_proxy_2, ci).receive_shape(node_id, bytes, buf.data());
    else
        proxy_at<3>(g->partition_proxy_3, ci).receive_shape(node_id, bytes, buf.data());
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

template <typename T>
struct Bucket {
    std::vector<int64_t> runs;
    std::vector<T> data;
};

template <typename T>
std::vector<char> pack_runs(const Bucket<T>& b) {
    ShapeMsgHeader h = {};
    h.nruns = static_cast<int64_t>(b.runs.size() / 2);
    h.count = static_cast<int64_t>(b.data.size());
    size_t run_bytes = b.runs.size() * sizeof(int64_t);
    std::vector<char> buf(sizeof(h) + run_bytes + b.data.size() * sizeof(T));
    memcpy(buf.data(), &h, sizeof(h));
    memcpy(buf.data() + sizeof(h), b.runs.data(), run_bytes);
    memcpy(buf.data() + sizeof(h) + run_bytes, b.data.data(), b.data.size() * sizeof(T));
    return buf;
}

using Outbox = std::vector<std::pair<std::array<int, 3>, std::vector<char>>>;

/// RESHAPE: split source rows into runs inside one result row and tile:
/// fn(result chare, offset in its tile, length, source storage offset).
template <int N, typename Fn>
void for_each_reshape_run(const SrcBlock<N>& b, const OutGeom& o, Fn&& fn) {
    int64_t vfs[3] = {1, 1, 1};
    for (int d = N - 2; d >= 0; --d)
        vfs[d] = vfs[d + 1] * b.vsize[d + 1];
    int tn = o.nd;
    struct TileGeom {
        int cstart[3] = {0, 0, 0};
        int64_t lstride[3] = {0, 0, 0};
    };
    std::map<std::array<int, 3>, TileGeom> tiles;

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
            int last = tn - 1;
            int64_t n = std::min<int64_t>(len - pos, o.shape[last] - t[last]);
            n = std::min<int64_t>(n, o.tile - (t[last] + o.off[last]) % o.tile);
            std::array<int, 3> c = {0, 0, 0};
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
            fn(c, lo, n, s0 + pos * sstep);
            pos += n;
        }
    });
}

template <int N, typename T>
void reshape_send(const T* src, const SrcBlock<N>& b, const OutGeom& o, Outbox& out) {
    std::map<std::array<int, 3>, Bucket<T>> buckets;
    int64_t sstep = b.vstride[N - 1];
    for_each_reshape_run<N>(b, o, [&](const std::array<int, 3>& c, int64_t lo, int64_t n,
                                      int64_t soff) {
        Bucket<T>& bk = buckets[c];
        size_t nr = bk.runs.size();
        if (nr > 0 && bk.runs[nr - 2] + bk.runs[nr - 1] == lo) {
            bk.runs[nr - 1] += n;
        } else {
            bk.runs.push_back(lo);
            bk.runs.push_back(n);
        }
        const T* p = src + soff;
        if (sstep == 1) {
            bk.data.insert(bk.data.end(), p, p + n);
        } else {
            for (int64_t i = 0; i < n; ++i)
                bk.data.push_back(p[i * sstep]);
        }
    });
    for (auto& [c, bk] : buckets)
        out.emplace_back(c, pack_runs<T>(bk));
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
        out.emplace_back(c, std::move(buf));
    });
}

#ifdef USE_KOKKOS
/// Strided runs: run r copies len[r] elements from in[ibase[r] + j * istep]
/// to out[obase[r] + j * ostep]; adjacent runs that continue each other merge.
struct Runs {
    std::vector<int64_t> obase, ibase, len;
    int64_t total = 0;
    void add(int64_t o, int64_t i, int64_t n, int64_t ostep, int64_t istep) {
        total += n;
        if (!len.empty() && obase.back() + len.back() * ostep == o &&
            ibase.back() + len.back() * istep == i) {
            len.back() += n;
            return;
        }
        obase.push_back(o);
        ibase.push_back(i);
        len.push_back(n);
    }
};

template <int N, typename T>
void copy_runs(PartitionImpl<N>* p, T* out, int64_t ostep, const T* in, int64_t istep,
               const Runs& r) {
    int64_t nr = static_cast<int64_t>(r.len.size());
    if (nr == 0)
        return;
    // obase | ibase | exclusive prefix of len (nr + 1)
    std::vector<int64_t> h(3 * nr + 1);
    for (int64_t k = 0; k < nr; ++k) {
        h[k] = r.obase[k];
        h[nr + k] = r.ibase[k];
        h[2 * nr + k + 1] = h[2 * nr + k] + r.len[k];
    }
    Kokkos::View<int64_t*, DeviceSpace> d("ct_shape_runs", h.size());
    Kokkos::deep_copy(d, Kokkos::View<const int64_t*, HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h.data(), h.size()));
    const int64_t* db = d.data();
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, r.total), KOKKOS_LAMBDA(int64_t i) {
            const int64_t* pre = db + 2 * nr;
            int64_t lo = 0, hi = nr;
            while (hi - lo > 1) {
                int64_t m = (lo + hi) / 2;
                if (pre[m] <= i)
                    lo = m;
                else
                    hi = m;
            }
            int64_t j = i - pre[lo];
            out[db[lo] + j * ostep] = in[db[nr + lo] + j * istep];
        });
    compute_fence(p); // before `d` is freed
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

/// Message for result chare c: header, `runs`, then the elements of the device
/// buffer dbuf, which travels as a pointer when c lives in this process.
template <int N, typename T>
void push_device_msg(PartitionImpl<N>* p, ArrayDAGGroup* g, const OutGeom& o,
                     const std::array<int, 3>& c, ShapeMsgHeader h,
                     const std::vector<int64_t>& runs, T* dbuf, Outbox& out) {
    h.nruns = static_cast<int64_t>(runs.size() / 2);
    h.src_node = CkMyNode();
    bool local = chare_node(g, o.nd, c) == CkMyNode();
    size_t head = sizeof(h) + runs.size() * sizeof(int64_t);
    std::vector<char> buf(head + (local ? 0 : h.count * sizeof(T)));
    if (local) {
        h.dbuf = reinterpret_cast<int64_t>(dbuf);
    } else {
        Kokkos::deep_copy(
            Kokkos::View<T*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                reinterpret_cast<T*>(buf.data() + head), h.count),
            Kokkos::View<T*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(dbuf, h.count));
        Kokkos::kokkos_free<DeviceSpace>(dbuf);
    }
    memcpy(buf.data(), &h, sizeof(h));
    memcpy(buf.data() + sizeof(h), runs.data(), runs.size() * sizeof(int64_t));
    out.emplace_back(c, std::move(buf));
}

template <int N, typename T>
void shape_send_device(PartitionImpl<N>* p, ArrayDAGGroup* g, const T* src, const SrcBlock<N>& b,
                       const ShapeSpec& s, const OutGeom& o, Outbox& out) {
    struct Plan {
        std::vector<int64_t> runs; // (offset in the result tile, length)
        Runs gather;
    };
    std::vector<std::pair<std::array<int, 3>, ShapeMsgHeader>> boxes;
    std::map<std::array<int, 3>, Plan> plans;
    int64_t sstep = b.vstride[N - 1];
    if (s.op == Opcode::RESHAPE) {
        for_each_reshape_run<N>(b, o, [&](const std::array<int, 3>& c, int64_t lo, int64_t n,
                                          int64_t soff) {
            Plan& pl = plans[c];
            size_t nr = pl.runs.size();
            if (nr > 0 && pl.runs[nr - 2] + pl.runs[nr - 1] == lo) {
                pl.runs[nr - 1] += n;
            } else {
                pl.runs.push_back(lo);
                pl.runs.push_back(n);
            }
            pl.gather.add(pl.gather.total, soff, n, 1, sstep);
        });
    } else {
        for_each_broadcast_box<N>(b, o, [&](const std::array<int, 3>& c, const ShapeMsgHeader& h) {
            boxes.emplace_back(c, h);
        });
    }
    std::vector<std::pair<std::array<int, 3>, T*>> bufs;
    for (auto& [c, pl] : plans) {
        T* d = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(pl.gather.total * sizeof(T)));
        copy_runs<N, T>(p, d, 1, src, sstep, pl.gather);
        bufs.emplace_back(c, d);
    }
    for (auto& [c, h] : boxes) {
        T* d = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(h.count * sizeof(T)));
        int size[3];
        int64_t os[3], ibase = b.base;
        for (int d0 = 0; d0 < N; ++d0) {
            size[d0] = h.stop[d0] - h.start[d0];
            ibase += h.start[d0] * b.vstride[d0];
        }
        os[N - 1] = 1;
        for (int d0 = N - 2; d0 >= 0; --d0)
            os[d0] = os[d0 + 1] * size[d0 + 1];
        copy_box<N, T>(p, d, 0, os, src, ibase, b.vstride, size, N);
        bufs.emplace_back(c, d);
    }
    compute_fence(p); // receivers read the buffers on their own streams
    size_t k = 0;
    for (auto& [c, pl] : plans) {
        ShapeMsgHeader h = {};
        h.count = pl.gather.total;
        push_device_msg<N, T>(p, g, o, c, h, pl.runs, bufs[k++].second, out);
    }
    for (auto& [c, h] : boxes)
        push_device_msg<N, T>(p, g, o, c, h, {}, bufs[k++].second, out);
}
#endif

template <int N, typename T>
void shape_send_typed(PartitionImpl<N>* partition, ArrayDAGGroup* g, CTArrayBase<N>* arr,
                      const ShapeSpec& s, const OutGeom& o, Outbox& out) {
    SrcBlock<N> b;
    if (!src_block<N>(partition, arr, s.src_region, b))
        return;
#ifdef USE_KOKKOS
    shape_send_device<N, T>(partition, g, static_cast<const T*>(arr->device_data_ptr()), b, s, o,
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
    compute_fence(partition);
}

/// Apply one received message to the result tile (RESHAPE) or the device
/// block (BROADCAST). Returns a device buffer to free once the copies are done.
template <int M, typename T>
T* apply_msg_device(PartitionImpl<M>* partition, ShapeState<M>& st, const ShapeSpec& s,
                    CTArrayBase<M>* out, const std::vector<char>& msg, const int* pstart,
                    const int* psize) {
    ShapeMsgHeader h;
    memcpy(&h, msg.data(), sizeof(h));
    const auto* runs = reinterpret_cast<const int64_t*>(msg.data() + sizeof(h));
    size_t head = sizeof(h) + h.nruns * 2 * sizeof(int64_t);
    T* data;
    if (h.dbuf != 0) {
        if (h.src_node != CkMyNode())
            CkAbort("reshape/broadcast_to: device buffer from another process");
        data = reinterpret_cast<T*>(h.dbuf);
    } else {
        data = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(h.count * sizeof(T)));
        Kokkos::deep_copy(
            Kokkos::View<T*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(data, h.count),
            Kokkos::View<const T*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                reinterpret_cast<const T*>(msg.data() + head), h.count));
    }
    if (s.op == Opcode::RESHAPE) {
        Runs r;
        int64_t pos = 0;
        for (int64_t k = 0; k < h.nruns; ++k) {
            r.add(runs[2 * k], pos, runs[2 * k + 1], 1, 1);
            pos += runs[2 * k + 1];
        }
        copy_runs<M, T>(partition, static_cast<T*>(out->device_data_ptr()), 1, data, 1, r);
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
        copy_box<M, T>(partition, reinterpret_cast<T*>(st.d_block), obase, bst, data, 0, is, size,
                       nd);
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
    std::vector<void*> to_free;
    for (auto& msg : st.stash) {
        ShapeMsgHeader h;
        memcpy(&h, msg.data(), sizeof(h));
        with_elem_type(s.dt, [&](auto tag) {
            to_free.push_back(apply_msg_device<M, decltype(tag)>(partition, st, s, out, msg,
                                                                 pstart, psize));
        });
        st.covered += h.count;
    }
    if (!to_free.empty()) {
        compute_fence(partition);
        for (void* p : to_free)
            Kokkos::kokkos_free<DeviceSpace>(p);
    }
#else
    int es = dtype_size(s.dt);
    for (auto& msg : st.stash) {
        ShapeMsgHeader h;
        memcpy(&h, msg.data(), sizeof(h));
        if (s.op == Opcode::RESHAPE) {
            const auto* runs = reinterpret_cast<const int64_t*>(msg.data() + sizeof(h));
            const char* data = msg.data() + sizeof(h) + h.nruns * 2 * sizeof(int64_t);
            char* dst = static_cast<char*>(out->data_ptr());
            for (int64_t r = 0; r < h.nruns; ++r) {
                memcpy(dst + runs[2 * r] * es, data, runs[2 * r + 1] * es);
                data += runs[2 * r + 1] * es;
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
#endif
    st.stash.clear();
    if (st.covered < st.expected)
        return false;

    with_elem_type(s.dt, [&](auto tag) {
        finish_typed<M, decltype(tag)>(partition, st, s, o, out, mine, vsize);
    });
#ifdef USE_KOKKOS
    if (st.d_block) {
        Kokkos::kokkos_free<DeviceSpace>(st.d_block);
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
            for (auto& [c, buf] : outbox) {
                bool self = s.out_nd == N;
                for (int k = 0; k < N && self; ++k)
                    self = c[k] == nd_idx[k];
                if (self) {
                    shape_state[node->id].stash.push_back(std::move(buf));
                    continue;
                }
                switch (s.out_nd) {
                case 1:
                    send_to<1>(g, c, node->id, buf);
                    break;
                case 2:
                    send_to<2>(g, c, node->id, buf);
                    break;
                default:
                    send_to<3>(g, c, node->id, buf);
                    break;
                }
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
                st.d_block = static_cast<char*>(
                    Kokkos::kokkos_malloc<DeviceSpace>(st.expected * dtype_size(s.dt)));
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
void PartitionImpl<N>::receive_shape(int node_id, int64_t size, char* data) {
    executor->on_shape_msg(node_id, data, size);
}

template void ArrayDAGExecutorND<1>::execute_shape_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_shape_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_shape_node(DAGNode*);

template void ArrayDAGExecutorND<1>::on_shape_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<2>::on_shape_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<3>::on_shape_msg(int, char*, int64_t);

template void PartitionImpl<1>::receive_shape(int, int64_t, char*);
template void PartitionImpl<2>::receive_shape(int, int64_t, char*);
template void PartitionImpl<3>::receive_shape(int, int64_t, char*);
