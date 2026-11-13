#include "backend_internal.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>
#include <climits>
#include <cstdint>
#include <type_traits>

// Indexing with data-dependent communication.
//
// GATHER / SCATTER (the frontend linearizes every index expression into one
// integer array L of flat offsets into the root of the indexed array):
//   GATHER   out[e] = src.flat[L[b] + sum_k s_k * stride_k]
//   SCATTER  dst.flat[L[b] + sum_k s_k * stride_k] = v[e]   (v array or scalar)
// over e = (pre slice dims, b, post slice dims); see the GATHER / SCATTER
// section at the end of this file.
//
// EXPAND writes the flat position i of every element of a view count(x[i])
// times, in order (count = x != 0, or x itself): nonzero and repeat; see the
// EXPAND section.

#ifdef USE_KOKKOS
#define CT_IX_FN KOKKOS_INLINE_FUNCTION
#else
#define CT_IX_FN inline
#endif

namespace {

enum IxKind : int32_t { IX_VAL = 1, IX_SEG = 2, IX_SEGOFF = 3 };

/// Header of a receive_index_msg payload, followed by count * width int64 words.
struct IxHdr {
    int32_t kind;
    int32_t width;
    int32_t from[3]; // sender chare index (EXPAND segment lists)
    int32_t pad;
    int64_t count;
};

/// An operand: root array name/rank and the viewed region (array coordinates).
struct IxView {
    int name = -1;
    int nd = 1;
    int start[3] = {0, 0, 0};
    int step[3] = {1, 1, 1};
    int shape[3] = {1, 1, 1};
};

struct IxSpec {
    Opcode opc = Opcode::GATHER;
    IxView data; // GATHER source / SCATTER target (whole root array)
    IxView L;    // flat base offsets; EXPAND input
    IxView V;    // SCATTER values (name < 0: the scalar below)
    double scalar = 0.0;
    int npre = 0, npost = 0, ndE = 1;
    int64_t E[3] = {1, 1, 1};
    int64_t stride[3] = {0, 0, 0}; // per slice dim: pre dims, then post dims
    int sdim[3] = {0, 0, 0};       // per slice dim: root dim it walks
    int64_t sstart[3] = {0, 0, 0}; // per slice dim: root coordinate at s = 0
    int out_name = -1;
    DType dt = DType::FLOAT64; // element type of the moved values
    int mode = 0;              // EXPAND: 0 counts x != 0, 1 counts x
    int64_t total = 0;         // EXPAND output length
};

using MetaMap = std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>;

IxView view_of(ASTNode* root, int i, const MetaMap& meta) {
    IxView v;
    ASTNode* x = root->operands[i];
    v.name = x->result_name;
    v.nd = x->ndims;
    auto it = meta.find(v.name);
    for (int d = 0; d < v.nd; ++d)
        v.shape[d] = it != meta.end() ? it->second.global_shape[d] : 1;
    Region* r = root->get_operand_region(i);
    if (r == nullptr || r->is_global)
        return v;
    auto fill = [&](auto* ar) {
        for (int d = 0; d < v.nd; ++d) {
            v.start[d] = ar->start[d];
            v.step[d] = ar->step[d];
            v.shape[d] = ar->size(d);
        }
    };
    switch (v.nd) {
    case 1:
        fill(static_cast<ArrayRegion<1>*>(r));
        break;
    case 2:
        fill(static_cast<ArrayRegion<2>*>(r));
        break;
    case 3:
        fill(static_cast<ArrayRegion<3>*>(r));
        break;
    }
    return v;
}

int64_t scalar_at(ASTNode* root, int i) {
    if (i >= (int)root->operands.size() || !root->operands[i]->is_scalar)
        CkAbort("index op %d: missing scalar operand %d", root->opcode, i);
    return static_cast<int64_t>(root->operands[i]->scalar);
}

/// Decode a GATHER/SCATTER/EXPAND root. Operands:
///   GATHER  [src, L, npre, npost, ndE, E..., strides..., sdims..., sstarts...]
///   SCATTER [dst, L, v, npre, npost, ndE, E..., strides..., sdims..., sstarts...]
///   EXPAND  [x, mode, total]
///   COMPRESS [x, mask, total], MASK_SCATTER [x, mask, v, total]
/// Only depends on the AST and array metadata (also used by the receive pass).
IxSpec parse_index(ASTNode* root, const MetaMap& meta) {
    IxSpec s;
    s.opc = static_cast<Opcode>(root->opcode);
    s.out_name = root->result_name;
    if (is_mask_op(s.opc)) {
        s.data = view_of(root, 0, meta);
        s.L = view_of(root, 1, meta);
        int p = 2;
        if (s.opc == Opcode::MASK_SCATTER)
            s.V = view_of(root, p++, meta);
        s.total = scalar_at(root, p);
        s.ndE = 1;
        s.E[0] = s.total;
        s.dt = root->dtype;
        return s;
    }
    if (s.opc == Opcode::EXPAND) {
        s.L = view_of(root, 0, meta);
        s.mode = (int)scalar_at(root, 1);
        s.total = scalar_at(root, 2);
        s.dt = DType::INT64;
        s.ndE = 1;
        s.E[0] = s.total;
        return s;
    }
    s.data = view_of(root, 0, meta);
    s.L = view_of(root, 1, meta);
    int p = 2;
    if (s.opc == Opcode::SCATTER) {
        if (root->operands[2]->is_scalar)
            s.scalar = root->operands[2]->scalar;
        else
            s.V = view_of(root, 2, meta);
        p = 3;
    }
    s.npre = (int)scalar_at(root, p);
    s.npost = (int)scalar_at(root, p + 1);
    s.ndE = (int)scalar_at(root, p + 2);
    for (int d = 0; d < s.ndE; ++d)
        s.E[d] = scalar_at(root, p + 3 + d);
    int ns = s.npre + s.npost;
    for (int k = 0; k < ns; ++k) {
        s.stride[k] = scalar_at(root, p + 3 + s.ndE + k);
        s.sdim[k] = (int)scalar_at(root, p + 3 + s.ndE + ns + k);
        s.sstart[k] = scalar_at(root, p + 3 + s.ndE + 2 * ns + k);
    }
    s.dt = root->dtype;
    return s;
}

/// Tile decomposition of one array (any rank), in array coordinates.
struct IxLayout {
    int nd = 1;
    int64_t shape[3] = {1, 1, 1};
    int off[3] = {0, 0, 0};
    int tile = 1;
    int c0[3] = {0, 0, 0}, c1[3] = {1, 1, 1}; // chares with a non-empty tile

    int64_t size() const { return shape[0] * shape[1] * shape[2]; }
    int64_t lo(int d, int c) const { return std::max<int64_t>((int64_t)c * tile, off[d]) - off[d]; }
    int64_t hi(int d, int c) const {
        return std::min<int64_t>((int64_t)(c + 1) * tile, off[d] + shape[d]) - off[d];
    }
    int64_t num_chares() const {
        int64_t n = 1;
        for (int d = 0; d < nd; ++d)
            n *= std::max(0, c1[d] - c0[d]);
        return n;
    }
    /// k-th chare in row-major order.
    void chare(int64_t k, int ci[3]) const {
        ci[0] = ci[1] = ci[2] = 0;
        for (int d = nd - 1; d >= 0; --d) {
            int n = c1[d] - c0[d];
            ci[d] = c0[d] + (int)(k % n);
            k /= n;
        }
    }
    bool holds(const int ci[3]) const {
        for (int d = 0; d < nd; ++d)
            if (ci[d] < c0[d] || ci[d] >= c1[d])
                return false;
        return true;
    }
    static int64_t pack(const int ci[3]) {
        return ((int64_t)ci[0] << 40) | ((int64_t)ci[1] << 20) | (int64_t)ci[2];
    }
};

IxLayout layout_of(ArrayDAGGroup* g, int name, int nd) {
    auto it = g->array_meta.find(name);
    if (it == g->array_meta.end())
        CkAbort("index op: no metadata for array %d", name);
    const auto& m = it->second;
    IxLayout l;
    l.nd = nd;
    l.tile = m.tile;
    for (int d = 0; d < nd; ++d) {
        l.shape[d] = m.global_shape[d];
        l.off[d] = m.offset[d];
        l.c0[d] = m.offset[d] / m.tile;
        l.c1[d] = m.global_shape[d] > 0 ? (m.offset[d] + m.global_shape[d] - 1) / m.tile + 1
                                        : l.c0[d];
    }
    return l;
}

/// The part of view v inside chare ci's tile: view coords [k0, k0 + cnt) per dim.
bool view_block(const IxView& v, const IxLayout& l, const int ci[3], int64_t k0[3],
                int64_t cnt[3]) {
    if (!l.holds(ci))
        return false;
    for (int d = 0; d < 3; ++d) {
        k0[d] = 0;
        cnt[d] = 1;
    }
    for (int d = 0; d < v.nd; ++d) {
        int64_t lo = l.lo(d, ci[d]), hi = l.hi(d, ci[d]);
        int64_t first = lo <= v.start[d] ? 0 : (lo - v.start[d] + v.step[d] - 1) / v.step[d];
        int64_t last = (hi - 1 - v.start[d]) >= 0 ? (hi - 1 - v.start[d]) / v.step[d] : -1;
        last = std::min<int64_t>(last, v.shape[d] - 1);
        if (last < first)
            return false;
        k0[d] = first;
        cnt[d] = last - first + 1;
    }
    return true;
}

/// Chares holding a non-empty block of view v (row-major order).
std::vector<std::array<int, 3>> view_chares(const IxView& v, const IxLayout& l) {
    std::vector<std::array<int, 3>> out;
    int64_t n = l.num_chares();
    for (int64_t k = 0; k < n; ++k) {
        std::array<int, 3> ci;
        l.chare(k, ci.data());
        int64_t k0[3], cnt[3];
        if (view_block(v, l, ci.data(), k0, cnt))
            out.push_back(ci);
    }
    return out;
}

/// f(B{}) with B the unsigned integer of dt's element size: values move as raw bits.
template <typename F>
void dispatch_bits(DType dt, F&& f) {
    switch (dtype_size(dt)) {
    case 1: f(uint8_t{}); break;
    case 2: f(uint16_t{}); break;
    case 4: f(uint32_t{}); break;
    case 8: f(uint64_t{}); break;
    default: CkAbort("index op: dtype %d is not supported (elements over 8 bytes)", (int)dt);
    }
}

template <typename T>
int64_t to_bits(T v) {
    int64_t b = 0;
    memcpy(&b, &v, sizeof(T));
    return b;
}

template <typename T>
T from_bits(int64_t b) {
    T v;
    memcpy(&v, &b, sizeof(T));
    return v;
}

// ---- device kernels (functors: nvcc cannot take KOKKOS_LAMBDA in generic lambdas) ----

template <typename T>
struct IxGatherFn {
    const T* src;
    const int64_t* off;
    T* out;
    CT_IX_FN void operator()(const int64_t i) const { out[i] = src[off[i]]; }
};

#ifdef USE_KOKKOS
template <int N, typename D, typename S>
void ix_copy(PartitionImpl<N>* p, const D& dst, const S& src) {
#ifdef USE_NVIDIA
    Kokkos::deep_copy(p->compute_exec, dst, src);
#else
    (void)p;
    Kokkos::deep_copy(dst, src);
#endif
}

template <typename T>
using HostUnmanaged = Kokkos::View<T*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
#endif

/// out[i] = data[off[i]] for a tile on the device (or host).
template <int N, typename T>
std::vector<T> gather_tile(PartitionImpl<N>* p, CTArrayBase<N>* arr,
                           const std::vector<int64_t>& off) {
    int64_t n = (int64_t)off.size();
    std::vector<T> out(n);
    if (n == 0)
        return out;
    const T* src = static_cast<const T*>(arr->device_data_ptr());
#ifdef USE_KOKKOS
    Kokkos::View<int64_t*, DeviceSpace> d_off(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "ix_off"), n);
    Kokkos::View<T*, DeviceSpace> d_out(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "ix_vals"), n);
    ix_copy(p, d_off, HostUnmanaged<const int64_t>(off.data(), n));
    Kokkos::parallel_for("ct_ix_gather", CT_COMPUTE_POLICY(p, n),
                         IxGatherFn<T>{src, d_off.data(), d_out.data()});
    ix_copy(p, HostUnmanaged<T>(out.data(), n), d_out);
    compute_fence(p);
#else
    (void)p;
    IxGatherFn<T> f{src, off.data(), out.data()};
    for (int64_t i = 0; i < n; ++i)
        f(i);
#endif
    return out;
}

std::vector<char> make_msg(int32_t kind, int32_t width, const int from[3],
                           const std::vector<int64_t>& words) {
    IxHdr h = {};
    h.kind = kind;
    h.width = width;
    for (int d = 0; d < 3; ++d)
        h.from[d] = from ? from[d] : 0;
    h.count = width > 0 ? (int64_t)words.size() / width : 0;
    std::vector<char> buf(sizeof(h) + words.size() * sizeof(int64_t));
    memcpy(buf.data(), &h, sizeof(h));
    if (!words.empty())
        memcpy(buf.data() + sizeof(h), words.data(), words.size() * sizeof(int64_t));
    return buf;
}

void send_msg(ArrayDAGGroup* g, int nd, const int ci[3], int node_id, std::vector<char>& buf) {
    int64_t size = (int64_t)buf.size();
    switch (nd) {
    case 1:
        ct_send_bytes(proxy_at<1>(g->partition_proxy_1, ChareIndex<1>{{ci[0]}}),
                      MSG_INDEX, node_id, size, buf.data());
        break;
    case 2:
        ct_send_bytes(proxy_at<2>(g->partition_proxy_2, ChareIndex<2>{{ci[0], ci[1]}}),
                      MSG_INDEX, node_id, size, buf.data());
        break;
    case 3:
        ct_send_bytes(proxy_at<3>(g->partition_proxy_3, ChareIndex<3>{{ci[0], ci[1], ci[2]}}),
                      MSG_INDEX, node_id, size, buf.data());
        break;
    }
}

template <size_t N>
void self_index(const std::array<int, N>& a, int ci[3]) {
    for (size_t d = 0; d < 3; ++d)
        ci[d] = d < N ? a[d] : 0;
}

} // namespace

/// Result ndims/shape of an index node (0 for SCATTER, which writes in place).
int index_result_meta(ASTNode* root, const MetaMap& meta, std::array<int, 3>& shape_out) {
    IxSpec s = parse_index(root, meta);
    shape_out = {0, 0, 0};
    if (s.opc == Opcode::SCATTER || s.opc == Opcode::MASK_SCATTER)
        return 0;
    for (int d = 0; d < s.ndE; ++d)
        shape_out[d] = (int)s.E[d];
    return s.ndE;
}

/// Ranks of the partitions taking part in an index node.
std::vector<int> index_op_ndims(ASTNode* root) {
    std::vector<int> nds;
    for (ASTNode* x : root->operands)
        if (!x->is_scalar)
            nds.push_back(x->ndims);
    auto op = static_cast<Opcode>(root->opcode);
    if (op != Opcode::SCATTER && op != Opcode::MASK_SCATTER)
        nds.push_back(root->ndims);
    return nds;
}

// ---- executor ----

namespace {

struct IxCtx {
    IxSpec s;
    IxLayout L_l, out_l;
    std::vector<std::array<int, 3>> drivers; // chares holding a block of x
};

Opcode node_index_op(DAGNode* node) {
    for (ASTNode* root : node->ast->roots)
        if (is_index_op(static_cast<Opcode>(root->opcode)))
            return static_cast<Opcode>(root->opcode);
    return Opcode::NOOP;
}

/// EXPAND context.
IxCtx make_ctx(DAGNode* node, ArrayDAGGroup* g) {
    IxCtx c;
    for (ASTNode* root : node->ast->roots)
        if (is_index_op(static_cast<Opcode>(root->opcode))) {
            c.s = parse_index(root, g->array_meta);
            break;
        }
    c.L_l = layout_of(g, c.s.L.name, c.s.L.nd);
    c.drivers = view_chares(c.s.L, c.L_l);
    c.out_l = layout_of(g, c.s.out_name, 1);
    return c;
}

template <int N>
CTArrayBase<N>* local_array(PartitionImpl<N>* p, int name) {
    auto it = p->arrays.find(name);
    if (it == p->arrays.end())
        CkAbort("index op: array %d has no tile on this chare", name);
    return it->second;
}

/// EXPAND coordinator: order every segment by its flat start, give each one
/// its exclusive prefix count and send the offsets back to the drivers.
template <int N>
void answer_segments(ArrayDAGGroup* g, IndexState<N>& st, const IxCtx& c, int node_id) {
    struct Seg {
        int64_t flat, count;
        int msg;
        int64_t idx;
    };
    std::vector<Seg> segs;
    std::vector<IxHdr> hdrs(st.segs.size());
    for (size_t m = 0; m < st.segs.size(); ++m) {
        memcpy(&hdrs[m], st.segs[m].data(), sizeof(IxHdr));
        const auto* w = reinterpret_cast<const int64_t*>(st.segs[m].data() + sizeof(IxHdr));
        for (int64_t i = 0; i < hdrs[m].count; ++i)
            segs.push_back({w[2 * i], w[2 * i + 1], (int)m, i});
    }
    std::sort(segs.begin(), segs.end(), [](const Seg& a, const Seg& b) { return a.flat < b.flat; });
    std::vector<std::vector<int64_t>> offs(st.segs.size());
    for (size_t m = 0; m < st.segs.size(); ++m)
        offs[m].resize(hdrs[m].count);
    int64_t pos = 0;
    for (auto& sg : segs) {
        offs[sg.msg][sg.idx] = pos;
        pos += sg.count;
    }
    if (pos != c.s.total)
        CkAbort("EXPAND: counts add up to %lld, expected %lld", (long long)pos,
                (long long)c.s.total);
    for (size_t m = 0; m < st.segs.size(); ++m) {
        auto buf = make_msg(IX_SEGOFF, 1, nullptr, offs[m]);
        send_msg(g, c.s.L.nd, hdrs[m].from, node_id, buf);
    }
    st.segs.clear();
}

} // namespace

// ---- COMPRESS / MASK_SCATTER: x[mask] and x[mask] = v, mask of x's view shape ----
//
// The chares holding a block of x ("x chares") scan their mask block on the
// device and send the per-row counts to the first x chare, which answers with
// each row's output offset (row-major over the view). COMPRESS then compacts
// the block and sends the runs in bulk to the result tiles; MASK_SCATTER pulls
// each row's run of v from v's chares, then writes the selected elements. A
// mask laid out differently from x is first moved onto x's blocks.

namespace {

/// 16-byte element (complex128); elements only move.
struct IxElem16 {
    uint64_t w[2];
};

template <typename F>
void dispatch_elem(DType dt, F&& f) {
    switch (dtype_size(dt)) {
    case 1: f(uint8_t{}); break;
    case 2: f(uint16_t{}); break;
    case 4: f(uint32_t{}); break;
    case 8: f(uint64_t{}); break;
    default: f(IxElem16{}); break;
    }
}

enum MxKind : int32_t { MX_MASK = 10, MX_SEG = 11, MX_SEGOFF = 12, MX_DATA = 13, MX_REQ = 14, MX_VAL = 15 };

/// Header of a mask-op message: `nruns` runs of `width` int64 words, then
/// `count` elements (inline, or in device buffer `dbuf` of process `src_node`).
struct MxHdr {
    int32_t kind; // first, as in IxHdr
    int32_t width;
    int32_t from[3];
    int32_t src_node;
    int64_t start[3], stop[3]; // MX_MASK: box in view coordinates
    int64_t nruns;
    int64_t count;
    int64_t dbuf;
};

/// Block of a view on one chare, right-aligned to 3 dims: element (i0, i1, i2)
/// is at base + sum i_k * str[k] in the chare's tile.
struct MxBlock {
    int64_t k0[3] = {0, 0, 0}, cnt[3] = {1, 1, 1}, str[3] = {0, 0, 0}, base = 0;
    int64_t size() const { return cnt[0] * cnt[1] * cnt[2]; }
    int64_t rows() const { return cnt[0] * cnt[1]; }
    void contiguous() {
        base = 0;
        str[2] = 1;
        str[1] = cnt[2];
        str[0] = cnt[1] * cnt[2];
    }
};

bool mx_block(const IxView& v, const IxLayout& l, const int ci[3], MxBlock& b) {
    int64_t k0[3], cnt[3];
    if (!view_block(v, l, ci, k0, cnt))
        return false;
    int sh = 3 - v.nd;
    int64_t lstride = 1;
    for (int d = v.nd - 1; d >= 0; --d) {
        int64_t lo = l.lo(d, ci[d]);
        b.k0[d + sh] = k0[d];
        b.cnt[d + sh] = cnt[d];
        b.base += (v.start[d] + k0[d] * v.step[d] - lo) * lstride;
        b.str[d + sh] = v.step[d] * lstride;
        lstride *= l.hi(d, ci[d]) - lo;
    }
    return true;
}

/// Scan of a mask block. MODE 0: rowpos only; 1: compact x into buf; 2: write buf into x.
template <typename T, int MODE>
struct MxScanFn {
    using value_type = int64_t;
    const uint8_t* m;
    int64_t mbase, ms0, ms1, ms2;
    T* x;
    int64_t xbase, xs0, xs1, xs2;
    T* buf;
    int64_t c1, c2, nrows;
    int64_t* rowpos;
    CT_IX_FN void operator()(const int64_t i, int64_t& upd, const bool final) const {
        int64_t i2 = i, r = 0, i1 = 0, i0 = 0;
        if (nrows > 1) {
            i2 = i % c2;
            r = i / c2;
            i1 = r % c1;
            i0 = r / c1;
        }
        bool f = m[mbase + i0 * ms0 + i1 * ms1 + i2 * ms2] != 0;
        if (final) {
            if (MODE != 2 && i2 == 0)
                rowpos[r] = upd;
            if (MODE != 0 && f) {
                int64_t o = xbase + i0 * xs0 + i1 * xs1 + i2 * xs2;
                if (MODE == 1)
                    buf[upd] = x[o];
                else
                    x[o] = buf[upd];
            }
        }
        if (f)
            upd += 1;
    }
};

template <typename T>
struct MxRunsFn {
    T* out;
    const T* in;
    const int64_t* d; // obase[nr] | ibase[nr] | exclusive prefix of the lengths [nr + 1]
    int64_t nr, istep;
    CT_IX_FN void operator()(const int64_t i) const {
        const int64_t* pre = d + 2 * nr;
        int64_t lo = 0, hi = nr;
        while (hi - lo > 1) {
            int64_t mid = (lo + hi) / 2;
            if (pre[mid] <= i)
                lo = mid;
            else
                hi = mid;
        }
        int64_t j = i - pre[lo];
        out[d[lo] + j] = in[d[nr + lo] + j * istep];
    }
};

/// Element runs: run r is len[r] elements from in[ib[r] + j * istep] to out[ob[r] + j].
struct MxRuns {
    std::vector<int64_t> ob, ib, len;
    int64_t total = 0;
    void add(int64_t o, int64_t i, int64_t n, int64_t istep) {
        total += n;
        if (!len.empty() && ob.back() + len.back() == o && ib.back() + len.back() * istep == i) {
            len.back() += n;
            return;
        }
        ob.push_back(o);
        ib.push_back(i);
        len.push_back(n);
    }
};

char* mx_alloc(int64_t bytes) {
#ifdef USE_KOKKOS
    return static_cast<char*>(Kokkos::kokkos_malloc<DeviceSpace>(std::max<int64_t>(bytes, 1)));
#else
    return new char[std::max<int64_t>(bytes, 1)];
#endif
}

template <int N>
void mx_to_device(PartitionImpl<N>* p, char* dst, const char* src, int64_t bytes) {
#ifdef USE_KOKKOS
    ix_copy(p, Kokkos::View<char*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(dst, bytes),
            HostUnmanaged<const char>(src, bytes));
    compute_fence(p);
#else
    (void)p;
    memcpy(dst, src, bytes);
#endif
}

template <int N>
void mx_to_host(PartitionImpl<N>* p, char* dst, const char* src, int64_t bytes) {
#ifdef USE_KOKKOS
    ix_copy(p, HostUnmanaged<char>(dst, bytes),
            Kokkos::View<const char*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(src,
                                                                                           bytes));
    compute_fence(p);
#else
    (void)p;
    memcpy(dst, src, bytes);
#endif
}

/// Queues the copies of `r` on the compute stream (the caller fences before
/// reading `out` on the host or freeing `in`).
template <int N, typename T>
void mx_copy_runs(PartitionImpl<N>* p, T* out, const T* in, int64_t istep, const MxRuns& r) {
    int64_t nr = (int64_t)r.len.size();
    if (nr == 0)
        return;
#ifdef USE_KOKKOS
    using DevU = Kokkos::View<T*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using CDevU = Kokkos::View<const T*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    if (nr == 1 && istep == 1) {
        ix_copy(p, DevU(out + r.ob[0], r.len[0]), CDevU(in + r.ib[0], r.len[0]));
        return;
    }
    std::vector<int64_t> h(3 * nr + 1, 0);
    for (int64_t k = 0; k < nr; ++k) {
        h[k] = r.ob[k];
        h[nr + k] = r.ib[k];
        h[2 * nr + k + 1] = h[2 * nr + k] + r.len[k];
    }
    Kokkos::View<int64_t*, DeviceSpace> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "ct_mx_runs"),
                                          h.size());
    ix_copy(p, d, HostUnmanaged<const int64_t>(h.data(), h.size()));
    Kokkos::parallel_for("ct_mx_runs", CT_COMPUTE_POLICY(p, r.total),
                         MxRunsFn<T>{out, in, d.data(), nr, istep});
    compute_fence(p); // before `d` and `h` go away
#else
    (void)p;
    for (int64_t k = 0; k < nr; ++k)
        for (int64_t j = 0; j < r.len[k]; ++j)
            out[r.ob[k] + j] = in[r.ib[k] + j * istep];
#endif
}

template <int N, typename T, int MODE>
int64_t mx_scan(PartitionImpl<N>* p, const uint8_t* m, const MxBlock& mb, T* x, const MxBlock& xb,
                T* buf, int64_t* rowpos) {
    int64_t total = 0;
#ifdef USE_KOKKOS
    MxScanFn<T, MODE> f{m,        mb.base,   mb.str[0], mb.str[1], mb.str[2], x,
                        xb.base,  xb.str[0], xb.str[1], xb.str[2], buf,       xb.cnt[1],
                        xb.cnt[2], xb.rows(), rowpos};
    Kokkos::parallel_scan("ct_mask_scan", CT_COMPUTE_POLICY(p, xb.size()), f, total);
#else
    (void)p;
    int64_t r = 0, n2 = xb.cnt[2], ms = mb.str[2], xs = xb.str[2];
    for (int64_t i0 = 0; i0 < xb.cnt[0]; ++i0)
        for (int64_t i1 = 0; i1 < xb.cnt[1]; ++i1, ++r) {
            const uint8_t* mr = m + mb.base + i0 * mb.str[0] + i1 * mb.str[1];
            int64_t xo = xb.base + i0 * xb.str[0] + i1 * xb.str[1];
            if (MODE != 2)
                rowpos[r] = total;
            for (int64_t i2 = 0; i2 < n2; ++i2) {
                bool f = mr[i2 * ms] != 0;
                if (MODE == 1 && f)
                    buf[total] = x[xo + i2 * xs];
                else if (MODE == 2 && f)
                    x[xo + i2 * xs] = buf[total];
                total += f;
            }
        }
#endif
    return total;
}

/// Process currently holding chare c of the nd-D partition.
int mx_chare_node(ArrayDAGGroup* g, int nd, const int c[3]) {
    int pe;
    if (nd == 1)
        pe = g->partition_proxy_1.ckLocalBranch()->lastKnown(CkArrayIndex1D(c[0]));
    else if (nd == 2)
        pe = g->partition_proxy_2.ckLocalBranch()->lastKnown(CkArrayIndex2D(c[0], c[1]));
    else
        pe = g->partition_proxy_3.ckLocalBranch()->lastKnown(CkArrayIndex3D(c[0], c[1], c[2]));
    return CkNodeOf(pe);
}

/// Sends h + runs + `count` elements of `elems` (taken over: a device buffer in
/// GPU builds, passed as a pointer when the receiver is in this process).
template <int N>
void mx_send(PartitionImpl<N>* p, ArrayDAGGroup* g, int nd, const int dest[3], int node_id,
             MxHdr h, const std::vector<int64_t>& runs, char* elems, int64_t ebytes) {
    h.nruns = h.width > 0 ? (int64_t)runs.size() / h.width : 0;
    h.src_node = CkMyNode();
    size_t head = sizeof(h) + runs.size() * sizeof(int64_t);
    int64_t bytes = h.count * ebytes;
    bool inline_data = true;
#ifdef USE_KOKKOS
    inline_data = elems == nullptr || mx_chare_node(g, nd, dest) != CkMyNode();
#endif
    std::vector<char> buf(head + (inline_data ? bytes : 0));
    if (elems != nullptr) {
        if (inline_data) {
            mx_to_host(p, buf.data() + head, elems, bytes);
            free_buffer(elems);
        } else {
            h.dbuf = reinterpret_cast<int64_t>(elems);
        }
    }
    memcpy(buf.data(), &h, sizeof(h));
    if (!runs.empty())
        memcpy(buf.data() + sizeof(h), runs.data(), runs.size() * sizeof(int64_t));
    send_msg(g, nd, dest, node_id, buf);
}

/// A received message's elements in device memory (GPU builds); *owned is set
/// to the buffer to free once the copies reading it are done.
template <int N>
const char* mx_elems(PartitionImpl<N>* p, const std::vector<char>& msg, const MxHdr& h,
                     int64_t ebytes, char** owned) {
    const char* data = msg.data() + sizeof(h) + h.nruns * h.width * sizeof(int64_t);
    *owned = nullptr;
#ifdef USE_KOKKOS
    if (h.dbuf != 0) {
        if (h.src_node != CkMyNode())
            CkAbort("mask op: device buffer from another process");
        *owned = reinterpret_cast<char*>(h.dbuf);
        return *owned;
    }
    if (h.count > 0) {
        *owned = mx_alloc(h.count * ebytes);
        mx_to_device(p, *owned, data, h.count * ebytes);
        return *owned;
    }
#else
    (void)p;
    (void)ebytes;
#endif
    return data;
}

struct MxCtx {
    IxSpec s;
    IxLayout x_l, m_l, v_l, out_l;
    std::vector<std::array<int, 3>> xs, vs; // chares with a block of x / of v
    bool aligned = true;                    // every x chare holds the same mask block
};

MxCtx make_mx_ctx(DAGNode* node, ArrayDAGGroup* g) {
    MxCtx c;
    for (ASTNode* root : node->ast->roots)
        if (is_mask_op(static_cast<Opcode>(root->opcode))) {
            c.s = parse_index(root, g->array_meta);
            break;
        }
    const IxSpec& s = c.s;
    c.x_l = layout_of(g, s.data.name, s.data.nd);
    c.m_l = layout_of(g, s.L.name, s.L.nd);
    c.xs = view_chares(s.data, c.x_l);
    if (s.opc == Opcode::COMPRESS) {
        c.out_l = layout_of(g, s.out_name, 1);
    } else {
        c.v_l = layout_of(g, s.V.name, s.V.nd);
        c.vs = view_chares(s.V, c.v_l);
    }
    // MASK_SCATTER writes x: a mask sharing its storage is read from a copy
    c.aligned = s.data.nd == s.L.nd && (s.opc == Opcode::COMPRESS || s.data.name != s.L.name) &&
                view_chares(s.L, c.m_l) == c.xs;
    for (size_t k = 0; c.aligned && k < c.xs.size(); ++k) {
        int64_t a0[3], an[3], b0[3], bn[3];
        view_block(s.data, c.x_l, c.xs[k].data(), a0, an);
        view_block(s.L, c.m_l, c.xs[k].data(), b0, bn);
        for (int d = 0; d < 3; ++d)
            c.aligned = c.aligned && a0[d] == b0[d] && an[d] == bn[d];
    }
    return c;
}

/// Mask chare (unaligned case): send each x chare the part of its block held here.
template <int N>
void mx_send_mask(PartitionImpl<N>* p, ArrayDAGGroup* g, const MxCtx& c, int node_id,
                  const int me[3]) {
    const IxSpec& s = c.s;
    int64_t m0[3], mn[3];
    if (!view_block(s.L, c.m_l, me, m0, mn))
        return;
    CTArrayBase<N>* arr = local_array(p, s.L.name);
    int64_t tsize[3] = {1, 1, 1};
    for (int d = 0; d < s.L.nd; ++d)
        tsize[d] = c.m_l.hi(d, me[d]) - c.m_l.lo(d, me[d]);
    for (const auto& xc : c.xs) {
        int64_t x0[3], xn[3];
        view_block(s.data, c.x_l, xc.data(), x0, xn);
        MxHdr h = {};
        h.kind = MX_MASK;
        h.count = 1;
        for (int d = 0; d < 3; ++d) {
            h.start[d] = std::max(m0[d], x0[d]);
            h.stop[d] = std::min(m0[d] + mn[d], x0[d] + xn[d]);
            h.count *= std::max<int64_t>(0, h.stop[d] - h.start[d]);
        }
        if (h.count == 0)
            continue;
        // tile offsets of the box (row-major)
        std::vector<int64_t> offs;
        offs.reserve(h.count);
        for (int64_t i = h.start[0]; i < h.stop[0]; ++i)
            for (int64_t j = h.start[1]; j < h.stop[1]; ++j)
                for (int64_t k = h.start[2]; k < h.stop[2]; ++k) {
                    int64_t v[3] = {i, j, k}, o = 0;
                    for (int d = 0; d < s.L.nd; ++d)
                        o = o * tsize[d] + s.L.start[d] + v[d] * s.L.step[d] - c.m_l.lo(d, me[d]);
                    offs.push_back(o);
                }
        std::vector<uint8_t> vals = gather_tile<N, uint8_t>(p, arr, offs);
        std::vector<char> buf(sizeof(h) + vals.size());
        memcpy(buf.data(), &h, sizeof(h));
        memcpy(buf.data() + sizeof(h), vals.data(), vals.size());
        send_msg(g, s.data.nd, xc.data(), node_id, buf);
    }
}

/// x chare: place a received part of the mask block.
template <int N>
void mx_place_mask(IndexState<N>& st, const MxCtx& c, const int me[3], const std::vector<char>& msg) {
    MxHdr h;
    memcpy(&h, msg.data(), sizeof(h));
    int64_t x0[3], xn[3];
    view_block(c.s.data, c.x_l, me, x0, xn);
    const uint8_t* src = reinterpret_cast<const uint8_t*>(msg.data() + sizeof(h));
    for (int64_t i = h.start[0]; i < h.stop[0]; ++i)
        for (int64_t j = h.start[1]; j < h.stop[1]; ++j)
            for (int64_t k = h.start[2]; k < h.stop[2]; ++k)
                st.mblock[((i - x0[0]) * xn[1] + (j - x0[1])) * xn[2] + (k - x0[2])] = *src++;
    st.m_covered += h.count;
}

/// x chare: count the selected elements of every block row (COMPRESS: also
/// compact them into st.cbuf) and send the row counts to the coordinator.
template <int N, typename T>
void mx_scan_block(PartitionImpl<N>* p, ArrayDAGGroup* g, IndexState<N>& st, const MxCtx& c,
                   int node_id, const int me[3]) {
    const IxSpec& s = c.s;
    MxBlock xb, mb;
    mx_block(s.data, c.x_l, me, xb);
    const uint8_t* m;
    char* mtmp = nullptr;
    if (c.aligned) {
        mx_block(s.L, c.m_l, me, mb);
        m = static_cast<const uint8_t*>(local_array(p, s.L.name)->device_data_ptr());
    } else {
        mb = xb;
        mb.contiguous();
        mtmp = mx_alloc(xb.size());
        mx_to_device(p, mtmp, reinterpret_cast<const char*>(st.mblock.data()), xb.size());
        if (s.opc == Opcode::COMPRESS)
            std::vector<uint8_t>().swap(st.mblock);
        m = reinterpret_cast<const uint8_t*>(mtmp);
    }
    T* x = static_cast<T*>(local_array(p, s.data.name)->device_data_ptr());
    int64_t nrows = xb.rows();
    st.rowpos.assign(nrows + 1, 0);
    char* drow = mx_alloc(nrows * sizeof(int64_t));
    int64_t* rowpos = reinterpret_cast<int64_t*>(drow);
#ifndef USE_KOKKOS
    rowpos = st.rowpos.data();
#endif
    int64_t total = mx_scan<N, T, 0>(p, m, mb, x, xb, nullptr, rowpos);
    if (s.opc == Opcode::COMPRESS) {
        st.cbuf = mx_alloc(total * sizeof(T));
        mx_scan<N, T, 1>(p, m, mb, x, xb, reinterpret_cast<T*>(st.cbuf), rowpos);
    }
#ifdef USE_KOKKOS
    mx_to_host(p, reinterpret_cast<char*>(st.rowpos.data()), drow, nrows * sizeof(int64_t));
#endif
    free_buffer(drow);
    if (mtmp)
        free_buffer(mtmp);
    st.rowpos[nrows] = total;
    st.mx_scanned = true;

    // (view flat index, count) per row
    int sh = 3 - s.data.nd;
    int64_t vs[3] = {1, 1, 1};
    for (int d = 0; d < s.data.nd; ++d)
        vs[d + sh] = s.data.shape[d];
    std::vector<int64_t> words(2 * nrows);
    for (int64_t r = 0; r < nrows; ++r) {
        int64_t i0 = r / xb.cnt[1], i1 = r % xb.cnt[1];
        words[2 * r] = ((xb.k0[0] + i0) * vs[1] + xb.k0[1] + i1) * vs[2] + xb.k0[2];
        words[2 * r + 1] = st.rowpos[r + 1] - st.rowpos[r];
    }
    MxHdr h = {};
    h.kind = MX_SEG;
    h.width = 2;
    for (int d = 0; d < 3; ++d)
        h.from[d] = me[d];
    mx_send(p, g, s.data.nd, c.xs[0].data(), node_id, h, words, nullptr, 0);
}

/// Coordinator: every row's output offset, in view row-major order.
template <int N>
void mx_answer(PartitionImpl<N>* p, ArrayDAGGroup* g, IndexState<N>& st, const MxCtx& c,
               int node_id) {
    struct Seg {
        int64_t flat, count;
        int msg;
        int64_t idx;
    };
    std::vector<Seg> segs;
    std::vector<MxHdr> hdrs(st.segs.size());
    for (size_t k = 0; k < st.segs.size(); ++k) {
        memcpy(&hdrs[k], st.segs[k].data(), sizeof(MxHdr));
        const auto* w = reinterpret_cast<const int64_t*>(st.segs[k].data() + sizeof(MxHdr));
        for (int64_t i = 0; i < hdrs[k].nruns; ++i)
            segs.push_back({w[2 * i], w[2 * i + 1], (int)k, i});
    }
    std::sort(segs.begin(), segs.end(), [](const Seg& a, const Seg& b) { return a.flat < b.flat; });
    std::vector<std::vector<int64_t>> offs(st.segs.size());
    for (size_t k = 0; k < st.segs.size(); ++k)
        offs[k].resize(hdrs[k].nruns);
    int64_t pos = 0;
    for (auto& sg : segs) {
        offs[sg.msg][sg.idx] = pos;
        pos += sg.count;
    }
    if (pos != c.s.total)
        CkAbort("mask op: the mask selects %lld elements, expected %lld", (long long)pos,
                (long long)c.s.total);
    for (size_t k = 0; k < st.segs.size(); ++k) {
        MxHdr h = {};
        h.kind = MX_SEGOFF;
        h.width = 1;
        mx_send(p, g, c.s.data.nd, hdrs[k].from, node_id, h, offs[k], nullptr, 0);
    }
    st.segs.clear();
}

/// COMPRESS x chare: send the compacted rows to the result tiles.
template <int N, typename T>
void mx_route(PartitionImpl<N>* p, ArrayDAGGroup* g, IndexState<N>& st, const MxCtx& c,
              int node_id, const int64_t* offs) {
    const IxLayout& o = c.out_l;
    struct Plan {
        std::vector<int64_t> runs; // (offset in the result tile, length)
        MxRuns gather;
    };
    std::map<int, Plan> plans;
    int64_t nrows = (int64_t)st.rowpos.size() - 1;
    for (int64_t r = 0; r < nrows; ++r) {
        int64_t src = st.rowpos[r], n = st.rowpos[r + 1] - src, gpos = offs[r];
        while (n > 0) {
            int ch = (int)((gpos + o.off[0]) / o.tile);
            int64_t lo = o.lo(0, ch);
            int64_t len = std::min(n, o.hi(0, ch) - gpos);
            Plan& pl = plans[ch];
            size_t nr = pl.runs.size();
            if (nr > 0 && pl.runs[nr - 2] + pl.runs[nr - 1] == gpos - lo)
                pl.runs[nr - 1] += len;
            else
                pl.runs.insert(pl.runs.end(), {gpos - lo, len});
            pl.gather.add(pl.gather.total, src, len, 1);
            src += len;
            gpos += len;
            n -= len;
        }
    }
    T* buf = reinterpret_cast<T*>(st.cbuf);
    std::vector<std::pair<int, char*>> bufs;
    bool whole = plans.size() == 1 && plans.begin()->second.gather.len.size() == 1 &&
                 plans.begin()->second.gather.ib[0] == 0;
    for (auto& [ch, pl] : plans) {
        if (whole) {
            bufs.emplace_back(ch, st.cbuf);
            break;
        }
        char* d = mx_alloc(pl.gather.total * sizeof(T));
        mx_copy_runs<N, T>(p, reinterpret_cast<T*>(d), buf, 1, pl.gather);
        bufs.emplace_back(ch, d);
    }
    compute_fence(p); // receivers read the buffers on their own streams
    if (!whole)
        free_buffer(st.cbuf);
    st.cbuf = nullptr;
    size_t k = 0;
    for (auto& [ch, pl] : plans) {
        MxHdr h = {};
        h.kind = MX_DATA;
        h.width = 2;
        h.count = pl.gather.total;
        int dest[3] = {ch, 0, 0};
        mx_send(p, g, 1, dest, node_id, h, pl.runs, bufs[k++].second, sizeof(T));
    }
}

/// MASK_SCATTER x chare: ask every v chare for the parts of v this block takes.
template <int N, typename T>
void mx_request(PartitionImpl<N>* p, ArrayDAGGroup* g, IndexState<N>& st, const MxCtx& c,
                int node_id, const int me[3], const int64_t* offs) {
    const IxView& v = c.s.V;
    const IxLayout& vl = c.v_l;
    std::map<int, std::vector<int64_t>> req; // v chare -> (view index, length, position) triples
    int64_t nrows = (int64_t)st.rowpos.size() - 1;
    for (int64_t r = 0; r < nrows; ++r) {
        int64_t dst = st.rowpos[r], n = st.rowpos[r + 1] - dst, j = offs[r];
        while (n > 0) {
            int64_t a = v.start[0] + j * v.step[0];
            int ch = (int)((a + vl.off[0]) / vl.tile);
            int64_t len = std::min(n, (vl.hi(0, ch) - a + v.step[0] - 1) / v.step[0]);
            req[ch].insert(req[ch].end(), {j, len, dst});
            j += len;
            dst += len;
            n -= len;
        }
    }
    st.cbuf = mx_alloc(st.rowpos[nrows] * sizeof(T));
    st.val_left = (int)c.vs.size();
    static const std::vector<int64_t> none;
    for (const auto& vc : c.vs) {
        auto it = req.find(vc[0]);
        MxHdr h = {};
        h.kind = MX_REQ;
        h.width = 3;
        for (int d = 0; d < 3; ++d)
            h.from[d] = me[d];
        mx_send(p, g, 1, vc.data(), node_id, h, it == req.end() ? none : it->second, nullptr, 0);
    }
}

/// v holder: answer every request (once all have arrived, so no x chare writes
/// before every read of v is done).
template <int N, typename T>
void mx_serve(PartitionImpl<N>* p, ArrayDAGGroup* g, IndexState<N>& st, const MxCtx& c,
              int node_id, const int me[3]) {
    const IxView& v = c.s.V;
    const T* src = static_cast<const T*>(local_array(p, v.name)->device_data_ptr());
    int64_t lo = c.v_l.lo(0, me[0]);
    std::vector<std::pair<MxHdr, std::vector<int64_t>>> replies;
    std::vector<char*> bufs;
    for (auto& m : st.reqs) {
        MxHdr rq;
        memcpy(&rq, m.data(), sizeof(rq));
        const auto* w = reinterpret_cast<const int64_t*>(m.data() + sizeof(rq));
        MxRuns runs;
        std::vector<int64_t> back; // (position, length)
        for (int64_t k = 0; k < rq.nruns; ++k) {
            runs.add(runs.total, v.start[0] + w[3 * k] * v.step[0] - lo, w[3 * k + 1], v.step[0]);
            back.insert(back.end(), {w[3 * k + 2], w[3 * k + 1]});
        }
        char* d = runs.total ? mx_alloc(runs.total * sizeof(T)) : nullptr;
        if (d)
            mx_copy_runs<N, T>(p, reinterpret_cast<T*>(d), src, v.step[0], runs);
        MxHdr h = {};
        h.kind = MX_VAL;
        h.width = 2;
        h.count = runs.total;
        memcpy(h.from, rq.from, sizeof(h.from));
        replies.emplace_back(h, std::move(back));
        bufs.push_back(d);
    }
    compute_fence(p);
    for (size_t k = 0; k < replies.size(); ++k)
        mx_send(p, g, c.s.data.nd, replies[k].first.from, node_id, replies[k].first,
                replies[k].second, bufs[k], sizeof(T));
    st.reqs.clear();
}

/// MASK_SCATTER x chare: place one reply's values into st.cbuf.
template <int N, typename T>
void mx_take_values(PartitionImpl<N>* p, IndexState<N>& st, const std::vector<char>& msg) {
    MxHdr h;
    memcpy(&h, msg.data(), sizeof(h));
    if (h.count == 0)
        return;
    const auto* w = reinterpret_cast<const int64_t*>(msg.data() + sizeof(h));
    char* owned;
    const T* data = reinterpret_cast<const T*>(mx_elems(p, msg, h, sizeof(T), &owned));
    MxRuns runs;
    int64_t pos = 0;
    for (int64_t k = 0; k < h.nruns; ++k) {
        runs.add(w[2 * k], pos, w[2 * k + 1], 1);
        pos += w[2 * k + 1];
    }
    mx_copy_runs<N, T>(p, reinterpret_cast<T*>(st.cbuf), data, 1, runs);
    if (owned) {
        compute_fence(p);
        free_buffer(owned);
    }
}

/// MASK_SCATTER x chare: write the gathered values into the selected elements.
template <int N, typename T>
void mx_write(PartitionImpl<N>* p, IndexState<N>& st, const MxCtx& c, const int me[3]) {
    const IxSpec& s = c.s;
    MxBlock xb, mb;
    mx_block(s.data, c.x_l, me, xb);
    const uint8_t* m;
    char* mtmp = nullptr;
    if (c.aligned) {
        mx_block(s.L, c.m_l, me, mb);
        m = static_cast<const uint8_t*>(local_array(p, s.L.name)->device_data_ptr());
    } else {
        mb = xb;
        mb.contiguous();
        mtmp = mx_alloc(xb.size());
        mx_to_device(p, mtmp, reinterpret_cast<const char*>(st.mblock.data()), xb.size());
        m = reinterpret_cast<const uint8_t*>(mtmp);
    }
    T* x = static_cast<T*>(local_array(p, s.data.name)->device_data_ptr());
    mx_scan<N, T, 2>(p, m, mb, x, xb, reinterpret_cast<T*>(st.cbuf), nullptr);
    compute_fence(p);
    if (mtmp)
        free_buffer(mtmp);
    free_buffer(st.cbuf);
    st.cbuf = nullptr;
}

/// COMPRESS result chare: place one message's runs into the result tile.
template <int N, typename T>
void mx_place_data(PartitionImpl<N>* p, IndexState<N>& st, const MxCtx& c,
                   const std::vector<char>& msg) {
    MxHdr h;
    memcpy(&h, msg.data(), sizeof(h));
    const auto* w = reinterpret_cast<const int64_t*>(msg.data() + sizeof(h));
    char* owned;
    const T* data = reinterpret_cast<const T*>(mx_elems(p, msg, h, sizeof(T), &owned));
    MxRuns runs;
    int64_t pos = 0;
    for (int64_t k = 0; k < h.nruns; ++k) {
        runs.add(w[2 * k], pos, w[2 * k + 1], 1);
        pos += w[2 * k + 1];
    }
    T* out = static_cast<T*>(local_array(p, c.s.out_name)->device_data_ptr());
    mx_copy_runs<N, T>(p, out, data, 1, runs);
    if (owned) {
        compute_fence(p);
        free_buffer(owned);
    }
    st.out_covered += h.count;
}

} // namespace

template <int N>
void ArrayDAGExecutorND<N>::execute_mask_node(DAGNode* node) {
    auto* g = static_cast<ArrayDAGGroup*>(group);
    MxCtx c = make_mx_ctx(node, g);
    const IxSpec& s = c.s;
    int me[3];
    self_index(partition->nd_index(), me);
    std::array<int, 3> mine = {me[0], me[1], me[2]};
    IndexState<N>& st = ix_state[node->id];
    st.node = node;
    st.started = true;

    if (s.opc == Opcode::COMPRESS && N == 1 && c.out_l.holds(me)) {
        ArrayDecomp<N> decomp = g->array_meta[s.out_name].template decomp<N>();
        ArrayRegion<N> tile = decomp.chare_region_local(partition->nd_index());
        std::array<int, N> gs{}, zeros{}, sizes{}, ones{};
        gs[0] = (int)s.total;
        sizes[0] = tile.size(0);
        ones[0] = 1;
        partition->ensure_array(ArrayRegion<N>(zeros, sizes, ones), gs, s.out_name, s.dt, decomp);
        st.out_expected = c.out_l.hi(0, me[0]) - c.out_l.lo(0, me[0]);
    }
    if (s.opc == Opcode::MASK_SCATTER && N == s.V.nd &&
        std::find(c.vs.begin(), c.vs.end(), mine) != c.vs.end())
        st.req_left = (int)c.xs.size();
    if (N == s.data.nd && !c.xs.empty() && c.xs[0] == mine)
        st.seg_left = (int)c.xs.size();
    MxBlock xb;
    if (N == s.data.nd && mx_block(s.data, c.x_l, me, xb)) {
        st.mx_pending = true;
        if (!c.aligned) {
            st.m_expected = xb.size();
            st.mblock.assign(xb.size(), 0);
        }
    }
    if (!c.aligned && N == s.L.nd)
        mx_send_mask(partition, g, c, node->id, me);
    mask_progress(node->id);
}

template <int N>
void ArrayDAGExecutorND<N>::mask_progress(int node_id) {
    IndexState<N>& st = ix_state[node_id];
    auto* g = static_cast<ArrayDAGGroup*>(group);
    MxCtx c = make_mx_ctx(st.node, g);
    int me[3];
    self_index(partition->nd_index(), me);

    std::vector<std::vector<char>> msgs;
    msgs.swap(st.stash);
    for (auto& m : msgs) {
        MxHdr h;
        memcpy(&h, m.data(), sizeof(h));
        switch (h.kind) {
        case MX_MASK:
            mx_place_mask(st, c, me, m);
            break;
        case MX_SEG:
            st.segs.push_back(std::move(m));
            if (--st.seg_left == 0)
                mx_answer(partition, g, st, c, node_id);
            break;
        case MX_SEGOFF: {
            const auto* offs = reinterpret_cast<const int64_t*>(m.data() + sizeof(h));
            dispatch_elem(c.s.dt, [&](auto tag) {
                using T = decltype(tag);
                if (c.s.opc == Opcode::COMPRESS) {
                    mx_route<N, T>(partition, g, st, c, node_id, offs);
                    st.mx_pending = false;
                } else {
                    mx_request<N, T>(partition, g, st, c, node_id, me, offs);
                }
            });
            break;
        }
        case MX_DATA:
            dispatch_elem(c.s.dt, [&](auto tag) {
                mx_place_data<N, decltype(tag)>(partition, st, c, m);
            });
            break;
        case MX_REQ:
            st.reqs.push_back(std::move(m));
            if (--st.req_left == 0)
                dispatch_elem(c.s.dt, [&](auto tag) {
                    mx_serve<N, decltype(tag)>(partition, g, st, c, node_id, me);
                });
            break;
        case MX_VAL:
            dispatch_elem(c.s.dt, [&](auto tag) {
                using T = decltype(tag);
                mx_take_values<N, T>(partition, st, m);
                if (--st.val_left == 0) {
                    mx_write<N, T>(partition, st, c, me);
                    st.mx_pending = false;
                }
            });
            break;
        default:
            CkAbort("mask op: unexpected message kind %d", h.kind);
        }
    }
    if (st.mx_pending && !st.mx_scanned && st.m_covered == st.m_expected)
        dispatch_elem(c.s.dt, [&](auto tag) {
            mx_scan_block<N, decltype(tag)>(partition, g, st, c, node_id, me);
        });
    if (!st.mx_pending && st.seg_left == 0 && st.req_left == 0 && st.val_left == 0 &&
        st.out_covered == st.out_expected) {
        ix_state.erase(node_id);
#ifdef USE_NVIDIA
        auto* p = new ComputeDoneParam<N>{partition, node_id};
        CkCallback hcb(compute_done_cb<N>, p);
        hapiAddCallback(partition->compute_stream_raw, &hcb);
#else
        CT_KOKKOS_FENCE();
        node_finished(node_id);
#endif
    }
}

// ---- GATHER / SCATTER ----
//
// Slice k of the iteration space walks root dim rk of the data array from
// root coordinate ak with step tk, so one entry b (one L value) and a box of
// slice coordinates address a box of data elements; all per-element work runs
// on the device and only per-chare metadata is handled on the host.
//
// GATHER: the chares of L send each result chare ("target") the L values of
// its b range. The target finds, per b, the data chare along the advanced
// (non-slice) dims and its offset in the result tile, groups the entries by
// that chare and sends one request to every data chare ("holder") whose slice
// range meets its box. The holder copies the requested box of values straight
// into the target's tile when both live in one process, else replies with it.
// SCATTER: the chares of L group their entries by destination chare and send
// each group there. A destination keeps the last write per element (NumPy's
// last-write-wins: the largest b) and fills in a scalar, or pulls the values
// from v's chares, which copy them straight into its tile within a process.
// Holders and v chares know from the layouts how many requests to serve.
// Entries outside the array are dropped with a warning (device index arrays
// are not bounds-checked by the frontend).

namespace {

// Message kinds of one index node share receive_index_msg: IxKind 1-3 (EXPAND),
// MxKind 10-15 (COMPRESS / MASK_SCATTER), GxKind 20-29; GX_FREE is handled
// before any node state.
enum GxKind : int32_t {
    GX_LPIECE = 20,
    GX_BUCKET = 21,
    GX_REQ = 22,
    GX_VAL = 23,
    GX_DONE = 24,
    GX_FREE = 25, // return an IPC block to its process
};

struct GxHdr {
    int32_t kind; // first, as in IxHdr
    int32_t src_node;
    int32_t from[3];
    int32_t owned;  // the receiver frees dbuf
    int32_t direct; // REQ: copy into dst_ptr instead of replying
    int32_t from_nd;
    int64_t n;                // entries
    int64_t box0[3], box1[3]; // LPIECE: box of b; REQ / VAL: slice piece
    int64_t ent_off;          // REQ / VAL: first entry of the requester's group
    int64_t bytes;            // payload
    int64_t dbuf;             // payload in device memory of this process (else inline)
    int64_t dst_ptr, ent_ptr; // REQ: requester's tile and entry pairs (same process)
    int64_t dco[2];           // direct REQ: destination offset per unit of each slice coordinate
    int64_t ipc_ptr, ipc_off; // payload in an IPC slab of the sender (same host)
    int32_t ipc_slab, pad;
    char ipc_handle[64];
    int64_t units; // share of the receiver's expected total carried by this part
};

/// Requests and scatter buckets each count kGxUnits, split among their parts.
constexpr int64_t kGxUnits = int64_t(1) << 30;

#ifdef USE_KOKKOS
#define CT_IX_ATOMIC_ADD(p, v) Kokkos::atomic_add((p), (v))
#define CT_IX_ATOMIC_MAX(p, v) Kokkos::atomic_max((p), (v))
#define CT_IX_FETCH_ADD(p, v) Kokkos::atomic_fetch_add((p), (v))
#define CT_IX_CAS(p, o, v) Kokkos::atomic_compare_exchange((p), (o), (v))
#else
#define CT_IX_CAS(p, o, v) ix_cas((p), (o), (v))
#define CT_IX_ATOMIC_ADD(p, v) (*(p) += (v))
#define CT_IX_ATOMIC_MAX(p, v) (*(p) = std::max(*(p), (v)))
#define CT_IX_FETCH_ADD(p, v) ((*(p) += (v)) - (v))
#endif

inline int64_t ix_cas(int64_t* p, int64_t o, int64_t v) {
    int64_t c = *p;
    if (c == o)
        *p = v;
    return c;
}

template <int N, typename F>
void gx_for(PartitionImpl<N>* p, int64_t n, const F& f) {
    if (n <= 0)
        return;
#ifdef USE_KOKKOS
    Kokkos::parallel_for("ct_gx", CT_COMPUTE_POLICY(p, n), f);
#else
    (void)p;
    for (int64_t i = 0; i < n; ++i)
        f(i);
#endif
}

template <int N, typename F>
int64_t gx_scan(PartitionImpl<N>* p, int64_t n, const F& f) {
    int64_t total = 0;
    if (n <= 0)
        return 0;
#ifdef USE_KOKKOS
    Kokkos::parallel_scan("ct_gx_scan", CT_COMPUTE_POLICY(p, n), f, total);
#else
    (void)p;
    for (int64_t i = 0; i < n; ++i)
        f(i, total, true);
#endif
    return total;
}

char* gx_alloc(int64_t bytes) {
#ifdef USE_KOKKOS
    return static_cast<char*>(comm_alloc(std::max<int64_t>(bytes, 16)));
#else
    return new char[std::max<int64_t>(bytes, 16)];
#endif
}

/// Frees a gx_alloc buffer; the caller has fenced every kernel that uses it.
void gx_free(char* b) {
    if (b == nullptr)
        return;
#ifdef USE_KOKKOS
    comm_free(b);
#else
    delete[] b;
#endif
}

template <int N>
void gx_d2d(PartitionImpl<N>* p, char* dst, const char* src, int64_t bytes) {
    if (bytes <= 0)
        return;
#ifdef USE_KOKKOS
    using DevU = Kokkos::View<char*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using CDevU = Kokkos::View<const char*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    ix_copy(p, DevU(dst, bytes), CDevU(src, bytes));
#else
    (void)p;
    memcpy(dst, src, bytes);
#endif
}

template <typename T>
struct GxSetFn {
    T* ptr;
    T val;
    CT_IX_FN void operator()(const int64_t i) const { ptr[i] = val; }
};

/// Offset of (entry j, piece position u) on one side of a copy.
struct GxSide {
    const int64_t* base = nullptr; // per entry, every bstride words; else j * mul
    int64_t bstride = 1, mul = 0, cst = 0;
    int64_t co[2] = {0, 0};
    CT_IX_FN int64_t at(int64_t j, int64_t u0, int64_t u1) const {
        return (base ? base[j * bstride] : j * mul) + cst + u0 * co[0] + u1 * co[1];
    }
};

/// Iteration over nb entries times a piece of up to two slice dims (pc0 x pc1).
struct GxIter {
    int64_t nb = 1, psize = 1, pc1 = 1;
    bool jinner = true; // consecutive threads take consecutive entries
    int64_t size() const { return nb * psize; }
    CT_IX_FN void split(int64_t i, int64_t& j, int64_t& u0, int64_t& u1) const {
        int64_t r;
        int64_t d = jinner ? nb : psize;
        if (nb * psize <= (int64_t)UINT32_MAX) { // 32-bit division
            uint32_t q = (uint32_t)i / (uint32_t)d;
            r = jinner ? q : (uint32_t)i - q * (uint32_t)d;
            j = jinner ? (uint32_t)i - q * (uint32_t)d : q;
        } else {
            int64_t q = i / d;
            r = jinner ? q : i - q * d;
            j = jinner ? i - q * d : q;
        }
        if (pc1 == 1) {
            u0 = r;
            u1 = 0;
        } else {
            u0 = r / pc1;
            u1 = r - u0 * pc1;
        }
    }
};

template <typename DT, typename ST>
struct GxCopyFn {
    DT* dst;
    const ST* src;
    GxSide d, s;
    GxIter it;
    CT_IX_FN void operator()(const int64_t i) const {
        int64_t j, u0, u1;
        it.split(i, j, u0, u1);
        dst[d.at(j, u0, u1)] = static_cast<DT>(src[s.at(j, u0, u1)]);
    }
};

template <typename T>
struct GxFillFn {
    T* dst;
    GxSide d;
    GxIter it;
    T val;
    CT_IX_FN void operator()(const int64_t i) const {
        int64_t j, u0, u1;
        it.split(i, j, u0, u1);
        dst[d.at(j, u0, u1)] = val;
    }
};

/// Data array geometry: which entries are valid and which chare (along the
/// advanced dims) holds them.
struct GxGeo {
    int nd = 1;
    int64_t shape[3] = {1, 1, 1}, off[3] = {0, 0, 0}, tile = 1;
    int64_t c0[3] = {0, 0, 0}, cn[3] = {1, 1, 1};
    int adv[3] = {1, 1, 1};
    int ns = 0, rk[2] = {0, 0};
    int64_t ak[2] = {0, 0};
    int64_t size = 1;
    /// Bucket of flat offset f, or -1 if it is outside the array or the slices.
    CT_IX_FN int64_t bucket(int64_t f) const {
        if (f < 0 || f >= size)
            return -1;
        int64_t c[3] = {0, 0, 0};
        for (int d = nd - 1; d >= 0; --d) {
            c[d] = f % shape[d];
            f /= shape[d];
        }
        for (int q = 0; q < ns; ++q)
            if (c[rk[q]] != ak[q])
                return -1;
        int64_t k = 0;
        for (int d = 0; d < nd; ++d)
            if (adv[d])
                k = k * cn[d] + (c[d] + off[d]) / tile - c0[d];
        return k;
    }
};

/// Entry i of a row-major block of L: pair (flat offset, wcst + sum_d b_d * wco[d])
/// with b the entry's global b coordinates; returns its bucket (nbk: invalid).
struct GxKeyFn {
    const int64_t* L;
    GxGeo g;
    int nB;
    int64_t bcnt[3], bk0[3], wco[3], wcst;
    int64_t nbk;
    int32_t* key;
    int64_t* ent;
    CT_IX_FN int64_t operator()(int64_t i) const {
        int64_t f = L[i];
        int64_t k = g.bucket(f);
        if (k < 0)
            k = nbk;
        int64_t w = wcst, r = i;
        for (int d = nB - 1; d >= 0; --d) {
            w += (bk0[d] + r % bcnt[d]) * wco[d];
            r /= bcnt[d];
        }
        key[i] = (int32_t)k;
        ent[2 * i] = f;
        ent[2 * i + 1] = w;
        return k;
    }
};

/// v chare (along the b dims) of entry j's b position (word 0).
struct GxVKeyFn {
    const int64_t* ent;
    int32_t* key;
    int nB;
    int64_t bshape[3], vstart[3], vstep[3], voff[3], vtile, vc0[3], vcn[3];
    CT_IX_FN int64_t operator()(int64_t j) const {
        int64_t r = ent[2 * j], e[3] = {0, 0, 0};
        for (int d = nB - 1; d >= 0; --d) {
            e[d] = r % bshape[d];
            r /= bshape[d];
        }
        int64_t k = 0;
        for (int d = 0; d < nB; ++d)
            k = k * vcn[d] + (vstart[d] + e[d] * vstep[d] + voff[d]) / vtile - vc0[d];
        key[j] = (int32_t)k;
        return k;
    }
};

template <typename K>
struct GxHistFn {
    using value_type = int64_t[];
    unsigned value_count;
    K k;
    CT_IX_FN void init(int64_t* dst) const {
        for (unsigned b = 0; b < value_count; ++b)
            dst[b] = 0;
    }
    CT_IX_FN void join(int64_t* dst, const int64_t* src) const {
        for (unsigned b = 0; b < value_count; ++b)
            dst[b] += src[b];
    }
    CT_IX_FN void operator()(const int64_t i, int64_t* acc) const { acc[k(i)] += 1; }
};

template <typename K>
struct GxHistAtomicFn {
    K k;
    int64_t* hist;
    CT_IX_FN void operator()(const int64_t i) const { CT_IX_ATOMIC_ADD(&hist[k(i)], (int64_t)1); }
};

/// Runs k over [0, n) and counts its buckets (the last one: invalid entries).
template <int N, typename K>
std::vector<int64_t> gx_hist(PartitionImpl<N>* p, int64_t n, int64_t nbk, const K& k) {
    std::vector<int64_t> h(nbk + 1, 0);
    if (n <= 0)
        return h;
#ifdef USE_KOKKOS
    if (nbk + 1 <= 32) {
        Kokkos::View<int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(h.data(),
                                                                                    nbk + 1);
        Kokkos::parallel_reduce("ct_gx_hist", CT_COMPUTE_POLICY(p, n),
                                GxHistFn<K>{(unsigned)(nbk + 1), k}, hv);
        compute_fence(p);
    } else {
        int64_t* d = reinterpret_cast<int64_t*>(gx_alloc((nbk + 1) * sizeof(int64_t)));
        gx_for(p, nbk + 1, GxSetFn<int64_t>{d, 0});
        gx_for(p, n, GxHistAtomicFn<K>{k, d});
        mx_to_host(p, reinterpret_cast<char*>(h.data()), reinterpret_cast<const char*>(d),
                   (nbk + 1) * sizeof(int64_t));
        gx_free(reinterpret_cast<char*>(d));
    }
#else
    (void)p;
    for (int64_t i = 0; i < n; ++i)
        h[k(i)]++;
#endif
    return h;
}

struct GxPartFn {
    using value_type = int64_t;
    const int32_t* key;
    const int64_t* ent;
    int64_t* out;
    int32_t k;
    int64_t start;
    CT_IX_FN void operator()(const int64_t i, int64_t& upd, const bool final) const {
        if (key[i] != k)
            return;
        if (final) {
            int64_t o = 2 * (start + upd);
            out[o] = ent[2 * i];
            out[o + 1] = ent[2 * i + 1];
        }
        upd += 1;
    }
};

struct GxPlaceFn {
    const int32_t* key;
    const int64_t* ent;
    int64_t* out;
    int64_t* cur;
    int32_t nbk;
    CT_IX_FN void operator()(const int64_t i) const {
        int32_t k = key[i];
        if (k >= nbk)
            return;
        int64_t o = 2 * CT_IX_FETCH_ADD(&cur[k], (int64_t)1);
        out[o] = ent[2 * i];
        out[o + 1] = ent[2 * i + 1];
    }
};

/// Entry pairs grouped by bucket (in entry order for up to 16 buckets), invalid
/// ones dropped; starts[k] is bucket k's first entry. Returns `ent` itself if it
/// is grouped already, else a new buffer.
template <int N>
int64_t* gx_group(PartitionImpl<N>* p, int64_t n, const int32_t* key, int64_t* ent,
                  const std::vector<int64_t>& h, std::vector<int64_t>& starts) {
    int64_t nbk = (int64_t)h.size() - 1;
    starts.assign(nbk + 1, 0);
    int nonempty = 0;
    for (int64_t k = 0; k < nbk; ++k) {
        starts[k + 1] = starts[k] + h[k];
        nonempty += h[k] > 0;
    }
    if (h[nbk] == 0 && nonempty <= 1)
        return ent;
    int64_t* out = reinterpret_cast<int64_t*>(gx_alloc(starts[nbk] * 2 * sizeof(int64_t)));
    if (nbk <= 16) {
        for (int64_t k = 0; k < nbk; ++k)
            if (h[k] > 0)
                gx_scan(p, n, GxPartFn{key, ent, out, (int32_t)k, starts[k]});
    } else {
        int64_t* cur = reinterpret_cast<int64_t*>(gx_alloc(nbk * sizeof(int64_t)));
        mx_to_device(p, reinterpret_cast<char*>(cur), reinterpret_cast<const char*>(starts.data()),
                     nbk * sizeof(int64_t));
        gx_for(p, n, GxPlaceFn{key, ent, out, cur, (int32_t)nbk});
        compute_fence(p);
        gx_free(reinterpret_cast<char*>(cur));
    }
    compute_fence(p);
    return out;
}

/// out[j] = cst + offset in a chare's tile of the advanced coordinates of entry
/// j's flat offset (word 0); akey[j] (optional): their compact index.
struct GxBaseFn {
    const int64_t* ent;
    int64_t* out;
    int64_t* akey;
    int nd;
    int64_t shape[3], lo[3], tstr[3], aco[3];
    int adv[3];
    int64_t cst;
    CT_IX_FN void operator()(const int64_t j) const {
        int64_t f = ent[2 * j], o = cst, a = 0;
        for (int d = nd - 1; d >= 0; --d) {
            int64_t c = f % shape[d] - lo[d];
            f /= shape[d];
            if (adv[d]) {
                o += c * tstr[d];
                a += c * aco[d];
            }
        }
        out[j] = o;
        if (akey)
            akey[j] = a;
    }
};

/// out[j] = cst + offset in a v tile of entry j's b position (word 0).
struct GxVBaseFn {
    const int64_t* ent;
    int64_t* out;
    int nB;
    int64_t bshape[3], vco[3], cst;
    CT_IX_FN void operator()(const int64_t j) const {
        int64_t r = ent[2 * j], o = cst;
        for (int d = nB - 1; d >= 0; --d) {
            o += (r % bshape[d]) * vco[d];
            r /= bshape[d];
        }
        out[j] = o;
    }
};

/// akey[j] -> its slot in an open-addressing table (keys start at -1), for scatters
/// with far fewer entries than tile elements.
struct GxSlotFn {
    int64_t* akey;
    int64_t* keys;
    int64_t mask;
    CT_IX_FN void operator()(const int64_t j) const {
        int64_t k = akey[j];
        uint64_t h = (uint64_t)k * 0x9e3779b97f4a7c15ull;
        int64_t s = (int64_t)(h ^ (h >> 31)) & mask;
        while (true) {
            int64_t prev = CT_IX_CAS(&keys[s], (int64_t)-1, k);
            if (prev == -1 || prev == k)
                break;
            s = (s + 1) & mask;
        }
        akey[j] = s;
    }
};

struct GxWinFn {
    const int64_t* ent;
    const int64_t* akey;
    int64_t* win;
    CT_IX_FN void operator()(const int64_t j) const { CT_IX_ATOMIC_MAX(&win[akey[j]], ent[2 * j + 1]); }
};

/// Keeps the last write of every element: pairs (b position, destination base).
struct GxKeepFn {
    using value_type = int64_t;
    const int64_t* ent;
    const int64_t* akey;
    const int64_t* win;
    const int64_t* dbase;
    int64_t* out;
    CT_IX_FN void operator()(const int64_t j, int64_t& upd, const bool final) const {
        if (win[akey[j]] != ent[2 * j + 1])
            return;
        if (final) {
            out[2 * upd] = ent[2 * j + 1];
            out[2 * upd + 1] = dbase[j];
        }
        upd += 1;
    }
};

/// Largest reply payload (CNUM_GX_CHUNK_MB, default 64): replies are split into parts.
int64_t gx_chunk_bytes() {
    static int64_t mb = [] {
        const char* e = getenv("CNUM_GX_CHUNK_MB");
        return e && atoll(e) > 0 ? atoll(e) : 64;
    }();
    return mb << 20;
}

/// Splits n items of `ibytes` into parts of at most gx_chunk_bytes(): f(first, count, units),
/// with the units of all parts adding up to `total`.
template <typename F>
void gx_split(int64_t n, int64_t ibytes, int64_t total, F&& f) {
    int64_t per = std::max<int64_t>(1, gx_chunk_bytes() / std::max<int64_t>(ibytes, 1));
    int64_t parts = std::max<int64_t>(1, (n + per - 1) / per);
    for (int64_t k = 0; k < parts; ++k) {
        int64_t a = std::min(n, k * per);
        f(a, std::min(n, a + per) - a, total / parts + (k == 0 ? total % parts : 0));
    }
}

int64_t gx_fdiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

/// [s0, s1): the s in [0, cnt) with a + s * t in [lo, hi).
void gx_srange(int64_t a, int64_t t, int64_t lo, int64_t hi, int64_t cnt, int64_t& s0,
               int64_t& s1) {
    if (t > 0) {
        s0 = -gx_fdiv(a - lo, t);
        s1 = -gx_fdiv(a - hi, t);
    } else {
        s0 = gx_fdiv(a - hi, -t) + 1;
        s1 = gx_fdiv(a - lo, -t) + 1;
    }
    s0 = std::max<int64_t>(s0, 0);
    s1 = std::max(std::min(s1, cnt), s0);
}

/// Start (array coordinates) and row-major strides of chare ci's tile.
void gx_tile(const IxLayout& l, const int ci[3], int64_t lo[3], int64_t tstr[3]) {
    int64_t m = 1;
    for (int d = 2; d >= 0; --d) {
        lo[d] = 0;
        tstr[d] = 0;
        if (d < l.nd) {
            lo[d] = l.lo(d, ci[d]);
            tstr[d] = m;
            m *= l.hi(d, ci[d]) - lo[d];
        }
    }
}

struct GxCtx {
    IxSpec s;
    IxLayout D, L, V, O; // data root, index array, values (SCATTER), result (GATHER)
    std::vector<std::array<int, 3>> lch; // chares with a block of L
    int nB = 1, ns = 0, ek[2] = {0, 0};
    int64_t tk[2] = {1, 1};
    GxGeo g;
    int64_t nbk = 1;
    bool jinner = true;
    bool alias = false; // SCATTER v shares the destination's storage
};

GxCtx gx_ctx(DAGNode* node, ArrayDAGGroup* g) {
    GxCtx c;
    for (ASTNode* root : node->ast->roots) {
        auto op = static_cast<Opcode>(root->opcode);
        if (op == Opcode::GATHER || op == Opcode::SCATTER) {
            c.s = parse_index(root, g->array_meta);
            break;
        }
    }
    const IxSpec& s = c.s;
    c.D = layout_of(g, s.data.name, s.data.nd);
    c.L = layout_of(g, s.L.name, s.L.nd);
    c.lch = view_chares(s.L, c.L);
    if (s.opc == Opcode::GATHER)
        c.O = layout_of(g, s.out_name, s.ndE);
    else if (s.V.name >= 0)
        c.V = layout_of(g, s.V.name, s.V.nd);
    c.alias = s.opc == Opcode::SCATTER && s.V.name == s.data.name;
    c.ns = s.npre + s.npost;
    c.nB = s.ndE - c.ns;
    c.jinner = s.npost == 0;
    GxGeo& G = c.g;
    G.nd = s.data.nd;
    G.tile = c.D.tile;
    G.size = c.D.size();
    int64_t rs[3] = {1, 1, 1};
    for (int d = G.nd - 2; d >= 0; --d)
        rs[d] = rs[d + 1] * c.D.shape[d + 1];
    for (int d = 0; d < 3; ++d) {
        G.shape[d] = c.D.shape[d];
        G.off[d] = c.D.off[d];
        G.c0[d] = c.D.c0[d];
        G.cn[d] = std::max(1, c.D.c1[d] - c.D.c0[d]);
        G.adv[d] = d < G.nd ? 1 : 0;
    }
    G.ns = c.ns;
    for (int k = 0; k < c.ns; ++k) {
        c.ek[k] = k < s.npre ? k : c.nB + k;
        G.rk[k] = s.sdim[k];
        G.ak[k] = s.sstart[k];
        c.tk[k] = s.stride[k] / rs[s.sdim[k]];
        G.adv[s.sdim[k]] = 0;
    }
    c.nbk = 1;
    for (int d = 0; d < G.nd; ++d)
        if (G.adv[d])
            c.nbk *= G.cn[d];
    return c;
}

/// Bucket of data chare ci (its coordinates along the advanced dims).
int64_t gx_bucket_of(const GxCtx& c, const int ci[3]) {
    int64_t k = 0;
    for (int d = 0; d < c.g.nd; ++d)
        if (c.g.adv[d])
            k = k * c.g.cn[d] + ci[d] - c.g.c0[d];
    return k;
}

/// Slice box [s0, s1) of data chare ci; false if it holds no element of the slices.
bool gx_data_box(const GxCtx& c, const int ci[3], int64_t s0[2], int64_t s1[2]) {
    if (!c.D.holds(ci))
        return false;
    for (int k = 0; k < c.ns; ++k) {
        int r = c.g.rk[k];
        gx_srange(c.g.ak[k], c.tk[k], c.D.lo(r, ci[r]), c.D.hi(r, ci[r]), c.s.E[c.ek[k]], s0[k],
                  s1[k]);
        if (s0[k] >= s1[k])
            return false;
    }
    return true;
}

/// Box [e0, e1) of the iteration space held by result chare ci (GATHER) or v chare ci.
bool gx_e_box(const GxCtx& c, bool values, const int ci[3], int64_t e0[3], int64_t e1[3]) {
    int64_t k0[3], cnt[3];
    if (values) {
        if (!view_block(c.s.V, c.V, ci, k0, cnt))
            return false;
    } else {
        if (!c.O.holds(ci))
            return false;
        for (int d = 0; d < 3; ++d) {
            k0[d] = d < c.O.nd ? c.O.lo(d, ci[d]) : 0;
            cnt[d] = d < c.O.nd ? c.O.hi(d, ci[d]) - k0[d] : 1;
            if (cnt[d] <= 0)
                return false;
        }
    }
    for (int d = 0; d < 3; ++d) {
        e0[d] = k0[d];
        e1[d] = k0[d] + cnt[d];
    }
    return true;
}

/// Slice piece shared by a data chare's slice box and an iteration-space box.
bool gx_piece(const GxCtx& c, const int64_t s0[2], const int64_t s1[2], const int64_t e0[3],
              const int64_t e1[3], int64_t p0[3], int64_t p1[3]) {
    for (int k = 0; k < 3; ++k) {
        p0[k] = 0;
        p1[k] = 1;
    }
    for (int k = 0; k < c.ns; ++k) {
        p0[k] = std::max(s0[k], e0[c.ek[k]]);
        p1[k] = std::min(s1[k], e1[c.ek[k]]);
        if (p0[k] >= p1[k])
            return false;
    }
    return true;
}

GxIter gx_iter(const GxCtx& c, int64_t n, const int64_t p0[3], const int64_t p1[3]) {
    GxIter it;
    it.nb = std::max<int64_t>(n, 1);
    it.psize = 1;
    for (int k = 0; k < c.ns; ++k)
        it.psize *= p1[k] - p0[k];
    it.pc1 = c.ns == 2 ? p1[1] - p0[1] : 1;
    it.jinner = c.jinner;
    if (n == 0)
        it.nb = 0;
    return it;
}

/// A contiguous buffer laid out in iteration order.
GxSide gx_contig(const GxIter& it) {
    GxSide s;
    if (it.jinner) {
        s.mul = 1;
        s.co[0] = it.nb * it.pc1;
        s.co[1] = it.nb;
    } else {
        s.mul = it.psize;
        s.co[0] = it.pc1;
        s.co[1] = 1;
    }
    return s;
}

/// Tile side: per-entry bases (every bstride words), slice coefficients co.
GxSide gx_tile_side(const GxCtx& c, const int64_t* base, int64_t bstride, const int64_t co[2],
                    const int64_t p0[3]) {
    GxSide s;
    s.base = base;
    s.bstride = bstride;
    for (int k = 0; k < c.ns; ++k) {
        s.co[k] = co[k];
        s.cst += p0[k] * co[k];
    }
    return s;
}

enum GxPay { GX_GIVE, GX_COPY, GX_BORROW };

int gx_chare_pe(ArrayDAGGroup* g, int nd, const int c[3]) {
    if (nd == 1)
        return g->partition_proxy_1.ckLocalBranch()->lastKnown(CkArrayIndex1D(c[0]));
    if (nd == 2)
        return g->partition_proxy_2.ckLocalBranch()->lastKnown(CkArrayIndex2D(c[0], c[1]));
    return g->partition_proxy_3.ckLocalBranch()->lastKnown(CkArrayIndex3D(c[0], c[1], c[2]));
}

/// Sends h with `bytes` of device payload: by pointer within the process (given
/// away, copied for the receiver, or borrowed while the sender waits), through
/// a CUDA IPC slab to another process on this host, else inline. The payload's
/// kernels have completed.
template <int N>
void gx_send(PartitionImpl<N>* p, ArrayDAGGroup* g, int nd, const int dest[3], int node_id,
             GxHdr h, char* payload, int64_t bytes, GxPay how) {
    h.src_node = CkMyNode();
    h.from_nd = N;
    for (int d = 0; d < 3; ++d)
        h.from[d] = d < N ? p->index[d] : 0;
    h.bytes = payload != nullptr ? bytes : 0;
    int pe = gx_chare_pe(g, nd, dest);
    bool local = CkNodeOf(pe) == CkMyNode();
    bool inl = !local && h.bytes > 0;
#ifdef USE_NVIDIA
    static_assert(sizeof(cudaIpcMemHandle_t) <= sizeof(h.ipc_handle), "IPC handle size");
    if (inl && gpu_ipc_peer(pe)) {
        int slab;
        int64_t off;
        if (void* blk = gpu_ipc_alloc(h.bytes, slab, off, false, true)) {
            hapiCheck(cudaMemcpyAsync(blk, payload, h.bytes, cudaMemcpyDeviceToDevice,
                                      p->compute_stream_raw));
            compute_fence(p);
            h.ipc_ptr = reinterpret_cast<int64_t>(blk);
            h.ipc_off = off;
            h.ipc_slab = slab;
            cudaIpcMemHandle_t hd = gpu_ipc_handle(slab);
            memcpy(h.ipc_handle, &hd, sizeof(hd));
            inl = false;
        }
    }
#endif
    std::vector<char> buf(sizeof(h) + (inl ? h.bytes : 0));
    if (h.bytes > 0 && local) {
        char* d = payload;
        if (how == GX_COPY) {
            d = gx_alloc(bytes);
            gx_d2d(p, d, payload, bytes);
            compute_fence(p);
        }
        h.dbuf = reinterpret_cast<int64_t>(d);
        h.owned = how != GX_BORROW;
    } else {
        if (inl)
            mx_to_host(p, buf.data() + sizeof(h), payload, bytes);
        if (payload != nullptr && how == GX_GIVE)
            gx_free(payload);
    }
    memcpy(buf.data(), &h, sizeof(h));
    send_msg(g, nd, dest, node_id, buf);
}

/// A received payload in device memory; *owned is the buffer to free after use.
template <int N>
char* gx_payload(PartitionImpl<N>* p, std::vector<char>& msg, const GxHdr& h, char** owned) {
    *owned = nullptr;
    if (h.bytes == 0)
        return nullptr;
#ifdef USE_NVIDIA
    if (h.ipc_ptr != 0) {
        char* base = gpu_ipc_peer_slab(h.src_node, h.ipc_slab, h.ipc_handle);
        if (base == nullptr)
            CkAbort("index op: IPC slab of process %d is gone", h.src_node);
        return base + h.ipc_off;
    }
#endif
    if (h.dbuf != 0) {
        if (h.src_node != CkMyNode())
            CkAbort("index op: device buffer from another process");
        if (h.owned)
            *owned = reinterpret_cast<char*>(h.dbuf);
        return reinterpret_cast<char*>(h.dbuf);
    }
#ifdef USE_KOKKOS
    char* d = gx_alloc(h.bytes);
    mx_to_device(p, d, msg.data() + sizeof(h), h.bytes);
    *owned = d;
    return d;
#else
    (void)p;
    return msg.data() + sizeof(h);
#endif
}

/// Once the kernels reading a payload have run: free it, or hand the sender's IPC block back.
template <int N>
void gx_release(PartitionImpl<N>* p, ArrayDAGGroup* g, const GxHdr& h, char* owned) {
    if (owned == nullptr && h.ipc_ptr == 0)
        return;
    compute_fence(p);
    gx_free(owned);
    if (h.ipc_ptr != 0) {
        GxHdr f = {};
        f.kind = GX_FREE;
        f.ipc_ptr = h.ipc_ptr;
        std::vector<char> buf(sizeof(f));
        memcpy(buf.data(), &f, sizeof(f));
        int dest[3] = {h.from[0], h.from[1], h.from[2]};
        send_msg(g, h.from_nd, dest, 0, buf);
    }
}

/// The b box [b0, b1) (view coordinates of L, inside this chare's block) as
/// row-major int64 values in a new buffer.
template <int N>
char* gx_pack_L(PartitionImpl<N>* p, const GxCtx& c, const int me[3], const int64_t b0[3],
                const int64_t b1[3]) {
    const IxView& v = c.s.L;
    int64_t lo[3], tstr[3], cnt[3] = {1, 1, 1};
    gx_tile(c.L, me, lo, tstr);
    int64_t n = 1;
    for (int d = 0; d < v.nd; ++d) {
        cnt[d] = b1[d] - b0[d];
        n *= cnt[d];
    }
    char* out = gx_alloc(n * sizeof(int64_t));
    GxIter it;
    it.nb = cnt[0];
    it.pc1 = cnt[2];
    it.psize = cnt[1] * cnt[2];
    it.jinner = false;
    GxSide src, dst;
    for (int d = 0; d < v.nd; ++d)
        src.cst += (v.start[d] + b0[d] * v.step[d] - lo[d]) * tstr[d];
    src.mul = v.step[0] * tstr[0];
    src.co[0] = v.nd > 1 ? v.step[1] * tstr[1] : 0;
    src.co[1] = v.nd > 2 ? v.step[2] * tstr[2] : 0;
    dst.mul = cnt[1] * cnt[2];
    dst.co[0] = cnt[2];
    dst.co[1] = 1;
    CTArrayBase<N>* arr = local_array(p, v.name);
    cn_dispatch(arr->dtype, [&](auto tag) {
        using T = decltype(tag);
        if constexpr (std::is_integral<T>::value) {
            gx_for(p, n,
                   GxCopyFn<int64_t, T>{reinterpret_cast<int64_t*>(out),
                                        static_cast<const T*>(arr->device_data_ptr()), dst, src,
                                        it});
        } else {
            CkAbort("index op: index array of dtype %d", (int)arr->dtype);
        }
    });
    compute_fence(p);
    return out;
}

/// Places a received L piece into the target's block of L (row-major over its b box).
template <int N>
void gx_place_L(PartitionImpl<N>* p, ArrayDAGGroup* g, const GxCtx& c, IndexState<N>& st,
                std::vector<char>& msg, const GxHdr& h, const int64_t e0[3], const int64_t e1[3]) {
    char* owned;
    const int64_t* src = reinterpret_cast<const int64_t*>(gx_payload(p, msg, h, &owned));
    int nB = c.nB, b = c.s.npre;
    int64_t ext[3] = {1, 1, 1}, str[3] = {0, 0, 0}, cnt[3] = {1, 1, 1};
    for (int d = 0; d < nB; ++d)
        ext[d] = e1[b + d] - e0[b + d];
    int64_t m = 1;
    for (int d = nB - 1; d >= 0; --d) {
        str[d] = m;
        m *= ext[d];
    }
    GxSide dst, s;
    int64_t n = 1;
    for (int d = 0; d < nB; ++d) {
        cnt[d] = h.box1[d] - h.box0[d];
        n *= cnt[d];
        dst.cst += (h.box0[d] - e0[b + d]) * str[d];
    }
    dst.mul = str[0];
    dst.co[0] = str[1];
    dst.co[1] = str[2];
    s.mul = cnt[1] * cnt[2];
    s.co[0] = cnt[2];
    s.co[1] = 1;
    GxIter it;
    it.nb = cnt[0];
    it.pc1 = cnt[2];
    it.psize = cnt[1] * cnt[2];
    it.jinner = false;
    if (src != nullptr)
        gx_for(p, n,
               GxCopyFn<int64_t, int64_t>{reinterpret_cast<int64_t*>(st.gx_lblk), src, dst, s, it});
    gx_release(p, g, h, owned);
}

/// Holder of a request (GATHER data chare / SCATTER v chare): copy the values
/// of its entries and slice piece into the requester's tile or a reply.
template <int N, typename T>
void gx_serve(PartitionImpl<N>* p, ArrayDAGGroup* g, const GxCtx& c, int node_id,
              const int me[3], std::vector<char>& msg, const GxHdr& rq) {
    const IxSpec& s = c.s;
    bool gather = s.opc == Opcode::GATHER;
    char* owned = nullptr;
    const int64_t* ent = rq.dbuf != 0 || rq.bytes > 0
                             ? reinterpret_cast<const int64_t*>(gx_payload(p, msg, rq, &owned))
                             : reinterpret_cast<const int64_t*>(rq.ent_ptr);
    int64_t n = rq.n;
    int64_t p0[3] = {rq.box0[0], rq.box0[1], 0}, p1[3] = {rq.box1[0], rq.box1[1], 1};
    GxIter it = gx_iter(c, n, p0, p1);
    GxHdr h = {};
    h.n = n;
    h.ent_off = rq.ent_off;
    memcpy(h.box0, rq.box0, sizeof(h.box0));
    memcpy(h.box1, rq.box1, sizeof(h.box1));
    if (n > 0) {
        int64_t lo[3], tstr[3], sco[2] = {0, 0};
        int64_t* base = reinterpret_cast<int64_t*>(gx_alloc(n * sizeof(int64_t)));
        if (gather) {
            gx_tile(c.D, me, lo, tstr);
            GxBaseFn f{ent, base, nullptr, c.g.nd, {}, {}, {}, {0, 0, 0}, {}, 0};
            for (int d = 0; d < 3; ++d) {
                f.shape[d] = c.g.shape[d];
                f.lo[d] = lo[d];
                f.tstr[d] = tstr[d];
                f.adv[d] = c.g.adv[d];
            }
            for (int k = 0; k < c.ns; ++k) {
                int r = c.g.rk[k];
                f.cst += (c.g.ak[k] - lo[r]) * tstr[r];
                sco[k] = c.tk[k] * tstr[r];
            }
            gx_for(p, n, f);
        } else {
            const IxView& v = s.V;
            gx_tile(c.V, me, lo, tstr);
            GxVBaseFn f{ent, base, c.nB, {1, 1, 1}, {0, 0, 0}, 0};
            for (int d = 0; d < s.ndE; ++d)
                f.cst += (v.start[d] - lo[d]) * tstr[d];
            for (int d = 0; d < c.nB; ++d) {
                int e = s.npre + d;
                f.bshape[d] = s.E[e];
                f.vco[d] = v.step[e] * tstr[e];
            }
            for (int k = 0; k < c.ns; ++k)
                sco[k] = v.step[c.ek[k]] * tstr[c.ek[k]];
            gx_for(p, n, f);
        }
        const T* src = static_cast<const T*>(local_array(p, gather ? s.data.name : s.V.name)
                                                 ->device_data_ptr());
        int nd = gather ? s.ndE : s.data.nd;
        int dest[3] = {rq.from[0], rq.from[1], rq.from[2]};
        if (rq.direct) {
            const int64_t* rent = reinterpret_cast<const int64_t*>(rq.ent_ptr);
            GxSide ds = gx_tile_side(c, rent + 1, 2, rq.dco, p0);
            gx_for(p, it.size(),
                   GxCopyFn<T, T>{reinterpret_cast<T*>(rq.dst_ptr), src, ds,
                                  gx_tile_side(c, base, 1, sco, p0), it});
            compute_fence(p);
        } else {
            // parts of at most gx_chunk_bytes(): rows of the first slice, else entries
            bool rows = it.jinner && c.ns > 0;
            int64_t outer = rows ? p1[0] - p0[0] : n;
            int64_t unit = it.size() / outer * (int64_t)sizeof(T);
            int64_t step = std::max<int64_t>(1, gx_chunk_bytes() / std::max<int64_t>(unit, 1));
            for (int64_t a = 0; a < outer; a += step) {
                int64_t b = std::min(outer, a + step), j0 = rows ? 0 : a;
                int64_t q0[3] = {p0[0], p0[1], p0[2]}, q1[3] = {p1[0], p1[1], p1[2]};
                if (rows) {
                    q0[0] = p0[0] + a;
                    q1[0] = p0[0] + b;
                }
                GxIter pit = gx_iter(c, rows ? n : b - a, q0, q1);
                char* part = gx_alloc(pit.size() * sizeof(T));
                gx_for(p, pit.size(),
                       GxCopyFn<T, T>{reinterpret_cast<T*>(part), src, gx_contig(pit),
                                      gx_tile_side(c, base + j0, 1, sco, q0), pit});
                compute_fence(p);
                GxHdr v = h;
                v.kind = GX_VAL;
                v.n = pit.nb;
                v.ent_off = rq.ent_off + j0;
                memcpy(v.box0, q0, sizeof(v.box0));
                memcpy(v.box1, q1, sizeof(v.box1));
                gx_send(p, g, nd, dest, node_id, v, part, pit.size() * sizeof(T), GX_GIVE);
            }
        }
        gx_free(reinterpret_cast<char*>(base));
    }
    gx_release(p, g, rq, owned);
    if (rq.direct) {
        int nd = gather ? s.ndE : s.data.nd;
        int dest[3] = {rq.from[0], rq.from[1], rq.from[2]};
        h.kind = GX_DONE;
        gx_send(p, g, nd, dest, node_id, h, nullptr, 0, GX_GIVE);
    }
}

/// Requester: write a reply's values into its tile.
template <int N, typename T>
void gx_place_values(PartitionImpl<N>* p, ArrayDAGGroup* g, const GxCtx& c, IndexState<N>& st,
                     const int me[3], std::vector<char>& msg) {
    GxHdr h;
    memcpy(&h, msg.data(), sizeof(h));
    if (h.n == 0)
        return;
    char* owned;
    const T* src = reinterpret_cast<const T*>(gx_payload(p, msg, h, &owned));
    int64_t p0[3] = {h.box0[0], h.box0[1], 0}, p1[3] = {h.box1[0], h.box1[1], 1};
    GxIter it = gx_iter(c, h.n, p0, p1);
    bool gather = c.s.opc == Opcode::GATHER;
    int64_t lo[3], tstr[3], dco[2] = {0, 0};
    gx_tile(gather ? c.O : c.D, me, lo, tstr);
    for (int k = 0; k < c.ns; ++k)
        dco[k] = gather ? tstr[c.ek[k]] : c.tk[k] * tstr[c.g.rk[k]];
    const int64_t* ent = reinterpret_cast<const int64_t*>(st.gx_ent) + 2 * h.ent_off;
    GxSide ds = gx_tile_side(c, ent + 1, 2, dco, p0);
    T* dst = static_cast<T*>(local_array(p, gather ? c.s.out_name : c.s.data.name)->device_data_ptr());
    gx_for(p, it.size(), GxCopyFn<T, T>{dst, src, ds, gx_contig(it), it});
    gx_release(p, g, h, owned);
}

/// Elements of a request or reply (entries times slice piece).
int64_t gx_elems(const GxCtx& c, const GxHdr& h) {
    int64_t e = h.n;
    for (int k = 0; k < c.ns; ++k)
        e *= h.box1[k] - h.box0[k];
    return e;
}

/// A request to another process: its entry pairs inline, in parts.
template <int N>
void gx_send_req(PartitionImpl<N>* p, ArrayDAGGroup* g, int nd, const int dest[3], int node_id,
                 const GxHdr& h, char* pairs) {
    gx_split(h.n, 2 * sizeof(int64_t), kGxUnits, [&](int64_t a, int64_t m, int64_t u) {
        GxHdr q = h;
        q.n = m;
        q.ent_off = h.ent_off + a;
        q.units = u;
        gx_send(p, g, nd, dest, node_id, q, pairs + a * 2 * sizeof(int64_t),
                m * 2 * sizeof(int64_t), GX_BORROW);
    });
}

void gx_warn_dropped(int64_t n) {
    if (n > 0)
        CkPrintf("charmnumeric: warning: %lld out-of-bounds indices ignored\n", (long long)n);
}

} // namespace

template <int N>
void ArrayDAGExecutorND<N>::gx_start(DAGNode* node) {
    auto* g = static_cast<ArrayDAGGroup*>(group);
    IndexState<N>& st = ix_state[node->id];
    auto ctx = std::make_shared<GxCtx>(gx_ctx(node, g));
    st.ctx = ctx;
    const GxCtx& c = *ctx;
    const IxSpec& s = c.s;
    bool gather = s.opc == Opcode::GATHER;
    int me[3];
    self_index(partition->nd_index(), me);
    st.node = node;
    st.started = true;
    int64_t s0[2], s1[2], e0[3], e1[3], p0[3], p1[3];

    if (gather) {
        // Target: allocate the result tile; count the L pieces and the holders to ask.
        if (N == s.ndE && gx_e_box(c, false, me, e0, e1)) {
            st.gx_target = true;
            ArrayDecomp<N> decomp = g->array_meta[s.out_name].template decomp<N>();
            ArrayRegion<N> mine = decomp.chare_region_local(partition->nd_index());
            std::array<int, N> gs{}, zeros{}, sizes{}, ones{};
            for (int d = 0; d < N; ++d) {
                gs[d] = (int)s.E[d];
                sizes[d] = mine.size(d);
                ones[d] = 1;
            }
            partition->ensure_array(ArrayRegion<N>(zeros, sizes, ones), gs, s.out_name, s.dt,
                                    decomp);
            int64_t nb = 1;
            for (int d = 0; d < c.nB; ++d)
                nb *= e1[s.npre + d] - e0[s.npre + d];
            st.gx_lblk = gx_alloc(nb * sizeof(int64_t));
            st.lmsg_left = nb; // L elements to come

        }
        // Holder: one request from every result chare whose slice box meets ours.
        if (N == s.data.nd && gx_data_box(c, me, s0, s1))
            for (int64_t k = 0; k < c.O.num_chares(); ++k) {
                int ci[3];
                c.O.chare(k, ci);
                st.req_left += (gx_e_box(c, false, ci, e0, e1) && gx_piece(c, s0, s1, e0, e1, p0, p1)) *
                               kGxUnits;
            }
        // L chare: send every result chare the part of its b range held here.
        int64_t k0[3], cnt[3];
        if (N == s.L.nd && view_block(s.L, c.L, me, k0, cnt))
            for (int64_t k = 0; k < c.O.num_chares(); ++k) {
                int ci[3];
                c.O.chare(k, ci);
                if (!gx_e_box(c, false, ci, e0, e1))
                    continue;
                GxHdr h = {};
                h.kind = GX_LPIECE;
                bool meets = true;
                for (int d = 0; d < 3; ++d) {
                    h.box0[d] = d < c.nB ? std::max(k0[d], e0[s.npre + d]) : 0;
                    h.box1[d] = d < c.nB ? std::min(k0[d] + cnt[d], e1[s.npre + d]) : 1;
                    meets = meets && h.box0[d] < h.box1[d];
                }
                if (!meets)
                    continue;
                int64_t row = 1;
                for (int d = 1; d < c.nB; ++d)
                    row *= h.box1[d] - h.box0[d];
                int64_t r0 = h.box0[0];
                gx_split(h.box1[0] - r0, row * sizeof(int64_t), 0, [&](int64_t a, int64_t m, int64_t) {
                    GxHdr q = h;
                    q.box0[0] = r0 + a;
                    q.box1[0] = r0 + a + m;
                    q.n = q.units = m * row;
                    char* buf = gx_pack_L(partition, c, me, q.box0, q.box1);
                    gx_send(partition, g, s.ndE, ci, node->id, q, buf, q.n * sizeof(int64_t),
                            GX_GIVE);
                });
            }
    } else {
        // Destination: one bucket from every L chare, then one reply per v chare asked.
        if (N == s.data.nd && gx_data_box(c, me, s0, s1)) {
            st.gx_target = true;
            st.lmsg_left = (int64_t)c.lch.size() * kGxUnits;
        }
        // v chare: one request from every destination whose slice box meets ours.
        if (s.V.name >= 0 && N == s.V.nd && gx_e_box(c, true, me, e0, e1))
            for (int64_t k = 0; k < c.D.num_chares(); ++k) {
                int ci[3];
                c.D.chare(k, ci);
                st.req_left += (gx_data_box(c, ci, s0, s1) && gx_piece(c, s0, s1, e0, e1, p0, p1)) *
                               kGxUnits;
            }
        // L chare: group the block's entries by destination chare and send the groups.
        int64_t k0[3], cnt[3];
        if (N == s.L.nd && view_block(s.L, c.L, me, k0, cnt)) {
            int64_t b1[3], n = 1;
            for (int d = 0; d < 3; ++d) {
                b1[d] = k0[d] + cnt[d];
                n *= cnt[d];
            }
            char* lp = gx_pack_L(partition, c, me, k0, b1);
            int32_t* key = reinterpret_cast<int32_t*>(gx_alloc(n * sizeof(int32_t)));
            int64_t* ent = reinterpret_cast<int64_t*>(gx_alloc(n * 2 * sizeof(int64_t)));
            GxKeyFn kf{reinterpret_cast<const int64_t*>(lp), c.g, c.nB, {1, 1, 1}, {0, 0, 0},
                       {0, 0, 0}, 0, c.nbk, key, ent};
            int64_t m = 1;
            for (int d = c.nB - 1; d >= 0; --d) {
                kf.bcnt[d] = cnt[d];
                kf.bk0[d] = k0[d];
                kf.wco[d] = m;
                m *= s.E[s.npre + d];
            }
            std::vector<int64_t> hist = gx_hist(partition, n, c.nbk, kf), starts;
            gx_warn_dropped(hist[c.nbk]);
            int64_t* grouped = gx_group(partition, n, key, ent, hist, starts);
            for (int64_t k = 0; k < c.D.num_chares(); ++k) {
                int ci[3];
                c.D.chare(k, ci);
                if (!gx_data_box(c, ci, s0, s1))
                    continue;
                int64_t b = gx_bucket_of(c, ci);
                gx_split(hist[b], 2 * sizeof(int64_t), kGxUnits, [&](int64_t a, int64_t m, int64_t u) {
                    GxHdr h = {};
                    h.kind = GX_BUCKET;
                    h.n = m;
                    h.units = u;
                    gx_send(partition, g, s.data.nd, ci, node->id, h,
                            reinterpret_cast<char*>(grouped + 2 * (starts[b] + a)),
                            m * 2 * sizeof(int64_t), GX_COPY);
                });
            }
            if (grouped != ent)
                gx_free(reinterpret_cast<char*>(grouped));
            gx_free(reinterpret_cast<char*>(ent));
            gx_free(reinterpret_cast<char*>(key));
            gx_free(lp);
        }
    }
    gx_progress(node->id);
}

template <int N>
void ArrayDAGExecutorND<N>::gx_progress(int node_id) {
    IndexState<N>& st = ix_state[node_id];
    if (!st.started)
        return;
    auto* g = static_cast<ArrayDAGGroup*>(group);
    std::shared_ptr<void> keep = st.ctx;
    const GxCtx& c = *static_cast<const GxCtx*>(keep.get());
    const IxSpec& s = c.s;
    bool gather = s.opc == Opcode::GATHER;
    int me[3];
    self_index(partition->nd_index(), me);
    int64_t s0[2] = {0, 0}, s1[2] = {1, 1}, e0[3] = {0, 0, 0}, e1[3] = {1, 1, 1}, p0[3], p1[3];
    if (st.gx_target) {
        if (gather)
            gx_e_box(c, false, me, e0, e1);
        else
            gx_data_box(c, me, s0, s1);
    }

    std::vector<std::vector<char>> msgs;
    msgs.swap(st.stash);
    for (auto& m : msgs) {
        GxHdr h;
        memcpy(&h, m.data(), sizeof(h));
        switch (h.kind) {
        case GX_LPIECE:
            gx_place_L(partition, g, c, st, m, h, e0, e1);
            st.lmsg_left -= h.units;
            break;
        case GX_BUCKET:
            st.gx_msgs.push_back(std::move(m));
            st.lmsg_left -= h.units;
            break;
        case GX_REQ:
            dispatch_elem(s.dt, [&](auto tag) {
                gx_serve<N, decltype(tag)>(partition, g, c, node_id, me, m, h);
            });
            st.req_left -= h.units;
            break;
        case GX_VAL:
            st.rep_elems -= gx_elems(c, h);
            if (c.alias && st.req_left > 0)
                st.gx_late.push_back(std::move(m));
            else
                dispatch_elem(s.dt, [&](auto tag) {
                    gx_place_values<N, decltype(tag)>(partition, g, c, st, me, m);
                });
            break;
        case GX_DONE:
            --st.rep_left;
            break;
        default:
            CkAbort("index op: unexpected message kind %d", h.kind);
        }
    }

    if (st.gx_target && !st.gx_issued && st.lmsg_left == 0) {
        st.gx_issued = true;
        if (gather) {
            // Entries of the b range: holder bucket, flat offset, offset in the result tile.
            int64_t lo[3], ostr[3];
            gx_tile(c.O, me, lo, ostr);
            int64_t n = 1;
            for (int d = 0; d < c.nB; ++d)
                n *= e1[s.npre + d] - e0[s.npre + d];
            int32_t* key = reinterpret_cast<int32_t*>(gx_alloc(n * sizeof(int32_t)));
            int64_t* ent = reinterpret_cast<int64_t*>(gx_alloc(n * 2 * sizeof(int64_t)));
            GxKeyFn kf{reinterpret_cast<const int64_t*>(st.gx_lblk), c.g, c.nB, {1, 1, 1},
                       {0, 0, 0}, {0, 0, 0}, 0, c.nbk, key, ent};
            for (int d = 0; d < s.ndE; ++d)
                kf.wcst -= e0[d] * ostr[d];
            for (int d = 0; d < c.nB; ++d) {
                kf.bcnt[d] = e1[s.npre + d] - e0[s.npre + d];
                kf.bk0[d] = e0[s.npre + d];
                kf.wco[d] = ostr[s.npre + d];
            }
            std::vector<int64_t> hist = gx_hist(partition, n, c.nbk, kf), starts;
            int64_t* grouped = gx_group(partition, n, key, ent, hist, starts);
            if (grouped != ent)
                gx_free(reinterpret_cast<char*>(ent));
            gx_free(reinterpret_cast<char*>(key));
            gx_free(st.gx_lblk);
            st.gx_lblk = nullptr;
            st.gx_ent = reinterpret_cast<char*>(grouped);
            CTArrayBase<N>* out = local_array(partition, s.out_name);
            if (hist[c.nbk] > 0) {
                gx_warn_dropped(hist[c.nbk]);
                dispatch_elem(s.dt, [&](auto tag) {
                    using T = decltype(tag);
                    gx_for(partition, out->local_size(),
                           GxSetFn<T>{static_cast<T*>(out->device_data_ptr()), T{}});
                });
                compute_fence(partition);
            }
            for (int64_t k = 0; k < c.D.num_chares(); ++k) {
                int ci[3];
                c.D.chare(k, ci);
                if (!gx_data_box(c, ci, s0, s1) || !gx_piece(c, s0, s1, e0, e1, p0, p1))
                    continue;
                int64_t b = gx_bucket_of(c, ci);
                GxHdr h = {};
                h.kind = GX_REQ;
                h.n = hist[b];
                h.ent_off = starts[b];
                memcpy(h.box0, p0, sizeof(h.box0));
                memcpy(h.box1, p1, sizeof(h.box1));
                char* pairs = reinterpret_cast<char*>(grouped + 2 * starts[b]);
                if (mx_chare_node(g, s.data.nd, ci) == CkMyNode()) {
                    h.direct = 1;
                    ++st.rep_left;
                    h.dst_ptr = reinterpret_cast<int64_t>(out->device_data_ptr());
                    h.ent_ptr = reinterpret_cast<int64_t>(pairs);
                    for (int q = 0; q < c.ns; ++q)
                        h.dco[q] = ostr[c.ek[q]];
                    h.units = kGxUnits;
                    gx_send(partition, g, s.data.nd, ci, st.node->id, h, nullptr, 0, GX_BORROW);
                } else {
                    st.rep_elems += gx_elems(c, h);
                    gx_send_req(partition, g, s.data.nd, ci, st.node->id, h, pairs);
                }
            }
        } else {
            // All entries landing here: offsets in the tile, then the last write per element.
            int64_t n = 0;
            for (auto& m : st.gx_msgs) {
                GxHdr h;
                memcpy(&h, m.data(), sizeof(h));
                n += h.n;
            }
            int64_t* ent = reinterpret_cast<int64_t*>(gx_alloc(n * 2 * sizeof(int64_t)));
            int64_t at = 0;
            for (auto& m : st.gx_msgs) {
                GxHdr h;
                memcpy(&h, m.data(), sizeof(h));
                char* owned;
                char* src = gx_payload(partition, m, h, &owned);
                gx_d2d(partition, reinterpret_cast<char*>(ent + 2 * at), src, h.n * 2 * sizeof(int64_t));
                at += h.n;
                gx_release(partition, g, h, owned);
            }
            std::vector<std::vector<char>>().swap(st.gx_msgs);
            int64_t lo[3], tstr[3], dco[2] = {0, 0};
            gx_tile(c.D, me, lo, tstr);
            GxBaseFn bf{ent, nullptr, nullptr, c.g.nd, {}, {}, {}, {0, 0, 0}, {}, 0};
            int64_t aext = 1;
            for (int d = 2; d >= 0; --d) {
                bf.shape[d] = c.g.shape[d];
                bf.lo[d] = lo[d];
                bf.tstr[d] = tstr[d];
                bf.adv[d] = c.g.adv[d];
                if (d < c.g.nd && c.g.adv[d]) {
                    bf.aco[d] = aext;
                    aext *= c.D.hi(d, me[d]) - lo[d];
                }
            }
            for (int k = 0; k < c.ns; ++k) {
                int r = c.g.rk[k];
                bf.cst += (c.g.ak[k] - lo[r]) * tstr[r];
                dco[k] = c.tk[k] * tstr[r];
            }
            int64_t* dbase = reinterpret_cast<int64_t*>(gx_alloc(n * sizeof(int64_t)));
            bf.out = dbase;
            CTArrayBase<N>* dst = local_array(partition, s.data.name);
            for (int k = 0; k < c.ns; ++k) {
                p0[k] = s0[k];
                p1[k] = s1[k];
            }
            if (s.V.name < 0) {
                gx_for(partition, n, bf);
                int64_t bits = 0;
                cn_dispatch(s.dt, [&](auto tag) {
                    using T = decltype(tag);
                    if constexpr (sizeof(T) <= sizeof(int64_t))
                        bits = to_bits<T>(cn_cast<T>(s.scalar));
                    else
                        CkAbort("SCATTER: scalar of dtype %d", (int)s.dt);
                });
                GxIter it = gx_iter(c, n, p0, p1);
                GxSide ds = gx_tile_side(c, dbase, 1, dco, p0);
                dispatch_elem(s.dt, [&](auto tag) {
                    using T = decltype(tag);
                    if constexpr (sizeof(T) <= sizeof(int64_t))
                        gx_for(partition, it.size(),
                               GxFillFn<T>{static_cast<T*>(dst->device_data_ptr()), ds, it,
                                           from_bits<T>(bits)});
                });
                compute_fence(partition);
                gx_free(reinterpret_cast<char*>(dbase));
                gx_free(reinterpret_cast<char*>(ent));
            } else {
                int64_t* akey = reinterpret_cast<int64_t*>(gx_alloc(n * sizeof(int64_t)));
                int64_t cap = 16;
                while (cap < 2 * n)
                    cap <<= 1;
                int64_t wn = aext > 2 * cap ? cap : aext;
                int64_t* win = reinterpret_cast<int64_t*>(gx_alloc(wn * sizeof(int64_t)));
                int64_t* kept = reinterpret_cast<int64_t*>(gx_alloc(n * 2 * sizeof(int64_t)));
                bf.akey = akey;
                gx_for(partition, n, bf);
                if (wn != aext) {
                    int64_t* keys = reinterpret_cast<int64_t*>(gx_alloc(cap * sizeof(int64_t)));
                    gx_for(partition, cap, GxSetFn<int64_t>{keys, -1});
                    gx_for(partition, n, GxSlotFn{akey, keys, cap - 1});
                    compute_fence(partition);
                    gx_free(reinterpret_cast<char*>(keys));
                }
                gx_for(partition, wn, GxSetFn<int64_t>{win, -1});
                gx_for(partition, n, GxWinFn{ent, akey, win});
                int64_t nk = gx_scan(partition, n, GxKeepFn{ent, akey, win, dbase, kept});
                compute_fence(partition);
                gx_free(reinterpret_cast<char*>(win));
                gx_free(reinterpret_cast<char*>(akey));
                gx_free(reinterpret_cast<char*>(dbase));
                gx_free(reinterpret_cast<char*>(ent));
                // group by v chare along the b dims
                const IxView& v = s.V;
                int32_t* key = reinterpret_cast<int32_t*>(gx_alloc(nk * sizeof(int32_t)));
                GxVKeyFn kf{kept, key, c.nB, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0},
                            c.V.tile, {0, 0, 0}, {1, 1, 1}};
                int64_t nvb = 1;
                for (int d = 0; d < c.nB; ++d) {
                    int e = s.npre + d;
                    kf.bshape[d] = s.E[e];
                    kf.vstart[d] = v.start[e];
                    kf.vstep[d] = v.step[e];
                    kf.voff[d] = c.V.off[e];
                    kf.vc0[d] = c.V.c0[e];
                    kf.vcn[d] = std::max(1, c.V.c1[e] - c.V.c0[e]);
                    nvb *= kf.vcn[d];
                }
                std::vector<int64_t> hist = gx_hist(partition, nk, nvb, kf), starts;
                int64_t* grouped = gx_group(partition, nk, key, kept, hist, starts);
                if (grouped != kept)
                    gx_free(reinterpret_cast<char*>(kept));
                gx_free(reinterpret_cast<char*>(key));
                st.gx_ent = reinterpret_cast<char*>(grouped);
                for (int64_t k = 0; k < c.V.num_chares(); ++k) {
                    int ci[3];
                    c.V.chare(k, ci);
                    if (!gx_e_box(c, true, ci, e0, e1) || !gx_piece(c, s0, s1, e0, e1, p0, p1))
                        continue;
                    int64_t b = 0;
                    for (int d = 0; d < c.nB; ++d)
                        b = b * kf.vcn[d] + ci[s.npre + d] - kf.vc0[d];
                    GxHdr h = {};
                    h.kind = GX_REQ;
                    h.n = hist[b];
                    h.ent_off = starts[b];
                    memcpy(h.box0, p0, sizeof(h.box0));
                    memcpy(h.box1, p1, sizeof(h.box1));
                    char* pairs = reinterpret_cast<char*>(grouped + 2 * starts[b]);
                    if (mx_chare_node(g, s.V.nd, ci) == CkMyNode()) {
                        h.direct = !c.alias;
                        if (h.direct)
                            ++st.rep_left;
                        else
                            st.rep_elems += gx_elems(c, h);
                        h.dst_ptr = reinterpret_cast<int64_t>(dst->device_data_ptr());
                        h.ent_ptr = reinterpret_cast<int64_t>(pairs);
                        for (int q = 0; q < c.ns; ++q)
                            h.dco[q] = dco[q];
                        h.units = kGxUnits;
                        gx_send(partition, g, s.V.nd, ci, st.node->id, h, nullptr, 0, GX_BORROW);
                    } else {
                        st.rep_elems += gx_elems(c, h);
                        gx_send_req(partition, g, s.V.nd, ci, st.node->id, h, pairs);
                    }
                }
            }
        }
    }
    if (c.alias && st.req_left == 0 && !st.gx_late.empty()) {
        for (auto& m : st.gx_late)
            dispatch_elem(s.dt, [&](auto tag) {
                gx_place_values<N, decltype(tag)>(partition, g, c, st, me, m);
            });
        st.gx_late.clear();
    }

    if (st.req_left == 0 && st.lmsg_left == 0 && st.rep_left == 0 && st.rep_elems == 0 &&
        st.gx_late.empty() &&
        (!st.gx_target || st.gx_issued)) {
        compute_fence(partition);
        gx_free(st.gx_ent);
        gx_free(st.gx_lblk);
        ix_state.erase(node_id);
#ifdef USE_NVIDIA
        auto* p = new ComputeDoneParam<N>{partition, node_id};
        CkCallback hcb(compute_done_cb<N>, p);
        hapiAddCallback(partition->compute_stream_raw, &hcb);
#else
        CT_KOKKOS_FENCE();
        node_finished(node_id);
#endif
    }
}

// ---- EXPAND ----
//
// Each chare holding a block of x scans it on the device: the count of every
// element (x != 0, or max(x, 0)), every block row's exclusive prefix and the
// flat positions, each repeated count times. The row counts go to the first x
// chare, which answers with every row's output offset (row-major over the
// view); the positions then move in runs to the result tiles, as for COMPRESS.
// Limit: a block's positions are one device buffer (its whole output count).

namespace {

template <typename T, bool COUNT, bool WRITE>
struct ExScanFn {
    using value_type = int64_t;
    const T* x;
    int64_t xbase, xs0, xs1, xs2, c1, c2;
    int64_t f0, fs0, fs1; // flat view position of block element (i0, i1, i2)
    int64_t* rowpos;
    int64_t* out;
    int64_t* pre; // optional: every element's exclusive prefix
    CT_IX_FN void operator()(const int64_t i, int64_t& upd, const bool final) const {
        int64_t i2 = i % c2, r = i / c2, i1 = r % c1, i0 = r / c1;
        T v = x[xbase + i0 * xs0 + i1 * xs1 + i2 * xs2];
        int64_t k;
        if constexpr (COUNT) {
            k = cn_cast<int64_t>(v);
            k = k > 0 ? k : 0;
        } else {
            k = cn_nonzero(v) ? 1 : 0;
        }
        if (final) {
            if (!WRITE && i2 == 0)
                rowpos[r] = upd;
            if (pre)
                pre[i] = upd;
            if (WRITE) {
                int64_t f = f0 + i0 * fs0 + i1 * fs1 + i2;
                for (int64_t q = 0; q < k; ++q)
                    out[upd + q] = f;
            }
        }
        upd += k;
    }
};

/// Output position o of a repeat-count block: the element whose prefix range holds o.
struct ExFillFn {
    const int64_t* pre;
    int64_t n, c1, c2, f0, fs0, fs1;
    int64_t* out;
    CT_IX_FN void operator()(const int64_t o) const {
        int64_t lo = 0, hi = n;
        while (hi - lo > 1) {
            int64_t mid = (lo + hi) / 2;
            if (pre[mid] <= o)
                lo = mid;
            else
                hi = mid;
        }
        int64_t i2 = lo % c2, r = lo / c2;
        out[o] = f0 + (r / c1) * fs0 + (r % c1) * fs1 + i2;
    }
};

/// Total count and positions (in *cbuf, one buffer holding the block's whole output)
/// of a block. Repeat counts fill the output in parallel from the per-element prefixes.
template <int N, typename T, bool COUNT>
int64_t ex_scan(PartitionImpl<N>* p, ExScanFn<T, COUNT, false> f, int64_t n, char** cbuf) {
    char* pre = COUNT ? mx_alloc(n * sizeof(int64_t)) : nullptr;
    f.pre = reinterpret_cast<int64_t*>(pre);
    int64_t total = gx_scan(p, n, f);
    *cbuf = mx_alloc(total * sizeof(int64_t));
    if (COUNT) {
        gx_for(p, total,
               ExFillFn{f.pre, n, f.c1, f.c2, f.f0, f.fs0, f.fs1, reinterpret_cast<int64_t*>(*cbuf)});
        compute_fence(p);
        free_buffer(pre);
    } else {
        ExScanFn<T, COUNT, true> w{f.x,  f.xbase, f.xs0, f.xs1, f.xs2,   f.c1,
                                   f.c2, f.f0,    f.fs0, f.fs1, nullptr, reinterpret_cast<int64_t*>(*cbuf),
                                   nullptr};
        gx_scan(p, n, w);
    }
    return total;
}

/// x chare: positions of the block into st.cbuf, row counts to the coordinator.
template <int N, typename T>
void ex_scan_block(PartitionImpl<N>* p, ArrayDAGGroup* g, IndexState<N>& st, const IxCtx& c,
                   int node_id, const int me[3]) {
    const IxSpec& s = c.s;
    MxBlock xb;
    mx_block(s.L, c.L_l, me, xb);
    int sh = 3 - s.L.nd;
    int64_t vs[3] = {1, 1, 1};
    for (int d = 0; d < s.L.nd; ++d)
        vs[d + sh] = s.L.shape[d];
    int64_t nrows = xb.rows();
    st.rowpos.assign(nrows + 1, 0);
    char* drow = mx_alloc(nrows * sizeof(int64_t));
    int64_t* rowpos = reinterpret_cast<int64_t*>(drow);
#ifndef USE_KOKKOS
    rowpos = st.rowpos.data();
#endif
    const T* x = static_cast<const T*>(local_array(p, s.L.name)->device_data_ptr());
    int64_t f0 = (xb.k0[0] * vs[1] + xb.k0[1]) * vs[2] + xb.k0[2];
    int64_t total;
    if (s.mode == 1)
        total = ex_scan<N, T, true>(p,
                                    {x, xb.base, xb.str[0], xb.str[1], xb.str[2], xb.cnt[1],
                                     xb.cnt[2], f0, vs[1] * vs[2], vs[2], rowpos, nullptr, nullptr},
                                    xb.size(), &st.cbuf);
    else
        total = ex_scan<N, T, false>(p,
                                     {x, xb.base, xb.str[0], xb.str[1], xb.str[2], xb.cnt[1],
                                      xb.cnt[2], f0, vs[1] * vs[2], vs[2], rowpos, nullptr, nullptr},
                                     xb.size(), &st.cbuf);
#ifdef USE_KOKKOS
    mx_to_host(p, reinterpret_cast<char*>(st.rowpos.data()), drow, nrows * sizeof(int64_t));
#endif
    compute_fence(p);
    free_buffer(drow);
    st.rowpos[nrows] = total;
    std::vector<int64_t> words(2 * nrows);
    for (int64_t r = 0; r < nrows; ++r) {
        int64_t i0 = r / xb.cnt[1], i1 = r % xb.cnt[1];
        words[2 * r] = f0 + i0 * vs[1] * vs[2] + i1 * vs[2];
        words[2 * r + 1] = st.rowpos[r + 1] - st.rowpos[r];
    }
    auto buf = make_msg(IX_SEG, 2, me, words);
    send_msg(g, s.L.nd, c.drivers[0].data(), node_id, buf);
}

MxCtx ex_mx_ctx(const IxCtx& c) {
    MxCtx m;
    m.s = c.s;
    m.out_l = c.out_l;
    return m;
}

} // namespace

template <int N>
void ArrayDAGExecutorND<N>::execute_index_node(DAGNode* node) {
    Opcode op = node_index_op(node);
    if (is_mask_op(op)) {
        execute_mask_node(node);
        return;
    }
    if (op != Opcode::EXPAND) {
        gx_start(node);
        return;
    }
    auto* g = static_cast<ArrayDAGGroup*>(group);
    IndexState<N>& st = ix_state[node->id];
    auto ctx = std::make_shared<IxCtx>(make_ctx(node, g));
    st.ctx = ctx;
    const IxCtx& c = *ctx;
    const IxSpec& s = c.s;
    int me[3];
    self_index(partition->nd_index(), me);
    st.node = node;
    st.started = true;

    if (N == 1 && c.out_l.holds(me)) {
        ArrayDecomp<N> decomp = g->array_meta[s.out_name].template decomp<N>();
        ArrayRegion<N> mine = decomp.chare_region_local(partition->nd_index());
        std::array<int, N> gs{}, zeros{}, sizes{}, ones{};
        for (int d = 0; d < N; ++d) {
            gs[d] = (int)s.E[d];
            sizes[d] = mine.size(d);
            ones[d] = 1;
        }
        partition->ensure_array(ArrayRegion<N>(zeros, sizes, ones), gs, s.out_name, s.dt, decomp);
        st.out_expected = c.out_l.hi(0, me[0]) - c.out_l.lo(0, me[0]);
    }
    if (N == s.L.nd && !c.drivers.empty() && c.drivers[0] == std::array<int, 3>{me[0], me[1], me[2]})
        st.seg_left = (int)c.drivers.size();
    int64_t k0[3], cnt[3];
    if (N == s.L.nd && view_block(s.L, c.L_l, me, k0, cnt)) {
        st.wait_offsets = true;
        cn_dispatch(s.L.name >= 0 ? local_array(partition, s.L.name)->dtype : DType::INT64,
                    [&](auto tag) {
                        ex_scan_block<N, decltype(tag)>(partition, g, st, c, node->id, me);
                    });
    }
    index_progress(node->id);
}

template <int N>
void ArrayDAGExecutorND<N>::on_index_msg(int node_id, char* data, int64_t size) {
    int32_t kind;
    memcpy(&kind, data, sizeof(kind));
    if (kind == GX_FREE) {
#ifdef USE_NVIDIA
        GxHdr h;
        memcpy(&h, data, sizeof(h));
        gpu_ipc_free(reinterpret_cast<void*>(h.ipc_ptr));
#endif
        return;
    }
    ix_state[node_id].stash.emplace_back(data, data + size);
    index_progress(node_id);
}

/// Handle stashed messages once the chare has reached the node; finish the
/// node when every role of this chare is complete.
template <int N>
void ArrayDAGExecutorND<N>::index_progress(int node_id) {
    IndexState<N>& st = ix_state[node_id];
    if (!st.started)
        return;
    Opcode op = node_index_op(st.node);
    if (is_mask_op(op)) {
        mask_progress(node_id);
        return;
    }
    if (op != Opcode::EXPAND) {
        gx_progress(node_id);
        return;
    }
    auto* g = static_cast<ArrayDAGGroup*>(group);
    std::shared_ptr<void> keep = st.ctx;
    const IxCtx& c = *static_cast<const IxCtx*>(keep.get());
    std::vector<std::vector<char>> msgs;
    msgs.swap(st.stash);
    for (auto& m : msgs) {
        int32_t kind;
        memcpy(&kind, m.data(), sizeof(kind));
        switch (kind) {
        case IX_SEG:
            st.segs.push_back(std::move(m));
            if (--st.seg_left == 0)
                answer_segments(g, st, c, node_id);
            break;
        case IX_SEGOFF:
            mx_route<N, int64_t>(partition, g, st, ex_mx_ctx(c), node_id,
                                 reinterpret_cast<const int64_t*>(m.data() + sizeof(IxHdr)));
            st.wait_offsets = false;
            break;
        case MX_DATA:
            mx_place_data<N, int64_t>(partition, st, ex_mx_ctx(c), m);
            break;
        default:
            CkAbort("EXPAND: unexpected message kind %d", kind);
        }
    }
    if (st.seg_left == 0 && !st.wait_offsets && st.out_covered == st.out_expected) {
        compute_fence(partition);
        ix_state.erase(node_id);
#ifdef USE_NVIDIA
        auto* p = new ComputeDoneParam<N>{partition, node_id};
        CkCallback hcb(compute_done_cb<N>, p);
        hapiAddCallback(partition->compute_stream_raw, &hcb);
#else
        CT_KOKKOS_FENCE();
        node_finished(node_id);
#endif
    }
}

template <int N>
void PartitionImpl<N>::receive_index_msg(int node_id, int64_t size, char* data) {
    executor->on_index_msg(node_id, data, size);
}

template void ArrayDAGExecutorND<1>::execute_index_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_index_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_index_node(DAGNode*);

template void ArrayDAGExecutorND<1>::execute_mask_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_mask_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_mask_node(DAGNode*);

template void ArrayDAGExecutorND<1>::on_index_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<2>::on_index_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<3>::on_index_msg(int, char*, int64_t);

template void PartitionImpl<1>::receive_index_msg(int, int64_t, char*);
template void PartitionImpl<2>::receive_index_msg(int, int64_t, char*);
template void PartitionImpl<3>::receive_index_msg(int, int64_t, char*);
