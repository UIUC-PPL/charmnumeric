#include "backend_internal.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>

// GATHER / SCATTER / EXPAND: indexing with data-dependent communication.
//
// The frontend linearizes every index expression into one integer array L of
// flat offsets into the root of the indexed array, so a node is
//   GATHER   out[e] = src.flat[L[b] + sum_k s_k * stride_k]
//   SCATTER  dst.flat[L[b] + sum_k s_k * stride_k] = v[e]   (v array or scalar)
// over the iteration space e = (pre slice dims, b, post slice dims). The
// chares that hold L ("drivers") walk their block of L, work out which chare
// holds each element, and send one request list to every chare of the value
// holder (src, or v for SCATTER). Holders read the values and send one value
// list to every chare of the target (out, or dst), which writes them. Fixed
// all-to-all message counts tell each chare when its part is done, so a holder
// serves every request before the DAG can move past the node.
//
// EXPAND writes the flat position i of every element of a view count(x[i])
// times, in order (count = x != 0, or x itself): nonzero and repeat. Drivers
// send the counts of their row segments to one coordinator chare, which
// answers with each segment's exclusive prefix (the output position), and the
// drivers then send the positions to the output chares.
//
// Limits: the drivers do all address arithmetic, so the work is balanced by
// the layout of L, not of the data (a small L with large slice dims, e.g.
// take(a, idx, axis=1) with a tall a, runs on the few chares of L); every
// node costs |drivers| x |holders| + |holders| x |targets| messages; each
// element travels as 2-3 int64 words; EXPAND's scan is centralized. Index
// bounds of device index arrays are not checked: offsets outside the array
// are dropped (with a warning).

#ifdef USE_KOKKOS
#define CT_IX_FN KOKKOS_INLINE_FUNCTION
#else
#define CT_IX_FN inline
#endif

namespace {

enum IxKind : int32_t { IX_REQ = 0, IX_VAL = 1, IX_SEG = 2, IX_SEGOFF = 3 };

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
///   GATHER  [src, L, npre, npost, ndE, E..., strides...]
///   SCATTER [dst, L, v, npre, npost, ndE, E..., strides...]
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
    for (int k = 0; k < s.npre + s.npost; ++k)
        s.stride[k] = scalar_at(root, p + 3 + s.ndE + k);
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
    /// Owner chare (packed key) and offset in its tile of a flat element index.
    bool locate_flat(int64_t flat, int64_t& key, int64_t& local) const {
        if (flat < 0 || flat >= size())
            return false;
        int64_t x[3] = {0, 0, 0};
        for (int d = nd - 1; d >= 0; --d) {
            x[d] = flat % shape[d];
            flat /= shape[d];
        }
        locate(x, key, local);
        return true;
    }
    void locate(const int64_t* x, int64_t& key, int64_t& local) const {
        int ci[3] = {0, 0, 0};
        local = 0;
        for (int d = 0; d < nd; ++d) {
            ci[d] = (int)((x[d] + off[d]) / tile);
            local = local * (hi(d, ci[d]) - lo(d, ci[d])) + (x[d] - lo(d, ci[d]));
        }
        key = pack(ci);
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

/// Tile offsets of the elements of a block (row-major over the block).
std::vector<int64_t> block_offsets(const IxView& v, const IxLayout& l, const int ci[3],
                                   const int64_t k0[3], const int64_t cnt[3]) {
    int64_t n = cnt[0] * cnt[1] * cnt[2];
    std::vector<int64_t> offs(n);
    int64_t tsize[3] = {1, 1, 1};
    for (int d = 0; d < v.nd; ++d)
        tsize[d] = l.hi(d, ci[d]) - l.lo(d, ci[d]);
    for (int64_t e = 0; e < n; ++e) {
        int64_t rem = e, o = 0, mul = 1;
        for (int d = v.nd - 1; d >= 0; --d) {
            int64_t j = rem % cnt[d];
            rem /= cnt[d];
            int64_t x = v.start[d] + (k0[d] + j) * v.step[d];
            o += (x - l.lo(d, ci[d])) * mul;
            mul *= tsize[d];
        }
        offs[e] = o;
    }
    return offs;
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

template <typename T>
struct IxScatterFn {
    T* dst;
    const int64_t* off;
    const T* val;
    CT_IX_FN void operator()(const int64_t i) const { dst[off[i]] = val[i]; }
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

/// data[off[i]] = val[i] for a tile on the device (or host); offsets are distinct.
template <int N, typename T>
void scatter_tile(PartitionImpl<N>* p, CTArrayBase<N>* arr, const std::vector<int64_t>& off,
                  const std::vector<T>& val) {
    int64_t n = (int64_t)off.size();
    if (n == 0)
        return;
    T* dst = static_cast<T*>(arr->device_data_ptr());
#ifdef USE_KOKKOS
    Kokkos::View<int64_t*, DeviceSpace> d_off(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "ix_off"), n);
    Kokkos::View<T*, DeviceSpace> d_val(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "ix_vals"), n);
    ix_copy(p, d_off, HostUnmanaged<const int64_t>(off.data(), n));
    ix_copy(p, d_val, HostUnmanaged<const T>(val.data(), n));
    Kokkos::parallel_for("ct_ix_scatter", CT_COMPUTE_POLICY(p, n),
                         IxScatterFn<T>{dst, d_off.data(), d_val.data()});
    compute_fence(p);
#else
    (void)p;
    IxScatterFn<T> f{dst, off.data(), val.data()};
    for (int64_t i = 0; i < n; ++i)
        f(i);
#endif
}

/// Values of a tile at `off`, converted to int64 (EXPAND mode 0: x != 0).
template <int N>
std::vector<int64_t> read_ints(PartitionImpl<N>* p, CTArrayBase<N>* arr,
                               const std::vector<int64_t>& off, bool nonzero = false) {
    std::vector<int64_t> out(off.size());
    dispatch_bits(arr->dtype, [&](auto btag) {
        using B = decltype(btag);
        std::vector<B> v = gather_tile<N, B>(p, arr, off);
        cn_dispatch(arr->dtype, [&](auto tag) {
            using T = decltype(tag);
            if constexpr (sizeof(T) == sizeof(B)) {
                for (size_t i = 0; i < v.size(); ++i) {
                    T x;
                    memcpy(&x, &v[i], sizeof(T));
                    out[i] = nonzero ? (cn_nonzero(x) ? 1 : 0) : cn_cast<int64_t>(x);
                }
            }
        });
    });
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
        proxy_at<1>(g->partition_proxy_1, ChareIndex<1>{{ci[0]}})
            .receive_index_msg(node_id, size, buf.data());
        break;
    case 2:
        proxy_at<2>(g->partition_proxy_2, ChareIndex<2>{{ci[0], ci[1]}})
            .receive_index_msg(node_id, size, buf.data());
        break;
    case 3:
        proxy_at<3>(g->partition_proxy_3, ChareIndex<3>{{ci[0], ci[1], ci[2]}})
            .receive_index_msg(node_id, size, buf.data());
        break;
    }
}

/// map[key] with a one-entry cache: consecutive elements mostly share a chare.
struct PerChare {
    std::unordered_map<int64_t, std::vector<int64_t>>& m;
    int64_t last = -1;
    std::vector<int64_t>* v = nullptr;
    std::vector<int64_t>& operator[](int64_t key) {
        if (v == nullptr || key != last) {
            v = &m[key];
            last = key;
        }
        return *v;
    }
};

/// One message (possibly empty) to every chare of layout l.
void send_to_all(ArrayDAGGroup* g, const IxLayout& l, int node_id, int32_t kind, int32_t width,
                 std::unordered_map<int64_t, std::vector<int64_t>>& per_chare) {
    static const std::vector<int64_t> none;
    int64_t n = l.num_chares();
    for (int64_t k = 0; k < n; ++k) {
        int ci[3];
        l.chare(k, ci);
        auto it = per_chare.find(IxLayout::pack(ci));
        auto buf = make_msg(kind, width, nullptr, it == per_chare.end() ? none : it->second);
        send_msg(g, l.nd, ci, node_id, buf);
    }
    per_chare.clear();
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
    IxLayout data_l, L_l, V_l, out_l;
    std::vector<std::array<int, 3>> drivers; // chares holding a block of L
};

Opcode node_index_op(DAGNode* node) {
    for (ASTNode* root : node->ast->roots)
        if (is_index_op(static_cast<Opcode>(root->opcode)))
            return static_cast<Opcode>(root->opcode);
    return Opcode::NOOP;
}

IxCtx make_ctx(DAGNode* node, ArrayDAGGroup* g) {
    IxCtx c;
    for (ASTNode* root : node->ast->roots)
        if (is_index_op(static_cast<Opcode>(root->opcode))) {
            c.s = parse_index(root, g->array_meta);
            break;
        }
    c.L_l = layout_of(g, c.s.L.name, c.s.L.nd);
    c.drivers = view_chares(c.s.L, c.L_l);
    if (c.s.opc == Opcode::EXPAND) {
        c.out_l = layout_of(g, c.s.out_name, 1);
        return c;
    }
    c.data_l = layout_of(g, c.s.data.name, c.s.data.nd);
    if (c.s.opc == Opcode::GATHER)
        c.out_l = layout_of(g, c.s.out_name, c.s.ndE);
    else if (c.s.V.name >= 0)
        c.V_l = layout_of(g, c.s.V.name, c.s.V.nd);
    return c;
}

/// Calls f(eflat, e, flat) for every element of a driver block: e is the
/// iteration-space coordinate (row-major index eflat), flat the offset into
/// the data root.
template <typename F>
void for_each_element(const IxSpec& s, const int64_t k0[3], const int64_t cnt[3],
                      const std::vector<int64_t>& Lv, F&& f) {
    int nB = s.L.nd;
    int ns = s.npre + s.npost;
    int64_t sdim[3] = {1, 1, 1};
    for (int k = 0; k < s.npre; ++k)
        sdim[k] = s.E[k];
    for (int k = 0; k < s.npost; ++k)
        sdim[s.npre + k] = s.E[s.npre + nB + k];
    int64_t nslice = 1;
    for (int k = 0; k < ns; ++k)
        nslice *= sdim[k];
    int64_t estr[3] = {1, 1, 1};
    for (int d = s.ndE - 2; d >= 0; --d)
        estr[d] = estr[d + 1] * s.E[d + 1];

    int64_t e[3] = {0, 0, 0};
    for (int64_t ib = 0; ib < (int64_t)Lv.size(); ++ib) {
        int64_t rem = ib;
        for (int d = nB - 1; d >= 0; --d) {
            e[s.npre + d] = k0[d] + rem % cnt[d];
            rem /= cnt[d];
        }
        for (int64_t is = 0; is < nslice; ++is) {
            int64_t r = is, flat = Lv[ib];
            for (int k = ns - 1; k >= 0; --k) {
                int64_t sk = r % sdim[k];
                r /= sdim[k];
                e[k < s.npre ? k : nB + k] = sk;
                flat += sk * s.stride[k];
            }
            int64_t eflat = 0;
            for (int d = 0; d < s.ndE; ++d)
                eflat += e[d] * estr[d];
            f(eflat, e, flat);
        }
    }
}

void drive(ArrayDAGGroup* g, int node_id, const IxCtx& c, const int64_t k0[3],
           const int64_t cnt[3], const std::vector<int64_t>& Lv) {
    const IxSpec& s = c.s;
    std::unordered_map<int64_t, std::vector<int64_t>> per_map;
    PerChare per{per_map};
    int64_t dropped = 0;

    if (s.opc == Opcode::GATHER) {
        // (offset in the source tile, output element) to the source chares
        for_each_element(s, k0, cnt, Lv, [&](int64_t eflat, const int64_t*, int64_t flat) {
            int64_t key, local;
            if (!c.data_l.locate_flat(flat, key, local)) {
                ++dropped;
                return;
            }
            auto& w = per[key];
            w.push_back(local);
            w.push_back(eflat);
        });
        send_to_all(g, c.data_l, node_id, IX_REQ, 2, per_map);
    } else if (s.V.name >= 0) {
        // (offset in the value tile, target flat index, order) to the value chares
        for_each_element(s, k0, cnt, Lv, [&](int64_t eflat, const int64_t* e, int64_t flat) {
            if (flat < 0 || flat >= c.data_l.size()) {
                ++dropped;
                return;
            }
            int64_t x[3] = {0, 0, 0};
            for (int d = 0; d < s.V.nd; ++d)
                x[d] = s.V.start[d] + e[d] * s.V.step[d];
            int64_t key, local;
            c.V_l.locate(x, key, local);
            auto& w = per[key];
            w.push_back(local);
            w.push_back(flat);
            w.push_back(eflat);
        });
        send_to_all(g, c.V_l, node_id, IX_REQ, 3, per_map);
    } else {
        // scalar value: (target offset, order, value) straight to the targets
        int64_t bits = 0;
        cn_dispatch(s.dt, [&](auto tag) {
            using T = decltype(tag);
            if constexpr (sizeof(T) <= sizeof(int64_t))
                bits = to_bits<T>(cn_cast<T>(s.scalar));
            else
                CkAbort("SCATTER: dtype %d is not supported (elements over 8 bytes)", (int)s.dt);
        });
        for_each_element(s, k0, cnt, Lv, [&](int64_t eflat, const int64_t*, int64_t flat) {
            int64_t key, local;
            if (!c.data_l.locate_flat(flat, key, local)) {
                ++dropped;
                return;
            }
            auto& w = per[key];
            w.push_back(local);
            w.push_back(eflat);
            w.push_back(bits);
        });
        send_to_all(g, c.data_l, node_id, IX_VAL, 3, per_map);
    }
    if (dropped > 0)
        CkPrintf("charmnumeric: warning: %lld out-of-bounds indices ignored\n", (long long)dropped);
}

template <int N>
CTArrayBase<N>* local_array(PartitionImpl<N>* p, int name) {
    auto it = p->arrays.find(name);
    if (it == p->arrays.end())
        CkAbort("index op: array %d has no tile on this chare", name);
    return it->second;
}

/// Holder: answer one request list (values go to the per-target lists).
template <int N>
void serve_requests(PartitionImpl<N>* p, IndexState<N>& st, const IxCtx& c, const IxHdr& h,
                    const int64_t* w) {
    const IxSpec& s = c.s;
    bool gather = s.opc == Opcode::GATHER;
    int64_t n = h.count;
    std::vector<int64_t> offs(n);
    for (int64_t i = 0; i < n; ++i)
        offs[i] = w[i * h.width];
    CTArrayBase<N>* arr = local_array(p, gather ? s.data.name : s.V.name);
    int64_t estr[3] = {1, 1, 1};
    for (int d = s.ndE - 2; d >= 0; --d)
        estr[d] = estr[d + 1] * s.E[d + 1];
    PerChare outgoing{st.outgoing};
    dispatch_bits(arr->dtype, [&](auto tag) {
        using T = decltype(tag);
        std::vector<T> vals = gather_tile<N, T>(p, arr, offs);
        for (int64_t i = 0; i < n; ++i) {
            int64_t key, local;
            if (gather) {
                int64_t e[3] = {0, 0, 0}, rem = w[i * 2 + 1];
                for (int d = 0; d < s.ndE; ++d) {
                    e[d] = rem / estr[d];
                    rem %= estr[d];
                }
                c.out_l.locate(e, key, local);
                auto& o = outgoing[key];
                o.push_back(local);
                o.push_back(to_bits<T>(vals[i]));
            } else {
                c.data_l.locate_flat(w[i * 3 + 1], key, local);
                auto& o = outgoing[key];
                o.push_back(local);
                o.push_back(w[i * 3 + 2]);
                o.push_back(to_bits<T>(vals[i]));
            }
        }
    });
}

/// Target: write the collected (offset[, order], value) entries. Repeated
/// offsets (SCATTER) keep the value with the highest order, as in NumPy.
template <int N>
void apply_values(PartitionImpl<N>* p, IndexState<N>& st, const IxCtx& c) {
    const IxSpec& s = c.s;
    CTArrayBase<N>* arr = local_array(p, s.opc == Opcode::SCATTER ? s.data.name : s.out_name);
    int w = st.upd_width;
    int64_t n = w > 0 ? (int64_t)st.updates.size() / w : 0;
    std::vector<int64_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    const int64_t* u = st.updates.data();
    if (w == 3) {
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return u[a * 3] != u[b * 3] ? u[a * 3] < u[b * 3] : u[a * 3 + 1] < u[b * 3 + 1];
        });
        int64_t m = 0;
        for (int64_t i = 0; i < n; ++i)
            if (i + 1 == n || u[order[i] * 3] != u[order[i + 1] * 3])
                order[m++] = order[i];
        order.resize(m);
    }
    std::vector<int64_t> offs(order.size());
    for (size_t i = 0; i < order.size(); ++i)
        offs[i] = u[order[i] * w];
    dispatch_bits(arr->dtype, [&](auto tag) {
        using T = decltype(tag);
        std::vector<T> vals(order.size());
        for (size_t i = 0; i < order.size(); ++i)
            vals[i] = from_bits<T>(u[order[i] * w + w - 1]);
        scatter_tile<N, T>(p, arr, offs, vals);
    });
    std::vector<int64_t>().swap(st.updates);
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

/// EXPAND driver: write the flat position of each element count times,
/// starting at its segment's offset.
template <int N>
void emit_positions(ArrayDAGGroup* g, IndexState<N>& st, const IxCtx& c, int node_id,
                    const int64_t* offs) {
    std::unordered_map<int64_t, std::vector<int64_t>> per_map;
    PerChare per{per_map};
    int64_t nrows = (int64_t)st.seg_start.size();
    for (int64_t r = 0; r < nrows; ++r) {
        int64_t pos = offs[r];
        for (int64_t j = 0; j < st.row; ++j) {
            int64_t f = st.seg_start[r] + j;
            for (int64_t k = 0; k < st.counts[r * st.row + j]; ++k) {
                int64_t key, local;
                c.out_l.locate_flat(pos++, key, local);
                auto& w = per[key];
                w.push_back(local);
                w.push_back(f);
            }
        }
    }
    send_to_all(g, c.out_l, node_id, IX_VAL, 2, per_map);
    std::vector<int64_t>().swap(st.counts);
}

} // namespace

template <int N>
void ArrayDAGExecutorND<N>::execute_index_node(DAGNode* node) {
    if (is_mask_op(node_index_op(node))) {
        execute_mask_node(node);
        return;
    }
    auto* g = static_cast<ArrayDAGGroup*>(group);
    IxCtx c = make_ctx(node, g);
    const IxSpec& s = c.s;
    int me[3];
    self_index(partition->nd_index(), me);

    IndexState<N>& st = ix_state[node->id];
    st.node = node;
    st.started = true;

    // Target: allocate the result tile (GATHER/EXPAND); count the value lists to expect.
    bool scatter = s.opc == Opcode::SCATTER;
    const IxLayout& tl = scatter ? c.data_l : c.out_l;
    if (N == (scatter ? s.data.nd : s.ndE) && tl.holds(me)) {
        if (s.opc == Opcode::GATHER)
            st.val_left = (int)c.data_l.num_chares();
        else if (scatter && s.V.name >= 0)
            st.val_left = (int)c.V_l.num_chares();
        else
            st.val_left = (int)c.drivers.size();
        if (!scatter) {
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
        }
    }

    // Holder (GATHER source, SCATTER values): one request list per driver.
    if ((s.opc == Opcode::GATHER && N == s.data.nd && c.data_l.holds(me)) ||
        (scatter && s.V.name >= 0 && N == s.V.nd && c.V_l.holds(me))) {
        st.holder = true;
        st.req_left = (int)c.drivers.size();
    }

    int64_t k0[3], cnt[3];
    bool driver = N == s.L.nd && view_block(s.L, c.L_l, me, k0, cnt);
    if (s.opc == Opcode::EXPAND && N == s.L.nd && !c.drivers.empty() &&
        c.drivers[0] == std::array<int, 3>{me[0], me[1], me[2]})
        st.seg_left = (int)c.drivers.size();

    if (driver) {
        CTArrayBase<N>* L_arr = local_array(partition, s.L.name);
        auto offs = block_offsets(s.L, c.L_l, me, k0, cnt);
        if (s.opc == Opcode::EXPAND) {
            // Send the count of every row segment of the block to the coordinator.
            st.wait_offsets = true;
            st.counts = read_ints(partition, L_arr, offs, s.mode == 0);
            int nd = s.L.nd;
            int64_t vstride[3] = {1, 1, 1};
            for (int d = nd - 2; d >= 0; --d)
                vstride[d] = vstride[d + 1] * s.L.shape[d + 1];
            st.row = cnt[nd - 1];
            int64_t nrows = (int64_t)st.counts.size() / st.row;
            st.seg_start.assign(nrows, 0);
            std::vector<int64_t> words;
            for (int64_t r = 0; r < nrows; ++r) {
                int64_t rem = r, flat = k0[nd - 1];
                for (int d = nd - 2; d >= 0; --d) {
                    flat += (k0[d] + rem % cnt[d]) * vstride[d];
                    rem /= cnt[d];
                }
                int64_t sum = 0;
                for (int64_t j = 0; j < st.row; ++j) {
                    int64_t& k = st.counts[r * st.row + j];
                    k = std::max<int64_t>(k, 0);
                    sum += k;
                }
                st.seg_start[r] = flat;
                words.push_back(flat);
                words.push_back(sum);
            }
            auto buf = make_msg(IX_SEG, 2, me, words);
            send_msg(g, nd, c.drivers[0].data(), node->id, buf);
        } else {
            drive(g, node->id, c, k0, cnt, read_ints(partition, L_arr, offs));
        }
    }
    index_progress(node->id);
}

template <int N>
void ArrayDAGExecutorND<N>::on_index_msg(int node_id, char* data, int64_t size) {
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
    if (is_mask_op(node_index_op(st.node))) {
        mask_progress(node_id);
        return;
    }
    auto* g = static_cast<ArrayDAGGroup*>(group);
    IxCtx c = make_ctx(st.node, g);
    bool scatter = c.s.opc == Opcode::SCATTER;

    if (st.holder && st.req_left == 0 && !st.flushed) {
        st.flushed = true;
        send_to_all(g, scatter ? c.data_l : c.out_l, node_id, IX_VAL, scatter ? 3 : 2,
                    st.outgoing);
    }
    std::vector<std::vector<char>> msgs;
    msgs.swap(st.stash);
    for (auto& m : msgs) {
        IxHdr h;
        memcpy(&h, m.data(), sizeof(h));
        const auto* w = reinterpret_cast<const int64_t*>(m.data() + sizeof(h));
        switch (h.kind) {
        case IX_REQ:
            serve_requests(partition, st, c, h, w);
            if (--st.req_left == 0) {
                st.flushed = true;
                send_to_all(g, scatter ? c.data_l : c.out_l, node_id, IX_VAL, scatter ? 3 : 2,
                            st.outgoing);
            }
            break;
        case IX_VAL:
            if (h.count > 0) {
                st.upd_width = h.width;
                st.updates.insert(st.updates.end(), w, w + h.count * h.width);
            }
            if (--st.val_left == 0)
                apply_values(partition, st, c);
            break;
        case IX_SEG:
            st.segs.push_back(std::move(m));
            if (--st.seg_left == 0)
                answer_segments(g, st, c, node_id);
            break;
        case IX_SEGOFF:
            emit_positions(g, st, c, node_id, w);
            st.wait_offsets = false;
            break;
        }
    }
    if (st.req_left == 0 && st.val_left == 0 && st.seg_left == 0 && !st.wait_offsets) {
        ix_state.erase(node_id);
#ifdef USE_NVIDIA
        auto* p = new ComputeDoneParam<N>{partition, node_id, false};
        CkCallback hcb(compute_done_cb<N>, p);
        hapiAddCallback(partition->compute_stream_raw, &hcb);
#else
        CT_KOKKOS_FENCE();
        node_finished(node_id);
#endif
    }
}

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
        auto* p = new ComputeDoneParam<N>{partition, node_id, false};
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
