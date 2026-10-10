#include "backend_internal.hpp"

#include <algorithm>
#include <cstring>
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
/// Only depends on the AST and array metadata (also used by the receive pass).
IxSpec parse_index(ASTNode* root, const MetaMap& meta) {
    IxSpec s;
    s.opc = static_cast<Opcode>(root->opcode);
    s.out_name = root->result_name;
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
    if (s.opc == Opcode::SCATTER)
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
    if (static_cast<Opcode>(root->opcode) != Opcode::SCATTER)
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

template <int N>
void PartitionImpl<N>::receive_index_msg(int node_id, int64_t size, char* data) {
    executor->on_index_msg(node_id, data, size);
}

template void ArrayDAGExecutorND<1>::execute_index_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_index_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_index_node(DAGNode*);

template void ArrayDAGExecutorND<1>::on_index_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<2>::on_index_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<3>::on_index_msg(int, char*, int64_t);

template void PartitionImpl<1>::receive_index_msg(int, int64_t, char*);
template void PartitionImpl<2>::receive_index_msg(int, int64_t, char*);
template void PartitionImpl<3>::receive_index_msg(int, int64_t, char*);
