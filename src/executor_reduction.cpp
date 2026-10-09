#include "backend_internal.hpp"

#include <cmath>
#include <cstring>

// REDUCTION and BINCOUNT executors.
//
// Every chare of the input partition that owns part of the operand's view
// folds its block into per-output-element accumulators (on the device in
// Kokkos builds), then sends each piece of that small partial straight to the
// output chare that owns it (receive_reduce_partial). An output chare knows it
// is done when the partials it has folded in cover all input elements of its
// output region: covered == |region| * reduced extent. That needs no knowledge
// of the input's decomposition on the receiving side and no Charm++ reduction,
// so independent reductions in one DAG may run in any order on any chare.
//
// Full reductions are the special case where every axis is reduced; the result
// is a (1,) array on 1-D chare 0 (or a (1,...,1) array with keepdims).

#ifdef USE_KOKKOS
#define CT_RED_FN KOKKOS_INLINE_FUNCTION
#else
#define CT_RED_FN inline
#endif

/// Reduction kinds. Must match charmnumeric/reductions.py (_RED_*).
enum RedOp : int {
    RED_SUM = 0,
    RED_PROD = 1,
    RED_MAX = 2,
    RED_MIN = 3,
    RED_ARGMAX = 4,
    RED_ARGMIN = 5,
    RED_COUNT_NONZERO = 6,
    RED_ALL = 7,
    RED_ANY = 8,
    RED_MEAN = 9,
    RED_VAR = 10,
    RED_STD = 11,
    RED_NORM2 = 12,      // sqrt(sum(x*x))
    RED_NORM1 = 13,      // sum(|x|)
    RED_NORM_INF = 14,   // max(|x|)
    RED_NORM_NEGINF = 15 // min(|x|)
};

/// Accumulator for one output element. A is int64_t for exact integer
/// reductions and double otherwise.
template <typename A>
struct RedAcc {
    A v;         // sum / product / extremum / running mean (VAR, STD)
    double m2;   // sum of squared deviations from the mean (VAR, STD)
    int64_t idx; // index of the extremum (ARGMAX, ARGMIN)
    int64_t n;   // number of input elements folded in
};

/// Wire header of a receive_reduce_partial message, followed by `count`
/// RedAcc<A> entries in row-major order over [start, stop).
struct RedMsgHeader {
    int out_nd;
    int pad;
    int start[3];
    int stop[3];
    int64_t depth; // input elements folded into each entry
    int64_t count;
};

static bool red_acc_is_int(int op, DType in_dt) {
    if (in_dt != DType::INT32 && in_dt != DType::INT64)
        return false;
    return op != RED_MEAN && op != RED_VAR && op != RED_STD && op != RED_NORM2;
}

template <typename A>
CT_RED_FN bool red_isnan(A x) {
    return x != x; // always false for integers
}

template <typename A>
CT_RED_FN A red_abs(A x) {
    return x < A(0) ? -x : x;
}

template <typename A>
CT_RED_FN void red_init(int op, RedAcc<A>& a) {
    a.v = (op == RED_PROD) ? A(1) : A(0);
    a.m2 = 0.0;
    a.idx = 0;
    a.n = 0;
}

/// True if (x, ix) should replace the current extremum (y, iy). NaN wins over
/// numbers (NumPy propagates NaN through max/min and argmax/argmin returns the
/// first NaN); equal values keep the lower index (first occurrence).
template <typename A>
CT_RED_FN bool red_prefer(bool max_like, A x, int64_t ix, A y, int64_t iy) {
    bool nx = red_isnan(x), ny = red_isnan(y);
    if (nx || ny)
        return (nx && ny) ? ix < iy : nx;
    if (x == y)
        return ix < iy;
    return max_like ? (x > y) : (x < y);
}

template <typename A>
CT_RED_FN void red_accumulate(int op, RedAcc<A>& a, A x, int64_t ix) {
    switch (op) {
    case RED_SUM:
    case RED_MEAN:
        a.v += x;
        break;
    case RED_NORM2:
        a.v += x * x;
        break;
    case RED_NORM1:
        a.v += red_abs(x);
        break;
    case RED_PROD:
        a.v *= x;
        break;
    case RED_COUNT_NONZERO:
    case RED_ALL:
    case RED_ANY:
        a.v += (x != A(0)) ? A(1) : A(0);
        break;
    case RED_MAX:
    case RED_MIN:
    case RED_ARGMAX:
    case RED_ARGMIN:
    case RED_NORM_INF:
    case RED_NORM_NEGINF: {
        bool max_like = (op == RED_MAX || op == RED_ARGMAX || op == RED_NORM_INF);
        if (op == RED_NORM_INF || op == RED_NORM_NEGINF)
            x = red_abs(x);
        if (a.n == 0 || red_prefer(max_like, x, ix, a.v, a.idx)) {
            a.v = x;
            a.idx = ix;
        }
        break;
    }
    case RED_VAR:
    case RED_STD: {
        // Welford update
        double xd = static_cast<double>(x);
        double mean = static_cast<double>(a.v);
        double delta = xd - mean;
        mean += delta / static_cast<double>(a.n + 1);
        a.m2 += delta * (xd - mean);
        a.v = static_cast<A>(mean);
        break;
    }
    }
    a.n += 1;
}

template <typename A>
CT_RED_FN void red_join(int op, RedAcc<A>& a, const RedAcc<A>& b) {
    if (b.n == 0)
        return;
    if (a.n == 0) {
        a = b;
        return;
    }
    switch (op) {
    case RED_PROD:
        a.v *= b.v;
        break;
    case RED_MAX:
    case RED_MIN:
    case RED_ARGMAX:
    case RED_ARGMIN:
    case RED_NORM_INF:
    case RED_NORM_NEGINF: {
        bool max_like = (op == RED_MAX || op == RED_ARGMAX || op == RED_NORM_INF);
        if (red_prefer(max_like, b.v, b.idx, a.v, a.idx)) {
            a.v = b.v;
            a.idx = b.idx;
        }
        break;
    }
    case RED_VAR:
    case RED_STD: {
        // Chan et al. pairwise combination
        double na = static_cast<double>(a.n), nb = static_cast<double>(b.n);
        double ma = static_cast<double>(a.v), mb = static_cast<double>(b.v);
        double delta = mb - ma;
        double nt = na + nb;
        a.v = static_cast<A>(ma + delta * nb / nt);
        a.m2 = a.m2 + b.m2 + delta * delta * na * nb / nt;
        break;
    }
    default: // sums and counts
        a.v += b.v;
        break;
    }
    a.n += b.n;
}

/// Final value of one output element.
template <typename A, typename T>
static T red_finalize(int op, const RedAcc<A>& a, int64_t extent, int ddof) {
    switch (op) {
    case RED_ARGMAX:
    case RED_ARGMIN:
        return static_cast<T>(a.idx);
    case RED_ALL:
        return static_cast<T>(static_cast<int64_t>(a.v) == extent ? 1 : 0);
    case RED_ANY:
        return static_cast<T>(a.v > A(0) ? 1 : 0);
    case RED_MEAN:
        return static_cast<T>(static_cast<double>(a.v) / static_cast<double>(extent));
    case RED_NORM2:
        return static_cast<T>(std::sqrt(static_cast<double>(a.v)));
    case RED_VAR:
    case RED_STD: {
        double dof = static_cast<double>(std::max<int64_t>(a.n - ddof, 0));
        double var = a.m2 / dof;
        return static_cast<T>(op == RED_STD ? std::sqrt(var) : var);
    }
    default:
        return static_cast<T>(a.v);
    }
}

/// Geometry of one chare's block of a (possibly strided) view: which elements
/// of the local tile it covers and where they sit in the view.
struct RedGeom {
    int nd;
    int red_mask;       // bit d set: dim d is reduced
    int b0[3];          // first block element along each dim (local tile coords)
    int bstep[3];       // step between block elements (the view's step)
    int bn[3];          // block extent
    int vb0[3];         // view index of the first block element
    int64_t bstride[3]; // row-major strides of the local tile
    int64_t vstride[3]; // row-major strides of the view (flat indices)
};

/// Every axis reduced: fold the whole block into one accumulator.
template <typename T, typename A>
struct RedFullFunctor {
    using value_type = RedAcc<A>;
    const T* data;
    RedGeom g;
    int op;

    CT_RED_FN void operator()(const int e, value_type& acc) const {
        int64_t rem = e, off = 0, flat = 0;
        for (int d = g.nd - 1; d >= 0; --d) {
            int c = static_cast<int>(rem % g.bn[d]);
            rem /= g.bn[d];
            off += static_cast<int64_t>(g.b0[d] + c * g.bstep[d]) * g.bstride[d];
            flat += static_cast<int64_t>(g.vb0[d] + c) * g.vstride[d];
        }
        red_accumulate<A>(op, acc, static_cast<A>(data[off]), flat);
    }
    CT_RED_FN void join(value_type& dst, const value_type& src) const {
        red_join<A>(op, dst, src);
    }
    CT_RED_FN void init(value_type& v) const { red_init<A>(op, v); }
};

/// Some axes reduced: one work item per kept-coordinate, looping over the
/// reduced axes of the block. arg_axis >= 0 reports the position along that
/// axis (ARGMAX/ARGMIN with axis=k), otherwise the flat view index.
template <typename T, typename A>
struct RedPartialFunctor {
    const T* data;
    RedAcc<A>* out;
    RedGeom g;
    int op;
    int arg_axis;

    CT_RED_FN void operator()(const int p) const {
        int c[3] = {0, 0, 0};
        int64_t rem = p;
        int64_t q_total = 1;
        for (int d = g.nd - 1; d >= 0; --d) {
            if ((g.red_mask >> d) & 1) {
                q_total *= g.bn[d];
            } else {
                c[d] = static_cast<int>(rem % g.bn[d]);
                rem /= g.bn[d];
            }
        }
        RedAcc<A> acc;
        red_init<A>(op, acc);
        for (int64_t q = 0; q < q_total; ++q) {
            int64_t r = q;
            for (int d = g.nd - 1; d >= 0; --d) {
                if ((g.red_mask >> d) & 1) {
                    c[d] = static_cast<int>(r % g.bn[d]);
                    r /= g.bn[d];
                }
            }
            int64_t off = 0, flat = 0;
            for (int d = 0; d < g.nd; ++d) {
                off += static_cast<int64_t>(g.b0[d] + c[d] * g.bstep[d]) * g.bstride[d];
                flat += static_cast<int64_t>(g.vb0[d] + c[d]) * g.vstride[d];
            }
            int64_t ix = arg_axis >= 0 ? static_cast<int64_t>(g.vb0[arg_axis] + c[arg_axis]) : flat;
            red_accumulate<A>(op, acc, static_cast<A>(data[off]), ix);
        }
        out[p] = acc;
    }
};

/// Local histogram of a 1-D block (BINCOUNT). w == nullptr counts ones.
template <typename T, typename W, typename A>
struct BincountFunctor {
    const T* x;
    const W* w;
    A* hist;
    int b0;
    int bstep;
    int64_t length;

    CT_RED_FN void operator()(const int e) const {
        int64_t off = static_cast<int64_t>(b0) + static_cast<int64_t>(e) * bstep;
        int64_t bin = static_cast<int64_t>(x[off]);
        if (bin < 0 || bin >= length)
            return;
        A inc = w ? static_cast<A>(w[off]) : A(1);
#ifdef USE_KOKKOS
        Kokkos::atomic_add(&hist[bin], inc);
#else
        hist[bin] += inc;
#endif
    }
};

// ---- Node description shared by the send and receive sides ----

namespace {

struct RedSpec {
    Opcode opc;
    int op = RED_SUM;
    int red_mask = 0;
    bool keepdims = false;
    int ddof = 0;
    DType in_dt = DType::FLOAT64;
    DType out_dt = DType::FLOAT64;
    bool acc_int = false;
    int in_name = -1;
    int in_nd = 1;
    std::array<int, 3> view_shape = {1, 1, 1};
    Region* in_region = nullptr;
    int w_name = -1; // BINCOUNT weights (-1: none)
    Region* w_region = nullptr;
    DType w_dt = DType::FLOAT64;
    int64_t length = 0; // BINCOUNT output length
    int out_name = -1;
    int out_nd = 1;
    std::array<int, 3> out_shape = {1, 1, 1};
    int64_t extent = 1; // input elements folded into each output element
};

template <int N>
std::array<int, 3> region_shape(Region* r, const std::array<int, 3>& fallback) {
    std::array<int, 3> s = fallback;
    if (r != nullptr && !r->is_global) {
        auto* ar = static_cast<ArrayRegion<N>*>(r);
        for (int d = 0; d < N; ++d)
            s[d] = ar->size(d);
    }
    return s;
}

std::array<int, 3> operand_shape(ASTNode* root, int op_idx, int nd,
                                 const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& meta) {
    std::array<int, 3> fallback = {1, 1, 1};
    auto it = meta.find(root->operands[op_idx]->result_name);
    if (it != meta.end())
        fallback = it->second.global_shape;
    Region* r = root->get_operand_region(op_idx);
    switch (nd) {
    case 1:
        return region_shape<1>(r, fallback);
    case 2:
        return region_shape<2>(r, fallback);
    case 3:
        return region_shape<3>(r, fallback);
    }
    return fallback;
}

} // namespace

/// Decode a REDUCTION/BINCOUNT root. Also used by the receive pass to size the
/// result array, so it must only depend on the AST and array metadata.
static RedSpec parse_reduction(ASTNode* root,
                               const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& meta) {
    RedSpec s;
    s.opc = static_cast<Opcode>(root->opcode);
    s.out_name = root->result_name;
    s.out_dt = root->dtype;
    ASTNode* x = root->operands[0];
    s.in_name = x->result_name;
    s.in_nd = x->ndims;
    s.in_region = root->get_operand_region(0);
    s.view_shape = operand_shape(root, 0, s.in_nd, meta);

    if (s.opc == Opcode::REDUCTION) {
        // operands: x, op, red_mask, keepdims, ddof, in_dtype
        s.op = static_cast<int>(root->operands[1]->scalar);
        s.red_mask = static_cast<int>(root->operands[2]->scalar);
        s.keepdims = root->operands[3]->scalar != 0.0;
        s.ddof = static_cast<int>(root->operands[4]->scalar);
        s.in_dt = static_cast<DType>(static_cast<int>(root->operands[5]->scalar));
        s.acc_int = red_acc_is_int(s.op, s.in_dt);
        int m = 0;
        s.extent = 1;
        for (int d = 0; d < s.in_nd; ++d) {
            if ((s.red_mask >> d) & 1) {
                s.extent *= s.view_shape[d];
                if (s.keepdims)
                    s.out_shape[m++] = 1;
            } else {
                s.out_shape[m++] = s.view_shape[d];
            }
        }
        if (m == 0) {
            s.out_shape = {1, 1, 1};
            m = 1;
        }
        s.out_nd = m;
    } else {
        // operands: x, [weights,] length, x_dtype, w_dtype
        int i = 1;
        if (!root->operands[1]->is_scalar) {
            s.w_name = root->operands[1]->result_name;
            s.w_region = root->get_operand_region(1);
            i = 2;
        }
        s.op = RED_SUM;
        s.length = static_cast<int64_t>(root->operands[i]->scalar);
        s.in_dt = static_cast<DType>(static_cast<int>(root->operands[i + 1]->scalar));
        if (s.w_name >= 0)
            s.w_dt = static_cast<DType>(static_cast<int>(root->operands[i + 2]->scalar));
        s.acc_int = (s.w_name < 0);
        s.extent = 1;
        for (int d = 0; d < s.in_nd; ++d)
            s.extent *= s.view_shape[d];
        s.out_nd = 1;
        s.out_shape = {static_cast<int>(s.length), 1, 1};
    }
    return s;
}

/// Result shape/ndims of a reduction node (used by the receive pass).
int reduction_result_meta(ASTNode* root,
                          const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& meta,
                          std::array<int, 3>& shape_out) {
    RedSpec s = parse_reduction(root, meta);
    shape_out = {0, 0, 0};
    for (int d = 0; d < s.out_nd; ++d)
        shape_out[d] = s.out_shape[d];
    return s.out_nd;
}

// ---- Send side ----

/// Split a partial over output box [start, stop) by the output decomposition
/// and send each piece to its owner.
template <int M, typename A>
static void send_partial_pieces(ArrayDAGGroup* dag_group, int node_id, const RedSpec& s,
                                const std::array<int, 3>& start, const std::array<int, 3>& stop,
                                const RedAcc<A>* part, int64_t depth) {
    ArrayDecomp<M> out_decomp = dag_group->array_meta[s.out_name].template decomp<M>();
    std::array<int, M> bs{}, be{}, one{};
    for (int d = 0; d < M; ++d) {
        bs[d] = start[d];
        be[d] = stop[d];
        one[d] = 1;
    }
    ArrayRegion<M> box(bs, be, one);
    std::array<int64_t, M> box_stride{};
    box_stride[M - 1] = 1;
    for (int d = M - 2; d >= 0; --d)
        box_stride[d] = box_stride[d + 1] * box.size(d + 1);

    auto pieces = decompose(out_decomp.to_global(box), out_decomp);
    for (auto& [ci, piece_global] : pieces) {
        ArrayRegion<M> piece = out_decomp.to_local(piece_global);
        int64_t count = piece.size();
        if (count <= 0)
            continue;
        int64_t bytes = sizeof(RedMsgHeader) + count * sizeof(RedAcc<A>);
        std::vector<char> buf(bytes);
        RedMsgHeader h = {};
        h.out_nd = M;
        for (int d = 0; d < M; ++d) {
            h.start[d] = piece.start[d];
            h.stop[d] = piece.stop[d];
        }
        h.depth = depth;
        h.count = count;
        memcpy(buf.data(), &h, sizeof(h));
        auto* dst = reinterpret_cast<RedAcc<A>*>(buf.data() + sizeof(h));

        std::array<int, M> idx{};
        for (int64_t k = 0; k < count; ++k) {
            int64_t src = 0;
            for (int d = 0; d < M; ++d)
                src += static_cast<int64_t>(piece.start[d] + idx[d] - box.start[d]) * box_stride[d];
            dst[k] = part[src];
            for (int d = M - 1; d >= 0; --d) {
                if (++idx[d] < piece.size(d))
                    break;
                idx[d] = 0;
            }
        }

        if constexpr (M == 1)
            proxy_at<1>(dag_group->partition_proxy_1, ci).receive_reduce_partial(node_id, bytes, buf.data());
        else if constexpr (M == 2)
            proxy_at<2>(dag_group->partition_proxy_2, ci).receive_reduce_partial(node_id, bytes, buf.data());
        else
            proxy_at<3>(dag_group->partition_proxy_3, ci).receive_reduce_partial(node_id, bytes, buf.data());
    }
}

template <typename A>
static void send_partial(ArrayDAGGroup* dag_group, int node_id, const RedSpec& s,
                         const std::array<int, 3>& start, const std::array<int, 3>& stop,
                         const RedAcc<A>* part, int64_t depth) {
    switch (s.out_nd) {
    case 1:
        send_partial_pieces<1, A>(dag_group, node_id, s, start, stop, part, depth);
        break;
    case 2:
        send_partial_pieces<2, A>(dag_group, node_id, s, start, stop, part, depth);
        break;
    case 3:
        send_partial_pieces<3, A>(dag_group, node_id, s, start, stop, part, depth);
        break;
    }
}

/// Locate this chare's block of the view `vr` (array coordinates) inside
/// the local tile of `arr`. Returns false if the chare holds none of it.
template <int N>
static bool block_geometry(PartitionImpl<N>* partition, CTArrayBase<N>* arr, Region* vr_base,
                           int red_mask, RedGeom& g) {
    ArrayRegion<N> view;
    if (vr_base == nullptr || vr_base->is_global) {
        std::array<int, N> s{}, e{}, st{};
        for (int d = 0; d < N; ++d) {
            e[d] = arr->global_shape[d];
            st[d] = 1;
        }
        view = ArrayRegion<N>(s, e, st);
    } else {
        view = *static_cast<ArrayRegion<N>*>(vr_base);
    }
    ArrayRegion<N> tile = arr->decomp.chare_region_local(partition->nd_index());
    for (int d = 0; d < N; ++d)
        if (tile.size(d) <= 0)
            return false;
    auto [blk, ok] = intersect(view, tile);
    if (!ok || blk.size() <= 0)
        return false;

    g.nd = N;
    g.red_mask = red_mask;
    int64_t bs = 1, vs = 1;
    for (int d = N - 1; d >= 0; --d) {
        g.b0[d] = blk.start[d] - tile.start[d];
        g.bstep[d] = view.step[d];
        g.bn[d] = blk.size(d);
        g.vb0[d] = (blk.start[d] - view.start[d]) / view.step[d];
        g.bstride[d] = bs;
        g.vstride[d] = vs;
        bs *= arr->region.size(d);
        vs *= view.size(d);
    }
    for (int d = N; d < 3; ++d) {
        g.b0[d] = 0;
        g.bstep[d] = 1;
        g.bn[d] = 1;
        g.vb0[d] = 0;
        g.bstride[d] = 0;
        g.vstride[d] = 0;
    }
    return true;
}

template <int N, typename T, typename A>
static void reduce_block_and_send(PartitionImpl<N>* partition, ArrayDAGGroup* dag_group,
                                  int node_id, const RedSpec& s, CTArrayBase<N>* arr,
                                  const RedGeom& g) {
    const T* data = static_cast<const T*>(arr->device_data_ptr());
    int all_mask = (1 << N) - 1;
    std::array<int, 3> start = {0, 0, 0}, stop = {1, 1, 1};
    int64_t depth = 1;
    int m = 0;
    for (int d = 0; d < N; ++d) {
        if ((s.red_mask >> d) & 1) {
            depth *= g.bn[d];
            if (s.keepdims) {
                start[m] = 0;
                stop[m] = 1;
                ++m;
            }
        } else {
            start[m] = g.vb0[d];
            stop[m] = g.vb0[d] + g.bn[d];
            ++m;
        }
    }

    if ((s.red_mask & all_mask) == all_mask) {
        int64_t total = depth;
        RedFullFunctor<T, A> f{data, g, s.op};
        RedAcc<A> result;
#ifdef USE_KOKKOS
        Kokkos::parallel_reduce("ct_reduce_full", CT_COMPUTE_POLICY(partition, total), f, result);
#else
        f.init(result);
        for (int e = 0; e < total; ++e)
            f(e, result);
#endif
        send_partial<A>(dag_group, node_id, s, start, stop, &result, depth);
        return;
    }

    int arg_axis = -1;
    if (s.op == RED_ARGMAX || s.op == RED_ARGMIN) {
        for (int d = 0; d < N; ++d)
            if ((s.red_mask >> d) & 1)
                arg_axis = d;
    }
    int64_t P = 1;
    for (int d = 0; d < N; ++d)
        if (!((s.red_mask >> d) & 1))
            P *= g.bn[d];

    std::vector<RedAcc<A>> part(P);
#ifdef USE_KOKKOS
    Kokkos::View<RedAcc<A>*, DeviceSpace> d_part("ct_reduce_partial", P);
    RedPartialFunctor<T, A> f{data, d_part.data(), g, s.op, arg_axis};
    Kokkos::parallel_for("ct_reduce_axis", CT_COMPUTE_POLICY(partition, P), f);
    Kokkos::View<RedAcc<A>*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_part(part.data(), P);
#ifdef USE_NVIDIA
    Kokkos::deep_copy(partition->compute_exec, h_part, d_part);
    partition->compute_exec.fence();
#else
    Kokkos::deep_copy(h_part, d_part);
#endif
#else
    RedPartialFunctor<T, A> f{data, part.data(), g, s.op, arg_axis};
    for (int p = 0; p < P; ++p)
        f(p);
#endif
    send_partial<A>(dag_group, node_id, s, start, stop, part.data(), depth);
}

template <int N, typename T, typename W, typename A>
static void bincount_block_and_send(PartitionImpl<N>* partition, ArrayDAGGroup* dag_group,
                                    int node_id, const RedSpec& s, CTArrayBase<N>* x_arr,
                                    CTArrayBase<N>* w_arr, const RedGeom& g) {
    const T* x = static_cast<const T*>(x_arr->device_data_ptr());
    const W* w = w_arr ? static_cast<const W*>(w_arr->device_data_ptr()) : nullptr;
    int64_t len = s.length;
    int64_t nelem = g.bn[0];
    std::vector<A> hist(len, A(0));
#ifdef USE_KOKKOS
    Kokkos::View<A*, DeviceSpace> d_hist("ct_bincount", len);
    BincountFunctor<T, W, A> f{x, w, d_hist.data(), g.b0[0], g.bstep[0], len};
    Kokkos::parallel_for("ct_bincount", CT_COMPUTE_POLICY(partition, nelem), f);
    Kokkos::View<A*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_hist(hist.data(), len);
#ifdef USE_NVIDIA
    Kokkos::deep_copy(partition->compute_exec, h_hist, d_hist);
    partition->compute_exec.fence();
#else
    Kokkos::deep_copy(h_hist, d_hist);
#endif
#else
    BincountFunctor<T, W, A> f{x, w, hist.data(), g.b0[0], g.bstep[0], len};
    for (int e = 0; e < nelem; ++e)
        f(e);
#endif
    std::vector<RedAcc<A>> part(len);
    for (int64_t i = 0; i < len; ++i)
        part[i] = {hist[i], 0.0, 0, nelem};
    send_partial<A>(dag_group, node_id, s, {0, 0, 0}, {static_cast<int>(len), 1, 1}, part.data(),
                    nelem);
}

template <int N, typename T, typename A>
static void dispatch_bincount_weights(PartitionImpl<N>* partition, ArrayDAGGroup* dag_group,
                                      int node_id, const RedSpec& s, CTArrayBase<N>* x_arr,
                                      CTArrayBase<N>* w_arr, const RedGeom& g) {
    if (w_arr == nullptr) {
        bincount_block_and_send<N, T, T, A>(partition, dag_group, node_id, s, x_arr, nullptr, g);
        return;
    }
    switch (w_arr->dtype) {
    case DType::FLOAT32:
        bincount_block_and_send<N, T, float, A>(partition, dag_group, node_id, s, x_arr, w_arr, g);
        break;
    case DType::FLOAT64:
        bincount_block_and_send<N, T, double, A>(partition, dag_group, node_id, s, x_arr, w_arr, g);
        break;
    case DType::INT32:
        bincount_block_and_send<N, T, int32_t, A>(partition, dag_group, node_id, s, x_arr, w_arr, g);
        break;
    case DType::INT64:
        bincount_block_and_send<N, T, int64_t, A>(partition, dag_group, node_id, s, x_arr, w_arr, g);
        break;
    }
}

template <int N, typename A>
static void dispatch_send_side(PartitionImpl<N>* partition, ArrayDAGGroup* dag_group, int node_id,
                               const RedSpec& s, CTArrayBase<N>* arr, CTArrayBase<N>* w_arr,
                               const RedGeom& g) {
    if (s.opc == Opcode::BINCOUNT) {
        if constexpr (N == 1) {
            if (arr->dtype == DType::INT32)
                dispatch_bincount_weights<N, int32_t, A>(partition, dag_group, node_id, s, arr, w_arr, g);
            else if (arr->dtype == DType::INT64)
                dispatch_bincount_weights<N, int64_t, A>(partition, dag_group, node_id, s, arr, w_arr, g);
            else
                CkAbort("bincount: input must be int32 or int64");
        }
        return;
    }
    switch (arr->dtype) {
    case DType::FLOAT32:
        reduce_block_and_send<N, float, A>(partition, dag_group, node_id, s, arr, g);
        break;
    case DType::FLOAT64:
        reduce_block_and_send<N, double, A>(partition, dag_group, node_id, s, arr, g);
        break;
    case DType::INT32:
        reduce_block_and_send<N, int32_t, A>(partition, dag_group, node_id, s, arr, g);
        break;
    case DType::INT64:
        reduce_block_and_send<N, int64_t, A>(partition, dag_group, node_id, s, arr, g);
        break;
    }
}

// ---- Receive side ----

template <int N, typename A>
static void fold_partial(ReductionState<N>& st, int op, const char* data) {
    RedMsgHeader h;
    memcpy(&h, data, sizeof(h));
    const auto* src = reinterpret_cast<const RedAcc<A>*>(data + sizeof(h));
    auto* acc = reinterpret_cast<RedAcc<A>*>(st.acc.data());

    std::array<int64_t, N> mine_stride{};
    mine_stride[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        mine_stride[d] = mine_stride[d + 1] * st.mine.size(d + 1);

    std::array<int, N> idx{};
    for (int64_t k = 0; k < h.count; ++k) {
        int64_t dst = 0;
        for (int d = 0; d < N; ++d)
            dst += static_cast<int64_t>(h.start[d] + idx[d] - st.mine.start[d]) * mine_stride[d];
        RedAcc<A> b;
        memcpy(&b, src + k, sizeof(b));
        red_join<A>(op, acc[dst], b);
        for (int d = N - 1; d >= 0; --d) {
            if (++idx[d] < h.stop[d] - h.start[d])
                break;
            idx[d] = 0;
        }
    }
    st.covered += h.count * h.depth;
}

template <int N, typename A, typename T>
static void write_result(PartitionImpl<N>* partition, ReductionState<N>& st, const RedSpec& s) {
    auto* arr = partition->arrays[s.out_name];
    T* out = static_cast<T*>(arr->data_ptr());
    const auto* acc = reinterpret_cast<const RedAcc<A>*>(st.acc.data());
    int64_t n = st.mine.size();
    for (int64_t i = 0; i < n; ++i)
        out[i] = red_finalize<A, T>(s.op, acc[i], s.extent, s.ddof);
    arr->copyToDevice();
}

template <int N, typename A>
static void write_result_dispatch(PartitionImpl<N>* partition, ReductionState<N>& st,
                                  const RedSpec& s) {
    switch (s.out_dt) {
    case DType::FLOAT32:
        write_result<N, A, float>(partition, st, s);
        break;
    case DType::FLOAT64:
        write_result<N, A, double>(partition, st, s);
        break;
    case DType::INT32:
        write_result<N, A, int32_t>(partition, st, s);
        break;
    case DType::INT64:
        write_result<N, A, int64_t>(partition, st, s);
        break;
    }
}

/// Fold stashed partials and finish the node once every input element of
/// this chare's output region has been accounted for.
template <int N>
static bool reduction_try_finish(PartitionImpl<N>* partition, ReductionState<N>& st,
                                 const RedSpec& s) {
    for (auto& msg : st.stash) {
        if (s.acc_int)
            fold_partial<N, int64_t>(st, s.op, msg.data());
        else
            fold_partial<N, double>(st, s.op, msg.data());
    }
    st.stash.clear();
    if (st.covered < st.expected)
        return false;
    if (s.acc_int)
        write_result_dispatch<N, int64_t>(partition, st, s);
    else
        write_result_dispatch<N, double>(partition, st, s);
    return true;
}

static RedSpec spec_of(DAGNode* node, ArrayDAGGroup* dag_group) {
    for (ASTNode* root : node->ast->roots)
        if (is_reduction_op(static_cast<Opcode>(root->opcode)))
            return parse_reduction(root, dag_group->array_meta);
    CkAbort("execute_reduction_node: node %d has no reduction root", node->id);
    return RedSpec{};
}

template <int N>
void ArrayDAGExecutorND<N>::execute_reduction_node(DAGNode* node) {
    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    RedSpec s = spec_of(node, dag_group);
    auto nd_idx = partition->nd_index();

    // Send side: fold this chare's block of the operand and ship the partial.
    if (N == s.in_nd) {
        auto it = partition->arrays.find(s.in_name);
        RedGeom g;
        if (it != partition->arrays.end() &&
            block_geometry<N>(partition, it->second, s.in_region, s.red_mask, g)) {
            CTArrayBase<N>* w_arr = nullptr;
            if (s.w_name >= 0) {
                auto w_it = partition->arrays.find(s.w_name);
                if (w_it == partition->arrays.end())
                    CkAbort("bincount: weights array %d is not on the chare that holds x", s.w_name);
                w_arr = w_it->second;
                bool aligned = w_arr->decomp.tile == it->second->decomp.tile &&
                               w_arr->decomp.offset == it->second->decomp.offset &&
                               w_arr->region.size() == it->second->region.size();
                if (s.w_region && s.in_region && !s.w_region->is_global && !s.in_region->is_global) {
                    auto* wr = static_cast<ArrayRegion<N>*>(s.w_region);
                    auto* xr = static_cast<ArrayRegion<N>*>(s.in_region);
                    aligned = aligned && wr->start == xr->start && wr->step == xr->step;
                }
                if (!aligned)
                    CkAbort("bincount: weights must have the same layout as x "
                            "(materialize both with .copy())");
            }
            if (s.acc_int)
                dispatch_send_side<N, int64_t>(partition, dag_group, node->id, s, it->second, w_arr, g);
            else
                dispatch_send_side<N, double>(partition, dag_group, node->id, s, it->second, w_arr, g);
        }
    }

    // Receive side: does this chare own part of the result?
    bool receiver = false;
    if (N == s.out_nd) {
        ArrayDecomp<N> out_decomp = dag_group->array_meta[s.out_name].template decomp<N>();
        ArrayRegion<N> mine = out_decomp.chare_region_local(nd_idx);
        receiver = true;
        for (int d = 0; d < N; ++d)
            if (mine.size(d) <= 0)
                receiver = false;
        if (receiver) {
            std::array<int, N> gs{}, zeros{}, sizes{}, ones{};
            for (int d = 0; d < N; ++d) {
                gs[d] = s.out_shape[d];
                sizes[d] = mine.size(d);
                ones[d] = 1;
            }
            ArrayRegion<N> local(zeros, sizes, ones);
            partition->ensure_array(local, gs, s.out_name, s.out_dt, out_decomp);

            ReductionState<N>& st = red_state[node->id];
            st.node = node;
            st.started = true;
            st.mine = mine;
            st.expected = mine.size() * s.extent;
            size_t acc_size = s.acc_int ? sizeof(RedAcc<int64_t>) : sizeof(RedAcc<double>);
            st.acc.assign(mine.size() * acc_size, 0);
            for (int64_t i = 0; i < mine.size(); ++i) {
                if (s.acc_int)
                    red_init<int64_t>(s.op, reinterpret_cast<RedAcc<int64_t>*>(st.acc.data())[i]);
                else
                    red_init<double>(s.op, reinterpret_cast<RedAcc<double>*>(st.acc.data())[i]);
            }
            if (reduction_try_finish<N>(partition, st, s)) {
                red_state.erase(node->id);
                node_finished(node->id);
            }
            return;
        }
    }
    node_finished(node->id);
}

template <int N>
void ArrayDAGExecutorND<N>::on_reduce_partial(int node_id, char* data, int64_t size) {
    ReductionState<N>& st = red_state[node_id];
    st.stash.emplace_back(data, data + size);
    if (!st.started)
        return; // this chare has not reached the node yet
    RedSpec s = spec_of(st.node, static_cast<ArrayDAGGroup*>(group));
    if (reduction_try_finish<N>(partition, st, s)) {
        red_state.erase(node_id);
        node_finished(node_id);
    }
}

template <int N>
void PartitionImpl<N>::receive_reduce_partial(int node_id, int64_t size, char* data) {
    executor->on_reduce_partial(node_id, data, size);
}

template void ArrayDAGExecutorND<1>::execute_reduction_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_reduction_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_reduction_node(DAGNode*);

template void ArrayDAGExecutorND<1>::on_reduce_partial(int, char*, int64_t);
template void ArrayDAGExecutorND<2>::on_reduce_partial(int, char*, int64_t);
template void ArrayDAGExecutorND<3>::on_reduce_partial(int, char*, int64_t);

template void PartitionImpl<1>::receive_reduce_partial(int, int64_t, char*);
template void PartitionImpl<2>::receive_reduce_partial(int, int64_t, char*);
template void PartitionImpl<3>::receive_reduce_partial(int, int64_t, char*);
