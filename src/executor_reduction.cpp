#include "backend_internal.hpp"

#include <climits>
#include <cmath>
#include <cstring>

// REDUCTION and BINCOUNT executors.
//
// Each input chare folds its block of the operand into per-output-element
// accumulators and sends each piece of the partial to the output chare that owns
// it (receive_reduce_partial). An output chare is done when the partials folded
// in cover its region: covered == |region| * reduced extent. No Charm++
// reduction is needed, so independent reductions in one DAG may run in any order.
//
// Full reductions reduce every axis: the result is a (1,) array on chare 0 (or
// (1,...,1) with keepdims).

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

/// Accumulator for one output element. A is int64_t (uint64_t for uint64
/// input) for exact integer reductions, cn_c128 for complex input and double
/// otherwise.
template <typename A>
struct RedAcc {
    A v;       // sum / product / extremum / running mean (VAR, STD) / mid-range sum of squares (NORM2)
    double m2; // sum of squared deviations from the mean (VAR, STD); scaled big squares (NORM2)
    union {
        int64_t idx; // index of the extremum (ARGMAX, ARGMIN)
        double m3;   // scaled small squares (NORM2)
    };
    int64_t n; // number of input elements folded in
};

// NORM2 thresholds and scales of Blue's algorithm (LAPACK dnrm2): squares of values in
// [NRM_TSML, NRM_TBIG] are summed unscaled, others with exact power-of-two scaling.
#define NRM_TSML 0x1p-511
#define NRM_TBIG 0x1p486
#define NRM_SSML 0x1p537
#define NRM_SBIG 0x1p-538

enum RedMsgKind : int {
    MSG_PARTIAL = 0,    // packed accumulators over [start, stop), each folding `depth` inputs
    MSG_BIN_DENSE = 1,  // BINCOUNT: `count` bin values for bins [start, start + count)
    MSG_BIN_SPARSE = 2, // BINCOUNT: `count` (int64 bin, value) pairs
    MSG_WEIGHTS = 3     // BINCOUNT: `count` double weights for x's view indices [start, ...)
};

/// Wire header of a receive_reduce_partial message. Every BINCOUNT sender sends one
/// MSG_BIN_* message (possibly empty) to every output chare; `depth` is its x count.
struct RedMsgHeader {
    int out_nd;
    int kind;
    int64_t start[3];
    int64_t stop[3];
    int64_t depth;
    int64_t count;
};

enum RedAccKind : int { ACC_I64 = 0, ACC_U64 = 1, ACC_F64 = 2, ACC_C128 = 3 };

/// Complex input supports SUM/PROD/MEAN/MAX/MIN/ARG*/COUNT_NONZERO/ALL/ANY; the
/// frontend rewrites complex norms and var/std on real arrays.
static int red_acc_kind(int op, DType in_dt) {
    if (dtype_is_complex(in_dt))
        return ACC_C128;
    bool exact = op != RED_MEAN && op != RED_VAR && op != RED_STD && op != RED_NORM2;
    if (exact && (dtype_is_int(in_dt) || in_dt == DType::BOOL))
        return in_dt == DType::UINT64 ? ACC_U64 : ACC_I64;
    return ACC_F64;
}

/// Call f(A{}) for an accumulator kind.
template <typename F>
static void with_acc(int kind, F&& f) {
    switch (kind) {
    case ACC_I64: f(int64_t{}); break;
    case ACC_U64: f(uint64_t{}); break;
    case ACC_F64: f(double{}); break;
    default: f(cn_c128{}); break;
    }
}

template <typename A>
CT_RED_FN bool red_isnan(A x) {
    return cn_isnan(x); // always false for integers
}

template <typename A>
CT_RED_FN A red_abs(A x) {
    if constexpr (cn_is_complex_v<A> || std::is_unsigned_v<A>)
        return x; // complex norms are rewritten by the frontend
    else
        return x < A(0) ? -x : x;
}

template <typename A>
CT_RED_FN void red_init(int op, RedAcc<A>& a) {
    a.v = (op == RED_PROD) ? A(1) : A(0);
    a.m2 = 0.0;
    if (op == RED_NORM2)
        a.m3 = 0.0;
    else
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
    return max_like ? cn_sort_lt(y, x) : cn_sort_lt(x, y);
}

template <typename A>
CT_RED_FN void red_accumulate(int op, RedAcc<A>& a, A x, int64_t ix) {
    switch (op) {
    case RED_SUM:
    case RED_MEAN:
        a.v += x;
        break;
    case RED_NORM2:
        if constexpr (std::is_same_v<A, double>) {
            double ax = x < 0.0 ? -x : x;
            if (ax > NRM_TBIG) {
                double t = ax * NRM_SBIG;
                a.m2 += t * t;
            } else if (ax < NRM_TSML) {
                double t = ax * NRM_SSML;
                a.m3 += t * t;
            } else {
                a.v += x * x;
            }
        } else {
            a.v += x * x;
        }
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
        double xd = cn_to_double(x);
        double mean = cn_to_double(a.v);
        double delta = xd - mean;
        mean += delta / static_cast<double>(a.n + 1);
        a.m2 += delta * (xd - mean);
        a.v = cn_cast<A>(mean);
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
        double ma = cn_to_double(a.v), mb = cn_to_double(b.v);
        double delta = mb - ma;
        double nt = na + nb;
        a.v = cn_cast<A>(ma + delta * nb / nt);
        a.m2 = a.m2 + b.m2 + delta * delta * na * nb / nt;
        break;
    }
    case RED_NORM2:
        a.v += b.v;
        a.m2 += b.m2;
        a.m3 += b.m3;
        break;
    default: // sums and counts
        a.v += b.v;
        break;
    }
    a.n += b.n;
}

/// Blue's combination of the three NORM2 sums (LAPACK dnrm2).
static double norm2_finalize(const RedAcc<double>& a) {
    double amed = a.v, abig = a.m2, asml = a.m3;
    if (std::isnan(amed))
        return amed;
    if (abig > 0.0) {
        if (amed > 0.0)
            abig += (amed * NRM_SBIG) * NRM_SBIG;
        return std::sqrt(abig) / NRM_SBIG;
    }
    if (asml > 0.0) {
        if (amed > 0.0) {
            double ym = std::sqrt(amed), ys = std::sqrt(asml) / NRM_SSML;
            double ymin = std::min(ym, ys), ymax = std::max(ym, ys);
            double r = ymin / ymax;
            return ymax * std::sqrt(1.0 + r * r);
        }
        return std::sqrt(asml) / NRM_SSML;
    }
    return std::sqrt(amed);
}

/// Accumulator fields a MSG_PARTIAL entry carries besides v (n is the header's depth).
static bool red_ships_m2(int op) { return op == RED_VAR || op == RED_STD || op == RED_NORM2; }
static bool red_ships_slot(int op) {
    return op == RED_ARGMAX || op == RED_ARGMIN || op == RED_NORM2; // idx or m3
}

template <typename A>
static int64_t red_entry_bytes(int op) {
    return sizeof(A) + (red_ships_m2(op) ? 8 : 0) + (red_ships_slot(op) ? 8 : 0);
}

template <typename A>
static char* red_pack(int op, const RedAcc<A>& a, char* p) {
    memcpy(p, &a.v, sizeof(A));
    p += sizeof(A);
    if (red_ships_m2(op)) {
        memcpy(p, &a.m2, 8);
        p += 8;
    }
    if (red_ships_slot(op)) {
        memcpy(p, &a.idx, 8);
        p += 8;
    }
    return p;
}

template <typename A>
static const char* red_unpack(int op, const char* p, int64_t n, RedAcc<A>& a) {
    red_init<A>(op, a);
    memcpy(&a.v, p, sizeof(A));
    p += sizeof(A);
    if (red_ships_m2(op)) {
        memcpy(&a.m2, p, 8);
        p += 8;
    }
    if (red_ships_slot(op)) {
        memcpy(&a.idx, p, 8);
        p += 8;
    }
    a.n = n;
    return p;
}

/// Final value of one output element.
template <typename A, typename T>
static T red_finalize(int op, const RedAcc<A>& a, int64_t extent, int ddof) {
    switch (op) {
    case RED_ARGMAX:
    case RED_ARGMIN:
        return cn_cast<T>(a.idx);
    case RED_ALL:
        return cn_cast<T>(cn_cast<int64_t>(a.v) == extent ? 1 : 0);
    case RED_ANY:
        return cn_cast<T>(cn_nonzero(a.v) ? 1 : 0);
    case RED_MEAN:
        if constexpr (cn_is_complex_v<A>)
            return cn_cast<T>(a.v / A(static_cast<double>(extent)));
        else
            return cn_cast<T>(cn_to_double(a.v) / static_cast<double>(extent));
    case RED_NORM2:
        if constexpr (std::is_same_v<A, double>)
            return cn_cast<T>(norm2_finalize(a));
        else
            return cn_cast<T>(std::sqrt(cn_to_double(a.v)));
    case RED_VAR:
    case RED_STD: {
        double dof = static_cast<double>(std::max<int64_t>(a.n - ddof, 0));
        double var = a.m2 / dof;
        return cn_cast<T>(op == RED_STD ? std::sqrt(var) : var);
    }
    default:
        return cn_cast<T>(a.v);
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

    CT_RED_FN void operator()(const int64_t e, value_type& acc) const {
        int64_t rem = e, off = 0, flat = 0;
        for (int d = g.nd - 1; d >= 0; --d) {
            int c = static_cast<int>(rem % g.bn[d]);
            rem /= g.bn[d];
            off += static_cast<int64_t>(g.b0[d] + c * g.bstep[d]) * g.bstride[d];
            flat += static_cast<int64_t>(g.vb0[d] + c) * g.vstride[d];
        }
        red_accumulate<A>(op, acc, cn_cast<A>(data[off]), flat);
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

    CT_RED_FN void operator()(const int64_t p) const { out[p] = fold(p, 0, -1); }

    /// Accumulator of output p over reduced positions [q0, q1) (q1 < 0: all).
    CT_RED_FN RedAcc<A> fold(int64_t p, int64_t q0, int64_t q1) const {
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
        if (q1 < 0 || q1 > q_total)
            q1 = q_total;
        for (int64_t q = q0; q < q1; ++q) {
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
            red_accumulate<A>(op, acc, cn_cast<A>(data[off]), ix);
        }
        return acc;
    }
};

/// Few outputs over long reductions: item i folds chunk i % chunks of output
/// i / chunks into tmp; RedChunkJoin then joins each output's chunks in order.
template <typename T, typename A>
struct RedChunkFunctor {
    RedPartialFunctor<T, A> f;
    RedAcc<A>* tmp;
    int chunks;
    int64_t chunk_len;

    CT_RED_FN void operator()(const int64_t i) const {
        int64_t q0 = (i % chunks) * chunk_len;
        tmp[i] = f.fold(i / chunks, q0, q0 + chunk_len);
    }
};

template <typename A>
struct RedChunkJoin {
    const RedAcc<A>* tmp;
    RedAcc<A>* out;
    int chunks;
    int op;

    CT_RED_FN void operator()(const int64_t p) const {
        RedAcc<A> acc = tmp[p * chunks];
        for (int k = 1; k < chunks; ++k)
            red_join<A>(op, acc, tmp[p * chunks + k]);
        out[p] = acc;
    }
};

struct BinRange {
    int64_t lo, hi;
};

/// Smallest and largest in-range bin of a 1-D block (BINCOUNT).
template <typename T>
struct BinRangeFunctor {
    using value_type = BinRange;
    const T* x;
    int64_t b0;
    int64_t bstep;
    int64_t length;

    CT_RED_FN void operator()(const int64_t e, value_type& r) const {
        int64_t bin = cn_cast<int64_t>(x[b0 + e * bstep]);
        if (bin < 0 || bin >= length)
            return;
        if (bin < r.lo)
            r.lo = bin;
        if (bin > r.hi)
            r.hi = bin;
    }
    CT_RED_FN void join(value_type& d, const value_type& s) const {
        if (s.lo < d.lo)
            d.lo = s.lo;
        if (s.hi > d.hi)
            d.hi = s.hi;
    }
    CT_RED_FN void init(value_type& v) const {
        v.lo = length;
        v.hi = -1;
    }
};

/// Local histogram of a 1-D block over bins [lo, lo + n) (BINCOUNT). w == nullptr
/// counts ones; weight e is w[w0 + e * wstep].
template <typename T, typename W, typename A>
struct BincountFunctor {
    const T* x;
    const W* w;
    A* hist;
    int64_t b0;
    int64_t bstep;
    int64_t w0;
    int64_t wstep;
    int64_t lo;
    int64_t length;

    CT_RED_FN void operator()(const int64_t e) const {
        int64_t bin = cn_cast<int64_t>(x[b0 + e * bstep]);
        if (bin < 0 || bin >= length)
            return;
        A inc = w ? cn_cast<A>(w[w0 + e * wstep]) : A(1);
#ifdef USE_KOKKOS
        Kokkos::atomic_add(&hist[bin - lo], inc);
#else
        hist[bin - lo] += inc;
#endif
    }
};

/// Strided weights block converted to double (BINCOUNT weights moved to x's layout).
template <typename W>
struct WeightPackFunctor {
    const W* w;
    double* out;
    int64_t b0;
    int64_t bstep;

    CT_RED_FN void operator()(const int64_t e) const { out[e] = cn_cast<double>(w[b0 + e * bstep]); }
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
    int in_name = -1;
    int in_nd = 1;
    std::array<int, 3> view_shape = {1, 1, 1};
    Region* in_region = nullptr;
    int w_name = -1; // BINCOUNT weights (-1: none)
    Region* w_region = nullptr;
    DType w_dt = DType::FLOAT64;
    int acc = ACC_F64; // RedAccKind
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
        s.acc = red_acc_kind(s.op, s.in_dt);
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
        double length = root->operands[i]->scalar;
        if (!(length >= 0.0 && length <= INT_MAX))
            CkAbort("bincount: output length %.0f is outside [0, 2^31 - 1]", length);
        s.length = static_cast<int64_t>(length);
        s.in_dt = static_cast<DType>(static_cast<int>(root->operands[i + 1]->scalar));
        if (s.w_name >= 0)
            s.w_dt = static_cast<DType>(static_cast<int>(root->operands[i + 2]->scalar));
        s.acc = s.w_name < 0 ? ACC_I64 : ACC_F64;
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

template <int M>
static void send_reduce_msg(ArrayDAGGroup* dag_group, const ChareIndex<M>& ci, int node_id,
                            std::vector<char>& buf) {
    int64_t bytes = static_cast<int64_t>(buf.size());
    if constexpr (M == 1)
        proxy_at<1>(dag_group->partition_proxy_1, ci).receive_reduce_partial(node_id, bytes, buf.data());
    else if constexpr (M == 2)
        proxy_at<2>(dag_group->partition_proxy_2, ci).receive_reduce_partial(node_id, bytes, buf.data());
    else
        proxy_at<3>(dag_group->partition_proxy_3, ci).receive_reduce_partial(node_id, bytes, buf.data());
}

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
    const int64_t esz = red_entry_bytes<A>(s.op);

    auto pieces = decompose(out_decomp.to_global(box), out_decomp);
    for (auto& [ci, piece_global] : pieces) {
        ArrayRegion<M> piece = out_decomp.to_local(piece_global);
        int64_t count = piece.size();
        if (count <= 0)
            continue;
        std::vector<char> buf(sizeof(RedMsgHeader) + count * esz);
        RedMsgHeader h = {};
        h.out_nd = M;
        h.kind = MSG_PARTIAL;
        for (int d = 0; d < M; ++d) {
            h.start[d] = piece.start[d];
            h.stop[d] = piece.stop[d];
        }
        h.depth = depth;
        h.count = count;
        memcpy(buf.data(), &h, sizeof(h));
        char* dst = buf.data() + sizeof(h);

        std::array<int, M> idx{};
        for (int64_t k = 0; k < count; ++k) {
            int64_t src = 0;
            for (int d = 0; d < M; ++d)
                src += static_cast<int64_t>(piece.start[d] + idx[d] - box.start[d]) * box_stride[d];
            dst = red_pack<A>(s.op, part[src], dst);
            for (int d = M - 1; d >= 0; --d) {
                if (++idx[d] < piece.size(d))
                    break;
                idx[d] = 0;
            }
        }
        send_reduce_msg<M>(dag_group, ci, node_id, buf);
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
        for (int64_t e = 0; e < total; ++e)
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
    using AccView = Kokkos::View<RedAcc<A>*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    AccView d_part(static_cast<RedAcc<A>*>(comm_alloc(P * sizeof(RedAcc<A>))), P);
    RedPartialFunctor<T, A> f{data, d_part.data(), g, s.op, arg_axis};
    int64_t q_total = 1;
    for (int d = 0; d < N; ++d)
        if ((s.red_mask >> d) & 1)
            q_total *= g.bn[d];
    // one thread per output starves the GPU when there are few outputs
    int chunks = (int)std::min<int64_t>({std::max<int64_t>(1, 65536 / P), (q_total + 255) / 256,
                                         1024});
    RedAcc<A>* tmp = nullptr;
    if (chunks > 1) {
        int64_t chunk_len = (q_total + chunks - 1) / chunks;
        tmp = static_cast<RedAcc<A>*>(comm_alloc(P * chunks * sizeof(RedAcc<A>)));
        Kokkos::parallel_for("ct_reduce_chunks", CT_COMPUTE_POLICY(partition, P * chunks),
                             RedChunkFunctor<T, A>{f, tmp, chunks, chunk_len});
        Kokkos::parallel_for("ct_reduce_join", CT_COMPUTE_POLICY(partition, P),
                             RedChunkJoin<A>{tmp, d_part.data(), chunks, s.op});
    } else {
        Kokkos::parallel_for("ct_reduce_axis", CT_COMPUTE_POLICY(partition, P), f);
    }
    Kokkos::View<RedAcc<A>*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_part(part.data(), P);
#ifdef USE_NVIDIA
    Kokkos::deep_copy(partition->compute_exec, h_part, d_part);
    partition->compute_exec.fence();
#else
    Kokkos::deep_copy(h_part, d_part);
#endif
    if (tmp)
        comm_free(tmp);
    comm_free(d_part.data());
#else
    RedPartialFunctor<T, A> f{data, part.data(), g, s.op, arg_axis};
    for (int64_t p = 0; p < P; ++p)
        f(p);
#endif
    send_partial<A>(dag_group, node_id, s, start, stop, part.data(), depth);
}

/// Send a local histogram over bins [lo, lo + hist.size()) to every output chare, as
/// dense values over the nonzero range of its piece or as (bin, value) pairs.
template <typename A>
static void send_bincount_pieces(ArrayDAGGroup* dag_group, int node_id, const RedSpec& s,
                                 const std::vector<A>& hist, int64_t lo, int64_t depth) {
    ArrayDecomp<1> out_decomp = dag_group->array_meta[s.out_name].template decomp<1>();
    ArrayRegion<1> all({0}, {static_cast<int>(s.length)}, {1});
    const int64_t hi = lo + static_cast<int64_t>(hist.size());
    for (auto& [ci, piece_global] : decompose(out_decomp.to_global(all), out_decomp)) {
        ArrayRegion<1> piece = out_decomp.to_local(piece_global);
        if (piece.size() <= 0)
            continue;
        int64_t a = std::max<int64_t>(piece.start[0], lo);
        int64_t b = std::min<int64_t>(piece.stop[0], hi);
        while (a < b && hist[a - lo] == A(0))
            ++a;
        while (b > a && hist[b - 1 - lo] == A(0))
            --b;
        int64_t nz = 0;
        for (int64_t i = a; i < b; ++i)
            nz += hist[i - lo] != A(0);
        bool sparse = nz * static_cast<int64_t>(8 + sizeof(A)) <
                      (b - a) * static_cast<int64_t>(sizeof(A));
        RedMsgHeader h = {};
        h.out_nd = 1;
        h.kind = sparse ? MSG_BIN_SPARSE : MSG_BIN_DENSE;
        h.start[0] = a;
        h.stop[0] = b;
        h.depth = depth;
        h.count = sparse ? nz : b - a;
        std::vector<char> buf(sizeof(h) + h.count * (sparse ? 8 + sizeof(A) : sizeof(A)));
        memcpy(buf.data(), &h, sizeof(h));
        char* p = buf.data() + sizeof(h);
        if (sparse) {
            for (int64_t i = a; i < b; ++i) {
                if (hist[i - lo] == A(0))
                    continue;
                memcpy(p, &i, 8);
                memcpy(p + 8, &hist[i - lo], sizeof(A));
                p += 8 + sizeof(A);
            }
        } else if (b > a) {
            memcpy(p, hist.data() + (a - lo), (b - a) * sizeof(A));
        }
        send_reduce_msg<1>(dag_group, ci, node_id, buf);
    }
}

/// Histogram of this chare's block of x over the bin range it hits; weight e of the
/// block is w[w0 + e * wstep] (w == nullptr: counts).
template <typename T, typename W, typename A>
static void bincount_block_and_send(PartitionImpl<1>* partition, ArrayDAGGroup* dag_group,
                                    int node_id, const RedSpec& s, CTArrayBase<1>* x_arr,
                                    const W* w, int64_t w0, int64_t wstep, const RedGeom& g) {
    const T* x = static_cast<const T*>(x_arr->device_data_ptr());
    const int64_t len = s.length, nelem = g.bn[0], b0 = g.b0[0], bstep = g.bstep[0];
    BinRangeFunctor<T> rf{x, b0, bstep, len};
    BinRange r;
#ifdef USE_KOKKOS
    Kokkos::parallel_reduce("ct_bincount_range", CT_COMPUTE_POLICY(partition, nelem), rf, r);
#else
    rf.init(r);
    for (int64_t e = 0; e < nelem; ++e)
        rf(e, r);
#endif
    const int64_t lo = r.lo, nb = r.hi >= r.lo ? r.hi - r.lo + 1 : 0;
    std::vector<A> hist(nb, A(0));
    if (nb > 0) {
#ifdef USE_KOKKOS
        Kokkos::View<A*, DeviceSpace> d_hist("ct_bincount", nb);
        BincountFunctor<T, W, A> f{x, w, d_hist.data(), b0, bstep, w0, wstep, lo, len};
        Kokkos::parallel_for("ct_bincount", CT_COMPUTE_POLICY(partition, nelem), f);
        Kokkos::View<A*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_hist(hist.data(), nb);
#ifdef USE_NVIDIA
        Kokkos::deep_copy(partition->compute_exec, h_hist, d_hist);
        partition->compute_exec.fence();
#else
        Kokkos::deep_copy(h_hist, d_hist);
#endif
#else
        BincountFunctor<T, W, A> f{x, w, hist.data(), b0, bstep, w0, wstep, lo, len};
        for (int64_t e = 0; e < nelem; ++e)
            f(e);
#endif
    }
    send_bincount_pieces<A>(dag_group, node_id, s, hist, lo, nelem);
}

template <int N, typename T, typename A>
static void dispatch_bincount_weights(PartitionImpl<N>* partition, ArrayDAGGroup* dag_group,
                                      int node_id, const RedSpec& s, CTArrayBase<N>* x_arr,
                                      CTArrayBase<N>* w_arr, const RedGeom& g) {
    if (w_arr == nullptr) {
        bincount_block_and_send<T, T, A>(partition, dag_group, node_id, s, x_arr, nullptr, 0, 0,
                                         g);
        return;
    }
    cn_dispatch(w_arr->dtype, [&](auto tag) {
        using W = decltype(tag);
        if constexpr (!cn_is_complex_v<W>)
            bincount_block_and_send<T, W, A>(partition, dag_group, node_id, s, x_arr,
                                             static_cast<const W*>(w_arr->device_data_ptr()),
                                             g.b0[0], g.bstep[0], g);
    });
}

/// View start/step of a 1-D operand (global: the whole array).
static void view_start_step(Region* r, int64_t& start, int64_t& step) {
    start = 0;
    step = 1;
    if (r != nullptr && !r->is_global) {
        auto* ar = static_cast<ArrayRegion<1>*>(r);
        start = ar->start[0];
        step = ar->step[0];
    }
}

/// True if the BINCOUNT weights can be read at x's local offsets (same tiles and view).
static bool bincount_weights_aligned(ArrayDAGGroup* dag_group, const RedSpec& s) {
    auto xm = dag_group->array_meta.find(s.in_name);
    auto wm = dag_group->array_meta.find(s.w_name);
    if (xm == dag_group->array_meta.end() || wm == dag_group->array_meta.end())
        CkAbort("bincount: no metadata for x (%d) or weights (%d)", s.in_name, s.w_name);
    int64_t xs, xst, ws, wst;
    view_start_step(s.in_region, xs, xst);
    view_start_step(s.w_region, ws, wst);
    return xm->second.tile == wm->second.tile && xm->second.offset[0] == wm->second.offset[0] &&
           xs == ws && xst == wst;
}

/// BINCOUNT weights laid out unlike x: send this chare's block of the weights view, as
/// doubles, to the chares holding the matching elements of x.
template <typename W>
static void push_weights(PartitionImpl<1>* partition, ArrayDAGGroup* dag_group, int node_id,
                         const RedSpec& s, CTArrayBase<1>* w_arr, const RedGeom& gw) {
    const int64_t n = gw.bn[0];
    std::vector<double> vals(n);
    WeightPackFunctor<W> f{static_cast<const W*>(w_arr->device_data_ptr()), nullptr, gw.b0[0],
                           gw.bstep[0]};
#ifdef USE_KOKKOS
    Kokkos::View<double*, DeviceSpace> d_vals("ct_bincount_w", n);
    f.out = d_vals.data();
    Kokkos::parallel_for("ct_bincount_wpack", CT_COMPUTE_POLICY(partition, n), f);
    Kokkos::View<double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_vals(vals.data(), n);
#ifdef USE_NVIDIA
    Kokkos::deep_copy(partition->compute_exec, h_vals, d_vals);
    partition->compute_exec.fence();
#else
    Kokkos::deep_copy(h_vals, d_vals);
#endif
#else
    (void)partition;
    f.out = vals.data();
    for (int64_t e = 0; e < n; ++e)
        f(e);
#endif
    int64_t xs, xst;
    view_start_step(s.in_region, xs, xst);
    ArrayDecomp<1> xd = dag_group->array_meta[s.in_name].template decomp<1>();
    const int64_t a = gw.vb0[0];
    ArrayRegion<1> span({static_cast<int>(xs + a * xst)}, {static_cast<int>(xs + (a + n - 1) * xst + 1)},
                        {static_cast<int>(xst)});
    for (auto& [ci, piece_global] : decompose(xd.to_global(span), xd)) {
        ArrayRegion<1> piece = xd.to_local(piece_global);
        int64_t cnt = piece.size();
        if (cnt <= 0)
            continue;
        RedMsgHeader h = {};
        h.out_nd = 1;
        h.kind = MSG_WEIGHTS;
        h.start[0] = (piece.start[0] - xs) / xst;
        h.stop[0] = h.start[0] + cnt;
        h.count = cnt;
        std::vector<char> buf(sizeof(h) + cnt * sizeof(double));
        memcpy(buf.data(), &h, sizeof(h));
        memcpy(buf.data() + sizeof(h), vals.data() + (h.start[0] - a), cnt * sizeof(double));
        send_reduce_msg<1>(dag_group, ci, node_id, buf);
    }
}

/// Input/accumulator pairs red_acc_kind can produce (limits instantiations).
template <typename T, typename A>
constexpr bool red_combo_ok() {
    if constexpr (cn_is_complex_v<T> || cn_is_complex_v<A>)
        return cn_is_complex_v<T> && cn_is_complex_v<A>;
    else if constexpr (std::is_same_v<A, uint64_t>)
        return std::is_same_v<T, uint64_t>;
    else if constexpr (std::is_same_v<A, int64_t>)
        return !cn_is_real_float_v<T>;
    else
        return true;
}

template <typename F>
static void dispatch_bin_input(DType dt, F&& f) {
    if (!dtype_is_int(dt) && dt != DType::BOOL)
        CkAbort("bincount: input must have an integer dtype");
    cn_dispatch(dt, [&](auto tag) {
        using T = decltype(tag);
        if constexpr (!cn_is_complex_v<T> && !cn_is_real_float_v<T>)
            f(tag);
    });
}

template <int N, typename A>
static void dispatch_send_side(PartitionImpl<N>* partition, ArrayDAGGroup* dag_group, int node_id,
                               const RedSpec& s, CTArrayBase<N>* arr, CTArrayBase<N>* w_arr,
                               const RedGeom& g) {
    if (s.opc == Opcode::BINCOUNT) {
        if constexpr (N == 1 && (std::is_same_v<A, int64_t> || std::is_same_v<A, double>)) {
            dispatch_bin_input(arr->dtype, [&](auto tag) {
                dispatch_bincount_weights<N, decltype(tag), A>(partition, dag_group, node_id, s,
                                                               arr, w_arr, g);
            });
        }
        return;
    }
    cn_dispatch(arr->dtype, [&](auto tag) {
        using T = decltype(tag);
        if constexpr (red_combo_ok<T, A>())
            reduce_block_and_send<N, T, A>(partition, dag_group, node_id, s, arr, g);
    });
}

/// x's chare has all its moved weights: histogram its block with them.
static void bincount_with_moved_weights(PartitionImpl<1>* partition, ArrayDAGGroup* dag_group,
                                        int node_id, const RedSpec& s, std::vector<double>& wts) {
    auto it = partition->arrays.find(s.in_name);
    RedGeom g;
    if (it == partition->arrays.end() || !block_geometry<1>(partition, it->second, s.in_region, 0, g))
        CkAbort("bincount: chare lost its block of x");
    const int64_t n = static_cast<int64_t>(wts.size());
#ifdef USE_KOKKOS
    Kokkos::View<double*, DeviceSpace> d_w("ct_bincount_wx", n);
    Kokkos::View<double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_w(wts.data(), n);
#ifdef USE_NVIDIA
    Kokkos::deep_copy(partition->compute_exec, d_w, h_w);
#else
    Kokkos::deep_copy(d_w, h_w);
#endif
    const double* w = d_w.data();
#else
    const double* w = wts.data();
#endif
    dispatch_bin_input(it->second->dtype, [&](auto tag) {
        bincount_block_and_send<decltype(tag), double, double>(partition, dag_group, node_id, s,
                                                               it->second, w, 0, 1, g);
    });
}

// ---- Receive side ----

template <int N, typename A>
static void fold_partial(ReductionState<N>& st, int op, const RedMsgHeader& h, const char* p) {
    auto* acc = reinterpret_cast<RedAcc<A>*>(st.acc.data());
    if (h.kind == MSG_BIN_DENSE || h.kind == MSG_BIN_SPARSE) {
        for (int64_t k = 0; k < h.count; ++k) {
            int64_t bin = h.start[0] + k;
            if (h.kind == MSG_BIN_SPARSE) {
                memcpy(&bin, p, 8);
                p += 8;
            }
            A v;
            memcpy(&v, p, sizeof(A));
            p += sizeof(A);
            acc[bin - st.mine.start[0]].v += v;
        }
        st.covered += st.mine.size() * h.depth;
        return;
    }

    std::array<int64_t, N> mine_stride{};
    mine_stride[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        mine_stride[d] = mine_stride[d + 1] * st.mine.size(d + 1);

    std::array<int64_t, N> idx{};
    for (int64_t k = 0; k < h.count; ++k) {
        int64_t dst = 0;
        for (int d = 0; d < N; ++d)
            dst += (h.start[d] + idx[d] - st.mine.start[d]) * mine_stride[d];
        RedAcc<A> b;
        p = red_unpack<A>(op, p, h.depth, b);
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
    cn_dispatch(s.out_dt, [&](auto tag) {
        write_result<N, A, decltype(tag)>(partition, st, s);
    });
}

/// Handle stashed messages; true once this chare is done with the node (its moved
/// weights are used and every input element of its output region is folded in).
template <int N>
static bool reduction_progress(PartitionImpl<N>* partition, ArrayDAGGroup* dag_group, int node_id,
                               ReductionState<N>& st, const RedSpec& s) {
    std::vector<std::vector<char>> msgs;
    msgs.swap(st.stash);
    for (auto& msg : msgs) {
        RedMsgHeader h;
        memcpy(&h, msg.data(), sizeof(h));
        const char* body = msg.data() + sizeof(h);
        if (h.kind == MSG_WEIGHTS) {
            int64_t off = h.start[0] - st.w_base;
            if (!st.sending || off < 0 || off + h.count > static_cast<int64_t>(st.weights.size()))
                CkAbort("bincount: unexpected weights [%lld, +%lld) for node %d", (long long)h.start[0],
                        (long long)h.count, node_id);
            memcpy(st.weights.data() + off, body, h.count * sizeof(double));
            st.w_got += h.count;
            continue;
        }
        if (!st.receiving)
            CkAbort("reduction: partial for node %d on a chare that owns no result", node_id);
        with_acc(s.acc, [&](auto tag) { fold_partial<N, decltype(tag)>(st, s.op, h, body); });
    }
    if (st.sending && st.w_got == static_cast<int64_t>(st.weights.size())) {
        st.sending = false;
        if constexpr (N == 1)
            bincount_with_moved_weights(partition, dag_group, node_id, s, st.weights);
        std::vector<double>().swap(st.weights);
    }
    if (st.receiving && st.covered >= st.expected) {
        with_acc(s.acc, [&](auto tag) { write_result_dispatch<N, decltype(tag)>(partition, st, s); });
        st.receiving = false;
    }
    return !st.sending && !st.receiving;
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
    ReductionState<N>& st = red_state[node->id];
    st.node = node;
    st.started = true;
    const bool move_weights = s.w_name >= 0 && !bincount_weights_aligned(dag_group, s);

    // Send side: fold this chare's block of the operand and ship the partial.
    if (N == s.in_nd) {
        auto it = partition->arrays.find(s.in_name);
        RedGeom g;
        if (it != partition->arrays.end() &&
            block_geometry<N>(partition, it->second, s.in_region, s.red_mask, g)) {
            if (move_weights) {
                st.sending = true;
                st.w_base = g.vb0[0];
                st.w_got = 0;
                st.weights.assign(g.bn[0], 0.0);
            } else {
                CTArrayBase<N>* w_arr = nullptr;
                if (s.w_name >= 0) {
                    auto w_it = partition->arrays.find(s.w_name);
                    if (w_it == partition->arrays.end())
                        CkAbort("bincount: weights array %d is not on the chare that holds x",
                                s.w_name);
                    w_arr = w_it->second;
                }
                with_acc(s.acc, [&](auto tag) {
                    dispatch_send_side<N, decltype(tag)>(partition, dag_group, node->id, s,
                                                         it->second, w_arr, g);
                });
            }
        }
    }

    if constexpr (N == 1) {
        if (move_weights) {
            auto w_it = partition->arrays.find(s.w_name);
            RedGeom gw;
            if (w_it != partition->arrays.end() &&
                block_geometry<1>(partition, w_it->second, s.w_region, 0, gw)) {
                cn_dispatch(w_it->second->dtype, [&](auto tag) {
                    using W = decltype(tag);
                    if constexpr (!cn_is_complex_v<W>)
                        push_weights<W>(partition, dag_group, node->id, s, w_it->second, gw);
                });
            }
        }
    }

    // Receive side: does this chare own part of the result?
    if (N == s.out_nd) {
        ArrayDecomp<N> out_decomp = dag_group->array_meta[s.out_name].template decomp<N>();
        ArrayRegion<N> mine = out_decomp.chare_region_local(nd_idx);
        bool receiver = true;
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

            st.receiving = true;
            st.mine = mine;
            st.expected = mine.size() * s.extent;
            with_acc(s.acc, [&](auto tag) {
                using A = decltype(tag);
                st.acc.assign(mine.size() * sizeof(RedAcc<A>), 0);
                for (int64_t i = 0; i < mine.size(); ++i)
                    red_init<A>(s.op, reinterpret_cast<RedAcc<A>*>(st.acc.data())[i]);
            });
        }
    }

    if (reduction_progress<N>(partition, dag_group, node->id, st, s)) {
        red_state.erase(node->id);
        node_finished(node->id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::on_reduce_partial(int node_id, char* data, int64_t size) {
    ReductionState<N>& st = red_state[node_id];
    st.stash.emplace_back(data, data + size);
    if (!st.started)
        return; // this chare has not reached the node yet
    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    RedSpec s = spec_of(st.node, dag_group);
    if (reduction_progress<N>(partition, dag_group, node_id, st, s)) {
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
