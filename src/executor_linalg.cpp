#include "backend_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef USE_NVIDIA
#include "dispatch.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#else
#include <Eigen/Dense>
#endif

// LINALG (cnp.linalg): operand blocks are gathered to chare 0 of the result's
// partition, factored there (Eigen; cuSOLVER/cuBLAS on NVIDIA), and result tiles
// are scattered back. Not scalable: the problem must fit on one PE/GPU.

/// Kinds. Must match charmnumeric/linalg.py (_LA_*).
enum LaOp : int {
    LA_CHOLESKY = 0,         // p0: upper
    LA_QR = 1,               // p0: 0 reduced (packed), 1 'r', 2 complete (packed)
    LA_SOLVE = 2,            // a, b
    LA_INV = 3,
    LA_EIGH = 4,             // p0: uplo (0 'L', 1 'U'), p1: eigenvectors (packed)
    LA_SVD = 5,              // p0: full_matrices, p1: compute_uv (packed)
    LA_SOLVE_TRIANGULAR = 6, // a, b; p0: trans, p1: lower, p2: unit_diagonal
    LA_DET = 7,
    LA_SLOGDET = 8,          // (sign, logabsdet)
    LA_LSTSQ = 9,            // a, b; p0: rcond (packed: x, s, rank)
    LA_RESHAPE = 10,         // row-major copy into the result shape
    LA_TSSVD = 11            // packed: s, vh, v / s
};

/// Message header, followed by `count` elements in row-major block order.
/// Gather: block of operand `operand` in view coordinates, element type `dtype`.
/// Scatter: block of the result in array coordinates, element type of the result.
struct LaMsgHeader {
    int kind; // 0 gather, 1 scatter
    int operand;
    int dtype;
    int nd;
    int start[3];
    int size[3];
    int64_t count;
};

namespace {

struct LaOperand {
    int name = -1;
    int nd = 1;
    Region* region = nullptr;
    std::array<int, 3> shape = {1, 1, 1}; // view shape
    int rows() const { return shape[0]; }
    int cols() const { return nd >= 2 ? shape[1] : 1; }
    int64_t size() const { return static_cast<int64_t>(rows()) * cols(); }
};

struct LaSpec {
    int op = 0;
    DType dt = DType::FLOAT64;
    int out_name = -1;
    int out_nd = 1;
    std::array<int, 3> out_shape = {1, 1, 1};
    int nin = 0;
    LaOperand in[2];
    double p[3] = {0, 0, 0};
    int out_rows() const { return out_shape[0]; }
    int out_cols() const { return out_nd >= 2 ? out_shape[1] : 1; }
};

template <int N>
std::array<int, 3> la_region_shape(Region* r, std::array<int, 3> s) {
    if (r != nullptr && !r->is_global) {
        auto* ar = static_cast<ArrayRegion<N>*>(r);
        for (int d = 0; d < N; ++d)
            s[d] = ar->size(d);
    }
    return s;
}

} // namespace

/// operands: arrays..., op, out_nd, out0, out1, p0, p1, p2
static LaSpec parse_linalg(ASTNode* root,
                           const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& meta) {
    LaSpec s;
    s.out_name = root->result_name;
    s.dt = root->dtype;
    int i = 0;
    while (i < (int)root->operands.size() && !root->operands[i]->is_scalar) {
        if (s.nin >= 2)
            CkAbort("linalg: at most two array operands");
        ASTNode* x = root->operands[i];
        LaOperand& o = s.in[s.nin++];
        o.name = x->result_name;
        o.nd = x->ndims;
        o.region = root->get_operand_region(i);
        auto it = meta.find(o.name);
        std::array<int, 3> fallback = {1, 1, 1};
        if (it != meta.end())
            fallback = it->second.global_shape;
        o.shape = o.nd == 1 ? la_region_shape<1>(o.region, fallback)
                            : la_region_shape<2>(o.region, fallback);
        if (o.nd > 2)
            CkAbort("linalg: operands must be 1-D or 2-D");
        ++i;
    }
    s.op = static_cast<int>(root->operands[i]->scalar);
    s.out_nd = static_cast<int>(root->operands[i + 1]->scalar);
    s.out_shape = {static_cast<int>(root->operands[i + 2]->scalar),
                   static_cast<int>(root->operands[i + 3]->scalar), 0};
    if (s.out_nd == 1)
        s.out_shape[1] = 0;
    for (int k = 0; k < 3; ++k)
        s.p[k] = root->operands[i + 4 + k]->scalar;
    return s;
}

int linalg_result_meta(ASTNode* root,
                       const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& meta,
                       std::array<int, 3>& shape_out) {
    LaSpec s = parse_linalg(root, meta);
    shape_out = s.out_shape;
    return s.out_nd;
}

// ---- Profiling (CNUM_LA_TIME=1) ----

static bool la_timing() {
    static const bool on = getenv("CNUM_LA_TIME") != nullptr;
    return on;
}

struct LaProf {
    double t0 = -1, send = 0, write = 0, todev = 0;
    bool done = false;
    std::vector<std::pair<const char*, double>> marks;
};

thread_local int la_cur_node = -1;

static LaProf& la_prof(int node_id = la_cur_node) {
    thread_local std::unordered_map<int, LaProf> p;
    return p[node_id];
}

static void la_mark(const char* what) {
    if (la_timing())
        la_prof().marks.emplace_back(what, CkWallTimer());
}

static void la_report(int node_id, const char* who) {
    LaProf& p = la_prof(node_id);
    if (!la_timing() || p.t0 < 0)
        return;
    std::string s;
    double prev = p.t0;
    for (auto& [w, t] : p.marks) {
        s += " " + std::string(w) + "=" + std::to_string(1e3 * (t - prev)).substr(0, 8);
        prev = t;
    }
    p.done = true;
    CkPrintf("[la] pe %d node %d %s: total=%.1f ms send=%.1f write=%.1f todev=%.1f |%s\n",
             CkMyPe(), node_id, who, 1e3 * (CkWallTimer() - p.t0), 1e3 * p.send, 1e3 * p.write,
             1e3 * p.todev, s.c_str());
}

// ---- Messaging ----

template <int M>
static void la_send_to(ArrayDAGGroup* g, const ChareIndex<M>& ci, int node_id,
                       std::vector<char>& buf) {
    int64_t bytes = static_cast<int64_t>(buf.size());
    if constexpr (M == 1)
        proxy_at<1>(g->partition_proxy_1, ci).receive_linalg(node_id, bytes, buf.data());
    else if constexpr (M == 2)
        proxy_at<2>(g->partition_proxy_2, ci).receive_linalg(node_id, bytes, buf.data());
    else
        proxy_at<3>(g->partition_proxy_3, ci).receive_linalg(node_id, bytes, buf.data());
}

static void la_send_root(ArrayDAGGroup* g, int out_nd, int node_id, std::vector<char>& buf) {
    switch (out_nd) {
    case 1:
        la_send_to<1>(g, ChareIndex<1>{{0}}, node_id, buf);
        break;
    case 2:
        la_send_to<2>(g, ChareIndex<2>{{0, 0}}, node_id, buf);
        break;
    default:
        la_send_to<3>(g, ChareIndex<3>{{0, 0, 0}}, node_id, buf);
        break;
    }
}

/// Send this chare's block of operand `opi` to the root.
template <int N, typename S>
static void la_send_block(PartitionImpl<N>* partition, ArrayDAGGroup* g, int node_id,
                          const LaSpec& s, int opi, CTArrayBase<N>* arr) {
    const LaOperand& o = s.in[opi];
    ArrayRegion<N> view;
    if (o.region == nullptr || o.region->is_global) {
        std::array<int, N> st{}, e{}, one{};
        for (int d = 0; d < N; ++d) {
            e[d] = arr->global_shape[d];
            one[d] = 1;
        }
        view = ArrayRegion<N>(st, e, one);
    } else {
        view = *static_cast<ArrayRegion<N>*>(o.region);
    }
    ArrayRegion<N> tile = arr->decomp.chare_region_local(partition->nd_index());
    for (int d = 0; d < N; ++d)
        if (tile.size(d) <= 0)
            return;
    auto [blk, ok] = intersect(view, tile);
    if (!ok || blk.size() <= 0)
        return;

    double ts = CkWallTimer();
    arr->copyToHost();
    const S* data = static_cast<const S*>(arr->data_ptr());
    std::array<int64_t, N> tstride{};
    tstride[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        tstride[d] = tstride[d + 1] * arr->region.size(d + 1);

    LaMsgHeader h = {};
    h.kind = 0;
    h.operand = opi;
    h.dtype = static_cast<int>(arr->dtype);
    h.nd = N;
    h.count = blk.size();
    for (int d = 0; d < N; ++d) {
        h.start[d] = (blk.start[d] - view.start[d]) / view.step[d];
        h.size[d] = blk.size(d);
    }
    std::vector<char> buf(sizeof(h) + h.count * sizeof(S));
    memcpy(buf.data(), &h, sizeof(h));
    S* dst = reinterpret_cast<S*>(buf.data() + sizeof(h));
    std::array<int, N> c{};
    for (int64_t k = 0; k < h.count; ++k) {
        int64_t off = 0;
        for (int d = 0; d < N; ++d)
            off += static_cast<int64_t>(blk.start[d] - tile.start[d] + c[d] * view.step[d]) *
                   tstride[d];
        dst[k] = data[off];
        for (int d = N - 1; d >= 0; --d) {
            if (++c[d] < h.size[d])
                break;
            c[d] = 0;
        }
    }
    la_send_root(g, s.out_nd, node_id, buf);
    la_prof(node_id).send += CkWallTimer() - ts;
}

/// Cut the column-major result into the output decomposition and send each tile.
template <int M, typename T>
static void la_scatter(ArrayDAGGroup* g, int node_id, const LaSpec& s, const std::vector<T>& out) {
    ArrayDecomp<M> decomp = g->array_meta[s.out_name].template decomp<M>();
    std::array<int, M> bs{}, be{}, one{};
    for (int d = 0; d < M; ++d) {
        be[d] = s.out_shape[d];
        one[d] = 1;
    }
    ArrayRegion<M> box(bs, be, one);
    int R = s.out_rows();
    for (auto& [ci, piece_global] : decompose(decomp.to_global(box), decomp)) {
        ArrayRegion<M> piece = decomp.to_local(piece_global);
        LaMsgHeader h = {};
        h.kind = 1;
        h.dtype = static_cast<int>(s.dt);
        h.nd = M;
        h.count = piece.size();
        if (h.count <= 0)
            continue;
        for (int d = 0; d < M; ++d) {
            h.start[d] = piece.start[d];
            h.size[d] = piece.size(d);
        }
        std::vector<char> buf(sizeof(h) + h.count * sizeof(T));
        memcpy(buf.data(), &h, sizeof(h));
        T* dst = reinterpret_cast<T*>(buf.data() + sizeof(h));
        if constexpr (M == 1) {
            for (int i = 0; i < h.size[0]; ++i)
                dst[i] = out[h.start[0] + i];
        } else {
            int64_t k = 0;
            for (int i = 0; i < h.size[0]; ++i)
                for (int j = 0; j < h.size[1]; ++j)
                    dst[k++] = out[(h.start[0] + i) + static_cast<int64_t>(h.start[1] + j) * R];
        }
        la_send_to<M>(g, ci, node_id, buf);
    }
}

// ---- Dense kernels on column-major buffers ----

namespace {

template <typename T>
struct LaProblem {
    std::vector<T> a, b;      // column-major operands
    int m = 0, n = 0;         // a is m x n
    int bn = 0;               // b is n_rows x bn (b rows = m)
    std::vector<T> out;       // column-major result, out_rows x out_cols
    int R = 0, C = 0;
    T& at(int i, int j) { return out[i + static_cast<int64_t>(j) * R]; }
};

/// Real type of T (float for complex<float>).
template <typename T>
using la_real_t = decltype(cn_real_part(T()));

template <typename T>
T la_nan() {
    la_real_t<T> q = std::numeric_limits<la_real_t<T>>::quiet_NaN();
    if constexpr (cn_is_complex_v<T>)
        return T(q, q);
    else
        return q;
}

template <typename T>
void fill_nan(std::vector<T>& v) {
    std::fill(v.begin(), v.end(), la_nan<T>());
}

template <typename T>
T la_conj(const T& x) {
    if constexpr (cn_is_complex_v<T>)
        return T(x.real(), -x.imag());
    else
        return x;
}

/// Packed tssvd output from the ascending eigenpairs (w, V) of A^T A.
template <typename T>
void tssvd_pack(LaProblem<T>& P, const std::vector<T>& w, const std::vector<T>& V) {
    int n = P.n;
    for (int c = 0; c < n; ++c) {
        int src = n - 1 - c; // descending singular values
        T sv = std::sqrt(w[src]);
        P.at(0, c) = sv;
        for (int i = 0; i < n; ++i) {
            T v = V[i + static_cast<int64_t>(src) * n];
            P.at(1 + c, i) = v;         // vh[c, i]
            P.at(1 + n + i, c) = v / sv; // (V / s)[i, c]
        }
    }
}

/// det/slogdet from the diagonal of an LU factor (element i at diag[i * stride]).
template <typename T>
void slogdet_from_lu(LaProblem<T>& P, const T* diag, int64_t stride, int n, int perm_sign) {
    if constexpr (cn_is_complex_v<T>) {
        T det = T(static_cast<la_real_t<T>>(perm_sign), 0); // det only (slogdet is real-only)
        for (int i = 0; i < n; ++i)
            det = det * diag[i * stride];
        P.out[0] = det;
    } else {
        T sign = static_cast<T>(perm_sign), logdet = 0;
        for (int i = 0; i < n; ++i) {
            T d = diag[i * stride];
            if (d == T(0)) {
                sign = 0;
                logdet = -std::numeric_limits<T>::infinity();
                break;
            }
            if (d < 0)
                sign = -sign;
            logdet += std::log(std::abs(d));
        }
        if (P.R == 1) {
            P.out[0] = sign == T(0) ? T(0) : sign * std::exp(logdet);
        } else {
            P.out[0] = sign;
            P.out[1] = logdet;
        }
    }
}

/// Packed lstsq result: x in rows [0, n), s in row n, rank at (n + 1, 0).
template <typename T>
void lstsq_pack(LaProblem<T>& P, const std::vector<T>& x, const std::vector<T>& sv, int rank) {
    int n = P.n, k = std::min(P.m, P.n);
    for (int j = 0; j < P.bn; ++j)
        for (int i = 0; i < n; ++i)
            P.at(i, j) = x[i + static_cast<int64_t>(j) * n];
    for (int i = 0; i < k; ++i)
        P.at(n, i) = sv[i];
    P.at(n + 1, 0) = static_cast<T>(rank);
}

#ifndef USE_NVIDIA

template <typename T>
using LaMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using LaMap = Eigen::Map<LaMat<T>>;

template <typename T>
void put(LaProblem<T>& P, int r0, int c0, const LaMat<T>& X) {
    for (int j = 0; j < X.cols(); ++j)
        for (int i = 0; i < X.rows(); ++i)
            P.at(r0 + i, c0 + j) = X(i, j);
}

template <typename T>
void la_compute_real(const LaSpec& s, LaProblem<T>& P);

template <typename T>
void la_compute(const LaSpec& s, LaProblem<T>& P) {
    using Mat = LaMat<T>;
    LaMap<T> A(P.a.data(), P.m, P.n);
    int m = P.m, n = P.n, k = std::min(m, n);
    switch (s.op) {
    case LA_CHOLESKY: {
        Mat L;
        bool ok;
        if (s.p[0] != 0) {
            Eigen::LLT<Mat, Eigen::Upper> llt(A);
            ok = llt.info() == Eigen::Success;
            L = llt.matrixU();
        } else {
            Eigen::LLT<Mat, Eigen::Lower> llt(A);
            ok = llt.info() == Eigen::Success;
            L = llt.matrixL();
        }
        put(P, 0, 0, L);
        if (!ok)
            fill_nan(P.out);
        break;
    }
    case LA_QR: {
        int mode = static_cast<int>(s.p[0]);
        int rr = mode == 2 ? m : k;
        Eigen::HouseholderQR<Mat> qr(A);
        Mat R = qr.matrixQR().topRows(rr).template triangularView<Eigen::Upper>();
        if (mode == 1) {
            put(P, 0, 0, R);
            break;
        }
        Mat Q = Mat::Identity(m, mode == 2 ? m : k);
        Q.applyOnTheLeft(qr.householderQ());
        put(P, 0, 0, Q);
        put(P, m, 0, R);
        break;
    }
    case LA_SOLVE:
    case LA_INV: {
        Eigen::PartialPivLU<Mat> lu(A);
        if (s.op == LA_INV) {
            put(P, 0, 0, Mat(lu.inverse()));
        } else {
            LaMap<T> B(P.b.data(), n, P.bn);
            put(P, 0, 0, Mat(lu.solve(B)));
        }
        break;
    }
    case LA_DET:
    case LA_SLOGDET: {
        Eigen::PartialPivLU<Mat> lu(A);
        Mat LU = lu.matrixLU();
        int psign = static_cast<int>(lu.permutationP().determinant());
        slogdet_from_lu(P, LU.data(), n + 1, n, psign);
        break;
    }
    case LA_EIGH: {
        Mat S = s.p[0] != 0 ? Mat(A.template selfadjointView<Eigen::Upper>())
                            : Mat(A.template selfadjointView<Eigen::Lower>());
        bool vecs = s.p[1] != 0;
        Eigen::SelfAdjointEigenSolver<Mat> es(S, vecs ? Eigen::ComputeEigenvectors
                                                      : Eigen::EigenvaluesOnly);
        for (int i = 0; i < n; ++i)
            P.out[static_cast<int64_t>(i) * (vecs ? P.R : 1)] = es.eigenvalues()(i);
        if (vecs)
            put(P, 1, 0, Mat(es.eigenvectors()));
        if (es.info() != Eigen::Success)
            fill_nan(P.out);
        break;
    }
    case LA_SVD: {
        bool full = s.p[0] != 0, uv = s.p[1] != 0;
        if (!uv) {
            Eigen::BDCSVD<Mat> svd(A);
            for (int i = 0; i < k; ++i)
                P.out[i] = svd.singularValues()(i);
            break;
        }
        unsigned opts = full ? (Eigen::ComputeFullU | Eigen::ComputeFullV)
                             : (Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::BDCSVD<Mat> svd(A, opts);
        put(P, 0, 0, Mat(svd.matrixU()));
        for (int i = 0; i < k; ++i)
            P.at(m, i) = svd.singularValues()(i);
        put(P, m + 1, 0, Mat(svd.matrixV().adjoint()));
        break;
    }
    case LA_SOLVE_TRIANGULAR: {
        Mat X = LaMap<T>(P.b.data(), n, P.bn);
        bool trans = s.p[0] != 0, lower = s.p[1] != 0, unit = s.p[2] != 0;
        // A^T of a lower triangle is upper and vice versa.
        bool solve_lower = lower != trans;
        if (trans) {
            auto At = A.transpose();
            if (solve_lower)
                unit ? At.template triangularView<Eigen::UnitLower>().solveInPlace(X)
                     : At.template triangularView<Eigen::Lower>().solveInPlace(X);
            else
                unit ? At.template triangularView<Eigen::UnitUpper>().solveInPlace(X)
                     : At.template triangularView<Eigen::Upper>().solveInPlace(X);
        } else {
            if (solve_lower)
                unit ? A.template triangularView<Eigen::UnitLower>().solveInPlace(X)
                     : A.template triangularView<Eigen::Lower>().solveInPlace(X);
            else
                unit ? A.template triangularView<Eigen::UnitUpper>().solveInPlace(X)
                     : A.template triangularView<Eigen::Upper>().solveInPlace(X);
        }
        put(P, 0, 0, X);
        break;
    }
    case LA_LSTSQ:
    case LA_TSSVD:
        if constexpr (cn_is_complex_v<T>)
            CkAbort("linalg: lstsq/tssvd do not support complex input");
        else
            la_compute_real(s, P);
        break;
    }
}

/// lstsq and tssvd (real dtypes only).
template <typename T>
void la_compute_real(const LaSpec& s, LaProblem<T>& P) {
    using Mat = LaMat<T>;
    LaMap<T> A(P.a.data(), P.m, P.n);
    int m = P.m, n = P.n, k = std::min(m, n);
    switch (s.op) {
    case LA_LSTSQ: {
        Eigen::BDCSVD<Mat> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        LaMap<T> B(P.b.data(), m, P.bn);
        auto sv = svd.singularValues();
        T cut = static_cast<T>(s.p[0]) * (k > 0 ? sv(0) : T(0));
        Eigen::Matrix<T, Eigen::Dynamic, 1> inv(k);
        int rank = 0;
        for (int i = 0; i < k; ++i) {
            inv(i) = sv(i) > cut ? T(1) / sv(i) : T(0);
            rank += sv(i) > cut;
        }
        Mat X = svd.matrixV() * (inv.asDiagonal() * (svd.matrixU().transpose() * B));
        std::vector<T> x(X.data(), X.data() + X.size()), svh(sv.data(), sv.data() + k);
        lstsq_pack(P, x, svh, rank);
        break;
    }
    case LA_TSSVD: {
        Mat G = A.transpose() * A;
        Eigen::SelfAdjointEigenSolver<Mat> es(G);
        std::vector<T> w(es.eigenvalues().data(), es.eigenvalues().data() + n);
        Mat V = es.eigenvectors();
        tssvd_pack(P, w, std::vector<T>(V.data(), V.data() + V.size()));
        break;
    }
    }
}

#else // USE_NVIDIA

#define CT_CUDA_CHECK(x)                                                                       \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess)                                                                 \
            CkAbort("linalg: %s failed: %s", #x, cudaGetErrorString(e_));                     \
    } while (0)
#define CT_CUSOLVER_CHECK(x)                                                                   \
    do {                                                                                       \
        cusolverStatus_t s_ = (x);                                                             \
        if (s_ != CUSOLVER_STATUS_SUCCESS)                                                     \
            CkAbort("linalg: %s failed (status %d)", #x, static_cast<int>(s_));               \
    } while (0)
#define CT_CUBLAS_CHECK(x)                                                                     \
    do {                                                                                       \
        cublasStatus_t s_ = (x);                                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                       \
            CkAbort("linalg: %s failed (status %d)", #x, static_cast<int>(s_));               \
    } while (0)

/// Device buffer on the root's compute stream.
template <typename T>
struct DevBuf {
    T* p = nullptr;
    size_t n = 0;
    cudaStream_t st;
    DevBuf(size_t n_, cudaStream_t st_) : n(n_), st(st_) {
        if (n > 0)
            CT_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    }
    DevBuf(const std::vector<T>& h, cudaStream_t st_) : DevBuf(h.size(), st_) { upload(h); }
    ~DevBuf() {
        if (p)
            cudaFree(p);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    void upload(const std::vector<T>& h) {
        if (n > 0)
            CT_CUDA_CHECK(cudaMemcpyAsync(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice, st));
    }
    std::vector<T> download(size_t count) const {
        std::vector<T> h(count);
        if (count > 0) {
            CT_CUDA_CHECK(cudaMemcpyAsync(h.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost, st));
            CT_CUDA_CHECK(cudaStreamSynchronize(st));
        }
        return h;
    }
    std::vector<T> download() const { return download(n); }
};

struct LaHandles {
    cusolverDnHandle_t solver = nullptr;
    cublasHandle_t blas = nullptr;
};

static LaHandles& la_slot() {
    thread_local LaHandles h; // per PE
    return h;
}

/// Creates the handles unless they exist; false (nothing kept) on failure.
static bool la_try_create() {
    LaHandles& h = la_slot();
    if (h.solver != nullptr)
        return true;
    cusolverDnHandle_t solver = nullptr;
    cublasHandle_t blas = nullptr;
    if (cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS)
        return false;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        cusolverDnDestroy(solver);
        return false;
    }
    h.solver = solver;
    h.blas = blas;
    return true;
}

static LaHandles& la_handles(cudaStream_t st) {
    LaHandles& h = la_slot();
    if (h.solver == nullptr) {
        CT_CUSOLVER_CHECK(cusolverDnCreate(&h.solver));
        CT_CUBLAS_CHECK(cublasCreate(&h.blas));
    }
    CT_CUSOLVER_CHECK(cusolverDnSetStream(h.solver, st));
    CT_CUBLAS_CHECK(cublasSetStream(h.blas, st));
    return h;
}

// float/double overloads of the cuSOLVER/cuBLAS routines used below.
#define CT_LA_OVERLOADS(T, P)                                                                    \
    inline cusolverStatus_t potrf_bs(cusolverDnHandle_t h, cublasFillMode_t u, int n, T* A,     \
                                     int lda, int* lw) {                                         \
        return cusolverDn##P##potrf_bufferSize(h, u, n, A, lda, lw);                             \
    }                                                                                            \
    inline cusolverStatus_t potrf(cusolverDnHandle_t h, cublasFillMode_t u, int n, T* A,        \
                                  int lda, T* w, int lw, int* info) {                            \
        return cusolverDn##P##potrf(h, u, n, A, lda, w, lw, info);                               \
    }                                                                                            \
    inline cusolverStatus_t getrf_bs(cusolverDnHandle_t h, int m, int n, T* A, int lda,         \
                                     int* lw) {                                                  \
        return cusolverDn##P##getrf_bufferSize(h, m, n, A, lda, lw);                             \
    }                                                                                            \
    inline cusolverStatus_t getrf(cusolverDnHandle_t h, int m, int n, T* A, int lda, T* w,      \
                                  int* ipiv, int* info) {                                        \
        return cusolverDn##P##getrf(h, m, n, A, lda, w, ipiv, info);                             \
    }                                                                                            \
    inline cusolverStatus_t getrs(cusolverDnHandle_t h, int n, int nrhs, const T* A, int lda,   \
                                  const int* ipiv, T* B, int ldb, int* info) {                   \
        return cusolverDn##P##getrs(h, CUBLAS_OP_N, n, nrhs, A, lda, ipiv, B, ldb, info);        \
    }                                                                                            \
    inline cusolverStatus_t geqrf_bs(cusolverDnHandle_t h, int m, int n, T* A, int lda,         \
                                     int* lw) {                                                  \
        return cusolverDn##P##geqrf_bufferSize(h, m, n, A, lda, lw);                             \
    }                                                                                            \
    inline cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, T* A, int lda, T* tau,    \
                                  T* w, int lw, int* info) {                                     \
        return cusolverDn##P##geqrf(h, m, n, A, lda, tau, w, lw, info);                          \
    }                                                                                            \
    inline cusolverStatus_t orgqr_bs(cusolverDnHandle_t h, int m, int n, int k, const T* A,     \
                                     int lda, const T* tau, int* lw) {                           \
        return cusolverDn##P##orgqr_bufferSize(h, m, n, k, A, lda, tau, lw);                     \
    }                                                                                            \
    inline cusolverStatus_t orgqr(cusolverDnHandle_t h, int m, int n, int k, T* A, int lda,     \
                                  const T* tau, T* w, int lw, int* info) {                       \
        return cusolverDn##P##orgqr(h, m, n, k, A, lda, tau, w, lw, info);                       \
    }                                                                                            \
    inline cusolverStatus_t syevd_bs(cusolverDnHandle_t h, cusolverEigMode_t j,                 \
                                     cublasFillMode_t u, int n, const T* A, int lda,             \
                                     const T* W, int* lw) {                                      \
        return cusolverDn##P##syevd_bufferSize(h, j, u, n, A, lda, W, lw);                       \
    }                                                                                            \
    inline cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t j, cublasFillMode_t u, \
                                  int n, T* A, int lda, T* W, T* w, int lw, int* info) {         \
        return cusolverDn##P##syevd(h, j, u, n, A, lda, W, w, lw, info);                         \
    }                                                                                            \
    inline cusolverStatus_t gesvd_bs(cusolverDnHandle_t h, int m, int n, T*, int* lw) {         \
        return cusolverDn##P##gesvd_bufferSize(h, m, n, lw);                                     \
    }                                                                                            \
    inline cusolverStatus_t gesvd(cusolverDnHandle_t h, signed char ju, signed char jv, int m,  \
                                  int n, T* A, int lda, T* S, T* U, int ldu, T* VT, int ldvt,   \
                                  T* w, int lw, T* rw, int* info) {                              \
        return cusolverDn##P##gesvd(h, ju, jv, m, n, A, lda, S, U, ldu, VT, ldvt, w, lw, rw,     \
                                    info);                                                       \
    }                                                                                            \
    inline cublasStatus_t trsm(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,       \
                               cublasDiagType_t d, int m, int n, const T* A, int lda, T* B,     \
                               int ldb) {                                                        \
        const T one = 1;                                                                         \
        return cublas##P##trsm(h, CUBLAS_SIDE_LEFT, u, t, d, m, n, &one, A, lda, B, ldb);        \
    }                                                                                            \
    inline cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,    \
                               int m, int n, int k, const T* A, int lda, const T* B, int ldb,   \
                               T* C, int ldc) {                                                  \
        const T one = 1, zero = 0;                                                               \
        return cublas##P##gemm(h, ta, tb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc);         \
    }                                                                                            \
    inline cublasStatus_t dgmm(cublasHandle_t h, int m, int n, const T* A, int lda,             \
                               const T* x, T* C, int ldc) {                                      \
        return cublas##P##dgmm(h, CUBLAS_SIDE_LEFT, m, n, A, lda, x, 1, C, ldc);                 \
    }                                                                                            \
    inline cublasStatus_t geam_t(cublasHandle_t h, int m, int n, const T* A, int lda, T* C,     \
                                 int ldc) {                                                      \
        const T one = 1, zero = 0;                                                               \
        return cublas##P##geam(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &one, A, lda, &zero, C, ldc,   \
                               C, ldc);                                                          \
    }                                                                                            \
    inline cublasStatus_t gemm_sub_tn(cublasHandle_t h, int m, int n, int k, const T* A, int lda,\
                                      const T* B, int ldb, T* C, int ldc) {                      \
        const T minus = -1, one = 1;                                                             \
        return cublas##P##gemm(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &minus, A, lda, B, ldb,     \
                               &one, C, ldc);                                                    \
    }                                                                                            \
    inline cublasStatus_t syrk_sub_t(cublasHandle_t h, int n, int k, const T* A, int lda, T* C,  \
                                     int ldc) {                                                  \
        const T minus = -1, one = 1;                                                             \
        return cublas##P##syrk(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, k, &minus, A, lda,     \
                               &one, C, ldc);                                                    \
    }

// Complex overloads: T is cn_complex (layout of CT, the CUDA type), R the real
// type of S/W/rwork. orgqr/syevd become ungqr/heevd.
#define CT_LA_OVERLOADS_C(T, R, CT, P)                                                           \
    inline CT* cu(T* p) { return reinterpret_cast<CT*>(p); }                                     \
    inline const CT* cu(const T* p) { return reinterpret_cast<const CT*>(p); }                   \
    inline cusolverStatus_t potrf_bs(cusolverDnHandle_t h, cublasFillMode_t u, int n, T* A,     \
                                     int lda, int* lw) {                                         \
        return cusolverDn##P##potrf_bufferSize(h, u, n, cu(A), lda, lw);                         \
    }                                                                                            \
    inline cusolverStatus_t potrf(cusolverDnHandle_t h, cublasFillMode_t u, int n, T* A,        \
                                  int lda, T* w, int lw, int* info) {                            \
        return cusolverDn##P##potrf(h, u, n, cu(A), lda, cu(w), lw, info);                       \
    }                                                                                            \
    inline cusolverStatus_t getrf_bs(cusolverDnHandle_t h, int m, int n, T* A, int lda,         \
                                     int* lw) {                                                  \
        return cusolverDn##P##getrf_bufferSize(h, m, n, cu(A), lda, lw);                         \
    }                                                                                            \
    inline cusolverStatus_t getrf(cusolverDnHandle_t h, int m, int n, T* A, int lda, T* w,      \
                                  int* ipiv, int* info) {                                        \
        return cusolverDn##P##getrf(h, m, n, cu(A), lda, cu(w), ipiv, info);                     \
    }                                                                                            \
    inline cusolverStatus_t getrs(cusolverDnHandle_t h, int n, int nrhs, const T* A, int lda,   \
                                  const int* ipiv, T* B, int ldb, int* info) {                   \
        return cusolverDn##P##getrs(h, CUBLAS_OP_N, n, nrhs, cu(A), lda, ipiv, cu(B), ldb,       \
                                    info);                                                       \
    }                                                                                            \
    inline cusolverStatus_t geqrf_bs(cusolverDnHandle_t h, int m, int n, T* A, int lda,         \
                                     int* lw) {                                                  \
        return cusolverDn##P##geqrf_bufferSize(h, m, n, cu(A), lda, lw);                         \
    }                                                                                            \
    inline cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, T* A, int lda, T* tau,    \
                                  T* w, int lw, int* info) {                                     \
        return cusolverDn##P##geqrf(h, m, n, cu(A), lda, cu(tau), cu(w), lw, info);              \
    }                                                                                            \
    inline cusolverStatus_t orgqr_bs(cusolverDnHandle_t h, int m, int n, int k, const T* A,     \
                                     int lda, const T* tau, int* lw) {                           \
        return cusolverDn##P##ungqr_bufferSize(h, m, n, k, cu(A), lda, cu(tau), lw);             \
    }                                                                                            \
    inline cusolverStatus_t orgqr(cusolverDnHandle_t h, int m, int n, int k, T* A, int lda,     \
                                  const T* tau, T* w, int lw, int* info) {                       \
        return cusolverDn##P##ungqr(h, m, n, k, cu(A), lda, cu(tau), cu(w), lw, info);           \
    }                                                                                            \
    inline cusolverStatus_t syevd_bs(cusolverDnHandle_t h, cusolverEigMode_t j,                 \
                                     cublasFillMode_t u, int n, const T* A, int lda,             \
                                     const R* W, int* lw) {                                      \
        return cusolverDn##P##heevd_bufferSize(h, j, u, n, cu(A), lda, W, lw);                   \
    }                                                                                            \
    inline cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t j, cublasFillMode_t u, \
                                  int n, T* A, int lda, R* W, T* w, int lw, int* info) {         \
        return cusolverDn##P##heevd(h, j, u, n, cu(A), lda, W, cu(w), lw, info);                 \
    }                                                                                            \
    inline cusolverStatus_t gesvd_bs(cusolverDnHandle_t h, int m, int n, T*, int* lw) {         \
        return cusolverDn##P##gesvd_bufferSize(h, m, n, lw);                                     \
    }                                                                                            \
    inline cusolverStatus_t gesvd(cusolverDnHandle_t h, signed char ju, signed char jv, int m,  \
                                  int n, T* A, int lda, R* S, T* U, int ldu, T* VT, int ldvt,   \
                                  T* w, int lw, R* rw, int* info) {                              \
        return cusolverDn##P##gesvd(h, ju, jv, m, n, cu(A), lda, S, cu(U), ldu, cu(VT), ldvt,   \
                                    cu(w), lw, rw, info);                                        \
    }                                                                                            \
    inline cublasStatus_t trsm(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,       \
                               cublasDiagType_t d, int m, int n, const T* A, int lda, T* B,     \
                               int ldb) {                                                        \
        const T one(1, 0);                                                                       \
        return cublas##P##trsm(h, CUBLAS_SIDE_LEFT, u, t, d, m, n, cu(&one), cu(A), lda, cu(B),  \
                               ldb);                                                             \
    }                                                                                            \
    inline cublasStatus_t geam_t(cublasHandle_t h, int m, int n, const T* A, int lda, T* C,     \
                                 int ldc) {                                                      \
        const T one(1, 0), zero(0, 0);                                                           \
        return cublas##P##geam(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, cu(&one), cu(A), lda,          \
                               cu(&zero), cu(C), ldc, cu(C), ldc);                               \
    }

namespace cula {
CT_LA_OVERLOADS(float, S)
CT_LA_OVERLOADS(double, D)
CT_LA_OVERLOADS_C(cn_c64, float, cuComplex, C)
CT_LA_OVERLOADS_C(cn_c128, double, cuDoubleComplex, Z)
} // namespace cula

static int la_info(const DevBuf<int>& info) { return info.download(1)[0]; }

static void la_smark(cudaStream_t st, const char* what) {
    if (la_timing()) {
        cudaStreamSynchronize(st);
        la_mark(what);
    }
}

/// SVD of a host column-major m x n matrix. U is m x (full ? m : k), VT is
/// (full ? n : k) x n. gesvd needs m >= n, so m < n factors A^T instead.
template <typename T>
void gpu_svd(LaHandles& h, cudaStream_t st, const std::vector<T>& A, int m, int n, bool full,
             bool uv, std::vector<T>& U, std::vector<la_real_t<T>>& sv, std::vector<T>& VT) {
    using R = la_real_t<T>;
    bool tr = m < n; // factor A^H
    int mm = tr ? n : m, nn = tr ? m : n, k = nn;
    std::vector<T> At = A;
    if (tr)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                At[j + static_cast<int64_t>(i) * n] = la_conj(A[i + static_cast<int64_t>(j) * m]);
    DevBuf<T> dA(At, st), dU(uv ? static_cast<size_t>(mm) * (full ? mm : k) : 1, st),
        dVT(uv ? static_cast<size_t>(nn) * nn : 1, st);
    DevBuf<R> dS(k, st), dRW(k, st);
    DevBuf<int> dInfo(1, st);
    int lw = 0;
    CT_CUSOLVER_CHECK(cula::gesvd_bs(h.solver, mm, nn, dA.p, &lw));
    DevBuf<T> dW(std::max(lw, 1), st);
    signed char ju = uv ? (full ? 'A' : 'S') : 'N', jv = uv ? 'A' : 'N';
    CT_CUSOLVER_CHECK(cula::gesvd(h.solver, ju, jv, mm, nn, dA.p, mm, dS.p, dU.p, mm, dVT.p, nn,
                                  dW.p, lw, dRW.p, dInfo.p));
    sv = dS.download();
    if (la_info(dInfo) != 0)
        fill_nan(sv);
    if (!uv)
        return;
    int uc = full ? mm : k;
    std::vector<T> Ud = dU.download(static_cast<size_t>(mm) * uc), Vd = dVT.download();
    if (!tr) {
        U = std::move(Ud);
        VT = std::move(Vd);
        return;
    }
    // A = (A^H)^H = V' S U'^H: U = V'^H^H (m x m), VT = U'^H (uc x n)
    U.assign(static_cast<size_t>(m) * m, T(0));
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i)
            U[i + static_cast<int64_t>(j) * m] = la_conj(Vd[j + static_cast<int64_t>(i) * nn]);
    VT.assign(static_cast<size_t>(uc) * n, T(0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < uc; ++i)
            VT[i + static_cast<int64_t>(j) * uc] = la_conj(Ud[j + static_cast<int64_t>(i) * mm]);
}

template <typename T, int N>
void la_compute(const LaSpec& s, LaProblem<T>& P, PartitionImpl<N>* partition) {
    cudaStream_t st = partition->compute_stream_raw;
    LaHandles& h = la_handles(st);
    int m = P.m, n = P.n, k = std::min(m, n);
    DevBuf<int> dInfo(1, st);
    switch (s.op) {
    case LA_CHOLESKY: {
        bool upper = s.p[0] != 0;
        cublasFillMode_t u = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        DevBuf<T> dA(P.a, st);
        la_smark(st, "h2d");
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::potrf_bs(h.solver, u, n, dA.p, n, &lw));
        DevBuf<T> dW(std::max(lw, 1), st);
        CT_CUSOLVER_CHECK(cula::potrf(h.solver, u, n, dA.p, n, dW.p, lw, dInfo.p));
        la_smark(st, "factor");
        std::vector<T> L = dA.download();
        la_mark("d2h");
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                P.at(i, j) = (upper ? i <= j : i >= j) ? L[i + static_cast<int64_t>(j) * n] : T(0);
        if (la_info(dInfo) != 0)
            fill_nan(P.out);
        break;
    }
    case LA_QR: {
        int mode = static_cast<int>(s.p[0]);
        int rr = mode == 2 ? m : k, qc = mode == 2 ? m : k;
        DevBuf<T> dA(P.a, st), dTau(std::max(k, 1), st);
        la_smark(st, "h2d");
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::geqrf_bs(h.solver, m, n, dA.p, m, &lw));
        DevBuf<T> dW(std::max(lw, 1), st);
        CT_CUSOLVER_CHECK(cula::geqrf(h.solver, m, n, dA.p, m, dTau.p, dW.p, lw, dInfo.p));
        la_smark(st, "geqrf");
        std::vector<T> F = dA.download();
        la_mark("d2h");
        int r0 = mode == 1 ? 0 : m;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < rr; ++i)
                P.at(r0 + i, j) = (i <= j && i < m) ? F[i + static_cast<int64_t>(j) * m] : T(0);
        if (mode == 1)
            break;
        DevBuf<T> dQ(static_cast<size_t>(m) * qc, st);
        CT_CUDA_CHECK(cudaMemcpyAsync(dQ.p, dA.p, sizeof(T) * m * std::min(qc, n),
                                      cudaMemcpyDeviceToDevice, st));
        int lq = 0;
        CT_CUSOLVER_CHECK(cula::orgqr_bs(h.solver, m, qc, k, dQ.p, m, dTau.p, &lq));
        DevBuf<T> dWq(std::max(lq, 1), st);
        CT_CUSOLVER_CHECK(cula::orgqr(h.solver, m, qc, k, dQ.p, m, dTau.p, dWq.p, lq, dInfo.p));
        la_smark(st, "orgqr");
        std::vector<T> Q = dQ.download();
        la_mark("d2h");
        for (int j = 0; j < qc; ++j)
            for (int i = 0; i < m; ++i)
                P.at(i, j) = Q[i + static_cast<int64_t>(j) * m];
        break;
    }
    case LA_SOLVE:
    case LA_INV:
    case LA_DET:
    case LA_SLOGDET: {
        DevBuf<T> dA(P.a, st);
        DevBuf<int> dPiv(std::max(n, 1), st);
        la_smark(st, "h2d");
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::getrf_bs(h.solver, n, n, dA.p, n, &lw));
        DevBuf<T> dW(std::max(lw, 1), st);
        CT_CUSOLVER_CHECK(cula::getrf(h.solver, n, n, dA.p, n, dW.p, dPiv.p, dInfo.p));
        la_smark(st, "getrf");
        if (s.op == LA_DET || s.op == LA_SLOGDET) {
            std::vector<T> LU = dA.download();
            std::vector<int> piv = dPiv.download();
            int psign = 1;
            for (int i = 0; i < n; ++i)
                if (piv[i] != i + 1)
                    psign = -psign;
            slogdet_from_lu(P, LU.data(), n + 1, n, psign);
            break;
        }
        std::vector<T> Bh;
        int nrhs;
        if (s.op == LA_INV) {
            nrhs = n;
            Bh.assign(static_cast<size_t>(n) * n, T(0));
            for (int i = 0; i < n; ++i)
                Bh[i + static_cast<int64_t>(i) * n] = T(1);
        } else {
            nrhs = P.bn;
            Bh = P.b;
        }
        DevBuf<T> dB(Bh, st);
        CT_CUSOLVER_CHECK(cula::getrs(h.solver, n, nrhs, dA.p, n, dPiv.p, dB.p, n, dInfo.p));
        la_smark(st, "getrs");
        P.out = dB.download();
        break;
    }
    case LA_EIGH:
    case LA_TSSVD: {
        using R = la_real_t<T>;
        if constexpr (cn_is_complex_v<T>)
            if (s.op == LA_TSSVD)
                CkAbort("linalg: tssvd does not support complex input");
        bool vecs = s.op == LA_TSSVD || s.p[1] != 0;
        cusolverEigMode_t jz = vecs ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
        cublasFillMode_t u = (s.op == LA_EIGH && s.p[0] != 0) ? CUBLAS_FILL_MODE_UPPER
                                                                : CUBLAS_FILL_MODE_LOWER;
        DevBuf<T> dG(static_cast<size_t>(n) * n, st);
        DevBuf<R> dWv(std::max(n, 1), st);
        if (s.op == LA_TSSVD) {
            if constexpr (!cn_is_complex_v<T>) {
                DevBuf<T> dA(P.a, st);
                la_smark(st, "h2d");
                CT_CUBLAS_CHECK(cula::gemm(h.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, dA.p, m,
                                           dA.p, m, dG.p, n));
                la_smark(st, "gram");
            }
        } else {
            dG.upload(P.a);
        }
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::syevd_bs(h.solver, jz, u, n, dG.p, n, dWv.p, &lw));
        DevBuf<T> dW(std::max(lw, 1), st);
        CT_CUSOLVER_CHECK(cula::syevd(h.solver, jz, u, n, dG.p, n, dWv.p, dW.p, lw, dInfo.p));
        la_smark(st, "syevd");
        std::vector<R> w = dWv.download(n);
        std::vector<T> V = vecs ? dG.download() : std::vector<T>();
        if constexpr (!cn_is_complex_v<T>)
            if (s.op == LA_TSSVD) {
                tssvd_pack(P, w, V);
                break;
            }
        for (int i = 0; i < n; ++i)
            P.out[static_cast<int64_t>(i) * (vecs ? P.R : 1)] = T(w[i]);
        if (vecs)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    P.at(1 + i, j) = V[i + static_cast<int64_t>(j) * n];
        if (la_info(dInfo) != 0)
            fill_nan(P.out);
        break;
    }
    case LA_SVD: {
        bool full = s.p[0] != 0, uv = s.p[1] != 0;
        std::vector<T> U, VT;
        std::vector<la_real_t<T>> sv;
        gpu_svd(h, st, P.a, m, n, full, uv, U, sv, VT);
        if (!uv) {
            for (int i = 0; i < k; ++i)
                P.out[i] = T(sv[i]);
            break;
        }
        int uc = full ? m : k, vr = full ? n : k;
        for (int j = 0; j < uc; ++j)
            for (int i = 0; i < m; ++i)
                P.at(i, j) = U[i + static_cast<int64_t>(j) * m];
        for (int i = 0; i < k; ++i)
            P.at(m, i) = T(sv[i]);
        int ldvt = static_cast<int>(VT.size() / std::max(n, 1));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < vr; ++i)
                P.at(m + 1 + i, j) = VT[i + static_cast<int64_t>(j) * ldvt];
        break;
    }
    case LA_SOLVE_TRIANGULAR: {
        bool trans = s.p[0] != 0, lower = s.p[1] != 0, unit = s.p[2] != 0;
        DevBuf<T> dA(P.a, st), dB(P.b, st);
        CT_CUBLAS_CHECK(cula::trsm(h.blas, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                                   trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                   unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, n, P.bn, dA.p,
                                   n, dB.p, n));
        P.out = dB.download();
        break;
    }
    case LA_LSTSQ: {
        if constexpr (cn_is_complex_v<T>) {
            CkAbort("linalg: lstsq does not support complex input");
        } else {
        std::vector<T> U, sv, VT;
        gpu_svd(h, st, P.a, m, n, false, true, U, sv, VT);
        T cut = static_cast<T>(s.p[0]) * (k > 0 ? sv[0] : T(0));
        std::vector<T> inv(k);
        int rank = 0;
        for (int i = 0; i < k; ++i) {
            inv[i] = sv[i] > cut ? T(1) / sv[i] : T(0);
            rank += sv[i] > cut;
        }
        DevBuf<T> dU(U, st), dVT(VT, st), dB(P.b, st), dInv(inv, st),
            dT(static_cast<size_t>(k) * P.bn, st), dX(static_cast<size_t>(n) * P.bn, st);
        // x = V diag(inv) U^T b
        CT_CUBLAS_CHECK(cula::gemm(h.blas, CUBLAS_OP_T, CUBLAS_OP_N, k, P.bn, m, dU.p, m, dB.p, m,
                                   dT.p, k));
        CT_CUBLAS_CHECK(cula::dgmm(h.blas, k, P.bn, dT.p, k, dInv.p, dT.p, k));
        CT_CUBLAS_CHECK(cula::gemm(h.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, P.bn, k, dVT.p, k, dT.p, k,
                                   dX.p, n));
        lstsq_pack(P, dX.download(), sv, rank);
        }
        break;
    }
    }
}

#endif // USE_NVIDIA

template <typename T, typename S>
void place_typed(std::vector<T>& dst, int rows, const LaMsgHeader& h, const char* payload) {
    const S* src = reinterpret_cast<const S*>(payload);
    int64_t k = 0;
    int cols = h.nd >= 2 ? h.size[1] : 1;
    for (int i = 0; i < h.size[0]; ++i)
        for (int j = 0; j < cols; ++j) {
            int r = h.start[0] + i, c = h.nd >= 2 ? h.start[1] + j : 0;
            S v;
            memcpy(&v, src + k++, sizeof(S));
            dst[r + static_cast<int64_t>(c) * rows] = cn_cast<T>(v);
        }
}

template <typename T>
void place_gather(std::vector<T>& dst, int rows, const LaMsgHeader& h, const char* payload) {
    cn_dispatch(static_cast<DType>(h.dtype), [&](auto tag) {
        place_typed<T, decltype(tag)>(dst, rows, h, payload);
    });
}

} // namespace

// ---- Root: assemble, factor, scatter ----

template <int N, typename T>
static void la_root_compute(PartitionImpl<N>* partition, ArrayDAGGroup* g, int node_id,
                            const LaSpec& s, LinalgState<N>& st) {
    LaProblem<T> P;
    P.m = s.in[0].rows();
    P.n = s.in[0].cols();
    P.a.assign(s.in[0].size(), T(0));
    if (s.nin > 1) {
        P.b.assign(s.in[1].size(), T(0));
        P.bn = s.in[1].cols();
    }
    for (auto& msg : st.inputs) {
        LaMsgHeader h;
        memcpy(&h, msg.data(), sizeof(h));
        const char* payload = msg.data() + sizeof(h);
        if (h.operand == 0)
            place_gather<T>(P.a, s.in[0].rows(), h, payload);
        else
            place_gather<T>(P.b, s.in[1].rows(), h, payload);
    }
    st.inputs.clear();
    la_mark("place");
    P.R = s.out_rows();
    P.C = s.out_cols();
    P.out.assign(static_cast<size_t>(P.R) * P.C, T(0));
    if (s.op == LA_RESHAPE) {
        int ir = s.in[0].rows(), ic = s.in[0].cols();
        for (int64_t l = 0; l < static_cast<int64_t>(P.out.size()); ++l)
            P.out[(l / P.C) + (l % P.C) * P.R] = P.a[(l / ic) + (l % ic) * ir];
    } else {
#ifdef USE_NVIDIA
        la_compute<T, N>(s, P, partition);
#else
        la_compute<T>(s, P);
#endif
    }
    la_mark("compute");
    if (s.out_nd == 1)
        la_scatter<1, T>(g, node_id, s, P.out);
    else
        la_scatter<2, T>(g, node_id, s, P.out);
    la_mark("scatter");
}

template <int N, typename T>
static void la_write_piece(PartitionImpl<N>* partition, const LaSpec& s, LinalgState<N>& st,
                           const LaMsgHeader& h, const char* payload) {
    T* out = static_cast<T*>(partition->arrays[s.out_name]->data_ptr());
    const T* src = reinterpret_cast<const T*>(payload);
    int64_t k = 0;
    if constexpr (N == 1) {
        for (int i = 0; i < h.size[0]; ++i)
            memcpy(out + (h.start[0] + i - st.mine.start[0]), src + k++, sizeof(T));
    } else if constexpr (N == 2) {
        int tc = st.mine.size(1);
        for (int i = 0; i < h.size[0]; ++i)
            for (int j = 0; j < h.size[1]; ++j) {
                int64_t dst = static_cast<int64_t>(h.start[0] + i - st.mine.start[0]) * tc +
                              (h.start[1] + j - st.mine.start[1]);
                memcpy(out + dst, src + k++, sizeof(T));
            }
    }
    st.out_covered += h.count;
}

/// f(T{}) for the factorization dtypes (float32/64, complex64/128).
template <typename F>
static void la_dispatch(DType dt, F&& f) {
    switch (dt) {
    case DType::FLOAT32: f(float{}); break;
    case DType::FLOAT64: f(double{}); break;
    case DType::COMPLEX64: f(cn_c64{}); break;
    case DType::COMPLEX128: f(cn_c128{}); break;
    default: CkAbort("linalg: unsupported result dtype %d", (int)dt);
    }
}

static LaSpec la_spec_of(DAGNode* node, ArrayDAGGroup* g) {
    for (ASTNode* root : node->ast->roots)
        if (static_cast<Opcode>(root->opcode) == Opcode::LINALG)
            return parse_linalg(root, g->array_meta);
    CkAbort("execute_linalg_node: node %d has no LINALG root", node->id);
    return LaSpec{};
}

#ifdef USE_NVIDIA

// ---- Device path (NVIDIA) ----
// Operand blocks travel as device chunks (CUDA IPC / device messages, at most
// LA_CHUNK bytes each) to chare 0 of the operands' partition, which assembles
// column-major operands in device memory, factors them in place with
// cuSOLVER/cuBLAS and sends row-major result chunks to the output tiles.
// tssvd with all columns in one tile skips the gather: each chare sends the Gram
// matrix of its rows. Lower cholesky of a whole real array is a tiled right-looking
// factorization in place in the result tiles (la_chol_*).

enum LaDevKind : int {
    LA_DK_A = 0,
    LA_DK_B = 1,
    LA_DK_GRAM = 2,
    LA_DK_OUT = 3,
    LA_DK_PANEL = 4,
    LA_DK_TSR = 5,
    LA_DK_TSQ = 6
};

constexpr int64_t LA_CHUNK = int64_t(1) << 26; // largest pooled comm buffer

/// A factored tile Y_xk = L_xk^T (column-major, i.e. L_xk row-major) received in chunks.
struct LaPanel {
    char* p = nullptr;
    bool comm = false; // p is the (single) received chunk
    int64_t got = 0;
};

struct LaDevRoot {
    char* a = nullptr;
    char* b = nullptr;
    char* g = nullptr;
    int lda = 0;
    // distributed cholesky: this chare holds tile (ta, tb) of an nt x nt grid of t x t tiles
    bool dist = false, factored = false, status = false, failed = false;
    int ta = 0, tb = 0, nt = 0, t = 0, n = 0, applied = 0;
    std::map<std::pair<int, int>, LaPanel> pan; // (x, k) -> Y_xk
    // TSQR: this chare's block (reflectors in a, tau; rows [qr0, qr0 + qh)); root's R_i
    bool tsqr = false, tsq_wait = false;
    char* tau = nullptr;
    int qr0 = 0, qh = 0;
    std::map<int, LaDevChunk> parts; // by first row; c0, c1: the sender's chare index
};

static LaDevRoot* la_dev_state(LaDevRoot*& d) {
    if (d == nullptr)
        d = new LaDevRoot();
    return d;
}

namespace {

void* la_dalloc(size_t bytes, cudaStream_t st) {
    void* p = nullptr;
    CT_CUDA_CHECK(cudaMallocAsync(&p, std::max<size_t>(bytes, 1), st));
    return p;
}

void la_dfree(void* p, cudaStream_t st) {
    if (p)
        CT_CUDA_CHECK(cudaFreeAsync(p, st));
}

/// HAPI callback: return received chunks to the comm pool.
void la_free_cb(void* param, void*) {
    auto* v = static_cast<std::vector<void*>*>(param);
    for (void* p : *v)
        comm_free(p);
    delete v;
}

template <int N>
void la_free_after(PartitionImpl<N>* p, std::vector<void*>&& bufs) {
    if (bufs.empty())
        return;
    auto* v = new std::vector<void*>(std::move(bufs));
    CkCallback cb(la_free_cb, v);
    hapiAddCallback(p->compute_stream_raw, &cb);
}

/// The comm stream waits for the work queued so far on the compute stream.
template <int N>
void la_comm_after_compute(PartitionImpl<N>* p) {
    CT_CUDA_CHECK(cudaEventRecord(p->join_event, p->compute_stream_raw));
    CT_CUDA_CHECK(cudaStreamWaitEvent(p->comm_stream_raw, p->join_event, 0));
}

/// dst (h x w, column-major) = T(src[base + i * rs + j * cs]).
template <int N, typename S, typename T>
void la_k_pack(PartitionImpl<N>* p, T* dst, int h, int w, const S* src, int64_t base,
               int64_t rs, int64_t cs) {
    int64_t total = static_cast<int64_t>(h) * w;
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, total), KOKKOS_LAMBDA(int64_t k) {
            int64_t i = k % h, j = k / h;
            dst[k] = cn_cast<T>(src[base + i * rs + j * cs]);
        });
}

/// Keep the upper (or lower) triangle of the rows x cols block at src, zero the rest.
template <int N, typename T>
void la_k_tri(PartitionImpl<N>* p, T* dst, int64_t ldd, const T* src, int64_t lds, int rows,
              int cols, bool upper) {
    int64_t total = static_cast<int64_t>(rows) * cols;
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, total), KOKKOS_LAMBDA(int64_t k) {
            int64_t i = k % rows, j = k / rows;
            bool keep = upper ? i <= j : i >= j;
            dst[i + j * ldd] = keep ? src[i + j * lds] : T(0);
        });
}

/// out[i * stride] = T(w[i]).
template <int N, typename T, typename R>
void la_k_from_real(PartitionImpl<N>* p, T* out, int64_t stride, const R* w, int n) {
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, n), KOKKOS_LAMBDA(int64_t i) { out[i * stride] = T(w[i]); });
}

/// Fill the rows x cols block with `v` if *info != 0.
template <int N, typename T>
void la_k_nan_if(PartitionImpl<N>* p, T* out, int64_t ld, int rows, int cols, const int* info,
                 T v) {
    int64_t total = static_cast<int64_t>(rows) * cols;
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, total), KOKKOS_LAMBDA(int64_t k) {
            if (*info != 0)
                out[(k % rows) + (k / rows) * ld] = v;
        });
}

template <int N, typename T>
void la_k_fill_nan(PartitionImpl<N>* p, T* out, int64_t n) {
    T v = la_nan<T>();
    Kokkos::parallel_for(CT_COMPUTE_POLICY(p, n), KOKKOS_LAMBDA(int64_t i) { out[i] = v; });
}

template <int N, typename T>
void la_k_eye(PartitionImpl<N>* p, T* out, int n) {
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, n), KOKKOS_LAMBDA(int64_t i) { out[i + i * n] = T(1); });
}

template <int N, typename T>
void la_k_add(PartitionImpl<N>* p, T* dst, const T* src, int64_t n) {
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, n), KOKKOS_LAMBDA(int64_t i) { dst[i] = dst[i] + src[i]; });
}

/// Row-major copy of the ir x ic matrix a into the R x C matrix out (both column-major).
template <int N, typename T>
void la_k_reshape(PartitionImpl<N>* p, T* out, int R, int C, const T* a, int ir, int ic) {
    int64_t total = static_cast<int64_t>(R) * C;
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, total), KOKKOS_LAMBDA(int64_t l) {
            out[(l / C) + (l % C) * R] = a[(l / ic) + (l % ic) * ir];
        });
}

int la_root_nd(const LaSpec& s) {
    int rn = 1;
    for (int i = 0; i < s.nin; ++i)
        rn = std::max(rn, s.in[i].nd);
    return rn;
}

/// View of operand i in its array's local coordinates.
template <int N>
ArrayRegion<N> la_view(const LaOperand& o, const std::array<int, 3>& gshape) {
    if (o.region == nullptr || o.region->is_global) {
        std::array<int, N> st{}, e{}, one{};
        for (int d = 0; d < N; ++d) {
            e[d] = gshape[d];
            one[d] = 1;
        }
        return ArrayRegion<N>(st, e, one);
    }
    return *static_cast<ArrayRegion<N>*>(o.region);
}

/// tssvd whose view columns lie in one column of tiles: every chare holding rows holds
/// whole rows, so per-chare Gram matrices sum to A^T A.
/// Number of tiles along dimension d that the (2-D) view of operand 0 spans; 0 if unknown.
int la_tiles_spanned(const LaSpec& s, ArrayDAGGroup* g, int d) {
    auto it = g->array_meta.find(s.in[0].name);
    if (s.in[0].nd != 2 || it == g->array_meta.end() || it->second.tile <= 0)
        return 0;
    ArrayDecomp<2> dec = it->second.template decomp<2>();
    ArrayRegion<2> v = la_view<2>(s.in[0], it->second.global_shape);
    int cnt = s.in[0].shape[d];
    int first = v.start[d] + dec.offset[d];
    int last = v.start[d] + (cnt - 1) * v.step[d] + dec.offset[d];
    return last / dec.tile - first / dec.tile + 1;
}

bool la_use_gram(const LaSpec& s, ArrayDAGGroup* g) {
    return s.op == LA_TSSVD && !getenv("CNUM_LA_GATHER_TSSVD") && la_tiles_spanned(s, g, 1) == 1;
}

template <int N, int M>
void la_dev_send_to(PartitionImpl<N>* p, ArrayDAGGroup* g, const ChareIndex<M>& ci, int node_id,
                    int kind, int name, const int* rd, int64_t bytes, char* buf) {
    int region[6];
    for (int k = 0; k < 3 * M; ++k)
        region[k] = rd[k];
    if constexpr (M == 1)
        device_pack_send<N, 1>(p, g->partition_proxy_1, ci, node_id, LA_DEV_INPUT + kind, name,
                               region, bytes, buf);
    else
        device_pack_send<N, 2>(p, g->partition_proxy_2, ci, node_id, LA_DEV_INPUT + kind, name,
                               region, bytes, buf);
}

/// Send rows [r0, r1) x cols [c0, c1) (in rd) to chare 0 of the rn-D partition.
template <int N>
void la_dev_send_root(PartitionImpl<N>* p, ArrayDAGGroup* g, int rn, int node_id, int kind,
                      int name, const int* rd, int64_t bytes, char* buf) {
    la_comm_after_compute(p);
    if (rn == 1)
        la_dev_send_to<N, 1>(p, g, ChareIndex<1>{{0}}, node_id, kind, name, rd, bytes, buf);
    else
        la_dev_send_to<N, 2>(p, g, ChareIndex<2>{{0, 0}}, node_id, kind, name, rd, bytes, buf);
}

/// Send this chare's block of operand `opi` (as T) to the root, or its Gram matrix.
template <int N, typename S, typename T>
void la_dev_send_block(PartitionImpl<N>* p, ArrayDAGGroup* g, int node_id, const LaSpec& s,
                       int opi, CTArrayBase<N>* arr, bool gram) {
    if constexpr (N > 2) {
        CkAbort("linalg: operands must be 1-D or 2-D");
    } else {
        const LaOperand& o = s.in[opi];
        ArrayRegion<N> view = la_view<N>(o, {arr->global_shape[0],
                                             N > 1 ? arr->global_shape[N - 1] : 1, 1});
        ArrayRegion<N> tile = arr->decomp.chare_region_local(p->nd_index());
        for (int d = 0; d < N; ++d)
            if (tile.size(d) <= 0)
                return;
        auto [blk, ok] = intersect(view, tile);
        if (!ok || blk.size() <= 0)
            return;
        double ts = CkWallTimer();
        cudaStream_t st = p->compute_stream_raw;
        const S* src = static_cast<const S*>(arr->device_data_ptr());
        int64_t tc = N == 2 ? arr->region.size(N - 1) : 1;
        int h = blk.size(0), w = N == 2 ? blk.size(N - 1) : 1;
        int64_t rs = static_cast<int64_t>(view.step[0]) * tc;
        int64_t cs = N == 2 ? view.step[N - 1] : 1;
        int64_t base = static_cast<int64_t>(blk.start[0] - tile.start[0]) * tc;
        int vr0 = (blk.start[0] - view.start[0]) / view.step[0], vc0 = 0;
        if constexpr (N == 2) {
            base += blk.start[1] - tile.start[1];
            vc0 = (blk.start[1] - view.start[1]) / view.step[1];
        }
        int rn = la_root_nd(s);
        constexpr bool same = std::is_same_v<S, T>;
        if (gram) {
            if constexpr (!cn_is_complex_v<T>) {
                LaHandles& hd = la_handles(st);
                int64_t gbytes = static_cast<int64_t>(w) * w * sizeof(T);
                char* gbuf = static_cast<char*>(comm_alloc(gbytes));
                T* G = reinterpret_cast<T*>(gbuf);
                if (same && cs == 1) {
                    // the block is row-major with leading dimension rs: G = B^T B
                    const T* B = reinterpret_cast<const T*>(src) + base;
                    CT_CUBLAS_CHECK(cula::gemm(hd.blas, CUBLAS_OP_N, CUBLAS_OP_T, w, w, h, B,
                                               static_cast<int>(rs), B, static_cast<int>(rs), G,
                                               w));
                } else {
                    T* B = static_cast<T*>(la_dalloc(sizeof(T) * h * static_cast<int64_t>(w), st));
                    la_k_pack<N, S, T>(p, B, h, w, src, base, rs, cs);
                    CT_CUBLAS_CHECK(
                        cula::gemm(hd.blas, CUBLAS_OP_T, CUBLAS_OP_N, w, w, h, B, h, B, h, G, w));
                    la_dfree(B, st);
                }
                int rd[6] = {vr0, vr0 + h, 1, 0, w, 1};
                la_dev_send_root(p, g, rn, node_id, LA_DK_GRAM, s.out_name, rd, gbytes, gbuf);
            }
        } else {
            int64_t esz = sizeof(T);
            int rc = static_cast<int>(std::min<int64_t>(h, std::max<int64_t>(1, LA_CHUNK / esz)));
            for (int r = 0; r < h; r += rc) {
                int hh = std::min(rc, h - r);
                int cc = static_cast<int>(std::max<int64_t>(1, LA_CHUNK / (hh * esz)));
                for (int c = 0; c < w; c += cc) {
                    int ww = std::min(cc, w - c);
                    int64_t bytes = static_cast<int64_t>(hh) * ww * esz;
                    char* buf = static_cast<char*>(comm_alloc(bytes));
                    T* dst = reinterpret_cast<T*>(buf);
                    int64_t off = base + r * rs + c * cs;
                    bool done = false;
                    if constexpr (same) {
                        const T* sp = reinterpret_cast<const T*>(src) + off;
                        if (ww == 1 && rs == 1) {
                            CT_CUDA_CHECK(cudaMemcpyAsync(dst, sp, bytes,
                                                          cudaMemcpyDeviceToDevice, st));
                            done = true;
                        } else if (cs == 1 && rs >= ww) {
                            LaHandles& hd = la_handles(st);
                            CT_CUBLAS_CHECK(cula::geam_t(hd.blas, hh, ww, sp,
                                                         static_cast<int>(rs), dst, hh));
                            done = true;
                        }
                    }
                    if (!done)
                        la_k_pack<N, S, T>(p, dst, hh, ww, src, off, rs, cs);
                    int rd[6] = {vr0 + r, vr0 + r + hh, 1, vc0 + c, vc0 + c + ww, 1};
                    la_dev_send_root(p, g, rn, node_id, opi == 0 ? LA_DK_A : LA_DK_B,
                                     s.out_name, rd, bytes, buf);
                }
            }
        }
        la_prof(node_id).send += CkWallTimer() - ts;
    }
}

/// Root: allocate the column-major operands (QR: in place in the packed result).
template <int N, typename T>
void la_dev_root_alloc(PartitionImpl<N>* p, const LaSpec& s, LinalgState<N>& st, bool gram) {
    cudaStream_t cs = p->compute_stream_raw;
    LaDevRoot* D = la_dev_state(st.dev);
    int m = s.in[0].rows(), n = s.in[0].cols();
    if (D->tsqr)
        return;
    if (gram) {
        D->g = static_cast<char*>(la_dalloc(sizeof(T) * n * static_cast<int64_t>(n), cs));
        CT_CUDA_CHECK(cudaMemsetAsync(D->g, 0, sizeof(T) * n * static_cast<int64_t>(n), cs));
        return;
    }
    int64_t rows = m, cols = n;
    if (s.op == LA_QR && static_cast<int>(s.p[0]) != 1) {
        rows = s.out_rows();
        cols = s.out_cols();
    }
    D->lda = static_cast<int>(rows);
    int64_t bytes = sizeof(T) * rows * cols;
    D->a = static_cast<char*>(la_dalloc(bytes, cs));
    if (rows != m || cols != n)
        CT_CUDA_CHECK(cudaMemsetAsync(D->a, 0, bytes, cs));
    if (s.nin > 1)
        D->b = static_cast<char*>(la_dalloc(sizeof(T) * s.in[1].size(), cs));
}

template <int N, typename T>
void la_dev_place(PartitionImpl<N>* p, const LaSpec& s, LinalgState<N>& st, const LaDevChunk& c) {
    cudaStream_t cs = p->compute_stream_raw;
    LaDevRoot* D = st.dev;
    int hh = c.r1 - c.r0, ww = c.c1 - c.c0;
    if (c.kind == LA_DK_GRAM) {
        la_k_add<N, T>(p, reinterpret_cast<T*>(D->g), reinterpret_cast<const T*>(c.buf),
                       static_cast<int64_t>(ww) * ww);
    } else {
        char* dst = c.kind == LA_DK_A ? D->a : D->b;
        int64_t ld = c.kind == LA_DK_A ? D->lda : s.in[1].rows();
        CT_CUDA_CHECK(cudaMemcpy2DAsync(dst + (c.r0 + c.c0 * ld) * sizeof(T), ld * sizeof(T),
                                        c.buf, hh * sizeof(T), hh * sizeof(T), ww,
                                        cudaMemcpyDeviceToDevice, cs));
    }
    la_free_after(p, std::vector<void*>{c.buf});
    st.in_covered += static_cast<int64_t>(hh) * ww;
}

/// Owner: copy a row-major result chunk into this chare's tile.
template <int N, typename T>
void la_dev_write(PartitionImpl<N>* p, const LaSpec& s, LinalgState<N>& st, const LaDevChunk& c) {
    T* out = static_cast<T*>(p->arrays[s.out_name]->device_data_ptr());
    int hh = c.r1 - c.r0, ww = c.c1 - c.c0;
    int64_t tc = N == 2 ? st.mine.size(N - 1) : 1;
    int64_t off = static_cast<int64_t>(c.r0 - st.mine.start[0]) * tc;
    if constexpr (N == 2)
        off += c.c0 - st.mine.start[1];
    CT_CUDA_CHECK(cudaMemcpy2DAsync(out + off, tc * sizeof(T), c.buf, ww * sizeof(T),
                                    ww * sizeof(T), hh, cudaMemcpyDeviceToDevice,
                                    p->compute_stream_raw));
    la_free_after(p, std::vector<void*>{c.buf});
    st.out_covered += static_cast<int64_t>(hh) * ww;
}

/// Cut rows [r0, r1) x cols [c0, c1) of the result, held column-major at `out` (element
/// (r0, c0), leading dimension ldo), into the output tiles and send row-major chunks.
template <int N, int M, typename T>
void la_dev_scatter(PartitionImpl<N>* p, ArrayDAGGroup* g, int node_id, const LaSpec& s,
                    const T* out, int64_t ldo, int r0 = 0, int r1 = -1, int c0 = 0, int c1 = -1) {
    cudaStream_t cs = p->compute_stream_raw;
    ArrayDecomp<M> decomp = g->array_meta[s.out_name].template decomp<M>();
    std::array<int, M> bs{}, be{}, one{};
    bs[0] = r0;
    be[0] = r1 < 0 ? s.out_shape[0] : r1;
    if constexpr (M == 2) {
        bs[1] = c0;
        be[1] = c1 < 0 ? s.out_shape[1] : c1;
    }
    for (int d = 0; d < M; ++d)
        one[d] = 1;
    ArrayRegion<M> box(bs, be, one);
    int64_t esz = sizeof(T);
    for (auto& [ci, piece_global] : decompose(decomp.to_global(box), decomp)) {
        ArrayRegion<M> piece = decomp.to_local(piece_global);
        if (piece.size() <= 0)
            continue;
        int h = piece.size(0), w = M == 2 ? piece.size(M - 1) : 1;
        int pr0 = piece.start[0], pc0 = M == 2 ? piece.start[M - 1] : 0;
        int rc = static_cast<int>(std::max<int64_t>(1, LA_CHUNK / (w * esz)));
        for (int r = 0; r < h; r += rc) {
            int hh = std::min(rc, h - r);
            int64_t bytes = static_cast<int64_t>(hh) * w * esz;
            char* buf = static_cast<char*>(comm_alloc(bytes));
            const T* src = out + (pr0 + r - r0) + (pc0 - (M == 2 ? c0 : 0)) * ldo;
            if (w == 1) {
                CT_CUDA_CHECK(cudaMemcpyAsync(buf, src, bytes, cudaMemcpyDeviceToDevice, cs));
            } else {
                LaHandles& hd = la_handles(cs);
                CT_CUBLAS_CHECK(cula::geam_t(hd.blas, w, hh, src, static_cast<int>(ldo),
                                             reinterpret_cast<T*>(buf), w));
            }
            int rd[6] = {pr0 + r, pr0 + r + hh, 1, pc0, pc0 + w, 1};
            la_comm_after_compute(p);
            la_dev_send_to<N, M>(p, g, ci, node_id, LA_DK_OUT, s.out_name, rd, bytes, buf);
        }
    }
}

// ---- TSQR (qr 'reduced'/'r' of a real tall matrix whose columns lie in one tile column) ----
// Each chare factors its row block (B_i = Q_i R_i) and sends R_i to the root, which factors
// the stacked R_i (= Q_s R), sends R to the result and Q_s's row slices back; each chare
// then sends its rows Q_i Q_s,i of Q.

bool la_use_tsqr(const LaSpec& s, ArrayDAGGroup* g) {
    int mode = static_cast<int>(s.p[0]);
    int m = s.in[0].rows(), n = s.in[0].cols();
    if (s.op != LA_QR || mode == 2 || s.in[0].nd != 2 || m < n || dtype_is_complex(s.dt))
        return false;
    const char* env = getenv("CNUM_LA_TSQR");
    if (env && atoi(env) == 0)
        return false;
    int64_t esz = s.dt == DType::FLOAT32 ? 4 : 8;
    return static_cast<int64_t>(n) * n * esz <= LA_CHUNK && la_tiles_spanned(s, g, 1) == 1 &&
           la_tiles_spanned(s, g, 0) >= 2;
}

/// This chare's row block: geqrf, R_i to the root; keep the reflectors for Q (reduced).
template <int N, typename S, typename T>
void la_tsqr_local(PartitionImpl<N>* p, ArrayDAGGroup* g, int node_id, const LaSpec& s,
                   CTArrayBase<N>* arr, LaDevRoot*& Dref) {
    if constexpr (N == 2 && !cn_is_complex_v<T>) {
        const LaOperand& o = s.in[0];
        ArrayRegion<2> view = la_view<2>(o, {arr->global_shape[0], arr->global_shape[1], 1});
        ArrayRegion<2> tile = arr->decomp.chare_region_local(p->nd_index());
        if (tile.size(0) <= 0 || tile.size(1) <= 0)
            return;
        auto [blk, ok] = intersect(view, tile);
        if (!ok || blk.size() <= 0)
            return;
        cudaStream_t cs = p->compute_stream_raw;
        LaHandles& h = la_handles(cs);
        int64_t tc = arr->region.size(1);
        int hh = blk.size(0), n = blk.size(1), k = std::min(hh, n);
        int64_t rs = static_cast<int64_t>(view.step[0]) * tc, cstep = view.step[1];
        int64_t base = static_cast<int64_t>(blk.start[0] - tile.start[0]) * tc +
                       (blk.start[1] - tile.start[1]);
        int vr0 = (blk.start[0] - view.start[0]) / view.step[0];
        T* B = static_cast<T*>(la_dalloc(sizeof(T) * hh * static_cast<int64_t>(n), cs));
        const S* src = static_cast<const S*>(arr->device_data_ptr());
        if constexpr (std::is_same_v<S, T>) {
            if (cstep == 1)
                CT_CUBLAS_CHECK(cula::geam_t(h.blas, hh, n, src + base, static_cast<int>(rs), B, hh));
            else
                la_k_pack<N, S, T>(p, B, hh, n, src, base, rs, cstep);
        } else {
            la_k_pack<N, S, T>(p, B, hh, n, src, base, rs, cstep);
        }
        T* tau = static_cast<T*>(la_dalloc(sizeof(T) * k, cs));
        int* info = static_cast<int*>(la_dalloc(sizeof(int), cs));
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::geqrf_bs(h.solver, hh, n, B, hh, &lw));
        T* w = static_cast<T*>(la_dalloc(sizeof(T) * std::max(lw, 1), cs));
        CT_CUSOLVER_CHECK(cula::geqrf(h.solver, hh, n, B, hh, tau, w, lw, info));
        la_dfree(w, cs);
        la_dfree(info, cs);
        int64_t bytes = sizeof(T) * k * static_cast<int64_t>(n);
        char* buf = static_cast<char*>(comm_alloc(bytes));
        la_k_tri<N, T>(p, reinterpret_cast<T*>(buf), k, B, hh, k, n, true);
        auto idx = p->nd_index();
        int rd[6] = {vr0, vr0 + hh, 0, idx[0], idx[1], 0};
        la_dev_send_root(p, g, 2, node_id, LA_DK_TSR, s.out_name, rd, bytes, buf);
        if (static_cast<int>(s.p[0]) == 1) {
            la_dfree(B, cs);
            la_dfree(tau, cs);
            return;
        }
        LaDevRoot* D = la_dev_state(Dref);
        D->tsq_wait = true;
        D->a = reinterpret_cast<char*>(B);
        D->tau = reinterpret_cast<char*>(tau);
        D->qr0 = vr0;
        D->qh = hh;
    }
}

/// Root: all R_i are in; factor the stack, send R and the Q_s slices.
template <int N, typename T>
void la_tsqr_root(PartitionImpl<N>* p, ArrayDAGGroup* g, int node_id, const LaSpec& s,
                  LaDevRoot* D) {
    if constexpr (N == 2 && !cn_is_complex_v<T>) {
        cudaStream_t cs = p->compute_stream_raw;
        LaHandles& h = la_handles(cs);
        int m = s.in[0].rows(), n = s.in[0].cols(), mode = static_cast<int>(s.p[0]);
        int K = 0;
        for (auto& [r, c] : D->parts)
            K += std::min(c.r1 - c.r0, n);
        T* S_ = static_cast<T*>(la_dalloc(sizeof(T) * K * static_cast<int64_t>(n), cs));
        int off = 0;
        for (auto& [r, c] : D->parts) {
            int k = std::min(c.r1 - c.r0, n);
            CT_CUDA_CHECK(cudaMemcpy2DAsync(S_ + off, K * sizeof(T), c.buf, k * sizeof(T),
                                            k * sizeof(T), n, cudaMemcpyDeviceToDevice, cs));
            la_free_after(p, std::vector<void*>{c.buf});
            off += k;
        }
        la_smark(cs, "gathered");
        T* tau = static_cast<T*>(la_dalloc(sizeof(T) * n, cs));
        int* info = static_cast<int*>(la_dalloc(sizeof(int), cs));
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::geqrf_bs(h.solver, K, n, S_, K, &lw));
        T* w = static_cast<T*>(la_dalloc(sizeof(T) * std::max(lw, 1), cs));
        CT_CUSOLVER_CHECK(cula::geqrf(h.solver, K, n, S_, K, tau, w, lw, info));
        la_dfree(w, cs);
        T* R = static_cast<T*>(la_dalloc(sizeof(T) * n * static_cast<int64_t>(n), cs));
        la_k_tri<N, T>(p, R, n, S_, K, n, n, true);
        la_dev_scatter<N, 2, T>(p, g, node_id, s, R, n, mode == 1 ? 0 : m, mode == 1 ? n : m + n,
                                0, n);
        if (mode != 1) {
            CT_CUSOLVER_CHECK(cula::orgqr_bs(h.solver, K, n, n, S_, K, tau, &lw));
            w = static_cast<T*>(la_dalloc(sizeof(T) * std::max(lw, 1), cs));
            CT_CUSOLVER_CHECK(cula::orgqr(h.solver, K, n, n, S_, K, tau, w, lw, info));
            la_dfree(w, cs);
            la_smark(cs, "tsqr");
            off = 0;
            for (auto& [r, c] : D->parts) {
                int k = std::min(c.r1 - c.r0, n);
                int64_t bytes = sizeof(T) * k * static_cast<int64_t>(n);
                char* buf = static_cast<char*>(comm_alloc(bytes));
                CT_CUDA_CHECK(cudaMemcpy2DAsync(buf, k * sizeof(T), S_ + off, K * sizeof(T),
                                                k * sizeof(T), n, cudaMemcpyDeviceToDevice, cs));
                int rd[6] = {c.r0, c.r1, 1, 0, k, 1};
                la_comm_after_compute(p);
                la_dev_send_to<N, 2>(p, g, ChareIndex<2>{{c.c0, c.c1}}, node_id, LA_DK_TSQ,
                                     s.out_name, rd, bytes, buf);
                off += k;
            }
        }
        D->parts.clear();
        la_dfree(R, cs);
        la_dfree(tau, cs);
        la_dfree(info, cs);
        la_dfree(S_, cs);
    }
}

/// This chare: Q_i Q_s,i into the result rows of its block.
template <int N, typename T>
void la_tsqr_finish(PartitionImpl<N>* p, ArrayDAGGroup* g, int node_id, const LaSpec& s,
                    LaDevRoot* D, const LaDevChunk& c) {
    if constexpr (N == 2 && !cn_is_complex_v<T>) {
        cudaStream_t cs = p->compute_stream_raw;
        LaHandles& h = la_handles(cs);
        int n = s.in[0].cols(), hh = D->qh, k = std::min(hh, n);
        T* B = reinterpret_cast<T*>(D->a);
        T* tau = reinterpret_cast<T*>(D->tau);
        int* info = static_cast<int*>(la_dalloc(sizeof(int), cs));
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::orgqr_bs(h.solver, hh, k, k, B, hh, tau, &lw));
        T* w = static_cast<T*>(la_dalloc(sizeof(T) * std::max(lw, 1), cs));
        CT_CUSOLVER_CHECK(cula::orgqr(h.solver, hh, k, k, B, hh, tau, w, lw, info));
        T* Q = static_cast<T*>(la_dalloc(sizeof(T) * hh * static_cast<int64_t>(n), cs));
        CT_CUBLAS_CHECK(cula::gemm(h.blas, CUBLAS_OP_N, CUBLAS_OP_N, hh, n, k, B, hh,
                                   reinterpret_cast<const T*>(c.buf), k, Q, hh));
        la_free_after(p, std::vector<void*>{c.buf});
        la_dev_scatter<N, 2, T>(p, g, node_id, s, Q, hh, D->qr0, D->qr0 + hh, 0, n);
        for (void* q : {static_cast<void*>(Q), static_cast<void*>(w), static_cast<void*>(info),
                        static_cast<void*>(B), static_cast<void*>(tau)})
            la_dfree(q, cs);
        D->a = D->tau = nullptr;
        D->tsq_wait = false;
    }
}

/// Host fallback (svd, lstsq, gathered tssvd): download, factor as before, upload.
template <int N, typename T>
T* la_dev_host_compute(PartitionImpl<N>* p, const LaSpec& s, LaDevRoot* D) {
    cudaStream_t cs = p->compute_stream_raw;
    LaProblem<T> P;
    P.m = s.in[0].rows();
    P.n = s.in[0].cols();
    P.a.resize(s.in[0].size());
    CT_CUDA_CHECK(cudaMemcpy2DAsync(P.a.data(), P.m * sizeof(T), D->a, D->lda * sizeof(T),
                                    P.m * sizeof(T), P.n, cudaMemcpyDeviceToHost, cs));
    if (s.nin > 1) {
        P.b.resize(s.in[1].size());
        P.bn = s.in[1].cols();
        CT_CUDA_CHECK(cudaMemcpyAsync(P.b.data(), D->b, P.b.size() * sizeof(T),
                                      cudaMemcpyDeviceToHost, cs));
    }
    CT_CUDA_CHECK(cudaStreamSynchronize(cs));
    P.R = s.out_rows();
    P.C = s.out_cols();
    P.out.assign(static_cast<size_t>(P.R) * P.C, T(0));
    la_compute<T, N>(s, P, p);
    T* out = static_cast<T*>(la_dalloc(sizeof(T) * P.out.size(), cs));
    CT_CUDA_CHECK(cudaMemcpyAsync(out, P.out.data(), sizeof(T) * P.out.size(),
                                  cudaMemcpyHostToDevice, cs));
    CT_CUDA_CHECK(cudaStreamSynchronize(cs)); // P.out is pageable and goes away
    return out;
}

/// Factor in place on the device; returns the column-major result and its leading
/// dimension. `tmp` collects buffers to free after the scatter.
template <int N, typename T>
T* la_dev_compute(PartitionImpl<N>* p, const LaSpec& s, LaDevRoot* D, int64_t& ldo,
                  std::vector<void*>& tmp) {
    using R = la_real_t<T>;
    cudaStream_t cs = p->compute_stream_raw;
    LaHandles& h = la_handles(cs);
    int m = s.in[0].rows(), n = s.in[0].cols(), k = std::min(m, n);
    T* A = reinterpret_cast<T*>(D->a);
    T* B = reinterpret_cast<T*>(D->b);
    int lda = D->lda;
    int* info = static_cast<int*>(la_dalloc(sizeof(int), cs));
    tmp.push_back(info);
    CT_CUDA_CHECK(cudaMemsetAsync(info, 0, sizeof(int), cs));
    auto work = [&](int lw) {
        T* w = static_cast<T*>(la_dalloc(sizeof(T) * std::max(lw, 1), cs));
        tmp.push_back(w);
        return w;
    };
    la_smark(cs, "gathered");
    ldo = s.out_rows();
    switch (s.op) {
    case LA_CHOLESKY: {
        bool upper = s.p[0] != 0;
        cublasFillMode_t u = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::potrf_bs(h.solver, u, n, A, lda, &lw));
        CT_CUSOLVER_CHECK(cula::potrf(h.solver, u, n, A, lda, work(lw), lw, info));
        la_smark(cs, "factor");
        la_k_tri<N, T>(p, A, lda, A, lda, n, n, upper);
        la_k_nan_if<N, T>(p, A, lda, n, n, info, la_nan<T>());
        ldo = lda;
        return A;
    }
    case LA_QR: {
        int mode = static_cast<int>(s.p[0]);
        int rr = mode == 2 ? m : k, qc = mode == 2 ? m : k;
        T* tau = static_cast<T*>(la_dalloc(sizeof(T) * std::max(k, 1), cs));
        tmp.push_back(tau);
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::geqrf_bs(h.solver, m, n, A, lda, &lw));
        CT_CUSOLVER_CHECK(cula::geqrf(h.solver, m, n, A, lda, tau, work(lw), lw, info));
        la_smark(cs, "geqrf");
        if (mode == 1) {
            T* out = static_cast<T*>(la_dalloc(sizeof(T) * k * static_cast<int64_t>(n), cs));
            tmp.push_back(out);
            la_k_tri<N, T>(p, out, k, A, lda, k, n, true);
            ldo = k;
            return out;
        }
        // packed result: Q in rows [0, m), R in rows [m, m + rr), both in A's buffer
        la_k_tri<N, T>(p, A + m, lda, A, lda, rr, n, true);
        int lq = 0;
        CT_CUSOLVER_CHECK(cula::orgqr_bs(h.solver, m, qc, k, A, lda, tau, &lq));
        CT_CUSOLVER_CHECK(cula::orgqr(h.solver, m, qc, k, A, lda, tau, work(lq), lq, info));
        la_smark(cs, "orgqr");
        if (qc < s.out_cols())
            CT_CUDA_CHECK(cudaMemset2DAsync(A + static_cast<int64_t>(qc) * lda, lda * sizeof(T),
                                            0, m * sizeof(T), s.out_cols() - qc, cs));
        ldo = lda;
        return A;
    }
    case LA_SOLVE:
    case LA_INV:
    case LA_DET:
    case LA_SLOGDET: {
        int* piv = static_cast<int*>(la_dalloc(sizeof(int) * std::max(n, 1), cs));
        tmp.push_back(piv);
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::getrf_bs(h.solver, n, n, A, lda, &lw));
        CT_CUSOLVER_CHECK(cula::getrf(h.solver, n, n, A, lda, work(lw), piv, info));
        la_smark(cs, "getrf");
        if (s.op == LA_DET || s.op == LA_SLOGDET) {
            std::vector<T> diag(n);
            std::vector<int> hp(n);
            CT_CUDA_CHECK(cudaMemcpy2DAsync(diag.data(), sizeof(T), A, (lda + 1) * sizeof(T),
                                            sizeof(T), n, cudaMemcpyDeviceToHost, cs));
            CT_CUDA_CHECK(cudaMemcpyAsync(hp.data(), piv, sizeof(int) * n,
                                          cudaMemcpyDeviceToHost, cs));
            CT_CUDA_CHECK(cudaStreamSynchronize(cs));
            int psign = 1;
            for (int i = 0; i < n; ++i)
                if (hp[i] != i + 1)
                    psign = -psign;
            LaProblem<T> P;
            P.R = s.out_rows();
            P.C = s.out_cols();
            P.out.assign(static_cast<size_t>(P.R) * P.C, T(0));
            slogdet_from_lu(P, diag.data(), 1, n, psign);
            T* out = static_cast<T*>(la_dalloc(sizeof(T) * P.out.size(), cs));
            tmp.push_back(out);
            CT_CUDA_CHECK(cudaMemcpyAsync(out, P.out.data(), sizeof(T) * P.out.size(),
                                          cudaMemcpyHostToDevice, cs));
            CT_CUDA_CHECK(cudaStreamSynchronize(cs));
            return out;
        }
        int nrhs = s.op == LA_INV ? n : s.in[1].cols();
        if (s.op == LA_INV) {
            B = static_cast<T*>(la_dalloc(sizeof(T) * n * static_cast<int64_t>(n), cs));
            tmp.push_back(B);
            CT_CUDA_CHECK(cudaMemsetAsync(B, 0, sizeof(T) * n * static_cast<int64_t>(n), cs));
            la_k_eye<N, T>(p, B, n);
        }
        CT_CUSOLVER_CHECK(cula::getrs(h.solver, n, nrhs, A, lda, piv, B, n, info));
        la_smark(cs, "getrs");
        ldo = n;
        return B;
    }
    case LA_EIGH: {
        bool vecs = s.p[1] != 0;
        cusolverEigMode_t jz = vecs ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
        cublasFillMode_t u = s.p[0] != 0 ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        R* w = static_cast<R*>(la_dalloc(sizeof(R) * std::max(n, 1), cs));
        tmp.push_back(w);
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::syevd_bs(h.solver, jz, u, n, A, lda, w, &lw));
        CT_CUSOLVER_CHECK(cula::syevd(h.solver, jz, u, n, A, lda, w, work(lw), lw, info));
        la_smark(cs, "syevd");
        int R_ = s.out_rows(), C_ = s.out_cols();
        T* out = static_cast<T*>(la_dalloc(sizeof(T) * R_ * static_cast<int64_t>(C_), cs));
        tmp.push_back(out);
        la_k_from_real<N, T, R>(p, out, vecs ? R_ : 1, w, n);
        if (vecs)
            CT_CUDA_CHECK(cudaMemcpy2DAsync(out + 1, R_ * sizeof(T), A, lda * sizeof(T),
                                            n * sizeof(T), n, cudaMemcpyDeviceToDevice, cs));
        la_k_nan_if<N, T>(p, out, R_, R_, C_, info, la_nan<T>());
        ldo = R_;
        return out;
    }
    case LA_SOLVE_TRIANGULAR: {
        bool trans = s.p[0] != 0, lower = s.p[1] != 0, unit = s.p[2] != 0;
        CT_CUBLAS_CHECK(cula::trsm(h.blas, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                                   trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                   unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, n,
                                   s.in[1].cols(), A, lda, B, n));
        la_smark(cs, "trsm");
        ldo = n;
        return B;
    }
    case LA_RESHAPE: {
        int R_ = s.out_rows(), C_ = s.out_cols();
        T* out = static_cast<T*>(la_dalloc(sizeof(T) * R_ * static_cast<int64_t>(C_), cs));
        tmp.push_back(out);
        la_k_reshape<N, T>(p, out, R_, C_, A, m, n);
        ldo = R_;
        return out;
    }
    case LA_TSSVD:
        if (D->g != nullptr) {
            if constexpr (cn_is_complex_v<T>) {
                CkAbort("linalg: tssvd does not support complex input");
            } else {
                T* G = reinterpret_cast<T*>(D->g);
                T* w = static_cast<T*>(la_dalloc(sizeof(T) * n, cs));
                tmp.push_back(w);
                int lw = 0;
                CT_CUSOLVER_CHECK(cula::syevd_bs(h.solver, CUSOLVER_EIG_MODE_VECTOR,
                                                 CUBLAS_FILL_MODE_LOWER, n, G, n, w, &lw));
                CT_CUSOLVER_CHECK(cula::syevd(h.solver, CUSOLVER_EIG_MODE_VECTOR,
                                              CUBLAS_FILL_MODE_LOWER, n, G, n, w, work(lw), lw,
                                              info));
                la_smark(cs, "syevd");
                LaProblem<T> P;
                P.m = m;
                P.n = n;
                P.R = s.out_rows();
                P.C = s.out_cols();
                P.out.assign(static_cast<size_t>(P.R) * P.C, T(0));
                std::vector<T> hw(n), V(static_cast<size_t>(n) * n);
                CT_CUDA_CHECK(cudaMemcpyAsync(hw.data(), w, sizeof(T) * n,
                                              cudaMemcpyDeviceToHost, cs));
                CT_CUDA_CHECK(cudaMemcpyAsync(V.data(), G, sizeof(T) * V.size(),
                                              cudaMemcpyDeviceToHost, cs));
                CT_CUDA_CHECK(cudaStreamSynchronize(cs));
                tssvd_pack(P, hw, V);
                T* out = static_cast<T*>(la_dalloc(sizeof(T) * P.out.size(), cs));
                tmp.push_back(out);
                CT_CUDA_CHECK(cudaMemcpyAsync(out, P.out.data(), sizeof(T) * P.out.size(),
                                              cudaMemcpyHostToDevice, cs));
                CT_CUDA_CHECK(cudaStreamSynchronize(cs));
                return out;
            }
        }
        [[fallthrough]];
    default: {
        T* out = la_dev_host_compute<N, T>(p, s, D);
        tmp.push_back(out);
        la_smark(cs, "host");
        return out;
    }
    }
}

// ---- Distributed cholesky (lower, real, whole operand, result tiled like the operand) ----
// Tile (a, b) works on Y_ab = L_ab^T, column-major, which is L_ab in the row-major tile:
// Y_aa: potrf(upper); Y_ab: trsm by Y_bb^T after the updates Y_ab -= Y_bk^T Y_ak, k < b.

bool la_use_dist(const LaSpec& s, ArrayDAGGroup* g, int& nt, int& t) {
    if (s.op != LA_CHOLESKY || s.p[0] != 0 ||
        (s.dt != DType::FLOAT32 && s.dt != DType::FLOAT64))
        return false;
    const char* env = getenv("CNUM_LA_DIST");
    if (env && atoi(env) == 0)
        return false;
    const LaOperand& o = s.in[0];
    auto ia = g->array_meta.find(o.name), io = g->array_meta.find(s.out_name);
    if (o.nd != 2 || ia == g->array_meta.end() || io == g->array_meta.end())
        return false;
    const auto &ma = ia->second, &mo = io->second;
    int n = s.out_rows();
    if (ma.ndims != 2 || ma.global_shape[0] != n || ma.global_shape[1] != n || ma.tile <= 0 ||
        ma.tile != mo.tile || ma.offset[0] || ma.offset[1] || mo.offset[0] || mo.offset[1])
        return false;
    if (o.region != nullptr && !o.region->is_global) {
        auto* r = static_cast<ArrayRegion<2>*>(o.region);
        for (int d = 0; d < 2; ++d)
            if (r->start[d] != 0 || r->step[d] != 1 || r->size(d) != n)
                return false;
    }
    t = ma.tile;
    nt = (n + t - 1) / t;
    return nt >= 2;
}

int la_tdim(const LaDevRoot* D, int x) { return std::min(D->t, D->n - x * D->t); }

/// flag = 1 if *info != 0 or the diagonal of the n x n tile Y holds a NaN.
template <int N, typename T>
void la_k_bad(PartitionImpl<N>* p, const T* Y, int n, const int* info, int* flag) {
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, n), KOKKOS_LAMBDA(int64_t i) {
            if ((i == 0 && *info != 0) || cn_isnan(Y[i + i * n]))
                *flag = 1;
        });
}

template <int N, typename T>
void la_chol_send(PartitionImpl<N>* p, ArrayDAGGroup* g, int node_id, int name, const T* Y,
                  int64_t count, int x, int k, int ti, int tj) {
    if constexpr (N == 2) {
        cudaStream_t cs = p->compute_stream_raw;
        int64_t per = LA_CHUNK / sizeof(T);
        for (int64_t lo = 0; lo < count; lo += per) {
            int64_t hi = std::min(count, lo + per);
            int64_t bytes = (hi - lo) * sizeof(T);
            char* buf = static_cast<char*>(comm_alloc(bytes));
            CT_CUDA_CHECK(cudaMemcpyAsync(buf, Y + lo, bytes, cudaMemcpyDeviceToDevice, cs));
            int rd[6] = {x, k, 0, static_cast<int>(lo), static_cast<int>(hi), 0};
            la_comm_after_compute(p);
            la_dev_send_to<2, 2>(p, g, ChareIndex<2>{{ti, tj}}, node_id, LA_DK_PANEL, name, rd,
                                 bytes, buf);
        }
    }
}

template <int N, typename T>
void la_chol_recv(PartitionImpl<N>* p, LaDevRoot* D, const LaDevChunk& c) {
    cudaStream_t cs = p->compute_stream_raw;
    LaPanel& P = D->pan[{c.r0, c.r1}];
    int64_t count = static_cast<int64_t>(la_tdim(D, c.r0)) * la_tdim(D, c.r1);
    if (c.c0 == 0 && c.c1 == count) {
        P.p = c.buf;
        P.comm = true;
    } else {
        if (P.p == nullptr)
            P.p = static_cast<char*>(la_dalloc(count * sizeof(T), cs));
        CT_CUDA_CHECK(cudaMemcpyAsync(P.p + c.c0 * sizeof(T), c.buf, (c.c1 - c.c0) * sizeof(T),
                                      cudaMemcpyDeviceToDevice, cs));
        la_free_after(p, std::vector<void*>{c.buf});
    }
    P.got += c.c1 - c.c0;
}

template <int N>
const char* la_chol_ready(LaDevRoot* D, int x, int k) {
    auto it = D->pan.find({x, k});
    if (it == D->pan.end() ||
        it->second.got < static_cast<int64_t>(la_tdim(D, x)) * la_tdim(D, k))
        return nullptr;
    return it->second.p;
}

template <int N>
void la_chol_release(PartitionImpl<N>* p, LaDevRoot* D, int x, int k) {
    auto it = D->pan.find({x, k});
    if (it->second.comm)
        la_free_after(p, std::vector<void*>{it->second.p});
    else
        la_dfree(it->second.p, p->compute_stream_raw);
    D->pan.erase(it);
}

/// Apply the available updates, factor the tile once they are all in, send it on.
template <int N, typename T>
void la_chol_step(PartitionImpl<N>* p, ArrayDAGGroup* g, int node_id, const LaSpec& s,
                  LinalgState<N>& st) {
    if constexpr (!cn_is_complex_v<T>) {
        cudaStream_t cs = p->compute_stream_raw;
        LaDevRoot* D = st.dev;
        la_cur_node = node_id;
        int a = D->ta, b = D->tb, ba = la_tdim(D, a), bb = la_tdim(D, b);
        T* Y = static_cast<T*>(p->arrays[s.out_name]->device_data_ptr());
        LaHandles& h = la_handles(cs);
        for (int k = 0; k < b && !D->factored; ++k) {
            const T* Yak = reinterpret_cast<const T*>(la_chol_ready<N>(D, a, k));
            const T* Ybk = reinterpret_cast<const T*>(la_chol_ready<N>(D, b, k));
            if (Yak == nullptr || Ybk == nullptr)
                continue;
            int bk = la_tdim(D, k);
            if (a == b) {
                CT_CUBLAS_CHECK(cula::syrk_sub_t(h.blas, ba, bk, Yak, bk, Y, ba));
            } else {
                CT_CUBLAS_CHECK(cula::gemm_sub_tn(h.blas, bb, ba, bk, Ybk, bk, Yak, bk, Y, bb));
                la_chol_release(p, D, b, k);
            }
            la_chol_release(p, D, a, k);
            ++D->applied;
        }
        if (D->factored || D->applied < b)
            return;
        int64_t count = static_cast<int64_t>(ba) * bb;
        if (a == b) {
            int* dev = static_cast<int*>(la_dalloc(2 * sizeof(int), cs));
            CT_CUDA_CHECK(cudaMemsetAsync(dev, 0, 2 * sizeof(int), cs));
            int lw = 0;
            CT_CUSOLVER_CHECK(cula::potrf_bs(h.solver, CUBLAS_FILL_MODE_UPPER, ba, Y, ba, &lw));
            T* w = static_cast<T*>(la_dalloc(sizeof(T) * std::max(lw, 1), cs));
            CT_CUSOLVER_CHECK(
                cula::potrf(h.solver, CUBLAS_FILL_MODE_UPPER, ba, Y, ba, w, lw, dev));
            la_dfree(w, cs);
            la_k_tri<N, T>(p, Y, ba, Y, ba, ba, ba, true);
            la_k_nan_if<N, T>(p, Y, ba, ba, ba, dev, la_nan<T>()); // poisons the later tiles
            for (int i = a + 1; i < D->nt; ++i)
                la_chol_send<N, T>(p, g, node_id, s.out_name, Y, count, a, a, i, a);
            if (a == D->nt - 1) {
                la_k_bad<N, T>(p, Y, ba, dev, dev + 1);
                int flag = 0;
                CT_CUDA_CHECK(
                    cudaMemcpyAsync(&flag, dev + 1, sizeof(int), cudaMemcpyDeviceToHost, cs));
                CT_CUDA_CHECK(cudaStreamSynchronize(cs));
                LaMsgHeader hd = {};
                hd.kind = 2;
                hd.count = flag;
                std::vector<char> msg(sizeof(hd));
                memcpy(msg.data(), &hd, sizeof(hd));
                for (int i = 0; i < D->nt; ++i)
                    for (int j = 0; j < D->nt; ++j)
                        la_send_to<2>(g, ChareIndex<2>{{i, j}}, node_id, msg);
            }
            la_dfree(dev, cs);
            la_smark(cs, "potrf");
        } else {
            const T* U = reinterpret_cast<const T*>(la_chol_ready<N>(D, b, b));
            if (U == nullptr)
                return;
            CT_CUBLAS_CHECK(cula::trsm(h.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                       CUBLAS_DIAG_NON_UNIT, bb, ba, U, bb, Y, bb));
            la_chol_release(p, D, b, b);
            for (int j = b + 1; j <= a; ++j)
                la_chol_send<N, T>(p, g, node_id, s.out_name, Y, count, a, b, a, j);
            for (int j = a + 1; j < D->nt; ++j)
                la_chol_send<N, T>(p, g, node_id, s.out_name, Y, count, a, b, j, a);
        }
        D->factored = true;
    }
}

/// Chare (a, b) of the grid: the result tile starts as A's tile (zero above the diagonal).
template <int N, typename S, typename T>
void la_chol_start(PartitionImpl<N>* p, const LaSpec& s, LaDevRoot* D) {
    cudaStream_t cs = p->compute_stream_raw;
    T* Y = static_cast<T*>(p->arrays[s.out_name]->device_data_ptr());
    int64_t count = static_cast<int64_t>(la_tdim(D, D->ta)) * la_tdim(D, D->tb);
    if (D->ta < D->tb) {
        CT_CUDA_CHECK(cudaMemsetAsync(Y, 0, count * sizeof(T), cs));
        D->factored = true;
        return;
    }
    const S* src = static_cast<const S*>(p->arrays[s.in[0].name]->device_data_ptr());
    if constexpr (std::is_same_v<S, T>)
        CT_CUDA_CHECK(cudaMemcpyAsync(Y, src, count * sizeof(T), cudaMemcpyDeviceToDevice, cs));
    else
        la_k_pack<N, S, T>(p, Y, static_cast<int>(count), 1, src, 0, 1, 0);
}

} // namespace

template <int N, typename T>
static void la_dev_root_compute(PartitionImpl<N>* p, ArrayDAGGroup* g, int node_id,
                                const LaSpec& s, LinalgState<N>& st) {
    cudaStream_t cs = p->compute_stream_raw;
    LaDevRoot* D = st.dev;
    la_cur_node = node_id;
    if (D->tsqr) {
        la_tsqr_root<N, T>(p, g, node_id, s, D);
        return;
    }
    std::vector<void*> tmp;
    int64_t ldo = 0;
    T* out = la_dev_compute<N, T>(p, s, D, ldo, tmp);
    la_smark(cs, "compute");
    if (s.out_nd == 1)
        la_dev_scatter<N, 1, T>(p, g, node_id, s, out, ldo);
    else
        la_dev_scatter<N, 2, T>(p, g, node_id, s, out, ldo);
    la_smark(cs, "scatter");
    for (void* q : tmp)
        la_dfree(q, cs);
    la_dfree(D->a, cs);
    la_dfree(D->b, cs);
    la_dfree(D->g, cs);
    D->a = D->b = D->g = nullptr;
}

/// Device path: place stashed chunks, factor once everything arrived; true when done.
template <int N>
static bool la_dev_progress(PartitionImpl<N>* p, ArrayDAGGroup* g, int node_id, const LaSpec& s,
                            LinalgState<N>& st) {
    if (st.dev != nullptr && st.dev->dist) {
        LaDevRoot* D = st.dev;
        for (auto& msg : st.stash) {
            LaMsgHeader h;
            memcpy(&h, msg.data(), sizeof(h));
            D->status = true;
            D->failed = h.count != 0;
        }
        st.stash.clear();
        std::vector<LaDevChunk> chunks;
        chunks.swap(st.dev_stash);
        la_dispatch(s.dt, [&](auto tag) {
            using T = decltype(tag);
            for (auto& c : chunks)
                la_chol_recv<N, T>(p, D, c);
            la_chol_step<N, T>(p, g, node_id, s, st);
            if (D->factored && D->status && D->failed) {
                T* Y = static_cast<T*>(p->arrays[s.out_name]->device_data_ptr());
                int64_t count = static_cast<int64_t>(la_tdim(D, D->ta)) * la_tdim(D, D->tb);
                la_k_fill_nan<N, T>(p, Y, count);
            }
        });
        if (!D->factored || !D->status)
            return false;
        delete D;
        st.dev = nullptr;
        la_report(node_id, "tile");
        return true;
    }
    std::vector<LaDevChunk> chunks;
    chunks.swap(st.dev_stash);
    for (auto& c : chunks)
        la_dispatch(s.dt, [&](auto tag) {
            if (c.kind == LA_DK_OUT) {
                double tw = CkWallTimer();
                la_dev_write<N, decltype(tag)>(p, s, st, c);
                la_prof(node_id).write += CkWallTimer() - tw;
            } else if (c.kind == LA_DK_TSR) {
                st.dev->parts[c.r0] = c;
                st.in_covered += static_cast<int64_t>(c.r1 - c.r0) * s.in[0].cols();
            } else if (c.kind == LA_DK_TSQ) {
                la_tsqr_finish<N, decltype(tag)>(p, g, node_id, s, st.dev, c);
            } else {
                la_dev_place<N, decltype(tag)>(p, s, st, c);
            }
        });
    if (st.root && !st.computed && st.in_covered == st.in_expected) {
        st.computed = true;
        la_dispatch(s.dt, [&](auto tag) {
            la_dev_root_compute<N, decltype(tag)>(p, g, node_id, s, st);
        });
    }
    if (st.root && !st.computed)
        return false;
    if (st.owner && st.out_covered < st.out_expected)
        return false;
    if (st.dev != nullptr && st.dev->tsq_wait)
        return false;
    delete st.dev;
    st.dev = nullptr;
    la_report(node_id, st.root ? "root" : "owner");
    return true;
}

template <int N>
static void la_chol_execute(ArrayDAGExecutorND<N>* ex, PartitionImpl<N>* p, ArrayDAGGroup* g,
                            DAGNode* node, const LaSpec& s, int nt, int t) {
    auto idx = p->nd_index();
    if (idx[0] >= nt || idx[N - 1] >= nt) {
        ex->node_finished(node->id);
        return;
    }
    ArrayDecomp<N> out_decomp = g->array_meta[s.out_name].template decomp<N>();
    ArrayRegion<N> mine = out_decomp.chare_region_local(idx);
    std::array<int, N> gs{}, zeros{}, sizes{}, ones{};
    for (int d = 0; d < N; ++d) {
        gs[d] = s.out_shape[d];
        sizes[d] = mine.size(d);
        ones[d] = 1;
    }
    p->ensure_array(ArrayRegion<N>(zeros, sizes, ones), gs, s.out_name, s.dt, out_decomp, false);
    p->compute_after_others();
    LinalgState<N>& st = ex->la_state[node->id];
    st.node = node;
    st.started = true;
    st.owner = true;
    st.mine = mine;
    auto* D = new LaDevRoot();
    st.dev = D;
    D->dist = true;
    D->ta = idx[0];
    D->tb = idx[N - 1];
    D->nt = nt;
    D->t = t;
    D->n = s.out_rows();
    cn_dispatch(p->arrays[s.in[0].name]->dtype, [&](auto stag) {
        la_dispatch(s.dt, [&](auto ttag) {
            la_chol_start<N, decltype(stag), decltype(ttag)>(p, s, D);
        });
    });
    if (la_dev_progress<N>(p, g, node->id, s, st)) {
        ex->la_state.erase(node->id);
        ex->node_finished(node->id);
    }
}

template <int N>
static void la_dev_execute(ArrayDAGExecutorND<N>* ex, PartitionImpl<N>* p, ArrayDAGGroup* g,
                           DAGNode* node, const LaSpec& s) {
    int nt = 0, t = 0;
    if (N == 2 && la_use_dist(s, g, nt, t)) {
        la_chol_execute<N>(ex, p, g, node, s, nt, t);
        return;
    }
    bool gram = la_use_gram(s, g), tsqr = N == 2 && la_use_tsqr(s, g);
    LaDevRoot* D = nullptr;
    for (int i = 0; i < s.nin; ++i) {
        if (s.in[i].nd != N)
            continue;
        auto it = p->arrays.find(s.in[i].name);
        if (it == p->arrays.end())
            continue;
        cn_dispatch(it->second->dtype, [&](auto stag) {
            la_dispatch(s.dt, [&](auto ttag) {
                using S = decltype(stag);
                using T = decltype(ttag);
                if (tsqr)
                    la_tsqr_local<N, S, T>(p, g, node->id, s, it->second, D);
                else
                    la_dev_send_block<N, S, T>(p, g, node->id, s, i, it->second, gram && i == 0);
            });
        });
    }
    auto nd_idx = p->nd_index();
    bool zero_idx = true;
    for (int d = 0; d < N; ++d)
        zero_idx = zero_idx && nd_idx[d] == 0;
    bool root = zero_idx && N == la_root_nd(s);
    bool owner = false;
    ArrayRegion<N> mine;
    if (N == s.out_nd) {
        ArrayDecomp<N> out_decomp = g->array_meta[s.out_name].template decomp<N>();
        mine = out_decomp.chare_region_local(nd_idx);
        owner = true;
        for (int d = 0; d < N; ++d)
            if (mine.size(d) <= 0)
                owner = false;
        if (owner) {
            std::array<int, N> gs{}, zeros{}, sizes{}, ones{};
            for (int d = 0; d < N; ++d) {
                gs[d] = s.out_shape[d];
                sizes[d] = mine.size(d);
                ones[d] = 1;
            }
            p->ensure_array(ArrayRegion<N>(zeros, sizes, ones), gs, s.out_name, s.dt, out_decomp,
                            false);
            p->compute_after_others();
        }
    }
    if (!root && !owner && D == nullptr) {
        ex->node_finished(node->id);
        return;
    }
    LinalgState<N>& st = ex->la_state[node->id];
    st.dev = D;
    if (root && tsqr)
        la_dev_state(st.dev)->tsqr = true;
    st.node = node;
    st.started = true;
    st.owner = owner;
    st.root = root;
    st.mine = mine;
    st.out_expected = owner ? mine.size() : 0;
    if (root) {
        st.in_expected = 0;
        for (int i = 0; i < s.nin; ++i)
            st.in_expected += s.in[i].size();
        la_dispatch(s.dt, [&](auto tag) { la_dev_root_alloc<N, decltype(tag)>(p, s, st, gram); });
    }
    if (la_dev_progress<N>(p, g, node->id, s, st)) {
        ex->la_state.erase(node->id);
        ex->node_finished(node->id);
    }
}

#endif // USE_NVIDIA

/// Process stashed messages; returns true once this chare is done with the node.
template <int N>
static bool la_progress(PartitionImpl<N>* partition, ArrayDAGGroup* g, int node_id,
                        const LaSpec& s, LinalgState<N>& st) {
    std::vector<std::vector<char>> msgs;
    msgs.swap(st.stash);
    for (auto& msg : msgs) {
        LaMsgHeader h;
        memcpy(&h, msg.data(), sizeof(h));
        if (h.kind == 0) {
            st.in_covered += h.count;
            st.inputs.push_back(std::move(msg));
        } else {
            double tw = CkWallTimer();
            la_dispatch(s.dt, [&](auto tag) {
                la_write_piece<N, decltype(tag)>(partition, s, st, h, msg.data() + sizeof(h));
            });
            la_prof(node_id).write += CkWallTimer() - tw;
        }
    }
    if (st.root && !st.computed && st.in_covered == st.in_expected) {
        st.computed = true;
        la_cur_node = node_id;
        la_mark("gathered");
        la_dispatch(s.dt, [&](auto tag) {
            la_root_compute<N, decltype(tag)>(partition, g, node_id, s, st);
        });
    }
    if (!st.owner || st.out_covered < st.out_expected)
        return false;
    double td = CkWallTimer();
    partition->arrays[s.out_name]->copyToDevice();
    la_prof(node_id).todev += CkWallTimer() - td;
    la_report(node_id, st.root ? "root" : "owner");
    return true;
}

template <int N, typename S>
static void la_send_dispatch(PartitionImpl<N>* partition, ArrayDAGGroup* g, int node_id,
                             const LaSpec& s, int opi, CTArrayBase<N>* arr) {
    la_send_block<N, S>(partition, g, node_id, s, opi, arr);
}

template <int N>
void ArrayDAGExecutorND<N>::execute_linalg_node(DAGNode* node) {
    auto* g = static_cast<ArrayDAGGroup*>(group);
    LaSpec s = la_spec_of(node, g);
    if (la_timing() && (la_prof(node->id).t0 < 0 || la_prof(node->id).done)) {
        la_prof(node->id) = LaProf();
        la_prof(node->id).t0 = CkWallTimer();
    }
    if (s.dt != DType::FLOAT32 && s.dt != DType::FLOAT64 && !dtype_is_complex(s.dt))
        CkAbort("linalg: unsupported result dtype %d", (int)s.dt);
#ifdef USE_NVIDIA
    la_dev_execute<N>(this, partition, g, node, s);
    return;
#endif
    auto nd_idx = partition->nd_index();

    for (int i = 0; i < s.nin; ++i) {
        if (s.in[i].nd != N)
            continue;
        auto it = partition->arrays.find(s.in[i].name);
        if (it == partition->arrays.end())
            continue;
        cn_dispatch(it->second->dtype, [&](auto tag) {
            la_send_dispatch<N, decltype(tag)>(partition, g, node->id, s, i, it->second);
        });
    }

    if (N != s.out_nd) {
        node_finished(node->id);
        return;
    }
    ArrayDecomp<N> out_decomp = g->array_meta[s.out_name].template decomp<N>();
    ArrayRegion<N> mine = out_decomp.chare_region_local(nd_idx);
    bool owner = true;
    for (int d = 0; d < N; ++d)
        if (mine.size(d) <= 0)
            owner = false;
    if (!owner) {
        node_finished(node->id);
        return;
    }
    std::array<int, N> gs{}, zeros{}, sizes{}, ones{};
    for (int d = 0; d < N; ++d) {
        gs[d] = s.out_shape[d];
        sizes[d] = mine.size(d);
        ones[d] = 1;
    }
    partition->ensure_array(ArrayRegion<N>(zeros, sizes, ones), gs, s.out_name, s.dt, out_decomp);

    LinalgState<N>& st = la_state[node->id];
    st.node = node;
    st.started = true;
    st.owner = true;
    st.mine = mine;
    st.out_expected = mine.size();
    bool is_root = true;
    for (int d = 0; d < N; ++d)
        if (nd_idx[d] != 0)
            is_root = false;
    st.root = is_root;
    st.in_expected = 0;
    for (int i = 0; i < s.nin; ++i)
        st.in_expected += s.in[i].size();
    if (la_progress<N>(partition, g, node->id, s, st)) {
        la_state.erase(node->id);
        node_finished(node->id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::on_linalg_msg(int node_id, char* data, int64_t size) {
    LinalgState<N>& st = la_state[node_id];
    st.stash.emplace_back(data, data + size);
    if (!st.started)
        return;
    auto* g = static_cast<ArrayDAGGroup*>(group);
    LaSpec s = la_spec_of(st.node, g);
#ifdef USE_NVIDIA
    if (la_dev_progress<N>(partition, g, node_id, s, st)) {
#else
    if (la_progress<N>(partition, g, node_id, s, st)) {
#endif
        la_state.erase(node_id);
        node_finished(node_id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::on_linalg_device(int node_id, int kind, const int* region,
                                             char* buf) {
#ifdef USE_NVIDIA
    LinalgState<N>& st = la_state[node_id];
    st.dev_stash.push_back({kind, region[0], region[1], N == 2 ? region[3] : 0,
                            N == 2 ? region[4] : 1, buf});
    if (!st.started)
        return;
    auto* g = static_cast<ArrayDAGGroup*>(group);
    LaSpec s = la_spec_of(st.node, g);
    if (la_dev_progress<N>(partition, g, node_id, s, st)) {
        la_state.erase(node_id);
        node_finished(node_id);
    }
#else
    CkAbort("linalg: device chunks need an NVIDIA build");
#endif
}

template <int N>
void PartitionImpl<N>::receive_linalg(int node_id, int64_t size, char* data) {
    executor->on_linalg_msg(node_id, data, size);
}

template void ArrayDAGExecutorND<1>::execute_linalg_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_linalg_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_linalg_node(DAGNode*);

template void ArrayDAGExecutorND<1>::on_linalg_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<2>::on_linalg_msg(int, char*, int64_t);
template void ArrayDAGExecutorND<3>::on_linalg_msg(int, char*, int64_t);

template void ArrayDAGExecutorND<1>::on_linalg_device(int, int, const int*, char*);
template void ArrayDAGExecutorND<2>::on_linalg_device(int, int, const int*, char*);
template void ArrayDAGExecutorND<3>::on_linalg_device(int, int, const int*, char*);

template void PartitionImpl<1>::receive_linalg(int, int64_t, char*);
template void PartitionImpl<2>::receive_linalg(int, int64_t, char*);
template void PartitionImpl<3>::receive_linalg(int, int64_t, char*);

#ifdef USE_NVIDIA
double ct_prewarm_blas() {
    double t0 = CkWallTimer();
    bool la = la_try_create();
    cublasHandle_t h = ct_try_cublas();
    double t1 = CkWallTimer();
    // a first GEMM initializes cuBLAS internals and loads its kernels
    void* p = nullptr;
    if (la && h && cublasSetStream(h, nullptr) == CUBLAS_STATUS_SUCCESS &&
        cudaMalloc(&p, 12 * sizeof(double)) == cudaSuccess) {
        cudaMemset(p, 0, 12 * sizeof(double));
        auto* f = static_cast<float*>(p);
        auto* d = static_cast<double*>(p);
        const float f1 = 1, f0 = 0;
        const double d1 = 1, d0 = 0;
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &f1, f, 2, f + 4, 2, &f0, f + 8, 2);
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &d1, d, 2, d + 4, 2, &d0, d + 8, 2);
        cudaStreamSynchronize(nullptr);
        cudaFree(p);
    }
    cudaGetLastError(); // a failure above must not show up in later error checks
    return 1e3 * (t1 - t0);
}
#endif
