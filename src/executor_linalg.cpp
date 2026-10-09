#include "backend_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef USE_NVIDIA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#else
#include <Eigen/Dense>
#endif

// LINALG (cnp.linalg): operand blocks are gathered to chare 0 of the result's
// partition, factored there (Eigen; cuSOLVER/cuBLAS on NVIDIA) and the result
// tiles are scattered back. Not scalable: the problem must fit on one PE/GPU,
// which factors it alone; gather/scatter messages are staged through host memory.

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

template <typename T>
void fill_nan(std::vector<T>& v) {
    std::fill(v.begin(), v.end(), std::numeric_limits<T>::quiet_NaN());
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

template <typename T>
void slogdet_from_lu(LaProblem<T>& P, const T* lu, int n, int perm_sign) {
    T sign = static_cast<T>(perm_sign), logdet = 0;
    for (int i = 0; i < n; ++i) {
        T d = lu[i + static_cast<int64_t>(i) * n];
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
        slogdet_from_lu(P, LU.data(), n, psign);
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
        put(P, m + 1, 0, Mat(svd.matrixV().transpose()));
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

static LaHandles& la_handles(cudaStream_t st) {
    static LaHandles h;
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
    }

namespace cula {
CT_LA_OVERLOADS(float, S)
CT_LA_OVERLOADS(double, D)
} // namespace cula

static int la_info(const DevBuf<int>& info) { return info.download(1)[0]; }

/// SVD of a host column-major m x n matrix. U is m x (full ? m : k), VT is
/// (full ? n : k) x n. gesvd needs m >= n, so m < n factors A^T instead.
template <typename T>
void gpu_svd(LaHandles& h, cudaStream_t st, const std::vector<T>& A, int m, int n, bool full,
             bool uv, std::vector<T>& U, std::vector<T>& sv, std::vector<T>& VT) {
    bool tr = m < n;
    int mm = tr ? n : m, nn = tr ? m : n, k = nn;
    std::vector<T> At = A;
    if (tr)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                At[j + static_cast<int64_t>(i) * n] = A[i + static_cast<int64_t>(j) * m];
    DevBuf<T> dA(At, st), dS(k, st), dU(uv ? static_cast<size_t>(mm) * (full ? mm : k) : 1, st),
        dVT(uv ? static_cast<size_t>(nn) * nn : 1, st), dRW(k, st);
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
    // A = (A^T)^T = V' S U'^T: U = V'^T^T (m x m), VT = U'^T (uc x n)
    U.assign(static_cast<size_t>(m) * m, T(0));
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i)
            U[i + static_cast<int64_t>(j) * m] = Vd[j + static_cast<int64_t>(i) * nn];
    VT.assign(static_cast<size_t>(uc) * n, T(0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < uc; ++i)
            VT[i + static_cast<int64_t>(j) * uc] = Ud[j + static_cast<int64_t>(i) * mm];
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
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::potrf_bs(h.solver, u, n, dA.p, n, &lw));
        DevBuf<T> dW(std::max(lw, 1), st);
        CT_CUSOLVER_CHECK(cula::potrf(h.solver, u, n, dA.p, n, dW.p, lw, dInfo.p));
        std::vector<T> L = dA.download();
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
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::geqrf_bs(h.solver, m, n, dA.p, m, &lw));
        DevBuf<T> dW(std::max(lw, 1), st);
        CT_CUSOLVER_CHECK(cula::geqrf(h.solver, m, n, dA.p, m, dTau.p, dW.p, lw, dInfo.p));
        std::vector<T> F = dA.download();
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
        std::vector<T> Q = dQ.download();
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
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::getrf_bs(h.solver, n, n, dA.p, n, &lw));
        DevBuf<T> dW(std::max(lw, 1), st);
        CT_CUSOLVER_CHECK(cula::getrf(h.solver, n, n, dA.p, n, dW.p, dPiv.p, dInfo.p));
        if (s.op == LA_DET || s.op == LA_SLOGDET) {
            std::vector<T> LU = dA.download();
            std::vector<int> piv = dPiv.download();
            int psign = 1;
            for (int i = 0; i < n; ++i)
                if (piv[i] != i + 1)
                    psign = -psign;
            slogdet_from_lu(P, LU.data(), n, psign);
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
        P.out = dB.download();
        break;
    }
    case LA_EIGH:
    case LA_TSSVD: {
        bool vecs = s.op == LA_TSSVD || s.p[1] != 0;
        cusolverEigMode_t jz = vecs ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
        cublasFillMode_t u = (s.op == LA_EIGH && s.p[0] != 0) ? CUBLAS_FILL_MODE_UPPER
                                                                : CUBLAS_FILL_MODE_LOWER;
        DevBuf<T> dG(static_cast<size_t>(n) * n, st), dWv(std::max(n, 1), st);
        if (s.op == LA_TSSVD) {
            DevBuf<T> dA(P.a, st);
            CT_CUBLAS_CHECK(cula::gemm(h.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, dA.p, m, dA.p, m,
                                       dG.p, n));
        } else {
            dG.upload(P.a);
        }
        int lw = 0;
        CT_CUSOLVER_CHECK(cula::syevd_bs(h.solver, jz, u, n, dG.p, n, dWv.p, &lw));
        DevBuf<T> dW(std::max(lw, 1), st);
        CT_CUSOLVER_CHECK(cula::syevd(h.solver, jz, u, n, dG.p, n, dWv.p, dW.p, lw, dInfo.p));
        std::vector<T> w = dWv.download(n);
        std::vector<T> V = vecs ? dG.download() : std::vector<T>();
        if (s.op == LA_TSSVD) {
            tssvd_pack(P, w, V);
            break;
        }
        for (int i = 0; i < n; ++i)
            P.out[static_cast<int64_t>(i) * (vecs ? P.R : 1)] = w[i];
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
        std::vector<T> U, sv, VT;
        gpu_svd(h, st, P.a, m, n, full, uv, U, sv, VT);
        if (!uv) {
            for (int i = 0; i < k; ++i)
                P.out[i] = sv[i];
            break;
        }
        int uc = full ? m : k, vr = full ? n : k;
        for (int j = 0; j < uc; ++j)
            for (int i = 0; i < m; ++i)
                P.at(i, j) = U[i + static_cast<int64_t>(j) * m];
        for (int i = 0; i < k; ++i)
            P.at(m, i) = sv[i];
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
            dst[r + static_cast<int64_t>(c) * rows] = static_cast<T>(v);
        }
}

template <typename T>
void place_gather(std::vector<T>& dst, int rows, const LaMsgHeader& h, const char* payload) {
    switch (static_cast<DType>(h.dtype)) {
    case DType::FLOAT32:
        place_typed<T, float>(dst, rows, h, payload);
        break;
    case DType::FLOAT64:
        place_typed<T, double>(dst, rows, h, payload);
        break;
    case DType::INT32:
        place_typed<T, int32_t>(dst, rows, h, payload);
        break;
    case DType::INT64:
        place_typed<T, int64_t>(dst, rows, h, payload);
        break;
    }
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
    if (s.out_nd == 1)
        la_scatter<1, T>(g, node_id, s, P.out);
    else
        la_scatter<2, T>(g, node_id, s, P.out);
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

static LaSpec la_spec_of(DAGNode* node, ArrayDAGGroup* g) {
    for (ASTNode* root : node->ast->roots)
        if (static_cast<Opcode>(root->opcode) == Opcode::LINALG)
            return parse_linalg(root, g->array_meta);
    CkAbort("execute_linalg_node: node %d has no LINALG root", node->id);
    return LaSpec{};
}

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
        } else if (s.dt == DType::FLOAT32) {
            la_write_piece<N, float>(partition, s, st, h, msg.data() + sizeof(h));
        } else {
            la_write_piece<N, double>(partition, s, st, h, msg.data() + sizeof(h));
        }
    }
    if (st.root && !st.computed && st.in_covered == st.in_expected) {
        st.computed = true;
        if (s.dt == DType::FLOAT32)
            la_root_compute<N, float>(partition, g, node_id, s, st);
        else
            la_root_compute<N, double>(partition, g, node_id, s, st);
    }
    if (!st.owner || st.out_covered < st.out_expected)
        return false;
    partition->arrays[s.out_name]->copyToDevice();
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
    if (s.dt != DType::FLOAT32 && s.dt != DType::FLOAT64)
        CkAbort("linalg: result dtype must be float32 or float64");
    auto nd_idx = partition->nd_index();

    for (int i = 0; i < s.nin; ++i) {
        if (s.in[i].nd != N)
            continue;
        auto it = partition->arrays.find(s.in[i].name);
        if (it == partition->arrays.end())
            continue;
        switch (it->second->dtype) {
        case DType::FLOAT32:
            la_send_dispatch<N, float>(partition, g, node->id, s, i, it->second);
            break;
        case DType::FLOAT64:
            la_send_dispatch<N, double>(partition, g, node->id, s, i, it->second);
            break;
        case DType::INT32:
            la_send_dispatch<N, int32_t>(partition, g, node->id, s, i, it->second);
            break;
        case DType::INT64:
            la_send_dispatch<N, int64_t>(partition, g, node->id, s, i, it->second);
            break;
        }
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
    if (la_progress<N>(partition, g, node_id, s, st)) {
        la_state.erase(node_id);
        node_finished(node_id);
    }
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

template void PartitionImpl<1>::receive_linalg(int, int64_t, char*);
template void PartitionImpl<2>::receive_linalg(int, int64_t, char*);
template void PartitionImpl<3>::receive_linalg(int, int64_t, char*);
