#pragma once

#include "backend.hpp"
#include "dispatch.hpp"

#include <algorithm>
#include <type_traits>

#ifdef USE_KOKKOS
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#else
#include <Eigen/Dense>
#endif

#ifdef USE_NVIDIA
#include <cublas_v2.h>

/// cuBLAS handle of this PE (non-SMP: one PE per process).
inline cublasHandle_t ct_cublas() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
            CkAbort("cublasCreate failed");
        return h;
    }();
    return handle;
}
#endif

/// Row-major C[m x n] = A[m x k] * B[k x n] + beta * C, leading dims lda/ldb/ldc.
/// Under USE_KOKKOS all pointers are device pointers; float/double use cuBLAS on `stream`.
template <typename T>
inline void local_gemm(const T* a, int lda, const T* b, int ldb, T* c, int ldc, int m, int n,
                       int k, T beta
#ifdef USE_NVIDIA
                       ,
                       cudaStream_t stream
#endif
) {
    if (m <= 0 || n <= 0)
        return;
#ifdef USE_KOKKOS
#ifdef USE_NVIDIA
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        // inputs may come from work on the default instance's stream
        Kokkos::DefaultExecutionSpace().fence("ct_gemm inputs");
        // column-major view of the row-major product: C^T = B^T * A^T
        cublasHandle_t h = ct_cublas();
        cublasSetStream(h, stream);
        const T one = T(1);
        cublasStatus_t st;
        if constexpr (std::is_same_v<T, float>)
            st = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, b, ldb, a, lda, &beta, c,
                             ldc);
        else
            st = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, b, ldb, a, lda, &beta, c,
                             ldc);
        if (st != CUBLAS_STATUS_SUCCESS)
            CkAbort("cuBLAS gemm failed (%d)", (int)st);
        return;
    }
#endif
    using SMat = Kokkos::View<T**, Kokkos::LayoutStride, DeviceSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    SMat d_A(const_cast<T*>(a), Kokkos::LayoutStride(m, lda, k, 1));
    SMat d_B(const_cast<T*>(b), Kokkos::LayoutStride(k, ldb, n, 1));
    SMat d_C(c, Kokkos::LayoutStride(m, ldc, n, 1));
#ifdef USE_NVIDIA
    // KokkosBlas runs on the default instance, which is not ordered with `stream`
    cudaStreamSynchronize(stream);
#endif
    KokkosBlas::gemm("N", "N", T(1), d_A, d_B, beta, d_C);
#ifdef USE_NVIDIA
    Kokkos::DefaultExecutionSpace().fence("ct_gemm result");
#endif
#else
    using RMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using CStride = Eigen::Stride<Eigen::Dynamic, 1>;
    Eigen::Map<const RMat, 0, CStride> A_map(a, m, k, CStride(lda, 1));
    Eigen::Map<const RMat, 0, CStride> B_map(b, k, n, CStride(ldb, 1));
    Eigen::Map<RMat, 0, CStride> C_map(c, m, n, CStride(ldc, 1));
    if (beta == T(0))
        C_map.noalias() = A_map * B_map;
    else
        C_map.noalias() += A_map * B_map;
#endif
}

/// Concatenate received vector pieces in index order into one buffer (device memory
/// under Kokkos, like the pieces); free it with free_vector_pieces.
template <typename T, typename Buffers>
inline T* concat_vector_pieces(Buffers& buffers, int& vec_len) {
    std::sort(buffers.begin(), buffers.end(),
              [](auto& a, auto& b) { return a.region.start[0] < b.region.start[0]; });
    vec_len = 0;
    for (auto& rb : buffers)
        vec_len += rb.byte_size / sizeof(T);
#ifdef USE_KOKKOS
    T* out = static_cast<T*>(comm_alloc((int64_t)vec_len * sizeof(T)));
    using U = Kokkos::View<T*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    int64_t off = 0;
    for (auto& rb : buffers) {
        int64_t len = rb.byte_size / sizeof(T);
        Kokkos::deep_copy(U(out + off, len), U(reinterpret_cast<T*>(rb.data), len));
        off += len;
    }
    Kokkos::fence("ct_concat_vector");
#else
    T* out = new T[vec_len];
    int64_t off = 0;
    for (auto& rb : buffers) {
        memcpy(out + off, rb.data, rb.byte_size);
        off += rb.byte_size / sizeof(T);
    }
#endif
    return out;
}

template <typename T>
inline void free_vector_pieces(T* p) {
    if (p == nullptr)
        return;
#ifdef USE_KOKKOS
    comm_free(p);
#else
    delete[] p;
#endif
}

#ifdef USE_KOKKOS
/// Item i: double-precision dot product of row i / chunks over column chunk i % chunks
/// (float32 dot products accumulated in float drift from NumPy's BLAS on long rows).
template <typename T>
struct GemvChunkFunctor {
    const T* mat;
    int64_t ld;
    int cols, chunks, chunk_len;
    const T* vec;
    double* tmp; // per (row, chunk); unused if chunks == 1
    T* res;

    KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
        int r = i / chunks, c0 = (i % chunks) * chunk_len;
        int c1 = c0 + chunk_len < cols ? c0 + chunk_len : cols;
        const T* row = mat + r * ld;
        double s = 0.0;
        for (int c = c0; c < c1; ++c)
            s += static_cast<double>(row[c]) * static_cast<double>(vec[c]);
        if (chunks == 1)
            res[r] = static_cast<T>(s);
        else
            tmp[i] = s;
    }
};

template <typename T>
struct GemvJoinFunctor {
    const double* tmp;
    int chunks;
    T* res;

    KOKKOS_INLINE_FUNCTION void operator()(const int r) const {
        double s = 0.0;
        for (int k = 0; k < chunks; ++k)
            s += tmp[(int64_t)r * chunks + k];
        res[r] = static_cast<T>(s);
    }
};

/// Row-major res[rows] = mat[rows x cols] * vec (device pointers, row stride ld), on `exec`.
template <typename T, typename Exec>
inline void local_gemv(const Exec& exec, const T* mat, int ld, int rows, int cols, const T* vec,
                       T* res) {
    if (rows <= 0)
        return;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        // inputs may come from the default instance's stream; callers send the result
        // from other streams right after, so finish before returning
        Kokkos::DefaultExecutionSpace().fence("ct_gemv inputs");
        int chunks = (int)std::min<int64_t>(
            {std::max<int64_t>(1, 65536 / rows), (cols + 255) / 256, 1024});
        chunks = std::max(chunks, 1);
        int chunk_len = (cols + chunks - 1) / chunks;
        Kokkos::View<double*, DeviceSpace> tmp;
        if (chunks > 1)
            tmp = Kokkos::View<double*, DeviceSpace>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "ct_gemv"), (int64_t)rows * chunks);
        Kokkos::parallel_for("ct_gemv", Kokkos::RangePolicy<Exec>(exec, 0, rows * chunks),
                             GemvChunkFunctor<T>{mat, ld, cols, chunks, chunk_len, vec,
                                                 tmp.data(), res});
        if (chunks > 1)
            Kokkos::parallel_for("ct_gemv_join", Kokkos::RangePolicy<Exec>(exec, 0, rows),
                                 GemvJoinFunctor<T>{tmp.data(), chunks, res});
        exec.fence("ct_gemv");
    } else {
        Kokkos::View<T**, Kokkos::LayoutStride, DeviceSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            d_mat(const_cast<T*>(mat), Kokkos::LayoutStride(rows, ld, cols, 1));
        Kokkos::View<T*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d_vec(
            const_cast<T*>(vec), cols);
        Kokkos::View<T*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d_res(res, rows);
        Kokkos::DefaultExecutionSpace().fence("ct_gemv inputs");
        KokkosBlas::gemv(exec, "N", T(1), d_mat, d_vec, T(0), d_res);
        exec.fence("ct_gemv");
    }
}
#endif

/// SUMMA panel update: C[c_row:c_row+a_rows, c_col:c_col+b_cols] +=
///   A[:, a_col_off:a_col_off+k_size] * B[b_row_off:b_row_off+k_size, :].
/// A is a_rows x a_k_size, B is b_k_size x b_cols, C is c_rows x c_cols, all row-major.
/// Under USE_KOKKOS all three pointers are device pointers.
template <typename T>
inline void panel_gemm_accumulate(T* a_data, int a_rows, int a_k_size, int a_col_off,
                                  T* b_data, int b_k_size, int b_cols, int b_row_off,
                                  int k_size, T* c_data, int c_rows, int c_cols,
                                  int c_row, int c_col
#ifdef USE_NVIDIA
                                  ,
                                  cudaStream_t stream
#endif
) {
    (void)c_rows;
    (void)b_k_size;
    local_gemm<T>(a_data + a_col_off, a_k_size, b_data + (int64_t)b_row_off * b_cols, b_cols,
                  c_data + (int64_t)c_row * c_cols + c_col, c_cols, a_rows, b_cols, k_size, T(1)
#ifdef USE_NVIDIA
                  ,
                  stream
#endif
    );
}

/// Eigen-based gemv: res = mat[:, :actual_cols] * vec.
/// Matrix is row-major with `local_cols` stride.
#ifndef USE_KOKKOS
template <typename T>
inline void eigen_gemv(T* mat_data, int local_rows, int local_cols,
                       T* vec_data, int actual_cols, T* res_data) {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat(mat_data, local_rows, local_cols);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(vec_data, actual_cols);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> res(res_data, local_rows);
    res.noalias() = mat.leftCols(actual_cols) * vec;
}

/// Eigen-based sub-matrix gemv: res = mat[row_offset:row_offset+sub_rows,
///                                      col_offset:col_offset+sub_cols] * vec.
/// full_local_cols is the row stride of the full local matrix (row-major).
template <typename T>
inline void eigen_gemv_sub(T* mat_data, int full_local_cols,
                           int row_offset, int col_offset,
                           int sub_rows, int sub_cols,
                           T* vec_data, T* res_data) {
    typedef Eigen::Stride<Eigen::Dynamic, 1> RowStride;
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
               0, RowStride>
        sub_mat(mat_data + row_offset * full_local_cols + col_offset,
                sub_rows, sub_cols, RowStride(full_local_cols, 1));
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(vec_data, sub_cols);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> res(res_data, sub_rows);
    res.noalias() = sub_mat * vec;
}

/// Eigen-based gemv on a 2D sub-matrix extracted from a 3D tile.
/// The 3D tile has local shape (local_d0, local_d1, local_d2) in row-major order.
/// dropped_dim identifies which dimension is the singleton (0, 1, or 2).
/// dd_local_offset is the local index in the dropped dimension.
/// row_offset/col_offset are local offsets in the remaining 2 dims.
template <typename T>
inline void eigen_gemv_sub_3d(T* mat_data,
                               int local_d0, int local_d1, int local_d2,
                               int dropped_dim, int dd_local_offset,
                               int row_offset, int col_offset,
                               int sub_rows, int sub_cols,
                               T* vec_data, T* res_data) {
    T* base;
    int row_stride, col_stride;

    if (dropped_dim == 0) {
        // Sub-matrix at (dd, row, col): contiguous 2D block
        base = mat_data + dd_local_offset * local_d1 * local_d2
             + row_offset * local_d2 + col_offset;
        row_stride = local_d2;
        col_stride = 1;
    } else if (dropped_dim == 1) {
        // Sub-matrix at (row, dd, col)
        base = mat_data + row_offset * local_d1 * local_d2
             + dd_local_offset * local_d2 + col_offset;
        row_stride = local_d1 * local_d2;
        col_stride = 1;
    } else {
        // Sub-matrix at (row, col, dd)
        base = mat_data + row_offset * local_d1 * local_d2
             + col_offset * local_d2 + dd_local_offset;
        row_stride = local_d1 * local_d2;
        col_stride = local_d2;
    }

    typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> GenStride;
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
               0, GenStride>
        sub_mat(base, sub_rows, sub_cols, GenStride(row_stride, col_stride));
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(vec_data, sub_cols);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> res(res_data, sub_rows);
    res.noalias() = sub_mat * vec;
}

/// Eigen-based dot product: returns a.dot(b) for vectors of length n.
template <typename T>
inline T eigen_dot(T* a_data, T* b_data, int n) {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> a(a_data, n);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> b(b_data, n);
    if constexpr (cn_is_complex_v<T>)
        return a.conjugate().dot(b); // Eigen's dot conjugates a; NumPy's does not
    else
        return a.dot(b);
}

/// Eigen-based sub-matrix gemm: C += A_sub * B_sub (accumulate).
/// Extracts sub-blocks from row-major A and B by offset, accumulates into C.
/// A has row stride a_full_cols, B has row stride b_full_cols.
/// C is contiguous (sub_rows × sub_cols).
template <typename T>
inline void eigen_gemm_sub(T* a_data, int a_full_cols,
                           int a_row_offset, int a_col_offset,
                           int sub_rows, int k_size,
                           T* b_data, int b_full_cols,
                           int b_row_offset, int b_col_offset,
                           int sub_cols,
                           T* c_data) {
    typedef Eigen::Stride<Eigen::Dynamic, 1> RowStride;
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
               0, RowStride>
        A_sub(a_data + a_row_offset * a_full_cols + a_col_offset,
              sub_rows, k_size, RowStride(a_full_cols, 1));
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
               0, RowStride>
        B_sub(b_data + b_row_offset * b_full_cols + b_col_offset,
              k_size, sub_cols, RowStride(b_full_cols, 1));
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        C(c_data, sub_rows, sub_cols);
    C.noalias() += A_sub * B_sub;
}
#endif

/// Round up to the next power of 2 (returns v if already a power of 2).
inline int next_pow2(int v) {
    if (v <= 1)
        return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/// Minimum tile size per dimension to keep tile volume >= ~1M elements.
/// Returns a power of 2.
inline int ct_min_tile(int ndims) {
    switch (ndims) {
    case 1:
        return CT_MIN_TILE_1D;
    case 2:
        return CT_MIN_TILE_2D;
    case 3:
        return CT_MIN_TILE_3D;
    default:
        return 1;
    }
}

/// Get the tile size for array `name` from array_meta.
/// Falls back to ct_min_tile(ndims) if the array has no metadata entry or tile == 0.
/// Use this in all executor communication paths so that per-array tile sizes
/// assigned by compute_decompositions are respected.
inline int array_tile(const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& meta,
                      int name, int ndims) {
    auto it = meta.find(name);
    if (it != meta.end() && it->second.tile > 0)
        return it->second.tile;
    return ct_min_tile(ndims);
}

/// Array whose decomposition defines r_out of extract_regions_nd(): the last
/// SET_REGION root's target if any, else the first non-temp output root.
/// Do not substitute a fresh output's decomposition for a SET_REGION region.
inline int fused_output_decomp_name(DAGNode* node) {
    auto output_name = [](ASTNode* root) {
        if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION && !root->operands.empty() &&
            root->operands[0] != nullptr)
            return root->operands[0]->result_name;
        return root->result_name;
    };
    for (auto it = node->ast->roots.rbegin(); it != node->ast->roots.rend(); ++it)
        if (static_cast<Opcode>((*it)->opcode) == Opcode::SET_REGION)
            return output_name(*it);
    for (ASTNode* root : node->ast->roots) {
        auto opc = static_cast<Opcode>(root->opcode);
        if (opc != Opcode::NOOP && opc != Opcode::CREATE && !root->is_temp)
            return output_name(root);
    }
    return output_name(node->ast->roots[0]);
}

/// Walk the AST to extract input and output regions for communication.
/// Returns true if region operations were found.
/// `meta` resolves the global shape on chares that hold none of the node's arrays.
template <int N>
bool extract_regions_nd(DAGNode* node, std::unordered_map<int, CTArrayBase<N>*>& arrays,
                        ArrayRegion<N>& r_out, std::vector<ArrayRegion<N>>& input_regions,
                        std::vector<int>& input_source_names, std::array<int, N>& global_shape_out,
                        const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>* meta = nullptr);

/// Determine the DType for a DAG node from its AST or the arrays map.
template <int N>
DType determine_dtype(DAGNode* node, std::unordered_map<int, CTArrayBase<N>*>& arrays);

/// Scratch buffer for point-to-point messages: device memory under USE_KOKKOS, host otherwise.
inline char* alloc_buffer(int64_t bytes) {
    bytes = std::max<int64_t>(bytes, 1);
#ifdef USE_KOKKOS
    return static_cast<char*>(Kokkos::kokkos_malloc<DeviceSpace>(bytes));
#else
    return new char[bytes];
#endif
}

inline void free_buffer(char* buf) {
#ifdef USE_KOKKOS
    Kokkos::kokkos_free<DeviceSpace>(buf);
#else
    delete[] buf;
#endif
}

/// Wait for every kernel this chare queued on its compute stream.
template <int N>
void compute_fence(PartitionImpl<N>* partition) {
#if defined(USE_NVIDIA)
    partition->compute_exec.fence();
#elif defined(USE_KOKKOS)
    Kokkos::fence();
#else
    (void)partition;
#endif
}

/// TRANSPOSE receive side (executor_transpose.cpp, N = 2, 3).
template <int N, typename T>
void transpose_place(ArrayDAGGroup* dag_group, PartitionImpl<N>* partition, ASTNode* root,
                     PendingComm<N>* comm);

/// Send matmul result from a 3D partition back to a 1D partition.
template <typename T>
inline void cross_matmul_send_result_3d_to_1d(PartitionImpl<3>* partition, DAGNode* node,
                                              int result_name, const ChareIndex<1>& target_ci,
                                              T* result_data, int result_rows, int row_start,
                                              CProxy_Partition1D& proxy_1d) {
    int64_t byte_size = result_rows * sizeof(T);

    // Encode as 1D region: [row_start, row_start + result_rows)
    int region_data[1 * 3];
    region_data[0] = row_start;
    region_data[1] = row_start + result_rows;
    region_data[2] = 1;

#ifndef NDEBUG
    partition->comm_bytes_sent += byte_size;
#endif

#ifndef USE_KOKKOS
    T* send_buf = new T[result_rows];
    memcpy(send_buf, result_data, byte_size);
    proxy_at<1>(proxy_1d, target_ci)
        .receive_data(node->id, /*input_index=*/0, result_name, 1, region_data, byte_size,
                      reinterpret_cast<char*>(send_buf));
    delete[] send_buf;
#else
    T* send_buf = static_cast<T*>(comm_alloc(byte_size));
    Kokkos::parallel_for(
        CT_COMM_POLICY(partition, result_rows),
        KOKKOS_LAMBDA(int i) { send_buf[i] = result_data[i]; });
    device_pack_send<3, 1>(partition, proxy_1d, target_ci, node->id, 0, result_name, region_data,
                           byte_size, send_buf);
#endif
}

/// Send matmul result from a 2D partition (column 0) back to a 1D partition.
template <typename T>
inline void cross_matmul_send_result_2d_to_1d(PartitionImpl<2>* partition, DAGNode* node,
                                              int result_name, const ChareIndex<1>& target_ci,
                                              T* result_data, int result_rows, int row_start,
                                              CProxy_Partition1D& proxy_1d) {
    int64_t byte_size = result_rows * sizeof(T);

    // Encode as 1D region: [row_start, row_start + result_rows)
    int region_data[1 * 3];
    region_data[0] = row_start;
    region_data[1] = row_start + result_rows;
    region_data[2] = 1;

#ifndef NDEBUG
    partition->comm_bytes_sent += byte_size;
#endif

#ifndef USE_KOKKOS
    T* send_buf = new T[result_rows];
    memcpy(send_buf, result_data, byte_size);
    proxy_at<1>(proxy_1d, target_ci)
        .receive_data(node->id, /*input_index=*/0, result_name, 1, region_data, byte_size,
                      reinterpret_cast<char*>(send_buf));
    delete[] send_buf;
#else
    T* send_buf = static_cast<T*>(comm_alloc(byte_size));
    Kokkos::parallel_for(
        CT_COMM_POLICY(partition, result_rows),
        KOKKOS_LAMBDA(int i) { send_buf[i] = result_data[i]; });
    device_pack_send<2, 1>(partition, proxy_1d, target_ci, node->id, 0, result_name, region_data,
                           byte_size, send_buf);
#endif
}


/// ---------------------------------------------------------------------------
/// Partial dot product sent to chare 0 (carries metadata + value)
/// ---------------------------------------------------------------------------

/// Data contributed by each chare for a REDUCE (dot product) operation.
/// Layout: [node_id, result_name, dtype, pad, value[16]] = 32 bytes.
struct ReduceContrib {
    int node_id;
    int result_name;
    int dtype_int;
    int pad;
    char value[16]; // one element of any BLAS dtype (complex128 included)
};
static_assert(sizeof(ReduceContrib) == 32, "ReduceContrib must be 32 bytes");

/// ---------------------------------------------------------------------------
/// Cross-partition SET_REGION send helpers
/// ---------------------------------------------------------------------------

/// Lattice mapping of a cross-ndims SET_REGION target[tgt_region] = source[src_view].
/// Non-singleton dims are paired in order (k-th lattice point to k-th); other dims
/// are fixed. Coordinates are decomp-global. Sender and receiver both derive from
/// this map, so their message counts agree.
template <int N_src, int N_tgt>
struct CrossSetRegionMap {
    int src_dim[N_tgt];
    int vs[N_src], vstep[N_src], vn[N_src]; // source view lattice
    int ts[N_tgt], tstep[N_tgt], tn[N_tgt]; // target region lattice

    CrossSetRegionMap(Region* src_view_base, const std::array<int, N_src>& src_shape,
                      const std::array<int, N_src>& src_off, Region* tgt_region_base,
                      const std::array<int, N_tgt>& tgt_shape,
                      const std::array<int, N_tgt>& tgt_off) {
        auto* sv = static_cast<ArrayRegion<N_src>*>(src_view_base);
        bool sv_full = (sv == nullptr || src_view_base->is_global);
        for (int d = 0; d < N_src; ++d) {
            vs[d] = (sv_full ? 0 : sv->start[d]) + src_off[d];
            vstep[d] = sv_full ? 1 : sv->step[d];
            vn[d] = sv_full ? src_shape[d] : sv->size(d);
        }
        auto* tr = static_cast<ArrayRegion<N_tgt>*>(tgt_region_base);
        bool tr_full = (tr == nullptr || tgt_region_base->is_global);
        for (int d = 0; d < N_tgt; ++d) {
            ts[d] = (tr_full ? 0 : tr->start[d]) + tgt_off[d];
            tstep[d] = tr_full ? 1 : tr->step[d];
            tn[d] = tr_full ? tgt_shape[d] : tr->size(d);
        }
        int sd = 0;
        for (int td = 0; td < N_tgt; ++td) {
            src_dim[td] = -1;
            if (tn[td] <= 1)
                continue;
            while (sd < N_src && vn[sd] <= 1)
                ++sd;
            if (sd < N_src)
                src_dim[td] = sd++;
        }
    }

    bool is_mapped_src(int sd) const {
        for (int td = 0; td < N_tgt; ++td)
            if (src_dim[td] == sd)
                return true;
        return false;
    }

    /// Target region (global) written from the source chare covering
    /// src_chare (global).  Returns false if that chare holds none of the view.
    bool mapped_region(const ArrayRegion<N_src>& src_chare, ArrayRegion<N_tgt>& out) const {
        int k_lo[N_src] = {}, k_hi[N_src] = {};
        for (int d = 0; d < N_src; ++d) {
            if (!is_mapped_src(d)) {
                // Fixed coordinate: this chare must own it.
                if (vs[d] < src_chare.start[d] || vs[d] >= src_chare.stop[d])
                    return false;
                continue;
            }
            auto ceil_k = [&](int coord) {
                int num = coord - vs[d];
                return num <= 0 ? 0 : (num + vstep[d] - 1) / vstep[d];
            };
            k_lo[d] = ceil_k(src_chare.start[d]);
            k_hi[d] = std::min(vn[d], ceil_k(src_chare.stop[d]));
            if (k_lo[d] >= k_hi[d])
                return false;
        }
        for (int td = 0; td < N_tgt; ++td) {
            int sd = src_dim[td];
            out.step[td] = tstep[td];
            if (sd < 0) {
                out.start[td] = ts[td];
                out.stop[td] = ts[td] + (tn[td] - 1) * tstep[td] + 1;
            } else {
                out.start[td] = ts[td] + k_lo[sd] * tstep[td];
                out.stop[td] = ts[td] + (k_hi[sd] - 1) * tstep[td] + 1;
            }
        }
        return true;
    }

    /// Source coordinate (global) along source dim sd for target point t.
    int src_coord(int sd, const int* t) const {
        for (int td = 0; td < N_tgt; ++td)
            if (src_dim[td] == sd)
                return vs[sd] + ((t[td] - ts[td]) / tstep[td]) * vstep[sd];
        return vs[sd];
    }
};

/// Number of source chares that send a fragment to the target chare owning
/// my_chare (global) — the receiver side of cross_set_region_send.
template <int N_src, int N_tgt>
inline int count_cross_set_region_msgs(const CrossSetRegionMap<N_src, N_tgt>& xmap,
                                       const ArrayDecomp<N_src>& src_decomp,
                                       const ArrayRegion<N_tgt>& my_chare) {
    int lo[N_src], hi[N_src];
    for (int d = 0; d < N_src; ++d) {
        int last = xmap.vs[d] + (std::max(xmap.vn[d], 1) - 1) * xmap.vstep[d];
        lo[d] = xmap.vs[d] / src_decomp.tile;
        hi[d] = last / src_decomp.tile;
    }
    std::array<int, N_src> ci;
    for (int d = 0; d < N_src; ++d)
        ci[d] = lo[d];
    int count = 0;
    while (true) {
        ArrayRegion<N_tgt> mapped;
        auto src_chare = src_decomp.chare_region_global(ci);
        bool nonempty = true;
        for (int d = 0; d < N_src; ++d)
            if (src_chare.stop[d] <= src_chare.start[d])
                nonempty = false;
        if (nonempty && xmap.mapped_region(src_chare, mapped)) {
            auto [ov, has] = intersect(mapped, my_chare);
            (void)ov;
            if (has)
                ++count;
        }
        int d = N_src - 1;
        for (; d >= 0; --d) {
            if (++ci[d] <= hi[d])
                break;
            ci[d] = lo[d];
        }
        if (d < 0)
            break;
    }
    return count;
}

/// Send local source data from PartitionImpl<N_src> to target chares on Partition<N_tgt>.
/// The data is packed and sent with region encoded in N_tgt-dimensional target
/// (decomp-global) coordinates.
template <int N_src, int N_tgt, typename T>
inline void cross_set_region_send(PartitionImpl<N_src>* partition, DAGNode* node, int source_name,
                                  const CrossSetRegionMap<N_src, N_tgt>& xmap,
                                  typename PartitionTraits<N_tgt>::ProxyType& target_proxy,
                                  const ArrayDecomp<N_tgt>& tgt_decomp) {
    auto src_it = partition->arrays.find(source_name);
    if (src_it == partition->arrays.end() || src_it->second->local_size() == 0)
        return;

    auto* src = static_cast<Array<N_src, T>*>(src_it->second);

    auto src_nd = partition->nd_index();
    auto& src_decomp = src->decomp;
    auto src_chare_global = src_decomp.chare_region_global(src_nd);

    // Compute source local region in global coordinates
    std::array<int, N_src> src_cs;
    for (int d = 0; d < N_src; ++d)
        src_cs[d] = src_chare_global.start[d];

    // Target region this chare's part of the source view is written to
    ArrayRegion<N_tgt> mapped_region;
    if (!xmap.mapped_region(src_chare_global, mapped_region))
        return;

    // Decompose into target chares using the target array's decomposition
    auto chare_map = decompose(mapped_region, tgt_decomp);

    // Compute source strides
    std::array<int, N_src> src_strides;
    src_strides[N_src - 1] = 1;
    for (int d = N_src - 2; d >= 0; --d)
        src_strides[d] = src_strides[d + 1] * src->region.size(d + 1);

    for (auto& [ci, cr] : chare_map) {
        // Compute the size of this fragment
        int64_t total_size = cr.size();
        int64_t byte_size = total_size * sizeof(T);

        // Encode region in N_tgt coordinates
        int region_data[N_tgt * 3];
        for (int d = 0; d < N_tgt; ++d) {
            region_data[d * 3 + 0] = cr.start[d];
            region_data[d * 3 + 1] = cr.stop[d];
            region_data[d * 3 + 2] = cr.step[d];
        }

#ifndef NDEBUG
        partition->comm_bytes_sent += byte_size;
#endif

#ifndef USE_KOKKOS
        src->copyToHost();
        T* send_buf = new T[total_size];
        T* src_data = static_cast<T*>(src->data_ptr());

        int idx[N_tgt];
        for (int d = 0; d < N_tgt; ++d)
            idx[d] = cr.start[d];
        for (int64_t flat = 0; flat < total_size; ++flat) {
            int src_flat = 0;
            for (int sd = 0; sd < N_src; ++sd)
                src_flat += (xmap.src_coord(sd, idx) - src_cs[sd]) * src_strides[sd];
            send_buf[flat] = src_data[src_flat];

            // Advance odometer
            for (int d = N_tgt - 1; d >= 0; --d) {
                idx[d] += cr.step[d];
                if (idx[d] < cr.stop[d])
                    break;
                idx[d] = cr.start[d];
            }
        }

        proxy_at<N_tgt>(target_proxy, ci)
            .receive_data(node->id, /*input_index=*/0, source_name, N_tgt, region_data, byte_size,
                          reinterpret_cast<char*>(send_buf));
        delete[] send_buf;
#else
        T* send_buf = static_cast<T*>(comm_alloc(byte_size));
        T* src_device = static_cast<T*>(src->device_data_ptr());

        int dm[N_tgt], cr_start[N_tgt], cr_sizes[N_tgt], cr_step[N_tgt], t_s[N_tgt], t_st[N_tgt];
        int sc[N_src], ss[N_src], v_s[N_src], v_st[N_src];
        for (int d = 0; d < N_tgt; ++d) {
            dm[d] = xmap.src_dim[d];
            cr_start[d] = cr.start[d];
            cr_sizes[d] = cr.size(d);
            cr_step[d] = cr.step[d];
            t_s[d] = xmap.ts[d];
            t_st[d] = xmap.tstep[d];
        }
        for (int d = 0; d < N_src; ++d) {
            sc[d] = src_cs[d];
            ss[d] = src_strides[d];
            v_s[d] = xmap.vs[d];
            v_st[d] = xmap.vstep[d];
        }

        Kokkos::parallel_for(
            CT_COMM_POLICY(partition, total_size), KOKKOS_LAMBDA(int flat_idx) {
                int remaining = flat_idx;
                int tgt_coords[N_tgt];
                for (int d = N_tgt - 1; d >= 0; --d) {
                    tgt_coords[d] = cr_start[d] + (remaining % cr_sizes[d]) * cr_step[d];
                    remaining /= cr_sizes[d];
                }
                int src_flat = 0;
                for (int sd = 0; sd < N_src; ++sd) {
                    int coord = v_s[sd];
                    for (int td = 0; td < N_tgt; ++td) {
                        if (dm[td] == sd) {
                            coord = v_s[sd] + ((tgt_coords[td] - t_s[td]) / t_st[td]) * v_st[sd];
                            break;
                        }
                    }
                    src_flat += (coord - sc[sd]) * ss[sd];
                }
                send_buf[flat_idx] = src_device[src_flat];
            });
        device_pack_send<N_src, N_tgt>(partition, target_proxy, ci, node->id, 0, source_name,
                                       region_data, byte_size, send_buf);
#endif
    }
}
/// Dispatch cross_set_region_send by DType.
template <int N_src, int N_tgt>
inline void dispatch_cross_set_region_send(DType dt, PartitionImpl<N_src>* partition, DAGNode* node,
                                           int source_name,
                                           const CrossSetRegionMap<N_src, N_tgt>& xmap,
                                           typename PartitionTraits<N_tgt>::ProxyType& target_proxy,
                                           const ArrayDecomp<N_tgt>& tgt_decomp) {
    cn_dispatch(dt, [&](auto tag) {
        using T = decltype(tag);
        cross_set_region_send<N_src, N_tgt, T>(partition, node, source_name, xmap,
                                                    target_proxy, tgt_decomp);
    });
}
