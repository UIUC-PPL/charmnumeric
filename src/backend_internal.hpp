#pragma once

#include "backend.hpp"
#include "dispatch.hpp"

#ifdef USE_KOKKOS
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas3_gemm.hpp>
#else
#include <Eigen/Dense>
#endif

/// SUMMA panel update: C[c_row:c_row+a_rows, c_col:c_col+b_cols] +=
///   A[:, a_col_off:a_col_off+k_size] * B[b_row_off:b_row_off+k_size, :].
/// A is a_rows x a_k_size, B is b_k_size x b_cols, C is c_rows x c_cols, all row-major.
/// Under USE_KOKKOS all three pointers are device pointers.
template <typename T>
inline void panel_gemm_accumulate(T* a_data, int a_rows, int a_k_size, int a_col_off,
                                  T* b_data, int b_k_size, int b_cols, int b_row_off,
                                  int k_size, T* c_data, int c_rows, int c_cols,
                                  int c_row, int c_col) {
#ifdef USE_KOKKOS
    using UMat = Kokkos::View<T**, Kokkos::LayoutRight, DeviceSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    UMat d_A_full(a_data, a_rows, a_k_size);
    auto d_A_sub = Kokkos::subview(d_A_full, Kokkos::ALL,
                                   Kokkos::make_pair(a_col_off, a_col_off + k_size));
    UMat d_B_full(b_data, b_k_size, b_cols);
    auto d_B_sub = Kokkos::subview(d_B_full, Kokkos::make_pair(b_row_off, b_row_off + k_size),
                                   Kokkos::ALL);
    UMat d_C_full(c_data, c_rows, c_cols);
    auto d_C_sub = Kokkos::subview(d_C_full, Kokkos::make_pair(c_row, c_row + a_rows),
                                   Kokkos::make_pair(c_col, c_col + b_cols));
    KokkosBlas::gemm("N", "N", T(1), d_A_sub, d_B_sub, T(1), d_C_sub);
#else
    (void)c_rows;
    (void)b_k_size;
    using RMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using CStride = Eigen::Stride<Eigen::Dynamic, 1>;
    Eigen::Map<RMat, 0, CStride> A_map(a_data + a_col_off, a_rows, k_size, CStride(a_k_size, 1));
    Eigen::Map<RMat, 0, CStride> B_map(b_data + b_row_off * b_cols, k_size, b_cols,
                                       CStride(b_cols, 1));
    Eigen::Map<RMat, 0, CStride> C_map(c_data + c_row * c_cols + c_col, a_rows, b_cols,
                                       CStride(c_cols, 1));
    C_map.noalias() += A_map * B_map;
#endif
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

/// Name of the array whose coordinate system/decomposition the output region
/// r_out of extract_regions_nd() refers to.  extract_regions_nd() takes r_out
/// from the last SET_REGION root (in its target's coordinates) when there is
/// one, so its target must also provide the output decomposition; otherwise
/// the first non-temp output root does.  Using a fresh output's decomposition
/// for a SET_REGION region (e.g. a kernel writing CALL[2:4] that also exports
/// an intermediate) mis-sized the outputs and crashed / dropped them.
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
template <int N>
bool extract_regions_nd(DAGNode* node, std::unordered_map<int, CTArrayBase<N>*>& arrays,
                        ArrayRegion<N>& r_out, std::vector<ArrayRegion<N>>& input_regions,
                        std::vector<int>& input_source_names, std::array<int, N>& global_shape_out);

/// Determine the DType for a DAG node from its AST or the arrays map.
template <int N>
DType determine_dtype(DAGNode* node, std::unordered_map<int, CTArrayBase<N>*>& arrays);

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
    T* send_buf = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(byte_size));
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
    T* send_buf = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(byte_size));
    Kokkos::parallel_for(
        CT_COMM_POLICY(partition, result_rows),
        KOKKOS_LAMBDA(int i) { send_buf[i] = result_data[i]; });
    device_pack_send<2, 1>(partition, proxy_1d, target_ci, node->id, 0, result_name, region_data,
                           byte_size, send_buf);
#endif
}


/// ---------------------------------------------------------------------------
/// Custom Charm++ reduction for dot product (carries metadata + value)
/// ---------------------------------------------------------------------------

/// Data contributed by each chare for a REDUCE (dot product) operation.
/// Layout: [node_id, result_name, dtype, pad, value[8]] = 24 bytes.
struct ReduceContrib {
    int node_id;
    int result_name;
    int dtype_int;
    int pad;
    char value[8]; // large enough for float, double, int32_t, int64_t
};
static_assert(sizeof(ReduceContrib) == 24, "ReduceContrib must be 24 bytes");

/// Custom reducer: sums the value field based on dtype, preserving metadata.
CkReductionMsg* reduce_dot_sum(int nMsg, CkReductionMsg** msgs);

/// Global reducer type handle — registered once during init.
extern CkReduction::reducerType reduce_dot_sum_type;

/// Call once to register the custom reducer.
void register_reduce_dot_sum();

/// ---------------------------------------------------------------------------
/// Cross-partition SET_REGION send helpers
/// ---------------------------------------------------------------------------

/// Lattice mapping of a cross-ndims SET_REGION  target[tgt_region] = source[src_view].
///
/// Non-singleton dims of the source view and of the target region are paired
/// in order: the k-th lattice point of source dim src_dim[td] lands on the
/// k-th lattice point of target dim td.  Other dims are fixed (the singleton
/// coordinate of the view / of the target region).  All coordinates are
/// decomp-global (array-local + decomp offset).
///
/// Both the sending (source) and receiving (target) chares derive everything
/// from this one map, so target offsets (e.g. Hin[t, :, ws:] = prev2d), source
/// views (H[t, :, a:b]) and offset decompositions stay consistent with the
/// receiver's expected message count.
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

        // Pack data: iterate over the target fragment lattice, map back to source coords
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
        T* send_buf = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(byte_size));
        T* src_device = static_cast<T*>(src->device_data_ptr());

        // Capture plain arrays for the device lambda
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
    switch (dt) {
    case DType::FLOAT32:
        cross_set_region_send<N_src, N_tgt, float>(partition, node, source_name, xmap,
                                                    target_proxy, tgt_decomp);
        break;
    case DType::FLOAT64:
        cross_set_region_send<N_src, N_tgt, double>(partition, node, source_name, xmap,
                                                     target_proxy, tgt_decomp);
        break;
    case DType::INT32:
        cross_set_region_send<N_src, N_tgt, int32_t>(partition, node, source_name, xmap,
                                                      target_proxy, tgt_decomp);
        break;
    case DType::INT64:
        cross_set_region_send<N_src, N_tgt, int64_t>(partition, node, source_name, xmap,
                                                      target_proxy, tgt_decomp);
        break;
    }
}
