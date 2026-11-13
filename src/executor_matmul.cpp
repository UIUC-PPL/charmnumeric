#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <cstring>

template <typename T>
static void cross_matmul_send_vector_slice_1d_to_2d(
    PartitionImpl<1>* partition, DAGNode* node, int vec_name,
    const ChareIndex<2>& target_ci, int local_offset, int send_len, int col_start, int col_end,
    CProxy_Partition2D& proxy_2d) {
    auto vec_it = partition->arrays.find(vec_name);
    if (vec_it == partition->arrays.end())
        return;
    Array<1, T>* vec = static_cast<Array<1, T>*>(vec_it->second);
    int64_t byte_size = send_len * sizeof(T);

    int region_data[2 * 3];
    region_data[0] = col_start;
    region_data[1] = col_end;
    region_data[2] = 1;
    region_data[3] = 0;
    region_data[4] = 1;
    region_data[5] = 1;

#ifndef NDEBUG
    partition->comm_bytes_sent += byte_size;
#endif

#ifdef USE_KOKKOS
    T* send_buf = static_cast<T*>(comm_alloc(byte_size));
    T* src_device = static_cast<T*>(vec->device_data_ptr()) + local_offset;
    Kokkos::parallel_for(
        CT_COMM_POLICY(partition, send_len),
        KOKKOS_LAMBDA(int i) { send_buf[i] = src_device[i]; });
    device_pack_send<1, 2>(partition, proxy_2d, target_ci, node->id, 0, vec_name, region_data,
                           byte_size, send_buf);
#else
    vec->copyToHost();
    T* send_buf = new T[send_len];
    memcpy(send_buf, static_cast<T*>(vec->data_ptr()) + local_offset, byte_size);
    proxy_at<2>(proxy_2d, target_ci)
        .receive_data(node->id, /*input_index=*/0, vec_name, 2, region_data, byte_size,
                      reinterpret_cast<char*>(send_buf));
    delete[] send_buf;
#endif
}

template <typename T>
static void cross_matmul_send_vector_slice_1d_to_3d(
    PartitionImpl<1>* partition, DAGNode* node, int vec_name,
    const ChareIndex<3>& target_ci, int local_offset, int send_len, int col_start, int col_end,
    CProxy_Partition3D& proxy_3d) {
    auto vec_it = partition->arrays.find(vec_name);
    if (vec_it == partition->arrays.end())
        return;
    Array<1, T>* vec = static_cast<Array<1, T>*>(vec_it->second);
    int64_t byte_size = send_len * sizeof(T);

    int region_data[3 * 3];
    region_data[0] = col_start;
    region_data[1] = col_end;
    region_data[2] = 1;
    region_data[3] = 0;
    region_data[4] = 1;
    region_data[5] = 1;
    region_data[6] = 0;
    region_data[7] = 1;
    region_data[8] = 1;

#ifndef NDEBUG
    partition->comm_bytes_sent += byte_size;
#endif

#ifdef USE_KOKKOS
    T* send_buf = static_cast<T*>(comm_alloc(byte_size));
    T* src_device = static_cast<T*>(vec->device_data_ptr()) + local_offset;
    Kokkos::parallel_for(
        CT_COMM_POLICY(partition, send_len),
        KOKKOS_LAMBDA(int i) { send_buf[i] = src_device[i]; });
    device_pack_send<1, 3>(partition, proxy_3d, target_ci, node->id, 0, vec_name, region_data,
                           byte_size, send_buf);
#else
    vec->copyToHost();
    T* send_buf = new T[send_len];
    memcpy(send_buf, static_cast<T*>(vec->data_ptr()) + local_offset, byte_size);
    proxy_at<3>(proxy_3d, target_ci)
        .receive_data(node->id, /*input_index=*/0, vec_name, 3, region_data, byte_size,
                      reinterpret_cast<char*>(send_buf));
    delete[] send_buf;
#endif
}

static void dispatch_cross_matmul_send_vector_slice_3d(
    DType dt, PartitionImpl<1>* partition, DAGNode* node, int vec_name,
    const ChareIndex<3>& target_ci, int local_offset, int send_len, int col_start, int col_end,
    CProxy_Partition3D& proxy_3d) {
    cn_dispatch_blas(dt, [&](auto tag) {
        using T = decltype(tag);
        cross_matmul_send_vector_slice_1d_to_3d<T>(partition, node, vec_name, target_ci,
                                                        local_offset, send_len, col_start, col_end,
                                                        proxy_3d);
    });
}

static void dispatch_cross_matmul_send_vector_slice(
    DType dt, PartitionImpl<1>* partition, DAGNode* node, int vec_name,
    const ChareIndex<2>& target_ci, int local_offset, int send_len, int col_start, int col_end,
    CProxy_Partition2D& proxy_2d) {
    cn_dispatch_blas(dt, [&](auto tag) {
        using T = decltype(tag);
        cross_matmul_send_vector_slice_1d_to_2d<T>(partition, node, vec_name, target_ci,
                                                        local_offset, send_len, col_start, col_end,
                                                        proxy_2d);
    });
}

template <int N>
void ArrayDAGExecutorND<N>::execute_matmul_node(DAGNode* node) {
    // Parse AST to find operand names and detect cross-partition case
    ASTNode* matmul_root = nullptr;
    for (ASTNode* root : node->ast->roots) {
        if (static_cast<Opcode>(root->opcode) == Opcode::MATMUL) {
            matmul_root = root;
            break;
        }
    }
    int mat_name = matmul_root->operands[0]->result_name;
    int vec_name = matmul_root->operands[1]->result_name;
    int result_name = matmul_root->result_name;

    auto* dag_group = static_cast<ArrayDAGGroup*>(group);

    if constexpr (N == 1) {
        // --- 1D partition role in cross-partition matmul ---

        auto nd_idx = partition->nd_index();
        int k = nd_idx[0];

        DType dt = determine_dtype<1>(node, partition->arrays);

        // Look up the matrix global shape from array_meta
        auto meta_it = dag_group->array_meta.find(mat_name);
        if (meta_it == dag_group->array_meta.end()) {
            CkAbort("Cross-partition MATMUL: could not find matrix metadata for name=%d", mat_name);
            return;
        }
        int mat_ndims = meta_it->second.ndims;

        auto vec_it = partition->arrays.find(vec_name);
        bool has_vec = (vec_it != partition->arrays.end() && vec_it->second->local_size() > 0);
        int local_size = has_vec ? vec_it->second->local_size() : 0;

        // Compute vec global size from array_meta
        int vec_s = 0, vec_e = 0;
        auto vec_meta_it = dag_group->array_meta.find(vec_name);
        if (vec_meta_it != dag_group->array_meta.end())
            vec_e = vec_meta_it->second.global_shape[0];

        // Get decomps for global-space translation
        ArrayDecomp<1> vec_decomp;
        if (vec_meta_it != dag_group->array_meta.end())
            vec_decomp = vec_meta_it->second.template decomp<1>();
        auto result_meta_1d = dag_group->array_meta.find(result_name);
        ArrayDecomp<1> result_decomp;
        if (result_meta_1d != dag_group->array_meta.end())
            result_decomp = result_meta_1d->second.template decomp<1>();

        if (mat_ndims == 3) {
            // --- 3D matrix path (dimension-dropped matvec) ---
            int tile_3d = array_tile(dag_group->array_meta, mat_name, 3);

            // operand_regions[0] is a 3D region, operand_regions[1] is 1D vec region
            auto* mat_region_3d = static_cast<ArrayRegion<3>*>(matmul_root->operand_regions[0]);
            auto* vec_region = static_cast<ArrayRegion<1>*>(matmul_root->operand_regions[1]);
            vec_s = vec_region->start[0];
            vec_e = vec_region->stop[0];

            // Identify dropped dim (size-1 range), row dim, col dim
            int dd = matmul_drop_dim(matmul_root, 0);
            int rd = (dd == 0) ? 1 : 0;
            int cd = (dd <= 1) ? 2 : 1;

            int dd_s = mat_region_3d->start[dd], dd_e = mat_region_3d->stop[dd];
            int rd_s = mat_region_3d->start[rd], rd_e = mat_region_3d->stop[rd];
            int cd_s = mat_region_3d->start[cd], cd_e = mat_region_3d->stop[cd];

            // Translate AST regions to global space
            auto mat_decomp_3d = meta_it->second.template decomp<3>();
            dd_s += mat_decomp_3d.offset[dd];
            dd_e += mat_decomp_3d.offset[dd];
            rd_s += mat_decomp_3d.offset[rd];
            rd_e += mat_decomp_3d.offset[rd];
            cd_s += mat_decomp_3d.offset[cd];
            cd_e += mat_decomp_3d.offset[cd];
            vec_s += vec_decomp.offset[0];
            vec_e += vec_decomp.offset[0];

            int slice_rows = rd_e - rd_s;

            // Determine which 3D chares overlap the region
            int first_dd_chare = dd_s / tile_3d;
            int last_dd_chare = (dd_e > 0) ? (dd_e - 1) / tile_3d : 0;
            int first_row_chare = rd_s / tile_3d;
            int last_row_chare = (rd_e > 0) ? (rd_e - 1) / tile_3d : 0;
            int first_col_chare = cd_s / tile_3d;
            int last_col_chare = (cd_e > 0) ? (cd_e - 1) / tile_3d : 0;

            DBG_PRINT("[1D Chare %d] cross_matmul_3d: node=%d dd=%d rd=%d cd=%d "
                      "dd=[%d:%d] rd=[%d:%d] cd=[%d:%d] vec=[%d:%d]\n",
                      partition->index[0], node->id, dd, rd, cd,
                      dd_s, dd_e, rd_s, rd_e, cd_s, cd_e, vec_s, vec_e);

            // Phase 1: Send vector data to relevant 3D chares
            if (has_vec) {
                int my_vec_start = vec_decomp.chare_start_global(0, k);
                int my_vec_end = my_vec_start + local_size;
                int rel_start = std::max(my_vec_start, vec_s);
                int rel_end = std::min(my_vec_end, vec_e);
                if (rel_start < rel_end) {
                    int mat_col_lo = cd_s + (rel_start - vec_s);
                    int mat_col_hi = cd_s + (rel_end - vec_s);

                    int fc = mat_col_lo / tile_3d;
                    int lc = (mat_col_hi - 1) / tile_3d;

                    for (int tc = fc; tc <= lc; tc++) {
                        int chare_col_start = std::max(mat_col_lo, tc * tile_3d);
                        int chare_col_end = std::min(mat_col_hi, (tc + 1) * tile_3d);
                        int send_len = chare_col_end - chare_col_start;
                        int local_offset = (chare_col_start - cd_s + vec_s) - my_vec_start;

                        for (int tr = first_row_chare; tr <= last_row_chare; tr++) {
                            for (int tb = first_dd_chare; tb <= last_dd_chare; tb++) {
                                ChareIndex<3> target_ci;
                                target_ci.idx[dd] = tb;
                                target_ci.idx[rd] = tr;
                                target_ci.idx[cd] = tc;
                                dispatch_cross_matmul_send_vector_slice_3d(
                                    dt, partition, node, vec_name, target_ci,
                                    local_offset, send_len, chare_col_start, chare_col_end,
                                    dag_group->partition_proxy_3);
                            }
                        }
                    }
                }
            }

            // Phase 2: Wait for partial results from 3D chares
            auto r_local_3d = result_decomp.chare_region_local(nd_idx);
            int my_result_start = r_local_3d.start[0];
            int my_result_end = std::min(r_local_3d.stop[0], slice_rows);
            int total_expected = 0;
            if (my_result_start < slice_rows) {
                for (int tb = first_dd_chare; tb <= last_dd_chare; tb++) {
                    for (int tc = first_col_chare; tc <= last_col_chare; tc++) {
                        for (int tr = first_row_chare; tr <= last_row_chare; tr++) {
                            int global_row_lo = std::max(tr * tile_3d, rd_s);
                            int global_row_hi = std::min((tr + 1) * tile_3d, rd_e);
                            int result_lo = global_row_lo - rd_s;
                            int result_hi = global_row_hi - rd_s;
                            if (result_lo < my_result_end && result_hi > my_result_start)
                                total_expected++;
                        }
                    }
                }
            }

            auto pre_it = pending.find(node->id);
            int pre_arrived = 0;
            std::unordered_map<int, std::vector<RemoteBuffer<1>>> pre_buffers;
            if (pre_it != pending.end()) {
                pre_arrived = -(pre_it->second.expected_msgs);
                pre_buffers = std::move(pre_it->second.remote_buffers);
            }
            int remaining = total_expected - pre_arrived;
            pending[node->id] = {node, remaining, {}, std::move(pre_buffers)};
            DBG_PRINT("[1D Chare %d] cross_matmul_3d: total_expected=%d remaining=%d\n",
                      partition->index[0], total_expected, remaining);
            if (remaining <= 0)
                on_comm_done(node->id);

        } else {
            // --- 2D matrix path (existing) ---
            int tile_2d = array_tile(dag_group->array_meta, mat_name, 2);

            int mat_global_rows = meta_it->second.global_shape[0];
            int mat_global_cols = meta_it->second.global_shape[1];

            int mat_rs = 0, mat_re = mat_global_rows;
            int mat_cs = 0, mat_ce = mat_global_cols;
            if (matmul_root->operand_regions.size() >= 2) {
                auto* mat_region = static_cast<ArrayRegion<2>*>(matmul_root->operand_regions[0]);
                auto* vec_region = static_cast<ArrayRegion<1>*>(matmul_root->operand_regions[1]);
                mat_rs = mat_region->start[0];
                mat_re = mat_region->stop[0];
                mat_cs = mat_region->start[1];
                mat_ce = mat_region->stop[1];
                vec_s = vec_region->start[0];
                vec_e = vec_region->stop[0];
            }

            // Translate AST regions to global space
            auto mat_decomp_2d = meta_it->second.template decomp<2>();
            mat_rs += mat_decomp_2d.offset[0];
            mat_re += mat_decomp_2d.offset[0];
            mat_cs += mat_decomp_2d.offset[1];
            mat_ce += mat_decomp_2d.offset[1];
            vec_s += vec_decomp.offset[0];
            vec_e += vec_decomp.offset[0];

            int slice_rows = mat_re - mat_rs;

            int first_row_chare = mat_rs / tile_2d;
            int last_row_chare = (mat_re > 0) ? (mat_re - 1) / tile_2d : 0;
            int first_col_chare = mat_cs / tile_2d;
            int last_col_chare = (mat_ce > 0) ? (mat_ce - 1) / tile_2d : 0;

            DBG_PRINT("[1D Chare %d] cross_matmul: node=%d vec=%d mat=%d result=%d "
                      "slice=[%d:%d,%d:%d] vec=[%d:%d] has_vec=%d\n",
                      partition->index[0], node->id, vec_name, mat_name, result_name,
                      mat_rs, mat_re, mat_cs, mat_ce, vec_s, vec_e, (int)has_vec);

            if (has_vec) {
                int my_vec_start = vec_decomp.chare_start_global(0, k);
                int my_vec_end = my_vec_start + local_size;
                int rel_start = std::max(my_vec_start, vec_s);
                int rel_end = std::min(my_vec_end, vec_e);
                if (rel_start < rel_end) {
                    int mat_col_lo = mat_cs + (rel_start - vec_s);
                    int mat_col_hi = mat_cs + (rel_end - vec_s);

                    int fc = mat_col_lo / tile_2d;
                    int lc = (mat_col_hi - 1) / tile_2d;
                    for (int tc = fc; tc <= lc; tc++) {
                        int chare_col_start = std::max(mat_col_lo, tc * tile_2d);
                        int chare_col_end = std::min(mat_col_hi, (tc + 1) * tile_2d);
                        int send_len = chare_col_end - chare_col_start;
                        int local_offset = (chare_col_start - mat_cs + vec_s) - my_vec_start;

                        for (int tr = first_row_chare; tr <= last_row_chare; tr++) {
                            ChareIndex<2> target_ci;
                            target_ci.idx[0] = tr;
                            target_ci.idx[1] = tc;
                            dispatch_cross_matmul_send_vector_slice(
                                dt, partition, node, vec_name, target_ci,
                                local_offset, send_len, chare_col_start, chare_col_end,
                                dag_group->partition_proxy_2);
                        }
                    }
                }
            }

            auto r_local_2d = result_decomp.chare_region_local(nd_idx);
            int my_result_start = r_local_2d.start[0];
            int my_result_end = std::min(r_local_2d.stop[0], slice_rows);
            int total_expected = 0;
            if (my_result_start < slice_rows) {
                for (int tc = first_col_chare; tc <= last_col_chare; tc++) {
                    for (int tr = first_row_chare; tr <= last_row_chare; tr++) {
                        int global_row_lo = std::max(tr * tile_2d, mat_rs);
                        int global_row_hi = std::min((tr + 1) * tile_2d, mat_re);
                        int result_lo = global_row_lo - mat_rs;
                        int result_hi = global_row_hi - mat_rs;
                        if (result_lo < my_result_end && result_hi > my_result_start)
                            total_expected++;
                    }
                }
            }

            auto pre_it = pending.find(node->id);
            int pre_arrived = 0;
            std::unordered_map<int, std::vector<RemoteBuffer<1>>> pre_buffers;
            if (pre_it != pending.end()) {
                pre_arrived = -(pre_it->second.expected_msgs);
                pre_buffers = std::move(pre_it->second.remote_buffers);
            }
            int remaining = total_expected - pre_arrived;
            pending[node->id] = {node, remaining, {}, std::move(pre_buffers)};
            DBG_PRINT("[1D Chare %d] cross_matmul: node=%d total_expected=%d pre_arrived=%d remaining=%d\n",
                      partition->index[0], node->id, total_expected, pre_arrived, remaining);
            if (remaining <= 0)
                on_comm_done(node->id);
        }

    } else if constexpr (N == 2) {
        int tile_1d = array_tile(dag_group->array_meta, vec_name, 1);

        auto nd_idx = partition->nd_index();


        auto mat_it = partition->arrays.find(mat_name);
        if (mat_it == partition->arrays.end()) {
            node_finished(node->id);
            return;
        }
        int mat_global_rows = mat_it->second->global_shape[0];
        int mat_global_cols = mat_it->second->global_shape[1];

        // Extract slice regions if present
        int mat_rs = 0, mat_re = mat_global_rows;
        int mat_cs = 0, mat_ce = mat_global_cols;
        int vec_s = 0, vec_e = mat_ce - mat_cs; // default: full range
        if (matmul_root->operand_regions.size() >= 2) {
            auto* mat_region = static_cast<ArrayRegion<2>*>(matmul_root->operand_regions[0]);
            auto* vec_region = static_cast<ArrayRegion<1>*>(matmul_root->operand_regions[1]);
            mat_rs = mat_region->start[0];
            mat_re = mat_region->stop[0];
            mat_cs = mat_region->start[1];
            mat_ce = mat_region->stop[1];
            vec_s = vec_region->start[0];
            vec_e = vec_region->stop[0];
        }

        // Translate AST regions to global space
        auto mat_decomp = mat_it->second->decomp;
        mat_rs += mat_decomp.offset[0];
        mat_re += mat_decomp.offset[0];
        mat_cs += mat_decomp.offset[1];
        mat_ce += mat_decomp.offset[1];
        auto vec_decomp_1d = dag_group->array_meta[vec_name].template decomp<1>();
        vec_s += vec_decomp_1d.offset[0];
        vec_e += vec_decomp_1d.offset[0];

        // Check if this 2D chare's tile overlaps the matrix slice
        auto mat_chare = mat_decomp.chare_region_global(nd_idx);
        int row_lo = std::max(mat_chare.start[0], mat_rs);
        int row_hi = std::min(mat_chare.stop[0], mat_re);
        int col_lo = std::max(mat_chare.start[1], mat_cs);
        int col_hi = std::min(mat_chare.stop[1], mat_ce);
        if (row_lo >= row_hi || col_lo >= col_hi) {
            node_finished(node->id);
            return;
        }

        // Count how many vector messages to expect.
        int vec_idx_start = vec_s + (col_lo - mat_cs);
        int vec_idx_end = vec_s + (col_hi - mat_cs);
        int first_vec_chare = vec_idx_start / tile_1d;
        int last_vec_chare = (vec_idx_end - 1) / tile_1d;
        int vec_msgs = last_vec_chare - first_vec_chare + 1;

        auto pre_it = pending.find(node->id);
        int pre_arrived = 0;
        std::unordered_map<int, std::vector<RemoteBuffer<N>>> pre_buffers;
        if (pre_it != pending.end()) {
            pre_arrived = -(pre_it->second.expected_msgs);
            pre_buffers = std::move(pre_it->second.remote_buffers);
        }
        int remaining = vec_msgs - pre_arrived;
        pending[node->id] = {node, remaining, {}, std::move(pre_buffers)};
        if (pre_arrived > 0 || remaining <= 0) {
            DBG_PRINT("[2D Chare %d] matmul node=%d: pre_arrived=%d remaining=%d vec_msgs=%d\n",
                      partition->index[0], node->id, pre_arrived, remaining, vec_msgs);
        }

        if (remaining <= 0)
            on_comm_done(node->id);

    } else if constexpr (N == 3) {
        // --- 3D partition role in cross-partition matmul (dimension-dropped) ---
        int tile_1d = array_tile(dag_group->array_meta, vec_name, 1);

        auto nd_idx = partition->nd_index();


        auto mat_it = partition->arrays.find(mat_name);
        if (mat_it == partition->arrays.end()) {
            node_finished(node->id);
            return;
        }

        // operand_regions[0] is a 3D region
        auto* mat_region_3d = static_cast<ArrayRegion<3>*>(matmul_root->operand_regions[0]);
        auto* vec_region = static_cast<ArrayRegion<1>*>(matmul_root->operand_regions[1]);

        // Identify dropped dim (size-1 range), row dim, col dim
        int dd = matmul_drop_dim(matmul_root, 0);
        int rd = (dd == 0) ? 1 : 0;
        int cd = (dd <= 1) ? 2 : 1;

        int dd_s = mat_region_3d->start[dd], dd_e = mat_region_3d->stop[dd];
        int rd_s = mat_region_3d->start[rd], rd_e = mat_region_3d->stop[rd];
        int cd_s = mat_region_3d->start[cd], cd_e = mat_region_3d->stop[cd];
        int vec_s = vec_region->start[0], vec_e = vec_region->stop[0];

        // Translate AST regions to global space
        auto mat_decomp = mat_it->second->decomp;
        dd_s += mat_decomp.offset[dd];
        dd_e += mat_decomp.offset[dd];
        rd_s += mat_decomp.offset[rd];
        rd_e += mat_decomp.offset[rd];
        cd_s += mat_decomp.offset[cd];
        cd_e += mat_decomp.offset[cd];
        auto vec_decomp_1d = dag_group->array_meta[vec_name].template decomp<1>();
        vec_s += vec_decomp_1d.offset[0];
        vec_e += vec_decomp_1d.offset[0];

        // Check if this 3D chare overlaps the region in ALL dims
        auto mat_chare = mat_decomp.chare_region_global(nd_idx);
        int dd_lo = std::max(mat_chare.start[dd], dd_s);
        int dd_hi = std::min(mat_chare.stop[dd], dd_e);
        int row_lo = std::max(mat_chare.start[rd], rd_s);
        int row_hi = std::min(mat_chare.stop[rd], rd_e);
        int col_lo = std::max(mat_chare.start[cd], cd_s);
        int col_hi = std::min(mat_chare.stop[cd], cd_e);
        if (dd_lo >= dd_hi || row_lo >= row_hi || col_lo >= col_hi) {
            node_finished(node->id);
            return;
        }

        // Count how many vector messages to expect
        int vec_idx_start = vec_s + (col_lo - cd_s);
        int vec_idx_end = vec_s + (col_hi - cd_s);
        int first_vec_chare = vec_idx_start / tile_1d;
        int last_vec_chare = (vec_idx_end - 1) / tile_1d;
        int vec_msgs = last_vec_chare - first_vec_chare + 1;

        auto pre_it = pending.find(node->id);
        int pre_arrived = 0;
        std::unordered_map<int, std::vector<RemoteBuffer<N>>> pre_buffers;
        if (pre_it != pending.end()) {
            pre_arrived = -(pre_it->second.expected_msgs);
            pre_buffers = std::move(pre_it->second.remote_buffers);
        }
        int remaining = vec_msgs - pre_arrived;
        pending[node->id] = {node, remaining, {}, std::move(pre_buffers)};
        DBG_PRINT("[3D Chare %d] matmul node=%d: dd=%d rd=%d cd=%d "
                  "dd_lo=%d dd_hi=%d row_lo=%d row_hi=%d col_lo=%d col_hi=%d "
                  "vec_msgs=%d remaining=%d\n",
                  partition->index[0], node->id, dd, rd, cd,
                  dd_lo, dd_hi, row_lo, row_hi, col_lo, col_hi,
                  vec_msgs, remaining);

        if (remaining <= 0)
            on_comm_done(node->id);

    } else {
        CkAbort("MATMUL only supported for 1D, 2D, and 3D partitions (N=%d)", N);
    }
}

template void ArrayDAGExecutorND<1>::execute_matmul_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_matmul_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_matmul_node(DAGNode*);
