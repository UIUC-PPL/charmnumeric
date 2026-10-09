#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <cstring>

#ifdef USE_KOKKOS
#include <KokkosBlas2_gemv.hpp>
#endif

/// Collect memrefLeaves, broadcastLeaves, and outputRoots from a DAG node's AST.
static void collect_leaves_and_outputs(DAGNode* node, std::vector<ASTNode*>& memrefLeaves,
                                       std::vector<ASTNode*>& outputRoots,
                                       std::vector<ASTNode*>& broadcastLeaves) {
    std::unordered_set<int> internalResults;
    for (ASTNode* root : node->ast->roots)
        internalResults.insert(root->result_name);

    std::unordered_set<int> seenBroadcasts;

    for (ASTNode* root : node->ast->roots) {
        auto root_opc = static_cast<Opcode>(root->opcode);

        for (int op_idx = 0; op_idx < (int)root->operands.size(); ++op_idx) {
            ASTNode* operand = root->operands[op_idx];
            if (!operand)
                continue;
            if (internalResults.count(operand->result_name))
                continue;
            if (root_opc == Opcode::SET_REGION && op_idx == 0)
                continue;
            auto opc = static_cast<Opcode>(operand->opcode);
            if ((opc == Opcode::NOOP || opc == Opcode::CREATE) && !operand->is_scalar) {
                if (operand->is_broadcast) {
                    if (seenBroadcasts.insert(operand->result_name).second)
                        broadcastLeaves.push_back(operand);
                } else {
                    memrefLeaves.push_back(operand);
                }
            }
        }
    }

    for (ASTNode* root : node->ast->roots) {
        auto opc = static_cast<Opcode>(root->opcode);
        if (opc != Opcode::NOOP && opc != Opcode::CREATE && !root->is_temp)
            outputRoots.push_back(root);
    }
}

// ---- Unified execute_node_nd ----

/// Build a MemRef descriptor for an input array.
template <int N, typename T = float>
static MemRef<N, T> make_input_memref(Array<N, T>* arr) {
    MemRef<N, T> desc;
    T* ptr = static_cast<T*>(arr->device_data_ptr());
    desc.allocated = ptr;
    desc.aligned = ptr;
    // Compute row-major strides and offset
    desc.strides[N - 1] = 1LL;
    for (int d = N - 2; d >= 0; --d)
        desc.strides[d] = desc.strides[d + 1] * (int64_t)arr->region.size(d + 1);
    desc.offset = 0;
    for (int d = 0; d < N; ++d) {
        desc.sizes[d] = (int64_t)arr->region.size(d);
        desc.offset += (int64_t)arr->region.start[d] * desc.strides[d];
    }
    return desc;
}

/// Build an output MemRef descriptor.
template <int N, typename T = float>
static MemRef<N, T> make_output_memref(Array<N, T>* arr) {
    MemRef<N, T> desc;
    T* ptr = static_cast<T*>(arr->device_data_ptr());
    desc.allocated = ptr;
    desc.aligned = ptr;
    desc.offset = 0LL;
    desc.strides[N - 1] = 1LL;
    for (int d = N - 2; d >= 0; --d)
        desc.strides[d] = desc.strides[d + 1] * (int64_t)arr->region.size(d + 1);
    for (int d = 0; d < N; ++d)
        desc.sizes[d] = (int64_t)arr->region.size(d);
    return desc;
}

template <int N, typename T>
void ArrayDAGGroup::execute_node_nd(DAGNode* node, PartitionImpl<N>* partition, PendingComm<N>* comm) {
    auto nd_idx = partition->nd_index();
    int chare_idx = nd_idx[0];
    if constexpr (N >= 2)
        chare_idx = chare_idx * partition_grid[N].grid[1] + nd_idx[1];
    if constexpr (N >= 3)
        chare_idx = chare_idx * partition_grid[N].grid[2] + nd_idx[2];
    auto& ameta = this->array_meta;


    // DBG_PRINT("[Chare %d] execute_node_nd<%d>: node_id=%d identifier=%lld fusible=%d comm=%s\n",
    //           chare_idx, N, node->id, (long long)node->identifier, (int)node->fusible,
    //           comm ? "yes" : "no");

    auto it = compile_cache.find(node->identifier);

    std::vector<ASTNode*> memrefLeaves_check;
    std::vector<ASTNode*> outputRoots_check;
    std::vector<ASTNode*> broadcastLeaves_check;
    collect_leaves_and_outputs(node, memrefLeaves_check, outputRoots_check, broadcastLeaves_check);
    bool force_interpreter = memrefLeaves_check.empty() && broadcastLeaves_check.empty();

    if (force_interpreter || it == compile_cache.end() || !it->second) {
        // Interpreter fallback
        // DBG_PRINT("[Chare %d]   -> interpreter fallback\n", chare_idx);
        bool handled = false;
        if (comm) {
            for (auto ast_node : node->ast->roots) {
                if (static_cast<Opcode>(ast_node->opcode) == Opcode::MATMUL) {
                    if constexpr (N == 2) {
                        // --- 2D MATMUL interpreter with communication data ---
                        int mat_name = ast_node->operands[0]->result_name;
                        int vec_name = ast_node->operands[1]->result_name;
                        int result_name = ast_node->result_name;

                        // Get local matrix
                        auto mat_it = partition->arrays.find(mat_name);
                        if (mat_it == partition->arrays.end()) {
                            handled = true;
                            continue;
                        }
                        Array<N, T>* mat = static_cast<Array<N, T>*>(mat_it->second);
                        int local_rows = mat->region.size(0);
                        int local_cols = mat->region.size(1);

                        // Extract slice regions if present
                        int mat_rs = 0, mat_re = mat->global_shape[0];
                        int mat_cs = 0, mat_ce = mat->global_shape[1];
                        if (ast_node->operand_regions.size() >= 2) {
                            auto* mr = static_cast<ArrayRegion<2>*>(ast_node->operand_regions[0]);
                            mat_rs = mr->start[0]; mat_re = mr->stop[0];
                            mat_cs = mr->start[1]; mat_ce = mr->stop[1];
                        }

                        // Translate AST regions to global space
                        auto mat_decomp = mat_it->second->decomp;
                        mat_rs += mat_decomp.offset[0]; mat_re += mat_decomp.offset[0];
                        mat_cs += mat_decomp.offset[1]; mat_ce += mat_decomp.offset[1];

                        // Compute row/col overlap with local tile
                        auto mat_chare = mat_decomp.chare_region_global(nd_idx);
                        int row_lo = std::max(mat_chare.start[0], mat_rs);
                        int row_hi = std::min(mat_chare.stop[0], mat_re);
                        int col_lo = std::max(mat_chare.start[1], mat_cs);
                        int col_hi = std::min(mat_chare.stop[1], mat_ce);
                        int sub_rows = row_hi - row_lo;
                        int sub_cols = col_hi - col_lo;
                        int row_offset = row_lo - mat_chare.start[0];
                        int col_offset = col_lo - mat_chare.start[1];

                        // Reassemble vector from remote buffers (may be multiple)
                        T* vec_data = nullptr;
                        int vec_len = 0;
                        T* assembled_vec = nullptr;
                        {
                            auto rb_it = comm->remote_buffers.find(0);
                            if (rb_it != comm->remote_buffers.end() && !rb_it->second.empty()) {
                                auto& buffers = rb_it->second;
                                if (buffers.size() == 1) {
                                    vec_data = reinterpret_cast<T*>(buffers[0].data);
                                    vec_len = buffers[0].byte_size / sizeof(T);
                                } else {
                                    std::sort(buffers.begin(), buffers.end(),
                                        [](auto& a, auto& b) {
                                            return a.region.start[0] < b.region.start[0];
                                        });
                                    for (auto& rb : buffers)
                                        vec_len += rb.byte_size / sizeof(T);
                                    assembled_vec = new T[vec_len];
                                    int offset = 0;
                                    for (auto& rb : buffers) {
                                        int len = rb.byte_size / sizeof(T);
                                        memcpy(assembled_vec + offset,
                                               reinterpret_cast<T*>(rb.data),
                                               rb.byte_size);
                                        offset += len;
                                    }
                                    vec_data = assembled_vec;
                                }
                            }
                        }

                        if (!vec_data || sub_rows <= 0 || sub_cols <= 0) {
                            delete[] assembled_vec;
                            handled = true;
                            continue;
                        }

                        int slice_rows = mat_re - mat_rs;
                        std::array<int, N> out_start = {}, out_stop, out_step, out_gs;
                        out_stop[0] = sub_rows; out_step[0] = 1; out_gs[0] = slice_rows;
                        out_stop[1] = 1; out_step[1] = 1; out_gs[1] = 1;
                        ArrayRegion<N> out_region(out_start, out_stop, out_step);
                        ArrayDecomp<N> out_decomp = array_meta[result_name].decomp<N>();
                        Array<N, T>* result = partition->template ensure_array_typed<T>(
                            out_region, out_gs, result_name, out_decomp);

                        int actual_cols = std::min(sub_cols, vec_len);
#ifdef USE_KOKKOS
                        {
                            Kokkos::View<T**, Kokkos::LayoutRight, DeviceSpace,
                                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                                d_mat(static_cast<T*>(mat->device_data_ptr()),
                                      local_rows, local_cols);
                            auto d_sub_mat = Kokkos::subview(d_mat,
                                Kokkos::make_pair(row_offset, row_offset + sub_rows),
                                Kokkos::make_pair(col_offset, col_offset + actual_cols));
                            Kokkos::View<T*, DeviceSpace,
                                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                                d_vec(vec_data, actual_cols);
                            Kokkos::View<T*, DeviceSpace,
                                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                                d_res(static_cast<T*>(result->device_data_ptr()),
                                      sub_rows);
                            KokkosBlas::gemv("N", T(1), d_sub_mat, d_vec, T(0), d_res);
                        }
                        CT_KOKKOS_FENCE();
#else
                        eigen_gemv_sub(static_cast<T*>(mat->data_ptr()), local_cols,
                                       row_offset, col_offset,
                                       sub_rows, actual_cols,
                                       vec_data, static_cast<T*>(result->data_ptr()));
#endif
                        delete[] assembled_vec;

                        // Send partial result to 1D partition(s)
                        {
                            auto* dag_group = static_cast<ArrayDAGGroup*>(
                                partition->dag_proxy.ckLocalBranch());
                            int tile_1d = array_tile(ameta, result_name, 1);


                            // result_lo/hi in result-local space (0-based)
                            int result_lo = row_lo - mat_rs;
                            int result_hi = row_hi - mat_rs;
                            // Translate to global for chare index computation
                            auto res_meta = dag_group->array_meta.find(result_name);
                            int res_offset = (res_meta != dag_group->array_meta.end())
                                                 ? res_meta->second.offset[0]
                                                 : 0;
                            int result_lo_g = result_lo + res_offset;
                            int result_hi_g = result_hi + res_offset;
                            int first_result_chare = result_lo_g / tile_1d;
                            int last_result_chare = (result_hi_g - 1) / tile_1d;

                            T* result_data = static_cast<T*>(result->device_data_ptr());
                            for (int rk = first_result_chare; rk <= last_result_chare; rk++) {
                                // Tile boundaries in result-local space
                                int send_start = std::max(result_lo, rk * tile_1d - res_offset);
                                int send_end = std::min(result_hi, (rk + 1) * tile_1d - res_offset);
                                int send_len = send_end - send_start;
                                int data_offset = send_start - result_lo;

                                ChareIndex<1> target_ci;
                                target_ci.idx[0] = rk;
                                cross_matmul_send_result_2d_to_1d<T>(
                                    partition, node, result_name, target_ci,
                                    result_data + data_offset, send_len, send_start,
                                    dag_group->partition_proxy_1);
                            }

                            partition->retire_array(result_name);
                        }

                        handled = true;
                        continue;

                    } else if constexpr (N == 3) {
                        // --- 3D MATMUL interpreter: dimension-dropped matvec ---
                        int mat_name = ast_node->operands[0]->result_name;
                        int vec_name = ast_node->operands[1]->result_name;
                        int result_name = ast_node->result_name;

                        auto mat_it = partition->arrays.find(mat_name);
                        if (mat_it == partition->arrays.end()) {
                            handled = true;
                            continue;
                        }
                        Array<N, T>* mat = static_cast<Array<N, T>*>(mat_it->second);
                        int local_d0 = mat->region.size(0);
                        int local_d1 = mat->region.size(1);
                        int local_d2 = mat->region.size(2);

                        // Extract 3D region: identify dropped/row/col dims
                        auto* mr = static_cast<ArrayRegion<3>*>(ast_node->operand_regions[0]);
                        int dd = -1;
                        for (int d = 0; d < 3; d++) {
                            if (mr->stop[d] - mr->start[d] == 1) {
                                dd = d; break;
                            }
                        }
                        int rd = (dd == 0) ? 1 : 0;
                        int cd = (dd <= 1) ? 2 : 1;

                        int dd_s = mr->start[dd], dd_e = mr->stop[dd];
                        int rd_s = mr->start[rd], rd_e = mr->stop[rd];
                        int cd_s = mr->start[cd], cd_e = mr->stop[cd];

                        // Translate AST regions to global space
                        auto mat_decomp = mat_it->second->decomp;
                        dd_s += mat_decomp.offset[dd]; dd_e += mat_decomp.offset[dd];
                        rd_s += mat_decomp.offset[rd]; rd_e += mat_decomp.offset[rd];
                        cd_s += mat_decomp.offset[cd]; cd_e += mat_decomp.offset[cd];

                        // Local tile sizes for each dim
                        int local_dims[3] = {local_d0, local_d1, local_d2};

                        // Compute overlap in each dim using decomp
                        auto mat_chare = mat_decomp.chare_region_global(nd_idx);
                        int dd_lo = std::max(mat_chare.start[dd], dd_s);
                        int dd_hi = std::min(mat_chare.stop[dd], dd_e);
                        int row_lo = std::max(mat_chare.start[rd], rd_s);
                        int row_hi = std::min(mat_chare.stop[rd], rd_e);
                        int col_lo = std::max(mat_chare.start[cd], cd_s);
                        int col_hi = std::min(mat_chare.stop[cd], cd_e);

                        int sub_rows = row_hi - row_lo;
                        int sub_cols = col_hi - col_lo;
                        int dd_local_offset = dd_lo - mat_chare.start[dd];
                        int row_offset = row_lo - mat_chare.start[rd];
                        int col_offset = col_lo - mat_chare.start[cd];

                        // Reassemble vector from remote buffers
                        T* vec_data = nullptr;
                        int vec_len = 0;
                        T* assembled_vec = nullptr;
                        {
                            auto rb_it = comm->remote_buffers.find(0);
                            if (rb_it != comm->remote_buffers.end() && !rb_it->second.empty()) {
                                auto& buffers = rb_it->second;
                                if (buffers.size() == 1) {
                                    vec_data = reinterpret_cast<T*>(buffers[0].data);
                                    vec_len = buffers[0].byte_size / sizeof(T);
                                } else {
                                    std::sort(buffers.begin(), buffers.end(),
                                        [](auto& a, auto& b) {
                                            return a.region.start[0] < b.region.start[0];
                                        });
                                    for (auto& rb : buffers)
                                        vec_len += rb.byte_size / sizeof(T);
                                    assembled_vec = new T[vec_len];
                                    int offset = 0;
                                    for (auto& rb : buffers) {
                                        int len = rb.byte_size / sizeof(T);
                                        memcpy(assembled_vec + offset,
                                               reinterpret_cast<T*>(rb.data),
                                               rb.byte_size);
                                        offset += len;
                                    }
                                    vec_data = assembled_vec;
                                }
                            }
                        }

                        if (!vec_data || sub_rows <= 0 || sub_cols <= 0) {
                            delete[] assembled_vec;
                            handled = true;
                            continue;
                        }

                        int slice_rows = rd_e - rd_s;

                        // Create temporary result array for sub_rows
                        std::array<int, N> out_start = {}, out_stop = {}, out_step = {}, out_gs = {};
                        out_stop[0] = sub_rows; out_step[0] = 1; out_gs[0] = slice_rows;
                        out_stop[1] = 1; out_step[1] = 1; out_gs[1] = 1;
                        out_stop[2] = 1; out_step[2] = 1; out_gs[2] = 1;
                        ArrayRegion<N> out_region(out_start, out_stop, out_step);
                        ArrayDecomp<N> out_decomp = array_meta[result_name].decomp<N>();
                        Array<N, T>* result = partition->template ensure_array_typed<T>(
                            out_region, out_gs, result_name, out_decomp);

                        int actual_cols = std::min(sub_cols, vec_len);

                        // Perform 3D sub-matrix GEMV (zero-copy from 3D tile)
#ifndef USE_KOKKOS
                        eigen_gemv_sub_3d(static_cast<T*>(mat->data_ptr()),
                                          local_d0, local_d1, local_d2,
                                          dd, dd_local_offset,
                                          row_offset, col_offset,
                                          sub_rows, actual_cols,
                                          vec_data, static_cast<T*>(result->data_ptr()));
#else
                        // Kokkos: use 3D subview then reshape to 2D for gemv
                        {
                            Kokkos::View<T***, Kokkos::LayoutRight, DeviceSpace,
                                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                                d_mat(static_cast<T*>(mat->device_data_ptr()),
                                      local_d0, local_d1, local_d2);

                            // Create subview for the 2D slice
                            int r0_lo, r0_hi, r1_lo, r1_hi, r2_lo, r2_hi;
                            if (dd == 0) {
                                r0_lo = dd_local_offset; r0_hi = dd_local_offset + 1;
                                r1_lo = row_offset; r1_hi = row_offset + sub_rows;
                                r2_lo = col_offset; r2_hi = col_offset + actual_cols;
                            } else if (dd == 1) {
                                r0_lo = row_offset; r0_hi = row_offset + sub_rows;
                                r1_lo = dd_local_offset; r1_hi = dd_local_offset + 1;
                                r2_lo = col_offset; r2_hi = col_offset + actual_cols;
                            } else {
                                r0_lo = row_offset; r0_hi = row_offset + sub_rows;
                                r1_lo = col_offset; r1_hi = col_offset + actual_cols;
                                r2_lo = dd_local_offset; r2_hi = dd_local_offset + 1;
                            }
                            auto d_sub = Kokkos::subview(d_mat,
                                Kokkos::make_pair(r0_lo, r0_hi),
                                Kokkos::make_pair(r1_lo, r1_hi),
                                Kokkos::make_pair(r2_lo, r2_hi));

                            // Manual gemv on the subview
                            Kokkos::View<T*, DeviceSpace,
                                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                                d_vec(vec_data, actual_cols);
                            Kokkos::View<T*, DeviceSpace,
                                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                                d_res(static_cast<T*>(result->device_data_ptr()),
                                      sub_rows);
                            // Flatten the 3D subview to 2D for KokkosBlas
                            // For dd==0: subview is (1, sub_rows, actual_cols) -> reshape to (sub_rows, actual_cols)
                            // For other cases, strides may differ; use manual parallel_for
                            Kokkos::parallel_for(
                                Kokkos::RangePolicy<>(0, sub_rows),
                                KOKKOS_LAMBDA(int i) {
                                    T sum = T(0);
                                    for (int j = 0; j < actual_cols; j++) {
                                        if (dd == 0)
                                            sum += d_sub(0, i, j) * d_vec(j);
                                        else if (dd == 1)
                                            sum += d_sub(i, 0, j) * d_vec(j);
                                        else
                                            sum += d_sub(i, j, 0) * d_vec(j);
                                    }
                                    d_res(i) = sum;
                                });
                        }
                        CT_KOKKOS_FENCE();
#endif
                        delete[] assembled_vec;

                        // Send partial result to 1D partition(s)
                        {
                            auto* dag_group = static_cast<ArrayDAGGroup*>(
                                partition->dag_proxy.ckLocalBranch());
                            int tile_1d = array_tile(ameta, result_name, 1);


                            // result_lo/hi in result-local space (0-based)
                            int result_lo = row_lo - rd_s;
                            int result_hi = row_hi - rd_s;
                            // Translate to global for chare index computation
                            auto res_meta = dag_group->array_meta.find(result_name);
                            int res_offset = (res_meta != dag_group->array_meta.end())
                                                 ? res_meta->second.offset[0]
                                                 : 0;
                            int result_lo_g = result_lo + res_offset;
                            int result_hi_g = result_hi + res_offset;
                            int first_result_chare = result_lo_g / tile_1d;
                            int last_result_chare = (result_hi_g - 1) / tile_1d;

                            T* result_data = static_cast<T*>(result->device_data_ptr());
                            for (int rk = first_result_chare; rk <= last_result_chare; rk++) {
                                // Tile boundaries in result-local space
                                int send_start = std::max(result_lo, rk * tile_1d - res_offset);
                                int send_end = std::min(result_hi, (rk + 1) * tile_1d - res_offset);
                                int send_len = send_end - send_start;
                                int data_offset = send_start - result_lo;

                                ChareIndex<1> target_ci;
                                target_ci.idx[0] = rk;
                                cross_matmul_send_result_3d_to_1d<T>(
                                    partition, node, result_name, target_ci,
                                    result_data + data_offset, send_len, send_start,
                                    dag_group->partition_proxy_1);
                            }

                            partition->retire_array(result_name);
                        }

                        handled = true;
                        continue;
                    } else if constexpr (N == 1) {
                        // --- 1D partition: accumulate partial results from 2D/3D chares ---
                        int result_name = ast_node->result_name;
                        int mat_name = ast_node->operands[0]->result_name;
                        auto* dag_group =
                            static_cast<ArrayDAGGroup*>(partition->dag_proxy.ckLocalBranch());

                        // Extract slice region for result size (works for 2D or 3D regions)
                        int slice_rows = 0;
                        if (ast_node->operand_regions.size() >= 1) {
                            auto meta_it = dag_group->array_meta.find(mat_name);
                            int mat_ndims = (meta_it != dag_group->array_meta.end()) ? meta_it->second.ndims : 2;
                            if (mat_ndims == 3) {
                                auto* mr = static_cast<ArrayRegion<3>*>(ast_node->operand_regions[0]);
                                // Find dropped dim and extract row dim size
                                int dd = -1;
                                for (int d = 0; d < 3; d++) {
                                    if (mr->stop[d] - mr->start[d] == 1) { dd = d; break; }
                                }
                                int rd = (dd == 0) ? 1 : 0;
                                slice_rows = mr->stop[rd] - mr->start[rd];
                            } else {
                                auto* mr = static_cast<ArrayRegion<2>*>(ast_node->operand_regions[0]);
                                slice_rows = mr->stop[0] - mr->start[0];
                            }
                        } else {
                            // Fallback: use matrix global rows
                            auto meta_it = dag_group->array_meta.find(mat_name);
                            if (meta_it != dag_group->array_meta.end())
                                slice_rows = meta_it->second.global_shape[0];
                        }

                        int tile_1d = array_tile(ameta, result_name, 1);

                        int k = partition->nd_index()[0];
                        // Decomp-aware result chare start (in result-local space)
                        auto res_meta_1d = dag_group->array_meta.find(result_name);
                        ArrayDecomp<1> res_decomp_1d;
                        if (res_meta_1d != dag_group->array_meta.end())
                            res_decomp_1d = res_meta_1d->second.decomp<1>();
                        std::array<int, 1> nd_1d = {k};
                        auto res_chare_1d = res_decomp_1d.chare_region_local(nd_1d);
                        int my_result_start = res_chare_1d.start[0];
                        int local_result_size =
                            std::min(res_chare_1d.stop[0] - my_result_start,
                                     slice_rows - my_result_start);
                        if (local_result_size <= 0) {
                            handled = true;
                            continue;
                        }

                        // Create result array
                        std::array<int, 1> out_start = {0};
                        std::array<int, 1> out_stop = {local_result_size};
                        std::array<int, 1> out_step = {1};
                        std::array<int, 1> out_gs = {slice_rows};
                        ArrayRegion<1> out_region(out_start, out_stop, out_step);
                        ArrayDecomp<1> out_decomp = array_meta[result_name].decomp<1>();
                        partition->template ensure_array_typed<T>(
                            out_region, out_gs, result_name, out_decomp);

                        auto rb_it = comm->remote_buffers.find(0);
                        if (rb_it != comm->remote_buffers.end() && !rb_it->second.empty()) {
                            auto& buffers = rb_it->second;
                            T* res_data = static_cast<T*>(partition->arrays[result_name]->data_ptr());

                            // Accumulate partials with offset awareness.
                            // Each partial has a 1D region [start, stop) in output space.
                            // Result array is zero-initialized, so always accumulate with +=.
                            for (auto& rb : buffers) {
                                T* received = reinterpret_cast<T*>(rb.data);
                                int partial_start = rb.region.start[0];
                                int partial_size = rb.byte_size / sizeof(T);
                                int local_offset = partial_start - my_result_start;
#ifdef USE_KOKKOS
                                Kokkos::View<T*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                                    d_partial(received, partial_size);
                                Kokkos::View<T*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                                    d_res(static_cast<T*>(
                                        partition->arrays[result_name]->device_data_ptr()) + local_offset,
                                        partial_size);
                                KokkosBlas::axpy(T(1), d_partial, d_res);
#else
                                Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>
                                    eigen_res(res_data + local_offset, partial_size);
                                Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>
                                    eigen_partial(received, partial_size);
                                eigen_res += eigen_partial;
#endif
                            }
                            DBG_PRINT("[1D Chare %d]   MATMUL: accumulated %d partials (%d elements)\n",
                                      chare_idx, (int)buffers.size(), local_result_size);
                        }
                        handled = true;
                        continue;
                    } else {
                        CkAbort("MATMUL only supported for N >= 2");
                    }
                }
                if (static_cast<Opcode>(ast_node->opcode) == Opcode::MATMATMUL) {
                    if constexpr (N == 2) {
                        // Incremental mode: all computation was done in on_matmatmul_receive
                        // as panels arrived. C array was allocated in execute_matmatmul_node.
                        // Nothing to do here — just mark as handled.
                        DBG_PRINT("[2D Chare %d]   MATMATMUL: incremental done\n", chare_idx);
                        handled = true;
                        continue;
                    } else {
                        CkAbort("MATMATMUL only supported for N == 2");
                    }
                }
                if (static_cast<Opcode>(ast_node->opcode) == Opcode::DIAG) {
                    int input_name = ast_node->operands[0]->result_name;
                    int result_name = ast_node->result_name;
                    int k_offset = 0;
                    if (ast_node->operands.size() >= 2 && ast_node->operands[1]->is_scalar)
                        k_offset = (int)ast_node->operands[1]->scalar;

                    auto* dag_group = static_cast<ArrayDAGGroup*>(
                        partition->dag_proxy.ckLocalBranch());
                    auto result_meta_it = dag_group->array_meta.find(result_name);

                    if constexpr (N == 2) {
                        // 1D → 2D: construct diagonal matrix from received vector elements
                        int out_n = result_meta_it->second.global_shape[0];
                        auto result_decomp = result_meta_it->second.decomp<2>();
                        auto result_chare = result_decomp.chare_region_global(nd_idx);
                        int local_rows = result_chare.stop[0] - result_chare.start[0];
                        int local_cols = result_chare.stop[1] - result_chare.start[1];

                        // Create zero-initialized output array
                        std::array<int, 2> out_start = {0, 0};
                        std::array<int, 2> out_stop = {local_rows, local_cols};
                        std::array<int, 2> out_step = {1, 1};
                        std::array<int, 2> out_gs = {out_n, out_n};
                        ArrayRegion<2> out_region(out_start, out_stop, out_step);
                        partition->template ensure_array_typed<T>(
                            out_region, out_gs, result_name, result_decomp);
                        int row_start = result_chare.start[0];
                        int col_start = result_chare.start[1];

                        // Place diagonal elements from remote buffers
                        // Array is already zero-initialized by constructor
#ifdef USE_KOKKOS
                        // Device-side placement: rb.data is a device pointer
                        if (comm) {
                            auto rb_it = comm->remote_buffers.find(0);
                            if (rb_it != comm->remote_buffers.end()) {
                                for (auto& rb : rb_it->second) {
                                    T* received = reinterpret_cast<T*>(rb.data);
                                    T* out_device = static_cast<T*>(
                                        partition->arrays[result_name]->device_data_ptr());
                                    int rb_row_start = rb.region.start[0];
                                    int rb_col_start = rb.region.start[1];
                                    int rb_count = rb.byte_size / sizeof(T);
                                    int rs_cap = row_start, cs_cap = col_start;
                                    int lr_cap = local_rows, lc_cap = local_cols;
                                    Kokkos::parallel_for(
                                        CT_COMPUTE_POLICY(partition, rb_count),
                                        KOKKOS_LAMBDA(int j) {
                                            int row = rb_row_start + j;
                                            int col = rb_col_start + j;
                                            int local_r = row - rs_cap;
                                            int local_c = col - cs_cap;
                                            if (local_r >= 0 && local_r < lr_cap &&
                                                local_c >= 0 && local_c < lc_cap) {
                                                out_device[local_r * lc_cap + local_c] =
                                                    received[j];
                                            }
                                        });
                                }
                            }
                        }
#else
                        T* out_data = static_cast<T*>(partition->arrays[result_name]->data_ptr());
                        std::memset(out_data, 0, local_rows * local_cols * sizeof(T));

                        if (comm) {
                            auto rb_it = comm->remote_buffers.find(0);
                            if (rb_it != comm->remote_buffers.end()) {
                                for (auto& rb : rb_it->second) {
                                    T* received = reinterpret_cast<T*>(rb.data);
                                    int rb_row_start = rb.region.start[0];
                                    int rb_col_start = rb.region.start[1];
                                    int rb_count = rb.byte_size / sizeof(T);
                                    for (int j = 0; j < rb_count; ++j) {
                                        int row = rb_row_start + j;
                                        int col = rb_col_start + j;
                                        int local_r = row - row_start;
                                        int local_c = col - col_start;
                                        if (local_r >= 0 && local_r < local_rows &&
                                            local_c >= 0 && local_c < local_cols) {
                                            out_data[local_r * local_cols + local_c] = received[j];
                                        }
                                    }
                                }
                            }
                        }
#endif
                        DBG_PRINT("[2D Chare %d]   DIAG: 1D->2D construct done\n", chare_idx);
                        handled = true;
                        continue;
                    } else if constexpr (N == 1) {
                        // 2D → 1D: place received diagonal elements into output
                        int diag_len = result_meta_it->second.global_shape[0];
                        auto result_decomp_1d = result_meta_it->second.decomp<1>();
                        auto result_chare_1d = result_decomp_1d.chare_region_global(nd_idx);
                        int my_start = result_chare_1d.start[0];
                        int local_size = result_chare_1d.stop[0] - my_start;

                        std::array<int, 1> out_start = {0};
                        std::array<int, 1> out_stop = {local_size};
                        std::array<int, 1> out_step = {1};
                        std::array<int, 1> out_gs = {diag_len};
                        ArrayRegion<1> out_region(out_start, out_stop, out_step);
                        partition->template ensure_array_typed<T>(
                            out_region, out_gs, result_name, result_decomp_1d);
                        // Array is already zero-initialized by constructor
#ifdef USE_KOKKOS
                        // Device-side placement: rb.data is a device pointer
                        if (comm) {
                            auto rb_it = comm->remote_buffers.find(0);
                            if (rb_it != comm->remote_buffers.end()) {
                                for (auto& rb : rb_it->second) {
                                    T* received = reinterpret_cast<T*>(rb.data);
                                    T* out_device = static_cast<T*>(
                                        partition->arrays[result_name]->device_data_ptr());
                                    int rb_start = rb.region.start[0];
                                    int rb_count = rb.byte_size / sizeof(T);
                                    int ms_cap = my_start, ls_cap = local_size;
                                    Kokkos::parallel_for(
                                        CT_COMPUTE_POLICY(partition, rb_count),
                                        KOKKOS_LAMBDA(int j) {
                                            int local_idx = (rb_start + j) - ms_cap;
                                            if (local_idx >= 0 && local_idx < ls_cap)
                                                out_device[local_idx] = received[j];
                                        });
                                }
                            }
                        }
#else
                        T* out_data = static_cast<T*>(partition->arrays[result_name]->data_ptr());
                        std::memset(out_data, 0, local_size * sizeof(T));

                        if (comm) {
                            auto rb_it = comm->remote_buffers.find(0);
                            if (rb_it != comm->remote_buffers.end()) {
                                for (auto& rb : rb_it->second) {
                                    T* received = reinterpret_cast<T*>(rb.data);
                                    int rb_start = rb.region.start[0];
                                    int rb_count = rb.byte_size / sizeof(T);
                                    for (int j = 0; j < rb_count; ++j) {
                                        int local_idx = (rb_start + j) - my_start;
                                        if (local_idx >= 0 && local_idx < local_size) {
                                            out_data[local_idx] = received[j];
                                        }
                                    }
                                }
                            }
                        }
#endif
                        DBG_PRINT("[1D Chare %d]   DIAG: 2D->1D extract done\n", chare_idx);
                        handled = true;
                        continue;
                    }
                }
                if (static_cast<Opcode>(ast_node->opcode) == Opcode::TILE) {
                    int input_name = ast_node->operands[0]->result_name;
                    int result_name = ast_node->result_name;
                    int out_ndims = ast_node->ndims;

                    // Extract reps
                    std::array<int, 3> reps = {1, 1, 1};
                    int num_reps = (int)ast_node->operands.size() - 1;
                    for (int d = 0; d < num_reps && d < 3; d++) {
                        if (ast_node->operands[d + 1]->is_scalar)
                            reps[d] = (int)ast_node->operands[d + 1]->scalar;
                    }

                    auto* dag_group = static_cast<ArrayDAGGroup*>(
                        partition->dag_proxy.ckLocalBranch());
                    auto result_meta_it = dag_group->array_meta.find(result_name);
                    auto input_meta_it = dag_group->array_meta.find(input_name);

                    if (result_meta_it != dag_group->array_meta.end() &&
                        input_meta_it != dag_group->array_meta.end() &&
                        N == out_ndims) {
                        auto result_decomp = result_meta_it->second.template decomp<N>();
                        auto result_chare = result_decomp.chare_region_global(nd_idx);
                        int input_ndims = input_meta_it->second.ndims;
                        int delta = out_ndims - input_ndims;

                        // Compute local sizes
                        std::array<int, N> local_sizes;
                        std::array<int, N> out_gs;
                        for (int d = 0; d < N; ++d) {
                            local_sizes[d] = result_chare.stop[d] - result_chare.start[d];
                            out_gs[d] = result_meta_it->second.global_shape[d];
                        }

                        // Create output array if not yet allocated
                        std::array<int, N> out_start, out_stop, out_step;
                        for (int d = 0; d < N; ++d) {
                            out_start[d] = 0;
                            out_stop[d] = local_sizes[d];
                            out_step[d] = 1;
                        }
                        ArrayRegion<N> out_region(out_start, out_stop, out_step);
                        partition->template ensure_array_typed<T>(
                            out_region, out_gs, result_name, result_decomp);

                        // Get input shape (padded to out_ndims)
                        std::array<int, 3> input_shape = input_meta_it->second.global_shape;

                        // Place received data using modular index mapping
                        // Each remote buffer contains source data from one source chare.
                        // The buffer's region encodes the source's global coords (with delta padding).
#ifndef USE_KOKKOS
                        T* out_data = static_cast<T*>(partition->arrays[result_name]->data_ptr());

                        if (comm) {
                            auto rb_it = comm->remote_buffers.find(0);
                            if (rb_it != comm->remote_buffers.end()) {
                                for (auto& rb : rb_it->second) {
                                    T* received = reinterpret_cast<T*>(rb.data);
                                    // rb.region encodes source global coords in N_tgt dims
                                    // For dims < delta: [0, 1) (prepended dim)
                                    // For dims >= delta: source chare's global range
                                    std::array<int, N> src_start, src_size;
                                    int src_total_size = 1;
                                    for (int d = 0; d < N; ++d) {
                                        src_start[d] = rb.region.start[d];
                                        src_size[d] = rb.region.stop[d] - rb.region.start[d];
                                        src_total_size *= src_size[d];
                                    }

                                    // For each output position in this chare's local region,
                                    // check if it maps (via mod) to this source's range
                                    int total_out = 1;
                                    for (int d = 0; d < N; ++d)
                                        total_out *= local_sizes[d];

                                    for (int flat = 0; flat < total_out; ++flat) {
                                        // Decompose flat index into N-dimensional local coords
                                        std::array<int, N> local_idx;
                                        int rem = flat;
                                        for (int d = N - 1; d >= 0; --d) {
                                            local_idx[d] = rem % local_sizes[d];
                                            rem /= local_sizes[d];
                                        }

                                        // Convert to global output coords
                                        std::array<int, N> global_out;
                                        for (int d = 0; d < N; ++d)
                                            global_out[d] = result_chare.start[d] + local_idx[d];

                                        // Map to input coords via mod
                                        bool from_this_src = true;
                                        std::array<int, N> src_local;
                                        for (int d = 0; d < N; ++d) {
                                            int in_size;
                                            if (d < delta)
                                                in_size = 1;
                                            else
                                                in_size = input_shape[d - delta];
                                            int inp_pos = global_out[d] % in_size;
                                            if (inp_pos < src_start[d] ||
                                                inp_pos >= src_start[d] + src_size[d]) {
                                                from_this_src = false;
                                                break;
                                            }
                                            src_local[d] = inp_pos - src_start[d];
                                        }
                                        if (!from_this_src)
                                            continue;

                                        // Compute flat index into received buffer
                                        int src_flat = 0;
                                        for (int d = 0; d < N; ++d) {
                                            src_flat = src_flat * src_size[d] + src_local[d];
                                        }

                                        out_data[flat] = received[src_flat];
                                    }
                                }
                            }
                        }
#else
                        if (comm) {
                            auto rb_it = comm->remote_buffers.find(0);
                            if (rb_it != comm->remote_buffers.end()) {
                                for (auto& rb : rb_it->second) {
                                    T* received = reinterpret_cast<T*>(rb.data);
                                    T* out_device = static_cast<T*>(
                                        partition->arrays[result_name]->device_data_ptr());

                                    std::array<int, N> src_start_arr, src_size_arr;
                                    for (int d = 0; d < N; ++d) {
                                        src_start_arr[d] = rb.region.start[d];
                                        src_size_arr[d] = rb.region.stop[d] - rb.region.start[d];
                                    }

                                    int total_out = 1;
                                    for (int d = 0; d < N; ++d)
                                        total_out *= local_sizes[d];

                                    // Capture arrays for lambda
                                    int lsz[N], ssz[N], sst[N], rcs[N], ishp[3];
                                    for (int d = 0; d < N; ++d) {
                                        lsz[d] = local_sizes[d];
                                        ssz[d] = src_size_arr[d];
                                        sst[d] = src_start_arr[d];
                                        rcs[d] = result_chare.start[d];
                                    }
                                    for (int d = 0; d < 3; ++d)
                                        ishp[d] = input_shape[d];
                                    int delta_cap = delta;

                                    Kokkos::parallel_for(
                                        CT_COMPUTE_POLICY(partition, total_out),
                                        KOKKOS_LAMBDA(int flat) {
                                            int local_idx[N];
                                            int r = flat;
                                            for (int d = N - 1; d >= 0; --d) {
                                                local_idx[d] = r % lsz[d];
                                                r /= lsz[d];
                                            }

                                            bool from_this_src = true;
                                            int src_local[N];
                                            for (int d = 0; d < N; ++d) {
                                                int global_out = rcs[d] + local_idx[d];
                                                int in_size = (d < delta_cap) ? 1 : ishp[d - delta_cap];
                                                int inp_pos = global_out % in_size;
                                                if (inp_pos < sst[d] || inp_pos >= sst[d] + ssz[d]) {
                                                    from_this_src = false;
                                                    break;
                                                }
                                                src_local[d] = inp_pos - sst[d];
                                            }
                                            if (!from_this_src)
                                                return;

                                            int src_flat = 0;
                                            for (int d = 0; d < N; ++d)
                                                src_flat = src_flat * ssz[d] + src_local[d];

                                            out_device[flat] = received[src_flat];
                                        });
                                }
                            }
                        }
#endif
                        DBG_PRINT("[%dD Chare %d]   TILE: assembly done\n", N, chare_idx);
                        handled = true;
                        continue;
                    }
                }
                if (static_cast<Opcode>(ast_node->opcode) == Opcode::SET_REGION) {
                    int target_name = ast_node->operands[0]->result_name;
                    int source_name = ast_node->operands[1]->result_name;
                    auto* region = static_cast<ArrayRegion<N>*>(ast_node->region);

                    auto tgt_it = partition->arrays.find(target_name);
                    if (tgt_it == partition->arrays.end()) {
                        // Target array not on this chare — nothing to write
                        handled = true;
                        continue;
                    }
                    Array<N, T>* target =
                        static_cast<Array<N, T>*>(tgt_it->second);
                    auto r_chare_g = target->decomp.chare_region_global(nd_idx);
                    auto dst_region_global = target->decomp.to_global(*region);

                    // Use the target array's decomp-global coordinates before
                    // intersecting with the owning chare. Without this shift,
                    // any nonzero target offset makes same-epoch SET_REGION
                    // writes land on the wrong cells even though the source
                    // communication and fragment kernels are otherwise correct.
                    auto [dst_global, has_dst_overlap] =
                        intersect(dst_region_global, r_chare_g);
                    if (!has_dst_overlap) {
                        handled = true;
                        continue;
                    }

                    if (!comm->my_inputs.empty()) {
                        auto& my_inp = comm->my_inputs[0];

                        // Compute row-major strides for target
                        std::array<int, N> tgt_strides;
                        tgt_strides[N - 1] = 1;
                        for (int d = N - 2; d >= 0; --d)
                            tgt_strides[d] = tgt_strides[d + 1] * target->region.size(d + 1);

                        // Chare starts for target and source arrays
                        std::array<int, N> cs_tgt, cs_src;
                        for (int d = 0; d < N; ++d)
                            cs_tgt[d] = r_chare_g.start[d];

                        // Source array's chare region for local intersection
                        auto src_it_tmp = partition->arrays.find(source_name);
                        ArrayRegion<N> r_chare_src;
                        if (src_it_tmp != partition->arrays.end()) {
                            r_chare_src = src_it_tmp->second->decomp.chare_region_global(nd_idx);
                            for (int d = 0; d < N; ++d)
                                cs_src[d] = src_it_tmp->second->decomp.chare_start_global(d, nd_idx[d]);
                        } else {
                            r_chare_src = r_chare_g;
                            cs_src = cs_tgt;
                        }

                        // Phase 1: copy remote buffers
                        for (auto& [ridx, rbufs] : comm->remote_buffers) {
                            for (auto& rb : rbufs) {
                                // Map the received source fragment back into this chare's
                                // destination slice using the same region algebra as the
                                // communication planner. The previous hand-rolled arithmetic
                                // was fragile for shifted decompositions.
                                ArrayRegion<N> rb_in_dst = map(rb.region, my_inp, dst_global);
                                auto [overlap, has_overlap] = intersect(rb_in_dst, dst_global);
                                if (!has_overlap)
                                    continue;

                                // Compute row-major strides for remote buffer (densely packed)
                                std::array<int, N> rb_strides;
                                rb_strides[N - 1] = 1;
                                for (int d = N - 2; d >= 0; --d)
                                    rb_strides[d] = rb_strides[d + 1] * rb.region.size(d + 1);

#ifdef USE_KOKKOS
                                // Device-side copy: remote buffer and target are both on device
                                T* tgt_device = static_cast<T*>(target->device_data_ptr());
                                T* rb_device = reinterpret_cast<T*>(rb.data);
                                int64_t overlap_total = overlap.size();

                                int cs_arr[N], tgt_s[N], rb_s[N];
                                int ol_start[N], ol_sizes[N], ol_step[N], rb_in_dst_start[N],
                                    rb_in_dst_step[N];
                                for (int d = 0; d < N; d++) {
                                    cs_arr[d] = cs_tgt[d];
                                    tgt_s[d] = tgt_strides[d];
                                    rb_s[d] = rb_strides[d];
                                    ol_start[d] = overlap.start[d];
                                    ol_sizes[d] = overlap.size(d);
                                    ol_step[d] = overlap.step[d];
                                    rb_in_dst_start[d] = rb_in_dst.start[d];
                                    rb_in_dst_step[d] = rb_in_dst.step[d];
                                }

                                Kokkos::parallel_for(
                                    CT_COMPUTE_POLICY(partition, overlap_total),
                                    KOKKOS_LAMBDA(int flat_idx) {
                                        int remaining = flat_idx;
                                        int dst_flat = 0, rb_flat = 0;
                                        for (int d = N - 1; d >= 0; --d) {
                                            int coord_d = remaining % ol_sizes[d];
                                            remaining /= ol_sizes[d];
                                            int global_d = ol_start[d] + coord_d * ol_step[d];
                                            dst_flat += (global_d - cs_arr[d]) * tgt_s[d];
                                            int rb_logical =
                                                (global_d - rb_in_dst_start[d]) / rb_in_dst_step[d];
                                            rb_flat += rb_logical * rb_s[d];
                                        }
                                        tgt_device[dst_flat] = rb_device[rb_flat];
                                    });
#else
                                // Host-side odometer over overlap (step-aware)
                                std::array<int, N> idx;
                                for (int d = 0; d < N; ++d)
                                    idx[d] = overlap.start[d];
                                while (true) {
                                    int dst_flat = 0, rb_flat = 0;
                                    for (int d = 0; d < N; ++d) {
                                        dst_flat += (idx[d] - cs_tgt[d]) * tgt_strides[d];
                                        int rb_logical =
                                            (idx[d] - rb_in_dst.start[d]) / rb_in_dst.step[d];
                                        rb_flat += rb_logical * rb_strides[d];
                                    }
                                    static_cast<T*>(target->data_ptr())[dst_flat] =
                                        reinterpret_cast<T*>(rb.data)[rb_flat];

                                    int d = N - 1;
                                    while (d >= 0) {
                                        idx[d] += overlap.step[d];
                                        if (idx[d] < overlap.stop[d])
                                            break;
                                        idx[d] = overlap.start[d];
                                        --d;
                                    }
                                    if (d < 0)
                                        break;
                                }
#endif
                            }
                        }

                        // Phase 2: copy local source
                        auto [local_inp, has_local] = intersect(my_inp, r_chare_src);
                        if (has_local) {
                            ArrayRegion<N> local_in_dst = map(local_inp, my_inp, dst_global);
                            auto [loc_overlap, has_loc_overlap] =
                                intersect(local_in_dst, dst_global);
                            if (has_loc_overlap) {
                                auto src_it = partition->arrays.find(source_name);
                                if (src_it != partition->arrays.end()) {
                                    // Compute row-major strides for source
                                    std::array<int, N> src_strides;
                                    src_strides[N - 1] = 1;
                                    for (int d = N - 2; d >= 0; --d)
                                        src_strides[d] =
                                            src_strides[d + 1] * src_it->second->region.size(d + 1);

#ifdef USE_KOKKOS
                                    // Device-side copy: local source and target both on device
                                    T* tgt_device = static_cast<T*>(target->device_data_ptr());
                                    T* src_device =
                                        static_cast<T*>(src_it->second->device_data_ptr());
                                    int64_t loc_total = loc_overlap.size();

                                    int cs_tgt_arr[N], cs_src_arr[N], tgt_s[N], src_s[N];
                                    int lo_start[N], lo_sizes[N], lo_step[N];
                                    int local_inp_start[N], local_inp_step[N];
                                    int local_in_dst_start[N], local_in_dst_step[N];
                                    for (int d = 0; d < N; d++) {
                                        cs_tgt_arr[d] = cs_tgt[d];
                                        cs_src_arr[d] = cs_src[d];
                                        tgt_s[d] = tgt_strides[d];
                                        src_s[d] = src_strides[d];
                                        lo_start[d] = loc_overlap.start[d];
                                        lo_sizes[d] = loc_overlap.size(d);
                                        lo_step[d] = loc_overlap.step[d];
                                        local_inp_start[d] = local_inp.start[d];
                                        local_inp_step[d] = local_inp.step[d];
                                        local_in_dst_start[d] = local_in_dst.start[d];
                                        local_in_dst_step[d] = local_in_dst.step[d];
                                    }

                                    Kokkos::parallel_for(
                                        CT_COMPUTE_POLICY(partition, loc_total),
                                        KOKKOS_LAMBDA(int flat_idx) {
                                            int remaining = flat_idx;
                                            int dst_flat = 0, src_flat = 0;
                                            for (int d = N - 1; d >= 0; --d) {
                                                int coord_d = remaining % lo_sizes[d];
                                                remaining /= lo_sizes[d];
                                                int global_d = lo_start[d] + coord_d * lo_step[d];
                                                dst_flat += (global_d - cs_tgt_arr[d]) * tgt_s[d];
                                                int logical_idx =
                                                    (global_d - local_in_dst_start[d]) /
                                                    local_in_dst_step[d];
                                                int local_src_d = local_inp_start[d] +
                                                                  logical_idx * local_inp_step[d] -
                                                                  cs_src_arr[d];
                                                src_flat += local_src_d * src_s[d];
                                            }
                                            tgt_device[dst_flat] = src_device[src_flat];
                                        });
#else
                                    // Host-side odometer over loc_overlap (step-aware)
                                    std::array<int, N> idx;
                                    for (int d = 0; d < N; ++d)
                                        idx[d] = loc_overlap.start[d];
                                    while (true) {
                                        int dst_flat = 0, src_flat = 0;
                                        for (int d = 0; d < N; ++d) {
                                            dst_flat += (idx[d] - cs_tgt[d]) * tgt_strides[d];
                                            int logical_idx =
                                                (idx[d] - local_in_dst.start[d]) /
                                                local_in_dst.step[d];
                                            int local_src_d = local_inp.start[d] +
                                                              logical_idx * local_inp.step[d] -
                                                              cs_src[d];
                                            src_flat += local_src_d * src_strides[d];
                                        }
                                        static_cast<T*>(target->data_ptr())[dst_flat] =
                                            static_cast<T*>(src_it->second->data_ptr())[src_flat];

                                        int d = N - 1;
                                        while (d >= 0) {
                                            idx[d] += loc_overlap.step[d];
                                            if (idx[d] < loc_overlap.stop[d])
                                                break;
                                            idx[d] = loc_overlap.start[d];
                                            --d;
                                        }
                                        if (d < 0)
                                            break;
                                    }
#endif
                                }
                            }
                        }
                        handled = true;
                    }
                } else {
                    partition->executor->ast_visitor(ast_node, dtype_of<T>());
                }
            }
        }
        if (!handled)
            for (auto ast_node : node->ast->roots)
                partition->executor->ast_visitor(ast_node, dtype_of<T>());
        return;
    }

    // ---- JIT path ----
    DBG_PRINT("[Chare %d]   -> JIT path\n", chare_idx);
    void* func_ptr = it->second;

    std::vector<ASTNode*> memrefLeaves;
    std::vector<ASTNode*> outputRoots;
    std::vector<ASTNode*> broadcastLeaves;
    collect_leaves_and_outputs(node, memrefLeaves, outputRoots, broadcastLeaves);

    int n_inputs = (int)memrefLeaves.size();
    int n_outputs = (int)outputRoots.size();
    int n_broadcast = (int)broadcastLeaves.size();
    bool has_comm = comm && !comm->my_inputs.empty();

    // Load broadcast scalar values from local arrays or received buffers.
    // Each broadcast leaf is a size-1 array; we load its single element as a T scalar.
    std::vector<T> broadcast_values(n_broadcast);
    std::vector<void*> broadcast_ptrs(n_broadcast);
    for (int i = 0; i < n_broadcast; ++i) {
        int bname = broadcastLeaves[i]->result_name;
        auto arr_it = partition->arrays.find(bname);
        if (arr_it != partition->arrays.end()) {
#ifdef USE_KOKKOS
            Kokkos::deep_copy(
                Kokkos::View<T, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    &broadcast_values[i]),
                Kokkos::View<T, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    static_cast<T*>(arr_it->second->device_data_ptr())));
#else
            arr_it->second->copyToHost();
            broadcast_values[i] = *static_cast<T*>(arr_it->second->data_ptr());
#endif
        } else if (comm) {
            // Look for broadcast value in received remote buffers
            // Convention: broadcast operands use input_index = memref count + broadcast index
            int bcast_input_idx = n_inputs + i;
            auto rb_it = comm->remote_buffers.find(bcast_input_idx);
            if (rb_it != comm->remote_buffers.end() && !rb_it->second.empty()) {
#ifdef USE_KOKKOS
                // rb.data is a device pointer under USE_KOKKOS
                Kokkos::deep_copy(
                    Kokkos::View<T, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        &broadcast_values[i]),
                    Kokkos::View<T, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        reinterpret_cast<T*>(rb_it->second[0].data)));
#else
                broadcast_values[i] = *reinterpret_cast<T*>(rb_it->second[0].data);
#endif
            }
        }
        broadcast_ptrs[i] = &broadcast_values[i];
        DBG_PRINT("[Chare %d]   broadcast[%d] name=%d value=%f\n",
                  chare_idx, i, bname, (double)broadcast_values[i]);
    }

    // For the fragment path, we need the AST regions. extract_regions_nd
    // returns array-local coordinates; convert them to global space once we
    // know the output decomp so fragment/output mapping stays aligned with the
    // communication planner.
    ArrayRegion<N> r_out;
    std::vector<ArrayRegion<N>> tmp_inp;
    bool has_regions = false;
    if (has_comm) {
        std::vector<int> tmp_names;
        std::array<int, N> tmp_gs;
        has_regions = extract_regions_nd<N>(node, partition->arrays, r_out, tmp_inp, tmp_names, tmp_gs);
    }

    if (!has_comm) {
        // ---- Fast path: no region operations ----
        DBG_PRINT("[Chare %d]   -> fast path (no comm)\n", chare_idx);
        int n_memrefs = n_inputs + n_outputs;
        std::vector<MemRef<N, T>> descs(n_memrefs);

        bool missing_input = false;
        for (int i = 0; i < n_inputs; ++i) {
            int arr_name = memrefLeaves[i]->result_name;
            auto arr_it = partition->arrays.find(arr_name);
            if (arr_it == partition->arrays.end()) {
                // Input array not on this chare — skip this node
                DBG_PRINT("[Chare %d] fast-path input %d: array name=%d not on this chare, skipping (node_id=%d)\n",
                          chare_idx, i, arr_name, node->id);
                missing_input = true;
                break;
            }
            descs[i] = make_input_memref(static_cast<Array<N, T>*>(arr_it->second));
        }
        if (missing_input)
            return;

        // Determine output size from first input
        std::array<int64_t, N> out_sizes = {};
        std::array<int, N> out_gs = {};
        if (n_inputs > 0) {
            for (int d = 0; d < N; ++d)
                out_sizes[d] = descs[0].sizes[d];
            int first_name = memrefLeaves[0]->result_name;
            out_gs = partition->arrays[first_name]->global_shape;
        } else if (n_broadcast > 0) {
            // Pure scalar-scalar op: output is a single element
            for (int d = 0; d < N; ++d) {
                out_sizes[d] = 1;
                out_gs[d] = 1;
            }
        }

        DBG_PRINT("[Chare %d]   fast-path: n_inputs=%d n_outputs=%d n_broadcast=%d node_id=%d\n",
                  chare_idx, n_inputs, n_outputs, n_broadcast, node->id);
        for (int i = 0; i < n_outputs; ++i) {
            int result_name = outputRoots[i]->result_name;
            DBG_PRINT("[Chare %d]   output[%d] result_name=%d is_temp=%d (node_id=%d)\n",
                      chare_idx, i, result_name, (int)outputRoots[i]->is_temp, node->id);
            std::array<int, N> out_start = {};
            std::array<int, N> out_stop, out_step;
            for (int d = 0; d < N; ++d) {
                out_stop[d] = (int)out_sizes[d];
                out_step[d] = 1;
            }
            ArrayRegion<N> out_region(out_start, out_stop, out_step);
            ArrayDecomp<N> out_decomp_alloc;
            {
                int lookup_name = result_name;
                if (static_cast<Opcode>(outputRoots[i]->opcode) == Opcode::SET_REGION &&
                    !outputRoots[i]->operands.empty())
                    lookup_name = outputRoots[i]->operands[0]->result_name;
                auto meta_it = array_meta.find(lookup_name);
                if (meta_it != array_meta.end())
                    out_decomp_alloc = meta_it->second.template decomp<N>();
            }
            Array<N, T>* output = partition->template ensure_array_typed<T>(
                out_region, out_gs, result_name, out_decomp_alloc);
            descs[n_inputs + i] = make_output_memref(output);
        }

        dispatch_kernel(func_ptr, n_memrefs, descs, n_inputs, broadcast_ptrs
#ifdef USE_NVIDIA
                        ,
                        partition->compute_stream_raw
#endif
        );
        // A SET_REGION whose right-hand side has only scalar/broadcast leaves
        // (e.g. a[1:2] = b[0:1] + 1) produced a 1-element result buffer; write it
        // into the target region like the fragment path does below.
        if (n_inputs == 0)
            for (ASTNode* root : node->ast->roots)
                if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION)
                    partition->executor->ast_visitor(root, dtype_of<T>());
        return;
    }

    // ---- Fragment path: align local input regions + remote buffers ----
    DBG_PRINT("[Chare %d]   -> fragment path (with comm)\n", chare_idx);

    // Output decomp: the array r_out refers to (same choice as the comm planner)
    ArrayDecomp<N> out_decomp;
    {
        int result_name = fused_output_decomp_name(node);
        auto arr_it = partition->arrays.find(result_name);
        if (arr_it != partition->arrays.end())
            out_decomp = arr_it->second->decomp;
        else {
            auto meta_it = array_meta.find(result_name);
            if (meta_it != array_meta.end())
                out_decomp = meta_it->second.template decomp<N>();
            else
                out_decomp = ArrayDecomp<N>::default_decomp(
                    partition->arrays.begin()->second->global_shape,
                    array_tile(array_meta, result_name, N));
        }
    }
    if (has_regions)
        r_out = out_decomp.to_global(r_out);
    std::array<int, N> cs_out;
    for (int d = 0; d < N; ++d)
        cs_out[d] = out_decomp.chare_start_global(d, nd_idx[d]);
    ArrayRegion<N> r_chare_out = out_decomp.chare_region_global(nd_idx);
    // Global output sub-region this chare owns (same coordinate system as r_out).
    auto [r_myout_global, has_r_myout_global] = intersect(r_out, r_chare_out);

    std::vector<std::vector<FragmentData<N, T>>> input_frag_data(n_inputs);
    std::vector<ArrayRegion<N>> parents;

    for (int i = 0; i < n_inputs; ++i) {
        int source_name = memrefLeaves[i]->result_name;
        auto src_it = partition->arrays.find(source_name);

        auto& my_inp = comm->my_inputs[i];

        if (src_it != partition->arrays.end()) {
            Array<N, T>* arr = static_cast<Array<N, T>*>(src_it->second);

            // Use this input array's decomp for its chare region
            ArrayRegion<N> r_chare_inp = arr->decomp.chare_region_global(nd_idx);
            std::array<int, N> cs_inp;
            for (int d = 0; d < N; ++d)
                cs_inp[d] = arr->decomp.chare_start_global(d, nd_idx[d]);

            auto [local_part, has_local] = intersect(my_inp, r_chare_inp);
            if (has_local) {
                {
                    std::array<int, N> arr_strides;
                    arr_strides[N - 1] = 1;
                    for (int d = N - 2; d >= 0; --d)
                        arr_strides[d] = arr_strides[d + 1] * arr->region.size(d + 1);

                    int64_t flat_offset = 0;
                    for (int d = 0; d < N; ++d)
                        flat_offset += (int64_t)(local_part.start[d] - cs_inp[d]) * arr_strides[d];

                    T* local_ptr = static_cast<T*>(arr->device_data_ptr()) + flat_offset;
                    FragmentData<N, T> fd;
                    fd.region = local_part;
                    fd.data = local_ptr;
                    for (int d = 0; d < N; ++d)
                        fd.src_strides[d] = arr_strides[d];
                    input_frag_data[i].push_back(fd);
                }
            }
        }

        auto rb_it = comm->remote_buffers.find(i);
        if (rb_it != comm->remote_buffers.end())
            for (auto& rb : rb_it->second)
                input_frag_data[i].push_back({rb.region, reinterpret_cast<T*>(rb.data)});

        parents.push_back(my_inp);
    }

    auto aligned = align_fragments(input_frag_data, parents);
    int M = aligned.empty() ? 0 : (int)aligned[0].size();

    {
        std::array<int, N> first_input_start = {};
        if (comm && !comm->my_inputs.empty())
            first_input_start = comm->my_inputs.front().start;

        for (int i = 0; i < n_outputs; ++i) {
            int result_name = outputRoots[i]->result_name;
            if (partition->arrays.find(result_name) == partition->arrays.end()) {
                std::array<int, N> out_gs, local_out, zeros = {}, ones;
                bool valid = true;
                for (int d = 0; d < N; ++d) {
                    out_gs[d] = r_out.size(d);
                    // Size this chare's output tile by intersect(r_out, chare tile), not
                    // min(tile, |r_out| - nd*tile). The latter assumes r_out starts at global 0;
                    // for slices like [16:48) on chare 2 it gives 0 and skips allocation → null
                    // out pointer and segfault in the fragment loop.
                    if (has_r_myout_global)
                        local_out[d] = r_myout_global.size(d);
                    else
                        local_out[d] = 0;
                    if (local_out[d] <= 0)
                        valid = false;
                    ones[d] = 1;
                }
                if (!valid)
                    continue;
                ArrayRegion<N> out_region(zeros, local_out, ones);
                ArrayDecomp<N> result_decomp;
                {
                    int lookup_name = result_name;
                    if (static_cast<Opcode>(outputRoots[i]->opcode) == Opcode::SET_REGION &&
                        !outputRoots[i]->operands.empty())
                        lookup_name = outputRoots[i]->operands[0]->result_name;
                    auto meta_it = array_meta.find(lookup_name);
                    if (meta_it != array_meta.end())
                        result_decomp = meta_it->second.template decomp<N>();
                    else
                        result_decomp = out_decomp;
                }
                partition->template ensure_array_typed<T>(
                    out_region, out_gs, result_name, result_decomp);
            }
        }

        for (int m = 0; m < M; ++m) {
            int n_memrefs = n_inputs + n_outputs;
            std::vector<MemRef<N, T>> descs(n_memrefs);

            for (int i = 0; i < n_inputs; ++i)
                descs[i] = aligned[i][m].memref;

            for (int i = 0; i < n_outputs; ++i) {
                auto out_it = partition->arrays.find(outputRoots[i]->result_name);
                if (out_it == partition->arrays.end() || out_it->second == nullptr)
                    CkAbort("[Chare %d] fragment path: missing output array name=%d node=%d",
                            chare_idx, outputRoots[i]->result_name, node->id);
                Array<N, T>* out = static_cast<Array<N, T>*>(out_it->second);
                int j = n_inputs + i;

                std::array<int64_t, N> out_strides;
                out_strides[N - 1] = 1;
                for (int d = N - 2; d >= 0; --d)
                    out_strides[d] = out_strides[d + 1] * (int64_t)out->region.size(d + 1);

                int64_t flat_offset = 0;
                if (n_inputs > 0 && has_r_myout_global) {
                    ArrayRegion<N> out_corner =
                        map(aligned[0][m].region, comm->my_inputs[0], r_myout_global);
                    for (int d = 0; d < N; ++d) {
                        // Fragment outputs are written into a packed local buffer
                        // whose origin is the start of this chare's owned output
                        // slice, not the start of the global chare tile. Using
                        // raw global coordinates here misplaces strided SET_REGION
                        // outputs such as prolongation writes.
                        int64_t out_local =
                            ((int64_t)out_corner.start[d] - (int64_t)r_myout_global.start[d]) /
                            (int64_t)r_myout_global.step[d];
                        flat_offset += out_local * out_strides[d];
                    }
                } else {
                    for (int d = 0; d < N; ++d)
                        flat_offset +=
                            ((int64_t)(aligned[0][m].region.start[d] - first_input_start[d]) -
                             (int64_t)cs_out[d]) *
                            out_strides[d];
                }

                T* ptr = static_cast<T*>(out->device_data_ptr()) + flat_offset;
                MemRef<N, T> desc;
                desc.allocated = ptr;
                desc.aligned = ptr;
                desc.offset = 0LL;
                for (int d = 0; d < N; ++d)
                    desc.sizes[d] = aligned[0][m].memref.sizes[d];
                desc.strides[N - 1] = 1LL;
                for (int d = N - 2; d >= 0; --d)
                    desc.strides[d] = desc.strides[d + 1] * (int64_t)out->region.size(d + 1);
                descs[j] = desc;
            }

            dispatch_kernel(func_ptr, n_memrefs, descs, n_inputs, broadcast_ptrs
#ifdef USE_NVIDIA
                            ,
                            partition->compute_stream_raw
#endif
            );

            // Debug: verify input and output data for this fragment
            DBG_PRINT("[Chare %d]   fragment %d/%d: n_inputs=%d n_outputs=%d\n",
                      chare_idx, m, M, n_inputs, n_outputs);
            for (int i = 0; i < n_inputs; ++i) {
                T first_val = T(0);
                if (descs[i].sizes[0] > 0 && (N < 2 || descs[i].sizes[1] > 0))
                    first_val = descs[i].aligned[0];
                DBG_PRINT("[Chare %d]     input[%d]: ptr=%p sizes=[%lld",
                          chare_idx, i, (void*)descs[i].aligned, (long long)descs[i].sizes[0]);
                for (int d = 1; d < N; ++d)
                    DBG_PRINT(",%lld", (long long)descs[i].sizes[d]);
                DBG_PRINT("] strides=[%lld", (long long)descs[i].strides[0]);
                for (int d = 1; d < N; ++d)
                    DBG_PRINT(",%lld", (long long)descs[i].strides[d]);
                DBG_PRINT("] first_val=%f\n", (double)first_val);
            }
            for (int i = 0; i < n_outputs; ++i) {
                int j = n_inputs + i;
                T first_val = T(0);
                if (descs[j].sizes[0] > 0 && (N < 2 || descs[j].sizes[1] > 0))
                    first_val = descs[j].aligned[0];
                T sum = T(0);
                int64_t total = 1;
                for (int d = 0; d < N; ++d)
                    total *= descs[j].sizes[d];
                int check_count = std::min((int64_t)16, total);
                for (int c = 0; c < check_count; ++c) {
                    int64_t flat = 0, rem = c;
                    for (int d = N - 1; d >= 0; --d) {
                        flat += (rem % descs[j].sizes[d]) * descs[j].strides[d];
                        rem /= descs[j].sizes[d];
                    }
                    sum += descs[j].aligned[flat];
                }
                DBG_PRINT("[Chare %d]     output[%d]: ptr=%p sizes=[%lld",
                          chare_idx, i, (void*)descs[j].aligned, (long long)descs[j].sizes[0]);
                for (int d = 1; d < N; ++d)
                    DBG_PRINT(",%lld", (long long)descs[j].sizes[d]);
                DBG_PRINT("] strides=[%lld", (long long)descs[j].strides[0]);
                for (int d = 1; d < N; ++d)
                    DBG_PRINT(",%lld", (long long)descs[j].strides[d]);
                DBG_PRINT("] first_val=%f sum_first16=%f\n", (double)first_val, (double)sum);
            }
        }
    }

    // Handle SET_REGION side-effects
    for (ASTNode* root : node->ast->roots)
        if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION)
            partition->executor->ast_visitor(root, dtype_of<T>());
}

// Explicit template instantiations for execute_node_nd
template void ArrayDAGGroup::execute_node_nd<1, float>(DAGNode*, PartitionImpl<1>*, PendingComm<1>*);
template void ArrayDAGGroup::execute_node_nd<1, double>(DAGNode*, PartitionImpl<1>*, PendingComm<1>*);
template void ArrayDAGGroup::execute_node_nd<1, int32_t>(DAGNode*, PartitionImpl<1>*, PendingComm<1>*);
template void ArrayDAGGroup::execute_node_nd<1, int64_t>(DAGNode*, PartitionImpl<1>*, PendingComm<1>*);
template void ArrayDAGGroup::execute_node_nd<2, float>(DAGNode*, PartitionImpl<2>*, PendingComm<2>*);
template void ArrayDAGGroup::execute_node_nd<2, double>(DAGNode*, PartitionImpl<2>*, PendingComm<2>*);
template void ArrayDAGGroup::execute_node_nd<2, int32_t>(DAGNode*, PartitionImpl<2>*, PendingComm<2>*);
template void ArrayDAGGroup::execute_node_nd<2, int64_t>(DAGNode*, PartitionImpl<2>*, PendingComm<2>*);
template void ArrayDAGGroup::execute_node_nd<3, float>(DAGNode*, PartitionImpl<3>*, PendingComm<3>*);
template void ArrayDAGGroup::execute_node_nd<3, double>(DAGNode*, PartitionImpl<3>*, PendingComm<3>*);
template void ArrayDAGGroup::execute_node_nd<3, int32_t>(DAGNode*, PartitionImpl<3>*, PendingComm<3>*);
template void ArrayDAGGroup::execute_node_nd<3, int64_t>(DAGNode*, PartitionImpl<3>*, PendingComm<3>*);
