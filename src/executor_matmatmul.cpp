#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <type_traits>

#ifdef USE_KOKKOS
/// Gather a (count / ncols) x ncols panel out of a 3-D device tile into panel_buf.
/// Element (r, c) is read from idx3d with idx3d[dd_dim] = dd,
/// idx3d[rd_dim] = rd_base + r, idx3d[cd_dim] = cd_base + c.
/// Kept outside the dtype-dispatch generic lambdas: nvcc does not allow
/// extended device lambdas inside generic lambdas.
template <typename T, typename Policy>
static void pack_3d_panel(const Policy& policy, T* panel_buf, const T* d_data, int ncols,
                          int dd_dim, int dd, int rd_dim, int rd_base, int cd_dim, int cd_base,
                          int s0, int s1, int s2) {
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(int i) {
            int r = i / ncols;
            int c = i % ncols;
            int idx3d[3];
            idx3d[dd_dim] = dd;
            idx3d[rd_dim] = rd_base + r;
            idx3d[cd_dim] = cd_base + c;
            panel_buf[i] = d_data[idx3d[0] * s0 + idx3d[1] * s1 + idx3d[2] * s2];
        });
}
#endif

template <typename T>
static void cross_matmatmul_send_panel_2d(
    PartitionImpl<2>* partition, DAGNode* node, int arr_name, int input_index,
    const ChareIndex<2>& target_ci, T* local_data, int local_cols,
    int row_offset, int col_offset, int sub_rows, int sub_cols,
    int k_start, int k_end, int c_pos,
    CProxy_Partition2D& proxy_2d) {
    int64_t send_count = (int64_t)sub_rows * sub_cols;
    int64_t byte_size = send_count * sizeof(T);

#ifndef NDEBUG
    partition->comm_bytes_sent += byte_size;
#endif

    // For A panels (input_index=0): dim 1 = [c_pos, c_pos + sub_rows) = C-space row range
    // For B panels (input_index=1): dim 1 = [c_pos, c_pos + sub_cols) = C-space col range
    int region_data[2 * 3];
    region_data[0] = k_start;
    region_data[1] = k_end;
    region_data[2] = 1;
    region_data[3] = c_pos;
    region_data[4] = c_pos + ((input_index == 0) ? sub_rows : sub_cols);
    region_data[5] = 1;

#ifndef USE_KOKKOS
    // Pack sub-block into contiguous send buffer
    T* send_buf = new T[send_count];
    for (int r = 0; r < sub_rows; r++)
        memcpy(send_buf + r * sub_cols,
               local_data + (row_offset + r) * local_cols + col_offset,
               sub_cols * sizeof(T));
    proxy_at<2>(proxy_2d, target_ci)
        .receive_data(node->id, input_index, arr_name, 2, region_data, byte_size,
                      reinterpret_cast<char*>(send_buf));
    delete[] send_buf;
#else
    T* send_buf = static_cast<T*>(comm_alloc(byte_size));
    T* src = local_data + row_offset * local_cols + col_offset;
    int lc = local_cols;
    int sc = sub_cols;
    Kokkos::parallel_for(
        CT_COMM_POLICY(partition, sub_rows),
        KOKKOS_LAMBDA(int r) {
            for (int c = 0; c < sc; c++)
                send_buf[r * sc + c] = src[r * lc + c];
        });
    device_pack_send<2>(partition, proxy_2d, target_ci, node->id, input_index, arr_name,
                        region_data, byte_size, send_buf);
#endif
}

static void dispatch_cross_matmatmul_send_panel_2d(
    DType dt, PartitionImpl<2>* partition, DAGNode* node, int arr_name, int input_index,
    const ChareIndex<2>& target_ci, int local_cols,
    int row_offset, int col_offset, int sub_rows, int sub_cols,
    int k_start, int k_end, int c_pos,
    CProxy_Partition2D& proxy_2d) {
    // Under USE_KOKKOS the packing lambda operates on device, so pass device_data_ptr.
    // Under CPU-only, pass host data_ptr.
    auto arr_ptr = [&](auto* dummy) -> void* {
        (void)dummy;
#ifdef USE_KOKKOS
        return partition->arrays[arr_name]->device_data_ptr();
#else
        return partition->arrays[arr_name]->data_ptr();
#endif
    };
    cn_dispatch_blas(dt, [&](auto tag) {
        using T = decltype(tag);
        cross_matmatmul_send_panel_2d<T>(
            partition, node, arr_name, input_index, target_ci,
            static_cast<T*>(arr_ptr((T*)nullptr)),
            local_cols, row_offset, col_offset, sub_rows, sub_cols,
            k_start, k_end, c_pos, proxy_2d);
    });
}

template <int N>
void ArrayDAGExecutorND<N>::execute_matmatmul_node(DAGNode* node) {
    ASTNode* mm_root = nullptr;
    for (ASTNode* root : node->ast->roots) {
        if (static_cast<Opcode>(root->opcode) == Opcode::MATMATMUL) {
            mm_root = root;
            break;
        }
    }
    int a_name = mm_root->operands[0]->result_name;
    int b_name = mm_root->operands[1]->result_name;
    int result_name = mm_root->result_name;

    auto* dag_group = static_cast<ArrayDAGGroup*>(group);

    if constexpr (N == 2) {
        // A, B and C may use different tiles: each chare-index computation must use
        // the tile of the array whose chares it enumerates.
        int c_tile = array_tile(dag_group->array_meta, result_name, 2);
        int a_tile = c_tile, b_tile = c_tile;

        auto nd_idx = partition->nd_index();

        DType dt = determine_dtype<N>(node, partition->arrays);

        // Look up operand metadata
        auto a_meta_it = dag_group->array_meta.find(a_name);
        auto b_meta_it = dag_group->array_meta.find(b_name);
        auto c_meta_it = dag_group->array_meta.find(result_name);

        // Extract A and B slice regions (or full range)
        int a_rs = 0, a_re, a_cs = 0, a_ce;
        int b_rs = 0, b_re, b_cs = 0, b_ce;
        if (a_meta_it != dag_group->array_meta.end()) {
            a_re = a_meta_it->second.global_shape[0];
            a_ce = a_meta_it->second.global_shape[1];
        } else {
            CkAbort("MATMATMUL: could not find A metadata for name=%d", a_name);
            return;
        }
        if (b_meta_it != dag_group->array_meta.end()) {
            b_re = b_meta_it->second.global_shape[0];
            b_ce = b_meta_it->second.global_shape[1];
        } else {
            CkAbort("MATMATMUL: could not find B metadata for name=%d", b_name);
            return;
        }

        if (mm_root->operand_regions.size() >= 2) {
            // Extract effective 2D ranges from operand regions.
            // Operands may be 3D (dimension-dropped), so check ndims.
            int a_ndims = a_meta_it->second.ndims;
            int b_ndims = b_meta_it->second.ndims;

            if (a_ndims == 3) {
                auto* ar = static_cast<ArrayRegion<3>*>(mm_root->operand_regions[0]);
                int dd = matmul_drop_dim(mm_root, 0);
                int rd = (dd == 0) ? 1 : 0;
                int cd = (dd <= 1) ? 2 : 1;
                a_rs = ar->start[rd]; a_re = ar->stop[rd];
                a_cs = ar->start[cd]; a_ce = ar->stop[cd];
            } else {
                auto* ar = static_cast<ArrayRegion<2>*>(mm_root->operand_regions[0]);
                a_rs = ar->start[0]; a_re = ar->stop[0];
                a_cs = ar->start[1]; a_ce = ar->stop[1];
            }

            if (b_ndims == 3) {
                auto* br = static_cast<ArrayRegion<3>*>(mm_root->operand_regions[1]);
                int dd = matmul_drop_dim(mm_root, 1);
                int rd = (dd == 0) ? 1 : 0;
                int cd = (dd <= 1) ? 2 : 1;
                b_rs = br->start[rd]; b_re = br->stop[rd];
                b_cs = br->start[cd]; b_ce = br->stop[cd];
            } else {
                auto* br = static_cast<ArrayRegion<2>*>(mm_root->operand_regions[1]);
                b_rs = br->start[0]; b_re = br->stop[0];
                b_cs = br->start[1]; b_ce = br->stop[1];
            }
        }

        // Translate A/B regions to global space
        if (mm_root->operand_regions.size() >= 2) {
            int a_ndims = a_meta_it->second.ndims;
            if (a_ndims == 3) {
                auto ad = a_meta_it->second.template decomp<3>();
                auto* ar3 = static_cast<ArrayRegion<3>*>(mm_root->operand_regions[0]);
                int dd = matmul_drop_dim(mm_root, 0);
                int rd = (dd == 0) ? 1 : 0, cd = (dd <= 1) ? 2 : 1;
                a_rs += ad.offset[rd]; a_re += ad.offset[rd];
                a_cs += ad.offset[cd]; a_ce += ad.offset[cd];
            } else {
                auto ad = a_meta_it->second.template decomp<2>();
                a_rs += ad.offset[0]; a_re += ad.offset[0];
                a_cs += ad.offset[1]; a_ce += ad.offset[1];
            }
            int b_ndims = b_meta_it->second.ndims;
            if (b_ndims == 3) {
                auto bd = b_meta_it->second.template decomp<3>();
                auto* br3 = static_cast<ArrayRegion<3>*>(mm_root->operand_regions[1]);
                int dd = matmul_drop_dim(mm_root, 1);
                int rd = (dd == 0) ? 1 : 0, cd = (dd <= 1) ? 2 : 1;
                b_rs += bd.offset[rd]; b_re += bd.offset[rd];
                b_cs += bd.offset[cd]; b_ce += bd.offset[cd];
            } else {
                auto bd = b_meta_it->second.template decomp<2>();
                b_rs += bd.offset[0]; b_re += bd.offset[0];
                b_cs += bd.offset[1]; b_ce += bd.offset[1];
            }
        } else {
            // No operand regions — both A and B are 2D
            auto ad = a_meta_it->second.template decomp<2>();
            a_rs += ad.offset[0]; a_re += ad.offset[0];
            a_cs += ad.offset[1]; a_ce += ad.offset[1];
            auto bd = b_meta_it->second.template decomp<2>();
            b_rs += bd.offset[0]; b_re += bd.offset[0];
            b_cs += bd.offset[1]; b_ce += bd.offset[1];
        }

        int M = a_re - a_rs;        // result rows
        int N_cols = b_ce - b_cs;   // result cols

        // C result range in global coords
        ArrayDecomp<2> c_decomp;
        if (c_meta_it != dag_group->array_meta.end())
            c_decomp = c_meta_it->second.template decomp<2>();
        int c_rs = c_decomp.offset[0], c_re = c_decomp.offset[0] + M;
        int c_cs = c_decomp.offset[1], c_ce = c_decomp.offset[1] + N_cols;

        // Chare ranges for A, B, and C
        a_tile = array_tile(dag_group->array_meta, a_name, a_meta_it->second.ndims);
        b_tile = array_tile(dag_group->array_meta, b_name, b_meta_it->second.ndims);
        int first_a_col_chare = a_cs / a_tile;
        int last_a_col_chare = (a_ce > 0) ? (a_ce - 1) / a_tile : 0;
        int first_b_row_chare = b_rs / b_tile;
        int last_b_row_chare = (b_re > 0) ? (b_re - 1) / b_tile : 0;
        int first_c_row_chare = c_rs / c_tile;
        int last_c_row_chare = (c_re > 0) ? (c_re - 1) / c_tile : 0;
        int first_c_col_chare = c_cs / c_tile;
        int last_c_col_chare = (c_ce > 0) ? (c_ce - 1) / c_tile : 0;

        // Fast path: this chare holds A's and B's slices and all of C, so one local
        // GEMM replaces the panel exchange (no other chare has work).
        {
            auto ai = partition->arrays.find(a_name);
            auto bi = partition->arrays.find(b_name);
            auto covers = [](const ArrayRegion<2>& r, int r0, int r1, int c0, int c1) {
                return r.start[0] <= r0 && r.stop[0] >= r1 && r.start[1] <= c0 && r.stop[1] >= c1;
            };
            int K = a_ce - a_cs;
            if (ai != partition->arrays.end() && bi != partition->arrays.end() &&
                a_meta_it->second.ndims == 2 && b_meta_it->second.ndims == 2 && M > 0 &&
                N_cols > 0 && K > 0 && K == b_re - b_rs && ai->second->local_size() > 0 &&
                bi->second->local_size() > 0) {
                auto a_ch = ai->second->decomp.chare_region_global(nd_idx);
                auto b_ch = bi->second->decomp.chare_region_global(nd_idx);
                auto c_ch = c_decomp.chare_region_global(nd_idx);
                if (covers(a_ch, a_rs, a_re, a_cs, a_ce) && covers(b_ch, b_rs, b_re, b_cs, b_ce) &&
                    covers(c_ch, c_rs, c_re, c_cs, c_ce) && c_ch.start[0] == c_rs &&
                    c_ch.start[1] == c_cs) {
                    int c_rows = M, c_cols = N_cols;
                    if (partition->arrays.find(result_name) == partition->arrays.end()) {
                        ArrayRegion<2> out_region({0, 0}, {c_rows, c_cols}, {1, 1});
                        ArrayDecomp<2> out_decomp =
                            dag_group->array_meta[result_name].template decomp<2>();
                        partition->arrays[result_name] = partition->allocate_or_reuse(
                            out_region, {M, N_cols}, result_name, dt, out_decomp);
                    }
                    CTArrayBase<2>* A = ai->second;
                    CTArrayBase<2>* B = bi->second;
                    CTArrayBase<2>* C = partition->arrays[result_name];
                    int lda = A->region.size(1), ldb = B->region.size(1), ldc = C->region.size(1);
                    int64_t a_off = (int64_t)(a_rs - a_ch.start[0]) * lda + (a_cs - a_ch.start[1]);
                    int64_t b_off = (int64_t)(b_rs - b_ch.start[0]) * ldb + (b_cs - b_ch.start[1]);
                    cn_dispatch_blas(dt, [&](auto tag) {
                        using T = decltype(tag);
#ifdef USE_KOKKOS
                        T* a = static_cast<T*>(A->device_data_ptr());
                        T* b = static_cast<T*>(B->device_data_ptr());
                        T* c = static_cast<T*>(C->device_data_ptr());
#else
                        T* a = static_cast<T*>(A->data_ptr());
                        T* b = static_cast<T*>(B->data_ptr());
                        T* c = static_cast<T*>(C->data_ptr());
#endif
                        local_gemm<T>(a + a_off, lda, b + b_off, ldb, c, ldc, M, N_cols,
                                      K, T(0)
#ifdef USE_NVIDIA
                                      ,
                                      partition->compute_stream_raw
#endif
                        );
                    });
                    pending[node->id] = {node, 0, {}, {}, /*incremental=*/true};
                    on_comm_done(node->id);
                    return;
                }
            }
        }

        // --- Phase 1: Send A panels ---
        auto a_it = partition->arrays.find(a_name);
        if (a_it != partition->arrays.end() && a_it->second->local_size() > 0) {
            int local_cols = a_it->second->region.size(1);

            // Overlap of this chare's tile with A's slice region
            auto a_decomp = a_it->second->decomp;
            auto a_chare = a_decomp.chare_region_global(nd_idx);
            int a_row_lo = std::max(a_chare.start[0], a_rs);
            int a_row_hi = std::min(a_chare.stop[0], a_re);
            int a_col_lo = std::max(a_chare.start[1], a_cs);
            int a_col_hi = std::min(a_chare.stop[1], a_ce);

            if (a_row_lo < a_row_hi && a_col_lo < a_col_hi) {
                int sub_cols = a_col_hi - a_col_lo;
                int row_offset = a_row_lo - a_chare.start[0];
                int col_offset = a_col_lo - a_chare.start[1];

                // k-range in K-space: [a_col_lo - a_cs, a_col_hi - a_cs)
                int k_start = a_col_lo - a_cs;
                int k_end = a_col_hi - a_cs;

                // Which result row chare does this A block map to?
                int c_row_start = c_rs + (a_row_lo - a_rs);
                int c_row_end = c_rs + (a_row_hi - a_rs);
                int first_dest_row_chare = c_row_start / c_tile;
                int last_dest_row_chare = (c_row_end - 1) / c_tile;

#ifndef USE_KOKKOS
                a_it->second->copyToHost();
#endif
                for (int tj = first_c_col_chare; tj <= last_c_col_chare; tj++) {
                    for (int ti = first_dest_row_chare; ti <= last_dest_row_chare; ti++) {
                        ChareIndex<2> target_ci;
                        target_ci.idx[0] = ti;
                        target_ci.idx[1] = tj;

                        int dest_row_start = std::max(c_row_start, ti * c_tile);
                        int dest_row_end = std::min(c_row_end, (ti + 1) * c_tile);
                        int send_rows = dest_row_end - dest_row_start;
                        int send_row_offset = row_offset + (dest_row_start - c_row_start);

                        dispatch_cross_matmatmul_send_panel_2d(
                            dt, partition, node, a_name, /*input_index=*/0,
                            target_ci, local_cols,
                            send_row_offset, col_offset, send_rows, sub_cols,
                            k_start, k_end, /*c_pos=*/dest_row_start,
                            dag_group->partition_proxy_2);
                    }
                }
            }
        }

        // --- Phase 2: Send B panels ---
        auto b_it = partition->arrays.find(b_name);
        if (b_it != partition->arrays.end() && b_it->second->local_size() > 0) {
            int local_cols = b_it->second->region.size(1);

            auto b_decomp = b_it->second->decomp;
            auto b_chare = b_decomp.chare_region_global(nd_idx);
            int b_row_lo = std::max(b_chare.start[0], b_rs);
            int b_row_hi = std::min(b_chare.stop[0], b_re);
            int b_col_lo = std::max(b_chare.start[1], b_cs);
            int b_col_hi = std::min(b_chare.stop[1], b_ce);

            if (b_row_lo < b_row_hi && b_col_lo < b_col_hi) {
                int sub_rows = b_row_hi - b_row_lo;
                int row_offset = b_row_lo - b_chare.start[0];
                int col_offset = b_col_lo - b_chare.start[1];

                // k-range in K-space: [b_row_lo - b_rs, b_row_hi - b_rs)
                int k_start = b_row_lo - b_rs;
                int k_end = b_row_hi - b_rs;

                // B cols [b_col_lo, b_col_hi) map to C cols in global C space
                int c_col_start = c_cs + (b_col_lo - b_cs);
                int c_col_end = c_cs + (b_col_hi - b_cs);
                int first_dest_col_chare = c_col_start / c_tile;
                int last_dest_col_chare = (c_col_end - 1) / c_tile;

#ifndef USE_KOKKOS
                b_it->second->copyToHost();
#endif
                for (int ti = first_c_row_chare; ti <= last_c_row_chare; ti++) {
                    for (int tj = first_dest_col_chare; tj <= last_dest_col_chare; tj++) {
                        ChareIndex<2> target_ci;
                        target_ci.idx[0] = ti;
                        target_ci.idx[1] = tj;

                        int dest_col_start = std::max(c_col_start, tj * c_tile);
                        int dest_col_end = std::min(c_col_end, (tj + 1) * c_tile);
                        int send_cols = dest_col_end - dest_col_start;
                        int send_col_offset = col_offset + (dest_col_start - c_col_start);

                        dispatch_cross_matmatmul_send_panel_2d(
                            dt, partition, node, b_name, /*input_index=*/1,
                            target_ci, local_cols,
                            row_offset, send_col_offset, sub_rows, send_cols,
                            k_start, k_end, /*c_pos=*/dest_col_start,
                            dag_group->partition_proxy_2);
                    }
                }
            }
        }

        // --- Phase 3: Allocate C, count expected messages, enable incremental ---
        auto c_chare = c_decomp.chare_region_global(nd_idx);
        int my_c_row_lo = std::max(c_chare.start[0], c_rs);
        int my_c_row_hi = std::min(c_chare.stop[0], c_re);
        int my_c_col_lo = std::max(c_chare.start[1], c_cs);
        int my_c_col_hi = std::min(c_chare.stop[1], c_ce);

        int total_expected = 0;
        if (my_c_row_lo < my_c_row_hi && my_c_col_lo < my_c_col_hi) {
            int sub_rows = my_c_row_hi - my_c_row_lo;
            int sub_cols = my_c_col_hi - my_c_col_lo;

            // Allocate C upfront so incremental receives can accumulate into it
            if (partition->arrays.find(result_name) == partition->arrays.end()) {
                std::array<int, 2> out_start = {0, 0};
                std::array<int, 2> out_stop = {sub_rows, sub_cols};
                std::array<int, 2> out_step = {1, 1};
                std::array<int, 2> out_gs = {M, N_cols};
                ArrayRegion<2> out_region(out_start, out_stop, out_step);
                ArrayDecomp<2> out_decomp = dag_group->array_meta[result_name].template decomp<2>();
                partition->arrays[result_name] =
                    partition->allocate_or_reuse(out_region, out_gs, result_name, dt, out_decomp);
            }

            int a_k_chares = 0;
            for (int tk = first_a_col_chare; tk <= last_a_col_chare; tk++) {
                // A rows feeding my C rows (C rows are offset by c_rs).
                int a_row_for_c_lo = a_rs + (my_c_row_lo - c_rs);
                int a_row_for_c_hi = a_rs + (my_c_row_hi - c_rs);
                int fa = a_row_for_c_lo / a_tile;
                int la = (a_row_for_c_hi - 1) / a_tile;
                a_k_chares += (la - fa + 1);
            }

            int b_k_chares = 0;
            for (int tk = first_b_row_chare; tk <= last_b_row_chare; tk++) {
                int b_col_for_c_lo = b_cs + (my_c_col_lo - c_cs);
                int b_col_for_c_hi = b_cs + (my_c_col_hi - c_cs);
                int fb = b_col_for_c_lo / b_tile;
                int lb = (b_col_for_c_hi - 1) / b_tile;
                b_k_chares += (lb - fb + 1);
            }

            total_expected = a_k_chares + b_k_chares;
        }

        auto pre_it = pending.find(node->id);
        int pre_arrived = 0;
        std::unordered_map<int, std::vector<RemoteBuffer<2>>> pre_buffers;
        if (pre_it != pending.end()) {
            pre_arrived = -(pre_it->second.expected_msgs);
            pre_buffers = std::move(pre_it->second.remote_buffers);
        }
        int remaining = total_expected - pre_arrived;
        pending[node->id] = {node, remaining, {}, std::move(pre_buffers), /*incremental=*/true};

        DBG_PRINT("[2D Chare %d] matmatmul node=%d: A=[%d:%d,%d:%d] B=[%d:%d,%d:%d] "
                  "C_tile=[%d:%d,%d:%d] total_expected=%d remaining=%d\n",
                  partition->index[0], node->id,
                  a_rs, a_re, a_cs, a_ce, b_rs, b_re, b_cs, b_ce,
                  my_c_row_lo, my_c_row_hi, my_c_col_lo, my_c_col_hi,
                  total_expected, remaining);

        // Process any pre-arrived panels that came before incremental mode was set
        if (pre_arrived > 0) {
            PendingComm<2>& comm = pending[node->id];
            auto a_it2 = comm.remote_buffers.find(0);
            auto b_it2 = comm.remote_buffers.find(1);
            if (a_it2 != comm.remote_buffers.end() && b_it2 != comm.remote_buffers.end()) {
                auto& a_bufs = a_it2->second;
                auto& b_bufs = b_it2->second;
                auto arr_it = partition->arrays.find(result_name);
                if (arr_it != partition->arrays.end()) {
                    auto nd_idx2 = partition->nd_index();
                    int c_row_lo2 = c_decomp.chare_start_global(0, nd_idx2[0]);
                    int c_col_lo2 = c_decomp.chare_start_global(1, nd_idx2[1]);
                    int sub_cols2 = arr_it->second->region.size(1);

                    auto compute_pre = [&](auto* dummy) {
                        using T = std::remove_pointer_t<decltype(dummy)>;
#ifdef USE_KOKKOS
                        T* c_data = static_cast<T*>(arr_it->second->device_data_ptr());
#else
                        T* c_data = static_cast<T*>(arr_it->second->data_ptr());
#endif
                        int c_rows2 = arr_it->second->region.size(0);
                        for (auto& a_buf : a_bufs) {
                            for (auto& b_buf : b_bufs) {
                                int a_k_s = a_buf.region.start[0], a_k_e = a_buf.region.stop[0];
                                int b_k_s = b_buf.region.start[0], b_k_e = b_buf.region.stop[0];
                                int k_lo = std::max(a_k_s, b_k_s);
                                int k_hi = std::min(a_k_e, b_k_e);
                                if (k_lo >= k_hi)
                                    continue;
                                int a_rows = a_buf.region.stop[1] - a_buf.region.start[1];
                                int a_k_size = a_k_e - a_k_s;
                                int b_cols = b_buf.region.stop[1] - b_buf.region.start[1];
                                int c_lr = a_buf.region.start[1] - c_row_lo2;
                                int c_lc = b_buf.region.start[1] - c_col_lo2;
                                T* a_data = reinterpret_cast<T*>(a_buf.data);
                                T* b_data = reinterpret_cast<T*>(b_buf.data);
                                int k_size = k_hi - k_lo;
                                int a_co = k_lo - a_k_s;
                                int b_ro = k_lo - b_k_s;
                                panel_gemm_accumulate<T>(a_data, a_rows, a_k_size, a_co, b_data,
                                                         b_k_e - b_k_s, b_cols, b_ro, k_size,
                                                         c_data, c_rows2, sub_cols2, c_lr, c_lc
#ifdef USE_NVIDIA
                                                         ,
                                                         partition->compute_stream_raw
#endif
                                );
                            }
                        }
                    };
                    cn_dispatch_blas(dt, [&](auto tag) { decltype(tag)* d = nullptr; compute_pre(d); });
                }
            }
        }

        if (remaining <= 0)
            on_comm_done(node->id);

    } else if constexpr (N == 3) {
        // 3D partition: one operand is a dimension-dropped 3D array.
        int tile_2d = array_tile(dag_group->array_meta, result_name, 2);

        auto nd_idx = partition->nd_index();
        DType dt = determine_dtype<N>(node, partition->arrays);

        // Determine which operand (0=A, 1=B) is the 3D one
        for (int op_idx = 0; op_idx < 2; op_idx++) {
            int op_name = mm_root->operands[op_idx]->result_name;
            auto op_it = partition->arrays.find(op_name);
            if (op_it == partition->arrays.end() || op_it->second->local_size() <= 0)
                continue;

            if (mm_root->operand_regions.size() < 2)
                continue;
            auto* r3d = static_cast<ArrayRegion<3>*>(mm_root->operand_regions[op_idx]);

            int dd = matmul_drop_dim(mm_root, op_idx);
            if (dd < 0)
                continue;
            int rd = (dd == 0) ? 1 : 0;
            int cd = (dd <= 1) ? 2 : 1;

            int dd_s = r3d->start[dd], dd_e = r3d->stop[dd];
            int rd_s = r3d->start[rd], rd_e = r3d->stop[rd];
            int cd_s = r3d->start[cd], cd_e = r3d->stop[cd];

            // Translate AST regions to global space
            auto op_decomp = op_it->second->decomp;
            dd_s += op_decomp.offset[dd];
            dd_e += op_decomp.offset[dd];
            rd_s += op_decomp.offset[rd];
            rd_e += op_decomp.offset[rd];
            cd_s += op_decomp.offset[cd];
            cd_e += op_decomp.offset[cd];

            int local_dims[3];
            for (int d = 0; d < 3; d++)
                local_dims[d] = op_it->second->region.size(d);

            auto op_chare = op_decomp.chare_region_global(nd_idx);
            int dd_lo = std::max(op_chare.start[dd], dd_s);
            int dd_hi = std::min(op_chare.stop[dd], dd_e);
            int row_lo = std::max(op_chare.start[rd], rd_s);
            int row_hi = std::min(op_chare.stop[rd], rd_e);
            int col_lo = std::max(op_chare.start[cd], cd_s);
            int col_hi = std::min(op_chare.stop[cd], cd_e);

            if (dd_lo >= dd_hi || row_lo >= row_hi || col_lo >= col_hi)
                continue;

            int sub_rows = row_hi - row_lo;
            int sub_cols = col_hi - col_lo;
            int dd_local_offset = dd_lo - op_chare.start[dd];
            int row_offset = row_lo - op_chare.start[rd];
            int col_offset = col_lo - op_chare.start[cd];

#ifndef USE_KOKKOS
            op_it->second->copyToHost();
#endif

            auto c3_meta_it = dag_group->array_meta.find(result_name);
            ArrayDecomp<2> c3_decomp;
            if (c3_meta_it != dag_group->array_meta.end())
                c3_decomp = c3_meta_it->second.template decomp<2>();
            int c_M = c3_decomp.offset[0] + c3_decomp.global_shape[0];
            int c_N_cols_g = c3_decomp.offset[1] + c3_decomp.global_shape[1];

            int first_c_row_chare = c3_decomp.offset[0] / tile_2d;
            int last_c_row_chare = (c_M > 0) ? (c_M - 1) / tile_2d : 0;
            int first_c_col_chare = c3_decomp.offset[1] / tile_2d;
            int last_c_col_chare = (c_N_cols_g > 0) ? (c_N_cols_g - 1) / tile_2d : 0;

            if (op_idx == 0) {
                // This 3D chare holds A. Send A panels to C chares.
                int k_start = col_lo - cd_s;
                int k_end = col_hi - cd_s;
                int c_row_start = c3_decomp.offset[0] + (row_lo - rd_s);
                int c_row_end = c3_decomp.offset[0] + (row_hi - rd_s);
                int first_dest_row_chare = c_row_start / tile_2d;
                int last_dest_row_chare = (c_row_end - 1) / tile_2d;

                auto send_3d_panel = [&](auto* dummy) {
                    using T = std::remove_pointer_t<decltype(dummy)>;
                    int s0 = local_dims[1] * local_dims[2];
                    int s1 = local_dims[2];
                    int s2 = 1;
                    int strides[3] = {s0, s1, s2};

#ifdef USE_KOKKOS
                    T* d_data = static_cast<T*>(op_it->second->device_data_ptr());
                    int cap_dd = dd_local_offset, cap_rd = row_offset, cap_cd = col_offset;
                    int cap_dd_dim = dd, cap_rd_dim = rd, cap_cd_dim = cd;
                    int cap_sub_cols = sub_cols;

                    for (int tj = first_c_col_chare; tj <= last_c_col_chare; tj++) {
                        for (int ti = first_dest_row_chare; ti <= last_dest_row_chare; ti++) {
                            int dest_row_start_c = std::max(c_row_start, ti * tile_2d);
                            int dest_row_end_c = std::min(c_row_end, (ti + 1) * tile_2d);
                            int send_rows = dest_row_end_c - dest_row_start_c;
                            int send_row_off = dest_row_start_c - c_row_start;

                            ChareIndex<2> target_ci;
                            target_ci.idx[0] = ti;
                            target_ci.idx[1] = tj;

                            int64_t panel_count = (int64_t)send_rows * sub_cols;
                            int64_t panel_bytes = panel_count * sizeof(T);
                            T* panel_buf = static_cast<T*>(
                                comm_alloc(panel_bytes));
                            pack_3d_panel<T>(CT_COMM_POLICY(partition, panel_count), panel_buf,
                                             d_data, cap_sub_cols, cap_dd_dim, cap_dd,
                                             cap_rd_dim, cap_rd + send_row_off, cap_cd_dim,
                                             cap_cd, strides[0], strides[1], strides[2]);

                            int region_data[2 * 3] = {k_start, k_end, 1,
                                                      dest_row_start_c, dest_row_end_c, 1};

#ifndef NDEBUG
                            partition->comm_bytes_sent += panel_bytes;
#endif
                            device_pack_send<3, 2>(partition, dag_group->partition_proxy_2,
                                                   target_ci, node->id, 0, op_name,
                                                   region_data, panel_bytes, panel_buf);
                        }
                    }
#else
                    T* data = static_cast<T*>(op_it->second->data_ptr());
                    int64_t send_count = (int64_t)sub_rows * sub_cols;
                    T* send_buf = new T[send_count];

                    for (int r = 0; r < sub_rows; r++) {
                        for (int c = 0; c < sub_cols; c++) {
                            int idx3d[3];
                            idx3d[dd] = dd_local_offset;
                            idx3d[rd] = row_offset + r;
                            idx3d[cd] = col_offset + c;
                            send_buf[r * sub_cols + c] =
                                data[idx3d[0] * strides[0] + idx3d[1] * strides[1] + idx3d[2] * strides[2]];
                        }
                    }

                    for (int tj = first_c_col_chare; tj <= last_c_col_chare; tj++) {
                        for (int ti = first_dest_row_chare; ti <= last_dest_row_chare; ti++) {
                            int dest_row_start_c = std::max(c_row_start, ti * tile_2d);
                            int dest_row_end_c = std::min(c_row_end, (ti + 1) * tile_2d);
                            int send_rows = dest_row_end_c - dest_row_start_c;
                            int send_row_off = dest_row_start_c - c_row_start;

                            ChareIndex<2> target_ci;
                            target_ci.idx[0] = ti;
                            target_ci.idx[1] = tj;

                            int region_data[2 * 3];
                            region_data[0] = k_start;
                            region_data[1] = k_end;
                            region_data[2] = 1;
                            region_data[3] = dest_row_start_c;
                            region_data[4] = dest_row_end_c;
                            region_data[5] = 1;

                            int64_t panel_count = (int64_t)send_rows * sub_cols;
                            int64_t panel_bytes = panel_count * sizeof(T);
                            T* panel_buf = new T[panel_count];
                            for (int r = 0; r < send_rows; r++)
                                memcpy(panel_buf + r * sub_cols,
                                       send_buf + (send_row_off + r) * sub_cols,
                                       sub_cols * sizeof(T));

#ifndef NDEBUG
                            partition->comm_bytes_sent += panel_bytes;
#endif
                            proxy_at<2>(dag_group->partition_proxy_2, target_ci)
                                .receive_data(node->id, /*input_index=*/0, op_name, 2,
                                              region_data, panel_bytes,
                                              reinterpret_cast<char*>(panel_buf));
                            delete[] panel_buf;
                        }
                    }
                    delete[] send_buf;
#endif
                };
                cn_dispatch_blas(dt, [&](auto tag) { decltype(tag)* d = nullptr; send_3d_panel(d); });
            } else {
                // This 3D chare holds B. Send B panels to C chares.
                int k_start = row_lo - rd_s;
                int k_end = row_hi - rd_s;
                int c_col_start = c3_decomp.offset[1] + (col_lo - cd_s);
                int c_col_end = c3_decomp.offset[1] + (col_hi - cd_s);
                int first_dest_col_chare = c_col_start / tile_2d;
                int last_dest_col_chare = (c_col_end - 1) / tile_2d;

                auto send_3d_panel = [&](auto* dummy) {
                    using T = std::remove_pointer_t<decltype(dummy)>;
                    int s0 = local_dims[1] * local_dims[2];
                    int s1 = local_dims[2];
                    int s2 = 1;
                    int strides[3] = {s0, s1, s2};

#ifdef USE_KOKKOS
                    T* d_data = static_cast<T*>(op_it->second->device_data_ptr());
                    int cap_dd = dd_local_offset, cap_rd = row_offset, cap_cd = col_offset;
                    int cap_dd_dim = dd, cap_rd_dim = rd, cap_cd_dim = cd;

                    for (int ti = first_c_row_chare; ti <= last_c_row_chare; ti++) {
                        for (int tj = first_dest_col_chare; tj <= last_dest_col_chare; tj++) {
                            int dest_col_start_c = std::max(c_col_start, tj * tile_2d);
                            int dest_col_end_c = std::min(c_col_end, (tj + 1) * tile_2d);
                            int send_cols = dest_col_end_c - dest_col_start_c;
                            int send_col_off = dest_col_start_c - c_col_start;

                            ChareIndex<2> target_ci;
                            target_ci.idx[0] = ti;
                            target_ci.idx[1] = tj;

                            int64_t panel_count = (int64_t)sub_rows * send_cols;
                            int64_t panel_bytes = panel_count * sizeof(T);
                            T* panel_buf = static_cast<T*>(
                                comm_alloc(panel_bytes));
                            int cap_send_cols = send_cols;
                            pack_3d_panel<T>(CT_COMM_POLICY(partition, panel_count), panel_buf,
                                             d_data, cap_send_cols, cap_dd_dim, cap_dd,
                                             cap_rd_dim, cap_rd, cap_cd_dim,
                                             cap_cd + send_col_off, strides[0], strides[1],
                                             strides[2]);

                            int region_data[2 * 3] = {k_start, k_end, 1,
                                                      dest_col_start_c, dest_col_end_c, 1};
#ifndef NDEBUG
                            partition->comm_bytes_sent += panel_bytes;
#endif
                            device_pack_send<3, 2>(partition, dag_group->partition_proxy_2,
                                                   target_ci, node->id, 1, op_name,
                                                   region_data, panel_bytes, panel_buf);
                        }
                    }
#else
                    T* data = static_cast<T*>(op_it->second->data_ptr());
                    int64_t send_count = (int64_t)sub_rows * sub_cols;
                    T* send_buf = new T[send_count];

                    for (int r = 0; r < sub_rows; r++) {
                        for (int c = 0; c < sub_cols; c++) {
                            int idx3d[3];
                            idx3d[dd] = dd_local_offset;
                            idx3d[rd] = row_offset + r;
                            idx3d[cd] = col_offset + c;
                            send_buf[r * sub_cols + c] =
                                data[idx3d[0] * strides[0] + idx3d[1] * strides[1] + idx3d[2] * strides[2]];
                        }
                    }

                    for (int ti = first_c_row_chare; ti <= last_c_row_chare; ti++) {
                        for (int tj = first_dest_col_chare; tj <= last_dest_col_chare; tj++) {
                            int dest_col_start_c = std::max(c_col_start, tj * tile_2d);
                            int dest_col_end_c = std::min(c_col_end, (tj + 1) * tile_2d);
                            int send_cols = dest_col_end_c - dest_col_start_c;
                            int send_col_off = dest_col_start_c - c_col_start;

                            ChareIndex<2> target_ci;
                            target_ci.idx[0] = ti;
                            target_ci.idx[1] = tj;

                            int region_data[2 * 3];
                            region_data[0] = k_start;
                            region_data[1] = k_end;
                            region_data[2] = 1;
                            region_data[3] = dest_col_start_c;
                            region_data[4] = dest_col_end_c;
                            region_data[5] = 1;

                            int64_t panel_count = (int64_t)sub_rows * send_cols;
                            int64_t panel_bytes = panel_count * sizeof(T);
                            T* panel_buf = new T[panel_count];
                            for (int r = 0; r < sub_rows; r++)
                                for (int c = 0; c < send_cols; c++)
                                    panel_buf[r * send_cols + c] =
                                        send_buf[r * sub_cols + (send_col_off + c)];

#ifndef NDEBUG
                            partition->comm_bytes_sent += panel_bytes;
#endif
                            proxy_at<2>(dag_group->partition_proxy_2, target_ci)
                                .receive_data(node->id, /*input_index=*/1, op_name, 2,
                                              region_data, panel_bytes,
                                              reinterpret_cast<char*>(panel_buf));
                            delete[] panel_buf;
                        }
                    }
                    delete[] send_buf;
#endif
                };
                cn_dispatch_blas(dt, [&](auto tag) { decltype(tag)* d = nullptr; send_3d_panel(d); });
            }
        }

        // 3D chares only send — they don't hold C, so just finish
        node_finished(node->id);

    } else {
        CkAbort("MATMATMUL only supported for 2D and 3D partitions (N=%d)", N);
    }
}

template void ArrayDAGExecutorND<1>::execute_matmatmul_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_matmatmul_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_matmatmul_node(DAGNode*);
