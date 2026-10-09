#include "backend_internal.hpp"

#include <algorithm>
#include <cstring>

#ifdef USE_KOKKOS
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#endif

template <int N>
int ArrayDAGExecutorND<N>::ast_visitor(ASTNode* node, DType dtype) {
    auto nd_idx = partition->nd_index();

    switch (static_cast<Opcode>(node->opcode)) {
    case Opcode::CREATE: {
        auto* dag_group = static_cast<ArrayDAGGroup*>(group);
        auto meta_it = dag_group->array_meta.find(node->result_name);
        auto decomp = meta_it->second.template decomp<N>();
        return partition->create(static_cast<ArrayRegion<N>*>(node->region), node->result_name,
                                 node->dtype, decomp);
    }
    case Opcode::SET_REGION: {
        // Get target array (type-erased) — may not exist on this chare
        // if the array is smaller than the chare grid.
        auto tgt_it = partition->arrays.find(node->operands[0]->result_name);
        if (tgt_it == partition->arrays.end())
            return node->result_name;
        CTArrayBase<N>* target = tgt_it->second;
        int esz = target->elem_size();

        auto* region = static_cast<ArrayRegion<N>*>(node->region);
        ASTNode* rhs_node = node->operands.size() > 1 ? node->operands[1] : nullptr;
        Region* rhs_region_base = node->get_operand_region(1);

        // Decomp-aware chare region and region-to-global translation
        auto tgt_chare = target->decomp.chare_region_global(nd_idx);
        std::array<int, N> g_reg_start, g_reg_stop;
        for (int d = 0; d < N; ++d) {
            g_reg_start[d] = region->start[d] + target->decomp.offset[d];
            g_reg_stop[d] = region->stop[d] + target->decomp.offset[d];
        }
        ArrayRegion<N> dst_region_global(g_reg_start, g_reg_stop, region->step);
        auto [overlap, has_overlap] = intersect(dst_region_global, tgt_chare);

        CTArrayBase<N>* source = nullptr;
        ArrayRegion<N> source_region_global;
        bool has_source_array = false;
        bool source_is_result_buffer = false;
        auto full_source_region_global = [&](CTArrayBase<N>* candidate) {
            std::array<int, N> start{}, stop{}, step{};
            for (int d = 0; d < N; ++d) {
                start[d] = candidate->decomp.offset[d];
                stop[d] = candidate->decomp.offset[d] + candidate->global_shape[d];
                step[d] = 1;
            }
            return ArrayRegion<N>(start, stop, step);
        };
        auto set_source = [&](CTArrayBase<N>* candidate, ArrayRegion<N> const& candidate_region) {
            source = candidate;
            source_region_global = candidate_region;
            has_source_array = true;
            source_is_result_buffer = false;
        };

        if (!has_source_array) {
            auto temp_it = partition->arrays.find(node->result_name);
            if (temp_it != partition->arrays.end()) {
                // Prefer the communicated/caller-visible SET_REGION result buffer
                // when it exists. For mixed decompositions (for example 65->129
                // prolongation), that buffer is already laid out on the target
                // slice's chare grid, while the raw RHS array may only contain
                // this chare's local source tile.
                source = temp_it->second;
                has_source_array = true;
                source_is_result_buffer = true;
            }
        }

        if (!has_source_array && rhs_node) {
            auto rhs_it = partition->arrays.find(rhs_node->result_name);
            auto rhs_op = static_cast<Opcode>(rhs_node->opcode);
            if (rhs_it != partition->arrays.end() &&
                (rhs_op == Opcode::NOOP || rhs_op == Opcode::CREATE)) {
                if (rhs_region_base && !rhs_region_base->is_global) {
                    auto* rhs_region = static_cast<ArrayRegion<N>*>(rhs_region_base);
                    set_source(rhs_it->second, rhs_it->second->decomp.to_global(*rhs_region));
                } else {
                    set_source(rhs_it->second, full_source_region_global(rhs_it->second));
                }
            }
        }

        if (!has_source_array && rhs_node) {
            auto rhs_it = partition->arrays.find(rhs_node->result_name);
            if (rhs_it != partition->arrays.end()) {
                if (rhs_region_base && !rhs_region_base->is_global) {
                    auto* rhs_region = static_cast<ArrayRegion<N>*>(rhs_region_base);
                    set_source(rhs_it->second, rhs_it->second->decomp.to_global(*rhs_region));
                } else {
                    set_source(rhs_it->second, full_source_region_global(rhs_it->second));
                }
            }
        }

        // Compute row-major strides for target
        std::array<int, N> tgt_strides;
        tgt_strides[N - 1] = 1;
        for (int d = N - 2; d >= 0; --d)
            tgt_strides[d] = tgt_strides[d + 1] * target->region.size(d + 1);

        if (!has_source_array) {
            // Scalar fill — convert scalar to target dtype
            if (has_overlap && node->operands.size() > 1 && node->operands[1]->is_scalar) {
                char scalar_buf[8];
                double s = node->operands[1]->scalar;
                switch (dtype) {
                case DType::FLOAT32: {
                    float v = (float)s;
                    memcpy(scalar_buf, &v, sizeof(v));
                    break;
                }
                case DType::FLOAT64: {
                    memcpy(scalar_buf, &s, sizeof(s));
                    break;
                }
                case DType::INT32: {
                    int32_t v = (int32_t)s;
                    memcpy(scalar_buf, &v, sizeof(v));
                    break;
                }
                case DType::INT64: {
                    int64_t v = (int64_t)s;
                    memcpy(scalar_buf, &v, sizeof(v));
                    break;
                }
                }

#ifdef USE_KOKKOS
                // Device-side scalar fill
                char* tgt_device = static_cast<char*>(target->device_data_ptr());
                // Copy scalar_buf to device (8 bytes as int64_t for capture)
                int64_t scalar_bits;
                memcpy(&scalar_bits, scalar_buf, sizeof(scalar_bits));

                int64_t overlap_total = overlap.size();
                int ol_start[N], ol_step[N], tgt_s[N], ol_sizes[N], tgt_cs_arr[N];
                for (int d = 0; d < N; d++) {
                    ol_start[d] = overlap.start[d];
                    ol_step[d] = overlap.step[d];
                    tgt_s[d] = tgt_strides[d];
                    ol_sizes[d] = overlap.size(d);
                    tgt_cs_arr[d] = tgt_chare.start[d];
                }

                Kokkos::parallel_for(
                    CT_COMPUTE_POLICY(partition, overlap_total), KOKKOS_LAMBDA(int flat_idx) {
                        int remaining = flat_idx;
                        int flat = 0;
                        for (int d = N - 1; d >= 0; --d) {
                            int coord_d = remaining % ol_sizes[d];
                            remaining /= ol_sizes[d];
                            int global_d = ol_start[d] + coord_d * ol_step[d];
                            flat += (global_d - tgt_cs_arr[d]) * tgt_s[d];
                        }
                        char* dst = tgt_device + flat * esz;
                        const char* src = reinterpret_cast<const char*>(&scalar_bits);
                        for (int b = 0; b < esz; b++)
                            dst[b] = src[b];
                    });
#else
                // Host-side odometer over the overlap region
                char* tgt_data = static_cast<char*>(target->data_ptr());
                std::array<int, N> idx;
                for (int d = 0; d < N; ++d)
                    idx[d] = overlap.start[d];
                while (true) {
                    int flat = 0;
                    for (int d = 0; d < N; ++d)
                        flat += (idx[d] - tgt_chare.start[d]) * tgt_strides[d];
                    memcpy(tgt_data + flat * esz, scalar_buf, esz);

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
            return node->result_name;
        }

        if (has_overlap) {
            // Compute row-major strides for source
            std::array<int, N> src_strides;
            src_strides[N - 1] = 1;
            for (int d = N - 2; d >= 0; --d)
                src_strides[d] = src_strides[d + 1] * source->region.size(d + 1);

            auto src_chare = source->decomp.chare_region_global(nd_idx);

#ifdef USE_KOKKOS
            // Device-side element copy
            char* tgt_device = static_cast<char*>(target->device_data_ptr());
            char* src_device = static_cast<char*>(source->device_data_ptr());

            int64_t overlap_total = overlap.size();
            int ol_start[N], ol_step[N], tgt_s[N], src_s[N], ol_sizes[N], tgt_cs_arr2[N];
            int g_reg_s[N], reg_step[N], src_chare_s[N], src_reg_s[N], src_reg_step[N],
                src_region_sizes[N];
            for (int d = 0; d < N; d++) {
                ol_start[d] = overlap.start[d];
                ol_step[d] = overlap.step[d];
                tgt_s[d] = tgt_strides[d];
                src_s[d] = src_strides[d];
                ol_sizes[d] = overlap.size(d);
                tgt_cs_arr2[d] = tgt_chare.start[d];
                g_reg_s[d] = g_reg_start[d];
                reg_step[d] = region->step[d];
                src_chare_s[d] = src_chare.start[d];
                src_reg_s[d] = source_region_global.start[d];
                src_reg_step[d] = source_region_global.step[d];
                src_region_sizes[d] = source->region.size(d);
            }

            Kokkos::parallel_for(
                CT_COMPUTE_POLICY(partition, overlap_total), KOKKOS_LAMBDA(int flat_idx) {
                    int remaining = flat_idx;
                    int dst_flat = 0, src_flat = 0;
                    bool src_valid = true;
                    for (int d = N - 1; d >= 0; --d) {
                        int coord_d = remaining % ol_sizes[d];
                        remaining /= ol_sizes[d];
                        int global_d = ol_start[d] + coord_d * ol_step[d];
                        dst_flat += (global_d - tgt_cs_arr2[d]) * tgt_s[d];
                        if (source_is_result_buffer) {
                            // A size-1 result buffer (scalar RHS) broadcasts.
                            int src_local = src_region_sizes[d] == 1
                                                ? 0
                                                : (global_d - ol_start[d]) / ol_step[d];
                            if (src_local < 0 || src_local >= src_region_sizes[d]) {
                                src_valid = false;
                                break;
                            }
                            src_flat += src_local * src_s[d];
                        } else {
                            int logical_idx = (global_d - g_reg_s[d]) / reg_step[d];
                            int src_global = src_reg_s[d] + logical_idx * src_reg_step[d];
                            int src_local = src_global - src_chare_s[d];
                            if (src_local < 0 || src_local >= src_region_sizes[d]) {
                                src_valid = false;
                                break;
                            }
                            src_flat += src_local * src_s[d];
                        }
                    }
                    if (src_valid) {
                        char* dst = tgt_device + dst_flat * esz;
                        char* src = src_device + src_flat * esz;
                        for (int b = 0; b < esz; b++)
                            dst[b] = src[b];
                    }
                });
#else
            // Host-side odometer over the overlap region
            char* tgt_data = static_cast<char*>(target->data_ptr());
            char* src_data = static_cast<char*>(source->data_ptr());
            std::array<int, N> idx;
            for (int d = 0; d < N; ++d)
                idx[d] = overlap.start[d];
            while (true) {
                int dst_flat = 0, src_flat = 0;
                bool src_valid = true;
                for (int d = 0; d < N; ++d) {
                    dst_flat += (idx[d] - tgt_chare.start[d]) * tgt_strides[d];
                    if (source_is_result_buffer) {
                        // A size-1 result buffer (scalar RHS) broadcasts.
                        int src_local = source->region.size(d) == 1
                                            ? 0
                                            : (idx[d] - overlap.start[d]) / overlap.step[d];
                        if (src_local < 0 || src_local >= source->region.size(d)) {
                            src_valid = false;
                            break;
                        }
                        src_flat += src_local * src_strides[d];
                    } else {
                        int logical_idx = (idx[d] - g_reg_start[d]) / region->step[d];
                        int src_global = source_region_global.start[d] +
                                         logical_idx * source_region_global.step[d];
                        int src_local = src_global - src_chare.start[d];
                        if (src_local < 0 || src_local >= source->region.size(d)) {
                            src_valid = false;
                            break;
                        }
                        src_flat += src_local * src_strides[d];
                    }
                }
                if (src_valid)
                    memcpy(tgt_data + dst_flat * esz, src_data + src_flat * esz, esz);

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
        return node->result_name;
    }
    case Opcode::REDUCE: {
        // REDUCE is fully handled by execute_reduce_node + Charm++ reduction.
        // The ast_visitor should never be reached for REDUCE nodes.
        return node->result_name;
    }
    case Opcode::MATMUL: {
        if constexpr (N >= 2) {
            // Local matmul in ast_visitor (no-comm path, e.g. single chare)
            // This is reached when execute_matmul_node had total_expected == 0
            int mat_name = node->operands[0]->result_name;
            int vec_name = node->operands[1]->result_name;
            int result_name = node->result_name;

            auto mat_it = partition->arrays.find(mat_name);
            auto vec_it = partition->arrays.find(vec_name);
            if (mat_it == partition->arrays.end() || vec_it == partition->arrays.end()) {
                DBG_PRINT("[Chare %d] MATMUL ast_visitor: missing operand arrays\n",
                          partition->index[0]);
                return result_name;
            }

            CTArrayBase<N>* mat_base = mat_it->second;
            CTArrayBase<N>* vec_base = vec_it->second;
            int local_rows = mat_base->region.size(0);
            int local_cols = mat_base->region.size(1);
            int vec_len = vec_base->local_size();

            // Create result array if needed
            if (partition->arrays.find(result_name) == partition->arrays.end()) {
                std::array<int, N> out_start = {}, out_stop, out_step, out_gs;
                out_stop[0] = local_rows;
                out_step[0] = 1;
                out_gs[0] = mat_base->global_shape[0];
                if (N > 1) {
                    out_stop[1] = 1;
                    out_step[1] = 1;
                    out_gs[1] = 1;
                }
                ArrayRegion<N> out_region(out_start, out_stop, out_step);
                ArrayDecomp<N> out_decomp =
                    static_cast<ArrayDAGGroup*>(group)->array_meta[result_name].template decomp<N>();
                // Use dtype dispatch to create the correctly typed result array
                partition->arrays[result_name] =
                    partition->allocate_or_reuse(out_region, out_gs, result_name, dtype, out_decomp);
            }

            CTArrayBase<N>* result_base = partition->arrays[result_name];
            int actual_cols = std::min(local_cols, vec_len);

#ifdef USE_KOKKOS
            // Perform matmul on device using KokkosBlas::gemv
            auto kokkos_gemv = [&](auto dummy) {
                using VT = decltype(dummy);
                Kokkos::View<VT**, Kokkos::LayoutRight, DeviceSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    d_mat(static_cast<VT*>(mat_base->device_data_ptr()), local_rows, local_cols);
                auto d_sub =
                    Kokkos::subview(d_mat, Kokkos::ALL, Kokkos::make_pair(0, actual_cols));
                Kokkos::View<VT*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    d_vec(static_cast<VT*>(vec_base->device_data_ptr()), actual_cols);
                Kokkos::View<VT*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    d_res(static_cast<VT*>(result_base->device_data_ptr()), local_rows);
                KokkosBlas::gemv("N", VT(1), d_sub, d_vec, VT(0), d_res);
            };
            switch (dtype) {
            case DType::FLOAT32:
                kokkos_gemv(float{});
                break;
            case DType::FLOAT64:
                kokkos_gemv(double{});
                break;
            case DType::INT32:
                kokkos_gemv(int32_t{});
                break;
            case DType::INT64:
                kokkos_gemv(int64_t{});
                break;
            }
#else
            // Perform matmul on host using Eigen gemv
            mat_base->copyToHost();
            vec_base->copyToHost();

            switch (dtype) {
            case DType::FLOAT32:
                eigen_gemv(static_cast<float*>(mat_base->data_ptr()), local_rows, local_cols,
                           static_cast<float*>(vec_base->data_ptr()), actual_cols,
                           static_cast<float*>(result_base->data_ptr()));
                break;
            case DType::FLOAT64:
                eigen_gemv(static_cast<double*>(mat_base->data_ptr()), local_rows, local_cols,
                           static_cast<double*>(vec_base->data_ptr()), actual_cols,
                           static_cast<double*>(result_base->data_ptr()));
                break;
            case DType::INT32:
                eigen_gemv(static_cast<int32_t*>(mat_base->data_ptr()), local_rows, local_cols,
                           static_cast<int32_t*>(vec_base->data_ptr()), actual_cols,
                           static_cast<int32_t*>(result_base->data_ptr()));
                break;
            case DType::INT64:
                eigen_gemv(static_cast<int64_t*>(mat_base->data_ptr()), local_rows, local_cols,
                           static_cast<int64_t*>(vec_base->data_ptr()), actual_cols,
                           static_cast<int64_t*>(result_base->data_ptr()));
                break;
            }
#endif
            DBG_PRINT("[Chare %d] MATMUL ast_visitor: %dx%d @ %d done\n", partition->index[0],
                      local_rows, actual_cols, vec_len);
            return result_name;
        } else {
            // N=1: MATMUL on 1D partition is handled via comm path, not ast_visitor
            return node->result_name;
        }
    }
    case Opcode::MATMATMUL: {
        if constexpr (N == 2) {
            // Local matmatmul in ast_visitor (no-comm path, e.g. single chare)
            int a_name = node->operands[0]->result_name;
            int b_name = node->operands[1]->result_name;
            int result_name = node->result_name;

            auto a_it = partition->arrays.find(a_name);
            auto b_it = partition->arrays.find(b_name);
            if (a_it == partition->arrays.end() || b_it == partition->arrays.end()) {
                DBG_PRINT("[Chare %d] MATMATMUL ast_visitor: missing operand arrays\n",
                          partition->index[0]);
                return result_name;
            }

            CTArrayBase<N>* a_base = a_it->second;
            CTArrayBase<N>* b_base = b_it->second;

            // Determine effective sub-block sizes from regions
            int a_local_rows = a_base->region.size(0);
            int a_local_cols = a_base->region.size(1);
            int b_local_rows = b_base->region.size(0);
            int b_local_cols = b_base->region.size(1);

            int sub_rows = a_local_rows;
            int k_size = std::min(a_local_cols, b_local_rows);
            int sub_cols = b_local_cols;

            // Handle slice regions if present
            int a_row_offset = 0, a_col_offset = 0;
            int b_row_offset = 0, b_col_offset = 0;
            if (node->operand_regions.size() >= 2) {
                auto* ar = static_cast<ArrayRegion<2>*>(node->operand_regions[0]);
                auto* br = static_cast<ArrayRegion<2>*>(node->operand_regions[1]);
                // Translate regions to global space using array decomps
                auto a_decomp = a_base->decomp;
                auto b_decomp = b_base->decomp;
                int a_rs = ar->start[0] + a_decomp.offset[0];
                int a_re = ar->stop[0] + a_decomp.offset[0];
                int a_cs = ar->start[1] + a_decomp.offset[1];
                int a_ce = ar->stop[1] + a_decomp.offset[1];
                int b_rs = br->start[0] + b_decomp.offset[0];
                int b_re = br->stop[0] + b_decomp.offset[0];
                int b_cs = br->start[1] + b_decomp.offset[1];
                int b_ce = br->stop[1] + b_decomp.offset[1];

                auto a_chare = a_decomp.chare_region_global(nd_idx);
                int chare_a_row_lo = std::max(a_chare.start[0], a_rs);
                int chare_a_row_hi = std::min(a_chare.stop[0], a_re);
                int chare_a_col_lo = std::max(a_chare.start[1], a_cs);
                int chare_a_col_hi = std::min(a_chare.stop[1], a_ce);
                a_row_offset = chare_a_row_lo - a_chare.start[0];
                a_col_offset = chare_a_col_lo - a_chare.start[1];
                sub_rows = chare_a_row_hi - chare_a_row_lo;

                auto b_chare = b_decomp.chare_region_global(nd_idx);
                int chare_b_row_lo = std::max(b_chare.start[0], b_rs);
                int chare_b_row_hi = std::min(b_chare.stop[0], b_re);
                int chare_b_col_lo = std::max(b_chare.start[1], b_cs);
                int chare_b_col_hi = std::min(b_chare.stop[1], b_ce);
                b_row_offset = chare_b_row_lo - b_chare.start[0];
                b_col_offset = chare_b_col_lo - b_chare.start[1];
                k_size =
                    std::min(chare_a_col_hi - chare_a_col_lo, chare_b_row_hi - chare_b_row_lo);
                sub_cols = chare_b_col_hi - chare_b_col_lo;
            }

            if (sub_rows <= 0 || k_size <= 0 || sub_cols <= 0)
                return result_name;

            // Create result array
            auto* dag_group2 = static_cast<ArrayDAGGroup*>(group);
            auto c_meta = dag_group2->array_meta.find(result_name);
            int c_M = c_meta->second.global_shape[0];
            int c_N = c_meta->second.global_shape[1];

            if (partition->arrays.find(result_name) == partition->arrays.end()) {
                std::array<int, 2> out_start = {0, 0};
                std::array<int, 2> out_stop = {sub_rows, sub_cols};
                std::array<int, 2> out_step = {1, 1};
                std::array<int, 2> out_gs = {c_M, c_N};
                ArrayRegion<2> out_region(out_start, out_stop, out_step);
                ArrayDecomp<2> out_decomp =
                    dag_group2->array_meta[result_name].template decomp<2>();
                partition->arrays[result_name] =
                    partition->allocate_or_reuse(out_region, out_gs, result_name, dtype, out_decomp);
            }

#ifdef USE_KOKKOS
            // Perform matmatmul on device using KokkosBlas::gemm
            auto kokkos_gemm = [&](auto dummy) {
                using VT = decltype(dummy);
                Kokkos::View<VT**, Kokkos::LayoutRight, DeviceSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    d_a(static_cast<VT*>(a_base->device_data_ptr()), a_local_rows, a_local_cols);
                auto d_a_sub = Kokkos::subview(d_a,
                                               Kokkos::make_pair(a_row_offset, a_row_offset + sub_rows),
                                               Kokkos::make_pair(a_col_offset, a_col_offset + k_size));
                Kokkos::View<VT**, Kokkos::LayoutRight, DeviceSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    d_b(static_cast<VT*>(b_base->device_data_ptr()), b_local_rows, b_local_cols);
                auto d_b_sub = Kokkos::subview(d_b,
                                               Kokkos::make_pair(b_row_offset, b_row_offset + k_size),
                                               Kokkos::make_pair(b_col_offset, b_col_offset + sub_cols));
                Kokkos::View<VT**, Kokkos::LayoutRight, DeviceSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    d_c(static_cast<VT*>(partition->arrays[result_name]->device_data_ptr()), sub_rows,
                        sub_cols);
                KokkosBlas::gemm("N", "N", VT(1), d_a_sub, d_b_sub, VT(1), d_c);
            };
            switch (dtype) {
            case DType::FLOAT32:
                kokkos_gemm(float{});
                break;
            case DType::FLOAT64:
                kokkos_gemm(double{});
                break;
            case DType::INT32:
                kokkos_gemm(int32_t{});
                break;
            case DType::INT64:
                kokkos_gemm(int64_t{});
                break;
            }
#else
            a_base->copyToHost();
            b_base->copyToHost();

            switch (dtype) {
            case DType::FLOAT32:
                eigen_gemm_sub(static_cast<float*>(a_base->data_ptr()), a_local_cols, a_row_offset,
                               a_col_offset, sub_rows, k_size,
                               static_cast<float*>(b_base->data_ptr()), b_local_cols, b_row_offset,
                               b_col_offset, sub_cols,
                               static_cast<float*>(partition->arrays[result_name]->data_ptr()));
                break;
            case DType::FLOAT64:
                eigen_gemm_sub(static_cast<double*>(a_base->data_ptr()), a_local_cols, a_row_offset,
                               a_col_offset, sub_rows, k_size,
                               static_cast<double*>(b_base->data_ptr()), b_local_cols, b_row_offset,
                               b_col_offset, sub_cols,
                               static_cast<double*>(partition->arrays[result_name]->data_ptr()));
                break;
            case DType::INT32:
                eigen_gemm_sub(static_cast<int32_t*>(a_base->data_ptr()), a_local_cols, a_row_offset,
                               a_col_offset, sub_rows, k_size,
                               static_cast<int32_t*>(b_base->data_ptr()), b_local_cols, b_row_offset,
                               b_col_offset, sub_cols,
                               static_cast<int32_t*>(partition->arrays[result_name]->data_ptr()));
                break;
            case DType::INT64:
                eigen_gemm_sub(static_cast<int64_t*>(a_base->data_ptr()), a_local_cols, a_row_offset,
                               a_col_offset, sub_rows, k_size,
                               static_cast<int64_t*>(b_base->data_ptr()), b_local_cols, b_row_offset,
                               b_col_offset, sub_cols,
                               static_cast<int64_t*>(partition->arrays[result_name]->data_ptr()));
                break;
            }
#endif
            DBG_PRINT("[Chare %d] MATMATMUL ast_visitor: (%dx%d) @ (%dx%d) done\n",
                      partition->index[0], sub_rows, k_size, k_size, sub_cols);
            return result_name;
        } else {
            return node->result_name;
        }
    }
    case Opcode::DIAG:
        // DIAG is always handled via the comm path (execute_diag_node).
        return node->result_name;
    case Opcode::TILE:
        // TILE is always handled via the comm path (execute_tile_node).
        return node->result_name;
    case Opcode::REDUCTION:
    case Opcode::BINCOUNT:
        // Always handled by execute_reduction_node.
        return node->result_name;
    default:
        CkAbort("Unknown opcode in interpreter: %d", node->opcode);
    }
}

template int ArrayDAGExecutorND<1>::ast_visitor(ASTNode*, DType);
template int ArrayDAGExecutorND<2>::ast_visitor(ASTNode*, DType);
template int ArrayDAGExecutorND<3>::ast_visitor(ASTNode*, DType);
