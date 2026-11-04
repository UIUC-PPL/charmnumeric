#include "backend_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef USE_KOKKOS
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#endif

/// Strided N-D box copy in elements: dst[dst0 + sum c*dinc] = src[src0 + sum c*sinc], 0 <= c < n.
template <int N>
struct RegionCopyPlan {
    int n[N];
    int64_t dst0, src0, dinc[N], sinc[N];
};

struct alignas(16) CopyBytes16 {
    uint64_t lo, hi;
};

#ifdef USE_KOKKOS
template <int N, typename E, typename I>
static void region_copy_kernel(PartitionImpl<N>* partition, E* dst, const E* src,
                               const RegionCopyPlan<N>& p, int64_t total) {
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, total), KOKKOS_LAMBDA(int64_t i) {
            I flat = static_cast<I>(i);
            int64_t di = p.dst0, si = p.src0;
            for (int d = N - 1; d >= 0; --d) {
                I c = flat % static_cast<I>(p.n[d]);
                flat /= static_cast<I>(p.n[d]);
                di += c * p.dinc[d];
                si += c * p.sinc[d];
            }
            dst[di] = src[si];
        });
}
#endif

template <int N>
static bool region_copy_spans_overlap(const RegionCopyPlan<N>& p) {
    int64_t dlo = p.dst0, dhi = p.dst0, slo = p.src0, shi = p.src0;
    for (int d = 0; d < N; ++d) {
        int64_t dx = (int64_t)(p.n[d] - 1) * p.dinc[d], sx = (int64_t)(p.n[d] - 1) * p.sinc[d];
        (dx < 0 ? dlo : dhi) += dx;
        (sx < 0 ? slo : shi) += sx;
    }
    return dlo <= shi && slo <= dhi;
}

/// `alias`: src and dst are the same buffer; overlapping copies then go through a packed
/// temporary, since the elements are copied in no particular order.
template <int N, typename E>
static void region_copy_typed(PartitionImpl<N>* partition, E* dst, const E* src,
                              const RegionCopyPlan<N>& p, bool alias = false) {
    int64_t total = 1;
    for (int d = 0; d < N; ++d)
        total *= p.n[d];
    if (alias && region_copy_spans_overlap(p)) {
        RegionCopyPlan<N> in = p, out = p;
        int64_t stride = 1;
        for (int d = N - 1; d >= 0; --d) {
            in.dinc[d] = out.sinc[d] = stride;
            stride *= p.n[d];
        }
        in.dst0 = out.src0 = 0;
#ifdef USE_KOKKOS
        Kokkos::View<E*, DeviceSpace> tmp(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "ct_copy_stage"), total);
        region_copy_typed<N>(partition, tmp.data(), src, in);
        region_copy_typed<N>(partition, dst, tmp.data(), out);
        CT_COMPUTE_EXEC(partition).fence("ct_copy_stage");
#else
        std::vector<E> tmp(total);
        region_copy_typed<N>(partition, tmp.data(), src, in);
        region_copy_typed<N>(partition, dst, tmp.data(), out);
#endif
        return;
    }
#ifdef USE_KOKKOS
    if (total <= INT32_MAX)
        region_copy_kernel<N, E, uint32_t>(partition, dst, src, p, total);
    else
        region_copy_kernel<N, E, uint64_t>(partition, dst, src, p, total);
#else
    (void)partition;
    int64_t rows = total / p.n[N - 1];
    bool contiguous = p.dinc[N - 1] == 1 && p.sinc[N - 1] == 1;
    for (int64_t r = 0; r < rows; ++r) {
        int64_t di = p.dst0, si = p.src0, rem = r;
        for (int d = N - 2; d >= 0; --d) {
            int64_t c = rem % p.n[d];
            rem /= p.n[d];
            di += c * p.dinc[d];
            si += c * p.sinc[d];
        }
        if (contiguous) {
            memmove(dst + di, src + si, p.n[N - 1] * sizeof(E));
        } else {
            for (int c = 0; c < p.n[N - 1]; ++c)
                dst[di + c * p.dinc[N - 1]] = src[si + c * p.sinc[N - 1]];
        }
    }
#endif
}

template <int N>
static void region_copy(PartitionImpl<N>* partition, int esz, char* dst, const char* src,
                        const RegionCopyPlan<N>& p, bool alias) {
    switch (esz) {
    case 1:
        return region_copy_typed<N>(partition, reinterpret_cast<uint8_t*>(dst),
                                    reinterpret_cast<const uint8_t*>(src), p, alias);
    case 2:
        return region_copy_typed<N>(partition, reinterpret_cast<uint16_t*>(dst),
                                    reinterpret_cast<const uint16_t*>(src), p, alias);
    case 4:
        return region_copy_typed<N>(partition, reinterpret_cast<uint32_t*>(dst),
                                    reinterpret_cast<const uint32_t*>(src), p, alias);
    case 8:
        return region_copy_typed<N>(partition, reinterpret_cast<uint64_t*>(dst),
                                    reinterpret_cast<const uint64_t*>(src), p, alias);
    case 16:
        return region_copy_typed<N>(partition, reinterpret_cast<CopyBytes16*>(dst),
                                    reinterpret_cast<const CopyBytes16*>(src), p, alias);
    default:
        CkAbort("region_copy: unsupported element size %d", esz);
    }
}

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
                char scalar_buf[16];
                double s = node->operands[1]->scalar;
                cn_dispatch(target->dtype, [&](auto tag) {
                    auto v = cn_cast<decltype(tag)>(s);
                    memcpy(scalar_buf, &v, sizeof(v));
                });

#ifdef USE_KOKKOS
                // Device-side scalar fill
                char* tgt_device = static_cast<char*>(target->device_data_ptr());
                int64_t bits0, bits1; // captured by value (up to 16 bytes)
                memcpy(&bits0, scalar_buf, 8);
                memcpy(&bits1, scalar_buf + 8, 8);

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
                        int64_t bits[2] = {bits0, bits1};
                        const char* src = reinterpret_cast<const char*>(bits);
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

            // Both offsets are affine in the overlap lattice coordinates; cells whose
            // source lies outside this chare's tile are clipped per dim.
            RegionCopyPlan<N> plan;
            plan.dst0 = plan.src0 = 0;
            bool empty = false;
            for (int d = 0; d < N; ++d) {
                int64_t lo = 0, hi = overlap.size(d), a, b;
                if (source_is_result_buffer) {
                    bool bcast = source->region.size(d) == 1;
                    a = 0;
                    b = bcast ? 0 : 1;
                    if (!bcast)
                        hi = std::min<int64_t>(hi, source->region.size(d));
                } else {
                    int64_t l0 = (overlap.start[d] - g_reg_start[d]) / region->step[d];
                    int64_t k = overlap.step[d] / region->step[d];
                    a = source_region_global.start[d] + l0 * source_region_global.step[d] -
                        src_chare.start[d];
                    b = k * source_region_global.step[d];
                    int64_t n = source->region.size(d);
                    if (b > 0) {
                        lo = std::max<int64_t>(lo, a >= 0 ? 0 : (-a + b - 1) / b);
                        hi = std::min<int64_t>(hi, a >= n ? 0 : (n - a + b - 1) / b);
                    } else if (b < 0) {
                        lo = std::max<int64_t>(lo, a < n ? 0 : (a - n) / -b + 1);
                        hi = std::min<int64_t>(hi, a < 0 ? 0 : a / -b + 1);
                    } else if (a < 0 || a >= n) {
                        hi = 0;
                    }
                }
                if (lo >= hi) {
                    empty = true;
                    break;
                }
                plan.n[d] = (int)(hi - lo);
                plan.dinc[d] = (int64_t)overlap.step[d] * tgt_strides[d];
                plan.sinc[d] = b * src_strides[d];
                plan.dst0 += ((int64_t)overlap.start[d] - tgt_chare.start[d]) * tgt_strides[d] +
                             lo * plan.dinc[d];
                plan.src0 += a * src_strides[d] + lo * plan.sinc[d];
            }
            if (!empty)
                region_copy(partition, esz, static_cast<char*>(target->device_data_ptr()),
                            static_cast<const char*>(source->device_data_ptr()), plan,
                            source == target);
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
                out_stop.fill(1);
                out_step.fill(1);
                out_gs.fill(1);
                out_stop[0] = local_rows;
                out_gs[0] = mat_base->global_shape[0];
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
                local_gemv<VT>(CT_COMPUTE_EXEC(partition),
                               static_cast<VT*>(mat_base->device_data_ptr()), local_cols,
                               local_rows, actual_cols,
                               static_cast<VT*>(vec_base->device_data_ptr()),
                               static_cast<VT*>(result_base->device_data_ptr()));
            };
            cn_dispatch_blas(dtype, kokkos_gemv);
#ifdef USE_NVIDIA
            partition->join_streams();
#endif
#else
            // Perform matmul on host using Eigen gemv
            mat_base->copyToHost();
            vec_base->copyToHost();

            cn_dispatch_blas(dtype, [&](auto tag) {
                using VT = decltype(tag);
                eigen_gemv(static_cast<VT*>(mat_base->data_ptr()), local_rows, local_cols,
                           static_cast<VT*>(vec_base->data_ptr()), actual_cols,
                           static_cast<VT*>(result_base->data_ptr()));
            });
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
#ifdef USE_NVIDIA
                // default instance is not ordered with the chare's compute stream
                cudaStreamSynchronize(partition->compute_stream_raw);
#endif
                KokkosBlas::gemm("N", "N", VT(1), d_a_sub, d_b_sub, VT(1), d_c);
#ifdef USE_NVIDIA
                Kokkos::DefaultExecutionSpace().fence("ct_gemm result");
#endif
            };
            cn_dispatch_blas(dtype, kokkos_gemm);
#else
            a_base->copyToHost();
            b_base->copyToHost();

            cn_dispatch_blas(dtype, [&](auto tag) {
                using VT = decltype(tag);
                eigen_gemm_sub(static_cast<VT*>(a_base->data_ptr()), a_local_cols, a_row_offset,
                               a_col_offset, sub_rows, k_size, static_cast<VT*>(b_base->data_ptr()),
                               b_local_cols, b_row_offset, b_col_offset, sub_cols,
                               static_cast<VT*>(partition->arrays[result_name]->data_ptr()));
            });
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
    case Opcode::TRANSPOSE:
    case Opcode::CONVOLVE:
    case Opcode::FLIP:
    case Opcode::BATCHED_MATMUL:
    case Opcode::RESHAPE:
    case Opcode::BROADCAST:
        // Handled by their own executors.
        return node->result_name;
    case Opcode::LINALG:
        // Always handled by execute_linalg_node.
        return node->result_name;
    case Opcode::GATHER:
    case Opcode::SCATTER:
    case Opcode::EXPAND:
    case Opcode::COMPRESS:
    case Opcode::MASK_SCATTER:
        // Always handled by execute_index_node.
        return node->result_name;
    default:
        CkAbort("Unknown opcode in interpreter: %d", node->opcode);
    }
}

template int ArrayDAGExecutorND<1>::ast_visitor(ASTNode*, DType);
template int ArrayDAGExecutorND<2>::ast_visitor(ASTNode*, DType);
template int ArrayDAGExecutorND<3>::ast_visitor(ASTNode*, DType);
