#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <cstring>
#include <functional>
#include <set>
#include <unordered_set>

template <int N_tgt>
static typename PartitionTraits<N_tgt>::ProxyType& get_partition_proxy(ArrayDAGGroup* dag_group);

template <>
CProxy_Partition1D& get_partition_proxy<1>(ArrayDAGGroup* dag_group) {
    return dag_group->partition_proxy_1;
}
template <>
CProxy_Partition2D& get_partition_proxy<2>(ArrayDAGGroup* dag_group) {
    return dag_group->partition_proxy_2;
}
template <>
CProxy_Partition3D& get_partition_proxy<3>(ArrayDAGGroup* dag_group) {
    return dag_group->partition_proxy_3;
}

/// Build the source->target lattice map of a cross-ndims SET_REGION from metadata.
template <int N_src, int N_tgt>
static CrossSetRegionMap<N_src, N_tgt>
make_cross_set_region_map(const ArrayDAGGroup::ArrayMetadata& src_meta,
                          const ArrayDAGGroup::ArrayMetadata& tgt_meta, ASTNode* sr_root) {
    std::array<int, N_src> src_shape, src_off;
    for (int d = 0; d < N_src; ++d) {
        src_shape[d] = src_meta.global_shape[d];
        src_off[d] = src_meta.offset[d];
    }
    std::array<int, N_tgt> tgt_shape, tgt_off;
    for (int d = 0; d < N_tgt; ++d) {
        tgt_shape[d] = tgt_meta.global_shape[d];
        tgt_off[d] = tgt_meta.offset[d];
    }
    return CrossSetRegionMap<N_src, N_tgt>(sr_root->get_operand_region(1), src_shape, src_off,
                                           sr_root->region, tgt_shape, tgt_off);
}

template <int N>
static void dispatch_cross_send(DType dt, PartitionImpl<N>* partition, DAGNode* node, int source_name,
                                int target_name, ASTNode* sr_root, ArrayDAGGroup* dag_group) {
    int target_ndims = sr_root->ndims;

    auto src_it = partition->arrays.find(source_name);
    if (src_it == partition->arrays.end() || src_it->second->local_size() == 0)
        return;

    // The source array's physical decomposition is authoritative on the sender.
    ArrayDAGGroup::ArrayMetadata src_meta{};
    src_meta.ndims = N;
    for (int d = 0; d < N; ++d) {
        src_meta.global_shape[d] = src_it->second->global_shape[d];
        src_meta.offset[d] = src_it->second->decomp.offset[d];
    }
    src_meta.tile = src_it->second->decomp.tile;

    // Look up target array decomposition from array_meta
    auto tgt_meta_it = dag_group->array_meta.find(target_name);
    if (tgt_meta_it == dag_group->array_meta.end())
        CkAbort("Cross-partition SET_REGION: no metadata for target name=%d", target_name);
    const auto& tgt_meta = tgt_meta_it->second;

    // Dispatch to the correct N_tgt template
    if (target_ndims == 1) {
        auto xmap = make_cross_set_region_map<N, 1>(src_meta, tgt_meta, sr_root);
        dispatch_cross_set_region_send<N, 1>(dt, partition, node, source_name, xmap,
                                             get_partition_proxy<1>(dag_group),
                                             tgt_meta.template decomp<1>());
    } else if (target_ndims == 2) {
        auto xmap = make_cross_set_region_map<N, 2>(src_meta, tgt_meta, sr_root);
        dispatch_cross_set_region_send<N, 2>(dt, partition, node, source_name, xmap,
                                             get_partition_proxy<2>(dag_group),
                                             tgt_meta.template decomp<2>());
    } else if (target_ndims == 3) {
        auto xmap = make_cross_set_region_map<N, 3>(src_meta, tgt_meta, sr_root);
        dispatch_cross_set_region_send<N, 3>(dt, partition, node, source_name, xmap,
                                             get_partition_proxy<3>(dag_group),
                                             tgt_meta.template decomp<3>());
    }
}

/// Receiver side: number of fragments the source partition sends to my chare.
template <int N_src, int N_tgt>
static int expected_cross_set_region_msgs(const ArrayDAGGroup::ArrayMetadata& src_meta,
                                          const ArrayDAGGroup::ArrayMetadata& tgt_meta,
                                          ASTNode* sr_root, const ArrayRegion<N_tgt>& my_chare) {
    auto xmap = make_cross_set_region_map<N_src, N_tgt>(src_meta, tgt_meta, sr_root);
    return count_cross_set_region_msgs<N_src, N_tgt>(xmap, src_meta.template decomp<N_src>(),
                                                     my_chare);
}

template <int N>
void ArrayDAGExecutorND<N>::execute_cross_set_region_node(DAGNode* node) {
    ASTNode* sr_root = nullptr;
    for (ASTNode* root : node->ast->roots) {
        if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION) {
            sr_root = root;
            break;
        }
    }
    if (!sr_root) {
        node_finished(node->id);
        return;
    }

    int target_name = sr_root->operands[0]->result_name;
    int source_name = sr_root->operands[1]->result_name;
    int target_ndims = sr_root->ndims;
    int source_ndims = sr_root->operands[1]->ndims;

    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    DType dt = determine_dtype<N>(node, partition->arrays);

    DBG_PRINT("[Chare %d] execute_cross_set_region_node<%d>: node_id=%d target=%d(nd=%d) "
              "source=%d(nd=%d)\n",
              partition->index[0], N, node->id, target_name, target_ndims, source_name,
              source_ndims);

    if (N == source_ndims) {
        // === SOURCE SIDE: send data to target partition ===
        dispatch_cross_send<N>(dt, partition, node, source_name, target_name, sr_root, dag_group);
        node_finished(node->id);

    } else if (N == target_ndims) {
        // === TARGET SIDE: set up to receive data from source partition ===
        auto tgt_it = partition->arrays.find(target_name);
        if (tgt_it == partition->arrays.end()) {
            node_finished(node->id);
            return;
        }

        auto nd_idx = partition->nd_index();
        auto tgt_decomp = tgt_it->second->decomp;
        auto r_chare_global = tgt_decomp.chare_region_global(nd_idx);
        auto* tgt_region = static_cast<ArrayRegion<N>*>(sr_root->region);

        // Preserve the target lattice for stepped target slices.  The wire
        // region is array-local; chare regions are decomp-global.
        auto [my_input, has_overlap] =
            intersect(tgt_decomp.to_global(*tgt_region), r_chare_global);
        if (!has_overlap) {
            node_finished(node->id);
            return;
        }

        // Compute expected messages from source partition
        auto meta_it = dag_group->array_meta.find(source_name);
        if (meta_it == dag_group->array_meta.end()) {
            CkAbort("Cross-partition SET_REGION: could not find source metadata for name=%d",
                    source_name);
            return;
        }
        int src_ndims = meta_it->second.ndims;

        // Count exactly the source chares whose share of the source view maps
        // onto this chare, using the same lattice map as the sender.
        ArrayDAGGroup::ArrayMetadata tgt_meta{};
        tgt_meta.ndims = N;
        for (int d = 0; d < N; ++d) {
            tgt_meta.global_shape[d] = tgt_decomp.global_shape[d];
            tgt_meta.offset[d] = tgt_decomp.offset[d];
        }
        tgt_meta.tile = tgt_decomp.tile;
        int expected_msgs = 0;
        switch (src_ndims) {
        case 1:
            expected_msgs = expected_cross_set_region_msgs<1, N>(meta_it->second, tgt_meta,
                                                                 sr_root, r_chare_global);
            break;
        case 2:
            expected_msgs = expected_cross_set_region_msgs<2, N>(meta_it->second, tgt_meta,
                                                                 sr_root, r_chare_global);
            break;
        case 3:
            expected_msgs = expected_cross_set_region_msgs<3, N>(meta_it->second, tgt_meta,
                                                                 sr_root, r_chare_global);
            break;
        }

        DBG_PRINT("[Chare %d]   target side: expected_msgs=%d\n", partition->index[0],
                  expected_msgs);

        auto pre_it = pending.find(node->id);
        int pre_arrived = 0;
        std::unordered_map<int, std::vector<RemoteBuffer<N>>> pre_buffers;
        if (pre_it != pending.end()) {
            pre_arrived = -(pre_it->second.expected_msgs);
            pre_buffers = std::move(pre_it->second.remote_buffers);
        }
        int remaining = expected_msgs - pre_arrived;
        pending[node->id] = {node, remaining, {my_input}, std::move(pre_buffers)};
        if (remaining <= 0)
            on_comm_done(node->id);

    } else {
        node_finished(node->id);
    }
}

template <int N_src, typename T>
static void diag_send(PartitionImpl<N_src>* partition, DAGNode* node, int source_name,
                      int result_name, int k_offset, ArrayDAGGroup* dag_group) {
    auto src_it = partition->arrays.find(source_name);
    if (src_it == partition->arrays.end() || src_it->second->local_size() == 0)
        return;

    auto* src = static_cast<Array<N_src, T>*>(src_it->second);

    if constexpr (N_src == 1) {
        int tile_2d = array_tile(dag_group->array_meta, result_name, 2);

        auto nd_idx = partition->nd_index();
        int my_start = src_it->second->decomp.chare_start_global(0, nd_idx[0]);
        int local_size = src->region.size(0);

        struct ChareRange {
            ChareIndex<2> target_ci;
            int first_vec_idx, last_vec_idx;
            int first_local, count;
        };
        std::vector<ChareRange> ranges;
        for (int i = 0; i < local_size;) {
            int vec_idx = my_start + i;
            int row = (k_offset >= 0) ? vec_idx : vec_idx - k_offset;
            int col = (k_offset >= 0) ? vec_idx + k_offset : vec_idx;
            int target_row_chare = row / tile_2d;
            int target_col_chare = col / tile_2d;
            ChareIndex<2> target_ci;
            target_ci.idx[0] = target_row_chare;
            target_ci.idx[1] = target_col_chare;

            int j = i + 1;
            while (j < local_size) {
                int vj = my_start + j;
                int rj = (k_offset >= 0) ? vj : vj - k_offset;
                int cj = (k_offset >= 0) ? vj + k_offset : vj;
                if (rj / tile_2d != target_row_chare || cj / tile_2d != target_col_chare)
                    break;
                ++j;
            }
            ranges.push_back({target_ci, vec_idx, my_start + j - 1, i, j - i});
            i = j;
        }

        for (auto& r : ranges) {
            int first_row = (k_offset >= 0) ? r.first_vec_idx : r.first_vec_idx - k_offset;
            int last_row = (k_offset >= 0) ? r.last_vec_idx : r.last_vec_idx - k_offset;
            int first_col = (k_offset >= 0) ? r.first_vec_idx + k_offset : r.first_vec_idx;
            int last_col = (k_offset >= 0) ? r.last_vec_idx + k_offset : r.last_vec_idx;

            int region_data[2 * 3];
            region_data[0] = first_row;
            region_data[1] = last_row + 1;
            region_data[2] = 1;
            region_data[3] = first_col;
            region_data[4] = last_col + 1;
            region_data[5] = 1;

            int64_t byte_size = r.count * sizeof(T);
#ifndef NDEBUG
            partition->comm_bytes_sent += byte_size;
#endif
#ifdef USE_KOKKOS
            T* send_buf = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(byte_size));
            T* src_device = static_cast<T*>(src->device_data_ptr()) + r.first_local;
            Kokkos::parallel_for(
                CT_COMM_POLICY(partition, r.count),
                KOKKOS_LAMBDA(int j) { send_buf[j] = src_device[j]; });
            device_pack_send<1, 2>(partition, dag_group->partition_proxy_2, r.target_ci,
                                   node->id, 0, source_name, region_data, byte_size, send_buf);
#else
            src->copyToHost();
            T* src_data = static_cast<T*>(src->data_ptr());
            T* send_buf = new T[r.count];
            for (int j = 0; j < r.count; ++j)
                send_buf[j] = src_data[r.first_local + j];
            proxy_at<2>(dag_group->partition_proxy_2, r.target_ci)
                .receive_data(node->id, /*input_index=*/0, source_name, 2, region_data,
                              byte_size, reinterpret_cast<char*>(send_buf));
            delete[] send_buf;
#endif
        }
    } else if constexpr (N_src == 2) {
        int tile_1d = array_tile(dag_group->array_meta, result_name, 1);

        auto nd_idx = partition->nd_index();
        auto src_decomp = src_it->second->decomp;
        int row_start = src_decomp.chare_start_global(0, nd_idx[0]);
        int col_start = src_decomp.chare_start_global(1, nd_idx[1]);
        int local_rows = src->region.size(0);
        int local_cols = src->region.size(1);

        int row_lo = row_start;
        int row_hi = row_start + local_rows;
        int col_lo = col_start;
        int col_hi = col_start + local_cols;

        int diag_lo, diag_hi;
        if (k_offset >= 0) {
            diag_lo = std::max(row_lo, col_lo - k_offset);
            diag_hi = std::min(row_hi, col_hi - k_offset);
        } else {
            diag_lo = std::max(row_lo + k_offset, col_lo);
            diag_hi = std::min(row_hi + k_offset, col_hi);
        }

        if (diag_lo >= diag_hi)
            return;

        struct ChareRange {
            ChareIndex<1> target_ci;
            int first_vec_idx;
            int count;
        };
        std::vector<ChareRange> ranges;
        for (int vec_idx = diag_lo; vec_idx < diag_hi;) {
            int target_chare = vec_idx / tile_1d;
            ChareIndex<1> target_ci;
            target_ci.idx[0] = target_chare;
            int end = std::min(diag_hi, (target_chare + 1) * tile_1d);
            ranges.push_back({target_ci, vec_idx, end - vec_idx});
            vec_idx = end;
        }

        for (auto& r : ranges) {
            int region_data[1 * 3];
            region_data[0] = r.first_vec_idx;
            region_data[1] = r.first_vec_idx + r.count;
            region_data[2] = 1;

            int64_t byte_size = r.count * sizeof(T);
#ifndef NDEBUG
            partition->comm_bytes_sent += byte_size;
#endif
#ifdef USE_KOKKOS
            T* send_buf = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(byte_size));
            T* src_device = static_cast<T*>(src->device_data_ptr());
            int rs_cap = row_start, cs_cap = col_start, lc_cap = local_cols;
            int ko_cap = k_offset;
            int fvi_cap = r.first_vec_idx;
            Kokkos::parallel_for(
                CT_COMM_POLICY(partition, r.count),
                KOKKOS_LAMBDA(int j) {
                    int vec_idx = fvi_cap + j;
                    int row = (ko_cap >= 0) ? vec_idx : vec_idx - ko_cap;
                    int col = (ko_cap >= 0) ? vec_idx + ko_cap : vec_idx;
                    int local_row = row - rs_cap;
                    int local_col = col - cs_cap;
                    send_buf[j] = src_device[local_row * lc_cap + local_col];
                });
            device_pack_send<2, 1>(partition, dag_group->partition_proxy_1, r.target_ci,
                                   node->id, 0, source_name, region_data, byte_size, send_buf);
#else
            src->copyToHost();
            T* src_data = static_cast<T*>(src->data_ptr());
            T* send_buf = new T[r.count];
            for (int j = 0; j < r.count; ++j) {
                int vec_idx = r.first_vec_idx + j;
                int row, col;
                if (k_offset >= 0) {
                    row = vec_idx;
                    col = vec_idx + k_offset;
                } else {
                    row = vec_idx - k_offset;
                    col = vec_idx;
                }
                int local_row = row - row_start;
                int local_col = col - col_start;
                send_buf[j] = src_data[local_row * local_cols + local_col];
            }
            proxy_at<1>(dag_group->partition_proxy_1, r.target_ci)
                .receive_data(node->id, /*input_index=*/0, source_name, 1, region_data,
                              byte_size, reinterpret_cast<char*>(send_buf));
            delete[] send_buf;
#endif
        }
    }
}

template <int N_src>
static void dispatch_diag_send(DType dt, PartitionImpl<N_src>* partition, DAGNode* node,
                               int source_name, int result_name, int k_offset,
                               ArrayDAGGroup* dag_group) {
    switch (dt) {
    case DType::FLOAT32:
        diag_send<N_src, float>(partition, node, source_name, result_name, k_offset, dag_group);
        break;
    case DType::FLOAT64:
        diag_send<N_src, double>(partition, node, source_name, result_name, k_offset, dag_group);
        break;
    case DType::INT32:
        diag_send<N_src, int32_t>(partition, node, source_name, result_name, k_offset, dag_group);
        break;
    case DType::INT64:
        diag_send<N_src, int64_t>(partition, node, source_name, result_name, k_offset, dag_group);
        break;
    }
}

template <int N>
void ArrayDAGExecutorND<N>::execute_diag_node(DAGNode* node) {
    ASTNode* diag_root = nullptr;
    for (ASTNode* root : node->ast->roots) {
        if (static_cast<Opcode>(root->opcode) == Opcode::DIAG) {
            diag_root = root;
            break;
        }
    }
    if (!diag_root) {
        node_finished(node->id);
        return;
    }

    int input_name = diag_root->operands[0]->result_name;
    int result_name = diag_root->result_name;
    int k_offset = 0;
    if (diag_root->operands.size() >= 2 && diag_root->operands[1]->is_scalar)
        k_offset = (int)diag_root->operands[1]->scalar;

    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    DType dt = determine_dtype<N>(node, partition->arrays);

    auto meta_it = dag_group->array_meta.find(input_name);
    if (meta_it == dag_group->array_meta.end()) {
        node_finished(node->id);
        return;
    }
    int input_ndims = meta_it->second.ndims;

    DBG_PRINT("[Chare %d] execute_diag_node<%d>: node_id=%d input=%d(nd=%d) k=%d\n",
              partition->index[0], N, node->id, input_name, input_ndims, k_offset);

    if (input_ndims == 1) {
        if constexpr (N == 1) {
            dispatch_diag_send<1>(dt, partition, node, input_name, result_name, k_offset, dag_group);
            node_finished(node->id);
        } else if constexpr (N == 2) {
            int tile_1d = array_tile(dag_group->array_meta, input_name, 1);

            auto nd_idx = partition->nd_index();
            auto result_meta_it = dag_group->array_meta.find(result_name);
            if (result_meta_it == dag_group->array_meta.end()) {
                node_finished(node->id);
                return;
            }
            int out_n = result_meta_it->second.global_shape[0];
            auto result_decomp = result_meta_it->second.template decomp<2>();
            auto result_chare = result_decomp.chare_region_global(nd_idx);
            int row_start = result_chare.start[0];
            int col_start = result_chare.start[1];
            int local_rows = result_chare.stop[0] - row_start;
            int local_cols = result_chare.stop[1] - col_start;
            if (local_rows <= 0 || local_cols <= 0) {
                node_finished(node->id);
                return;
            }

            int vec_len = meta_it->second.global_shape[0];
            int diag_lo, diag_hi;
            if (k_offset >= 0) {
                diag_lo = std::max(row_start, col_start - k_offset);
                diag_hi = std::min(row_start + local_rows, col_start + local_cols - k_offset);
            } else {
                diag_lo = std::max(row_start + k_offset, col_start);
                diag_hi = std::min(row_start + local_rows + k_offset, col_start + local_cols);
            }
            diag_lo = std::max(diag_lo, 0);
            diag_hi = std::min(diag_hi, vec_len);

            if (diag_lo >= diag_hi) {
                std::array<int, 2> out_start = {0, 0};
                std::array<int, 2> out_stop = {local_rows, local_cols};
                std::array<int, 2> out_step = {1, 1};
                std::array<int, 2> out_gs = {out_n, out_n};
                ArrayRegion<2> out_region(out_start, out_stop, out_step);
                ArrayDecomp<2> out_decomp = dag_group->array_meta[result_name].template decomp<2>();
                partition->arrays[result_name] =
                    partition->allocate_or_reuse(out_region, out_gs, result_name, dt, out_decomp);
#ifndef USE_KOKKOS
                std::memset(partition->arrays[result_name]->data_ptr(), 0,
                            local_rows * local_cols * dtype_size(dt));
#endif
                node_finished(node->id);
                return;
            }

            int first_1d_chare = diag_lo / tile_1d;
            int last_1d_chare = (diag_hi - 1) / tile_1d;
            int expected_msgs = last_1d_chare - first_1d_chare + 1;

            std::array<int, 2> my_start = {row_start, col_start};
            std::array<int, 2> my_stop = {row_start + local_rows, col_start + local_cols};
            std::array<int, 2> my_step = {1, 1};
            ArrayRegion<2> my_input(my_start, my_stop, my_step);

            auto pre_it = pending.find(node->id);
            int pre_arrived = 0;
            std::unordered_map<int, std::vector<RemoteBuffer<2>>> pre_buffers;
            if (pre_it != pending.end()) {
                pre_arrived = -(pre_it->second.expected_msgs);
                pre_buffers = std::move(pre_it->second.remote_buffers);
            }
            int remaining = expected_msgs - pre_arrived;
            pending[node->id] = {node, remaining, {my_input}, std::move(pre_buffers)};
            if (remaining <= 0)
                on_comm_done(node->id);
        }
    } else if (input_ndims == 2) {
        if constexpr (N == 2) {
            dispatch_diag_send<2>(dt, partition, node, input_name, result_name, k_offset, dag_group);
            node_finished(node->id);
        } else if constexpr (N == 1) {
            int tile_2d = array_tile(dag_group->array_meta, input_name, 2);

            auto nd_idx = partition->nd_index();
            auto result_meta_it = dag_group->array_meta.find(result_name);
            if (result_meta_it == dag_group->array_meta.end()) {
                node_finished(node->id);
                return;
            }
            int diag_len = result_meta_it->second.global_shape[0];
            auto result_decomp = result_meta_it->second.template decomp<1>();
            auto result_chare = result_decomp.chare_region_global(nd_idx);
            int my_start_idx = result_chare.start[0];
            int local_size = result_chare.stop[0] - my_start_idx;
            if (local_size <= 0) {
                node_finished(node->id);
                return;
            }

            int my_end_idx = my_start_idx + local_size;

            std::unordered_set<ChareIndex<2>, ChareIndexHash<2>> source_chares;
            int M = meta_it->second.global_shape[0];
            int N_cols = meta_it->second.global_shape[1];
            for (int vec_idx = my_start_idx; vec_idx < my_end_idx; ++vec_idx) {
                int row, col;
                if (k_offset >= 0) {
                    row = vec_idx;
                    col = vec_idx + k_offset;
                } else {
                    row = vec_idx - k_offset;
                    col = vec_idx;
                }
                if (row >= 0 && row < M && col >= 0 && col < N_cols) {
                    int src_row_chare = row / tile_2d;
                    int src_col_chare = col / tile_2d;
                    ChareIndex<2> src_ci;
                    src_ci.idx[0] = src_row_chare;
                    src_ci.idx[1] = src_col_chare;
                    source_chares.insert(src_ci);
                }
            }
            int expected_msgs = (int)source_chares.size();

            std::array<int, 1> my_start_arr = {my_start_idx};
            std::array<int, 1> my_stop_arr = {my_end_idx};
            std::array<int, 1> my_step_arr = {1};
            ArrayRegion<1> my_input(my_start_arr, my_stop_arr, my_step_arr);

            auto pre_it = pending.find(node->id);
            int pre_arrived = 0;
            std::unordered_map<int, std::vector<RemoteBuffer<1>>> pre_buffers;
            if (pre_it != pending.end()) {
                pre_arrived = -(pre_it->second.expected_msgs);
                pre_buffers = std::move(pre_it->second.remote_buffers);
            }
            int remaining = expected_msgs - pre_arrived;
            pending[node->id] = {node, remaining, {my_input}, std::move(pre_buffers)};
            if (remaining <= 0)
                on_comm_done(node->id);
        }
    } else {
        node_finished(node->id);
    }
}

template <int N_tgt>
struct PartitionProxyHelper;

template <>
struct PartitionProxyHelper<1> {
    static auto& get(ArrayDAGGroup* g) { return g->partition_proxy_1; }
};
template <>
struct PartitionProxyHelper<2> {
    static auto& get(ArrayDAGGroup* g) { return g->partition_proxy_2; }
};
template <>
struct PartitionProxyHelper<3> {
    static auto& get(ArrayDAGGroup* g) { return g->partition_proxy_3; }
};

template <int N_src, int N_tgt, typename T>
static void tile_send(PartitionImpl<N_src>* partition, DAGNode* node, int source_name,
                      int result_name, std::array<int, 3> const& reps,
                      std::array<int, 3> const& input_shape, int out_ndims,
                      ArrayDAGGroup* dag_group) {
    auto src_it = partition->arrays.find(source_name);
    if (src_it == partition->arrays.end() || src_it->second->local_size() == 0)
        return;

    auto* src = static_cast<Array<N_src, T>*>(src_it->second);
    auto nd_idx = partition->nd_index();
    auto src_decomp = src_it->second->decomp;

    std::array<int, N_src> src_start, src_stop;
    for (int d = 0; d < N_src; ++d) {
        src_start[d] = src_decomp.chare_start_global(d, nd_idx[d]);
        src_stop[d] = std::min(src_start[d] + src->region.size(d),
                               src_decomp.offset[d] + src_decomp.global_shape[d]);
    }

    auto result_meta_it = dag_group->array_meta.find(result_name);
    if (result_meta_it == dag_group->array_meta.end())
        return;
    auto result_decomp = result_meta_it->second.template decomp<N_tgt>();

    int delta = out_ndims - N_src;
    std::array<int, N_tgt> out_shape;
    for (int d = 0; d < N_tgt; ++d)
        out_shape[d] = result_meta_it->second.global_shape[d];

    int result_tile = result_meta_it->second.tile;

    std::array<std::vector<int>, N_tgt> chare_indices_per_dim;
    for (int d = 0; d < N_tgt; ++d) {
        int num_chares_d = result_decomp.num_chares(d);
        if (d < delta) {
            for (int ci = 0; ci < num_chares_d; ++ci)
                chare_indices_per_dim[d].push_back(ci);
        } else {
            int src_d = d - delta;
            int in_size = input_shape[src_d];
            for (int ci = 0; ci < num_chares_d; ++ci) {
                int chare_lo = std::max(ci * result_tile, result_decomp.offset[d]);
                int chare_hi = std::min((ci + 1) * result_tile,
                                        result_decomp.offset[d] + out_shape[d]);
                if (chare_lo >= chare_hi)
                    continue;
                bool overlaps = false;
                if (chare_hi - chare_lo >= in_size) {
                    overlaps = true;
                } else {
                    int mod_lo = chare_lo % in_size;
                    int mod_hi = (chare_hi - 1) % in_size;
                    if (mod_lo <= mod_hi) {
                        overlaps = !(mod_hi < src_start[src_d] || mod_lo >= src_stop[src_d]);
                    } else {
                        overlaps = !(mod_hi < src_start[src_d] && mod_lo >= src_stop[src_d]);
                    }
                }
                if (overlaps)
                    chare_indices_per_dim[d].push_back(ci);
            }
        }
    }

    int local_size = src->local_size();
    int64_t byte_size = local_size * sizeof(T);

    std::function<void(int, ChareIndex<N_tgt>&)> send_to_chares;
    send_to_chares = [&](int dim, ChareIndex<N_tgt>& ci) {
        if (dim == N_tgt) {
            int region_data[N_tgt * 3];
            for (int d = 0; d < N_tgt; ++d) {
                if (d < delta) {
                    region_data[d * 3 + 0] = 0;
                    region_data[d * 3 + 1] = 1;
                    region_data[d * 3 + 2] = 1;
                } else {
                    int src_d = d - delta;
                    region_data[d * 3 + 0] = src_start[src_d];
                    region_data[d * 3 + 1] = src_stop[src_d];
                    region_data[d * 3 + 2] = 1;
                }
            }

#ifndef USE_KOKKOS
            src->copyToHost();
            T* src_data = static_cast<T*>(src->data_ptr());
            T* send_buf = new T[local_size];
            std::memcpy(send_buf, src_data, byte_size);
#ifndef NDEBUG
            partition->comm_bytes_sent += byte_size;
#endif
            proxy_at<N_tgt>(PartitionProxyHelper<N_tgt>::get(dag_group), ci)
                .receive_data(node->id, /*input_index=*/0, source_name, N_tgt, region_data,
                              byte_size, reinterpret_cast<char*>(send_buf));
            delete[] send_buf;
#else
            T* send_buf = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(byte_size));
            T* src_device = static_cast<T*>(src->device_data_ptr());
            Kokkos::parallel_for(
                CT_COMM_POLICY(partition, local_size),
                KOKKOS_LAMBDA(int j) { send_buf[j] = src_device[j]; });
            device_pack_send<N_src, N_tgt>(partition,
                                           PartitionProxyHelper<N_tgt>::get(dag_group), ci,
                                           node->id, 0, source_name, region_data, byte_size,
                                           send_buf);
#endif
            return;
        }
        for (int idx : chare_indices_per_dim[dim]) {
            ci.idx[dim] = idx;
            send_to_chares(dim + 1, ci);
        }
    };

    ChareIndex<N_tgt> ci;
    send_to_chares(0, ci);
}

template <int N_src, int N_tgt>
static void dispatch_tile_send(DType dt, PartitionImpl<N_src>* partition, DAGNode* node,
                               int source_name, int result_name,
                               std::array<int, 3> const& reps,
                               std::array<int, 3> const& input_shape, int out_ndims,
                               ArrayDAGGroup* dag_group) {
    switch (dt) {
    case DType::FLOAT32:
        tile_send<N_src, N_tgt, float>(partition, node, source_name, result_name, reps,
                                       input_shape, out_ndims, dag_group);
        break;
    case DType::FLOAT64:
        tile_send<N_src, N_tgt, double>(partition, node, source_name, result_name, reps,
                                        input_shape, out_ndims, dag_group);
        break;
    case DType::INT32:
        tile_send<N_src, N_tgt, int32_t>(partition, node, source_name, result_name, reps,
                                         input_shape, out_ndims, dag_group);
        break;
    case DType::INT64:
        tile_send<N_src, N_tgt, int64_t>(partition, node, source_name, result_name, reps,
                                         input_shape, out_ndims, dag_group);
        break;
    }
}

template <int N>
void ArrayDAGExecutorND<N>::execute_tile_node(DAGNode* node) {
    ASTNode* tile_root = nullptr;
    for (ASTNode* root : node->ast->roots) {
        if (static_cast<Opcode>(root->opcode) == Opcode::TILE) {
            tile_root = root;
            break;
        }
    }
    if (!tile_root) {
        node_finished(node->id);
        return;
    }

    int input_name = tile_root->operands[0]->result_name;
    int result_name = tile_root->result_name;
    int out_ndims = tile_root->ndims;

    std::array<int, 3> reps = {1, 1, 1};
    int num_reps = (int)tile_root->operands.size() - 1;
    for (int d = 0; d < num_reps && d < 3; d++) {
        if (tile_root->operands[d + 1]->is_scalar)
            reps[d] = (int)tile_root->operands[d + 1]->scalar;
    }

    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    DType dt = determine_dtype<N>(node, partition->arrays);

    auto meta_it = dag_group->array_meta.find(input_name);
    if (meta_it == dag_group->array_meta.end()) {
        node_finished(node->id);
        return;
    }
    int input_ndims = meta_it->second.ndims;
    std::array<int, 3> input_shape = meta_it->second.global_shape;

    DBG_PRINT("[Chare %d] execute_tile_node<%d>: node_id=%d input=%d(nd=%d) result_nd=%d "
              "reps=(%d,%d,%d)\n",
              partition->index[0], N, node->id, input_name, input_ndims, out_ndims,
              reps[0], reps[1], reps[2]);

    if (N == input_ndims) {
        if (input_ndims == out_ndims) {
            switch (N) {
            case 1:
                dispatch_tile_send<1, 1>(dt, reinterpret_cast<PartitionImpl<1>*>(partition),
                                         node, input_name, result_name, reps, input_shape,
                                         out_ndims, dag_group);
                break;
            case 2:
                dispatch_tile_send<2, 2>(dt, reinterpret_cast<PartitionImpl<2>*>(partition),
                                         node, input_name, result_name, reps, input_shape,
                                         out_ndims, dag_group);
                break;
            case 3:
                dispatch_tile_send<3, 3>(dt, reinterpret_cast<PartitionImpl<3>*>(partition),
                                         node, input_name, result_name, reps, input_shape,
                                         out_ndims, dag_group);
                break;
            }
        } else {
            if (input_ndims == 1 && out_ndims == 2) {
                dispatch_tile_send<1, 2>(dt, reinterpret_cast<PartitionImpl<1>*>(partition),
                                         node, input_name, result_name, reps, input_shape,
                                         out_ndims, dag_group);
            } else if (input_ndims == 1 && out_ndims == 3) {
                dispatch_tile_send<1, 3>(dt, reinterpret_cast<PartitionImpl<1>*>(partition),
                                         node, input_name, result_name, reps, input_shape,
                                         out_ndims, dag_group);
            } else if (input_ndims == 2 && out_ndims == 3) {
                dispatch_tile_send<2, 3>(dt, reinterpret_cast<PartitionImpl<2>*>(partition),
                                         node, input_name, result_name, reps, input_shape,
                                         out_ndims, dag_group);
            }
        }
        if (N != out_ndims) {
            node_finished(node->id);
        }
    }

    if (N == out_ndims) {
        auto nd_idx = partition->nd_index();

        auto result_meta_it = dag_group->array_meta.find(result_name);
        if (result_meta_it == dag_group->array_meta.end()) {
            node_finished(node->id);
            return;
        }
        auto result_decomp = result_meta_it->second.template decomp<N>();
        auto result_chare = result_decomp.chare_region_global(nd_idx);

        bool has_extent = true;
        for (int d = 0; d < N; ++d) {
            if (result_chare.stop[d] <= result_chare.start[d]) {
                has_extent = false;
                break;
            }
        }
        if (!has_extent) {
            node_finished(node->id);
            return;
        }

        int delta = out_ndims - input_ndims;
        auto input_decomp_meta = dag_group->array_meta.find(input_name);
        int input_tile = input_decomp_meta->second.tile;

        int expected_msgs = 1;
        for (int src_d = 0; src_d < input_ndims; ++src_d) {
            int out_d = src_d + delta;
            int in_size = input_shape[src_d];
            int chare_lo = result_chare.start[out_d];
            int chare_hi = result_chare.stop[out_d];

            std::set<int> needed_src_chares;
            for (int p = chare_lo; p < chare_hi;) {
                int inp_pos = p % in_size;
                int src_chare = inp_pos / input_tile;
                needed_src_chares.insert(src_chare);
                int next_boundary = (src_chare + 1) * input_tile - inp_pos + p;
                int next_wrap = p + (in_size - inp_pos);
                p = std::min({next_boundary, next_wrap, chare_hi});
            }
            expected_msgs *= (int)needed_src_chares.size();
        }

        ArrayRegion<N> my_input;
        for (int d = 0; d < N; ++d) {
            my_input.start[d] = result_chare.start[d];
            my_input.stop[d] = result_chare.stop[d];
            my_input.step[d] = 1;
        }

        auto pre_it = pending.find(node->id);
        int pre_arrived = 0;
        std::unordered_map<int, std::vector<RemoteBuffer<N>>> pre_buffers;
        if (pre_it != pending.end()) {
            pre_arrived = -(pre_it->second.expected_msgs);
            pre_buffers = std::move(pre_it->second.remote_buffers);
        }
        int remaining = expected_msgs - pre_arrived;
        pending[node->id] = {node, remaining, {my_input}, std::move(pre_buffers)};
        if (remaining <= 0)
            on_comm_done(node->id);
    }
}

template void ArrayDAGExecutorND<1>::execute_cross_set_region_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_cross_set_region_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_cross_set_region_node(DAGNode*);

template void ArrayDAGExecutorND<1>::execute_diag_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_diag_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_diag_node(DAGNode*);

template void ArrayDAGExecutorND<1>::execute_tile_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_tile_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_tile_node(DAGNode*);
