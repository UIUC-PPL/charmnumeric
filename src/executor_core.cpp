#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <cstring>
#include <unordered_set>

template <int N, typename T>
static void send_remote_input(PartitionImpl<N>* partition, DAGNode* node, const RemoteSend<N>& send,
                              int inp_name, const std::array<int, N>& nd_idx) {
    auto arr_it = partition->arrays.find(inp_name);
    if (arr_it == partition->arrays.end())
        return;
    Array<N, T>* arr = static_cast<Array<N, T>*>(arr_it->second);

    // Use the input array's own decomp for chare region and local offset
    ArrayRegion<N> r_chare_inp = arr->decomp.chare_region_global(nd_idx);
    auto [overlap, has_overlap] = intersect(send.region, r_chare_inp);
    if (!has_overlap)
        return;

    std::array<int, N> cs;
    for (int d = 0; d < N; ++d)
        cs[d] = arr->decomp.chare_start_global(d, nd_idx[d]);
    int64_t total_size = overlap.size();

    {
        bool out_of_bounds = false;
        for (int d = 0; d < N; ++d) {
            int local_start_d = overlap.start[d] - cs[d];
            int phys_extent = local_start_d + (overlap.size(d) - 1) * overlap.step[d] + 1;
            if (local_start_d < 0 || phys_extent > arr->region.size(d)) {
                out_of_bounds = true;
                break;
            }
        }
        if (out_of_bounds)
            return;
    }

    std::array<int, N> arr_strides;
    arr_strides[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        arr_strides[d] = arr_strides[d + 1] * arr->region.size(d + 1);

    int region_data[N * 3];
    for (int d = 0; d < N; ++d) {
        region_data[d * 3 + 0] = overlap.start[d];
        region_data[d * 3 + 1] = overlap.stop[d];
        region_data[d * 3 + 2] = overlap.step[d];
    }
    int64_t byte_size = total_size * sizeof(T);

#ifndef NDEBUG
    partition->comm_bytes_sent += byte_size;
#endif

#ifdef USE_KOKKOS
    // Pack on device and send via direct GPU messaging
    T* send_buf = static_cast<T*>(comm_alloc(total_size * sizeof(T)));
    T* src_device = static_cast<T*>(arr->device_data_ptr());

    // Capture strides/offsets in plain arrays for KOKKOS_LAMBDA
    int overlap_sizes[N], local_starts[N], src_strides_arr[N], overlap_steps[N];
    for (int d = 0; d < N; d++) {
        overlap_sizes[d] = overlap.size(d);
        local_starts[d] = overlap.start[d] - cs[d];
        src_strides_arr[d] = arr_strides[d];
        overlap_steps[d] = overlap.step[d];
    }

    Kokkos::parallel_for(
        CT_COMM_POLICY(partition, total_size), KOKKOS_LAMBDA(int flat_idx) {
            int remaining = flat_idx;
            int src_flat = 0;
            for (int d = N - 1; d >= 0; --d) {
                int coord_d = remaining % overlap_sizes[d];
                remaining /= overlap_sizes[d];
                src_flat += (local_starts[d] + coord_d * overlap_steps[d]) * src_strides_arr[d];
            }
            send_buf[flat_idx] = src_device[src_flat];
        });

    device_pack_send<N>(partition, partition->thisProxy, send.target,
                        node->id, send.input_index, inp_name, region_data,
                        byte_size, send_buf);
#else
    // Host path: pack on CPU and send via regular Charm++ messaging
    arr->copyToHost();
    T* send_buf = new T[total_size];
    T* host_data = static_cast<T*>(arr->data_ptr());

    if (overlap.step[N - 1] == 1) {
        // Fast path: innermost dimension is contiguous, use memcpy
        int64_t inner_size = overlap.size(N - 1);
        int64_t buf_offset = 0;
        std::array<int, N> idx = {};
        while (true) {
            int64_t src_flat = 0;
            for (int d = 0; d < N; ++d)
                src_flat += (int64_t)(overlap.start[d] + idx[d] * overlap.step[d] - cs[d]) *
                            arr_strides[d];
            memcpy(send_buf + buf_offset, host_data + src_flat, inner_size * sizeof(T));
            buf_offset += inner_size;

            int d = N - 2;
            while (d >= 0) {
                if (++idx[d] < overlap.size(d))
                    break;
                idx[d] = 0;
                --d;
            }
            if (d < 0)
                break;
        }
    } else {
        // Element-wise packing for non-unit innermost step
        int64_t buf_offset = 0;
        std::array<int, N> idx = {};
        while (true) {
            int64_t src_flat = 0;
            for (int d = 0; d < N; ++d)
                src_flat += (int64_t)(overlap.start[d] + idx[d] * overlap.step[d] - cs[d]) *
                            arr_strides[d];
            send_buf[buf_offset++] = host_data[src_flat];

            int d = N - 1;
            while (d >= 0) {
                if (++idx[d] < overlap.size(d))
                    break;
                idx[d] = 0;
                --d;
            }
            if (d < 0)
                break;
        }
    }

    proxy_at<N>(partition->thisProxy, send.target)
        .receive_data(node->id, send.input_index, inp_name, N, region_data, byte_size,
                      reinterpret_cast<char*>(send_buf));
    delete[] send_buf;
#endif
}

/// Sends an input array's elements in that array's own dtype (kernels may mix dtypes).
template <int N>
static void dispatch_send(PartitionImpl<N>* partition, DAGNode* node, const RemoteSend<N>& send,
                          int inp_name, const std::array<int, N>& nd_idx) {
    auto arr_it = partition->arrays.find(inp_name);
    if (arr_it == partition->arrays.end())
        return;
    cn_dispatch(arr_it->second->dtype, [&](auto tag) {
        send_remote_input<N, decltype(tag)>(partition, node, send, inp_name, nd_idx);
    });
}

template <int N>
static void dispatch_execute(DType dt, ArrayDAGGroup* group, DAGNode* node,
                             PartitionImpl<N>* partition, PendingComm<N>* comm) {
    cn_dispatch(dt, [&](auto tag) {
        group->execute_node_nd<N, decltype(tag)>(node, partition, comm);
    });
}

template <int N>
void ArrayDAGExecutorND<N>::delete_array(int name) {
    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    const int had_live_meta = dag_group->live_array_meta.count(name) ? 1 : 0;
    DBG_PRINT("[PE %d] Partition<%d> delete_array epoch=%d name=%d live_meta=%d\n",
              CkMyPe(), N, epoch, name, had_live_meta);
    partition->retire_array(name);
    dag_group->live_array_meta.erase(name);
}

/// True if chare idx of the NT-D partition owns part of array `name` in its
/// decomposition. Only owners compute, send broadcasts and answer gets.
template <int NT>
static bool chare_owns_array(ArrayDAGGroup* group, int name, const std::array<int, NT>& idx) {
    auto meta_it = group->array_meta.find(name);
    if (meta_it == group->array_meta.end() || meta_it->second.ndims != NT)
        return false;
    auto chare_region = meta_it->second.template decomp<NT>().chare_region_global(idx);
    for (int d = 0; d < NT; ++d)
        if (chare_region.size(d) <= 0)
            return false;
    return true;
}

/// True if chare idx of the NT-D partition owns part of root's output.
template <int NT>
static bool chare_owns_root_result(ArrayDAGGroup* group, ASTNode* root,
                                   const std::array<int, NT>& idx) {
    if (root == nullptr || root->ndims != NT)
        return false;

    int output_name = root->result_name;
    if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION && !root->operands.empty() &&
        root->operands[0] != nullptr)
        output_name = root->operands[0]->result_name;
    if (!chare_owns_array<NT>(group, output_name, idx))
        return false;

    if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION && root->region != nullptr &&
        !root->region->is_global) {
        auto decomp = group->array_meta.find(output_name)->second.template decomp<NT>();
        auto chare_region = decomp.chare_region_global(idx);
        auto* out_region = static_cast<ArrayRegion<NT>*>(root->region);
        auto out_region_global = decomp.to_global(*out_region);
        auto [overlap, has_overlap] = intersect(out_region_global, chare_region);
        return has_overlap && overlap.size() > 0;
    }

    return true;
}

/// True if chare idx of the nd-D partition owns part of any output of node.
static bool chare_owns_node_result(ArrayDAGGroup* group, DAGNode* node, int nd,
                                   const std::array<int, 3>& idx) {
    for (ASTNode* root : node->ast->roots) {
        switch (nd) {
        case 1:
            if (chare_owns_root_result<1>(group, root, {idx[0]}))
                return true;
            break;
        case 2:
            if (chare_owns_root_result<2>(group, root, {idx[0], idx[1]}))
                return true;
            break;
        case 3:
            if (chare_owns_root_result<3>(group, root, {idx[0], idx[1], idx[2]}))
                return true;
            break;
        }
    }
    return false;
}

template <int N>
void ArrayDAGExecutorND<N>::execute_dag_node(DAGNode* node) {
    // Check if this node is relevant to our partition dimensionality
    bool relevant = false;
    bool is_matmul_node = false;
    bool is_matmatmul_node = false;
    bool is_reduce_node = false;
    bool is_cross_set_region = false;
    bool is_diag_node = false;
    bool is_tile_node = false;
    bool is_reduction_node = false;
    bool is_creation_node = false;
    bool is_transpose_node = false;
    bool is_castscansort_node = false;
    bool is_linalg_node = false;
    bool is_signal_node = false;
    bool is_index_node = false;
    bool is_shape_node = false;
    // Broadcast operands: size-1 arrays from another partition used as scalars
    struct BroadcastInfo {
        int source_name;
        int source_ndims;
        int target_ndims;
    };
    std::vector<BroadcastInfo> broadcast_ops;
    int n_expected_broadcasts = 0; // broadcast values this chare receives
    bool sender_only = false;      // owns no output: only sends its input fragments
    auto* dag_group_meta = static_cast<ArrayDAGGroup*>(group);
    auto nd_idx = partition->nd_index();

    auto chare_owns_result = [&](ASTNode* root) -> bool {
        return chare_owns_root_result<N>(dag_group_meta, root, nd_idx);
    };
    std::array<int, 3> nd_idx3 = {};
    for (int d = 0; d < N; ++d)
        nd_idx3[d] = nd_idx[d];

    std::unordered_set<int> internal_results;
    for (ASTNode* root : node->ast->roots)
        internal_results.insert(root->result_name);

    for (ASTNode* root : node->ast->roots) {
        auto opc = static_cast<Opcode>(root->opcode);
        if (opc == Opcode::CREATE) {
            if (root->ndims == N)
                relevant = true;
            continue;
        }
        if (is_reduction_op(opc)) {
            // Input chares send partials; output chares wait for them.
            is_reduction_node = true;
            int input_ndims = root->operands[0]->ndims;
            if (N == input_ndims || N == root->ndims)
                relevant = true;
            continue;
        }
        if (is_shape_op(opc)) {
            // Source chares send, result chares receive.
            is_shape_node = true;
            if (N == root->operands[0]->ndims || N == root->ndims)
                relevant = true;
            continue;
        }
        if (opc == Opcode::LINALG) {
            // Operand chares send their blocks to the root; result chares wait.
            is_linalg_node = true;
            if (N == root->ndims)
                relevant = true;
            for (ASTNode* operand : root->operands)
                if (!operand->is_scalar && operand->ndims == N)
                    relevant = true;
            continue;
        }
        if (is_index_op(opc)) {
            // All chares of these partitions run the node (they may get messages).
            is_index_node = true;
            for (int nd : index_op_ndims(root))
                if (nd == N)
                    relevant = true;
            continue;
        }
        if (is_fill_creation(opc)) {
            // UPLOAD / RANDOM / IOTA: each chare fills its own tile.
            is_creation_node = true;
            if (root->ndims == N)
                relevant = true;
            continue;
        }
        if (opc == Opcode::REDUCE) {
            is_reduce_node = true;
            if (N == 1)
                relevant = true;
            continue;
        }
        if (opc == Opcode::MATMUL) {
            is_matmul_node = true;
            // MATMUL is relevant for 1D (vector send/result receive),
            // 2D (matrix compute), and 3D (dimension-dropped matvec)
            if (N == 1 || N == 2 || N == 3)
                relevant = true;
            continue;
        }
        if (opc == Opcode::MATMATMUL) {
            is_matmatmul_node = true;
            // MATMATMUL is relevant for 2D chares that hold A, B, or will hold C,
            // and 3D chares with dim-dropped operands
            if constexpr (N == 2) {
                // Check if this chare holds A or B
                for (ASTNode* operand : root->operands)
                    if (!operand->is_scalar && !operand->is_broadcast &&
                        partition->arrays.count(operand->result_name))
                        relevant = true;
                // Check if this chare will hold part of C
                auto* dag_group_tmp = static_cast<ArrayDAGGroup*>(group);
                auto c_meta = dag_group_tmp->array_meta.find(root->result_name);
                if (c_meta != dag_group_tmp->array_meta.end()) {
                    auto nd = partition->nd_index();
                    auto c_decomp = c_meta->second.template decomp<2>();
                    auto c_chare = c_decomp.chare_region_global(nd);
                    if (c_chare.stop[0] > c_chare.start[0] &&
                        c_chare.stop[1] > c_chare.start[1])
                        relevant = true;
                }
            } else if constexpr (N == 3) {
                relevant = true;
            }
            continue;
        }
        if (opc == Opcode::DIAG) {
            is_diag_node = true;
            // DIAG is relevant for 1D and 2D partitions
            if (N == 1 || N == 2)
                relevant = true;
            continue;
        }
        if (opc == Opcode::TILE) {
            is_tile_node = true;
            int input_ndims = root->operands[0]->ndims;
            int result_ndims = root->ndims;
            if (N == input_ndims || N == result_ndims)
                relevant = true;
            continue;
        }
        if (opc == Opcode::TRANSPOSE) {
            is_transpose_node = true;
            if (N == root->ndims)
                relevant = true;
            continue;
        }
        if (is_signal_op(opc)) {
            is_signal_node = true;
            if (N == root->ndims)
                relevant = true;
            continue;
        }
        if (is_castscansort(opc)) {
            // All chares of the partition take part; empty ones finish.
            is_castscansort_node = true;
            if (N == root->ndims)
                relevant = true;
            continue;
        }
        if (opc == Opcode::SET_REGION && root->operands.size() >= 2 &&
            !root->operands[1]->is_scalar && !root->operands[1]->is_broadcast) {
            int source_ndims = root->operands[1]->ndims;
            int target_ndims = root->ndims;
            if (source_ndims != target_ndims) {
                is_cross_set_region = true;
                // Relevant if we are the source or target partition
                if (N == source_ndims || N == target_ndims)
                    relevant = true;
                continue;
            }
        }
        // Detect broadcast operands (cross-partition or same-partition); results of
        // this node's own roots and SET_REGION targets are not inputs.
        for (int op_idx = 0; op_idx < (int)root->operands.size(); ++op_idx) {
            ASTNode* operand = root->operands[op_idx];
            bool is_input = (operand->opcode == static_cast<int>(Opcode::NOOP) ||
                             operand->opcode == static_cast<int>(Opcode::CREATE)) &&
                            !internal_results.count(operand->result_name) &&
                            !(opc == Opcode::SET_REGION && op_idx == 0);
            if (operand->is_broadcast && is_input) {
                auto* dag_group = static_cast<ArrayDAGGroup*>(group);
                auto meta_it = dag_group->array_meta.find(operand->result_name);
                if (meta_it != dag_group->array_meta.end()) {
                    int src_nd = meta_it->second.ndims;
                    int tgt_nd = root->ndims;
                    broadcast_ops.push_back({operand->result_name, src_nd, tgt_nd});
                    // Relevant for the chare holding the source and for chares owning a result
                    if (N == src_nd && chare_owns_array<N>(dag_group, operand->result_name, nd_idx))
                        relevant = true;
                    if (N == tgt_nd && chare_owns_node_result(dag_group, node, N, nd_idx3))
                        relevant = true;
                }
            }
        }
        // For other ops, check if any operand arrays exist on this partition
        for (ASTNode* operand : root->operands)
            if (!operand->is_scalar && !operand->is_broadcast &&
                partition->arrays.count(operand->result_name))
                relevant = true;
        if (partition->arrays.count(root->result_name))
            relevant = true;
        if (chare_owns_result(root))
            relevant = true;
    }
    if (!relevant) {
        DBG_PRINT("[PE %d] Partition<%d> chare %d: IRRELEVANT node %d\n",
                 CkMyPe(), N, partition->index[0], node->id);
        node_finished(node->id);
        return;
    }

    if (is_reduction_node) {
        execute_reduction_node(node);
        return;
    }

    if (is_linalg_node) {
        execute_linalg_node(node);
        return;
    }

    if (is_shape_node) {
        execute_shape_node(node);
        return;
    }

    if (is_creation_node) {
        execute_creation_node(node);
        return;
    }

    if (is_index_node) {
        execute_index_node(node);
        return;
    }

    // Check for REDUCE nodes — 1D dot product with cross-chare reduction
    if (is_reduce_node) {
        execute_reduce_node(node);
        return;
    }

    // Check for MATMUL nodes — these use a custom communication pattern
    if (is_matmul_node) {
        execute_matmul_node(node);
        return;
    }

    // Check for MATMATMUL nodes — SUMMA matrix-matrix multiply
    if (is_matmatmul_node) {
        execute_matmatmul_node(node);
        return;
    }

    // Check for cross-partition SET_REGION
    if (is_cross_set_region) {
        execute_cross_set_region_node(node);
        return;
    }

    // Check for DIAG nodes — cross-partition diagonal construction/extraction
    if (is_diag_node) {
        execute_diag_node(node);
        return;
    }

    // Check for TILE nodes — numpy.tile repetition with possible ndims change
    if (is_tile_node) {
        execute_tile_node(node);
        return;
    }

    // Check for TRANSPOSE nodes — point-to-point tile permutation
    if (is_transpose_node) {
        execute_transpose_node(node);
        return;
    }

    if (is_signal_node) {
        execute_signal_node(node);
        return;
    }

    // Check for SCAN/SORT/ARGSORT nodes (executor_castscansort.cpp)
    if (is_castscansort_node) {
        execute_castscansort_node(node);
        return;
    }

    // Handle cross-partition broadcast: send scalar values from source to target
    if (!broadcast_ops.empty()) {
        auto* dag_group = static_cast<ArrayDAGGroup*>(group);
        bool is_source_only = true;  // true if this partition only sends, doesn't compute

        // Compute broadcast input index for each broadcast op (shared logic).
        // Must match the receiver convention: n_memref_leaves + bcast_index.
        // First count all memref leaves, then find the broadcast index.
        //
        // The leaf walk mirrors collect_leaves_and_outputs() (execute_node.cpp)
        // and MLIRJitCompiler::collectLeaves(): operands produced by a root of
        // this fused node are internal (not inputs), and broadcast leaves are
        // de-duplicated by name among broadcast leaves only.  Counting
        // internal temps here shifted the index past the receiver's slot, so
        // the receiver never found the value and used 0 instead.
        auto compute_bcast_input_idx = [&](int source_name) -> int {
            std::unordered_set<int> internal;
            for (ASTNode* root : node->ast->roots)
                internal.insert(root->result_name);
            auto is_leaf_use = [&](ASTNode* root, int op_idx) {
                ASTNode* operand = root->operands[op_idx];
                if (!operand || operand->is_scalar || internal.count(operand->result_name))
                    return false;
                if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION && op_idx == 0)
                    return false;
                auto opc = static_cast<Opcode>(operand->opcode);
                return opc == Opcode::NOOP || opc == Opcode::CREATE;
            };

            int n_memrefs = 0;
            for (ASTNode* root : node->ast->roots)
                for (int i = 0; i < (int)root->operands.size(); ++i)
                    if (is_leaf_use(root, i) && !root->operands[i]->is_broadcast)
                        n_memrefs++;

            // Second pass: find the broadcast index for source_name
            std::unordered_set<int> seen;
            int bcast_idx = 0;
            for (ASTNode* root : node->ast->roots) {
                for (int i = 0; i < (int)root->operands.size(); ++i) {
                    ASTNode* operand = root->operands[i];
                    if (!is_leaf_use(root, i) || !operand->is_broadcast ||
                        !seen.insert(operand->result_name).second)
                        continue;
                    if (operand->result_name == source_name)
                        return n_memrefs + bcast_idx;
                    bcast_idx++;
                }
            }
            return n_memrefs;
        };

        // Helper: send a broadcast scalar to a target chare identified by ND index
        auto send_broadcast = [&](int tgt_nd, const std::array<int, 3>& tgt_idx,
                                  int bcast_input_idx, int source_name, char* val_ptr,
                                  int elem_sz) {
            std::vector<int> region_data(tgt_nd * 3, 0);
            for (int d = 0; d < tgt_nd; ++d) {
                region_data[d * 3 + 1] = 1;
                region_data[d * 3 + 2] = 1;
            }
#ifndef NDEBUG
            partition->comm_bytes_sent += elem_sz;
#endif
#ifdef USE_KOKKOS
            // receive_data takes a nocopydevice buffer; send_complete frees it.
            auto send_device = [&](auto& proxy, const auto& ci, auto nt) {
                constexpr int NT = decltype(nt)::value;
                char* dbuf = static_cast<char*>(comm_alloc(elem_sz));
                Kokkos::deep_copy(
                    Kokkos::View<char*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        dbuf, elem_sz),
                    Kokkos::View<char*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        val_ptr, elem_sz));
                device_pack_send<N, NT>(partition, proxy, ci, node->id, bcast_input_idx,
                                        source_name, region_data.data(), elem_sz, dbuf);
            };
            switch (tgt_nd) {
            case 1: {
                ChareIndex<1> ci;
                ci.idx[0] = tgt_idx[0];
                send_device(dag_group->partition_proxy_1, ci, std::integral_constant<int, 1>{});
                break;
            }
            case 2: {
                ChareIndex<2> ci;
                ci.idx[0] = tgt_idx[0];
                ci.idx[1] = tgt_idx[1];
                send_device(dag_group->partition_proxy_2, ci, std::integral_constant<int, 2>{});
                break;
            }
            case 3: {
                ChareIndex<3> ci;
                ci.idx[0] = tgt_idx[0];
                ci.idx[1] = tgt_idx[1];
                ci.idx[2] = tgt_idx[2];
                send_device(dag_group->partition_proxy_3, ci, std::integral_constant<int, 3>{});
                break;
            }
            }
#else
            char* send_buf = new char[elem_sz];
            memcpy(send_buf, val_ptr, elem_sz);
            switch (tgt_nd) {
            case 1: {
                ChareIndex<1> ci;
                ci.idx[0] = tgt_idx[0];
                proxy_at<1>(dag_group->partition_proxy_1, ci)
                    .receive_data(node->id, bcast_input_idx, source_name, tgt_nd,
                                  region_data.data(), elem_sz, send_buf);
                break;
            }
            case 2: {
                ChareIndex<2> ci;
                ci.idx[0] = tgt_idx[0];
                ci.idx[1] = tgt_idx[1];
                proxy_at<2>(dag_group->partition_proxy_2, ci)
                    .receive_data(node->id, bcast_input_idx, source_name, tgt_nd,
                                  region_data.data(), elem_sz, send_buf);
                break;
            }
            case 3: {
                ChareIndex<3> ci;
                ci.idx[0] = tgt_idx[0];
                ci.idx[1] = tgt_idx[1];
                ci.idx[2] = tgt_idx[2];
                proxy_at<3>(dag_group->partition_proxy_3, ci)
                    .receive_data(node->id, bcast_input_idx, source_name, tgt_nd,
                                  region_data.data(), elem_sz, send_buf);
                break;
            }
            }
            delete[] send_buf;
#endif
        };

        for (auto& bcast : broadcast_ops) {
            int bcast_input_idx = compute_bcast_input_idx(bcast.source_name);

            if (N == bcast.source_ndims) {
                // We are on the source partition — send if we own the array
                auto arr_it = partition->arrays.find(bcast.source_name);
                DBG_PRINT("[Chare %d]   bcast send check: name=%d found=%d local_size=%d\n",
                          partition->index[0], bcast.source_name,
                          (int)(arr_it != partition->arrays.end()),
                          (arr_it != partition->arrays.end()) ? arr_it->second->local_size() : -1);
                if (chare_owns_array<N>(dag_group, bcast.source_name, nd_idx) &&
                    arr_it != partition->arrays.end() &&
                    arr_it->second->local_size() > 0) {
                    int elem_sz = arr_it->second->elem_size();
#ifdef USE_KOKKOS
                    char host_scalar[16];
                    Kokkos::deep_copy(
                        Kokkos::View<char*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                            host_scalar, elem_sz),
                        Kokkos::View<char*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                            static_cast<char*>(arr_it->second->device_data_ptr()), elem_sz));
                    char* val_ptr = host_scalar;
#else
                    arr_it->second->copyToHost();
                    char* val_ptr = static_cast<char*>(arr_it->second->data_ptr());
#endif

                    if (bcast.source_ndims == bcast.target_ndims) {
                        auto pg_it = dag_group->partition_grid.find(N);
                        if (pg_it != dag_group->partition_grid.end()) {
                            int total_chares = 1;
                            for (int d = 0; d < N; ++d)
                                total_chares *= pg_it->second.grid[d];
                            for (int t = 0; t < total_chares; ++t) {
                                std::array<int, 3> tgt_idx = {};
                                int rem = t;
                                for (int d = N - 1; d >= 0; --d) {
                                    tgt_idx[d] = rem % pg_it->second.grid[d];
                                    rem /= pg_it->second.grid[d];
                                }
                                // Skip self
                                bool is_self = true;
                                for (int d = 0; d < N; ++d)
                                    if (tgt_idx[d] != nd_idx[d])
                                        is_self = false;
                                if (is_self || !chare_owns_node_result(dag_group, node, N, tgt_idx))
                                    continue;
                                send_broadcast(N, tgt_idx, bcast_input_idx, bcast.source_name,
                                               val_ptr, elem_sz);
                            }
                        }
                    } else {
                        int tgt_nd = bcast.target_ndims;
                        auto pg_it = dag_group->partition_grid.find(tgt_nd);
                        if (pg_it != dag_group->partition_grid.end()) {
                            int total_chares = 1;
                            for (int d = 0; d < tgt_nd; ++d)
                                total_chares *= pg_it->second.grid[d];
                            for (int t = 0; t < total_chares; ++t) {
                                std::array<int, 3> tgt_idx = {};
                                int rem = t;
                                for (int d = tgt_nd - 1; d >= 0; --d) {
                                    tgt_idx[d] = rem % pg_it->second.grid[d];
                                    rem /= pg_it->second.grid[d];
                                }
                                if (!chare_owns_node_result(dag_group, node, tgt_nd, tgt_idx))
                                    continue;
                                send_broadcast(tgt_nd, tgt_idx, bcast_input_idx,
                                               bcast.source_name, val_ptr, elem_sz);
                            }
                        }
                    }
                }
            }

            if (N == bcast.target_ndims && chare_owns_node_result(dag_group, node, N, nd_idx3))
                is_source_only = false;
        }

        if (is_source_only) {
            // No owned output: only sends broadcasts and, on the result partition, inputs.
            bool holds_input = false;
            for (ASTNode* root : node->ast->roots)
                for (ASTNode* operand : root->operands)
                    if (root->ndims == N && !operand->is_scalar && !operand->is_broadcast &&
                        partition->arrays.count(operand->result_name))
                        holds_input = true;
            if (!holds_input) {
                node_finished(node->id);
                return;
            }
            sender_only = true;
        }

        // Count expected broadcast messages for this chare
        // For same-partition: every chare but the source's owner expects 1 message per broadcast
        // For cross-partition: all target chares expect 1 message per broadcast
        for (auto& bcast : broadcast_ops) {
            if (N != bcast.target_ndims || sender_only)
                continue;
            if (bcast.source_ndims == bcast.target_ndims) {
                if (!chare_owns_array<N>(dag_group, bcast.source_name, nd_idx))
                    n_expected_broadcasts++;
            } else {
                // Cross-partition: we always expect a message
                n_expected_broadcasts++;
            }
        }
        DBG_PRINT("[Chare %d]   n_expected_broadcasts=%d is_source_only=%d\n",
                  partition->index[0], n_expected_broadcasts, (int)is_source_only);
        // Broadcast values arrive as extra messages of the regular path below.
    }

    // Wait for `expected` messages (remote fragments and broadcast values); some
    // may have arrived already (negative count in `pending`).
    auto wait_for_msgs = [&](int expected, std::vector<ArrayRegion<N>> my_inputs) {
        auto pre_it = pending.find(node->id);
        int pre_arrived = 0;
        std::unordered_map<int, std::vector<RemoteBuffer<N>>> pre_buffers;
        if (pre_it != pending.end()) {
            pre_arrived = -(pre_it->second.expected_msgs);
            pre_buffers = std::move(pre_it->second.remote_buffers);
        }

        int remaining = expected - pre_arrived;
        pending[node->id] = {node, remaining, std::move(my_inputs), std::move(pre_buffers)};

        if (remaining <= 0) {
            on_comm_done(node->id);
        } else {
            DBG_PRINT("[Chare %d]   -> waiting for %d remote messages\n", partition->index[0],
                      remaining);
        }
    };

    ArrayRegion<N> r_out;
    std::vector<ArrayRegion<N>> input_regions;
    std::vector<int> input_source_names;
    std::array<int, N> global_shape;

    DType dt = determine_dtype<N>(node, partition->arrays);

    if (!extract_regions_nd<N>(node, partition->arrays, r_out, input_regions, input_source_names,
                               global_shape, &dag_group_meta->array_meta) ||
        input_regions.empty()) {
        if (sender_only) {
            node_finished(node->id);
            return;
        }
        if (n_expected_broadcasts > 0) {
            wait_for_msgs(n_expected_broadcasts, {});
            return;
        }
        dispatch_execute<N>(dt, static_cast<ArrayDAGGroup*>(group), node, partition, nullptr);
#ifdef USE_NVIDIA
        {
            auto* p = new ComputeDoneParam<N>{partition, node->id, false};
            CkCallback hcb(compute_done_cb<N>, p);
            hapiAddCallback(partition->compute_stream_raw, &hcb);
        }
#else
        CT_KOKKOS_FENCE();
        node_finished(node->id);
#endif
        return;
    }

    // Build per-input decomps and output decomp from array metadata
    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    std::vector<ArrayDecomp<N>> input_decomps;
    for (int i = 0; i < (int)input_source_names.size(); ++i) {
        auto arr_it = partition->arrays.find(input_source_names[i]);
        if (arr_it != partition->arrays.end()) {
            input_decomps.push_back(arr_it->second->decomp);
        } else {
            auto meta_it = dag_group->array_meta.find(input_source_names[i]);
            if (meta_it != dag_group->array_meta.end())
                input_decomps.push_back(meta_it->second.template decomp<N>());
            else
                input_decomps.push_back(ArrayDecomp<N>::default_decomp(global_shape,
                    array_tile(dag_group->array_meta, input_source_names[i], N)));

        }
    }

    // Decomp of the array r_out refers to (see fused_output_decomp_name). Fused
    // ASTs can keep temp roots ahead of the real result, with other offsets.
    ArrayDecomp<N> output_decomp;
    {
        int result_name = fused_output_decomp_name(node);
        auto arr_it = partition->arrays.find(result_name);
        if (arr_it != partition->arrays.end()) {
            output_decomp = arr_it->second->decomp;
        } else {
            auto meta_it = dag_group->array_meta.find(result_name);
            if (meta_it != dag_group->array_meta.end())
                output_decomp = meta_it->second.template decomp<N>();
            else
                output_decomp = ArrayDecomp<N>::default_decomp(global_shape,
                    array_tile(dag_group->array_meta, result_name, N));

        }
    }

    // Translate regions from local space to global space
    ArrayRegion<N> r_out_global = output_decomp.to_global(r_out);
    std::vector<ArrayRegion<N>> input_regions_global;
    for (int i = 0; i < (int)input_regions.size(); ++i)
        input_regions_global.push_back(input_decomps[i].to_global(input_regions[i]));

    ArrayRegion<N> r_chare_out = output_decomp.chare_region_global(nd_idx);

    // Send remote inputs (type-dispatched)
    auto sends = send_remote_inputs<N>(r_out_global, input_regions_global, nd_idx, output_decomp,
                                       input_decomps);
    for (auto& send : sends) {
        int inp_name = input_source_names[send.input_index];
        dispatch_send<N>(partition, node, send, inp_name, nd_idx);
    }
    if (sender_only) {
        node_finished(node->id);
        return;
    }

    // Determine expected messages
    auto li = local_inputs<N>(r_out_global, input_regions_global, nd_idx, output_decomp,
                              input_decomps);

    if (li.expected_msgs + n_expected_broadcasts == 0) {
        if (li.my_inputs.empty()) {
            auto [r_myout, has_out] = intersect(r_out_global, r_chare_out);
            if (has_out && r_myout.size() > 0) {
                PendingComm<N> comm_local = {node, 0, {}, {}};
                dispatch_execute<N>(dt, static_cast<ArrayDAGGroup*>(group), node, partition,
                                    &comm_local);
            }
#ifdef USE_NVIDIA
            {
                auto* p = new ComputeDoneParam<N>{partition, node->id, false};
                CkCallback hcb(compute_done_cb<N>, p);
                hapiAddCallback(partition->compute_stream_raw, &hcb);
            }
#else
            CT_KOKKOS_FENCE();
            node_finished(node->id);
#endif
        } else {
            PendingComm<N> comm_local = {node, 0, std::move(li.my_inputs), {}};
            dispatch_execute<N>(dt, static_cast<ArrayDAGGroup*>(group), node, partition,
                                &comm_local);
#ifdef USE_NVIDIA
            {
                auto* p = new ComputeDoneParam<N>{partition, node->id, false};
                CkCallback hcb(compute_done_cb<N>, p);
                hapiAddCallback(partition->compute_stream_raw, &hcb);
            }
#else
            CT_KOKKOS_FENCE();
            node_finished(node->id);
#endif
        }
    } else {
        wait_for_msgs(li.expected_msgs + n_expected_broadcasts, std::move(li.my_inputs));
    }
}

template <int N>
void ArrayDAGExecutorND<N>::on_comm_done(int node_id) {
    auto it = pending.find(node_id);
    if (it == pending.end())
        return;

    PendingComm<N>& comm = it->second;
    DAGNode* node = comm.node;
    bool castscansort = false, signal = false;
    for (ASTNode* root : node->ast->roots) {
        castscansort |= is_castscansort(static_cast<Opcode>(root->opcode));
        signal |= is_signal_op(static_cast<Opcode>(root->opcode));
    }
    if (castscansort) {
        finish_castscansort_node(comm);
    } else if (signal) {
        finish_signal_node(comm);
    } else {
        DType dt = determine_dtype<N>(node, partition->arrays);
        dispatch_execute<N>(dt, static_cast<ArrayDAGGroup*>(group), node, partition, &comm);
    }

#ifdef USE_NVIDIA
    // Defer cleanup + node_finished until GPU work completes on compute_stream
    {
        auto* p = new ComputeDoneParam<N>{partition, node_id, true};
        CkCallback hcb(compute_done_cb<N>, p);
        hapiAddCallback(partition->compute_stream_raw, &hcb);
    }
#else
    CT_KOKKOS_FENCE();
    comm.clear_remote_buffers();
    pending.erase(it);
    // on_comm_done is high-volume; node_finished already prints
    node_finished(node_id);
#endif
}

template void ArrayDAGExecutorND<1>::delete_array(int);
template void ArrayDAGExecutorND<2>::delete_array(int);
template void ArrayDAGExecutorND<3>::delete_array(int);

template void ArrayDAGExecutorND<1>::execute_dag_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_dag_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_dag_node(DAGNode*);

template void ArrayDAGExecutorND<1>::on_comm_done(int);
template void ArrayDAGExecutorND<2>::on_comm_done(int);
template void ArrayDAGExecutorND<3>::on_comm_done(int);
