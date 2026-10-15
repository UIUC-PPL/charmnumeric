#include "backend_internal.hpp"

#ifdef USE_KOKKOS
#include <KokkosBlas1_dot.hpp>
#endif

#ifdef USE_KOKKOS
/// sum(x * y); KokkosBlas::dot conjugates a complex x, NumPy's dot does not.
template <typename T>
static T device_dot_nc(const T* x, const T* y, int n) {
    T sum = T(0);
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i, T& acc) { acc += x[i] * y[i]; }, sum);
    return sum;
}
#endif

template <int N>
void ArrayDAGExecutorND<N>::execute_reduce_node(DAGNode* node) {
    if constexpr (N != 1) {
        CkAbort("REDUCE only supported for 1D partitions (N=%d)", N);
        return;
    }

    if constexpr (N == 1) {
        ASTNode* reduce_root = nullptr;
        for (ASTNode* root : node->ast->roots) {
            if (static_cast<Opcode>(root->opcode) == Opcode::REDUCE) {
                reduce_root = root;
                break;
            }
        }
        int lhs_name = reduce_root->operands[0]->result_name;
        int rhs_name = reduce_root->operands[1]->result_name;
        int result_name = reduce_root->result_name;

        DType dt = determine_dtype<1>(node, partition->arrays);

        DBG_PRINT("[1D Chare %d] execute_reduce_node: lhs=%d rhs=%d result=%d\n",
                  partition->index[0], lhs_name, rhs_name, result_name);

        // Compute local dot product (0 if this chare has no data for either operand)
        auto lhs_it = partition->arrays.find(lhs_name);
        auto rhs_it = partition->arrays.find(rhs_name);

        // Build ReduceContrib with metadata + local partial value
        ReduceContrib contrib = {};
        contrib.node_id = node->id;
        contrib.result_name = result_name;
        contrib.dtype_int = static_cast<int>(dt);

        if (lhs_it != partition->arrays.end() && rhs_it != partition->arrays.end()) {
            int local_size =
                std::min(lhs_it->second->local_size(), rhs_it->second->local_size());

#ifdef USE_KOKKOS
            auto kokkos_dot = [&](auto dummy) {
                using VT = decltype(dummy);
                Kokkos::View<VT*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    d_lhs(static_cast<VT*>(lhs_it->second->device_data_ptr()), local_size);
                Kokkos::View<VT*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    d_rhs(static_cast<VT*>(rhs_it->second->device_data_ptr()), local_size);
                VT val;
                if constexpr (cn_is_complex_v<VT>)
                    val = device_dot_nc<VT>(d_lhs.data(), d_rhs.data(), local_size);
                else
                    val = KokkosBlas::dot(d_lhs, d_rhs);
                memcpy(contrib.value, &val, sizeof(VT));
            };
            cn_dispatch_blas(dt, kokkos_dot);
#else
            lhs_it->second->copyToHost();
            rhs_it->second->copyToHost();

            cn_dispatch_blas(dt, [&](auto tag) {
                using T = decltype(tag);
                T val = eigen_dot(static_cast<T*>(lhs_it->second->data_ptr()),
                                  static_cast<T*>(rhs_it->second->data_ptr()), local_size);
                memcpy(contrib.value, &val, sizeof(T));
            });
#endif

            DBG_PRINT("[1D Chare %d]   local dot computed (local_size=%d)\n",
                      partition->index[0], local_size);
        }
        // else: contrib.value is all zeros — contributes 0 to the reduction

        // Partials go to chare 0 (not a Charm++ reduction: chares added by a later
        // DAG restart their reduction numbering, which mismatches the older ones).
        auto* dag_group = static_cast<ArrayDAGGroup*>(group);
        ArrayDecomp<1> in_decomp = dag_group->array_meta[lhs_name].decomp<1>();
        int first = in_decomp.offset[0] / in_decomp.tile;
        int idx = partition->nd_index()[0];
        bool holder = idx >= first && idx < first + in_decomp.num_chares(0);
        if (idx == 0) {
            DotState& st = dot_state[node->id];
            st.started = true;
            st.expected = in_decomp.num_chares(0);
            if (holder)
                on_dot_partial(node->id, reinterpret_cast<char*>(&contrib), sizeof(contrib));
            else
                finish_dot(node->id);
            return; // finishes when every partial is in
        }
        if (holder) {
            ChareIndex<1> target_ci;
            target_ci.idx[0] = 0;
            proxy_at<1>(dag_group->partition_proxy_1, target_ci)
                .reduce_result(node->id, sizeof(contrib), reinterpret_cast<char*>(&contrib));
        }
        node_finished(node->id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::on_dot_partial(int node_id, char* data, int64_t size) {
    if constexpr (N == 1) {
        DotState& st = dot_state[node_id];
        ReduceContrib in;
        memcpy(&in, data, sizeof(in));
        if (st.got++ == 0) {
            st.acc.assign(data, data + size);
        } else {
            ReduceContrib acc;
            memcpy(&acc, st.acc.data(), sizeof(acc));
            cn_dispatch_blas(static_cast<DType>(acc.dtype_int), [&](auto tag) {
                using T = decltype(tag);
                T a, b;
                memcpy(&a, acc.value, sizeof(T));
                memcpy(&b, in.value, sizeof(T));
                a += b;
                memcpy(acc.value, &a, sizeof(T));
            });
            memcpy(st.acc.data(), &acc, sizeof(acc));
        }
        if (st.started)
            finish_dot(node_id);
    }
}

/// Chare 0: store the sum once all partials arrived.
template <int N>
void ArrayDAGExecutorND<N>::finish_dot(int node_id) {
    if constexpr (N == 1) {
        DotState& st = dot_state[node_id];
        if (st.got < st.expected)
            return;
        ReduceContrib result;
        memcpy(&result, st.acc.data(), sizeof(result));
        int result_name = result.result_name;
        DType dt = static_cast<DType>(result.dtype_int);
        auto& arrays = partition->arrays;

        // Create the scalar result array (size 1) on chare 0
        if (arrays.find(result_name) == arrays.end()) {
            auto* dag_group = static_cast<ArrayDAGGroup*>(group);
            ArrayDecomp<1> out_decomp = dag_group->array_meta[result_name].decomp<1>();
            std::array<int, 1> out_start = {0};
            std::array<int, 1> out_stop = {1};
            std::array<int, 1> out_step = {1};
            std::array<int, 1> out_gs = {1};
            ArrayRegion<1> out_region(out_start, out_stop, out_step);
            arrays[result_name] =
                partition->allocate_or_reuse(out_region, out_gs, result_name, dt, out_decomp);
        }

        int elem_size = dtype_size(dt);
#ifdef USE_KOKKOS
        Kokkos::deep_copy(
            Kokkos::View<char*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                static_cast<char*>(arrays[result_name]->device_data_ptr()), elem_size),
            Kokkos::View<char*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                result.value, elem_size));
#else
        memcpy(arrays[result_name]->data_ptr(), result.value, elem_size);
#endif
        dot_state.erase(node_id);
        node_finished(node_id);
    }
}

template void ArrayDAGExecutorND<1>::execute_reduce_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_reduce_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_reduce_node(DAGNode*);

template void ArrayDAGExecutorND<1>::on_dot_partial(int, char*, int64_t);
template void ArrayDAGExecutorND<2>::on_dot_partial(int, char*, int64_t);
template void ArrayDAGExecutorND<3>::on_dot_partial(int, char*, int64_t);
template void ArrayDAGExecutorND<1>::finish_dot(int);
template void ArrayDAGExecutorND<2>::finish_dot(int);
template void ArrayDAGExecutorND<3>::finish_dot(int);
