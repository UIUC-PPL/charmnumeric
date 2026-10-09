#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <cstring>

#ifdef USE_KOKKOS
static std::vector<int> recv_key(int node_id, int input_index, int name, int ndims,
                                 const int* region_data) {
    std::vector<int> key = {node_id, input_index, name, ndims};
    key.insert(key.end(), region_data, region_data + ndims * 3);
    return key;
}
#endif

#ifdef USE_KOKKOS
template <int N>
void PartitionImpl<N>::receive_data(int& node_id, int& input_index, int& name, int& ndims,
                                    int*& region_data, int64_t& size, char*& data,
                                    CkDeviceBufferPost* devicePost) {
    // Post entry method: allocate device buffer for incoming GPU data. Charm++
    // invokes the regular entry with a null data pointer once the transfer is
    // done, so remember the buffer under the message's identity.
    data = static_cast<char*>(Kokkos::kokkos_malloc<DeviceSpace>(size));
    posted_recv_bufs[recv_key(node_id, input_index, name, ndims, region_data)].push_back(data);
}

template <int N>
void PartitionImpl<N>::send_complete(int send_id) {
    auto it = pending_sends.find(send_id);
    if (it != pending_sends.end()) {
        Kokkos::kokkos_free<DeviceSpace>(it->second);
        pending_sends.erase(it);
    }
}
#endif

template <int N>
void PartitionImpl<N>::receive_data(int node_id, int input_index, int name, int ndims,
                                    int* region_data, int64_t size, char* data) {
    PendingComm<N>& comm = executor->pending[node_id];

#ifdef USE_KOKKOS
    // The payload landed in the device buffer allocated by the post entry above.
    char* buf = data;
    if (buf == nullptr) {
        auto key = recv_key(node_id, input_index, name, ndims, region_data);
        auto it = posted_recv_bufs.find(key);
        if (it == posted_recv_bufs.end() || it->second.empty())
            CkAbort("receive_data: no posted device buffer for node=%d input=%d name=%d",
                    node_id, input_index, name);
        buf = it->second.front();
        it->second.pop_front();
        if (it->second.empty())
            posted_recv_bufs.erase(it);
    }
#else
    char* buf = new char[size];
    memcpy(buf, data, size);
#endif

    // Unpack region from region_data: [start0, stop0, step0, start1, stop1, step1, ...]
    std::array<int, N> start, stop, step;
    for (int d = 0; d < N; ++d) {
        start[d] = region_data[d * 3 + 0];
        stop[d] = region_data[d * 3 + 1];
        step[d] = region_data[d * 3 + 2];
    }
    ArrayRegion<N> region(start, stop, step);

    comm.remote_buffers[input_index].push_back({buf, region, size});

    if (comm.incremental) {
        // Incremental mode: compute with this panel immediately
        executor->on_matmatmul_receive(node_id, input_index);
    }

    if (comm.expected_msgs > 0) {
        comm.expected_msgs--;
        if (comm.expected_msgs == 0)
            executor->on_comm_done(node_id);
    } else {
        comm.expected_msgs--;
    }
}

template <int N>
void PartitionImpl<N>::comm_done(int node_id) {
    executor->on_comm_done(node_id);
}

template <int N>
void PartitionImpl<N>::reduce_result(CkReductionMsg* msg) {
    if constexpr (N == 1) {
        // Extract metadata and summed value from the custom reducer's output
        ReduceContrib result;
        memcpy(&result, msg->getData(), sizeof(ReduceContrib));
        int node_id = result.node_id;
        int result_name = result.result_name;
        DType dt = static_cast<DType>(result.dtype_int);

        // Create the scalar result array (size 1) on chare 0
        if (arrays.find(result_name) == arrays.end()) {
            auto* dag_group = static_cast<ArrayDAGGroup*>(dag_proxy.ckLocalBranch());
            ArrayDecomp<1> out_decomp = dag_group->array_meta[result_name].decomp<1>();
            std::array<int, 1> out_start = {0};
            std::array<int, 1> out_stop = {1};
            std::array<int, 1> out_step = {1};
            std::array<int, 1> out_gs = {1};
            ArrayRegion<1> out_region(out_start, out_stop, out_step);
            arrays[result_name] =
                allocate_or_reuse(out_region, out_gs, result_name, dt, out_decomp);
        }

        // Store the reduced value
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

        DBG_PRINT("[1D Chare %d] reduce_result: node=%d result_name=%d stored\n",
                  index[0], node_id, result_name);

        delete msg;
        executor->node_finished(node_id);
    } else {
        delete msg;
    }
}

#ifdef USE_KOKKOS
template void PartitionImpl<1>::receive_data(int&, int&, int&, int&, int*&, int64_t&, char*&,
                                             CkDeviceBufferPost*);
template void PartitionImpl<2>::receive_data(int&, int&, int&, int&, int*&, int64_t&, char*&,
                                             CkDeviceBufferPost*);
template void PartitionImpl<3>::receive_data(int&, int&, int&, int&, int*&, int64_t&, char*&,
                                             CkDeviceBufferPost*);

template void PartitionImpl<1>::send_complete(int);
template void PartitionImpl<2>::send_complete(int);
template void PartitionImpl<3>::send_complete(int);
#endif

template void PartitionImpl<1>::receive_data(int, int, int, int, int*, int64_t, char*);
template void PartitionImpl<2>::receive_data(int, int, int, int, int*, int64_t, char*);
template void PartitionImpl<3>::receive_data(int, int, int, int, int*, int64_t, char*);

template void PartitionImpl<1>::comm_done(int);
template void PartitionImpl<2>::comm_done(int);
template void PartitionImpl<3>::comm_done(int);

template void PartitionImpl<1>::reduce_result(CkReductionMsg*);
template void PartitionImpl<2>::reduce_result(CkReductionMsg*);
template void PartitionImpl<3>::reduce_result(CkReductionMsg*);
