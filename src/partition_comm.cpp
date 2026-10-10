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
    // Post entry method: allocate device buffer for incoming GPU data
    // receive_data later gets a null data pointer, so the buffer is keyed by message identity.
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
    // Payload is already in the buffer from the post entry.
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
void PartitionImpl<N>::reduce_result(int node_id, int64_t size, char* data) {
    if constexpr (N == 1)
        executor->on_dot_partial(node_id, data, size);
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

template void PartitionImpl<1>::reduce_result(int, int64_t, char*);
template void PartitionImpl<2>::reduce_result(int, int64_t, char*);
template void PartitionImpl<3>::reduce_result(int, int64_t, char*);
