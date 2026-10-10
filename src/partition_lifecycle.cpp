#include "backend_internal.hpp"
#include "dispatch.hpp"

#ifdef USE_NVIDIA
#define CT_CUDA_CHECK(call)                                                                        \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess)                                                                   \
            CkAbort("CUDA error '%s' at %s:%d", cudaGetErrorString(err_), __FILE__, __LINE__);     \
    } while (0)
#endif

template <int N>
void PartitionImpl<N>::init(typename PartitionTraits<N>::ProxyType proxy, std::array<int, N> idx,
                            CProxy_ArrayDAGGroup dag_proxy_) {
    thisProxy = proxy;
    index = idx;
    dag_proxy = dag_proxy_;
#ifdef USE_NVIDIA
    CT_CUDA_CHECK(cudaStreamCreate(&compute_stream_raw));
    CT_CUDA_CHECK(cudaStreamCreate(&comm_stream_raw));
    compute_exec = Kokkos::Cuda(compute_stream_raw);
    comm_exec = Kokkos::Cuda(comm_stream_raw);
#endif
    auto* dag_group = static_cast<ArrayDAGGroup*>(dag_proxy_.ckLocalBranch());
    executor = new ArrayDAGExecutorND<N>(dag_group, this);
    // Skip to the epoch where this partition was first created.
    // start_epoch - 1 so that next_epoch() advances to start_epoch.
    int se = dag_group->partition_grid[N].start_epoch;
    if (se > 0)
        executor->epoch = se - 1;
    ChareIndex<N> ci;
    for (int d = 0; d < N; ++d)
        ci.idx[d] = index[d];
    proxy_at<N>(thisProxy, ci).run();
}

template <int N>
PartitionImpl<N>::~PartitionImpl() {
    delete executor;
    for (auto& [name, arr] : arrays)
        delete arr;
    for (auto& [key, bucket] : free_arrays)
        for (auto* arr : bucket)
            delete arr;
#ifdef USE_KOKKOS
    for (auto& [id, ptr] : pending_sends)
        Kokkos::kokkos_free<DeviceSpace>(ptr);
    for (auto& [key, bufs] : posted_recv_bufs)
        for (char* p : bufs)
            Kokkos::kokkos_free<DeviceSpace>(p);
#endif
#ifdef USE_NVIDIA
    cudaStreamDestroy(compute_stream_raw);
    cudaStreamDestroy(comm_stream_raw);
#endif
}

template <int N>
int PartitionImpl<N>::create(ArrayRegion<N>* region, int name, DType dtype,
                             const ArrayDecomp<N>& decomp) {

    auto nd_idx = this->nd_index();
    auto* dag_group = static_cast<ArrayDAGGroup*>(dag_proxy.ckLocalBranch());

    std::array<int, N> global_shape{};
    std::array<int, N> local_sizes{};
    std::array<int, N> zeros{};
    std::array<int, N> steps{};
    zeros.fill(0);

    for (int d = 0; d < N; ++d) {
        global_shape[d] = region->size(d);
        steps[d] = region->step[d];
    }

    ArrayDAGGroup::ArrayMetadata live_meta = {};
    live_meta.ndims = N;
    live_meta.global_shape = {0, 0, 0};
    live_meta.offset = {0, 0, 0};
    live_meta.tile = decomp.tile;
    for (int d = 0; d < N; ++d) {
        live_meta.global_shape[d] = global_shape[d];
        live_meta.offset[d] = decomp.offset[d];
    }
    dag_group->live_array_meta[name] = live_meta;

    // Re-create replaces any old buffer, even if this chare no longer owns a tile.
    auto old_it = arrays.find(name);
    if (old_it != arrays.end()) {
        delete old_it->second;
        arrays.erase(old_it);
    }

    auto local_region = decomp.chare_region_local(nd_idx);
    for (int d = 0; d < N; ++d) {
        local_sizes[d] = local_region.size(d);
        if (local_sizes[d] <= 0)
            return name;
    }

    ArrayRegion<N> chare_region(zeros, local_sizes, steps);

    arrays[name] = allocate_or_reuse(chare_region, global_shape, name, dtype, decomp);
    return name;
}

template <int N>
void PartitionImpl<N>::run() {
    // If a DAG is currently executing (not all nodes done), ignore this
    // stale run() from the NONE polling loop.  The DAG completion callback
    // will re-invoke run() when the DAG finishes.
    if (executor->dag != nullptr &&
        executor->dag->num_nodes_done < executor->dag->num_nodes) {
        return;
    }

    // A chare inserted by PE 0 can be constructed on another PE before that
    // PE's receive_dag() has recorded the partition's start epoch, so init()
    // left it at epoch -1 and it would wait forever for DAGs of epochs that
    // predate this partition (flaky hang at +p2, e.g. diag.py).  The epoch
    // is known by the time receive_dag() wakes the chares with run().
    if (executor->dag == nullptr) {
        auto* dag_group = static_cast<ArrayDAGGroup*>(dag_proxy.ckLocalBranch());
        int se = dag_group->partition_grid[N].start_epoch;
        if (se > 0 && executor->epoch < se - 1)
            executor->epoch = se - 1;
    }

#ifndef NDEBUG
    // Print cumulative communication volume at the end of each DAG epoch
    if (executor->dag != nullptr &&
        executor->dag->num_nodes_done >= executor->dag->num_nodes) {
        DBG_PRINT("[PE %d] Partition<%d> chare (%d",
                  CkMyPe(), N, index[0]);
        for (int d = 1; d < N; ++d)
            CkPrintf(",%d", index[d]);
        CkPrintf("): epoch %d done, cumulative comm_bytes_sent=%lld\n",
                 executor->epoch, (long long)comm_bytes_sent);
    }
#endif

    ChareIndex<N> ci;
    for (int d = 0; d < N; ++d)
        ci.idx[d] = index[d];

    EpochType type = executor->next_epoch();
    DBG_PRINT("[PE %d] Partition<%d> chare %d: run() epoch=%d type=%s\n",
              CkMyPe(), N, index[0], executor->epoch,
              type == EpochType::DAG ? "DAG" : type == EpochType::GET ? "GET" : "NONE");
    switch (type) {
    case EpochType::DAG:
        executor->execute_dag(
            CkCallback(PartitionTraits<N>::CkIndexType::run(), proxy_at<N>(thisProxy, ci)));
        break;
    case EpochType::GET:
        process_get(executor->epoch);
        proxy_at<N>(thisProxy, ci).run();
        break;
    case EpochType::NONE:
        break;
    }
}

template void PartitionImpl<1>::init(CProxy_Partition1D, std::array<int, 1>, CProxy_ArrayDAGGroup);
template void PartitionImpl<2>::init(CProxy_Partition2D, std::array<int, 2>, CProxy_ArrayDAGGroup);
template void PartitionImpl<3>::init(CProxy_Partition3D, std::array<int, 3>, CProxy_ArrayDAGGroup);

template PartitionImpl<1>::~PartitionImpl();
template PartitionImpl<2>::~PartitionImpl();
template PartitionImpl<3>::~PartitionImpl();

template int PartitionImpl<1>::create(ArrayRegion<1>*, int, DType, const ArrayDecomp<1>&);
template int PartitionImpl<2>::create(ArrayRegion<2>*, int, DType, const ArrayDecomp<2>&);
template int PartitionImpl<3>::create(ArrayRegion<3>*, int, DType, const ArrayDecomp<3>&);

template void PartitionImpl<1>::run();
template void PartitionImpl<2>::run();
template void PartitionImpl<3>::run();
