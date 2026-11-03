#include "backend_internal.hpp"

#include <malloc.h>

namespace {

struct MemTally {
    int64_t live = 0, live_dev = 0, live_host = 0;
    int64_t freed = 0, free_dev = 0, free_host = 0, free_keys = 0;
    int64_t states = 0, sends = 0;
};

template <int N>
void mem_tally(MemTally& t) {
    for (PartitionImpl<N>* p : local_partitions<N>()) {
        for (auto& [name, arr] : p->arrays) {
            ++t.live;
            t.live_dev += arr->local_size() * arr->elem_size();
            t.live_host += arr->host_bytes();
        }
        t.free_keys += p->free_arrays.size();
        for (auto& [key, bucket] : p->free_arrays)
            for (auto* arr : bucket) {
                ++t.freed;
                t.free_dev += arr->local_size() * arr->elem_size();
                t.free_host += arr->host_bytes();
            }
        auto* e = p->executor;
        t.states += e->pending.size() + e->red_state.size() + e->dot_state.size() +
                    e->la_state.size() + e->ix_state.size() + e->shape_state.size() +
                    e->upload_wait.size();
#ifdef USE_KOKKOS
        t.sends += p->pending_sends.size() + p->posted_recv_bufs.size();
#endif
    }
}

} // namespace

bool mem_stats_on() {
    static const bool on = [] {
        const char* e = getenv("CNUM_MEM_STATS");
        return e != nullptr && *e != '\0' && *e != '0';
    }();
    return on;
}

void mem_stats_report(ArrayDAGGroup* g, int epoch) {
    MemTally t;
    mem_tally<1>(t);
    mem_tally<2>(t);
    mem_tally<3>(t);
    int64_t dags = g->stored_dags();
    struct mallinfo2 mi = mallinfo2();
    const double MB = 1 << 20;
    double dev_used = 0, pool = 0, large = 0, pinned = 0;
    int64_t ipc_slabs = 0, ipc_live = 0;
#ifdef USE_KOKKOS
    {
        auto& cp = comm_pool();
        std::lock_guard<std::mutex> lock(cp.mu);
        pool = cp.cached / MB;
        large = cp.large_cached / MB;
    }
#endif
#ifdef USE_NVIDIA
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    dev_used = (tot - fr) / MB;
    {
        auto& pp = pinned_pool();
        std::lock_guard<std::mutex> lock(pp.mu);
        pinned = pp.cached / MB;
    }
    {
        auto& a = gpu_ipc_arena();
        std::lock_guard<std::mutex> lock(a.mu);
        ipc_slabs = a.slabs.size();
        for (auto& [p, sc] : a.live)
            ipc_live += int64_t(1) << sc.second;
    }
#endif
    CkPrintf("[mem] pe %d epoch %d dev_used %.0f | live %lld %.0f/%.0f MB | free %lld in %lld "
             "keys %.0f/%.0f MB | comm_pool %.0f+%.0f pinned %.0f MB | ipc %lld slabs %.0f MB "
             "live | states %lld sends %lld | dags %lld kernels %zu | heap %.0f MB\n",
             CkMyPe(), epoch, dev_used, (long long)t.live, t.live_dev / MB, t.live_host / MB,
             (long long)t.freed, (long long)t.free_keys, t.free_dev / MB, t.free_host / MB, pool,
             large, pinned, (long long)ipc_slabs, ipc_live / MB, (long long)t.states,
             (long long)t.sends, (long long)dags, g->compile_cache.size(),
             (mi.uordblks + mi.hblkhd) / MB);
}

template <int N>
void PartitionImpl<N>::process_get(int epoch) {

    auto nd_idx = this->nd_index();

    DBG_PRINT("Partition<%d> processing get for epoch %d\n", N, epoch);
    auto* req = executor->group->get_get_request(N, epoch);
    if (!req) {
        CkAbort("No get request found for epoch %d", epoch);
        return;
    }
    int name = req->name;
    if (name == kBarrierName) {
        // chares inserted after the barrier replay it: only those in its grid arrive
        auto* group = static_cast<ArrayDAGGroup*>(executor->group);
        const auto& grid = group->barrier_grids[epoch][N];
        for (int d = 0; d < N; ++d)
            if (nd_idx[d] >= grid.grid[d])
                return;
#ifdef USE_KOKKOS
        // nodes finish once their GPU work is queued
        Kokkos::fence("ct_barrier");
#endif
        dag_proxy[0].barrier_arrive(epoch);
        return;
    }
    if (arrays.find(name) == arrays.end()) {
        // This partition doesn't own any portion of the array (e.g. the array
        // is smaller than the chare grid).  Nothing to contribute.
        DBG_PRINT("Partition<%d>: array %d not local, skipping get for epoch %d\n", N, name,
                  epoch);
        return;
    }

    // Position in the gathered (array-local, row-major) result: the chare
    // region is decomp-global, so remove the decomposition offset (nonzero
    // when the solver shifted the array to align with a sliced use, e.g.
    // X in lstm_forward); otherwise fragments land past the buffer end.
    // Only the chares owning part of the array in its decomposition contribute.
    auto global_region = arrays[name]->decomp.chare_region_local(nd_idx);
    for (int d = 0; d < N; ++d)
        if (global_region.size(d) <= 0)
            return;

    arrays[name]->copyToHost();

    {
        CTArrayBase<N>* arr = arrays[name];
        int esz = arr->elem_size();

        std::array<int64_t, N> global_strides;
        global_strides[N - 1] = 1;
        for (int d = N - 2; d >= 0; --d)
            global_strides[d] = global_strides[d + 1] * (int64_t)arr->global_shape[d + 1];

        std::array<int64_t, N> local_strides;
        local_strides[N - 1] = 1;
        for (int d = N - 2; d >= 0; --d)
            local_strides[d] = local_strides[d + 1] * (int64_t)arr->region.size(d + 1);

        int64_t inner_size = arr->region.size(N - 1);

        std::array<int, N> idx = {};
        while (true) {
            int64_t global_offset = 0;
            int64_t local_offset = 0;
            for (int d = 0; d < N; ++d) {
                global_offset += ((int64_t)global_region.start[d] + idx[d]) * global_strides[d];
                local_offset += (int64_t)idx[d] * local_strides[d];
            }
            // Send as bytes: offset and size in bytes
            int64_t byte_offset = global_offset * esz;
            int64_t byte_size = inner_size * esz;
            char* base = static_cast<char*>(arr->data_ptr());
            dag_proxy[0].gather(epoch, name, byte_offset, byte_size, base + local_offset * esz);

            int d = N - 2;
            while (d >= 0) {
                if (++idx[d] < arr->region.size(d))
                    break;
                idx[d] = 0;
                --d;
            }
            if (d < 0)
                break;
        }
    }
}

template void PartitionImpl<1>::process_get(int);
template void PartitionImpl<2>::process_get(int);
template void PartitionImpl<3>::process_get(int);
