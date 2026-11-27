#include "backend_internal.hpp"

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
