#include "backend_internal.hpp"
#include "decomposition_solver.hpp"
#include "jit_api.hpp"

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_set>

#ifdef USE_KOKKOS
namespace {
std::once_flag kokkos_once;   // SMP: the PEs of a process share one Kokkos instance
std::atomic<int> kokkos_users{0};
} // namespace
#endif

ArrayDAGGroup::ArrayDAGGroup() {
#ifdef USE_KOKKOS
    std::call_once(kokkos_once, [] {
        if (!Kokkos::is_initialized())
            Kokkos::initialize();
    });
    ++kokkos_users;
#ifdef USE_NVIDIA
    // SMP: each PE thread uses the device Kokkos chose for this process
    if (cudaSetDevice(Kokkos::device_id()) != cudaSuccess)
        CkAbort("PE %d: cudaSetDevice(%d) failed", CkMyPe(), Kokkos::device_id());
#endif
#endif
    if (CkMyPe() == 0) {
        CkArrayOptions opts;
        opts.setMap(CProxy_PartitionMap::ckNew());
        partition_proxy_1 = CProxy_Partition1D::ckNew(opts);
        partition_proxy_2 = CProxy_Partition2D::ckNew(opts);
        partition_proxy_3 = CProxy_Partition3D::ckNew(opts);
        thisProxy.set_proxies(partition_proxy_1, partition_proxy_2, partition_proxy_3);
    }
}

void ArrayDAGGroup::set_proxies(CkArrayID p1, CkArrayID p2, CkArrayID p3) {
    partition_proxy_1 = CProxy_Partition1D(p1);
    partition_proxy_2 = CProxy_Partition2D(p2);
    partition_proxy_3 = CProxy_Partition3D(p3);
#ifdef USE_NVIDIA
    ipc_probe_start();
#endif
    CkCallback cb(CkReductionTarget(ArrayDAGGroup, proxies_ready), thisProxy[0]);
    contribute(cb);
}

void ArrayDAGGroup::proxies_ready() {
    Server<CProxy_ArrayDAGGroup>::initialize(thisProxy);
    thisProxy.prewarm();
    jit_cache_maintain();
}

#ifdef USE_NVIDIA
/// Pre-warm uses the GPU only if this thread's current device is the one Kokkos uses (no
/// context is made on another GPU, e.g. GPU 0 from an SMP PE thread).
static bool prewarm_device() {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev != Kokkos::Cuda().cuda_device()) {
        cudaGetLastError();
        return false;
    }
    return true;
}
#endif

// CCS requests that arrive meanwhile wait in the scheduler queue.
void ArrayDAGGroup::prewarm() {
    const char* on = getenv("CNUM_PREWARM");
    if (on && std::string(on) == "0")
        return;
    double budget = 250.0;
    if (const char* ms = getenv("CNUM_PREWARM_MS")) {
        char* end = nullptr;
        double v = strtod(ms, &end);
        if (end != ms && *end == '\0' && v >= 0)
            budget = v;
    }
    double t0 = CkWallTimer(), handles = -1;
    bool gpu = true;
#ifdef USE_NVIDIA
    size_t free = 0, total = 0;
    gpu = budget > 0 && prewarm_device() && cudaMemGetInfo(&free, &total) == cudaSuccess &&
          free >= (size_t(1) << 30);
    if (gpu)
        handles = ct_prewarm_blas();
    cudaGetLastError();
#endif
    double t1 = CkWallTimer();
    // kernels keep a fifth of the budget even when the BLAS part overran
    int kernels = gpu ? jit_prewarm(std::max(budget - 1e3 * (t1 - t0), 0.2 * budget)) : 0;
    if (getenv("CNUM_JIT_TIME"))
        CkPrintf("[%d] prewarm: blas %.1f ms (handles %.1f), %d cached kernels %.1f ms\n",
                 CkMyPe(), 1e3 * (t1 - t0), handles, kernels, 1e3 * (CkWallTimer() - t1));
}

ArrayDAGGroup::~ArrayDAGGroup() {
    std::unordered_set<void*> freed;
    for (auto& [id, handle] : module_cache)
        if (handle && freed.insert(handle).second)
            jit_free_module(handle);
#ifdef USE_KOKKOS
    if (--kokkos_users == 0 && Kokkos::is_initialized()) {
#ifdef USE_NVIDIA
        gpu_comm_shutdown();
#endif
        Kokkos::finalize();
    }
#endif
}

void ArrayDAGGroup::compute_decompositions(DAG* dag) {
    // Solve over the arrays this DAG references only: array_meta keeps every array
    // ever created, and the others are all decomp_final (singleton classes).
    std::map<int, ArrayMetadata> sub; // ordered: every PE solves alike
    std::unordered_set<ASTNode*> seen;
    std::vector<ASTNode*> stack;
    for (auto& [id, node] : dag->nodes)
        for (ASTNode* root : node->ast->roots)
            stack.push_back(root);
    while (!stack.empty()) {
        ASTNode* n = stack.back();
        stack.pop_back();
        if (n == nullptr || !seen.insert(n).second)
            continue;
        if (!n->is_scalar) {
            auto it = array_meta.find(n->result_name);
            if (it != array_meta.end())
                sub.emplace(it->first, it->second);
        }
        for (ASTNode* op : n->operands)
            stack.push_back(op);
    }
    static thread_local const bool log = getenv("CNUM_DECOMP_LOG") != nullptr && CkMyPe() == 0;
    std::unordered_set<int> was_final;
    for (auto& [name, meta] : sub)
        if (log && meta.decomp_final)
            was_final.insert(name);
    decomposition_solver::compute_decompositions(sub, dag, odf, CkNumPes());
    for (auto& [name, meta] : sub)
        array_meta[name] = meta;

    if (!log)
        return;
    for (auto& [name, meta] : sub)
        if (!was_final.count(name))
            CkPrintf("decomp: array %d shape (%d,%d,%d) tile %d offset (%d,%d,%d)\n", name,
                     meta.global_shape[0], meta.global_shape[1], meta.global_shape[2],
                     meta.tile, meta.offset[0], meta.offset[1], meta.offset[2]);
}

// Copies a piece of one chare's tile into the row-major result. meta (from process_get):
// [nouter, row bytes, piece offset in the tile, tile offset in the result,
//  outer extents[nouter], outer byte strides[nouter]].
static void gather_scatter(char* out, const int64_t* meta, int64_t size, const char* src) {
    int nouter = static_cast<int>(meta[0]);
    int64_t row_bytes = meta[1], pos = meta[2], base = meta[3];
    const int64_t* extent = meta + 4;
    const int64_t* stride = meta + 4 + nouter;
    if (row_bytes <= 0)
        return;
    int64_t row = pos / row_bytes, in_row = pos % row_bytes;
    int64_t idx[3] = {0, 0, 0};
    for (int d = nouter - 1; d >= 0; --d) {
        idx[d] = row % extent[d];
        row /= extent[d];
    }
    char* run = nullptr;
    int64_t run_len = 0;
    const char* run_src = src;
    for (int64_t done = 0; done < size;) {
        int64_t dst = base + in_row;
        for (int d = 0; d < nouter; ++d)
            dst += idx[d] * stride[d];
        int64_t n = std::min(row_bytes - in_row, size - done);
        if (run != nullptr && out + dst == run + run_len) {
            run_len += n;
        } else {
            if (run != nullptr)
                memcpy(run, run_src, run_len);
            run = out + dst;
            run_src = src + done;
            run_len = n;
        }
        done += n;
        in_row = 0;
        for (int d = nouter - 1; d >= 0; --d) {
            if (++idx[d] < extent[d])
                break;
            idx[d] = 0;
        }
    }
    if (run != nullptr)
        memcpy(run, run_src, run_len);
}

void ArrayDAGGroup::receive_get_request(int ndims, int epoch, int name, int64_t size, int dtype) {
    epoch_arrived(epoch, [=] { receive_get_request_now(ndims, epoch, name, size, dtype); });
}

void ArrayDAGGroup::receive_get_request_now(int ndims, int epoch, int name, int64_t size,
                                            int dtype) {
    if (name == kBarrierName) {
        int64_t chares = 0;
        for (auto& [nd, pg] : partition_grid) {
            if (pg.grid[0] <= 0)
                continue;
            DAGGroup::receive_get_request(nd, epoch, name, 0, dtype);
            barrier_grids[epoch][nd] = pg;
            int64_t n = 1;
            for (int d = 0; d < nd; ++d)
                n *= pg.grid[d];
            chares += n;
        }
        partition_proxy_1.run();
        partition_proxy_2.run();
        partition_proxy_3.run();
        if (CkMyPe() == 0) {
            barrier_pending[epoch] += chares + 1; // early arrivals made it negative
            barrier_arrive(epoch);
        }
        return;
    }
    DAGGroup::receive_get_request(ndims, epoch, name, size, dtype);

    // Store empty DAGs for ndims that have active partitions but aren't involved in this GET.
    for (auto& [nd, pg] : partition_grid) {
        if (pg.grid[0] > 0 && nd != ndims) {
            add_dag(nd, epoch, new DAG());
        }
    }

    // Wake up all partition chares so they can check for new work.
    partition_proxy_1.run();
    partition_proxy_2.run();
    partition_proxy_3.run();

    // Only PE 0 allocates the gather buffer for assembly
    // size is element count; convert to bytes using the wire dtype
    int elem_size = dtype_size(static_cast<DType>(dtype));
    int64_t byte_size = size * elem_size;
    if (CkMyPe() == 0) {
        gather_buffers[epoch] = new char[byte_size];
        gather_total[epoch] = byte_size;
        if (gather_counts.find(epoch) == gather_counts.end())
            gather_counts[epoch] = 0;

        // Flush any gather fragments that arrived before this allocation
        auto early_it = gather_early.find(epoch);
        if (early_it != gather_early.end()) {
            for (auto& frag : early_it->second) {
                gather_scatter(gather_buffers[epoch], frag.meta.data(), frag.size, frag.data);
                delete[] frag.data;
            }
            gather_early.erase(early_it);
            if (gather_counts[epoch] >= gather_total[epoch]) {
                reply_to_client(epoch, byte_size, gather_buffers[epoch], /*fetch=*/true);
                gather_counts.erase(epoch);
                gather_total.erase(epoch);
                gather_buffers.erase(epoch);
            }
        }
    }
}

void ArrayDAGGroup::mem_report(int epoch) {
    mem_stats_report(this, epoch);
}

void ArrayDAGGroup::check_layout_hash(int epoch, int64_t hash) {
    auto [it, fresh] = layout_hashes.try_emplace(epoch, hash);
    if (fresh)
        return;
    if (it->second != hash)
        CkAbort("receive_dag epoch=%d: PEs %d and %d solved the DAG's layouts differently",
                epoch, (CkMyPe() + CkNumPes() - 1) % CkNumPes(), CkMyPe());
    layout_hashes.erase(it);
}

void ArrayDAGGroup::reply_to_client(int epoch, int64_t byte_size, char* data, bool fetch) {
    if (client_error.empty()) {
        if (fetch)
            Server<CProxy_ArrayDAGGroup>::send_fetch_reply(epoch, byte_size, data);
        else
            Server<CProxy_ArrayDAGGroup>::send_reply(epoch, byte_size, data);
        return;
    }
    if (fetch)
        delete[] data;
    std::string msg = std::string(kClientErrorTag, kClientErrorTagLen) + client_error;
    client_error.clear();
    Server<CProxy_ArrayDAGGroup>::send_reply(epoch, (int64_t)msg.size(), &msg[0]);
}

void ArrayDAGGroup::barrier_arrive(int epoch) {
    if (--barrier_pending[epoch] != 0)
        return;
    barrier_pending.erase(epoch);
    if (mem_stats_on())
        thisProxy.mem_report(epoch);
    char none = 0;
    reply_to_client(epoch, 1, &none); // CCS needs a non-empty reply
}

void ArrayDAGGroup::gather(int epoch, int name, int nmeta, int64_t* meta, int64_t size,
                           char* data) {
    if (gather_buffers.find(epoch) == gather_buffers.end()) {
        char* buf = new char[size];
        memcpy(buf, data, size);
        gather_early[epoch].push_back({std::vector<int64_t>(meta, meta + nmeta), size, buf});
        gather_counts[epoch] += size;
        return;
    }

    gather_scatter(gather_buffers[epoch], meta, size, data);
    gather_counts[epoch] += size;
    if (gather_counts[epoch] >= gather_total[epoch]) {
        reply_to_client(epoch, gather_total[epoch], gather_buffers[epoch], /*fetch=*/true);
        gather_counts.erase(epoch);
        gather_total.erase(epoch);
        gather_buffers.erase(epoch);
    }
}
