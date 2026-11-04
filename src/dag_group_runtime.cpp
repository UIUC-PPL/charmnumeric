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
    static const bool log = getenv("CNUM_DECOMP_LOG") != nullptr && CkMyPe() == 0;
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
                memcpy(gather_buffers[epoch] + frag.offset, frag.data, frag.size);
                delete[] frag.data;
            }
            gather_early.erase(early_it);
            if (gather_counts[epoch] >= gather_total[epoch]) {
                Server<CProxy_ArrayDAGGroup>::send_fetch_reply(epoch, byte_size,
                                                               gather_buffers[epoch]);
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

void ArrayDAGGroup::barrier_arrive(int epoch) {
    if (--barrier_pending[epoch] != 0)
        return;
    barrier_pending.erase(epoch);
    if (mem_stats_on())
        thisProxy.mem_report(epoch);
    char none = 0;
    Server<CProxy_ArrayDAGGroup>::send_reply(epoch, 1, &none); // CCS needs a non-empty reply
}

void ArrayDAGGroup::gather(int epoch, int name, int64_t offset, int64_t size, char* data) {
    // offset and size are in bytes
    if (gather_buffers.find(epoch) == gather_buffers.end()) {
        char* buf = new char[size];
        memcpy(buf, data, size);
        gather_early[epoch].push_back({offset, size, buf});
        gather_counts[epoch] += size;
        return;
    }

    memcpy(gather_buffers[epoch] + offset, data, size);
    gather_counts[epoch] += size;
    if (gather_counts[epoch] >= gather_total[epoch]) {
        Server<CProxy_ArrayDAGGroup>::send_fetch_reply(epoch, gather_total[epoch],
                                                       gather_buffers[epoch]);
        gather_counts.erase(epoch);
        gather_total.erase(epoch);
        gather_buffers.erase(epoch);
    }
}
