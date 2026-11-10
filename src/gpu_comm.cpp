#include "backend.hpp"

// Startup probe for CUDA IPC between the processes of a host. Rank 0 of each process exports
// a small buffer; the processes on the same host (same hostname and boot id) map each peer's
// buffer and read it back; the pairs that worked are gathered, and only those use IPC
// (gpu_ipc_peer). Messages sent before the probe completes are staged. Never aborts.
// CNUM_GPU_IPC_PROBE_FAIL="src-dst,...": process dst fails to map process src's buffer.

#ifdef USE_KOKKOS
#ifdef USE_NVIDIA
#include <cstdio>
#include <string>
#include <unistd.h>

namespace {
struct ProbeRec {
    int node;
    int exported;
    cudaIpcMemHandle_t handle;
    char host[64];
    char boot[48];
    char pci[32]; // the process's GPU
};
struct ProbePair {
    int src, dst;
};
constexpr unsigned char kProbeByte = 0x5a;
void* probe_buf = nullptr;
int probe_expected = 0;

void host_identity(char* host, char* boot) {
    gethostname(host, 63);
    if (FILE* f = std::fopen("/proc/sys/kernel/random/boot_id", "r")) {
        if (!std::fgets(boot, 47, f))
            boot[0] = 0;
        std::fclose(f);
    }
}

bool same_host(const ProbeRec& a, const ProbeRec& b) {
    return std::strcmp(a.host, b.host) == 0 && std::strcmp(a.boot, b.boot) == 0;
}

bool injected_failure(int src, int dst) {
    const char* env = std::getenv("CNUM_GPU_IPC_PROBE_FAIL");
    if (!env)
        return false;
    std::string s(env);
    for (size_t at = 0; at < s.size();) {
        size_t end = s.find(',', at);
        int a = -1, b = -1;
        if (std::sscanf(s.substr(at, end - at).c_str(), "%d-%d", &a, &b) == 2 && a == src &&
            b == dst)
            return true;
        at = end == std::string::npos ? s.size() : end + 1;
    }
    return false;
}

/// Map a peer's probe buffer and read it back.
bool probe_peer(const ProbeRec& r) {
    void* p = nullptr;
    if (injected_failure(r.node, CkMyNode()) ||
        cudaIpcOpenMemHandle(&p, r.handle, cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    unsigned char byte = 0;
    bool ok = cudaMemcpy(&byte, p, 1, cudaMemcpyDeviceToHost) == cudaSuccess && byte == kProbeByte;
    cudaIpcCloseMemHandle(p);
    cudaGetLastError();
    return ok;
}
} // namespace
#endif

void ArrayDAGGroup::ipc_probe_start() {
    std::vector<char> out;
#ifdef USE_NVIDIA
    if (CkMyRank() == 0) {
        cudaSetDevice(Kokkos::Cuda().cuda_device());
        ProbeRec rec{};
        rec.node = CkMyNode();
        host_identity(rec.host, rec.boot);
        cudaDeviceGetPCIBusId(rec.pci, sizeof(rec.pci), Kokkos::Cuda().cuda_device());
        if (gpu_ipc_enabled() && cudaMalloc(&probe_buf, 4096) == cudaSuccess)
            rec.exported = cudaMemset(probe_buf, kProbeByte, 4096) == cudaSuccess &&
                           cudaDeviceSynchronize() == cudaSuccess &&
                           cudaIpcGetMemHandle(&rec.handle, probe_buf) == cudaSuccess;
        cudaGetLastError();
        out.assign(reinterpret_cast<char*>(&rec), reinterpret_cast<char*>(&rec + 1));
    }
#endif
    contribute(out.size(), out.data(), CkReduction::concat,
               CkCallback(CkReductionTarget(ArrayDAGGroup, ipc_probe_exports), thisProxy));
}

void ArrayDAGGroup::ipc_probe_exports(int n, char* data) {
    std::vector<char> out;
#ifdef USE_NVIDIA
    if (CkMyRank() == 0) {
        cudaSetDevice(Kokkos::Cuda().cuda_device());
        int count = n / (int)sizeof(ProbeRec);
        std::vector<ProbeRec> recs(count);
        std::memcpy(recs.data(), data, count * sizeof(ProbeRec));
        const ProbeRec* me = nullptr;
        for (int i = 0; i < count; ++i)
            if (recs[i].node == CkMyNode())
                me = &recs[i];
        if (me && me->pci[0]) {
            int sharers = 0;
            for (int i = 0; i < count; ++i)
                sharers += same_host(recs[i], *me) && std::strcmp(recs[i].pci, me->pci) == 0;
            gpu_sharers().store(std::max(sharers, 1));
            if (sharers > 1)
                trim_device_caches();
        }
        for (int i = 0; i < count; ++i)
            for (int j = 0; CkMyPe() == 0 && j < count; ++j)
                probe_expected += i != j && recs[i].exported && recs[j].exported &&
                                  same_host(recs[i], recs[j]);
        for (int i = 0; me && me->exported && i < count; ++i) {
            const ProbeRec& r = recs[i];
            if (r.node == CkMyNode() || !r.exported || !same_host(r, *me))
                continue;
            if (probe_peer(r)) {
                ProbePair pr{r.node, CkMyNode()};
                out.insert(out.end(), reinterpret_cast<char*>(&pr),
                           reinterpret_cast<char*>(&pr + 1));
            } else {
                CkPrintf("[%d] gpu ipc: cannot map the GPU memory of process %d; its messages "
                         "here are staged through host memory\n",
                         CkMyPe(), r.node);
            }
        }
    }
#endif
    contribute(out.size(), out.data(), CkReduction::concat,
               CkCallback(CkReductionTarget(ArrayDAGGroup, ipc_probe_results), thisProxy));
}

void ArrayDAGGroup::ipc_probe_results(int n, char* data) {
#ifdef USE_NVIDIA
    if (CkMyRank() != 0)
        return;
    int count = n / (int)sizeof(ProbePair);
    std::vector<ProbePair> pairs(count);
    std::memcpy(pairs.data(), data, count * sizeof(ProbePair));
    auto& a = gpu_ipc_arena();
    a.peer_ok.assign(CkNumNodes(), 0);
    for (int i = 0; i < count; ++i)
        if (pairs[i].src == CkMyNode())
            a.peer_ok[pairs[i].dst] = 1;
    a.probed.store(true, std::memory_order_release);
    if (probe_buf) {
        cudaFree(probe_buf);
        probe_buf = nullptr;
    }
    if (CkMyPe() == 0 && probe_expected > 0)
        CkPrintf("gpu ipc: %d of %d same-host process pairs use CUDA IPC\n", count,
                 probe_expected);
#endif
}
#endif
