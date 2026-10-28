#pragma once

// GPU messages between processes on one host through CUDA IPC, bypassing Charm++'s
// CkDeviceBuffer host staging (a cudaMallocHost, a blocking stream sync and a cudaFreeHost per
// message). Each process carves send blocks out of a few cudaMalloc slabs whose IPC handles
// travel with the messages; the receiver maps a peer slab once and copies device to device.

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hapi.h"
#include <cuda_runtime.h>

struct GpuIpcArena {
    static constexpr int kMinClass = 12, kSlabClass = 28; // 4 KB blocks .. 256 MB slabs
    struct Slab {
        char* base;
        int64_t used;
        cudaIpcMemHandle_t handle;
    };
    std::vector<Slab> slabs;
    std::vector<std::pair<int, int64_t>> free_lists[kSlabClass + 1]; // (slab, offset)
    std::unordered_map<void*, int> live;                              // block -> size class
    std::map<std::pair<int, int>, char*> peers; // (node, slab) -> mapped peer slab
    int64_t limit = 0;
    int enabled = -1;
    std::mutex mu;
};

inline GpuIpcArena& gpu_ipc_arena() {
    static GpuIpcArena arena;
    return arena;
}

/// On unless CNUM_GPU_IPC=0; CNUM_GPU_IPC_MB bounds the slabs (default 1024).
inline bool gpu_ipc_enabled() {
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    if (a.enabled < 0) {
        const char* env = std::getenv("CNUM_GPU_IPC");
        a.enabled = !env || std::atoi(env) != 0;
        const char* mb = std::getenv("CNUM_GPU_IPC_MB");
        a.limit = (mb ? std::atoll(mb) : 1024) << 20;
    }
    return a.enabled;
}

/// `pe` (in another process) shares this host and CUDA IPC is on.
inline bool gpu_ipc_peer(int pe) {
    return CmiPeOnSamePhysicalNode(pe, CkMyPe()) && gpu_ipc_enabled();
}

/// Block of at least `bytes` in an exported slab, or nullptr (too large or arena full).
inline void* gpu_ipc_alloc(int64_t bytes, int& slab, int64_t& offset) {
    int cls = GpuIpcArena::kMinClass;
    while (cls <= GpuIpcArena::kSlabClass && (int64_t(1) << cls) < bytes)
        ++cls;
    if (cls > GpuIpcArena::kSlabClass)
        return nullptr;
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    const int64_t sz = int64_t(1) << cls;
    if (!a.free_lists[cls].empty()) {
        std::tie(slab, offset) = a.free_lists[cls].back();
        a.free_lists[cls].pop_back();
    } else {
        // bump allocation; blocks are aligned to their size inside a slab
        slab = -1;
        for (int s = 0; s < (int)a.slabs.size() && slab < 0; ++s) {
            int64_t off = (a.slabs[s].used + sz - 1) & ~(sz - 1);
            if (off + sz <= (int64_t(1) << GpuIpcArena::kSlabClass)) {
                slab = s;
                offset = off;
            }
        }
        if (slab < 0) {
            const int64_t slab_bytes = int64_t(1) << GpuIpcArena::kSlabClass;
            if (int64_t(a.slabs.size() + 1) * slab_bytes > a.limit)
                return nullptr;
            GpuIpcArena::Slab s;
            if (cudaMalloc(&s.base, slab_bytes) != cudaSuccess) {
                cudaGetLastError();
                return nullptr;
            }
            if (cudaIpcGetMemHandle(&s.handle, s.base) != cudaSuccess) {
                cudaGetLastError();
                cudaFree(s.base);
                a.enabled = 0;
                return nullptr;
            }
            s.used = 0;
            a.slabs.push_back(s);
            slab = (int)a.slabs.size() - 1;
            offset = 0;
        }
        a.slabs[slab].used = offset + sz;
    }
    void* p = a.slabs[slab].base + offset;
    a.live[p] = cls;
    return p;
}

/// Return a block to the arena; false if `ptr` is not an arena block.
inline bool gpu_ipc_free(void* ptr) {
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    auto it = a.live.find(ptr);
    if (it == a.live.end())
        return false;
    for (int s = 0; s < (int)a.slabs.size(); ++s) {
        char* b = a.slabs[s].base;
        if (static_cast<char*>(ptr) >= b &&
            static_cast<char*>(ptr) < b + (int64_t(1) << GpuIpcArena::kSlabClass)) {
            a.free_lists[it->second].push_back({s, static_cast<char*>(ptr) - b});
            break;
        }
    }
    a.live.erase(it);
    return true;
}

inline cudaIpcMemHandle_t gpu_ipc_handle(int slab) {
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    return a.slabs[slab].handle;
}

/// Device pointer to slab `slab` of process `node`, mapped on first use.
inline char* gpu_ipc_peer_slab(int node, int slab, const char* handle) {
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    char*& p = a.peers[{node, slab}];
    if (!p) {
        cudaIpcMemHandle_t h;
        std::memcpy(&h, handle, sizeof(h));
        void* q = nullptr;
        cudaError_t err = cudaIpcOpenMemHandle(&q, h, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess)
            CkAbort("CUDA IPC: cannot map a GPU buffer of process %d (%s); set CNUM_GPU_IPC=0",
                    node, cudaGetErrorString(err));
        p = static_cast<char*>(q);
    }
    return p;
}

