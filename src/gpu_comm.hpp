#pragma once

// GPU messages between processes, bypassing Charm++'s CkDeviceBuffer host staging (a
// cudaMallocHost, a blocking stream sync and a cudaFreeHost per message).
// - Same host: CUDA IPC. Each process carves send blocks out of a few cudaMalloc slabs whose
//   IPC handles travel with the messages; the receiver maps a peer slab once and copies
//   device to device. Used only for pairs that passed the startup probe (gpu_comm.cpp).
// - Other hosts, failed pairs, or CNUM_GPU_IPC=0: staged through pooled pinned host buffers
//   and sent as a plain message.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hapi.h"
#include <cuda_runtime.h>

/// Integer environment variable clamped to [lo, hi], or `def` when unset.
inline int64_t gpu_comm_env(const char* name, int64_t def, int64_t lo, int64_t hi) {
    const char* v = std::getenv(name);
    return v ? std::min(std::max<int64_t>(std::atoll(v), lo), hi) : def;
}

struct GpuIpcArena {
    static constexpr int kMinClass = 12, kSlabClass = 28; // 4 KB blocks .. 256 MB slabs
    struct Slab {
        char* base;
        int64_t used;
        cudaIpcMemHandle_t handle;
    };
    std::vector<Slab> slabs;
    std::set<std::pair<int, int64_t>> free_lists[kSlabClass + 1]; // (slab, offset)
    std::unordered_map<void*, std::pair<int, int>> live;              // block -> (slab, class)
    std::map<std::pair<int, int>, char*> peers; // (node, slab) -> mapped peer slab
    std::vector<char> peer_ok;                  // by process: receives our IPC blocks
    std::atomic<bool> probed{false};
    int64_t limit = 0;
    int enabled = -1;
    bool slab_failed = false, warned = false, closed = false;
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
        a.enabled = gpu_comm_env("CNUM_GPU_IPC", 1, 0, 1) != 0;
        a.limit = gpu_comm_env("CNUM_GPU_IPC_MB", 1024, 0, 1 << 20) << 20;
    }
    return a.enabled;
}

/// `pe` (in another process) passed the startup probe: it can map our slabs.
inline bool gpu_ipc_peer(int pe) {
    auto& a = gpu_ipc_arena();
    return a.probed.load(std::memory_order_acquire) && a.peer_ok[CmiNodeOf(pe)] &&
           gpu_ipc_enabled() && !a.closed;
}

inline void gpu_ipc_warn(GpuIpcArena& a, const char* why) {
    if (!a.warned) {
        a.warned = true;
        CkPrintf("[%d] gpu ipc: %s; such messages are staged through host memory\n", CkMyPe(),
                 why);
    }
}

/// Free block `cls` at `offset`, merged with its free buddies: freed blocks otherwise stay split
/// and the arena fills up with pieces too small for later messages.
inline void gpu_ipc_release(GpuIpcArena& a, int slab, int64_t offset, int cls) {
    for (; cls < GpuIpcArena::kSlabClass; ++cls) {
        auto it = a.free_lists[cls].find({slab, offset ^ (int64_t(1) << cls)});
        if (it == a.free_lists[cls].end())
            break;
        a.free_lists[cls].erase(it);
        offset &= ~(int64_t(1) << cls);
    }
    a.free_lists[cls].insert({slab, offset});
}

/// Block of at least `bytes` in an exported slab, or nullptr (too large or arena full).
inline void* gpu_ipc_alloc(int64_t bytes, int& slab, int64_t& offset) {
    auto& a = gpu_ipc_arena();
    int cls = GpuIpcArena::kMinClass;
    while (cls <= GpuIpcArena::kSlabClass && (int64_t(1) << cls) < bytes)
        ++cls;
    std::lock_guard<std::mutex> lock(a.mu);
    if (a.closed || a.limit == 0)
        return nullptr;
    if (cls > GpuIpcArena::kSlabClass) {
        gpu_ipc_warn(a, "a message is larger than a 256 MB slab");
        return nullptr;
    }
    const int64_t sz = int64_t(1) << cls, slab_bytes = int64_t(1) << GpuIpcArena::kSlabClass;
    int k = cls;
    while (k <= GpuIpcArena::kSlabClass && a.free_lists[k].empty())
        ++k;
    if (k <= GpuIpcArena::kSlabClass) {
        // free block, split down to this class
        std::tie(slab, offset) = *a.free_lists[k].begin();
        a.free_lists[k].erase(a.free_lists[k].begin());
        for (int j = k - 1; j >= cls; --j)
            a.free_lists[j].insert({slab, offset + (int64_t(1) << j)});
    } else {
        // bump allocation; blocks are aligned to their size, the gap goes to smaller classes
        slab = -1;
        for (int s = 0; s < (int)a.slabs.size() && slab < 0; ++s) {
            int64_t off = (a.slabs[s].used + sz - 1) & ~(sz - 1);
            if (off + sz <= slab_bytes) {
                slab = s;
                offset = off;
            }
        }
        if (slab < 0) {
            if (a.slab_failed || int64_t(a.slabs.size() + 1) * slab_bytes > a.limit) {
                gpu_ipc_warn(a, "the IPC slabs are full (CNUM_GPU_IPC_MB)");
                return nullptr;
            }
            GpuIpcArena::Slab s;
            if (cudaMalloc(&s.base, slab_bytes) != cudaSuccess) {
                cudaGetLastError();
                a.slab_failed = true;
                gpu_ipc_warn(a, "cannot allocate another IPC slab");
                return nullptr;
            }
            if (cudaIpcGetMemHandle(&s.handle, s.base) != cudaSuccess) {
                cudaGetLastError();
                cudaFree(s.base);
                a.slab_failed = true;
                gpu_ipc_warn(a, "cannot export an IPC slab");
                return nullptr;
            }
            s.used = 0;
            a.slabs.push_back(s);
            slab = (int)a.slabs.size() - 1;
            offset = 0;
        }
        for (int64_t u = a.slabs[slab].used; u < offset;) {
            int j = GpuIpcArena::kMinClass;
            while (j < cls && u % (int64_t(2) << j) == 0 && u + (int64_t(2) << j) <= offset)
                ++j;
            gpu_ipc_release(a, slab, u, j);
            u += int64_t(1) << j;
        }
        a.slabs[slab].used = offset + sz;
    }
    void* p = a.slabs[slab].base + offset;
    a.live[p] = {slab, cls};
    return p;
}

/// Return a block to the arena; false if `ptr` is not an arena block.
inline bool gpu_ipc_free(void* ptr) {
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    auto it = a.live.find(ptr);
    if (it == a.live.end())
        return false;
    if (a.closed)
        return true;
    auto [s, cls] = it->second;
    gpu_ipc_release(a, s, static_cast<char*>(ptr) - a.slabs[s].base, cls);
    a.live.erase(it);
    return true;
}

/// Drop an arena block without reusing it (a receiver may still be reading it).
inline bool gpu_ipc_forget(void* ptr) {
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    return a.closed ? a.live.count(ptr) > 0 : a.live.erase(ptr) > 0;
}

inline cudaIpcMemHandle_t gpu_ipc_handle(int slab) {
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    cudaIpcMemHandle_t h{};
    if (slab >= 0 && slab < (int)a.slabs.size())
        h = a.slabs[slab].handle;
    return h;
}

/// Device pointer to slab `slab` of process `node`, mapped on first use; nullptr after
/// gpu_comm_shutdown.
inline char* gpu_ipc_peer_slab(int node, int slab, const char* handle) {
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    char*& p = a.peers[{node, slab}];
    if (!p && !a.closed) {
        cudaIpcMemHandle_t h;
        std::memcpy(&h, handle, sizeof(h));
        void* q = nullptr;
        cudaError_t err = cudaIpcOpenMemHandle(&q, h, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess)
            CkAbort("CUDA IPC: cannot map IPC slab %d (%d MB) of process %d on process %d (%s), "
                    "although the startup probe could map a small buffer; the device may be out "
                    "of memory or the mapping limit reached. Rerun with CNUM_GPU_IPC=0",
                    slab, 1 << (GpuIpcArena::kSlabClass - 20), node, CkMyNode(),
                    cudaGetErrorString(err));
        p = static_cast<char*>(q);
    }
    return p;
}

/// Pinned host buffers by power-of-two size class. A block is released from the stream
/// callback that follows its last async copy, so reuse never races a copy.
struct PinnedPool {
    static constexpr int kMinClass = 12, kMaxClass = 28; // 4 KB .. 256 MB
    std::vector<void*> free_lists[kMaxClass + 1];
    std::unordered_map<void*, int> live;
    int64_t cached = 0, limit = -1;
    bool on = true;
    std::mutex mu;
};

inline PinnedPool& pinned_pool() {
    static PinnedPool pool;
    return pool;
}

/// Staged sends on unless CNUM_GPU_STAGE=0. CNUM_GPU_STAGE_MB caps only the blocks cached
/// for reuse (default 512), not those in flight. nullptr: use the CkDeviceBuffer path.
inline void* pinned_alloc(int64_t bytes) {
    auto& pool = pinned_pool();
    int cls = PinnedPool::kMinClass;
    while (cls <= PinnedPool::kMaxClass && (int64_t(1) << cls) < bytes)
        ++cls;
    void* p = nullptr;
    {
        std::lock_guard<std::mutex> lock(pool.mu);
        if (pool.limit < 0) {
            pool.on = gpu_comm_env("CNUM_GPU_STAGE", 1, 0, 1) != 0;
            pool.limit = gpu_comm_env("CNUM_GPU_STAGE_MB", 512, 0, 1 << 20) << 20;
        }
        if (cls > PinnedPool::kMaxClass || !pool.on)
            return nullptr;
        if (!pool.free_lists[cls].empty()) {
            p = pool.free_lists[cls].back();
            pool.free_lists[cls].pop_back();
            pool.cached -= int64_t(1) << cls;
            pool.live[p] = cls;
            return p;
        }
    }
    if (cudaMallocHost(&p, size_t(1) << cls) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(pool.mu);
    pool.live[p] = cls;
    return p;
}

inline void pinned_free(void* p) {
    auto& pool = pinned_pool();
    {
        std::lock_guard<std::mutex> lock(pool.mu);
        auto it = pool.live.find(p);
        if (it == pool.live.end())
            return;
        int cls = it->second;
        pool.live.erase(it);
        if (pool.cached + (int64_t(1) << cls) <= pool.limit) {
            pool.free_lists[cls].push_back(p);
            pool.cached += int64_t(1) << cls;
            return;
        }
    }
    cudaFreeHost(p);
}

/// Process exit: no new IPC sends or mappings, and late frees become no-ops. Slabs, peer
/// mappings and pinned blocks stay until the process ends (a peer may still be copying).
inline void gpu_comm_shutdown() {
    auto& a = gpu_ipc_arena();
    std::lock_guard<std::mutex> lock(a.mu);
    a.closed = true;
}
