#pragma once

#include "backend.hpp"
#include "big_msg.hpp"
#include "jit_api.hpp"

#include <limits>

// ---- Stream policy helpers (USE_KOKKOS) ----
// Under USE_NVIDIA: Kokkos::Cuda exec space instances bound to per-chare streams.
// Under USE_KOKKOS without USE_NVIDIA: default execution space + Kokkos::fence().

#ifdef USE_KOKKOS
#ifdef USE_NVIDIA
#define CT_COMPUTE_POLICY(partition, n)                                                            \
    Kokkos::RangePolicy<Kokkos::Cuda>((partition)->compute_exec, 0, (n))
#define CT_COMM_POLICY(partition, n)                                                               \
    Kokkos::RangePolicy<Kokkos::Cuda>((partition)->comm_exec, 0, (n))
#define CT_COMPUTE_EXEC(partition) ((partition)->compute_exec)
#else
#define CT_COMPUTE_POLICY(partition, n) Kokkos::RangePolicy<>(0, (n))
#define CT_COMM_POLICY(partition, n) Kokkos::RangePolicy<>(0, (n))
#define CT_COMPUTE_EXEC(partition) Kokkos::DefaultExecutionSpace()
#endif
#endif

// Helper: fence Kokkos on non-NVIDIA GPU builds (no-op on CPU-only and NVIDIA).
#if defined(USE_KOKKOS) && !defined(USE_NVIDIA)
#define CT_KOKKOS_FENCE() Kokkos::fence()
#else
#define CT_KOKKOS_FENCE() ((void)0)
#endif

// ---- HAPI callback structs and functions (USE_NVIDIA only) ----

#ifdef USE_NVIDIA
/// Param struct for compute-done callbacks (hapiAddCallback).
/// Heap-allocated, freed by the callback function.
template <int N>
struct ComputeDoneParam {
    PartitionImpl<N>* partition;
    int node_id;
};

/// C callback invoked by HAPI when all prior compute_stream work completes.
template <int N>
static void compute_done_cb(void* param, void* msg) {
    auto* p = static_cast<ComputeDoneParam<N>*>(param);
    p->partition->executor->node_finished(p->node_id);
    delete p;
}

template <int N>
using RemoteBuffers = std::unordered_map<int, std::vector<RemoteBuffer<N>>>;

/// HAPI callback freeing a finished node's received buffers once its kernels have run.
template <int N>
static void free_remote_buffers_cb(void* param, void* msg) {
    auto* bufs = static_cast<RemoteBuffers<N>*>(param);
    for (auto& [idx, list] : *bufs)
        for (auto& rb : list)
            comm_free(rb.data);
    delete bufs;
}

/// Param struct for deferred-send callbacks (hapiAddCallback).
/// N_sender is the sender partition dimension, N_target is the target partition dimension.
/// For same-dimension sends, N_sender == N_target (the default).
template <int N_sender, int N_target = N_sender>
struct DeferredSendParam {
    PartitionImpl<N_sender>* partition;
    typename PartitionTraits<N_target>::ProxyType target_proxy;
    int node_id;
    int input_index;
    int inp_name;
    ChareIndex<N_target> target_ci;
    int region_data[N_target * 3];
    int64_t byte_size;
    void* send_buf;
    int send_id;
    void* ipc_buf = nullptr;  // CUDA IPC copy of send_buf (another process on this host)
    int ipc_slab;
    int64_t ipc_offset;
    void* host_buf = nullptr; // pinned copy of send_buf (another host)
    bool local = true;        // target in this process

    /// Same-dimension send constructor: target_proxy defaults to partition's own proxy.
    DeferredSendParam(PartitionImpl<N_sender>* p)
        : partition(p), target_proxy(p->thisProxy) {}

    /// Cross-dimensional send constructor: target_proxy is explicitly provided.
    DeferredSendParam(PartitionImpl<N_sender>* p, typename PartitionTraits<N_target>::ProxyType tp)
        : partition(p), target_proxy(tp) {}
};

/// C callback invoked by HAPI when comm_stream packing completes.
/// Performs the actual CkDeviceBuffer send.
template <int N_sender, int N_target = N_sender>
static void deferred_send_cb(void* param, void* msg) {
    auto* p = static_cast<DeferredSendParam<N_sender, N_target>*>(param);
    ChareIndex<N_sender> self_ci;
    for (int d = 0; d < N_sender; ++d)
        self_ci.idx[d] = p->partition->index[d];
    // send_id is the refnum: the source callback fires on the receiver, empty.
    CkCallback cb(PartitionTraits<N_sender>::CkIndexType::send_complete(nullptr),
                  proxy_at<N_sender>(p->partition->thisProxy, self_ci));
    cb.setRefnum(static_cast<CMK_REFNUM_TYPE>(p->send_id));
    if (p->host_buf) {
        auto target = proxy_at<N_target>(p->target_proxy, p->target_ci);
        if (p->byte_size > ct_msg_max_bytes())
            ct_send_frags(target, MSG_STAGED, p->node_id,
                          ct_data_meta(p->input_index, p->inp_name, N_target, p->region_data),
                          p->byte_size, static_cast<const char*>(p->host_buf));
        else
            target.receive_staged(p->node_id, p->input_index, p->inp_name, N_target,
                                  p->region_data, p->byte_size, static_cast<char*>(p->host_buf));
        pinned_free(p->host_buf);
        p->partition->send_complete(p->send_id);
    } else if (p->ipc_buf) {
        comm_free(p->send_buf);
        cudaIpcMemHandle_t h = gpu_ipc_handle(p->ipc_slab);
        proxy_at<N_target>(p->target_proxy, p->target_ci).receive_ipc(
            p->node_id, p->input_index, p->inp_name, N_target, p->region_data, p->byte_size,
            CkMyNode(), p->ipc_slab, p->ipc_offset, (int)sizeof(h),
            reinterpret_cast<const char*>(&h), cb);
    } else if (p->local && p->byte_size > ct_msg_max_bytes()) {
        // charmxi posts nocopydevice sizes as int: same process, so pass the pointer itself
        proxy_at<N_target>(p->target_proxy, p->target_ci).receive_ipc(
            p->node_id, p->input_index, p->inp_name, N_target, p->region_data, p->byte_size,
            CkMyNode(), -1, reinterpret_cast<int64_t>(p->send_buf), 0, nullptr, cb);
    } else if (p->byte_size > ct_msg_max_bytes()) {
        // another process, no IPC block or pinned block: stream the payload as fragments
        // through one reusable pinned chunk (marshalling copies each fragment at the call)
        auto target = proxy_at<N_target>(p->target_proxy, p->target_ci);
        std::vector<int> meta =
            ct_data_meta(p->input_index, p->inp_name, N_target, p->region_data);
        int64_t chunk = std::min<int64_t>(ct_msg_max_bytes(), int64_t(64) << 20);
        char* pin = static_cast<char*>(pinned_alloc(chunk));
        std::unique_ptr<char[]> pageable;
        if (!pin) {
            pageable.reset(new char[chunk]);
            pin = pageable.get();
        }
        int64_t seq = ct_next_frag_seq();
        for (int64_t off = 0; off < p->byte_size; off += chunk) {
            int64_t n = std::min(chunk, p->byte_size - off);
            hapiCheck(cudaMemcpy(pin, static_cast<char*>(p->send_buf) + off, n,
                                 cudaMemcpyDeviceToHost));
            target.receive_frag(p->node_id, MSG_STAGED, CkMyPe(), seq, (int)meta.size(),
                                meta.data(), p->byte_size, off, n, pin);
        }
        if (!pageable)
            pinned_free(pin);
        p->partition->send_complete(p->send_id);
    } else {
        proxy_at<N_target>(p->target_proxy, p->target_ci).receive_data(
            p->node_id, p->input_index, p->inp_name, N_target, p->region_data, p->byte_size,
            CkDeviceBuffer(reinterpret_cast<char*>(p->send_buf), cb));
    }
    delete p;
}

#endif

#ifdef USE_NVIDIA
/// Copy a packed send to pinned host memory (or leave it to CkDeviceBuffer), then send it
/// from the stream callback.
template <int N_sender, int N_target>
static void staged_send_start(DeferredSendParam<N_sender, N_target>* ds) {
    cudaStream_t st = ds->partition->comm_stream_raw;
    if ((ds->host_buf = pinned_alloc(ds->byte_size)))
        hapiCheck(cudaMemcpyAsync(ds->host_buf, ds->send_buf, ds->byte_size,
                                  cudaMemcpyDeviceToHost, st));
    CkCallback hapi_cb(deferred_send_cb<N_sender, N_target>, ds);
    hapiAddCallback(st, &hapi_cb);
}

/// Copy a packed send into IPC block `buf`, then send it from the stream callback.
template <int N_sender, int N_target>
static void ipc_send_start(DeferredSendParam<N_sender, N_target>* ds, void* buf, int slab,
                           int64_t offset) {
    ds->ipc_buf = buf;
    ds->ipc_slab = slab;
    ds->ipc_offset = offset;
    cudaStream_t st = ds->partition->comm_stream_raw;
    hapiCheck(cudaMemcpyAsync(buf, ds->send_buf, ds->byte_size, cudaMemcpyDeviceToDevice, st));
    ds->partition->pending_sends[ds->send_id] = buf;
    CkCallback hapi_cb(deferred_send_cb<N_sender, N_target>, ds);
    hapiAddCallback(st, &hapi_cb);
}
#endif

#ifdef USE_KOKKOS
/// Helper: complete a device-side pack-and-send after a parallel_for on comm_stream.
/// On NVIDIA: defers the send via hapiAddCallback on comm_stream.
/// On non-NVIDIA Kokkos: fences then sends synchronously.
template <int N_sender, int N_target>
static void device_pack_send(PartitionImpl<N_sender>* partition,
                             typename PartitionTraits<N_target>::ProxyType& target_proxy,
                             const ChareIndex<N_target>& target_ci, int node_id, int input_index,
                             int name, int* region_data, int64_t byte_size, void* send_buf) {
    auto* ds = new DeferredSendParam<N_sender, N_target>(partition, target_proxy);
    ds->node_id = node_id;
    ds->input_index = input_index;
    ds->inp_name = name;
    ds->target_ci = target_ci;
    memcpy(ds->region_data, region_data, sizeof(int) * N_target * 3);
    ds->byte_size = byte_size;
    ds->send_buf = send_buf;
    // The id travels as a CMK_REFNUM_TYPE refnum: skip ids of sends still in flight.
    auto& pending = partition->pending_sends;
    if (pending.size() > std::numeric_limits<CMK_REFNUM_TYPE>::max())
        CkAbort("device_pack_send: too many sends in flight");
    do
        ds->send_id = static_cast<CMK_REFNUM_TYPE>(partition->next_send_id++);
    while (pending.count(ds->send_id));
    pending[ds->send_id] = send_buf;
#ifdef USE_NVIDIA
    auto target = proxy_at<N_target>(target_proxy, target_ci);
    // partitions do not migrate: lastKnown is the target's PE
    int target_pe = target.ckLocalBranch()->lastKnown(target.ckGetIndex());
    if (CmiNodeOf(target_pe) != CkMyNode()) {
        ds->local = false;
        if (gpu_ipc_peer(target_pe)) {
            int slab;
            int64_t offset;
            if (void* buf = gpu_ipc_alloc(byte_size, slab, offset, true)) {
                ipc_send_start<N_sender, N_target>(ds, buf, slab, offset);
                return;
            }
            // arena full, or earlier sends waiting: queue behind them rather than stage
            if (gpu_ipc_wait(
                    byte_size, partition,
                    [ds](void* buf, int slab, int64_t offset) {
                        if (buf)
                            ipc_send_start<N_sender, N_target>(ds, buf, slab, offset);
                        else
                            staged_send_start<N_sender, N_target>(ds);
                    },
                    [ds] { delete ds; }))
                return;
        }
        if ((ds->host_buf = pinned_alloc(byte_size))) {
            hapiCheck(cudaMemcpyAsync(ds->host_buf, send_buf, byte_size, cudaMemcpyDeviceToHost,
                                      partition->comm_stream_raw));
        }
    }
    CkCallback hapi_cb(deferred_send_cb<N_sender, N_target>, ds);
    hapiAddCallback(partition->comm_stream_raw, &hapi_cb);
#else
    Kokkos::fence();
    deferred_send_cb<N_sender, N_target>(ds, nullptr);
#endif
}
#endif

/// Dispatch a JIT-compiled kernel with the given packed memref descriptors
/// and optional broadcast scalar arguments.
///
/// Function signature layout (matching JIT buildFromAST):
///   [n_input_memrefs memref args] [n_scalars scalar args] [n_output_memrefs memref args]
///
/// @param n_input_memrefs  Number of input memrefs (descs[0..n_input_memrefs-1])
/// @param scalar_args      Pointers to scalar values (broadcast leaves), inserted
///                         between input and output memref args
/// @param descs            All memref descriptors: inputs first, then outputs
template <int N, typename T = float>
static void dispatch_kernel(void* func_ptr, int n_memrefs, std::vector<MemRef<N, T>>& descs,
                            int n_input_memrefs = -1,
                            const std::vector<void*>& scalar_args = {}
#ifdef USE_NVIDIA
                            ,
                            cudaStream_t stream = nullptr
#endif
) {
    if (n_input_memrefs < 0)
        n_input_memrefs = n_memrefs; // backward compat: all are inputs (no outputs separate)

    constexpr int FIELDS_PER_MEMREF = MemRef<N, T>::fields_per_memref();
    int n_scalar_args = (int)scalar_args.size();

    // Pack args: input memrefs, then scalar args, then output memrefs
    std::vector<void*> args;
    args.reserve(n_memrefs * FIELDS_PER_MEMREF + n_scalar_args);

    // Input memrefs
    auto pack_memref = [&](int i) {
        args.push_back(&descs[i].allocated);
        args.push_back(&descs[i].aligned);
        args.push_back(&descs[i].offset);
        for (int d = 0; d < N; ++d)
            args.push_back(&descs[i].sizes[d]);
        for (int d = 0; d < N; ++d)
            args.push_back(&descs[i].strides[d]);
    };

    for (int i = 0; i < n_input_memrefs; ++i)
        pack_memref(i);

    // Broadcast scalar arguments
    for (int i = 0; i < n_scalar_args; ++i)
        args.push_back(scalar_args[i]);

    // Output memrefs
    for (int i = n_input_memrefs; i < n_memrefs; ++i)
        pack_memref(i);

    auto total_elements = [&]() -> int64_t {
        if (n_memrefs == 0)
            return 0;
        int64_t n = 1;
        for (int d = 0; d < N; ++d)
            n *= descs[0].sizes[d];
        return n;
    };

#if defined(USE_NVIDIA)
    {
        // func_ptr is the JIT'd host launcher (MLIRJitCompiler::generateNVIDIAHost);
        // it queues kernels on this chare's compute stream without waiting.
        jit_set_stream(stream);
        static const bool check_memrefs = getenv("CNUM_CHECK_MEMREFS") != nullptr;
        if (check_memrefs) {
            // Debug aid: each descriptor's range must lie in its device allocation.
            for (int i = 0; i < n_memrefs; ++i) {
                const auto& m = descs[i];
                int64_t lo = m.offset, hi = m.offset;
                for (int d = 0; d < N; ++d) {
                    int64_t ext = (m.sizes[d] > 0 ? m.sizes[d] - 1 : 0) * m.strides[d];
                    (ext < 0 ? lo : hi) += ext;
                }
                CUdeviceptr base = 0;
                size_t bytes = 0;
                CUresult r = cuMemGetAddressRange(
                    &base, &bytes, reinterpret_cast<CUdeviceptr>(m.aligned + lo));
                bool ok = (r == CUDA_SUCCESS) &&
                          reinterpret_cast<CUdeviceptr>(m.aligned + hi + 1) <= base + bytes;
                if (!ok) {
                    CkPrintf("[PE %d] memref %d/%d (%s) BAD: res=%d ptr=%p off=%lld sizes=", CkMyPe(),
                             i, n_memrefs, i < n_input_memrefs ? "in" : "out", (int)r,
                             (void*)m.aligned, (long long)m.offset);
                    for (int d = 0; d < N; ++d)
                        CkPrintf("%lld,", (long long)m.sizes[d]);
                    CkPrintf(" strides=");
                    for (int d = 0; d < N; ++d)
                        CkPrintf("%lld,", (long long)m.strides[d]);
                    CkPrintf(" alloc=[%p,+%zu) need=[%lld,%lld] elems\n", (void*)base, bytes,
                             (long long)lo, (long long)hi);
                }
            }
        }
        using PackedFunc = void (*)(void**);
        reinterpret_cast<PackedFunc>(func_ptr)(args.data());
    }
#elif defined(USE_AMD)
    {
        hipFunction_t hipFunc = reinterpret_cast<hipFunction_t>(func_ptr);
        int64_t n_elements = total_elements();
        int blockSize = 256;
        int gridSize = ((int)n_elements + blockSize - 1) / blockSize;
        hipModuleLaunchKernel(hipFunc, gridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args.data(),
                              nullptr);
        hipDeviceSynchronize();
    }
#elif defined(USE_INTEL)
    {
        ze_kernel_handle_t zeKernel = reinterpret_cast<ze_kernel_handle_t>(func_ptr);
        int64_t n_elements = total_elements();
        uint32_t groupSizeX = 256;
        zeKernelSetGroupSize(zeKernel, groupSizeX, 1, 1);

        uint32_t argIdx = 0;
        // Input memrefs
        for (int i = 0; i < n_input_memrefs; ++i) {
            zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(T*), &descs[i].allocated);
            zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(T*), &descs[i].aligned);
            zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(int64_t), &descs[i].offset);
            for (int d = 0; d < N; ++d)
                zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(int64_t), &descs[i].sizes[d]);
            for (int d = 0; d < N; ++d)
                zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(int64_t), &descs[i].strides[d]);
        }
        // Broadcast scalars
        for (int i = 0; i < n_scalar_args; ++i)
            zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(T), scalar_args[i]);
        // Output memrefs
        for (int i = n_input_memrefs; i < n_memrefs; ++i) {
            zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(T*), &descs[i].allocated);
            zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(T*), &descs[i].aligned);
            zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(int64_t), &descs[i].offset);
            for (int d = 0; d < N; ++d)
                zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(int64_t), &descs[i].sizes[d]);
            for (int d = 0; d < N; ++d)
                zeKernelSetArgumentValue(zeKernel, argIdx++, sizeof(int64_t), &descs[i].strides[d]);
        }

        ze_group_count_t dispatchArgs = {(uint32_t)((n_elements + groupSizeX - 1) / groupSizeX), 1,
                                         1};

        ze_result_t res = zeInit(ZE_INIT_FLAG_GPU_ONLY);
        uint32_t driverCount = 1;
        ze_driver_handle_t driver;
        zeDriverGet(&driverCount, &driver);
        uint32_t deviceCount = 1;
        ze_device_handle_t device;
        zeDeviceGet(driver, &deviceCount, &device);
        ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
        ze_context_handle_t zeContext;
        zeContextCreate(driver, &ctxDesc, &zeContext);

        ze_command_queue_desc_t cmdQueueDesc = {};
        cmdQueueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
        ze_command_list_handle_t cmdList;
        zeCommandListCreateImmediate(zeContext, device, &cmdQueueDesc, &cmdList);

        zeCommandListAppendLaunchKernel(cmdList, zeKernel, &dispatchArgs, nullptr, 0, nullptr);
        zeCommandListDestroy(cmdList);
        zeContextDestroy(zeContext);
    }
#else
    using PackedFunc = void (*)(void**);
    reinterpret_cast<PackedFunc>(func_ptr)(args.data());
#endif
}
