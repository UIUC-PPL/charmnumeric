#pragma once

#include "backend.hpp"

// ---- Stream policy helpers (USE_KOKKOS) ----
// Under USE_NVIDIA: Kokkos::Cuda exec space instances bound to per-chare streams.
// Under USE_KOKKOS without USE_NVIDIA: default execution space + Kokkos::fence().

#ifdef USE_KOKKOS
#ifdef USE_NVIDIA
#define CT_COMPUTE_POLICY(partition, n)                                                            \
    Kokkos::RangePolicy<Kokkos::Cuda>((partition)->compute_exec, 0, (n))
#define CT_COMM_POLICY(partition, n)                                                               \
    Kokkos::RangePolicy<Kokkos::Cuda>((partition)->comm_exec, 0, (n))
#else
#define CT_COMPUTE_POLICY(partition, n) Kokkos::RangePolicy<>(0, (n))
#define CT_COMM_POLICY(partition, n) Kokkos::RangePolicy<>(0, (n))
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
    bool has_comm;
};

/// C callback invoked by HAPI when all prior compute_stream work completes.
template <int N>
static void compute_done_cb(void* param, void* msg) {
    auto* p = static_cast<ComputeDoneParam<N>*>(param);
    if (p->has_comm) {
        auto it = p->partition->executor->pending.find(p->node_id);
        if (it != p->partition->executor->pending.end()) {
            it->second.clear_remote_buffers();
            p->partition->executor->pending.erase(it);
        }
    }
    p->partition->executor->node_finished(p->node_id);
    delete p;
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
    p->partition->pending_sends[p->send_id] = p->send_buf;
    ChareIndex<N_sender> self_ci;
    for (int d = 0; d < N_sender; ++d)
        self_ci.idx[d] = p->partition->index[d];
    // The source callback fires on the receiver with an empty message, so the
    // send_id rides along as the callback's refnum.
    CkCallback cb(PartitionTraits<N_sender>::CkIndexType::send_complete(nullptr),
                  proxy_at<N_sender>(p->partition->thisProxy, self_ci));
    cb.setRefnum(static_cast<CMK_REFNUM_TYPE>(p->send_id));
    proxy_at<N_target>(p->target_proxy, p->target_ci).receive_data(
        p->node_id, p->input_index, p->inp_name, N_target, p->region_data, p->byte_size,
        CkDeviceBuffer(reinterpret_cast<char*>(p->send_buf), cb));
    delete p;
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
    ds->send_id = static_cast<CMK_REFNUM_TYPE>(partition->next_send_id++);
#ifdef USE_NVIDIA
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
        // func_ptr is the JIT'd host launcher (see MLIRJitCompiler::generateNVIDIAHost):
        // it takes the same packed arguments as the CPU path and launches the
        // kernels synchronously on its own stream, so first drain this chare's
        // stream (inputs may still be in flight there).
        cudaStreamSynchronize(stream);
        static const bool check_memrefs = getenv("CNUM_CHECK_MEMREFS") != nullptr;
        if (check_memrefs) {
            // Debug aid: verify each descriptor's accessed range lies inside the
            // device allocation its pointer belongs to.
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
