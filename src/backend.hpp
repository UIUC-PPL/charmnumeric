#pragma once

#include <functional>
#include <mutex>
#include <deque>
#include <map>

#include "array_region.hpp"
#include "opcodes.hpp"
#include <charmtyles/core/charmtyles.hpp>
#include <charmtyles/core/server.hpp>

#include "charmnumeric.decl.h"

#ifdef USE_KOKKOS
#define CT_MIN_TILE_1D 1048576
#define CT_MIN_TILE_2D 1024
#define CT_MIN_TILE_3D 128
#else
#define CT_MIN_TILE_1D 262144
#define CT_MIN_TILE_2D 512
#define CT_MIN_TILE_3D 64
#endif


// ---- Partition traits: map N -> concrete Charm++ chare/proxy/index types ----

template <int N>
struct PartitionTraits;

template <>
struct PartitionTraits<1> {
    using ProxyType = CProxy_Partition1D;
    using CkIndexType = CkIndex_Partition1D;
};
template <>
struct PartitionTraits<2> {
    using ProxyType = CProxy_Partition2D;
    using CkIndexType = CkIndex_Partition2D;
};
template <>
struct PartitionTraits<3> {
    using ProxyType = CProxy_Partition3D;
    using CkIndexType = CkIndex_Partition3D;
};

/// Access an element of an N-D chare array proxy by ChareIndex<N>.
template <int N>
inline auto proxy_at(typename PartitionTraits<N>::ProxyType& proxy, const ChareIndex<N>& ci);

template <>
inline auto proxy_at<1>(CProxy_Partition1D& proxy, const ChareIndex<1>& ci) {
    return proxy[ci.idx[0]];
}
template <>
inline auto proxy_at<2>(CProxy_Partition2D& proxy, const ChareIndex<2>& ci) {
    return proxy(ci.idx[0], ci.idx[1]);
}
template <>
inline auto proxy_at<3>(CProxy_Partition3D& proxy, const ChareIndex<3>& ci) {
    return proxy(ci.idx[0], ci.idx[1], ci.idx[2]);
}

/// Extent-1 dim of a dimension-dropped 3-D matrix region (rows/cols are the other two, in
/// order) that gives `want_rows` x `want_cols` (-1: any); -1 if no dim has extent 1.
inline int drop_dim_3d(const ArrayRegion<3>& r, int want_rows = -1, int want_cols = -1) {
    int first = -1;
    for (int dd = 0; dd < 3; ++dd) {
        if (r.stop[dd] - r.start[dd] != 1)
            continue;
        int rd = dd == 0 ? 1 : 0, cd = dd <= 1 ? 2 : 1;
        if (first < 0)
            first = dd;
        if ((want_rows < 0 || r.stop[rd] - r.start[rd] == want_rows) &&
            (want_cols < 0 || r.stop[cd] - r.start[cd] == want_cols))
            return dd;
    }
    return first;
}

/// (rows, cols) of a MATMUL/MATMATMUL operand region; a vector is one column.
inline std::array<int, 2> matmul_region_extents(Region* rb) {
    if (auto* r = dynamic_cast<ArrayRegion<1>*>(rb))
        return {r->stop[0] - r->start[0], 1};
    if (auto* r = dynamic_cast<ArrayRegion<2>*>(rb))
        return {r->stop[0] - r->start[0], r->stop[1] - r->start[1]};
    if (auto* r = dynamic_cast<ArrayRegion<3>*>(rb)) {
        int dd = drop_dim_3d(*r);
        if (dd >= 0) {
            int rd = dd == 0 ? 1 : 0, cd = dd <= 1 ? 2 : 1;
            return {r->stop[rd] - r->start[rd], r->stop[cd] - r->start[cd]};
        }
    }
    return {-1, -1};
}

/// Dropped dim of 3-D operand `idx` of a MATMUL/MATMATMUL root. When several dims have
/// extent 1, the one matching the other operand (A's cols == B's rows / vector length).
inline int matmul_drop_dim(ASTNode* root, int idx) {
    auto* r = static_cast<ArrayRegion<3>*>(root->operand_regions[idx]);
    int want_rows = -1, want_cols = -1;
    if (root->operand_regions.size() >= 2) {
        auto other = matmul_region_extents(root->operand_regions[1 - idx]);
        if (idx == 0)
            want_cols = other[0];
        else
            want_rows = other[1];
    }
    return drop_dim_3d(*r, want_rows, want_cols);
}

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostSpace = Kokkos::HostSpace;
#ifdef USE_NVIDIA
#include "hapi.h"
#include <cuda_runtime.h>
#endif

/// Device message buffers, cached by power-of-two size class: cudaMalloc/cudaFree per
/// message cost tens of us, and cudaFree also synchronizes the device. Only for buffers
/// freed after their last GPU use (send_complete, compute-done callbacks).
struct CommPool {
    static constexpr int kMinClass = 8, kMaxClass = 26; // 256 B .. 64 MB
    static constexpr int64_t kMaxCached = int64_t(1) << 30;
    std::unordered_map<void*, int> live; // pooled block -> size class
    std::vector<void*> free_lists[kMaxClass + 1];
    int64_t cached = 0;
    std::mutex mu; // shared by the PEs of a process (SMP)
};

inline CommPool& comm_pool() {
    static CommPool pool;
    return pool;
}

inline void* comm_alloc(int64_t bytes) {
    int cls = CommPool::kMinClass;
    while (cls <= CommPool::kMaxClass && (int64_t(1) << cls) < bytes)
        ++cls;
    if (cls > CommPool::kMaxClass)
        return Kokkos::kokkos_malloc<DeviceSpace>(bytes);
    auto& pool = comm_pool();
    std::lock_guard<std::mutex> lock(pool.mu);
    void* ptr;
    if (!pool.free_lists[cls].empty()) {
        ptr = pool.free_lists[cls].back();
        pool.free_lists[cls].pop_back();
        pool.cached -= int64_t(1) << cls;
    } else {
        ptr = Kokkos::kokkos_malloc<DeviceSpace>(size_t(1) << cls);
    }
    pool.live[ptr] = cls;
    return ptr;
}

inline void comm_free(void* ptr) {
    auto& pool = comm_pool();
    std::lock_guard<std::mutex> lock(pool.mu);
    auto it = pool.live.find(ptr);
    if (it == pool.live.end()) {
        Kokkos::kokkos_free<DeviceSpace>(ptr);
        return;
    }
    int cls = it->second;
    pool.live.erase(it);
    if (pool.cached + (int64_t(1) << cls) > CommPool::kMaxCached) {
        Kokkos::kokkos_free<DeviceSpace>(ptr);
        return;
    }
    pool.free_lists[cls].push_back(ptr);
    pool.cached += int64_t(1) << cls;
}
#endif

#include "dtypes.hpp"

/// Type-erased base class for N-D arrays.
template <int N>
class CTArrayBase {
  public:
    int name;
    ArrayRegion<N> region;
    ArrayDecomp<N> decomp;
    std::array<int, N> global_shape;
    int global_size; // total elements across all dimensions
    bool owner;
    DType dtype;

    virtual ~CTArrayBase() = default;
    int64_t local_size() const { return region.size(); }
    virtual void copyToHost() = 0;
    virtual void copyToDevice() = 0;
    virtual void* data_ptr() = 0;        // host pointer
    virtual void* device_data_ptr() = 0; // device pointer (or host if no GPU)
    virtual int elem_size() const = 0;
};

template <int N, typename T = float>
class Array : public CTArrayBase<N> {
  public:
#ifdef USE_KOKKOS
    Kokkos::View<T*, DeviceSpace> d_view;
    typename Kokkos::View<T*, DeviceSpace>::HostMirror h_view; // zeroed, allocated on first host use
#else
    T* data; // host pointer (always allocated, used for gather/get)
#endif

    /// Construct from a local region + global shape + decomposition.
    Array(ArrayRegion<N> region_, std::array<int, N> global_shape_, int name_,
          ArrayDecomp<N> decomp_) {
        this->name = name_;
        this->region = region_;
        this->global_shape = global_shape_;
        this->decomp = decomp_;
        this->owner = true;
        this->dtype = dtype_of<T>();
        this->global_size = 1;
        for (int d = 0; d < N; ++d)
            this->global_size *= global_shape_[d];
        int n = this->region.size();
#ifdef USE_KOKKOS
        d_view = Kokkos::View<T*, DeviceSpace>("array_device", n);
#else
        data = new T[n];
        for (int i = 0; i < n; i++)
            data[i] = T(0);
#endif
    }

    /// Non-owning constructor (wraps existing buffer).
    Array(T* data_, int name_, int size_) {
        this->name = name_;
        this->owner = false;
        this->dtype = dtype_of<T>();
        this->global_size = size_;
#ifdef USE_KOKKOS
        h_view = Kokkos::View<T*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(data_, size_);
        // No device allocation for non-owning arrays
#else
        data = data_;
        for (int i = 0; i < size_; i++)
            data[i] = T(0);
#endif
    }

    ~Array() override {
#ifndef USE_KOKKOS
        if (data != NULL && this->owner)
            delete[] data;
#endif
        // Kokkos Views are RAII — automatic cleanup
    }

#ifdef USE_KOKKOS
    void ensure_host() {
        if (h_view.data() == nullptr && d_view.extent(0) > 0)
            h_view = Kokkos::create_mirror_view(HostSpace{}, d_view);
    }
#endif

    void* data_ptr() override {
#ifdef USE_KOKKOS
        ensure_host();
        return h_view.data();
#else
        return data;
#endif
    }

    void* device_data_ptr() override {
#ifdef USE_KOKKOS
        return d_view.data();
#else
        return data; // CPU-only: host pointer is the "device" pointer
#endif
    }

    int elem_size() const override { return sizeof(T); }

    void copyToHost() override {
#ifdef USE_KOKKOS
        ensure_host();
        Kokkos::deep_copy(h_view, d_view);
#endif
    }

    void copyToDevice() override {
#ifdef USE_KOKKOS
        ensure_host();
        Kokkos::deep_copy(d_view, h_view);
#endif
    }
};

// ---- Communication Structures (templated on dimensionality only) ----

template <int N>
struct RemoteBuffer {
    char* data; // host pointer (CPU) or device pointer (USE_KOKKOS)
    ArrayRegion<N> region;
    int64_t byte_size;
};

template <int N>
struct PendingComm {
    DAGNode* node;
    int expected_msgs;
    std::vector<ArrayRegion<N>> my_inputs;
    std::unordered_map<int, std::vector<RemoteBuffer<N>>> remote_buffers;
    bool incremental = false; // When true, compute as panels arrive

    void clear_remote_buffers() {
#ifdef USE_KOKKOS
        // pooled buffers are reused at once; kernels on the default instance may still read them
        if (!remote_buffers.empty())
            Kokkos::DefaultExecutionSpace().fence("ct_remote_buffers");
#endif
        for (auto& [idx, bufs] : remote_buffers)
            for (auto& rb : bufs) {
#ifdef USE_KOKKOS
                comm_free(rb.data);
#else
                delete[] rb.data;
#endif
            }
        remote_buffers.clear();
    }
};

// ---- Forward declarations ----

template <int N>
class PartitionImpl;

/// State of a REDUCTION/BINCOUNT node on one chare (executor_reduction.cpp).
/// Messages that arrive before the chare reaches the node wait in `stash`.
template <int N>
struct ReductionState {
    DAGNode* node = nullptr;
    bool started = false;
    bool receiving = false; // owns part of the result
    bool sending = false;   // BINCOUNT: waiting for weights moved onto x's layout
    std::vector<std::vector<char>> stash;
    ArrayRegion<N> mine;    // output elements owned by this chare (output array coords)
    std::vector<char> acc;  // one accumulator per element of `mine`
    int64_t covered = 0;    // input elements folded in, summed over `mine`
    int64_t expected = 0;   // mine.size() * reduced extent
    std::vector<double> weights; // BINCOUNT weights for x's block, by view index - w_base
    int64_t w_base = 0;
    int64_t w_got = 0;
};

/// REDUCE (dot) state on chare 0; partials are summed as they arrive.
struct DotState {
    std::vector<char> acc;   // ReduceContrib: metadata of the first partial, summed value
    int got = 0;
    int expected = 0;
    bool started = false;
};

/// LINALG state on one result chare; early messages wait in `stash`.
template <int N>
struct LinalgState {
    DAGNode* node = nullptr;
    bool started = false;
    bool owner = false;
    bool root = false;     // chare 0: gathers the operands and factors them
    bool computed = false;
    std::vector<std::vector<char>> stash;
    std::vector<std::vector<char>> inputs; // gathered operand blocks (root)
    ArrayRegion<N> mine;
    int64_t in_covered = 0, in_expected = 0;
    int64_t out_covered = 0, out_expected = 0;
};

/// GATHER/SCATTER/EXPAND state per chare; early messages wait in `stash`.
template <int N>
struct IndexState {
    DAGNode* node = nullptr;
    bool started = false;
    std::vector<std::vector<char>> stash;
    bool holder = false, flushed = false;
    int req_left = 0;  // holder: request lists still expected
    int val_left = 0;  // target: value lists still expected
    int seg_left = 0;  // EXPAND coordinator: segment lists still expected
    bool wait_offsets = false; // EXPAND driver: segment offsets not yet received
    std::unordered_map<int64_t, std::vector<int64_t>> outgoing; // holder: value lists per target
    std::vector<int64_t> updates; // target: received entries
    int upd_width = 0;
    std::vector<std::vector<char>> segs; // coordinator: received segment lists
    std::vector<int64_t> counts, seg_start; // EXPAND driver: element counts, row starts
    int64_t row = 1;
    // COMPRESS / MASK_SCATTER
    bool mx_pending = false; // this chare holds a block of x that is not done yet
    bool mx_scanned = false;
    int64_t m_covered = 0, m_expected = 0; // mask elements moved onto x's block
    std::vector<uint8_t> mblock;           // that mask block (row-major)
    std::vector<int64_t> rowpos;           // compacted offset of each block row, then the total
    char* cbuf = nullptr;                  // compacted values (device memory in GPU builds)
    int64_t out_covered = 0, out_expected = 0;
    std::vector<std::vector<char>> reqs;   // v holder: requests, served once all arrived
};

/// RESHAPE/BROADCAST state per result chare; early messages wait in `stash`.
template <int N>
struct ShapeState {
    DAGNode* node = nullptr;
    bool started = false;
    std::vector<std::vector<char>> stash;
    int64_t covered = 0;
    int64_t expected = 0;
    std::vector<char> block; // BROADCAST: the source block this chare expands
    char* d_block = nullptr; // its device copy (GPU builds)
};

// ---- ArrayDAGGroup ----

/// GET of this array name: a barrier, answered (with no data) once every chare
/// of every partition has finished all earlier epochs.
constexpr int kBarrierName = -1;

class ArrayDAGGroup : public CBase_ArrayDAGGroup {
  public:
    // Per-ndims partition state: current chare grid and proxy.
    struct PartitionGrid {
        int grid[3] = {0, 0, 0};
        int start_epoch = -1; // first epoch this partition was expanded
    };
    std::unordered_map<int, PartitionGrid> partition_grid;
    CProxy_Partition1D partition_proxy_1;
    CProxy_Partition2D partition_proxy_2;
    CProxy_Partition3D partition_proxy_3;
    std::unordered_map<int64_t, void*> compile_cache;
    std::unordered_map<int64_t, void*> module_cache;
    std::unordered_map<int, char*> gather_buffers;
    std::unordered_map<int, int64_t> gather_counts; // bytes gathered so far
    std::unordered_map<int, int64_t> gather_total;  // total bytes expected
    struct GatherFragment {
        int64_t offset;
        int64_t size;
        char* data; // owned copy (bytes)
    };
    std::unordered_map<int, std::vector<GatherFragment>> gather_early;
    // Barrier GETs (name kBarrierName): the chare grids they wait for, and on
    // PE 0 the number of chares still to arrive, per epoch
    std::unordered_map<int, std::unordered_map<int, PartitionGrid>> barrier_grids;
    std::unordered_map<int, int64_t> barrier_pending;

    // Global metadata for arrays referenced by the current DAG.
    // Maps array name -> metadata (ndims, global_shape, decomposition).
    struct ArrayMetadata {
        int ndims;
        std::array<int, 3> global_shape; // padded to 3 dims
        std::array<int, 3> offset;       // decomposition offset (padded to 3 dims)
        int tile;                        // tile size
        bool decomp_final = false;       // true once offset has been computed and must not change

        /// Construct an ArrayDecomp<N> from the stored metadata.
        template <int N>
        ArrayDecomp<N> decomp() const {
            std::array<int, N> shape{}, off{};
            for (int d = 0; d < N; ++d) {
                shape[d] = global_shape[d];
                off[d] = offset[d];
            }
            return ArrayDecomp<N>::offset_decomp(shape, off, tile);
        }
    };
    std::unordered_map<int, ArrayMetadata> array_meta;
    // Physical decomposition of arrays that have already been materialized.
    // This persists across epochs so later DAGs do not reinterpret old data
    // using freshly recomputed metadata unless the array is recreated.
    std::unordered_map<int, ArrayMetadata> live_array_meta;

    // UPLOAD payloads (held by PE 0 only) cut into per-chare tiles by stage_uploads() and
    // sent to each chare's home PE, keyed by (array name, chare index). A chare takes (and
    // frees) its tile when it executes the UPLOAD node, fetching it if it lives elsewhere.
    struct UploadTile {
        std::vector<char> data;
        ArrayMetadata meta;
    };
    std::map<std::array<int, 4>, UploadTile> upload_tiles;
    std::map<std::array<int, 4>, std::pair<int, int>> upload_fetches; // key -> (node, ndims)
    void stage_uploads(DAG* dag, int epoch);
    void stage_upload_tile(int name, int nd, int i0, int i1, int i2, int nmeta, int* meta,
                           int64_t size, char* data);
    void serve_upload_fetch(const std::array<int, 4>& key, int nd);
    void fetch_upload_tile(int node_id, int name, int nd, int i0, int i1, int i2);
    void send_upload_tile(const std::array<int, 4>& key, int nd, int node_id,
                          std::vector<char>& data);

    ArrayDAGGroup();
    ArrayDAGGroup(CkMigrateMessage* m) {}
    ~ArrayDAGGroup();

    // Broadcast partition proxies from PE 0 to all PEs
    void set_proxies(CkArrayID p1, CkArrayID p2, CkArrayID p3);
    // Reduction target: all PEs have proxies, register CCS handlers
    void proxies_ready();

    // Unified DAG handling — dispatches to all relevant partition types
    void receive_dag(int epoch, int size, char* serialized_dag);
    void receive_get_request(int ndims, int epoch, int name, int size, int dtype = 0);
    void receive_dag_now(int epoch, int size, char* serialized_dag);
    void receive_get_request_now(int ndims, int epoch, int name, int size, int dtype);
    // Broadcasts from PE 0 may arrive out of order (e.g. a small GET overtaking a large
    // DAG); epochs are processed in client order, later ones wait here.
    int next_epoch = 0;
    std::map<int, std::function<void()>> early_epochs;
    void epoch_arrived(int epoch, std::function<void()> run);
    void gather(int epoch, int name, int64_t offset, int64_t size, char* data);
    void barrier_arrive(int epoch);

    // Templated execute_node — dispatched by dtype at call site
    template <int N, typename T = float>
    void execute_node_nd(DAGNode* node, PartitionImpl<N>* partition, PendingComm<N>* comm = nullptr);

    // Determine the decomposition for each array in the DAG.
    // The runtime may later override entries with live_array_meta for arrays
    // that already exist physically from an earlier epoch.
    void compute_decompositions(DAG* dag);

    // Shared
    void* compile_node(DAGNode* node);
    void compile(DAG* dag);
};

/// Result ndims and shape of a REDUCTION/BINCOUNT root (executor_reduction.cpp).
int reduction_result_meta(ASTNode* root,
                          const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& meta,
                          std::array<int, 3>& shape_out);

/// Result ndims and shape of a LINALG root (executor_linalg.cpp).
int linalg_result_meta(ASTNode* root,
                       const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& meta,
                       std::array<int, 3>& shape_out);
/// Result ndims/shape of a GATHER/EXPAND root (0 for SCATTER), and the ranks of
/// the partitions taking part in an index node (executor_indexing.cpp).
int index_result_meta(ASTNode* root,
                      const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& meta,
                      std::array<int, 3>& shape_out);
std::vector<int> index_op_ndims(ASTNode* root);

// ---- Unified Executor (templated on N only) ----

template <int N>
class ArrayDAGExecutorND : public DAGExecutor {
  private:
    PartitionImpl<N>* partition;

  public:
    std::unordered_map<int, PendingComm<N>> pending;
    std::unordered_map<int, ReductionState<N>> red_state;
    std::unordered_map<int, DotState> dot_state;
    std::unordered_map<int, LinalgState<N>> la_state;
    std::unordered_map<int, IndexState<N>> ix_state;
    std::unordered_map<int, ShapeState<N>> shape_state;
    std::unordered_map<int, std::pair<DAGNode*, int>> upload_wait; // node -> tiles still fetched

    ArrayDAGExecutorND(ArrayDAGGroup* group_, PartitionImpl<N>* partition_)
        : DAGExecutor(group_, N), partition(partition_) {}
    virtual ~ArrayDAGExecutorND() = default;

    void execute_dag_node(DAGNode* node) override;
    void delete_array(int name) override;
    void execute_matmul_node(DAGNode* node);
    void execute_matmatmul_node(DAGNode* node);
    void execute_reduce_node(DAGNode* node);
    void on_dot_partial(int node_id, char* data, int64_t size);
    void finish_dot(int node_id);
    void execute_cross_set_region_node(DAGNode* node);
    void execute_diag_node(DAGNode* node);
    void execute_tile_node(DAGNode* node);
    void execute_reduction_node(DAGNode* node);
    void on_reduce_partial(int node_id, char* data, int64_t size);
    void execute_linalg_node(DAGNode* node);
    void on_linalg_msg(int node_id, char* data, int64_t size);
    void execute_creation_node(DAGNode* node);
    void on_upload_tile(int node_id, int name, char* data, int64_t size);
    void finish_creation_node(int node_id);
    void execute_transpose_node(DAGNode* node);
    void execute_shape_node(DAGNode* node);
    void on_shape_msg(int node_id, char* data, int64_t size);
    void execute_castscansort_node(DAGNode* node);
    void finish_castscansort_node(PendingComm<N>& comm);
    void execute_signal_node(DAGNode* node);
    void finish_signal_node(PendingComm<N>& comm);
    void execute_index_node(DAGNode* node);
    void on_index_msg(int node_id, char* data, int64_t size);
    void index_progress(int node_id);
    void execute_mask_node(DAGNode* node);
    void mask_progress(int node_id);
    void on_comm_done(int node_id);
    void on_matmatmul_receive(int node_id, int input_index);
    void on_matmul_partial(int node_id, RemoteBuffer<N>& partial_buf);
    void matmul_check_finalize(int node_id);
    int ast_visitor(ASTNode* node, DType dtype);
};

// ---- PartitionImpl: all partition logic, independent of Charm++ chare base ----

template <int N>
class PartitionImpl {
    friend class ArrayDAGExecutorND<N>;

  public:
    ArrayDAGExecutorND<N>* executor;
    typename PartitionTraits<N>::ProxyType thisProxy;
    CProxy_ArrayDAGGroup dag_proxy;
    std::unordered_map<int, CTArrayBase<N>*> arrays; // mixed-type array storage
    std::array<int, N> index;                        // N-D chare index (set by wrapper)

#ifndef NDEBUG
    int64_t comm_bytes_sent = 0; // cumulative communication volume (bytes) sent from this chare
#endif

    // Free list for array reuse: retired arrays keyed by exact buffer metadata.
    // A buffer is only reusable when dtype, shape, and decomposition match.
    static constexpr int FREE_LIST_MAX = 4; // max entries per key
    struct FreeKey {
        DType dtype;
        std::array<int, N> local_shape;
        std::array<int, N> global_shape;
        std::array<int, N> decomp_offset;
        std::array<int, N> decomp_global_shape;
        int decomp_tile;

        bool operator==(const FreeKey& o) const {
            return dtype == o.dtype &&
                   local_shape == o.local_shape &&
                   global_shape == o.global_shape &&
                   decomp_offset == o.decomp_offset &&
                   decomp_global_shape == o.decomp_global_shape &&
                   decomp_tile == o.decomp_tile;
        }
    };
    struct FreeKeyHash {
        static inline void hash_combine(std::size_t& seed, std::size_t value) {
            seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        static inline void hash_int_array(std::size_t& seed, const std::array<int, N>& values) {
            for (int value : values)
                hash_combine(seed, std::hash<int>()(value));
        }

        std::size_t operator()(const FreeKey& k) const {
            std::size_t seed = std::hash<int>()(static_cast<int>(k.dtype));
            hash_int_array(seed, k.local_shape);
            hash_int_array(seed, k.global_shape);
            hash_int_array(seed, k.decomp_offset);
            hash_int_array(seed, k.decomp_global_shape);
            hash_combine(seed, std::hash<int>()(k.decomp_tile));
            return seed;
        }
    };
    std::unordered_map<FreeKey, std::vector<CTArrayBase<N>*>, FreeKeyHash> free_arrays;

    static FreeKey make_free_key(DType dtype,
                                 const ArrayRegion<N>& region,
                                 const std::array<int, N>& global_shape,
                                 const ArrayDecomp<N>& decomp) {
        std::array<int, N> local_shape{};
        for (int d = 0; d < N; ++d)
            local_shape[d] = region.size(d);
        return FreeKey{
            dtype,
            local_shape,
            global_shape,
            decomp.offset,
            decomp.global_shape,
            decomp.tile,
        };
    }

    /// Move an array from `arrays` into the free list (or delete it if the
    /// free list for its key is full).
    void retire_array(int name) {
        auto it = arrays.find(name);
        if (it == arrays.end()) {
            DBG_PRINT("[PE %d] Partition<%d> retire_array name=%d: no local buffer\n",
                      CkMyPe(), N, name);
            return;
        }
        CTArrayBase<N>* arr = it->second;
        arrays.erase(it);
        FreeKey key = make_free_key(arr->dtype, arr->region, arr->global_shape, arr->decomp);
        auto& bucket = free_arrays[key];
        if (static_cast<int>(bucket.size()) < FREE_LIST_MAX) {
            bucket.push_back(arr);
            DBG_PRINT("[PE %d] Partition<%d> retire_array name=%d: moved to freelist "
                      "(dtype=%d local_size=%lld bucket=%d/%d)\n",
                      CkMyPe(), N, name, static_cast<int>(arr->dtype), (long long)arr->local_size(),
                      static_cast<int>(bucket.size()), FREE_LIST_MAX);
        } else {
            DBG_PRINT("[PE %d] Partition<%d> retire_array name=%d: deleting local buffer "
                      "(dtype=%d local_size=%lld freelist_full=%d)\n",
                      CkMyPe(), N, name, static_cast<int>(arr->dtype), (long long)arr->local_size(),
                      FREE_LIST_MAX);
            delete arr;
        }
    }

    /// Try to pop a reusable buffer from the free list.
    /// Returns nullptr if none available.
    CTArrayBase<N>* try_reuse(DType dtype,
                              const ArrayRegion<N>& region,
                              const std::array<int, N>& global_shape,
                              const ArrayDecomp<N>& decomp) {
        FreeKey key = make_free_key(dtype, region, global_shape, decomp);
        auto it = free_arrays.find(key);
        if (it == free_arrays.end() || it->second.empty())
            return nullptr;
        CTArrayBase<N>* arr = it->second.back();
        it->second.pop_back();
        if (it->second.empty())
            free_arrays.erase(it);
        return arr;
    }

    template <typename T>
    Array<N, T>* allocate_or_reuse_typed(const ArrayRegion<N>& region,
                                         const std::array<int, N>& global_shape,
                                         int name,
                                         const ArrayDecomp<N>& decomp) {
        CTArrayBase<N>* reused = try_reuse(dtype_of<T>(), region, global_shape, decomp);
        if (reused == nullptr)
            return new Array<N, T>(region, global_shape, name, decomp);

        auto* arr = static_cast<Array<N, T>*>(reused);
        arr->name = name;
        arr->region = region;
        arr->decomp = decomp;
        arr->global_shape = global_shape;
        arr->global_size = 1;
        for (int d = 0; d < N; ++d)
            arr->global_size *= global_shape[d];
        arr->owner = true;
        arr->dtype = dtype_of<T>();

#ifdef USE_KOKKOS
        Kokkos::deep_copy(arr->d_view, T(0));
        arr->h_view = {};
#else
        T* data = static_cast<T*>(arr->data_ptr());
        for (int64_t i = 0; i < arr->local_size(); ++i)
            data[i] = T(0);
#endif
        return arr;
    }

    CTArrayBase<N>* allocate_or_reuse(const ArrayRegion<N>& region,
                                      const std::array<int, N>& global_shape,
                                      int name,
                                      DType dtype,
                                      const ArrayDecomp<N>& decomp) {
        CTArrayBase<N>* arr = nullptr;
        cn_dispatch(dtype, [&](auto tag) {
            arr = allocate_or_reuse_typed<decltype(tag)>(region, global_shape, name, decomp);
        });
        return arr;
    }

    template <typename T>
    Array<N, T>* ensure_array_typed(const ArrayRegion<N>& region,
                                    const std::array<int, N>& global_shape,
                                    int name,
                                    const ArrayDecomp<N>& decomp) {
        auto it = arrays.find(name);
        if (it == arrays.end()) {
            auto* arr = allocate_or_reuse_typed<T>(region, global_shape, name, decomp);
            arrays[name] = arr;
            return arr;
        }
        return static_cast<Array<N, T>*>(it->second);
    }

    CTArrayBase<N>* ensure_array(const ArrayRegion<N>& region,
                                 const std::array<int, N>& global_shape,
                                 int name,
                                 DType dtype,
                                 const ArrayDecomp<N>& decomp) {
        auto it = arrays.find(name);
        if (it == arrays.end()) {
            auto* arr = allocate_or_reuse(region, global_shape, name, dtype, decomp);
            arrays[name] = arr;
            return arr;
        }
        return it->second;
    }
#ifdef USE_KOKKOS
    std::unordered_map<int, void*> pending_sends; // send_id → device ptr
    // Posted device recv buffers, keyed by message identity; nocopydevice passes a null data pointer.
    std::map<std::vector<int>, std::deque<char*>> posted_recv_bufs;
    unsigned next_send_id = 0;
#ifdef USE_NVIDIA
    cudaStream_t compute_stream_raw;
    cudaStream_t comm_stream_raw;
    Kokkos::Cuda compute_exec;
    Kokkos::Cuda comm_exec;
#endif
#endif

    /// Return the N-D chare index (stored directly, no delinearization needed).
    std::array<int, N> nd_index() const { return index; }

    /// Callback to delegate contribute() to the owning chare.
    std::function<void(int, void*, CkReduction::reducerType, CkCallback)> contribute_fn;
    void contribute(int size, void* data, CkReduction::reducerType type, CkCallback cb) {
        contribute_fn(size, data, type, cb);
    }

    void init(typename PartitionTraits<N>::ProxyType proxy, std::array<int, N> idx,
              CProxy_ArrayDAGGroup dag_proxy_);
    ~PartitionImpl();

    int create(ArrayRegion<N>* region, int name, DType dtype, const ArrayDecomp<N>& decomp);
    void run();
    void receive_data(int node_id, int input_index, int name, int ndims, int* region_data,
                      int64_t size, char* data);
#ifdef USE_KOKKOS
    void receive_data(int& node_id, int& input_index, int& name, int& ndims, int*& region_data,
                      int64_t& size, char*& data, CkDeviceBufferPost* devicePost);
    void send_complete(int send_id);
#endif
    void comm_done(int node_id);
    void reduce_result(int node_id, int64_t size, char* data);
    void receive_reduce_partial(int node_id, int64_t size, char* data);
    void receive_linalg(int node_id, int64_t size, char* data);
    void receive_index_msg(int node_id, int64_t size, char* data);
    void receive_shape(int node_id, int64_t size, char* data);
    void receive_upload_tile(int node_id, int name, int64_t size, char* data);
    void process_get(int epoch);
};

// ---- Thin wrapper chare classes ----

class Partition1D : public CBase_Partition1D {
  public:
    PartitionImpl<1> impl;

    Partition1D(CProxy_ArrayDAGGroup dag_proxy) {
        impl.contribute_fn = [this](int sz, void* d, CkReduction::reducerType t, CkCallback cb) {
            this->contribute(sz, d, t, cb);
        };
        impl.init(thisProxy, {thisIndex}, dag_proxy);
    }
    Partition1D(CkMigrateMessage* m) {}
    ~Partition1D() {}

    void run() { impl.run(); }
    void receive_data(int node_id, int input_index, int name, int ndims, int* region_data,
                      int64_t size, char* data) {
        impl.receive_data(node_id, input_index, name, ndims, region_data, size, data);
    }
#ifdef USE_KOKKOS
    void receive_data(int& node_id, int& input_index, int& name, int& ndims, int*& region_data,
                      int64_t& size, char*& data, CkDeviceBufferPost* devicePost) {
        impl.receive_data(node_id, input_index, name, ndims, region_data, size, data, devicePost);
    }
    void send_complete(CkDataMsg* msg) {
        impl.send_complete(CkGetRefNum(msg));
        CkFreeMsg(msg);
    }
#endif
    void comm_done(int node_id) { impl.comm_done(node_id); }
    void reduce_result(int node_id, int64_t size, char* data) {
        impl.reduce_result(node_id, size, data);
    }
    void receive_reduce_partial(int node_id, int64_t size, char* data) {
        impl.receive_reduce_partial(node_id, size, data);
    }
    void receive_linalg(int node_id, int64_t size, char* data) {
        impl.receive_linalg(node_id, size, data);
    }
    void receive_index_msg(int node_id, int64_t size, char* data) {
        impl.receive_index_msg(node_id, size, data);
    }
    void receive_shape(int node_id, int64_t size, char* data) {
        impl.receive_shape(node_id, size, data);
    }
    void receive_upload_tile(int node_id, int name, int64_t size, char* data) {
        impl.receive_upload_tile(node_id, name, size, data);
    }
};

class Partition2D : public CBase_Partition2D {
  public:
    PartitionImpl<2> impl;

    Partition2D(CProxy_ArrayDAGGroup dag_proxy) {
        impl.contribute_fn = [this](int sz, void* d, CkReduction::reducerType t, CkCallback cb) {
            this->contribute(sz, d, t, cb);
        };
        impl.init(thisProxy, {thisIndex.x, thisIndex.y}, dag_proxy);
    }
    Partition2D(CkMigrateMessage* m) {}
    ~Partition2D() {}

    void run() { impl.run(); }
    void receive_data(int node_id, int input_index, int name, int ndims, int* region_data,
                      int64_t size, char* data) {
        impl.receive_data(node_id, input_index, name, ndims, region_data, size, data);
    }
#ifdef USE_KOKKOS
    void receive_data(int& node_id, int& input_index, int& name, int& ndims, int*& region_data,
                      int64_t& size, char*& data, CkDeviceBufferPost* devicePost) {
        impl.receive_data(node_id, input_index, name, ndims, region_data, size, data, devicePost);
    }
    void send_complete(CkDataMsg* msg) {
        impl.send_complete(CkGetRefNum(msg));
        CkFreeMsg(msg);
    }
#endif
    void comm_done(int node_id) { impl.comm_done(node_id); }
    void reduce_result(int node_id, int64_t size, char* data) {
        impl.reduce_result(node_id, size, data);
    }
    void receive_reduce_partial(int node_id, int64_t size, char* data) {
        impl.receive_reduce_partial(node_id, size, data);
    }
    void receive_linalg(int node_id, int64_t size, char* data) {
        impl.receive_linalg(node_id, size, data);
    }
    void receive_index_msg(int node_id, int64_t size, char* data) {
        impl.receive_index_msg(node_id, size, data);
    }
    void receive_shape(int node_id, int64_t size, char* data) {
        impl.receive_shape(node_id, size, data);
    }
    void receive_upload_tile(int node_id, int name, int64_t size, char* data) {
        impl.receive_upload_tile(node_id, name, size, data);
    }
};

class Partition3D : public CBase_Partition3D {
  public:
    PartitionImpl<3> impl;

    Partition3D(CProxy_ArrayDAGGroup dag_proxy) {
        impl.contribute_fn = [this](int sz, void* d, CkReduction::reducerType t, CkCallback cb) {
            this->contribute(sz, d, t, cb);
        };
        impl.init(thisProxy, {thisIndex.x, thisIndex.y, thisIndex.z}, dag_proxy);
    }
    Partition3D(CkMigrateMessage* m) {}
    ~Partition3D() {}

    void run() { impl.run(); }
    void receive_data(int node_id, int input_index, int name, int ndims, int* region_data,
                      int64_t size, char* data) {
        impl.receive_data(node_id, input_index, name, ndims, region_data, size, data);
    }
#ifdef USE_KOKKOS
    void receive_data(int& node_id, int& input_index, int& name, int& ndims, int*& region_data,
                      int64_t& size, char*& data, CkDeviceBufferPost* devicePost) {
        impl.receive_data(node_id, input_index, name, ndims, region_data, size, data, devicePost);
    }
    void send_complete(CkDataMsg* msg) {
        impl.send_complete(CkGetRefNum(msg));
        CkFreeMsg(msg);
    }
#endif
    void comm_done(int node_id) { impl.comm_done(node_id); }
    void reduce_result(int node_id, int64_t size, char* data) {
        impl.reduce_result(node_id, size, data);
    }
    void receive_reduce_partial(int node_id, int64_t size, char* data) {
        impl.receive_reduce_partial(node_id, size, data);
    }
    void receive_linalg(int node_id, int64_t size, char* data) {
        impl.receive_linalg(node_id, size, data);
    }
    void receive_index_msg(int node_id, int64_t size, char* data) {
        impl.receive_index_msg(node_id, size, data);
    }
    void receive_shape(int node_id, int64_t size, char* data) {
        impl.receive_shape(node_id, size, data);
    }
    void receive_upload_tile(int node_id, int name, int64_t size, char* data) {
        impl.receive_upload_tile(node_id, name, size, data);
    }
};
