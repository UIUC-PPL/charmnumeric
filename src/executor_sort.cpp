#ifdef USE_NVIDIA
// Before the Charm++ headers: their ALIGN_BYTES macro breaks cub/thrust.
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#endif

#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

// Distributed 1-D SORT / ARGSORT (sample sort) and SELECT (order statistics by radix
// selection); see execute_ds_node. Elements are ordered by (radix key, array index),
// so the order is stable and every split is exact.

#if defined(USE_NVIDIA)
#define DS_FN __host__ __device__ inline
#elif defined(USE_KOKKOS)
#define DS_FN KOKKOS_INLINE_FUNCTION
#else
#define DS_FN inline
#endif

struct DsState;

namespace ds_detail {

enum : int32_t {
    S_SAMPLES = 1,
    S_SPLIT,
    S_COUNTS,
    S_NB,
    S_G,
    S_PIECE,
    Q_HIST,
    Q_REQ,
    Q_GATHER,
    Q_CANDS,
    Q_DONE,
    Q_RESULT,
};

struct Hdr {
    int32_t kind, src;
    int32_t phase, ints;
    int64_t a, b, c, d;
};

// ---- radix keys: unsigned integers ordered like NumPy's sort (NaNs last) ----

template <int B>
struct UIntOf;
template <>
struct UIntOf<1> {
    using type = uint8_t;
};
template <>
struct UIntOf<2> {
    using type = uint16_t;
};
template <>
struct UIntOf<4> {
    using type = uint32_t;
};
template <>
struct UIntOf<8> {
    using type = uint64_t;
};
template <>
struct UIntOf<16> {
    using type = unsigned __int128;
};

template <typename K, typename T>
DS_FN K to_bits(const T& x) {
    K k;
    memcpy(&k, &x, sizeof(K));
    return k;
}

template <typename T, typename K>
DS_FN T from_bits(K k) {
    T x;
    memcpy(&x, &k, sizeof(K));
    return x;
}

/// Real types only. `canon`: -0.0 sorts equal to +0.0 (ties then keep index order).
template <typename T>
struct Key {
    using K = typename UIntOf<sizeof(T)>::type;
    static constexpr int W = 8 * (int)sizeof(T);
    static constexpr K SIGN = K(K(1) << (W - 1));

    static DS_FN K enc(const T& x, bool canon) {
        if constexpr (std::is_same_v<T, bool>) {
            return x ? K(1) : K(0);
        } else if constexpr (cn_is_real_float_v<T>) {
            if (cn_isnan(x))
                return K(~K(0));
            K b = to_bits<K>(x);
            if (canon && b == SIGN)
                b = 0;
            return (b & SIGN) ? K(~b) : K(b | SIGN);
        } else if constexpr (std::is_signed_v<T>) {
            return K(to_bits<K>(x) ^ SIGN);
        } else {
            return K(x);
        }
    }

    static DS_FN T dec(K k) {
        if constexpr (std::is_same_v<T, bool>) {
            return k != 0;
        } else if constexpr (cn_is_real_float_v<T>) {
            if (k == K(~K(0)))
                return from_bits<T>(K(~SIGN));
            return from_bits<T>((k & SIGN) ? K(k & ~SIGN) : K(~k));
        } else if constexpr (std::is_signed_v<T>) {
            return from_bits<T>(K(k ^ SIGN));
        } else {
            return T(k);
        }
    }
};

/// Complex: (real, imag) keys side by side; NaN parts take the top high words, so the
/// order is R+Rj < R+nanj (by real) < nan+Rj (by imag) < nan+nanj, as in NumPy. Zero
/// parts are always canonical (-0.0 must tie with 0.0 for the next part to decide).
template <typename F>
struct Key<cn_complex<F>> {
    using KF = Key<F>;
    using H = typename KF::K;
    using K = typename UIntOf<2 * sizeof(F)>::type;
    static constexpr int W = 2 * KF::W;
    static constexpr H TOP = H(~H(0));

    static DS_FN K join(H hi, H lo) { return K(K(hi) << KF::W) | K(lo); }

    static DS_FN K enc(const cn_complex<F>& x, bool) {
        F re = x.real(), im = x.imag();
        bool rn = re != re, in = im != im;
        if (rn && in)
            return join(TOP, TOP);
        if (rn)
            return join(H(TOP - 1), KF::enc(im, true));
        if (in)
            return join(H(TOP - 2), KF::enc(re, true));
        return join(KF::enc(re, true), KF::enc(im, true));
    }

    static DS_FN cn_complex<F> dec(K k) {
        H hi = H(k >> KF::W), lo = H(k);
        F nan = KF::dec(TOP);
        if (hi == TOP)
            return cn_complex<F>(nan, nan);
        if (hi == H(TOP - 1))
            return cn_complex<F>(nan, KF::dec(lo));
        if (hi == H(TOP - 2))
            return cn_complex<F>(KF::dec(lo), nan);
        return cn_complex<F>(KF::dec(hi), KF::dec(lo));
    }
};

template <typename K>
DS_FN int bucket_of(K k, int64_t g, const K* sk, const int64_t* sg, int nspl) {
    int lo = 0, hi = nspl;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (sk[mid] < k || (sk[mid] == k && sg[mid] <= g))
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

/// Group of key k at `depth` determined top bits: index into sorted prefixes, or -1.
template <typename K>
DS_FN int group_of(K k, int W, int depth, const K* grp, int G) {
    if (depth == 0)
        return 0;
    K p = K(k >> (W - depth));
    int lo = 0, hi = G - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (grp[mid] == p)
            return mid;
        if (grp[mid] < p)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

/// Byte offset of n keys stored after n int64 values (16-byte keys stay aligned).
DS_FN int64_t keys_after(int64_t n) {
    return (8 * n + 15) / 16 * 16;
}

DS_FN uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// ---- buffers: device memory in GPU builds ----
// NVIDIA: scratch comes from the device's stream-ordered pool on the stream of the
// chare being served (ds_stream_scope), so large buffers are cached instead of
// cudaMalloc/cudaFree per call; buffers for device_pack_send come from comm_alloc.

#ifdef USE_NVIDIA
thread_local cudaStream_t ds_cur_stream = nullptr;

struct DsPool {
    std::mutex mu;
    std::unordered_set<void*> live;
};

inline DsPool& ds_pool() {
    static DsPool pool;
    return pool;
}

/// Per device: keep up to CNUM_SORT_POOL_MB (default a fifth of the device) cached in the pool.
inline void ds_pool_init() {
    static std::mutex mu;
    static std::unordered_set<int> done;
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lock(mu);
    if (!done.insert(dev).second)
        return;
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess)
        return;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    uint64_t keep = total_b / 5;
    if (const char* e = getenv("CNUM_SORT_POOL_MB"))
        keep = (uint64_t)atoll(e) << 20;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
}

/// Makes room for what the pool cannot serve from the memory it holds.
inline void ds_make_room(int64_t bytes) {
    int dev = 0;
    cudaMemPool_t pool;
    uint64_t reserved = 0, used = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    }
    int64_t held = reserved > used ? static_cast<int64_t>(reserved - used) : 0;
    if (bytes > held)
        device_make_room(bytes - held);
}

/// Device attribute of the current device, cached per device.
inline int ds_dev_attr(cudaDeviceAttr attr) {
    static std::mutex mu;
    static std::map<std::pair<int, int>, int> cache;
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lock(mu);
    auto key = std::make_pair(dev, (int)attr);
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;
    int v = 0;
    cudaDeviceGetAttribute(&v, attr, dev);
    return cache[key] = v;
}
#endif

struct ds_stream_scope {
#ifdef USE_NVIDIA
    cudaStream_t prev;
    explicit ds_stream_scope(cudaStream_t s) : prev(ds_cur_stream) { ds_cur_stream = s; }
    ~ds_stream_scope() { ds_cur_stream = prev; }
#else
    explicit ds_stream_scope(int) {}
#endif
};

#ifdef USE_NVIDIA
#define DS_SCOPE(p) ds_stream_scope ds_scope_((p)->compute_stream_raw)
#else
#define DS_SCOPE(p) ds_stream_scope ds_scope_(0)
#endif

/// `comm`: the buffer may be passed to device_pack_send (freed with comm_free).
inline char* ds_alloc(int64_t bytes, bool comm = false) {
    bytes = std::max<int64_t>(bytes, 8);
#if defined(USE_NVIDIA)
    if (!comm) {
        ds_pool_init();
        ds_make_room(bytes);
        void* p = nullptr;
        if (cudaMallocAsync(&p, bytes, ds_cur_stream) != cudaSuccess) {
            cudaGetLastError();
            release_device_caches();
            if (cudaMallocAsync(&p, bytes, ds_cur_stream) != cudaSuccess)
                device_oom(bytes);
        }
        auto& pool = ds_pool();
        std::lock_guard<std::mutex> lock(pool.mu);
        pool.live.insert(p);
        return static_cast<char*>(p);
    }
    return static_cast<char*>(comm_alloc(bytes));
#elif defined(USE_KOKKOS)
    (void)comm;
    return static_cast<char*>(comm_alloc(bytes));
#else
    (void)comm;
    return new char[bytes];
#endif
}

/// Frees ds_alloc and received (comm_alloc) buffers once the serving stream is past them.
inline void ds_free(void* p) {
    if (p == nullptr)
        return;
#if defined(USE_NVIDIA)
    {
        auto& pool = ds_pool();
        std::lock_guard<std::mutex> lock(pool.mu);
        auto it = pool.live.find(p);
        if (it != pool.live.end()) {
            pool.live.erase(it);
            hapiCheck(cudaFreeAsync(p, ds_cur_stream));
            return;
        }
    }
    comm_free(p);
#elif defined(USE_KOKKOS)
    comm_free(p);
#else
    delete[] static_cast<char*>(p);
#endif
}

#ifdef USE_NVIDIA
template <int N>
cudaStream_t ds_stream(PartitionImpl<N>* p) {
    return p->compute_stream_raw;
}
#endif

template <int N>
void ds_copy(PartitionImpl<N>* p, void* dst, const void* src, int64_t bytes) {
    if (bytes <= 0)
        return;
#ifdef USE_NVIDIA
    hapiCheck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, ds_stream(p)));
#else
    (void)p;
    std::memcpy(dst, src, bytes);
#endif
}

/// Host -> device (host -> host in CPU builds).
template <int N>
void ds_to_dev(PartitionImpl<N>* p, void* dst, const void* src, int64_t bytes) {
    if (bytes <= 0)
        return;
#if defined(USE_NVIDIA)
    hapiCheck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, ds_stream(p)));
    hapiCheck(cudaStreamSynchronize(ds_stream(p)));
#elif defined(USE_KOKKOS)
    (void)p;
    using U = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::deep_copy(Kokkos::View<char*, DeviceSpace, U>(static_cast<char*>(dst), bytes),
                      Kokkos::View<const char*, HostSpace, U>(static_cast<const char*>(src),
                                                              bytes));
#else
    (void)p;
    std::memcpy(dst, src, bytes);
#endif
}

/// Device -> host (host -> host in CPU builds).
template <int N>
void ds_to_host(PartitionImpl<N>* p, void* dst, const void* src, int64_t bytes) {
    if (bytes <= 0)
        return;
#if defined(USE_NVIDIA)
    hapiCheck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, ds_stream(p)));
    hapiCheck(cudaStreamSynchronize(ds_stream(p)));
#elif defined(USE_KOKKOS)
    (void)p;
    using U = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::deep_copy(Kokkos::View<char*, HostSpace, U>(static_cast<char*>(dst), bytes),
                      Kokkos::View<const char*, DeviceSpace, U>(static_cast<const char*>(src),
                                                                bytes));
#else
    (void)p;
    std::memcpy(dst, src, bytes);
#endif
}

template <int N>
void ds_fence(PartitionImpl<N>* p) {
#if defined(USE_NVIDIA)
    hapiCheck(cudaStreamSynchronize(ds_stream(p)));
#elif defined(USE_KOKKOS)
    (void)p;
    Kokkos::fence();
#else
    (void)p;
#endif
}

// ---- local kernels ----

#ifdef USE_NVIDIA
#define DS_LAUNCHED() hapiCheck(cudaPeekAtLastError())
constexpr int DS_BLOCK = 256, DS_WARPS = DS_BLOCK / 32, DS_ITEMS = 8;
constexpr int DS_TILE = DS_BLOCK * DS_ITEMS;

__device__ inline unsigned lanemask_lt() {
    unsigned m;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

template <typename K>
__device__ void load_splitters(char* smem, const K* spk, const int64_t* spg, int nspl,
                               K*& sk, int64_t*& sg) {
    sg = reinterpret_cast<int64_t*>(smem);
    sk = reinterpret_cast<K*>(smem + keys_after(nspl));
    for (int i = threadIdx.x; i < nspl; i += blockDim.x) {
        sg[i] = spg[i];
        sk[i] = spk[i];
    }
}

DS_FN size_t split_smem(int nspl, int ksize) {
    return ((size_t)keys_after(nspl) + (size_t)nspl * ksize + 7) / 8 * 8;
}

/// hist[b * nblocks + block] = elements of the block going to bucket b.
template <typename T>
__global__ void ds_count_kernel(const T* in, int64_t n, int64_t g0,
                                const typename Key<T>::K* spk, const int64_t* spg, int nspl,
                                bool canon, unsigned* hist, int nblocks) {
    using K = typename Key<T>::K;
    extern __shared__ __align__(16) char smem[];
    K* sk;
    int64_t* sg;
    load_splitters(smem, spk, spg, nspl, sk, sg);
    unsigned* cnt = reinterpret_cast<unsigned*>(smem + split_smem(nspl, sizeof(K)));
    int P = nspl + 1;
    for (int b = threadIdx.x; b < P; b += blockDim.x)
        cnt[b] = 0;
    __syncthreads();
    int64_t base = (int64_t)blockIdx.x * DS_TILE;
    for (int it = 0; it < DS_ITEMS; ++it) {
        int64_t i = base + it * DS_BLOCK + threadIdx.x;
        int b = P;
        if (i < n)
            b = bucket_of<K>(Key<T>::enc(in[i], canon), g0 + i, sk, sg, nspl);
        unsigned peers = __match_any_sync(0xffffffffu, b);
        if (b < P && (peers & lanemask_lt()) == 0)
            atomicAdd(&cnt[b], (unsigned)__popc(peers));
    }
    __syncthreads();
    for (int b = threadIdx.x; b < P; b += blockDim.x)
        hist[(int64_t)b * nblocks + blockIdx.x] = cnt[b];
}

/// Stable scatter of the block's elements into the per-bucket buffers dk[b] (and dg[b]).
template <typename T, bool ARG>
__global__ void ds_scatter_kernel(const T* in, int64_t n, int64_t g0,
                                  const typename Key<T>::K* spk, const int64_t* spg, int nspl,
                                  bool canon, const unsigned* offs, int nblocks,
                                  const int64_t* bstart, typename Key<T>::K* const* dk,
                                  int64_t* const* dg) {
    using K = typename Key<T>::K;
    extern __shared__ __align__(16) char smem[];
    K* sk;
    int64_t* sg;
    load_splitters(smem, spk, spg, nspl, sk, sg);
    int P = nspl + 1;
    unsigned* wh = reinterpret_cast<unsigned*>(smem + split_smem(nspl, sizeof(K)));
    for (int i = threadIdx.x; i < DS_WARPS * P; i += blockDim.x)
        wh[i] = 0;
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int64_t wbase = (int64_t)blockIdx.x * DS_TILE + (int64_t)warp * 32 * DS_ITEMS;
    K keys[DS_ITEMS];
    int bk[DS_ITEMS];
    unsigned peers[DS_ITEMS];
    unsigned lt = lanemask_lt();
#pragma unroll
    for (int it = 0; it < DS_ITEMS; ++it) {
        int64_t i = wbase + it * 32 + lane;
        bk[it] = P;
        if (i < n) {
            keys[it] = Key<T>::enc(in[i], canon);
            bk[it] = bucket_of<K>(keys[it], g0 + i, sk, sg, nspl);
        }
        peers[it] = __match_any_sync(0xffffffffu, bk[it]);
        if (bk[it] < P && (peers[it] & lt) == 0)
            wh[warp * P + bk[it]] += __popc(peers[it]);
        __syncwarp();
    }
    __syncthreads();
    for (int b = threadIdx.x; b < P; b += blockDim.x) {
        unsigned acc = offs[(int64_t)b * nblocks + blockIdx.x] - (unsigned)bstart[b];
        for (int w = 0; w < DS_WARPS; ++w) {
            unsigned t = wh[w * P + b];
            wh[w * P + b] = acc;
            acc += t;
        }
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < DS_ITEMS; ++it) {
        int b = bk[it];
        unsigned pos = 0;
        if (b < P)
            pos = wh[warp * P + b] + __popc(peers[it] & lt);
        __syncwarp();
        if (b < P) {
            dk[b][pos] = keys[it];
            if constexpr (ARG)
                dg[b][pos] = g0 + wbase + it * 32 + lane;
            if ((peers[it] & lt) == 0)
                wh[warp * P + b] += __popc(peers[it]);
        }
        __syncwarp();
    }
}

/// Many buckets (no shared-memory split): bucket id and local index of every element.
template <typename T>
__global__ void ds_bid_kernel(const T* in, int64_t n, int64_t g0, const typename Key<T>::K* spk,
                              const int64_t* spg, int nspl, bool canon, unsigned* bid,
                              unsigned* idx, unsigned* cnt) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int b = bucket_of(Key<T>::enc(in[i], canon), g0 + i, spk, spg, nspl);
    bid[i] = b;
    idx[i] = (unsigned)i;
    atomicAdd(&cnt[b], 1u);
}

/// Elements sorted by bucket id (stable) into the per-bucket buffers.
template <typename T, bool ARG>
__global__ void ds_gscatter_kernel(const T* in, int64_t n, int64_t g0, bool canon,
                                   const unsigned* bid, const unsigned* idx,
                                   const int64_t* bstart, typename Key<T>::K* const* dk,
                                   int64_t* const* dg) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n)
        return;
    unsigned b = bid[j], i = idx[j];
    int64_t pos = j - bstart[b];
    dk[b][pos] = Key<T>::enc(in[i], canon);
    if constexpr (ARG)
        dg[b][pos] = g0 + i;
}

__global__ void ds_bucket_starts(const unsigned* offs, int nblocks, int P, int64_t* bstart) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < P)
        bstart[b] = offs[(int64_t)b * nblocks];
}

template <typename T>
__global__ void ds_sample_kernel(const T* in, int64_t n, int64_t g0, uint64_t seed, int s,
                                 bool canon, typename Key<T>::K* sk, int64_t* sg) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= s)
        return;
    int64_t i = (int64_t)(mix64(seed + (uint64_t)k) % (uint64_t)n);
    sk[k] = Key<T>::enc(in[i], canon);
    sg[k] = g0 + i;
}

template <typename T>
__global__ void ds_encode_kernel(const T* in, int64_t n, bool canon, typename Key<T>::K* k,
                                 int64_t* g, int64_t g0) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    k[i] = Key<T>::enc(in[i], canon);
    if (g)
        g[i] = g0 + i;
}

template <typename T>
__global__ void ds_decode_kernel(const typename Key<T>::K* k, int64_t n, T* out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = Key<T>::dec(k[i]);
}

template <typename T>
__global__ void dq_hist_kernel(const T* in, int64_t n, int depth, int b,
                               const typename Key<T>::K* groups, int G, unsigned* ghist) {
    using K = typename Key<T>::K;
    constexpr int W = Key<T>::W;
    extern __shared__ __align__(16) char smem[];
    K* grp = reinterpret_cast<K*>(smem);
    unsigned* h = reinterpret_cast<unsigned*>(smem + ((size_t)G * sizeof(K) + 7) / 8 * 8);
    int nb = G << b;
    for (int i = threadIdx.x; i < G; i += blockDim.x)
        grp[i] = groups[i];
    for (int i = threadIdx.x; i < nb; i += blockDim.x)
        h[i] = 0;
    __syncthreads();
    K mask = K((K(1) << b) - 1);
    if (b == W)
        mask = K(~K(0));
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        K k = Key<T>::enc(in[i], false);
        int g = group_of<K>(k, W, depth, grp, G);
        if (g < 0)
            continue;
        int d = (int)(K(k >> (W - depth - b)) & mask);
        atomicAdd(&h[(g << b) + d], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nb; i += blockDim.x)
        if (h[i])
            atomicAdd(&ghist[i], h[i]);
}

template <typename T>
__global__ void dq_compact_kernel(const T* in, int64_t n, int depth,
                                  const typename Key<T>::K* groups, int G,
                                  typename Key<T>::K* out, unsigned* cursor) {
    using K = typename Key<T>::K;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        K k = Key<T>::enc(in[i], false);
        if (group_of<K>(k, Key<T>::W, depth, groups, G) >= 0)
            out[atomicAdd(cursor, 1u)] = k;
    }
}

inline int grid_for(int64_t n, int block = 256) {
    return (int)std::max<int64_t>(1, std::min<int64_t>((n + block - 1) / block, 1 << 30));
}
#endif

/// Device (GPU builds) copy of a small host array, freed with ds_free.
template <typename X>
X* ds_upload(PartitionImpl<1>* p, const std::vector<X>& v) {
    X* d = reinterpret_cast<X*>(ds_alloc((int64_t)v.size() * sizeof(X)));
    ds_copy(p, d, v.data(), (int64_t)v.size() * sizeof(X));
    return d;
}

/// keys[i] = enc(in[i]) (and g[i] = g0 + i).
template <typename T>
void ds_encode(PartitionImpl<1>* p, const T* in, int64_t n, bool canon,
               typename Key<T>::K* k, int64_t* g, int64_t g0) {
#ifdef USE_NVIDIA
    if (n > 0) {
        ds_encode_kernel<T><<<grid_for(n), 256, 0, ds_stream(p)>>>(in, n, canon, k, g, g0);
        DS_LAUNCHED();
    }
#else
    (void)p;
    for (int64_t i = 0; i < n; ++i) {
        k[i] = Key<T>::enc(in[i], canon);
        if (g)
            g[i] = g0 + i;
    }
#endif
}

template <typename T>
void ds_decode(PartitionImpl<1>* p, const typename Key<T>::K* k, int64_t n, T* out) {
#ifdef USE_NVIDIA
    if (n > 0) {
        ds_decode_kernel<T><<<grid_for(n), 256, 0, ds_stream(p)>>>(k, n, out);
        DS_LAUNCHED();
    }
#else
    (void)p;
    for (int64_t i = 0; i < n; ++i)
        out[i] = Key<T>::dec(k[i]);
#endif
}

/// Stable sort of keys (with values g when non-null) in place; a/ga are scratch of size n.
/// Returns true when the result is in the scratch buffers.
template <typename K>
bool ds_sort_local(PartitionImpl<1>* p, K* k, K* ka, int64_t* g, int64_t* ga, int64_t n) {
    if (n <= 1)
        return false;
#ifdef USE_NVIDIA
    cub::DoubleBuffer<K> dk(k, ka);
    cub::DoubleBuffer<int64_t> dv(g, ga);
    size_t tb = 0;
    int64_t nn = n;
    if (g)
        hapiCheck(cub::DeviceRadixSort::SortPairs(nullptr, tb, dk, dv, nn, 0, 8 * sizeof(K),
                                                  ds_stream(p)));
    else
        hapiCheck(cub::DeviceRadixSort::SortKeys(nullptr, tb, dk, nn, 0, 8 * sizeof(K),
                                                 ds_stream(p)));
    char* tmp = ds_alloc((int64_t)tb);
    if (g)
        hapiCheck(cub::DeviceRadixSort::SortPairs(tmp, tb, dk, dv, nn, 0, 8 * sizeof(K),
                                                  ds_stream(p)));
    else
        hapiCheck(cub::DeviceRadixSort::SortKeys(tmp, tb, dk, nn, 0, 8 * sizeof(K),
                                                 ds_stream(p)));
    ds_fence(p);
    ds_free(tmp);
    return dk.selector == 1;
#else
    (void)p;
    (void)ka;
    (void)ga;
    if (g) {
        std::vector<std::pair<K, int64_t>> v(n);
        for (int64_t i = 0; i < n; ++i)
            v[i] = {k[i], g[i]};
        std::sort(v.begin(), v.end());
        for (int64_t i = 0; i < n; ++i) {
            k[i] = v[i].first;
            g[i] = v[i].second;
        }
    } else {
        std::sort(k, k + n);
    }
    return false;
#endif
}

/// Row-major tile viewed as `lines` lines of length `len` along `axis`.
struct AxisGeom {
    int64_t sizes[3], strides[3];
    int nd, axis;
    int64_t len, lines;

    DS_FN int64_t phys(int64_t l, int64_t p) const {
        int64_t idx = p * strides[axis];
        for (int d = nd - 1; d >= 0; --d) {
            if (d == axis)
                continue;
            idx += (l % sizes[d]) * strides[d];
            l /= sizes[d];
        }
        return idx;
    }
};

/// Lines [0, lines) x [0, len) of a packed block (axis last).
inline AxisGeom block_geom(int64_t lines, int64_t len) {
    return AxisGeom{{lines, len, 1}, {len, 1, 1}, 2, 1, len, lines};
}

} // namespace ds_detail

using namespace ds_detail;

/// Per-node state of one chare: 1-D sorts and SELECT, and N-D axis sorts (ax_*).
struct DsState {
    struct Piece {
        int phase = 0, src = 0;
        int64_t start = 0, count = 0;
        char* buf = nullptr; // index values (ARGSORT), then keys; phase 1: output values
    };
    DAGNode* node = nullptr;
    bool started = false;
    std::vector<std::vector<char>> stash;
    std::vector<Piece> early; // data pieces that arrived before the start

    Opcode op = Opcode::SORT;
    DType dt = DType::FLOAT64;
    int in_name = 0, out_name = 0;
    int first = 0, P = 0, rank = -1; // participating chares first .. first+P-1
    std::vector<int64_t> pstart;     // array start of each participant's tile, then the size
    int64_t ksize = 0;

    // SORT / ARGSORT
    bool arg = false;
    std::vector<std::vector<char>> samples;
    int samples_got = 0;
    bool split_done = false;
    std::vector<int64_t> col;  // elements each source sends to this bucket
    int ncol = 0;
    int64_t nb = 0, G = -1;    // bucket size, global rank of its first element
    std::vector<int64_t> nbs;  // first chare: bucket sizes
    int nbs_got = 0;
    std::vector<int> owner;    // participant rank holding each bucket
    int bucket = -1;           // the bucket this chare holds, if any
    std::vector<Piece> pieces; // phase 0, by source
    bool sorted = false;
    std::vector<Piece> rebal; // phase 1
    int64_t rebal_got = 0;

    // SELECT
    int coord = -1, result_chare = -1; // chare indices
    std::vector<int64_t> want;         // requested ranks
    int depth = 0, bits = 0;
    std::vector<uint64_t> groups;      // prefixes of the last histogram (sorted)
    std::vector<unsigned> hist;        // this chare's last histogram
    std::vector<std::vector<char>> replies;
    int replies_got = 0;
    // coordinator: per distinct rank
    std::vector<int64_t> uranks, resid;
    std::vector<uint64_t> prefix;
    std::vector<char> host_in; // non-NVIDIA Kokkos builds: host copy of the tile

    // N-D axis sort: rank / P / pstart are this chare's place among the tiles along the axis
    std::vector<int64_t> lst;            // P + 1: the lines each of them sorts whole
    std::vector<std::array<int, 3>> grp; // their chare indices
    AxisGeom tg{};                       // this tile's lines along the axis
    char* lines = nullptr;               // this chare's lines (whole), being assembled
    int64_t esize = 0;
    int need1 = 0, got1 = 0, need2 = 0, got2 = 0;
};

namespace ds_detail {

/// The tile as the local kernels read it (a host copy in non-NVIDIA Kokkos builds).
template <typename T>
const T* ds_in(const DsState& s, const T* ip) {
    return s.host_in.empty() ? ip : reinterpret_cast<const T*>(s.host_in.data());
}

template <typename X>
void put(std::vector<char>& b, const X* p, int64_t n) {
    size_t off = b.size();
    b.resize(off + n * sizeof(X));
    if (n > 0)
        memcpy(b.data() + off, p, n * sizeof(X));
}

std::vector<char> make_msg(Hdr h) {
    std::vector<char> b(sizeof(Hdr));
    memcpy(b.data(), &h, sizeof(h));
    return b;
}

void send_ctrl(PartitionImpl<1>* p, int chare, int node_id, const std::vector<char>& b) {
    p->thisProxy[chare].receive_sort_msg(node_id, (int64_t)b.size(),
                                         const_cast<char*>(b.data()));
}

int pe_of(PartitionImpl<1>* p, int chare) {
    auto target = p->thisProxy[chare];
    return target.ckLocalBranch()->lastKnown(target.ckGetIndex());
}

/// Hands `buf` (ownership) to `chare`: by pointer on this PE, else as a data message.
void send_piece(PartitionImpl<1>* p, int chare, int node_id, int phase, int src, int64_t start,
                int64_t count, char* buf, int64_t bytes) {
    if (pe_of(p, chare) == CkMyPe()) {
        Hdr h{S_PIECE, src, phase, 0, start, count, bytes, reinterpret_cast<int64_t>(buf)};
        send_ctrl(p, chare, node_id, make_msg(h));
        return;
    }
    // phase 0 pieces start at 0 and carry their count; phase 1 pieces carry their start
    int64_t v = phase == 0 ? count : start;
    int region_data[3] = {src, (int)(uint32_t)v, (int)(v >> 32)};
#ifdef USE_KOKKOS
    ChareIndex<1> ci;
    ci.idx[0] = chare;
    device_pack_send<1, 1>(p, p->thisProxy, ci, node_id, kDsDataTag + phase, 0, region_data,
                           bytes, buf);
#else
    ct_send_data(p->thisProxy[chare], node_id, kDsDataTag + phase, 0, 1, region_data, bytes, buf);
    delete[] buf;
#endif
}

struct Layout {
    int first, P;
    std::vector<int64_t> start; // P + 1
};

Layout layout_of(const ArrayDecomp<1>& dec) {
    Layout L;
    int64_t n = dec.global_shape[0];
    L.first = n > 0 ? dec.offset[0] / dec.tile : 0;
    int last = n > 0 ? (int)((dec.offset[0] + n - 1) / dec.tile) : -1;
    L.P = last - L.first + 1;
    for (int i = 0; i < L.P; ++i)
        L.start.push_back(dec.chare_region_global({L.first + i}).start[0] - dec.offset[0]);
    L.start.push_back(n);
    return L;
}

// ------------------------------------------------------------------ SORT

template <typename T>
void sort_single(PartitionImpl<1>* p, DsState& s, const T* in, void* outp, int64_t n) {
    using K = typename Key<T>::K;
    char* kb = ds_alloc(2 * n * sizeof(K));
    K* k = reinterpret_cast<K*>(kb);
    K* ka = k + n;
    char* gb = s.arg ? ds_alloc(2 * n * sizeof(int64_t)) : nullptr;
    int64_t* g = reinterpret_cast<int64_t*>(gb);
    int64_t* ga = g ? g + n : nullptr;
    ds_encode<T>(p, in, n, s.arg, k, g, 0);
    bool alt = ds_sort_local<K>(p, k, ka, g, ga, n);
    if (s.arg)
        ds_copy(p, outp, alt ? ga : g, n * sizeof(int64_t));
    else
        ds_decode<T>(p, alt ? ka : k, n, static_cast<T*>(outp));
    ds_fence(p);
    ds_free(kb);
    ds_free(gb);
}

constexpr int kSamplesPerTile = 1024;
constexpr int kAllBuckets = 64, kMaxBuckets = 256;

template <typename T>
void sort_send_samples(PartitionImpl<1>* p, DsState& s, const T* in, int64_t n) {
    using K = typename Key<T>::K;
    int64_t tile = 0;
    for (int i = 0; i < s.P; ++i)
        tile = std::max(tile, s.pstart[i + 1] - s.pstart[i]);
    int64_t per = std::max<int64_t>(64, std::min<int64_t>(kSamplesPerTile, (1 << 22) / s.P));
    int64_t want = (per * n + tile - 1) / tile;
    int ns = (int)std::max<int64_t>(1, std::min<int64_t>(n, want));
    int64_t g0 = s.pstart[s.rank];
    uint64_t seed = mix64(((uint64_t)s.node->id << 32) ^ (uint64_t)s.rank);
    std::vector<K> hk(ns);
    std::vector<int64_t> hg(ns);
#ifdef USE_NVIDIA
    char* d = ds_alloc(keys_after(ns) + (int64_t)ns * sizeof(K));
    int64_t* dg = reinterpret_cast<int64_t*>(d);
    K* dk = reinterpret_cast<K*>(d + keys_after(ns));
    ds_sample_kernel<T><<<grid_for(ns), 256, 0, ds_stream(p)>>>(in, n, g0, seed, ns, s.arg, dk,
                                                               dg);
    DS_LAUNCHED();
    ds_to_host(p, hk.data(), dk, ns * sizeof(K));
    ds_to_host(p, hg.data(), dg, ns * sizeof(int64_t));
    ds_free(d);
#else
    (void)p;
    for (int k = 0; k < ns; ++k) {
        int64_t i = (int64_t)(mix64(seed + (uint64_t)k) % (uint64_t)n);
        hk[k] = Key<T>::enc(in[i], s.arg);
        hg[k] = g0 + i;
    }
#endif
    auto msg = make_msg(Hdr{S_SAMPLES, s.rank, 0, 0, ns, 0, 0, 0});
    put(msg, hg.data(), ns);
    put(msg, hk.data(), ns);
    send_ctrl(p, s.first, s.node->id, msg);
}

/// Root: splitters at the participants' tile starts, from the pooled samples.
template <typename T>
void sort_make_splitters(PartitionImpl<1>* p, DsState& s) {
    using K = typename Key<T>::K;
    std::vector<std::pair<K, int64_t>> all;
    for (auto& m : s.samples) {
        Hdr h;
        memcpy(&h, m.data(), sizeof(h));
        std::vector<int64_t> g(h.a);
        std::vector<K> k(h.a);
        memcpy(g.data(), m.data() + sizeof(h), 8 * h.a);
        memcpy(k.data(), m.data() + sizeof(h) + 8 * h.a, sizeof(K) * h.a);
        for (int64_t i = 0; i < h.a; ++i)
            all.push_back({k[i], g[i]});
    }
    s.samples.clear();
    std::sort(all.begin(), all.end());
    // Every chare holds a bucket when there are few; else the first chare of each PE
    // (at most kMaxBuckets), so the exchange is O(P * buckets) messages, not O(P^2).
    std::vector<int64_t> owner;
    if (s.P <= kAllBuckets) {
        for (int i = 0; i < s.P; ++i)
            owner.push_back(i);
    } else {
        std::unordered_set<int> pes;
        for (int i = 0; i < s.P && (int)owner.size() < kMaxBuckets; ++i)
            if (pes.insert(pe_of(p, s.first + i)).second)
                owner.push_back(i);
    }
    int B = (int)owner.size();
    int64_t S = (int64_t)all.size(), n = s.pstart[s.P];
    std::vector<int64_t> sg(B - 1);
    std::vector<K> sk(B - 1);
    for (int j = 1; j < B; ++j) {
        double f = (double)s.pstart[owner[j]] / n;
        int64_t idx = std::min<int64_t>(S - 1, (int64_t)(f * S));
        sk[j - 1] = all[idx].first;
        sg[j - 1] = all[idx].second;
    }
    auto msg = make_msg(Hdr{S_SPLIT, s.rank, 0, 0, B - 1, 0, 0, 0});
    put(msg, sg.data(), B - 1);
    put(msg, sk.data(), B - 1);
    put(msg, owner.data(), B);
    for (int i = 0; i < s.P; ++i)
        send_ctrl(p, s.first + i, s.node->id, msg);
}

/// Bucket this chare's tile by the splitters and send the buckets and the counts.
template <typename T>
void sort_partition(PartitionImpl<1>* p, DsState& s, const T* in, int64_t n,
                    const std::vector<char>& m) {
    using K = typename Key<T>::K;
    Hdr h;
    memcpy(&h, m.data(), sizeof(h));
    int nspl = (int)h.a, P = nspl + 1; // buckets
    std::vector<int64_t> vsg(nspl);
    std::vector<K> vsk(nspl);
    memcpy(vsg.data(), m.data() + sizeof(h), 8 * nspl);
    memcpy(vsk.data(), m.data() + sizeof(h) + 8 * nspl, sizeof(K) * nspl);
    const int64_t* how = reinterpret_cast<const int64_t*>(m.data() + sizeof(h) +
                                                          (8 + sizeof(K)) * nspl);
    s.owner.assign(how, how + P);
    for (int b = 0; b < P; ++b)
        if (s.owner[b] == s.rank)
            s.bucket = b;
    int64_t g0 = s.pstart[s.rank];
    std::vector<int64_t> cnt(P, 0);
    std::vector<char*> bufs(P, nullptr);
    // bucket buffers: index values (ARGSORT), then keys at koff(count)
    auto koff = [&](int64_t c) { return s.arg ? keys_after(c) : 0; };
    auto pbytes = [&](int64_t c) { return koff(c) + c * (int64_t)sizeof(K); };
#ifdef USE_NVIDIA
    cudaStream_t st = ds_stream(p);
    int64_t* dsg = ds_upload(p, vsg);
    K* dsk = ds_upload(p, vsk);
    size_t sm1 = split_smem(nspl, sizeof(K)) + P * sizeof(unsigned);
    size_t sm2 = split_smem(nspl, sizeof(K)) + (size_t)DS_WARPS * P * sizeof(unsigned);
    static const int max_parts = getenv("CNUM_SORT_SMEM_PARTS")
                                     ? atoi(getenv("CNUM_SORT_SMEM_PARTS"))
                                     : 1024;
    size_t optin = (size_t)ds_dev_attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
    std::vector<K*> pk(P);
    std::vector<int64_t*> pg(P);
    auto make_bufs = [&](const std::vector<int64_t>& hb) {
        for (int b = 0; b < P; ++b)
            cnt[b] = (b + 1 < P ? hb[b + 1] : n) - hb[b];
        for (int b = 0; b < P; ++b) {
            bufs[b] = ds_alloc(pbytes(cnt[b]), pe_of(p, s.first + s.owner[b]) != CkMyPe());
            pk[b] = reinterpret_cast<K*>(bufs[b] + koff(cnt[b]));
            pg[b] = reinterpret_cast<int64_t*>(bufs[b]);
        }
    };
    if (P > max_parts || sm1 > optin || sm2 > optin) {
        // bucket ids, stable radix sort by id, then a gather into the buckets
        int bits = 1;
        while ((1 << bits) < P)
            ++bits;
        char* work = ds_alloc(16 * n + (int64_t)(2 * P + 1) * 8);
        unsigned* bid = reinterpret_cast<unsigned*>(work);
        unsigned* bid2 = bid + n;
        unsigned* idx = bid2 + n;
        unsigned* idx2 = idx + n;
        unsigned* dcnt = idx2 + n;
        int64_t* bstart = reinterpret_cast<int64_t*>(work + 16 * n + ((P * 4 + 7) / 8) * 8);
        hapiCheck(cudaMemsetAsync(dcnt, 0, P * sizeof(unsigned), st));
        ds_bid_kernel<T><<<grid_for(n), 256, 0, st>>>(in, n, g0, dsk, dsg, nspl, s.arg, bid, idx,
                                                     dcnt);
        DS_LAUNCHED();
        cub::DoubleBuffer<unsigned> kb(bid, bid2), vb(idx, idx2);
        size_t tb = 0;
        hapiCheck(cub::DeviceRadixSort::SortPairs(nullptr, tb, kb, vb, n, 0, bits, st));
        char* tmp = ds_alloc((int64_t)tb);
        hapiCheck(cub::DeviceRadixSort::SortPairs(tmp, tb, kb, vb, n, 0, bits, st));
        std::vector<unsigned> hc(P);
        ds_to_host(p, hc.data(), dcnt, P * sizeof(unsigned));
        std::vector<int64_t> hb(P);
        int64_t acc = 0;
        for (int b = 0; b < P; ++b) {
            hb[b] = acc;
            acc += hc[b];
        }
        make_bufs(hb);
        ds_copy(p, bstart, hb.data(), P * sizeof(int64_t));
        K** dpk = ds_upload(p, pk);
        int64_t** dpg = ds_upload(p, pg);
        if (s.arg)
            ds_gscatter_kernel<T, true><<<grid_for(n), 256, 0, st>>>(
                in, n, g0, s.arg, kb.Current(), vb.Current(), bstart, dpk, dpg);
        else
            ds_gscatter_kernel<T, false><<<grid_for(n), 256, 0, st>>>(
                in, n, g0, s.arg, kb.Current(), vb.Current(), bstart, dpk, dpg);
        DS_LAUNCHED();
        ds_fence(p);
        ds_free(tmp);
        ds_free(work);
        ds_free(dsg);
        ds_free(dsk);
        ds_free(dpk);
        ds_free(dpg);
    } else {
        int nblocks = (int)std::max<int64_t>(1, (n + DS_TILE - 1) / DS_TILE);
        int64_t hn = (int64_t)P * nblocks;
        char* work = ds_alloc(2 * hn * sizeof(unsigned) + (P + 1) * sizeof(int64_t));
        unsigned* hist = reinterpret_cast<unsigned*>(work);
        unsigned* offs = hist + hn;
        int64_t* bstart = reinterpret_cast<int64_t*>(offs + hn);
        if (sm1 > 48 * 1024)
            hapiCheck(cudaFuncSetAttribute(
                ds_count_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm1));
        if (sm2 > 48 * 1024) {
            hapiCheck(cudaFuncSetAttribute(
                ds_scatter_kernel<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm2));
            hapiCheck(cudaFuncSetAttribute(
                ds_scatter_kernel<T, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)sm2));
        }
        ds_count_kernel<T><<<nblocks, DS_BLOCK, sm1, st>>>(in, n, g0, dsk, dsg, nspl, s.arg,
                                                            hist, nblocks);
        DS_LAUNCHED();
        size_t tb = 0;
        hapiCheck(cub::DeviceScan::ExclusiveSum(nullptr, tb, hist, offs, (int)hn, st));
        char* tmp = ds_alloc((int64_t)tb);
        hapiCheck(cub::DeviceScan::ExclusiveSum(tmp, tb, hist, offs, (int)hn, st));
        ds_bucket_starts<<<(P + 255) / 256, 256, 0, st>>>(offs, nblocks, P, bstart);
        DS_LAUNCHED();
        std::vector<int64_t> hb(P);
        ds_to_host(p, hb.data(), bstart, P * sizeof(int64_t));
        ds_free(tmp);
        make_bufs(hb);
        K** dpk = ds_upload(p, pk);
        int64_t** dpg = ds_upload(p, pg);
        if (s.arg)
            ds_scatter_kernel<T, true><<<nblocks, DS_BLOCK, sm2, st>>>(
                in, n, g0, dsk, dsg, nspl, s.arg, offs, nblocks, bstart, dpk, dpg);
        else
            ds_scatter_kernel<T, false><<<nblocks, DS_BLOCK, sm2, st>>>(
                in, n, g0, dsk, dsg, nspl, s.arg, offs, nblocks, bstart, dpk, dpg);
        DS_LAUNCHED();
        ds_fence(p);
        ds_free(work);
        ds_free(dsg);
        ds_free(dsk);
        ds_free(dpk);
        ds_free(dpg);
    }
#else
    (void)p;
    std::vector<int> bk(n);
    for (int64_t i = 0; i < n; ++i) {
        bk[i] = bucket_of<K>(Key<T>::enc(in[i], s.arg), g0 + i, vsk.data(), vsg.data(), nspl);
        cnt[bk[i]]++;
    }
    std::vector<int64_t> pos(P, 0);
    for (int b = 0; b < P; ++b)
        bufs[b] = ds_alloc(pbytes(cnt[b]));
    for (int64_t i = 0; i < n; ++i) {
        int b = bk[i];
        char* base = bufs[b];
        reinterpret_cast<K*>(base + koff(cnt[b]))[pos[b]] = Key<T>::enc(in[i], s.arg);
        if (s.arg)
            reinterpret_cast<int64_t*>(base)[pos[b]] = g0 + i;
        pos[b]++;
    }
#endif
    for (int b = 0; b < P; ++b)
        send_ctrl(p, s.first + s.owner[b], s.node->id,
                  make_msg(Hdr{S_COUNTS, s.rank, 0, 0, cnt[b]}));
    for (int b = 0; b < P; ++b) {
        if (b == s.bucket) {
            s.pieces[s.rank] = DsState::Piece{0, s.rank, 0, cnt[b], bufs[b]};
            continue;
        }
        if (cnt[b] > 0)
            send_piece(p, s.first + s.owner[b], s.node->id, 0, s.rank, 0, cnt[b], bufs[b],
                       pbytes(cnt[b]));
        else
            ds_free(bufs[b]);
    }
}

/// All buckets here: sort them and send each output tile its part.
template <typename T>
void sort_bucket(PartitionImpl<1>* p, DsState& s, void* outp) {
    using K = typename Key<T>::K;
    int P = s.P, me = s.rank;
    int64_t nb = s.nb, G = s.G;
    char* kb = ds_alloc(2 * nb * sizeof(K));
    K* k = reinterpret_cast<K*>(kb);
    K* ka = k + nb;
    char* gb = s.arg ? ds_alloc(2 * nb * 8) : nullptr;
    int64_t* g = reinterpret_cast<int64_t*>(gb);
    int64_t* ga = g ? g + nb : nullptr;
    int64_t at = 0;
    for (int i = 0; i < P; ++i) {
        auto& pc = s.pieces[i];
        if (pc.count > 0) {
            ds_copy(p, k + at, pc.buf + (s.arg ? keys_after(pc.count) : 0),
                    pc.count * sizeof(K));
            if (s.arg)
                ds_copy(p, g + at, pc.buf, pc.count * 8);
            at += pc.count;
        }
    }
    bool alt = ds_sort_local<K>(p, k, ka, g, ga, nb);
    K* ks = alt ? ka : k;
    int64_t* gs = alt ? ga : g;
    int64_t osize = s.arg ? 8 : sizeof(T);
    // ranks [G, G + nb) go to the tiles they overlap
    std::vector<std::pair<int, char*>> sends;
    std::vector<std::pair<int64_t, int64_t>> spans;
    for (int c = 0; c < P; ++c) {
        int64_t lo = std::max(G, s.pstart[c]), hi = std::min(G + nb, s.pstart[c + 1]);
        if (lo >= hi)
            continue;
        char* dst = c == me ? static_cast<char*>(outp) + (lo - s.pstart[c]) * osize
                            : ds_alloc((hi - lo) * osize, pe_of(p, s.first + c) != CkMyPe());
        if (s.arg)
            ds_copy(p, dst, gs + (lo - G), (hi - lo) * 8);
        else
            ds_decode<T>(p, ks + (lo - G), hi - lo, reinterpret_cast<T*>(dst));
        if (c != me) {
            sends.push_back({c, dst});
            spans.push_back({lo, hi});
        }
    }
    ds_fence(p);
    for (size_t i = 0; i < sends.size(); ++i)
        send_piece(p, s.first + sends[i].first, s.node->id, 1, me, spans[i].first,
                   spans[i].second - spans[i].first, sends[i].second,
                   (spans[i].second - spans[i].first) * osize);
    for (auto& pc : s.pieces) {
        ds_free(pc.buf);
        pc.buf = nullptr;
    }
    ds_free(kb);
    ds_free(gb);
    s.sorted = true;
}

// ------------------------------------------------------------------ SELECT

constexpr int64_t kMaxCandidates = 1 << 16;

/// Digit width of the next histogram: G * 2^bits counters fit in 40 KB of shared memory.
template <typename T>
int select_bits(int G, int depth) {
    int b = 12;
    while (b > 1 && (int64_t)G * 8 + ((int64_t(G) << b) * 4) > 40 * 1024)
        --b;
    return std::min(b, Key<T>::W - depth);
}

/// Histogram of the next `bits` key bits below each prefix in s.groups (at s.depth).
template <typename T>
void select_hist(PartitionImpl<1>* p, DsState& s, const T* in, int64_t n) {
    using K = typename Key<T>::K;
    constexpr int W = Key<T>::W;
    int G = (int)s.groups.size(), b = s.bits;
    std::vector<K> grp(s.groups.begin(), s.groups.end());
    s.hist.assign((size_t)G << b, 0);
#ifdef USE_NVIDIA
    if (s.host_in.empty()) {
        K* dg = ds_upload(p, grp);
        unsigned* dh = reinterpret_cast<unsigned*>(ds_alloc(s.hist.size() * sizeof(unsigned)));
        hapiCheck(cudaMemsetAsync(dh, 0, s.hist.size() * sizeof(unsigned), ds_stream(p)));
        size_t sm = ((size_t)G * sizeof(K) + 7) / 8 * 8 + s.hist.size() * sizeof(unsigned);
        int grid = std::min(grid_for(n), 8 * ds_dev_attr(cudaDevAttrMultiProcessorCount));
        if (n > 0) {
            dq_hist_kernel<T><<<grid, 256, sm, ds_stream(p)>>>(in, n, s.depth, b, dg, G, dh);
            DS_LAUNCHED();
        }
        ds_to_host(p, s.hist.data(), dh, s.hist.size() * sizeof(unsigned));
        ds_free(dg);
        ds_free(dh);
        return;
    }
#endif
    (void)p;
    K mask = b == W ? K(~K(0)) : K((K(1) << b) - 1);
    for (int64_t i = 0; i < n; ++i) {
        K k = Key<T>::enc(in[i], false);
        int g = group_of<K>(k, W, s.depth, grp.data(), G);
        if (g < 0)
            continue;
        int d = (int)(K(k >> (W - s.depth - b)) & mask);
        s.hist[((size_t)g << b) + d]++;
    }
}

/// Keys of this chare under the prefixes `pre` at depth `depth` (one level below s.groups).
template <typename T>
std::vector<typename Key<T>::K> select_compact(PartitionImpl<1>* p, DsState& s, const T* in,
                                               int64_t n, const std::vector<uint64_t>& pre,
                                               int depth) {
    using K = typename Key<T>::K;
    constexpr int W = Key<T>::W;
    int64_t total = 0;
    for (uint64_t q : pre) {
        uint64_t parent = s.bits == W ? 0 : q >> s.bits;
        auto it = std::lower_bound(s.groups.begin(), s.groups.end(), parent);
        if (it != s.groups.end() && *it == parent)
            total += s.hist[((size_t)(it - s.groups.begin()) << s.bits) +
                            (q & ((uint64_t(1) << s.bits) - 1))];
    }
    std::vector<K> grp(pre.begin(), pre.end());
    std::vector<K> out(total);
#ifdef USE_NVIDIA
    if (s.host_in.empty()) {
        if (total > 0) {
            K* dg = ds_upload(p, grp);
            char* d = ds_alloc(total * sizeof(K) + 8);
            unsigned* cur = reinterpret_cast<unsigned*>(d);
            K* dk = reinterpret_cast<K*>(d + 8);
            hapiCheck(cudaMemsetAsync(cur, 0, 4, ds_stream(p)));
            int grid = std::min(grid_for(n), 8 * ds_dev_attr(cudaDevAttrMultiProcessorCount));
            dq_compact_kernel<T><<<grid, 256, 0, ds_stream(p)>>>(
                in, n, depth, dg, (int)grp.size(), dk, cur);
            DS_LAUNCHED();
            ds_to_host(p, out.data(), dk, total * sizeof(K));
            ds_free(dg);
            ds_free(d);
        }
        return out;
    }
#endif
    (void)p;
    int64_t at = 0;
    for (int64_t i = 0; i < n && at < total; ++i) {
        K k = Key<T>::enc(in[i], false);
        if (group_of<K>(k, W, depth, grp.data(), (int)grp.size()) >= 0)
            out[at++] = k;
    }
    return out;
}

// ------------------------------------------------------------------ axis sorts
// A tile holding whole lines sorts them locally. Otherwise the tiles along the axis
// trade data so that each sorts a share of their lines whole, then send the results
// back: every element crosses the network at most twice, with no splitter rounds.

constexpr int64_t kAxisBatch = int64_t(1) << 25; // elements per segmented sort

#ifdef USE_NVIDIA
template <typename T>
__global__ void ax_gather_kernel(const T* in, AxisGeom g, int64_t l0, int n, bool canon,
                                 typename Key<T>::K* k, int64_t* v) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int64_t l = l0 + i / g.len, p = i % g.len;
    k[i] = Key<T>::enc(in[g.phys(l, p)], canon);
    if (v)
        v[i] = p;
}

template <typename T>
__global__ void ax_scatter_kernel(const typename Key<T>::K* k, const int64_t* v, AxisGeom g,
                                  int64_t l0, int n, void* out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int64_t o = g.phys(l0 + i / g.len, i % g.len);
    if (v)
        static_cast<int64_t*>(out)[o] = v[i];
    else
        static_cast<T*>(out)[o] = Key<T>::dec(k[i]);
}

struct SegOff {
    int64_t len, add;
    DS_FN int operator()(int64_t l) const { return (int)((l + add) * len); }
};
#endif

template <typename T>
void ds_sort_axis_host(const T* in, void* out, const AxisGeom& g, bool arg) {
    using K = typename Key<T>::K;
    std::vector<std::pair<K, int64_t>> line(g.len);
    for (int64_t l = 0; l < g.lines; ++l) {
        for (int64_t q = 0; q < g.len; ++q)
            line[q] = {Key<T>::enc(in[g.phys(l, q)], arg), q};
        std::sort(line.begin(), line.end());
        for (int64_t q = 0; q < g.len; ++q) {
            int64_t o = g.phys(l, q);
            if (arg)
                static_cast<int64_t*>(out)[o] = line[q].second;
            else
                static_cast<T*>(out)[o] = Key<T>::dec(line[q].first);
        }
    }
}

/// SORT/ARGSORT of every line of a block that holds them whole (ARGSORT: stable).
template <int N, typename T>
void ds_sort_axis_local(PartitionImpl<N>* p, const T* in, void* out, AxisGeom g, bool arg) {
    if (g.lines * g.len == 0)
        return;
#if defined(USE_NVIDIA)
    using K = typename Key<T>::K;
    cudaStream_t st = p->compute_stream_raw;
    int64_t bl = std::max<int64_t>(1, std::min<int64_t>(g.lines, kAxisBatch / g.len));
    int64_t cap = bl * g.len;
    char* kb = ds_alloc(2 * cap * sizeof(K));
    char* vb = arg ? ds_alloc(2 * cap * sizeof(int64_t)) : nullptr;
    K* k = reinterpret_cast<K*>(kb);
    int64_t* v = reinterpret_cast<int64_t*>(vb);
    char* tmp = nullptr;
    size_t tcap = 0;
    thrust::counting_iterator<int64_t> seg(0);
    auto b0 = thrust::make_transform_iterator(seg, SegOff{g.len, 0});
    auto b1 = thrust::make_transform_iterator(seg, SegOff{g.len, 1});
    if (cap > INT_MAX)
        CkAbort("axis sort: a line of %lld elements exceeds the segmented sort's int range",
                (long long)g.len);
    for (int64_t l0 = 0; l0 < g.lines; l0 += bl) {
        int nl = (int)std::min(bl, g.lines - l0), n = (int)(nl * g.len); // <= cap
        ax_gather_kernel<T><<<grid_for(n), 256, 0, st>>>(in, g, l0, n, arg, k, v);
        DS_LAUNCHED();
        size_t tb = 0;
        auto run = [&](void* t) {
            return arg ? cub::DeviceSegmentedSort::StableSortPairs(t, tb, k, k + cap, v, v + cap,
                                                                   n, nl, b0, b1, st)
                       : cub::DeviceSegmentedSort::SortKeys(t, tb, k, k + cap, n, nl, b0, b1, st);
        };
        hapiCheck(run(nullptr));
        if (tb > tcap) {
            ds_free(tmp);
            tmp = ds_alloc((int64_t)tb);
            tcap = tb;
        }
        hapiCheck(run(tmp));
        ax_scatter_kernel<T><<<grid_for(n), 256, 0, st>>>(k + cap, arg ? v + cap : nullptr, g, l0,
                                                          n, out);
        DS_LAUNCHED();
    }
    hapiCheck(cudaStreamSynchronize(st));
    ds_free(tmp);
    ds_free(kb);
    ds_free(vb);
#elif defined(USE_KOKKOS)
    int64_t total = g.lines * g.len, osize = arg ? 8 : (int64_t)sizeof(T);
    std::vector<T> hin(total);
    std::vector<char> hout(total * osize);
    ds_to_host(p, hin.data(), in, total * (int64_t)sizeof(T));
    ds_sort_axis_host<T>(hin.data(), hout.data(), g, arg);
    ds_to_dev(p, out, hout.data(), total * osize);
#else
    (void)p;
    ds_sort_axis_host<T>(in, out, g, arg);
#endif
}

struct alignas(16) Bytes16 {
    uint64_t a, b;
};

/// f(E{}) for a type of `esize` bytes (the exchange only moves elements).
template <typename F>
void by_size(int64_t esize, F&& f) {
    switch (esize) {
    case 1: f(uint8_t{}); return;
    case 2: f(uint16_t{}); return;
    case 4: f(uint32_t{}); return;
    case 8: f(uint64_t{}); return;
    case 16: f(Bytes16{}); return;
    default: CkAbort("axis sort: element size %lld", (long long)esize);
    }
}

/// lines[(l - l0) * ld + off + q] <-> tile[g.phys(l, q)] for l in [l0, l1), q < g.len.
template <int N, typename E>
void ax_tile_xfer(PartitionImpl<N>* p, E* tile, AxisGeom g, int64_t l0, int64_t l1, E* lines,
                  int64_t ld, int64_t off, bool to_tile) {
    int64_t n = (l1 - l0) * g.len;
    if (n <= 0)
        return;
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, n), KOKKOS_LAMBDA(int64_t i) {
            int64_t l = l0 + i / g.len, q = i % g.len;
            int64_t a = g.phys(l, q), b = (l - l0) * ld + off + q;
            if (to_tile)
                tile[a] = lines[b];
            else
                lines[b] = tile[a];
        });
#else
    (void)p;
    for (int64_t i = 0; i < n; ++i) {
        int64_t l = l0 + i / g.len, q = i % g.len;
        int64_t a = g.phys(l, q), b = (l - l0) * ld + off + q;
        if (to_tile)
            tile[a] = lines[b];
        else
            lines[b] = tile[a];
    }
#endif
}

/// dst[r * dld + doff + q] = src[r * sld + soff + q] for r < rows, q < w.
template <int N, typename E>
void ax_copy2d(PartitionImpl<N>* p, const E* src, int64_t sld, int64_t soff, E* dst, int64_t dld,
               int64_t doff, int64_t rows, int64_t w) {
    int64_t n = rows * w;
    if (n <= 0)
        return;
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(p, n), KOKKOS_LAMBDA(int64_t i) {
            int64_t r = i / w, q = i % w;
            dst[r * dld + doff + q] = src[r * sld + soff + q];
        });
#else
    (void)p;
    for (int64_t r = 0; r < rows; ++r)
        memcpy(dst + r * dld + doff, src + r * sld + soff, w * sizeof(E));
#endif
}

template <int N>
void ax_tile(PartitionImpl<N>* p, int64_t esize, void* tile, const AxisGeom& g, int64_t l0,
             int64_t l1, void* lines, int64_t ld, int64_t off, bool to_tile) {
    by_size(esize, [&](auto e) {
        using E = decltype(e);
        ax_tile_xfer<N, E>(p, static_cast<E*>(tile), g, l0, l1, static_cast<E*>(lines), ld, off,
                           to_tile);
    });
}

template <int N>
void ax_copy(PartitionImpl<N>* p, int64_t esize, const void* src, int64_t sld, int64_t soff,
             void* dst, int64_t dld, int64_t doff, int64_t rows, int64_t w) {
    by_size(esize, [&](auto e) {
        using E = decltype(e);
        ax_copy2d<N, E>(p, static_cast<const E*>(src), sld, soff, static_cast<E*>(dst), dld, doff,
                        rows, w);
    });
}

template <int N>
auto ax_proxy(PartitionImpl<N>* p, const std::array<int, 3>& c) {
    ChareIndex<N> ci;
    for (int d = 0; d < N; ++d)
        ci.idx[d] = c[d];
    return proxy_at<N>(p->thisProxy, ci);
}

template <int N>
bool ax_on_pe(PartitionImpl<N>* p, const std::array<int, 3>& c) {
    auto t = ax_proxy<N>(p, c);
    return t.ckLocalBranch()->lastKnown(t.ckGetIndex()) == CkMyPe();
}

/// Hands `buf` (ownership) to chare c: by pointer on this PE, else as a data message.
template <int N>
void ax_send(PartitionImpl<N>* p, const std::array<int, 3>& c, int node_id, int phase, int src,
             char* buf, int64_t bytes) {
    auto t = ax_proxy<N>(p, c);
    if (ax_on_pe<N>(p, c)) {
        auto m = make_msg(Hdr{S_PIECE, src, phase, 0, 0, 0, bytes, reinterpret_cast<int64_t>(buf)});
        t.receive_sort_msg(node_id, (int64_t)m.size(), m.data());
        return;
    }
    int region_data[N * 3] = {};
    region_data[0] = src;
#ifdef USE_KOKKOS
    ChareIndex<N> ci;
    for (int d = 0; d < N; ++d)
        ci.idx[d] = c[d];
    device_pack_send<N, N>(p, p->thisProxy, ci, node_id, kDsDataTag + phase, 0, region_data,
                           bytes, buf);
#else
    ct_send_data(t, node_id, kDsDataTag + phase, 0, N, region_data, bytes, buf);
    delete[] buf;
#endif
}

} // namespace ds_detail

template <int N>
bool ds_axis_whole(CTArrayBase<N>* in, int axis) {
    return in->decomp.num_chares(axis) == 1;
}

/// Small arrays split along the axis are sorted on one chare (sort_root).
template <int N>
bool ds_axis_gather(CTArrayBase<N>* in) {
    static const int64_t limit = getenv("CNUM_SORT_GATHER_MAX")
                                     ? atoll(getenv("CNUM_SORT_GATHER_MAX"))
                                     : int64_t(1) << 16;
    int64_t total = 1;
    for (int d = 0; d < N; ++d)
        total *= in->decomp.global_shape[d];
    return total <= limit;
}

template <int N>
void ds_axis_local(PartitionImpl<N>* p, CTArrayBase<N>* in, CTArrayBase<N>* out, int axis,
                   bool arg) {
    DS_SCOPE(p);
    AxisGeom g;
    g.nd = N;
    g.axis = axis;
    for (int d = 0; d < 3; ++d)
        g.sizes[d] = d < N ? in->region.size(d) : 1;
    g.strides[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        g.strides[d] = g.strides[d + 1] * g.sizes[d + 1];
    g.len = g.sizes[axis];
    g.lines = in->local_size() / std::max<int64_t>(1, g.len);
    cn_dispatch(in->dtype, [&](auto tag) {
        using T = decltype(tag);
        ds_sort_axis_local<N, T>(p, static_cast<const T*>(in->device_data_ptr()),
                                 out->device_data_ptr(), g, arg);
    });
}

#define DS_AXIS_INST(N)                                                                            \
    template bool ds_axis_whole<N>(CTArrayBase<N>*, int);                                          \
    template bool ds_axis_gather<N>(CTArrayBase<N>*);                                              \
    template void ds_axis_local<N>(PartitionImpl<N>*, CTArrayBase<N>*, CTArrayBase<N>*, int, bool);
DS_AXIS_INST(1)
DS_AXIS_INST(2)
DS_AXIS_INST(3)

template <int N>
void ArrayDAGExecutorND<N>::execute_ax_node(DAGNode* node, int in_name, int out_name, int axis,
                                            bool arg) {
    if constexpr (N == 1) {
        CkAbort("execute_ax_node: N-D partitions only");
    } else {
        DS_SCOPE(partition);
        auto& slot = ds_state[node->id];
        if (!slot)
            slot = std::make_shared<DsState>();
        auto keep = slot;
        DsState& s = *keep;
        CTArrayBase<N>* in = partition->arrays.at(in_name);
        const ArrayDecomp<N>& dec = in->decomp;
        auto me = partition->nd_index();
        s.node = node;
        s.op = arg ? Opcode::ARGSORT : Opcode::SORT;
        s.arg = arg;
        s.dt = in->dtype;
        s.in_name = in_name;
        s.out_name = out_name;
        s.esize = in->elem_size();
        int first = dec.offset[axis] / dec.tile, P = dec.num_chares(axis);
        s.first = first;
        s.P = P;
        s.rank = me[axis] - first;
        s.grp.assign(P, {0, 0, 0});
        s.pstart.assign(P + 1, dec.global_shape[axis]);
        for (int r = 0; r < P; ++r) {
            for (int d = 0; d < N; ++d)
                s.grp[r][d] = d == axis ? first + r : me[d];
            s.pstart[r] = dec.chare_start_global(axis, first + r) - dec.offset[axis];
        }
        AxisGeom& g = s.tg;
        g.nd = N;
        g.axis = axis;
        for (int d = 0; d < 3; ++d)
            g.sizes[d] = d < N ? in->region.size(d) : 1;
        g.strides[N - 1] = 1;
        for (int d = N - 2; d >= 0; --d)
            g.strides[d] = g.strides[d + 1] * g.sizes[d + 1];
        g.len = g.sizes[axis];
        g.lines = in->local_size() / g.len;
        s.lst.resize(P + 1);
        for (int r = 0; r <= P; ++r)
            s.lst[r] = g.lines * r / P;
        int64_t L = s.pstart[P], mine = s.lst[s.rank + 1] - s.lst[s.rank];
        s.need1 = mine > 0 ? P - 1 : 0;
        for (int r = 0; r < P; ++r)
            s.need2 += r != s.rank && s.lst[r + 1] > s.lst[r];
        if (mine > 0)
            s.lines = ds_alloc(mine * L * s.esize);
        char* ip = static_cast<char*>(in->device_data_ptr());
        std::vector<std::pair<int, char*>> sends;
        for (int r = 0; r < P; ++r) {
            if (s.lst[r + 1] == s.lst[r])
                continue;
            if (r == s.rank) {
                ax_tile(partition, s.esize, ip, g, s.lst[r], s.lst[r + 1], s.lines, L,
                        s.pstart[r], false);
                continue;
            }
            char* buf = ds_alloc((s.lst[r + 1] - s.lst[r]) * g.len * s.esize,
                                 !ax_on_pe(partition, s.grp[r]));
            ax_tile(partition, s.esize, ip, g, s.lst[r], s.lst[r + 1], buf, g.len, 0, false);
            sends.push_back({r, buf});
        }
        ds_fence(partition);
        for (auto& sb : sends)
            ax_send(partition, s.grp[sb.first], node->id, 0, s.rank, sb.second,
                    (s.lst[sb.first + 1] - s.lst[sb.first]) * g.len * s.esize);
        s.started = true;
        std::vector<DsState::Piece> early;
        early.swap(s.early);
        for (auto& pc : early)
            ax_take_piece(node->id, pc.phase, pc.src, pc.buf, pc.count);
        if (ds_state.count(node->id))
            ax_progress(node->id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::ax_take_piece(int node_id, int phase, int src, char* buf,
                                          int64_t bytes) {
    if constexpr (N != 1) {
        DS_SCOPE(partition);
        auto& slot = ds_state[node_id];
        if (!slot)
            slot = std::make_shared<DsState>();
        auto keep = slot;
        DsState& s = *keep;
        if (!s.started) {
            s.early.push_back(DsState::Piece{phase, src, 0, bytes, buf});
            return;
        }
        int me = s.rank;
        if (phase == 0) {
            int64_t w = s.pstart[src + 1] - s.pstart[src];
            ax_copy(partition, s.esize, buf, w, 0, s.lines, s.pstart[s.P], s.pstart[src],
                    s.lst[me + 1] - s.lst[me], w);
            ++s.got1;
        } else {
            CTArrayBase<N>* out = partition->arrays.at(s.out_name);
            ax_tile(partition, out->elem_size(), out->device_data_ptr(), s.tg, s.lst[src],
                    s.lst[src + 1], buf, s.tg.len, 0, true);
            ++s.got2;
        }
        ds_fence(partition);
        ds_free(buf);
        ax_progress(node_id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::ax_progress(int node_id) {
    if constexpr (N != 1) {
        DS_SCOPE(partition);
        auto keep = ds_state.at(node_id);
        DsState& s = *keep;
        int me = s.rank, P = s.P;
        int64_t L = s.pstart[P];
        CTArrayBase<N>* out = partition->arrays.at(s.out_name);
        if (!s.sorted && s.got1 == s.need1) {
            s.sorted = true;
            int64_t mine = s.lst[me + 1] - s.lst[me];
            if (mine > 0) {
                int64_t osize = out->elem_size();
                char* res = ds_alloc(mine * L * osize);
                cn_dispatch(s.dt, [&](auto tag) {
                    using T = decltype(tag);
                    ds_sort_axis_local<N, T>(partition, reinterpret_cast<const T*>(s.lines), res,
                                             block_geom(mine, L), s.arg);
                });
                ds_free(s.lines);
                s.lines = nullptr;
                std::vector<std::pair<int, char*>> sends;
                for (int r = 0; r < P; ++r) {
                    int64_t w = s.pstart[r + 1] - s.pstart[r];
                    if (r == me) {
                        ax_tile(partition, osize, out->device_data_ptr(), s.tg, s.lst[me],
                                s.lst[me + 1], res, L, s.pstart[me], true);
                        continue;
                    }
                    char* buf = ds_alloc(mine * w * osize, !ax_on_pe(partition, s.grp[r]));
                    ax_copy(partition, osize, res, L, s.pstart[r], buf, w, 0, mine, w);
                    sends.push_back({r, buf});
                }
                ds_fence(partition);
                ds_free(res);
                for (auto& sb : sends)
                    ax_send(partition, s.grp[sb.first], node_id, 1, me, sb.second,
                            mine * (s.pstart[sb.first + 1] - s.pstart[sb.first]) * osize);
            }
        }
        if (s.sorted && s.got2 == s.need2) {
            ds_fence(partition);
            ds_state.erase(node_id);
            node_finished(node_id);
        }
    }
}

// ------------------------------------------------------------------ executor

namespace {

ASTNode* ds_root(DAGNode* node) {
    for (ASTNode* root : node->ast->roots) {
        Opcode op = static_cast<Opcode>(root->opcode);
        if (op == Opcode::SORT || op == Opcode::ARGSORT || op == Opcode::SELECT)
            return root;
    }
    return nullptr;
}

} // namespace

/// Used by execute_castscansort_node: 1-D SORT/ARGSORT and SELECT.
bool ds_handles(ASTNode* root, DType dt, int ndims) {
    (void)dt;
    Opcode op = static_cast<Opcode>(root->opcode);
    if (op == Opcode::SELECT)
        return true;
    if (ndims != 1)
        return false;
#if defined(USE_KOKKOS) && !defined(USE_NVIDIA)
    return false;
#else
    return op == Opcode::SORT || op == Opcode::ARGSORT;
#endif
}

template <int N>
void ArrayDAGExecutorND<N>::execute_ds_node(DAGNode* node) {
    DS_SCOPE(partition);
    if constexpr (N != 1) {
        CkAbort("execute_ds_node: 1-D partitions only");
    } else {
        auto* g = static_cast<ArrayDAGGroup*>(group);
        ASTNode* root = ds_root(node);
        auto st = std::make_shared<DsState>();
        auto it = ds_state.find(node->id);
        if (it != ds_state.end())
            st = it->second;
        else
            ds_state[node->id] = st;
        DsState& s = *st;
        s.node = node;
        s.op = static_cast<Opcode>(root->opcode);
        s.in_name = root->operands[0]->result_name;
        s.out_name = root->result_name;
        ArrayDecomp<1> dec = g->array_meta.at(s.in_name).template decomp<1>();
        Layout L = layout_of(dec);
        s.first = L.first;
        s.P = L.P;
        s.pstart = L.start;
        int me = partition->nd_index()[0];
        s.rank = (me >= s.first && me < s.first + s.P) ? me - s.first : -1;

        if (s.op == Opcode::SELECT) {
            ArrayDecomp<1> odec = g->array_meta.at(s.out_name).template decomp<1>();
            s.result_chare = odec.offset[0] / odec.tile;
            s.coord = s.first;
            for (size_t i = 1; i < root->operands.size(); ++i)
                s.want.push_back((int64_t)root->operands[i]->scalar);
            s.dt = root->dtype;
            if (me == s.result_chare) {
                ArrayRegion<1> r = odec.chare_region_local({me});
                partition->ensure_array(r, odec.global_shape, s.out_name, s.dt, odec, false);
            }
            if (s.rank < 0 && me != s.result_chare) {
                ds_state.erase(node->id);
                node_finished(node->id);
                return;
            }
        } else {
            if (s.rank < 0 || s.pstart[s.P] == 0) {
                ds_state.erase(node->id);
                node_finished(node->id);
                return;
            }
            CTArrayBase<1>* in = partition->arrays.at(s.in_name);
            s.dt = in->dtype;
            s.arg = s.op == Opcode::ARGSORT;
            partition->ensure_array(in->region, in->global_shape, s.out_name,
                                    s.arg ? DType::INT64 : in->dtype, in->decomp, false);
        }
        s.started = true;
        s.pieces.assign(s.P, DsState::Piece{});
        s.col.assign(s.P, 0);
        s.nbs.assign(s.P, 0);
        ds_progress_start(node->id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::ds_progress_start(int node_id) {
    if constexpr (N == 1) {
        auto st = ds_state.at(node_id);
        DsState& s = *st;
        CTArrayBase<1>* in = nullptr;
        if (s.rank >= 0)
            in = partition->arrays.at(s.in_name);
        cn_dispatch(s.dt, [&](auto tag) {
            using T = decltype(tag);
            {
                const T* ip = in ? static_cast<const T*>(in->device_data_ptr()) : nullptr;
                int64_t n = in ? in->local_size() : 0;
                if (s.op == Opcode::SELECT) {
                    if constexpr (cn_is_complex_v<T>) {
                        CkAbort("select: complex dtype");
                    } else if (s.rank >= 0) {
#if defined(USE_KOKKOS) && !defined(USE_NVIDIA)
                        s.host_in.resize(n * sizeof(T));
                        Kokkos::deep_copy(
                            Kokkos::View<char*, HostSpace>(s.host_in.data(), n * sizeof(T)),
                            Kokkos::View<const char*, DeviceSpace,
                                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                                reinterpret_cast<const char*>(ip), n * sizeof(T)));
#endif
                        s.depth = 0;
                        s.groups = {0};
                        s.bits = select_bits<T>(1, 0);
                        select_hist<T>(partition, s, ds_in<T>(s, ip), n);
                        auto msg = make_msg(Hdr{Q_HIST, s.rank, s.depth, s.bits, n, 0, 0, 0});
                        put(msg, s.hist.data(), (int64_t)s.hist.size());
                        send_ctrl(partition, s.coord, node_id, msg);
                    }
                } else if (s.P == 1) {
                    sort_single<T>(partition, s, ip,
                                   partition->arrays.at(s.out_name)->device_data_ptr(), n);
                    ds_state.erase(node_id);
                    node_finished(node_id);
                    return;
                } else {
                    sort_send_samples<T>(partition, s, ip, n);
                }
            }
        });
        if (!ds_state.count(node_id))
            return;
        std::vector<std::vector<char>> msgs;
        msgs.swap(s.stash);
        for (auto& m : msgs)
            on_ds_msg(node_id, m.data(), (int64_t)m.size());
        std::vector<DsState::Piece> early;
        early.swap(s.early);
        for (auto& pc : early)
            ds_take_piece(node_id, pc.phase, pc.src, pc.start, pc.count, pc.buf);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::on_ds_data(int node_id, int phase, const int* region_data, char* buf,
                                       int64_t size) {
    if constexpr (N == 1) {
        int64_t v = (int64_t)(uint32_t)region_data[1] | ((int64_t)region_data[2] << 32);
        if (phase == 0) {
            ds_take_piece(node_id, phase, region_data[0], 0, v, buf);
        } else {
            // phase 1 follows this chare's own start, so its output array exists
            DsState& s = *ds_state.at(node_id);
            ds_take_piece(node_id, phase, region_data[0], v,
                          size / partition->arrays.at(s.out_name)->elem_size(), buf);
        }
    } else
        ax_take_piece(node_id, phase, region_data[0], buf, size);
}

template <int N>
void ArrayDAGExecutorND<N>::ds_take_piece(int node_id, int phase, int src, int64_t start,
                                          int64_t count, char* buf) {
    DS_SCOPE(partition);
    DsState::Piece pc{phase, src, start, count, buf};
    auto& st = ds_state[node_id];
    if (!st)
        st = std::make_shared<DsState>();
    DsState& s = *st;
    if (!s.started) {
        s.early.push_back(pc);
        return;
    }
    if (pc.phase == 0) {
        s.pieces[pc.src] = pc;
    } else {
        s.rebal.push_back(pc);
        s.rebal_got += pc.count;
    }
    ds_sort_progress(node_id);
}

template <int N>
void ArrayDAGExecutorND<N>::on_ds_msg(int node_id, char* data, int64_t size) {
    DS_SCOPE(partition);
    if constexpr (N != 1) {
        Hdr h;
        memcpy(&h, data, sizeof(h));
        if (h.kind != S_PIECE)
            CkAbort("on_ds_msg: unexpected message %d on a %d-D partition", h.kind, N);
        ax_take_piece(node_id, h.phase, h.src, reinterpret_cast<char*>(h.d), h.c);
        return;
    }
    auto& st = ds_state[node_id];
    if (!st)
        st = std::make_shared<DsState>();
    auto keep = st;
    DsState& s = *keep;
    if (!s.started) {
        s.stash.emplace_back(data, data + size);
        return;
    }
    Hdr h;
    memcpy(&h, data, sizeof(h));
    if constexpr (N == 1) {
        switch (h.kind) {
        case S_PIECE:
            ds_take_piece(node_id, h.phase, h.src, h.a, h.b, reinterpret_cast<char*>(h.d));
            return;
        case S_SAMPLES:
            s.samples.emplace_back(data, data + size);
            if (++s.samples_got == s.P)
                cn_dispatch(s.dt, [&](auto tag) {
                    sort_make_splitters<decltype(tag)>(partition, s);
                });
            return;
        case S_SPLIT: {
            std::vector<char> m(data, data + size);
            CTArrayBase<1>* in = partition->arrays.at(s.in_name);
            cn_dispatch(s.dt, [&](auto tag) {
                using T = decltype(tag);
                sort_partition<T>(partition, s, static_cast<const T*>(in->device_data_ptr()),
                                  in->local_size(), m);
            });
            s.split_done = true;
            ds_sort_progress(node_id);
            return;
        }
        case S_COUNTS:
            s.col[h.src] = h.a;
            s.nb += h.a;
            if (++s.ncol == s.P)
                send_ctrl(partition, s.first, node_id,
                          make_msg(Hdr{S_NB, s.rank, s.bucket, 0, s.nb}));
            return;
        case S_NB:
            s.nbs[h.phase] = h.a;
            if (++s.nbs_got == (int)s.owner.size()) {
                int64_t G = 0;
                for (size_t j = 0; j < s.owner.size(); ++j) {
                    send_ctrl(partition, s.first + s.owner[j], node_id,
                              make_msg(Hdr{S_G, s.rank, 0, 0, G}));
                    G += s.nbs[j];
                }
            }
            return;
        case S_G:
            s.G = h.a;
            ds_sort_progress(node_id);
            return;
        default:
            ds_select_msg(node_id, data, size);
        }
    }
}

template <int N>
void ArrayDAGExecutorND<N>::ds_sort_progress(int node_id) {
    if constexpr (N == 1) {
        auto keep = ds_state.at(node_id);
        DsState& s = *keep;
        if (!s.split_done || (s.bucket >= 0 && s.G < 0))
            return;
        int P = s.P, me = s.rank;
        if (s.bucket >= 0 && !s.sorted) {
            for (int i = 0; i < P; ++i)
                if (s.col[i] > 0 && s.pieces[i].buf == nullptr)
                    return;
            void* outp = partition->arrays.at(s.out_name)->device_data_ptr();
            cn_dispatch(s.dt, [&](auto tag) { sort_bucket<decltype(tag)>(partition, s, outp); });
        }
        // elements of my tile that other buckets hold
        int64_t G = s.G, nb = s.nb;
        int64_t t0 = s.pstart[me], t1 = s.pstart[me + 1];
        int64_t own = s.bucket < 0 ? 0
                                   : std::max<int64_t>(0, std::min(G + nb, t1) - std::max(G, t0));
        if (s.rebal_got < (t1 - t0) - own)
            return;
        CTArrayBase<1>* out = partition->arrays.at(s.out_name);
        char* outp = static_cast<char*>(out->device_data_ptr());
        int64_t osize = out->elem_size();
        for (auto& pc : s.rebal)
            ds_copy(partition, outp + (pc.start - t0) * osize, pc.buf, pc.count * osize);
        ds_fence(partition);
        for (auto& pc : s.rebal)
            ds_free(pc.buf);
        ds_state.erase(node_id);
        node_finished(node_id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::ds_select_msg(int node_id, char* data, int64_t size) {
    if constexpr (N == 1) {
        Hdr h;
        memcpy(&h, data, sizeof(h));
        auto keep = ds_state.at(node_id);
        DsState& s = *keep;
        int me = partition->nd_index()[0];
        cn_dispatch(s.dt, [&](auto tag) {
            using T = decltype(tag);
            if constexpr (!cn_is_complex_v<T>) {
                using K = typename Key<T>::K;
                constexpr int W = Key<T>::W;
                CTArrayBase<1>* in = s.rank >= 0 ? partition->arrays.at(s.in_name) : nullptr;
                const T* ip = in ? ds_in<T>(s, static_cast<const T*>(in->device_data_ptr()))
                                 : nullptr;
                int64_t n = in ? in->local_size() : 0;
                auto finish_participant = [&]() {
                    if (me != s.coord && me != s.result_chare) {
                        ds_state.erase(node_id);
                        node_finished(node_id);
                    }
                };
                switch (h.kind) {
                case Q_REQ: {
                    const uint64_t* q = reinterpret_cast<const uint64_t*>(data + sizeof(h));
                    s.groups.assign(q, q + h.a);
                    s.depth = h.phase;
                    s.bits = h.ints;
                    select_hist<T>(partition, s, ip, n);
                    auto msg = make_msg(Hdr{Q_HIST, s.rank, s.depth, s.bits, n, 0, 0, 0});
                    put(msg, s.hist.data(), (int64_t)s.hist.size());
                    send_ctrl(partition, s.coord, node_id, msg);
                    return;
                }
                case Q_GATHER: {
                    const uint64_t* q = reinterpret_cast<const uint64_t*>(data + sizeof(h));
                    std::vector<uint64_t> pre(q, q + h.a);
                    auto keys = select_compact<T>(partition, s, ip, n, pre, h.phase);
                    auto msg = make_msg(Hdr{Q_CANDS, s.rank, 0, 0, (int64_t)keys.size(), 0, 0, 0});
                    put(msg, keys.data(), (int64_t)keys.size());
                    send_ctrl(partition, s.coord, node_id, msg);
                    finish_participant();
                    return;
                }
                case Q_DONE:
                    finish_participant();
                    return;
                case Q_RESULT: {
                    CTArrayBase<1>* out = partition->arrays.at(s.out_name);
                    const T* v = reinterpret_cast<const T*>(data + sizeof(h));
                    ds_to_dev(partition, out->device_data_ptr(), v, h.a * sizeof(T));
                    ds_state.erase(node_id);
                    node_finished(node_id);
                    return;
                }
                case Q_HIST:
                case Q_CANDS:
                    break;
                }
                // coordinator
                s.replies.emplace_back(data, data + size);
                if (++s.replies_got < s.P)
                    return;
                s.replies_got = 0;
                std::vector<std::vector<char>> reps;
                reps.swap(s.replies);
                bool first_round = s.uranks.empty() && h.kind == Q_HIST && s.prefix.empty();
                if (first_round) {
                    s.uranks = s.want;
                    std::sort(s.uranks.begin(), s.uranks.end());
                    s.uranks.erase(std::unique(s.uranks.begin(), s.uranks.end()),
                                   s.uranks.end());
                    s.resid = s.uranks;
                    s.prefix.assign(s.uranks.size(), 0);
                }
                std::vector<K> vals(s.uranks.size());
                bool done = false;
                if (h.kind == Q_HIST) {
                    Hdr h0;
                    memcpy(&h0, reps[0].data(), sizeof(h0));
                    int depth = h0.phase, b = h0.ints;
                    size_t nbins = s.groups.size() << b;
                    std::vector<int64_t> sum(nbins, 0);
                    for (auto& r : reps) {
                        auto* hh = reinterpret_cast<const unsigned*>(r.data() + sizeof(Hdr));
                        for (size_t i = 0; i < nbins; ++i)
                            sum[i] += hh[i];
                    }
                    int64_t cand = 0;
                    std::vector<uint64_t> next;
                    for (size_t t = 0; t < s.uranks.size(); ++t) {
                        size_t gi = std::lower_bound(s.groups.begin(), s.groups.end(),
                                                     s.prefix[t]) - s.groups.begin();
                        int64_t acc = 0;
                        int d = 0;
                        if (gi == s.groups.size())
                            CkAbort("SELECT: unknown prefix");
                        for (; d < (1 << b); ++d) {
                            int64_t c = sum[(gi << b) + d];
                            if (s.resid[t] < acc + c)
                                break;
                            acc += c;
                        }
                        if (d == (1 << b))
                            CkAbort("SELECT: rank %lld not in the histogram",
                                    (long long)s.uranks[t]);
                        s.resid[t] -= acc;
                        s.prefix[t] = (b == 64 ? 0 : (s.prefix[t] << b)) | (uint64_t)d;
                        if (next.empty() || next.back() != s.prefix[t]) {
                            if (std::find(next.begin(), next.end(), s.prefix[t]) == next.end()) {
                                next.push_back(s.prefix[t]);
                                cand += sum[(gi << b) + d];
                            }
                        }
                    }
                    std::sort(next.begin(), next.end());
                    int nd = depth + b;
                    if (nd == W) {
                        for (size_t t = 0; t < s.uranks.size(); ++t)
                            vals[t] = K(s.prefix[t]);
                        done = true;
                        auto msg = make_msg(Hdr{Q_DONE, -1, 0, 0, 0, 0, 0, 0});
                        for (int i = 0; i < s.P; ++i)
                            if (s.first + i != me)
                                send_ctrl(partition, s.first + i, node_id, msg);
                    } else if (cand <= kMaxCandidates) {
                        int64_t G = (int64_t)next.size();
                        auto msg = make_msg(Hdr{Q_GATHER, -1, nd, 0, G, 0, 0, 0});
                        put(msg, next.data(), (int64_t)next.size());
                        s.depth = nd;
                        for (int i = 0; i < s.P; ++i)
                            send_ctrl(partition, s.first + i, node_id, msg);
                        return;
                    } else {
                        int nb2 = select_bits<T>((int)next.size(), nd);
                        auto msg = make_msg(Hdr{Q_REQ, -1, nd, nb2, (int64_t)next.size(), 0, 0, 0});
                        put(msg, next.data(), (int64_t)next.size());
                        s.groups = next;
                        for (int i = 0; i < s.P; ++i)
                            send_ctrl(partition, s.first + i, node_id, msg);
                        return;
                    }
                } else {
                    std::vector<K> cands;
                    for (auto& r : reps) {
                        Hdr hr;
                        memcpy(&hr, r.data(), sizeof(hr));
                        const K* k = reinterpret_cast<const K*>(r.data() + sizeof(Hdr));
                        cands.insert(cands.end(), k, k + hr.a);
                    }
                    std::sort(cands.begin(), cands.end());
                    int sh = W - s.depth;
                    for (size_t t = 0; t < s.uranks.size(); ++t) {
                        // candidates are sorted, so a prefix is a contiguous run
                        auto lo = std::partition_point(cands.begin(), cands.end(), [&](K k) {
                            return (uint64_t)(sh == W ? 0 : (k >> sh)) < s.prefix[t];
                        });
                        if (cands.end() - lo <= s.resid[t])
                            CkAbort("SELECT: missing candidates");
                        vals[t] = *(lo + s.resid[t]);
                    }
                    done = true;
                }
                if (!done)
                    return;
                int64_t m = (int64_t)s.want.size();
                auto msg = make_msg(Hdr{Q_RESULT, -1, 0, 0, m, 0, 0, 0});
                msg.resize(sizeof(Hdr) + m * sizeof(T));
                for (int64_t i = 0; i < m; ++i) {
                    size_t t = std::lower_bound(s.uranks.begin(), s.uranks.end(), s.want[i]) -
                               s.uranks.begin();
                    T v = Key<T>::dec(vals[t]);
                    memcpy(msg.data() + sizeof(Hdr) + i * sizeof(T), &v, sizeof(T));
                }
                if (s.result_chare != me) {
                    send_ctrl(partition, s.result_chare, node_id, msg);
                    ds_state.erase(node_id);
                    node_finished(node_id);
                } else {
                    ds_select_msg(node_id, msg.data(), (int64_t)msg.size());
                }
            }
        });
    }
}

template <int N>
void PartitionImpl<N>::receive_sort_msg(int node_id, int64_t size, char* data) {
    executor->on_ds_msg(node_id, data, size);
}

#define DS_INST(N)                                                                                 \
    template void ArrayDAGExecutorND<N>::execute_ds_node(DAGNode*);                                \
    template void ArrayDAGExecutorND<N>::on_ds_msg(int, char*, int64_t);                           \
    template void ArrayDAGExecutorND<N>::on_ds_data(int, int, const int*, char*, int64_t);         \
    template void ArrayDAGExecutorND<N>::execute_ax_node(DAGNode*, int, int, int, bool);           \
    template void PartitionImpl<N>::receive_sort_msg(int, int64_t, char*);
DS_INST(1)
DS_INST(2)
DS_INST(3)
