#ifdef USE_KOKKOS
// Before the Charm++ headers: their ALIGN_BYTES macro breaks cub/thrust.
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#ifdef USE_NVIDIA
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#endif
#endif

#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef USE_KOKKOS
#define CSS_FN KOKKOS_INLINE_FUNCTION
#else
#define CSS_FN inline
#endif

// SCAN / SORT / ARGSORT executors (opcodes 81-83). The result has the
// input's decomposition (pinned in dag_group_receive.cpp). 1-D sorts of real
// types and SELECT run in executor_sort.cpp; the other sorts gather to the
// chare holding element 0.

// Named namespace: these types are captured by device lambdas.
namespace castscansort_detail {

ASTNode* castscansort_root(DAGNode* node) {
    for (ASTNode* root : node->ast->roots)
        if (is_castscansort(static_cast<Opcode>(root->opcode)))
            return root;
    return nullptr;
}

/// Integer scalar operand `idx` of the root (axis, scan kind).
int scalar_param(ASTNode* root, int idx, int dflt) {
    if ((int)root->operands.size() > idx && root->operands[idx]->is_scalar)
        return (int)root->operands[idx]->scalar;
    return dflt;
}

/// Row-major block viewed as `lines` lines of length `len` along `axis`.
template <int N>
struct LineGeom {
    int64_t sizes[N];
    int64_t strides[N];
    int axis;
    int64_t len;
    int64_t lines;

    LineGeom(const int64_t* sizes_, int axis_) : axis(axis_) {
        for (int d = 0; d < N; ++d)
            sizes[d] = sizes_[d];
        strides[N - 1] = 1;
        for (int d = N - 2; d >= 0; --d)
            strides[d] = strides[d + 1] * sizes[d + 1];
        len = sizes[axis];
        lines = 1;
        for (int d = 0; d < N; ++d)
            if (d != axis)
                lines *= sizes[d];
    }

    CSS_FN int64_t phys(int64_t line, int64_t pos) const {
        int64_t idx = pos * strides[axis];
        int64_t rem = line;
        for (int d = N - 1; d >= 0; --d) {
            if (d == axis)
                continue;
            idx += (rem % sizes[d]) * strides[d];
            rem /= sizes[d];
        }
        return idx;
    }
};

template <int N>
LineGeom<N> tile_geom(CTArrayBase<N>* arr, int axis) {
    int64_t sizes[N];
    for (int d = 0; d < N; ++d)
        sizes[d] = arr->region.size(d);
    return LineGeom<N>(sizes, axis);
}

template <typename T>
struct ScanOp {
    int kind; // 0 cumsum, 1 cumprod, 2 nancumsum, 3 nancumprod

    CSS_FN bool is_mul() const { return kind == 1 || kind == 3; }
    CSS_FN T identity() const { return is_mul() ? T(1) : T(0); }
    // nan*: NaN acts as the identity.
    CSS_FN T input(T x) const { return (kind >= 2 && cn_isnan(x)) ? identity() : x; }
    CSS_FN T apply(T a, T b) const { return is_mul() ? T(a * b) : T(a + b); }
};

/// Order of flat indices: line, then key (NumPy order, NaNs last), then position (stable).
template <typename T>
struct SortLess {
    const T* keys;
    int64_t axis_stride;
    int64_t axis_size;

    CSS_FN int64_t line_key(int64_t i) const {
        return i - ((i / axis_stride) % axis_size) * axis_stride;
    }
    CSS_FN bool operator()(int64_t a, int64_t b) const {
        int64_t la = line_key(a), lb = line_key(b);
        if (la != lb)
            return la < lb;
        T x = keys[a], y = keys[b];
        if (cn_sort_lt(x, y))
            return true;
        if (cn_sort_lt(y, x))
            return false;
        return a < b;
    }
};

/// Array-index region of chare ci.
template <int N>
ArrayRegion<N> chare_array_region(const ArrayDecomp<N>& dec, const std::array<int, N>& ci) {
    return dec.to_local(dec.chare_region_global(ci));
}

/// All chares that hold part of an array with decomposition dec.
template <int N>
std::vector<std::array<int, N>> nonempty_chares(const ArrayDecomp<N>& dec) {
    std::array<int, N> first, last, ci;
    for (int d = 0; d < N; ++d) {
        first[d] = dec.offset[d] / dec.tile;
        last[d] = (dec.offset[d] + dec.global_shape[d] - 1) / dec.tile;
        ci[d] = first[d];
    }
    std::vector<std::array<int, N>> out;
    while (true) {
        out.push_back(ci);
        int d = N - 1;
        for (; d >= 0; --d) {
            if (++ci[d] <= last[d])
                break;
            ci[d] = first[d];
        }
        if (d < 0)
            break;
    }
    return out;
}

template <int N>
void region_to_data(const ArrayRegion<N>& r, int* region_data) {
    for (int d = 0; d < N; ++d) {
        region_data[d * 3 + 0] = r.start[d];
        region_data[d * 3 + 1] = r.stop[d];
        region_data[d * 3 + 2] = 1;
    }
}

// ---- device/host buffer helpers ----

template <int N>
void copy_bytes(PartitionImpl<N>* partition, char* dst, const char* src, int64_t bytes) {
#ifdef USE_KOKKOS
    using U = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<char*, DeviceSpace, U> d(dst, bytes);
    Kokkos::View<const char*, DeviceSpace, U> s(src, bytes);
#ifdef USE_NVIDIA
    Kokkos::deep_copy(partition->compute_exec, d, s);
#else
    Kokkos::deep_copy(d, s);
#endif
#else
    (void)partition;
    std::memcpy(dst, src, bytes);
#endif
}

/// Send a ready alloc_buffer() buffer to `target`; takes ownership.
template <int N>
void send_buffer(PartitionImpl<N>* partition, const std::array<int, N>& target, int node_id,
                 int input_index, int name, const ArrayRegion<N>& region, char* buf,
                 int64_t bytes) {
    ChareIndex<N> ci;
    for (int d = 0; d < N; ++d)
        ci.idx[d] = target[d];
    int region_data[N * 3];
    region_to_data<N>(region, region_data);
#ifndef NDEBUG
    partition->comm_bytes_sent += bytes;
#endif
#ifdef USE_KOKKOS
    device_pack_send<N, N>(partition, partition->thisProxy, ci, node_id, input_index, name,
                           region_data, bytes, buf);
#else
    proxy_at<N>(partition->thisProxy, ci)
        .receive_data(node_id, input_index, name, N, region_data, bytes, buf);
    delete[] buf;
#endif
}

/// Small message/scratch buffers: pooled device memory in GPU builds. Free only
/// after the GPU is done with them (comm_free reuses them at once).
inline char* pooled_alloc(int64_t bytes) {
#ifdef USE_KOKKOS
    return static_cast<char*>(comm_alloc(std::max<int64_t>(bytes, 8)));
#else
    return alloc_buffer(bytes);
#endif
}

inline void pooled_free(char* p) {
#ifdef USE_KOKKOS
    comm_free(p);
#else
    free_buffer(p);
#endif
}

// ---- SCAN ----
// Each chare sends its line totals to the later chares along the axis, then scans
// its tile once with the folded carry. The input may be a narrower integer type
// than the result (cast on the fly).

template <typename Tout, typename Tin>
struct ScanIn {
    const Tin* in;
    int64_t base, stride;
    ScanOp<Tout> op;
    const Tout* carry;

    CSS_FN Tout operator()(int64_t pos) const {
        Tout x = op.input(cn_cast<Tout>(in[base + pos * stride]));
        if (carry != nullptr && pos == 0)
            x = op.apply(*carry, x);
        return x;
    }
};

template <typename T>
struct ScanBin {
    ScanOp<T> op;
    CSS_FN T operator()(const T& a, const T& b) const { return op.apply(a, b); }
};

struct Strided {
    int64_t s;
    CSS_FN int64_t operator()(int64_t i) const { return i * s; }
};

/// totals[l] = op-fold of line l (one thread per line).
template <int N, typename Tout, typename Tin>
void totals_per_thread(PartitionImpl<N>* partition, const Tin* in, Tout* totals,
                       const LineGeom<N>& g, ScanOp<Tout> op) {
    int64_t astride = g.strides[g.axis];
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, g.lines), KOKKOS_LAMBDA(int64_t l) {
            ScanIn<Tout, Tin> f{in, g.phys(l, 0), astride, op, nullptr};
            Tout acc = op.identity();
            for (int64_t p = 0; p < g.len; ++p)
                acc = op.apply(acc, f(p));
            totals[l] = acc;
        });
#else
    (void)partition;
    for (int64_t l = 0; l < g.lines; ++l) {
        ScanIn<Tout, Tin> f{in, g.phys(l, 0), astride, op, nullptr};
        Tout acc = op.identity();
        for (int64_t p = 0; p < g.len; ++p)
            acc = op.apply(acc, f(p));
        totals[l] = acc;
    }
#endif
}

/// out = carry (op) inclusive scan of cast(in), per line (one thread per line).
template <int N, typename Tout, typename Tin>
void scan_per_thread(PartitionImpl<N>* partition, const Tin* in, Tout* out, const Tout* carry,
                     const LineGeom<N>& g, ScanOp<Tout> op) {
    int64_t astride = g.strides[g.axis];
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, g.lines), KOKKOS_LAMBDA(int64_t l) {
            int64_t base = g.phys(l, 0);
            ScanIn<Tout, Tin> f{in, base, astride, op, nullptr};
            Tout acc = carry ? carry[l] : op.identity();
            for (int64_t p = 0; p < g.len; ++p) {
                acc = op.apply(acc, f(p));
                out[base + p * astride] = acc;
            }
        });
#else
    (void)partition;
    for (int64_t l = 0; l < g.lines; ++l) {
        int64_t base = g.phys(l, 0);
        ScanIn<Tout, Tin> f{in, base, astride, op, nullptr};
        Tout acc = carry ? carry[l] : op.identity();
        for (int64_t p = 0; p < g.len; ++p) {
            acc = op.apply(acc, f(p));
            out[base + p * astride] = acc;
        }
    }
#endif
}

#ifdef USE_NVIDIA
/// Few long lines: one CUB reduction / single-pass scan per line.
template <int N, typename Tout, typename Tin>
void lines_cub(PartitionImpl<N>* partition, const Tin* in, Tout* out, Tout* totals,
               const Tout* carry, const LineGeom<N>& g, ScanOp<Tout> op) {
    cudaStream_t st = partition->compute_stream_raw;
    int64_t astride = g.strides[g.axis];
    int len = (int)g.len;
    ScanBin<Tout> bin{op};
    auto it_of = [&](int64_t l) {
        return thrust::make_transform_iterator(
            thrust::counting_iterator<int64_t>(0),
            ScanIn<Tout, Tin>{in, g.phys(l, 0), astride, op, carry ? carry + l : nullptr});
    };
    size_t tb = 0;
    if (totals)
        hapiCheck(cub::DeviceReduce::Reduce(nullptr, tb, it_of(0), totals, len, bin,
                                            op.identity(), st));
    else if (astride == 1)
        hapiCheck(cub::DeviceScan::InclusiveScan(nullptr, tb, it_of(0), out, bin, len, st));
    else
        hapiCheck(cub::DeviceScan::InclusiveScan(
            nullptr, tb, it_of(0),
            thrust::make_permutation_iterator(
                out, thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                                     Strided{astride})),
            bin, len, st));
    char* tmp = pooled_alloc((int64_t)tb);
    for (int64_t l = 0; l < g.lines; ++l) {
        if (totals) {
            hapiCheck(cub::DeviceReduce::Reduce(tmp, tb, it_of(l), totals + l, len, bin,
                                                op.identity(), st));
        } else {
            Tout* o = out + g.phys(l, 0);
            if (astride == 1)
                hapiCheck(cub::DeviceScan::InclusiveScan(tmp, tb, it_of(l), o, bin, len, st));
            else
                hapiCheck(cub::DeviceScan::InclusiveScan(
                    tmp, tb, it_of(l),
                    thrust::make_permutation_iterator(
                        o, thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                                           Strided{astride})),
                    bin, len, st));
        }
    }
    compute_fence(partition);
    pooled_free(tmp);
}
#endif

template <int N, typename Tout, typename Tin>
void line_totals(PartitionImpl<N>* partition, const Tin* in, Tout* totals, const LineGeom<N>& g,
                 ScanOp<Tout> op) {
#ifdef USE_NVIDIA
    if (g.lines < g.len) {
        lines_cub<N, Tout, Tin>(partition, in, nullptr, totals, nullptr, g, op);
        return;
    }
#endif
    totals_per_thread<N, Tout, Tin>(partition, in, totals, g, op);
}

template <int N, typename Tout, typename Tin>
void scan_lines(PartitionImpl<N>* partition, const Tin* in, Tout* out, const Tout* carry,
                const LineGeom<N>& g, ScanOp<Tout> op) {
#ifdef USE_NVIDIA
    if (g.lines < g.len) {
        lines_cub<N, Tout, Tin>(partition, in, out, nullptr, carry, g, op);
        return;
    }
#endif
    scan_per_thread<N, Tout, Tin>(partition, in, out, carry, g, op);
}

/// carry[l] = op(carry[l] or identity, totals[l]).
template <int N, typename T>
void fold_totals(PartitionImpl<N>* partition, T* carry, const T* totals, int64_t lines,
                 ScanOp<T> op, bool init) {
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, lines), KOKKOS_LAMBDA(int64_t l) {
            carry[l] = op.apply(init ? op.identity() : carry[l], totals[l]);
        });
#else
    (void)partition;
    for (int64_t l = 0; l < lines; ++l)
        carry[l] = op.apply(init ? op.identity() : carry[l], totals[l]);
#endif
}

/// Send line totals to later chares. Returns messages to expect.
template <int N, typename Tout, typename Tin>
int scan_execute(PartitionImpl<N>* partition, DAGNode* node, int out_name, CTArrayBase<N>* in,
                 int axis, int kind) {
    LineGeom<N> g = tile_geom<N>(in, axis);
    ScanOp<Tout> op{kind};
    const ArrayDecomp<N>& dec = in->decomp;
    auto me = partition->nd_index();
    int first = dec.offset[axis] / dec.tile;
    int last = (dec.offset[axis] + dec.global_shape[axis] - 1) / dec.tile;
    int n_targets = last - me[axis];
    if (n_targets > 0) {
        int64_t bytes = g.lines * (int64_t)sizeof(Tout);
        char* totals = pooled_alloc(bytes);
        line_totals<N, Tout, Tin>(partition, static_cast<const Tin*>(in->device_data_ptr()),
                                  reinterpret_cast<Tout*>(totals), g, op);
        std::vector<char*> bufs(n_targets);
        for (int t = 0; t < n_targets; ++t) {
            bufs[t] = pooled_alloc(bytes);
            copy_bytes<N>(partition, bufs[t], totals, bytes);
        }
        compute_fence(partition);
        pooled_free(totals);
        ArrayRegion<N> my_region = chare_array_region<N>(dec, me);
        for (int t = 0; t < n_targets; ++t) {
            auto target = me;
            target[axis] = me[axis] + 1 + t;
            send_buffer<N>(partition, target, node->id, 0, out_name, my_region, bufs[t], bytes);
        }
    }
    return me[axis] - first;
}

/// Fold the received totals into a carry and scan the tile with it.
template <int N, typename Tout, typename Tin>
void scan_finish(PartitionImpl<N>* partition, CTArrayBase<N>* in, CTArrayBase<N>* out, int axis,
                 int kind, PendingComm<N>& comm) {
    LineGeom<N> g = tile_geom<N>(in, axis);
    ScanOp<Tout> op{kind};
    char* carry_buf = nullptr;
    auto rb_it = comm.remote_buffers.find(0);
    if (rb_it != comm.remote_buffers.end() && !rb_it->second.empty()) {
        carry_buf = pooled_alloc(g.lines * (int64_t)sizeof(Tout));
        bool init = true;
        for (auto& rb : rb_it->second) {
            fold_totals<N, Tout>(partition, reinterpret_cast<Tout*>(carry_buf),
                                 reinterpret_cast<const Tout*>(rb.data), g.lines, op, init);
            init = false;
        }
    }
    scan_lines<N, Tout, Tin>(partition, static_cast<const Tin*>(in->device_data_ptr()),
                             static_cast<Tout*>(out->device_data_ptr()),
                             reinterpret_cast<const Tout*>(carry_buf), g, op);
    if (carry_buf) {
        compute_fence(partition);
        pooled_free(carry_buf);
    }
}

/// f(Tout{}, Tin{}) for the supported (result, input) dtype pairs: equal, or an
/// integer/bool input narrower than an int64/uint64 result.
template <typename F>
void scan_dispatch(DType out_dt, DType in_dt, F&& f) {
    if (out_dt == in_dt) {
        cn_dispatch(out_dt, [&](auto t) { f(t, t); });
        return;
    }
    auto inner = [&](auto o) {
        switch (in_dt) {
        case DType::BOOL: f(o, bool{}); return;
        case DType::INT8: f(o, int8_t{}); return;
        case DType::INT16: f(o, int16_t{}); return;
        case DType::INT32: f(o, int32_t{}); return;
        case DType::UINT8: f(o, uint8_t{}); return;
        case DType::UINT16: f(o, uint16_t{}); return;
        case DType::UINT32: f(o, uint32_t{}); return;
        default: CkAbort("SCAN: input dtype %d for result dtype %d", (int)in_dt, (int)out_dt);
        }
    };
    if (out_dt == DType::INT64)
        inner(int64_t{});
    else if (out_dt == DType::UINT64)
        inner(uint64_t{});
    else
        CkAbort("SCAN: result dtype %d != input dtype %d", (int)out_dt, (int)in_dt);
}

// ---- SORT / ARGSORT ----

/// Copy between packed block `piece` (region r) and full array `full`.
template <int N, typename T>
void copy_piece(PartitionImpl<N>* partition, T* full, T* piece, const ArrayRegion<N>& r,
                const std::array<int, N>& gs, bool to_full) {
    int64_t start[N], size[N], fstride[N];
    int64_t n = 1;
    for (int d = 0; d < N; ++d) {
        start[d] = r.start[d];
        size[d] = r.stop[d] - r.start[d];
        n *= size[d];
    }
    fstride[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        fstride[d] = fstride[d + 1] * gs[d + 1];
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, n), KOKKOS_LAMBDA(int64_t k) {
            int64_t rem = k, f = 0;
            for (int d = N - 1; d >= 0; --d) {
                f += (start[d] + rem % size[d]) * fstride[d];
                rem /= size[d];
            }
            if (to_full)
                full[f] = piece[k];
            else
                piece[k] = full[f];
        });
#else
    (void)partition;
    for (int64_t k = 0; k < n; ++k) {
        int64_t rem = k, f = 0;
        for (int d = N - 1; d >= 0; --d) {
            f += (start[d] + rem % size[d]) * fstride[d];
            rem /= size[d];
        }
        if (to_full)
            full[f] = piece[k];
        else
            piece[k] = full[f];
    }
#endif
}

/// Sort each line's flat indices by key into idx[l*len + p].
template <int N, typename T>
void sort_lines(PartitionImpl<N>* partition, const T* keys, int64_t* idx, const LineGeom<N>& g) {
    int64_t total = g.lines * g.len;
    SortLess<T> less{keys, g.strides[g.axis], g.len};
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, total),
        KOKKOS_LAMBDA(int64_t k) { idx[k] = g.phys(k / g.len, k % g.len); });
    // One sort of all lines; lines come out in order l.
    Kokkos::View<int64_t*, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v(idx, total);
#ifdef USE_NVIDIA
    Kokkos::sort(partition->compute_exec, v, less);
#else
    Kokkos::sort(Kokkos::DefaultExecutionSpace(), v, less);
#endif
#else
    (void)partition;
    for (int64_t k = 0; k < total; ++k)
        idx[k] = g.phys(k / g.len, k % g.len);
    for (int64_t l = 0; l < g.lines; ++l)
        std::sort(idx + l * g.len, idx + (l + 1) * g.len, less);
#endif
}

/// res[phys(l, p)] = sorted value (SORT) or position along the axis (ARGSORT).
template <int N, typename T, typename Tout>
void gather_sorted(PartitionImpl<N>* partition, const T* full, const int64_t* idx, Tout* res,
                   const LineGeom<N>& g, bool arg) {
    int64_t total = g.lines * g.len;
    int64_t astride = g.strides[g.axis];
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, total), KOKKOS_LAMBDA(int64_t k) {
            int64_t l = k / g.len;
            int64_t base = g.phys(l, 0);
            int64_t src = idx[k];
            res[g.phys(l, k % g.len)] =
                arg ? cn_cast<Tout>((src - base) / astride) : cn_cast<Tout>(full[src]);
        });
#else
    (void)partition;
    for (int64_t k = 0; k < total; ++k) {
        int64_t l = k / g.len;
        int64_t base = g.phys(l, 0);
        int64_t src = idx[k];
        res[g.phys(l, k % g.len)] =
            arg ? cn_cast<Tout>((src - base) / astride) : cn_cast<Tout>(full[src]);
    }
#endif
}

/// Root: assemble, sort, keep own tile, send the others back.
template <int N, typename T, typename Tout>
void sort_root(PartitionImpl<N>* partition, int node_id, int out_name, CTArrayBase<N>* in,
               CTArrayBase<N>* out, int axis, bool arg, PendingComm<N>& comm) {
    const ArrayDecomp<N>& dec = in->decomp;
    std::array<int, N> gs = dec.global_shape;
    int64_t sizes[N];
    int64_t total = 1;
    for (int d = 0; d < N; ++d) {
        sizes[d] = gs[d];
        total *= gs[d];
    }
    LineGeom<N> g(sizes, axis);

    T* full = reinterpret_cast<T*>(alloc_buffer(total * (int64_t)sizeof(T)));
    int64_t* idx = reinterpret_cast<int64_t*>(alloc_buffer(total * (int64_t)sizeof(int64_t)));
    Tout* res = reinterpret_cast<Tout*>(alloc_buffer(total * (int64_t)sizeof(Tout)));

    auto me = partition->nd_index();
    copy_piece<N, T>(partition, full, static_cast<T*>(in->device_data_ptr()),
                     chare_array_region<N>(dec, me), gs, true);
    auto rb_it = comm.remote_buffers.find(0);
    if (rb_it != comm.remote_buffers.end())
        for (auto& rb : rb_it->second)
            copy_piece<N, T>(partition, full, reinterpret_cast<T*>(rb.data), rb.region, gs, true);

    sort_lines<N, T>(partition, full, idx, g);
    gather_sorted<N, T, Tout>(partition, full, idx, res, g, arg);

    for (auto& ci : nonempty_chares<N>(dec)) {
        ArrayRegion<N> r = chare_array_region<N>(dec, ci);
        if (ci == me) {
            copy_piece<N, Tout>(partition, res, static_cast<Tout*>(out->device_data_ptr()), r, gs,
                                false);
            continue;
        }
        int64_t bytes = r.size() * (int64_t)sizeof(Tout);
        char* buf = alloc_buffer(bytes);
        copy_piece<N, Tout>(partition, res, reinterpret_cast<Tout*>(buf), r, gs, false);
        compute_fence(partition);
        send_buffer<N>(partition, ci, node_id, 1, out_name, r, buf, bytes);
    }
    compute_fence(partition);
    free_buffer(reinterpret_cast<char*>(full));
    free_buffer(reinterpret_cast<char*>(idx));
    free_buffer(reinterpret_cast<char*>(res));
}

} // namespace castscansort_detail

using namespace castscansort_detail;

bool ds_handles(ASTNode* root, DType dt, int ndims);
template <int N>
bool ds_axis_whole(CTArrayBase<N>* in, int axis);
template <int N>
void ds_axis_local(PartitionImpl<N>* p, CTArrayBase<N>* in, CTArrayBase<N>* out, int axis,
                   bool arg);

template <int N>
void ArrayDAGExecutorND<N>::execute_castscansort_node(DAGNode* node) {
    ASTNode* root = castscansort_root(node);
    Opcode op = static_cast<Opcode>(root->opcode);
    if (op != Opcode::SCAN && ds_handles(root, root->operands[0]->dtype, N)) {
        execute_ds_node(node);
        return;
    }
    int in_name = root->operands[0]->result_name;
    int out_name = root->result_name;

    auto in_it = partition->arrays.find(in_name);
    if (in_it == partition->arrays.end() || in_it->second->local_size() == 0) {
        // No tile here.
        node_finished(node->id);
        return;
    }
    CTArrayBase<N>* in = in_it->second;
    DType out_dt = op == Opcode::ARGSORT ? DType::INT64
                   : op == Opcode::SCAN  ? root->dtype
                                         : in->dtype;
    CTArrayBase<N>* out = partition->ensure_array(in->region, in->global_shape, out_name, out_dt,
                                                  in->decomp, false);

    int expected = 0;
    if (op == Opcode::SCAN) {
        int axis = scalar_param(root, 1, 0);
        int kind = scalar_param(root, 2, 0);
        scan_dispatch(out_dt, in->dtype, [&](auto to, auto ti) {
            expected = scan_execute<N, decltype(to), decltype(ti)>(partition, node, out_name, in,
                                                                   axis, kind);
        });
    } else if (ds_axis_whole<N>(in, scalar_param(root, 1, 0))) {
        ds_axis_local<N>(partition, in, out, scalar_param(root, 1, 0), op == Opcode::ARGSORT);
    } else if (op == Opcode::SORT || op == Opcode::ARGSORT) {
        const ArrayDecomp<N>& dec = in->decomp;
        auto me = partition->nd_index();
        std::array<int, N> root_ci;
        for (int d = 0; d < N; ++d)
            root_ci[d] = dec.offset[d] / dec.tile;
        if (me == root_ci) {
            expected = (int)nonempty_chares<N>(dec).size() - 1;
        } else {
            int64_t bytes = in->local_size() * (int64_t)in->elem_size();
            char* buf = alloc_buffer(bytes);
            copy_bytes<N>(partition, buf, static_cast<const char*>(in->device_data_ptr()), bytes);
            compute_fence(partition);
            send_buffer<N>(partition, root_ci, node->id, 0, in_name,
                           chare_array_region<N>(dec, me), buf, bytes);
            expected = 1;
        }
    }

    DBG_PRINT("[PE %d] Partition<%d> castscansort node=%d op=%d in=%d out=%d expected=%d\n",
              CkMyPe(), N, node->id, (int)op, in_name, out_name, expected);

    auto pre_it = pending.find(node->id);
    int pre_arrived = 0;
    std::unordered_map<int, std::vector<RemoteBuffer<N>>> pre_buffers;
    if (pre_it != pending.end()) {
        pre_arrived = -(pre_it->second.expected_msgs);
        pre_buffers = std::move(pre_it->second.remote_buffers);
    }
    int remaining = expected - pre_arrived;
    pending[node->id] = {node, remaining, {}, std::move(pre_buffers)};
    if (remaining <= 0)
        on_comm_done(node->id);
}

template <int N>
void ArrayDAGExecutorND<N>::finish_castscansort_node(PendingComm<N>& comm) {
    ASTNode* root = castscansort_root(comm.node);
    Opcode op = static_cast<Opcode>(root->opcode);
    CTArrayBase<N>* in = partition->arrays.at(root->operands[0]->result_name);
    CTArrayBase<N>* out = partition->arrays.at(root->result_name);

    int axis = scalar_param(root, 1, 0);
    if (op == Opcode::SCAN) {
        int kind = scalar_param(root, 2, 0);
        scan_dispatch(out->dtype, in->dtype, [&](auto to, auto ti) {
            scan_finish<N, decltype(to), decltype(ti)>(partition, in, out, axis, kind, comm);
        });
        return;
    }

    // SORT / ARGSORT
    if (ds_axis_whole<N>(in, axis))
        return;
    const ArrayDecomp<N>& dec = in->decomp;
    auto me = partition->nd_index();
    bool is_root = true;
    for (int d = 0; d < N; ++d)
        is_root = is_root && me[d] == dec.offset[d] / dec.tile;
    if (!is_root) {
        auto rb_it = comm.remote_buffers.find(1);
        if (rb_it == comm.remote_buffers.end() || rb_it->second.size() != 1)
            CkAbort("SORT: missing result tile on chare");
        auto& rb = rb_it->second[0];
        copy_bytes<N>(partition, static_cast<char*>(out->device_data_ptr()), rb.data,
                      rb.byte_size);
        return;
    }
    bool arg = op == Opcode::ARGSORT;
    cn_dispatch(in->dtype, [&](auto tag) {
        using T = decltype(tag);
        if (arg)
            sort_root<N, T, int64_t>(partition, comm.node->id, root->result_name, in, out, axis,
                                     true, comm);
        else
            sort_root<N, T, T>(partition, comm.node->id, root->result_name, in, out, axis, false,
                               comm);
    });
}

template void ArrayDAGExecutorND<1>::execute_castscansort_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_castscansort_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_castscansort_node(DAGNode*);

template void ArrayDAGExecutorND<1>::finish_castscansort_node(PendingComm<1>&);
template void ArrayDAGExecutorND<2>::finish_castscansort_node(PendingComm<2>&);
template void ArrayDAGExecutorND<3>::finish_castscansort_node(PendingComm<3>&);
