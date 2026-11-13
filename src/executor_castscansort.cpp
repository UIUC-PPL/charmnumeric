#ifdef USE_KOKKOS
// Before the Charm++ headers: their ALIGN_BYTES macro breaks cub/thrust.
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
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
// input's decomposition (pinned in dag_group_receive.cpp).
// SORT is not scalable: it gathers to the chare holding element 0.

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

// ---- SCAN ----

#ifdef USE_KOKKOS
/// Device scan of one line (sum/product are commutative, join order is free).
template <int N, typename T>
struct LineScanFunctor {
    using value_type = T;
    const T* in;
    T* out;
    LineGeom<N> g;
    ScanOp<T> op;
    int64_t line;

    KOKKOS_INLINE_FUNCTION void init(T& v) const { v = op.identity(); }
    KOKKOS_INLINE_FUNCTION void join(T& dst, const T& src) const { dst = op.apply(dst, src); }
    KOKKOS_INLINE_FUNCTION void operator()(const int64_t pos, T& update, const bool final) const {
        int64_t i = g.phys(line, pos);
        update = op.apply(update, op.input(in[i]));
        if (final)
            out[i] = update;
    }
};
#endif

/// out = inclusive scan of in along g.axis, per line (no carry yet).
template <int N, typename T>
void scan_tile(PartitionImpl<N>* partition, const T* in, T* out, const LineGeom<N>& g,
               ScanOp<T> op) {
#ifdef USE_KOKKOS
    if (g.lines >= g.len) {
        // Many short lines: one thread per line.
        Kokkos::parallel_for(
            CT_COMPUTE_POLICY(partition, g.lines), KOKKOS_LAMBDA(int64_t l) {
                T acc = op.identity();
                for (int64_t p = 0; p < g.len; ++p) {
                    int64_t i = g.phys(l, p);
                    acc = op.apply(acc, op.input(in[i]));
                    out[i] = acc;
                }
            });
    } else {
        // Few long lines: a device scan per line.
        for (int64_t l = 0; l < g.lines; ++l)
            Kokkos::parallel_scan("ct_scan_line", CT_COMPUTE_POLICY(partition, g.len),
                                  LineScanFunctor<N, T>{in, out, g, op, l});
    }
#else
    (void)partition;
    for (int64_t l = 0; l < g.lines; ++l) {
        T acc = op.identity();
        for (int64_t p = 0; p < g.len; ++p) {
            int64_t i = g.phys(l, p);
            acc = op.apply(acc, op.input(in[i]));
            out[i] = acc;
        }
    }
#endif
}

template <int N, typename T>
void pack_line_totals(PartitionImpl<N>* partition, const T* out, T* buf, const LineGeom<N>& g) {
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, g.lines),
        KOKKOS_LAMBDA(int64_t l) { buf[l] = out[g.phys(l, g.len - 1)]; });
#else
    (void)partition;
    for (int64_t l = 0; l < g.lines; ++l)
        buf[l] = out[g.phys(l, g.len - 1)];
#endif
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

template <int N, typename T>
void apply_carry(PartitionImpl<N>* partition, T* out, const T* carry, const LineGeom<N>& g,
                 ScanOp<T> op) {
    int64_t total = g.lines * g.len;
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, total), KOKKOS_LAMBDA(int64_t k) {
            int64_t l = k / g.len;
            int64_t i = g.phys(l, k % g.len);
            out[i] = op.apply(carry[l], out[i]);
        });
#else
    (void)partition;
    for (int64_t k = 0; k < total; ++k) {
        int64_t l = k / g.len;
        int64_t i = g.phys(l, k % g.len);
        out[i] = op.apply(carry[l], out[i]);
    }
#endif
}

/// Local scan + send line totals to later chares. Returns messages to expect.
template <int N, typename T>
int scan_execute(PartitionImpl<N>* partition, DAGNode* node, int out_name, CTArrayBase<N>* in,
                 CTArrayBase<N>* out, int axis, int kind) {
    LineGeom<N> g = tile_geom<N>(in, axis);
    ScanOp<T> op{kind};
    T* out_ptr = static_cast<T*>(out->device_data_ptr());
    scan_tile<N, T>(partition, static_cast<const T*>(in->device_data_ptr()), out_ptr, g, op);

    const ArrayDecomp<N>& dec = in->decomp;
    auto me = partition->nd_index();
    int first = dec.offset[axis] / dec.tile;
    int last = (dec.offset[axis] + dec.global_shape[axis] - 1) / dec.tile;
    int n_targets = last - me[axis];
    if (n_targets > 0) {
        int64_t bytes = g.lines * (int64_t)sizeof(T);
        std::vector<char*> bufs(n_targets);
        for (int t = 0; t < n_targets; ++t) {
            bufs[t] = alloc_buffer(bytes);
            pack_line_totals<N, T>(partition, out_ptr, reinterpret_cast<T*>(bufs[t]), g);
        }
        compute_fence(partition);
        ArrayRegion<N> my_region = chare_array_region<N>(dec, me);
        for (int t = 0; t < n_targets; ++t) {
            auto target = me;
            target[axis] = me[axis] + 1 + t;
            send_buffer<N>(partition, target, node->id, 0, out_name, my_region, bufs[t], bytes);
        }
    }
    return me[axis] - first;
}

/// Fold received totals into a carry and apply it.
template <int N, typename T>
void scan_finish(PartitionImpl<N>* partition, CTArrayBase<N>* out, int axis, int kind,
                 PendingComm<N>& comm) {
    auto rb_it = comm.remote_buffers.find(0);
    if (rb_it == comm.remote_buffers.end() || rb_it->second.empty())
        return;
    LineGeom<N> g = tile_geom<N>(out, axis);
    ScanOp<T> op{kind};
    char* carry_buf = alloc_buffer(g.lines * (int64_t)sizeof(T));
    T* carry = reinterpret_cast<T*>(carry_buf);
    bool init = true;
    for (auto& rb : rb_it->second) {
        fold_totals<N, T>(partition, carry, reinterpret_cast<const T*>(rb.data), g.lines, op,
                          init);
        init = false;
    }
    apply_carry<N, T>(partition, static_cast<T*>(out->device_data_ptr()), carry, g, op);
    compute_fence(partition);
    free_buffer(carry_buf);
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

template <int N>
void ArrayDAGExecutorND<N>::execute_castscansort_node(DAGNode* node) {
    ASTNode* root = castscansort_root(node);
    Opcode op = static_cast<Opcode>(root->opcode);
    int in_name = root->operands[0]->result_name;
    int out_name = root->result_name;

    auto in_it = partition->arrays.find(in_name);
    if (in_it == partition->arrays.end() || in_it->second->local_size() == 0) {
        // No tile here.
        node_finished(node->id);
        return;
    }
    CTArrayBase<N>* in = in_it->second;
    if (op == Opcode::SCAN && root->dtype != in->dtype)
        CkAbort("SCAN: result dtype %d != input dtype %d", (int)root->dtype, (int)in->dtype);
    DType out_dt = op == Opcode::ARGSORT ? DType::INT64 : in->dtype;
    CTArrayBase<N>* out =
        partition->ensure_array(in->region, in->global_shape, out_name, out_dt, in->decomp);

    int expected = 0;
    if (op == Opcode::SCAN) {
        int axis = scalar_param(root, 1, 0);
        int kind = scalar_param(root, 2, 0);
        cn_dispatch(in->dtype, [&](auto tag) {
            using T = decltype(tag);
            expected = scan_execute<N, T>(partition, node, out_name, in, out, axis, kind);
        });
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
        cn_dispatch(out->dtype, [&](auto tag) {
            using T = decltype(tag);
            scan_finish<N, T>(partition, out, axis, kind, comm);
        });
        return;
    }

    // SORT / ARGSORT
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
