#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <cstring>

// TRANSPOSE: result axis k reads source axis perm[k]. Source chares send permuted
// blocks to each overlapping result chare; any tiles/offsets work.

template <int N>
struct TransposeGeom {
    ArrayDecomp<N> src;        // source array decomposition
    ArrayDecomp<N> dst;        // result decomposition
    std::array<int, N> vstart; // view start along each source axis, source global coords
    std::array<int, N> vstep;
    std::array<int, N> vsize;  // view extent (elements) along each source axis
    std::array<int, N> perm;   // result axis k <- source axis perm[k]
    std::array<int, N> inv;    // source axis d -> result axis inv[d]
};

template <int N>
static bool make_transpose_geom(ArrayDAGGroup* dag_group, ASTNode* root,
                                TransposeGeom<N>& g) {
    int input_name = root->operands[0]->result_name;
    auto in_it = dag_group->array_meta.find(input_name);
    auto out_it = dag_group->array_meta.find(root->result_name);
    if (in_it == dag_group->array_meta.end() || out_it == dag_group->array_meta.end())
        return false;
    if ((int)root->operands.size() < N + 1)
        return false;

    g.src = in_it->second.template decomp<N>();
    g.dst = out_it->second.template decomp<N>();

    Region* view = root->get_operand_region(0);
    for (int d = 0; d < N; ++d) {
        if (view == nullptr || view->is_global) {
            g.vstart[d] = g.src.offset[d];
            g.vstep[d] = 1;
            g.vsize[d] = g.src.global_shape[d];
        } else {
            auto* r = static_cast<ArrayRegion<N>*>(view);
            g.vstart[d] = r->start[d] + g.src.offset[d];
            g.vstep[d] = r->step[d];
            g.vsize[d] = r->size(d);
        }
    }
    for (int k = 0; k < N; ++k) {
        g.perm[k] = (int)root->operands[k + 1]->scalar;
        g.inv[g.perm[k]] = k;
    }
    return true;
}

/// View-index range [lo, hi) along source axis d held by source chare index c.
template <int N>
static bool src_block_1d(const TransposeGeom<N>& g, int d, int c, int& lo, int& hi) {
    return g.src.view_block(d, c, g.vstart[d], g.vstep[d], g.vsize[d], lo, hi);
}

/// Result chares [first, last] along axis k covering local indices [lo, hi).
template <int N>
static void dst_chare_range(const TransposeGeom<N>& g, int k, int lo, int hi, int& first,
                            int& last) {
    first = (lo + g.dst.offset[k]) / g.dst.tile;
    last = (hi - 1 + g.dst.offset[k]) / g.dst.tile;
}

/// Pack `sub` (result coords) in result order and send it. sstride[k]: source
/// stride per step along result axis k.
template <int N, typename T>
static void transpose_pack_send(PartitionImpl<N>* partition, Array<N, T>* src, DAGNode* node,
                         int input_name, const ChareIndex<N>& target,
                         const ArrayRegion<N>& sub, int64_t src_base,
                         const std::array<int64_t, N>& sstride) {
    int region_data[N * 3];
    int ext[N];
    int64_t total = 1;
    for (int k = 0; k < N; ++k) {
        region_data[k * 3 + 0] = sub.start[k];
        region_data[k * 3 + 1] = sub.stop[k];
        region_data[k * 3 + 2] = 1;
        ext[k] = sub.stop[k] - sub.start[k];
        total *= ext[k];
    }
    int64_t byte_size = total * sizeof(T);
#ifndef NDEBUG
    partition->comm_bytes_sent += byte_size;
#endif

#ifdef USE_KOKKOS
    T* send_buf = static_cast<T*>(Kokkos::kokkos_malloc<DeviceSpace>(byte_size));
    T* src_device = static_cast<T*>(src->device_data_ptr());
    int64_t ss[N];
    for (int k = 0; k < N; ++k)
        ss[k] = sstride[k];
    Kokkos::parallel_for(
        CT_COMM_POLICY(partition, total), KOKKOS_LAMBDA(int flat) {
            int rem = flat;
            int64_t s = src_base;
            for (int k = N - 1; k >= 0; --k) {
                s += (int64_t)(rem % ext[k]) * ss[k];
                rem /= ext[k];
            }
            send_buf[flat] = src_device[s];
        });
    device_pack_send<N, N>(partition, partition->thisProxy, target, node->id, 0, input_name,
                           region_data, byte_size, send_buf);
#else
    src->copyToHost();
    const T* src_data = static_cast<T*>(src->data_ptr());
    T* send_buf = new T[total];
    int inner = ext[N - 1];
    int64_t inner_stride = sstride[N - 1];
    int64_t outer = total / inner;
    for (int64_t o = 0; o < outer; ++o) {
        int64_t rem = o;
        int64_t s = src_base;
        for (int k = N - 2; k >= 0; --k) {
            s += (rem % ext[k]) * sstride[k];
            rem /= ext[k];
        }
        T* dst = send_buf + o * inner;
        if (inner_stride == 1) {
            std::memcpy(dst, src_data + s, inner * sizeof(T));
        } else {
            for (int i = 0; i < inner; ++i)
                dst[i] = src_data[s + i * inner_stride];
        }
    }
    proxy_at<N>(partition->thisProxy, target)
        .receive_data(node->id, /*input_index=*/0, input_name, N, region_data, byte_size,
                      reinterpret_cast<char*>(send_buf));
    delete[] send_buf;
#endif
}

/// Send one permuted block to every result chare overlapping this chare's part.
template <int N, typename T>
static void transpose_send(PartitionImpl<N>* partition, DAGNode* node, int input_name,
                    const TransposeGeom<N>& g) {
    auto arr_it = partition->arrays.find(input_name);
    if (arr_it == partition->arrays.end() || arr_it->second->local_size() == 0)
        return;
    auto* src = static_cast<Array<N, T>*>(arr_it->second);
    auto nd_idx = partition->nd_index();

    std::array<int, N> blo, bhi;
    for (int d = 0; d < N; ++d)
        if (!src_block_1d(g, d, nd_idx[d], blo[d], bhi[d]))
            return;

    // Source local layout (row-major over the chare's tile).
    std::array<int64_t, N> lstride;
    lstride[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        lstride[d] = lstride[d + 1] * (int64_t)src->region.size(d + 1);
    std::array<int, N> cstart;
    for (int d = 0; d < N; ++d)
        cstart[d] = g.src.chare_start_global(d, nd_idx[d]);

    // Same block in result coords: local [rlo, rhi) along result axis k.
    std::array<int, N> rlo, rhi, cfirst, clast;
    std::array<int64_t, N> sstride;
    for (int k = 0; k < N; ++k) {
        int d = g.perm[k];
        rlo[k] = blo[d];
        rhi[k] = bhi[d];
        dst_chare_range(g, k, rlo[k], rhi[k], cfirst[k], clast[k]);
        sstride[k] = (int64_t)g.vstep[d] * lstride[d];
    }

    std::array<int, N> t = cfirst;
    while (true) {
        ChareIndex<N> target;
        ArrayRegion<N> sub;
        std::array<int, N> tci;
        for (int k = 0; k < N; ++k)
            tci[k] = t[k];
        auto tchare = g.dst.chare_region_global(tci);
        int64_t src_base = 0;
        std::array<int, N> v0;
        for (int k = 0; k < N; ++k) {
            target.idx[k] = t[k];
            sub.start[k] = std::max(rlo[k] + g.dst.offset[k], tchare.start[k]);
            sub.stop[k] = std::min(rhi[k] + g.dst.offset[k], tchare.stop[k]);
            sub.step[k] = 1;
            v0[g.perm[k]] = sub.start[k] - g.dst.offset[k];
        }
        for (int d = 0; d < N; ++d)
            src_base += (int64_t)(g.vstart[d] + v0[d] * g.vstep[d] - cstart[d]) * lstride[d];
        transpose_pack_send<N, T>(partition, src, node, input_name, target, sub, src_base,
                                  sstride);

        int k = N - 1;
        while (k >= 0) {
            if (++t[k] <= clast[k])
                break;
            t[k] = cfirst[k];
            --k;
        }
        if (k < 0)
            break;
    }
}

template <int N>
static void dispatch_transpose_send(DType dt, PartitionImpl<N>* partition, DAGNode* node,
                             int input_name, const TransposeGeom<N>& g) {
    cn_dispatch(dt, [&](auto tag) {
        using T = decltype(tag);
        transpose_send<N, T>(partition, node, input_name, g);
    });
}

/// Messages the result chare owning `mine` expects; must match transpose_send.
template <int N>
static int transpose_expected_msgs(const TransposeGeom<N>& g, const ArrayRegion<N>& mine) {
    int expected = 1;
    for (int d = 0; d < N; ++d) {
        int k = g.inv[d];
        int need_lo = mine.start[k] - g.dst.offset[k];
        int need_hi = mine.stop[k] - g.dst.offset[k];
        int c_first = (g.vstart[d] + need_lo * g.vstep[d]) / g.src.tile;
        int c_last = (g.vstart[d] + (need_hi - 1) * g.vstep[d]) / g.src.tile;
        int count = 0;
        for (int c = c_first; c <= c_last; ++c) {
            int lo, hi;
            if (src_block_1d(g, d, c, lo, hi) && lo < need_hi && hi > need_lo)
                ++count;
        }
        expected *= count;
    }
    return expected;
}

template <int N>
void ArrayDAGExecutorND<N>::execute_transpose_node(DAGNode* node) {
    ASTNode* root = nullptr;
    for (ASTNode* r : node->ast->roots) {
        if (static_cast<Opcode>(r->opcode) == Opcode::TRANSPOSE) {
            root = r;
            break;
        }
    }
    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    TransposeGeom<N> g;
    if (root == nullptr || root->ndims != N || !make_transpose_geom<N>(dag_group, root, g)) {
        node_finished(node->id);
        return;
    }
    int input_name = root->operands[0]->result_name;
    DType dt = determine_dtype<N>(node, partition->arrays);
    auto nd_idx = partition->nd_index();

    DBG_PRINT("[PE %d] execute_transpose_node<%d>: node_id=%d input=%d result=%d\n", CkMyPe(),
              N, node->id, input_name, root->result_name);

    dispatch_transpose_send<N>(dt, partition, node, input_name, g);

    auto mine = g.dst.chare_region_global(nd_idx);
    for (int k = 0; k < N; ++k) {
        if (mine.stop[k] <= mine.start[k]) {
            node_finished(node->id);
            return;
        }
    }
    int expected_msgs = transpose_expected_msgs<N>(g, mine);

    auto pre_it = pending.find(node->id);
    int pre_arrived = 0;
    std::unordered_map<int, std::vector<RemoteBuffer<N>>> pre_buffers;
    if (pre_it != pending.end()) {
        pre_arrived = -(pre_it->second.expected_msgs);
        pre_buffers = std::move(pre_it->second.remote_buffers);
    }
    int remaining = expected_msgs - pre_arrived;
    pending[node->id] = {node, remaining, {mine}, std::move(pre_buffers)};
    if (remaining <= 0)
        on_comm_done(node->id);
}

template <int N, typename T>
void transpose_place(ArrayDAGGroup* dag_group, PartitionImpl<N>* partition, ASTNode* root,
                     PendingComm<N>* comm) {
    int result_name = root->result_name;
    auto meta_it = dag_group->array_meta.find(result_name);
    if (meta_it == dag_group->array_meta.end())
        return;
    auto decomp = meta_it->second.template decomp<N>();
    auto mine = decomp.chare_region_global(partition->nd_index());

    std::array<int, N> zeros{}, sizes{}, ones{}, gs{};
    for (int k = 0; k < N; ++k) {
        sizes[k] = mine.stop[k] - mine.start[k];
        ones[k] = 1;
        gs[k] = decomp.global_shape[k];
    }
    ArrayRegion<N> local(zeros, sizes, ones);
    auto* out = partition->template ensure_array_typed<T>(local, gs, result_name, decomp);
    if (comm == nullptr)
        return;
    auto rb_it = comm->remote_buffers.find(0);
    if (rb_it == comm->remote_buffers.end())
        return;

    for (auto& rb : rb_it->second) {
        // rb.region: the block's extent in result global coords, packed row-major.
        int ext[N];
        int64_t base = 0, ostride[N];
        ostride[N - 1] = 1;
        for (int k = N - 2; k >= 0; --k)
            ostride[k] = ostride[k + 1] * sizes[k + 1];
        int64_t total = 1;
        for (int k = 0; k < N; ++k) {
            ext[k] = rb.region.stop[k] - rb.region.start[k];
            base += (int64_t)(rb.region.start[k] - mine.start[k]) * ostride[k];
            total *= ext[k];
        }
        const T* received = reinterpret_cast<const T*>(rb.data);
#ifdef USE_KOKKOS
        T* out_device = static_cast<T*>(out->device_data_ptr());
        Kokkos::parallel_for(
            CT_COMPUTE_POLICY(partition, total), KOKKOS_LAMBDA(int flat) {
                int rem = flat;
                int64_t o = base;
                for (int k = N - 1; k >= 0; --k) {
                    o += (int64_t)(rem % ext[k]) * ostride[k];
                    rem /= ext[k];
                }
                out_device[o] = received[flat];
            });
#else
        T* out_data = static_cast<T*>(out->data_ptr());
        int inner = ext[N - 1];
        int64_t outer = total / inner;
        for (int64_t o = 0; o < outer; ++o) {
            int64_t rem = o;
            int64_t off = base;
            for (int k = N - 2; k >= 0; --k) {
                off += (rem % ext[k]) * ostride[k];
                rem /= ext[k];
            }
            std::memcpy(out_data + off, received + o * inner, inner * sizeof(T));
        }
#endif
    }
}

template void ArrayDAGExecutorND<1>::execute_transpose_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_transpose_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_transpose_node(DAGNode*);

#define CN_INST_TRANSPOSE_PLACE(T)                                                                 \
    template void transpose_place<2, T>(ArrayDAGGroup*, PartitionImpl<2>*, ASTNode*,                 \
                                        PendingComm<2>*);                                          \
    template void transpose_place<3, T>(ArrayDAGGroup*, PartitionImpl<3>*, ASTNode*,                 \
                                        PendingComm<3>*);
CN_FOR_EACH_TYPE(CN_INST_TRANSPOSE_PLACE)
