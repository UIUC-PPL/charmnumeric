#include "backend_internal.hpp"
#include "creation_kernels.hpp"
#include "dispatch.hpp"

#include <cstring>

// Array-creation ops (UPLOAD, RANDOM, IOTA) fill each array chare-locally, without communication.
// RANDOM/IOTA derive elements from the global index. UPLOAD tiles are cut from the DAG
// payload in stage_uploads and freed after use.

namespace {

/// Parameter word k of a creation node: two exact SCALAR operands (hi, lo).
uint64_t param_word(ASTNode* root, int k) {
    int hi = 2 * k, lo = 2 * k + 1;
    if (lo >= (int)root->operands.size())
        CkAbort("creation op %d: missing parameter %d", root->opcode, k);
    return ((uint64_t)root->operands[hi]->scalar << 32) | (uint64_t)root->operands[lo]->scalar;
}

template <int N>
std::array<int, 4> upload_key(int name, const std::array<int, N>& idx) {
    std::array<int, 4> key = {name, 0, 0, 0};
    for (int d = 0; d < N; ++d)
        key[1 + d] = idx[d];
    return key;
}

template <int N>
CkArrayIndex chare_array_index(const std::array<int, N>& idx) {
    if constexpr (N == 1)
        return CkArrayIndex1D(idx[0]);
    else if constexpr (N == 2)
        return CkArrayIndex2D(idx[0], idx[1]);
    else
        return CkArrayIndex3D(idx[0], idx[1], idx[2]);
}

template <int N>
void stage_upload_nd(ArrayDAGGroup* group, ASTNode* root, CkLocMgr* loc_mgr) {
    int name = root->result_name;
    auto meta_it = group->array_meta.find(name);
    if (meta_it == group->array_meta.end())
        CkAbort("UPLOAD: no metadata for array %d", name);
    const ArrayDAGGroup::ArrayMetadata& meta = meta_it->second;
    ArrayDecomp<N> decomp = meta.template decomp<N>();
    const int esz = dtype_size(root->dtype);

    std::array<int64_t, N> gstrides;
    gstrides[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        gstrides[d] = gstrides[d + 1] * (int64_t)decomp.global_shape[d + 1];
    int64_t total = gstrides[0] * (int64_t)decomp.global_shape[0];
    if ((int64_t)root->payload.size() != total * esz)
        CkAbort("UPLOAD: array %d payload has %lld bytes, expected %lld", name,
                (long long)root->payload.size(), (long long)(total * esz));

    const int* grid = group->partition_grid[N].grid;
    int num_chares = 1;
    for (int d = 0; d < N; ++d)
        num_chares *= grid[d];

    for (int c = 0; c < num_chares; ++c) {
        std::array<int, N> ci;
        int rem = c;
        for (int d = N - 1; d >= 0; --d) {
            ci[d] = rem % grid[d];
            rem /= grid[d];
        }
        ArrayRegion<N> local = decomp.chare_region_local(ci);
        bool empty = false;
        for (int d = 0; d < N; ++d)
            if (local.size(d) <= 0)
                empty = true;
        if (empty || loc_mgr->homePe(chare_array_index<N>(ci)) != CkMyPe())
            continue;

        ArrayDAGGroup::UploadTile& tile = group->upload_tiles[upload_key<N>(name, ci)];
        tile.meta = meta;
        int64_t row = local.size(N - 1);
        int64_t rows = local.size() / row;
        tile.data.resize((size_t)(local.size() * esz));
        std::array<int, N> idx = {};
        for (int64_t r = 0; r < rows; ++r) {
            int64_t src = 0;
            for (int d = 0; d < N; ++d)
                src += (int64_t)(local.start[d] + idx[d]) * gstrides[d];
            memcpy(tile.data.data() + r * row * esz, root->payload.data() + src * esz,
                   (size_t)(row * esz));
            for (int d = N - 2; d >= 0; --d) {
                if (++idx[d] < local.size(d))
                    break;
                idx[d] = 0;
            }
        }
    }
}

// Fill kernels are named templates: nvcc rejects KOKKOS_LAMBDA in generic lambdas.

template <int N>
struct TileGeometry {
    int64_t start[3]; // array index of the tile's first element
    int64_t size[3];  // tile extent
};

template <int N>
CT_HD void tile_coords(const TileGeometry<N>& t, int64_t k, int64_t* idx) {
    for (int d = 2; d >= N; --d)
        idx[d] = 0;
    for (int d = N - 1; d >= 0; --d) {
        idx[d] = t.start[d] + k % t.size[d];
        k /= t.size[d];
    }
}

template <int N, typename T>
void fill_random(PartitionImpl<N>* partition, Array<N, T>* arr, const TileGeometry<N>& t,
                 const ct_create::RandomParams& rp) {
    int64_t n = arr->local_size();
    T* out = static_cast<T*>(arr->device_data_ptr());
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, n), KOKKOS_LAMBDA(int64_t k) {
            int64_t idx[3];
            tile_coords<N>(t, k, idx);
            uint64_t g = 0;
            for (int d = 0; d < N; ++d)
                g += (uint64_t)idx[d] * (uint64_t)rp.index_strides[d];
            out[k] = ct_create::random_value<T>(rp, g);
        });
#else
    (void)partition;
    for (int64_t k = 0; k < n; ++k) {
        int64_t idx[3];
        tile_coords<N>(t, k, idx);
        uint64_t g = 0;
        for (int d = 0; d < N; ++d)
            g += (uint64_t)idx[d] * (uint64_t)rp.index_strides[d];
        out[k] = ct_create::random_value<T>(rp, g);
    }
#endif
}

template <int N, typename T>
void fill_iota(PartitionImpl<N>* partition, Array<N, T>* arr, const TileGeometry<N>& t,
               const ct_create::IotaParams& ip) {
    int64_t n = arr->local_size();
    T* out = static_cast<T*>(arr->device_data_ptr());
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, n), KOKKOS_LAMBDA(int64_t k) {
            int64_t idx[3];
            tile_coords<N>(t, k, idx);
            out[k] = ct_create::iota_value<T>(ip, idx);
        });
#else
    (void)partition;
    for (int64_t k = 0; k < n; ++k) {
        int64_t idx[3];
        tile_coords<N>(t, k, idx);
        out[k] = ct_create::iota_value<T>(ip, idx);
    }
#endif
}

template <int N, typename T>
void fill_upload(Array<N, T>* arr, const std::vector<char>& tile) {
    if ((int64_t)tile.size() != (int64_t)arr->local_size() * (int64_t)sizeof(T))
        CkAbort("UPLOAD: tile of array %d has %lld bytes, chare holds %lld", arr->name,
                (long long)tile.size(), (long long)arr->local_size() * (long long)sizeof(T));
    memcpy(arr->data_ptr(), tile.data(), tile.size());
    arr->copyToDevice();
}

template <int N, typename T>
void fill_typed(Opcode opc, PartitionImpl<N>* partition, CTArrayBase<N>* base, ASTNode* root,
                const TileGeometry<N>& t, std::vector<char>& tile) {
    auto* arr = static_cast<Array<N, T>*>(base);
    if (opc == Opcode::UPLOAD) {
        fill_upload<N, T>(arr, tile);
    } else if (opc == Opcode::RANDOM) {
        ct_create::RandomParams rp;
        rp.kind = (int)param_word(root, 0);
        rp.seed = param_word(root, 1);
        rp.stream = param_word(root, 2);
        rp.a = param_word(root, 3);
        rp.b = param_word(root, 4);
        for (int d = 0; d < 3; ++d)
            rp.index_strides[d] = (int64_t)param_word(root, 5 + d);
        fill_random<N, T>(partition, arr, t, rp);
    } else {
        ct_create::IotaParams ip;
        ip.mode = (int)param_word(root, 0);
        ip.axis = (int)param_word(root, 1);
        ip.flags = (int)param_word(root, 2);
        ip.num = (int64_t)param_word(root, 3);
        ip.div = (int64_t)param_word(root, 4);
        ip.v0 = param_word(root, 5);
        ip.v1 = param_word(root, 6);
        ip.v2 = param_word(root, 7);
        ip.v3 = param_word(root, 8);
        if (ip.axis < 0 || ip.axis >= N)
            CkAbort("IOTA: axis %d out of range for a %d-D array", ip.axis, N);
        fill_iota<N, T>(partition, arr, t, ip);
    }
}

} // namespace

void ArrayDAGGroup::stage_uploads(DAG* dag) {
    for (auto& [id, dag_node] : dag->nodes) {
        for (ASTNode* root : dag_node->ast->roots) {
            if (static_cast<Opcode>(root->opcode) != Opcode::UPLOAD)
                continue;
            switch (root->ndims) {
            case 1:
                stage_upload_nd<1>(this, root, partition_proxy_1.ckLocMgr());
                break;
            case 2:
                stage_upload_nd<2>(this, root, partition_proxy_2.ckLocMgr());
                break;
            case 3:
                stage_upload_nd<3>(this, root, partition_proxy_3.ckLocMgr());
                break;
            default:
                CkAbort("UPLOAD: unsupported ndims=%d", root->ndims);
            }
            // The DAG (and its ASTs) outlive the epoch; drop the full payload now.
            std::vector<char>().swap(root->payload);
        }
    }
}

template <int N>
void ArrayDAGExecutorND<N>::execute_creation_node(DAGNode* node) {
    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    auto nd_idx = partition->nd_index();

    for (ASTNode* root : node->ast->roots) {
        auto opc = static_cast<Opcode>(root->opcode);
        if (!is_fill_creation(opc) || root->ndims != N)
            continue;
        int name = root->result_name;

        ArrayDecomp<N> decomp;
        std::vector<char> tile;
        bool have_tile = false;
        if (opc == Opcode::UPLOAD) {
            auto tile_it = dag_group->upload_tiles.find(upload_key<N>(name, nd_idx));
            if (tile_it != dag_group->upload_tiles.end()) {
                tile = std::move(tile_it->second.data);
                decomp = tile_it->second.meta.template decomp<N>();
                dag_group->upload_tiles.erase(tile_it);
                have_tile = true;
            }
        }
        if (!have_tile) {
            auto meta_it = dag_group->array_meta.find(name);
            if (meta_it == dag_group->array_meta.end())
                CkAbort("creation op %d: no metadata for array %d", root->opcode, name);
            decomp = meta_it->second.template decomp<N>();
        }

        partition->create(static_cast<ArrayRegion<N>*>(root->region), name, root->dtype, decomp);
        auto arr_it = partition->arrays.find(name);
        if (arr_it == partition->arrays.end())
            continue; // this chare holds no part of the array
        if (opc == Opcode::UPLOAD && !have_tile)
            CkAbort("UPLOAD: no staged tile for array %d on PE %d", name, CkMyPe());

        TileGeometry<N> t;
        ArrayRegion<N> local = decomp.chare_region_local(nd_idx);
        for (int d = 0; d < 3; ++d) {
            t.start[d] = d < N ? local.start[d] : 0;
            t.size[d] = d < N ? local.size(d) : 1;
        }

        cn_dispatch(root->dtype, [&](auto tag) {
            using T = decltype(tag);
            fill_typed<N, T>(opc, partition, arr_it->second, root, t, tile);
        });
    }

#ifdef USE_NVIDIA
    {
        auto* p = new ComputeDoneParam<N>{partition, node->id, false};
        CkCallback hcb(compute_done_cb<N>, p);
        hapiAddCallback(partition->compute_stream_raw, &hcb);
    }
#else
    CT_KOKKOS_FENCE();
    node_finished(node->id);
#endif
}

template void ArrayDAGExecutorND<1>::execute_creation_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_creation_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_creation_node(DAGNode*);
