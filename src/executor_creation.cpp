#include "backend_internal.hpp"
#include "creation_kernels.hpp"
#include "dispatch.hpp"

#include <cstring>

// Array-creation ops (UPLOAD, RANDOM, IOTA) fill each array chare-locally.
// RANDOM/IOTA derive elements from the global index. UPLOAD bytes reach only PE 0, which
// cuts them into tiles and sends each to its chare's home PE; a chare living elsewhere
// fetches its tile from there.

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

/// PE that stages chare idx's UPLOAD tiles: its home PE (CNUM_UPLOAD_STAGE_SHIFT moves
/// staging off the chares to test the fetch path).
template <int N>
int upload_stage_pe(CkLocMgr* loc_mgr, const std::array<int, N>& idx) {
    static const int shift = [] {
        const char* e = getenv("CNUM_UPLOAD_STAGE_SHIFT");
        return e ? atoi(e) : 0;
    }();
    return (loc_mgr->homePe(chare_array_index<N>(idx)) + shift) % CkNumPes();
}

template <int N>
CkLocMgr* partition_loc_mgr(ArrayDAGGroup* group) {
    if constexpr (N == 1)
        return group->partition_proxy_1.ckLocMgr();
    else if constexpr (N == 2)
        return group->partition_proxy_2.ckLocMgr();
    else
        return group->partition_proxy_3.ckLocMgr();
}

constexpr int kTileMetaInts = 8; // ndims, global_shape[3], offset[3], tile

template <int N>
void stage_upload_nd(ArrayDAGGroup* group, ASTNode* root, const char* payload) {
    CkLocMgr* loc_mgr = partition_loc_mgr<N>(group);
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
    if (root->payload_size != total * esz)
        CkAbort("UPLOAD: array %d payload has %lld bytes, expected %lld", name,
                (long long)root->payload_size, (long long)(total * esz));
    int meta_ints[kTileMetaInts] = {meta.ndims, meta.global_shape[0], meta.global_shape[1],
                                    meta.global_shape[2], meta.offset[0], meta.offset[1],
                                    meta.offset[2], meta.tile};
    std::vector<char> remote; // a tile on its way to another PE

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
        if (empty)
            continue;

        int pe = upload_stage_pe<N>(loc_mgr, ci);
        auto key = upload_key<N>(name, ci);
        std::vector<char>* dst = &remote;
        if (pe == CkMyPe()) {
            ArrayDAGGroup::UploadTile& tile = group->upload_tiles[key];
            tile.meta = meta;
            dst = &tile.data;
        }
        int64_t row = local.size(N - 1);
        int64_t rows = local.size() / row;
        dst->resize((size_t)(local.size() * esz));
        std::array<int, N> idx = {};
        for (int64_t r = 0; r < rows; ++r) {
            int64_t src = 0;
            for (int d = 0; d < N; ++d)
                src += (int64_t)(local.start[d] + idx[d]) * gstrides[d];
            memcpy(dst->data() + r * row * esz, payload + src * esz, (size_t)(row * esz));
            for (int d = N - 2; d >= 0; --d) {
                if (++idx[d] < local.size(d))
                    break;
                idx[d] = 0;
            }
        }
        if (pe == CkMyPe())
            group->serve_upload_fetch(key, N);
        else
            group->thisProxy[pe].stage_upload_tile(name, N, key[1], key[2], key[3], kTileMetaInts,
                                                   meta_ints, (int64_t)remote.size(),
                                                   remote.data());
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

void ArrayDAGGroup::stage_uploads(DAG* dag, int epoch) {
    for (auto& [id, dag_node] : dag->nodes) {
        for (ASTNode* root : dag_node->ast->roots) {
            if (static_cast<Opcode>(root->opcode) != Opcode::UPLOAD)
                continue;
            const char* payload = inline_bytes(epoch, root);
            if (payload == nullptr)
                continue; // not PE 0: tiles arrive in stage_upload_tile
            switch (root->ndims) {
            case 1:
                stage_upload_nd<1>(this, root, payload);
                break;
            case 2:
                stage_upload_nd<2>(this, root, payload);
                break;
            case 3:
                stage_upload_nd<3>(this, root, payload);
                break;
            default:
                CkAbort("UPLOAD: unsupported ndims=%d", root->ndims);
            }
        }
    }
    release_inline_bulk(epoch);
}

void ArrayDAGGroup::stage_upload_tile(int name, int nd, int i0, int i1, int i2, int nmeta,
                                      int* meta, int64_t size, char* data) {
    if (nmeta != kTileMetaInts)
        CkAbort("UPLOAD: tile of array %d has %d metadata words", name, nmeta);
    std::array<int, 4> key = {name, i0, i1, i2};
    UploadTile& tile = upload_tiles[key];
    tile.meta.ndims = meta[0];
    for (int d = 0; d < 3; ++d) {
        tile.meta.global_shape[d] = meta[1 + d];
        tile.meta.offset[d] = meta[4 + d];
    }
    tile.meta.tile = meta[7];
    tile.meta.decomp_final = true;
    tile.data.assign(data, data + size);
    serve_upload_fetch(key, nd);
}

void ArrayDAGGroup::serve_upload_fetch(const std::array<int, 4>& key, int nd) {
    auto f = upload_fetches.find(key);
    if (f == upload_fetches.end())
        return;
    auto it = upload_tiles.find(key);
    send_upload_tile(key, nd, f->second.first, it->second.data);
    upload_fetches.erase(f);
    upload_tiles.erase(it);
}

void ArrayDAGGroup::fetch_upload_tile(int node_id, int name, int nd, int i0, int i1, int i2) {
    std::array<int, 4> key = {name, i0, i1, i2};
    auto it = upload_tiles.find(key);
    if (it == upload_tiles.end()) {
        upload_fetches[key] = {node_id, nd}; // not staged yet: served by stage_upload_nd
        return;
    }
    send_upload_tile(key, nd, node_id, it->second.data);
    upload_tiles.erase(it);
}

void ArrayDAGGroup::send_upload_tile(const std::array<int, 4>& key, int nd, int node_id,
                                     std::vector<char>& data) {
    int64_t size = static_cast<int64_t>(data.size());
    if (nd == 1)
        partition_proxy_1(key[1]).receive_upload_tile(node_id, key[0], size, data.data());
    else if (nd == 2)
        partition_proxy_2(key[1], key[2]).receive_upload_tile(node_id, key[0], size, data.data());
    else
        partition_proxy_3(key[1], key[2], key[3])
            .receive_upload_tile(node_id, key[0], size, data.data());
}

template <int N>
void ArrayDAGExecutorND<N>::on_upload_tile(int node_id, int name, char* data, int64_t size) {
    auto w = upload_wait.find(node_id);
    if (w == upload_wait.end())
        CkAbort("UPLOAD: unexpected tile of array %d for node %d", name, node_id);
    ASTNode* root = nullptr;
    for (ASTNode* r : w->second.first->ast->roots)
        if (r->result_name == name && static_cast<Opcode>(r->opcode) == Opcode::UPLOAD)
            root = r;
    auto arr_it = partition->arrays.find(name);
    if (root == nullptr || arr_it == partition->arrays.end())
        CkAbort("UPLOAD: tile for unknown array %d", name);
    std::vector<char> tile(data, data + size);
    cn_dispatch(root->dtype, [&](auto tag) {
        using T = decltype(tag);
        fill_upload<N, T>(static_cast<Array<N, T>*>(arr_it->second), tile);
    });
    if (--w->second.second == 0) {
        upload_wait.erase(w);
        finish_creation_node(node_id);
    }
}

template <int N>
void ArrayDAGExecutorND<N>::finish_creation_node(int node_id) {
#ifdef USE_NVIDIA
    auto* p = new ComputeDoneParam<N>{partition, node_id, false};
    CkCallback hcb(compute_done_cb<N>, p);
    hapiAddCallback(partition->compute_stream_raw, &hcb);
#else
    CT_KOKKOS_FENCE();
    node_finished(node_id);
#endif
}

template <int N>
void PartitionImpl<N>::receive_upload_tile(int node_id, int name, int64_t size, char* data) {
    executor->on_upload_tile(node_id, name, data, size);
}

template <int N>
void ArrayDAGExecutorND<N>::execute_creation_node(DAGNode* node) {
    auto* dag_group = static_cast<ArrayDAGGroup*>(group);
    auto nd_idx = partition->nd_index();
    int fetches = 0;

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
        if (opc == Opcode::UPLOAD && !have_tile) {
            // staged on another PE: fetch it (on_upload_tile fills the array)
            int pe = upload_stage_pe<N>(partition_loc_mgr<N>(dag_group), nd_idx);
            std::array<int, 3> ci = {0, 0, 0};
            for (int d = 0; d < N; ++d)
                ci[d] = nd_idx[d];
            dag_group->thisProxy[pe].fetch_upload_tile(node->id, name, N, ci[0], ci[1], ci[2]);
            ++fetches;
            continue;
        }

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

    if (fetches > 0)
        upload_wait[node->id] = {node, fetches};
    else
        finish_creation_node(node->id);
}

template void ArrayDAGExecutorND<1>::execute_creation_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_creation_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_creation_node(DAGNode*);

template void PartitionImpl<1>::receive_upload_tile(int, int, int64_t, char*);
template void PartitionImpl<2>::receive_upload_tile(int, int, int64_t, char*);
template void PartitionImpl<3>::receive_upload_tile(int, int, int64_t, char*);
