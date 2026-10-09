#include "backend.hpp"
#include "dag_group_internal.hpp"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <set>
#include <unordered_set>

namespace {

template <int N>
Region* deserialize_array_region_impl(char*& msg) {
    std::array<int, N> s, e, st;
    for (int d = 0; d < N; ++d) {
        s[d] = extract<int>(msg);
        e[d] = extract<int>(msg);
        st[d] = extract<int>(msg);
    }
    return new ArrayRegion<N>(s, e, st);
}

Region* deserialize_array_region(int ndims, char*& msg) {
    switch (ndims) {
    case 1:
        return deserialize_array_region_impl<1>(msg);
    case 2:
        return deserialize_array_region_impl<2>(msg);
    case 3:
        return deserialize_array_region_impl<3>(msg);
    default:
        CkAbort("Unsupported ndims=%d in deserialize_array_region", ndims);
        return nullptr;
    }
}

bool region_to_shape(Region* region, int ndims, std::array<int, 3>& shape_out) {
    shape_out = {0, 0, 0};
    if (region == nullptr || region->is_global)
        return false;

    switch (ndims) {
    case 1: {
        auto* r = static_cast<ArrayRegion<1>*>(region);
        shape_out[0] = r->size(0);
        return true;
    }
    case 2: {
        auto* r = static_cast<ArrayRegion<2>*>(region);
        shape_out[0] = r->size(0);
        shape_out[1] = r->size(1);
        return true;
    }
    case 3: {
        auto* r = static_cast<ArrayRegion<3>*>(region);
        shape_out[0] = r->size(0);
        shape_out[1] = r->size(1);
        shape_out[2] = r->size(2);
        return true;
    }
    default:
        return false;
    }
}

bool infer_result_shape_from_operands(
    ASTNode* ast_node, const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>& array_meta,
    std::array<int, 3>& shape_out) {
    shape_out = {0, 0, 0};
    bool found = false;

    for (int op_idx = 0; op_idx < (int)ast_node->operands.size(); ++op_idx) {
        ASTNode* operand = ast_node->operands[op_idx];
        if (operand == nullptr || operand->is_scalar)
            continue;

        std::array<int, 3> candidate = {0, 0, 0};
        bool have_candidate = false;

        Region* operand_region = ast_node->get_operand_region(op_idx);
        if (operand_region && !operand_region->is_global)
            have_candidate = region_to_shape(operand_region, operand->ndims, candidate);

        if (!have_candidate) {
            auto meta_it = array_meta.find(operand->result_name);
            if (meta_it != array_meta.end()) {
                candidate = meta_it->second.global_shape;
                have_candidate = true;
            }
        }

        if (!have_candidate)
            continue;

        shape_out = candidate;
        found = true;

        // Prefer a non-broadcast operand when available; size-1 broadcast
        // operands are only a fallback when every array-like operand is scalar-shaped.
        if (!operand->is_broadcast)
            return true;
    }

    return found;
}

template <int N>
void insert_chare(typename PartitionTraits<N>::ProxyType& part_proxy, CProxy_ArrayDAGGroup proxy,
                  const std::array<int, N>& idx) {
    ChareIndex<N> ci;
    for (int d = 0; d < N; ++d)
        ci.idx[d] = idx[d];
    proxy_at<N>(part_proxy, ci).insert(proxy);
}

template <int N>
void expand_partition_nd(CProxy_ArrayDAGGroup proxy, typename PartitionTraits<N>::ProxyType& part_proxy,
                         int* current_grid, Region* region_base, int tile) {
    auto* region = static_cast<ArrayRegion<N>*>(region_base);
    int needed[N];

    bool needs_expansion = false;
    for (int d = 0; d < N; ++d) {
        needed[d] = (region->size(d) + tile - 1) / tile;
        if (needed[d] > current_grid[d])
            needs_expansion = true;
    }

    if (!needs_expansion)
        return;

    int expanded[N];
    for (int d = 0; d < N; ++d)
        expanded[d] = std::max(current_grid[d], needed[d]);

    if (CkMyPe() == 0) {
        part_proxy.beginInserting();

        // Insert new chares by iterating dimension-by-dimension over the
        // "L-shaped" expansion region.  For each dimension d where the grid
        // grew, iterate over the slab [old..expanded) in that dimension
        // with [0..expanded) in dimensions > d and [0..old) in dimensions < d
        // (the latter are already covered by earlier slabs).
        int inserted = 0;
        for (int dim = 0; dim < N; ++dim) {
            if (expanded[dim] <= current_grid[dim])
                continue;

            int slab_total = 1;
            int slab_extent[N];
            int slab_start[N];
            for (int d = 0; d < N; ++d) {
                if (d < dim) {
                    slab_start[d] = 0;
                    slab_extent[d] = current_grid[d];
                } else if (d == dim) {
                    slab_start[d] = current_grid[d];
                    slab_extent[d] = expanded[d] - current_grid[d];
                } else {
                    slab_start[d] = 0;
                    slab_extent[d] = expanded[d];
                }
                slab_total *= slab_extent[d];
            }

            std::array<int, N> idx;
            for (int d = 0; d < N; ++d)
                idx[d] = slab_start[d];

            for (int i = 0; i < slab_total; ++i) {
                insert_chare<N>(part_proxy, proxy, idx);
                inserted++;

                for (int d = N - 1; d >= 0; --d) {
                    if (++idx[d] < slab_start[d] + slab_extent[d])
                        break;
                    idx[d] = slab_start[d];
                }
            }
        }

        part_proxy.doneInserting();
        DBG_PRINT("Partition<%d>: expanded grid, inserted %d chares\n", N, inserted);
    }

    for (int d = 0; d < N; ++d)
        current_grid[d] = expanded[d];
}

} // namespace

void ArrayDAGGroup::receive_dag(int epoch, int size, char* serialized_dag) {
    // charmtyles (cache_dag) prefixes every DAG with an int64 plan signature for
    // DAGGroup::replay_dag_template. The template cache is not used here yet, so
    // skip the signature and deserialize the full DAG that follows it.
    int64_t plan_signature;
    memcpy(&plan_signature, serialized_dag, sizeof(plan_signature));
    (void)plan_signature;
    char* payload = serialized_dag + sizeof(plan_signature);
    DAG* dag = DAG::deserialize(payload, deserialize_array_region);

    // Collect which ndims are used in this DAG
    std::set<int> active_ndims;
    for (auto& [id, dag_node] : dag->nodes) {
        for (ASTNode* ast_node : dag_node->ast->roots) {
            if (ast_node->ndims >= 1 && ast_node->ndims <= 3)
                active_ndims.insert(ast_node->ndims);
            for (ASTNode* operand : ast_node->operands)
                if (!operand->is_scalar && !operand->is_broadcast && operand->ndims >= 1 &&
                    operand->ndims <= 3)
                    active_ndims.insert(operand->ndims);
            if (static_cast<Opcode>(ast_node->opcode) == Opcode::DIAG) {
                active_ndims.insert(1);
                active_ndims.insert(2);
            }
            if (is_reduction_op(static_cast<Opcode>(ast_node->opcode))) {
                // The operand may be a size-1 broadcast view; both sides still run.
                active_ndims.insert(ast_node->operands[0]->ndims);
                active_ndims.insert(ast_node->ndims);
            }
            if (static_cast<Opcode>(ast_node->opcode) == Opcode::TILE) {
                int input_nd = ast_node->operands[0]->ndims;
                int result_nd = ast_node->ndims;
                active_ndims.insert(input_nd);
                active_ndims.insert(result_nd);
            }
        }
    }
    DBG_PRINT("[PE %d] receive_dag epoch=%d: active_ndims={", CkMyPe(), epoch);
    for (int nd : active_ndims)
        DBG_PRINT(" %d", nd);
    DBG_PRINT(" }, num_nodes=%d\n", (int)dag->nodes.size());

    // Pass 1: populate array_meta shapes in topological order.
    dag_group_for_each_node_topo(dag, [&](DAGNode* dag_node) {
        for (ASTNode* ast_node : dag_node->ast->roots) {
            Opcode op = static_cast<Opcode>(ast_node->opcode);

            if (is_creation(op) && ast_node->region) {
                int nd = ast_node->ndims;
                switch (nd) {
                case 1: {
                    auto* r = static_cast<ArrayRegion<1>*>(ast_node->region);
                    array_meta[ast_node->result_name] = {nd, {r->size(0), 0, 0}, {}, 0};
                    break;
                }
                case 2: {
                    auto* r = static_cast<ArrayRegion<2>*>(ast_node->region);
                    array_meta[ast_node->result_name] = {nd, {r->size(0), r->size(1), 0}, {}, 0};
                    break;
                }
                case 3: {
                    auto* r = static_cast<ArrayRegion<3>*>(ast_node->region);
                    array_meta[ast_node->result_name] =
                        {nd, {r->size(0), r->size(1), r->size(2)}, {}, 0};
                    break;
                }
                default:
                    CkAbort("Unsupported ndims=%d for partition creation", nd);
                }
            } else if (op == Opcode::REDUCE) {
                array_meta[ast_node->result_name] = {1, {1, 0, 0}, {}, 0};
            } else if (is_reduction_op(op)) {
                std::array<int, 3> out_shape;
                int out_nd = reduction_result_meta(ast_node, array_meta, out_shape);
                array_meta[ast_node->result_name] = {out_nd, out_shape, {}, 0};
            } else if (op == Opcode::MATMUL) {
                int mat_op_name = ast_node->operands[0]->result_name;
                auto mat_it = array_meta.find(mat_op_name);
                if (mat_it != array_meta.end()) {
                    int mat_ndims = mat_it->second.ndims;
                    int mat_rows = mat_it->second.global_shape[0];
                    if (ast_node->operand_regions.size() >= 1) {
                        if (mat_ndims == 3) {
                            auto* mr =
                                static_cast<ArrayRegion<3>*>(ast_node->operand_regions[0]);
                            int dropped = -1;
                            for (int d = 0; d < 3; d++) {
                                if (mr->stop[d] - mr->start[d] == 1) {
                                    dropped = d;
                                    break;
                                }
                            }
                            int row_dim = (dropped == 0) ? 1 : 0;
                            mat_rows = mr->stop[row_dim] - mr->start[row_dim];
                        } else {
                            auto* mr =
                                static_cast<ArrayRegion<2>*>(ast_node->operand_regions[0]);
                            mat_rows = mr->stop[0] - mr->start[0];
                        }
                    }
                    array_meta[ast_node->result_name] = {1, {mat_rows, 0, 0}, {}, 0};
                }
            } else if (op == Opcode::MATMATMUL) {
                int lhs_name = ast_node->operands[0]->result_name;
                int rhs_name = ast_node->operands[1]->result_name;
                auto lhs_it = array_meta.find(lhs_name);
                auto rhs_it = array_meta.find(rhs_name);
                if (lhs_it != array_meta.end() && rhs_it != array_meta.end()) {
                    int M, N_cols;
                    if (ast_node->operand_regions.size() >= 2) {
                        int lhs_ndims = lhs_it->second.ndims;
                        int rhs_ndims = rhs_it->second.ndims;
                        if (lhs_ndims == 3) {
                            auto* lr =
                                static_cast<ArrayRegion<3>*>(ast_node->operand_regions[0]);
                            int dd = -1;
                            for (int d = 0; d < 3; d++)
                                if (lr->stop[d] - lr->start[d] == 1) {
                                    dd = d;
                                    break;
                                }
                            int rd = (dd == 0) ? 1 : 0;
                            M = lr->stop[rd] - lr->start[rd];
                        } else {
                            auto* lr =
                                static_cast<ArrayRegion<2>*>(ast_node->operand_regions[0]);
                            M = lr->stop[0] - lr->start[0];
                        }
                        if (rhs_ndims == 3) {
                            auto* rr =
                                static_cast<ArrayRegion<3>*>(ast_node->operand_regions[1]);
                            int dd = -1;
                            for (int d = 0; d < 3; d++)
                                if (rr->stop[d] - rr->start[d] == 1) {
                                    dd = d;
                                    break;
                                }
                            int cd = (dd <= 1) ? 2 : 1;
                            N_cols = rr->stop[cd] - rr->start[cd];
                        } else {
                            auto* rr =
                                static_cast<ArrayRegion<2>*>(ast_node->operand_regions[1]);
                            N_cols = rr->stop[1] - rr->start[1];
                        }
                    } else {
                        M = lhs_it->second.global_shape[0];
                        N_cols = rhs_it->second.global_shape[1];
                    }
                    array_meta[ast_node->result_name] = {2, {M, N_cols, 0}, {}, 0};
                }
            } else if (op == Opcode::DIAG) {
                int input_name = ast_node->operands[0]->result_name;
                auto input_it = array_meta.find(input_name);
                if (input_it != array_meta.end()) {
                    int input_ndims = input_it->second.ndims;
                    int k_offset = 0;
                    if (ast_node->operands.size() >= 2 && ast_node->operands[1]->is_scalar)
                        k_offset = (int)ast_node->operands[1]->scalar;
                    if (input_ndims == 1) {
                        int vec_len = input_it->second.global_shape[0];
                        int n = vec_len + std::abs(k_offset);
                        array_meta[ast_node->result_name] = {2, {n, n, 0}, {}, 0};
                    } else if (input_ndims == 2) {
                        int M = input_it->second.global_shape[0];
                        int N_cols = input_it->second.global_shape[1];
                        int diag_len;
                        if (k_offset >= 0)
                            diag_len = std::max(0, std::min(M, N_cols - k_offset));
                        else
                            diag_len = std::max(0, std::min(M + k_offset, N_cols));
                        array_meta[ast_node->result_name] = {1, {diag_len, 0, 0}, {}, 0};
                    }
                }
            } else if (op == Opcode::TILE) {
                int input_name = ast_node->operands[0]->result_name;
                auto input_it = array_meta.find(input_name);
                if (input_it != array_meta.end()) {
                    int input_ndims = input_it->second.ndims;
                    int out_ndims = ast_node->ndims;
                    std::array<int, 3> reps = {1, 1, 1};
                    int num_reps = (int)ast_node->operands.size() - 1;
                    for (int d = 0; d < num_reps && d < 3; d++) {
                        if (ast_node->operands[d + 1]->is_scalar)
                            reps[d] = (int)ast_node->operands[d + 1]->scalar;
                    }
                    int delta = out_ndims - input_ndims;
                    std::array<int, 3> out_shape = {0, 0, 0};
                    for (int d = 0; d < out_ndims; d++) {
                        int in_dim =
                            (d < delta) ? 1 : input_it->second.global_shape[d - delta];
                        out_shape[d] = in_dim * reps[d];
                    }
                    array_meta[ast_node->result_name] = {out_ndims, out_shape, {}, 0};
                }
            } else if (op == Opcode::SET_REGION) {
                // SET_REGION writes into the target array's partition.
            } else if (is_elementwise(op) || op == Opcode::COPY) {
                if (ast_node->is_temp)
                    continue;
                std::array<int, 3> inferred_shape = {0, 0, 0};
                if (infer_result_shape_from_operands(ast_node, array_meta, inferred_shape)) {
                    array_meta[ast_node->result_name] = {ast_node->ndims, inferred_shape, {}, 0};
                } else {
                    array_meta[ast_node->result_name] = {1, {1, 0, 0}, {}, 0};
                }
            } else if (op != Opcode::NOOP) {
                bool found = false;
                for (ASTNode* operand : ast_node->operands) {
                    if (operand->is_scalar || operand->is_broadcast)
                        continue;
                    auto it = array_meta.find(operand->result_name);
                    if (it != array_meta.end()) {
                        array_meta[ast_node->result_name] = it->second;
                        found = true;
                        break;
                    }
                }
                if (!found)
                    array_meta[ast_node->result_name] = {1, {1, 0, 0}, {}, 0};
            }
        }
    });
    DBG_PRINT("[PE %d] receive_dag epoch=%d: metadata pass complete\n", CkMyPe(), epoch);

    compute_decompositions(dag);
    for (auto& [name, live_meta] : live_array_meta) {
        auto meta_it = array_meta.find(name);
        if (meta_it == array_meta.end()) {
            array_meta[name] = live_meta;
            continue;
        }
        meta_it->second.ndims = live_meta.ndims;
        meta_it->second.global_shape = live_meta.global_shape;
        meta_it->second.tile = live_meta.tile;
        meta_it->second.offset = live_meta.offset;
    }

    // Pass 2: expand partitions now that tile sizes have been assigned.
    dag_group_for_each_node_topo(dag, [&](DAGNode* dag_node) {
        for (ASTNode* ast_node : dag_node->ast->roots) {
            Opcode op = static_cast<Opcode>(ast_node->opcode);

            if (is_creation(op) && ast_node->region) {
                int nd = ast_node->ndims;
                int tile = array_meta[ast_node->result_name].tile;
                if (partition_grid[nd].start_epoch < 0)
                    partition_grid[nd].start_epoch = epoch;
                switch (nd) {
                case 1:
                    expand_partition_nd<1>(thisProxy, partition_proxy_1, partition_grid[nd].grid,
                                           ast_node->region, tile);
                    break;
                case 2:
                    expand_partition_nd<2>(thisProxy, partition_proxy_2, partition_grid[nd].grid,
                                           ast_node->region, tile);
                    break;
                case 3:
                    expand_partition_nd<3>(thisProxy, partition_proxy_3, partition_grid[nd].grid,
                                           ast_node->region, tile);
                    break;
                default:
                    CkAbort("Unsupported ndims=%d in expansion pass", nd);
                }
            } else if (op == Opcode::MATMUL) {
                auto res_it = array_meta.find(ast_node->result_name);
                if (res_it != array_meta.end()) {
                    int mat_rows = res_it->second.global_shape[0];
                    int tile_1d = res_it->second.tile;
                    std::array<int, 1> s = {0}, e = {mat_rows}, st = {1};
                    ArrayRegion<1> result_region(s, e, st);
                    // The result may be the first 1-D array: new chares must
                    // start at this epoch (set before inserting, since a local
                    // insert constructs the chare immediately).
                    if (partition_grid[1].start_epoch < 0)
                        partition_grid[1].start_epoch = epoch;
                    expand_partition_nd<1>(thisProxy, partition_proxy_1, partition_grid[1].grid,
                                           &result_region, tile_1d);
                }
            } else if (op == Opcode::MATMATMUL) {
                auto res_it = array_meta.find(ast_node->result_name);
                if (res_it != array_meta.end()) {
                    int M = res_it->second.global_shape[0];
                    int N_cols = res_it->second.global_shape[1];
                    int tile_2d = res_it->second.tile;
                    std::array<int, 2> s = {0, 0}, e = {M, N_cols}, st = {1, 1};
                    ArrayRegion<2> result_region(s, e, st);
                    // The result may be the first 2-D array (e.g. A3d[k] @
                    // B3d[k]); without a start epoch the new chares start at
                    // -1, never run this DAG, and the result never appears.
                    if (partition_grid[2].start_epoch < 0)
                        partition_grid[2].start_epoch = epoch;
                    expand_partition_nd<2>(thisProxy, partition_proxy_2, partition_grid[2].grid,
                                           &result_region, tile_2d);
                }
            } else if (op == Opcode::DIAG) {
                auto res_it = array_meta.find(ast_node->result_name);
                if (res_it != array_meta.end()) {
                    int res_nd = res_it->second.ndims;
                    int tile = res_it->second.tile;
                    if (res_nd == 2) {
                        int n = res_it->second.global_shape[0];
                        std::array<int, 2> s = {0, 0}, e = {n, n}, st = {1, 1};
                        ArrayRegion<2> result_region(s, e, st);
                        if (partition_grid[2].start_epoch < 0)
                            partition_grid[2].start_epoch = epoch;
                        expand_partition_nd<2>(thisProxy, partition_proxy_2, partition_grid[2].grid,
                                               &result_region, tile);
                    } else {
                        int diag_len = res_it->second.global_shape[0];
                        std::array<int, 1> s = {0}, e = {diag_len}, st = {1};
                        ArrayRegion<1> result_region(s, e, st);
                        if (partition_grid[1].start_epoch < 0)
                            partition_grid[1].start_epoch = epoch;
                        expand_partition_nd<1>(thisProxy, partition_proxy_1, partition_grid[1].grid,
                                               &result_region, tile);
                    }
                }
            } else if (op == Opcode::TILE || is_reduction_op(op)) {
                auto res_it = array_meta.find(ast_node->result_name);
                if (res_it != array_meta.end()) {
                    int res_nd = res_it->second.ndims;
                    int tile = res_it->second.tile;
                    auto& sh = res_it->second.global_shape;
                    if (partition_grid[res_nd].start_epoch < 0)
                        partition_grid[res_nd].start_epoch = epoch;
                    switch (res_nd) {
                    case 1: {
                        std::array<int, 1> s = {0}, e = {sh[0]}, st = {1};
                        ArrayRegion<1> result_region(s, e, st);
                        expand_partition_nd<1>(thisProxy, partition_proxy_1,
                                               partition_grid[res_nd].grid, &result_region, tile);
                        break;
                    }
                    case 2: {
                        std::array<int, 2> s = {0, 0}, e = {sh[0], sh[1]}, st = {1, 1};
                        ArrayRegion<2> result_region(s, e, st);
                        expand_partition_nd<2>(thisProxy, partition_proxy_2,
                                               partition_grid[res_nd].grid, &result_region, tile);
                        break;
                    }
                    case 3: {
                        std::array<int, 3> s = {0, 0, 0}, e = {sh[0], sh[1], sh[2]}, st = {1, 1, 1};
                        ArrayRegion<3> result_region(s, e, st);
                        expand_partition_nd<3>(thisProxy, partition_proxy_3,
                                               partition_grid[res_nd].grid, &result_region, tile);
                        break;
                    }
                    }
                }
            } else if (is_elementwise(op) || op == Opcode::COPY) {
                auto res_it = array_meta.find(ast_node->result_name);
                if (res_it != array_meta.end() && res_it->second.global_shape[0] > 0) {
                    int nd = res_it->second.ndims;
                    int tile = res_it->second.tile;
                    auto& sh = res_it->second.global_shape;
                    if (partition_grid[nd].start_epoch < 0)
                        partition_grid[nd].start_epoch = epoch;
                    switch (nd) {
                    case 1: {
                        std::array<int, 1> s = {0}, e = {sh[0]}, st = {1};
                        ArrayRegion<1> region(s, e, st);
                        expand_partition_nd<1>(thisProxy, partition_proxy_1, partition_grid[nd].grid,
                                               &region, tile);
                        break;
                    }
                    case 2: {
                        std::array<int, 2> s = {0, 0}, e = {sh[0], sh[1]}, st = {1, 1};
                        ArrayRegion<2> region(s, e, st);
                        expand_partition_nd<2>(thisProxy, partition_proxy_2, partition_grid[nd].grid,
                                               &region, tile);
                        break;
                    }
                    case 3: {
                        std::array<int, 3> s = {0, 0, 0}, e = {sh[0], sh[1], sh[2]}, st = {1, 1, 1};
                        ArrayRegion<3> region(s, e, st);
                        expand_partition_nd<3>(thisProxy, partition_proxy_3, partition_grid[nd].grid,
                                               &region, tile);
                        break;
                    }
                    }
                }
            }
        }
    });
    DBG_PRINT("[PE %d] receive_dag epoch=%d: partition expansion pass complete\n", CkMyPe(),
              epoch);

    // Cut UPLOAD payloads into the tiles of this PE's chares, now that the
    // decompositions and the chare grids are final.
    stage_uploads(dag);

    compile(dag);

    bool first = true;
    for (int nd : active_ndims) {
        if (first) {
            add_dag(nd, epoch, dag);
            first = false;
        } else {
            add_dag(nd, epoch, dag->copy());
        }
    }

    for (auto& [nd, pg] : partition_grid) {
        if (pg.grid[0] > 0 && active_ndims.find(nd) == active_ndims.end()) {
            add_dag(nd, epoch, new DAG());
            DBG_PRINT("[PE %d] receive_dag epoch=%d: stored empty DAG for ndims=%d\n", CkMyPe(),
                      epoch, nd);
        }
    }

    partition_proxy_1.run();
    partition_proxy_2.run();
    partition_proxy_3.run();
}
