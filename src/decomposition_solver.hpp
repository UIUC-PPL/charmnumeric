#pragma once

#include "array_region.hpp"
#include "opcodes.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef CT_MIN_TILE_1D
#ifdef USE_KOKKOS
#define CT_MIN_TILE_1D 1048576
#define CT_MIN_TILE_2D 1024
#define CT_MIN_TILE_3D 128
#else
#define CT_MIN_TILE_1D 262144
#define CT_MIN_TILE_2D 512
#define CT_MIN_TILE_3D 64
#endif
#endif

namespace decomposition_solver {

namespace detail {

inline int next_pow2(int v) {
    if (v <= 1)
        return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

inline int ct_min_tile(int ndims) {
    switch (ndims) {
    case 1:
        return CT_MIN_TILE_1D;
    case 2:
        return CT_MIN_TILE_2D;
    case 3:
        return CT_MIN_TILE_3D;
    default:
        return 1;
    }
}

template <typename Fn>
inline void for_each_node_topo(DAG* dag, Fn&& fn) {
    if (dag == nullptr)
        return;

    std::queue<DAGNode*> topo_queue;
    std::unordered_map<DAGNode*, int> remaining_parents;
    remaining_parents.reserve(dag->nodes.size());

    for (auto& [id, node] : dag->nodes) {
        remaining_parents[node] = node->num_parents;
        if (node->num_parents == 0)
            topo_queue.push(node);
    }

    int processed = 0;
    while (!topo_queue.empty()) {
        DAGNode* dag_node = topo_queue.front();
        topo_queue.pop();
        processed++;
        fn(dag_node);

        for (DAGNode* child : dag_node->children) {
            auto it = remaining_parents.find(child);
            if (it == remaining_parents.end())
                continue;
            if (--it->second == 0)
                topo_queue.push(child);
        }
    }

    assert(processed == static_cast<int>(dag->nodes.size()));
}

} // namespace detail

template <typename ArrayMetaMap>
void compute_decompositions(ArrayMetaMap& array_meta, DAG* dag, int odf, int num_pes) {
    using ArrayMetadata = typename ArrayMetaMap::mapped_type;

    auto zero_offset = std::array<int, 3>{0, 0, 0};

    auto compute_tile_count = [](const ArrayMetadata& m) -> int {
        int count = 1;
        for (int d = 0; d < m.ndims; d++)
            count *= (m.global_shape[d] + m.tile - 1) / m.tile;
        return count;
    };

    auto clamp_tile_for_max_count = [&](ArrayMetadata& m) {
        int max_tiles = std::max(1, odf) * std::max(1, num_pes);
        while (compute_tile_count(m) > max_tiles)
            m.tile *= 2;
    };

    for (auto& [name, meta] : array_meta) {
        if (meta.tile == 0)
            meta.tile = detail::ct_min_tile(meta.ndims);

        if (!meta.decomp_final) {
            meta.tile = std::max(detail::next_pow2(meta.tile), detail::ct_min_tile(meta.ndims));
            clamp_tile_for_max_count(meta);
            meta.offset = zero_offset;
        }
    }

    if (dag != nullptr) {
        std::unordered_set<int> matmul_arrays;

        auto positive_mod = [](int value, int mod) {
            int r = value % mod;
            return r < 0 ? r + mod : r;
        };

        auto gcd_positive = [](int a, int b) {
            if (a < 0)
                a = -a;
            if (b < 0)
                b = -b;
            while (b != 0) {
                int t = a % b;
                a = b;
                b = t;
            }
            return a;
        };

        auto region_stride_factor = [&](Region* region_base, int ndims) {
            if (region_base == nullptr || region_base->is_global)
                return 1;

            int factor = 0;
            auto fold_step = [&](int step) {
                if (step < 0)
                    step = -step;
                if (step <= 1)
                    return;
                factor = (factor == 0) ? step : gcd_positive(factor, step);
            };

            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                fold_step(r->step[0]);
                break;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                fold_step(r->step[0]);
                fold_step(r->step[1]);
                break;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                fold_step(r->step[0]);
                fold_step(r->step[1]);
                fold_step(r->step[2]);
                break;
            }
            default:
                break;
            }

            return factor > 1 ? factor : 1;
        };

        auto region_is_dense = [](Region* region_base, int ndims) {
            if (region_base == nullptr || region_base->is_global)
                return true;

            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                return r->step[0] == 1;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                return r->step[0] == 1 && r->step[1] == 1;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                return r->step[0] == 1 && r->step[1] == 1 && r->step[2] == 1;
            }
            default:
                return false;
            }
        };

        auto tile_fits_grid = [](const ArrayMetadata& meta, int tile) {
            (void)meta;
            return tile > 0;
        };

        auto try_assign_tile = [&](int array_name, int candidate_tile) {
            auto array_it = array_meta.find(array_name);
            if (array_it == array_meta.end())
                return false;

            auto& meta = array_it->second;
            if (meta.decomp_final)
                return false;
            if (candidate_tile <= 0)
                return false;

            candidate_tile =
                std::max(detail::next_pow2(candidate_tile), detail::ct_min_tile(meta.ndims));

            if (meta.tile > 0 && candidate_tile >= meta.tile)
                return false;
            if (!tile_fits_grid(meta, candidate_tile))
                return false;

            meta.tile = candidate_tile;
            return true;
        };

        auto propose_tile_from_access = [&](int array_name, int ndims, int source_name,
                                            Region* access_region) {
            if (source_name == array_name)
                return false;

            auto src_it = array_meta.find(source_name);
            if (src_it == array_meta.end() || src_it->second.ndims != ndims)
                return false;

            int candidate_tile = src_it->second.tile;
            int stride_factor = region_stride_factor(access_region, ndims);
            if (stride_factor > 1)
                candidate_tile = std::max(1, candidate_tile / stride_factor);

            return try_assign_tile(array_name, candidate_tile);
        };

        // Each temp is expanded once per walk: fused chains reuse temps, and
        // re-expanding per use is exponential in chain length.
        auto collect_tile_candidates =
            [&](auto&& self, int array_name, int ndims, ASTNode* node, Region* carried_region,
                const std::unordered_map<int, ASTNode*>& temp_defs,
                std::unordered_set<int>& visited_temps) -> bool {
            if (node == nullptr || node->is_scalar || node->is_broadcast)
                return false;

            auto meta_it = array_meta.find(node->result_name);
            if (meta_it != array_meta.end()) {
                if (meta_it->second.ndims == ndims)
                    return propose_tile_from_access(array_name, ndims, node->result_name,
                                                    carried_region);
                return false;
            }

            if (node->operands.empty()) {
                auto def_it = temp_defs.find(node->result_name);
                if (def_it != temp_defs.end()) {
                    if (!visited_temps.insert(node->result_name).second)
                        return false;
                    ASTNode* def_root = def_it->second;
                    bool changed = false;
                    for (int i = 0; i < (int)def_root->operands.size(); ++i) {
                        changed = self(self, array_name, ndims, def_root->operands[i],
                                       def_root->get_operand_region(i), temp_defs,
                                       visited_temps) ||
                                  changed;
                    }
                    return changed;
                }
                return false;
            }

            bool changed = false;
            for (int child_idx = 0; child_idx < (int)node->operands.size(); ++child_idx) {
                changed = self(self, array_name, ndims, node->operands[child_idx],
                               node->get_operand_region(child_idx), temp_defs, visited_temps) ||
                          changed;
            }
            return changed;
        };

        auto is_ast_leaf = [](ASTNode* node) -> bool {
            for (ASTNode* child : node->operands)
                if (child != nullptr && !child->is_scalar && !child->is_broadcast)
                    return false;
            return true;
        };

        bool tiles_changed = true;
        int max_tile_passes = std::max(64, static_cast<int>(array_meta.size()));
        for (int iter = 0; iter < max_tile_passes && tiles_changed; ++iter) {
            tiles_changed = false;

            detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
                std::unordered_map<int, ASTNode*> temp_defs;
                for (ASTNode* root : dag_node->ast->roots) {
                    if (root->is_temp)
                        temp_defs[root->result_name] = root;
                }

                for (ASTNode* ast_node : dag_node->ast->roots) {
                    Opcode op = static_cast<Opcode>(ast_node->opcode);

                    if (op == Opcode::SET_REGION) {
                        if (ast_node->operands.size() < 2)
                            continue;

                        int dst_name = ast_node->operands[0]->result_name;
                        auto dst_it = array_meta.find(dst_name);
                        if (dst_it == array_meta.end())
                            continue;

                        if (region_is_dense(ast_node->region, dst_it->second.ndims)) {
                            for (int op_idx = 1; op_idx < (int)ast_node->operands.size();
                                 ++op_idx) {
                                std::unordered_set<int> visited_temps;
                                tiles_changed =
                                    collect_tile_candidates(collect_tile_candidates, dst_name,
                                                            dst_it->second.ndims,
                                                            ast_node->operands[op_idx],
                                                            ast_node->get_operand_region(op_idx),
                                                            temp_defs, visited_temps) ||
                                    tiles_changed;
                            }
                        }

                        for (int op_idx = 1; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                            ASTNode* src_op = ast_node->operands[op_idx];
                            if (!is_ast_leaf(src_op))
                                continue;
                            auto src_it_cd = array_meta.find(src_op->result_name);
                            if (src_it_cd == array_meta.end())
                                continue;
                            if (src_it_cd->second.ndims == dst_it->second.ndims)
                                continue;
                            int higher_tile;
                            int lower_name_cd;
                            if (src_it_cd->second.ndims > dst_it->second.ndims) {
                                higher_tile = src_it_cd->second.tile;
                                lower_name_cd = dst_name;
                            } else {
                                higher_tile = dst_it->second.tile;
                                lower_name_cd = src_op->result_name;
                            }
                            tiles_changed =
                                try_assign_tile(lower_name_cd, higher_tile) || tiles_changed;
                        }
                        continue;
                    }

                    if (is_elementwise(op) || op == Opcode::COPY) {
                        auto out_it = array_meta.find(ast_node->result_name);
                        if (out_it == array_meta.end())
                            continue;

                        for (int op_idx = 0; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                            std::unordered_set<int> visited_temps;
                            tiles_changed =
                                collect_tile_candidates(collect_tile_candidates,
                                                        ast_node->result_name,
                                                        out_it->second.ndims,
                                                        ast_node->operands[op_idx],
                                                        ast_node->get_operand_region(op_idx),
                                                        temp_defs, visited_temps) ||
                                tiles_changed;
                        }
                    }
                }
            });
        }

        for (auto& [name, meta] : array_meta) {
            if (!meta.decomp_final)
                clamp_tile_for_max_count(meta);
        }

        auto extract_region_start = [](Region* region_base, int ndims) {
            std::array<int, 3> region_start = {0, 0, 0};
            if (region_base == nullptr || region_base->is_global)
                return region_start;
            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                region_start[0] = r->start[0];
                break;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                region_start[0] = r->start[0];
                region_start[1] = r->start[1];
                break;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                region_start[0] = r->start[0];
                region_start[1] = r->start[1];
                region_start[2] = r->start[2];
                break;
            }
            default:
                break;
            }
            return region_start;
        };

        auto extract_region_step = [](Region* region_base, int ndims) {
            std::array<int, 3> region_step = {1, 1, 1};
            if (region_base == nullptr || region_base->is_global)
                return region_step;
            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                region_step[0] = r->step[0];
                break;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                region_step[0] = r->step[0];
                region_step[1] = r->step[1];
                break;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                region_step[0] = r->step[0];
                region_step[1] = r->step[1];
                region_step[2] = r->step[2];
                break;
            }
            default:
                break;
            }
            return region_step;
        };

        auto extract_region_stop = [](Region* region_base, int ndims,
                                      const std::array<int, 3>& global_shape) {
            std::array<int, 3> region_stop = {0, 0, 0};
            if (region_base == nullptr || region_base->is_global) {
                for (int d = 0; d < ndims; ++d)
                    region_stop[d] = global_shape[d];
                return region_stop;
            }
            switch (ndims) {
            case 1: {
                auto* r = static_cast<ArrayRegion<1>*>(region_base);
                region_stop[0] = r->stop[0];
                break;
            }
            case 2: {
                auto* r = static_cast<ArrayRegion<2>*>(region_base);
                region_stop[0] = r->stop[0];
                region_stop[1] = r->stop[1];
                break;
            }
            case 3: {
                auto* r = static_cast<ArrayRegion<3>*>(region_base);
                region_stop[0] = r->stop[0];
                region_stop[1] = r->stop[1];
                region_stop[2] = r->stop[2];
                break;
            }
            default:
                break;
            }
            return region_stop;
        };

        auto circ_dist = [](int a, int b, int mod) -> int {
            int d = ((a - b) % mod + mod) % mod;
            return std::min(d, mod - d);
        };

        std::unordered_map<int, int> uf_parent, uf_rank;

        auto uf_make = [&](int x) {
            if (uf_parent.find(x) == uf_parent.end()) {
                uf_parent[x] = x;
                uf_rank[x] = 0;
            }
        };

        std::function<int(int)> uf_find = [&](int x) -> int {
            if (uf_parent[x] != x)
                uf_parent[x] = uf_find(uf_parent[x]);
            return uf_parent[x];
        };

        auto uf_unite = [&](int x, int y) {
            int rx = uf_find(x), ry = uf_find(y);
            if (rx == ry)
                return;
            if (uf_rank[rx] < uf_rank[ry])
                std::swap(rx, ry);
            uf_parent[ry] = rx;
            if (uf_rank[rx] == uf_rank[ry])
                uf_rank[rx]++;
        };

        for (auto& [name, meta] : array_meta)
            uf_make(name);

        std::unordered_set<int> source_arrays;

        detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
            for (ASTNode* ast_node : dag_node->ast->roots) {
                Opcode op = static_cast<Opcode>(ast_node->opcode);
                // TRANSPOSE results get a fixed offset (align_transpose_results)
                if (is_creation(op) || op == Opcode::TRANSPOSE)
                    source_arrays.insert(ast_node->result_name);
            }
        });

        auto collect_unionable_nonfixed_leaves = [&](auto&& self, ASTNode* node, int out_name,
                                                     int nd, int out_tile,
                                                     Region* carried_region,
                                                     std::vector<int>& leaves) -> void {
            if (node == nullptr || node->is_scalar || node->is_broadcast)
                return;

            if (is_ast_leaf(node)) {
                int src_name = node->result_name;
                if (src_name == out_name)
                    return;

                auto src_it = array_meta.find(src_name);
                if (src_it == array_meta.end() || src_it->second.ndims != nd ||
                    src_it->second.tile != out_tile)
                    return;

                bool is_fixed_source = src_it->second.decomp_final || source_arrays.count(src_name);
                if (is_fixed_source)
                    return;

                if (carried_region != nullptr && !carried_region->is_global) {
                    auto rstart = extract_region_start(carried_region, nd);
                    auto rstep = extract_region_step(carried_region, nd);
                    for (int d = 0; d < nd; ++d) {
                        if (rstart[d] != 0 || rstep[d] != 1)
                            return;
                    }
                }

                leaves.push_back(src_name);
                return;
            }

            for (int i = 0; i < (int)node->operands.size(); ++i) {
                ASTNode* child = node->operands[i];
                if (child == nullptr || child->is_scalar || child->is_broadcast)
                    continue;
                Region* child_region = node->get_operand_region(i);
                self(self, child, out_name, nd, out_tile,
                     child_region ? child_region : carried_region, leaves);
            }
        };

        detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
            for (ASTNode* ast_node : dag_node->ast->roots) {
                Opcode op = static_cast<Opcode>(ast_node->opcode);

                if (is_creation(op)) {
                    continue;
                }

                if (is_elementwise(op) || op == Opcode::COPY) {
                    int out_name = ast_node->result_name;
                    auto out_it = array_meta.find(out_name);
                    if (out_it == array_meta.end())
                        continue;
                    int nd = out_it->second.ndims;
                    int out_tile = out_it->second.tile;

                    std::vector<int> union_leaves;
                    for (int op_idx = 0; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                        ASTNode* operand = ast_node->operands[op_idx];
                        if (operand == nullptr || operand->is_scalar || operand->is_broadcast)
                            continue;
                        Region* op_region = ast_node->get_operand_region(op_idx);
                        collect_unionable_nonfixed_leaves(collect_unionable_nonfixed_leaves,
                                                          operand, out_name, nd, out_tile,
                                                          op_region, union_leaves);
                    }
                    for (int src_name : union_leaves)
                        uf_unite(out_name, src_name);
                    continue;
                }

                if (op == Opcode::MATMUL || op == Opcode::MATMATMUL) {
                    for (ASTNode* operand : ast_node->operands)
                        if (operand != nullptr && !operand->is_scalar)
                            matmul_arrays.insert(operand->result_name);
                    matmul_arrays.insert(ast_node->result_name);
                }

                // These results are routed by a zero-offset decomposition; BINCOUNT
                // also needs x and weights laid out identically.
                if (is_reduction_op(op) || op == Opcode::LINALG || is_shape_op(op) ||
                    is_signal_op(op) || op == Opcode::GATHER || op == Opcode::EXPAND) {
                    matmul_arrays.insert(ast_node->result_name);
                    if (op == Opcode::BINCOUNT)
                        for (ASTNode* operand : ast_node->operands)
                            if (operand != nullptr && !operand->is_scalar)
                                matmul_arrays.insert(operand->result_name);
                }
            }
        });

        // TRANSPOSE result: source tile, permuted (source offset + view start),
        // so source tiles map 1:1 onto result tiles. Matmul operands stay at 0.
        auto align_transpose_results = [&]() {
            detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
                for (ASTNode* ast_node : dag_node->ast->roots) {
                    if (static_cast<Opcode>(ast_node->opcode) != Opcode::TRANSPOSE ||
                        ast_node->operands.empty())
                        continue;
                    auto in_it = array_meta.find(ast_node->operands[0]->result_name);
                    auto out_it = array_meta.find(ast_node->result_name);
                    if (in_it == array_meta.end() || out_it == array_meta.end() ||
                        out_it->second.decomp_final)
                        continue;
                    auto& in = in_it->second;
                    auto& out = out_it->second;
                    int nd = out.ndims;
                    if (in.ndims != nd || (int)ast_node->operands.size() < nd + 1)
                        continue;
                    Region* view = ast_node->get_operand_region(0);
                    auto vstart = extract_region_start(view, nd);
                    auto vstep = extract_region_step(view, nd);
                    out.tile = in.tile;
                    out.offset = zero_offset;
                    if (!matmul_arrays.count(ast_node->result_name)) {
                        for (int k = 0; k < nd; ++k) {
                            int d = (int)ast_node->operands[k + 1]->scalar;
                            if (vstep[d] == 1)
                                out.offset[k] = positive_mod(in.offset[d] + vstart[d], in.tile);
                        }
                    }
                }
            });
        };
        align_transpose_results();

        struct ClassInfo {
            int tile = 0;
            int ndims = 0;
            bool has_fixed_offset = false;
            std::array<int, 3> fixed_offset = {0, 0, 0};
        };
        std::unordered_map<int, ClassInfo> class_info;

        for (auto& [name, meta] : array_meta) {
            int rep = uf_find(name);
            auto it = class_info.find(rep);
            bool is_fixed = meta.decomp_final || source_arrays.count(name);
            if (it == class_info.end()) {
                class_info[rep] = {meta.tile, meta.ndims, is_fixed, meta.offset};
            } else if (is_fixed) {
                it->second.has_fixed_offset = true;
                it->second.fixed_offset = meta.offset;
            }
        }

        struct ShiftEdge {
            int from_class;
            int to_class;
            std::array<int, 3> shift;
            std::array<int, 3> stride;
            int64_t weight;
            bool forward;
            std::array<int, 3> from_to_to = {0, 1, 2};
            std::array<int, 3> to_to_from = {0, 1, 2};
        };
        std::vector<ShiftEdge> shift_edges;

        auto add_shift_edge_if_needed = [&](int out_name, int nd, ASTNode* leaf, Region* region) {
            int src_name = leaf->result_name;
            if (src_name == out_name)
                return;
            auto src_it = array_meta.find(src_name);
            if (src_it == array_meta.end() || src_it->second.ndims != nd)
                return;

            int class_src = uf_find(src_name);
            int class_out = uf_find(out_name);
            if (class_src == class_out)
                return;

            std::array<int, 3> rstart = {0, 0, 0};
            std::array<int, 3> rstep = {1, 1, 1};
            if (region != nullptr && !region->is_global) {
                rstart = extract_region_start(region, nd);
                rstep = extract_region_step(region, nd);
            }

            int64_t weight = 1;
            for (int d = 0; d < nd; ++d)
                weight *= static_cast<int64_t>(src_it->second.global_shape[d]);

            shift_edges.push_back({class_src, class_out, rstart, rstep, weight, true});
        };

        auto visit_leaves_for_edges = [&](auto&& self, ASTNode* node, int out_name, int nd,
                                          Region* carried_region) -> void {
            if (node == nullptr || node->is_scalar || node->is_broadcast)
                return;
            if (is_ast_leaf(node)) {
                add_shift_edge_if_needed(out_name, nd, node, carried_region);
                return;
            }
            for (int i = 0; i < (int)node->operands.size(); ++i) {
                ASTNode* child = node->operands[i];
                if (child == nullptr || child->is_scalar || child->is_broadcast)
                    continue;
                Region* child_region = node->get_operand_region(i);
                if (is_ast_leaf(child)) {
                    add_shift_edge_if_needed(out_name, nd, child,
                                             child_region ? child_region : carried_region);
                } else {
                    self(self, child, out_name, nd, child_region);
                }
            }
        };

        auto visit_set_region_leaves = [&](auto&& self, ASTNode* node, int target_name, int nd,
                                           const std::array<int, 3>& region_start) -> void {
            if (node == nullptr || node->is_scalar || node->is_broadcast)
                return;
            if (is_ast_leaf(node)) {
                int src_name = node->result_name;
                if (src_name == target_name)
                    return;
                auto src_it = array_meta.find(src_name);
                if (src_it == array_meta.end() || src_it->second.ndims != nd)
                    return;

                int class_src = uf_find(src_name);
                int class_tgt = uf_find(target_name);
                if (class_src == class_tgt)
                    return;

                int64_t weight = 1;
                for (int d = 0; d < nd; ++d)
                    weight *= static_cast<int64_t>(src_it->second.global_shape[d]);

                shift_edges.push_back(
                    {class_tgt, class_src, region_start, {1, 1, 1}, weight, true});
                return;
            }
            for (ASTNode* child : node->operands)
                self(self, child, target_name, nd, region_start);
        };

        detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
            for (ASTNode* ast_node : dag_node->ast->roots) {
                Opcode op = static_cast<Opcode>(ast_node->opcode);

                if (op == Opcode::SET_REGION) {
                    if (ast_node->operands.size() < 2 || ast_node->region == nullptr)
                        continue;
                    int target_name = ast_node->operands[0]->result_name;
                    auto target_it = array_meta.find(target_name);
                    if (target_it == array_meta.end())
                        continue;
                    int nd = target_it->second.ndims;
                    auto region_start = extract_region_start(ast_node->region, nd);

                    for (int op_idx = 1; op_idx < (int)ast_node->operands.size(); ++op_idx)
                        visit_set_region_leaves(visit_set_region_leaves,
                                                ast_node->operands[op_idx], target_name, nd,
                                                region_start);

                    for (int op_idx = 1; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                        ASTNode* src_op = ast_node->operands[op_idx];
                        if (!is_ast_leaf(src_op))
                            continue;
                        int src_name_cd = src_op->result_name;
                        if (src_name_cd == target_name)
                            continue;
                        auto src_it_cd = array_meta.find(src_name_cd);
                        if (src_it_cd == array_meta.end())
                            continue;
                        int nd_src = src_it_cd->second.ndims;
                        if (nd_src == nd)
                            continue;

                        std::array<int, 3> from_to_to_cd = {-1, -1, -1};
                        std::array<int, 3> to_to_from_cd = {-1, -1, -1};

                        if (nd < nd_src) {
                            int td = 0;
                            for (int sd = 0; sd < nd_src && td < nd; ++sd) {
                                if (src_it_cd->second.global_shape[sd] > 1) {
                                    from_to_to_cd[td] = sd;
                                    to_to_from_cd[sd] = td;
                                    ++td;
                                }
                            }
                            if (td != nd)
                                continue;
                        } else {
                            auto region_stop = extract_region_stop(
                                ast_node->region, nd, target_it->second.global_shape);
                            int sd = 0;
                            for (int td = 0; td < nd && sd < nd_src; ++td) {
                                int size = region_stop[td] - region_start[td];
                                if (size > 1) {
                                    from_to_to_cd[td] = sd;
                                    to_to_from_cd[sd] = td;
                                    ++sd;
                                }
                            }
                            if (sd != nd_src)
                                continue;
                        }

                        int class_tgt_cd = uf_find(target_name);
                        int class_src_cd = uf_find(src_name_cd);
                        if (class_tgt_cd == class_src_cd)
                            continue;

                        int64_t weight_cd = 1;
                        int nd_lo = std::min(nd, nd_src);
                        auto& lo_meta = (nd <= nd_src) ? target_it->second : src_it_cd->second;
                        for (int d = 0; d < nd_lo; ++d)
                            weight_cd *= static_cast<int64_t>(lo_meta.global_shape[d]);

                        shift_edges.push_back({class_tgt_cd, class_src_cd, region_start,
                                               {1, 1, 1}, weight_cd, true, from_to_to_cd,
                                               to_to_from_cd});
                    }
                    continue;
                }

                if (is_elementwise(op) || op == Opcode::COPY) {
                    int out_name = ast_node->result_name;
                    auto out_it = array_meta.find(out_name);
                    if (out_it == array_meta.end())
                        continue;
                    int nd = out_it->second.ndims;

                    for (int op_idx = 0; op_idx < (int)ast_node->operands.size(); ++op_idx) {
                        ASTNode* operand = ast_node->operands[op_idx];
                        if (operand == nullptr || operand->is_scalar || operand->is_broadcast)
                            continue;
                        Region* op_region = ast_node->get_operand_region(op_idx);
                        if (is_ast_leaf(operand)) {
                            add_shift_edge_if_needed(out_name, nd, operand, op_region);
                        } else {
                            visit_leaves_for_edges(visit_leaves_for_edges, operand, out_name, nd,
                                                   op_region);
                        }
                    }
                }
            }
        });

        struct AdjEdge {
            int neighbor;
            std::array<int, 3> shift;
            std::array<int, 3> stride;
            std::array<int, 3> neighbor_dim = {0, 1, 2};
            int64_t weight;
            bool forward;
            int from_class;
            int to_class;
        };
        std::unordered_map<int, std::vector<AdjEdge>> adj;

        for (auto& e : shift_edges) {
            if (e.from_class == e.to_class)
                continue;
            adj[e.from_class].push_back(
                {e.to_class, e.shift, e.stride, e.from_to_to, e.weight, true, e.from_class,
                 e.to_class});
            adj[e.to_class].push_back(
                {e.from_class, e.shift, e.stride, e.to_to_from, e.weight, false, e.from_class,
                 e.to_class});
        }

        std::unordered_map<int, int> comp_id;
        std::unordered_map<int, int> tree_parent;
        std::unordered_map<int, std::vector<int>> tree_children;
        std::unordered_set<int> cyclic_components;
        int num_components = 0;

        std::unordered_set<int> all_classes;
        for (auto& [name, _] : array_meta)
            all_classes.insert(uf_find(name));

        std::unordered_map<int, int> comp_root;

        for (int cls : all_classes) {
            if (comp_id.count(cls))
                continue;
            int cid = num_components++;
            comp_root[cid] = cls;

            std::queue<int> bfs;
            bfs.push(cls);
            comp_id[cls] = cid;
            tree_parent[cls] = -1;

            while (!bfs.empty()) {
                int v = bfs.front();
                bfs.pop();
                std::unordered_set<int> seen_neighbors;
                for (auto& e : adj[v]) {
                    if (seen_neighbors.count(e.neighbor))
                        continue;
                    seen_neighbors.insert(e.neighbor);
                    if (!comp_id.count(e.neighbor)) {
                        comp_id[e.neighbor] = cid;
                        tree_parent[e.neighbor] = v;
                        tree_children[v].push_back(e.neighbor);
                        bfs.push(e.neighbor);
                    } else if (e.neighbor != tree_parent[v]) {
                        cyclic_components.insert(cid);
                    }
                }
                auto ci = class_info.find(v);
                if (ci != class_info.end() && ci->second.has_fixed_offset)
                    comp_root[cid] = v;
            }
        }

        tree_parent.clear();
        tree_children.clear();

        for (auto& [cid, root] : comp_root) {
            std::queue<int> bfs;
            bfs.push(root);
            tree_parent[root] = -1;

            std::unordered_set<int> visited;
            visited.insert(root);

            while (!bfs.empty()) {
                int v = bfs.front();
                bfs.pop();
                for (auto& e : adj[v]) {
                    if (!visited.count(e.neighbor) && comp_id[e.neighbor] == cid) {
                        visited.insert(e.neighbor);
                        tree_parent[e.neighbor] = v;
                        tree_children[v].push_back(e.neighbor);
                        bfs.push(e.neighbor);
                    }
                }
            }
        }

        constexpr int64_t INF = std::numeric_limits<int64_t>::max() / 2;

        std::unordered_map<int, std::array<std::vector<int>, 3>> cands;

        auto add_cand = [](std::vector<int>& v, int o) {
            auto it = std::lower_bound(v.begin(), v.end(), o);
            if (it == v.end() || *it != o)
                v.insert(it, o);
        };

        auto compute_desired = [&](int o_from, int shift_d, int stride_d, int tile_to) -> int {
            int step = stride_d > 1 ? stride_d : 1;
            return positive_mod((o_from + shift_d) / step, tile_to);
        };

        auto reverse_desired = [&](int o_target, int shift_d, int stride_d, int tile_from,
                                   int tile_to) -> std::vector<int> {
            int step = stride_d > 1 ? stride_d : 1;
            std::vector<int> result;
            int base = o_target * step - shift_d;
            int period = tile_to * step;
            if (period <= 0)
                period = 1;
            int num_periods = (tile_from + period - 1) / period;
            for (int k = 0; k <= num_periods; ++k) {
                for (int delta = 0; delta < step; ++delta) {
                    int raw = base + delta + k * period;
                    int cand = ((raw % tile_from) + tile_from) % tile_from;
                    if (cand >= 0 && cand < tile_from &&
                        compute_desired(cand, shift_d, stride_d, tile_to) == o_target) {
                        result.push_back(cand);
                    }
                }
            }
            std::sort(result.begin(), result.end());
            result.erase(std::unique(result.begin(), result.end()), result.end());
            return result;
        };

        std::vector<int> post_order;
        for (auto& [cid, root] : comp_root) {
            if (cyclic_components.count(cid))
                continue;

            std::stack<std::pair<int, bool>> stk;
            stk.push({root, false});
            while (!stk.empty()) {
                auto [v, processed] = stk.top();
                stk.pop();
                if (processed) {
                    post_order.push_back(v);
                    continue;
                }
                stk.push({v, true});
                for (int c : tree_children[v])
                    stk.push({c, false});
            }
        }

        for (int v : post_order) {
            auto ci = class_info.find(v);
            if (ci == class_info.end())
                continue;
            int nd = ci->second.ndims;
            int tile_v = ci->second.tile;
            if (tile_v <= 0)
                tile_v = 1;

            auto& cands_v = cands[v];

            if (ci->second.has_fixed_offset) {
                for (int d = 0; d < nd; ++d)
                    add_cand(cands_v[d], positive_mod(ci->second.fixed_offset[d], tile_v));
            }

            for (int c : tree_children[v]) {
                auto cc = class_info.find(c);
                if (cc == class_info.end())
                    continue;
                int tile_c = cc->second.tile;
                if (tile_c <= 0)
                    tile_c = 1;

                for (auto& e : adj[v]) {
                    if (e.neighbor != c)
                        continue;
                    for (int d = 0; d < nd; ++d) {
                        int d_c = e.neighbor_dim[d];
                        if (d_c < 0)
                            continue;
                        int d_from = e.forward ? d : d_c;
                        for (int oc : cands[c][d_c]) {
                            if (e.forward) {
                                for (int rv : reverse_desired(oc, e.shift[d_from],
                                                              e.stride[d_from], tile_v, tile_c))
                                    add_cand(cands_v[d], rv);
                            } else {
                                add_cand(cands_v[d], compute_desired(oc, e.shift[d_from],
                                                                     e.stride[d_from], tile_v));
                            }
                        }
                    }
                }
            }

            for (int d = 0; d < nd; ++d) {
                if (cands_v[d].empty())
                    add_cand(cands_v[d], 0);
            }
        }

        {
            std::queue<int> work;
            for (auto& [cid, root] : comp_root) {
                if (!cyclic_components.count(cid))
                    work.push(root);
            }

            while (!work.empty()) {
                int v = work.front();
                work.pop();
                auto ci = class_info.find(v);
                if (ci == class_info.end())
                    continue;
                int nd = ci->second.ndims;
                int tile_v = ci->second.tile;
                if (tile_v <= 0)
                    tile_v = 1;

                for (int c : tree_children[v]) {
                    auto cc = class_info.find(c);
                    if (cc == class_info.end())
                        continue;
                    int tile_c = cc->second.tile;
                    if (tile_c <= 0)
                        tile_c = 1;

                    for (auto& e : adj[v]) {
                        if (e.neighbor != c)
                            continue;
                        for (int d = 0; d < nd; ++d) {
                            int d_c = e.neighbor_dim[d];
                            if (d_c < 0)
                                continue;
                            int d_from = e.forward ? d : d_c;
                            for (int ov : cands[v][d]) {
                                if (e.forward) {
                                    add_cand(cands[c][d_c],
                                             compute_desired(ov, e.shift[d_from],
                                                             e.stride[d_from], tile_c));
                                } else {
                                    for (int rc : reverse_desired(ov, e.shift[d_from],
                                                                  e.stride[d_from], tile_c, tile_v))
                                        add_cand(cands[c][d_c], rc);
                                }
                            }
                        }
                    }
                    work.push(c);
                }
            }
        }

        std::unordered_map<int, std::array<std::vector<int64_t>, 3>> dp;

        for (int v : post_order) {
            auto ci = class_info.find(v);
            if (ci == class_info.end())
                continue;
            int nd = ci->second.ndims;
            int tile_v = ci->second.tile;
            if (tile_v <= 0)
                tile_v = 1;

            auto& dp_v = dp[v];
            auto& cands_v = cands[v];

            for (int d = 0; d < 3; ++d) {
                int nc = (int)cands_v[d].size();
                dp_v[d].assign(nc, 0);
                if (d < nd && ci->second.has_fixed_offset) {
                    int fixed = positive_mod(ci->second.fixed_offset[d], tile_v);
                    for (int i = 0; i < nc; ++i)
                        dp_v[d][i] = (cands_v[d][i] == fixed) ? 0 : INF;
                }
            }

            for (int c : tree_children[v]) {
                auto cc = class_info.find(c);
                if (cc == class_info.end())
                    continue;
                int tile_c = cc->second.tile;
                if (tile_c <= 0)
                    tile_c = 1;

                std::vector<const AdjEdge*> edges_vc;
                for (auto& e : adj[v])
                    if (e.neighbor == c)
                        edges_vc.push_back(&e);

                auto& cands_c = cands[c];
                auto& dp_c = dp[c];

                for (int d = 0; d < nd; ++d) {
                    int d_c = d;
                    for (auto* ep : edges_vc) {
                        if (ep->neighbor_dim[d] != d) {
                            d_c = ep->neighbor_dim[d];
                            break;
                        }
                    }
                    if (d_c < 0)
                        continue;

                    int nc_v = (int)cands_v[d].size();
                    int nc_cd = (int)cands_c[d_c].size();

                    std::vector<int64_t> h(nc_v, INF);
                    for (int iv = 0; iv < nc_v; ++iv) {
                        int ov = cands_v[d][iv];
                        for (int ic = 0; ic < nc_cd; ++ic) {
                            int oc = cands_c[d_c][ic];
                            int64_t cost = dp_c[d_c][ic];
                            if (cost >= INF)
                                continue;
                            for (auto* ep : edges_vc) {
                                int ep_dc = ep->neighbor_dim[d];
                                if (ep_dc < 0)
                                    continue;
                                int d_from = ep->forward ? d : ep_dc;
                                int o_from, tile_to, o_to;
                                if (ep->forward) {
                                    o_from = ov;
                                    tile_to = tile_c;
                                    o_to = oc;
                                } else {
                                    o_from = oc;
                                    tile_to = tile_v;
                                    o_to = ov;
                                }
                                int step = ep->stride[d_from] > 1 ? ep->stride[d_from] : 1;
                                int desired = positive_mod(
                                    (o_from + ep->shift[d_from]) / step, tile_to);
                                cost += ep->weight * circ_dist(desired, o_to, tile_to);
                                if (cost >= INF)
                                    break;
                            }
                            if (cost < h[iv])
                                h[iv] = cost;
                        }
                    }
                    for (int iv = 0; iv < nc_v; ++iv) {
                        if (dp_v[d][iv] < INF && h[iv] < INF)
                            dp_v[d][iv] += h[iv];
                        else
                            dp_v[d][iv] = INF;
                    }
                }
            }
        }

        std::unordered_map<int, std::array<int, 3>> class_phase;

        for (auto& [cid, root] : comp_root) {
            if (cyclic_components.count(cid))
                continue;
            auto ci = class_info.find(root);
            if (ci == class_info.end())
                continue;
            int nd = ci->second.ndims;

            std::array<int, 3> best = {0, 0, 0};
            for (int d = 0; d < nd; ++d) {
                int64_t best_cost = INF;
                auto& cv = cands[root][d];
                auto& dv = dp[root][d];
                for (int i = 0; i < (int)cv.size(); ++i) {
                    if (dv[i] < best_cost) {
                        best_cost = dv[i];
                        best[d] = cv[i];
                    }
                }
            }
            class_phase[root] = best;
        }

        {
            std::queue<int> work;
            for (auto& [cid, root] : comp_root) {
                if (!cyclic_components.count(cid))
                    work.push(root);
            }

            while (!work.empty()) {
                int v = work.front();
                work.pop();
                auto ci_v = class_info.find(v);
                if (ci_v == class_info.end())
                    continue;
                int nd = ci_v->second.ndims;
                int tile_v = ci_v->second.tile;
                if (tile_v <= 0)
                    tile_v = 1;

                for (int c : tree_children[v]) {
                    auto ci_c = class_info.find(c);
                    if (ci_c == class_info.end())
                        continue;
                    int tile_c = ci_c->second.tile;
                    if (tile_c <= 0)
                        tile_c = 1;

                    std::vector<const AdjEdge*> edges_vc;
                    for (auto& e : adj[v])
                        if (e.neighbor == c)
                            edges_vc.push_back(&e);

                    auto& cands_c = cands[c];
                    auto& dp_c = dp[c];
                    int nd_c = ci_c->second.ndims;

                    std::array<int, 3> best_c = {0, 0, 0};
                    for (int d_c = 0; d_c < nd_c; ++d_c) {
                        int ov = -1;
                        int d_parent = -1;
                        for (int dp_d = 0; dp_d < nd; ++dp_d) {
                            for (auto* ep : edges_vc) {
                                if (ep->neighbor_dim[dp_d] == d_c) {
                                    d_parent = dp_d;
                                    ov = class_phase[v][dp_d];
                                    break;
                                }
                            }
                            if (d_parent >= 0)
                                break;
                        }

                        if (ov < 0) {
                            int64_t best_cost = INF;
                            for (int ic = 0; ic < (int)cands_c[d_c].size(); ++ic) {
                                if (dp_c[d_c][ic] < best_cost) {
                                    best_cost = dp_c[d_c][ic];
                                    best_c[d_c] = cands_c[d_c][ic];
                                }
                            }
                            continue;
                        }

                        int64_t best_cost = INF;
                        int nc_cd = (int)cands_c[d_c].size();
                        for (int ic = 0; ic < nc_cd; ++ic) {
                            int oc = cands_c[d_c][ic];
                            int64_t cost = dp_c[d_c][ic];
                            if (cost >= INF)
                                continue;
                            for (auto* ep : edges_vc) {
                                int ep_dc = ep->neighbor_dim[d_parent];
                                if (ep_dc < 0 || ep_dc != d_c)
                                    continue;
                                int d_from = ep->forward ? d_parent : d_c;
                                int o_from, tile_to, o_to;
                                if (ep->forward) {
                                    o_from = ov;
                                    tile_to = tile_c;
                                    o_to = oc;
                                } else {
                                    o_from = oc;
                                    tile_to = tile_v;
                                    o_to = ov;
                                }
                                int step = ep->stride[d_from] > 1 ? ep->stride[d_from] : 1;
                                int desired = positive_mod(
                                    (o_from + ep->shift[d_from]) / step, tile_to);
                                cost += ep->weight * circ_dist(desired, o_to, tile_to);
                                if (cost >= INF)
                                    break;
                            }
                            if (cost < best_cost) {
                                best_cost = cost;
                                best_c[d_c] = oc;
                            }
                        }
                    }
                    class_phase[c] = best_c;
                    work.push(c);
                }
            }
        }

        auto positive_mod64 = [](int64_t value, int mod) -> int {
            if (mod <= 0)
                return static_cast<int>(value);
            int64_t r = value % mod;
            if (r < 0)
                r += mod;
            return static_cast<int>(r);
        };

        auto abs_dist64 = [](int64_t a, int64_t b) -> int64_t {
            return (a >= b) ? (a - b) : (b - a);
        };

        auto nearest_congruent = [&](int64_t anchor, int phase, int tile) -> int {
            if (tile <= 0)
                return static_cast<int>(anchor);

            int target = positive_mod(phase, tile);
            int64_t lower = target;
            if (anchor >= target) {
                lower += ((anchor - target) / tile) * static_cast<int64_t>(tile);
            } else {
                lower -= ((target - anchor + tile - 1) / tile) * static_cast<int64_t>(tile);
            }
            int64_t upper = lower + tile;
            return (abs_dist64(upper, anchor) < abs_dist64(lower, anchor))
                       ? static_cast<int>(upper)
                       : static_cast<int>(lower);
        };

        auto reverse_absolute = [&](int64_t target_abs, int shift_d, int stride_d, int phase_from,
                                    int tile_from) -> int {
            int step = stride_d > 1 ? stride_d : 1;
            int64_t low = target_abs * step - shift_d;
            int64_t center2 = 2 * low + (step - 1);
            bool found = false;
            int64_t best = low;
            int64_t best_score = 0;
            int phase = positive_mod(phase_from, tile_from);

            for (int delta = 0; delta < step; ++delta) {
                int64_t cand = low + delta;
                if (positive_mod64(cand, tile_from) != phase)
                    continue;
                if ((cand + shift_d) / step != target_abs)
                    continue;
                int64_t score = abs_dist64(2 * cand, center2);
                if (!found || score < best_score || (score == best_score && cand < best)) {
                    found = true;
                    best = cand;
                    best_score = score;
                }
            }

            if (found)
                return static_cast<int>(best);

            return nearest_congruent(low, phase_from, tile_from);
        };

        struct AbsoluteProposal {
            int value = 0;
            int64_t weight = 0;
        };

        auto choose_absolute = [&](const std::vector<AbsoluteProposal>& proposals, int phase,
                                   int tile) -> int {
            std::vector<int> candidates;
            candidates.push_back(nearest_congruent(phase, phase, tile));
            for (auto const& p : proposals) {
                auto it = std::lower_bound(candidates.begin(), candidates.end(), p.value);
                if (it == candidates.end() || *it != p.value)
                    candidates.insert(it, p.value);
            }

            int best = candidates.front();
            int64_t best_cost = std::numeric_limits<int64_t>::max();
            for (int cand : candidates) {
                int64_t cost = 0;
                for (auto const& p : proposals)
                    cost += p.weight * abs_dist64(cand, p.value);
                if (cost < best_cost || (cost == best_cost && cand < best)) {
                    best_cost = cost;
                    best = cand;
                }
            }
            return best;
        };

        std::unordered_map<int, std::array<int, 3>> class_abs_offset;

        for (auto& [cid, root] : comp_root) {
            if (cyclic_components.count(cid))
                continue;

            auto ci = class_info.find(root);
            auto phase_it = class_phase.find(root);
            if (ci == class_info.end() || phase_it == class_phase.end())
                continue;

            std::array<int, 3> root_abs = {0, 0, 0};
            int tile_root = ci->second.tile;
            if (tile_root <= 0)
                tile_root = 1;

            for (int d = 0; d < ci->second.ndims; ++d) {
                if (ci->second.has_fixed_offset)
                    root_abs[d] = ci->second.fixed_offset[d];
                else
                    root_abs[d] =
                        nearest_congruent(phase_it->second[d], phase_it->second[d], tile_root);
            }
            class_abs_offset[root] = root_abs;
        }

        {
            std::queue<int> work;
            for (auto& [cid, root] : comp_root) {
                if (!cyclic_components.count(cid) && class_abs_offset.count(root))
                    work.push(root);
            }

            while (!work.empty()) {
                int v = work.front();
                work.pop();

                auto ci_v = class_info.find(v);
                auto abs_v_it = class_abs_offset.find(v);
                if (ci_v == class_info.end() || abs_v_it == class_abs_offset.end())
                    continue;
                int nd_v = ci_v->second.ndims;

                for (int c : tree_children[v]) {
                    auto ci_c = class_info.find(c);
                    auto phase_c_it = class_phase.find(c);
                    if (ci_c == class_info.end() || phase_c_it == class_phase.end())
                        continue;

                    int tile_c = ci_c->second.tile;
                    if (tile_c <= 0)
                        tile_c = 1;

                    std::vector<const AdjEdge*> edges_vc;
                    for (auto& e : adj[v])
                        if (e.neighbor == c)
                            edges_vc.push_back(&e);

                    std::array<int, 3> abs_c = {0, 0, 0};
                    for (int d_c = 0; d_c < ci_c->second.ndims; ++d_c) {
                        if (ci_c->second.has_fixed_offset) {
                            abs_c[d_c] = ci_c->second.fixed_offset[d_c];
                            continue;
                        }

                        std::vector<AbsoluteProposal> proposals;
                        for (auto* ep : edges_vc) {
                            for (int d_parent = 0; d_parent < nd_v; ++d_parent) {
                                if (ep->neighbor_dim[d_parent] != d_c)
                                    continue;

                                int value = 0;
                                if (ep->forward) {
                                    int d_from = d_parent;
                                    int step = ep->stride[d_from] > 1 ? ep->stride[d_from] : 1;
                                    int64_t anchor =
                                        (static_cast<int64_t>(abs_v_it->second[d_parent]) +
                                         ep->shift[d_from]) /
                                        step;
                                    value =
                                        nearest_congruent(anchor, phase_c_it->second[d_c], tile_c);
                                } else {
                                    int d_from = d_c;
                                    value = reverse_absolute(abs_v_it->second[d_parent],
                                                             ep->shift[d_from],
                                                             ep->stride[d_from],
                                                             phase_c_it->second[d_c], tile_c);
                                }
                                proposals.push_back({value, ep->weight});
                            }
                        }

                        abs_c[d_c] =
                            choose_absolute(proposals, phase_c_it->second[d_c], tile_c);
                    }

                    class_abs_offset[c] = abs_c;
                    work.push(c);
                }
            }
        }

        for (auto& [name, meta] : array_meta) {
            if (meta.decomp_final)
                continue;
            if (matmul_arrays.count(name))
                continue;
            int rep = uf_find(name);
            auto comp_it = comp_id.find(rep);
            if (comp_it != comp_id.end() && cyclic_components.count(comp_it->second))
                continue;
            auto it = class_abs_offset.find(rep);
            if (it != class_abs_offset.end())
                meta.offset = it->second;
        }

        if (!cyclic_components.empty()) {
            struct DesiredOffset {
                std::array<int, 3> desired_phase;
                std::array<int, 3> desired_anchor;
                int64_t weight;
            };
            std::unordered_map<int, std::vector<DesiredOffset>> all_desired;

            auto collect_fallback_producer =
                [&](auto&& self, ASTNode* node, int out_name, int nd, int out_tile) -> void {
                if (node == nullptr || node->is_scalar || node->is_broadcast)
                    return;
                if (is_ast_leaf(node)) {
                    int src_name = node->result_name;
                    if (src_name == out_name)
                        return;
                    auto src_it = array_meta.find(src_name);
                    if (src_it == array_meta.end() || src_it->second.ndims != nd)
                        return;
                    if (out_tile <= 0)
                        return;
                    std::array<int, 3> desired_phase = {0, 0, 0};
                    std::array<int, 3> desired_anchor = {0, 0, 0};
                    for (int d = 0; d < nd; ++d) {
                        desired_anchor[d] = src_it->second.offset[d];
                        desired_phase[d] = positive_mod(src_it->second.offset[d], out_tile);
                    }
                    int64_t weight = 1;
                    for (int d = 0; d < nd; ++d)
                        weight *= static_cast<int64_t>(src_it->second.global_shape[d]);
                    all_desired[out_name].push_back({desired_phase, desired_anchor, weight});
                    return;
                }
                for (int i = 0; i < (int)node->operands.size(); ++i) {
                    ASTNode* child = node->operands[i];
                    if (child == nullptr || child->is_scalar || child->is_broadcast)
                        continue;
                    Region* child_region = node->get_operand_region(i);
                    if (is_ast_leaf(child) && child_region != nullptr && !child_region->is_global) {
                        int src_name = child->result_name;
                        if (src_name == out_name)
                            continue;
                        auto src_it = array_meta.find(src_name);
                        if (src_it == array_meta.end() || src_it->second.ndims != nd)
                            continue;
                        if (out_tile <= 0)
                            continue;
                        auto rstart = extract_region_start(child_region, nd);
                        auto rstep = extract_region_step(child_region, nd);
                        std::array<int, 3> desired_phase = {0, 0, 0};
                        std::array<int, 3> desired_anchor = {0, 0, 0};
                        for (int d = 0; d < nd; ++d) {
                            int step = (rstep[d] > 1) ? rstep[d] : 1;
                            int raw = src_it->second.offset[d] + rstart[d];
                            desired_anchor[d] = (step > 1) ? (raw / step) : raw;
                            desired_phase[d] = (step > 1)
                                                   ? positive_mod(raw / step, out_tile)
                                                   : positive_mod(raw, out_tile);
                        }
                        int64_t weight = 1;
                        for (int d = 0; d < nd; ++d)
                            weight *= static_cast<int64_t>(src_it->second.global_shape[d]);
                        all_desired[out_name].push_back(
                            {desired_phase, desired_anchor, weight});
                    } else {
                        self(self, child, out_name, nd, out_tile);
                    }
                }
            };

            auto collect_fallback_consumer =
                [&](auto&& self, ASTNode* node, int target_name, int nd,
                    const ArrayMetadata& target_meta, const std::array<int, 3>& region_start)
                -> void {
                if (node == nullptr || node->is_scalar || node->is_broadcast)
                    return;
                if (is_ast_leaf(node)) {
                    int src_name = node->result_name;
                    if (src_name == target_name)
                        return;
                    auto src_it = array_meta.find(src_name);
                    if (src_it == array_meta.end() || src_it->second.ndims != nd)
                        return;
                    int tile = src_it->second.tile;
                    if (tile <= 0)
                        return;
                    std::array<int, 3> desired_phase = {0, 0, 0};
                    std::array<int, 3> desired_anchor = {0, 0, 0};
                    for (int d = 0; d < nd; ++d) {
                        desired_anchor[d] = target_meta.offset[d] + region_start[d];
                        desired_phase[d] =
                            positive_mod(target_meta.offset[d] + region_start[d], tile);
                    }
                    int64_t weight = 1;
                    for (int d = 0; d < nd; ++d)
                        weight *= static_cast<int64_t>(src_it->second.global_shape[d]);
                    all_desired[src_name].push_back({desired_phase, desired_anchor, weight});
                    return;
                }
                for (ASTNode* child : node->operands)
                    self(self, child, target_name, nd, target_meta, region_start);
            };

            auto is_cyclic_array = [&](int name) {
                int rep = uf_find(name);
                auto cit = comp_id.find(rep);
                return cit != comp_id.end() && cyclic_components.count(cit->second);
            };

            detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
                for (ASTNode* ast_node : dag_node->ast->roots) {
                    Opcode op = static_cast<Opcode>(ast_node->opcode);

                    if (op == Opcode::SET_REGION) {
                        if (ast_node->operands.size() < 2 || !ast_node->region)
                            continue;
                        int target_name = ast_node->operands[0]->result_name;
                        auto target_it = array_meta.find(target_name);
                        if (target_it == array_meta.end())
                            continue;
                        int nd = target_it->second.ndims;
                        auto region_start = extract_region_start(ast_node->region, nd);
                        for (int i = 1; i < (int)ast_node->operands.size(); ++i)
                            collect_fallback_consumer(collect_fallback_consumer,
                                                      ast_node->operands[i], target_name, nd,
                                                      target_it->second, region_start);
                        continue;
                    }

                    if (is_elementwise(op) || op == Opcode::COPY) {
                        int out_name = ast_node->result_name;
                        if (!is_cyclic_array(out_name))
                            continue;
                        auto out_it = array_meta.find(out_name);
                        if (out_it == array_meta.end())
                            continue;
                        int nd = out_it->second.ndims;
                        int out_tile = out_it->second.tile;
                        for (int i = 0; i < (int)ast_node->operands.size(); ++i) {
                            ASTNode* operand = ast_node->operands[i];
                            if (!operand || operand->is_scalar || operand->is_broadcast)
                                continue;
                            collect_fallback_producer(collect_fallback_producer, operand, out_name,
                                                      nd, out_tile);
                        }
                    }
                }
            });

            for (auto& [name, constraints] : all_desired) {
                if (constraints.empty())
                    continue;
                if (!is_cyclic_array(name))
                    continue;
                auto meta_it = array_meta.find(name);
                if (meta_it == array_meta.end() || meta_it->second.decomp_final)
                    continue;
                if (matmul_arrays.count(name))
                    continue;

                int nd = meta_it->second.ndims;
                int tile = meta_it->second.tile;
                if (tile <= 0)
                    continue;

                std::array<int, 3> best_phase = {0, 0, 0};
                for (int d = 0; d < nd; ++d) {
                    std::vector<int> fb_cands;
                    fb_cands.push_back(0);
                    for (auto& c : constraints) {
                        int o = positive_mod(c.desired_phase[d], tile);
                        auto it2 = std::lower_bound(fb_cands.begin(), fb_cands.end(), o);
                        if (it2 == fb_cands.end() || *it2 != o)
                            fb_cands.insert(it2, o);
                    }

                    int64_t best_cost = std::numeric_limits<int64_t>::max();
                    int best_o = 0;
                    for (int o : fb_cands) {
                        int64_t cost = 0;
                        for (auto& c : constraints)
                            cost += c.weight * circ_dist(o, c.desired_phase[d], tile);
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_o = o;
                        }
                    }
                    best_phase[d] = best_o;
                }

                std::array<int, 3> best_offset = {0, 0, 0};
                for (int d = 0; d < nd; ++d) {
                    std::vector<AbsoluteProposal> proposals;
                    for (auto& c : constraints) {
                        proposals.push_back(
                            {nearest_congruent(c.desired_anchor[d], best_phase[d], tile),
                             c.weight});
                    }
                    best_offset[d] = choose_absolute(proposals, best_phase[d], tile);
                }
                meta_it->second.offset = best_offset;
            }
        }

        // sources' offsets may have moved since
        align_transpose_results();

        for (int name : matmul_arrays) {
            auto meta_it = array_meta.find(name);
            if (meta_it != array_meta.end() && !meta_it->second.decomp_final)
                meta_it->second.offset = zero_offset;
        }

        // The executor allocates a fused kernel's other kept results at the
        // shifted set_region shape, so place them to match.
        detail::for_each_node_topo(dag, [&](DAGNode* dag_node) {
            for (ASTNode* root : dag_node->ast->roots) {
                if (static_cast<Opcode>(root->opcode) != Opcode::SET_REGION ||
                    root->operands.empty() || root->region == nullptr || root->region->is_global)
                    continue;
                auto target_it = array_meta.find(root->operands[0]->result_name);
                if (target_it == array_meta.end())
                    continue;
                const auto& target = target_it->second;
                int nd = target.ndims;
                auto start = extract_region_start(root->region, nd);
                auto step = extract_region_step(root->region, nd);
                auto stop = extract_region_stop(root->region, nd, target.global_shape);
                bool shifted = false, unit = true;
                for (int d = 0; d < nd; ++d) {
                    shifted |= start[d] != 0;
                    unit &= step[d] == 1;
                }
                if (!shifted || !unit)
                    continue;
                for (ASTNode* other : dag_node->ast->roots) {
                    if (other == root || other->is_temp ||
                        static_cast<Opcode>(other->opcode) == Opcode::SET_REGION)
                        continue;
                    auto it = array_meta.find(other->result_name);
                    if (it == array_meta.end() || it->second.decomp_final ||
                        it->second.ndims != nd)
                        continue;
                    bool same_shape = true;
                    for (int d = 0; d < nd; ++d)
                        same_shape &= it->second.global_shape[d] == stop[d] - start[d];
                    if (!same_shape)
                        continue;
                    for (int d = 0; d < nd; ++d)
                        it->second.offset[d] = target.offset[d] + start[d];
                    it->second.tile = target.tile;
                }
            }
        });
    }

    // Chare indices are global / tile, so a negative offset (from a view that
    // starts left of its anchor) moves to the same tile phase in [0, tile).
    for (auto& [name, meta] : array_meta) {
        if (meta.decomp_final || meta.tile <= 0)
            continue;
        for (int d = 0; d < meta.ndims; ++d)
            if (meta.offset[d] < 0)
                meta.offset[d] = ((meta.offset[d] % meta.tile) + meta.tile) % meta.tile;
    }

    for (auto& [name, meta] : array_meta)
        meta.decomp_final = true;
}

} // namespace decomposition_solver
