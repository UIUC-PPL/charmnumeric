#include "backend_internal.hpp"

template <int N>
bool extract_regions_nd(DAGNode* node, std::unordered_map<int, CTArrayBase<N>*>& arrays,
                        ArrayRegion<N>& r_out, std::vector<ArrayRegion<N>>& input_regions,
                        std::vector<int>& input_source_names,
                        std::array<int, N>& global_shape_out,
                        const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>* meta) {
    bool has_regions = false;
    bool has_output = false;
    std::array<int, N> global_shape = {};

    // Resolve global_shape from the node's operands
    for (ASTNode* root : node->ast->roots) {
        auto opc = static_cast<Opcode>(root->opcode);
        if (opc == Opcode::SET_REGION && root->operands.size() > 0) {
            int ref_name = root->operands[0]->result_name;
            auto ait = arrays.find(ref_name);
            if (ait != arrays.end() && ait->second->global_size > 0) {
                global_shape = ait->second->global_shape;
                break;
            }
        }
    }
    if (global_shape[0] == 0) {
        for (ASTNode* root : node->ast->roots) {
            for (ASTNode* operand : root->operands) {
                if (operand->is_scalar || operand->is_broadcast)
                    continue;
                auto ait = arrays.find(operand->result_name);
                if (ait != arrays.end() && ait->second->global_size > 0) {
                    global_shape = ait->second->global_shape;
                    break;
                }
            }
            if (global_shape[0] != 0)
                break;
        }
    }
    if (global_shape[0] == 0 && meta != nullptr) {
        // a chare that only owns part of the output (e.g. past an offset tile edge)
        auto it = meta->find(fused_output_decomp_name(node));
        if (it != meta->end() && it->second.ndims == N)
            for (int d = 0; d < N; ++d)
                global_shape[d] = it->second.global_shape[d];
    }
    if (global_shape[0] == 0) {
        int best_gs = 0;
        for (auto& [name, arr] : arrays) {
            if (arr->global_size > best_gs) {
                best_gs = arr->global_size;
                global_shape = arr->global_shape;
            }
        }
    }
    global_shape_out = global_shape;
    if (global_shape[0] == 0)
        return false;

    std::unordered_set<int> internal_names;
    for (ASTNode* root : node->ast->roots)
        internal_names.insert(root->result_name);

    for (ASTNode* root : node->ast->roots) {
        auto opc = static_cast<Opcode>(root->opcode);

        if (opc == Opcode::SET_REGION) {
            has_regions = true;
            has_output = true;
            auto* region = static_cast<ArrayRegion<N>*>(root->region);
            r_out = ArrayRegion<N>(region->start, region->stop, region->step);

            ASTNode* rhs = root->operands[1];
            if (!rhs->is_scalar && !rhs->is_broadcast &&
                !internal_names.count(rhs->result_name)) {
                // Use inline region from the RHS operand if available
                Region* rhs_r = root->get_operand_region(1);
                if (rhs_r && !rhs_r->is_global) {
                    auto* inp_r = static_cast<ArrayRegion<N>*>(rhs_r);
                    input_regions.emplace_back(inp_r->start, inp_r->stop, inp_r->step);
                } else {
                    std::array<int, N> inp_start = {};
                    std::array<int, N> inp_stop;
                    std::array<int, N> inp_step;
                    for (int d = 0; d < N; ++d) {
                        inp_stop[d] = r_out.size(d);
                        inp_step[d] = 1;
                    }
                    input_regions.emplace_back(inp_start, inp_stop, inp_step);
                }
                input_source_names.push_back(rhs->result_name);
            }
        } else if (is_elementwise(opc)) {
            for (int op_idx = 0; op_idx < (int)root->operands.size(); ++op_idx) {
                ASTNode* operand = root->operands[op_idx];
                if (operand->is_scalar || operand->is_broadcast)
                    continue;
                if (internal_names.count(operand->result_name))
                    continue;

                // Use inline region from operand_regions if available
                Region* op_r = root->get_operand_region(op_idx);
                if (op_r && !op_r->is_global) {
                    has_regions = true;
                    auto* inp_r = static_cast<ArrayRegion<N>*>(op_r);
                    input_regions.emplace_back(inp_r->start, inp_r->stop, inp_r->step);
                } else {
                    std::array<int, N> inp_start = {};
                    std::array<int, N> inp_step;
                    for (int d = 0; d < N; ++d)
                        inp_step[d] = 1;
                    input_regions.emplace_back(inp_start, global_shape, inp_step);
                }
                input_source_names.push_back(operand->result_name);
            }

            if (!has_output) {
                if (has_regions && !input_regions.empty()) {
                    std::array<int, N> out_start = {};
                    std::array<int, N> out_stop;
                    std::array<int, N> out_step;
                    for (int d = 0; d < N; ++d) {
                        out_stop[d] = input_regions.front().size(d);
                        out_step[d] = 1;
                    }
                    r_out = ArrayRegion<N>(out_start, out_stop, out_step);
                } else {
                    std::array<int, N> out_start = {};
                    std::array<int, N> out_step;
                    for (int d = 0; d < N; ++d)
                        out_step[d] = 1;
                    r_out = ArrayRegion<N>(out_start, global_shape, out_step);
                }
                has_output = true;
            }
        }
    }

    return has_regions;
}

template bool extract_regions_nd<1>(DAGNode*, std::unordered_map<int, CTArrayBase<1>*>&,
                                    ArrayRegion<1>&, std::vector<ArrayRegion<1>>&,
                                    std::vector<int>&, std::array<int, 1>&,
                                    const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>*);
template bool extract_regions_nd<2>(DAGNode*, std::unordered_map<int, CTArrayBase<2>*>&,
                                    ArrayRegion<2>&, std::vector<ArrayRegion<2>>&,
                                    std::vector<int>&, std::array<int, 2>&,
                                    const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>*);
template bool extract_regions_nd<3>(DAGNode*, std::unordered_map<int, CTArrayBase<3>*>&,
                                    ArrayRegion<3>&, std::vector<ArrayRegion<3>>&,
                                    std::vector<int>&, std::array<int, 3>&,
                                    const std::unordered_map<int, ArrayDAGGroup::ArrayMetadata>*);
