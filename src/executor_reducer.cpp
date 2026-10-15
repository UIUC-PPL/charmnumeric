#include "backend_internal.hpp"

template <int N>
DType determine_dtype(DAGNode* node, std::unordered_map<int, CTArrayBase<N>*>& arrays) {
    // Check AST roots for dtype info
    for (ASTNode* root : node->ast->roots)
        if (root->dtype != DType::FLOAT32)
            return root->dtype;
    // Fall back to first referenced array's dtype
    for (ASTNode* root : node->ast->roots)
        for (ASTNode* operand : root->operands)
            if (!operand->is_scalar && !operand->is_broadcast) {
                auto it = arrays.find(operand->result_name);
                if (it != arrays.end())
                    return it->second->dtype;
            }
    return DType::FLOAT32;
}

template DType determine_dtype<1>(DAGNode*, std::unordered_map<int, CTArrayBase<1>*>&);
template DType determine_dtype<2>(DAGNode*, std::unordered_map<int, CTArrayBase<2>*>&);
template DType determine_dtype<3>(DAGNode*, std::unordered_map<int, CTArrayBase<3>*>&);
