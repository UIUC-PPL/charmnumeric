#include "backend.hpp"
#include "jit_api.hpp"

void* ArrayDAGGroup::compile_node(DAGNode* node, std::string& error) {
    void* moduleHandle = nullptr;
    void* funcPtr = jit_compile_ast(node->ast, &moduleHandle, &error);
    if (!funcPtr)
        return nullptr;
    module_cache[node->identifier] = moduleHandle;
    return funcPtr;
}

/// Cross-ndims SET_REGION runs as a copy, not a JIT kernel; compiling it fails with MLIR rank errors.
static bool is_cross_ndims_set_region(DAGNode* node) {
    for (ASTNode* root : node->ast->roots)
        if (static_cast<Opcode>(root->opcode) == Opcode::SET_REGION && root->operands.size() >= 2 &&
            !root->operands[1]->is_scalar && !root->operands[1]->is_broadcast &&
            root->operands[1]->ndims != root->ndims)
            return true;
    return false;
}

void ArrayDAGGroup::compile(DAG* dag) {
    std::vector<DAGNode*> todo;
    std::vector<void*> asts;
    for (auto& [id, node] : dag->nodes) {
        auto it = compile_cache.find(node->identifier);
        if (it == compile_cache.end() && node->fusible) {
            if (is_cross_ndims_set_region(node)) {
                compile_cache[node->identifier] = nullptr;
                continue;
            }
            compile_cache[node->identifier] = nullptr; // compiled below, once per identifier
            todo.push_back(node);
            asts.push_back(node->ast);
        }
    }
    std::vector<void*> fns(todo.size()), handles(todo.size());
    std::vector<std::string> errors(todo.size());
    if (!todo.empty())
        jit_compile_batch(asts.data(), (int)todo.size(), fns.data(), handles.data(),
                          errors.data());
    for (size_t i = 0; i < todo.size(); ++i) {
        compile_cache[todo[i]->identifier] = fns[i];
        if (fns[i])
            module_cache[todo[i]->identifier] = handles[i];
        if (!errors[i].empty())
            compile_errors[todo[i]->identifier] = errors[i];
    }
    if (CkMyPe() == 0 && client_error.empty() && !compile_errors.empty())
        for (auto& [id, node] : dag->nodes) {
            auto err = compile_errors.find(node->identifier);
            if (err != compile_errors.end()) {
                client_error = err->second;
                break;
            }
        }
    DBG_PRINT("[PE %d] compile: %d new kernels, %d cached kernels total\n",
              CkMyPe(), (int)todo.size(), (int)compile_cache.size());
}
