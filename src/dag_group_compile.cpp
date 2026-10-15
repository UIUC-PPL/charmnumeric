#include "backend.hpp"
#include "jit_api.hpp"

#include <pthread.h>

#if !defined(USE_NVIDIA) && !defined(USE_AMD) && !defined(USE_INTEL)
namespace {
// LLVM codegen recursion grows with loop count and can overflow the 8 MB main
// stack. CPU builds compile on a large-stack helper thread. GPU builds stay on
// the calling thread, which owns the device context.
constexpr size_t JIT_COMPILE_STACK_BYTES = size_t(1) << 30;

struct JitCompileJob {
    void* ast;
    void* module_handle;
    void* func_ptr;
};

void* jit_compile_thread(void* arg) {
    auto* job = static_cast<JitCompileJob*>(arg);
    job->func_ptr = jit_compile_ast(job->ast, &job->module_handle);
    return nullptr;
}

void* jit_compile_ast_big_stack(void* ast, void** moduleHandle) {
    JitCompileJob job{ast, nullptr, nullptr};
    pthread_attr_t attr;
    pthread_t thread;
    bool ran = false;
    if (pthread_attr_init(&attr) == 0) {
        if (pthread_attr_setstacksize(&attr, JIT_COMPILE_STACK_BYTES) == 0 &&
            pthread_create(&thread, &attr, jit_compile_thread, &job) == 0) {
            pthread_join(thread, nullptr);
            ran = true;
        }
        pthread_attr_destroy(&attr);
    }
    if (!ran)
        jit_compile_thread(&job);
    *moduleHandle = job.module_handle;
    return job.func_ptr;
}
} // namespace
#endif

void* ArrayDAGGroup::compile_node(DAGNode* node) {
    void* moduleHandle = nullptr;
#if !defined(USE_NVIDIA) && !defined(USE_AMD) && !defined(USE_INTEL)
    void* funcPtr = jit_compile_ast_big_stack(node->ast, &moduleHandle);
#else
    void* funcPtr = jit_compile_ast(node->ast, &moduleHandle);
#endif
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
    int newly_compiled = 0;
    for (auto& [id, node] : dag->nodes) {
        auto it = compile_cache.find(node->identifier);
        if (it == compile_cache.end() && node->fusible) {
            if (is_cross_ndims_set_region(node)) {
                compile_cache[node->identifier] = nullptr;
                continue;
            }
            newly_compiled++;
            void* compiled_fn = compile_node(node);
            compile_cache[node->identifier] = compiled_fn;
        }
    }
    DBG_PRINT("[PE %d] compile: %d new kernels, %d cached kernels total\n",
              CkMyPe(), newly_compiled, (int)compile_cache.size());
}
