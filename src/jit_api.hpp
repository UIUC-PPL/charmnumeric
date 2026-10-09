#pragma once

// Narrow interface to the MLIR JIT. Translation units that include Kokkos are
// compiled by nvcc on NVIDIA builds, and nvcc cannot parse the MLIR headers,
// so those TUs use these entry points instead of including jit.hpp.

class MLIRJitCompiler;

MLIRJitCompiler* jit_create();
void jit_destroy(MLIRJitCompiler* jit);

/// JIT-compile a fusible node's AST into a "fused_kernel" function.
/// Returns the function pointer (nullptr on failure) and the owning module
/// handle through moduleHandle.
void* jit_compile_ast(void* ast, void** moduleHandle);
