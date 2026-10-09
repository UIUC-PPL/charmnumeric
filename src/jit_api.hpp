#pragma once

// Narrow interface to the MLIR JIT. Translation units that include Kokkos are
// compiled by nvcc on NVIDIA builds, and nvcc cannot parse the MLIR headers,
// so those TUs use these entry points instead of including jit.hpp.

/// JIT-compile a fusible node's AST into a "fused_kernel" function.
/// Returns the function pointer (nullptr on failure) and the owning module
/// handle through moduleHandle.
void* jit_compile_ast(void* ast, void** moduleHandle);

/// Free a module returned by jit_compile_ast; identical kernels share one (free each once). No-op on GPU builds.
void jit_free_module(void* moduleHandle);
