#pragma once

// Narrow MLIR JIT interface for TUs nvcc compiles (it cannot parse MLIR headers).

#include <string>

/// Returns the "fused_kernel" pointer (nullptr on failure); the module handle goes to moduleHandle.
/// If an op has no kernel body for its dtype, *error says so and the kernel writes zeros there.
void* jit_compile_ast(void* ast, void** moduleHandle, std::string* error = nullptr);

/// Identical kernels share one module: free each once. No-op on GPU builds.
void jit_free_module(void* moduleHandle);

/// GPU builds: the stream the next JIT kernel calls on this thread enqueue on.
void jit_set_stream(void* stream);
