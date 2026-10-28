#pragma once

// Narrow MLIR JIT interface for TUs nvcc compiles (it cannot parse MLIR headers).

/// Returns the "fused_kernel" pointer (nullptr on failure); the module handle goes to moduleHandle.
void* jit_compile_ast(void* ast, void** moduleHandle);

/// Identical kernels share one module: free each once. No-op on GPU builds.
void jit_free_module(void* moduleHandle);

/// Loads the most recently used kernels of the disk cache until budget_ms has passed;
/// returns how many this process holds. Later compiles of them skip the disk.
int jit_prewarm(double budget_ms);

/// GPU builds: the stream the next JIT kernel calls on this thread enqueue on.
void jit_set_stream(void* stream);
