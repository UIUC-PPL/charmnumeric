#pragma once

// Narrow MLIR JIT interface for TUs nvcc compiles (it cannot parse MLIR headers).

/// Returns the "fused_kernel" pointer (nullptr on failure); the module handle goes to moduleHandle.
void* jit_compile_ast(void* ast, void** moduleHandle);

/// Identical kernels share one module: free each once. No-op on GPU builds.
void jit_free_module(void* moduleHandle);

/// Loads the most recently used kernels of this target's disk cache (of the first 1024
/// entries listed) until budget_ms has passed, listing included; returns how many this
/// process holds. Later compiles of them skip the disk. Load failures are not fatal.
int jit_prewarm(double budget_ms);

/// Bounds the disk cache in a background thread, one pass an hour per cache: removes
/// build/target directories unused for CNUM_JIT_CACHE_DAYS (7), stale temporaries and
/// pre-target-directory entries, then the oldest entries over CNUM_JIT_CACHE_MAX_FILES (5000).
void jit_cache_maintain();

/// GPU builds: the stream the next JIT kernel calls on this thread enqueue on.
void jit_set_stream(void* stream);
