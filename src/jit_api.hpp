#pragma once

// Narrow MLIR JIT interface for TUs nvcc compiles (it cannot parse MLIR headers).

#include <string>

/// Returns the "fused_kernel" pointer (nullptr on failure); the module handle goes to moduleHandle.
/// If an op has no kernel body for its dtype, *error says so and the kernel writes zeros there.
void* jit_compile_ast(void* ast, void** moduleHandle, std::string* error = nullptr);

/// jit_compile_ast for n kernels at once: misses compile in parallel (CNUM_JIT_THREADS), and
/// servers sharing the disk cache compile each kernel once (the others load it). errors, if
/// given, holds n strings: each kernel's error as jit_compile_ast reports it.
void jit_compile_batch(void* const* asts, int n, void** fns, void** moduleHandles,
                       std::string* errors = nullptr);

/// Identical kernels share one module: free each once. No-op on GPU builds.
void jit_free_module(void* moduleHandle);

/// Loads the most recently used kernels of this target's disk cache until budget_ms has
/// passed, listing included; returns how many this process holds. Only the first 1024
/// entries in directory order are considered, so with more entries the choice is not
/// strictly most recent first. Later compiles of them skip the disk. Load failures are not
/// fatal; only invalid entries are removed.
int jit_prewarm(double budget_ms);

/// Bounds the disk cache in a background thread, one pass an hour per cache: removes
/// entries of build/target directories unused for CNUM_JIT_CACHE_DAYS (7), stale temporaries
/// and pre-target-directory entries, then the oldest entries over CNUM_JIT_CACHE_MAX_FILES
/// (5000). Only roots marked as dedicated caches are touched; CNUM_JIT_CACHE_EVICT=0 skips it.
void jit_cache_maintain();

/// GPU builds: the stream the next JIT kernel calls on this thread enqueue on.
void jit_set_stream(void* stream);
