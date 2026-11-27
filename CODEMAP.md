# Codemap: Charmnumeric (repository root)

Distributed N-dimensional array DSL built on charmtyles. The backend stores 1-3D tile-decomposed arrays; arrays with more than 3 dims are folded into 3-D storage by the frontend (`folded.py`). Python `Array`/`ArrayView` classes extend `FrontendObject`; C++ backend uses Charm++ chare arrays (`Partition<N>`) with optional MLIR JIT and GPU (Kokkos) support.

## Python (`charmnumeric/`)

### `charmnumeric/charmnumeric.py` — Core array types & operations (1063 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `DType` | 39 | Wire dtype codes — FLOAT32=0, FLOAT64=1, INT32=2, INT64=3, BOOL=4, INT8=5, INT16=6, UINT8..UINT64=7-10, FLOAT16=11, COMPLEX64=12, COMPLEX128=13; `from_numpy()`, `to_numpy()` |
| `ArrayOperation` | 139 | Opcodes: add=0, sub=1, mul=2, div=3, matmul=4, tanh=5, exp=6, tile=7, reduce=8, matmatmul=9, diag=10; elementwise math/comparisons/where/clip 20-49; creation upload=50, random=51, iota=52; reduction=60, bincount=61; transpose=70; scan=81, sort=82, argsort=83; select=190; linalg=100; gather=120, scatter=121, expand=122, compress=123, mask_scatter=124; reshape=140, broadcast=141; convolve=150, flip=151; convert (fused astype)=160, real=161, imag=162, conj=163, bitwise_and/or/xor=164-166, invert=167, left_shift=168, right_shift=169; batched_matmul=170 |
| `ArrayRegion` | 153 | N-D region with start/stop/step per dim — `serialize()`:194, `shape()`:209, `compose()`:212, `overlaps()`:232, `covers()`:239, `intersect()`:246 |
| `Array(FrontendObject)` | 618 | Main array class — `wire_dtype()`:639, `get()`:683, `get_region()`/`__getitem__()`:690/697, `__setitem__()`:708, `dot()`:724, `matvec()`:727, `copy()`:730, `fill()`:735, `astype()`:738; operators: +,-,*,/,@ at 713-722 |
| `ArrayView(FrontendObjectView)` | 747 | View with same ops as Array — `get()`:796, `__getitem__()`:801, `__setitem__()`:812, `dot()`:828 |
| `_full_region()` | 258 | Get/cache full region for shape |
| `_expand_key()` | 272 | Normalize indexing keys (handles newaxis, ellipsis) |
| `_parse_key()` | 308 | Convert key → ArrayRegion |
| `_array_add/sub/mul/truediv()` | 394-489 | Binary elementwise operator implementations |
| `_ew()` / `_ufunc_dtypes()` / `_scalar_wire()` | — | Fused elementwise node; result dtype from NumPy's `ufunc.resolve_dtypes` (weak Python scalars); complex scalars and ints beyond 2**53 become size-1 arrays |
| `_astype()` / `_promote_operands()` / `_blas_dtype()` | — | Fused CONVERT; matmul/dot compute dtype (f16 -> f32, small/unsigned ints and bool -> int64) |
| `_compare()` / `_where()` / `_logical_*()` / `_real()` / `_imag()` | — | Comparisons return bool; where takes any condition dtype |
| `_compare_scalar_out_of_range()` | — | Exact int-array vs out-of-range / non-integral scalar comparison: constant masks |
| `_broadcast_operands()` / `_broadcast_to()` | — | NumPy broadcasting: non-size-1 operands of another shape become BROADCAST nodes |
| `_reshape()` / `_reshape3()` | — | C-order reshape (RESHAPE node; size-1-dim-only changes use a cross-ndim set_region); `reshape/ravel/squeeze/expand_dims/atleast_*d/broadcast_to/broadcast_arrays`, `Array.reshape/ravel/flatten/squeeze` |
| `_fit_assignment()` / `_inplace()` | — | Assignment broadcasting; in-place operators (`+=`, `/=`, ...) write into self |
| `_NDBase` / `_nd_aware()` | — | Marker of FoldedArray; wrapper that lets elementwise functions take >3-D operands |
| `_ElementwiseMethods` | 863 | Mixin for Array/ArrayView: round/clip and old-backend methods (sqrt, log(base), where, max(other), ...) |
| `_array_matmul()` | 492 | Matrix multiply dispatch (dot for 1D×1D, matvec for 2D×1D, matmatmul for 2D×2D) |
| `_matvec()` | 536 | Matrix-vector multiply |
| `_matmatmul()` | 595 | Matrix-matrix multiply (SUMMA) |
| `_unstrided()` | — | Copies a stepped view before matvec/SUMMA (the executors ignore view steps) |
| `_batched_matmul()` | — | N-D / FoldedArray / non-squeezable 3-D matmul via `einsum("...ij,...jk->...ik")` (one BATCHED_MATMUL node) |
| `create_array()` | 851 | Array creation factory |
| `empty/zeros/ones/full()` | 931-971 | Array constructors |
| `empty_like/zeros_like/ones_like/full_like()` | 975-994 | Like-constructors |
| `asarray/array()` | 997-1028 | Conversion functions |
| `arange()` | 1042 | Range array |
| `eye()/identity()` | 1048/1061 | Identity matrix |

### `charmnumeric/operations.py` — Standalone operation functions (180 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `tanh()` | 38 | Activation function |
| `exp()` | 45 | Exponential |
| `tile()` | 52 | Tile/repeat |
| `add/subtract/multiply/divide()` | 75-96 | Binary elementwise |
| `matmul()` | 99 | Matrix multiply |
| `dot()` | 105 | Dot product |
| `norm2()` | 116 | Squared L2 norm (1D) |
| `diag()` | 125 | Diagonal extraction/construction |
| `sqrt/log*/abs/floor/ceil/rint/round/sign/...` | 199-270 | Unary elementwise math (opcodes 20-36) |
| `power/maximum/minimum/fmax/fmin/remainder/floor_divide` | 287-305 | Binary elementwise math (37-43) |
| `less/.../not_equal`, `logical_*`, `where`, `clip` | 325-375 | Comparisons (44-47, bool results), where (48), clip (49) |

### `charmnumeric/reductions.py` — Native reductions
| Symbol | Description |
|--------|-------------|
| `_reduce()` | Builds a REDUCTION (opcode 60) node: operands `x, kind, axis_mask, keepdims, ddof, in_dtype` |
| `sum/prod/mean/var/std/amax/amin/argmax/argmin/count_nonzero/all/any` | NumPy-style API (`axis`, `keepdims`, `dtype`, `out`); also installed as `Array`/`ArrayView` methods |
| `bincount()` | BINCOUNT (opcode 61) node; needs `minlength` or `interface=` |
| `quantile()` / `percentile()` | 1-D (or axis=None) with up to 256 needed ranks: one SELECT node (`_select`), no sort; otherwise distributed sort + take |
| `norm()` | `cnp.linalg.norm` subset (vector 2/1/inf/-inf/0, matrix fro/inf/-inf/1/-1, axis) |

### `charmnumeric/linalg.py` — Dense linear algebra (`cnp.linalg`)
| Symbol | Description |
|--------|-------------|
| `_la()` | Builds a LINALG (opcode 100) node: operands `arrays..., kind, out_nd, out0, out1, p0, p1, p2` |
| `cholesky/qr/solve/inv/det/slogdet/eigh/eigvalsh/svd/svdvals/lstsq/solve_triangular/tssvd` | NumPy (and cuPyNumeric) API; multi-factor results are one packed 2-D node, sliced natively (`_block`, `_vec`) |
| `norm` | Re-exported from `reductions.py` |
| `_reference()` | NumPy model of a LINALG node (used by the CNUM_SIM simulator) |

### `charmnumeric/signal.py` — Convolution and flips
| Symbol | Description |
|--------|-------------|
| `convolve()` | cuPyNumeric `convolve(a, v, mode, method)`: N-D full/same/valid, direct/fft/auto; CONVOLVE (opcode 150) node `a, v, method, shift..., shape...`; computes in float32/float64/int32/int64 and casts to `result_type(a, v)` |
| `flip/flipud/fliplr()` | FLIP (opcode 151) node `x, axis_mask`; returns a copy |
| `split_negative_steps()` | Negative-step key -> (positive key, flipped axes); used by `__getitem__`/`__setitem__` |
| `reference()` | NumPy model of both nodes (CNUM_SIM simulator) |

### `charmnumeric/indexing.py` — Advanced indexing (opcodes 120-124)
| Symbol | Description |
|--------|-------------|
| `getitem()` / `setitem()` | `Array.__getitem__/__setitem__` for keys with index arrays or masks (slices in them may have negative steps): the key is linearized into an array L of flat offsets into the root (`_items`, `_plan`), then one GATHER / SCATTER node |
| `_lead_mask()` / `_mask_getitem()` / `_mask_setitem()` | `x[m]`, `x[m] = v` for a backend or host bool mask over leading dims of x: one COMPRESS / MASK_SCATTER node (a mask over fewer dims is broadcast first); `folded_getitem/setitem` use them for full-shape masks of >3-D arrays |
| `nonzero/flatnonzero/argwhere` | EXPAND node; output size from a `count_nonzero` get on `default_interface()` |
| `take/put/take_along_axis/put_along_axis/compress/repeat/choose` | NumPy API (also `Array` methods); bool arrays index as masks; GATHER/SCATTER operands carry each slice's root dim and start (`_slice_operands`) |

### `charmnumeric/folded.py` — Arrays with more than 3 dims
| Symbol | Description |
|--------|-------------|
| `FoldedArray` | N-D array (N > 3) stored as the 3-D array (s0*...*s[N-3], s[N-2], s[N-1]); limits in the module docstring |
| `reshape_nd()` / `broadcast_nd()` / `to_shape()` | Reshape and broadcasting with folded operands (storage-level when the pattern allows, else via grouped reshapes) |
| `FoldedArray.transpose/swapaxes/T`, `@` | Through `einsum.permute` (grouped 3-D transposes) and `_batched_matmul` |
| `apply_elementwise()` | Runs an elementwise function on the 3-D storage of broadcast operands |
| `getitem()` / `setitem()` | Basic indexing: views for rectangular storage blocks, block loops otherwise |
| `reduce_nd()` / `along_last_dims()` | Reductions (adjacent-dim grouping), scans/sorts along the last two axes |

### `charmnumeric/einsum.py` — einsum, tensordot, inner, outer (frontend lowering)
| Symbol | Description |
|--------|-------------|
| `einsum()` / `einsum_path()` | `numpy.einsum` (string or sublist form, `out=`, `dtype=`, `optimize=`); path from `numpy.einsum_path` on shape-only dummies |
| `_parse()` / `_from_sublists()` | Ellipsis expansion and implicit output; sublist form to subscripts |
| `_drop_broadcast()` / `_diagonals()` / `_sum_out()` | Per operand: size-1 broadcast dims, repeated labels (DIAG for 2-D, else permute + RESHAPE + strided view), single-operand sums (REDUCTION) |
| `_fold()` / `_contract_pair()` | Pairwise contraction (smallest intermediate first within a path step): label-space or grouped broadcast multiply + sum, native dot/matvec, SUMMA, or with batch labels one BATCHED_MATMUL node |
| `_bmm()` / `bmm_reference()` | BATCHED_MATMUL (opcode 170) node: operands `a, b, dims of a holding (batch, m, k), dims of b holding (batch, n, k), result shape`; any layout with the batch dim not last needs no transpose; NumPy model for the simulator |
| `permute()` | Any-rank transpose: drops size-1 axes, coalesces adjacent axes, >3-D via grouped 3-D transposes; `cnp.transpose`/`FoldedArray.T` use it |
| `tensordot()` / `inner()` / `outer()` | NumPy API on top of `einsum` |

### `charmnumeric/interface.py` — Cluster management (130 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `CharmNumericInterface(CCSInterface)` | 12 | Subclass with `from_bytes()` deserializer |
| `LocalCluster` | 26 | Local server launcher — `_run_server()`:74, `_connect_with_retry()`:94, `close()`:111 |

### `charmnumeric/random.py` — Native RNG (numpy.random subset)
| Symbol | Description |
|--------|-------------|
| `Generator` / `default_rng()` | random, uniform, normal, standard_normal, integers; one RANDOM node per draw |
| `seed`, `rand`, `random`, `randn`, `uniform`, `normal`, `randint` | Legacy module-level API on a global Generator (fixed default seed 0) |
| `transposed()` | Regenerate a draw with permuted indices (== x.T, no transpose op needed) |

### `charmnumeric/_rng_reference.py` — Host model of the creation kernels
Bit-exact NumPy mirror of `src/creation_kernels.hpp` (Philox4x32-10, software log/cos, iota). Used by the simulator and tests.

### Array creation in `charmnumeric/charmnumeric.py`
`_creation_op()` builds UPLOAD (50) / RANDOM (51) / IOTA (52) nodes: `[region, (hi, lo) scalar word pairs..., InlineData]`. `_upload()` (bulk host data), `arange`, `linspace`, `indices`, `meshgrid`, `eye` (all native).

## C++ Backend (`src/`)

### `backend.hpp` — Core backend types (673 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `CT_MIN_TILE_1D/2D/3D` | 11-17 | Minimum tile sizes |
| `PartitionTraits<N>` | 22 | Maps N → Charm++ proxy/index types (specializations for 1,2,3) |
| `proxy_at<N>()` | 44 | Access chare array element by index |
| `CTArrayBase<N>` | 93 | Type-erased N-D array base — name, region, decomp, global_shape, dtype |
| `Array<N,T>` | 113 | Typed array — Kokkos views (d_view, h_view), `data_ptr()`, `copyToHost/Device()` |
| `RemoteBuffer<N>` | 206 | Remote data buffer with region |
| `PendingComm<N>` | 213 | Pending communication state |
| `ArrayDAGGroup` | 240 | Charm++ group extending `DAGGroup` |
| — `PartitionGrid` | 245 | Grid dimensions + epoch |
| — `ArrayMetadata` | 267 | Per-array metadata with decomp — `decomp<N>()`:276 |
| — Key methods | | `receive_dag()`:303, `execute_node_nd<N,T>()`:308, `compute_decompositions()`:314, `compile_node()`:317, `compile()`:318, `gather()`:305 |
| `ArrayDAGExecutorND<N>` | 323 | N-D DAG executor — `execute_dag_node()`, `execute_matmul_node()`, `execute_matmatmul_node()`, `execute_reduce_node()`, `execute_diag_node()`, `execute_tile_node()`, `ast_visitor()` |
| `PartitionImpl<N>` | 352 | Partition logic (Charm++-independent) |
| — `FreeKey`/`FreeKeyHash` | — | Array reuse key (dtype + local element count); a reused array takes the new region, shape and decomposition |
| device caches (GPU) | — | Retired arrays (`retire_array`, `CNUM_FREE_LIST_MB`, default 1/2 of the device per sharing process) and freed message/scratch blocks (`comm_free`; >64 MB blocks `CNUM_DEVICE_CACHE_MB`, default 1/2; smaller ones `CNUM_COMM_CACHE_MB`, 1/16). Up to 1/8 of the device is always kept; beyond that a block is kept only while `CNUM_DEVICE_RESERVE_MB` (default 1/4 of the device) stays free. While the caches hold more than 1/8, a fresh array or large block first evicts cached ones, least recently freed first, until it fits with the reserve still free (`device_make_room`); a failed allocation releases all caches (and trims the stream-ordered pool) and retries once |
| — Key methods | | `init()`:567, `create()`:571, `run()`:572, `retire_array()`:428, `try_reuse()`:456, `allocate_or_reuse()`:502, `ensure_array()`:534, `process_get()`:582 |
| `Partition1D/2D/3D` | — | Thin Charm++ chare wrappers around PartitionImpl |
| `PartitionMap` | — | `CkArrayMap` group placing the partitions' chares by `chare_home_pe()` |

### `backend_internal.hpp` — Compute kernels & helpers (491 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `eigen_gemv/gemv_sub/gemv_sub_3d()` | 15/28/48 | Eigen GEMV kernels |
| `eigen_dot()` | 88 | Dot product |
| `eigen_gemm/gemm_sub()` | 98/114 | Eigen GEMM kernels |
| `ct_min_tile()` | 151 | Min tile by dimension |
| `array_tile()` | 168 | Tile size from metadata |
| `extract_regions_nd<N>()` | 179 | Extract I/O regions from AST |
| `determine_dtype<N>()` | 185 | Get dtype from node/arrays |
| `cross_matmul_send_result_3d_to_1d<T>()` | 188 | Cross-dim matmul send |
| `cross_matmul_send_result_2d_to_1d<T>()` | 223 | Cross-dim matmul send |
| `cross_set_region_send<N_src,N_tgt,T>()` | 320 | Cross-dim SET_REGION |
| `ReduceContrib` | 264 | 32-byte dot (REDUCE) contribution, sent to chare 0 |

### `decomposition_solver.hpp` — Standalone decomposition solver (2094 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `decomposition_solver::compute_decompositions()` | 191 | Charm-free decomposition solver used by the runtime and standalone tests; results do not depend on map iteration order (sorted names, node ids, class reps; anchor ties go to the lower name) |
| `detail::balanced_tile()` | 91 | Near-equal tiles: ceil(n/k) >= `CT_MIN_TILE_*` along some dim rounded to 1024/32/16 (1-/2-/3-D, at most the min tile), at most `odf * PEs` tiles; the one minimizing the busiest PE's elements under `chare_home_pe` (`busiest_pe_elems()`) replaces the old smallest power-of-two multiple of `CT_MIN_TILE_*` only when it unloads the busiest PE by 20% (never on 1 PE); exact per-PE sums when at most ~1M chare visits, else tiles per PE x full tile, for every candidate alike |
| `assign_class_tiles` (in `compute_decompositions`) | 411 | Tile classes: arrays joined by unit-stride elementwise/COPY/SET_REGION accesses and REDUCE operands share one tile (members under half the largest one's size keep their own) — the heaviest array from an earlier DAG (or TRANSPOSE result) in the class, else the balanced tile of the class's largest extents capped by strided accesses; the offset DP then aligns them |
| `detail::for_each_node_topo()` | 151 | Topological DAG traversal helper for solver passes |
| `collect_unionable_nonfixed_leaves()` | 758 | Union-find guard: only unions zero-shift, same-tile, non-fixed temp leaves |

### `array_region.hpp` — N-D regions & decomposition (449+ lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `ArrayRegion<N>` | 22 | N-D region (start/stop/step per dim) — `size()`:110, `deserialize()`:121, `overlaps()`:136, `covers()`:158, `intersect()`:186 |
| `ArrayDecomp<N>` | 294 | Tile decomposition — `default_decomp()`:303, `offset_decomp()`:312, `to_global/local()`:322/339, `num_chares()`:350, `chare_region_global/local()`:360/371 |
| `MemRef<N,T>` | 407 | MLIR memref descriptor (allocated, aligned, offset, sizes, strides) |
| `ChareIndex<N>` | — | N-D chare array index |
| `chare_home_pe()` | 454 | Home PE of a chare (`PartitionMap`): for 2^m PEs a GF(2)-linear map of the index bits, so aligned boxes (2x2, 4x4, 4x1, 2x2x2, ...) spread evenly; other PE counts a linear map with strides coprime to the PE count; 1-D round robin |

### `opcodes.hpp` — Operation codes
| Symbol | Line | Description |
|--------|------|-------------|
| `Opcode` | 3 | COPY=-5..DIAG=10; elementwise SQRT=20..CLIP=49 |
| `is_binary_elementwise()` | 23 | Binary arithmetic, math, comparison and bitwise ops |
| `is_unary_elementwise()` | 36 | TANH/EXP and unary math/predicates (20-36) |
| `is_ternary_elementwise()` | 109 | WHERE/CLIP |
| `is_elementwise()` | 47 | Binary or unary elementwise |

### `jit.hpp` / `jit.cpp` — MLIR JIT compiler (150+ lines each)
| Symbol | Line | Description |
|--------|------|-------------|
| `BinaryOpEntry` | 80 | Registry: float_body, int_body, optional uint_body, complex_body, bool_body |
| `UnaryOpEntry` | 87 | Registry: same bodies as BinaryOpEntry |
| `TernaryOpEntry` | 94 | Registry for 3-input ops (where, clip); `emitGenericNaryOp` builds the N-input linalg.generic |
| `MLIRJitCompiler` | 101 | AST → MLIR → executable — `buildFromAST()`, `optimizeAndFuse()`, `generateCPU()`, `loadCPU()`, `generateNVIDIA/AMD/Intel()` |
| kernel caches (jit.cpp) | — | In-process map keyed by the kernel IR, then a persistent object-file cache: `$CNUM_JIT_CACHE_DIR` (default `${XDG_CACHE_HOME:-~/.cache}/charmnumeric/jit`)/`<server build id>`/`<GPU target>[-<device sm>]-d<driver>-<host CPU>`/`<sha256(IR, target)>.o`, one file per kernel, loaded with an LLJIT (`loadCachedKernel`). NVIDIA objects embed the cubin (PTX assembled by the driver's JIT at compile time, `generateNVIDIAHost`), so the key also has the device's sm and driver version; `cubin/` (PTX-keyed cubins) is only used for kernels whose cubin could not be embedded. `CNUM_JIT_CACHE=0` disables it, `CNUM_JIT_TIME=1` prints compile phases and one `jit batch:` line per batch with misses |
| `jit_compile_batch()` (jit.cpp), `ArrayDAGGroup::compile()` | — | All new fusible nodes of a DAG compile as one batch. Misses compile in parallel on `CompilePool` threads (`CNUM_JIT_THREADS`, default min(8, CPUs of the process at startup); the threads run on all of those CPUs even if Charm++ pins the PE thread; 1 GiB stacks; one MLIR context per thread). Servers sharing the cache compile each kernel once: a compiler holds `<entry>.o.lock` (O_EXCL; host and pid inside; mtime refreshed every 2 s) and claims at most a pool's worth at a time, the others load the published entry. Waiters poll (opening files, 10-200 ms backoff), break a claim whose holder died (or is exiting or a zombie) on the same host or whose mtime stood still for 30 s (6 s for a holder on the same host or an empty lock, 5 s if it is 2 min old), and compile without a claim after `CNUM_JIT_WAIT_S` (600) s or when a published entry does not load here (loads never abort; invalid entries are removed and recompiled) |
| `jit_prewarm()` (jit.cpp), `ArrayDAGGroup::prewarm()` | — | Broadcast after "CCS server ready" (the line still marks readiness): every PE creates its cuBLAS/cuSOLVER handles and runs a first GEMM (GPU; skipped below 1 GiB free or when the thread's device is not Kokkos's), then loads the most recently used kernels of its target's cache. `CNUM_PREWARM_MS` (default 250) bounds the whole pre-warm: the BLAS part (~75 ms on an A100, not interruptible) counts against it and kernels load while budget remains (at least a fifth of it); both are skipped without the device or 1 GiB free; 0 skips both, as does `CNUM_PREWARM=0`. Failures are skipped. CCS requests that arrive meanwhile, including the first one, queue behind it on PE 0 |
| `jit_cache_maintain()` (jit.cpp) | — | PE 0 at startup, in a background thread, one completed pass an hour per cache (`.last-evict`; `.evict-lock` while a pass runs, taken over after 10 min if its server exited first): removes the entries of build/target directories unused for `CNUM_JIT_CACHE_DAYS` (7), `*.tmp` older than an hour and pre-target-directory `*.o`, then the oldest entries (`.o`, `.cubin`) over `CNUM_JIT_CACHE_MAX_FILES` (5000). Runs only on a root marked `.cnum-jit-cache` (written when the root is new or empty, has exactly the cache layout, or with `CNUM_JIT_CACHE_ADOPT=1`; never `/` or `$HOME`; an unmarked root is reported once), enters only build-id/target-tag directories and `cubin/` (removed once unused for the age limit), deletes only cache files (`<sha256>.o`/`.cubin`, their `cacheTmpName` temporaries and `.o.lock` claims, the last two once an hour old) and removes directories only once empty; `CNUM_JIT_CACHE_EVICT=0` skips it. The root is normalized (absolute, no trailing separator); the `/tmp/cnum-jit-<uid>` fallback is made 0700. Cache hits refresh an entry's mtime (`.o` and `.cubin`) at most hourly. Pre-warm load failures remove an entry only when it is invalid (bad object, `INVALID_IMAGE`/`NO_BINARY_FOR_GPU`/`INVALID_PTX`/`INVALID_SOURCE`/`UNSUPPORTED_PTX_VERSION`), not on e.g. out of memory |
| `cn_mgpu*` (jit.cpp, NVIDIA) | — | Overrides of the MLIR CUDA runtime wrappers: stream/alloc calls use the chare's stream, module load and launch use the current context |

### `dtypes.hpp` / `dtype_info.hpp` — Element types and dtype dispatch
| Symbol | Description |
|--------|-------------|
| `cn_half`, `cn_c64`, `cn_c128` | Element types (`_Float16`/`std::complex` on CPU; Kokkos `half_t`/`Kokkos::complex` with Kokkos) |
| `dtype_of<T>()`, `CN_FOR_EACH_TYPE` | Type -> DType; X-macro for explicit instantiations |
| `cn_dispatch(dt, f)`, `cn_dispatch_blas(dt, f)` | Call `f(T{})` for a dtype (all 14 / the BLAS types) |
| `cn_cast<To>()`, `cn_isnan()`, `cn_nonzero()`, `cn_sort_lt()` | Host/device helpers: NumPy astype, NaN test, truth, NumPy sort order (NaNs last, complex lexicographic) |
| `dtype_promote()` (dtype_info.hpp) | `numpy.promote_types` table (also used by the JIT, which cannot include Kokkos) |

### `dispatch.hpp` — Kernel dispatch & GPU callbacks (200+ lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `CT_COMPUTE_POLICY/CT_COMM_POLICY` | 11-19 | Kokkos execution policies |
| `dispatch_kernel<N,T>()` | 138 | Dispatch JIT kernel with memref descriptors |
| `compute_done_cb<N>()` | 42 | (GPU) HAPI callback for compute completion |
| `deferred_send_cb<N_sender,N_target>()` | 92 | (GPU) HAPI callback for deferred send: `receive_ipc` (same host), `receive_staged` (other host) or `receive_data` with a `CkDeviceBuffer` (same process, or fallback) |
| `device_pack_send<N_sender,N_target>()` | 129 | (GPU) Picks the transport by the target's process and host, queues the IPC / pinned copy on the comm stream |

### `gpu_comm.hpp` — GPU messages between processes (NVIDIA)
| Symbol | Description |
|--------|-------------|
| `gpu_ipc_peer()` | IPC only to processes that passed the startup probe (`peer_ok`); everything else is staged |
| `gpu_ipc_alloc()` / `gpu_ipc_free()` | Send blocks in lazily created 256 MB `cudaMalloc` slabs with CUDA IPC handles (`CNUM_GPU_IPC=0` disables, `CNUM_GPU_IPC_MB` bounds them, default 1024): free blocks of larger classes are split and freed blocks merge with their buddies, alignment gaps go to smaller classes, a failed slab allocation is not retried, one warning on the first fallback; when the slabs are full, or the PE already has sends waiting, a send queues (FIFO) for a block its PE frees (`gpu_ipc_wait`/`gpu_ipc_retry`, from `send_complete` and `gpu_ipc_forget`), is staged only once that PE has nothing in flight, and is dropped with its partition (`gpu_ipc_cancel`); `comm_free()` returns arena blocks here |
| `gpu_ipc_peer_slab()` | Receiver: maps a peer slab once (`cudaIpcOpenMemHandle`) and caches it |
| `pinned_alloc()` / `pinned_free()` | Pinned host pool by size class for cross-host messages (`CNUM_GPU_STAGE=0` disables, `CNUM_GPU_STAGE_MB` caps the cached blocks, default 512); released from the stream callback after the copy |
| `gpu_comm_shutdown()` | Before `Kokkos::finalize`: no new IPC sends or mappings, late frees become no-ops; slabs, mappings and pinned blocks are left to process exit (a peer may still be copying) |

### `gpu_comm.cpp` — CUDA IPC probe (`ArrayDAGGroup::ipc_probe_*`)
Started from `set_proxies`, off the CCS-ready path: rank 0 of each process exports a 4 KB buffer; processes with the same hostname and `/proc/sys/kernel/random/boot_id` map and read each peer's buffer; the working (sender, receiver) pairs are gathered into `peer_ok`. Never aborts; PE 0 prints `gpu ipc: k of n same-host process pairs use CUDA IPC`. `CNUM_GPU_IPC_PROBE_FAIL="src-dst,..."` injects failures (testing).

### Key C++ source files
| File | Description |
|------|-------------|
| `server.cpp` | Main chare, creates ArrayDAGGroup |
| `dag_group_runtime.cpp` | ArrayDAGGroup constructor (partitions use `PartitionMap`), `compute_decompositions()` (`CNUM_DECOMP_LOG=1`: PE 0 prints each new array's tile and offset), set_proxies, receive_get_request, gather |
| `dag_group_compile.cpp` | JIT compilation orchestration (one `jit_compile_batch` per DAG) |
| `dag_group_receive.cpp` | DAG reception and dispatch |
| `execute_node.cpp` | `execute_node_nd<N,T>()` — dispatches JIT kernels, handles MATMUL/REDUCE/TILE/DIAG |
| `execute_node_regions.cpp` | Region extraction from AST |
| `executor_core.cpp` | `execute_dag_node()`, remote input sends. Debug builds abort when an input/output buffer's decomposition (tile, shape, offset) differs from array_meta (`check_buffer_decomp`: receivers without the buffer plan from array_meta) or a planned send falls outside the sender's buffer (it would be dropped and the receiver would wait) |
| `executor_matmul.cpp` | Matrix-vector multiply execution |
| `executor_matmatmul.cpp` | Matrix-matrix multiply (SUMMA) execution |
| `executor_reduce.cpp` | REDUCE (1-D dot): local dot per chare, partials to chare 0. Operands with different layouts (tile, offset or shape): rhs first moves to lhs's tiles through the elementwise planner (`start_realigned_dot` in executor_core.cpp, `finish_realigned_dot`). Operands must have the node's dtype (the frontend promotes them); otherwise it aborts |
| `executor_reduction.cpp` | REDUCTION/BINCOUNT: per-chare partials (Kokkos on GPU) sent point-to-point to the output chares (`receive_reduce_partial`) |
| `executor_reducer.cpp` | `determine_dtype<N>()` (dtype of a node from its AST or arrays) |
| `executor_transfer.cpp` | Cross-partition data transfer |
| `executor_transpose.cpp` | TRANSPOSE (2-D/3-D axis permutation of an array or view): point-to-point pack/send, `transpose_place` receive side |
| `executor_castscansort.cpp` | SCAN (cumsum/cumprod/nan*; opcode 81): each chare sends its line totals to the later chares along the axis, then scans its tile once with the folded carry (CUB single-pass scan per long line on NVIDIA); narrow int/bool inputs are cast in the kernel. SORT/ARGSORT (82-83): dispatch to `executor_sort.cpp`; only arrays of at most `CNUM_SORT_GATHER_MAX` (65536) elements split along the axis gather to one chare (`sort_root`) |
| `executor_sort.cpp` | 1-D SORT/ARGSORT (every dtype; complex keys are (real, imag) radix keys in NumPy's NaN-class order): sample sort over the chares (pooled samples -> splitters on (key, index), stable multi-way split, buckets exchanged by pointer on the same PE or as device messages, CUB radix sort per bucket, then the sorted runs moved to the result tiles). SELECT (190): order statistics by radix selection (histogram rounds of the top key bits, coordinated by the first chare, then a gather of the few candidates); `receive_sort_msg` carries the control messages. Radix keys order like NumPy (NaNs last); sorted values are decoded from the keys, so NaNs come back canonical and -0.0 sorts before +0.0 (documented in `_sort`). N-D (`ds_axis_local`, `execute_ax_node`): a tile holding whole lines sorts them (CUB segmented sort in batches on NVIDIA, host elsewhere); along a split axis the P tiles sharing a line set split those lines P ways, send each owner its part of them, the owners sort whole lines and send the results back (two all-to-alls, pointer handoff on the same PE) |
| `executor_incremental.cpp` | Incremental computation |
| `executor_ast_visitor.cpp` | AST visitor for elementwise node execution |
| `partition_lifecycle.cpp` | `init()`, `create()`, `run()`, destructor |
| `partition_comm.cpp` | `receive_data()` inter-chare communication; `receive_ipc()`/`receive_staged()` copy into a device buffer on the comm stream, then deliver as `receive_data()` |
| `big_msg.hpp` | Payloads over `ct_msg_max_bytes()` (1 GiB; `CNUM_MSG_MAX_BYTES` lowers it for tests) go as `receive_frag` fragments, reassembled by the target chare (`PartitionImpl::receive_frag`, partition_comm.cpp) and delivered to the entry of their `MsgKind`: Charm++ marshalled messages hold less than 2 GiB. Senders: `ct_send_bytes` (linalg, index, shape, reduction partials, upload tiles), `ct_send_data` (host `receive_data`, CPU builds), staged GPU sends |
| `partition_get.cpp` | `process_get()` — gather results to PE 0 (rows cut into `ct_msg_max_bytes()` pieces); gets over 512 MiB reach the client in chunks (`Server::send_fetch_reply`, `get_chunk`) |
| `executor_signal.cpp` | CONVOLVE/FLIP: each result chare receives the input window (halo of the filter size) and the filter, or its mirrored tile, point-to-point; direct sum or per-chare FFT (host mixed radix, cuFFT on NVIDIA). BATCHED_MATMUL: each result chare cuts its batch range and K at the operands' chare boundaries and the batch further into chunks, so a (chunk, k panel) block is at most 64 MB (`CNUM_BMM_MSG_KB`); blocks arrive as one message per source chare (its own parts are read in place) and each chunk's GEMMs run in panel order as blocks complete (`on_signal_receive`), accumulating into C (cuBLAS `gemmStridedBatched` for float/complex on NVIDIA, Eigen on CPU, a naive Kokkos kernel for ints on NVIDIA and for every dtype on other Kokkos backends). `copy_box` is one cudaMemcpy2D/3D for unit-stride boxes |
| `executor_shape.cpp` | RESHAPE/BROADCAST (opcodes 140-141): source chares send row runs (reshape) or the projected block (broadcast) to each result chare (`receive_shape`, host-staged); broadcast blocks are expanded on the device |
| `executor_linalg.cpp` | LINALG: operand blocks gathered to chare 0 of the result partition (`receive_linalg`), factored there (Eigen on CPU, cuSOLVER/cuBLAS on NVIDIA), result tiles scattered back |
| `executor_indexing.cpp` | GATHER/SCATTER: an entry (one L value) plus a box of slice coordinates addresses a box of data elements; all per-element work in device kernels. GATHER: L chares send each result chare its L values, the result chare groups entries by data chare and requests each slice piece; data chares copy straight into the result tile in-process, else reply (CUDA IPC slab or host, parts of `CNUM_GX_CHUNK_MB`). SCATTER: L chares send entry buckets to the destination chares, which keep the last write per element (atomic max) and fill a scalar or pull values from v's chares. EXPAND: device scan of each x block, row offsets from one coordinator, positions routed in runs like COMPRESS. COMPRESS/MASK_SCATTER: device scan of each x block's mask, row offsets from one coordinator, compacted runs sent in bulk to the result tiles (or pulled from v's chares) |
| `executor_linalg.cpp` | LINALG. CPU: operand blocks gathered to chare 0 of the result partition (`receive_linalg`), factored with Eigen, result tiles scattered back. NVIDIA: device chunks (<= 64 MB, `receive_data` with input_index `LA_DEV_INPUT` + kind, CUDA IPC between processes) to chare 0 of the operands' partition, which assembles column-major operands in device memory, runs cuSOLVER/cuBLAS in place and sends row-major result chunks to the output tiles (`on_linalg_device`); tssvd with one tile column sums per-chare Gram matrices; lower real cholesky of a whole array is a tiled right-looking factorization in the result tiles (panel chunks between tile chares, NaN status broadcast from the last diagonal tile; `CNUM_LA_DIST=0` gathers instead). `CNUM_LA_TIME=1` prints per-phase times |
| `executor_indexing.cpp` | GATHER/SCATTER/EXPAND: index chares send request lists to the data chares, which send value lists to the target chares (`receive_index_msg`); EXPAND scans segment counts on one coordinator chare. COMPRESS/MASK_SCATTER: device scan of each x block's mask, row offsets from one coordinator, compacted runs sent in bulk to the result tiles (or pulled from v's chares) |
| `executor_creation.cpp` | UPLOAD/RANDOM/IOTA: `stage_uploads()` (PE 0 cuts the payload into tiles and sends them to the chares' home PEs), `execute_creation_node()` (create + fill locally) |
| `creation_kernels.hpp` | Host/device element functions: Philox RNG, Box-Muller with software log/cos, arange/linspace/indices/eye |
| `jit.cpp` | MLIRJitCompiler implementation — buildFromAST (per-leaf element types), emitOp (per-op compute dtype, `convertValue`), registries |

## Standalone C++ Tests (`tests/`)

### `test_compute_decompositions.cpp` — Standalone decomposition coverage (621 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `test_shifted_copy_uses_absolute_offset()` | 131 | Checks lifted absolute offset for a simple shifted copy |
| `test_shifted_expression_temp_uses_absolute_representative()` | 180 | Verifies shifted RHS temp keeps absolute representative `150` |
| `test_jacobi1d_temp_offset()` | 210 | 1D Jacobi temp regression: offset should be `1` |
| `test_jacobi2d_temp_offset()` | 262 | 2D Jacobi temp regression: offset should be `(1,1)` |
| `test_jacobi3d_temp_offset()` | 320 | 3D Jacobi temp regression: offset should be `(1,1,1)` |
| `test_strided_copy_reduces_tile_and_sets_offset()` | 377 | Stride-2 access regression: tile reduction and absolute offset lifting |
| `test_balanced_tiles()` | 453 | `balanced_tile`: 1100^3 on 4 PEs -> 560, 10000^2 -> 2528; powers of two kept unless 20% better (800^3, (2555,160,240)) and on 1 PE |
| `test_chare_home_pe_balance()` | 475 | `chare_home_pe` spreads 2x2, 4x4, 4x1, 2x2x2, ... chare boxes evenly on 4 and 8 PEs, and thin boxes over all of 3, 6 or 12 PEs |
| `test_misaligned_after_flush_takes_source_tile()` | 518 | A result read through shifted views of an earlier DAG's array gets that array's tile and the view offset |
| `test_reduce_operands_share_layout()` | 542 | A new REDUCE operand takes the other operand's tile and offset |
| `test_small_class_member_keeps_own_tile()` | 571 | A (160,365,240) array written into a slice of a (1120,365,240) one keeps its own tile |

### `backend.ci` — Charm++ interface (86 lines)
Module `charmnumeric` (depends on `charmtyles`). Mainchare `Main`. Group `ArrayDAGGroup : DAGGroup` with entries: `set_proxies()`, `proxies_ready()` [reduction], `receive_dag()`, `receive_get_request()`, `gather()`. Arrays `Partition1D/2D/3D` with entries: `run()`, `receive_data()` [nocopy/nocopydevice], `send_complete()`, `receive_ipc()`, `receive_staged()` (GPU), `comm_done()`, `reduce_result()`, `receive_reduce_partial()`, `receive_linalg()`, `receive_shape()`, `receive_frag()`, `receive_sort_msg()` (Partition1D).
