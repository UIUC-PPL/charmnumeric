# Codemap: Charmnumeric (`example/charmnumeric/`)

Distributed N-dimensional array DSL built on charmtyles. The backend stores 1-3D tile-decomposed arrays; arrays with more than 3 dims are folded into 3-D storage by the frontend (`folded.py`). Python `Array`/`ArrayView` classes extend `FrontendObject`; C++ backend uses Charm++ chare arrays (`Partition<N>`) with optional MLIR JIT and GPU (Kokkos) support.

## Python (`charmnumeric/`)

### `charmnumeric/charmnumeric.py` — Core array types & operations (1063 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `DType` | 39 | Wire dtype codes — FLOAT32=0, FLOAT64=1, INT32=2, INT64=3, BOOL=4, INT8=5, INT16=6, UINT8..UINT64=7-10, FLOAT16=11, COMPLEX64=12, COMPLEX128=13; `from_numpy()`, `to_numpy()` |
| `ArrayOperation` | 139 | Opcodes: add=0, sub=1, mul=2, div=3, matmul=4, tanh=5, exp=6, tile=7, reduce=8, matmatmul=9, diag=10; elementwise math/comparisons/where/clip 20-49; creation upload=50, random=51, iota=52; reduction=60, bincount=61; transpose=70; scan=81, sort=82, argsort=83; linalg=100; gather=120, scatter=121, expand=122, compress=123, mask_scatter=124; reshape=140, broadcast=141; convolve=150, flip=151; convert (fused astype)=160, real=161, imag=162, conj=163, bitwise_and/or/xor=164-166, invert=167, left_shift=168, right_shift=169 |
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
| `_batched_matmul()` | — | N-D / FoldedArray / non-squeezable 3-D matmul via `einsum("...ij,...jk->...ik")` |
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
| `take/put/take_along_axis/put_along_axis/compress/repeat/choose` | NumPy API (also `Array` methods); bool arrays index as masks; elements up to 8 bytes (no complex128) |

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
| `_fold()` / `_contract_pair()` | Pairwise contraction (smallest intermediate first within a path step): label-space or grouped broadcast multiply + sum, native dot/matvec, SUMMA, or one SUMMA per batch element on 3-D singleton views |
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

## C++ Backend (`example/charmnumeric/src/`)

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
| — `FreeKey`/`FreeKeyHash` | 370/387 | Array reuse key (dtype+region+shape) |
| — Key methods | | `init()`:567, `create()`:571, `run()`:572, `retire_array()`:428, `try_reuse()`:456, `allocate_or_reuse()`:502, `ensure_array()`:534, `process_get()`:582 |
| `Partition1D/2D/3D` | 587-672 | Thin Charm++ chare wrappers around PartitionImpl |

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

### `decomposition_solver.hpp` — Standalone decomposition solver (1716 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `decomposition_solver::compute_decompositions()` | 95 | Charm-free decomposition solver used by the runtime and standalone tests |
| `detail::for_each_node_topo()` | 59 | Topological DAG traversal helper for solver passes |
| `collect_unionable_nonfixed_leaves()` | 515 | Union-find guard: only unions zero-shift, same-tile, non-fixed temp leaves |

### `array_region.hpp` — N-D regions & decomposition (449+ lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `ArrayRegion<N>` | 22 | N-D region (start/stop/step per dim) — `size()`:110, `deserialize()`:121, `overlaps()`:136, `covers()`:158, `intersect()`:186 |
| `ArrayDecomp<N>` | 294 | Tile decomposition — `default_decomp()`:303, `offset_decomp()`:312, `to_global/local()`:322/339, `num_chares()`:350, `chare_region_global/local()`:360/371 |
| `MemRef<N,T>` | 407 | MLIR memref descriptor (allocated, aligned, offset, sizes, strides) |
| `ChareIndex<N>` | 437 | N-D chare array index |

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
| kernel caches (jit.cpp) | — | In-process map keyed by the kernel IR, then a persistent object-file cache: `$CNUM_JIT_CACHE_DIR` (default `${XDG_CACHE_HOME:-~/.cache}/charmnumeric/jit`)/`<server build id>`/`<sha256(IR, target)>.o`, loaded with an LLJIT (`loadCachedKernel`); NVIDIA cubins in `cubin/`, keyed by PTX, arch and driver; `CNUM_JIT_CACHE=0` disables it, `CNUM_JIT_TIME=1` prints compile phases |
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
| `deferred_send_cb<N_sender,N_target>()` | 83 | (GPU) HAPI callback for deferred send |

### Key C++ source files
| File | Description |
|------|-------------|
| `server.cpp` | Main chare, creates ArrayDAGGroup |
| `dag_group_runtime.cpp` | ArrayDAGGroup constructor, `compute_decompositions()`, set_proxies, receive_get_request, gather |
| `dag_group_compile.cpp` | JIT compilation orchestration |
| `dag_group_receive.cpp` | DAG reception and dispatch |
| `execute_node.cpp` | `execute_node_nd<N,T>()` — dispatches JIT kernels, handles MATMUL/REDUCE/TILE/DIAG |
| `execute_node_regions.cpp` | Region extraction from AST |
| `executor_core.cpp` | `execute_dag_node()`, remote input sends |
| `executor_matmul.cpp` | Matrix-vector multiply execution |
| `executor_matmatmul.cpp` | Matrix-matrix multiply (SUMMA) execution |
| `executor_reduce.cpp` | Reduce node execution |
| `executor_reduction.cpp` | REDUCTION/BINCOUNT: per-chare partials (Kokkos on GPU) sent point-to-point to the output chares (`receive_reduce_partial`) |
| `executor_reducer.cpp` | `determine_dtype<N>()` (dtype of a node from its AST or arrays) |
| `executor_transfer.cpp` | Cross-partition data transfer |
| `executor_transpose.cpp` | TRANSPOSE (2-D/3-D axis permutation of an array or view): point-to-point pack/send, `transpose_place` receive side |
| `executor_castscansort.cpp` | SCAN (cumsum/cumprod/nan*), SORT/ARGSORT executors (opcodes 81-83) |
| `executor_incremental.cpp` | Incremental computation |
| `executor_ast_visitor.cpp` | AST visitor for elementwise node execution |
| `partition_lifecycle.cpp` | `init()`, `create()`, `run()`, destructor |
| `partition_comm.cpp` | `receive_data()` inter-chare communication |
| `partition_get.cpp` | `process_get()` — gather results to PE 0 |
| `executor_signal.cpp` | CONVOLVE/FLIP: each result chare receives the input window (halo of the filter size) and the filter, or its mirrored tile, point-to-point; direct sum or per-chare FFT (host mixed radix, cuFFT on NVIDIA) |
| `executor_shape.cpp` | RESHAPE/BROADCAST (opcodes 140-141): source chares send row runs (reshape) or the projected block (broadcast) to each result chare (`receive_shape`, host-staged); broadcast blocks are expanded on the device |
| `executor_linalg.cpp` | LINALG: operand blocks gathered to chare 0 of the result partition (`receive_linalg`), factored there (Eigen on CPU, cuSOLVER/cuBLAS on NVIDIA), result tiles scattered back |
| `executor_indexing.cpp` | GATHER/SCATTER/EXPAND: index chares send request lists to the data chares, which send value lists to the target chares (`receive_index_msg`); EXPAND scans segment counts on one coordinator chare. COMPRESS/MASK_SCATTER: device scan of each x block's mask, row offsets from one coordinator, compacted runs sent in bulk to the result tiles (or pulled from v's chares) |
| `executor_creation.cpp` | UPLOAD/RANDOM/IOTA: `stage_uploads()` (PE 0 cuts the payload into tiles and sends them to the chares' home PEs), `execute_creation_node()` (create + fill locally) |
| `creation_kernels.hpp` | Host/device element functions: Philox RNG, Box-Muller with software log/cos, arange/linspace/indices/eye |
| `jit.cpp` | MLIRJitCompiler implementation — buildFromAST (per-leaf element types), emitOp (per-op compute dtype, `convertValue`), registries |

## Standalone C++ Tests (`example/charmnumeric/tests/`)

### `test_compute_decompositions.cpp` — Standalone decomposition coverage (416 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `test_shifted_copy_uses_absolute_offset()` | 128 | Checks lifted absolute offset for a simple shifted copy |
| `test_shifted_expression_temp_uses_absolute_representative()` | 177 | Verifies shifted RHS temp keeps absolute representative `150` |
| `test_jacobi1d_temp_offset()` | 207 | 1D Jacobi temp regression: offset should be `1` |
| `test_jacobi2d_temp_offset()` | 259 | 2D Jacobi temp regression: offset should be `(1,1)` |
| `test_jacobi3d_temp_offset()` | 317 | 3D Jacobi temp regression: offset should be `(1,1,1)` |
| `test_strided_copy_reduces_tile_and_sets_offset()` | 374 | Stride-2 access regression: tile reduction and absolute offset lifting |

### `backend.ci` — Charm++ interface (86 lines)
Module `charmnumeric` (depends on `charmtyles`). Mainchare `Main`. Group `ArrayDAGGroup : DAGGroup` with entries: `set_proxies()`, `proxies_ready()` [reduction], `receive_dag()`, `receive_get_request()`, `gather()`. Arrays `Partition1D/2D/3D` with entries: `run()`, `receive_data()` [nocopy/nocopydevice], `send_complete()`, `comm_done()`, `reduce_result()`, `receive_reduce_partial()`, `receive_linalg()`, `receive_shape()`.
