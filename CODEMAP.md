# Codemap: Charmnumeric (`example/charmnumeric/`)

Distributed N-dimensional array DSL built on charmtyles. Supports up to 3D arrays with tile decomposition. Python `Array`/`ArrayView` classes extend `FrontendObject`; C++ backend uses Charm++ chare arrays (`Partition<N>`) with optional MLIR JIT and GPU (Kokkos) support.

## Python (`charmnumeric/`)

### `charmnumeric/charmnumeric.py` — Core array types & operations (1063 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `DType` | 34 | Type encoding — FLOAT32=0, FLOAT64=1, INT32=2, INT64=3; `from_numpy()`, `to_numpy()`, `promote()` |
| `ArrayOperation` | 139 | Opcodes: add=0, sub=1, mul=2, div=3, matmul=4, tanh=5, exp=6, tile=7, reduce=8, matmatmul=9, diag=10; elementwise math/comparisons/where/clip 20-49 |
| `ArrayRegion` | 153 | N-D region with start/stop/step per dim — `serialize()`:194, `shape()`:209, `compose()`:212, `overlaps()`:232, `covers()`:239, `intersect()`:246 |
| `Array(FrontendObject)` | 618 | Main array class — `wire_dtype()`:639, `get()`:683, `get_region()`/`__getitem__()`:690/697, `__setitem__()`:708, `dot()`:724, `matvec()`:727, `copy()`:730, `fill()`:735, `astype()`:738; operators: +,-,*,/,@ at 713-722 |
| `ArrayView(FrontendObjectView)` | 747 | View with same ops as Array — `get()`:796, `__getitem__()`:801, `__setitem__()`:812, `dot()`:828 |
| `_full_region()` | 258 | Get/cache full region for shape |
| `_expand_key()` | 272 | Normalize indexing keys (handles newaxis, ellipsis) |
| `_parse_key()` | 308 | Convert key → ArrayRegion |
| `_array_add/sub/mul/truediv()` | 394-489 | Binary elementwise operator implementations |
| `_ew_operands()` / `_ew_unary/_binary/_ternary()` | 551-606 | Elementwise math, comparisons (0/1 masks in operand dtype), where/clip; dtype/broadcast checks |
| `_compare()` / `_compare_int_float()` | 669, 635 | Comparisons; exact int-array vs float-scalar rewriting |
| `_ElementwiseMethods` | 863 | Mixin for Array/ArrayView: round/clip and old-backend methods (sqrt, log(base), where, max(other), ...) |
| `_array_matmul()` | 492 | Matrix multiply dispatch (dot for 1D×1D, matvec for 2D×1D, matmatmul for 2D×2D) |
| `_matvec()` | 536 | Matrix-vector multiply |
| `_matmatmul()` | 595 | Matrix-matrix multiply (SUMMA) |
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
| `less/.../not_equal`, `logical_*`, `where`, `clip` | 325-375 | Comparisons (44-47) as 0/1 masks, where (48), clip (49) |

### `charmnumeric/reductions.py` — Native reductions
| Symbol | Description |
|--------|-------------|
| `_reduce()` | Builds a REDUCTION (opcode 60) node: operands `x, kind, axis_mask, keepdims, ddof, in_dtype` |
| `sum/prod/mean/var/std/amax/amin/argmax/argmin/count_nonzero/all/any` | NumPy-style API (`axis`, `keepdims`, `dtype`, `out`); also installed as `Array`/`ArrayView` methods |
| `bincount()` | BINCOUNT (opcode 61) node; needs `minlength` or `interface=` |
| `norm()` | `cnp.linalg.norm` subset (vector 2/1/inf/-inf/0, matrix fro/inf/-inf/1/-1, axis) |

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
| `ReduceContrib` | 264 | 24-byte dot reduction contribution |
| `reduce_dot_sum` | 274 | Custom reducer for dot products |

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
| `ArrayDecomp<N>` | 294 | Tile decomposition — `default_decomp()`:303, `offset_decomp()`:312, `to_global/local()`:322/339, `num_chares()`:350, `owning_chare()`:357, `chare_region_global/local()`:360/371 |
| `MemRef<N,T>` | 407 | MLIR memref descriptor (allocated, aligned, offset, sizes, strides) |
| `ChareIndex<N>` | 437 | N-D chare array index |

### `opcodes.hpp` — Operation codes (50 lines)
| Symbol | Line | Description |
|--------|------|-------------|
| `Opcode` | 3 | COPY=-5..DIAG=10; elementwise SQRT=20..CLIP=49 |
| `is_binary_elementwise()` | 23 | ADD/SUB/MUL/DIV |
| `is_unary_elementwise()` | 36 | TANH/EXP and unary math/predicates (20-36) |
| `is_ternary_elementwise()` | 109 | WHERE/CLIP |
| `is_elementwise()` | 47 | Binary or unary elementwise |

### `jit.hpp` / `jit.cpp` — MLIR JIT compiler (150+ lines each)
| Symbol | Line | Description |
|--------|------|-------------|
| `BinaryOpEntry` | 80 | Registry: float_body, int_body function pointers |
| `UnaryOpEntry` | 87 | Registry: float_body, int_body function pointers |
| `TernaryOpEntry` | 94 | Registry for 3-input ops (where, clip); `emitGenericNaryOp` builds the N-input linalg.generic |
| `MLIRJitCompiler` | 101 | AST → MLIR → executable — `buildFromAST()`, `optimizeAndFuse()`, `generateCPU()`, `loadCPU()`, `generateNVIDIA/AMD/Intel()` |

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
| `executor_reducer.cpp` | Custom reduction operations |
| `executor_transfer.cpp` | Cross-partition data transfer |
| `executor_incremental.cpp` | Incremental computation |
| `executor_ast_visitor.cpp` | AST visitor for elementwise node execution |
| `partition_lifecycle.cpp` | `init()`, `create()`, `run()`, destructor |
| `partition_comm.cpp` | `receive_data()` inter-chare communication |
| `partition_get.cpp` | `process_get()` — gather results to PE 0 |
| `executor_creation.cpp` | UPLOAD/RANDOM/IOTA: `stage_uploads()` (cut payload into this PE's chare tiles), `execute_creation_node()` (create + fill locally) |
| `creation_kernels.hpp` | Host/device element functions: Philox RNG, Box-Muller with software log/cos, arange/linspace/indices/eye |
| `jit.cpp` | MLIRJitCompiler implementation — buildFromAST, emitOp |

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
Module `charmnumeric` (depends on `charmtyles`). Mainchare `Main`. Group `ArrayDAGGroup : DAGGroup` with entries: `set_proxies()`, `proxies_ready()` [reduction], `receive_dag()`, `receive_get_request()`, `gather()`. Arrays `Partition1D/2D/3D` with entries: `run()`, `receive_data()` [nocopy/nocopydevice], `send_complete()`, `comm_done()`, `reduce_result()`.
