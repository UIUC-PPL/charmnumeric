# What it would take to support the missing ops natively

Companion to `README.md` (same directory). For every op or op family that the
cuPyNumeric ports emulate, run on the host, or cannot express, this file
estimates the cost of a native charmnumeric implementation. Each estimate is
grounded in the current code.

Paths below are relative to `charmtyles/`. `E/` means `example/charmnumeric/`.
Line numbers were taken on 2026-10-08 at about 18:25. `E/src/jit.cpp`,
`dag_group_receive.cpp` and `executor_core.cpp` were being edited during the
analysis (for example, the NVIDIA JIT now uses the upstream
`gpu-lower-to-nvvm` pipeline, `jit.cpp:431-484`), so re-grep before relying
on an exact line.

**Effort:** S = under 1 day, M = 1-3 days, L = about 1-2 weeks,
XL = research-level or open-ended.

**Layers:** FE = Python frontend (`E/charmnumeric/*.py`, `charmtyles/*.py`).
OPC = opcode in `E/src/opcodes.hpp` plus `ArrayOperation` in the frontend.
WIRE = DAG serialization (`charmtyles/ast.py` and
`include/charmtyles/core/dag.hpp`). SOLV = `E/src/decomposition_solver.hpp`.
RECV = shape and partition-expansion passes in `dag_group_receive.cpp`.
JIT = registry or `emitOp` in `jit.cpp`/`jit.hpp`. EXEC = new executor with
inter-chare messaging. KK = Kokkos or KokkosKernels device kernel (plus the
CPU path).

---

## 0. Two extension recipes (where things plug in)

### Recipe A: a new elementwise op (JIT only)

1. Add an enum value in `E/src/opcodes.hpp:3-20`. Add it to
   `is_binary_elementwise` (`:23`) or `is_unary_elementwise` (`:36`).
2. Add a registry entry in `E/src/jit.cpp`: `binaryOpRegistry()` at `:512`
   or `unaryOpRegistry()` at `:546`. An entry is a pair of `float_body` and
   `int_body` lambdas (`jit.hpp:80-91`). A body lambda may emit several ops,
   for example cmp+select.
3. Mirror the opcode in `ArrayOperation` (`E/charmnumeric/charmnumeric.py:139-150`).
   Add a function in `operations.py` modelled on `tanh`/`exp` (`:38-49`).
   Export it from `__all__` (`:151`) and `__init__.py:26`. For operators, add
   dunders on both `Array` and `ArrayView` (`charmnumeric.py:713-722` and
   `~817-826`).

Nothing else changes. The receive shape pass (`dag_group_receive.cpp:379,522`),
the solver (`decomposition_solver.hpp:357,567,790,1630`) and the
fragment/halo path in `execute_node.cpp` all key off `is_elementwise()`.
Shifted-view (stencil/halo) operands therefore work for free.

Limits of this recipe today:

- **One element type per kernel.** `buildFromAST` picks a single `elemType`
  from the first non-f32 root (`jit.cpp:65-86`). Every memref argument gets
  that type (`:96-104`). `dispatch_kernel<N,T>` (`dispatch.hpp:141`) and
  `execute_node_nd<N,T>` are templated on a single `T`. The wire does not
  carry operand dtypes either: a `FRONTEND_OBJECT` operand is
  name+ndims+region only (`ast.py:94-104`, `dag.hpp:137-141`). An op whose
  result dtype differs from its inputs (bool compare, astype, int+float
  promotion) cannot be expressed. See prerequisite P1.
- ~~**Unary ops assume a memref input.**~~ Fixed (2026-10-08): a unary op
  on a scalar/broadcast operand emits a plain scalar op, like the binary
  path (`emitOp` in `jit.cpp`).
- **No integer unary bodies for float-only math.** Unary `int_body == nullptr`
  makes the float body run on an int type, which produces invalid IR. The
  frontend therefore rejects int inputs to sqrt/log*/exp2/expm1/log1p/sin/cos
  (`_ew_unary(..., float_only=True)` in `charmnumeric.py`) until astype
  exists. abs/floor/ceil/rint/sign/isnan/isinf/isfinite have int bodies.
- ~~**Only two inputs.**~~ Ternary ops exist now: `TernaryOpEntry` +
  `ternaryOpRegistry()` (`jit.hpp`/`jit.cpp`), the N-ary
  `emitGenericNaryOp` (any input may be a scalar, wrapped as a 0-d memref),
  and `is_ternary_elementwise` in `opcodes.hpp` (WHERE, CLIP). The
  receive/solver/fragment passes already iterate over all operands. General
  broadcasting still needs projected indexing maps.

### Recipe B: a non-elementwise op with its own executor (templates: DIAG, TILE, REDUCE)

1. **OPC + FE.** Build the node with
   `nonfusible_op(x, *int_params, operation=..., shape=..., dtype=...)`, as
   `diag` does (`operations.py:125-148`). Integer parameters travel as
   `SCALAR` operands (doubles), so no wire change is needed
   (`dag.hpp:162-170`).
2. **RECV**, in `dag_group_receive.cpp`:
   - mark the participating ranks active (`:212-221`),
   - add the result to `array_meta` in the shape pass (DIAG `:333`,
     TILE `:356`),
   - expand the result partition (DIAG `:467`, TILE `:490`).
3. **SOLV.** Pin operands and result to zero offset by adding them to
   `matmul_arrays` (`decomposition_solver.hpp:590-595,1705`), unless the
   executor handles decomposition offsets.
4. **Relevance and dispatch**, in `executor_core.cpp`:
   - decide per rank which chares take part (REDUCE `:255`, DIAG `:295`,
     TILE `:302`; non-relevant chares return at `:347`),
   - dispatch to `execute_X_node` at `:355-390`.
5. **EXEC send side.** Declare `execute_X_node` next to the others in
   `backend.hpp:336-341`. Write `executor_X.cpp` modelled on
   `execute_diag_node` (`executor_transfer.cpp:380`, with a typed
   `diag_send` at `:180` and a DType switch at `:360`) or `execute_tile_node`
   (`:720`). Register the file in `E/src/CMakeLists.txt:141-150`.
6. **EXEC receive side.** Add a branch in the interpreter section of
   `execute_node_nd` (`execute_node.cpp:111`). Use DIAG `:600-753` or
   TILE `:755-942` as the template:
   - `ensure_array_typed` the output,
   - place `comm->remote_buffers` using `Kokkos::parallel_for` under
     `USE_KOKKOS`, and plain loops otherwise.
7. Add a no-op `case` in `executor_ast_visitor.cpp:596-601`. Without it,
   `default:` aborts.
8. **Reductions.** Copy `executor_reduce.cpp:7-119`:
   - local partial,
   - `contribute` with a custom reducer (`executor_reducer.cpp:11-61`,
     registered at `:59`),
   - a `ReduceContrib` payload (`backend_internal.hpp:301`),
   - the result lands in `PartitionImpl::reduce_result`
     (`partition_comm.cpp:69`).

   Charm++ reductions need **every element of the chare array** to
   contribute. REDUCE marks every 1-D chare relevant (`executor_core.cpp:255-259`).
   A 2-D or 3-D reduction must do the same for `Partition2D/3D`, or it hangs.

A full non-elementwise op costs about 6 files and 300-600 lines (DIAG is about
370 lines on the send side plus about 150 on the receive side). That is why
most such ops below are M or more.

---

## 1. Quick wins (registry + opcode + frontend function, or frontend only)

Each item below is S. Items 1-4 together are about 1-2 days including tests.

**Status 2026-10-08: Q1-Q6 are DONE** (opcodes 20-49, `src/opcodes.hpp`;
registries in `src/jit.cpp`; frontend in `charmnumeric/charmnumeric.py`
"Elementwise math, comparisons and selection" and `charmnumeric/operations.py`;
tests in `examples/ops/test_{unary,binary,compare,where}.py`, all PASS on the
CPU build at +p1 and +p2). Notes on what was built:
- Q1: sqrt, log (+`base=`), log2, log10, log1p, exp2, expm1, sin, cos,
  abs/absolute/fabs, floor, ceil, rint/round(decimals)/around (half-even),
  trunc/fix (frontend: where(x<0, ceil, floor)), sign, square, reciprocal,
  negative/positive. log1p/expm1 use the Goldberg/Kahan forms because the CPU
  math-to-llvm lowering is the naive log(1+x)/exp(x)-1. Old-backend methods
  (`x.sqrt()`, `x.log(base)`, `x.abs()`, `x.where(a, b)`, `x.max(other)`, ...)
  are on `Array`/`ArrayView`.
- Q2: power/`**`/`__rpow__`: `math.powf`; integers use a branch-free
  square-and-multiply (no `math.ipowi`, which needs math-to-funcs). `x**2`
  becomes `x*x` as in NumPy. `int ** negative scalar` raises like NumPy.
- Q3: maximum/minimum (NaN-propagating), fmax/fmin (NaN-ignoring).
- Q4: remainder/mod/`%`, floor_divide/`//`, divmod, reverse operators; exact
  NumPy semantics (npy_divmod for floats; ints: sign of divisor, x//0 = x%0 =
  0 without trapping, INT_MIN // -1 wraps).
- Q5: less/less_equal/greater/greater_equal/equal/not_equal and operators,
  isnan/isinf/isfinite, logical_and/or/xor/not (`& | ^ ~` are logical on
  masks). greater* reuse LESS/LESS_EQUAL with swapped operands. Results are
  0/1 masks in the operand dtype. An int array compared with a non-integral
  float scalar is rewritten exactly (x < 2.5 -> x < 3; NaN/out-of-range give
  constant masks).
- Q6: where (cond != 0, NaN truthy) and clip as real 3-input fused ops.
- Restrictions (single element type per kernel): array operands must share a
  dtype; a non-integral float scalar with an int array raises TypeError;
  only size-1 operands broadcast. Mask assignment `A[mask] = s` is
  `A[...] = cnp.where(mask, s, A)`.

| # | op(s) | what to do | examples helped |
|---|---|---|---|
| Q1 | `sqrt, log, abs, floor, ceil, round` (half-even), `log1p, expm1, rsqrt, sin, cos` | Recipe A, unary. Use `math::SqrtOp, LogOp, AbsFOp` (int: `AbsIOp`), `FloorOp, CeilOp, RoundEvenOp, Log1pOp, ExpM1Op, RsqrtOp, SinOp, CosOp`. Also apply the one-time scalar-input fix at `jit.cpp:375`. Float only until astype exists. | black_scholes, black_scholes_greeks, logreg (log, floor), tssvd (abs), quantiles (floor/ceil/round methods), solve check (abs for inf-norm) |
| Q2 | `power` / `**` | Recipe A, binary: `math::PowFOp` (int: `math::IPowIOp`). The frontend should also rewrite a literal `**2` to `x*x`. | cfd, linreg, lstm_backward, kmeans (`square`) |
| Q3 | `maximum, minimum` | Recipe A, binary: `arith::MaximumFOp`/`MinimumFOp` (NaN-propagating, like NumPy), `MaxSIOp`/`MinSIOp`. | kmeans, kmeans_slow. Also makes the sort/quantiles compare-exchange network exact (`lo=minimum(a,b)`, `hi=maximum(a,b)`) and removes the tanh-step's 2e-9 resolution limit. |
| Q4 | `mod` (`%`), `floor_divide` (`//`) | Recipe A, binary: `RemFOp`/`RemSIOp`, plus a sign fix-up so that `sign(r) == sign(b)` (cmp+select inside the body lambda). `floor_divide` is `DivSI` with a floor correction. | black_scholes_greeks, indexing_routines, `_cnum_util.rand_int` |
| Q5 | comparisons `< <= > >= == !=`, `isnan`, `isinf` as **numeric masks** | Recipe A, binary/unary: `arith::CmpFOp`/`CmpIOp` (`isnan` = `CmpF UNO(x,x)`, `isinf` = `abs(x)==inf`), then `UIToFP`/`ExtUI` back to the operand type. The result dtype equals the input dtype (1.0/0.0), not bool. Document this deviation. | black_scholes(+greeks) `d>0`, kmeans `not_equal`, kmeans_slow `labels==idx`, sort, quantiles, scan nan* (`isnan`), jacobi/logreg failure-path diagnostics |
| Q6 | `where(c, a, b)` | Interim, FE only (S): `b + c*(a-b)`. Exact when the unselected branch is finite. Proper version (S-M, about 1 day): a `TernaryOpEntry`, a 3-input `linalg.generic` next to `emitGenericArrayOp`, an `emitOp` case, and `is_elementwise` extended. With masks from Q5 this also gives `choose` (nested where) and **mask assignment** `A[mask] = s` → `A[...] = where(mask, s, A)`. | black_scholes(+greeks), kmeans_slow, scan nan*, indexing_routines (choose, ai1/ai2/ai3 mask assignments) |
| Q7 | FE-only API gaps | See the list below. | see list |
| Q8 | cross-dim `set_region` at a nonzero offset (a **bug**, not an op) | `cross_set_region_send` maps source coordinates straight to target coordinates (`tgt_start[td] = src_local_start[sd]`, `E/src/backend_internal.hpp:384-388`). It ignores `tgt_region->start[td]` in non-singleton dims, and it sends the whole source rather than the view region. Add the target-region start and clip to the source view. S-M. (README cites `:345-351`; that reference is stale.) | kmeans `centroids[:, idx] = bincount(...)`, lstm_forward `Hin[t,:,word_size:] = ...`, lstm_backward `dh0[0] += np.sum(...)` |

Q7, the frontend-only API gaps:

- **`hstack`/`vstack`/`concatenate`**: `create_array` plus slice `set_region`
  per piece. Native today. Needed by linreg and logreg.
- **`out=`**: `out[...] = result`. Needed by gemm, einsum and scan.
- **`eye`**: build it as `diag(ones(n))`. Today it issues N scalar ops
  (`charmnumeric.py:1048`).
- **Name limit**: raise `max_name` (`charmtyles/core.py:9`). Nothing in the
  backend depends on 2^16.
- **Views passed to REDUCE/TILE/DIAG**: auto-materialize them, because those
  executors ignore view regions (README issue 5).
- **`sum()` of 1-D/2-D arrays**: use REDUCE or matvec with `ones`, as
  `_cnum_util.vsum/msum` do today.
- **Uniform-count `repeat`**: strided slices.
- **`take` with host-known indices**: lower to column-slice copies, 496 for
  take_pairs' N=32. Cheaper and exact compared with the 0/1 selection-matrix
  SUMMA.
- **General broadcasting**: until it exists natively, materialize it with
  `tile`.

---

## 2. Cross-cutting prerequisites

These unlock several rows of the main table at once.

| id | prerequisite | effort | what changes |
|---|---|---|---|
| P1 | **Per-operand dtypes in fused kernels** (bool, astype fusion, int/float promotion) | L | WIRE: add the operand dtype to `FRONTEND_OBJECT`/`BROADCAST_OBJECT` (`ast.py:94-104`, `dag.hpp:137-141`) and include it in the node identifier (compile-cache key). JIT: per-leaf memref element types (`jit.cpp:96-104`), plus explicit `arith` casts where operand and result types differ. EXEC: `dispatch_kernel` must pack heterogeneous `MemRef<N,T>` descriptors (same layout, so pack them type-erased) and allocate outputs of a different `T` than the inputs (`execute_node.cpp:~1300-1330`). |
| P1b | **New DTypes** (bool/uint8, uint32, uint64, f16, c64/c128) | M-L each, after P1 | `DType` enum and `dtype_size` (`include/charmtyles/core/ast.hpp:8-29`); the frontend `DType` maps (`charmnumeric.py:34-75`, `interface.py:7-8`); about 22 `switch (dt)` sites in `E/src/*.cpp`; 12 more `execute_node_nd` instantiations per dtype (`execute_node.cpp:1588-1599`); the JIT `elemType` switch; reducer cases. Unsigned types also need signedness-aware registry bodies (`DivUI`, `CmpI ult`, `UIToFP`). Complex needs the MLIR `complex` dialect. f16 needs `Kokkos::Experimental::half_t` GEMM and f32-extended math on CPU. |
| P2 | **Generic full-array reduction** (any N, op kind ∈ {sum, prod, max, min, argmax, argmin, count_nonzero, any, all}) | M | Generalize REDUCE: relax N==1 (`executor_reduce.cpp:9-12`, `executor_core.cpp:255-259`; all chares must stay relevant); add `op_kind` and an `int64 index` to `ReduceContrib`; one reducer per kind (`executor_reducer.cpp`); local partials via `Kokkos::parallel_reduce` with `Sum/Max/Min/MaxLoc/MinLoc` (CPU: loops or Eigen). The callback still targets 1-D chare 0 (`executor_reduce.cpp:108-110`). |
| P3 | **"Local op" framework**: run a dense kernel on one chare, with gather→compute→scatter when the operand spans more than one tile | L (framework), then S-M per kernel | Comm: all-to-one gather and one-to-all scatter of tiles (like the `get` path: `partition_get.cpp:4-66`, `dag_group_runtime.cpp:55-116`). On GPU builds the minimum tiles are 1024² for 2-D, 1M for 1-D and 128³ for 3-D (`backend.hpp:10-17`). So most default example sizes already sit on one chare, and the gather is a no-op. Exceptions: take_pairs 10000×32; on CPU builds also qr 1000² and sort 1e6. It is not scalable, but it turns every remaining HOST-FALLBACK into a device-resident op with no host round trip. |
| P4 | **Bulk host→device upload** — **DONE (P4a)** | M | Implemented as an INLINE_DATA operand (type 4, `[nbytes:i64][bytes]`, `charmtyles/ast.py` `InlineData`, `include/charmtyles/core/dag.hpp`) on an UPLOAD node (opcode 50). `ArrayDAGGroup::stage_uploads` (`E/src/executor_creation.cpp`) cuts the payload into the tiles of the PE's own chares (home PE) right after the decomposition pass and frees the full payload; each chare moves its tile into the new array (`copyToDevice` on GPU builds). Ordering is the DAG's (the upload is an ordinary node writing a fresh array). Cost: P× replication of the DAG message; the size field is int32 (< 2 GiB). The frontend drops the payload after the flush. `_assign_host_data`'s per-element loop is gone. |

Library note for P3 and the factorization rows: the installed KokkosKernels
4.7.01 is built with **only the BLAS component and no TPLs**
(`.../kokkos-kernels-4.7.01-cuda/include/KokkosKernels_config.h:178-180`, all
`KOKKOSKERNELS_ENABLE_TPL_*` undefined). So there is no KokkosLapack and no
cuBLAS or cuSOLVER behind it. Two ways to get dense factorizations:

- **CPU:** Eigen, already a dependency (`E/src/CMakeLists.txt:51-54`):
  `LLT`, `PartialPivLU`, `HouseholderQR`, `SelfAdjointEigenSolver`, `BDCSVD`.
- **GPU:** link cuSOLVER directly (`libcusolver.so.11` exists under the HPC SDK
  `math_libs/12.9/lib64`), or rebuild KokkosKernels with the LAPACK component
  and the CUSOLVER TPL.

cuRAND and cuFFT are present too. Kokkos core ships `Kokkos_Sort.hpp`,
`Kokkos_Random.hpp` and `Kokkos_StdAlgorithms.hpp` (`inclusive_scan`).

---

## 3. Main table

The **Comm** column gives the cross-tile communication pattern of the native
distributed version.

| op / family | effort (interim → native distributed) | layers | comm | dtype issues | examples unblocked or simplified | cheap interim |
|---|---|---|---|---|---|---|
| unary math: sqrt, log, abs, floor, ceil, round, … | **DONE** (Q1) | OPC, JIT, FE | none | float-only math rejects int input until astype (P1) | black_scholes, black_scholes_greeks, logreg, tssvd, quantiles (methods), solve check | native |
| power / square | **DONE** (Q2) | OPC, JIT, FE | none | int `**` by square-and-multiply | cfd, linreg, lstm_backward, kmeans | native |
| maximum / minimum | **DONE** (Q3) | OPC, JIT, FE | none | none | kmeans, kmeans_slow; exact sort/quantiles network | native |
| mod, floor_divide | **DONE** (Q4) | OPC, JIT, FE | none | NumPy sign semantics | black_scholes_greeks, indexing_routines | native |
| comparisons, isnan, isinf | **DONE** as numeric masks (Q5) → **L** (true bool, P1+P1b) | OPC, JIT, FE (+WIRE, EXEC, all DType switches for bool) | none | Bool result is impossible today: single `elemType`, no bool DType | black_scholes(+greeks), kmeans(+slow), sort, quantiles, scan nan*, indexing_routines, jacobi/logreg diagnostics | 0/1 mask in the input dtype (native) |
| where / select / choose / mask-assign | **DONE** (Q6, ternary JIT op) | OPC, JIT (`TernaryOpEntry`), FE | none (halo for shifted views, as for any elementwise op) | cond must have the dtype of x/y until P1 | black_scholes(+greeks), kmeans_slow, scan nan*, indexing_routines (choose) | native; ai* scatters still need gather/scatter |
| astype | **M** (non-fused CAST executor) → **L** (fused, via P1) | OPC, FE, RECV (result meta = input meta), EXEC (local only), KK convert kernel | none: output decomposition = input decomposition | needs a 4×4 (later more) dtype dispatch; outputs allocated with a different T | quantiles (int), svd (int), scan (int), black_scholes_greeks (int→float), kmeans (labels/counts), indexing_routines (bool) | none native; host round trip today |
| native RNG: rand, random, randn, uniform, normal, randint, seed, default_rng; arange/linspace/indices/meshgrid/eye — **DONE** | M | OPC (51 RANDOM, 52 IOTA), FE (`random.py`, `charmnumeric.py`), RECV/SOLV (treated like CREATE: `is_creation()`), EXEC (`executor_creation.cpp`, kernels in `creation_kernels.hpp`) | none | randint takes exact int64 bounds (parameters travel as hi/lo 32-bit scalar pairs) | cholesky, qr, solve, svd, tssvd, take_pairs, jacobi, gemm, black_scholes, scan switched; the rest still use `_cnum_util` generators | Philox4x32-10 keyed by (seed, stream) on the global element index: independent of decomposition/PEs; CPU and GPU builds bit-identical (no FMA, software log/cos). Streams differ from NumPy's. |
| bulk host→device upload — **DONE (P4a)** | M | WIRE, OPC, EXEC, FE | none | none | cholesky, qr, solve, svd, take_pairs (host factors / selection matrices); removes the 65k-element cap | see P4 |
| full reductions: sum, prod, max, min, argmax/argmin (flat), count_nonzero, any/all, norm(inf/2) | **DONE** (REDUCTION opcode 60, `E/src/executor_reduction.cpp`, `E/charmnumeric/reductions.py`; also mean/var/std, views, 1-3-D, all dtypes). Was: **M** (P2) | OPC, FE, RECV, EXEC (`executor_reduce` generalized), reducer | reduction tree (Charm++ `contribute`) | argmax returns int64; result is a (1,) array on 1-D chare 0 | cfd, jacobi, linreg, logreg, kmeans (distance sum, `sum(changes)`), tssvd, take_pairs, indexing_routines warmup, qr check (`max`), solve check (`norm(·,inf)`) | dot/matvec with `ones` (sum only) |
| axis reductions: sum, mean, amin, amax, argmin, argmax, norm along one axis | **DONE** (REDUCTION opcode 60 with an axis mask: any axis or axis tuple of 1-3-D operands, keepdims; partials go point-to-point to the owning output chares). Was: **M** → **L** | OPC, FE, RECV, SOLV (pin), EXEC, KK | partial vectors reduced along one tile-grid dimension to the owning 1-D chares (exactly matvec's `cross_matmul_send_result_2d_to_1d`, `backend_internal.hpp:261`) | arg* output is int64, so the output dtype differs from the input; (value, index) pairs on the wire | kmeans, kmeans_slow (norm axis=1, argmin, amin, sum axis=0), lstm_backward (sum axis 0), tssvd (row sums), solve (pivot argmax), quantiles(axis) | sum = `A @ ones` (exact); argmin/amin need the Q3/Q5 one-hot trick |
| bincount (weights, minlength) | **DONE** (BINCOUNT opcode 61; length = minlength or max(x)+1 via `interface=`). Was: **M** after P2 | OPC, FE, EXEC, reducer (vector-valued) | local histogram, then an all-reduce of a length-C vector (reduction tree) | int labels with float weights: mixed dtype inside one op (fine in an executor) | kmeans, kmeans_slow | one-hot M with `ones @ M`, `dataT @ M` |
| transpose (2-D, materialized) | **M** (about 3 days) | OPC, FE (`.T`, `transpose()`, `swapaxes`), RECV (shape + expansion of the N×M grid), SOLV (pin zero offset), EXEC, KK local transpose | **point-to-point permutation**: square tiles of the same size for every dim (`array_region.hpp` `ArrayDecomp::tile`), so tile (i,j) goes to chare (j,i); diagonal tiles stay local | none | kmeans, kmeans_slow, linreg, logreg, lstm_backward, einsum, cholesky, qr, svd, tssvd | generate transposed inputs; symmetry tricks |
| vecmat (1-D @ 2-D) and transposed-operand matmul (`A.T @ B`, `A @ B.T`) | **M** (vecmat as a mirror of matvec) → **L** (SUMMA variants without materializing) | FE (`_array_matmul`, `charmnumeric.py:522`), EXEC (`executor_matmul.cpp` / `executor_matmatmul.cpp` flag) | vecmat: vector slices multicast along grid rows, partials reduced along grid columns. SUMMA-T: panel broadcast along the other grid axis. | none | linreg, logreg (`err.dot(X)`), lstm_backward, qr/svd (`A^T A`, `Q^T Q`) | materialized transpose + existing SUMMA |
| 3-D axis permutation (transpose for N=3) | **L** | as for 2-D, rank 3 | point-to-point (tile-index permutation); the 3-D grid may need expansion | none | einsum (general), lstm (if not reduced to 2-D) | per-slice 2-D transposes |
| dense factorizations: potrf, getrf+getrs, geqrf, syevd, gesvd, tssvd | **L** (P3 + Eigen/cuSOLVER, S-M per routine) → **XL** (distributed blocked: ScaLAPACK-style panels, pivot search across chares, TSQR trees, two-stage eigensolvers) | OPC, FE (`linalg` module), EXEC (P3), KK or cuSOLVER link, CMake | P3: gather/scatter. Distributed: panel broadcasts plus column-max reductions (getrf) and binary reduction trees (TSQR for tssvd). | s/d only (z with P1b) | cholesky, qr, solve, svd, tssvd | the ports' blocked algorithms with host-side bs×bs kernels |
| trsm / trtri | **S-M** on P3 (diagonal block) → **L** (distributed wavefront) | OPC, FE, EXEC | wavefront along the tile diagonal | none | cholesky, solve, qr | invert the small diagonal block (host) + SUMMA |
| gather / take / fancy indexing / row permutation from a **device** index array | **XL** | OPC, FE (`_parse_key` accepts only int/slice, `charmnumeric.py:308-343`), WIRE, EXEC | **data-dependent all-to-all** (request/reply). Communication can't be planned from regions at DAG-build time. | int64 index arrays | indexing_routines (ai1/ai2/ai3), take_pairs, solve (device pivots) | **host-known indices → M**: indices as SCALAR operands give a static plan like slicing. take_pairs: S via column-slice copies (Q7) |
| boolean-mask **read** `A[mask]` (compaction), variable-count `repeat`, `nonzero` | **L-XL** | as above + a scan (prefix count) + a host sync for the result shape | prefix scan of counts along the tile order + redistribution | bool (P1b) | indexing_routines (zipped `A2[b, b]`, `A3[idx, :, idx]`), `repeat(A2, R, axis=1)` with an array R | mask *assignment* is S via where (Q6) |
| reshape / flatten / ravel (1-3 D) | **M-L** | OPC, FE, RECV, EXEC | static redistribution: each source tile's contiguous row-major runs go to the 1-2 target tiles that own those linear ranges (all-to-all-ish). The linearization is the same as the `get` path (`partition_get.cpp:23-66`). | none | black_scholes_greeks, take_pairs, scan (axis=None), sort/quantiles/svd input reshapes | adding/removing singleton dims via cross-dim set_region (after Q8): **S** |
| more than 3 dims | **L-XL** | `PartitionTraits<4>`/`Partition4D` (`backend.hpp:22-43`, `backend.ci`), all `if constexpr (N==…)` and `switch(nd)` sites (`dag_group_receive.cpp` aborts on nd>3), solver, instantiations ×dtypes | same as now | none | black_scholes_greeks (4-D CALL/PUT), scan/sort/quantiles with ≥4-D `--shape`, einsum (≥4 modes) | FE folding of contiguous leading dims into dim 0 (what the greeks port does): S-M |
| sort / argsort | **M** (local op: `Kokkos::sort` / `sort_by_key`, `std::stable_sort`; arrays that fit one tile, plus segmented sort along an axis that lies inside a tile) → **L-XL** (distributed sample sort) | OPC, FE, EXEC, KK | sample sort: gather samples, broadcast splitters, **data-dependent all-to-all**, local merge, then rebalance to block tiling (a second all-to-all) | argsort returns int64; uint32 is the default dtype (P1b) | sort, quantiles | O(n²) odd-even network (exact once Q3 exists) |
| scan: cumsum, cumprod (nan* variants) | **S-M** (local op, `Kokkos::parallel_scan` / `std::inclusive_scan`) → **M-L** (distributed 1-D, 3 phases) → **L** (axis scans in 2-D/3-D) | OPC, FE, EXEC, KK | local scan, then an exclusive scan of per-tile totals along one grid dim (a chain or tree of scalars), then an elementwise add of the offset | nan* variants = `where(isnan(x), id, x)` first (Q5/Q6) | scan | Hillis-Steele with shifted slices (O(n log n) work) |
| convolve, N-D, mode="same", direct | **M-L** | OPC, FE, EXEC, KK | **halo exchange** of width f//2 per side, the same pattern as the fragment path for shifted views; then a local direct convolution | none | richardson_lucy | f³ shifted-view MACs (works today; about 64 views per call) |
| convolve via FFT (`--conv-method fft`, possibly also chosen by `auto`) | **M** (single-tile cuFFT/FFTW) → **XL** (distributed FFT with transposes) | EXEC, cuFFT link | all-to-all transposes | complex intermediates (P1b) | richardson_lucy (non-default) | direct convolution |
| flip / negative-step slicing | **M-L** | FE (`_parse_key` rejects step<0, `charmnumeric.py:321-322`), JIT (negative memref strides), EXEC (mirrored source tile: point-to-point, split across ≤2 tiles when misaligned) | point-to-point mirror | none | richardson_lucy (`np.flip(psf)`) | reversed-index generation (the port's approach) |
| general broadcasting, newaxis | **L** (S-M for the JIT part) | FE (`_expand_key` raises on `None`, `charmnumeric.py:292`), JIT (projected indexing maps in `emitGenericArrayOp`), SOLV (do not union lower-rank leaves with the output), EXEC | **multicast** of the lower-rank operand slice to every chare along the broadcast dim (like matvec's vector send, `executor_matmul.cpp:6-60`) | none | kmeans, kmeans_slow (`[:, newaxis]`), svd/tssvd check (`u * s`); also fixes README issue 4 (silent wrong results) | `tile()` to materialize (S, memory cost); SUMMA with ones |
| einsum | **L** after transpose + reshape + batched matmul; **XL** for arbitrary expressions | FE lowering (modes → permute/reshape/matmul/axis-sum) | inherits from the lowered ops | none | einsum | the 4 hand-lowered expressions |
| batched matmul (3-D) | **M-L** | EXEC (`executor_matmatmul.cpp` already has a 3-D dim-dropped path) | local `KokkosBatched` gemm when the contracted dims fit in a tile; otherwise per-batch SUMMA | none | einsum (`kac,kbc->kab`), lstm | loop of 2-D SUMMA calls (the port does this) |
| complex dtypes | **L** (after P1) | P1b + `complex` dialect + KK `Kokkos::complex` gemm + reducers | none | everything | scan, sort, svd, einsum (c64/c128) | none |
| float16 | **M-L** | P1b | none | f16 math must be extended to f32 on CPU | gemm, black_scholes, linreg, logreg, kmeans, einsum (`--precision 16`) | none |
| unsigned ints (uint32, uint64) | **M-L** | P1b + unsigned registry bodies | none | signedness in Cmp/Div/convert | sort and quantiles (**uint32 is the default dtype**), kmeans (uint64 counts) | int64 stand-in |
| exact int64 scalars | **S-M** | WIRE: a new `SCALAR_I64` operand (today every scalar is a double: `ast.py:105-107`, `dag.hpp:162-163`, `ast.hpp:38`) + JIT constant path (`jit.cpp:128`) | none | values above 2^53 are rounded today | sort/quantiles with `iinfo(int64)` bounds; randint | keep |values| < 2^53 |
| mixed-dtype elementwise promotion | **S** (FE: promote scalars, insert explicit astype) after astype; **L** natively (P1) | FE (`_operand_dtype` ignores scalars, `charmnumeric.py:361-366`) | none | **Likely silent bug today**: array-array int+float is fused with one `elemType`, so the int buffer is reinterpreted as float (`jit.cpp:65-104`, `dispatch.hpp:141`). Not verified at runtime. | black_scholes_greeks (`(temp_arr % n) * money_step` stays int64 instead of becoming float64), kmeans (counts / centroids) | keep everything in one dtype (the ports do) |

---

## 4. Per-op notes

**Q1-Q6, elementwise.** The current NVIDIA path runs
`gpu::buildLowerToNVVMPassPipeline` (`jit.cpp:479-484`), which includes the
NVVM/libdevice math patterns, so `math.log/pow/sqrt` should lower to `__nv_*`
calls. Compile each new op once for sm_80 anyway. The AMD path still runs
`ConvertMathToLLVM` before `ConvertGpuOpsToROCDLOps` and needs the same check.
Comparisons must convert `i1` back to `elemType` inside the body. Otherwise
`linalg.yield` mismatches the output memref type, which `emitOp` allocates
with the lhs element type (`jit.cpp:338-352`). Every new elementwise op fuses
with its neighbours automatically. Fusion is split only on output dtype
change (`charmtyles/ast.py:731`).

**astype (interim).** A non-fused CAST node has no communication. Copy the
input's decomposition into the result's `array_meta` in the receive shape
pass, and pin both in the solver. Then each chare runs
`partition->allocate_or_reuse(region, gs, name, dt_out, decomp)` plus a
`parallel_for` conversion. That is a 16-case (src, dst) dispatch. Making it
fusible needs P1.

**RNG.** Use a counter-based generator, such as Philox or splitmix64 hashed
on `(seed, global linear index)`. Results are then independent of the
decomposition and the odf, and every chare fills its tile with no
communication: `chare_region_global(nd_idx)` gives the global coordinates.
Do not use `Kokkos_Random` pools: their state is per thread, so the stream
depends on the decomposition. A generic JIT op cannot do this yet, because
kernels receive no tile origin, only memref descriptors and broadcast scalars
(`dispatch.hpp:141-180`). Adding the global offset as extra scalar arguments
would allow `linalg.index`-based `iota` and RNG inside fused kernels (M,
since it changes the JIT ABI). The executor route is simpler. The random
streams will not match NumPy's MT19937, which is fine because the ports
verify against fetched inputs.

**Reductions.** REDUCE today is a 1-D dot product only. Each chare contributes
a 24-byte `ReduceContrib`, the `reduce_dot_sum` reducer sums it, and the
result is a (1,) array on 1-D chare 0. A 2-D sum along axis 1 is the same
communication as matvec without the vector broadcast:
`cross_matmul_send_result_2d_to_1d` (`backend_internal.hpp:261`) plus the
1-D assembly in `execute_node.cpp:116-586`. So adding an op-kind parameter to
the matvec executor gives `sum/max/min(axis=1)` at about M. `argmin/argmax`
need (value, int64 index) pairs and an int64 output array. axis=0 is
vecmat-shaped.

**Transpose.** All dims of an array share one tile size, and the
`Partition2D` grid is shared by every 2-D array. A transposed array therefore
lives on the mirrored chares. The executor packs its tile, sends it to
`(j,i)` with `receive_data`, and the receiver writes the transposed block.
Use `Kokkos::parallel_for` with a 2-D MDRange on GPU; Eigen's `.transpose()`
works on CPU. When the array's decomposition offset is nonzero, the
permutation is no longer tile-aligned, so pin to zero offset
(`decomposition_solver.hpp:590-595`). Transposed-operand SUMMA (`A^T B`)
avoids the copy and is the better long-term target. It is also what
`tssvd`/`svd`/`qr` Gram products need.

**Factorizations.** The 5 HOST-FALLBACK examples already do the bulk work
natively (SUMMA). Only the bs×bs or n×n kernels go to the host. P3 removes
the host round trips without any distributed linear algebra. A truly
distributed potrf/getrf/geqrf/syevd is a ScaLAPACK or SLATE-sized effort on
top of the Charm++ tile model (XL). tssvd (a cuPyNumeric-specific API) is a
TSQR reduction tree of local QRs followed by a small SVD. Its tree maps well
onto a Charm++ reduction with a custom "stack-two-R-and-QR" reducer, which is
L once P3 exists.

**Gather/scatter.** Every existing executor plans communication
deterministically from regions and decompositions known when the DAG arrives
(`executor_core.cpp` send planning). A device-resident index array breaks
that: each chare must first learn which remote elements it needs, which
takes a request round, then a reply round, with message counts known only at
runtime. Charm++ handles this well, but it is a new executor shape: two
phases plus a quiescence or count-based completion. Hence XL for the general
case. When indices are known on the host, which covers every use in the
ports' defaults, a static plan is possible. Send them as SCALAR operands, or
as an INLINE_DATA payload after P4.

**Sort.** On GPU builds the 1-D minimum tile is 1,048,576 elements, so sort's
default of 1e6 and quantiles' default of 1000 both fit in one tile. A
single-tile `Kokkos::sort` / `Kokkos::Experimental::sort_by_key` op (M)
therefore makes both examples native at default sizes. Distributed sample
sort is needed only for scaling.

**Scan.** The single-tile version is S-M and covers scan's default (n=10).
The distributed 1-D version adds a small chain or tree over tile totals,
which is cheap. Most of the work is the 3-phase executor and nodes for the
axis variants.

**Convolve.** The existing fragment path already brings remote halos in for
shifted views. A CONVOLVE op can reuse that region machinery for the input:
it is a "stencil with a weights array". The filter can be broadcast as an
INLINE_DATA payload (P4) or fetched through the existing broadcast mechanism.

---

## 5. Corrections and additions to README's ranked list

Items used by the originals (including non-default and check paths) that the
README list misses or under-reports:

- **hstack/concatenate** (linreg, logreg). It appears in the feasibility table
  but not in the ranked list. It is a frontend-only quick win (Q7).
- **`out=`** is used by einsum (`--cupy-compatibility` off, the default
  path) and scan (`getattr(np, OP)(A, out=B, axis=ax)`), not only gemm.
  Frontend only.
- **Full reductions on the default (no-check) paths:**
  - jacobi: `math.isnan(np.sum(x))`, plus `count_nonzero(isnan/isinf)` in
    the assertion message,
  - logreg: same pattern on `weights`,
  - indexing_routines: warm-up `np.sum(zeros.dot(zeros))`,
  - take_pairs: `result.sum()`,
  - kmeans: `np.sum(np.amin(...))`, and `np.sum(np.not_equal(...))` every
    S iterations (bool sum).
- **Check-path ops:**
  - qr: `np.max(qtq_err)` and `q.T`,
  - solve: `np.linalg.norm(res, numpy.inf)`, i.e. max|·|,
  - svd/tssvd: `np.matmul(u * s, vh)`, which needs general broadcasting
    (m,k)*(k,).
- **FFT**: richardson_lucy `--conv-method fft` (`auto` may pick FFT in
  cuPyNumeric).
- **Unsigned dtypes are defaults**: uint32 is the *default* for both sort and
  quantiles; kmeans divides by `uint64` counts.
- **bool dtype**: indexing_routines `(B % 2).astype(bool)`; kmeans
  `np.not_equal`.
- **Mixed int/float promotion**: black_scholes_greeks computes
  `(temp_arr % n_money_steps) * money_step` (int64 × float → float64 in
  NumPy; charmnumeric keeps int64 because scalars never promote). The JIT
  likely miscompiles array-array int/float mixes (see the last row of §3).
- **indexing_routines extras:**
  - `choose(..., mode="wrap")`,
  - `repeat(A2, R, axis=1)` with an array `R` of counts,
  - zipped boolean indexing `A2[indx_bool, indx_bool] = 11` (a pointwise
    pair gather, not a 2-D mask),
  - mixed advanced indexing `A3[indx, :, indx] = 15` (NumPy moves the
    advanced dim first).
- **take_pairs**: `take(indices, axis=1)` with a 2-D index array gives a
  **3-D** (M, P, 2) result.
- **quantiles**: all 13 `--method` values. `nearest` needs round-half-even;
  `lower/higher/midpoint/inverted_cdf` need floor/ceil (Q1).
- **kmeans**: `np.linalg.norm(x, ord=2, axis=1)` (axis reduction + sqrt).
- **RNG API surface**: `random.seed`, `randint` with full
  `iinfo(int64)` bounds (needs exact int64 scalars), `uniform`.
- **Cross-dim set_region at an offset** (kmeans column assignment, lstm_*): a
  bug rather than a missing op (Q8). The README's line reference
  (`backend_internal.hpp:345-351`) is stale; the code is now at `:357-420`,
  with the faulty mapping at `:384-388`. Reading confirms the bug.
- **Not needed**:
  - cg's `generate_random` (uses `A.T`) is dead code; the call is commented
    out (`cg.py:112,204`),
  - `stencil_hint` is Legate-only,
  - in-place view updates (`V[:, i, :] += c`, `dC[t] += ...`) already work
    through `__getitem__`/`__setitem__`.

---

## 6. Suggested order (value per effort)

1. **Q1-Q6 + Q8** (about 3-4 days total). This removes every tanh-step,
   Newton and Halley emulation, makes sort/quantiles exact, and fixes the
   cross-dim set_region bug that blocks the natural kmeans/lstm code.
2. **Native RNG + iota, P4a bulk upload, `max_name`** (about 1 week). Real
   inputs, no 65k-name ceiling, and arange/reshape-free index grids.
3. **P2 full reductions, then 2-D axis reductions via the matvec path**
   (about 1 week). Covers cfd, jacobi, linreg, logreg, kmeans and
   lstm_backward without dot-with-ones tricks; adds argmin/argmax.
4. **2-D transpose + vecmat** (about 1 week). Unblocks 10 examples' natural
   code.
5. **CAST executor (astype)**, then **P3 local-op framework** with
   Eigen/cuSOLVER factorizations, sort and scan (about 2-3 weeks). Every
   HOST-FALLBACK becomes device-resident at default sizes.
6. Then the L/XL items, as research needs dictate: P1 typed kernels with
   bool/uint/complex/f16, general broadcasting, reshape, distributed
   sort/scan/gather, and distributed factorizations.
