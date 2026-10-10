# What it would take to support the missing ops natively

Companion to `README.md` (same directory). For every op or op family that the
cuPyNumeric ports emulate, run on the host, or cannot express, this file
estimates the cost of a native charmnumeric implementation. Each estimate is
grounded in the current code.

**Status (2026-10-09):** the ports are now the unmodified cuPyNumeric programs
and everything they use is native (`README.md`, "Library changes"); this file
remains the design record and the cost estimates for what is still open
(distributed sort/scan/factorizations, a fused transposed-operand SUMMA, public
`cnp.fft`).

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

- ~~**One element type per kernel.**~~ Fixed 2026-10-09 (P1). Historical: `buildFromAST` picked a single `elemType`
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
- ~~**No integer unary bodies for float-only math.**~~ Int inputs to
  sqrt/log*/exp*/sin/cos/tanh compute in float64 in-kernel (NumPy's loop dtypes).
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
   - a point-to-point send of a `ReduceContrib` payload to chare 0
     (`backend_internal.hpp`; not a Charm++ reduction, see the comment in
     `executor_reduce.cpp`),
   - the partials land in `PartitionImpl::reduce_result`
     (`partition_comm.cpp`).

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
  isnan/isinf/isfinite, logical_and/or/xor/not (`& | ^ ~` are bitwise, i.e.
  logical on bool). greater* reuse LESS/LESS_EQUAL with swapped operands.
  Results are bool. An int array compared with a non-integral or
  out-of-range scalar is rewritten exactly (x < 2.5 -> x < 3; NaN/out-of-range
  give constant results).
- Q6: where (cond != 0, NaN truthy) and clip as real 3-input fused ops.
- Restrictions: general broadcasting is materialized first (BROADCAST, opcode 141). Mask assignment `A[mask] = s` is
  `A[...] = cnp.where(mask, s, A)`.

| # | op(s) | what to do | examples helped |
|---|---|---|---|
| Q1 | `sqrt, log, abs, floor, ceil, round` (half-even), `log1p, expm1, rsqrt, sin, cos` | Recipe A, unary. Use `math::SqrtOp, LogOp, AbsFOp` (int: `AbsIOp`), `FloorOp, CeilOp, RoundEvenOp, Log1pOp, ExpM1Op, RsqrtOp, SinOp, CosOp`. Also apply the one-time scalar-input fix at `jit.cpp:375`. Float only until astype exists. | black_scholes, black_scholes_greeks, logreg (log, floor), tssvd (abs), quantiles (floor/ceil/round methods), solve check (abs for inf-norm) |
| Q2 | `power` / `**` | Recipe A, binary: `math::PowFOp` (int: `math::IPowIOp`). The frontend should also rewrite a literal `**2` to `x*x`. | cfd, linreg, lstm_backward, kmeans (`square`) |
| Q3 | `maximum, minimum` | Recipe A, binary: `arith::MaximumFOp`/`MinimumFOp` (NaN-propagating, like NumPy), `MaxSIOp`/`MinSIOp`. | kmeans, kmeans_slow. Also makes the sort/quantiles compare-exchange network exact (`lo=minimum(a,b)`, `hi=maximum(a,b)`) and removes the tanh-step's 2e-9 resolution limit. |
| Q4 | `mod` (`%`), `floor_divide` (`//`) | Recipe A, binary: `RemFOp`/`RemSIOp`, plus a sign fix-up so that `sign(r) == sign(b)` (cmp+select inside the body lambda). `floor_divide` is `DivSI` with a floor correction. | black_scholes_greeks, indexing_routines |
| Q5 | comparisons `< <= > >= == !=`, `isnan`, `isinf` as masks | Recipe A, binary/unary: `arith::CmpFOp`/`CmpIOp` (`isnan` = `CmpF UNO(x,x)`, `isinf` = `abs(x)==inf`). The result is bool (as built; the first design returned 1.0/0.0 in the input dtype). | black_scholes(+greeks) `d>0`, kmeans `not_equal`, kmeans_slow `labels==idx`, sort, quantiles, scan nan* (`isnan`), jacobi/logreg failure-path diagnostics |
| Q6 | `where(c, a, b)` | Interim, FE only (S): `b + c*(a-b)`. Exact when the unselected branch is finite. Proper version (S-M, about 1 day): a `TernaryOpEntry`, a 3-input `linalg.generic` next to `emitGenericArrayOp`, an `emitOp` case, and `is_elementwise` extended. With masks from Q5 this also gives `choose` (nested where) and **mask assignment** `A[mask] = s` → `A[...] = where(mask, s, A)`. | black_scholes(+greeks), kmeans_slow, scan nan*, indexing_routines (choose, ai1/ai2/ai3 mask assignments) |
| Q7 | FE-only API gaps | See the list below. | see list |
| Q8 | cross-dim `set_region` at a nonzero offset (a **bug**, not an op) | `cross_set_region_send` maps source coordinates straight to target coordinates (`tgt_start[td] = src_local_start[sd]`, `E/src/backend_internal.hpp:384-388`). It ignores `tgt_region->start[td]` in non-singleton dims, and it sends the whole source rather than the view region. Add the target-region start and clip to the source view. S-M. (README cites `:345-351`; that reference is stale.) | kmeans `centroids[:, idx] = bincount(...)`, lstm_forward `Hin[t,:,word_size:] = ...`, lstm_backward `dh0[0] += np.sum(...)` |

Q7, the frontend-only API gaps:

- **`hstack`/`vstack`/`concatenate`**: `create_array` plus slice `set_region`
  per piece. Needed by linreg and logreg. **DONE** (`operations.py`
  `concatenate/hstack/vstack/stack`, results up to 3-D; linreg/logreg use it).
- **`out=`**: `out[...] = result`. Needed by gemm, einsum and scan. **DONE**
  for add/subtract/multiply/divide/tanh/exp/matmul/dot/`Array.dot`
  (`charmnumeric.py` `_write_out`) and einsum; gemm and einsum use it. The
  native scans take `out=` too (scan.py uses `cumsum(A, out=B, axis=ax)`).
- **`eye`**: build it as `diag(ones(n))`. **DONE** (one DIAG op; non-square
  via a block copy).
- **Name limit**: raise `max_name` (`charmtyles/core.py:9`). Nothing in the
  backend depends on 2^16. **DONE**: int32 limit. Names are not recycled: the
  backend fills `array_meta`/`live_array_meta` by name when a DAG arrives,
  before older epochs that may still use a dead name finish, so reuse could
  clobber in-flight metadata. Note `array_meta` entries are never erased
  (about 80 B per name per PE).
- **Views passed to REDUCE/TILE/DIAG**: auto-materialize them, because those
  executors ignore view regions (README issue 5). **DONE.**
- **`sum()` of 1-D/2-D arrays**: use REDUCE or matvec with `ones`, as
  the old helpers did. **DONE**: native `cnp.sum` (REDUCTION opcode 60).
- **Uniform-count `repeat`**: strided slices.
- **`take` with host-known indices**: lower to column-slice copies, 496 for
  take_pairs' N=32. Cheaper and exact compared with the 0/1 selection-matrix
  SUMMA. Superseded: `cnp.take` / `Array.take` are a native GATHER
  (`indexing.py`) for host and device indices; take_pairs uses it.
- ~~**General broadcasting**~~: native (BROADCAST, see section 4).

---

## 2. Cross-cutting prerequisites

These unlock several rows of the main table at once.

| id | prerequisite | effort | what changes |
|---|---|---|---|
| P1 | **Per-operand dtypes in fused kernels** | **DONE** (2026-10-09) | WIRE: FRONTEND/BROADCAST_OBJECT carry the operand dtype, which is part of the node identifier. JIT: every memref, broadcast scalar and output has its own element type; each op converts its operands to its compute dtype in-kernel (`convertValue`, `emitOp` in `jit.cpp`); scalars travel as f64. EXEC: the JIT path of `execute_node_nd` is dtype-agnostic (outputs allocated per root dtype, `align_fragments_mixed` for mixed element sizes, remote inputs sent in the array's own dtype). The frontend no longer splits fusion on dtype changes. |
| P1b | **New DTypes** (bool, int8/16, uint8/16/32/64, f16, c64/c128) | **DONE** (2026-10-09) | `DType` in `ast.hpp` (codes 4-13); `src/dtypes.hpp` (C++ types, `cn_dispatch`, `cn_cast`, NumPy sort order) and `src/dtype_info.hpp` (`dtype_promote`). JIT: f16 math in f32, bool as i8 0/1, unsigned bodies (`uint_body`), complex dialect (`complex_body`, lowered by complex-to-standard). Executors dispatch over all dtypes (BLAS-backed ones over f32/f64/i32/i64/c64/c128; the frontend computes other dtypes in one of those). |
| P2 | **Generic full-array reduction** (any N, op kind ∈ {sum, prod, max, min, argmax, argmin, count_nonzero, any, all}) | M | Generalize REDUCE: relax N==1 (`executor_reduce.cpp:9-12`, `executor_core.cpp:255-259`; all chares must stay relevant); add `op_kind` and an `int64 index` to `ReduceContrib`; one reducer per kind (`executor_reducer.cpp`); local partials via `Kokkos::parallel_reduce` with `Sum/Max/Min/MaxLoc/MinLoc` (CPU: loops or Eigen). The callback still targets 1-D chare 0 (`executor_reduce.cpp:108-110`). |
| P3 | **DONE for dense linear algebra (LINALG, opcode 100; one chare, not scalable)** — **"Local op" framework**: run a dense kernel on one chare, with gather→compute→scatter when the operand spans more than one tile | L (framework), then S-M per kernel | Comm: all-to-one gather and one-to-all scatter of tiles (like the `get` path: `partition_get.cpp:4-66`, `dag_group_runtime.cpp:55-116`). On GPU builds the minimum tiles are 1024² for 2-D, 1M for 1-D and 128³ for 3-D (`backend.hpp:10-17`). So most default example sizes already sit on one chare, and the gather is a no-op. Exceptions: take_pairs 10000×32; on CPU builds also qr 1000² and sort 1e6. It is not scalable, but it turns every remaining HOST-FALLBACK into a device-resident op with no host round trip. |
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
| maximum / minimum | **DONE** (Q3) | OPC, JIT, FE | none | none | kmeans, kmeans_slow | native |
| mod, floor_divide | **DONE** (Q4) | OPC, JIT, FE | none | NumPy sign semantics | black_scholes_greeks, indexing_routines | native |
| comparisons, isnan, isinf | **DONE**: bool results, fused with their consumers | OPC, JIT, FE | none | complex `<` is lexicographic, as NumPy | black_scholes(+greeks), kmeans(+slow), sort, quantiles, scan nan*, indexing_routines, jacobi/logreg diagnostics | — |
| where / select / choose / mask-assign | **DONE** (Q6, ternary JIT op) | OPC, JIT (`TernaryOpEntry`), FE | none (halo for shifted views, as for any elementwise op) | none: cond is any dtype (bool or nonzero), x/y promote as NumPy | black_scholes(+greeks), kmeans_slow, scan nan*, indexing_routines (choose, `A[mask] = s`) | native; `cnp.choose` is a fused where chain (a gather when the choices' dtype differs from the index dtype) |
| astype | **DONE**: fused CONVERT (opcode 160) between all 14 dtypes, NumPy semantics (x->bool is x != 0, complex->real takes the real part) | OPC, JIT, FE | none | float->int out of range is platform-defined, as in NumPy | quantiles, svd, black_scholes_greeks, kmeans, indexing_routines | — |
| native RNG: rand, random, randn, uniform, normal, randint, seed, default_rng; arange/linspace/indices/meshgrid/eye — **DONE** | M | OPC (51 RANDOM, 52 IOTA), FE (`random.py`, `charmnumeric.py`), RECV/SOLV (treated like CREATE: `is_creation()`), EXEC (`executor_creation.cpp`, kernels in `creation_kernels.hpp`) | none | randint takes exact int64 bounds (parameters travel as hi/lo 32-bit scalar pairs) | cholesky, qr, solve, svd, tssvd, take_pairs, jacobi, gemm, black_scholes, scan switched; the rest still use `_cnum_util` generators | Philox4x32-10 keyed by (seed, stream) on the global element index: independent of decomposition/PEs; CPU and GPU builds bit-identical (no FMA, software log/cos). Streams differ from NumPy's. |
| bulk host→device upload — **DONE (P4a)** | M | WIRE, OPC, EXEC, FE | none | none | cholesky, qr, solve, svd (host factors); removes the 65k-element cap | see P4 |
| full reductions: sum, prod, max, min, argmax/argmin (flat), count_nonzero, any/all, norm(inf/2) | **DONE** (REDUCTION opcode 60, `E/src/executor_reduction.cpp`, `E/charmnumeric/reductions.py`; also mean/var/std, views, 1-3-D, all dtypes). Was: **M** (P2) | OPC, FE, RECV, EXEC (`executor_reduce` generalized), reducer | reduction tree (Charm++ `contribute`) | argmax returns int64; result is a (1,) array on 1-D chare 0 | cfd, jacobi, linreg, logreg, kmeans (distance sum, `sum(changes)`), tssvd, take_pairs, indexing_routines warmup, qr check (`max`), solve check (`norm(·,inf)`) | dot/matvec with `ones` (sum only) |
| axis reductions: sum, mean, amin, amax, argmin, argmax, norm along one axis | **DONE** (REDUCTION opcode 60 with an axis mask: any axis or axis tuple of 1-3-D operands, keepdims; partials go point-to-point to the owning output chares). Was: **M** → **L** | OPC, FE, RECV, SOLV (pin), EXEC, KK | partial vectors reduced along one tile-grid dimension to the owning 1-D chares (exactly matvec's `cross_matmul_send_result_2d_to_1d`, `backend_internal.hpp:261`) | arg* output is int64, so the output dtype differs from the input; (value, index) pairs on the wire | kmeans, kmeans_slow (norm axis=1, argmin, amin, sum axis=0), lstm_backward (sum axis 0), tssvd (row sums), solve (pivot argmax), quantiles(axis) | sum = `A @ ones` (exact); argmin/amin need the Q3/Q5 one-hot trick |
| bincount (weights, minlength) | **DONE** (BINCOUNT opcode 61; length = minlength or max(x)+1 via `interface=`). Was: **M** after P2 | OPC, FE, EXEC, reducer (vector-valued) | local histogram, then an all-reduce of a length-C vector (reduction tree) | int labels with float weights: mixed dtype inside one op (fine in an executor) | kmeans, kmeans_slow | one-hot M with `ones @ M`, `M.T @ data` |
| transpose (2-D, materialized) | **DONE** (opcode TRANSPOSE=70) | FE (`.T`, `transpose()`, `swapaxes`), RECV (shape + grid expansion incl. offset), SOLV (result takes the source tile and the permuted source offset+view start, fixed-offset source), EXEC `executor_transpose.cpp` (Kokkos pack/place on GPU, memcpy loops on CPU) | **point-to-point**: each source chare packs its part of the (possibly strided/offset) view in result order, once per overlapping result chare; with the solver alignment that is one message per tile | none | kmeans, kmeans_slow, linreg, logreg, lstm_backward, einsum, cholesky, qr, svd, tssvd (all switched) | — |
| vecmat (1-D @ 2-D) and transposed-operand matmul (`A.T @ B`, `A @ B.T`) | **DONE** as materialized transpose + matvec/SUMMA (vecmat caches A.T until A is written) → **L** (SUMMA variants without materializing) | FE (`_array_matmul`), later EXEC (`executor_matmul.cpp` / `executor_matmatmul.cpp` flag) | vecmat: transpose + matvec today. Fused: vector slices multicast along grid rows, partials reduced along grid columns; SUMMA-T: panel broadcast along the other grid axis. | none | linreg, logreg (`err.dot(X)`), lstm_backward, qr/svd (`A^T A`, `Q^T Q`) | — |
| 3-D axis permutation (transpose for N=3) | **DONE** (same executor, any axes permutation) | as for 2-D, rank 3 | point-to-point | none | einsum (batched forms use `transpose(x, (0, 2, 1))`) | — |
| dense factorizations: potrf, getrf+getrs, geqrf, syevd, gesvd, tssvd | **DONE (P3 interim; factors on one chare, not scalable)**: `cnp.linalg.cholesky/qr/solve/inv/det/slogdet/eigh/eigvalsh/svd/svdvals/lstsq/solve_triangular/tssvd` (`E/charmnumeric/linalg.py`, `E/src/executor_linalg.cpp`; Eigen on CPU, cuSOLVER/cuBLAS on NVIDIA). Next: **XL** (distributed blocked: ScaLAPACK-style panels, pivot search across chares, TSQR trees, two-stage eigensolvers) | OPC, FE, EXEC, cuSOLVER link, CMake | gather of the operand blocks to chare 0 of the result partition, factor there, scatter the result tiles (point-to-point, host-staged). **Limits:** the whole problem must fit in one chare's (PE's / GPU's) memory, the factorization runs serially on that one PE/GPU while the others wait, and all operand/result data crosses the network to/from that chare once. Distributed: panel broadcasts plus column-max reductions (getrf) and binary reduction trees (TSQR for tssvd). | s/d (int inputs factored in float64) | cholesky, qr, solve, svd, tssvd (all direct now) | — |
| trsm / trtri | **DONE on P3 (one chare, not scalable)**: `cnp.linalg.solve_triangular` (scipy/cuPyNumeric signature), `inv` → **L** (distributed wavefront) | OPC, FE, EXEC | wavefront along the tile diagonal | none | cholesky, solve, qr | — |
| gather / take / fancy indexing / row permutation from a **device** index array | **DONE** (GATHER 120 / SCATTER 121, `E/src/executor_indexing.cpp`, `E/charmnumeric/indexing.py`: `a[idx]`, `a[idx] = v`, multi-axis keys such as `a[i, :, j]`, take/put/take_along_axis/put_along_axis; host or device integer indices, views, negative and repeated indices). Was: **XL** | OPC, FE, RECV, SOLV (pin result), EXEC | **data-dependent all-to-all**: the frontend linearizes the key into one array of flat offsets; the chares holding it send request lists to every chare of the source, which send value lists to every chare of the target (fixed message counts give completion) | device index bounds are not checked (out-of-range offsets are dropped); device index arrays must have equal shapes; results ≤ 3-D | indexing_routines (ai1/ai2/ai3), take_pairs; solve (device pivots) not switched | — |
| boolean-mask **read** `A[mask]` (compaction), variable-count `repeat`, `nonzero` | **DONE** (EXPAND 122: nonzero/flatnonzero/argwhere, `A[mask]`, `A[mask] = array`, `A[b0, b1]`, compress, repeat with array counts; `A[mask] = scalar` stays an elementwise where). Was: **L-XL** | OPC, FE, RECV, EXEC | the size comes from a `count_nonzero`/`sum` get; in the node, every chare sends its row-segment counts to one coordinator chare, which returns each segment's exclusive prefix (output position); positions then go point-to-point to the output chares | masks are bool arrays; the scan is centralized | indexing_routines (`A2[b, b]`, `repeat(A2, R, axis=1)`, `A[mask] = s`) | — |
| reshape / flatten / ravel / squeeze / expand_dims / atleast_*d (1-3 D) | **DONE**: RESHAPE (opcode 140, `E/src/executor_shape.cpp`); `cnp.reshape/ravel/squeeze/expand_dims/atleast_1d/2d/3d`, `Array.reshape/ravel/flatten/squeeze`, C order, `-1`, views as sources; results are new arrays (a same-shape reshape returns the input) | OPC, FE, RECV, SOLV, EXEC | each source chare splits the rows of its block (view-index box) into runs that stay inside one result row and tile, merges runs that are contiguous in the result tile, and sends one message per result chare (`receive_shape`, host-staged on GPU); a result chare finishes when the runs cover its tile. Size-1-dim-only changes (squeeze/expand_dims/newaxis) use the cross-ndim set_region (device pack on GPU) | none | black_scholes_greeks, take_pairs, scan/sort/take/concatenate (axis=None), quantiles (axis=None, input reshape) | — |
| more than 3 dims | **DONE (frontend folding)**: `FoldedArray` (`E/charmnumeric/folded.py`): an N-D array (N > 3) is stored as the 3-D array (s0*...*s[N-3], s[N-2], s[N-1]). Creation, get, elementwise ops/operators with broadcasting, basic indexing (views for rectangular selections of the storage, copies otherwise) and assignment, reductions, reshape/ravel/squeeze/expand_dims, astype, scans/sorts along the last two axes or axis=None, stack/concatenate/take results, transpose (grouped 3-D transposes), einsum/tensordot. **Limits:** matmul/linalg/tile of >3-D arrays raise; broadcasting whose pattern changes inside the folded dims and reductions over leading dims reshape first (extra data movement); var/std/argmax/argmin over axes forming >3 groups of adjacent dims raise | FE only (RESHAPE/BROADCAST in the backend) | — | none | black_scholes_greeks (4-D CALL/PUT), scan/quantiles with >=4-D `--shape` | a native Partition4D: L-XL |
| sort / argsort | **DONE (not scalable)**: SORT/ARGSORT executors (opcodes 82/83) gather the whole array on the chare holding element 0, sort every line along any axis there (`std::sort` with a NaN-last, index-tie-broken comparator on CPU; `Kokkos::sort` with the same comparator, i.e. thrust, on GPU), and scatter the tiles back. Stable (= `kind='stable'`). → **L-XL** (distributed sample sort) | OPC, FE, EXEC, KK | today: all-to-one + one-to-all. Sample sort: gather samples, broadcast splitters, **data-dependent all-to-all**, local merge, then rebalance to block tiling (a second all-to-all) | argsort returns int64; uint32 is the default dtype (P1b) | sort, quantiles | — |
| scan: cumsum, cumprod, nancumsum, nancumprod | **DONE** (distributed, any axis of 1-D/2-D/3-D): SCAN executor (opcode 81) | OPC, FE, EXEC, KK | per-tile scan along the axis (GPU: one thread per line, or a `Kokkos::parallel_scan` per line when lines are few and long), each tile sends its per-line totals to every later tile of its grid line (P(P-1)/2 cross-section messages per line, one round), receivers fold them into a carry and add it | nan* treat NaN as the identity inside the executor (no isnan/where needed); int32 accumulates in int64 as in NumPy (implicit astype first) | scan | — |
| convolve, N-D (1-3-D), modes full/same/valid, direct | **DONE** (CONVOLVE opcode 150, `E/src/executor_signal.cpp`, `E/charmnumeric/signal.py`) | OPC, FE, RECV, SOLV (result pinned to offset 0), EXEC, KK | **halo exchange**: every result chare gets the input window around its tile (halo f-1, split over however many source tiles it spans, filters larger than a tile included) and the whole filter, point-to-point; then a local direct sum (Kokkos kernel on GPU) | ints direct only; computes in float32/float64/int32/int64, result `result_type(a, v)` | richardson_lucy | — |
| convolve via FFT (`method="fft"`, or `auto` for 4096+ taps) | **DONE** per chare (overlap-save on the same halo window: host mixed-radix FFT in double, cuFFT R2C/C2R on NVIDIA) → **XL** for one global FFT (distributed transposes) | EXEC, cuFFT link | same halo exchange as direct | real inputs only (complex convolve raises) | richardson_lucy (`--conv-method fft`) | — |
| flip / flipud / fliplr, negative-step slicing | **DONE** as copies (FLIP opcode 151; `a[::-k]` = flip of the positive-step view; `a[::-1] = x` writes the flipped rhs) | FE, RECV, SOLV, EXEC | point-to-point mirror (each result tile from the ≤2 source tiles per axis holding its mirror) | none | richardson_lucy (`np.flip(psf)`) | — |
| general broadcasting, newaxis | **DONE**: NumPy broadcasting in every elementwise op, comparison, where/clip and in assignment / `out=` / in-place operators (`+=`, `/=`, ...); `cnp.broadcast_to/broadcast_arrays/broadcast_shapes`; `x[:, newaxis]` | OPC, FE, RECV, SOLV, EXEC | BROADCAST (opcode 141, `E/src/executor_shape.cpp`) materializes each non-size-1 operand of a different shape: every source chare sends each result chare the part of its block the result tile reads (all of it along repeated dims), the result chare assembles that small block and expands it into its tile (Kokkos kernel on GPU). Communication is the minimum (each result chare receives its projected block once); the cost is one extra result-sized write. Chosen over zero-stride memrefs in the JIT because it needs no JIT, solver or remote-fetch changes (those assume equal-rank, positive-step regions). Size-1 operands stay backend scalars | none | kmeans, kmeans_slow (`[:, newaxis]`, `/= counts[:, newaxis]`, 3-way where), svd/tssvd check (`u * s`) | — |
| einsum, einsum_path, tensordot, inner, outer | **DONE (frontend lowering, `E/charmnumeric/einsum.py`)**: full `numpy.einsum` semantics (explicit/implicit output, `...` with broadcasting, size-1 broadcasting, repeated labels = traces/diagonals, single-operand sums, outer products, any number of operands contracted pairwise in `numpy.einsum_path` order (greedy default; `optimize=False` is left to right; explicit paths accepted), sublist form, scalars, `out=`, `dtype=`/`casting=`); `tensordot` (int and pair `axes`), `inner`, `outer`. **Limits:** results are new arrays (no views), scalar results are (1,) arrays, per-batch SUMMA loop for large batched matmuls | FE only: diagonals = DIAG (2-D) or permute + RESHAPE + strided view; sums = REDUCTION; pairs = TRANSPOSE/RESHAPE into (batch, free, contracted) blocks (orientation/batch position chosen to avoid transposes), then broadcast multiply (+ REDUCTION) when nothing is contracted or one side has no free labels, native dot/matvec for unbatched vectors, SUMMA for matrices, one SUMMA per batch element (3-D singleton views, written into the 3-D result) for batched matrices unless batch*M*N*K is small (then a broadcast multiply + sum); >3-D transposes are grouped 3-D transposes (`einsum.permute`, also `cnp.transpose` of FoldedArrays) | inherits from the lowered ops; broadcast intermediates are only materialized when they are at most 4x the operand + result sizes (or < 2^24 elements for > 64 batches) | none | einsum (the original program, unmodified) | — |
| batched matmul (3-D) | **DONE in the frontend**: N-D `@`/`cnp.matmul` (NumPy batch broadcasting, FoldedArrays) go through einsum; 3-D operands with one size-1 dim keep the old dim-dropped SUMMA/matvec. A native op is still **M-L** (the lowering runs one SUMMA per batch element, ~1 ms per element on CPU) | EXEC (`executor_matmatmul.cpp` already has a 3-D dim-dropped path) | local `KokkosBatched` gemm when the contracted dims fit in a tile; otherwise per-batch SUMMA | none | einsum (`kac,kbc->kab`), lstm | loop of 2-D SUMMA calls (einsum does this) |
| complex dtypes | **DONE** (2026-10-09): elementwise (complex dialect), reductions (complex accumulator; var/std/norms via real/imag/abs), scans, sort (NumPy order), matmul/dot (no conjugation), linalg (svd, solve, inv, det, qr, cholesky, eigh/eigvalsh on Eigen and cuSOLVER C/Z) | — | none | lstsq/slogdet/tssvd stay real-only | scan, sort, svd, einsum, cholesky, qr, solve | — |
| float16 | **DONE** (2026-10-09): stored as half (`_Float16` / Kokkos `half_t`), math computed in f32 and rounded per op (as NumPy); matmul computes in f32 | — | none | — | gemm, black_scholes, linreg, logreg, kmeans, einsum | — |
| unsigned ints, int8/16, bool | **DONE** (2026-10-09): unsigned compare/div/rem/max/min/pow/convert bodies; reductions accumulate uint64 in uint64; matmul computes small and unsigned ints in int64 (exact modulo 2^k) | — | none | — | sort and quantiles (uint32 default), kmeans | — |
| exact int64 scalars | **S-M** | WIRE: a new `SCALAR_I64` operand (today every scalar is a double: `ast.py:105-107`, `dag.hpp:162-163`, `ast.hpp:38`) + JIT constant path (`jit.cpp:128`) | none | values above 2^53 are rounded today | sort/quantiles with `iinfo(int64)` bounds; randint | keep |values| < 2^53 |
| mixed-dtype elementwise promotion | **DONE** natively (P1): result dtypes from `ufunc.resolve_dtypes` (NEP 50 weak scalars), conversions in-kernel; int/int true division is float64 as in NumPy | FE, JIT | none | — | black_scholes_greeks, kmeans | — |

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

**astype (done).** A fused CONVERT node (opcode 160): each operand is
converted in-kernel to the op's compute dtype, so astype needs no
communication and fuses with its neighbours. The original design was a
separate CAST executor (opcode 80, since removed; 80 is unused).

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
a 32-byte `ReduceContrib`, sent to 1-D chare 0, which sums them; the
result is a (1,) array on that chare. A 2-D sum along axis 1 is the same
communication as matvec without the vector broadcast:
`cross_matmul_send_result_2d_to_1d` (`backend_internal.hpp:261`) plus the
1-D assembly in `execute_node.cpp:116-586`. So adding an op-kind parameter to
the matvec executor gives `sum/max/min(axis=1)` at about M. `argmin/argmax`
need (value, int64 index) pairs and an int64 output array. axis=0 is
vecmat-shaped.

**Transpose (done).** `src/executor_transpose.cpp`. Result axis k reads
source axis perm[k]. Every chare of the partition runs the node: a source
chare computes, per axis, the view-index range its tile holds (strided views
included), maps it to result coordinates, and sends one block per overlapping
result chare, packed in result row-major order (Kokkos `parallel_for` on the
comm stream + `device_pack_send` on GPU, strided memcpy loops on CPU). A
result chare counts the expected blocks with the same per-axis rule and
copies them in (`transpose_place`, from the `execute_node_nd` interpreter).
Source and result may have different tiles and offsets. The solver gives the
result the source's tile and offset = (source offset + view start) permuted,
mod tile, and treats it as a fixed-offset source, so a full or step-1 view
transpose is a pure tile permutation (tile (i,j) -> (j,i) for zero offsets);
matmul operands keep the zero-offset pin. Transposed-operand SUMMA (`A^T B`)
would avoid the copy; it is still the better long-term target for the
`tssvd`/`svd`/`qr` Gram products.

**Factorizations.** (Status 2026-10-09: done as single-chare LINALG ops, see the main
table; the text below is the original analysis.) The 5 HOST-FALLBACK examples already do the bulk work
natively (SUMMA). Only the bs×bs or n×n kernels go to the host. P3 removes
the host round trips without any distributed linear algebra. A truly
distributed potrf/getrf/geqrf/syevd is a ScaLAPACK or SLATE-sized effort on
top of the Charm++ tile model (XL). tssvd (a cuPyNumeric-specific API) is a
TSQR reduction tree of local QRs followed by a small SVD. Its tree maps well
onto a Charm++ reduction with a custom "stack-two-R-and-QR" reducer, which is
L once P3 exists.

**Gather/scatter (done).** `E/src/executor_indexing.cpp` uses a request
round and a reply round with count-based completion: every index chare sends
one (possibly empty) request list to every source chare, and every source
chare one value list to every target chare. Limits: the index chares do all
address arithmetic (a small index array with large slice dims, e.g. take_pairs'
(P, 2) indices with M = 10000 rows, runs on one chare), P^2 messages per node,
2-3 int64 words per element, and a centralized scan in EXPAND. Host-known
indices travel as an uploaded offset array through the same path.

**Sort (done, not scalable).** On GPU builds the 1-D minimum tile is
1,048,576 elements, so sort's default of 1e6 and quantiles' default of 1000
both fit in one tile anyway; the gather/scatter is then a no-op. Distributed
sample sort is needed only for scaling.

**Scan (done).** Distributed along any axis; see the main table.

**Convolve (done).** CONVOLVE and FLIP share one executor: a result chare
needs a box of each source view (window + filter, or the mirrored tile),
source chares send what they hold point-to-point (any decomposition, strided
views) and the result chare computes locally. The FFT path is per chare, so
it scales like the direct one; a global distributed FFT (and public
`cnp.fft`) still needs complex dtypes.

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
  - svd/tssvd: `np.matmul(u * s, vh)` (general broadcasting; native now).
- ~~**FFT**: richardson_lucy `--conv-method fft`~~ (native per-chare FFT
  convolution; public `cnp.fft.*` is missing).
- **Unsigned dtypes are defaults**: uint32 is the *default* for both sort and
  quantiles; kmeans divides by `uint64` counts.
- ~~**bool dtype**~~: native (kmeans `np.not_equal`, indexing_routines masks).
- ~~**Mixed int/float promotion**~~: native (P1); black_scholes_greeks'
  `(temp_arr % n_money_steps) * money_step` is int64 × float -> float64 as in NumPy.
- ~~**indexing_routines extras**~~ (all native now): `choose(..., mode="wrap")`,
  `repeat(A2, R, axis=1)` with an array `R`, zipped boolean indexing
  `A2[indx_bool, indx_bool] = 11`, mixed advanced indexing `A3[indx, :, indx]`.
- ~~**take_pairs**~~: `take(indices, axis=1)` with a 2-D index array (a 3-D
  (M, P, 2) result) is native.
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
4. ~~**2-D transpose + vecmat**~~ Done (plus 3-D permutations); the 10
   examples use their natural code.
5. ~~**astype**, then **P3 local-op framework** with
   Eigen/cuSOLVER factorizations, sort and scan~~ Done: fused CONVERT
   (opcode 160), SCAN/SORT (opcodes 81-83) and LINALG (opcode 100). Factorizations and sort run on
   one chare (not scalable); the factorization HOST-FALLBACKs are gone.
6. Then the L/XL items, as research needs dictate: P1 typed kernels with
   bool/uint/complex/f16, general broadcasting, reshape, distributed
   sort/scan/gather, and distributed factorizations.
