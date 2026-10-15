# cuPyNumeric examples on charmnumeric

Every top-level program of
[`cupynumeric/examples`](https://github.com/nv-legate/cupynumeric/tree/main/examples)
except `benchmark.py` (the old harness) is here, unchanged: each file is the
upstream file plus one 2-line comment above its `from _benchmark import ...`
line. `upstream/` holds the unmodified originals (fetched 2026-10-09 from
`main`, with the `_benchmark` package they import), so
`diff upstream/<name>.py <name>.py` shows only that comment. The programs run
with `np` = charmnumeric through `_benchmark.py`, a stand-in for cuPyNumeric's
`_benchmark` package, which also checks every run against NumPy.

## Running

```bash
. /u/atahilyani/cnum-gpu/wt/ports/env_ports.sh   # or the main tree's environment
cd example/charmnumeric/examples/cupynumeric

# against a running backend (server.out ++server ++server-port $CNUM_PORT)
CNUM_HOST=127.0.0.1 CNUM_PORT=1234 CNUM_ODF=4 python cg.py --check
CNUM_AUTO_FLUSH=0 python cfd.py             # no automatic async flushes (default: every 256 ops)

# without a backend: host-side reference simulator (static/semantic check)
CNUM_SIM=1 python kmeans.py
python _static_check.py [--quick] [name ...]   # all ports + variants, tabulated

# NumPy only (no backend, no comparison)
python cg.py --package numpy
```

The programs take their original flags and the harness flags of
cuPyNumeric's `parse_with_harness` (`-b/--benchmark`, `--package`, ...). The
last line printed is exactly `PASS <name>` or `FAIL <name>: <reasons>`
(`<name>` is the file name); the exit status is non-zero on FAIL.

## Regression flags

Flags the regression passes to each program; "(defaults)" means none. A
runner reads them from the code span in the second column
(`wt/ports/run_suite.sh` splits the row on backticks and takes field 4;
"(defaults)" yields nothing). All upstream defaults finish in well under 2
minutes on the CPU build at +p1 and +p2 (times: one `harness/run_one.sh`
run, server start included), so no program needs smaller sizes.

| program | flags | +p1 | +p2 |
|---|---|---|---|
| `black_scholes` | (defaults) | PASS 2 s | PASS 2 s |
| `black_scholes_greeks` | (defaults) | PASS 26 s | PASS 34 s |
| `cfd` | (defaults) | PASS 70 s | PASS 74 s |
| `cg` | (defaults) | PASS 2 s | PASS 2 s |
| `cholesky` | (defaults) | PASS 2 s | PASS 2 s |
| `einsum` | (defaults) | PASS 4 s | PASS 5 s |
| `gemm` | (defaults) | PASS 2 s | PASS 3 s |
| `indexing_routines` | (defaults) | PASS 20 s | PASS 20 s |
| `jacobi` | (defaults) | PASS 3 s | PASS 3 s |
| `kmeans` | (defaults) | PASS 3 s | PASS 4 s |
| `kmeans_slow` | (defaults) | PASS 5 s | PASS 5 s |
| `linreg` | (defaults) | PASS 3 s | PASS 3 s |
| `logreg` | (defaults) | PASS 4 s | PASS 5 s |
| `lstm_backward` | (defaults) | PASS 2 s | PASS 3 s |
| `lstm_forward` | (defaults) | PASS 2 s | PASS 2 s |
| `qr` | (defaults) | PASS 3 s | PASS 3 s |
| `quantiles` | (defaults) | PASS 2 s | PASS 2 s |
| `richardson_lucy` | (defaults) | PASS 2 s | PASS 3 s |
| `scan` | (defaults) | PASS 2 s | PASS 3 s |
| `solve` | (defaults) | PASS 2 s | PASS 3 s |
| `sort` | (defaults) | PASS 3 s | PASS 2 s |
| `stencil` | (defaults) | PASS 3 s | PASS 2 s |
| `svd` | (defaults) | PASS 3 s | PASS 2 s |
| `take_pairs` | (defaults) | PASS 15 s | PASS 34 s |
| `tssvd` | (defaults) | PASS 5 s | PASS 5 s |

## The `_benchmark` shim

`_benchmark.py` provides what the programs import from cuPyNumeric's
`_benchmark` (`benchmark_info`, `get_benchmark_info`, `format_dtype`,
`get_numpy`, `parse_with_harness`, `timed_loop`, `get_timer`, `nthroot`) and
the harness object (`np`, `timer`, `package`, `repeat`, `run`, `run_timed`,
`run_timed_with_info`, ...).

- **Connect.** `parse_with_harness` adds the harness flags, parses with
  `parse_known_args` as upstream does, and connects
  (`_cnum_connect.get_interface`: `CNUM_HOST`/`CNUM_PORT`/`CNUM_ODF`, or the
  simulator with `CNUM_SIM=1`). `harness.package` is `"charmnumeric"`, so
  the `package == "legate"` branches (`stencil_hint`) are skipped.
- **Timing.** `timer.stop()` calls `charmnumeric.synchronize()`, a backend
  barrier that returns once every chare of every partition has finished all
  earlier work, so the times include backend completion.
- **NumPy comparison.** `run_timed*` runs the benchmark function with
  `np` = charmnumeric, then once more with `np` = NumPy, and compares:
  - *Inputs.* `np.random.*` draws (and `default_rng()` generator draws) of
    the charmnumeric run are recorded (a device copy, fetched at the end) and
    replayed in the NumPy run, which checks that NumPy's draw has the same
    shape and dtype. charmnumeric's counter-based streams differ from
    NumPy's, so both runs see the charmnumeric values.
  - *Results.* Every function of the program records its locals when it
    returns (`sys.monitoring`, events only on the program's own code objects,
    so the library runs at full speed). The last record of each function is
    compared between the runs: device arrays and NumPy values by shape, dtype
    and value, Python ints exactly; Python floats (timings) are skipped.
    Floating values are compared with `|got - want| <= rtol * (|want| + S)`,
    `S` the largest magnitude in the same record, `rtol` 1e-2/2e-5/1e-6 for
    float16/float32/float64 (and the complex types).
  - *Results NumPy leaves partly unspecified.* QR/SVD/TSSVD/eigh factors
    (signs) and `argsort` (order of ties unless `kind="stable"`): the NumPy
    run checks the charmnumeric result against its own input (Q R = A,
    orthonormal Q/U/V, triangular R, singular values / eigenvalues vs NumPy,
    U S V^H = A; `argsort` sorts its input and is a permutation) and then
    continues with it. `np.linalg.tssvd` (cuPyNumeric only) is checked as
    NumPy's thin SVD; `np.convolve(a, v, mode, method)` (cuPyNumeric's N-D
    convolve) runs as a direct N-D convolution in NumPy.
  - A line the program prints starting with `FAIL` (`FAIL!` of its own
    `--check`) or an exception also fails the run.
  The per-function records make the comparison generic: no program needs a
  hook.
- `CNUM_BENCH_TRACE=1` prints each fetched value to stderr (to locate a hang).

Support files:

| file | purpose |
|---|---|
| `_benchmark.py` | The harness stand-in described above. |
| `_cnum_connect.py` | `get_interface()`: connects with `cnp.CharmNumericInterface().connect(CNUM_HOST, CNUM_PORT, CNUM_ODF)`, or returns the simulator with `CNUM_SIM=1`. |
| `_cnum_sim.py` | Simulator. Hooks the charmtyles `ExecutionState` to evaluate every recorded op eagerly with NumPy; ops are still fused and serialized normally. Flags patterns the backend mishandles. |
| `_static_check.py` | Runs every program and variant (upstream flags) under the simulator and prints a table. |
| `_cnum_util.py` | Index-grid / pseudo-random input helpers and the `Checker`/`run` PASS/FAIL harness used by some `examples/ops` tests. |
| `upstream/` | The unmodified cuPyNumeric originals (`upstream/_benchmark/`: their harness package, for reference). |

## Programs

All 25 are upstream-exact (the file is the original plus the 2-line
comment). Notes list the NumPy behavior each one depends on; all of it runs
natively on the backend.

| program | upstream-exact | notes |
|---|---|---|
| black_scholes | yes | `random.random(N).astype(D)`, `absolute/exp/sqrt/log`, `where(d > 0, ...)`; float32 default, `--precision 16/64` |
| black_scholes_greeks | yes | 4-D CALL/PUT (FoldedArray); `CALL[g]` is a view, written with `out[...] =`; `arange(dtype=int) % n`, `.reshape(steps)`; `V[:, i, :] += ...` (integer index, written back) |
| cfd | yes | integer column indices (`u[1:-1, 0]` is 1-D), `np.sum(u)` 0-d, `math.isnan(udiff)` and f-strings of 0-d values, `empty_like`, `copy` |
| cg | yes | `diag(v, k)`, `full`, `eye`; `rsold`, `alpha`, `beta` are 0-d device scalars; `np.sqrt(rsnew) < threshold` is a host bool at check points; `--check` uses `allclose`, `--pre` the Jacobi preconditioner |
| cholesky | yes | `eye(n, dtype="float64")`, `linalg.cholesky`; `--check` runs `numpy.linalg.cholesky(get_numpy(...))` and `numpy.allclose` on a device array |
| einsum | yes | `einsum` (string form, `out` via cupy-compatibility path), `ones`; f16/f32/f64/c64/c128 |
| gemm | yes | `random.uniform(size=)`, `dot(A, B, out=C)` |
| indexing_routines | yes | `diag`, `choose(mode="wrap")`, `repeat` (scalar and array counts), `A1[indx] =`, `A1[mask] =`, `A2[b, b] =`, `A2[:, indx] =`, `A3[indx, :, indx] =`, `math.isnan(np.sum(...))` |
| jacobi | yes | `random.rand`, `eye`, `diag`, `dot`; NaN assertion through `float(np.sum(x))` |
| kmeans | yes | `linalg.norm(axis=1)`, `square`, `newaxis`, `.T`, `argmin`, `bincount(weights=data[:, idx])`, `amin`, `maximum(int64, uint64)` (float64), `str()` of 0-d values, `delta > 1 - 1e-6` |
| kmeans_slow | yes | `labels == idx`, `where(mask[..., newaxis], ...)`, `sum(axis=0)` into `centroids[idx, :]`, `pairwise_distances[:, idx]`, `0.0 += np.sum(...)` |
| linreg | yes | `randn/random`, `hstack`, `dot`, vecmat `error.dot(features)`, `power`, `len(features)` |
| logreg | yes | as linreg plus `floor`, `exp`, `log`, `astype(copy=False)`, NaN assertion |
| lstm_backward | yes | 3-D integer indices (`Ct[t]`, `dHin[t, :, w:]`), augmented assignment through integer keys, `transpose()`, `sum(axis=0)`, `dh0[0] +=` |
| lstm_forward | yes | `tile`, `Hin[t, :, w:] = prev`, `Hin[t].dot(WLSTM)`, `tanh`, `exp` |
| qr | yes | `rand(...).astype(dtype=)`, `linalg.qr`; `--check`: `matmul`, `allclose`, `max` |
| quantiles | yes | `random.seed/randint`, `.astype().reshape()`, `np.quantile(a, q, axis, method)` (all NumPy methods); `--check`: `get_numpy`, `allclose(host, device)` |
| richardson_lucy | yes | `rand(*shape)`, `flip`, `convolve(mode="same", method=)`, `im_deconv *=` |
| scan | yes | `randint(1000, size=, dtype=int32)`, `zeros(shape=)`, `cumsum/cumprod/nancumsum/nancumprod(A, out=B, axis=)`; `--check` runs NumPy on the device arrays |
| solve | yes | `linalg.solve`; `--check`: `linalg.norm(res, inf)` (0-d), `max_res < 1e-4`, f-string |
| sort | yes | `random.seed`, `iinfo`, `randint`, `sort`/`argsort` (uint32 default, 1M elements) |
| stencil | yes | views and in-place view assignment |
| svd | yes | `randint/random`, complex inputs `array(x + y * 1j).astype()`, `linalg.svd`; `--check`: `u[:, :k]`, `matmul(u * s, vh)`, `allclose` |
| take_pairs | yes | `arange(dtype=int32).reshape`, `array(list of tuples)`, `take(axis=1)`, `.sum()` (0-d int64), f-string |
| tssvd | yes | `random.seed/rand`, `mat[i, i] = 1.0 + np.sum(np.abs(mat[i, :]))`, `linalg.tssvd` |

Upstream bugs behave as upstream (the same exception as with cuPyNumeric):
`sort.py --check` (calls `check_sorted` with three arguments),
`scan.py -t float|complex` (`len()` of the int shape) and
`svd.py --check` with `m < n` and full matrices (shape mismatch in
`check_result`). `_static_check.py` avoids these combinations.

Not ported: the upstream subdirectories. `cpp/` (C++ programs against the
cuPyNumeric C++ API), `dlpack/` and `inline_task/` (Legate/DLPack interop and
Legate tasks) have no charmnumeric counterpart; `microbenchmarks/` is a
separate benchmark suite with its own driver (`main.py`), not example
programs; `_benchmark/` is the harness, replaced by `_benchmark.py`.

## Library changes made for these programs

All NumPy-exact, each with regression checks in `examples/ops`:

- **0-d arrays.** Full reductions, 1-D dot/inner, vector/matrix norms,
  `det`/`slogdet`, scalar `take` and integer-only indexing return 0-d arrays
  (stored as (1,) on the backend; `shape == ()`, `ndim == 0`). They combine
  with arrays of any rank, and `str`, `format`, `float`, `int`, `complex`,
  `bool`, `operator.index` and `__array__` fetch the value, so `if x < tol`,
  `math.isnan(x)` and f-strings work as in NumPy (`test_frontend.py
  zero_d_results`). `np.sum` etc. stay device values (NumPy returns host
  scalars), so they feed later expressions without a sync.
- **Basic indexing.** An integer index drops its dim (NumPy) instead of
  keeping it; integer-only keys give 0-d arrays, `x[()]` of a 0-d array a
  copy and `x[...]` the array. Results that NumPy returns as views but a
  charmnumeric view cannot express (a dropped dim, newaxis, a negative step)
  are copies linked to their array: writes to them go back to it, and they
  are refreshed from it before use when it has been written since
  (`_basic_getitem`, `_fresh`, `_wrote` in `charmnumeric.py`). `__setitem__`
  takes values shaped like NumPy's target (`test_frontend.py
  integer_indices`, `test_fuzz_frontend.py`).
- **Size-1 operands of another rank.** A fused kernel with a size-1 operand
  of a different storage rank hung the backend (cross-partition broadcast);
  such operands now move to the kernel's rank first (`_size1_to_rank`).
  `test_fuzz_frontend.py` no longer skips these cases (`KNOWN_BUGS_SKIP`
  removed).
- **NumPy interop.** `__array__` fetches (so NumPy functions accept device
  arrays), `__array_ufunc__` runs NumPy ufuncs natively (also
  `numpy_scalar * array`), and `add/subtract/multiply/divide` keep NumPy
  scalars typed (NEP 50) (`test_frontend.py numpy_interop`).
- **New functions.** `allclose`/`isclose` (NumPy's formula, `equal_nan`; device
  results), `quantile`/`percentile` (native sort, NumPy's index and
  interpolation rules for every method, `axis` None/int, `keepdims`, `out`),
  `true_divide`, `iinfo`/`finfo`, and `synchronize()` (backend barrier: a GET
  of the reserved name -1 answered once every chare of every partition has
  reached it; `src/partition_get.cpp`, `src/dag_group_runtime.cpp`).
- **Fixes.** `arange` takes its dtype from start, stop and step
  (`arange(4.0)` was int64); `lstsq` residuals of a vector right-hand side
  have shape (1,). Backend: a fused kernel that writes a shifted region
  (`P[96:112] = d + 1`, or `out[...] = ...` into `CALL[g]` in
  black_scholes_greeks) runs over that region and allocates its other kept
  results (`d`) there, but their decomposition placed them elsewhere, so a
  later read of `d` hung; the decomposition solver now places such results
  with the region (`src/decomposition_solver.hpp`, `test_frontend.py
  shifted_assignment_results`).

## Remaining deviations

None in the programs. Library-level differences from NumPy that the
programs do not observe:

- Linked copies (above) are not NumPy views: `y.base` is unset, a write to
  the array is seen by `y` only when `y` is next used by an operation (it is
  then refreshed), and each refresh is a copy.
- Full reductions return 0-d device arrays, not host scalars
  (`type(np.sum(x))` differs; values, printing and conversions match).
- `bincount` without `minlength` still needs `interface=` (data-dependent
  length); values outside `[0, minlength)` are ignored.

## Missing backend/frontend ops, ranked by examples they would unblock or simplify

1. ~~**Transpose (`.T`/`transpose`) or vecmat (1-D @ 2-D).**~~ Done: native
   2-D/3-D transpose and vecmat (opcode 70, `src/executor_transpose.cpp`).
   The 10 examples that needed it (kmeans, kmeans_slow, linreg, logreg,
   lstm_backward, einsum, cholesky, qr, svd, tssvd) now use `.T` / `x @ A`
   directly instead of generated transposed inputs or symmetry tricks. A
   fused transposed-operand SUMMA (no materialized copy) is still open.
2. ~~**Comparisons + `where`/select, `minimum`/`maximum`.**~~ Native since
   2026-10-08 (bool results since 2026-10-09). Ports switched: black_scholes,
   black_scholes_greeks, kmeans, kmeans_slow, logreg, sort, quantiles, scan,
   indexing_routines.
3. ~~**Bulk host->device upload and a native RNG**~~ Done: `cnp.array`
   uploads in bulk and `cnp.random` generates on the backend (opcodes 50-52,
   `src/executor_creation.cpp`). cholesky, qr, solve, svd, tssvd, take_pairs,
   jacobi, gemm, black_scholes and scan use them.
4. **Axis reductions: sum/mean/amin/argmin/argmax along an axis, plus a full
   `sum`.** NOW NATIVE (`charmnumeric/reductions.py`, REDUCTION/BINCOUNT
   executors in `src/executor_reduction.cpp`): cfd, jacobi, kmeans,
   kmeans_slow, linreg, logreg, lstm_backward, tssvd, take_pairs, solve
   (check), qr (check) use them. Was: 8 examples: cfd, kmeans, kmeans_slow, linreg, logreg,
   lstm_backward, tssvd, indexing_routines. sum is emulable with dot-with-ones;
   argmin/amin are not cheaply emulable.
5. ~~**Dense factorization kernels (potrf, getrf with pivoting, geqrf, syevd/gesvd)
   and trsm.**~~ Native (`cnp.linalg`, LINALG opcode 100, single-chare
   gather-factor-scatter): cholesky, solve, qr, svd, tssvd are direct ports.
   Limit: the operands are gathered and factored on one chare, so this is
   not scalable (distributed blocked factorizations are still open).
6. ~~**Elementwise math: sqrt, log, abs, pow (`__pow__`), floor.**~~ Native
   since 2026-10-08; black_scholes, black_scholes_greeks, logreg, tssvd, cfd,
   lstm_backward and linreg use them.
7. ~~**Gather/scatter: fancy and boolean-mask indexing, row permutation, take
   with device indices.**~~ Native (GATHER/SCATTER/EXPAND, opcodes 120-122,
   `src/executor_indexing.cpp`, `charmnumeric/indexing.py`); indexing_routines
   and take_pairs use them (bincount is native; kmeans keeps the one-hot form;
   solve's pivoting is inside `cnp.linalg.solve`).
8. ~~**reshape/flatten, more than 3 dims, arange on device**~~ Done:
   RESHAPE/BROADCAST (opcodes 140-141, `src/executor_shape.cpp`) and
   FoldedArray (`charmnumeric/folded.py`) for >3-D arrays. black_scholes_greeks,
   scan, quantiles, take_pairs use them.
9. ~~sort/argsort~~ (native since the castscansort ops; a distributed sample
   sort is still missing for scaling).
10. ~~astype (dtype conversion)~~ (fused CONVERT op; mixed dtypes are converted
    in-kernel with NumPy's promotion rules).
11. **One example each:** ~~scan (cumsum/cumprod)~~ (native), ~~convolve~~ (native),
    ~~einsum~~ (frontend lowering), ~~flip/negative-step slicing~~ (native, copies),
    ~~general broadcasting / newaxis (kmeans)~~ (native). ~~complex dtypes, float16,
    unsigned ints, bool~~ (native).

## Frontend issues found (file:line, paths relative to `charmtyles/`)

1. `example/charmnumeric/charmnumeric/charmnumeric.py:291-292`:
   `np.newaxis`/`None` in an index raises NotImplementedError, although
   `__init__.py:37` exports `newaxis = None`.
   **Fixed:** `x[:, newaxis]` returns a linked copy with the size-1 dims
   inserted (see "Basic indexing" below).
2. (fixed) `random.randn` used to return a zero-filled array; `cnp.random` is
   now a native generator.
3. `charmtyles/core.py:9,23-28`: names are never recycled. **Fixed:**
   `max_name` is now the int32 limit (names are int32 on the wire and in
   every backend map), and host data no longer costs one name per element
   (uploads, arange and eye are single nodes).
4. The frontend accepted full NumPy
   broadcasting, e.g. (4,8)+(8,). The backend only broadcasts size-1 operands
   (`BROADCAST_OBJECT` in `charmtyles/ast.py:95,101`). Such expressions are
   accepted silently and produce wrong results or crash in the backend.
   **Fixed:** such operands are materialized with a native BROADCAST node.
5. `charmnumeric.py:497-508` (1-D @ 1-D REDUCE), `operations.py:52` (tile) and
   `operations.py:125` (diag) accept views. The backend executors read the
   operands by array name and ignore the view region
   (`src/executor_reduce.cpp:32-42`, `src/execute_node.cpp:601,756`). So
   `x[0:4] @ x[4:8]` or `tile(A[0:2], ...)` silently use the whole array. The
   frontend should materialise views first. **Fixed:** partial views are copied
   first.
6. Fixed: 1-D @ 2-D (vecmat, e.g. `error.dot(features)`) used to raise
   "matmul requires a 2D matrix, got 1D". It is now `A.T @ x` with a native
   transpose of A that is cached until A is written.
7. `charmnumeric.py:526-530` (and 542-545): `A @ B` with B of shape (K,1)
   becomes a matvec. The result has shape (M,) instead of NumPy's (M,1), and
   the rewrite goes through a cross-ndim copy from a view (see the next item).
   **Fixed:** the result is (M, 1).
8. `charmnumeric.py:851-860`: `create_array` accepts more than 3 dims (e.g.
   `zeros((2,2,2,2))`), but the backend only has `Partition1D/2D/3D`.
   **Fixed:** arrays with more than 3 dims are FoldedArrays (3-D storage);
   0-d arrays are stored as (1,) arrays.
9. `charmtyles/interface.py:88`: `CCSInterface.connect` has no timeout. With no
   backend listening, a port hangs instead of failing.
10. `charmtyles/ast.py:494`: `_ancestor_bits` uses `1 << node_id` with global,
    never-reset node ids. The ints grow without bound over a long program.
    This is performance, not correctness.

Backend/serialization bugs found while testing the frontend ops. All of
them are fixed by the merged backend bug-fix patches (`wt-bugs/bugfixes.diff`,
`bugfixes-charmtyles-core.diff`), which the `frontend` worktree is based on:
size-1 views sent as broadcasts (`r[2:3] + 0` returned `r[0]`), reads after
an in-place set_region inside one fused kernel, the 1-D -> N-D rank promotion
in `dag.hpp` (cross-ndim set_region did nothing), cross-ndim set_region with
views / offsets / offset decompositions, `get()` of offset-decomposed arrays,
and partitions first used by a MATMUL/MATMATMUL result never starting (hang
in einsum's batched path). `examples/ops/test_frontend.py` covers each.

Backend bugs found while porting kmeans_slow (column views `pw[:, idx:idx+1]`
of a fresh result give it a column decomposition offset), fixed in the
`hard-shape` worktree: the partition grid ignored a result's decomposition
offset (tiles past the last chare were never computed), a chare holding none
of a kernel's arrays could not size its output tile (`extract_regions_nd` now
falls back to the array metadata), and the solver could pick a negative
offset (now moved into [0, tile)). Each one hung the run.

Found while testing einsum: SUMMA and matvec ignore the step of a strided
view operand (`A[::2] @ B` gave wrong rows, larger cases crashed the server
in `gather`). The frontend now copies such views first (`_unstrided` in
`charmnumeric.py`); offset views were already correct.

Unary math on an all-broadcast (size-1) kernel (`tanh(ones((1,)))`) used to
crash the JIT; it works now (`test_frontend.py zero_d_results`).

Suspected backend issue (fixed by the bug-fix patches; kept for reference):
`src/backend_internal.hpp:345-351` (`cross_set_region_send`) maps source-local
coordinates directly to target coordinates. It ignores the target region's start
in non-singleton dims, and it sends the whole source array rather than a view's
region. A cross-ndim `set_region` at a nonzero offset, such as
`Hin[t, :, word_size:] = prev2d` in `examples/lstm_forward.py:36`, would then
land at offset 0. The simulator flags this pattern as WARN. None of the ports
use it.

Backend issues reproduced on the CPU build while testing the creation ops
(all also happen with plain `cnp.zeros` + arithmetic, i.e. independent of the
new ops; `examples/ops/test_creation.py` avoids them):

- ~~After any size-1 1-D array (e.g. `cnp.zeros((1,)) + 1`) has been
  materialized, a later 1-D array that needs a larger 1-D chare grid (e.g.
  300007 elements) hangs its `.get()`.~~ Fixed: chares inserted later replay
  old DAGs and waited for size-1 broadcasts sent before they existed;
  broadcasts now use the chare grid of the DAG's epoch
  (`ArrayDAGGroup::grid_at`), and a size-1 `set_region` target is no longer
  treated as a broadcast input.
- ~~`.get()` of a 0-size array (e.g. `cnp.arange(3, 1)`) hangs.~~ Fixed in the
  frontend (`Array.get` returns an empty ndarray).

## Static check results

`python _static_check.py` runs 57 cases (all 25 programs with their default
flags and variants using the original flags: precisions, dtypes including
complex, `--check` paths, other routines/methods/axes) with `CNUM_SIM=1`:
**57/57 PASS, 0 simulator errors.** The simulator stores 0-d arrays as (1,)
arrays like the backend and answers the `synchronize()` barrier. Its
complex linalg model is incomplete (complex SVD and the complex cases of
`examples/ops/test_dtypes.py` fail under the simulator but pass on the
backend), so the complex SVD variant runs only on the backend.

This checks:

- frontend graph construction, shapes and dtypes,
- fusion and DAG serialization (every DAG is serialized and handed to
  `run_async`),
- the programs and their NumPy comparison under the intended charmnumeric
  semantics,
- the absence of patterns the backend is known not to support.

It does **not** run the C++ backend.

## Reference mode: cuPyNumeric instead of NumPy

`CNUM_REFERENCE=cupynumeric` makes the shim run the upstream program on real cuPyNumeric (a `legate` subprocess in a
separate conda env) with the recorded draws and factors replayed, and compare the locals as above; the reported times
of both sides are printed. `run_cupy_compare.sh <npes> <outdir>` runs all ports this way on a GPU node (charmnumeric
+p<npes> against cuPyNumeric on <npes> GPUs) and `_cupy_compare_table.py` formats results.jsonl. Setup and flags are
in the docstring of `_cupy_reference.py`; the NumPy mode is the default and unchanged.
