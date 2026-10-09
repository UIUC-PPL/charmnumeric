# cuPyNumeric examples ported to charmnumeric

These are ports of every top-level program in `cupynumeric/examples/`, excluding
`benchmark.py`, which is the timing harness. Each port keeps the same filename,
the same algorithm and loop structure, and the original argparse flags where
they make sense. Defaults are small so that a run takes seconds. Every port
checks itself against plain NumPy on the same inputs. The last line it prints
is exactly `PASS <name>` or `FAIL <name>: <reason>`, and it exits non-zero on
FAIL.

## Running

```bash
module load python/3.13.5-gcc13.3.1; . /u/atahilyani/cnum-gpu/venv/bin/activate
cd example/charmnumeric/examples/cupynumeric

# against a running backend (server.out ++server ++server-port $CNUM_PORT)
CNUM_HOST=127.0.0.1 CNUM_PORT=1234 CNUM_ODF=4 python stencil.py
CNUM_AUTO_FLUSH=1000 python cfd.py          # optional: charmtyles set_auto_flush

# without a backend: host-side reference simulator (static/semantic check)
CNUM_SIM=1 python kmeans.py
python _static_check.py [--quick] [name ...]   # all ports + variants, tabulated
```

Support files:

| file | purpose |
|---|---|
| `_cnum_connect.py` | `get_interface()`. Connects with `cnp.CharmNumericInterface().connect(CNUM_HOST, CNUM_PORT, CNUM_ODF)`. With `CNUM_SIM=1` it returns the simulator instead. |
| `_cnum_util.py` | Native input generators, thin wrappers over the native elementwise ops (kept for compatibility), reductions built from dots, the `upload` host-upload helper (bulk `cnp.array` into a block of a target), and the Checker/`run` PASS/FAIL harness. |
| `_cnum_sim.py` | Simulator. Hooks the charmtyles `ExecutionState` to evaluate every recorded op eagerly with NumPy. Ops are still fused and serialized normally. It also flags patterns that the frontend accepts but the backend mishandles. |
| `_static_check.py` | Runs every port and variant under the simulator and prints a table. |

### Inputs and verification

charmnumeric now has a native counter-based RNG (`cnp.random.rand`,
`random`, `randn`, `uniform`, `normal`, `randint`, `seed`, `default_rng`)
and a bulk host upload (`cnp.array` / `cnp.asarray` / `A[...] = ndarray` is one
UPLOAD DAG node carrying the data; each chare copies its tile). Ports whose
original calls `np.random.*` with a direct equivalent (cholesky, qr, solve,
svd, tssvd, jacobi, gemm, black_scholes, scan) use `cnp.random`, and
take_pairs builds `arange(M*N).reshape(M, N)` with `cnp.meshgrid` +
`cnp.arange`. The random streams differ from NumPy's, so every port fetches its
inputs with `.get()` and computes the NumPy reference from exactly those
values.

The other ports still use the older native generators
(`_cnum_util.uniform` / `randn` / `rand_int`):

- index grids are built with ceil(log2 n) shifted-slice adds,
- then a monotone warp,
- then 24 steps of the chaotic logistic map `x <- 3.99 x (1-x)`.

The resulting matrices are conditioned like `np.random.rand` output (checked:
cond ~2e3 to 4e3 for 128x128, no lag or cross-seed correlation). Each port then
fetches the generated inputs with `.get()` and computes the NumPy reference from
exactly those values.

Transposes (`X.T`, `transpose`, `swapaxes`) and vecmat (`x @ A`) are native
(TRANSPOSE executor, `src/executor_transpose.cpp`). Earlier versions of the
ports generated `X.T` of an input from the same draw with permuted indices
(`cnp.random.transposed(X)`, `perm=` for the `_cnum_util` generators) or used
symmetry tricks; they now use `.T` as the originals do.

## Feasibility table

Verdict meanings:

- **direct**: same NumPy calls, only trivial syntax changes.
- **workarounds**: fully native, using EMULATED primitives or transposed inputs.
- **needs backend**: the faithful port needs a missing primitive. It is ported
  with clearly marked `HOST-FALLBACK`s or a supported subset.

| example | NumPy ops needed | missing in charmnumeric | verdict | port status |
|---|---|---|---|---|
| black_scholes | random, sqrt, log, exp, abs, where(d>0) | none (random) | direct | native: cnp.sqrt/log/absolute/where, `d > 0` is a 0/1 mask |
| black_scholes_greeks | 4-D arrays, arange/%/reshape, sqrt, log, exp, abs, where | >3-D, reshape | workarounds | native. The greek axis is folded into dim 0; `arange(..).reshape` is the native int64 linear index, `%` is native, int * float promotes natively and K/T are converted with native `astype`. |
| cfd | slicing stencils, `**2`, `sum` | none | direct | native (`x**2`, native `cnp.sum`) |
| cg | diag(v,k), eye, matvec, dot, sqrt(scalar) | none (random) | direct | native. Convergence test is a host scalar. |
| cholesky | linalg.cholesky (input eye) | none | direct | native `cnp.linalg.cholesky` (LINALG executor). `--input spd` for a non-trivial input. |
| einsum | einsum (batched contraction), ones | einsum | needs backend | every single-contraction matrix product (2-D, or 3-D with a leading batch mode, any mode order) lowered to native transposes + (batched) matmul; 2-D ones with an untransposed output use `cnp.matmul(..., out=C)`. Others raise NotImplementedError. |
| gemm | dot(out=) | none | direct | native (`cnp.dot(A, B, out=C)`) |
| indexing_routines | diag, choose, repeat, fancy/bool-mask indexing, `%`, arange | choose, repeat, gather/scatter, mask indexing | needs backend | diag native. choose = native `%` + chain of native where (int64, exact). Uniform repeat via strided slices. ai1/2/3 index patterns lowered to slices (HOST-FALLBACK). |
| jacobi | rand, eye, diag, dot, sum (NaN assertion) | none (random) | direct | native (NaN assertion via native `cnp.sum`) |
| kmeans | norm(axis=1), newaxis broadcast, `.T`, argmin, bincount, amin, maximum, not_equal | broadcasting | workarounds | native. Native argmin(axis=1) (as float labels), one-hot via native `==`, native amin+sum, maximum and count_nonzero; counts int64 via native `astype`; centroid sums = native `M.T @ data`; native `.T` |
| kmeans_slow | same, plus `labels == idx`, where, sum(axis=0) | as above | workarounds | native (native argmin/sum/count_nonzero/`==`; mask = M@e_idx, where(m,x,0) = m*x, row sums = native vecmat `mask @ data`) |
| linreg | randn, hstack, matvec, vecmat (`err.dot(X)`), power, sum | none | workarounds | native (native vecmat `error.dot(features)`, `cnp.hstack` for the intercept, native power and `cnp.sum`) |
| logreg | as linreg + exp, log, floor | none | workarounds | native (native vecmat, `cnp.hstack`, log, floor and `cnp.sum`) |
| lstm_backward | 3-D slicing, `**2`, `.transpose()`, sum(axis 0), dot | none | workarounds | native (native `WLSTM.transpose()`, native `**2`, native batch `cnp.sum(axis=1)`) |
| lstm_forward | 3-D slicing, tile, exp, tanh, dot | none | direct | native (h0 tiled to 3-D (1,b,d)) |
| qr | linalg.qr | none | direct | native `cnp.linalg.qr` (Householder on one chare); allclose(a, q @ r) and Q^T Q (native `q.T @ q`) checked natively. |
| quantiles | quantile (sort + interpolation), astype | reshape | direct | native `cnp.sort` + native interpolation; int inputs via native `astype(float64)` |
| richardson_lucy | 3-D convolve(mode=same), flip, `/`, `*` | convolve, flip / negative strides | workarounds | native direct convolution (f^3 broadcast-scalar MACs); psf_mirror from the reversed index |
| scan | cumsum/cumprod/nancumsum/nancumprod, axis=None flatten | reshape | direct | native `cnp.cumsum/cumprod/nancumsum/nancumprod` (with `out=`). axis=None on 2-D composed from native axis scans (no reshape). |
| solve | linalg.solve | none | direct | native `cnp.linalg.solve` (partial-pivoted LU on one chare); residual norm native. |
| sort | sort/argsort (uint32 default) | uint32 | direct | native `cnp.sort/argsort` (gather-to-one-chare executor, stable). |
| stencil | slicing views, in-place view assignment | none | direct | native |
| svd | linalg.svd | general broadcasting (`u * s`), complex | direct | native `cnp.linalg.svd` (full_matrices; int inputs factored in float64, int/float promotion native in the check); `u * s` written `u @ diag(s)`. |
| take_pairs | arange.reshape, take(axis=1), sum | reshape | workarounds | data grid native (meshgrid of aranges). `data.take(indices, axis=1)` is native (host indices lowered to column-slice copies). Native 3-D `result.sum()`. |
| tssvd | rand, abs, sum, linalg.tssvd | general broadcasting (`u * s`) | direct | original make_random_matrix loop + native `cnp.linalg.tssvd` (eigh of A^T A on one chare, U = A V S^-1 by SUMMA). |

### Summary

- **Direct, fully native, no emulation (15):** stencil, jacobi, gemm, cg, cfd,
  lstm_forward, black_scholes, scan, sort, quantiles, cholesky, qr, solve,
  svd, tssvd.
- **Fully native with workarounds, no host step in the algorithm (9):**
  black_scholes_greeks, lstm_backward, linreg, logreg, kmeans, kmeans_slow,
  richardson_lucy, einsum (subset), take_pairs (native take with host
  indices).
- **Native bulk work with marked HOST-FALLBACKs (1):** indexing_routines
  (advanced-indexing patterns).
- **Partially blocked:**
  - einsum: anything but a single-contraction matrix product (2-D, or 3-D
    with a leading batch mode).
  - quantiles: axis=None on N-D.
  - sort/quantiles: uint32.
  - scan/sort/svd/tssvd/cholesky/qr/solve: complex dtypes.
  - --precision 16 everywhere.

## Per-example notes

- **Elementwise math, comparisons, where.** Native since 2026-10-08
  (opcodes 20-49, tests in `examples/ops/`). Comparisons and predicates
  return **0/1 masks in the operands' dtype**, not bool arrays (one element
  type per fused kernel, no bool dtype); masks work directly as `where`
  conditions and in arithmetic. The tanh-step, Newton/Halley and
  integer-division emulations are gone; the old `_cnum_util` helper names
  (`step`, `absolute`, `select`, `maximum`, `sqrt`, `log`, `int_mod`,
  `greater`, ...) remain as thin native wrappers.
- **black_scholes / greeks.** float32 by default, as in the original;
  tolerance rtol 2e-4. In greeks, `np.sqrt(T)` is computed once per call and
  reused. K and T stay in the working dtype; NumPy would promote them to
  float64.
- **cfd.** Integer column indices are written as length-1 slices so that the
  same function runs under NumPy and charmnumeric. In charmnumeric `u[1:-1, 0]`
  already keeps the dim.
- **cg.** The convergence test fetches `rsnew` only at check points. Verifies
  A (exact), `A.dot(x) ~ b`, and `x ~ linalg.solve`.
- **cholesky / qr / solve / svd / tssvd.** Same calls as the originals
  (`cnp.linalg.*`, see `examples/ops/test_linalg.py`), default sizes as in the
  originals, `nargs="+"` size lists. The verification always runs, natively
  where possible (`U.allclose`), plus a NumPy reference on the fetched inputs,
  with dtype-aware tolerances (`U.tolerances`; solve's fixed 1e-4 residual
  limit becomes n eps ||a|| ||x|| for float32). cholesky keeps the original
  identity input by default (`--input spd` for a random SPD matrix).
- **einsum.** Inputs are pseudo-random/N instead of ones so that the check is
  meaningful (`--ones` gives the original). A transposed operand is
  recomputed only when it changes; B never changes under the A<->C swap, so
  B.T is transposed once.
- **indexing_routines.** The ai2 and ai3 final states are fully overwritten
  (all-true masks), as in the original.
- **kmeans / kmeans_slow.** Default precision is 64. A float32 near-tie could
  make the native argmin differ from the NumPy reference. The frontend name
  limit is now the int32 limit, so the iteration count is not name-bound.
- **linreg / logreg.** `--num` is in thousands, default 1 (1000 points).
- **lstm_forward.** Tiling h0 to 3-D avoids a 2-D->3-D set_region at a nonzero
  column offset. See the suspected backend issue below. The existing
  `examples/lstm_forward.py` does exactly that pattern.
- **quantiles / sort.** `cnp.sort`/`cnp.argsort` gather the array on one
  chare, sort each line along the axis there (NaNs last, stable) and scatter
  it back: exact, O(n log n), but not scalable (no distributed sample sort).
  argsort is compared exactly with `numpy.argsort(kind="stable")`. The old
  O(n^2) odd-even network was removed from `_cnum_util.py`.
- **richardson_lucy.** Assumes cuPyNumeric's N-D `convolve(mode="same")`
  follows the scipy.signal.convolve centering (start index (f-1)//2).
- **scan.** Native SCAN executor. For `axis=None` on 2-D, the port computes
  the flattened-order scan from native axis scans but keeps the 2-D shape,
  since there is no reshape.
- **take_pairs.** `cnp.take` with the host index list (N=16, P=120 pairs):
  index column 1 is 15 contiguous runs (15 slice copies), index column 0
  repeats indices (120 single-column copies, which fuse); each (M, P) column
  result then goes into the 3-D result with one 2-D -> 3-D copy. The result
  sum is checked exactly.

## Missing backend/frontend ops, ranked by examples they would unblock or simplify

1. ~~**Transpose (`.T`/`transpose`) or vecmat (1-D @ 2-D).**~~ Done: native
   2-D/3-D transpose and vecmat (opcode 70, `src/executor_transpose.cpp`).
   The 10 examples that needed it (kmeans, kmeans_slow, linreg, logreg,
   lstm_backward, einsum, cholesky, qr, svd, tssvd) now use `.T` / `x @ A`
   directly instead of generated transposed inputs or symmetry tricks. A
   fused transposed-operand SUMMA (no materialized copy) is still open.
2. ~~**Comparisons + `where`/select, `minimum`/`maximum`.**~~ Native since
   2026-10-08 (0/1 masks in the operand dtype). Ports switched: black_scholes,
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
7. **Gather/scatter: fancy and boolean-mask indexing, row permutation, take
   with device indices.** 1 example: indexing_routines (bincount is native;
   kmeans uses the one-hot form; solve's pivoting is inside `cnp.linalg.solve`). `take` with host indices is native
   now; take_pairs uses it.
8. **reshape/flatten, more than 3 dims, arange on device** (`%` is native).
   4 examples:
   black_scholes_greeks, scan (axis=None), quantiles, take_pairs.
9. ~~sort/argsort~~ (native since the castscansort ops; a distributed sample
   sort is still missing for scaling).
10. ~~astype (dtype conversion)~~ (native CAST executor; mixed int/float
    operands are cast implicitly).
11. **One example each:** ~~scan (cumsum/cumprod)~~ (native), convolve (richardson_lucy),
    einsum, flip/negative-step slicing (richardson_lucy), general broadcasting
    / newaxis (kmeans), complex dtypes (scan/sort/svd), float16 (gemm,
    black_scholes, linreg, logreg, kmeans).

## Frontend issues found (file:line, paths relative to `charmtyles/`)

1. `example/charmnumeric/charmnumeric/charmnumeric.py:291-292`:
   `np.newaxis`/`None` in an index raises NotImplementedError, although
   `__init__.py:37` exports `newaxis = None`.
   **Fixed:** `x[:, newaxis]` returns a *copy* with the size-1 dims inserted
   (a view cannot change ndims); newaxis in a setitem key is dropped.
2. (fixed) `random.randn` used to return a zero-filled array; `cnp.random` is
   now a native generator.
3. `charmtyles/core.py:9,23-28`: names are never recycled. **Fixed:**
   `max_name` is now the int32 limit (names are int32 on the wire and in
   every backend map), and host data no longer costs one name per element
   (uploads, arange and eye are single nodes).
4. `charmnumeric.py:346-358`: `_broadcast_shape` accepts full NumPy
   broadcasting, e.g. (4,8)+(8,). The backend only broadcasts size-1 operands
   (`BROADCAST_OBJECT` in `charmtyles/ast.py:95,101`). Such expressions are
   accepted silently and produce wrong results or crash in the backend.
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
   **Fixed:** every Array with 0 or more than 3 dims raises NotImplementedError.
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

Known remaining backend limit: unary math on an all-broadcast (size-1) kernel
crashes the JIT (`tanh(ones((1,)))`), see MISSING_OPS Recipe A.

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

- After any size-1 1-D array (e.g. `cnp.zeros((1,)) + 1`) has been
  materialized, a later 1-D array that needs a larger 1-D chare grid (e.g.
  300007 elements) hangs its `.get()`.
- `.get()` of a 0-size array (e.g. `cnp.arange(3, 1)`) hangs.

## Static check results

`python _static_check.py` runs 46 cases (all 25 ports plus variants) with
`CNUM_SIM=1`. Result (2026-10-09, after merging native transpose/vecmat,
the frontend ops (hstack/vstack, take, `out=`), astype/scan/sort and
`cnp.linalg` into the native elementwise/creation/reduction tree): **46/46
PASS, 0 simulator errors, 0 warnings.** The simulator knows the elementwise
opcodes 20-49 (math, comparisons as 0/1 masks, where, clip), creation 50-52,
reductions 60-61, transpose 70, CAST/SCAN/SORT/ARGSORT 80-83 and LINALG 100.
This checks:

- frontend graph construction, shapes and dtypes,
- fusion and DAG serialization (every DAG is serialized and handed to
  `run_async`),
- the algorithms and their NumPy verification under the intended charmnumeric
  semantics,
- the absence of patterns the backend is known not to support.

It does **not** run the C++ backend. Backend semantics, such as fused-kernel
aliasing, must be confirmed by real runs (CPU build: see the native-op status
in `MISSING_OPS.md` section 1).
