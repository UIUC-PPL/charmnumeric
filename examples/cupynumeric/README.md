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
| `_cnum_util.py` | Native input generators, thin wrappers over the native elementwise ops (kept for compatibility), the sort network, reductions built from dots, the `upload` host-upload helper (bulk `cnp.array` into a block of a target), and the Checker/`run` PASS/FAIL harness. |
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

When an algorithm needs `X.T` of an *input* X, the port generates the transpose
natively from the same draw with permuted indices (`cnp.random.transposed(X)`
for `cnp.random` inputs, `perm=` for the `_cnum_util` generators). It also
checks that the result equals `X.T` exactly. This is how the missing transpose
is worked around.

## Feasibility table

Verdict meanings:

- **direct**: same NumPy calls, only trivial syntax changes.
- **workarounds**: fully native, using EMULATED primitives or transposed inputs.
- **needs backend**: the faithful port needs a missing primitive. It is ported
  with clearly marked `HOST-FALLBACK`s or a supported subset.

| example | NumPy ops needed | missing in charmnumeric | verdict | port status |
|---|---|---|---|---|
| black_scholes | random, sqrt, log, exp, abs, where(d>0) | none (random) | direct | native: cnp.sqrt/log/absolute/where, `d > 0` is a 0/1 mask |
| black_scholes_greeks | 4-D arrays, arange/%/reshape, sqrt, log, exp, abs, where | >3-D, reshape | workarounds | native. The greek axis is folded into dim 0; `arange(..).reshape` is the native linear index, `%` is native. |
| cfd | slicing stencils, `**2`, `sum` | none | direct | native (`x**2`, native `cnp.sum`) |
| cg | diag(v,k), eye, matvec, dot, sqrt(scalar) | none (random) | direct | native. Convergence test is a host scalar. |
| cholesky | linalg.cholesky (input eye) | potrf, trsm, transpose | needs backend | blocked right-looking. Native SUMMA updates. HOST-FALLBACK: bs x bs potrf+trtri. |
| einsum | einsum (batched contraction), ones | einsum, transpose | needs backend | 4 expressions lowered to (batched) matmul. Others raise NotImplementedError. |
| gemm | dot(out=) | `out=` | direct | native (`C[...] = A @ B`) |
| indexing_routines | diag, choose, repeat, fancy/bool-mask indexing, `%`, arange | choose, repeat, gather/scatter, mask indexing | needs backend | diag native. choose = native `%` + chain of native where (int64, exact). Uniform repeat via strided slices. ai1/2/3 index patterns lowered to slices (HOST-FALLBACK). |
| jacobi | rand, eye, diag, dot, sum (NaN assertion) | none (random) | direct | native (NaN assertion via native `cnp.sum`) |
| kmeans | norm(axis=1), newaxis broadcast, `.T`, argmin, bincount, amin, maximum, not_equal | broadcasting, transpose, astype | workarounds | native. Native argmin(axis=1) (as float labels), one-hot via native `==`, native amin+sum, maximum and count_nonzero; centroid sums = dataT @ M |
| kmeans_slow | same, plus `labels == idx`, where, sum(axis=0) | as above | workarounds | native (native argmin/sum/count_nonzero/`==`; mask = M@e_idx, where(m,x,0) = m*x) |
| linreg | randn, hstack, matvec, vecmat (`err.dot(X)`), power, sum | vecmat/transpose, hstack | workarounds | native (featuresT generated, `featuresT @ err`, native power and `cnp.sum`) |
| logreg | as linreg + exp, log, floor | as linreg | workarounds | native (native log, floor and `cnp.sum`) |
| lstm_backward | 3-D slicing, `**2`, `.transpose()`, sum(axis 0), dot | transpose | workarounds | native (WT generated, native `**2`, native batch `cnp.sum(axis=1)`) |
| lstm_forward | 3-D slicing, tile, exp, tanh, dot | none | direct | native (h0 tiled to 3-D (1,b,d)) |
| qr | linalg.qr | geqrf (or potrf+trsm), transpose | needs backend | CholeskyQR2. Native SUMMA. HOST-FALLBACK: n x n chol+inv of the Gram matrix. |
| quantiles | quantile (sort + interpolation), astype | sort, astype, reshape | workarounds | native O(n^2) odd-even sort network (native comparisons + where) + native interpolation (float only) |
| richardson_lucy | 3-D convolve(mode=same), flip, `/`, `*` | convolve, flip / negative strides | workarounds | native direct convolution (f^3 broadcast-scalar MACs); psf_mirror from the reversed index |
| scan | cumsum/cumprod/nancumsum/nancumprod, axis=None flatten | scan, reshape | workarounds | native Hillis-Steele scan; nan* variants use native where(isnan(A), identity, A). |
| solve | linalg.solve | getrf/getrs, argmax, row gather/permute | needs backend | blocked LU with partial pivoting. Native row swaps + SUMMA. HOST-FALLBACK: panel getrf, bs x bs inverses. |
| sort | sort/argsort (uint32 default) | sort/argsort, uint32 | workarounds | native O(n^2) odd-even transposition network built from native comparisons + where (exact for float and int keys). |
| stencil | slicing views, in-place view assignment | none | direct | native |
| svd | linalg.svd | gesvd/syevd, transpose | needs backend | Gram SVD. Native A^T A and A V S^-1. HOST-FALLBACK: k x k eigh. |
| take_pairs | arange.reshape, take(axis=1), sum | take/gather, reshape | needs backend | data grid native (meshgrid of aranges). take = matmul with 0/1 selection matrices (HOST-FALLBACK: built on the host, bulk upload). Native 3-D `result.sum()`. |
| tssvd | rand, abs, sum, linalg.tssvd | syevd/tssvd, transpose | needs backend | native make_random_matrix (row sums of abs = native `linalg.norm(ord=1, axis=1)`) + Gram SVD (HOST-FALLBACK eigh) |

### Summary

- **Direct, fully native, no emulation (7):** stencil, jacobi, gemm, cg, cfd,
  lstm_forward, black_scholes.
- **Fully native with workarounds or emulation, no host step in the algorithm
  (11):** black_scholes_greeks, lstm_backward, linreg, logreg, kmeans,
  kmeans_slow, sort, quantiles, richardson_lucy, einsum (subset), scan.
- **Native bulk work with marked HOST-FALLBACKs (7):** cholesky, qr, solve,
  svd, tssvd (small factorizations); indexing_routines (advanced-indexing
  patterns); take_pairs (selection-matrix fill).
- **Partially blocked:**
  - einsum: any expression other than the 4 lowered ones.
  - quantiles: integer dtypes, axis=None on N-D.
  - sort/quantiles: uint32.
  - scan/sort/svd: complex dtypes.
  - svd: integer dtypes; full_matrices with m != n.
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
- **cholesky.** Default input is the SPD matrix from cg.py's `generate_random`;
  `--input eye` gives the original identity input. `L21.T` comes from the upper
  block A12 (symmetry), because there is no transpose. Use a block size that
  divides n. Otherwise the frontend's `(K,1)` matmul special case kicks in;
  see the bugs list.
- **einsum.** Inputs are pseudo-random/N instead of ones so that the check is
  meaningful (`--ones` gives the original). B never changes under the A<->C
  swap, so B.T is generated once.
- **indexing_routines.** The ai2 and ai3 final states are fully overwritten
  (all-true masks), as in the original.
- **kmeans / kmeans_slow.** Default precision is 64. A float32 near-tie could
  make argmin differ from the NumPy reference, although the emulation itself
  is exact. About 400 frontend names per iteration, so roughly 150 iterations
  fit in the 65536-name budget.
- **kmeans relabel.** The pairwise "column a beats column b" mask is written
  `(P[:, b] - P[:, a]) > 0` rather than `P[:, b] > P[:, a]`: a fused kernel
  holding only one binary op on two column views of the same array hangs the
  backend inside kmeans_slow (also with a plain `-`), see the comment in
  `kmeans.py`.
- **linreg / logreg.** `--num` is in thousands, default 1 (1000 points).
- **lstm_forward.** Tiling h0 to 3-D avoids a 2-D->3-D set_region at a nonzero
  column offset. See the suspected backend issue below. The existing
  `examples/lstm_forward.py` does exactly that pattern.
- **qr.** Default is 128x32 (the original is 1000x1000 square). Square inputs
  work too (`-m 48 -n 48` is in the static check). `Q@R ~ A` and `Q^T Q ~ I`
  are checked natively.
- **quantiles / sort.** n phases of compare-exchange gives O(n^2) work. Each
  phase writes a fresh buffer, so no op reads and writes the same array. The
  compare-exchange is exact for float and integer keys (no NaN keys).
  argsort is verified by `a[perm] == sort(a)` plus permutation check, since
  ties may order differently from numpy.
- **richardson_lucy.** Assumes cuPyNumeric's N-D `convolve(mode="same")`
  follows the scipy.signal.convolve centering (start index (f-1)//2).
- **scan.** For `axis=None` on 2-D, the port computes the flattened-order scan
  but keeps the 2-D shape, since there is no reshape.
- **solve.** With nrhs == 1, b is kept 1-D so that every product is a matvec.
  This avoids the frontend's `(K,1)` special case.
- **svd / tssvd.** The Gram approach squares the condition number. That is
  fine for these inputs, and the reconstruction check holds regardless.
- **take_pairs.** int32 SUMMA with 0/1 selection matrices (one bulk upload
  each). The result sum is checked exactly.
- **HOST-FALLBACK factors** (cholesky, qr, solve, svd): the small host
  factors are bulk uploads (`U.upload`), no longer one scalar set per nonzero,
  so larger `-n`/block sizes no longer hit the 65536-name budget.

## Missing backend/frontend ops, ranked by examples they would unblock or simplify

1. **Transpose (`.T`/`transpose`) or vecmat (1-D @ 2-D).** 10 examples:
   kmeans, kmeans_slow, linreg, logreg, lstm_backward, einsum, cholesky, qr,
   svd, tssvd. All currently rely on generated transposed inputs or symmetry
   tricks.
2. ~~**Comparisons + `where`/select, `minimum`/`maximum`.**~~ Native since
   2026-10-08 (0/1 masks in the operand dtype). Ports switched: black_scholes,
   black_scholes_greeks, kmeans, kmeans_slow, logreg, sort, quantiles, scan,
   indexing_routines. solve's pivot search still needs argmax.
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
5. **Dense factorization kernels (potrf, getrf with pivoting, geqrf, syevd/gesvd)
   and trsm.** 5 examples: cholesky, solve, qr, svd, tssvd. These remove every
   HOST-FALLBACK there.
6. ~~**Elementwise math: sqrt, log, abs, pow (`__pow__`), floor.**~~ Native
   since 2026-10-08; black_scholes, black_scholes_greeks, logreg, tssvd, cfd,
   lstm_backward and linreg use them.
7. **Gather/scatter: take, fancy and boolean-mask indexing, bincount, row
   permutation.** 4 examples: indexing_routines, take_pairs, kmeans, solve.
8. **reshape/flatten, more than 3 dims, arange on device** (`%` is native).
   4 examples:
   black_scholes_greeks, scan (axis=None), quantiles, take_pairs.
9. **sort/argsort.** 2 examples: sort, quantiles. The O(n^2) network is only
   viable for n up to about 1e3.
10. **astype (dtype conversion).** 3 examples: quantiles (int), svd (int),
    scan.
11. **One example each:** scan (cumsum/cumprod), convolve (richardson_lucy),
    einsum, flip/negative-step slicing (richardson_lucy), general broadcasting
    / newaxis (kmeans), complex dtypes (scan/sort/svd), float16 (gemm,
    black_scholes, linreg, logreg, kmeans).

## Frontend issues found (file:line, paths relative to `charmtyles/`)

1. `example/charmnumeric/charmnumeric/charmnumeric.py:291-292`:
   `np.newaxis`/`None` in an index raises NotImplementedError, although
   `__init__.py:37` exports `newaxis = None`.
2. (fixed) `random.randn` used to return a zero-filled array; `cnp.random` is
   now a native generator.
3. `charmtyles/core.py:9,23-28`: names are never recycled and capped at
   `max_name = 1<<16`. Long iterative programs hit "Too many objects were
   created". (Host data no longer costs one name per element: uploads, arange
   and eye are single nodes now.)
4. `charmnumeric.py:346-358`: `_broadcast_shape` accepts full NumPy
   broadcasting, e.g. (4,8)+(8,). The backend only broadcasts size-1 operands
   (`BROADCAST_OBJECT` in `charmtyles/ast.py:95,101`). Such expressions are
   accepted silently and produce wrong results or crash in the backend.
5. `charmnumeric.py:497-508` (1-D @ 1-D REDUCE), `operations.py:52` (tile) and
   `operations.py:125` (diag) accept views. The backend executors read the
   operands by array name and ignore the view region
   (`src/executor_reduce.cpp:32-42`, `src/execute_node.cpp:601,756`). So
   `x[0:4] @ x[4:8]` or `tile(A[0:2], ...)` silently use the whole array. The
   frontend should materialise views first.
6. `charmnumeric.py:521-522`: 1-D @ 2-D (vecmat, e.g. `error.dot(features)`)
   raises "matmul requires a 2D matrix, got 1D".
7. `charmnumeric.py:526-530` (and 542-545): `A @ B` with B of shape (K,1)
   becomes a matvec. The result has shape (M,) instead of NumPy's (M,1), and
   the rewrite goes through a cross-ndim copy from a view (see the next item).
8. `charmnumeric.py:851-860`: `create_array` accepts more than 3 dims (e.g.
   `zeros((2,2,2,2))`), but the backend only has `Partition1D/2D/3D`.
9. `charmtyles/interface.py:88`: `CCSInterface.connect` has no timeout. With no
   backend listening, a port hangs instead of failing.
10. `charmtyles/ast.py:494`: `_ancestor_bits` uses `1 << node_id` with global,
    never-reset node ids. The ints grow without bound over a long program.
    This is performance, not correctness.

Suspected backend issue, not verified at runtime:
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
`CNUM_SIM=1`. Result (2026-10-08, after the switch to native elementwise
ops): **46/46 PASS, 0 simulator errors, 0 warnings.** Each run uses 75 to
6.4k frontend names. The simulator knows the elementwise opcodes 20-49
(math, comparisons as 0/1 masks, where, clip). This checks:

- frontend graph construction, shapes and dtypes,
- fusion and DAG serialization (every DAG is serialized and handed to
  `run_async`),
- the algorithms and their NumPy verification under the intended charmnumeric
  semantics,
- the absence of patterns the backend is known not to support.

It does **not** run the C++ backend. Backend semantics, such as fused-kernel
aliasing, must be confirmed by real runs (CPU build: see the native-op status
in `MISSING_OPS.md` section 1).
