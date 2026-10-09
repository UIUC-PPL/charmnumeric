"""Shared helpers for the cuPyNumeric -> charmnumeric ports.

Everything here is built from charmnumeric primitives (elementwise math,
comparisons and where, slicing/__setitem__, matvec/matmul/dot, tile, diag).
Nothing runs the algorithm on the host.

Three groups of helpers:

1. Legacy input generation done on the device: index grids (log2(n)
   shifted-slice adds) and a chaotic logistic map. These predate the native
   cnp.random generator and bulk cnp.array upload; the ports that mirror an
   np.random / np.arange call now use those directly (see README). The ports
   fetch their inputs with .get() and compute the NumPy reference from
   exactly those values.

2. Primitives that charmnumeric lacks, marked EMULATED in the ports:
     sums        via matvec / dot with a ones vector
   step/absolute/select/maximum/minimum/sqrt/log/int_mod/greater used to
   be emulated with tanh steps and Newton/Halley iterations. They are now
   thin wrappers over the native ops (comparisons give 0/1 masks in the
   operand dtype) and are kept for compatibility.

3. Verification / reporting (Checker, run).
"""

import sys
import traceback

import numpy as np

import charmnumeric as cnp
from charmnumeric import Array, ArrayView

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def materialize(x):
    """Return a whole Array for x (views are copied).

    The backend REDUCE/TILE/DIAG executors read their inputs by array name and
    ignore view regions, so views must be copied before those ops.
    """
    if isinstance(x, ArrayView):
        return x.copy()
    return x


def unit_vector(n, j, dtype=np.float64):
    e = cnp.zeros((n,), dtype=dtype)
    e[j : j + 1] = 1.0
    return e


def to_float(x, interface):
    """Fetch a 1-element charmnumeric array as a Python float (host sync)."""
    return float(np.asarray(x.get(interface)).reshape(-1)[0])


def allclose(a, b, interface, rtol=1e-05, atol=1e-08):
    """numpy.allclose(a, b) evaluated natively; only the verdict is fetched."""
    ok = cnp.all(cnp.abs(a - b) <= atol + rtol * cnp.abs(b))
    return to_float(ok, interface) != 0.0


def tolerances(dtype):
    """(rtol, atol) for allclose: NumPy's defaults for float64, scaled for float32."""
    if np.dtype(dtype) == np.float32:
        return 1e-04, 1e-05
    return 1e-05, 1e-08


# ---------------------------------------------------------------------------
# native index / pseudo-random generation
# ---------------------------------------------------------------------------
def iota(n, dtype=np.float64):
    """[0, 1, ..., n-1] built natively with ceil(log2 n) shifted-slice adds."""
    a = cnp.zeros((n,), dtype=dtype)
    k = 1
    while k < n:
        m = min(k, n - k)
        a[k : k + m] = a[0:m] + k
        k *= 2
    return a


def index_grid(shape, axis, dtype=np.float64):
    """g[i0, i1, ...] = i_axis, built natively (no host upload)."""
    shape = tuple(shape)
    g = cnp.zeros(shape, dtype=dtype)
    n = shape[axis]
    k = 1
    while k < n:
        m = min(k, n - k)
        dst = [slice(None)] * len(shape)
        src = [slice(None)] * len(shape)
        dst[axis] = slice(k, k + m)
        src[axis] = slice(0, m)
        g[tuple(dst)] = g[tuple(src)] + k
        k *= 2
    return g


def linear_index(shape, perm=None, dtype=np.float64):
    """Row-major linear index of each element of an array of shape `shape`.

    With perm, the result is laid out as np.transpose(L, perm), so
    linear_index(s, perm=(1, 0)) equals linear_index(s).T elementwise. This is
    how the ports obtain X and X.T for input data without a transpose op.
    """
    shape = tuple(shape)
    nd = len(shape)
    if perm is None:
        perm = tuple(range(nd))
    out_shape = tuple(shape[p] for p in perm)
    strides = [1] * nd
    for d in range(nd - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    L = None
    for out_axis, src_axis in enumerate(perm):
        if shape[src_axis] == 1:
            continue
        term = index_grid(out_shape, out_axis, dtype=dtype) * strides[src_axis]  # int: keeps int dtypes
        L = term if L is None else L + term
    if L is None:
        L = cnp.zeros(out_shape, dtype=dtype)
    return L


def uniform_from_index(L, total, seed=0, lo=0.0, hi=1.0, iters=24):
    """Pseudo-random values in [lo, hi) from an index array L.

    A smooth, strictly monotone, seed-dependent warp of L/total into
    (0.05, 0.45), followed by `iters` steps of the chaotic logistic map
    x <- 3.99 x (1 - x). Values stay in [0.0099, 0.9975]. Seeds are kept
    below 0.5 because the map is 2-to-1 (f(x) == f(1-x)). A symmetric seed
    range would make elements L and total-1-L identical.
    """
    # bounded slope + seed-dependent offset: monotone, never saturating
    c = 0.6 * ((seed * 0.6180339887498949) % 1.0) - 0.3
    x = 0.05 + 0.4 * (0.5 + 0.5 * cnp.tanh(2.0 * ((L + 0.5) / float(total) - 0.5) + c))
    for _ in range(iters):
        x = 3.99 * x * (1.0 - x)
    if lo != 0.0 or hi != 1.0:
        x = lo + (hi - lo) * x
    return x


def uniform(shape, seed=0, lo=0.0, hi=1.0, dtype=np.float64, perm=None, iters=24):
    shape = tuple(shape)
    total = int(np.prod(shape))
    L = linear_index(shape, perm=perm, dtype=dtype)
    return uniform_from_index(L, total, seed=seed, lo=lo, hi=hi, iters=iters)


def rand_int(shape, lower, upper, seed=0, dtype=np.int64):
    """Pseudo-random integers in [lower, upper), built natively with integer ops.

    hash(L) = c2*L^2 + c1*L + c0 (mod (upper - lower)), where L is the linear
    index. The mod is the native %.
    Constants are small for int32 so that L <= ~1e4 does not overflow.
    """
    shape = tuple(shape)
    dtype = np.dtype(dtype)
    L = cnp.zeros(shape, dtype=dtype)
    nd = len(shape)
    stride = 1
    for axis in range(nd - 1, -1, -1):
        if shape[axis] > 1:
            L = L + index_grid(shape, axis, dtype=dtype) * int(stride)
        stride *= shape[axis]
    if dtype == np.dtype(np.int32):
        c2, c1, c0 = 7, 131 + 2 * seed, 17 + seed
    else:
        c2, c1, c0 = 7919, 104729 + 31 * seed, 12345 + seed
    h = (L * L) * c2 + L * c1 + c0
    return int_mod(h, int(upper - lower)) + int(lower)


def randn(shape, seed=0, dtype=np.float64, perm=None):
    """Approximately standard-normal values (sum of 4 pseudo-uniform streams).

    np.random.randn would need a Box-Muller transform (log, sqrt, cos), which
    charmnumeric lacks. The distribution shape does not matter for these
    benchmarks.
    """
    shape = tuple(shape)
    total = int(np.prod(shape))
    L = linear_index(shape, perm=perm, dtype=dtype)
    s = None
    for k in range(4):
        u = uniform_from_index(L, total, seed=4 * seed + k + 1)
        s = u if s is None else s + u
    # logistic-map (r=3.99) samples have mean ~0.533 and variance ~0.115
    return (s - 4 * 0.533) * (1.0 / np.sqrt(4 * 0.115))


# ---------------------------------------------------------------------------
# elementwise primitives (formerly emulated; now native wrappers)
# ---------------------------------------------------------------------------
def step(x):
    """Indicator (x > 0) as a 0/1 mask in x's dtype (native comparison)."""
    return x > 0


def absolute(x):
    return cnp.absolute(x)


def select(h, a, b):
    """np.where(h, a, b) (native)."""
    return cnp.where(h, a, b)


def maximum(a, b):
    return cnp.maximum(a, b)


def minimum(a, b):
    return cnp.minimum(a, b)


def sqrt(x):
    return cnp.sqrt(x)


def log(x):
    return cnp.log(x)


def int_step_ge0(d):
    """Indicator (d >= 0) as a 0/1 mask (native comparison)."""
    return d >= 0


def int_mod(x, m):
    """x % m (native; NumPy sign semantics)."""
    return x % m


def greater(a, b):
    """Indicator (a > b) as a 0/1 mask (native comparison, exact)."""
    return a > b


# ---------------------------------------------------------------------------
# reductions built from matvec / dot
# ---------------------------------------------------------------------------
def ones_vec(n, dtype=np.float64):
    return cnp.ones((n,), dtype=dtype)


def vsum(v):
    """Sum of a 1-D array as a 1-element array (dot with ones)."""
    v = materialize(v)
    return v @ ones_vec(v.shape[0], v.dtype)


def row_sums(A):
    """A.sum(axis=1) for a 2-D array (or 3-D view with one singleton dim)."""
    if A.ndim == 3:
        singles = [i for i in range(3) if A.shape[i] == 1]
        cols = [A.shape[i] for i in range(3) if i != singles[0]][1]
    else:
        cols = A.shape[-1]
    return A @ ones_vec(cols, A.dtype)


def msum(A):
    """Sum of all elements of a 2-D array as a 1-element array."""
    rs = row_sums(A)
    return rs @ ones_vec(rs.shape[0], A.dtype)


def sum3(A):
    """Sum of all elements of a 3-D array (loop over the leading axis)."""
    total = None
    for k in range(A.shape[0]):
        s = msum(A[k])
        total = s if total is None else total + s
    return total


# ---------------------------------------------------------------------------
# host -> device upload of small factors (HOST-FALLBACK helper)
# ---------------------------------------------------------------------------
def upload(M, dtype=None, target=None, offset=(0, 0)):
    """Upload a host matrix (one bulk UPLOAD node, see cnp.array).

    If `target` is given, the block goes into target[offset + (0:h, 0:w)].
    (This used to issue one scalar set_region per nonzero.)
    """
    M = np.asarray(M, dtype=dtype if dtype is not None else None)
    if target is None:
        return cnp.array(M)
    r0, c0 = offset
    target[r0 : r0 + M.shape[0], c0 : c0 + M.shape[1]] = M.astype(target.dtype, copy=False)
    return target


# ---------------------------------------------------------------------------
# verification + reporting
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self, name):
        self.name = name
        self.failures = []

    def close(self, label, got, expected, rtol=1e-7, atol=1e-9):
        got = np.asarray(got, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if got.shape != expected.shape:
            if got.size == expected.size:
                got = got.reshape(expected.shape)
            else:
                msg = f"{label}: shape {got.shape} != expected {expected.shape}"
                print("  " + msg)
                self.failures.append(msg)
                return False
        err = np.abs(got - expected)
        max_err = float(np.max(err)) if err.size else 0.0
        denom = np.maximum(np.abs(expected), 1e-300)
        max_rel = float(np.max(err / denom)) if err.size else 0.0
        ok = bool(np.allclose(got, expected, rtol=rtol, atol=atol, equal_nan=True))
        print(
            f"  {label}: max|err|={max_err:.3e} max rel={max_rel:.3e} "
            f"(rtol={rtol:g}, atol={atol:g}) {'ok' if ok else 'MISMATCH'}"
        )
        if not ok:
            self.failures.append(f"{label} mismatch (max|err|={max_err:.3e})")
        return ok

    def true(self, label, cond, detail=""):
        print(f"  {label}: {'ok' if cond else 'FAILED'} {detail}")
        if not cond:
            self.failures.append(f"{label} {detail}".strip())
        return cond

    def finish(self):
        if self.failures:
            print(f"FAIL {self.name}: " + "; ".join(self.failures))
            sys.exit(1)
        print(f"PASS {self.name}")
        sys.exit(0)


def run(name, main):
    """Run main(checker), turning any exception into a FAIL line + exit 1."""
    checker = Checker(name)
    try:
        main(checker)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        msg = str(exc).splitlines()[0] if str(exc) else ""
        print(f"FAIL {name}: {type(exc).__name__}: {msg}")
        sys.exit(1)
    checker.finish()
