"""Index-grid and pseudo-random input generators, plus the Checker/run PASS/FAIL harness."""

import sys
import traceback

import numpy as np

import charmnumeric as cnp


# ---------------------------------------------------------------------------
# native index / pseudo-random generation
# ---------------------------------------------------------------------------
def iota(n, dtype=np.float64):
    """[0, 1, ..., n-1] built with ceil(log2 n) shifted-slice adds."""
    a = cnp.zeros((n,), dtype=dtype)
    k = 1
    while k < n:
        m = min(k, n - k)
        a[k : k + m] = a[0:m] + k
        k *= 2
    return a


def index_grid(shape, axis, dtype=np.float64):
    """g[i0, i1, ...] = i_axis, built on device (no host upload)."""
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
    """Row-major linear index of each element; with perm, equals np.transpose(L, perm)."""
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

    Warps L/total into (0.05, 0.45), then applies `iters` steps of the logistic
    map x <- 3.99 x (1 - x). Seeds must stay below 0.5: the map is 2-to-1
    (f(x) == f(1-x)), so a symmetric seed range makes L and total-1-L identical.
    """
    c =0.6 * ((seed * 0.6180339887498949) % 1.0) - 0.3
    x = 0.05 + 0.4 * (0.5 + 0.5 * cnp.tanh(2.0 * ((L + 0.5) / float(total) - 0.5) + c))
    for _ in range(iters):
        x = 3.99 * x * (1.0 - x)
    if lo != 0.0 or hi != 1.0:
        x = lo + (hi - lo) * x
    return x


def uniform(shape, seed=0, lo=0.0, hi=1.0, dtype=np.float64, perm=None, iters=24):
    if np.dtype(dtype) == np.float16:  # generate in float32, round once
        return uniform(shape, seed, lo, hi, np.float32, perm, iters).astype(np.float16)
    shape = tuple(shape)
    total = int(np.prod(shape))
    L = linear_index(shape, perm=perm, dtype=dtype)
    return uniform_from_index(L, total, seed=seed, lo=lo, hi=hi, iters=iters)


def int_mod(x, m):
    """x % m (native; NumPy sign semantics)."""
    return x % m


# ---------------------------------------------------------------------------
# verification + reporting
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self, name):
        self.name = name
        self.failures = []

    def close(self, label, got, expected, rtol=1e-7, atol=1e-9):
        cplx = np.iscomplexobj(got) or np.iscomplexobj(expected)
        got = np.asarray(got, dtype=np.complex128 if cplx else np.float64)
        expected = np.asarray(expected, dtype=got.dtype)
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
