#!/usr/bin/env python
"""Executor edge cases: bincount with weights laid out unlike x, sparse/dense bincount
partials, scaled 2-norms, matmul with dimension-dropped 3-D operands, uploads (set
CNUM_UPLOAD_STAGE_SHIFT=1 on the server to stage tiles away from their chares), and
set_region from a size-1 array of another rank.

Run as e.g. harness/run_one.sh server.out 2 33001 examples/ops/test_exec_robust.py
"""

import numpy as np

import charmnumeric as cnp
import charmnumeric.charmnumeric as cm
from charmtyles.core import execute
from _ops_common import Checker, connect


def bincount_layouts(chk, iface, rng):
    for n in (37, 1000, 300001, 600011):
        xv = rng.integers(0, 50, n)
        wv = rng.standard_normal(n)
        big = rng.standard_normal(3 * n + 11)
        X = cnp.array(xv)
        B = cnp.array(big)
        execute(iface)
        ref = np.bincount(xv, weights=wv, minlength=50)
        chk.check(f"n={n} plain", cnp.bincount(X, weights=cnp.array(wv), minlength=50), ref,
                  rtol=1e-12, atol=1e-9)
        # weights derived from a shifted slice of a larger array (other tile and offset)
        W1 = B[7:7 + n] * 1.0
        chk.check(f"n={n} shifted weights",
                  cnp.bincount(X, weights=W1, minlength=50),
                  np.bincount(xv, weights=big[7:7 + n], minlength=50), rtol=1e-12, atol=1e-9)
        W2 = B[n + 3:2 * n + 3] + 0.0
        execute(iface)
        X2 = cnp.array(np.concatenate([xv, xv]))[n:] + 0
        chk.check(f"n={n} both shifted",
                  cnp.bincount(X2, weights=W2, minlength=50),
                  np.bincount(xv, weights=big[n + 3:2 * n + 3], minlength=50), rtol=1e-12,
                  atol=1e-9)
        chk.check(f"n={n} int weights", cnp.bincount(X, weights=cnp.array(xv[::-1].copy()) + 0,
                                                     minlength=50),
                  np.bincount(xv, weights=xv[::-1], minlength=50))


def bincount_sparse(chk, iface, rng):
    n = 700001
    for hi, minlength in ((50, 3_000_000), (2_900_000, 3_000_000), (3_000_000, 10)):
        xv = rng.integers(0, hi, n)
        if hi == 50:
            xv[::1000] = 2_999_999  # a few far bins
        X = cnp.array(xv)
        ref = np.bincount(xv, minlength=minlength)
        L = max(minlength, ref.shape[0])
        chk.check(f"counts hi={hi} len={L}", cnp.bincount(X, minlength=L),
                  np.bincount(xv, minlength=L))
        wv = rng.standard_normal(n)
        chk.check(f"weights hi={hi} len={L}", cnp.bincount(X, weights=cnp.array(wv), minlength=L),
                  np.bincount(xv, weights=wv, minlength=L), rtol=1e-12, atol=1e-9)
    xv = rng.integers(-5, 5, 1000)
    chk.check("out of range only", cnp.bincount(cnp.array(xv) - 10, minlength=7),
              np.zeros(7, np.int64))


def scaled_norm(x, axis=None):
    m = np.max(np.abs(x), axis=axis, keepdims=True)
    m = np.where(m == 0, 1.0, m)
    r = np.squeeze(m, axis=axis) * np.linalg.norm(x / m, axis=axis)
    return r


def norms(chk, iface, rng):
    for scale in (1e200, 1e-200, 1e300, 1e-310):
        v = rng.standard_normal(600011) * scale
        chk.check(f"norm scale={scale:g}", cnp.linalg.norm(cnp.array(v)), scaled_norm(v),
                  rtol=1e-12)
    m = rng.standard_normal((300, 700))
    m[:, :5] *= 1e250
    m[:7] *= 1e-250
    M = cnp.array(m)
    chk.check("norm mixed fro", cnp.linalg.norm(M), scaled_norm(m.ravel()), rtol=1e-12)
    chk.check("norm mixed axis=1", cnp.linalg.norm(M, axis=1), scaled_norm(m, axis=1),
              rtol=1e-12)
    chk.check("norm mixed axis=0", cnp.linalg.norm(M, axis=0), scaled_norm(m, axis=0),
              rtol=1e-12)
    v = rng.standard_normal(1000)
    chk.check("norm normal values", cnp.linalg.norm(cnp.array(v)), np.sqrt(np.sum(v * v)),
              rtol=1e-15)
    for special in ([np.inf, 1.0, 2.0], [np.inf, np.inf], [np.nan, np.inf, 1e300],
                    [0.0, 0.0], [5e-324, 1e-320]):
        s = np.array(special)
        with np.errstate(all="ignore"):
            exp = scaled_norm(s) if np.all(np.isfinite(s)) else np.linalg.norm(s)
        chk.check(f"norm {special}", cnp.linalg.norm(cnp.array(s)), exp, rtol=1e-12)


def matmul_dim_drop(chk, iface, rng):
    def sq(s):
        return tuple(d for d in s if d != 1)
    for sa, sb in [((50, 1, 40), (40, 30)), ((1, 50, 40), (40, 30)), ((50, 40, 1), (40, 30)),
                   ((50, 40), (40, 1, 30)), ((50, 40), (1, 40, 30)), ((1, 50, 40), (40, 30, 1)),
                   ((300, 1, 600), (600, 500)), ((300, 600), (600, 1, 500)),
                   ((1, 300, 600), (600, 20)), ((3000, 1, 600), (600, 20))]:
        a, b = rng.standard_normal(sa), rng.standard_normal(sb)
        A, B = cnp.array(a), cnp.array(b)
        exp = a.reshape(sq(sa) if len(sa) == 3 else sa) @ b.reshape(sq(sb) if len(sb) == 3 else sb)
        chk.check(f"{sa} @ {sb}", cnp.reshape(A @ B, exp.shape), exp, rtol=1e-10, atol=1e-10)
        if len(sa) == 3:
            x = rng.standard_normal(exp.shape[0] and a.reshape(sq(sa)).shape[1])
            chk.check(f"{sa} @ vec", A @ cnp.array(x), a.reshape(sq(sa)) @ x, rtol=1e-10,
                      atol=1e-10)
    # (1, K, 1) operands: the dropped dim is fixed by the other operand (wire-level test)
    K = 40
    a = rng.standard_normal((2, K, 3))
    A = cnp.array(a)
    V = A[1:2, :, 1:2]
    x = rng.standard_normal(K)
    r = cm.nonfusible_op(V, cnp.array(x), operation=cm.ArrayOperation.matmul, shape=(1,),
                         dtype=np.dtype(np.float64))
    chk.check("(1,K,1) @ vec", r, a[1, :, 1][None, :] @ x, rtol=1e-12, atol=1e-12)
    b = rng.standard_normal((K, 30))
    r = cm.nonfusible_op(V, cnp.array(b), operation=cm.ArrayOperation.matmatmul, shape=(1, 30),
                         dtype=np.dtype(np.float64))
    chk.check("(1,K,1) @ (K,30)", r, a[1, :, 1][None, :] @ b, rtol=1e-12, atol=1e-12)
    c = rng.standard_normal((20, 1))
    r = cm.nonfusible_op(cnp.array(c), V, operation=cm.ArrayOperation.matmatmul, shape=(20, K),
                         dtype=np.dtype(np.float64))
    chk.check("(20,1) @ (1,K,1)", r, c @ a[1, :, 1][None, :], rtol=1e-12, atol=1e-12)


def uploads(chk, iface, rng):
    for shape in [(37,), (600011,), (37, 41), (1500, 1100), (5, 7, 9), (70, 65, 130)]:
        h = rng.standard_normal(shape)
        chk.check(f"upload {shape}", cnp.array(h), h)
        X = cnp.array(h)
        execute(iface)
        chk.check(f"upload {shape} then op", X * 2.0 + cnp.array(h), 3.0 * h, rtol=1e-15)
    hi = rng.integers(-9, 9, (300, 7)).astype(np.int32)
    chk.check("upload int32", cnp.array(hi), hi)


def broadcast_set_region(chk, iface, rng):
    """set_region whose rhs is a whole size-1 array of another rank (a scalar fill)."""
    for tshape, rshape in [((37, 41), (1,)), ((1500, 1100), (1,)), ((5, 7, 9), (1, 1)),
                           ((1000,), (1, 1)), ((70, 65, 130), (1,))]:
        T = cnp.zeros(tshape)
        s = cnp.array(np.full(rshape, 5.0))
        execute(iface)
        T.set_region(cm._full_region(tshape), s, shape=tshape, dtype=T.dtype)
        chk.check(f"fill {tshape} <- {rshape}", T, np.full(tshape, 5.0))
    T = cnp.zeros((37, 41))
    s = cnp.array(np.array([3.0]))
    execute(iface)
    U = T + 1.0
    U[2:9, 3:30].set_region(cm._full_region((7, 27)), s, shape=(7, 27), dtype=T.dtype)
    e = np.ones((37, 41))
    e[2:9, 3:30] = 3.0
    chk.check("fill sub-region then op", U * 2.0, 2.0 * e)


def main():
    iface = connect()
    chk = Checker("test_exec_robust", iface)
    rng = np.random.default_rng(7)
    bincount_layouts(chk, iface, rng)
    bincount_sparse(chk, iface, rng)
    norms(chk, iface, rng)
    matmul_dim_drop(chk, iface, rng)
    uploads(chk, iface, rng)
    broadcast_set_region(chk, iface, rng)
    chk.finish()


if __name__ == "__main__":
    main()
