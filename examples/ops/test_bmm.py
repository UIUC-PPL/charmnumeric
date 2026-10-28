#!/usr/bin/env python
"""Tests for the native batched matmul (BATCHED_MATMUL) behind einsum and N-D matmul.

Covers every operand layout (batch first or middle, transposed operands), views
(offset, stepped), multi-chare sizes, chained results and the BLAS dtypes.
Env: CNUM_HOST, CNUM_PORT, CNUM_ODF, or CNUM_SIM=1.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
from _ops_common import Checker, connect  # noqa: E402

_RNG = np.random.default_rng(11)


def host(shape, dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return _RNG.integers(0, 2, size=shape).astype(dtype)
    if dtype.kind in "iu":
        return _RNG.integers(-5 if dtype.kind == "i" else 0, 6, size=shape).astype(dtype)
    if dtype.kind == "c":
        return (_RNG.uniform(-1.0, 1.0, size=shape)
                + 1j * _RNG.uniform(-1.0, 1.0, size=shape)).astype(dtype)
    return _RNG.uniform(-1.0, 1.0, size=shape).astype(dtype)


def tol(dtype, k):
    dtype = np.dtype(dtype)
    if dtype.kind in "biu":
        return 0.0, 0.0
    eps = {2: 2e-3, 4: 2e-6, 8: 2e-6}.get(dtype.itemsize, 1e-13)
    return eps * 10, eps * 10 * max(1.0, np.sqrt(k))


# (subscripts, shapes): every layout of a (batch, m, k) x (batch, k, n) contraction
LAYOUTS = [
    ("bij,bjk->bik", [(6, 30, 40), (6, 40, 20)]),
    ("bij,bkj->bik", [(6, 30, 40), (6, 20, 40)]),
    ("bji,bjk->bik", [(6, 40, 30), (6, 40, 20)]),
    ("bji,bkj->bik", [(6, 40, 30), (6, 20, 40)]),
    ("ibj,bjk->bik", [(30, 6, 40), (6, 40, 20)]),
    ("jbi,jbk->bik", [(40, 6, 30), (40, 6, 20)]),
    ("bij,jbk->bki", [(6, 30, 40), (40, 6, 20)]),
    ("ijb,bjk->bik", [(30, 40, 6), (6, 40, 20)]),
    ("kac,kbc->kab", [(9, 10, 11), (9, 12, 11)]),
    ("kac,kbc->abk", [(9, 10, 11), (9, 12, 11)]),
    # grouped labels (>3-D operands) and two batch labels
    ("abij,abjk->abik", [(2, 3, 20, 30), (2, 3, 30, 10)]),
    ("bimj,bjn->bimn", [(4, 5, 6, 7), (4, 7, 8)]),
    ("bcij,cbjk->bcik", [(3, 4, 9, 10), (4, 3, 10, 11)]),
    ("bij,bjk", [(4, 5, 6), (4, 6, 7)]),
]

# multi-chare sizes (3-D tiles are 64 on CPU, 128 on GPU), not tile-aligned
BIG = [
    ("bij,bjk->bik", [(70, 130, 90), (70, 90, 110)], "float64"),
    ("kac,kbc->kab", [(100, 100, 100), (100, 100, 100)], "float32"),
    ("bij,bkj->bik", [(3, 150, 200), (3, 140, 200)], "float64"),
    ("ibj,jbk->bik", [(150, 3, 140), (140, 3, 130)], "float64"),
]

DTYPES = ("float64", "float32", "complex64", "complex128", "int64", "int32", "int8", "bool",
          "float16")


def check(chk, label, expr, hs, ops, dtype=None):
    got = cnp.einsum(expr, *ops)
    want = np.asarray(np.einsum(expr, *hs))
    k = max(1, int(np.prod([h.size for h in hs])) // max(1, want.size))
    rtol, atol = tol(want.dtype, k)
    chk.check(label, got, want, rtol=rtol, atol=atol, dtype=want.dtype)


def main():
    iface = connect()
    chk = Checker("test_bmm", iface)

    print("layouts")
    for expr, shapes in LAYOUTS:
        hs = [host(s, "float64") for s in shapes]
        check(chk, f"{expr} {shapes}", expr, hs, [cnp.array(h) for h in hs])

    print("multi-chare")
    for expr, shapes, dt in BIG:
        hs = [host(s, dt) for s in shapes]
        check(chk, f"{expr} {shapes} {dt}", expr, hs, [cnp.array(h) for h in hs])

    print("dtypes")
    for dt in DTYPES:
        for expr, shapes in (("bij,bjk->bik", [(7, 9, 11), (7, 11, 5)]),
                             ("kac,kbc->kab", [(70, 66, 65), (70, 67, 65)])):
            hs = [host(s, dt) for s in shapes]
            check(chk, f"{expr} {dt}", expr, hs, [cnp.array(h) for h in hs])

    print("views")
    ha, hb = host((80, 70, 90), "float64"), host((80, 90, 60), "float64")
    A, B = cnp.array(ha), cnp.array(hb)
    for ka, kb in (((slice(3, 50), slice(5, 45), slice(10, 70)),
                    (slice(3, 50), slice(10, 70), slice(0, 33))),
                   ((slice(None, None, 2), slice(1, 60), slice(None, None, 3)),
                    (slice(None, None, 2), slice(None, None, 3), slice(7, 60, 2))),
                   ((slice(70, 76), slice(None), slice(80, 90)),
                    (slice(70, 76), slice(80, 90), slice(None)))):
        check(chk, f"views {ka} {kb}", "bij,bjk->bik", [ha[ka], hb[kb]], [A[ka], B[kb]])
    check(chk, "transposed view", "bji,bkj->bik", [ha[:, 5:45, :], hb[:, 2:30, 10:50]],
          [A[:, 5:45, :], B[:, 2:30, 10:50]])

    print("matmul")
    for sa, sb in (((6, 30, 40), (6, 40, 20)), ((2, 3, 20, 30), (2, 3, 30, 10)),
                   ((2, 1, 6, 7), (3, 7, 5)), ((70, 130, 90), (70, 90, 110))):
        ha, hb = host(sa, "float64"), host(sb, "float64")
        chk.check(f"{sa} @ {sb}", cnp.matmul(cnp.array(ha), cnp.array(hb)), ha @ hb,
                  rtol=1e-12, atol=1e-11)
    ha, hb = host((5, 30, 40), "float32"), host((5, 40, 20), "float32")
    M = cnp.zeros((5, 30, 20), dtype="float32")
    cnp.matmul(cnp.array(ha), cnp.array(hb), out=M)
    chk.check("matmul out=", M, ha @ hb, rtol=2e-5, atol=2e-5)

    print("chained (results as operands, out=)")
    ha, hb = host((100, 30, 30), "float64"), host((100, 30, 30), "float64")
    A, B = cnp.array(ha), cnp.array(hb)
    C = cnp.zeros((100, 30, 30), dtype="float64")
    for _ in range(4):
        cnp.einsum("kac,kbc->kab", A, B, out=C)
        A, C = C, A
        ha = np.einsum("kac,kbc->kab", ha, hb)
    chk.check("chained einsum", A, ha, rtol=1e-10, atol=1e-10)
    X = cnp.array(host((50, 20, 20), "float64"))
    hx = chk.fetch(X)
    Y = X
    hy = hx
    for _ in range(3):
        Y = (Y @ X) * 0.5 + 1.0
        hy = (hy @ hx) * 0.5 + 1.0
    chk.check("chained matmul with elementwise", Y, hy, rtol=1e-10, atol=1e-10)

    chk.finish()


if __name__ == "__main__":
    main()
