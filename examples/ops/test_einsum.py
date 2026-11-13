#!/usr/bin/env python
"""Tests for cnp.einsum / einsum_path / tensordot / inner / outer against NumPy.

Inputs are uploaded host arrays, so the NumPy reference uses the same values.
Env: CNUM_HOST, CNUM_PORT, CNUM_ODF, or CNUM_SIM=1.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
from _ops_common import Checker, connect  # noqa: E402

_RNG = np.random.default_rng(7)


def host(shape, dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return _RNG.integers(0, 2, size=shape).astype(dtype)
    if dtype.kind == "u":
        return _RNG.integers(0, 6, size=shape).astype(dtype)
    if dtype.kind == "i":
        return _RNG.integers(-5, 6, size=shape).astype(dtype)
    if dtype.kind == "c":
        return (_RNG.uniform(-1.0, 1.0, size=shape)
                + 1j * _RNG.uniform(-1.0, 1.0, size=shape)).astype(dtype)
    return _RNG.uniform(-1.0, 1.0, size=shape).astype(dtype)


def tol(dtype, k=1):
    """rtol/atol for a result with contraction length ~k (atol scales with sqrt(k))."""
    dtype = np.dtype(dtype)
    if dtype.kind in "biu":
        return 0.0, 0.0
    eps = {2: 2e-3, 4: 2e-6, 8: 2e-6}.get(dtype.itemsize, 1e-13)
    return eps * 10, eps * 10 * max(1.0, np.sqrt(k))


# (subscripts, operand shapes, dtypes to test)
F64 = ("float64",)
F32 = ("float32",)
I64 = ("int64",)
ALL = ("float64", "float32", "int64")

CASES = [
    # 2-D matmul: every orientation, multi-chare
    ("ij,jk->ik", [(600, 70), (70, 530)], ALL),
    ("ij,kj->ik", [(37, 41), (29, 41)], ALL),
    ("ji,jk->ik", [(41, 37), (41, 29)], F64),
    ("ji,kj->ki", [(41, 37), (29, 41)], F64),
    ("ij,jk->ki", [(37, 41), (41, 29)], F64),
    ("ij,jk", [(530, 41), (41, 37)], F32),
    ("jk,ij", [(41, 37), (530, 41)], F64),
    # matvec / vecmat / dot
    ("ij,j->i", [(600, 37), (37,)], ALL),
    ("ji,j->i", [(37, 600), (37,)], F64),
    ("i,ij->j", [(37,), (37, 530)], F64),
    ("i,i->", [(300001,), (300001,)], F64),
    ("i,i", [(1000,), (1000,)], I64),
    # batched
    ("bij,bjk->bik", [(5, 30, 40), (5, 40, 20)], ALL),
    ("kac,kbc->kab", [(70, 66, 65), (70, 67, 65)], F64),
    ("bij,bjk->bik", [(3, 130, 70), (3, 70, 90)], F32),
    ("ibj,jbk->bik", [(30, 4, 20), (20, 4, 10)], F64),
    ("bij,bjk->bik", [(100, 4, 5), (100, 5, 3)], ALL),
    ("bij,bj->bi", [(70, 33, 65), (70, 65)], F64),
    ("bj,bjk->bk", [(70, 65), (70, 65, 33)], F64),
    ("bi,bi->b", [(70, 65), (70, 65)], F64),
    ("bik,bjk->bij", [(65, 7, 9), (65, 6, 9)], I64),
    # traces and diagonals
    ("ii->", [(530, 530)], ALL),
    ("ii", [(37, 37)], F64),
    ("ii->i", [(530, 530)], F64),
    ("iij->ij", [(9, 9, 70)], F64),
    ("iji->ij", [(9, 70, 9)], I64),
    ("iii->i", [(13, 13, 13)], F64),
    ("ijj,jk->ik", [(5, 6, 6), (6, 7)], F64),
    # implicit output
    ("ji", [(37, 41)], F64),
    ("ij", [(37, 41)], F64),
    ("ba,cb", [(5, 6), (7, 5)], F64),
    # ellipsis
    ("...ij,...jk->...ik", [(4, 30, 20), (4, 20, 10)], F64),
    ("...ij,jk", [(2, 3, 30, 20), (20, 10)], F64),
    ("...i,...i", [(2, 1, 3), (5, 3)], F64),
    ("i...->...", [(70, 66, 5)], F64),
    ("...ii->...i", [(5, 9, 9)], F64),
    ("ij...,jk...->ik...", [(6, 7, 3), (7, 5, 3)], F64),
    # 3 and 4 operands
    ("ij,jk,kl->il", [(30, 40), (40, 50), (50, 20)], ALL),
    ("i,ij,j->", [(37,), (37, 41), (41,)], F64),
    ("ab,bc,cd", [(5, 6), (6, 7), (7, 8)], F32),
    ("ij,jk,kl,lm->im", [(8, 9), (9, 10), (10, 11), (11, 12)], F64),
    ("ijk,jl,kl->il", [(6, 7, 8), (7, 9), (8, 9)], F64),
    ("bij,bjk,bkl->bil", [(3, 4, 5), (3, 5, 6), (3, 6, 7)], F64),
    # outer products
    ("i,j->ij", [(600,), (530,)], ALL),
    ("ij,k->ijk", [(5, 6), (70,)], F64),
    ("i,j,k->ijk", [(5,), (6,), (7,)], F64),
    # sums and transposes of one operand
    ("ij->", [(600, 37)], ALL),
    ("ij->i", [(600, 37)], F64),
    ("ij->j", [(600, 37)], I64),
    ("ijk->jk", [(70, 6, 5)], F64),
    ("ijk->ki", [(70, 6, 5)], F64),
    ("ij->ji", [(600, 530)], F64),
    ("ijk->kji", [(70, 6, 5)], F64),
    ("ijkl->ljik", [(3, 4, 5, 6)], F64),
    ("ijkl->jl", [(3, 4, 5, 6)], F64),
    # elementwise forms and size-1 broadcasting
    ("ij,ij->ij", [(600, 37), (600, 37)], ALL),
    ("ij,ij->", [(600, 37), (600, 37)], F64),
    ("ij,ji->ij", [(37, 41), (41, 37)], F64),
    ("ij,jk", [(5, 1), (3, 4)], F64),
    ("ij,ij->ij", [(5, 1), (5, 4)], F64),
    # >3-D operands
    ("abcd,cdef->abef", [(2, 3, 4, 5), (4, 5, 3, 2)], F64),
    ("abcd,dc->ab", [(3, 4, 5, 6), (6, 5)], F64),
    ("abcd->dcba", [(2, 3, 4, 5)], F64),
    ("abcd,abcd->bd", [(2, 3, 4, 5), (2, 3, 4, 5)], I64),
    # int32 keeps its dtype (no int64 accumulation, as in NumPy's einsum)
    ("ij,jk->ik", [(37, 41), (41, 29)], ("int32",)),
    ("bij,bjk->bik", [(100, 4, 5), (100, 5, 3)], ("int32",)),
    ("ii->", [(37, 37)], ("int32",)),
    ("ij,j->i", [(600, 37), (37,)], ("int32",)),
    # bool (or of ands), small and unsigned ints (wrap like NumPy), float16, complex
    ("ij,jk->ik", [(37, 41), (41, 29)], ("bool", "int8", "uint16", "float16", "complex64",
                                         "complex128")),
    ("bij,bjk->bik", [(100, 4, 5), (100, 5, 3)], ("bool", "int8", "complex64")),
    ("ii->i", [(37, 37)], ("bool", "uint16", "complex128")),
    ("ij->", [(37, 41)], ("bool", "int8", "float16")),
    ("i,j->ij", [(37,), (29,)], ("bool", "uint16", "complex64")),
    ("ij,j->i", [(600, 37), (37,)], ("bool", "float16")),
    ("abcd,dc->ab", [(3, 4, 5, 6), (6, 5)], ("bool", "int32")),
]


def check_case(chk, iface, expr, shapes, dtype):
    hs = [host(s, dtype) for s in shapes]
    ops = [cnp.array(h) for h in hs]
    got = cnp.einsum(expr, *ops)
    want = np.asarray(np.einsum(expr, *hs))
    k = max(1, int(np.prod([np.prod(s) for s in shapes])) // max(1, want.size))
    rtol, atol = tol(dtype, k)
    chk.check(f"{expr} {shapes} {dtype}", got, want, rtol=rtol, atol=atol, dtype=want.dtype)


def main():
    iface = connect()
    chk = Checker("test_einsum", iface)

    print("einsum expressions")
    for expr, shapes, dtypes in CASES:
        for dt in dtypes:
            check_case(chk, iface, expr, shapes, dt)

    print("optimize / paths")
    hs = [host(s, "float64") for s in [(30, 40), (40, 5), (5, 60)]]
    ops = [cnp.array(h) for h in hs]
    want = np.einsum("ij,jk,kl->il", *hs)
    for opt in (False, True, "greedy", "optimal", ["einsum_path", (1, 2), (0, 1)]):
        chk.check(f"optimize={opt!r}", cnp.einsum("ij,jk,kl->il", *ops, optimize=opt), want,
                  rtol=1e-12, atol=1e-12)
    path, _ = cnp.einsum_path("ij,jk,kl->il", *ops, optimize="greedy")
    chk.check("einsum_path", np.asarray(path[1:]), np.asarray(
        np.einsum_path("ij,jk,kl->il", *hs, optimize="greedy")[0][1:]))

    print("sublist form, scalars, out=, dtype=")
    a, b = host((37, 41), "float64"), host((41, 29), "float64")
    A, B = cnp.array(a), cnp.array(b)
    chk.check("sublist", cnp.einsum(A, [0, 1], B, [1, 2], [0, 2]), a @ b, rtol=1e-12, atol=1e-12)
    chk.check("sublist implicit+ellipsis", cnp.einsum(A, [Ellipsis, 1], b[:, 0].copy(), [1]),
              a @ b[:, 0], rtol=1e-12, atol=1e-12)
    chk.check("scalar operand", cnp.einsum("ij,,jk->ik", A, 2.5, B),
              np.einsum("ij,,jk->ik", a, 2.5, b), rtol=1e-12, atol=1e-12)
    C = cnp.zeros((37, 29))
    r = cnp.einsum("ij,jk->ik", A, B, out=C)
    chk.check("out= returns out", np.asarray([r is C]), np.asarray([True]))
    chk.check("out= value", C, a @ b, rtol=1e-12, atol=1e-12)
    Ct = cnp.zeros((29, 37))
    cnp.einsum("ij,jk->ki", A, B, out=Ct)
    chk.check("out= transposed", Ct, (a @ b).T, rtol=1e-12, atol=1e-12)
    a32 = host((37, 41), "float32")
    chk.check("dtype=float64", cnp.einsum("ij,jk", cnp.array(a32), B, dtype="float64"),
              np.einsum("ij,jk", a32, b, dtype="float64"), rtol=1e-6, atol=1e-6, dtype="float64")
    ai, bf = host((37, 41), "int64"), host((41, 29), "float32")
    chk.check("int64 x float32 -> float64", cnp.einsum("ij,jk", cnp.array(ai), cnp.array(bf)),
              np.einsum("ij,jk", ai, bf), rtol=1e-6, atol=1e-5, dtype="float64")
    chk.raises("dtype cast check", TypeError,
               lambda: cnp.einsum("ij,jk", A, B, dtype="float32"))
    chk.raises("bad subscripts", ValueError, lambda: cnp.einsum("ij,jk->il", A, B))
    chk.raises("shape mismatch", ValueError, lambda: cnp.einsum("ij,jk", A, A))
    chk.raises("operand count", ValueError, lambda: cnp.einsum("ij,jk", A))
    chk.raises("ellipsis dims missing from the output", ValueError,
               lambda: cnp.einsum("...ij->ij", cnp.array(host((2, 3, 4), "float64"))))

    print("tensordot / inner / outer")
    x, y = host((7, 6, 70), "float64"), host((6, 70, 5), "float64")
    X, Y = cnp.array(x), cnp.array(y)
    for axes in (2, ([1, 2], [0, 1]), ([2, 1], [1, 0]), ([2], [1])):
        chk.check(f"tensordot axes={axes}", cnp.tensordot(X, Y, axes=axes),
                  np.tensordot(x, y, axes=axes), rtol=1e-12, atol=1e-12)
    m, n = host((600, 37), "float32"), host((37, 41), "float32")
    chk.check("tensordot axes=1", cnp.tensordot(cnp.array(m), cnp.array(n), axes=1),
              np.tensordot(m, n, axes=1), rtol=2e-5, atol=2e-5)
    p, q = host((5, 6), "int64"), host((7,), "int64")
    chk.check("tensordot axes=0", cnp.tensordot(cnp.array(p), cnp.array(q), axes=0),
              np.tensordot(p, q, axes=0))
    chk.check("tensordot 4-D", cnp.tensordot(X, cnp.array(y[:, :, :2].copy()), axes=([1], [0])),
              np.tensordot(x, y[:, :, :2], axes=([1], [0])), rtol=1e-12, atol=1e-12)
    u, v = host((300001,), "float64"), host((300001,), "float64")
    chk.check("inner 1-D", cnp.inner(cnp.array(u), cnp.array(v)),
              np.asarray(np.inner(u, v)), rtol=1e-10, atol=1e-10)
    s, t = host((37, 41), "int64"), host((29, 41), "int64")
    chk.check("inner 2-D", cnp.inner(cnp.array(s), cnp.array(t)), np.inner(s, t))
    chk.check("inner 3-D x 2-D", cnp.inner(X, cnp.ones((3, 70))),
              np.inner(x, np.ones((3, 70))), rtol=1e-12, atol=1e-12)
    chk.check("inner scalar", cnp.inner(cnp.array(s), 3), np.inner(s, 3))
    w = host((600,), "float64")
    chk.check("outer", cnp.outer(cnp.array(w), cnp.array(u[:530].copy())),
              np.outer(w, u[:530]), rtol=1e-12, atol=1e-12)
    chk.check("outer 2-D inputs", cnp.outer(cnp.array(p), cnp.array(q)), np.outer(p, q))
    O = cnp.zeros((30, 35), dtype="int64")
    cnp.outer(cnp.array(p), cnp.array(q[:5].copy().repeat(7)), out=O)
    chk.check("outer out=", O, np.outer(p, q[:5].repeat(7)))

    print("batched matmul (@ / cnp.matmul on N-D operands)")
    for sa, sb in (((5, 30, 40), (5, 40, 20)), ((70, 66, 65), (65, 33)), ((65,), (4, 65, 7)),
                   ((4, 6, 7), (7,)), ((2, 1, 6, 7), (3, 7, 5)), ((2, 3, 4, 5), (2, 3, 5, 6))):
        ha, hb = host(sa, "float64"), host(sb, "float64")
        chk.check(f"{sa} @ {sb}", cnp.array(ha) @ cnp.array(hb), ha @ hb, rtol=1e-12, atol=1e-12)
    ha, hb = host((5, 30, 40), "int64"), host((5, 40, 20), "int64")
    M = cnp.zeros((5, 30, 20), dtype="int64")
    cnp.matmul(cnp.array(ha), cnp.array(hb), out=M)
    chk.check("matmul out= batched int64", M, ha @ hb)

    print("mixed dtypes")
    hb, hf = host((37, 41), "bool"), host((41, 29), "float32")
    chk.check("bool @ float32", cnp.array(hb) @ cnp.array(hf), hb @ hf, rtol=1e-5, atol=1e-5,
              dtype=np.float32)
    hi, hu = host((37, 41), "int32"), host((41,), "uint16")
    chk.check("einsum int32,uint16", cnp.einsum("ij,j->i", cnp.array(hi), cnp.array(hu)),
              np.einsum("ij,j->i", hi, hu), dtype=np.int32)
    chk.check("tensordot bool", cnp.tensordot(cnp.array(hb), cnp.array(hb.T), axes=1),
              np.tensordot(hb, hb.T, axes=1), dtype=np.bool_)
    h8a, h8b = host((7, 5), "int8"), host((3, 5), "int8")
    chk.check("inner int8", cnp.inner(cnp.array(h8a), cnp.array(h8b)), np.inner(h8a, h8b),
              dtype=np.int8)
    hba = host((2, 3, 4, 5), "bool")
    hbb = host((2, 3, 5, 2), "bool")
    chk.check("N-D @ bool", cnp.array(hba) @ cnp.array(hbb), hba @ hbb, dtype=np.bool_)

    print("views and >3-D transpose")
    big = host((600, 70), "float64")
    BIG = cnp.array(big)
    c57 = host((57, 29), "float64")
    chk.check("view operand", cnp.einsum("ij,jk->ik", BIG[10:500:2, 3:60], cnp.array(c57)),
              big[10:500:2, 3:60] @ c57, rtol=1e-12, atol=1e-12)
    chk.check("matmul of a strided view", BIG[10:500:2, 3:60] @ cnp.array(c57),
              big[10:500:2, 3:60] @ c57, rtol=1e-12, atol=1e-12)
    chk.check("view diag", cnp.einsum("ii->i", BIG[5:65, 1:61]), np.diag(big[5:65, 1:61]),
              rtol=0, atol=0)
    z = host((2, 3, 4, 5), "float64")
    chk.check("transpose 4-D", cnp.transpose(cnp.array(z), (2, 0, 3, 1)),
              np.transpose(z, (2, 0, 3, 1)))
    chk.check("FoldedArray.T", cnp.array(z).T, z.T)

    chk.finish()


if __name__ == "__main__":
    main()
