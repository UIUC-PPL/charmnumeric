#!/usr/bin/env python
"""Tests for the dense matmul paths: SUMMA (2-D @ 2-D), matvec, vecmat and dot.

Covers multi-chare and non-tile-aligned sizes, skinny operands, views and slices,
3-D operands with a dropped dim, out=, chained results and the BLAS dtypes.
Env: CNUM_HOST, CNUM_PORT, CNUM_ODF, or CNUM_SIM=1.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
from _ops_common import Checker, connect  # noqa: E402

_RNG = np.random.default_rng(5)


def host(shape, dtype="float64"):
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        return _RNG.integers(-5, 6, size=shape).astype(dtype)
    if dtype.kind == "c":
        return (_RNG.uniform(-1, 1, size=shape) + 1j * _RNG.uniform(-1, 1, size=shape)).astype(dtype)
    return _RNG.uniform(-1.0, 1.0, size=shape).astype(dtype)


def tol(dtype, k):
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        return 0.0, 0.0
    eps = 2e-6 if dtype in (np.float32, np.complex64) else 1e-13
    return eps * 10, eps * 10 * max(1.0, np.sqrt(k))


def mm(chk, label, ha, hb, dtype="float64"):
    ha, hb = ha.astype(dtype), hb.astype(dtype)
    rt, at = tol(dtype, ha.shape[-1])
    chk.check(label, cnp.array(ha) @ cnp.array(hb), ha @ hb, rtol=rt, atol=at)


def main():
    iface = connect()
    chk = Checker("test_matmul", iface)

    print("matmat")
    for sa, sb in (((37, 41), (41, 23)), ((1500, 1300), (1300, 900)), ((600, 3000), (3000, 7)),
                   ((2048, 2048), (2048, 2048)), ((1025, 513), (513, 1027)),
                   ((200000, 2), (2, 10)), ((3, 70000), (70000, 4))):
        mm(chk, f"{sa} @ {sb}", host(sa), host(sb))
    for dt in ("float32", "complex128", "int64"):
        mm(chk, f"(1100, 700) @ (700, 1200) {dt}", host((1100, 700)), host((700, 1200)), dt)

    print("matmat views")
    ha, hb = host((1600, 1400)), host((1400, 1200))
    A, B = cnp.array(ha), cnp.array(hb)
    for ka, kb in (((slice(10, 900), slice(5, 700)), (slice(5, 700), slice(3, 500))),
                   ((slice(600, 1600), slice(None)), (slice(None), slice(1100, 1200))),
                   ((slice(None), slice(700, 1400)), (slice(0, 700), slice(None)))):
        chk.check(f"views {ka} {kb}", A[ka] @ B[kb], ha[ka] @ hb[kb], rtol=1e-12, atol=1e-11)
    chk.check("A.T @ A", A.T @ A, ha.T @ ha, rtol=1e-12, atol=1e-10)
    chk.check("dot out=", cnp.dot(A[:1200, :1200], B[:1200, :1200], out=cnp.zeros((1200, 1200))),
              ha[:1200, :1200] @ hb[:1200, :1200], rtol=1e-12, atol=1e-10)

    print("matmat 3-D operand")
    hx, hw = host((6, 700, 900)), host((900, 1100))
    X, W = cnp.array(hx), cnp.array(hw)
    for t in (0, 5):
        chk.check(f"X[{t}] @ W", X[t] @ W, hx[t] @ hw, rtol=1e-12, atol=1e-10)
        chk.check(f"X[{t}] @ W.T[:900]", X[t] @ W.T[:900, :700], hx[t] @ hw.T[:900, :700],
                  rtol=1e-12, atol=1e-10)
    hy = np.zeros((3, 700, 1100))
    Y = cnp.zeros((3, 700, 1100))
    for t in range(3):
        Y[t] = X[t].dot(W)
        hy[t] = hx[t] @ hw
    chk.check("Y[t] = X[t].dot(W)", Y, hy, rtol=1e-12, atol=1e-10)

    print("chained matmat (gemm rotation)")
    n = 1300
    ha, hb = host((n, n)) / n, host((n, n)) / n
    A, B, C = cnp.array(ha), cnp.array(hb), cnp.zeros((n, n))
    hc = np.zeros((n, n))
    for _ in range(4):
        cnp.dot(A, B, out=C)
        hc = ha @ hb
        A, B, C = B, C, A
        ha, hb = hb, hc
    chk.check("rotation", B, hb, rtol=1e-11, atol=1e-13)

    print("matvec / vecmat / dot")
    for sm in ((300000, 5), (1500, 1300), (37, 41), (5, 300000), (2100, 2049)):
        hm, hv = host(sm), host(sm[1])
        M, v = cnp.array(hm), cnp.array(hv)
        rt, at = tol("float64", sm[1])
        chk.check(f"{sm} @ v", M @ v, hm @ hv, rtol=rt, atol=at)
        hu = host(sm[0])
        rt, at = tol("float64", sm[0])
        chk.check(f"u @ {sm}", cnp.array(hu) @ M, hu @ hm, rtol=rt, atol=at)
    for n in (37, 300001, 1000000):
        hx, hy = host(n), host(n)
        rt, at = tol("float64", n)
        chk.check(f"dot {n}", cnp.array(hx).dot(cnp.array(hy)), hx @ hy, rtol=rt, atol=at)
    hx = host(1000000, "float32")
    chk.check("dot f32", cnp.array(hx).dot(cnp.array(hx)), hx.astype(np.float64) @ hx,
              rtol=1e-5, atol=0, dtype=np.float32)
    hm, hv = host((1500, 1300)), host(1300)
    M, v = cnp.array(hm), cnp.array(hv)
    chk.check("M[100:900, 50:600] @ v[50:600]", M[100:900, 50:600] @ v[50:600],
              hm[100:900, 50:600] @ hv[50:600], rtol=1e-12, atol=1e-11)
    for dt in ("float32", "complex128", "int64"):
        hm2, hv2 = hm.astype(dt), hv.astype(dt)
        rt, at = tol(dt, 1300)
        chk.check(f"matvec {dt}", cnp.array(hm2) @ cnp.array(hv2), hm2 @ hv2, rtol=rt, atol=at)
    hx3 = host((4, 900, 700))
    chk.check("X3[2] @ v", cnp.array(hx3)[2] @ cnp.array(hv[:700]),
              hx3[2] @ hv[:700], rtol=1e-12, atol=1e-11)

    print("chained matvec (jacobi / cg)")
    n = 1700
    ha = host((n, n)) + n * np.eye(n)
    hb = host(n)
    A, b = cnp.array(ha), cnp.array(hb)
    d = cnp.diag(A)
    hd = np.diag(ha)
    R = A - cnp.diag(d)
    hr = ha - np.diag(hd)
    x, hx = cnp.zeros(n), np.zeros(n)
    for _ in range(6):
        x = (b - cnp.dot(R, x)) / d
        hx = (hb - hr @ hx) / hd
    chk.check("jacobi", x, hx, rtol=1e-11, atol=1e-12)
    x, hx = cnp.zeros(n), np.zeros(n)
    r, hr_ = b - A.dot(x), hb - ha @ hx
    p, hp = r, hr_
    rs, hrs = r.dot(r), hr_ @ hr_
    for _ in range(5):
        Ap, hAp = A.dot(p), ha @ hp
        alpha, halpha = rs / p.dot(Ap), hrs / (hp @ hAp)
        x, hx = x + alpha * p, hx + halpha * hp
        r, hr_ = r - alpha * Ap, hr_ - halpha * hAp
        rsn, hrsn = r.dot(r), hr_ @ hr_
        p, hp = r + (rsn / rs) * p, hr_ + (hrsn / hrs) * hp
        rs, hrs = rsn, hrsn
    chk.check("cg x", x, hx, rtol=1e-9, atol=1e-12)
    chk.check("cg rs", rs, hrs, rtol=1e-8, atol=0)
    hf, hw = host((200000, 32), "float32"), np.zeros(32, dtype=np.float32)
    F, w = cnp.array(hf), cnp.zeros(32, dtype="float32")
    ht = host(200000, "float32")
    t = cnp.array(ht)
    for _ in range(3):
        e = F.dot(w) - t
        w = w + 1e-3 * e.dot(F)
        he = hf @ hw - ht
        hw = hw + np.float32(1e-3) * (he @ hf)
    chk.check("linreg step", w, hw, rtol=1e-4, atol=1e-4)

    chk.finish()


if __name__ == "__main__":
    main()
