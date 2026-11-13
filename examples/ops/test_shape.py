#!/usr/bin/env python
"""Tests for reshape, broadcasting and >3-D (folded) arrays against NumPy.

Inputs have distinct values, so a misplaced element is caught exactly.
Env: CNUM_HOST, CNUM_PORT, CNUM_ODF; CNUM_SIM=1 for the simulator.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
from _ops_common import Checker, connect  # noqa: E402

DTYPES = (np.float64, np.float32, np.int64, np.int32)


def host(shape, dtype=np.float64, seed=0):
    """Distinct values (exact in every dtype for these sizes)."""
    n = int(np.prod(shape))
    v = (np.arange(n, dtype=np.int64) * 7 + seed * 1000003) % 16777213 - 5000
    return v.astype(dtype).reshape(shape)


def dev(h):
    return cnp.array(h)


def test_reshape(chk):
    pairs = [
        ((24,), (4, 6)), ((24,), (2, 3, 4)), ((4, 6), (6, 4)), ((4, 6), (24,)),
        ((2, 3, 4), (4, 6)), ((2, 3, 4), (3, 8)), ((2, 3, 4), (4, 3, 2)),
        ((37, 41), (41, 37)), ((5, 7, 9), (35, 9)), ((5, 7, 9), (315,)), ((1000, 3), (3, 1000)),
        ((420000,), (600, 700)), ((600, 700), (700, 600)), ((600, 700), (420000,)),
        ((600, 700), (70, 60, 100)), ((70, 60, 100), (600, 700)), ((130, 65, 10), (65, 10, 130)),
        ((300001,), (300001, 1)), ((300001, 1), (300001,)), ((1, 600, 700), (600, 700)),
        ((600, 1, 700), (1, 600, 700)),
    ]
    for src, dst in pairs:
        for dt in DTYPES if int(np.prod(src)) < 5000 else (np.float64, np.int32):
            h = host(src, dt)
            chk.check(f"reshape {src}->{dst} {np.dtype(dt).name}", dev(h).reshape(dst),
                      h.reshape(dst))
    h = host((6, 20))
    x = dev(h)
    chk.check("reshape -1", x.reshape(-1, 8), h.reshape(-1, 8))
    chk.check("cnp.reshape tuple", cnp.reshape(x, (3, 2, 20)), h.reshape(3, 2, 20))
    chk.check("reshape same shape is x", x.reshape(6, 20), h)
    chk.raises("reshape size mismatch", ValueError, lambda: x.reshape(7, 20))
    chk.raises("reshape two -1", ValueError, lambda: x.reshape(-1, -1))
    # views as sources
    h = host((600, 701))
    x = dev(h)
    chk.check("reshape view [1:-1, ::2]", x[1:-1, ::2].reshape(-1), h[1:-1, ::2].reshape(-1))
    chk.check("reshape view [3:303, 1:701]", x[3:303, 1:701].reshape(700, 300),
              h[3:303, 1:701].reshape(700, 300))
    h3 = host((70, 9, 130), np.int32)
    x3 = dev(h3)
    chk.check("reshape 3-D view", x3[1:, ::2, 3:].reshape(69, -1), h3[1:, ::2, 3:].reshape(69, -1))
    chk.check("reshape of a result", (x3 * 2).reshape(70 * 9, 130), (h3 * 2).reshape(70 * 9, 130))
    chk.check("chained reshape", dev(host((24,))).reshape(4, 6).reshape(2, 12).reshape(3, 8),
              host((24,)).reshape(3, 8))


def test_family(chk):
    h = host((5, 7, 9))
    x = dev(h)
    chk.check("ravel", x.ravel(), h.ravel())
    chk.check("cnp.ravel view", cnp.ravel(x[1:, 2:5]), np.ravel(h[1:, 2:5]))
    chk.check("flatten", x.flatten(), h.flatten())
    one = dev(host((5, 1, 9)))
    chk.check("squeeze", one.squeeze(), host((5, 1, 9)).squeeze())
    chk.check("cnp.squeeze axis", cnp.squeeze(dev(host((1, 6, 1))), axis=0), host((1, 6, 1)).squeeze(0))
    chk.raises("squeeze non-1 axis", ValueError, lambda: cnp.squeeze(x, axis=1))
    v = dev(host((700,)))
    chk.check("expand_dims 0", cnp.expand_dims(v, 0), np.expand_dims(host((700,)), 0))
    chk.check("expand_dims -1", cnp.expand_dims(v, -1), np.expand_dims(host((700,)), -1))
    chk.check("expand_dims (0,2)", cnp.expand_dims(v, (0, 2)), np.expand_dims(host((700,)), (0, 2)))
    m = dev(host((600, 7)))
    chk.check("expand_dims 2-D axis 1", cnp.expand_dims(m, 1), np.expand_dims(host((600, 7)), 1))
    chk.check("atleast_2d", cnp.atleast_2d(v), np.atleast_2d(host((700,))))
    chk.check("atleast_3d 1-D", cnp.atleast_3d(v), np.atleast_3d(host((700,))))
    chk.check("atleast_3d 2-D", cnp.atleast_3d(m), np.atleast_3d(host((600, 7))))
    chk.check("v[None]", v[None], host((700,))[None])
    chk.check("v[:, None]", v[:, None], host((700,))[:, None])
    chk.check("m[:, None, :]", m[:, None, :], host((600, 7))[:, None, :])
    chk.check("m[None, 2:5]", m[None, 2:5], host((600, 7))[None, 2:5])


def test_broadcast(chk):
    for dt in DTYPES:
        a = host((1100, 1), dt)
        b = host((1, 700), dt, seed=1)
        A, B = dev(a), dev(b)
        name = np.dtype(dt).name
        chk.check(f"(N,1)+(1,M) {name}", A + B, a + b)
        m = host((1100, 700), dt, seed=2)
        M = dev(m)
        r = host((700,), dt, seed=3)
        R = dev(r)
        chk.check(f"(N,M)*(M,) {name}", M * R, m * r)
        chk.check(f"(M,)-(N,M) {name}", R - M, r - m)
        chk.check(f"(N,M)-(N,1) {name}", M - A, m - a)
        chk.check(f"maximum (N,1),(M,) {name}", cnp.maximum(A, R), np.maximum(a, r))
        chk.check(f"(N,M) < (M,) {name}", M < R, m < r, dtype=np.bool_)
    a = host((1100,))
    c = host((40,), seed=5)
    A, C = dev(a), dev(c)
    chk.check("x[:, newaxis] + y[newaxis, :]", A[:, cnp.newaxis] + C[cnp.newaxis, :],
              a[:, None] + c[None, :])
    w = host((300001,), seed=6)
    W = dev(w)
    chk.check("(2,1) * (300001,) multi-chare", dev(host((2, 1))) * W, host((2, 1)) * w)
    # 3-D
    t = host((70, 9, 130), seed=7)
    T = dev(t)
    chk.check("(a,b,c)+(c,)", T + dev(host((130,), seed=8)), t + host((130,), seed=8))
    chk.check("(a,1,c)*(1,b,1)", dev(host((70, 1, 130))) * dev(host((1, 9, 1), seed=9)),
              host((70, 1, 130)) * host((1, 9, 1), seed=9))
    chk.check("(a,b,1)+(b,c)", dev(host((70, 9, 1))) + dev(host((9, 130), seed=4)),
              host((70, 9, 1)) + host((9, 130), seed=4))
    chk.check("3-D view (a,b,c)[::2] + (b,1)", T[::2] + dev(host((9, 1))), t[::2] + host((9, 1)))
    # mixed dtypes promote, then broadcast
    i32 = host((600, 1), np.int32)
    chk.check("int32 (N,1) + float64 (M,)", dev(i32) + dev(host((7,))), i32 + host((7,)))
    # where / clip / ternary broadcasting
    cond = (host((600, 1)) % 3 == 0).astype(np.float64)
    chk.check("where (N,1),(N,M),(1,M)",
              cnp.where(dev(cond), dev(host((600, 7))), dev(host((1, 7), seed=2))),
              np.where(cond, host((600, 7)), host((1, 7), seed=2)))
    chk.check("clip (N,M),(M,),(N,1)",
              cnp.clip(dev(host((600, 7))), dev(host((7,)) - 100), dev(host((600, 1)) + 4000)),
              np.clip(host((600, 7)), host((7,)) - 100, host((600, 1)) + 4000))
    # explicit broadcast_to / broadcast_arrays
    chk.check("broadcast_to (M,)->(N,M)", cnp.broadcast_to(dev(host((700,))), (1100, 700)),
              np.broadcast_to(host((700,)), (1100, 700)))
    chk.check("broadcast_to (N,1,1)->(N,b,c)", cnp.broadcast_to(dev(host((70, 1, 1))), (70, 9, 130)),
              np.broadcast_to(host((70, 1, 1)), (70, 9, 130)))
    chk.check("broadcast_to view", cnp.broadcast_to(dev(host((10, 700)))[3:4, 1:600], (90, 599)),
              np.broadcast_to(host((10, 700))[3:4, 1:600], (90, 599)))
    p, q = cnp.broadcast_arrays(dev(host((5, 1))), dev(host((3,))))
    pp, qq = np.broadcast_arrays(host((5, 1)), host((3,)))
    chk.check("broadcast_arrays[0]", p, pp)
    chk.check("broadcast_arrays[1]", q, qq)
    # Column views of a fresh result get an offset that spills into a second chare column.
    pw = dev(host((1000, 1))) + dev(host((1, 3), seed=3))
    pwh = host((1000, 1)) + host((1, 3), seed=3)
    acc = 0.0
    for idx in range(3):
        acc = acc + cnp.where(pw[:, idx:idx + 1] > 0, pw[:, idx:idx + 1], 0.0)
    chk.check("offset column views", acc, sum(np.where(pwh[:, i:i + 1] > 0, pwh[:, i:i + 1], 0.0)
                                               for i in range(3)))
    chk.check("argmin of offset result", cnp.argmin(pw, axis=1), np.argmin(pwh, axis=1))
    chk.raises("incompatible shapes", ValueError, lambda: dev(host((4, 3))) + dev(host((4,))))
    chk.raises("broadcast_to bad", ValueError, lambda: cnp.broadcast_to(dev(host((3,))), (4, 4)))


def test_assign(chk):
    m = host((1100, 700))
    M = dev(m)
    r = host((700,), seed=1)
    M[2:900] = dev(r)
    m[2:900] = r
    chk.check("A[2:900] = row", M, m)
    col = host((1100, 1), seed=2)
    M[...] = dev(col)
    m[...] = col
    chk.check("A[...] = (N,1)", M, m)
    M[:, 1:600] = dev(host((1, 599), seed=3))
    m[:, 1:600] = host((1, 599), seed=3)
    chk.check("A[:, 1:600] = (1,599)", M, m)
    x = dev(host((40, 1)))
    y = cnp.zeros((1, 40))
    y[...] = x
    chk.check("(1,40) = (40,1) reshape", y, host((40, 1)).reshape(1, 40))
    # in-place operators
    a = host((600, 7))
    A = dev(a)
    A += dev(host((7,), seed=4))
    a += host((7,), seed=4)
    chk.check("A += row", A, a)
    A -= 3.0
    a -= 3.0
    chk.check("A -= scalar", A, a)
    A[1:5] *= dev(host((4, 1), seed=5))
    a[1:5] *= host((4, 1), seed=5)
    chk.check("A[1:5] *= col", A, a)
    f32 = host((600, 7), np.float32)
    F = dev(f32)
    cnt = np.arange(1, 601, dtype=np.int64).reshape(600, 1)
    F /= dev(cnt)
    f32 /= cnt
    chk.check("float32 /= int64 col keeps float32", F, f32, rtol=1e-6, dtype=np.float32)
    B = A
    B += 1.0
    a += 1.0
    chk.check("in-place aliases", A, a)


def test_axis_none(chk):
    h = host((60, 70), np.int32)
    x = dev(h)
    chk.check("cumsum axis=None 2-D", cnp.cumsum(x), np.cumsum(h))
    h3 = (host((5, 6, 7)) % 13).astype(np.float64)
    chk.check("cumsum axis=None 3-D", cnp.cumsum(dev(h3)), np.cumsum(h3))
    out = cnp.zeros((60 * 70,), dtype=np.int32)
    cnp.cumsum(x, axis=None, out=out)
    chk.check("cumsum out= 1-D", out, np.cumsum(h).astype(np.int32))
    hs = host((30, 40), np.float64)[:, ::-1].copy()
    chk.check("sort axis=None", cnp.sort(dev(hs), axis=None), np.sort(hs, axis=None))
    chk.check("take axis=None", dev(h).take([5, 0, 4199]), h.take([5, 0, 4199]))
    chk.check("concatenate axis=None",
              cnp.concatenate([dev(host((3, 4))), dev(host((5,), seed=1))], axis=None),
              np.concatenate([host((3, 4)), host((5,), seed=1)], axis=None))
    lab = (np.arange(500) % 7).astype(np.int64)
    wts = host((500, 1), np.float64)
    chk.check("bincount weights (N,1)", cnp.bincount(dev(lab), weights=dev(wts)[:, 0], minlength=7),
              np.bincount(lab, weights=wts[:, 0], minlength=7))


def test_folded(chk):
    s4 = (7, 6, 9, 13)
    h = host(s4)
    X = dev(h)
    chk.check("4-D upload/get", X, h)
    chk.check("4-D zeros", cnp.zeros(s4, dtype=np.float32), np.zeros(s4, np.float32))
    chk.check("4-D full", cnp.full(s4, 2.5), np.full(s4, 2.5))
    chk.check("5-D ones", cnp.ones((2, 3, 4, 5, 6), dtype=np.int32), np.ones((2, 3, 4, 5, 6), np.int32))
    chk.check("arange.reshape 4-D", cnp.arange(7 * 6 * 9 * 13).reshape(s4),
              np.arange(7 * 6 * 9 * 13).reshape(s4))
    chk.check("4-D elementwise", cnp.sqrt(X * X + 1.0) - X / 3, np.sqrt(h * h + 1.0) - h / 3)
    chk.check("4-D + 2-D (c,d)", X + dev(host((9, 13), seed=1)), h + host((9, 13), seed=1))
    chk.check("4-D * (b,1,d)", X * dev(host((6, 1, 13), seed=2)), h * host((6, 1, 13), seed=2))
    chk.check("4-D + (a,1,1,1)", X + dev(host((7, 1, 1, 1), seed=3)), h + host((7, 1, 1, 1), seed=3))
    chk.check("4-D where", cnp.where(X > 100, X, -1.0), np.where(h > 100, h, -1.0))
    chk.check("4-D comparison", X >= dev(host((13,))), h >= host((13,)), dtype=np.bool_)
    chk.check("4-D astype", X.astype(np.int32), h.astype(np.int32))
    # indexing
    chk.check("X[2] (view)", X[2], h[2])
    chk.check("X[1:3]", X[1:3], h[1:3])
    chk.check("X[:, 4]", X[:, 4], h[:, 4])
    chk.check("X[2, 3]", X[2, 3], h[2, 3])
    chk.check("X[2, :, 1:5, ::3]", X[2, :, 1:5, ::3], h[2, :, 1:5, ::3])
    chk.check("X[::2] (blocks)", X[::2], h[::2])
    chk.check("X[:, 1:4] (blocks)", X[:, 1:4], h[:, 1:4])
    chk.check("X[..., 3]", X[..., 3], h[..., 3])
    chk.check("X[None] 5-D", X[None], h[None])
    chk.check("3-D [None] -> 4-D", dev(host((5, 6, 7)))[None], host((5, 6, 7))[None])
    # assignment
    Y = cnp.zeros(s4)
    y = np.zeros(s4)
    for g in range(7):
        Y[g] = dev(host((6, 9, 13), seed=g)) * g
        y[g] = host((6, 9, 13), seed=g) * g
    chk.check("Y[g] = 3-D expr", Y, y)
    Y[:, 1:4] = 5.0
    y[:, 1:4] = 5.0
    chk.check("Y[:, 1:4] = scalar", Y, y)
    Y[::3, :, 2] = dev(host((13,)))
    y[::3, :, 2] = host((13,))
    chk.check("Y[::3, :, 2] = row", Y, y)
    Y[1:3] = X[4:6]
    y[1:3] = h[4:6]
    chk.check("Y[1:3] = X[4:6]", Y, y)
    Y += X
    y += h
    chk.check("Y += X (4-D in-place)", Y, y)
    Z = X.copy()
    Z[:, :, 3, :] += 1.0
    z = h.copy()
    z[:, :, 3, :] += 1.0
    chk.check("Z[:, :, 3, :] += 1", Z, z)
    # reductions
    chk.check("sum()", X.sum(), np.asarray(h.sum()))
    for axis in (0, 1, 2, 3, (0, 1), (2, 3), (0, 2), (1, 3), (0, 1, 3)):
        chk.check(f"sum(axis={axis})", cnp.sum(X, axis=axis), h.sum(axis=axis))
    chk.check("sum keepdims (1,3)", X.sum(axis=(1, 3), keepdims=True), h.sum(axis=(1, 3), keepdims=True))
    chk.check("mean(axis=(0,2))", X.mean(axis=(0, 2)), h.mean(axis=(0, 2)), rtol=1e-12)
    chk.check("max(axis=1)", X.max(axis=1), h.max(axis=1))
    chk.check("argmin(axis=2)", cnp.argmin(X, axis=2), np.argmin(h, axis=2))
    chk.check("argmax()", cnp.argmax(X), np.asarray(np.argmax(h)))
    chk.check("var(axis=3)", X.var(axis=3), h.var(axis=3), rtol=1e-12)
    # reshape / squeeze / expand_dims
    chk.check("4-D -> 2-D", X.reshape(42, 117), h.reshape(42, 117))
    chk.check("4-D -> 5-D", X.reshape(7, 2, 3, 9, 13), h.reshape(7, 2, 3, 9, 13))
    chk.check("4-D -> 4-D", X.reshape(13, 9, 6, 7), h.reshape(13, 9, 6, 7))
    chk.check("4-D ravel", X.ravel(), h.ravel())
    chk.check("expand_dims to 4-D", cnp.expand_dims(dev(host((6, 9, 13))), 1),
              np.expand_dims(host((6, 9, 13)), 1))
    chk.check("squeeze 4-D", cnp.zeros((3, 1, 4, 5)).squeeze(), np.zeros((3, 4, 5)))
    chk.check("cumsum 4-D axis=None", cnp.cumsum(X), np.cumsum(h))
    chk.check("cumsum 4-D axis=-1", cnp.cumsum(X, axis=-1), np.cumsum(h, axis=-1))
    big = host((3, 70, 65, 10), np.float32)
    chk.check("multi-chare 4-D get/index", dev(big)[1, :, 3:60], big[1, :, 3:60])
    chk.check("multi-chare 4-D sum(axis=0)", dev(big).sum(axis=0), big.sum(axis=0), rtol=1e-6)


def small(shape, dtype, seed=0):
    """Values that are exact in every dtype (incl. bool, int8, float16, complex)."""
    v = host(shape, np.int64, seed) % 97 - 40
    if np.dtype(dtype).kind == "c":
        return (v + 1j * (v % 13)).astype(dtype)
    return v.astype(dtype)


def test_dtypes(chk):
    for dt in (np.bool_, np.int8, np.uint16, np.float16, np.complex64, np.complex128):
        name = np.dtype(dt).name
        h = small((600, 70), dt)
        x = dev(h)
        chk.check(f"reshape 2-D->3-D {name}", x.reshape(70, 60, 10), h.reshape(70, 60, 10),
                  dtype=dt)
        chk.check(f"reshape view {name}", x[1:, ::3].reshape(-1), h[1:, ::3].reshape(-1),
                  dtype=dt)
        r = small((70,), dt, seed=1)
        chk.check(f"broadcast_to {name}", cnp.broadcast_to(dev(r), (600, 70)),
                  np.broadcast_to(r, (600, 70)), dtype=dt)
        c = small((600, 1), dt, seed=2)
        chk.check(f"(N,M) == (N,1) {name}", x == dev(c), h == c, dtype=np.bool_)
        f = small((3, 4, 5, 6), dt, seed=3)
        F = dev(f)
        chk.check(f"4-D create/get {name}", F, f, dtype=dt)
        chk.check(f"4-D reshape {name}", F.reshape(6, 5, 4, 3), f.reshape(6, 5, 4, 3), dtype=dt)
        chk.check(f"4-D index {name}", F[1, :, 2:4], f[1, :, 2:4], dtype=dt)
        chk.check(f"4-D != {name}", F != F[:, :, :1, :], f != f[:, :, :1, :], dtype=np.bool_)
        if dt != np.bool_:
            chk.check(f"4-D + (6,) {name}", F + dev(small((6,), dt, seed=4)),
                      f + small((6,), dt, seed=4), dtype=dt)
        t = cnp.zeros((5, 6), dtype=dt)
        t[...] = dev(small((6,), dt, seed=5))
        chk.check(f"assign broadcast row {name}", t, np.broadcast_to(small((6,), dt, seed=5),
                                                                    (5, 6)), dtype=dt)
    b = dev(np.arange(24) % 3 == 0)
    chk.check("bool 4-D any(axis=1)", b.reshape(2, 3, 2, 2).any(axis=1),
              (np.arange(24) % 3 == 0).reshape(2, 3, 2, 2).any(axis=1), dtype=np.bool_)
    i = cnp.arange(10, dtype=np.int32)
    chk.raises("int32 /= int: same_kind", TypeError, lambda: i.__itruediv__(2))
    i += 3
    chk.check("int32 += int", i, np.arange(10, dtype=np.int32) + 3, dtype=np.int32)


def main():
    iface = connect()
    chk = Checker("test_shape", iface)
    for fn in (test_reshape, test_family, test_broadcast, test_assign, test_axis_none, test_folded,
               test_dtypes):
        sec = os.environ.get("TS_SECTIONS")
        if sec and fn.__name__[5:] not in sec.split(","):
            continue
        print(fn.__name__)
        fn(chk)
    chk.finish()


if __name__ == "__main__":
    main()
