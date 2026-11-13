#!/usr/bin/env python
"""Tests for astype, mixed-dtype promotion, scans (cumsum family) and sorts/argsort.

Inputs are generated natively. The big shapes span several chares.
Env: CNUM_HOST, CNUM_PORT, CNUM_ODF (default 4). Run as e.g.
    harness/run_one.sh server.out 2 35001 examples/ops/test_castscansort.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cupynumeric"))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402

DTYPES = [np.float32, np.float64, np.int32, np.int64]
BIG_1D = (600_001,)  # 3 tiles of 262144
BIG_2D = (1030, 1031)  # 2 x 2 tiles of 1024 (512 clamps to 1024 for <= 8 chares)
BIG_3D = (130, 131, 70)  # 2 x 2 x 1 tiles of 128


def linear_index(shape, dtype=np.float64):
    """Row-major linear index, built natively from cumsum(ones) along each axis."""
    shape = tuple(shape)
    L = None
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        if shape[axis] > 1:
            g = (cnp.cumsum(cnp.ones(shape, dtype=dtype), axis=axis) - 1) * stride
            L = g if L is None else L + g
        stride *= shape[axis]
    return cnp.zeros(shape, dtype=dtype) if L is None else L


def real_data(shape, seed, lo=-100.0, hi=100.0, dtype=np.float64):
    """Pseudo-random floats in [lo, hi) (logistic map of the linear index)."""
    L = linear_index(shape)
    if int(np.prod(shape)) == 1:
        # unary ops on size-1 arrays crash the JIT; avoid them
        x = L + (lo + (hi - lo) * ((seed * 0.6180339887498949) % 1.0))
    else:
        x = U.uniform_from_index(L, int(np.prod(shape)), seed=seed, lo=lo, hi=hi)
    return x if np.dtype(dtype) == np.float64 else x.astype(dtype)


def int_data(shape, seed, upper=50, dtype=np.int64):
    """Integers in [-upper/2, upper/2) from a quadratic hash of the index: many ties."""
    L = linear_index(shape, dtype=np.int64)
    h = (L * L) * 7919 + L * (104729 + 31 * seed) + (12345 + seed)
    x = U.int_mod(h, int(upper)) - upper // 2
    return x if np.dtype(dtype) == np.int64 else x.astype(dtype)


def main(chk):
    host = os.environ.get("CNUM_HOST", "127.0.0.1")
    port = int(os.environ.get("CNUM_PORT", "1234"))
    odf = int(os.environ.get("CNUM_ODF", "4"))
    iface = cnp.CharmNumericInterface()
    iface.connect(host, port, odf)
    print(f"connected {host}:{port} odf={odf}")
    only = set(os.environ.get("CSS_ONLY", "cast,promo,scan,sort").split(","))

    def check(label, arr, ref, rtol=0.0, atol=0.0, exact_dtype=True):
        ref = np.asarray(ref)
        if tuple(arr.shape) != ref.shape:
            return chk.true(label + " shape", False, f"{arr.shape} != {ref.shape}")
        if exact_dtype and np.dtype(arr.dtype) != ref.dtype:
            return chk.true(label + " dtype", False, f"{arr.dtype} != {ref.dtype}")
        got = arr.get(iface)
        if np.issubdtype(ref.dtype, np.integer) and rtol == 0 and atol == 0:
            ok = bool(np.array_equal(got, ref))
            bad = int(np.count_nonzero(got != ref)) if got.shape == ref.shape else -1
            return chk.true(label, ok, "" if ok else f"({bad} mismatches)")
        return chk.close(label, got, ref, rtol=rtol, atol=atol)

    # ------------------------------------------------------------------ astype
    if "cast" in only:
        for shape, seed in [((7,), 1), (BIG_1D, 2), ((5, 3), 3), (BIG_2D, 4), ((2, 3, 4), 5),
                            (BIG_3D, 6)]:
            src = {np.float64: real_data(shape, seed)}
            src[np.float32] = src[np.float64].astype(np.float32)
            src[np.int64] = int_data(shape, seed, upper=2000)
            src[np.int32] = src[np.int64].astype(np.int32)
            host = {dt: a.get(iface) for dt, a in src.items()}
            big = int(np.prod(shape)) > 1000
            for dfrom in DTYPES:
                for dto in DTYPES:
                    if big and dfrom == dto:
                        continue
                    check(f"astype {shape} {np.dtype(dfrom).name}->{np.dtype(dto).name}",
                          src[dfrom].astype(dto), host[dfrom].astype(dto))
        # views, copy=False, cnp.astype, asarray/array(dtype=), chains
        A = real_data((40, 30), 11)
        Ah = A.get(iface)
        check("astype view [3:37:2, 5:] f64->i32", A[3:37:2, 5:].astype(np.int32),
              Ah[3:37:2, 5:].astype(np.int32))
        check("astype same dtype copy", A.astype(np.float64), Ah)
        chk.true("astype(copy=False) same dtype is self", A.astype(np.float64, copy=False) is A)
        check("cnp.astype f64->f32", cnp.astype(A, np.float32), Ah.astype(np.float32))
        check("asarray(dtype=int64)", cnp.asarray(A, dtype=np.int64), Ah.astype(np.int64))
        check("array(dtype=float32)", cnp.array(A, dtype=np.float32), Ah.astype(np.float32))
        check("chain f64->i32->f32->i64", A.astype(np.int32).astype(np.float32).astype(np.int64),
              Ah.astype(np.int32).astype(np.float32).astype(np.int64))
        big = real_data((700_000,), 12)
        bigh = big.get(iface)
        check("astype offset view big[100003:650004] f64->i64",
              big[100003:650004].astype(np.int64), bigh[100003:650004].astype(np.int64))

    # -------------------------------------------------------------- promotion
    if "promo" in only:
        for shape, seed in [((8,), 21), (BIG_1D, 22), ((33, 17), 23), (BIG_2D, 24), (BIG_3D, 25)]:
            f64 = real_data(shape, seed)
            f32 = f64.astype(np.float32)
            i64 = int_data(shape, seed, upper=1000)
            i32 = i64.astype(np.int32)
            h = {"f64": f64.get(iface), "f32": f32.get(iface), "i64": i64.get(iface),
                 "i32": i32.get(iface)}
            d = {"f64": f64, "f32": f32, "i64": i64, "i32": i32}
            for a, b in [("i64", "f64"), ("i32", "f32"), ("f32", "i64"), ("i32", "i64"),
                         ("f32", "f64")]:
                check(f"{shape} {a}+{b}", d[a] + d[b], h[a] + h[b], rtol=1e-6)
                check(f"{shape} {b}*{a}", d[b] * d[a], h[b] * h[a], rtol=1e-6)
            check(f"{shape} f64-i32", d["f64"] - d["i32"], h["f64"] - h["i32"], rtol=1e-12)
            check(f"{shape} i64/f32", d["i64"] / d["f32"], h["i64"] / h["f32"], rtol=1e-6)
            check(f"{shape} i32*0.5 (Python float)", d["i32"] * 0.5, h["i32"] * 0.5)
            check(f"{shape} 2.5-i64 (Python float)", 2.5 - d["i64"], 2.5 - h["i64"])
            check(f"{shape} i32*3 (Python int stays int32)", d["i32"] * 3, h["i32"] * 3)
        n = 300
        M = real_data((n, n), 31)
        Mi = int_data((n, n), 32, upper=20)
        v = int_data((n,), 33, upper=20)
        Mh, Mih, vh = M.get(iface), Mi.get(iface), v.get(iface)
        check("matvec f64 @ i64", M @ v, Mh @ vh, rtol=1e-12, atol=1e-9)
        check("matmatmul i64 @ f64", Mi @ M, Mih @ Mh, rtol=1e-12, atol=1e-9)
        w = real_data((n,), 34)
        check("dot i64 . f64", v @ w, np.asarray(vh @ w.get(iface)), rtol=1e-12)
        T = cnp.zeros((n, n), dtype=np.float32)
        T[...] = Mi
        check("setitem i64 -> f32 array", T, Mih.astype(np.float32))
        T[10:20, 5:9] = M[0:10, 0:4]
        Th = Mih.astype(np.float32)
        Th[10:20, 5:9] = Mh[0:10, 0:4]
        check("setitem f64 view -> f32 slice", T, Th)

    # ------------------------------------------------------------------ scans
    if "scan" in only:
        cases = [((9,), (0, None)), (BIG_1D, (0,)), ((6, 5), (0, 1, -1)), (BIG_2D, (0, 1)),
                 ((3, 4, 5), (0, 1, 2)), (BIG_3D, (0, 1, 2))]
        for k, (shape, axes) in enumerate(cases):
            add = real_data(shape, 40 + k, lo=-1.0, hi=1.0)
            mul = real_data(shape, 50 + k, lo=0.9999, hi=1.0001)
            ints = int_data(shape, 60 + k, upper=100, dtype=np.int32)
            addh, mulh, intsh = add.get(iface), mul.get(iface), ints.get(iface)
            for ax in axes:
                tol = 1e-9
                check(f"cumsum {shape} axis={ax}", cnp.cumsum(add, axis=ax),
                      np.cumsum(addh, axis=ax), rtol=tol, atol=1e-9)
                check(f"cumprod {shape} axis={ax}", cnp.cumprod(mul, axis=ax),
                      np.cumprod(mulh, axis=ax), rtol=tol)
                check(f"cumsum int32->int64 {shape} axis={ax}", cnp.cumsum(ints, axis=ax),
                      np.cumsum(intsh, axis=ax))
            if int(np.prod(shape)) < 100:
                ax = axes[0]
                check(f"cumprod int32 {shape} axis={ax}", cnp.cumprod(ints, axis=ax),
                      np.cumprod(intsh, axis=ax))
        # NaNs, nan* variants, float32, dtype=, out=, methods, views
        A = real_data((50, 40), 71, lo=0.5, hi=1.5)
        A[1, 1] = np.nan
        A[7, 30] = np.nan
        Ah = A.get(iface)
        for ax in (0, 1):
            check(f"cumsum NaN axis={ax}", cnp.cumsum(A, axis=ax), np.cumsum(Ah, axis=ax),
                  rtol=1e-12)
            check(f"nancumsum axis={ax}", cnp.nancumsum(A, axis=ax), np.nancumsum(Ah, axis=ax),
                  rtol=1e-12)
            check(f"nancumprod axis={ax}", cnp.nancumprod(A, axis=ax),
                  np.nancumprod(Ah, axis=ax), rtol=1e-10)
        B = real_data(BIG_1D, 72, lo=-1.0, hi=1.0)
        B[5] = np.nan
        B[400_000] = np.nan
        Bh = B.get(iface)
        check("nancumsum big 1-D", cnp.nancumsum(B), np.nancumsum(Bh), rtol=1e-9, atol=1e-9)
        F = real_data((1000,), 73, lo=0.0, hi=1.0, dtype=np.float32)
        Fh = F.get(iface)
        check("cumsum float32", cnp.cumsum(F), np.cumsum(Fh), rtol=1e-5)
        check("cumsum dtype=float64 of float32", cnp.cumsum(F, dtype=np.float64),
              np.cumsum(Fh, dtype=np.float64), rtol=1e-12)
        I = int_data((20, 30), 74, upper=100, dtype=np.int64)
        Ih = I.get(iface)
        check("cumsum dtype=float32 of int64", cnp.cumsum(I, axis=1, dtype=np.float32),
              np.cumsum(Ih, axis=1, dtype=np.float32), rtol=1e-6)
        out = cnp.zeros((20, 30), dtype=np.int64)
        res = cnp.cumsum(I, axis=0, out=out)
        chk.true("cumsum out= returns out", res is out)
        check("cumsum out=", out, np.cumsum(Ih, axis=0))
        check("Array.cumsum method", I.cumsum(axis=1), Ih.cumsum(axis=1))
        check("Array.cumprod method (view)", I[0:5, :].cumprod(axis=0),
              Ih[0:5, :].cumprod(axis=0))
        check("cumsum view [2:18:3, 1:29]", cnp.cumsum(I[2:18:3, 1:29], axis=1),
              np.cumsum(Ih[2:18:3, 1:29], axis=1))
        big = real_data((700_000,), 75, lo=-1.0, hi=1.0)
        bigh = big.get(iface)
        check("cumsum offset view big[100003:650004]", cnp.cumsum(big[100003:650004]),
              np.cumsum(bigh[100003:650004]), rtol=1e-9, atol=1e-9)
        check("cumsum(axis=None) on 2-D", cnp.cumsum(I), np.cumsum(Ih))

    # ------------------------------------------------------------------ sorts
    if "sort" in only:
        cases = [((11,), (-1, 0, None)), (BIG_1D, (0,)), ((7, 9), (-1, 0)), (BIG_2D, (1, 0)),
                 ((3, 4, 5), (2, 1, 0)), (BIG_3D, (2, 0))]
        for k, (shape, axes) in enumerate(cases):
            R = real_data(shape, 80 + k)
            I = int_data(shape, 90 + k, upper=30)  # many ties
            Rh, Ih = R.get(iface), I.get(iface)
            for ax in axes:
                check(f"sort f64 {shape} axis={ax}", cnp.sort(R, axis=ax), np.sort(Rh, axis=ax))
                check(f"argsort f64 {shape} axis={ax}", cnp.argsort(R, axis=ax),
                      np.argsort(Rh, axis=ax, kind="stable"))
                check(f"sort i64 ties {shape} axis={ax}", cnp.sort(I, axis=ax),
                      np.sort(Ih, axis=ax))
                check(f"argsort i64 ties (stable) {shape} axis={ax}", cnp.argsort(I, axis=ax),
                      np.argsort(Ih, axis=ax, kind="stable"))
        for dt in (np.float32, np.int32):
            X = real_data((300, 7), 101).astype(dt)
            Xh = X.get(iface)
            check(f"sort {np.dtype(dt).name} axis=0", cnp.sort(X, axis=0), np.sort(Xh, axis=0))
            check(f"argsort {np.dtype(dt).name} axis=0", cnp.argsort(X, axis=0),
                  np.argsort(Xh, axis=0, kind="stable"))
        N = real_data((40, 25), 102)
        N[3, 4] = np.nan
        N[3, 9] = np.nan
        N[17, 4] = np.nan
        Nh = N.get(iface)
        for ax in (0, 1):
            check(f"sort NaNs last axis={ax}", cnp.sort(N, axis=ax), np.sort(Nh, axis=ax))
            check(f"argsort NaNs axis={ax}", cnp.argsort(N, axis=ax),
                  np.argsort(Nh, axis=ax, kind="stable"))
        V = int_data((60, 50), 103, upper=10)
        Vh = V.get(iface)
        check("sort view [5:55:2, 3:]", cnp.sort(V[5:55:2, 3:]), np.sort(Vh[5:55:2, 3:]))
        check("argsort view [5:55:2, 3:] axis=0", V[5:55:2, 3:].argsort(axis=0),
              np.argsort(Vh[5:55:2, 3:], axis=0, kind="stable"))
        W = V.copy()
        W.sort(axis=0)
        check("in-place Array.sort(axis=0)", W, np.sort(Vh, axis=0))
        W2 = V.copy()
        W2[10:20, :].sort()
        W2h = Vh.copy()
        W2h[10:20, :].sort()
        check("in-place ArrayView.sort()", W2, W2h)
        big = real_data((700_000,), 104)
        bigh = big.get(iface)
        check("sort offset view big[100003:650004]", cnp.sort(big[100003:650004]),
              np.sort(bigh[100003:650004]))
        check("argsort offset view big[100003:650004]", cnp.argsort(big[100003:650004]),
              np.argsort(bigh[100003:650004], kind="stable"))
        # consumers of the results in the same DAG
        Q = real_data((5000,), 105)
        Qh = Q.get(iface)
        check("sort feeds elementwise", cnp.sort(Q) * 2.0 + 1.0, np.sort(Qh) * 2.0 + 1.0)
        check("argsort -> float", cnp.argsort(Q).astype(np.float64) + 0.5,
              np.argsort(Qh, kind="stable") + 0.5)

    # ------------------------------------------------------- size-1 arrays
    # Run last: size-1 results on multi-chare partitions are wrong in the backend.
    x1 = cnp.full((1,), -34.75)
    i1 = cnp.full((1,), -648, dtype=np.int64)
    x1h, i1h = np.array([-34.75]), np.array([-648])
    for dt in DTYPES:
        if dt != np.float64:
            check(f"astype (1,) float64->{np.dtype(dt).name}", x1.astype(dt), x1h.astype(dt))
        if dt != np.int64:
            check(f"astype (1,) int64->{np.dtype(dt).name}", i1.astype(dt), i1h.astype(dt))
    check("cumsum (1,)", cnp.cumsum(x1), np.cumsum(x1h))
    check("cumprod (1,) int64", cnp.cumprod(i1), np.cumprod(i1h))
    check("sort (1,)", cnp.sort(x1), np.sort(x1h))
    check("argsort (1,)", cnp.argsort(i1), np.argsort(i1h))


if __name__ == "__main__":
    U.run("test_castscansort", main)
