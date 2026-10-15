#!/usr/bin/env python
"""Tests for native reductions (REDUCTION / BINCOUNT executors) against NumPy.

Covers full and axis reductions, argmax/argmin ties, NaN propagation, norm and bincount,
on offset and strided views. Env: CNUM_HOST, CNUM_PORT, CNUM_ODF (default 4).
Run as e.g. harness/run_one.sh server.out 2 33001 examples/ops/test_reductions.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cupynumeric"))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from charmtyles.core import execute  # noqa: E402


IFACE = None


def make(shape, dtype, seed):
    """Native pseudo-random input: floats clipped to [-0.97, 0.97], ints in [-50, 50).

    Each intermediate is execute()d before reuse, since a fused temporary read
    twice currently comes back as zeros. 1-D inputs are row sums of (n, 2) inputs.
    """
    dtype = np.dtype(dtype)
    if len(shape) == 1:
        return make((shape[0], 2), dtype, seed) @ cnp.ones((2,), dtype=dtype)
    if dtype.kind == "i":
        L = U.linear_index(shape, dtype=dtype)
        execute(IFACE)
        l2 = U.int_mod(L, 1009)
        execute(IFACE)
        h = (l2 * l2) * 13 + l2 * (7 + seed) + 3 * seed
        execute(IFACE)
        return U.int_mod(h, 100) - 50
    u = U.uniform(shape, seed=seed, dtype=dtype)
    x = 2.0 * u - 1.0
    execute(IFACE)
    hi = 0.5 + 0.5 * cnp.tanh(1e10 * (x - 0.97))
    lo = 0.5 + 0.5 * cnp.tanh(1e10 * (-0.97 - x))
    return x + hi * (0.97 - x) + lo * (-0.97 - x)


def small_int(shape, dtype, seed, m, lo=0):
    """Native ints in [lo, lo + m)."""
    P = make(shape, dtype, seed) + 50  # [0, 100)
    execute(IFACE)
    return U.int_mod(P, m) + lo


def main(chk):
    host = os.environ.get("CNUM_HOST", "127.0.0.1")
    port = int(os.environ.get("CNUM_PORT", "1234"))
    odf = int(os.environ.get("CNUM_ODF", "4"))
    global IFACE
    iface = cnp.CharmNumericInterface()
    iface.connect(host, port, odf)
    IFACE = iface
    print(f"connected {host}:{port} odf={odf}")

    def tol(dtype):
        return (2e-5, 1e-5) if np.dtype(dtype) == np.float32 else (1e-10, 1e-10)

    def eq(label, arr, ref, rtol=0.0, atol=0.0):
        ref = np.asarray(ref)
        exp_shape = ref.shape
        if tuple(arr.shape) != exp_shape:
            chk.true(label + " shape", False, f"{arr.shape} != {exp_shape}")
            return
        got = arr.get(iface)
        if np.dtype(arr.dtype) != np.dtype(got.dtype):
            chk.true(label + " dtype", False, f"{arr.dtype} vs fetched {got.dtype}")
        chk.close(label, got, ref.reshape(exp_shape), rtol=rtol, atol=atol)

    def dtype_check(label, arr, ref):
        exp = np.asarray(ref).dtype
        if label.startswith("all") or label.startswith("any"):
            return  # documented deviation: 0/1 in the input dtype
        if np.dtype(arr.dtype) != exp:
            chk.true(label + " dtype", False, f"{arr.dtype} != numpy {exp}")

    def check_all_ops(tag, A, a, axes_list, dtype):
        rtol, atol = tol(dtype)
        # guard against degenerate (e.g. all-zero) generated inputs
        chk.true(f"{tag} input nondegenerate", len(np.unique(a)) >= min(a.size, 5),
                 f"({len(np.unique(a))} distinct values)")
        n_red = a.size
        for axis in axes_list:
            for keepdims in (False, True):
                if axis is not None and keepdims and a.ndim == 1:
                    continue
                kw = dict(axis=axis, keepdims=keepdims)
                k = f"{tag} axis={axis} keep={int(keepdims)}"
                ext = n_red if axis is None else int(
                    np.prod([a.shape[x] for x in np.atleast_1d(axis)]))
                for name, cfn, nfn, rt in [
                    ("sum", cnp.sum, np.sum, rtol * max(1, ext / 1e3)),
                    ("mean", cnp.mean, np.mean, rtol * max(1, ext / 1e3)),
                    ("max", cnp.max, np.max, 0.0),
                    ("min", cnp.amin, np.amin, 0.0),
                    ("count_nonzero", cnp.count_nonzero, np.count_nonzero, 0.0),
                    ("all", cnp.all, np.all, 0.0),
                    ("any", cnp.any, np.any, 0.0),
                    ("var", cnp.var, np.var, rtol * 100),
                    ("std", cnp.std, np.std, rtol * 100),
                ]:
                    r = cfn(A, **kw)
                    ref = nfn(a, **kw)
                    dtype_check(name, r, ref)
                    eq(f"{name} {k}", r, ref, rtol=rt, atol=atol * ext if rt else 0.0)
                if axis is None or isinstance(axis, int):
                    for name, cfn, nfn in [("argmax", cnp.argmax, np.argmax),
                                           ("argmin", cnp.argmin, np.argmin)]:
                        r = cfn(A, **kw)
                        ref = nfn(a, **kw)
                        dtype_check(name, r, ref)
                        eq(f"{name} {k}", r, ref)

    # ---- full + axis reductions over shapes, dtypes and views ----
    cases = [
        ((1000,), [None, 0]),
        ((600011,), [None]),                       # 3 chares on CPU (1-D tile 262144)
        ((37, 53), [None, 0, 1, (0, 1)]),
        ((600, 1100), [None, 0, 1]),               # 2x3 tiles
        ((70, 65, 130), [None, 0, 1, 2, (0, 2), (1, 2)]),  # several 3-D tiles
    ]
    seed = 1
    for shape, axes in cases:
        dtypes = ["float64", "float32", "int64", "int32"]
        if int(np.prod(shape)) > 100000:
            dtypes = ["float64", "int32"]
        for dt in dtypes:
            seed += 1
            A = make(shape, dt, seed)
            a = A.get(iface)
            check_all_ops(f"{dt}{shape}", A, a, axes, dt)

    # ---- views: offset, strided, and integer-indexed (size-1 dim) ----
    A = make((600, 1100), "float64", 77)
    a = A.get(iface)
    for key in [np.s_[13:590:3, 501:1093:7], np.s_[100:101, :], np.s_[:, 517:518],
                np.s_[5:600, 3:1100], np.s_[511:513, 1023:1025]]:
        check_all_ops(f"view{key}", A[key], a[key], [None, 0, 1], "float64")
    B = make((70, 65, 130), "int64", 78)
    b = B.get(iface)
    for key in [np.s_[3:70:2, 1:64, 60:129:5], np.s_[69:70, :, 64:66]]:
        check_all_ops(f"view3d{key}", B[key], b[key], [None, 0, 2, (0, 1)], "int64")
    V = make((600011,), "float32", 79)
    v = V.get(iface)
    check_all_ops("view1d", V[1000:600000:7], v[1000:600000:7], [None], "float32")
    # view of a view, and a 1-element view (travels as a broadcast operand)
    W = A[10:500, 20:900][5:400:2, ::3]
    w = a[10:500, 20:900][5:400:2, ::3]
    eq("view-of-view sum", W.sum(), w.sum(), rtol=1e-12, atol=1e-9)
    eq("view-of-view argmax", W.argmax(), w.argmax())
    eq("1-elem view max", A[7:8, 9:10].max(), a[7:8, 9:10].max())
    eq("1-elem view argmin", A[7:8, 9:10].argmin(axis=0), a[7:8, 9:10].argmin(axis=0))

    # ---- prod (values near 1 to stay finite) ----
    for dt in ["float64", "float32"]:
        P = 1.0 + 0.001 * make((300, 700), dt, 90)
        p = P.get(iface)
        r = tol(dt)[0] * 100
        eq(f"prod {dt}", P.prod(), np.prod(p), rtol=r)
        eq(f"prod {dt} axis0", cnp.prod(P, axis=0), np.prod(p, axis=0), rtol=r)
        eq(f"prod {dt} axis1 view", P[3:200:2].prod(axis=1), np.prod(p[3:200:2], axis=1), rtol=r)
    I = small_int((40, 7), np.int64, 91, 3, lo=1)
    ii = I.get(iface)
    eq("prod int64 axis1", I.prod(axis=1), np.prod(ii, axis=1))
    eq("prod int64 view", I[0:20, 0:2].prod(), np.prod(ii[0:20, 0:2]))
    J = small_int((10, 3), np.int32, 92, 2, lo=1)
    jj = J.get(iface)
    eq("prod int32", J.prod(), np.prod(jj))

    # ---- NaN propagation and first-NaN argmax ----
    N = make((300, 700), "float64", 93)
    N[5:6, 600:601] = float("nan")
    N[200:201, 3:4] = float("nan")
    n = N.get(iface)
    eq("nan max", N.max(), np.max(n))
    eq("nan argmax", N.argmax(), np.argmax(n))
    eq("nan argmin", N.argmin(), np.argmin(n))
    eq("nan max axis0", N.max(axis=0), np.max(n, axis=0))
    eq("nan argmin axis1", N.argmin(axis=1), np.argmin(n, axis=1))
    eq("nan sum", N.sum(), np.sum(n))
    eq("nan count_nonzero", cnp.count_nonzero(N), np.count_nonzero(n))

    # ---- methods, dtype=, out=, ddof ----
    A2 = make((300, 700), "float32", 94)
    a2 = A2.get(iface)
    eq("method sum dtype=f64", A2.sum(dtype=np.float64), a2.sum(dtype=np.float64), rtol=1e-12)
    eq("method mean axis1", A2.mean(axis=1), a2.mean(axis=1), rtol=1e-5, atol=1e-6)
    eq("method var ddof1", A2.var(ddof=1), a2.var(ddof=1), rtol=1e-4)
    eq("method std axis0 ddof1", A2.std(axis=0, ddof=1), a2.std(axis=0, ddof=1), rtol=1e-4)
    eq("method min", A2[1:].min(), a2[1:].min())
    eq("method any", (A2 * 0.0).any(), np.any(a2 * 0.0))
    eq("method all", (A2 * 0.0 + 1.0).all(axis=0), np.all(a2 * 0.0 + 1.0, axis=0))
    out = cnp.zeros((700,), dtype=np.float32)
    A2.max(axis=0, out=out)
    eq("out= max axis0", out, a2.max(axis=0))
    I32 = make((40, 900), "int32", 95)
    i32 = I32.get(iface)
    eq("int32 sum -> int64", I32.sum(), i32.sum())
    eq("int32 sum dtype=int32", I32.sum(dtype=np.int32), i32.sum(dtype=np.int32))
    eq("int32 mean", I32.mean(axis=0), i32.mean(axis=0), rtol=1e-12)

    # ---- results feeding later expressions ----
    C = make((600, 1100), "float64", 96)
    c = C.get(iface)
    centered = C - C.mean()
    eq("x - x.mean() then sum", centered.sum(), np.sum(c - c.mean()), atol=1e-6)
    eq("x / x.max()", (C / C.max())[0:3, 0:4], (c / c.max())[0:3, 0:4], rtol=1e-14)
    eq("keepdims result broadcast", C - C.mean(keepdims=True), c - c.mean(keepdims=True),
       rtol=1e-12, atol=1e-12)
    rs = C.sum(axis=1)
    eq("axis result reused", rs * 2.0 + rs.max(), c.sum(axis=1) * 2.0 + c.sum(axis=1).max(),
       rtol=1e-12, atol=1e-9)
    eq("chained axis reductions", C.max(axis=0).argmin(), np.argmin(np.max(c, axis=0)))
    eq("reduce of temporary", (C * C).sum(axis=0), np.sum(c * c, axis=0), rtol=1e-12)
    D = make((70, 65, 130), "float64", 97)
    d = D.get(iface)
    eq("3-D sum(axis=0).sum(axis=1)", D.sum(axis=0).sum(axis=1), d.sum(axis=0).sum(axis=1),
       rtol=1e-12, atol=1e-9)
    eq("3-D mean over (0,1) keepdims", D.mean(axis=(0, 1), keepdims=True),
       d.mean(axis=(0, 1), keepdims=True), rtol=1e-12, atol=1e-12)

    # several independent reductions in one DAG (any execution order)
    rs = [C.sum(), C.max(axis=0), C[1:].argmin(axis=1), D.std(axis=2), C.min()]
    refs = [c.sum(), c.max(axis=0), c[1:].argmin(axis=1), d.std(axis=2), c.min()]
    for i, (r, ref) in enumerate(zip(rs, refs)):
        eq(f"batched #{i}", r, ref, rtol=1e-12, atol=1e-9)

    # ---- norm ----
    x = make((600011,), "float64", 98)
    xv = x.get(iface)
    for o in [None, 2, 1, np.inf, -np.inf, 0]:
        eq(f"norm vec ord={o}", cnp.linalg.norm(x, o), np.linalg.norm(xv, o), rtol=1e-12)
    for o in [None, "fro", np.inf, -np.inf, 1, -1]:
        eq(f"norm mat ord={o}", cnp.linalg.norm(C, o), np.linalg.norm(c, o), rtol=1e-12)
    eq("norm axis=1", cnp.linalg.norm(C, axis=1), np.linalg.norm(c, axis=1), rtol=1e-12)
    eq("norm axis=0 view", cnp.linalg.norm(C[3:500:2, 7:], axis=0),
       np.linalg.norm(c[3:500:2, 7:], axis=0), rtol=1e-12)
    eq("norm int", cnp.linalg.norm(I32), np.linalg.norm(i32), rtol=1e-12)
    eq("norm 3-D", cnp.linalg.norm(D), np.linalg.norm(d), rtol=1e-12)

    # ---- bincount ----
    for dt, n in [("int64", 1000), ("int32", 600011)]:
        L = small_int((n, 2), dt, 99, 7) @ cnp.ones((2,), dtype=dt)  # labels in [0, 12]
        lv = L.get(iface)
        eq(f"bincount {dt} n={n}", cnp.bincount(L, minlength=13), np.bincount(lv, minlength=13))
        eq(f"bincount {dt} minlength>max", cnp.bincount(L, minlength=20),
           np.bincount(lv, minlength=20))
        eq(f"bincount {dt} interface", cnp.bincount(L, interface=iface), np.bincount(lv))
        eq(f"bincount {dt} view", cnp.bincount(L[3:n:5], minlength=13),
           np.bincount(lv[3:n:5], minlength=13))
        Wt = make((n,), "float64", 100)
        wv = Wt.get(iface)
        eq(f"bincount {dt} weights", cnp.bincount(L, weights=Wt, minlength=13),
           np.bincount(lv, weights=wv, minlength=13), rtol=1e-12, atol=1e-9)
        eq(f"bincount {dt} weights views", cnp.bincount(L[1:n - 1], weights=Wt[0:n - 2], minlength=13),
           np.bincount(lv[1:n - 1], weights=wv[0:n - 2], minlength=13), rtol=1e-12, atol=1e-9)
    # kmeans-style: labels = argmin over centroids, counts/weighted sums by label
    dist = make((5000, 7), "float64", 101)
    dv = dist.get(iface)
    labels = cnp.argmin(dist, axis=1)
    eq("kmeans argmin(axis=1)", labels, np.argmin(dv, axis=1))
    eq("kmeans amin(axis=1).sum()", cnp.sum(cnp.amin(dist, axis=1)), np.sum(np.amin(dv, axis=1)),
       rtol=1e-12)
    eq("kmeans bincount(labels)", cnp.bincount(labels, minlength=7),
       np.bincount(np.argmin(dv, axis=1), minlength=7))
    feat = make((5000,), "float64", 102)
    fv = feat.get(iface)
    eq("kmeans bincount(labels, w)", cnp.bincount(labels, weights=feat, minlength=7),
       np.bincount(np.argmin(dv, axis=1), weights=fv, minlength=7), rtol=1e-12, atol=1e-9)

    # ---- small host-uploaded arrays with hand-checked ties ----
    h = np.array([[3, 1, 3, 2], [0, 3, 3, -1], [3, 0, 0, 3]], dtype=np.int32)
    H = cnp.array(h)
    eq("ties argmax", H.argmax(), np.argmax(h))
    eq("ties argmax axis0", H.argmax(axis=0), np.argmax(h, axis=0))
    eq("ties argmax axis1", H.argmax(axis=1), np.argmax(h, axis=1))
    eq("ties argmin axis1", H.argmin(axis=1), np.argmin(h, axis=1))
    eq("ties argmin keepdims", H.argmin(axis=0, keepdims=True), np.argmin(h, axis=0, keepdims=True))


U.run("test_reductions", main)
