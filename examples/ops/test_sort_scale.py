#!/usr/bin/env python
"""Distributed 1-D sort/argsort, cumsum family and quantile against NumPy.

Sizes span uneven tiles and many chares (+p4: up to 16), every dtype, duplicates,
NaNs, signed zeros, infinities, empty and size-1 arrays. Argsort is compared with
NumPy's stable argsort. Env: CNUM_HOST, CNUM_PORT, CNUM_ODF (default 4),
SSC_ONLY (sort,scan,quantile).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cupynumeric"))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402

REAL = [np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.int8, np.uint64,
        np.uint32, np.uint16, np.uint8, np.bool_]
SIZES = [1, 2, 7, 262_145, 1_000_003, 3_000_001]


def make(n, dt, rng, kind):
    dt = np.dtype(dt)
    if dt == np.bool_:
        return rng.random(n) < 0.3
    if dt.kind == "f":
        if kind == "dups":
            x = rng.integers(-5, 5, n).astype(dt)
        else:
            x = (rng.standard_normal(n) * 100).astype(dt)
        if n > 10:
            k = max(1, n // 1000)
            x[rng.integers(0, n, k)] = np.nan
            x[rng.integers(0, n, k)] = -0.0
            x[rng.integers(0, n, k)] = 0.0
            x[rng.integers(0, n, k)] = np.inf
            x[rng.integers(0, n, k)] = -np.inf
        return x
    info = np.iinfo(dt)
    if kind == "dups":
        lo, hi = max(info.min, -3), min(info.max, 3)
    else:
        lo, hi = info.min, info.max
    return rng.integers(lo, hi, n, dtype=dt, endpoint=True)


def main(chk):
    host = os.environ.get("CNUM_HOST", "127.0.0.1")
    port = int(os.environ.get("CNUM_PORT", "1234"))
    odf = int(os.environ.get("CNUM_ODF", "4"))
    iface = cnp.CharmNumericInterface()
    iface.connect(host, port, odf)
    print(f"connected {host}:{port} odf={odf}")
    only = set(os.environ.get("SSC_ONLY", "sort,scan,quantile").split(","))
    rng = np.random.default_rng(7)

    def same(label, arr, ref):
        ref = np.asarray(ref)
        got = arr.get(iface)
        if np.dtype(arr.dtype) != ref.dtype or got.shape != ref.shape:
            return chk.true(label, False, f"{arr.dtype}{got.shape} != {ref.dtype}{ref.shape}")
        if ref.dtype.kind in "fc":
            ok = np.array_equal(got, ref, equal_nan=True)
        else:
            ok = np.array_equal(got, ref)
        bad = "" if ok else f"({int(np.count_nonzero(got != ref))} differ)"
        return chk.true(label, ok, bad)

    if "sort" in only:
        for n in SIZES:
            for dt in REAL:
                if n > 1_000_003 and dt not in (np.float64, np.int32, np.uint8):
                    continue
                for kind in ("rand", "dups"):
                    h = make(n, dt, rng, kind)
                    a = cnp.asarray(h)
                    tag = f"{np.dtype(dt).name} n={n} {kind}"
                    same(f"sort {tag}", cnp.sort(a), np.sort(h))
                    same(f"argsort {tag}", cnp.argsort(a), np.argsort(h, kind="stable"))
        h = make(1_000_003, np.float64, rng, "dups")
        a = cnp.asarray(h)
        same("sort kind=stable", cnp.sort(a, kind="stable"), np.sort(h))
        same("argsort kind=quicksort (stable result)", cnp.argsort(a, kind="quicksort"),
             np.argsort(h, kind="stable"))
        same("Array.argsort", a.argsort(), np.argsort(h, kind="stable"))
        same("sort offset view", cnp.sort(a[100_003:650_004]), np.sort(h[100_003:650_004]))
        same("argsort offset view", cnp.argsort(a[100_003:650_004]),
             np.argsort(h[100_003:650_004], kind="stable"))
        same("sort(axis=None) of 2-D", cnp.sort(cnp.asarray(h[:600_000].reshape(600, 1000)),
                                                axis=None), np.sort(h[:600_000]))
        same("sort feeds elementwise", cnp.sort(a) * 2.0 + 1.0, np.sort(h) * 2.0 + 1.0)
        c = (rng.standard_normal(5000) + 1j * rng.integers(0, 3, 5000)).astype(np.complex64)
        c[17] = complex(np.nan, 1)
        same("sort complex64 (gather path)", cnp.sort(cnp.asarray(c)), np.sort(c))
        try:
            cnp.sort(a, kind="bogus")
            chk.true("bad kind raises", False)
        except ValueError:
            chk.true("bad kind raises", True)
        e = cnp.asarray(np.zeros(0))
        same("sort empty", cnp.sort(e), np.sort(np.zeros(0)))

    if "scan" in only:
        for n in SIZES:
            for dt in (np.int32, np.int64, np.uint8, np.bool_, np.float64, np.float32):
                h = make(n, dt, rng, "dups")
                if np.dtype(dt).kind == "f":
                    h = np.where(np.isnan(h), 0, h).astype(dt)
                    h = np.where(np.isinf(h), 1, h).astype(dt)
                a = cnp.asarray(h)
                tag = f"{np.dtype(dt).name} n={n}"
                ref = np.cumsum(h)
                if ref.dtype.kind == "f":
                    tol = 1e-3 if dt == np.float32 else 1e-9
                    chk.close(f"cumsum {tag}", cnp.cumsum(a).get(iface), ref, rtol=tol,
                              atol=tol * np.abs(h).sum())
                else:
                    same(f"cumsum {tag}", cnp.cumsum(a), ref)
                if np.dtype(dt).kind in "iu":
                    o = cnp.zeros(n, dtype=dt)
                    r = cnp.cumsum(a, out=o)
                    chk.true(f"cumsum out= is out {tag}", r is o)
                    oh = np.zeros(n, dtype=dt)
                    np.cumsum(h, out=oh)
                    same(f"cumsum out={np.dtype(dt).name} {tag}", o, oh)
                    same(f"cumprod {tag}", cnp.cumprod(a), np.cumprod(h))
        h = rng.standard_normal(1_000_003)
        h[rng.integers(0, h.size, 50)] = np.nan
        a = cnp.asarray(h)
        chk.close("nancumsum 1e6", cnp.nancumsum(a).get(iface), np.nancumsum(h), rtol=1e-9,
                  atol=1e-6)
        chk.close("cumsum NaN 1e6", cnp.cumsum(a).get(iface), np.cumsum(h), rtol=1e-9,
                  atol=1e-6)
        i8 = rng.integers(-100, 100, 1_000_003, dtype=np.int8)
        same("cumsum int8 dtype=int16", cnp.cumsum(cnp.asarray(i8), dtype=np.int16),
             np.cumsum(i8, dtype=np.int16))
        of = cnp.zeros(1_000_003, dtype=np.float64)
        cnp.cumsum(cnp.asarray(i8), out=of)
        same("cumsum int8 out=float64", of, np.cumsum(i8).astype(np.float64))

    if "quantile" in only:
        methods = ["linear", "lower", "higher", "nearest", "midpoint", "inverted_cdf",
                   "median_unbiased", "closest_observation"]
        qs = [0.5, np.array([0.0, 0.37, 0.42, 0.5, 0.67, 0.83, 0.99, 1.0]),
              np.array([[0.1, 0.9], [0.25, 0.75]])]
        for n in (1, 7, 262_145, 3_000_001):
            for dt in (np.float64, np.float32, np.uint32, np.int64, np.int8):
                h = make(n, dt, rng, "rand")
                if np.dtype(dt).kind == "f":
                    h = np.where(np.isnan(h), 1.5, h).astype(dt)
                a = cnp.asarray(h)
                for m in methods if n == 262_145 else ["linear"]:
                    for q in qs:
                        ref = np.quantile(h, q, method=m)
                        got = cnp.quantile(a, q, method=m)
                        g = np.asarray(got.get(iface)) if hasattr(got, "get") else np.asarray(got)
                        tag = f"{np.dtype(dt).name} n={n} {m} q{np.shape(q)}"
                        ok = g.shape == np.shape(ref) and np.allclose(g, ref, rtol=1e-6,
                                                                      equal_nan=True)
                        chk.true(f"quantile {tag}", ok, "" if ok else f"{g} vs {ref}")
        h = rng.standard_normal(1_000_003)
        h[12345] = np.nan
        a = cnp.asarray(h)
        g = cnp.quantile(a, [0.1, 0.5]).get(iface)
        chk.true("quantile with NaN is NaN", bool(np.all(np.isnan(g))), f"{g}")
        h = np.repeat(np.arange(5.0), 200_001)
        a = cnp.asarray(h)
        g = cnp.quantile(a, [0.0, 0.3, 0.5, 0.99, 1.0]).get(iface)
        chk.close("quantile all-duplicates", g, np.quantile(h, [0.0, 0.3, 0.5, 0.99, 1.0]))
        g = cnp.percentile(a, 40)
        chk.close("percentile scalar", np.asarray(g.get(iface)), np.percentile(h, 40))
        h2 = rng.standard_normal((300, 400))
        g = cnp.quantile(cnp.asarray(h2), 0.3, axis=1)
        chk.close("quantile axis=1 (sort path)", g.get(iface), np.quantile(h2, 0.3, axis=1))
        g = cnp.quantile(cnp.asarray(h2), [0.3, 0.6])
        chk.close("quantile 2-D axis=None", g.get(iface), np.quantile(h2, [0.3, 0.6]))


if __name__ == "__main__":
    U.run("test_sort_scale", main)
