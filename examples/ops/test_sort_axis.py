#!/usr/bin/env python
"""N-D sort/argsort along split axes, complex sorts and the quantile sort fallback vs NumPy.

Shapes span several chares along the sort axis (+p2/+p4), including tiles with fewer
lines than chares along the axis, offset views, empty and size-1 arrays. Every dtype;
floats carry NaNs, infinities and signed zeros; complex carries NaN parts. Argsort is
compared with NumPy's stable argsort. Env: CNUM_HOST, CNUM_PORT, CNUM_ODF (default 4),
SAX_BIG (also run a large 2-D case).
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
CPLX = [np.complex64, np.complex128]


def make(shape, dt, rng, dups=False):
    dt = np.dtype(dt)
    n = int(np.prod(shape))
    if dt == np.bool_:
        return (rng.random(n) < 0.3).reshape(shape)
    if dt.kind == "c":
        f = np.float32 if dt == np.complex64 else np.float64
        re = make(shape, f, rng, dups).reshape(-1)
        im = make(shape, f, rng, True).reshape(-1)
        x = (re + 1j * im).astype(dt)
        if n > 10:
            k = max(1, n // 500)
            x[rng.integers(0, n, k)] = complex(np.nan, 1.0)
            x[rng.integers(0, n, k)] = complex(2.0, np.nan)
            x[rng.integers(0, n, k)] = complex(np.nan, np.nan)
            x[rng.integers(0, n, k)] = complex(-0.0, 0.0)
            x[rng.integers(0, n, k)] = complex(np.inf, -np.inf)
        return x.reshape(shape)
    if dt.kind == "f":
        x = (rng.integers(-5, 5, n) if dups else rng.standard_normal(n) * 100).astype(dt)
        if n > 10:
            k = max(1, n // 1000)
            for v in (np.nan, -0.0, 0.0, np.inf, -np.inf):
                x[rng.integers(0, n, k)] = v
        return x.reshape(shape)
    info = np.iinfo(dt)
    lo, hi = (max(info.min, -3), min(info.max, 3)) if dups else (info.min, info.max)
    return rng.integers(lo, hi, n, dtype=dt, endpoint=True).reshape(shape)


def main(chk):
    host = os.environ.get("CNUM_HOST", "127.0.0.1")
    port = int(os.environ.get("CNUM_PORT", "1234"))
    odf = int(os.environ.get("CNUM_ODF", "4"))
    iface = cnp.CharmNumericInterface()
    iface.connect(host, port, odf)
    print(f"connected {host}:{port} odf={odf}")
    rng = np.random.default_rng(11)

    def same(label, arr, ref):
        ref = np.asarray(ref)
        got = np.asarray(arr.get(iface))
        if np.dtype(arr.dtype) != ref.dtype or got.shape != ref.shape:
            return chk.true(label, False, f"{arr.dtype}{got.shape} != {ref.dtype}{ref.shape}")
        if ref.dtype.kind == "c":
            ok = (np.array_equal(got.real, ref.real, equal_nan=True)
                  and np.array_equal(got.imag, ref.imag, equal_nan=True))
        elif ref.dtype.kind == "f":
            ok = np.array_equal(got, ref, equal_nan=True)
        else:
            ok = np.array_equal(got, ref)
        bad = "" if ok else f"({int(np.count_nonzero(got != ref))} differ)"
        return chk.true(label, ok, bad)

    def both(tag, a, h, axis):
        same(f"sort {tag} axis={axis}", cnp.sort(a, axis=axis), np.sort(h, axis=axis))
        same(f"argsort {tag} axis={axis}", cnp.argsort(a, axis=axis),
             np.argsort(h, axis=axis, kind="stable"))

    # every dtype, both axes of a 2-D array split along both
    for dt in REAL + CPLX:
        for dups in (False, True):
            if dups and np.dtype(dt).kind not in "fic":
                continue
            h = make((1500, 1300), dt, rng, dups)
            a = cnp.asarray(h)
            tag = f"{np.dtype(dt).name}{' dups' if dups else ''} (1500, 1300)"
            for axis in (0, 1):
                both(tag, a, h, axis)

    # 3-D, every axis
    for dt in (np.float64, np.int32, np.complex64, np.uint8):
        h = make((200, 150, 90), dt, rng, dt == np.uint8)
        a = cnp.asarray(h)
        for axis in (0, 1, 2, -1):
            both(f"{np.dtype(dt).name} (200, 150, 90)", a, h, axis)

    # fewer lines than tiles along the axis; long thin and tall
    for shape, axis in (((5, 200_003), 1), ((3, 300_007), -1), ((180_001, 7), 0),
                        ((2, 3, 150_001), 2), ((150_001, 2, 3), 0)):
        for dt in (np.float32, np.int64, np.complex128):
            h = make(shape, dt, rng)
            both(f"{np.dtype(dt).name} {shape}", cnp.asarray(h), h, axis)

    # few lines of 2**22+ elements: one 1-D sort per line
    for shape, axis in (((3, 4_194_311), 1), ((4_194_311, 1), 0)):
        h = make(shape, np.float32, rng, True)
        both(f"float32 {shape}", cnp.asarray(h), h, axis)

    # views (offset decompositions), uneven tiles
    h = make((2100, 1900), np.float64, rng, True)
    a = cnp.asarray(h)
    v, hv = a[37:2050, 101:1899], h[37:2050, 101:1899]
    for axis in (0, 1):
        both("float64 offset view", v, hv, axis)
    h3 = make((210, 170, 100), np.int16, rng)
    a3 = cnp.asarray(h3)
    for axis in (0, 1, 2):
        both("int16 3-D offset view", a3[3:, 5:160, 1:], h3[3:, 5:160, 1:], axis)

    # degenerate shapes
    for shape in ((0, 5), (5, 0), (1, 1), (1, 70_000), (70_000, 1), (1, 1, 1), (0, 3, 4)):
        h = make(shape, np.float64, rng)
        a = cnp.asarray(h)
        for axis in range(len(shape)):
            both(f"float64 {shape}", a, h, axis)

    # signed zeros / NaNs: values equal NumPy's, argsort stable
    z = np.tile(np.array([0.0, -0.0, np.nan, 1.0, -np.nan, -1.0]), (2000, 60))
    same("argsort signed zeros/NaNs axis=0", cnp.argsort(cnp.asarray(z), axis=0),
         np.argsort(z, axis=0, kind="stable"))
    same("sort signed zeros/NaNs axis=0", cnp.sort(cnp.asarray(z), axis=0), np.sort(z, axis=0))

    # complex 1-D (distributed sample sort) and complex N-D axis=None
    for n in (7, 262_145, 1_000_003, 3_000_001):  # 3e6: several chares on GPU builds too
        for dt in CPLX:
            h = make((n,), dt, rng, n > 10)
            both(f"{np.dtype(dt).name} 1-D n={n}", cnp.asarray(h), h, 0)
    h = make((900, 700), np.complex128, rng, True)
    same("sort complex128 2-D axis=None", cnp.sort(cnp.asarray(h), axis=None),
         np.sort(h, axis=None))
    c = np.array([complex(np.nan, np.nan), complex(1, np.nan), complex(np.nan, 0), 3 + 0j,
                  complex(2, np.nan), complex(np.nan, -1), -1 + 5j, -1 - 5j] * 10_000,
                 dtype=np.complex64)
    both("complex64 NaN classes", cnp.asarray(c), c, 0)
    both("complex64 NaN classes 2-D", cnp.asarray(c.reshape(8, -1).T.copy()),
         c.reshape(8, -1).T.copy(), 0)

    # known difference: complex zero parts compare equal, so sort() returns them as +0.0
    cz = np.array([complex(-0.0, 2), complex(0.0, -2), complex(1, -0.0), complex(-0.0, -0.0)] * 300)
    got = cnp.sort(cnp.asarray(cz)).get(iface)
    chk.true("sort complex signed zeros: values", np.array_equal(got, np.sort(cz)))
    chk.true("sort complex signed zeros: +0.0 parts (documented)",
             not np.any(np.signbit(got.real)) and not np.any(np.signbit(got.imag[got.imag == 0])))
    same("argsort complex signed zeros stable", cnp.argsort(cnp.asarray(cz)),
         np.argsort(cz, kind="stable"))

    # quantile/percentile through the sort: axis on N-D, many ranks
    h2 = rng.standard_normal((1700, 1200))
    a2 = cnp.asarray(h2)
    for axis in (0, 1):
        g = cnp.quantile(a2, [0.1, 0.5, 0.93], axis=axis)
        chk.close(f"quantile axis={axis}", g.get(iface), np.quantile(h2, [0.1, 0.5, 0.93],
                                                                         axis=axis))
    q = np.linspace(0, 1, 300)
    g = cnp.quantile(cnp.asarray(h2.reshape(-1)), q)
    chk.close("quantile 300 ranks (sort path)", g.get(iface), np.quantile(h2.reshape(-1), q))
    g = cnp.percentile(a2, q * 100, axis=1)
    chk.close("percentile 300 ranks axis=1", g.get(iface), np.percentile(h2, q * 100, axis=1))

    # sort feeding other ops, in-place method
    h = make((1500, 1300), np.float32, rng)
    a = cnp.asarray(h)
    same("sort axis=0 then elementwise", cnp.sort(a, axis=0) * 2 + 1, np.sort(h, axis=0) * 2 + 1)
    b = cnp.asarray(h)
    b.sort(axis=0)
    same("Array.sort(axis=0) in place", b, np.sort(h, axis=0))

    if os.environ.get("SAX_BIG"):
        h = make((4100, 4097), np.float32, rng)
        a = cnp.asarray(h)
        for axis in (0, 1):
            both("float32 (4100, 4097)", a, h, axis)


if __name__ == "__main__":
    U.run("test_sort_axis", main)
