"""Comparisons and predicates vs NumPy (bool results): less, less_equal,
greater, greater_equal, equal, not_equal (and < <= > >= == !=), isnan, isinf,
isfinite, logical_and/or/xor/not (and & | ^ ~)."""

import numpy as np

import charmnumeric as cnp
from _ops_common import SHAPES, Checker, connect, floats, halves, ints, views_of, with_specials

iface = connect()
C = Checker("test_compare", iface)


def m(x, dt=None):
    """NumPy's bool result (dt is ignored)."""
    return np.asarray(x).astype(bool)


CMP = [
    ("less", cnp.less, np.less, lambda x, y: x < y),
    ("less_equal", cnp.less_equal, np.less_equal, lambda x, y: x <= y),
    ("greater", cnp.greater, np.greater, lambda x, y: x > y),
    ("greater_equal", cnp.greater_equal, np.greater_equal, lambda x, y: x >= y),
    ("equal", cnp.equal, np.equal, lambda x, y: x == y),
    ("not_equal", cnp.not_equal, np.not_equal, lambda x, y: x != y),
]

with np.errstate(all="ignore"):
    for shape in SHAPES:
        a = halves(shape)  # many exact ties with b
        b = floats(shape, -2.0, 2.0, seed=1).round()
        ah, bh = C.fetch(a), C.fetch(b)
        s = f" {shape}"
        for name, f, nf, op in CMP:
            C.check(name + s, f(a, b), m(nf(ah, bh), ah.dtype), dtype=bool)
            C.check(f"op {name}" + s, op(a, b), m(nf(ah, bh), ah.dtype))
            C.check(f"{name}(a, 0.5)" + s, f(a, 0.5), m(nf(ah, 0.5), ah.dtype))
            C.check(f"{name}(0.5, a)" + s, f(0.5, a), m(nf(0.5, ah), ah.dtype))
            C.check(f"op 1.0 {name} a" + s, op(1.0, a), m(nf(1.0, ah), ah.dtype))
        for vname, key in views_of(a):
            C.check(f"less{vname}" + s, a[key] < b[key], m(ah[key] < bh[key], ah.dtype))
            C.check(f"equal{vname}" + s, a[key] == b[key], m(ah[key] == bh[key], ah.dtype))
        C.check("logical_and" + s, cnp.logical_and(a > 0, b < 0), m(np.logical_and(ah > 0, bh < 0), ah.dtype))
        C.check("logical_or" + s, cnp.logical_or(a, b), m(np.logical_or(ah, bh), ah.dtype))
        C.check("logical_xor" + s, cnp.logical_xor(a, b), m(np.logical_xor(ah, bh), ah.dtype))
        C.check("logical_not" + s, cnp.logical_not(a), m(np.logical_not(ah), ah.dtype))
        C.check("& |" + s, ((a > 0) & (b > 0)) | (a == b), m(((ah > 0) & (bh > 0)) | (ah == bh), ah.dtype))
        C.check("^ ~" + s, ~((a > 0) ^ (b > 0)), m(~((ah > 0) ^ (bh > 0)), ah.dtype))
        C.check("mask arithmetic" + s, (a < b) * a + (a >= b) * b, np.minimum(ah, bh))

    # NaN / inf semantics
    x = with_specials(floats((64,), -4.0, 4.0, seed=2))
    y = with_specials(floats((64,), -4.0, 4.0, seed=3), (np.nan, 1.0, -np.inf, 0.0, -0.0, np.inf))
    xh, yh = C.fetch(x), C.fetch(y)
    for name, f, nf, op in CMP:
        C.check(f"{name} specials", f(x, y), m(nf(xh, yh), xh.dtype))
        C.check(f"{name}(x, nan)", f(x, np.nan), m(nf(xh, np.nan), xh.dtype))
        C.check(f"{name}(x, inf)", f(x, np.inf), m(nf(xh, np.inf), xh.dtype))
    C.check("isnan", cnp.isnan(x), m(np.isnan(xh), xh.dtype))
    C.check("isinf", cnp.isinf(x), m(np.isinf(xh), xh.dtype))
    C.check("isfinite", cnp.isfinite(x), m(np.isfinite(xh), xh.dtype))
    C.check("logical_not nan", cnp.logical_not(x), m(np.logical_not(xh), xh.dtype))
    C.check("logical_and nan", cnp.logical_and(x, y), m(np.logical_and(xh, yh), xh.dtype))
    for vname, key in views_of(x):
        C.check(f"isnan{vname}", cnp.isnan(x[key]), m(np.isnan(xh[key]), xh.dtype))
    for shape in SHAPES[:4]:
        z = floats(shape, -1.0, 1.0, seed=4) / floats(shape, -1.0, 1.0, seed=4).round()
        zh = C.fetch(z)
        C.check(f"isnan/isinf/isfinite {shape}", cnp.isnan(z) + 2.0 * cnp.isinf(z) + 4.0 * cnp.isfinite(z),
                np.isnan(zh) + 2.0 * np.isinf(zh) + 4.0 * np.isfinite(zh))

    # float32
    a32 = halves((600, 7), dtype=np.float32)
    a32h = C.fetch(a32)
    C.check("less f32 (0.1 rounds to f32)", a32 < 0.1, m(a32h < np.float32(0.1), np.float32), dtype=bool)
    C.check("isfinite f32", cnp.isfinite(a32 / 0.0), m(np.isfinite(a32h / np.float32(0.0)), np.float32))

    # integers: exact; float scalars are compared exactly (not truncated)
    for dt in (np.int64, np.int32):
        a = ints((600, 7), -20, 20, dtype=dt)
        b = ints((600, 7), -3, 3, dtype=dt)
        ah, bh = C.fetch(a), C.fetch(b)
        n = np.dtype(dt).name
        for name, f, nf, op in CMP:
            C.check(f"{name} {n}", f(a, b), m(nf(ah, bh), dt), dtype=bool)
            for sc in (2, 2.5, -2.5, 3.0, 1e30, -1e30, np.inf, np.nan):
                C.check(f"{name}(a, {sc}) {n}", f(a, sc), m(nf(ah, sc), dt), dtype=bool)
                C.check(f"{name}({sc}, a) {n}", f(sc, a), m(nf(sc, ah), dt), dtype=bool)
        C.check(f"isnan {n}", cnp.isnan(a), np.zeros(ah.shape, bool), dtype=bool)
        C.check(f"isinf {n}", cnp.isinf(a), np.zeros(ah.shape, bool), dtype=bool)
        C.check(f"isfinite {n}", cnp.isfinite(a), np.ones(ah.shape, bool), dtype=bool)
        C.check(f"logical_xor {n}", cnp.logical_xor(a, b), m(np.logical_xor(ah, bh), dt), dtype=bool)

    # old-backend (fix-temporaries) method API
    a = floats((37, 41), -3.0, 3.0, seed=5)
    b = floats((37, 41), -3.0, 3.0, seed=6)
    ah, bh = C.fetch(a), C.fetch(b)
    C.check("a.greater_than(b)", a.greater_than(b), m(ah > bh, ah.dtype))
    C.check("a.less_than(b)", a.less_than(b), m(ah < bh, ah.dtype))
    C.check("a.equal(a)", a.equal(a), np.ones(ah.shape, bool))
    C.check("a.equal(b, epsilon=0.5)", a.equal(b, epsilon=0.5), m(np.abs(ah - bh) <= 0.5, ah.dtype))

C.finish()
