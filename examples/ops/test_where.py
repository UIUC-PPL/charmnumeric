"""where(cond, x, y) and clip(a, lo, hi) (3-input fused ops) vs NumPy, with
array/scalar/size-1 operands, views, NaN conditions and int arrays."""

import numpy as np

import charmnumeric as cnp
from _ops_common import SHAPES, Checker, connect, floats, ints, views_of, with_specials

iface = connect()
C = Checker("test_where", iface)

with np.errstate(all="ignore"):
    for shape in SHAPES:
        a = floats(shape, -3.0, 3.0, seed=1)
        b = floats(shape, -3.0, 3.0, seed=2)
        ah, bh = C.fetch(a), C.fetch(b)
        s = f" {shape}"
        C.check("where(a>b, a, b)" + s, cnp.where(a > b, a, b), np.where(ah > bh, ah, bh))
        C.check("where(a>0, a, 0.0)" + s, cnp.where(a > 0, a, 0.0), np.where(ah > 0, ah, 0.0))
        C.check("where(a>0, -1.0, b)" + s, cnp.where(a > 0, -1.0, b), np.where(ah > 0, -1.0, bh))
        C.check("where(mask, 1.0, 0.0)" + s, cnp.where(a < b, 1.0, 0.0), np.where(ah < bh, 1.0, 0.0))
        C.check("where(a, a, b) (nonzero)" + s, cnp.where(a - a.round(), a, b),
                np.where(ah - np.round(ah), ah, bh))
        C.check("where(True, a, b)" + s, cnp.where(True, a, b), ah)
        C.check("where(0, a, b)" + s, cnp.where(0, a, b), bh)
        C.check("clip" + s, cnp.clip(a, -1.0, 1.5), np.clip(ah, -1.0, 1.5))
        C.check("clip arrays" + s, cnp.clip(a, b, b + 1.0), np.clip(ah, bh, bh + 1.0))
        C.check("clip(min only)" + s, cnp.clip(a, -1.0, None), np.clip(ah, -1.0, None))
        C.check("clip(max only)" + s, cnp.clip(a, None, 1.0), np.clip(ah, None, 1.0))
        C.check("a.clip(min=, max=)" + s, a.clip(min=-0.5, max=0.5), ah.clip(min=-0.5, max=0.5))
        C.check("cnp.clip(min=, max=)" + s, cnp.clip(a, min=-0.5, max=0.5), np.clip(ah, -0.5, 0.5))
        for vname, key in views_of(a):
            C.check(f"where{vname}" + s, cnp.where(a[key] > 0, a[key], b[key]),
                    np.where(ah[key] > 0, ah[key], bh[key]))
            C.check(f"clip{vname}" + s, cnp.clip(a[key], -1.0, 1.0), np.clip(ah[key], -1.0, 1.0))
        if len(shape) == 1:
            C.check("where shifted" + s, cnp.where(a[1:] > a[:-1], a[1:], a[:-1]),
                    np.where(ah[1:] > ah[:-1], ah[1:], ah[:-1]))
        # where fused with a chain and written into a slice (mask assignment)
        t = a.copy()
        t[...] = cnp.where(cnp.isfinite(cnp.log(a)), cnp.log(a), -1.0)
        C.check("mask assignment" + s, t, np.where(np.isfinite(np.log(ah)), np.log(ah), -1.0),
                rtol=1e-14)

    # NaN as condition is truthy; NaN propagates through clip
    x = with_specials(floats((64,), -4.0, 4.0, seed=3))
    xh = C.fetch(x)
    C.check("where(nan cond)", cnp.where(x, 1.0, 0.0), np.where(xh, 1.0, 0.0))
    C.check("clip nan", cnp.clip(x, -1.0, 1.0), np.clip(xh, -1.0, 1.0))
    C.check("where(isnan, 0, x)", cnp.where(cnp.isnan(x), 0.0, x), np.where(np.isnan(xh), 0.0, xh))

    # size-1 (broadcast) operands, with the ndim of the other operands: a 1-D
    # (1,) array broadcast into a 2-D kernel hangs the backend (also b + x).
    lo = cnp.full((1, 1), -0.25)
    hi = cnp.full((1, 1), 0.75)
    C.fetch(lo)
    C.fetch(hi)
    a = floats((37, 41), -3.0, 3.0, seed=4)
    ah = C.fetch(a)
    C.check("clip(size-1 bounds)", cnp.clip(a, lo, hi), np.clip(ah, -0.25, 0.75))
    C.check("where(size-1 x)", cnp.where(a > 0, hi, a), np.where(ah > 0, 0.75, ah))
    C.check("where(size-1 cond)", cnp.where(hi, a, 0.0), ah)

    # float32 and ints
    a32 = floats((600, 7), -3.0, 3.0, seed=5, dtype=np.float32)
    a32h = C.fetch(a32)
    C.check("where f32", cnp.where(a32 > 0, a32, 0.0), np.where(a32h > 0, a32h, np.float32(0)))
    C.check("clip f32", cnp.clip(a32, -1.0, 1.0), np.clip(a32h, -1.0, 1.0))
    for dt in (np.int64, np.int32):
        a = ints((600, 7), -20, 20, dtype=dt)
        b = ints((600, 7), -3, 3, dtype=dt)
        ah, bh = C.fetch(a), C.fetch(b)
        n = np.dtype(dt).name
        C.check(f"where {n}", cnp.where(a > b, a, b), np.where(ah > bh, ah, bh), dtype=dt)
        C.check(f"where(b, a, 7) {n}", cnp.where(b, a, 7), np.where(bh, ah, 7), dtype=dt)
        C.check(f"clip {n}", cnp.clip(a, -5, 5), np.clip(ah, -5, 5), dtype=dt)
        C.check(f"clip arrays {n}", cnp.clip(a, b, b + 4), np.clip(ah, bh, bh + 4), dtype=dt)
        C.raises(f"where({n}, 0.5) rejected", TypeError, lambda: cnp.where(a > 0, a, 0.5))
    C.check("where with float64 cond", cnp.where(cnp.zeros((600, 7)), a32, a32 + 1),
            C.fetch(a32) + 1, rtol=1e-6, dtype=np.float32)

    # old-backend (fix-temporaries) API: cond.where(x, y)
    a = floats((37, 41), -3.0, 3.0, seed=6)
    ah = C.fetch(a)
    C.check("(a > 0).where(a, -a)", (a > 0).where(a, -a), np.where(ah > 0, ah, -ah))
    C.check("ndarray.where(a > 0, 1.0, 0.0)", cnp.ndarray.where(a > 0, 1.0, 0.0), np.where(ah > 0, 1.0, 0.0))

C.finish()
