"""Binary elementwise ops vs NumPy: power/**, maximum, minimum, fmax, fmin,
remainder/%, floor_divide/'//', divmod, reverse operators, int and float."""

import numpy as np

import charmnumeric as cnp
from _ops_common import SHAPES, Checker, connect, floats, ints, views_of, with_specials

iface = connect()
C = Checker("test_binary", iface)
TOL64 = dict(rtol=1e-14, atol=1e-300)

with np.errstate(all="ignore"):
    for shape in SHAPES:
        a = floats(shape, -6.0, 6.0, seed=1)
        b = floats(shape, -3.0, 3.0, seed=2)
        p = floats(shape, 0.1, 4.0, seed=3)
        ah, bh, ph = C.fetch(a), C.fetch(b), C.fetch(p)
        s = f" {shape}"
        C.check("power" + s, cnp.power(p, b), np.power(ph, bh), **TOL64)
        C.check("p ** b" + s, p ** b, ph ** bh, **TOL64)
        C.check("p ** 2.5" + s, p ** 2.5, ph ** 2.5, **TOL64)
        C.check("a ** 2" + s, a ** 2, ah ** 2)
        C.check("a ** 3" + s, a ** 3, ah ** 3, **TOL64)
        C.check("2 ** a" + s, 2.0 ** a, 2.0 ** ah, **TOL64)
        C.check("maximum" + s, cnp.maximum(a, b), np.maximum(ah, bh))
        C.check("minimum" + s, cnp.minimum(a, b), np.minimum(ah, bh))
        C.check("fmax" + s, cnp.fmax(a, b), np.fmax(ah, bh))
        C.check("fmin" + s, cnp.fmin(a, b), np.fmin(ah, bh))
        C.check("maximum(a, 0.5)" + s, cnp.maximum(a, 0.5), np.maximum(ah, 0.5))
        C.check("minimum(0.5, a)" + s, cnp.minimum(0.5, a), np.minimum(0.5, ah))
        C.check("a % b" + s, a % b, ah % bh)
        C.check("remainder" + s, cnp.remainder(a, b), np.remainder(ah, bh))
        C.check("a % 1.5" + s, a % 1.5, ah % 1.5)
        C.check("a % -1.5" + s, a % -1.5, ah % -1.5)
        C.check("7.0 % b" + s, 7.0 % b, 7.0 % bh)
        C.check("a // b" + s, a // b, ah // bh)
        C.check("floor_divide" + s, cnp.floor_divide(a, b), np.floor_divide(ah, bh))
        C.check("a // -0.7" + s, a // -0.7, ah // -0.7)
        C.check("5.0 // b" + s, 5.0 // b, 5.0 // bh)
        q, r = divmod(a, b)
        C.check("divmod[0]" + s, q, np.floor_divide(ah, bh))
        C.check("divmod[1]" + s, r, np.remainder(ah, bh))
        for vname, key in views_of(a):
            C.check(f"maximum{vname}" + s, cnp.maximum(a[key], b[key]), np.maximum(ah[key], bh[key]))
            C.check(f"%{vname}" + s, a[key] % b[key], ah[key] % bh[key])
            C.check(f"**{vname}" + s, p[key] ** b[key], ph[key] ** bh[key], **TOL64)
        # shifted views of the same array (halo exchange in the backend)
        if len(shape) == 1:
            C.check("maximum(a[1:], a[:-1])" + s, cnp.maximum(a[1:], a[:-1]), np.maximum(ah[1:], ah[:-1]))
        elif len(shape) == 2:
            C.check("minimum shifted 2d" + s, cnp.minimum(a[1:, 1:], a[:-1, :-1]),
                    np.minimum(ah[1:, 1:], ah[:-1, :-1]))
        # fused chain mixing new and old ops
        C.check("chain" + s, cnp.maximum(a, b) ** 2 - (a // 0.5) * (b % 0.25),
                np.maximum(ah, bh) ** 2 - (ah // 0.5) * (bh % 0.25), **TOL64)

    # NaN/inf/-0.0 semantics
    x = with_specials(floats((64,), -4.0, 4.0, seed=4))
    y = with_specials(floats((64,), -2.0, 2.0, seed=5), (1.0, np.nan, 2.0, 0.0, -0.0, np.inf, -np.inf))
    xh, yh = C.fetch(x), C.fetch(y)
    C.check("maximum nan", cnp.maximum(x, y), np.maximum(xh, yh))
    C.check("minimum nan", cnp.minimum(x, y), np.minimum(xh, yh))
    C.check("fmax nan", cnp.fmax(x, y), np.fmax(xh, yh))
    C.check("fmin nan", cnp.fmin(x, y), np.fmin(xh, yh))
    C.check("mod specials", x % y, np.remainder(xh, yh))
    C.check("floordiv specials", x // y, np.floor_divide(xh, yh))
    C.check("power specials", cnp.power(cnp.abs(x), y), np.power(np.abs(xh), yh), **TOL64)
    z = floats((64,), -4.0, 4.0, seed=6) * 0.0  # zero divisors
    zh = C.fetch(z)
    C.check("mod by zero", x % z, np.remainder(xh, zh))
    C.check("floordiv by zero", x // z, np.floor_divide(xh, zh))

    # float32
    a32 = floats((600, 7), -6.0, 6.0, seed=7, dtype=np.float32)
    b32 = floats((600, 7), 0.5, 3.0, seed=8, dtype=np.float32)
    a32h, b32h = C.fetch(a32), C.fetch(b32)
    C.check("power f32", cnp.power(b32, a32), np.power(b32h, a32h), rtol=2e-6)
    C.check("maximum f32", cnp.maximum(a32, b32), np.maximum(a32h, b32h))
    C.check("% f32", a32 % b32, a32h % b32h, rtol=1e-6, atol=1e-6)
    C.check("// f32", a32 // b32, a32h // b32h)

    # integers (int64 and int32), including negative operands and zero divisors
    for dt in (np.int64, np.int32):
        n = np.dtype(dt).name
        for shape in [(37,), (600, 7), (5, 7, 9)]:
            a = ints(shape, -40, 40, dtype=dt, mult=7)
            b = ints(shape, -6, 6, dtype=dt)
            e = ints(shape, 0, 9, dtype=dt)
            ah, bh, eh = C.fetch(a), C.fetch(b), C.fetch(e)
            s = f" {n} {shape}"
            nz = bh != 0
            C.check("a % b" + s, a % b, np.where(nz, np.remainder(ah, np.where(nz, bh, 1)), 0), dtype=dt)
            C.check("a // b" + s, a // b, np.where(nz, np.floor_divide(ah, np.where(nz, bh, 1)), 0), dtype=dt)
            C.check("a % 7" + s, a % 7, ah % 7, dtype=dt)
            C.check("a % -7" + s, a % -7, ah % -7, dtype=dt)
            C.check("a // 4" + s, a // 4, ah // 4, dtype=dt)
            C.check("a // -4" + s, a // -4, ah // -4, dtype=dt)
            C.check("100 // b" + s, 100 // b, np.where(nz, 100 // np.where(nz, bh, 1), 0), dtype=dt)
            C.check("a ** e" + s, a ** e, ah ** eh, dtype=dt)
            C.check("a ** 3" + s, a ** 3, ah ** 3, dtype=dt)
            C.check("2 ** e" + s, 2 ** e, 2 ** eh, dtype=dt)
            C.check("maximum" + s, cnp.maximum(a, b), np.maximum(ah, bh), dtype=dt)
            C.check("minimum(a, 3)" + s, cnp.minimum(a, 3), np.minimum(ah, 3), dtype=dt)
            C.check("maximum(a, 1.0)" + s, cnp.maximum(a, 1.0), np.maximum(ah, 1), dtype=dt)
            C.check("fmax" + s, cnp.fmax(a, b), np.fmax(ah, bh), dtype=dt)
        C.raises(f"int ** negative {n}", ValueError, lambda: a ** -1)
        C.raises(f"int % 1.5 {n}", TypeError, lambda: a % 1.5)
    # int64 wraparound and INT_MIN // -1 (NumPy wraps; must not trap)
    big = ints((40,), 0, 39, dtype=np.int64) + (2**62)
    bigh = C.fetch(big)
    C.check("int64 overflow power", big ** 2, bigh ** 2, dtype=np.int64)
    m1 = ints((40,), -1, -1, dtype=np.int64)
    C.check("x // -1", big // m1, bigh // -1, dtype=np.int64)

    # mixed array dtypes are rejected (single element type per kernel)
    C.raises("float64 % float32 rejected", TypeError, lambda: a32 % cnp.zeros((600, 7)))

    # old-backend (fix-temporaries) method API
    a = floats((37, 41), -3.0, 3.0, seed=9)
    b = floats((37, 41), 0.5, 2.0, seed=10)
    ah, bh = C.fetch(a), C.fetch(b)
    C.check("a.power(b)", b.power(a), bh ** ah, **TOL64)
    C.check("a.modulo(b)", a.modulo(b), ah % bh)
    C.check("a.max(b)", a.max(b), np.maximum(ah, bh))
    C.check("a.min(b)", a.min(b), np.minimum(ah, bh))
    C.check("a.add(b)", a.add(b), ah + bh)
    C.check("a.subtract(b)", a.subtract(b), ah - bh)
    C.check("a.multiply(b)", a.multiply(b), ah * bh)
    C.check("a.divide(b)", a.divide(b), ah / bh)
    C.check("a.weighted_average(b, 0.25, 0.75)", a.weighted_average(b, 0.25, 0.75), 0.25 * ah + 0.75 * bh)

C.finish()
