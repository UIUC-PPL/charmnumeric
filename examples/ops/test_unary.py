"""Unary elementwise ops vs NumPy: sqrt, log*, exp2, expm1, log1p, sin, cos,
abs, floor, ceil, rint/round, trunc, sign, square, reciprocal, negative,
and the old-backend Array methods (log(base), sqrt(), abs(), ...)."""

import numpy as np

import charmnumeric as cnp
from _ops_common import SHAPES, Checker, connect, floats, halves, ints, views_of, with_specials

iface = connect()
C = Checker("test_unary", iface)

TOL64 = dict(rtol=1e-13, atol=1e-300)
TOL32 = dict(rtol=2e-6, atol=1e-30)

# (name, cnp fn, numpy fn, input kind, exact)
FLOAT_OPS = [
    ("sqrt", cnp.sqrt, np.sqrt, "pos", False),
    ("log", cnp.log, np.log, "pos", False),
    ("log2", cnp.log2, np.log2, "pos", False),
    ("log10", cnp.log10, np.log10, "pos", False),
    ("log1p", cnp.log1p, np.log1p, "pos", False),
    ("exp2", cnp.exp2, np.exp2, "any", False),
    ("expm1", cnp.expm1, np.expm1, "any", False),
    ("sin", cnp.sin, np.sin, "any", False),
    ("cos", cnp.cos, np.cos, "any", False),
    ("abs", cnp.abs, np.abs, "any", True),
    ("fabs", cnp.fabs, np.fabs, "any", True),
    ("floor", cnp.floor, np.floor, "any", True),
    ("ceil", cnp.ceil, np.ceil, "any", True),
    ("rint", cnp.rint, np.rint, "halves", True),
    ("round", cnp.round, np.round, "halves", True),
    ("trunc", cnp.trunc, np.trunc, "any", True),
    ("sign", cnp.sign, np.sign, "any", True),
    ("square", cnp.square, np.square, "any", False),
    ("reciprocal", cnp.reciprocal, np.reciprocal, "any", False),
    ("negative", cnp.negative, np.negative, "any", True),
]

with np.errstate(all="ignore"):
    for shape in SHAPES:
        nd = len(shape)
        src = {
            "any": floats(shape, -4.0, 4.0, seed=1),
            "pos": floats(shape, 1e-3, 50.0, seed=2),
            "halves": halves(shape),
        }
        host = {k: C.fetch(v) for k, v in src.items()}
        for name, f, nf, kind, exact in FLOAT_OPS:
            tol = {} if exact else TOL64
            C.check(f"{name} {shape}", f(src[kind]), nf(host[kind]), **tol)
        # views (slices / strides) of the inputs
        for vname, key in views_of(src["any"]):
            for name, f, nf, kind, exact in FLOAT_OPS[:12:3]:
                tol = {} if exact else TOL64
                C.check(f"{name}{vname} {shape}", f(src[kind][key]), nf(host[kind][key]), **tol)
        # fused chain: several new ops in one kernel
        x, h = src["pos"], host["pos"]
        C.check(f"chain {shape}", cnp.sqrt(cnp.log1p(x)) * cnp.floor(x) - cnp.abs(cnp.sin(x)),
                np.sqrt(np.log1p(h)) * np.floor(h) - np.abs(np.sin(h)), rtol=1e-13, atol=1e-12)
        # round with decimals (NumPy's multiply / rint / divide algorithm)
        C.check(f"round(2) {shape}", cnp.round(src["any"], 2), np.round(host["any"], 2), **TOL64)
        C.check(f"round(-1) {shape}", (src["any"] * 10.0).round(-1), np.round(host["any"] * 10.0, -1), **TOL64)

    # special values (nan, +-inf, -0.0, 0.0) on a 1-D array
    x = with_specials(floats((64,), -4.0, 4.0, seed=3))
    xh = C.fetch(x)
    for name, f, nf, kind, exact in FLOAT_OPS:
        C.check(f"{name} specials", f(x), nf(xh), **({} if exact else TOL64))
    # tiny arguments where naive log(1+x) / exp(x)-1 lose all precision
    t = floats((100,), -1e-10, 1e-10, seed=4)
    th = C.fetch(t)
    C.check("log1p tiny", cnp.log1p(t), np.log1p(th), rtol=1e-14)
    C.check("expm1 tiny", cnp.expm1(t), np.expm1(th), rtol=1e-14)

    # float32
    x32 = floats((600, 7), 1e-2, 20.0, seed=5, dtype=np.float32)
    h32 = C.fetch(x32)
    for name, f, nf, kind, exact in FLOAT_OPS:
        if kind == "halves":
            continue
        C.check(f"{name} f32", f(x32), nf(h32), **({} if exact else TOL32))

    # broadcast (size-1) inputs: unary op on a scalar operand inside a kernel.
    # The size-1 array has the kernel's ndim: a 1-D (1,) array broadcast into a
    # 2-D kernel hangs the backend (also plain ``b + x``; not an op issue).
    b = cnp.full((1, 1), 2.25)
    C.fetch(b)
    xa = floats((37, 41), seed=6)
    xah = C.fetch(xa)
    C.check("x + sqrt(size-1)", xa + cnp.sqrt(b), xah + np.sqrt(2.25))
    C.check("sqrt(size-1)", cnp.sqrt(b), np.sqrt([[2.25]]))
    C.check("x * floor(size-1)", xa * cnp.floor(b), xah * np.floor(2.25))
    C.check("x - log(size-1)", xa - cnp.log(b), xah - np.log(2.25), **TOL64)

    # integer inputs: abs/floor/ceil/rint/sign/round/trunc keep the int dtype
    for dt in (np.int64, np.int32):
        xi = ints((600, 7), -50, 50, dtype=dt)
        hi = C.fetch(xi)
        C.check(f"abs {np.dtype(dt)}", cnp.abs(xi), np.abs(hi), dtype=dt)
        C.check(f"abs() op {np.dtype(dt)}", abs(xi), np.abs(hi), dtype=dt)
        C.check(f"sign {np.dtype(dt)}", cnp.sign(xi), np.sign(hi), dtype=dt)
        C.check(f"floor {np.dtype(dt)}", cnp.floor(xi), hi, dtype=dt)
        C.check(f"ceil {np.dtype(dt)}", cnp.ceil(xi), hi, dtype=dt)
        C.check(f"rint {np.dtype(dt)}", cnp.rint(xi), hi, dtype=dt)
        C.check(f"round {np.dtype(dt)}", cnp.round(xi), hi, dtype=dt)
        C.check(f"square {np.dtype(dt)}", cnp.square(xi), np.square(hi), dtype=dt)
        C.check(f"reciprocal {np.dtype(dt)}", cnp.reciprocal(xi),
                np.where(np.abs(hi) == 1, hi, 0), dtype=dt)
        C.raises(f"sqrt {np.dtype(dt)} rejected", TypeError, lambda: cnp.sqrt(xi))
        C.raises(f"log {np.dtype(dt)} rejected", TypeError, lambda: cnp.log(xi))

    # old-backend (fix-temporaries) method API
    x = floats((37, 41), 0.5, 9.0, seed=7)
    h = C.fetch(x)
    C.check("x.sqrt()", x.sqrt(), np.sqrt(h), **TOL64)
    C.check("x.log()", x.log(), np.log(h), **TOL64)
    C.check("x.log(2)", x.log(2), np.log2(h), **TOL64)
    C.check("x.log(10)", x.log(10), np.log10(h), **TOL64)
    C.check("x.log(3)", x.log(3), np.log(h) / np.log(3), **TOL64)
    C.check("x.log2()", x.log2(), np.log2(h), **TOL64)
    C.check("x.log10()", x.log10(), np.log10(h), **TOL64)
    C.check("x.abs()", (x - 4.0).abs(), np.abs(h - 4.0))
    C.check("x.absolute()", (x - 4.0).absolute(), np.abs(h - 4.0))
    C.check("x.negate()", x.negate(), -h)
    C.check("x.square()", x.square(), h * h)
    C.check("x.reciprocal()", x.reciprocal(), 1.0 / h)
    C.check("x.sin()", x.sin(), np.sin(h), **TOL64)
    C.check("x.cos()", x.cos(), np.cos(h), **TOL64)
    C.check("x.exp()", x.exp(), np.exp(h), **TOL64)
    C.check("x.relu()", (x - 4.0).relu(), np.maximum(h - 4.0, 0))
    C.check("x.scale(3)", x.scale(3.0), h * 3.0)
    C.check("x.add_constant(2)", x.add_constant(2.0), h + 2.0)
    C.check("cnp.log(x, base=2)", cnp.log(x, base=2), np.log2(h), **TOL64)
    C.check("view.sqrt()", x[1:-1, ::2].sqrt(), np.sqrt(h[1:-1, ::2]), **TOL64)

C.finish()
