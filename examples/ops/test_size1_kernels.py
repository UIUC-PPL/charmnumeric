"""Fused kernels with all-size-1 operands (scalar kernel args), per op family and dtype."""

import operator

import numpy as np

import charmnumeric as cnp
from _ops_common import Checker, connect

iface = connect()
C = Checker("test_size1_kernels", iface)

DTYPES = [np.dtype(t) for t in ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
                                "uint32", "uint64", "float16", "float32", "float64",
                                "complex64", "complex128")]
BINARY = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv,
    "floordiv": operator.floordiv, "mod": operator.mod, "pow": operator.pow,
    "lt": operator.lt, "eq": operator.eq,
    "maximum": (cnp.maximum, np.maximum), "minimum": (cnp.minimum, np.minimum),
}
UNARY = {
    "neg": (operator.neg, operator.neg), "abs": (abs, abs), "exp": (cnp.exp, np.exp),
    "sqrt": (cnp.sqrt, np.sqrt), "tanh": (cnp.tanh, np.tanh), "floor": (cnp.floor, np.floor),
    "sign": (cnp.sign, np.sign), "square": (cnp.square, np.square),
}


def tol(dt):
    return {"f2": 2e-3, "f4": 2e-6, "c8": 2e-6}.get(np.dtype(dt).str[1:], 1e-12)


def value(dt, v):
    dt = np.dtype(dt)
    if dt.kind == "b":
        return np.array(v != 0)
    if dt.kind == "u":
        v = abs(v)
    if dt.kind == "c":
        return np.array(complex(v, 0.5), dtype=dt)
    return np.array(v, dtype=dt)


def run(label, cfn, nfn, *hosts):
    with np.errstate(all="ignore"):
        try:
            want = np.asarray(nfn(*hosts))
        except Exception:  # noqa: BLE001  (NumPy rejects it: nothing to compare)
            return
    try:
        got = cfn(*[cnp.asarray(h) if isinstance(h, np.ndarray) else h for h in hosts])
    except (NotImplementedError, TypeError) as e:
        if "dtype" in str(e) or isinstance(e, NotImplementedError):
            return
        raise
    C.check(label, got, want, rtol=tol(want.dtype), atol=tol(want.dtype), dtype=want.dtype)


def fns(f):
    return f if isinstance(f, tuple) else (f, f)


for dt in DTYPES:
    a = value(dt, 7)
    b = value(dt, -3) if dt.kind not in "u" else value(dt, 2)
    for shape in ((1,), (1, 1, 1)):
        ha, hb = np.full(shape, a, dtype=dt), np.full(shape, b, dtype=dt)
        for name, f in BINARY.items():
            cf, nf = fns(f)
            run(f"{dt} {name} {shape}", cf, nf, ha, hb)
            run(f"{dt} {name} rev-scalar {shape}", lambda x, cf=cf: cf(3, x),
                lambda x, nf=nf: nf(3, x), ha)
        for name, f in UNARY.items():
            cf, nf = fns(f)
            run(f"{dt} {name} {shape}", cf, nf, ha)
        run(f"{dt} where {shape}", lambda x, y: cnp.where(x == y, x, y),
            lambda x, y: np.where(x == y, x, y), ha, hb)
        # a chain over size-1 operands only
        run(f"{dt} chain {shape}", lambda x, y: (x * y + x) % 5 - y,
            lambda x, y: (x * y + x) % 5 - y, ha, hb)

    # size-1 operand next to a full array, and a scalar setitem right-hand side
    full = np.arange(1, 1001).astype(dt) if dt.kind != "b" else (np.arange(1000) % 3 == 0)
    one = np.full((1,), b, dtype=dt)
    for name in ("add", "mod", "maximum", "lt"):
        cf, nf = fns(BINARY[name])
        run(f"{dt} {name} full*size1", cf, nf, full, one)
    with np.errstate(all="ignore"):
        try:
            want = full.copy()
            want[3:4] = one % one + one
        except Exception:  # noqa: BLE001
            want = None
    if want is not None:
        x = cnp.asarray(full)
        o = cnp.asarray(one)
        x[3:4] = o % o + o
        C.check(f"{dt} setitem size1 rhs", x, want, rtol=tol(dt), atol=tol(dt), dtype=dt)

C.finish()
