"""All dtypes vs NumPy: result dtypes and values across ops, casts and promotion."""

import numpy as np

import charmnumeric as cnp
from _ops_common import Checker, connect

iface = connect()
C = Checker("test_dtypes", iface)

DTYPES = [np.dtype(t) for t in ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
                                "uint32", "uint64", "float16", "float32", "float64",
                                "complex64", "complex128")]
NUMERIC = [d for d in DTYPES if d.kind != "b"]
SHAPES = [(37,), (300001,), (37, 41), (600, 7), (5, 7, 9)]


def tol(dt):
    dt = np.dtype(dt)
    if dt.kind in "biu":
        return 0.0
    if dt in (np.float16,):
        return 2e-3
    if dt in (np.float32, np.complex64):
        return 2e-6
    return 1e-13


def host(shape, dt, lo=-6, hi=6, seed=0):
    """Small exact values of dtype dt (non-negative for unsigned/bool)."""
    rng = np.random.default_rng(seed)
    dt = np.dtype(dt)
    if dt.kind == "b":
        return rng.integers(0, 2, shape).astype(dt)
    if dt.kind == "u":
        lo = 0
    v = rng.integers(lo, hi + 1, shape)
    if dt.kind == "f":
        v = v + rng.integers(0, 4, shape) * 0.25
    if dt.kind == "c":
        return (v + 1j * rng.integers(lo, hi + 1, shape)).astype(dt)
    return v.astype(dt)


def chk(label, got, exp, dtype=None, rtol=None):
    """Dtype must equal NumPy's; values within the dtype's tolerance."""
    exp = np.asarray(exp)
    dt = exp.dtype if dtype is None else np.dtype(dtype)
    r = tol(dt) if rtol is None else rtol
    if dt.kind != "c":
        C.check(label, got, exp, rtol=r, atol=r, dtype=dt)
        return
    g = C.fetch(got) if hasattr(got, "get") else np.asarray(got)
    if g.dtype != dt:
        C.check(label, g, exp, dtype=dt)  # reports the dtype mismatch
        return
    real = np.dtype(np.float32 if dt == np.complex64 else np.float64)
    flat = np.ascontiguousarray(exp.astype(dt)).reshape(-1)
    C.check(label, g.reshape(-1).view(real), flat.view(real), rtol=r,
            atol=r, dtype=real)


def up(h):
    return cnp.array(h)


with np.errstate(all="ignore"):
    # ---- creation, upload/get, astype between all pairs ----
    for dt in DTYPES:
        n = dt.name
        chk(f"zeros {n}", cnp.zeros((5, 3), dtype=dt), np.zeros((5, 3), dt))
        chk(f"ones {n}", cnp.ones((7,), dtype=dt), np.ones((7,), dt))
        chk(f"full {n}", cnp.full((4, 4), 1, dtype=dt), np.full((4, 4), 1, dt))
        h = host((300001,), dt, seed=1)
        chk(f"upload/get {n}", up(h), h)
        if dt.kind != "b":
            chk(f"arange {n}", cnp.arange(9, dtype=dt), np.arange(9, dtype=dt))
        chk(f"eye {n}", cnp.eye(4, dtype=dt), np.eye(4, dtype=dt))
    chk("full complex scalar", cnp.full((3, 2), 1 + 2j), np.full((3, 2), 1 + 2j))
    for src in DTYPES:
        h = host((37, 41), src, lo=0, hi=100, seed=2)
        a = up(h)
        for dst in DTYPES:
            chk(f"astype {src.name}->{dst.name}", a.astype(dst), h.astype(dst))
    # signed sources into signed targets keep negative values
    for src in ("int8", "int64", "float32", "float64"):
        h = host((600, 7), src, lo=-100, hi=100, seed=3)
        a = up(h)
        for dst in ("int8", "int16", "int32", "int64", "float16", "complex64", "bool"):
            chk(f"astype {src}->{dst} (negatives)", a.astype(dst), h.astype(dst))
    hn = np.array([np.nan, np.inf, -0.0, 1.5, -2.5, 0.0], np.float64)
    chk("astype nan/inf -> bool", up(hn).astype(bool), hn.astype(bool))
    chk("astype f64 -> f16 (inf, rounding)", up(np.array([7e4, 1.0009765625, -3e-8, 65504.0]))
        .astype(np.float16), np.array([7e4, 1.0009765625, -3e-8, 65504.0]).astype(np.float16))
    chk("astype complex -> float (real part)", up(np.array([1 + 2j, -3 - 4j])).astype(np.float32),
        np.array([1 + 2j, -3 - 4j]).real.astype(np.float32))
    v = np.arange(12).reshape(3, 4).astype(np.float64)
    av = up(v)
    chk("astype of a view", av[1:, ::2].astype(np.int16), v[1:, ::2].astype(np.int16))

    # ---- arithmetic over all dtype pairs (fused mixed-dtype kernels) ----
    for da in DTYPES:
        ha = host((37, 41), da, seed=4)
        a = up(ha)
        for db in DTYPES:
            hb = host((37, 41), db, seed=5)
            b = up(hb)
            label = f"{da.name}+{db.name}"
            chk(label, a + b, ha + hb)
            if da.kind == "b" and db.kind == "b":
                C.raises(f"{da.name}-{db.name}", TypeError, lambda: a - b)
            else:
                chk(f"{da.name}-{db.name}", a - b, ha - hb)
            chk(f"{da.name}*{db.name}", a * b, ha * hb)
    # all ops, same-dtype operands (divisors nonzero; NumPy int /, //, % semantics)
    for dt in NUMERIC:
        n = dt.name
        ha, hb = host((600, 7), dt, seed=6), host((600, 7), dt, lo=1, hi=5, seed=7)
        if dt.kind == "u":
            hb = hb + dt.type(1)
        a, b = up(ha), up(hb)
        chk(f"/ {n}", a / b, ha / hb)
        chk(f"maximum {n}", cnp.maximum(a, b), np.maximum(ha, hb))
        chk(f"minimum {n}", cnp.minimum(a, b), np.minimum(ha, hb))
        chk(f"neg {n}", -a, -ha)
        chk(f"abs {n}", abs(a), np.abs(ha))
        chk(f"square {n}", cnp.square(a), np.square(ha))
        if dt.kind != "c":
            chk(f"// {n}", a // b, ha // hb)
            chk(f"% {n}", a % b, ha % hb)
            chk(f"fmax {n}", cnp.fmax(a, b), np.fmax(ha, hb))
            chk(f"sign {n}", cnp.sign(a), np.sign(ha))
            chk(f"clip {n}", cnp.clip(a, 1, 4), np.clip(ha, 1, 4))
        if dt.kind in "iu":
            hp = host((600, 7), dt, lo=0, hi=3, seed=8)
            chk(f"** {n}", a ** up(hp), ha ** hp)
            chk(f"& {n}", a & b, ha & hb)
            chk(f"| {n}", a | b, ha | hb)
            chk(f"^ {n}", a ^ b, ha ^ hb)
            chk(f"~ {n}", ~a, ~ha)
        else:
            chk(f"** {n}", a ** 2.0, ha ** 2.0)
    # unsigned wraparound and division by zero (NumPy: 0)
    hu = np.array([0, 1, 250, 255], np.uint8)
    u = up(hu)
    chk("uint8 - 5 wraps", u - 5, hu - np.uint8(5))
    chk("uint8 * 3 wraps", u * 3, hu * np.uint8(3))
    chk("uint8 // 0", u // up(np.zeros(4, np.uint8)), np.zeros(4, np.uint8))
    hu64 = np.array([2**63 + 5, 3, 2**64 - 1], np.uint64)
    u64 = up(hu64)
    chk("uint64 > 2**63 compare", u64 > up(np.array([2**63, 4, 2**64 - 2], np.uint64)),
        hu64 > np.array([2**63, 4, 2**64 - 2], np.uint64))
    chk("uint64 // 2", u64 // 2, hu64 // np.uint64(2))
    chk("uint64 max", cnp.maximum(u64, up(np.array([1, 2**63, 1], np.uint64))),
        np.maximum(hu64, np.array([1, 2**63, 1], np.uint64)))
    chk("uint64 -> float64", u64.astype(np.float64), hu64.astype(np.float64))
    chk("float64 -> uint64 (>= 2**63)", up(np.array([2.0**63 * 1.5, 7.0])).astype(np.uint64),
        np.array([2.0**63 * 1.5, 7.0]).astype(np.uint64))

    # ---- Python scalars are weak (NEP 50) ----
    h8 = host((37,), np.int8, seed=9)
    i8 = up(h8)
    chk("int8 + 1", i8 + 1, h8 + 1)
    chk("int8 + 2.5 -> float64", i8 + 2.5, h8 + 2.5)
    chk("int8 * 1j -> complex128", i8 * 1j, h8 * 1j)
    chk("uint8 + 255", up(hu) + 255, hu + 255)
    C.raises("uint8 + 300 overflows", OverflowError, lambda: up(hu) + 300)
    h32 = host((37,), np.float32, seed=10)
    f32 = up(h32)
    chk("float32 + 0.1 stays float32", f32 + 0.1, h32 + 0.1)
    chk("float32 * (1+2j) -> complex64", f32 * (1 + 2j), h32 * (1 + 2j))
    h16 = host((37,), np.float16, seed=11)
    chk("float16 * 3.3", up(h16) * 3.3, h16 * 3.3)
    hb = np.array([2**60 + 1, -(2**61) - 3], np.int64)
    chk("int64 + 2**60 exact", up(hb) + 2**60, hb + 2**60)
    chk("bool + 1 -> int64", up(np.array([True, False])) + 1, np.array([True, False]) + 1)
    chk("int8 < 1000 (out of range)", i8 < 1000, h8 < 1000)
    chk("uint8 >= -1 (out of range)", up(hu) >= -1, hu >= -1)

    # ---- unary math (int -> float as NumPy: sqrt(int16) is float32, ...) ----
    for dt in DTYPES:
        n = dt.name
        h = host((5, 7, 9), dt, lo=0, hi=9, seed=12)
        a = up(h)
        for name in ("sqrt", "exp", "log1p", "sin", "cos", "tanh"):
            chk(f"{name} {n}", getattr(cnp, name)(a), getattr(np, name)(h),
                rtol=max(tol(getattr(np, name)(h).dtype), 1e-12))
        chk(f"isnan {n}", cnp.isnan(a), np.isnan(h))
        chk(f"isfinite {n}", cnp.isfinite(a), np.isfinite(h))
        if dt.kind != "c":
            chk(f"floor {n}", cnp.floor(a), np.floor(h))
        else:
            chk(f"log {n}", cnp.log(a + 1), np.log(h + 1), rtol=1e-5)
    hz = np.array([1 + 2j, -3.5 + 0.5j, np.nan + 1j, 1 + np.inf * 1j, 0j], np.complex128)
    z = up(hz)
    chk("complex abs -> float64", abs(z), np.abs(hz))
    chk("complex real", z.real, hz.real)
    chk("complex imag", cnp.imag(z), hz.imag)
    chk("complex conj", z.conj(), hz.conj())
    chk("complex isnan", cnp.isnan(z), np.isnan(hz))
    chk("complex isinf", cnp.isinf(z), np.isinf(hz))
    chk("complex sign", cnp.sign(z[0:2]), np.sign(hz[0:2]))
    chk("complex / complex", z[0:2] / up(np.array([2 - 1j, 0.5j])),
        hz[0:2] / np.array([2 - 1j, 0.5j]))
    chk("complex ** 2", z[0:2] ** 2, hz[0:2] ** 2)
    chk("complex exp", cnp.exp(z[0:2]), np.exp(hz[0:2]))
    chk("float real/imag", f32.real, h32.real)
    chk("float imag -> zeros", f32.imag, h32.imag)

    # ---- comparisons return bool; complex order is lexicographic ----
    for dt in DTYPES:
        ha, hb = host((300001,), dt, seed=13), host((300001,), dt, seed=14)
        a, b = up(ha), up(hb)
        for name in ("less", "less_equal", "greater", "equal", "not_equal"):
            chk(f"{name} {dt.name}", getattr(cnp, name)(a, b), getattr(np, name)(ha, hb))
        chk(f"a < 2 {dt.name}", a < 2, ha < 2)
    hc1 = np.array([1 + 1j, 1 + 2j, 2 + 0j, np.nan + 0j, 1 + 1j])
    hc2 = np.array([1 + 2j, 1 + 1j, 1 + 5j, 1 + 0j, 1 + 1j])
    for name in ("less", "less_equal", "greater_equal", "maximum", "minimum", "fmax"):
        chk(f"complex {name}", getattr(cnp, name)(up(hc1), up(hc2)),
            getattr(np, name)(hc1, hc2))
    chk("int8 vs uint8 compare", i8[:4] < up(hu), h8[:4] < hu)
    chk("int32 vs float32 compare", up(host((37,), np.int32)) == f32,
        host((37,), np.int32) == h32)

    # ---- where / logical / bitwise ----
    for dt in DTYPES:
        hx, hy = host((37, 41), dt, seed=15), host((37, 41), dt, seed=16)
        x, y = up(hx), up(hy)
        mask = x != y
        chk(f"where(bool, {dt.name}, {dt.name})", cnp.where(mask, x, y), np.where(hx != hy, hx, hy))
        chk(f"where({dt.name} cond)", cnp.where(x, 1.0, 0.0), np.where(hx, 1.0, 0.0))
        chk(f"logical_and {dt.name}", cnp.logical_and(x, y), np.logical_and(hx, hy))
        chk(f"logical_not {dt.name}", cnp.logical_not(x), np.logical_not(hx))
        chk(f"logical_xor {dt.name}", cnp.logical_xor(x, y), np.logical_xor(hx, hy))
    hm1, hm2 = host((600, 7), bool, seed=17), host((600, 7), bool, seed=18)
    m1, m2 = up(hm1), up(hm2)
    chk("bool & | ^ ~", (m1 & m2) | (~m1 ^ m2), (hm1 & hm2) | (~hm1 ^ hm2))
    chk("bool + bool (or)", m1 + m2, hm1 + hm2)
    chk("bool * float", m1 * 2.5, hm1 * 2.5)
    chk("where(int8, i8, f32) promotes", cnp.where(i8, i8, f32), np.where(h8, h8, h32))
    C.raises("float & float", TypeError, lambda: f32 & f32)
    hf = host((600, 7), np.float64, seed=19)
    f = up(hf)
    chk("fused cmp -> where -> sum", cnp.sum(cnp.where(f > 0, f, 0.0)),
        np.asarray(np.where(hf > 0, hf, 0.0).sum()), rtol=1e-12)
    t = f.copy()
    t[...] = cnp.where(f < 0, 0.0, f)
    chk("mask assignment", t, np.where(hf < 0, 0.0, hf))
    hu16 = host((600, 7), np.uint16, seed=30)
    key = (slice(1, None), slice(None, None, 2))
    chk("mixed dtypes on views", f[key] + m1[key] * up(hu16)[key], hf[key] + hm1[key] * hu16[key])

    # ---- reductions: NumPy's result dtypes ----
    for dt in DTYPES:
        n = dt.name
        for shape in ((300001,), (37, 41), (5, 7, 9)):
            h = host(shape, dt, lo=-3, hi=3, seed=20)
            a = up(h)
            s = f" {n} {shape}"
            chk("sum" + s, cnp.sum(a), np.asarray(np.sum(h)))
            chk("max" + s, cnp.max(a), np.asarray(np.max(h)))
            chk("argmin" + s, cnp.argmin(a), np.asarray(np.argmin(h)))
            chk("any" + s, cnp.any(a), np.asarray(np.any(h)))
            chk("all" + s, cnp.all(a), np.asarray(np.all(h)))
            chk("count_nonzero" + s, cnp.count_nonzero(a), np.asarray(np.count_nonzero(h)))
            if dt.kind != "b":
                chk("mean" + s, cnp.mean(a), np.asarray(np.mean(h)), rtol=max(tol(dt), 1e-9))
            if len(shape) > 1:
                chk("sum axis=0" + s, cnp.sum(a, axis=0), np.sum(h, axis=0))
                chk("argmax axis=1" + s, cnp.argmax(a, axis=1), np.argmax(h, axis=1))
        h = host((37, 41), dt, lo=-2, hi=2, seed=21)
        chk(f"prod {n}", cnp.prod(up(h[:, :5]), axis=1), np.prod(h[:, :5], axis=1))
        if dt.kind in "fc":
            chk(f"var {n}", cnp.var(up(h)), np.asarray(np.var(h)), rtol=max(tol(dt), 1e-9) * 10)
            chk(f"std axis {n}", cnp.std(up(h), axis=0, ddof=1), np.std(h, axis=0, ddof=1),
                rtol=max(tol(dt), 1e-9) * 10)
    chk("norm complex", cnp.linalg.norm(z[0:2]), np.asarray(np.linalg.norm(hz[0:2])))
    chk("sum(dtype=float32)", cnp.sum(i8, dtype=np.float32),
        np.asarray(np.sum(h8, dtype=np.float32)))
    chk("bincount uint8", cnp.bincount(up(hu[:2]), minlength=3), np.bincount(hu[:2], minlength=3))

    # ---- scans ----
    for dt in DTYPES:
        h = host((600, 7), dt, lo=-3, hi=3, seed=22)
        a = up(h)
        chk(f"cumsum {dt.name}", cnp.cumsum(a, axis=0), np.cumsum(h, axis=0),
            rtol=max(tol(dt), 1e-12) * 10)
        if dt.kind != "b":
            chk(f"cumprod {dt.name}", cnp.cumprod(a[:, :3].copy(), axis=1),
                np.cumprod(h[:, :3], axis=1))
    hcn = np.array([1 + 1j, np.nan + 0j, 2 - 1j, 3 + 0j], np.complex64)
    chk("nancumsum complex64", cnp.nancumsum(up(hcn)), np.nancumsum(hcn))

    # ---- sort / argsort ----
    for dt in DTYPES:
        for shape, axis in (((300001,), 0), ((37, 41), 1), ((5, 7, 9), 0)):
            h = host(shape, dt, lo=-50, hi=50, seed=23)
            a = up(h)
            chk(f"sort {dt.name} {shape}", cnp.sort(a, axis=axis), np.sort(h, axis=axis, kind="stable"))
            chk(f"argsort {dt.name} {shape}", cnp.argsort(a, axis=axis),
                np.argsort(h, axis=axis, kind="stable"))
    hcs = np.array([2 + 1j, 1 + np.nan * 1j, np.nan + 1j, 1 + 3j, 1 - 1j, np.nan + np.nan * 1j,
                    1 + 0j, -1 + 5j], np.complex128)
    chk("sort complex with NaNs", cnp.sort(up(hcs)), np.sort(hcs))
    chk("argsort complex with NaNs", cnp.argsort(up(hcs)), np.argsort(hcs, kind="stable"))
    hfs = np.array([3, np.nan, -1, np.inf, -0.0, 2], np.float16)
    chk("sort float16 with NaN", cnp.sort(up(hfs)), np.sort(hfs))

    # ---- matmul ----
    for dt in DTYPES:
        ha, hb = host((40, 30), dt, lo=-3, hi=3, seed=24), host((30, 20), dt, lo=-3, hi=3, seed=25)
        hv = host((30,), dt, lo=-3, hi=3, seed=26)
        a, b = up(ha), up(hb)
        r = 1e-2 if dt == np.float16 else max(tol(dt), 1e-12)
        chk(f"matmul {dt.name}", a @ b, ha @ hb, rtol=r)
        chk(f"matvec {dt.name}", a @ up(hv), ha @ hv, rtol=r)
        chk(f"dot {dt.name}", cnp.dot(up(hv), up(hv)), np.asarray(np.dot(hv, hv)), rtol=r)
    chk("matmul complex64 @ float32", up(host((8, 6), np.complex64)) @ up(host((6, 5), np.float32)),
        host((8, 6), np.complex64) @ host((6, 5), np.float32))

    # ---- random integers ----
    for dt, lo, hi in ((np.uint8, 0, 256), (np.uint32, 10, 2**32), (np.int8, -128, 128),
                       (np.uint64, 0, 2**64), (np.int16, -5, 5), (bool, 0, 2)):
        r = cnp.random.randint(lo, hi, size=(1000,), dtype=dt)
        rv = C.fetch(r)
        ok = rv.dtype == np.dtype(dt) and rv.min() >= lo and int(rv.max()) < hi and len(np.unique(rv)) > 1
        C.check(f"randint {np.dtype(dt).name} [{lo}, {hi})", np.array([ok]), np.array([True]))

    # ---- assignment converts to the target dtype ----
    t = cnp.zeros((6,), dtype=np.uint16)
    t[1:3] = 70000.0 - 69990.0
    t[3] = True
    t[4:] = up(np.array([2.9, 300.0]))
    chk("assign into uint16", t, np.array([0, 10, 10, 1, 2, 300], np.uint16))
    tc = cnp.zeros((4,), dtype=np.complex64)
    tc[0] = 1 + 2j
    tc[1:3] = 3.5
    tc[3] = up(np.array([2], np.int8))[0]
    chk("assign into complex64", tc, np.array([1 + 2j, 3.5, 3.5, 2], np.complex64))
    C.raises("complex into float", TypeError, lambda: f32.__setitem__(0, 1j))
    C.raises("300 into uint8", OverflowError, lambda: up(hu).__setitem__(0, 300))

    # ---- complex linalg ----
    rng = np.random.default_rng(27)
    for dt in (np.complex64, np.complex128):
        n = np.dtype(dt).name
        r = 1e-3 if dt == np.complex64 else 1e-10
        ha = (rng.random((12, 9)) + 1j * rng.random((12, 9))).astype(dt)
        a = up(ha)
        u, s, vh = cnp.linalg.svd(a, full_matrices=False)
        uh, sh, vhh = C.fetch(u), C.fetch(s), C.fetch(vh)
        chk(f"svd s {n}", s, np.linalg.svd(ha, compute_uv=False), rtol=r)
        chk(f"svd u s vh {n}", up(uh * sh) @ vh, ha, rtol=r)
        chk(f"svd u unitary {n}", up(uh.conj().T.copy()) @ u, np.eye(9, dtype=dt), rtol=r)
        hw = (rng.random((9, 12)) + 1j * rng.random((9, 12))).astype(dt)
        chk(f"svdvals wide {n}", cnp.linalg.svd(up(hw), compute_uv=False),
            np.linalg.svd(hw, compute_uv=False), rtol=r)
        hsq = (rng.random((10, 10)) + 1j * rng.random((10, 10)) + 4 * np.eye(10)).astype(dt)
        hrhs = (rng.random((10, 3)) + 1j * rng.random((10, 3))).astype(dt)
        chk(f"solve {n}", cnp.linalg.solve(up(hsq), up(hrhs)), np.linalg.solve(hsq, hrhs), rtol=r)
        chk(f"inv {n}", cnp.linalg.inv(up(hsq)), np.linalg.inv(hsq), rtol=r)
        chk(f"det {n}", cnp.linalg.det(up(hsq)), np.asarray(np.linalg.det(hsq)), rtol=r)
        hh = hsq @ hsq.conj().T
        chk(f"cholesky {n}", cnp.linalg.cholesky(up(hh)), np.linalg.cholesky(hh), rtol=r * 10)
        chk(f"eigvalsh {n}", cnp.linalg.eigvalsh(up(hh)), np.linalg.eigvalsh(hh), rtol=r * 10)
        q, rr = cnp.linalg.qr(a)
        chk(f"qr q @ r {n}", q @ rr, ha, rtol=r)
    C.raises("linalg float16", TypeError, lambda: cnp.linalg.inv(up(np.eye(3, dtype=np.float16))))

C.finish()
