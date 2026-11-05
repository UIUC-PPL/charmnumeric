#!/usr/bin/env python
"""Size limits and device-side creation: the 2**31-1 per-dimension guard, meshgrid/full/
linspace built on the device, einsum 'ij,i->j' without an n^2 temporary, stepped views in
matvec/matmul, and (LIMITS_BIG=1) kernels on tiles of more than 2**31 elements.

Env: CNUM_HOST, CNUM_PORT, CNUM_ODF; LIMITS_TESTS=guard,creation,einsum,stepped,big selects
sections (big only with LIMITS_BIG=1; run it with CNUM_ODF=1 so a chare holds > 2**31
elements); LIMITS_SHAPE=R,C (default 65536,32769).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cupynumeric"))

import charmnumeric as cnp  # noqa: E402
import _cnum_connect  # noqa: E402
import _cnum_util as U  # noqa: E402


def exact(chk, label, got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    ok = (got.dtype == expected.dtype and got.shape == expected.shape
          and np.array_equal(got, expected, equal_nan=got.dtype.kind in "fc"))
    detail = "" if ok else f"(got {got.dtype}{got.shape}, expected {expected.dtype}{expected.shape})"
    if not ok and got.shape == expected.shape:
        detail += f" {int(np.sum(got != expected))} differ"
    return chk.true(label, ok, detail)


def raises(chk, label, fn, exc=ValueError, match=None):
    try:
        fn()
    except exc as e:
        return chk.true(label, match is None or match in str(e), f"message: {e}")
    except Exception as e:  # noqa: BLE001
        return chk.true(label, False, f"raised {type(e).__name__}: {e}")
    return chk.true(label, False, "no exception")


def test_guard(chk, iface):
    print("-- per-dimension guard")
    big = 2**31
    raises(chk, "zeros((2**31,))", lambda: cnp.zeros((big,), dtype=np.int8))
    raises(chk, "empty((3, 2**31))", lambda: cnp.empty((3, big), dtype=np.int8))
    raises(chk, "arange(2**31 + 5)", lambda: cnp.arange(big + 5))
    raises(chk, "linspace num=2**31", lambda: cnp.linspace(0, 1, big))
    raises(chk, "5-D with folded dim >= 2**31", lambda: cnp.zeros((2**16, 2**15, 2, 1, 3)))
    raises(chk, "full((2**31,), 1)", lambda: cnp.full((big,), 1, dtype=np.int8))
    a = cnp.zeros((4, 5))
    raises(chk, "reshape to a 2**31 dim", lambda: a.reshape((big, -1)))
    e = cnp.zeros((0, 3))
    raises(chk, "reshape (0, 3) -> (2**32, 0)", lambda: e.reshape((2**32, 0)), match=str(2**32))
    raises(chk, "reshape (0, 3) -> 4-D with a 2**31 dim",
           lambda: e.reshape((2, big, 0, 1)), match=str(big))
    print("-- slice steps over 2**31-1")
    h = np.arange(40.0).reshape(8, 5)
    c = cnp.array(h)
    for label, key in [("[3:4:2**40]", (slice(3, 4, 2**40),)),
                       ("[3:3:2**40]", (slice(3, 3, 2**40),)),
                       ("[1::2**33, ::2**40]", (slice(1, None, 2**33), slice(None, None, 2**40))),
                       ("[:, 2::2**31]", (slice(None), slice(2, None, 2**31)))]:
        exact(chk, f"a{label}", c[key].get(iface), h[key])
    exact(chk, "a[[0, 6], 1:2:2**40]", c[[0, 6], 1:2:2**40].get(iface), h[[0, 6], 1:2:2**40])
    exact(chk, "a[::2**62, [4, 0]]", c[::2**62, [4, 0]].get(iface), h[::2**62, [4, 0]])
    exact(chk, "a[2:][1::2**40][:, 1:]", c[2:][1::2**40][:, 1:].get(iface), h[2:][1::2**40][:, 1:])
    w, hw = cnp.array(h), h.copy()
    w[5::2**35, 1:4:2**32] = -1.0
    hw[5::2**35, 1:4:2**32] = -1.0
    exact(chk, "a[5::2**35, 1:4:2**32] = -1", w.get(iface), hw)
    exact(chk, "2**31-1 dim shape accepted lazily", np.array(cnp.empty((0, big - 1)).shape),
          np.array((0, big - 1)))


def test_creation(chk, iface):
    print("-- meshgrid / full / linspace / array on the device")
    hx, hy, hz = np.arange(5) * 1.5, np.array([3, 1, 4, 1, 5, 9]), np.linspace(-1, 1, 4)
    for indexing in ("xy", "ij"):
        for sparse in (False, True):
            ref = np.meshgrid(hx, hy, indexing=indexing, sparse=sparse)
            got = cnp.meshgrid(hx, hy, indexing=indexing, sparse=sparse)
            for k, (g, r) in enumerate(zip(got, ref)):
                exact(chk, f"meshgrid host {indexing} sparse={sparse} [{k}]", g.get(iface), r)
            ref = np.meshgrid(hx, hy, hz, indexing=indexing, sparse=sparse)
            got = cnp.meshgrid(cnp.array(hx), cnp.array(hy), hz, indexing=indexing, sparse=sparse)
            for k, (g, r) in enumerate(zip(got, ref)):
                exact(chk, f"meshgrid device 3-D {indexing} sparse={sparse} [{k}]", g.get(iface), r)
    xs = cnp.arange(20.0)
    ref = np.meshgrid(np.arange(20.0)[::3], hy)
    for k, (g, r) in enumerate(zip(cnp.meshgrid(xs[::3], hy), ref)):
        exact(chk, f"meshgrid stepped view [{k}]", g.get(iface), r)
    xg, yg = cnp.meshgrid(np.arange(3000.0), np.arange(2000, dtype=np.float32))
    rx, ry = np.meshgrid(np.arange(3000.0), np.arange(2000, dtype=np.float32))
    exact(chk, "meshgrid 2000x3000 [0]", xg.get(iface), rx)
    exact(chk, "meshgrid 2000x3000 [1]", yg.get(iface), ry)

    for shape, fill in [((4, 5), [1, 2, 3, 4, 5]), ((3, 4, 5), [[1], [2], [3], [4]]),
                        ((300, 400), np.arange(400, dtype=np.int16)),
                        ((6,), np.float32(2.5)), ((2, 3), [[7, 7, 7], [7, 7, 7]])]:
        exact(chk, f"full {shape} fill {np.shape(fill)}", cnp.full(shape, fill).get(iface),
              np.full(shape, fill))
    raises(chk, "full bad broadcast", lambda: cnp.full((3, 4), [1, 2, 3]))

    for args, kw in [((0.1, 0.7, 9), {}), ((np.float32(0.1), np.float32(0.7), 9), {}),
                     ((np.float32(-3), 5.0, 1000), {"endpoint": False}),
                     ((np.float32(1), np.float32(1), 5), {}),
                     ((np.float32(0), np.float32(100), 33), {"dtype": np.int32}),
                     ((2, 3, 7), {"axis": -1}), ((np.float32(2), 9, 1), {})]:
        exact(chk, f"linspace{args} {kw}", cnp.linspace(*args, **kw).get(iface),
              np.linspace(*args, **kw))
    got, step = cnp.linspace(np.float32(0), np.float32(1), 7, retstep=True)
    rg, rstep = np.linspace(np.float32(0), np.float32(1), 7, retstep=True)
    chk.true("linspace float32 retstep", step == rstep and np.asarray(step).dtype == rstep.dtype,
             f"{step!r} vs {rstep!r}")
    for start, stop, kw in [([0, 1], [10, -1], {}), ([0.0, 1.0, 2.0], 5.0, {"axis": 1}),
                            (np.zeros((2, 3)), np.ones((2, 3)), {"axis": -1, "num": 6}),
                            ([1, 2], [3, 4], {"endpoint": False, "num": 4}),
                            ([1, 2], [5, 8], {"dtype": np.int64, "num": 5}),
                            (np.float32([1, 2]), np.float32([5, 7]), {"num": 9}),
                            ([0, 3], [0, 4], {"num": 1})]:
        kw = dict(kw)
        num = kw.pop("num", 11)
        ref = np.linspace(start, stop, num, **kw)
        got = cnp.linspace(start, stop, num, **kw).get(iface)
        ok = got.dtype == ref.dtype and got.shape == ref.shape and np.allclose(got, ref)
        chk.true(f"linspace arrays {np.shape(start)} {np.shape(stop)} {kw}", ok,
                 f"{got.dtype}{got.shape} vs {ref.dtype}{ref.shape}")
    g, s = cnp.linspace([0, 1], [2, 5], 5, retstep=True)
    rg, rs = np.linspace([0, 1], [2, 5], 5, retstep=True)
    chk.true("linspace arrays retstep", np.allclose(np.asarray(s), rs) and np.allclose(g.get(iface), rg))
    dev = cnp.linspace(cnp.array([0.0, 2.0]), 10.0, 6).get(iface)
    chk.true("linspace device endpoints", np.allclose(dev, np.linspace([0.0, 2.0], 10.0, 6)))

    for data in ([1, 2, 3, 3], [[5, 5], [5, 5]], np.arange(1000).reshape(10, 100) % 7,
                 np.full(100000, -0.0), np.array([0.0] * 50 + [-0.0]),
                 np.broadcast_to(np.arange(3.0), (4, 3)), np.ones((50, 60))[::2, ::3]):
        exact(chk, f"array {np.shape(data)}", cnp.array(data).get(iface), np.array(data))
    e = cnp.array(np.array([-0.0] * 9))
    chk.true("array of -0.0 keeps the sign", bool(np.all(np.signbit(e.get(iface)))))


def test_einsum(chk, iface):
    print("-- einsum vector-matrix contractions")
    g = np.random.default_rng(3)
    for m, n in [(7, 5), (300, 200), (1, 9), (9, 1), (1500, 1300)]:
        A = g.standard_normal((m, n))
        v = g.standard_normal(m)
        w = g.standard_normal(n)
        cA, cv, cw = cnp.array(A), cnp.array(v), cnp.array(w)
        for sub, ops, hops in [("ij,i->j", (cA, cv), (A, v)), ("i,ij->j", (cv, cA), (v, A)),
                               ("ij,j->i", (cA, cw), (A, w)), ("j,ij->i", (cw, cA), (w, A)),
                               ("ji,j", (cA, cv), (A, v))]:
            got = cnp.einsum(sub, *ops).get(iface)
            ref = np.einsum(sub, *hops)
            chk.close(f"einsum {sub} {m}x{n}", got, ref, rtol=1e-10, atol=1e-9)
    A = g.standard_normal((2500, 1700))
    v = g.standard_normal(2500)
    cA, cv = cnp.array(A), cnp.array(v)
    for label, fn in [("v @ A", lambda a, x: x @ a),
                      ("v[4:4] @ A[4:4]", lambda a, x: x[4:4] @ a[4:4]),
                      ("v[3:1203] @ A[3:1203, 5:]", lambda a, x: x[3:1203] @ a[3:1203, 5:]),
                      ("v[::2] @ A[::2, 100:900]", lambda a, x: x[::2] @ a[::2, 100:900]),
                      ("v[:1700] @ A[:1700, ::3]", lambda a, x: x[:1700] @ a[:1700, ::3]),
                      ("v[1500:] @ A[700:1700]", lambda a, x: x[1500:] @ a[700:1700])]:
        chk.close(f"vecmat {label}", fn(cA, cv).get(iface), fn(A, v), rtol=1e-10, atol=1e-9)
    Ac = (g.standard_normal((300, 200)) + 1j * g.standard_normal((300, 200))).astype(np.complex64)
    vc = (g.standard_normal(300) + 1j * g.standard_normal(300)).astype(np.complex64)
    chk.close("vecmat complex64", (cnp.array(vc) @ cnp.array(Ac)).get(iface), vc @ Ac,
              rtol=1e-4, atol=1e-4)
    A3 = g.standard_normal((4, 5, 6))
    v4 = g.standard_normal(4)
    chk.close("einsum ijk,i->jk", cnp.einsum("ijk,i->jk", cnp.array(A3), cnp.array(v4)).get(iface),
              np.einsum("ijk,i->jk", A3, v4), rtol=1e-10, atol=1e-9)
    Ai = (np.arange(30).reshape(6, 5) % 7).astype(np.int64)
    vi = np.arange(6, dtype=np.int64)
    exact(chk, "einsum ij,i->j int64", cnp.einsum("ij,i->j", cnp.array(Ai), cnp.array(vi)).get(iface),
          np.einsum("ij,i->j", Ai, vi))
    for sub, shapes in [("ij,jk->ik", [(3, 0), (0, 4)]), ("ij,j->i", [(5, 0), (0,)]),
                        ("ij,jk->ik", [(0, 3), (3, 4)]), ("i,i", [(0,), (0,)]),
                        ("...i,...i->...", [(2, 0), (2, 0)]), ("ij,i->j", [(0, 4), (0,)])]:
        hops = [np.zeros(s) for s in shapes]
        got = cnp.einsum(sub, *[cnp.zeros(s) for s in shapes]).get(iface)
        exact(chk, f"einsum empty {sub} {shapes}", got, np.einsum(sub, *hops))


def test_stepped(chk, iface):
    print("-- stepped views in matvec / matmul")
    g = np.random.default_rng(5)
    A = g.standard_normal((900, 700))
    B = g.standard_normal((700, 500))
    x = g.standard_normal(700)
    cA, cB, cx = cnp.array(A), cnp.array(B), cnp.array(x)
    cases = [
        ("A[::2] @ x", lambda a, b, v: a[::2] @ v),
        ("A[:, ::2] @ x[::2]", lambda a, b, v: a[:, ::2] @ v[::2]),
        ("A[1::3, 5:605:2] @ x[100:400]", lambda a, b, v: a[1::3, 5:605:2] @ v[100:400]),
        ("A[::2, ::5] @ B[::5, ::3]", lambda a, b, v: a[::2, ::5] @ b[::5, ::3]),
        ("A[3::4] @ B", lambda a, b, v: a[3::4] @ b),
        ("A[:, :700:7].T @ A[:, 3:703:7]", lambda a, b, v: a[:, :700:7].T @ a[:, 3:703:7]),
        ("x[::7] @ A[::9, ::7].T", lambda a, b, v: v[::7] @ a[::9, ::7].T),
    ]
    for label, fn in cases:
        chk.close(label, fn(cA, cB, cx).get(iface), fn(A, B, x), rtol=1e-9, atol=1e-9)
    # row steps larger than a tile: some row chares hold no selected row
    T = g.standard_normal((3000, 300))
    y = g.standard_normal(300)
    cT, cy = cnp.array(T), cnp.array(y)
    for sl in (slice(None, None, 2100), slice(7, 2990, 1100), slice(1500, None, 701)):
        chk.close(f"T[{sl.start}:{sl.stop}:{sl.step}] @ y", (cT[sl] @ cy).get(iface), T[sl] @ y,
                  rtol=1e-9, atol=1e-9)


def test_big(chk, iface):
    if os.environ.get("LIMITS_BIG", "0") in ("", "0"):
        print("-- big: skipped (LIMITS_BIG=1 enables it)")
        return
    R, C = (int(v) for v in os.environ.get("LIMITS_SHAPE", "65536,32769").split(","))
    print(f"-- big: int8 ({R}, {C}) = {R * C} elements ({R * C / 2**31:.3f} * 2**31)")
    ri, ci = np.arange(R) % 61, np.arange(C) % 67

    def ref(rows, cols):
        return (ri[rows][:, None] + ci[cols][None, :]).astype(np.int8)

    r = (cnp.arange(R) % 61).astype(np.int8).reshape((R, 1))
    c = (cnp.arange(C) % 67).astype(np.int8).reshape((1, C))
    x = r + c
    probes = [(slice(0, 3), slice(0, 4)), (slice(R - 3, R), slice(C - 4, C)),
              (slice(R // 2, R // 2 + 2), slice(C - 2, C)), (slice(R - 1, R), slice(0, 3))]
    for rs, cs in probes:
        exact(chk, f"x[{rs.start}:{rs.stop}, {cs.start}:{cs.stop}]", x[rs, cs].get(iface),
              ref(np.arange(R)[rs], np.arange(C)[cs]))
    raises(chk, "x.reshape(-1) beyond 2**31-1 raises", lambda: x.reshape(-1), match=str(R * C))
    raises(chk, "x.ravel() beyond 2**31-1 raises", lambda: x.ravel(), match=str(R * C))

    t = x.T
    for rs, cs in probes:
        exact(chk, f"x.T[{cs.start}:{cs.stop}, {rs.start}:{rs.stop}]", t[cs, rs].get(iface),
              ref(np.arange(R)[rs], np.arange(C)[cs]).T)
    del t

    n = min(R, C)
    d = cnp.diag(x)
    exact(chk, "diag(x)[-5:]", d[n - 5:].get(iface), (ri[n - 5:n] + ci[n - 5:n]).astype(np.int8))
    exact(chk, "diag(x)[:5]", d[:5].get(iface), (ri[:5] + ci[:5]).astype(np.int8))
    k = 7
    dk = cnp.diag(x, k=k)
    nk = min(R, C - k)
    exact(chk, "diag(x, 7)[-4:]", dk[nk - 4:].get(iface),
          (ri[nk - 4:nk] + ci[nk - 4 + k:nk + k]).astype(np.int8))
    dm = cnp.diag(x, k=-(R - n + 3))
    km = R - n + 3
    nm = min(R - km, C)
    exact(chk, f"diag(x, -{km})[-4:]", dm[nm - 4:].get(iface),
          (ri[nm - 4 + km:nm + km] + ci[nm - 4:nm]).astype(np.int8))

    c16 = x.astype(np.int16)
    for rs, cs in probes:
        exact(chk, f"astype int16 [{rs.start}:{rs.stop}, {cs.start}:{cs.stop}]",
              c16[rs, cs].get(iface), ref(np.arange(R)[rs], np.arange(C)[cs]).astype(np.int16))
    del c16

    s = x[::2, ::3].copy()
    exact(chk, "x[::2, ::3] copy tail", s[-2:, -3:].get(iface),
          ref(np.arange(R)[::2][-2:], np.arange(C)[::3][-3:]))
    del s
    y = cnp.zeros_like(x)
    y[1:, :] = x[:-1, :]
    exact(chk, "shifted set_region tail", y[R - 2:, C - 3:].get(iface),
          ref(np.arange(R - 3, R - 1), np.arange(C - 3, C)))
    y[R - 2:, :] = 7
    exact(chk, "scalar fill tail", y[R - 3:, C - 2:].get(iface),
          np.vstack([ref(np.arange(R - 4, R - 3), np.arange(C - 2, C)), np.full((2, 2), 7, np.int8)]))
    del y
    cs_ = cnp.cumsum(x, axis=1, dtype=np.int8)
    exact(chk, "cumsum axis=1 int8 last rows", cs_[R - 2:, C - 3:].get(iface),
          np.cumsum(ref(np.arange(R - 2, R), np.arange(C)), axis=1, dtype=np.int8)[:, -3:])
    del cs_
    tot = int(cnp.sum(x, dtype=np.int64).get(iface))
    exp = int(C * ri.sum() + R * ci.sum())
    chk.true("sum(x) int64", tot == exp, f"{tot} vs {exp}")

    m = 46342  # m*m > 2**31
    v = (cnp.arange(m) % 113).astype(np.int8)
    D = cnp.diag(v)
    exact(chk, "diag(v) 46342^2 tail", D[m - 3:, m - 3:].get(iface),
          np.diag((np.arange(m - 3, m) % 113).astype(np.int8)))
    exact(chk, "diag(v) 46342^2 last row", D[m - 1, m - 6:].get(iface),
          np.array([0] * 5 + [(m - 1) % 113], dtype=np.int8))


SECTIONS = {"guard": test_guard, "creation": test_creation, "einsum": test_einsum,
            "stepped": test_stepped, "big": test_big}


def main(chk):
    iface = _cnum_connect.get_interface()
    names = os.environ.get("LIMITS_TESTS", "guard,creation,einsum,stepped,big").split(",")
    for name in names:
        SECTIONS[name](chk, iface)


if __name__ == "__main__":
    U.run("test_limits", main)
