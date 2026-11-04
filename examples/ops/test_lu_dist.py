#!/usr/bin/env python
"""solve/inv/det/slogdet against NumPy, for the distributed LU path.

Matrices of at least CNUM_LA_LU_MIN rows (server env, default 2048) use the distributed
LU; run the server with CNUM_LA_LU_MIN=1 (and a small CNUM_LA_LU_NB) to send every case
here through it. Env: CNUM_HOST, CNUM_PORT, CNUM_ODF; LU_BIG=0 skips the n=2100 cases.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
from _ops_common import Checker, connect  # noqa: E402

IFACE = None


def get(x):
    return np.asarray(x.get(IFACE))


def tol(dt):
    return 2e-3 if np.dtype(dt) in (np.float32, np.complex64) else 1e-8


def close(chk, label, got, exp, t):
    chk.count += 1
    got, exp = np.asarray(got), np.asarray(exp)
    if got.shape != exp.shape:
        msg = f"shape {got.shape} != {exp.shape}"
    else:
        e = np.abs(got.astype(np.complex128) - exp).max() / max(1.0, np.abs(exp).max())
        msg = None if e <= t else f"rel err {e:.3e} > {t:.1e}"
    print(f"  PASS {label}" if msg is None else f"  FAIL {label}: {msg}")
    if msg:
        chk.failures.append(label)


def ok(chk, label, cond):
    chk.count += 1
    print(f"  {'PASS' if cond else 'FAIL'} {label}")
    if not cond:
        chk.failures.append(label)


def mat(n, dt, shift):
    if np.dtype(dt).kind == "c":
        r = np.dtype(dt).type(0).real.dtype
        a = cnp.random.rand(n, n, dtype=r) + 1j * cnp.random.rand(n, n, dtype=r)
        a = a.astype(dt)
    else:
        a = cnp.random.rand(n, n, dtype=dt)
    return a + shift * cnp.eye(n, dtype=dt) if shift else a


def check(chk, n, dt, shift=None):
    shift = float(n) / 4 if shift is None else shift
    a = mat(n, dt, shift)
    rdt = np.dtype(dt).type(0).real.dtype
    b1 = cnp.random.rand(n, dtype=rdt).astype(dt)
    b2 = cnp.random.rand(n, 3, dtype=rdt).astype(dt)
    ah, b1h, b2h = (get(x).astype(np.complex128) for x in (a, b1, b2))
    t = tol(dt) * max(1, n // 100)
    x1, x2 = cnp.linalg.solve(a, b1), cnp.linalg.solve(a, b2)
    ok(chk, f"solve dtype n={n} {np.dtype(dt)}", x1.dtype == np.dtype(dt) and x2.shape == (n, 3))
    close(chk, f"solve vec n={n} {np.dtype(dt)}", get(x1), np.linalg.solve(ah, b1h), t)
    close(chk, f"solve mat n={n} {np.dtype(dt)}", get(x2), np.linalg.solve(ah, b2h), t)
    close(chk, f"inv n={n} {np.dtype(dt)}", get(cnp.linalg.inv(a)), np.linalg.inv(ah), t)
    if n <= 40:  # larger determinants overflow
        close(chk, f"det n={n} {np.dtype(dt)}", get(cnp.linalg.det(a)) / np.linalg.det(ah), 1.0, t)
    if np.dtype(dt).kind != "c":
        s, ld = cnp.linalg.slogdet(a)
        sn, ldn = np.linalg.slogdet(ah.real)
        close(chk, f"slogdet n={n} {np.dtype(dt)}", [get(s), get(ld) / max(1.0, abs(ldn))],
              [sn, ldn / max(1.0, abs(ldn))], t)


def main():
    global IFACE
    IFACE = connect()
    chk = Checker("test_lu_dist", IFACE)
    for n in (1, 2, 7, 33, 100, 300):
        check(chk, n, np.float64)
    for n in (5, 64, 150):
        check(chk, n, np.float32)
    for n in (9, 70):
        check(chk, n, np.complex128)
        check(chk, n, np.complex64)
    # pivoting matters: no diagonal shift
    check(chk, 120, np.float64, shift=0.0)
    # strided views and an int64 matrix
    a = mat(80, np.float64, 20.0)
    b = cnp.random.rand(100)
    ah, bh = get(a), get(b)
    x = cnp.linalg.solve(a[0:80:2, 1:80:2], b[10:90:2])
    close(chk, "solve strided views", get(x), np.linalg.solve(ah[0:80:2, 1:80:2], bh[10:90:2]), 1e-8)
    ai = cnp.random.randint(-5, 6, size=(40, 40)) + 30 * cnp.eye(40, dtype=np.int64)
    aih = get(ai).astype(np.float64)
    ok(chk, "det int64 dtype", cnp.linalg.det(ai).dtype == np.float64)
    close(chk, "det int64", get(cnp.linalg.det(ai)) / np.linalg.det(aih), 1.0, 1e-9)
    close(chk, "solve int64", get(cnp.linalg.solve(ai, b[0:40])), np.linalg.solve(aih, bh[0:40]), 1e-9)
    # exactly singular: a zero row, a zero column
    for name, key in (("zero row", (7, slice(None))), ("zero column", (slice(None), 7))):
        s = mat(50, np.float64, 5.0)
        s[key] = 0.0
        x = get(cnp.linalg.solve(s, cnp.ones(50)))
        ok(chk, f"singular solve ({name}) not finite", not np.isfinite(x).all())
        ok(chk, f"singular det ({name}) == 0", get(cnp.linalg.det(s)) == 0.0)
        sg, ld = cnp.linalg.slogdet(s)
        ok(chk, f"singular slogdet ({name})", get(sg) == 0.0 and get(ld) == -np.inf)
    if os.environ.get("LU_BIG", "1") != "0":
        n = 2100
        a = mat(n, np.float64, 0.0)
        b = cnp.random.rand(n, 2)
        x = cnp.linalg.solve(a, b)
        res = cnp.linalg.norm(b - a @ x, np.inf)
        ok(chk, f"solve residual n={n}", get(res) < 1e-8)
        ah, bh = get(a), get(b)
        close(chk, f"solve n={n}", get(x), np.linalg.solve(ah, bh), 1e-6)
        sg, ld = cnp.linalg.slogdet(a)
        sn, ldn = np.linalg.slogdet(ah)
        close(chk, f"slogdet n={n}", [get(sg), get(ld) / abs(ldn)], [sn, 1.0], 1e-10)
    chk.finish()


if __name__ == "__main__":
    main()
