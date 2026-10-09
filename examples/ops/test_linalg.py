#!/usr/bin/env python
"""Tests for cnp.linalg factorizations and solves (LINALG executor).

cholesky (lower/upper), qr (reduced/complete/r), solve (vector and matrix
rhs), inv, det, slogdet, eigh/eigvalsh (UPLO L/U), svd (full/reduced,
compute_uv), svdvals, lstsq, solve_triangular (trans/lower/unit_diagonal) and
tssvd, against NumPy on the fetched inputs. Shapes include non-tile-aligned
sizes and sizes above the CPU 2-D minimum tile (512), so that +p2 runs gather
from several chares; operands include offset and strided views and int64
inputs; float32 and float64. Factors that are unique only up to signs
(Q, eigenvectors, singular vectors) are checked through their invariants.

Environment: CNUM_HOST, CNUM_PORT, CNUM_ODF (default 4).
Prints PASS test_linalg / FAIL test_linalg: ... (exit status 0 / 1).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
from _ops_common import Checker, connect  # noqa: E402

IFACE = None


def tol(dt):
    return 2e-4 if np.dtype(dt) == np.float32 else 1e-9


def rel_err(got, exp):
    got = np.asarray(got, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    return np.abs(got - exp).max() / max(1.0, np.abs(exp).max())


def close(chk, label, got, exp, t):
    chk.count += 1
    got = np.asarray(got)
    exp = np.asarray(exp)
    if got.shape != exp.shape:
        msg = f"shape {got.shape} != {exp.shape}"
    elif exp.size == 0:
        msg = None
    else:
        e = rel_err(got, exp)
        msg = None if e <= t else f"rel err {e:.3e} > {t:.1e}"
        if msg and got.size <= 4:
            msg += f" (got {got.ravel()}, expected {exp.ravel()})"
    if msg is None:
        print(f"  PASS {label}")
    else:
        print(f"  FAIL {label}: {msg}")
        chk.failures.append(label)


def dtype_ok(chk, label, x, dt):
    chk.count += 1
    if x.dtype == np.dtype(dt):
        print(f"  PASS {label} dtype")
    else:
        print(f"  FAIL {label} dtype: {x.dtype} != {np.dtype(dt)}")
        chk.failures.append(label + " dtype")


def get(x):
    return np.asarray(x.get(IFACE))


def rand(m, n, dt, shift=0.0):
    """Native random matrix; shift adds shift*I to make it well conditioned."""
    a = cnp.random.rand(m, n, dtype=dt)
    if shift:
        a = a + shift * cnp.eye(m, n, dtype=dt)
    return a


def spd(n, dt):
    r = cnp.random.rand(n, n, dtype=dt)
    return r @ cnp.random.transposed(r) + float(n) * cnp.eye(n, dtype=dt)


def check_cholesky(chk, n, dt):
    a = spd(n, dt)
    ah = get(a)
    L = cnp.linalg.cholesky(a)
    U = cnp.linalg.cholesky(a, upper=True)
    dtype_ok(chk, f"cholesky n={n} {dt}", L, dt)
    close(chk, f"cholesky n={n} {dt}", get(L), np.linalg.cholesky(ah.astype(np.float64)), tol(dt))
    close(chk, f"cholesky upper n={n} {dt}", get(U),
          np.linalg.cholesky(ah.astype(np.float64), upper=True), tol(dt))


def check_qr(chk, m, n, dt):
    a = rand(m, n, dt)
    ah = get(a).astype(np.float64)
    k = min(m, n)
    t = tol(dt) * 10
    for mode in ("reduced", "complete"):
        q, r = cnp.linalg.qr(a, mode=mode)
        qh, rh = get(q), get(r)
        qn, rn = np.linalg.qr(ah, mode=mode)
        lbl = f"qr {mode} {m}x{n} {dt}"
        if qh.shape != qn.shape or rh.shape != rn.shape:
            close(chk, lbl + " shapes", np.zeros(qh.shape + rh.shape), np.zeros(qn.shape + rn.shape), 0)
            continue
        close(chk, lbl + " Q@R", qh @ rh, ah, t)
        close(chk, lbl + " Q^TQ", qh.T @ qh, np.eye(qh.shape[1]), t)
        d = np.sign(np.diag(rh)[:k]) * np.sign(np.diag(rn)[:k])
        dd = np.ones(rh.shape[0])
        dd[:k] = d
        close(chk, lbl + " R (up to row signs)", rh, dd[:, None] * rn, t)
    r_only = cnp.linalg.qr(a, mode="r")
    close(chk, f"qr r {m}x{n} {dt}", np.abs(get(r_only)), np.abs(np.linalg.qr(ah, mode="r")), t)


def check_solve(chk, n, dt):
    a = rand(n, n, dt, shift=float(n) / 4)
    b1 = cnp.random.rand(n, dtype=dt)
    b2 = cnp.random.rand(n, 3, dtype=dt)
    ah, b1h, b2h = (get(x).astype(np.float64) for x in (a, b1, b2))
    x1 = cnp.linalg.solve(a, b1)
    x2 = cnp.linalg.solve(a, b2)
    dtype_ok(chk, f"solve n={n} {dt}", x1, dt)
    close(chk, f"solve vec n={n} {dt}", get(x1), np.linalg.solve(ah, b1h), tol(dt) * 10)
    close(chk, f"solve mat n={n} {dt}", get(x2), np.linalg.solve(ah, b2h), tol(dt) * 10)
    close(chk, f"inv n={n} {dt}", get(cnp.linalg.inv(a)), np.linalg.inv(ah), tol(dt) * 10)
    s, ld = cnp.linalg.slogdet(a)
    sn, ldn = np.linalg.slogdet(ah)
    close(chk, f"slogdet n={n} {dt}", np.concatenate([get(s), get(ld)]), [sn, ldn], tol(dt) * 10)
    # residual, natively: ||b - a x||_inf (the solve.py check)
    res = cnp.linalg.norm(b1 - a @ x1, np.inf)
    close(chk, f"solve native residual n={n} {dt}", get(res), [0.0], tol(dt) * 10)


def check_det(chk, n, dt):
    a = rand(n, n, dt)
    ah = get(a).astype(np.float64)
    d = cnp.linalg.det(a)
    close(chk, f"det n={n} {dt}", get(d), [np.linalg.det(ah)], tol(dt) * 10)


def check_eigh(chk, n, dt):
    a = spd(n, dt) - float(n) * cnp.eye(n, dtype=dt)  # indefinite symmetric
    ah = get(a).astype(np.float64)
    t = tol(dt) * 10
    for uplo in ("L", "U"):
        w, v = cnp.linalg.eigh(a, UPLO=uplo)
        wh, vh = get(w), get(v)
        lbl = f"eigh {uplo} n={n} {dt}"
        close(chk, lbl + " w", wh, np.linalg.eigvalsh(ah, UPLO=uplo), t)
        close(chk, lbl + " A V = V w", ah @ vh, vh * wh, t)
        close(chk, lbl + " V^TV", vh.T @ vh, np.eye(n), t)
    close(chk, f"eigvalsh n={n} {dt}", get(cnp.linalg.eigvalsh(a)), np.linalg.eigvalsh(ah), t)
    # only the named triangle is read
    au = a + cnp.random.rand(n, n, dtype=dt)  # nonsymmetric; triangles differ
    auh = get(au).astype(np.float64)
    close(chk, f"eigvalsh U triangle n={n} {dt}", get(cnp.linalg.eigvalsh(au, UPLO="U")),
          np.linalg.eigvalsh(auh, UPLO="U"), t)


def check_svd(chk, m, n, dt):
    a = rand(m, n, dt)
    ah = get(a).astype(np.float64)
    k = min(m, n)
    t = tol(dt) * 10
    sn = np.linalg.svd(ah, compute_uv=False)
    for full in (True, False):
        u, s, vh = cnp.linalg.svd(a, full_matrices=full)
        uh, sh, vhh = get(u), get(s), get(vh)
        un, _, vn = np.linalg.svd(ah, full_matrices=full)
        lbl = f"svd full={full} {m}x{n} {dt}"
        if uh.shape != un.shape or vhh.shape != vn.shape:
            close(chk, lbl + " shapes", np.zeros(uh.shape + vhh.shape), np.zeros(un.shape + vn.shape), 0)
            continue
        close(chk, lbl + " s", sh, sn, t)
        close(chk, lbl + " U S Vh", (uh[:, :k] * sh) @ vhh[:k], ah, t)
        close(chk, lbl + " U^TU", uh.T @ uh, np.eye(uh.shape[1]), t)
        close(chk, lbl + " Vh Vh^T", vhh @ vhh.T, np.eye(vhh.shape[0]), t)
    close(chk, f"svd compute_uv=False {m}x{n} {dt}", get(cnp.linalg.svd(a, compute_uv=False)), sn, t)
    close(chk, f"svdvals {m}x{n} {dt}", get(cnp.linalg.svdvals(a)), sn, t)


def check_lstsq(chk, m, n, dt):
    a = rand(m, n, dt)
    b1 = cnp.random.rand(m, dtype=dt)
    b2 = cnp.random.rand(m, 2, dtype=dt)
    ah, b1h, b2h = (get(x).astype(np.float64) for x in (a, b1, b2))
    t = tol(dt) * 100
    for b, bh, nm in ((b1, b1h, "vec"), (b2, b2h, "mat")):
        x, res, rank, s = cnp.linalg.lstsq(a, b)
        xn, resn, rankn, sn = np.linalg.lstsq(ah, bh, rcond=None)
        lbl = f"lstsq {nm} {m}x{n} {dt}"
        close(chk, lbl + " x", get(x), xn, t)
        close(chk, lbl + " s", get(s), sn, t)
        close(chk, lbl + " rank", get(rank), [rankn], 0)
        resh = np.asarray(res) if isinstance(res, np.ndarray) else get(res)
        close(chk, lbl + " residuals", resh, resn, t)


def check_solve_triangular(chk, n, dt):
    a = rand(n, n, dt, shift=2.0)
    b = cnp.random.rand(n, 2, dtype=dt)
    b1 = cnp.random.rand(n, dtype=dt)
    ah, bh, b1h = (get(x).astype(np.float64) for x in (a, b, b1))
    t = tol(dt) * 10
    for trans in (0, 1, "T"):
        for lower in (False, True):
            for unit in (False, True):
                tri = np.tril(ah) if lower else np.triu(ah)
                if unit:
                    np.fill_diagonal(tri, 1.0)
                op = tri if trans == 0 else tri.T
                x = cnp.linalg.solve_triangular(a, b, trans=trans, lower=lower, unit_diagonal=unit)
                close(chk, f"solve_triangular trans={trans} lower={lower} unit={unit} n={n} {dt}",
                      get(x), np.linalg.solve(op, bh), t)
    x1 = cnp.linalg.solve_triangular(a, b1, lower=True)
    close(chk, f"solve_triangular vec n={n} {dt}", get(x1), np.linalg.solve(np.tril(ah), b1h), t)


def check_tssvd(chk, m, n, dt):
    a = rand(m, n, dt, shift=float(n))
    ah = get(a).astype(np.float64)
    u, s, vh = cnp.linalg.tssvd(a)
    uh, sh, vhh = get(u), get(s), get(vh)
    t = tol(dt) * 100
    lbl = f"tssvd {m}x{n} {dt}"
    close(chk, lbl + " s", sh, np.linalg.svd(ah, compute_uv=False), t)
    close(chk, lbl + " (u*s)@vh", (uh * sh) @ vhh, ah, t)
    close(chk, lbl + " U^TU", uh.T @ uh, np.eye(n), t)


def check_views_and_ints(chk):
    big = spd(40, np.float64)
    view = big[3:30, 3:30]  # offset view of an SPD matrix is SPD
    vh = get(big)[3:30, 3:30]
    close(chk, "cholesky offset view", get(cnp.linalg.cholesky(view)), np.linalg.cholesky(vh), 1e-9)
    a = rand(50, 50, np.float64, shift=10.0)
    ah = get(a)
    sv = a[1::2, 2::2]  # strided 25x24
    close(chk, "svd strided view", get(cnp.linalg.svd(sv, compute_uv=False)),
          np.linalg.svd(ah[1::2, 2::2], compute_uv=False), 1e-9)
    b = cnp.random.rand(60)
    bh = get(b)
    x = cnp.linalg.solve(a[0:50:2, 0:50:2], b[10:60:2])
    close(chk, "solve strided views", get(x), np.linalg.solve(ah[0:50:2, 0:50:2], bh[10:60:2]), 1e-9)
    ai = cnp.random.randint(-5, 6, size=(9, 9))
    aih = get(ai)
    d = cnp.linalg.det(ai + 20 * cnp.eye(9, dtype=np.int64))
    dtype_ok(chk, "det int64", d, np.float64)
    close(chk, "det int64", get(d), [np.linalg.det(aih + 20 * np.eye(9))], 1e-9)
    # mixed float32/float64 operands promote to float64
    a32 = rand(12, 12, np.float32, shift=5.0)
    b64 = cnp.random.rand(12)
    x = cnp.linalg.solve(a32, b64)
    dtype_ok(chk, "solve f32/f64", x, np.float64)
    close(chk, "solve f32/f64", get(x),
          np.linalg.solve(get(a32).astype(np.float64), get(b64)), 1e-9)
    # results feed later native expressions
    q, r = cnp.linalg.qr(a)
    close(chk, "native Q @ R", get(q @ r), ah, 1e-9)
    w, v = cnp.linalg.eigh(big)
    close(chk, "native V diag(w) V^T", get(v @ cnp.diag(w) @ cnp.linalg.inv(v)), get(big), 1e-9)
    # 1x1 and 1-row/column operands
    one = cnp.full((1, 1), 4.0)
    close(chk, "cholesky 1x1", get(cnp.linalg.cholesky(one)), [[2.0]], 1e-12)
    close(chk, "inv 1x1", get(cnp.linalg.inv(one)), [[0.25]], 1e-12)
    row = cnp.random.rand(1, 5)
    close(chk, "svd 1x5", get(cnp.linalg.svd(row, compute_uv=False)),
          np.linalg.svd(get(row), compute_uv=False), 1e-12)
    # failures produce NaN instead of raising
    notpd = cnp.zeros((4, 4)) - cnp.eye(4)
    chk.count += 1
    if np.isnan(get(cnp.linalg.cholesky(notpd))).all():
        print("  PASS cholesky not-PD -> NaN")
    else:
        print("  FAIL cholesky not-PD -> NaN")
        chk.failures.append("cholesky not-PD")
    chk.raises("cholesky non-square", np.linalg.LinAlgError,
               lambda: cnp.linalg.cholesky(cnp.ones((3, 4))))
    chk.raises("solve 1-D a", np.linalg.LinAlgError,
               lambda: cnp.linalg.solve(cnp.ones((3,)), cnp.ones((3,))))
    chk.raises("solve rhs mismatch", ValueError,
               lambda: cnp.linalg.solve(cnp.eye(3), cnp.ones((4,))))
    chk.raises("qr raw", NotImplementedError, lambda: cnp.linalg.qr(cnp.eye(3), mode="raw"))


def main():
    global IFACE
    IFACE = connect()
    chk = Checker("test_linalg", IFACE)
    for dt in (np.float64, np.float32):
        for n in (2, 7, 64, 300):
            check_cholesky(chk, n, dt)
            check_solve(chk, n, dt)
            check_eigh(chk, n, dt)
            check_solve_triangular(chk, n, dt)
        for n in (2, 7, 30):
            check_det(chk, n, dt)
        for m, n in ((7, 7), (40, 9), (9, 40), (300, 77)):
            check_qr(chk, m, n, dt)
            check_svd(chk, m, n, dt)
            check_lstsq(chk, m, n, dt)
        for m, n in ((50, 5), (1000, 16)):
            check_tssvd(chk, m, n, dt)
    # several chares on the CPU build (2-D tile 512): non-aligned, rectangular
    check_cholesky(chk, 1100, np.float64)
    check_solve(chk, 777, np.float64)
    check_qr(chk, 1030, 600, np.float64)
    check_svd(chk, 600, 1030, np.float64)
    check_eigh(chk, 700, np.float64)
    check_solve_triangular(chk, 900, np.float32)
    check_tssvd(chk, 2000, 40, np.float64)
    # Size-1 operands are covered with constant inputs only: random 1x1 inputs
    # hit backend bugs outside linalg (after r + eye(1), a later DAG that grows
    # the 2-D grid hangs; 1x1 @ (1,) hangs; fetched 1x1 arrays can be garbage).
    check_views_and_ints(chk)
    chk.finish()


if __name__ == "__main__":
    main()
