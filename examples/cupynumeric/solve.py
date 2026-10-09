#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/solve.py (np.linalg.solve).

Verdict: needs backend features (getrf/getrs, or row gather + trsm). The
port is a blocked right-looking LU with partial pivoting, i.e. the
hybrid "panel on host, trailing update on device" scheme:
    for each panel k (bs columns):
        p, L11, U11 = getrf(LU[k:, k:k+bs])   HOST-FALLBACK: tall panel pivoting
                                              (needs argmax/compare, row swaps)
        swap rows natively                     one whole-row slice copy per moved row
        L21 = A21 @ U11^{-1}                   native SUMMA (U11^{-1}: bs x bs upload)
        U12 = L11^{-1} @ A12                   native SUMMA
        A22 -= L21 @ U12                       native SUMMA
    forward/back substitution, blocked, all native matvec/matmul with the
    uploaded inverse diagonal blocks
The panel is fetched to choose pivots. Only the bs x bs inverses of L11
and U11 are uploaded (bulk cnp.array uploads). a and b come from
cnp.random.rand, as in the original. With nrhs == 1, b is
kept 1-D so that every product is a plain matvec. The original check
(np.linalg.norm(b - A x, inf) < 1e-4) uses the native cnp.linalg.norm, and
x is also compared with numpy.linalg.solve.
"""

import argparse
import os
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from _cnum_connect import get_interface  # noqa: E402


def host_panel_getrf(panel, bs):
    """Partial-pivoted LU of a tall (rows x bs) panel. Returns (perm, L11, U11)."""
    P = panel.copy()
    rows = P.shape[0]
    perm = numpy.arange(rows)
    for j in range(bs):
        piv = j + int(numpy.argmax(numpy.abs(P[j:, j])))
        if P[piv, j] == 0.0:
            raise numpy.linalg.LinAlgError("singular matrix")
        if piv != j:
            P[[j, piv]] = P[[piv, j]]
            perm[[j, piv]] = perm[[piv, j]]
        P[j + 1 :, j] /= P[j, j]
        P[j + 1 :, j + 1 :] -= numpy.outer(P[j + 1 :, j], P[j, j + 1 :])
    L11 = numpy.tril(P[:bs, :bs], -1) + numpy.eye(bs)
    U11 = numpy.triu(P[:bs, :bs])
    return perm, L11, U11


def permute_rows(X, perm, k):
    """X[k + i] = X_old[k + perm[i]] for the rows that move (native slice copies)."""
    moved = [i for i in range(len(perm)) if perm[i] != i]
    if not moved:
        return
    Xc = X.copy()
    for i in moved:
        src = k + int(perm[i])
        if X.ndim == 1:
            X[k + i : k + i + 1] = Xc[src : src + 1]
        else:
            X[k + i : k + i + 1, :] = Xc[src : src + 1, :]


def lu_solve(A, B, n, bs, interface):
    LU = A.copy()
    X = B.copy()
    Linv, Uinv = [], []
    for k in range(0, n, bs):
        e = min(k + bs, n)
        w = e - k
        # HOST-FALLBACK: pivot selection + panel factorization on the host
        panel = LU[k:, k:e].get(interface)
        perm, L11, U11 = host_panel_getrf(panel, w)
        permute_rows(LU, perm, k)  # swap whole rows (incl. computed L columns), native
        permute_rows(X, perm, k)  # apply P to the right-hand side, native
        # HOST-FALLBACK: bs x bs triangular inverses (no trsm/trtri)
        Li = U.upload(numpy.tril(numpy.linalg.inv(L11)), dtype=A.dtype)
        Ui = U.upload(numpy.triu(numpy.linalg.inv(U11)), dtype=A.dtype)
        Linv.append(Li)
        Uinv.append(Ui)
        if e < n:
            L21 = LU[e:, k:e] @ Ui  # native
            U12 = Li @ LU[k:e, e:]  # native
            LU[e:, k:e] = L21
            LU[k:e, e:] = U12
            LU[e:, e:] = LU[e:, e:] - L21 @ U12  # native Schur update

    # forward substitution: L y = P b   (blocked, native)
    blocks = list(range(0, n, bs))
    for bi, k in enumerate(blocks):
        e = min(k + bs, n)
        yk = Linv[bi] @ X[k:e] if X.ndim == 1 else Linv[bi] @ X[k:e, :]
        if X.ndim == 1:
            X[k:e] = yk
            if e < n:
                X[e:] = X[e:] - LU[e:, k:e] @ X[k:e]
        else:
            X[k:e, :] = yk
            if e < n:
                X[e:, :] = X[e:, :] - LU[e:, k:e] @ X[k:e, :]
    # back substitution: U x = y
    for bi in reversed(range(len(blocks))):
        k = blocks[bi]
        e = min(k + bs, n)
        if X.ndim == 1:
            X[k:e] = Uinv[bi] @ X[k:e]
            if k > 0:
                X[:k] = X[:k] - LU[:k, k:e] @ X[k:e]
        else:
            X[k:e, :] = Uinv[bi] @ X[k:e, :]
            if k > 0:
                X[:k, :] = X[:k, :] - LU[:k, k:e] @ X[k:e, :]
    return X


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("-n", "--num", type=int, default=128, dest="n")
    parser.add_argument("-s", "--nrhs", type=int, default=1, dest="nrhs")
    parser.add_argument("-b", "--block", type=int, default=16, dest="bs")
    parser.add_argument("-d", "--dtype", default="float64", choices=["float64"], dest="dtype")
    parser.add_argument("--check", dest="check", action="store_true")
    args = parser.parse_args()
    n, nrhs = args.n, args.nrhs

    interface = get_interface()
    a = cnp.random.rand(n, n)
    # b = rand(n, nrhs); with nrhs == 1 keep it 1-D so all products are matvecs
    b = cnp.random.rand(n) if nrhs == 1 else cnp.random.rand(n, nrhs)
    t0 = time.perf_counter()
    x = lu_solve(a, b, n, args.bs, interface)
    x_h = x.get(interface)
    total = (time.perf_counter() - t0) * 1e3
    if args.timing:
        print(f"Elapsed Time: {total} ms (incl. sync)")
        flops = (n**3) * 2 / 3 + (n**2) * nrhs * 2 / 3
        print(f"{flops / total / 1000000} GOP/s")

    print("Checking result...")
    res = b - a @ x  # native residual
    max_res = U.to_float(cnp.linalg.norm(res, numpy.inf), interface)  # native norm
    chk.true("max-res < 1e-4", max_res < 1e-4, f"(max-res = {max_res:.3e})")
    chk.close("native norm(res, inf)", max_res, numpy.linalg.norm(res.get(interface), numpy.inf),
              rtol=0, atol=0)
    a_h, b_h = a.get(interface), b.get(interface)
    chk.close("x vs numpy.linalg.solve", x_h, numpy.linalg.solve(a_h, b_h), rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    U.run("solve", main)
