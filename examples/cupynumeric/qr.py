#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/qr.py.

Verdict: needs backend features (geqrf, or potrf + trsm). The port is
CholeskyQR2, the usual communication-avoiding QR for distributed tall
matrices. All O(m n^2) work is native SUMMA. Only n x n factorizations
of the Gram matrix run on the host:
    G1 = A^T A                (native)    R1 = chol(G1)^T     HOST-FALLBACK
    Q1 = A @ R1^{-1}          (native)    Q1^T = R1^{-T} @ A^T (native)
    G2 = Q1^T Q1              (native)    R2 = chol(G2)^T     HOST-FALLBACK
    Q  = Q1 @ R2^{-1}         (native)    R  = R2 @ R1        HOST-FALLBACK (n x n)
A = cnp.random.rand(m, n) as in the original. There is no transpose, so
A^T comes from cnp.random.transposed (the same draw with permuted indices,
checked against A.T), and Q1^T is carried along. CholeskyQR2 is
accurate for cond(A) up to about 1e7 in float64. The original uses
np.random.rand on a square 1000x1000 matrix. The default here is
-m 128 -n 32 (the n x n factors R^{-1} are bulk uploads). Square inputs
work too. The check (Q @ R ~ A and Q^T Q ~ I) runs natively.
float64 only (Gram conditioning).
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


def cholqr_pass(A, AT, interface):
    G = AT @ A  # native SUMMA, n x n Gram matrix
    # HOST-FALLBACK: n x n Cholesky + triangular inverse (no potrf/trsm in charmnumeric)
    Gh = G.get(interface)
    R = numpy.linalg.cholesky(Gh).T  # upper
    Rinv = numpy.linalg.inv(R)  # upper
    Rinv_d = U.upload(numpy.triu(Rinv), dtype=A.dtype)
    RinvT_d = U.upload(numpy.tril(Rinv.T), dtype=A.dtype)
    Q = A @ Rinv_d  # native
    QT = RinvT_d @ AT  # native, == Q.T
    return Q, QT, R


def qr(A, AT, interface):
    Q1, Q1T, R1 = cholqr_pass(A, AT, interface)
    Q, QT, R2 = cholqr_pass(Q1, Q1T, interface)
    R = R2 @ R1  # HOST-FALLBACK: n x n product of the small triangular factors
    return Q, QT, R


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("-m", "--rows", type=int, default=128, dest="m")
    parser.add_argument("-n", "--cols", type=int, default=32, dest="n")
    parser.add_argument("-d", "--dtype", default="float64", choices=["float64"], dest="dtype")
    parser.add_argument("--check", dest="check", action="store_true")
    args = parser.parse_args()
    m, n = args.m, args.n
    if m < n:
        raise NotImplementedError("CholeskyQR needs m >= n")

    interface = get_interface()
    a = cnp.random.rand(m, n)
    aT = cnp.random.transposed(a)  # == a.T
    t0 = time.perf_counter()
    q, qT, r = qr(a, aT, interface)
    q_h = q.get(interface)
    if args.timing:
        print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3} ms (incl. sync)")

    print("Checking result...")
    a_h, aT_h = a.get(interface), aT.get(interface)
    chk.close("generated a.T", aT_h, a_h.T, rtol=0, atol=0)
    r_d = U.upload(numpy.triu(r))
    qr_prod = (q @ r_d).get(interface)  # np.matmul(q, r), native
    chk.close("Q @ R vs A", qr_prod, a_h, rtol=1e-9, atol=1e-10)
    qtq_d = qT @ q
    qtq = qtq_d.get(interface)
    chk.close("Q^T Q vs I", qtq, numpy.eye(n), rtol=0, atol=1e-10)
    # the original's diagnostic, native: qtq_err = eye - Q^T Q; max(qtq_err)
    qtq_err = cnp.diag(cnp.ones((n,))) - qtq_d
    q_max = U.to_float(cnp.max(qtq_err), interface)
    q_min = U.to_float(cnp.min(qtq_err), interface)
    print(f"charmnumeric I-qTq max: {q_max}")
    chk.true("max |I - Q^T Q| < 1e-10 (native max/min)", max(q_max, -q_min) < 1e-10,
             f"({max(q_max, -q_min):.3e})")
    chk.close("native max(I - Q^T Q)", q_max, numpy.max(numpy.eye(n) - qtq), rtol=0, atol=0)
    chk.true("R upper triangular", bool(numpy.allclose(r, numpy.triu(r))))


if __name__ == "__main__":
    U.run("qr", main)
