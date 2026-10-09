#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/cholesky.py.

Verdict: needs backend features (potrf/trsm). The port is a blocked,
right-looking Cholesky with every O(n^3) update done natively in SUMMA
matmuls. Only the bs x bs diagonal blocks are factored on the host:
    for each block column k:
        L11  = chol(A11)                          HOST-FALLBACK (bs x bs potrf)
        W    = L11^{-T}                           HOST-FALLBACK (bs x bs trtri)
        L21  = A21 @ W                            native SUMMA (trsm via inverse)
        L21T = W^T @ A12                          native SUMMA (== L21.T, since A12 == A21.T)
        A22 -= L21 @ L21T                         native SUMMA (Schur update)
There is no transpose op, so the L21.T needed by the Schur update is
computed from the upper block A12, using the symmetry of A. The host
factors are uploaded in bulk (cnp.array, one UPLOAD node each).
The original factors np.eye(n), whose Cholesky factor is trivially I. For
a meaningful check the default input is the SPD matrix from cg.py's
generate_random: 0.5*(R + R.T) + n*I, with R = cnp.random.rand(n, n) and
R.T from cnp.random.transposed (R.T regenerated from the same draw). --input eye
reproduces the original.
Backend ops that would make it native: potrf (or sqrt + per-column
scalar ops), trsm/trtri, transpose.
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


def make_input(n, kind, dtype):
    eye = cnp.diag(cnp.ones((n,), dtype=dtype))
    if kind == "eye":
        return eye
    R = cnp.random.rand(n, n, dtype=dtype)
    RT = cnp.random.transposed(R)  # == R.T (same draw, permuted indices)
    return 0.5 * (R + RT) + float(n) * eye


def cholesky_blocked(A, n, bs, interface):
    A = A.copy()  # working copy; trailing blocks receive the Schur updates
    L = cnp.zeros((n, n), dtype=A.dtype)
    for k in range(0, n, bs):
        e = min(k + bs, n)
        # HOST-FALLBACK: factor the (bs x bs) diagonal block (no potrf in charmnumeric)
        Akk = A[k:e, k:e].get(interface)
        Lkk = numpy.linalg.cholesky(Akk)
        U.upload(Lkk, target=L, offset=(k, k))
        if e < n:
            # HOST-FALLBACK: bs x bs triangular inverse (no trsm/trtri in charmnumeric)
            W = numpy.linalg.inv(Lkk).T  # L11^{-T}, upper triangular
            W_d = U.upload(numpy.triu(W), dtype=A.dtype)
            WT_d = U.upload(numpy.tril(W.T), dtype=A.dtype)
            L21 = A[e:, k:e] @ W_d  # (n-e, bs)  native
            L21T = WT_d @ A[k:e, e:]  # (bs, n-e) native, == L21.T by symmetry of A
            L[e:, k:e] = L21
            A[e:, e:] = A[e:, e:] - L21 @ L21T  # native Schur complement update
    return L


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("-n", "--num", type=int, default=128, dest="n")
    parser.add_argument("-b", "--block", type=int, default=16, dest="bs")
    parser.add_argument("-d", "--dtype", default="float64", choices=["float32", "float64"], dest="dtype")
    parser.add_argument("--input", default="spd", choices=["spd", "eye"], dest="input")
    parser.add_argument("--check", dest="check", action="store_true")
    args = parser.parse_args()
    dtype = numpy.dtype(args.dtype)

    interface = get_interface()
    A = make_input(args.n, args.input, dtype)
    A_h = A.get(interface)
    t0 = time.perf_counter()
    L = cholesky_blocked(A, args.n, args.bs, interface)
    out1 = L.get(interface)
    total = (time.perf_counter() - t0) * 1e3
    if args.timing:
        print(f"Elapsed Time: {total} ms (incl. sync)")
        flops = (args.n**3) / 3 + 2 * args.n / 3
        print(f"{flops / total / 1000000} GOP/s")

    print("Checking result...")
    out2 = numpy.linalg.cholesky(A_h)
    rtol = 1e-4 if dtype == numpy.float32 else 1e-9
    chk.close("L vs numpy.linalg.cholesky", out1, out2, rtol=rtol, atol=rtol)
    chk.close("L @ L.T vs A", out1 @ out1.T, A_h, rtol=rtol, atol=rtol * abs(A_h).max())


if __name__ == "__main__":
    U.run("cholesky", main)
