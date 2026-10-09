#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/tssvd.py (tall-skinny SVD).

Verdict: needs backend features (syevd). Same Gram-matrix scheme as svd.py
(native A^T A and U = A V S^-1, host eigh on the n x n Gram matrix). That
is also how a tall-skinny SVD is normally built. make_random_matrix is
native (np.random.seed(6174); np.random.rand(m, n) -> cnp.random):
    mat[i, i] = 1 + sum(abs(mat[i, :]))  for i < n
becomes  mat[0:n, :] += diag(1 + rowsum(|mat[0:n, :]|) - diag(mat[0:n, :]))
using the native row reduction cnp.linalg.norm(mat[0:n, :], ord=1, axis=1)
(= sum(abs(.), axis=1), exact) and cnp.diag. The
same diagonal update is applied to matT = cnp.random.transposed(raw) (the
diagonal sits at the same positions). Each row's sum uses only the
original entries of that row, as in the sequential loop.
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
from svd import gram_svd  # noqa: E402


def make_random_matrix(m, n):
    """Random real full-column-rank m x n (m > n) matrix, diagonally dominant on top."""
    cnp.random.seed(6174)
    raw = cnp.random.rand(m, n)
    mat = raw.copy()
    matT = cnp.random.transposed(raw)  # == mat.T
    # strictly diagonally dominant: mat[i, i] = 1 + sum(|mat[i, :]|), i < n
    rs = cnp.linalg.norm(mat[0:n, :], ord=1, axis=1)  # (n,) sum(abs(row)), native
    top = U.materialize(mat[0:n, :])
    d = cnp.diag(top)  # current diagonal
    delta = cnp.diag(1.0 + rs - d)  # n x n diagonal correction
    mat[0:n, :] = top + delta
    matT[:, 0:n] = matT[:, 0:n] + delta
    return mat, matT, raw


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("-m", "--rows", type=int, default=256, dest="m")
    parser.add_argument("-n", "--cols", type=int, default=16, dest="n")
    parser.add_argument("--check", dest="check", action="store_true")
    args = parser.parse_args()
    m, n = args.m, args.n
    if m < n:
        raise ValueError("tssvd needs m >= n")

    interface = get_interface()
    A, AT, raw_d = make_random_matrix(m, n)
    t0 = time.perf_counter()
    u, s, vh = gram_svd(A, AT, interface)
    u_h = u.get(interface)
    if args.timing:
        print(f"TSSVD elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    print("Checking result...")
    A_h, AT_h = A.get(interface), AT.get(interface)
    chk.close("generated A.T", AT_h, A_h.T, rtol=0, atol=0)
    # make_random_matrix semantics vs the original sequential loop on the raw matrix
    raw = raw_d.get(interface)
    ref_mat = raw.copy()
    for i in range(n):
        ref_mat[i, i] = 1.0 + numpy.sum(numpy.abs(ref_mat[i, :]))
    chk.close("make_random_matrix", A_h, ref_mat, rtol=1e-12, atol=1e-12)
    a2 = numpy.matmul(u_h * s, vh)
    chk.close("(u * s) @ vh vs A", a2, A_h, rtol=1e-5, atol=1e-8)
    chk.close("singular values vs numpy", s, numpy.linalg.svd(A_h, compute_uv=False), rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    U.run("tssvd", main)
