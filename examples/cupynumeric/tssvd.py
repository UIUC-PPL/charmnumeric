#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/tssvd.py (tall-skinny SVD).

Verdict: direct. make_random_matrix is the original loop
(np.random.seed(6174); rand(m, n); mat[i, i] = 1 + sum(abs(mat[i, :]))) on
cnp.random, and u, s, vh = cnp.linalg.tssvd(A) is native with cuPyNumeric's
algorithm: A^T A and its eigh on one chare (A is gathered there), then
U = A @ (V S^-1) as a distributed SUMMA matmul. The original's check,
allclose(A, (u * s) @ vh), runs natively with u * s written as u @ diag(s);
s is also compared with numpy.linalg.svd.
"""

import argparse
import itertools
import os
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from _cnum_connect import get_interface  # noqa: E402
from charmtyles.core import sync  # noqa: E402


def check_result(chk, interface, a, u, s, vh):
    print("Checking result...")

    # (u * s) @ vh
    a2 = cnp.matmul(cnp.matmul(u, cnp.diag(s)), vh)
    rtol, atol = U.tolerances(a.dtype)
    ok = U.allclose(a, a2, interface, rtol=rtol, atol=atol * max(1, a.shape[1]) ** 0.5)
    chk.true(f"allclose(A, (u * s) @ vh) {a.shape[0]}x{a.shape[1]}", ok)
    s_ref = numpy.linalg.svd(a.get(interface), compute_uv=False)
    chk.close("singular values vs numpy", s.get(interface), s_ref, rtol=1e-8, atol=1e-10)


# make random real full column rank mxn, m>n matrix:
#
def make_random_matrix(m, n):
    cnp.random.seed(6174)

    mat = cnp.random.rand(m, n)

    # strictly diagonally dominant:
    #
    for i in range(n):
        mat[i, i] = 1.0 + cnp.sum(cnp.abs(mat[i, :]))

    return mat


def run_tssvd(chk, interface, m, n, print_timing=False):
    A = make_random_matrix(m, n)
    sync(interface)

    t0 = time.perf_counter()
    u, s, vh = cnp.linalg.tssvd(A)
    sync(interface)
    total = (time.perf_counter() - t0) * 1e3

    check_result(chk, interface, A, u, s, vh)

    if print_timing:
        print(f"TSSVD elapsed Time: {total:.3f} ms")
    return total


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true", help="perform timing")
    parser.add_argument("-m", "--rows", type=int, nargs="+", default=[10], dest="m",
                        help="number of rows in the matrix")
    parser.add_argument("-n", "--cols", type=int, nargs="+", default=[10], dest="n",
                        help="number of cols in the matrix")
    parser.add_argument("--check", dest="check", action="store_true",
                        help="compare result to numpy (always done)")
    args = parser.parse_args()

    interface = get_interface()
    for m, n in itertools.product(args.m, args.n):
        run_tssvd(chk, interface, m, n, args.timing)


if __name__ == "__main__":
    U.run("tssvd", main)
