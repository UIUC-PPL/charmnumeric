#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/svd.py.

Verdict: direct. a = random((m, n)) (randint for integer dtypes) and
u, s, vh = cnp.linalg.svd(a, full_matrices), as in the original (native
LINALG executor: divide-and-conquer SVD on CPU builds, gesvd on GPU builds).
The original's check, allclose(a, (u * s) @ vh), runs natively, with u * s
written as u @ diag(s) (no general broadcasting); s is also compared with
numpy.linalg.svd. The original trims vh only when k < m, which breaks m < n;
here vh is trimmed when k < n. Integer inputs are factored in float64, as in
NumPy; their reconstruction check runs natively too (a promotes to float64 in
a - a2). Complex dtypes are not supported.
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


def check_result(chk, interface, a, u, s, vh, dtype):
    print("Checking result...")

    # (u * s) @ vh
    m = a.shape[0]
    n = a.shape[1]
    k = min(m, n)

    u = u[:, :k] if k < m else u
    vh = vh[:k, :] if k < n else vh
    a2 = cnp.matmul(cnp.matmul(u, cnp.diag(s)), vh)
    work = numpy.dtype(dtype) if numpy.dtype(dtype).kind == "f" else numpy.dtype("float64")
    rtol, atol = U.tolerances(work)
    atol *= max(1, k) ** 0.5
    ok = U.allclose(a, a2, interface, rtol=rtol, atol=atol)  # int a promotes natively
    chk.true(f"allclose(a, (u * s) @ vh) {m}x{n} {dtype}", ok)
    s_ref = numpy.linalg.svd(a.get(interface).astype(numpy.float64), compute_uv=False)
    chk.close("singular values vs numpy", s.get(interface), s_ref, rtol=rtol * 10,
              atol=atol * max(1.0, float(s_ref.max())))


def make_input(m, n, dtype):
    if numpy.issubdtype(dtype, numpy.integer):
        return cnp.random.randint(0, 1000, size=(m, n), dtype=dtype)
    return cnp.random.random((m, n), dtype=dtype)


def svd(chk, interface, m, n, full_matrices, dtype, print_timing=False):
    a = make_input(m, n, dtype)
    sync(interface)

    t0 = time.perf_counter()
    u, s, vh = cnp.linalg.svd(a, full_matrices)
    sync(interface)
    total = (time.perf_counter() - t0) * 1e3

    check_result(chk, interface, a, u, s, vh, dtype)

    if print_timing:
        print(f"Elapsed Time: {total} ms")
    return total


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true", help="perform timing")
    parser.add_argument("-m", "--rows", type=int, nargs="+", default=[10], dest="m",
                        help="number of rows in the matrix")
    parser.add_argument("-n", "--cols", type=int, nargs="+", default=[10], dest="n",
                        help="number of cols in the matrix")
    parser.add_argument("--no-full-matrices", dest="full_matrices", action="store_false",
                        help="return u as (m,k)")
    parser.add_argument("-d", "--dtype", default="float64",
                        choices=["int32", "int64", "float32", "float64"], dest="dtype",
                        help="data type")
    parser.add_argument("--check", dest="check", action="store_true",
                        help="compare result to numpy (always done)")
    args = parser.parse_args()

    interface = get_interface()
    for m, n in itertools.product(args.m, args.n):
        svd(chk, interface, m, n, args.full_matrices, numpy.dtype(args.dtype), args.timing)


if __name__ == "__main__":
    U.run("svd", main)
