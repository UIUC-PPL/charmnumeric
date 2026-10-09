#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/jacobi.py (Jacobi linear solver).

Verdict: direct port. Changes from the original:
  * np.random.rand / np.eye are the native cnp.random.rand / cnp.eye (the
    random stream differs from NumPy's).
  * The NumPy reference runs the same iteration on the fetched A and b.
  * The original's default-path assertion `not math.isnan(np.sum(x))` uses
    the native full reduction cnp.sum. The NaN/inf counts in its message
    (count_nonzero(isnan(x))) are computed on the host, since there is no
    isnan yet.
"""

import argparse
import math
import os
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from _cnum_connect import get_interface  # noqa: E402


def generate_random(N):
    print("Generating %dx%d system..." % (N, N))
    # Generate a random matrix
    A = cnp.random.rand(N, N)
    # Make sure that it is diagonally dominate
    A = A + N * cnp.eye(N)
    # Generate a random vector
    b = cnp.random.rand(N)
    return A, b


def jacobi_iterations(np, A, b, N, iters, warmup):
    x = np.zeros(A.shape[1])
    d = np.diag(A)
    R = A - np.diag(d)
    for i in range(iters + warmup):
        x = (b - np.dot(R, x)) / d
    return x


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", dest="check", action="store_true")
    parser.add_argument("-i", "--iters", type=int, default=50, dest="iters")
    parser.add_argument("-w", "--warmup", type=int, default=5, dest="warmup")
    parser.add_argument("-n", "--num", type=int, default=128, dest="N")
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    args = parser.parse_args()
    N = args.N

    interface = get_interface()
    A, b = generate_random(N)

    print("Solving system...")
    t0 = time.perf_counter()
    x = jacobi_iterations(cnp, A, b, N, args.iters, args.warmup)
    x_val = x.get(interface)
    if args.timing:
        print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    print("Checking result...")
    Ax = (A @ x).get(interface)  # the original check: allclose(A.dot(x), b)
    A_h = A.get(interface)
    b_h = b.get(interface)
    sum_x = U.to_float(cnp.sum(x), interface)
    chk.true(
        "no NaN in x (native sum)",
        not math.isnan(sum_x),
        f"{numpy.count_nonzero(numpy.isnan(x_val))} NaNs, "
        f"{numpy.count_nonzero(numpy.isinf(x_val))} infs in x",
    )
    chk.close("sum(x)", sum_x, numpy.sum(x_val), rtol=1e-12)
    chk.close("A.dot(x) vs b", Ax, b_h, rtol=1e-5, atol=1e-8)
    x_ref = jacobi_iterations(numpy, A_h, b_h, N, args.iters, args.warmup)
    chk.close("x vs numpy", x_val, x_ref, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    U.run("jacobi", main)
