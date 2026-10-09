#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/solve.py (np.linalg.solve).

Verdict: direct. a = rand(n, n), b = rand(n, nrhs) and x = cnp.linalg.solve(a, b),
as in the original (native LINALG executor: partial-pivoted LU on one chare).
The original's check, norm(b - a @ x, inf) < 1e-4, runs natively, and x is also
compared with numpy.linalg.solve on the fetched inputs. For float32 the
residual threshold is the backward-error bound n eps ||a|| ||x|| when that
exceeds 1e-4. Complex dtypes are not supported.
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


def check_result(chk, interface, a, b, x, dtype):
    print("Checking result...")
    res = b - cnp.matmul(a, x)
    max_res = U.to_float(cnp.linalg.norm(res, numpy.inf), interface)
    limit = 1e-04
    if dtype == numpy.float32:
        # backward-error bound n eps ||a|| ||x|| (the fixed 1e-4 is a float64 threshold)
        norm_a = U.to_float(cnp.linalg.norm(a, numpy.inf), interface)
        norm_x = U.to_float(cnp.linalg.norm(x, numpy.inf), interface)
        limit = max(limit, a.shape[0] * float(numpy.finfo(dtype).eps) * norm_a * norm_x)
    chk.true(f"max-res < {limit:.3g}", max_res < limit, f"(max-res = {max_res})")
    ah = a.get(interface).astype(numpy.float64)
    bh = b.get(interface).astype(numpy.float64)
    rtol, atol = U.tolerances(dtype)
    cond = numpy.linalg.cond(ah)
    chk.close("x vs numpy.linalg.solve", x.get(interface), numpy.linalg.solve(ah, bh),
              rtol=rtol * 10 * max(1.0, cond * numpy.finfo(dtype).eps / rtol),
              atol=atol * 10 * max(1.0, cond * numpy.finfo(dtype).eps / atol))


def solve(chk, interface, n, nrhs, dtype, print_timing=False):
    a = cnp.random.rand(n, n, dtype=dtype)
    b = cnp.random.rand(n, nrhs, dtype=dtype)
    sync(interface)

    t0 = time.perf_counter()
    x = cnp.linalg.solve(a, b)
    sync(interface)
    total = (time.perf_counter() - t0) * 1e3

    check_result(chk, interface, a, b, x, dtype)

    if print_timing:
        print(f"Elapsed Time: {total} ms")
        getrf_flops = (n**3) * 2 / 3
        getrs_flops = (n**2) * nrhs * 2 / 3
        flops = getrf_flops + getrs_flops
        print(f"{flops / total / 1000000} GOP/s")
    return total


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true", help="perform timing")
    parser.add_argument("-n", "--num", type=int, nargs="+", default=[10], dest="n",
                        help="number of rows/columns in the matrix")
    parser.add_argument("-s", "--nrhs", type=int, nargs="+", default=[1], dest="nrhs",
                        help="number of right hand sides")
    parser.add_argument("-d", "--dtype", default="float64", choices=["float32", "float64"],
                        dest="dtype", help="data type")
    parser.add_argument("--check", dest="check", action="store_true",
                        help="compare result to numpy (always done)")
    args = parser.parse_args()

    interface = get_interface()
    for n, nrhs in itertools.product(args.n, args.nrhs):
        solve(chk, interface, n, nrhs, numpy.dtype(args.dtype), args.timing)


if __name__ == "__main__":
    U.run("solve", main)
