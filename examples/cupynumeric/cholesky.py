#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/cholesky.py.

Verdict: direct. Same input (np.eye(n)) and the same call,
cnp.linalg.cholesky(input), which is native (LINALG executor: gathered to one
chare and factored there). The check compares with numpy.linalg.cholesky, as
the original does, and always runs. --input spd factors a non-trivial SPD
matrix instead (0.5*(R + R.T) + n*I, with the native transpose R.T).
Complex dtypes are not supported.
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
from charmtyles.core import sync  # noqa: E402


def check_equal_numpy(chk, interface, input, output, dtype):
    print("Checking result...")
    out2 = numpy.linalg.cholesky(input.get(interface))
    rtol, atol = U.tolerances(dtype)
    chk.close(f"cholesky n={input.shape[0]} {dtype} vs numpy", output.get(interface), out2,
              rtol=rtol, atol=atol * max(1.0, float(numpy.abs(out2).max())))


def make_input(n, dtype, kind):
    if kind == "eye":
        return cnp.eye(n, dtype=dtype)
    R = cnp.random.rand(n, n, dtype=dtype)
    return 0.5 * (R + R.T) + float(n) * cnp.eye(n, dtype=dtype)


def cholesky(chk, interface, n, dtype, kind, print_timing=False):
    input = make_input(n, dtype, kind)
    sync(interface)

    t0 = time.perf_counter()
    out1 = cnp.linalg.cholesky(input)
    sync(interface)
    total = (time.perf_counter() - t0) * 1e3

    check_equal_numpy(chk, interface, input, out1, dtype)

    if print_timing:
        print(f"Elapsed Time: {total} ms")
        flops = (n**3) / 3 + 2 * n / 3
        print(f"{flops / total / 1000000} GOP/s")
    return total


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true", help="perform timing")
    parser.add_argument("-n", "--num", type=int, nargs="+", default=[10], dest="n",
                        help="number of rows/cols in the matrix")
    parser.add_argument("-d", "--dtype", default="float64", choices=["float32", "float64"],
                        dest="dtype", help="data type")
    parser.add_argument("--input", default="eye", choices=["eye", "spd"], dest="input",
                        help="eye (as the original) or a random SPD matrix")
    parser.add_argument("--check", dest="check", action="store_true",
                        help="compare result to numpy (always done)")
    args = parser.parse_args()

    interface = get_interface()
    for n in args.n:
        cholesky(chk, interface, n, numpy.dtype(args.dtype), args.input, args.timing)


if __name__ == "__main__":
    U.run("cholesky", main)
