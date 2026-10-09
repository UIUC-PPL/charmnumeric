#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/gemm.py (rotating GEMM).

Verdict: direct port.
  * np.dot(A, B, out=C) is native (cnp.dot with out=; the SUMMA result is
    copied into C).
  * np.random.uniform(...).astype(ft) is cnp.random.uniform(..., dtype=ft)
    (native generator; no astype yet).
  * --precision 16 is not supported (no float16 in charmnumeric).
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


def initialize(M, N, K, ft):
    A = cnp.random.uniform(size=(N, N), dtype=ft)
    B = cnp.random.uniform(size=(N, N), dtype=ft)
    C = cnp.zeros((N, N), dtype=ft)
    return A, B, C


def total_flops(M, N, K):
    return M * N * (2 * K - 1)


def gemm_loop(np, A, B, C, I, warmup):  # noqa: E741
    for idx in range(I + warmup):
        np.dot(A, B, out=C)
        # rotate the matrices (as in the original)
        A, B, C = B, C, A
    return A, B, C


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iter", type=int, default=4, dest="I")
    parser.add_argument("-w", "--warmup", type=int, default=1, dest="warmup")
    parser.add_argument("-n", "--num", type=int, default=128, dest="N")
    parser.add_argument("--precision", type=int, default=32, dest="P")
    args = parser.parse_args()

    if args.P == 32:
        ft = numpy.float32
    elif args.P == 64:
        ft = numpy.float64
    else:
        raise TypeError("Precision must be one of 32 or 64 (no float16 in charmnumeric)")
    N = args.N
    print(f"Problem Size:     M={N} N={N} K={N}")
    print(f"Total Iterations: {args.I}")
    print(f"Total Flops:      {total_flops(N, N, N) / 1e9} GFLOPS/iter")

    interface = get_interface()
    A, B, C = initialize(N, N, N, ft)
    A_h, B_h, C_h = A.get(interface), B.get(interface), C.get(interface)

    t0 = time.perf_counter()
    A, B, C = gemm_loop(cnp, A, B, C, args.I, args.warmup)
    out = [A.get(interface), B.get(interface), C.get(interface)]
    total = (time.perf_counter() - t0) * 1e3
    print(f"Elapsed Time:     {total:.3f} ms (incl. sync)")

    ref = gemm_loop(numpy, A_h.copy(), B_h.copy(), C_h.copy(), args.I, args.warmup)
    rtol = 1e-4 if ft == numpy.float32 else 1e-10
    for name, got, exp in zip("ABC", out, ref):
        chk.close(f"rotated {name}", got, exp, rtol=rtol, atol=0.0)


if __name__ == "__main__":
    U.run("gemm", main)
