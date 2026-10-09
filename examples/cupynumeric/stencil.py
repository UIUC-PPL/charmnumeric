#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/stencil.py (5-point Jacobi stencil).

Verdict: direct port. The algorithm function is shared with the NumPy
reference: run_stencil(np, ...) is called once with charmnumeric and once
with numpy.
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


def initialize(np, N):
    print("Initializing stencil grid...")
    grid = np.zeros((N + 2, N + 2))
    grid[:, 0] = -273.15
    grid[:, -1] = -273.15
    grid[-1, :] = -273.15
    grid[0, :] = 40.0
    return grid


def run_stencil(np, N, I, warmup):  # noqa: E741
    grid = initialize(np, N)

    print("Running Jacobi stencil...")
    center = grid[1:-1, 1:-1]
    north = grid[0:-2, 1:-1]
    east = grid[1:-1, 2:]
    west = grid[1:-1, 0:-2]
    south = grid[2:, 1:-1]

    for i in range(I + warmup):
        average = center + north + east + west + south
        work = 0.2 * average
        center[:] = work
    return grid


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iter", type=int, default=20, dest="I")
    parser.add_argument("-w", "--warmup", type=int, default=1, dest="warmup")
    parser.add_argument("-n", "--num", type=int, default=128, dest="N")
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    args = parser.parse_args()

    interface = get_interface()
    t0 = time.perf_counter()
    grid = run_stencil(cnp, args.N, args.I, args.warmup)
    result = grid.get(interface)
    if args.timing:
        print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    expected = run_stencil(numpy, args.N, args.I, args.warmup)
    chk.close("grid", result, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    U.run("stencil", main)
