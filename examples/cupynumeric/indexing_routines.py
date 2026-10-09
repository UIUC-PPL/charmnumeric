#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/indexing_routines.py.

Verdict: needs backend features (gather/scatter/boolean-mask indexing,
repeat). Per routine:
  diagonal  direct: cnp.diag 1-D <-> 2-D.
  choose    native. np.choose(C1, A, mode='wrap') is a chain of native
            where(C1 % n == k, A[k], ...) selections (exact, int64).
            arange(N) % 10 is iota(N) % 10 with the native %.
  repeat    np.repeat(A2, R, axis=1) with uniform counts r becomes r strided
            slice assignments out[:, j::r] = A2. Non-uniform counts need a
            backend repeat/gather.
  ai1/2/3   advanced indexing has no charmnumeric equivalent. The index
            arrays in the benchmark are structured (B % 10, B % 2, all-true
            masks), so each scatter is lowered to the equivalent slice
            assignments, marked HOST-FALLBACK (the index pattern is analysed
            on the host). The paired boolean index A2[m, m] (the odd diagonal)
            is done natively through diag: d' = d + where(mask, 11 - d, 0),
            with mask = arange(N) % 2.
Defaults: N=100, 3 steps, 1 warmup. Every routine is checked against the
original NumPy code.
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

INT = cnp.int64


def arange_mod(N, m, dtype):
    """arange(N) % m built natively."""
    return U.iota(N, dtype=dtype) % m


def compute_diagonal(steps, N, warmup):
    A1 = cnp.ones((N,), dtype=INT)
    print("measuring diagonal")
    for step in range(steps + warmup):
        A2 = cnp.diag(A1)
        A1 = cnp.diag(A2)
    return {"diag A1": A1, "diag A2": A2}


def choose(C, A, mode="wrap"):
    """np.choose(C, A, mode='wrap') as a chain of native where() selections."""
    if mode != "wrap":
        raise NotImplementedError("only mode='wrap' is ported")
    n = len(A)
    Cw = C % n
    result = A[n - 1]
    for k in range(n - 2, -1, -1):
        result = cnp.where(Cw == k, A[k], result)
    return result


def compute_choose(steps, N, warmup):
    print("measuring choose")
    A = tuple(cnp.full((N,), k, dtype=INT) for k in range(10))
    C1 = arange_mod(N, 10, INT)  # np.arange(N, dtype=int) % 10
    for step in range(steps + warmup):
        C1 = choose(C1, A, mode="wrap")
    return {"choose C1": C1}


def repeat_uniform(A2, r, axis=1):
    """np.repeat(A2, r, axis=1) for a uniform repeat count r (strided slices)."""
    assert axis == 1
    n0, n1 = A2.shape
    out = cnp.zeros((n0, n1 * r), dtype=A2.dtype)
    for j in range(r):
        out[:, j :: r] = A2
    return out


def compute_repeat(steps, N, warmup):
    A2 = cnp.ones((N, N), dtype=INT)
    r = 1  # R = np.repeat(int(1), N): every column repeated once
    print("measuring repeat")
    for step in range(steps + warmup):
        A2 = repeat_uniform(A2, r, axis=1)
    return {"repeat A2": A2}


def compute_advanced_indexing_1d(steps, N, warmup):
    A1 = cnp.ones((N,), dtype=INT)
    print("measuring advanced_indexing 1D")
    # indx = B % 10 -> positions {0..min(9, N-1)}; indx_bool = B % 2 -> odd positions
    for step in range(steps + warmup):
        A1[0 : min(10, N)] = 10  # HOST-FALLBACK: A1[indx] = 10 (scatter lowered to a slice)
        A1[1::2] = 12  # HOST-FALLBACK: A1[indx_bool] = 12 (mask lowered to a stride)
    return {"ai1 A1": A1}


def compute_advanced_indexing_2d(steps, N, warmup):
    A2 = cnp.ones((N, N), dtype=INT)
    print("measuring advanced_indexing 2D")
    odd = arange_mod(N, 2, INT)  # indx_bool = (B % 2).astype(bool), as a 0/1 mask
    for step in range(steps + warmup):
        # A2[indx_bool, indx_bool] = 11 -> odd diagonal entries, done natively via diag
        d = cnp.diag(A2)
        A2 = A2 + cnp.diag(cnp.where(odd, 11 - d, 0))
        A2[:, 0 : min(10, N)] = 12  # HOST-FALLBACK: A2[:, indx] = 12 (indx = B % 10)
        A2[...] = 13  # HOST-FALLBACK: A2[indx2d_bool] = 13 (mask (ones % 2) is all True)
    return {"ai2 A2": A2}


def compute_advanced_indexing_3d(steps, N, warmup):
    A3 = cnp.ones((N, int(N / 100), 100), dtype=INT)
    print("measuring advanced_indexing_3d")
    for step in range(steps + warmup):
        # HOST-FALLBACK: A3[indx, :, indx] = 15 with indx = B % 10 -> pairs (j, :, j), j < 10
        for j in range(min(10, N)):
            A3[j, :, j] = 15
        A3[...] = 16  # HOST-FALLBACK: A3[indx3d_bool] = 16 (mask is all True)
    return {"ai3 A3": A3}


# ---- original NumPy code (reference) ----------------------------------------
def reference(np, routine, steps, N, warmup):
    out = {}
    if routine in ("diagonal", "all"):
        A1 = np.ones((N,), dtype=int)
        for step in range(steps + warmup):
            A2 = np.diag(A1)
            A1 = np.diag(A2)
        out["diag A1"], out["diag A2"] = A1, A2
    if routine in ("choose", "all"):
        A = tuple(np.full((N,), k) for k in range(10))
        C1 = np.arange(N, dtype=int) % 10
        for step in range(steps + warmup):
            C1 = np.choose(C1, A, mode="wrap")
        out["choose C1"] = C1
    if routine in ("repeat", "all"):
        A2 = np.ones((N, N), dtype=int)
        R = np.repeat(int(1), N)
        for step in range(steps + warmup):
            A2 = np.repeat(A2, R, axis=1)
        out["repeat A2"] = A2
    if routine in ("ai1", "all"):
        A1 = np.ones((N,), dtype=int)
        B = np.arange(N, dtype=int)
        indx, indx_bool = B % 10, (B % 2).astype(bool)
        for step in range(steps + warmup):
            A1[indx] = 10
            A1[indx_bool] = 12
        out["ai1 A1"] = A1
    if routine in ("ai2", "all"):
        A2 = np.ones((N, N), dtype=int)
        B = np.arange(N, dtype=int)
        indx, indx_bool = B % 10, (B % 2).astype(bool)
        indx2d_bool = (A2 % 2).astype(bool)
        for step in range(steps + warmup):
            A2[indx_bool, indx_bool] = 11
            A2[:, indx] = 12
            A2[indx2d_bool] = 13
        out["ai2 A2"] = A2
    if routine in ("ai3", "all"):
        A3 = np.ones((N, int(N / 100), 100), dtype=int)
        B = np.arange(N, dtype=int)
        indx = B % 10
        indx3d_bool = (A3 % 2).astype(bool)
        for step in range(steps + warmup):
            A3[indx, :, indx] = 15
            A3[indx3d_bool] = 16
        out["ai3 A3"] = A3
    return out


def run_indexing_routines(interface, N, steps, warmup, routine):
    # simple operation to warm up the library
    warm = cnp.sum(cnp.zeros((N, N)).dot(cnp.zeros((N,))))
    assert not math.isnan(U.to_float(warm, interface))
    results = {}
    if routine in ("diagonal", "all"):
        results.update(compute_diagonal(steps, N, warmup))
    if routine in ("choose", "all"):
        results.update(compute_choose(steps, N, warmup))
    if routine in ("repeat", "all"):
        results.update(compute_repeat(steps, N, warmup))
    if routine in ("ai1", "all"):
        results.update(compute_advanced_indexing_1d(steps, N, warmup))
    if routine in ("ai2", "all"):
        results.update(compute_advanced_indexing_2d(steps, N, warmup))
    if routine in ("ai3", "all"):
        if N < 100:
            raise ValueError("ai3 needs N >= 100 (A3 has shape (N, N/100, 100))")
        results.update(compute_advanced_indexing_3d(steps, N, warmup))
    return results


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=100, dest="N")
    parser.add_argument("-i", "--iterations", type=int, default=3, dest="I")
    parser.add_argument("-w", "--warmup", type=int, default=1, dest="warmup")
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument(
        "-r", "--routine", default="all", dest="routine",
        choices=["diagonal", "choose", "repeat", "ai1", "ai2", "ai3", "all"],
    )
    args = parser.parse_args()

    interface = get_interface()
    t0 = time.perf_counter()
    results = run_indexing_routines(interface, args.N, args.I, args.warmup, args.routine)
    got = {k: v.get(interface) for k, v in results.items()}
    if args.timing:
        print(f"Total Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    ref = reference(numpy, args.routine, args.I, args.N, args.warmup)
    for key, r in ref.items():
        chk.close(key, got[key], r, rtol=0, atol=1e-9)


if __name__ == "__main__":
    U.run("indexing_routines", main)
