#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/scan.py (cumsum/cumprod/nan* variants).

Verdict: port with workarounds. cumsum/cumprod are native: a Hillis-Steele
scan of ceil(log2 n) shifted-slice steps along the axis,
    B_new[k:] = B[k:] (+|*) B[:-k],
double-buffered so no step reads and writes the same array. Gaps:
  * nancumsum/nancumprod first replace NaNs with the identity natively:
    where(isnan(A), 0 or 1, A).
  * axis=None on an N-D array needs flatten/reshape (missing). The port
    computes the row-major (flattened-order) scan in place and keeps the
    N-D shape. The check compares against numpy's 1-D result reshaped.
  * complex dtype is unsupported (no complex in charmnumeric).
  * Inputs come from the native cnp.random.randint / cnp.random.random.
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


def initialize(shape, dt, axis):
    if dt == "int":
        A = cnp.random.randint(1000, size=shape, dtype=cnp.int32)
        B = cnp.zeros(shape, dtype=cnp.int32)
        nan_pos = None
    elif dt == "float":
        A = cnp.random.random(shape, dtype=cnp.float32)
        # insert NAN at second element
        nan_pos = (1,) * len(shape)
        A[nan_pos] = numpy.nan
        B = cnp.zeros(shape, dtype=cnp.float32)
    else:
        raise NotImplementedError("complex dtypes are not supported by charmnumeric")
    return A, B, nan_pos


def _sl(nd, axis, s):
    key = [slice(None)] * nd
    key[axis] = s
    return tuple(key)


def scan_axis(A, axis, mul):
    """Inclusive scan along `axis` (Hillis-Steele, log2(n) native steps)."""
    nd = A.ndim
    n = A.shape[axis]
    B = A.copy()
    k = 1
    while k < n:
        Bn = B.copy()
        hi, lo = B[_sl(nd, axis, slice(k, n))], B[_sl(nd, axis, slice(0, n - k))]
        Bn[_sl(nd, axis, slice(k, n))] = (hi * lo) if mul else (hi + lo)
        B = Bn
        k *= 2
    return B


def scan_flat(A, mul):
    """Row-major (flattened) scan of an N-D array, kept in N-D shape."""
    if A.ndim == 1:
        return scan_axis(A, 0, mul)
    # scan the last axis, then carry the running total of the previous rows
    rows = scan_axis(A, A.ndim - 1, mul)
    if A.ndim != 2:
        raise NotImplementedError("axis=None scan implemented for 1-D/2-D only (needs reshape)")
    n0, n1 = A.shape
    last = rows[:, n1 - 1 : n1]  # (n0, 1) row totals
    carry = scan_axis(U.materialize(last), 0, mul)  # inclusive scan of row totals
    out = rows.copy()
    # row i gets combined with carry[i-1]
    tiled = cnp.tile(U.materialize(carry[0 : n0 - 1, :]), (1, n1))  # (n0-1, n1)
    if mul:
        out[1:, :] = rows[1:, :] * tiled
    else:
        out[1:, :] = rows[1:, :] + tiled
    return out


def run_scan(OP, shape, dt, ax):
    print(f"Problem Size:    shape={shape}")
    print(f"Problem Type:    OP={OP}")
    print(f"Axis:            axis={ax}")
    print(f"Data type:       dtype={dt}32")
    A, B, nan_pos = initialize(shape, dt, ax)
    src = A
    if OP.startswith("nan"):
        # nancumsum/nancumprod: NaNs count as the identity
        src = cnp.where(cnp.isnan(A), 1 if OP == "nancumprod" else 0, A)
    mul = OP in ("cumprod", "nancumprod")
    if ax is None:
        B[...] = scan_flat(src, mul)
    else:
        B[...] = scan_axis(src, ax % len(shape), mul)
    return A, B


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shape", type=int, nargs="+", default=[1000], dest="shape")
    parser.add_argument("-t", "--datatype", default="int", choices=["int", "float", "complex"], dest="dt")
    parser.add_argument("-a", "--axis", type=int, default=None, dest="axis")
    parser.add_argument("-o", "--operation", default="cumsum",
                        choices=["cumsum", "cumprod", "nancumsum", "nancumprod"], dest="OP")
    parser.add_argument("--check", dest="check", action="store_true")
    args = parser.parse_args()
    shape = tuple(args.shape)

    interface = get_interface()
    t0 = time.perf_counter()
    A, B = run_scan(args.OP, shape, args.dt, args.axis)
    Bh = B.get(interface)
    print(f"Elapsed Time:  {(time.perf_counter() - t0) * 1e3:.3f}ms (incl. sync)")

    Ah = A.get(interface)
    if args.axis is None:
        C = getattr(numpy, args.OP)(Ah, axis=None).reshape(shape)
    else:
        C = numpy.zeros(shape, dtype=Bh.dtype)
        getattr(numpy, args.OP)(Ah, out=C, axis=args.axis)
    print("Checking result...")
    if args.dt == "int":
        chk.true("exact integer scan", bool(numpy.array_equal(Bh, C.astype(Bh.dtype))))
    else:
        chk.close(args.OP, Bh, C, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    U.run("scan", main)
