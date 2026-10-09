#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/scan.py (cumsum/cumprod/nancumsum/nancumprod).

Verdict: direct for axis scans. cnp.cumsum/cumprod/nancumsum/nancumprod are
native (SCAN executor: per-tile scan along the axis, exclusive combination of
the per-tile line totals along the chare grid, offset add). Gaps:
  * axis=None on an N-D array needs flatten/reshape (missing). The port
    computes the row-major (flattened-order) scan from native scans and keeps
    the N-D shape: scan the last axis, scan the row totals, and combine the
    previous rows' carry into each row. The check compares against numpy's
    1-D result reshaped. Only 1-D/2-D for axis=None.
  * complex dtype is unsupported (no complex in charmnumeric).
  * Inputs come from the native cnp.random.randint / cnp.random.random.
    As in NumPy, int32 inputs accumulate in int64; B (the original's out=
    array) keeps the input dtype, and assigning into it casts natively.
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


_SCANS = {
    "cumsum": cnp.cumsum,
    "cumprod": cnp.cumprod,
    "nancumsum": cnp.nancumsum,
    "nancumprod": cnp.nancumprod,
}


def scan_flat(A, OP):
    """Row-major (flattened) scan of an N-D array, kept in N-D shape."""
    scan = _SCANS[OP]
    if A.ndim == 1:
        return scan(A, axis=0)
    if A.ndim != 2:
        raise NotImplementedError("axis=None scan implemented for 1-D/2-D only (needs reshape)")
    mul = OP in ("cumprod", "nancumprod")
    rows = scan(A, axis=1)  # native scan of each row
    n0, n1 = A.shape
    # inclusive scan of the row totals (already NaN-free for the nan* variants)
    carry = (cnp.cumprod if mul else cnp.cumsum)(rows[:, n1 - 1 : n1], axis=0)
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
    A, B, _ = initialize(shape, dt, ax)
    if ax is None:
        B[...] = scan_flat(A, OP)
    else:
        _SCANS[OP](A, out=B, axis=ax)  # native scan, cast into B like numpy's out=
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
