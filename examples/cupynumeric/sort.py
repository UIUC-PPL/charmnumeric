#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/sort.py (sort / argsort along an axis).

Verdict: port with workarounds (fully native). There is no sort op, so
the port uses a native odd-even transposition sorting network
(U.oddeven_sort). It has n phases of compare-exchange over strided views:
h = x > y (a native 0/1 mask), then where(h, y, x) / where(h, x, y), which
is exact for float and integer keys (equal keys are never swapped).
argsort carries an index array through the same network.
O(n^2) work instead of O(n log n). Defaults are sized for that (n=256).
Other changes:
  * uint32 (the original default) is not a charmnumeric dtype. The default
    here is float64. Use -d int64/int32 for integer keys (generated
    natively in [lower, upper)). complex is unsupported.
  * argsort is checked by applying it to the input: the result must be
    sorted and a permutation. Ties may legitimately order differently from
    numpy.
Backend ops that would make it direct: sort/argsort.
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


def make_input(shape, datatype, lower, upper):
    dt = numpy.dtype(datatype)
    if numpy.issubdtype(dt, numpy.integer):
        if dt not in (numpy.dtype(numpy.int32), numpy.dtype(numpy.int64)):
            raise NotImplementedError(f"{dt} is not a charmnumeric dtype (int32/int64 only)")
        lower = 0 if lower is None else int(lower)
        upper = 1000003 if upper is None else int(upper)
        return U.rand_int(shape, lower, upper, seed=5, dtype=dt)
    if numpy.issubdtype(dt, numpy.floating):
        if dt not in (numpy.dtype(numpy.float32), numpy.dtype(numpy.float64)):
            raise NotImplementedError(f"{dt} is not a charmnumeric dtype")
        return U.uniform(shape, seed=5, dtype=dt)
    raise NotImplementedError("UNKNOWN/unsupported type " + str(dt))


def index_along(shape, axis, dtype):
    return U.index_grid(shape, axis, dtype=dtype)


def run_sort(shape, axis, argsort, datatype, lower, upper):
    a = make_input(shape, datatype, lower, upper)
    idx = index_along(shape, axis % len(shape), a.dtype) if argsort else None
    a_sorted, perm = U.oddeven_sort(a, axis=axis, index=idx)
    return a, (perm if argsort else a_sorted)


def main(chk):
    def tuple_of_ints(arg):
        return tuple(map(int, arg.strip("()").split(",")))

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", dest="check", action="store_true")
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("-s", "--shape", type=tuple_of_ints, default=(256,), dest="shape")
    parser.add_argument("-d", "--datatype", type=str, default="float64", dest="datatype")
    parser.add_argument("-l", "--lower", type=int, default=None, dest="lower")
    parser.add_argument("-u", "--upper", type=int, default=None, dest="upper")
    parser.add_argument("-a", "--axis", type=int, default=-1, dest="axis")
    parser.add_argument("--argsort", dest="argsort", action="store_true")
    args = parser.parse_args()
    shape = tuple(args.shape)

    interface = get_interface()
    t0 = time.perf_counter()
    a, out = run_sort(shape, args.axis, args.argsort, args.datatype, args.lower, args.upper)
    out_h = out.get(interface)
    if args.timing:
        print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    a_h = a.get(interface)
    print("Checking result...")
    if not args.argsort:
        chk.close("sorted", out_h, numpy.sort(a_h, args.axis), rtol=0, atol=0)
    else:
        perm = numpy.rint(out_h).astype(numpy.int64)
        chk.true("argsort indices are integral", bool(numpy.all(perm == out_h)))
        chk.true(
            "argsort is a permutation",
            bool(numpy.all(numpy.sort(perm, axis=args.axis) == numpy.sort(numpy.argsort(a_h, args.axis), axis=args.axis))),
        )
        gathered = numpy.take_along_axis(a_h, perm, axis=args.axis)
        chk.close("a[argsort(a)] == sort(a)", gathered, numpy.sort(a_h, args.axis), rtol=0, atol=0)


if __name__ == "__main__":
    U.run("sort", main)
