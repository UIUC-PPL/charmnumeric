#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/sort.py (sort / argsort along an axis).

Verdict: direct. cnp.sort / cnp.argsort are native (SORT/ARGSORT executor).
The backend gathers the array on one chare, sorts every line along the axis
(std::sort on CPU, Kokkos::sort on GPU) and scatters the result back:
correct for any axis of 1-D..3-D arrays, but not scalable (no distributed
sample sort yet). NaNs sort last and ties keep their order (NumPy's
kind='stable'), so argsort is compared exactly with numpy.argsort(kind='stable').
Other changes:
  * uint32 (the original default) is not a charmnumeric dtype. The default
    here is float64. Use -d int64/int32 for integer keys (generated
    natively in [lower, upper)). complex is unsupported.
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


def run_sort(shape, axis, argsort, datatype, lower, upper):
    a = make_input(shape, datatype, lower, upper)
    out = cnp.argsort(a, axis=axis) if argsort else cnp.sort(a, axis=axis)
    return a, out


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
        perm = numpy.asarray(out_h)
        chk.true("argsort dtype is int64", perm.dtype == numpy.int64, str(perm.dtype))
        chk.true(
            "argsort == numpy.argsort(kind='stable')",
            bool(numpy.array_equal(perm, numpy.argsort(a_h, args.axis, kind="stable"))),
        )
        gathered = numpy.take_along_axis(a_h, perm, axis=args.axis)
        chk.close("a[argsort(a)] == sort(a)", gathered, numpy.sort(a_h, args.axis), rtol=0, atol=0)


if __name__ == "__main__":
    U.run("sort", main)
