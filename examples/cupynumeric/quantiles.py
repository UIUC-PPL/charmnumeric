#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/quantiles.py.

Verdict: direct data path (native sort + native interpolation).
np.quantile is a sort followed by interpolation at order-statistic positions.
The positions depend only on n and q, not on the data, so they are computed
on the host as plain Python scalars. The data path is native:
    s = cnp.sort(a, axis)                     (SORT executor)
    out[k] = s[lo] + t * (s[hi] - s[lo])      (size-1 views, native)
Integer inputs are sorted as integers and converted with a native astype to
float64 before the interpolation, as NumPy does.
Supported methods: linear (default), lower, higher, nearest, midpoint.
Supported layouts: 1-D (axis None/0) and 2-D with axis 0 or 1. axis=None on
N-D needs reshape (missing). For axis=1 the interpolation fills a
(rows, nq) array, and the native transpose returns NumPy's (nq, rows). The original default dtype uint32 is not a
charmnumeric dtype; the default here is float64 (-d int32/int64 also work).
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


def positions(n, q, method):
    """Host-side order-statistic positions (lo, hi, t) for each q (data independent)."""
    out = []
    for qq in q:
        h = (n - 1) * qq
        lo = int(math.floor(h))
        hi = min(lo + 1, n - 1)
        t = h - lo
        if method == "linear":
            pass
        elif method == "lower":
            hi, t = lo, 0.0
        elif method == "higher":
            lo = hi = int(math.ceil(h))
            t = 0.0
        elif method == "nearest":
            # numpy rounds half to even
            k = int(numpy.around(h))
            lo = hi = k
            t = 0.0
        elif method == "midpoint":
            t = 0.5 if hi != lo and h != lo else 0.0
        else:
            raise NotImplementedError(f"method {method!r} not ported")
        out.append((lo, hi, t))
    return out


def _sorted_float(a, axis):
    """Native sort along axis; integer data is converted to float64 afterwards."""
    s = cnp.sort(a, axis=axis)
    if numpy.issubdtype(numpy.dtype(s.dtype), numpy.integer):
        s = s.astype(numpy.float64)
    return s


def quantile_native(a, q, axis, method):
    if a.ndim == 1 or axis is None:
        if a.ndim != 1:
            raise NotImplementedError("axis=None on N-D input needs reshape (not in charmnumeric)")
        s = _sorted_float(a, 0)
        out = cnp.zeros((len(q),), dtype=s.dtype)
        for k, (lo, hi, t) in enumerate(positions(a.shape[0], q, method)):
            slo, shi = s[lo : lo + 1], s[hi : hi + 1]
            out[k : k + 1] = slo + t * (shi - slo) if t else slo
        return out
    if a.ndim != 2:
        raise NotImplementedError("only 1-D and 2-D inputs are ported")
    axis = axis % 2
    s = _sorted_float(a, axis)
    n = a.shape[axis]
    other = a.shape[1 - axis]
    pos = positions(n, q, method)
    if axis == 0:
        out = cnp.zeros((len(q), other), dtype=s.dtype)
        for k, (lo, hi, t) in enumerate(pos):
            slo, shi = s[lo : lo + 1, :], s[hi : hi + 1, :]
            out[k : k + 1, :] = slo + t * (shi - slo) if t else slo
    else:
        outT = cnp.zeros((other, len(q)), dtype=s.dtype)
        for k, (lo, hi, t) in enumerate(pos):
            slo, shi = s[:, lo : lo + 1], s[:, hi : hi + 1]
            outT[:, k : k + 1] = slo + t * (shi - slo) if t else slo
        out = outT.T  # native transpose: NumPy's (nq, rows)
    return out


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
    parser.add_argument("-a", "--axis", type=int, default=None, dest="axis")
    parser.add_argument("-m", "--method", type=str, default="linear", dest="method")
    args = parser.parse_args()
    shape = tuple(args.shape)
    dt = numpy.dtype(args.datatype)
    if dt not in (numpy.dtype(t) for t in (numpy.float32, numpy.float64, numpy.int32, numpy.int64)):
        raise NotImplementedError(f"{dt} is not a charmnumeric dtype")

    q = numpy.array([0.0, 0.37, 0.42, 0.5, 0.67, 0.83, 0.99, 1.0])
    interface = get_interface()
    if numpy.issubdtype(dt, numpy.integer):
        lower = 0 if args.lower is None else args.lower
        upper = 1000 if args.upper is None else args.upper
        a = U.rand_int(shape, lower, upper, seed=1729, dtype=dt)
    else:
        a = U.uniform(shape, seed=1729, dtype=dt)
    t0 = time.perf_counter()
    q_out = quantile_native(a, q, args.axis, args.method)
    got = q_out.get(interface)
    if args.timing:
        print("Elapsed Time: " + str((time.perf_counter() - t0) * 1e3) + " ms (incl. sync)")

    arr = a.get(interface)
    np_q_out = numpy.quantile(arr, q, axis=args.axis, method=args.method)
    print("Checking result...")
    eps = 1e-6 if dt == numpy.float32 else 1.0e-8
    chk.close("quantiles", got, np_q_out, rtol=0, atol=eps)


if __name__ == "__main__":
    U.run("quantiles", main)
