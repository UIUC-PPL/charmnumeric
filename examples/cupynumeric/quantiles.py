#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/quantiles.py.

Verdict: port with workarounds (native sort network built from native
comparisons and where). np.quantile is a sort followed by interpolation at
order-statistic positions. The positions depend only on n and q, not on
the data, so they are computed on the host as plain Python scalars. The
data path is native:
    s = oddeven_sort(a, axis)                 (see sort.py / _cnum_util.py)
    out[k] = s[lo] + t * (s[hi] - s[lo])      (size-1 views, native)
Supported methods: linear (default), lower, higher, nearest, midpoint.
Supported layouts: 1-D (axis None/0) and 2-D with axis 0 or 1. axis=None on
N-D needs reshape (missing). For axis=1 the result is produced as
(rows, nq). NumPy gives (nq, rows), and with no transpose op the check
compares against the transpose.
Integer inputs are not supported: the interpolated output must be float and
charmnumeric has no astype. The default dtype is float64 (the original
default uint32 is not a charmnumeric dtype).
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


def quantile_native(a, q, axis, method):
    if a.ndim == 1 or axis is None:
        if a.ndim != 1:
            raise NotImplementedError("axis=None on N-D input needs reshape (not in charmnumeric)")
        s, _ = U.oddeven_sort(a, axis=0)
        out = cnp.zeros((len(q),), dtype=a.dtype)
        for k, (lo, hi, t) in enumerate(positions(a.shape[0], q, method)):
            slo, shi = s[lo : lo + 1], s[hi : hi + 1]
            out[k : k + 1] = slo + t * (shi - slo) if t else slo
        return out
    if a.ndim != 2:
        raise NotImplementedError("only 1-D and 2-D inputs are ported")
    axis = axis % 2
    s, _ = U.oddeven_sort(a, axis=axis)
    n = a.shape[axis]
    other = a.shape[1 - axis]
    pos = positions(n, q, method)
    if axis == 0:
        out = cnp.zeros((len(q), other), dtype=a.dtype)
        for k, (lo, hi, t) in enumerate(pos):
            slo, shi = s[lo : lo + 1, :], s[hi : hi + 1, :]
            out[k : k + 1, :] = slo + t * (shi - slo) if t else slo
    else:
        out = cnp.zeros((other, len(q)), dtype=a.dtype)  # transposed w.r.t. numpy
        for k, (lo, hi, t) in enumerate(pos):
            slo, shi = s[:, lo : lo + 1], s[:, hi : hi + 1]
            out[:, k : k + 1] = slo + t * (shi - slo) if t else slo
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
    if dt not in (numpy.dtype(numpy.float32), numpy.dtype(numpy.float64)):
        raise NotImplementedError(f"{dt}: only float32/float64 are ported (no astype for int->float)")

    q = numpy.array([0.0, 0.37, 0.42, 0.5, 0.67, 0.83, 0.99, 1.0])
    interface = get_interface()
    a = U.uniform(shape, seed=1729, dtype=dt)
    t0 = time.perf_counter()
    q_out = quantile_native(a, q, args.axis, args.method)
    got = q_out.get(interface)
    if args.timing:
        print("Elapsed Time: " + str((time.perf_counter() - t0) * 1e3) + " ms (incl. sync)")

    arr = a.get(interface)
    np_q_out = numpy.quantile(arr, q, axis=args.axis, method=args.method)
    if args.axis is not None and len(shape) == 2 and args.axis % 2 == 1:
        np_q_out = np_q_out.T
    print("Checking result...")
    eps = 1.0e-8 if dt == numpy.float64 else 1e-6
    chk.close("quantiles", got, np_q_out, rtol=0, atol=eps)


if __name__ == "__main__":
    U.run("quantiles", main)
