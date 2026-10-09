#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/take_pairs.py.

Verdict: port with workarounds (fully native). Changes:
  * data_array = arange(M*N).reshape(M, N) is built natively, without
    reshape, as i*N + j from cnp.meshgrid(cnp.arange(N), cnp.arange(M)).
  * indices = itertools.combinations(range(N), 2) is a host-side list in the
    original too, so data_array.take(indices, axis=1) is native: cnp.take
    lowers host indices to column-slice copies (contiguous index runs become
    one slice copy each) and writes the (M, P) result for each index column
    into the 3-D result.
  * result.sum() is the native full reduction (int32 input, int64 result as
    in NumPy).
Backend ops that would make it direct: reshape (arange is native now).
"""

import argparse
import itertools
import os
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from _cnum_connect import get_interface  # noqa: E402

I32 = cnp.int32


def take_pairs(M, N):
    # np.arange(M * N, dtype=np.int32).reshape((M, N)), built natively (no reshape)
    jj, ii = cnp.meshgrid(cnp.arange(N, dtype=I32), cnp.arange(M, dtype=I32))
    data_array = ii * N + jj
    indices = numpy.array(list(a for a in itertools.combinations(range(N), 2)))

    # native take with host indices: lowered to column-slice copies
    result = data_array.take(indices, axis=1)  # (M, P, 2)
    value = result.sum()  # native full reduction
    return data_array, indices, result, value


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--num-rows", type=int, default=256, dest="M")
    parser.add_argument("-n", "--num-cols", type=int, default=16, dest="N")
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    args = parser.parse_args()

    interface = get_interface()
    t0 = time.perf_counter()
    data_array, indices, result, value = take_pairs(args.M, args.N)
    v = int(round(U.to_float(value, interface)))
    print(f"result sum: {v}")
    if args.timing:
        print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    ref_data = numpy.arange(args.M * args.N, dtype=numpy.int32).reshape((args.M, args.N))
    ref_result = ref_data.take(indices, axis=1)
    chk.close("data_array", data_array.get(interface), ref_data, rtol=0, atol=0)
    chk.close("take result", result.get(interface), ref_result, rtol=0, atol=0)
    chk.true("result sum", v == int(ref_result.sum(dtype=numpy.int64)), f"({v} vs {ref_result.sum()})")


if __name__ == "__main__":
    U.run("take_pairs", main)
