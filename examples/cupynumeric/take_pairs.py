#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/take_pairs.py.

Verdict: needs a backend feature (take/gather). Workaround:
  * data_array = arange(M*N).reshape(M, N) is built natively, without
    reshape, as i*N + j from cnp.meshgrid(cnp.arange(N), cnp.arange(M)).
  * indices = itertools.combinations(range(N), 2) is a host-side list in the
    original too. take(indices, axis=1) is expressed as two matmuls with 0/1
    selection matrices: R_c = data @ S_c, with S_c[j, p] = 1 iff
    indices[p, c] == j. S_c is built on the host from the index list and uploaded in bulk
    (cnp.array): HOST-FALLBACK (no native gather/scatter from an index array).
  * result[:, :, c] = R_c (2-D into a 3-D view, offset only in the
    singleton dim). result.sum() is the native full reduction (int32 input,
    int64 result as in NumPy).
Backend ops that would make it direct: take/gather along an axis and reshape
(arange is native now).
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
    P = indices.shape[0]

    # HOST-FALLBACK: gather lowered to selection matmuls; S_c built from the host index list
    S = []
    for c in range(2):
        S_c = numpy.zeros((N, P), dtype=numpy.int32)
        S_c[indices[:, c], numpy.arange(P)] = 1
        S.append(cnp.array(S_c))

    result = cnp.zeros((M, P, 2), dtype=I32)  # data_array.take(indices, axis=1)
    for c in range(2):
        result[:, :, c] = data_array @ S[c]
    value = result.sum()
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
