#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/qr.py.

Verdict: direct. a = cnp.random.rand(m, n) and q, r = cnp.linalg.qr(a), as in
the original (native LINALG executor: Householder QR on one chare). The
original's check, allclose(a, matmul(q, r)), runs natively; Q and R are also
compared with numpy.linalg.qr on the fetched input (R up to row signs).
On failure the original's I - Q^T Q diagnostic is printed, computed natively
(eye - q.T @ q, cnp.max). Complex dtypes are not supported.
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
from charmtyles.core import sync  # noqa: E402


def check_result(chk, interface, a, q, r, dtype):
    print("Checking result...")
    rtol, atol = U.tolerances(dtype)
    m, n = a.shape
    # matrix-product rounding grows with the inner dimension
    atol *= max(1, min(m, n)) ** 0.5
    ok = U.allclose(a, cnp.matmul(q, r), interface, rtol=rtol, atol=atol)
    qh, rh, ah = q.get(interface), r.get(interface), a.get(interface)
    qtq = cnp.matmul(q.T, q)  # native transpose + SUMMA
    if not ok:
        qtq_err = cnp.eye(q.shape[1], dtype=q.dtype) - qtq
        q_max = U.to_float(cnp.max(qtq_err), interface)
        print(f"charmnumeric I-qTq max: {q_max}")
    chk.true(f"allclose(a, q @ r) {m}x{n} {dtype} (native)", ok)
    qn, rn = numpy.linalg.qr(ah.astype(numpy.float64))
    k = min(m, n)
    d = numpy.sign(numpy.diag(rh)[:k]) * numpy.sign(numpy.diag(rn)[:k])
    chk.close("R vs numpy (up to row signs)", rh, d[:, None] * rn, rtol=rtol * 10,
              atol=atol * max(1.0, float(numpy.abs(rn).max())))
    chk.close("Q^T Q vs I (native q.T @ q)", qtq.get(interface), numpy.eye(k), rtol=0, atol=atol * 10)


def qr(chk, interface, m, n, dtype, print_timing=False):
    a = cnp.random.rand(m, n, dtype=dtype)
    sync(interface)

    t0 = time.perf_counter()
    q, r = cnp.linalg.qr(a)
    sync(interface)
    total = (time.perf_counter() - t0) * 1e3

    check_result(chk, interface, a, q, r, dtype)

    if print_timing:
        print(f"Elapsed Time: {total} ms")
    return total


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true", help="perform timing")
    parser.add_argument("-m", "--rows", type=int, nargs="+", default=[1000], dest="m",
                        help="number of rows in the matrix")
    parser.add_argument("-n", "--cols", type=int, nargs="+", default=[1000], dest="n",
                        help="number of cols in the matrix")
    parser.add_argument("-d", "--dtype", default="float64", choices=["float32", "float64"],
                        dest="dtype", help="data type")
    parser.add_argument("--check", dest="check", action="store_true",
                        help="compare result to numpy (always done)")
    args = parser.parse_args()

    interface = get_interface()
    for m, n in itertools.product(args.m, args.n):
        qr(chk, interface, m, n, numpy.dtype(args.dtype), args.timing)


if __name__ == "__main__":
    U.run("qr", main)
