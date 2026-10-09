#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/linreg.py (linear regression, GD).

Verdict: port with workarounds (fully native). Changes:
  * error.dot(features) is a 1-D x 2-D vecmat, which charmnumeric rejects
    ("matmul requires a 2D matrix, got 1D") and which would also need a
    transpose. The features are input data, so the port also generates
    featuresT natively (same closed form, permuted indices) and computes
    featuresT @ error, a matvec. A native vecmat or transpose would remove
    this.
  * np.hstack((intercept, features)) becomes slice assignment into a
    preallocated (N, F+1) array.
  * np.sum(np.power(error, 2)) is cnp.sum(cnp.power(error, 2)) (native
    power and native full reduction).
  * np.random.randn/random are replaced by the native generators.
  * --precision 16 is not supported.
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


def initialize(N, F, T):
    # random inputs, generated natively; xT is x.T (same values, permuted layout)
    x = U.randn((N, F), seed=1, dtype=T)
    xT = U.randn((N, F), seed=1, dtype=T, perm=(1, 0))
    y = U.uniform((N,), seed=2, dtype=T)
    return x, xT, y


def run_linear_regression(interface, N, F, T, I, warmup, S, B):  # noqa: E741
    print("Running linear regression...")
    print("Number of data points: " + str(N))
    print("Number of features: " + str(F))
    print("Number of iterations: " + str(I))

    learning_rate = 1e-5
    features, featuresT, target = initialize(N, F, T)
    if B:
        # np.hstack((intercept, features)) and its transpose
        f2 = cnp.zeros((N, F + 1), dtype=T)
        f2[:, 0:1] = 1.0
        f2[:, 1:] = features
        f2T = cnp.zeros((F + 1, N), dtype=T)
        f2T[0:1, :] = 1.0
        f2T[1:, :] = featuresT
        features, featuresT = f2, f2T
    weights = cnp.zeros((features.shape[1],), dtype=T)

    errors = []
    for step in range(-warmup, I):
        scores = cnp.dot(features, weights)
        error = scores - target
        gradient = -(1.0 / len(features)) * featuresT.dot(error)  # == error.dot(features)
        weights += learning_rate * gradient
        if step >= 0 and step % S == 0:
            e = U.to_float(cnp.sum(cnp.power(error, 2)), interface)
            errors.append(e)
            print("Error of step " + str(step) + ": " + str(e))
    return features, featuresT, target, weights, errors


def reference(features, target, T, I, warmup, S):  # noqa: E741
    learning_rate = 1e-5
    weights = numpy.zeros(features.shape[1], dtype=T)
    errors = []
    for step in range(-warmup, I):
        scores = numpy.dot(features, weights)
        error = scores - target
        gradient = -(1.0 / len(features)) * error.dot(features)
        weights += learning_rate * gradient
        if step >= 0 and step % S == 0:
            errors.append(numpy.sum(numpy.power(error, 2)))
    return weights, errors


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--intercept", dest="B", action="store_true")
    parser.add_argument("-f", "--features", type=int, default=32, dest="F")
    parser.add_argument("-i", "--iterations", type=int, default=20, dest="I")
    parser.add_argument("-w", "--warmup", type=int, default=2, dest="warmup")
    parser.add_argument("-n", "--num", type=int, default=1, dest="N",
                        help="number of data points in thousands")
    parser.add_argument("--precision", type=int, default=32, dest="P")
    parser.add_argument("-s", "--sample", type=int, default=10, dest="S")
    args = parser.parse_args()
    T = {32: numpy.float32, 64: numpy.float64}.get(args.P)
    if T is None:
        raise TypeError("Precision must be one of 32 or 64 (no float16 in charmnumeric)")

    interface = get_interface()
    t0 = time.perf_counter()
    features, featuresT, target, weights, errors = run_linear_regression(
        interface, args.N * 1000, args.F, T, args.I, args.warmup, args.S, args.B
    )
    w = weights.get(interface)
    print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    Fh, FTh, yh = features.get(interface), featuresT.get(interface), target.get(interface)
    chk.close("generated featuresT == features.T", FTh, Fh.T, rtol=0, atol=0)
    w_ref, err_ref = reference(Fh, yh, T, args.I, args.warmup, args.S)
    rtol = 1e-4 if T == numpy.float32 else 1e-10
    chk.close("weights", w, w_ref, rtol=rtol, atol=rtol * numpy.abs(w_ref).max())
    chk.close("sampled errors", errors, err_ref, rtol=rtol, atol=0)


if __name__ == "__main__":
    U.run("linreg", main)
