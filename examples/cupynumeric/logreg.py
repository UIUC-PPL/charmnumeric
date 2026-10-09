#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/logreg.py (logistic regression, GD).

Verdict: port with workarounds (fully native). Changes:
  * sigmoid is 1/(1+exp(-x)) (native exp), as in the original.
  * np.dot(error, features) is a vecmat. It is computed as
    featuresT @ error, where featuresT is generated natively from the same
    closed form as features (see linreg.py).
  * np.log and np.floor are native. np.sum is the native full reduction
    (cnp.sum).
  * np.hstack becomes slice assignment. --precision 16 is not supported.
Backend ops that would make this a direct port: a vecmat/transpose, sum.
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


def initialize(N, F, T):
    x = U.randn((N, F), seed=1, dtype=T)
    xT = U.randn((N, F), seed=1, dtype=T, perm=(1, 0))  # == x.T
    # Give the results either a 0 or 1 value
    y = cnp.floor(2.0 * U.uniform((N,), seed=2, dtype=T))
    return x, xT, y


def sigmoid(x):
    return 1.0 / (1.0 + cnp.exp(-x))


def log_likelihood(features, target, weights):
    scores = cnp.dot(features, weights)
    # np.sum(target*scores - np.log(1 + np.exp(scores)))
    return cnp.sum(target * scores - cnp.log(1.0 + cnp.exp(scores)))


def run_logistic_regression(interface, N, F, T, I, warmup, S, B):  # noqa: E741
    print("Running logistic regression...")
    print("Number of data points: " + str(N))
    print("Number of features: " + str(F))
    print("Number of iterations: " + str(I))

    learning_rate = 1e-5
    features, featuresT, target = initialize(N, F, T)
    if B:
        f2 = cnp.zeros((N, F + 1), dtype=T)
        f2[:, 0:1] = 1.0
        f2[:, 1:] = features
        f2T = cnp.zeros((F + 1, N), dtype=T)
        f2T[0:1, :] = 1.0
        f2T[1:, :] = featuresT
        features, featuresT = f2, f2T
    weights = cnp.zeros((features.shape[1],), dtype=T)

    lls = []
    for step in range(-warmup, I):
        scores = cnp.dot(features, weights)
        predictions = sigmoid(scores)
        error = target - predictions
        gradient = featuresT.dot(error)  # == np.dot(error, features)
        weights += learning_rate * gradient
        if step >= 0 and step % S == 0:
            ll = U.to_float(log_likelihood(features, target, weights), interface)
            lls.append(ll)
            print("Log Likelihood of step " + str(step) + ": " + str(ll))
    return features, featuresT, target, weights, lls


def reference(features, target, T, I, warmup, S):  # noqa: E741
    np = numpy
    learning_rate = 1e-5
    weights = np.zeros(features.shape[1], dtype=T)
    lls = []
    for step in range(-warmup, I):
        scores = np.dot(features, weights)
        predictions = 1.0 / (1.0 + np.exp(-scores))
        error = target - predictions
        gradient = np.dot(error, features)
        weights += learning_rate * gradient
        if step >= 0 and step % S == 0:
            s = np.dot(features, weights)
            lls.append(np.sum(target * s - np.log(1.0 + np.exp(s))))
    return weights, lls


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
    features, featuresT, target, weights, lls = run_logistic_regression(
        interface, args.N * 1000, args.F, T, args.I, args.warmup, args.S, args.B
    )
    w = weights.get(interface)
    print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    # the original's assertion, with the native sum
    chk.true("no NaN in weights (native sum)",
             not math.isnan(U.to_float(cnp.sum(weights), interface)))
    Fh, FTh, yh = features.get(interface), featuresT.get(interface), target.get(interface)
    chk.close("generated featuresT == features.T", FTh, Fh.T, rtol=0, atol=0)
    chk.true("target is exactly 0/1", bool(numpy.all((yh == 0) | (yh == 1))))
    w_ref, ll_ref = reference(Fh, yh, T, args.I, args.warmup, args.S)
    rtol = 1e-4 if T == numpy.float32 else 1e-10
    chk.close("weights", w, w_ref, rtol=rtol, atol=rtol * numpy.abs(w_ref).max())
    chk.close("log likelihood samples", lls, ll_ref, rtol=rtol, atol=0)


if __name__ == "__main__":
    U.run("logreg", main)
