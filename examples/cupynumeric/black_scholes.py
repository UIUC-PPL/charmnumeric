#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/black_scholes.py.

Verdict: direct port (fully native). sqrt, log, absolute and
where(d > 0, ...) are native charmnumeric ops; d > 0 is a 0/1 mask in the
working dtype (charmnumeric has no bool dtype), which where() accepts.
np.random.random(N).astype(D) is cnp.random.random(N, dtype=D) (native).
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

A1 = 0.31938153
A2 = -0.356563782
A3 = 1.781477937
A4 = -1.821255978
A5 = 1.330274429
RSQRT2PI = 0.39894228040143267793994605993438


def generate_random(N, min, max, D):
    diff = float(D(max) - D(min))
    rands = cnp.random.random(N, dtype=D)
    rands = rands * diff
    rands = rands + float(D(min))
    return rands


def initialize(N, D):
    S = generate_random(N, 5, 30, D)
    X = generate_random(N, 1, 100, D)
    T = generate_random(N, 0.25, 10, D)
    R = 0.02
    V = 0.3
    return S, X, T, R, V


def cnd(d):
    K = 1.0 / (1.0 + 0.2316419 * cnp.absolute(d))
    cnd = (
        RSQRT2PI
        * cnp.exp(-0.5 * d * d)
        * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    )
    return cnp.where(d > 0, 1.0 - cnd, cnd)


def black_scholes(S, X, T, R, V):
    sqrt_t = cnp.sqrt(T)
    d1 = cnp.log(S / X) + (R + 0.5 * V * V) * T / (V * sqrt_t)
    d2 = d1 - V * sqrt_t
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)
    exp_rt = cnp.exp(-R * T)
    call_result = S * cnd_d1 - X * exp_rt * cnd_d2
    put_result = X * exp_rt * (1.0 - cnd_d2) - S * (1.0 - cnd_d1)
    return call_result, put_result


# ---- plain NumPy reference (verbatim from the original) ---------------------
def cnd_ref(np, d):
    K = 1.0 / (1.0 + 0.2316419 * np.absolute(d))
    cnd = RSQRT2PI * np.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    return np.where(d > 0, 1.0 - cnd, cnd)


def black_scholes_ref(np, S, X, T, R, V):
    sqrt_t = np.sqrt(T)
    d1 = np.log(S / X) + (R + 0.5 * V * V) * T / (V * sqrt_t)
    d2 = d1 - V * sqrt_t
    cnd_d1 = cnd_ref(np, d1)
    cnd_d2 = cnd_ref(np, d2)
    exp_rt = np.exp(-R * T)
    call_result = S * cnd_d1 - X * exp_rt * cnd_d2
    put_result = X * exp_rt * (1.0 - cnd_d2) - S * (1.0 - cnd_d1)
    return call_result, put_result


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=10, dest="N",
                        help="number of options to price in thousands")
    parser.add_argument("--precision", type=int, default=32, dest="P")
    args = parser.parse_args()
    D = {32: numpy.float32, 64: numpy.float64}.get(args.P)
    if D is None:
        raise TypeError("Precision must be one of 32 or 64 (no float16 in charmnumeric)")

    interface = get_interface()
    print("Running black scholes on %dK options..." % args.N)
    N = args.N * 1000
    t0 = time.perf_counter()
    S, X, T, R, V = initialize(N, D)
    call, put = black_scholes(S, X, T, R, V)
    call_h, put_h = call.get(interface), put.get(interface)
    print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    Sh, Xh, Th = S.get(interface), X.get(interface), T.get(interface)
    call_r, put_r = black_scholes_ref(numpy, Sh, Xh, Th, R, V)
    rtol = 2e-4 if D == numpy.float32 else 1e-10
    chk.close("call", call_h, call_r, rtol=rtol, atol=rtol * numpy.abs(call_r).max())
    chk.close("put", put_h, put_r, rtol=rtol, atol=rtol * numpy.abs(put_r).max())


if __name__ == "__main__":
    U.run("black_scholes", main)
