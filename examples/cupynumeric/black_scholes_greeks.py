#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/black_scholes_greeks.py.

Verdict: port with workarounds (fully native). Changes:
  * CALL/PUT are 4-D (N_GREEKS, t, vol, money) in the original. charmnumeric
    stops at 3-D, so the greek axis is folded into the first axis
    (N_GREEKS*t, vol, money), and CALL[g] becomes the 3-D view
    CALL[g*t:(g+1)*t].
  * arange(..., dtype=int).reshape(...) is the native row-major linear index
    U.linear_index(steps), int64 as in the original; the % is native. As in
    the original, int * float promotes natively to float64, and K, T are then
    converted to the working dtype D with a native astype.
  * sqrt, log, absolute and where(d > 0) are native; d > 0 is a 0/1 mask in
    the working dtype. np.sqrt(T) is computed once per call and reused.
Backend ops that would make it direct: 4-D arrays (or reshape).
"""

import argparse
import os
import sys
import time
from enum import IntEnum

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from _cnum_connect import get_interface  # noqa: E402

vol_start = 0.1
vol_step = 0.01
t_start = 0.5
t_step = 1.0 / (365 * 10)
money_start = -0.4
money_step = 0.01

RISKFREE = 0.02
S0 = 100.0
N_GREEKS = 7
RSQRT2PI = 0.39894228040143267793994605993438
A1 = 0.31938153
A2 = -0.356563782
A3 = 1.781477937
A4 = -1.821255978
A5 = 1.330274429


class Greeks(IntEnum):
    PREM = 0
    DELTA = 1
    VEGA = 2
    GAMMA = 3
    VANNA = 4
    VOLGA = 5
    THETA = 6


def initialize(n_vol_steps, n_t_steps, n_money_steps, D):
    steps = (n_t_steps, n_vol_steps, n_money_steps)

    # 4-D (N_GREEKS,)+steps folded into 3-D
    CALL = cnp.zeros((N_GREEKS * n_t_steps, n_vol_steps, n_money_steps), dtype=D)
    PUT = cnp.zeros((N_GREEKS * n_t_steps, n_vol_steps, n_money_steps), dtype=D)
    S = cnp.full(steps, S0, dtype=D)
    # temp_arr = arange(nv*nt*nm, dtype=int).reshape(steps): the row-major
    # linear index (int64). int * float promotes natively to float64, then astype(D).
    temp_arr = U.linear_index(steps, dtype=cnp.int64)
    k_temp = (temp_arr % n_money_steps) * money_step
    K = ((k_temp + (1 + money_start)) * S0).astype(D)
    t_temp = (temp_arr % (n_vol_steps * n_money_steps)) * vol_step
    T = (t_temp + t_start).astype(D)
    R = 0.02
    V = cnp.full(steps, vol_start, dtype=D)
    for i in range(n_vol_steps):
        V[:, i, :] += i * vol_step
    return CALL, PUT, S, K, T, R, V


def normCDF(d):
    K = 1.0 / (1.0 + 0.2316419 * cnp.absolute(d))
    cnd = RSQRT2PI * cnp.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    return cnp.where(d > 0, 1.0 - cnd, cnd)


def normPDF(d):
    return RSQRT2PI * cnp.exp(-0.5 * d * d)


def black_scholes(out, S, K, R, T, V, CP, greek):
    sqrt_T = cnp.sqrt(T)
    stdev = V * sqrt_T
    df = cnp.exp(-R * T)
    d1 = (cnp.log(S / K) + (R + 0.5 * V * V) * T) / stdev
    d2 = d1 - stdev
    nd1 = normCDF(CP * d1)
    nd2 = normCDF(CP * d2)

    if greek == Greeks.PREM:
        out[...] = CP * (S * nd1 - K * df * nd2)
    elif greek == Greeks.DELTA:
        out[...] = CP * nd1
    elif greek == Greeks.VEGA:
        out[...] = S * sqrt_T * normPDF(d1)
    elif greek == Greeks.GAMMA:
        out[...] = normPDF(d1) / (S * V * sqrt_T)
    elif greek == Greeks.VANNA:
        out[...] = -d2 * normPDF(d1) / V
    elif greek == Greeks.VOLGA:
        out[...] = S * sqrt_T * d1 * d2 * normPDF(d1) / V
    elif greek == Greeks.THETA:
        out[...] = -(0.5 * S * V / sqrt_T * normPDF(d1) + CP * R * df * K * nd2)
    else:
        raise RuntimeError("Wrong greek name is passed")


# ---- plain NumPy reference (verbatim math from the original) ----------------
def black_scholes_ref(np, out, S, K, R, T, V, CP, greek):
    def normCDF_r(d):
        K_ = 1.0 / (1.0 + 0.2316419 * np.absolute(d))
        cnd = RSQRT2PI * np.exp(-0.5 * d * d) * (K_ * (A1 + K_ * (A2 + K_ * (A3 + K_ * (A4 + K_ * A5)))))
        return np.where(d > 0, 1.0 - cnd, cnd)

    def normPDF_r(d):
        return RSQRT2PI * np.exp(-0.5 * d * d)

    stdev = V * np.sqrt(T)
    df = np.exp(-R * T)
    d1 = (np.log(S / K) + (R + 0.5 * V * V) * T) / stdev
    d2 = d1 - stdev
    nd1 = normCDF_r(CP * d1)
    nd2 = normCDF_r(CP * d2)
    if greek == Greeks.PREM:
        out[...] = CP * (S * nd1 - K * df * nd2)
    elif greek == Greeks.DELTA:
        out[...] = CP * nd1
    elif greek == Greeks.VEGA:
        out[...] = S * np.sqrt(T) * normPDF_r(d1)
    elif greek == Greeks.GAMMA:
        out[...] = normPDF_r(d1) / (S * V * np.sqrt(T))
    elif greek == Greeks.VANNA:
        out[...] = -d2 * normPDF_r(d1) / V
    elif greek == Greeks.VOLGA:
        out[...] = S * np.sqrt(T) * d1 * d2 * normPDF_r(d1) / V
    elif greek == Greeks.THETA:
        out[...] = -(0.5 * S * V / np.sqrt(T) * normPDF_r(d1) + CP * R * df * K * nd2)


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vol-steps", type=int, default=8, dest="n_vol_steps")
    parser.add_argument("-t", "--time-steps", type=int, default=16, dest="n_time_steps")
    parser.add_argument("-m", "--money-steps", type=int, default=12, dest="n_money_steps")
    parser.add_argument("-i", "--iters", type=int, default=1, dest="iters")
    parser.add_argument("-w", "--warmup", type=int, default=1, dest="warmup")
    args = parser.parse_args()
    nv, nt, nm = args.n_vol_steps, args.n_time_steps, args.n_money_steps
    D = numpy.float32

    interface = get_interface()
    print("Start black_scholes")
    CALL, PUT, S, K, T, R, V = initialize(nv, nt, nm, D)
    t0 = time.perf_counter()
    for i in range(-args.warmup, args.iters):
        for g in Greeks:
            sl = slice(g.value * nt, (g.value + 1) * nt)
            black_scholes(CALL[sl], S, K, R, T, V, 1, g)
            black_scholes(PUT[sl], S, K, R, T, V, -1, g)
    call_h = CALL.get(interface).reshape(N_GREEKS, nt, nv, nm)
    put_h = PUT.get(interface).reshape(N_GREEKS, nt, nv, nm)
    print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    Sh, Kh, Th, Vh = (x.get(interface) for x in (S, K, T, V))
    # the initial grids must match the original arange/%/reshape construction
    temp = numpy.arange(nv * nt * nm)
    K0 = ((temp % nm) * money_step).reshape((nt, nv, nm)) + (1 + money_start)
    chk.close("K init", Kh, K0 * S0, rtol=1e-6, atol=1e-5)
    T0 = ((temp % (nv * nm)) * vol_step).reshape((nt, nv, nm)) + t_start
    chk.close("T init", Th, T0, rtol=1e-6, atol=1e-6)
    V0 = numpy.full((nt, nv, nm), vol_start)
    for i in range(nv):
        V0[:, i, :] += i * vol_step
    chk.close("V init", Vh, V0, rtol=1e-6, atol=1e-6)

    CALL_r = numpy.zeros((N_GREEKS, nt, nv, nm), dtype=D)
    PUT_r = numpy.zeros((N_GREEKS, nt, nv, nm), dtype=D)
    for g in Greeks:
        black_scholes_ref(numpy, CALL_r[g.value], Sh, Kh, R, Th, Vh, 1, g)
        black_scholes_ref(numpy, PUT_r[g.value], Sh, Kh, R, Th, Vh, -1, g)
    for g in Greeks:
        for name, got, ref in (("CALL", call_h, CALL_r), ("PUT", put_h, PUT_r)):
            r = ref[g.value]
            chk.close(f"{name}[{g.name}]", got[g.value], r, rtol=1e-3, atol=1e-3 * max(1.0, numpy.abs(r).max()))


if __name__ == "__main__":
    U.run("black_scholes_greeks", main)
