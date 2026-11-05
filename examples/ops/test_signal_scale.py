#!/usr/bin/env python
"""Multi-chare convolve and bincount: results spread over many chares per PE.

convolve: the filter goes to one chare per PE and FFTs run in overlap-save blocks
(set CNUM_CONV_BLOCK on the server, e.g. 48, to force several blocks per chare).
bincount: per-PE histogram merge, slices sent to the output chares.
Env: SCALE_SECTIONS=conv,bin runs a subset.
"""

import os

import numpy as np

from _ops_common import Checker, connect
from test_signal import ref_convolve, tol

import charmnumeric as cnp

_SECTIONS = os.environ.get("SCALE_SECTIONS")


def on(name):
    return _SECTIONS is None or name in _SECTIONS.split(",")


def main():
    iface = connect()
    chk = Checker("test_signal_scale", iface)
    rng = cnp.random.default_rng(11)

    def conv_case(label, ashape, vshape, modes, methods, dtype=np.float64, a_key=None):
        A = rng.random(ashape, dtype=dtype)
        V = rng.random(vshape, dtype=dtype)
        a = A if a_key is None else A[a_key]
        ha, hv = chk.fetch(A), chk.fetch(V)
        if a_key is not None:
            ha = ha[a_key]
        for mode in modes:
            if ha.ndim == 1:
                ref = np.convolve(ha.astype(np.float64), hv.astype(np.float64), mode)
            else:
                ref = ref_convolve(ha, hv, mode)
            for method in methods:
                got = cnp.convolve(a, V, mode=mode, method=method)
                chk.check(f"{label} {mode} {method} {np.dtype(dtype).name}", got, ref,
                          dtype=dtype, **tol(dtype, ref))

    if on("conv"):
        print("convolve: 1-D, many chares")
        conv_case("1d (1200007)*(33)", (1200007,), (33,), ("same", "valid"), ("direct", "fft"),
                  dtype=np.float32)
        conv_case("1d (1200007)*(5001)", (1200007,), (5001,), ("same",), ("fft",))
        conv_case("1d filter>tile (700001)*(600001)", (700001,), (600001,), ("full",), ("fft",))
        # float32 FFT rounding scales with the peak output; full mode's ends are ~a[0] * v[0]
        A = rng.random((700001,), dtype=np.float32)
        V = rng.random((600001,), dtype=np.float32)
        ha, hv = chk.fetch(A).astype(np.float64), chk.fetch(V).astype(np.float64)
        n = ha.size + hv.size - 1
        ref = np.fft.irfft(np.fft.rfft(ha, 1 << 21) * np.fft.rfft(hv, 1 << 21), 1 << 21)[:n]
        chk.check("1d filter>tile (700001)*(600001) full fft float32",
                  cnp.convolve(A, V, mode="full", method="fft"), ref, dtype=np.float32,
                  **tol(np.float32, ref))
        conv_case("1d view", (900001,), (17,), ("same",), ("direct", "fft"),
                  a_key=slice(5, None, 2))

        print("convolve: 2-D, many chares")
        conv_case("2d (1300,1100)*(31,17)", (1300, 1100), (31, 17), ("same", "valid"),
                  ("direct", "fft"), dtype=np.float32)
        conv_case("2d filter>tile (1100,700)*(900,9)", (1100, 700), (900, 9), ("same",),
                  ("fft",))

        print("convolve: 3-D, many chares")
        conv_case("3d (150,140,130)*(9,7,5)", (150, 140, 130), (9, 7, 5), ("same", "full"),
                  ("direct", "fft"), dtype=np.float32)
        conv_case("3d filter>tile (140,70,66)*(130,5,3)", (140, 70, 66), (130, 5, 3), ("same",),
                  ("fft",))

        print("convolve: integers and repeats")
        A = rng.integers(-9, 10, size=(1300, 600), dtype=np.int64)
        V = rng.integers(-3, 4, size=(5, 4), dtype=np.int64)
        ha, hv = chk.fetch(A), chk.fetch(V)
        chk.check("int64 (1300,600)*(5,4) same", cnp.convolve(A, V, mode="same"),
                  ref_convolve(ha, hv, "same", exact_int=True), dtype=np.int64)
        # the same shapes again: cached cuFFT plans, new shared filters
        img = rng.random((150, 140, 130), dtype=np.float32)
        psf = rng.random((6, 6, 6), dtype=np.float32)
        hi, hp = chk.fetch(img), chk.fetch(psf)
        est = cnp.full(img.shape, 0.5, dtype=np.float32)
        h_est = np.full(img.shape, 0.5)
        pm = cnp.flip(psf)
        for _ in range(2):
            conv = cnp.convolve(est, psf, mode="same", method="fft")
            est = est * cnp.convolve(img / conv, pm, mode="same", method="fft")
            hconv = ref_convolve(h_est, hp.astype(np.float64), "same")
            h_est = h_est * ref_convolve(hi / hconv, np.flip(hp).astype(np.float64), "same")
        chk.check("richardson-lucy x2 (150,140,130)", est, h_est, rtol=2e-4, atol=1e-5)

    if on("bin"):
        print("bincount: many x chares")
        n = 2_500_003
        for length, hi_val in ((7, 7), (300_000, 300_000), (3_000_000, 3_000_000),
                               (3_000_000, 1000)):
            X = rng.integers(0, hi_val, size=(n,), dtype=np.int64)
            hx = chk.fetch(X)
            ref = np.bincount(hx, minlength=length)[:length]
            chk.check(f"bincount n={n} values<{hi_val} length={length}",
                      cnp.bincount(X, minlength=length), ref, dtype=np.int64)
        X = rng.integers(-5, 400_000, size=(n,), dtype=np.int32)
        hx = chk.fetch(X)
        keep = (hx >= 0) & (hx < 350_000)
        chk.check("bincount out-of-range values ignored",
                  cnp.bincount(X, minlength=350_000),
                  np.bincount(hx[keep], minlength=350_000), dtype=np.int64)
        X = rng.integers(0, 200_000, size=(n,), dtype=np.int64)
        W = rng.random((n,))
        hx, hw = chk.fetch(X), chk.fetch(W)
        chk.check("bincount weights length=200000", cnp.bincount(X, weights=W, minlength=200_000),
                  np.bincount(hx, weights=hw, minlength=200_000), dtype=np.float64,
                  rtol=1e-12, atol=1e-9)
        chk.check("bincount view", cnp.bincount(X[7:n:3], minlength=200_000),
                  np.bincount(hx[7:n:3], minlength=200_000), dtype=np.int64)
        chk.check("bincount weights views", cnp.bincount(X[1:], weights=W[:-1], minlength=200_000),
                  np.bincount(hx[1:], weights=hw[:-1], minlength=200_000), dtype=np.float64,
                  rtol=1e-12, atol=1e-9)
        chk.check("bincount interface", cnp.bincount(X, interface=iface), np.bincount(hx),
                  dtype=np.int64)
        S = rng.integers(0, 10, size=(1000,), dtype=np.int64)
        hs = chk.fetch(S)
        chk.check("bincount small x, large length", cnp.bincount(S, minlength=2_000_000),
                  np.bincount(hs, minlength=2_000_000), dtype=np.int64)

    chk.finish()


if __name__ == "__main__":
    main()
