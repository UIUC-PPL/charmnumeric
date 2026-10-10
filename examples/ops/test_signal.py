#!/usr/bin/env python
"""Tests for cnp.convolve, cnp.flip/flipud/fliplr and negative-step slicing.

convolve is checked against numpy.convolve (1-D) and an N-D FFT reference
(scipy.signal.convolve semantics). Env: SIG_SECTIONS=conv,flip,neg,chain runs a subset.
"""

import os

import numpy as np

from _ops_common import Checker, connect

import charmnumeric as cnp

_SECTIONS = os.environ.get("SIG_SECTIONS")


def on(name):
    return _SECTIONS is None or name in _SECTIONS.split(",")


def ref_full(a, v):
    """Full N-D linear convolution in float64 via numpy FFTs."""
    shape = tuple(n + f - 1 for n, f in zip(a.shape, v.shape))
    axes = list(range(a.ndim))
    fa = np.fft.rfftn(a.astype(np.float64), shape, axes)
    fv = np.fft.rfftn(v.astype(np.float64), shape, axes)
    return np.fft.irfftn(fa * fv, shape, axes)


def ref_full_int(a, v):
    full = np.zeros(tuple(n + f - 1 for n, f in zip(a.shape, v.shape)), dtype=np.int64)
    for k in np.ndindex(*v.shape):
        full[tuple(slice(k[d], k[d] + a.shape[d]) for d in range(a.ndim))] += (
            int(v[k]) * a.astype(np.int64)
        )
    return full


def ref_convolve(a, v, mode, exact_int=False):
    """scipy.signal.convolve(a, v, mode) (no 1-D swap for 'same')."""
    if mode == "valid" and not all(n >= f for n, f in zip(a.shape, v.shape)):
        a, v = v, a
    full = ref_full_int(a, v) if exact_int else ref_full(a, v)
    if mode == "full":
        return full
    if mode == "same":
        out = a.shape
        st = [(f - 1) // 2 for f in v.shape]
    else:
        out = tuple(n - f + 1 for n, f in zip(a.shape, v.shape))
        st = [f - 1 for f in v.shape]
    return full[tuple(slice(s, s + n) for s, n in zip(st, out))]


def tol(dtype, ref):
    scale = float(np.max(np.abs(ref))) if ref.size else 1.0
    if np.dtype(dtype) == np.float32:
        return dict(rtol=2e-5, atol=2e-5 * scale)
    return dict(rtol=1e-11, atol=1e-11 * scale)


def main():
    iface = connect()
    chk = Checker("test_signal", iface)
    rng = cnp.random.default_rng(7)

    def rand(shape, dtype=np.float64):
        return rng.random(shape, dtype=dtype)

    def conv_case(label, ashape, vshape, modes, methods, dtype=np.float64, a_key=None,
                  v_key=None):
        A = rand(ashape, dtype)
        V = rand(vshape, dtype)
        a = A if a_key is None else A[a_key]
        v = V if v_key is None else V[v_key]
        ha, hv = chk.fetch(A), chk.fetch(V)
        if a_key is not None:
            ha = ha[a_key]
        if v_key is not None:
            hv = hv[v_key]
        for mode in modes:
            if ha.ndim == 1:
                ref = np.convolve(ha.astype(np.float64), hv.astype(np.float64), mode)
            else:
                ref = ref_convolve(ha, hv, mode)
            for method in methods:
                got = cnp.convolve(a, v, mode=mode, method=method)
                chk.check(f"{label} {mode} {method} {np.dtype(dtype).name}", got, ref,
                          dtype=dtype, **tol(dtype, ref))

    allm = ("full", "same", "valid")
    if on("conv"):
        print("convolve: 1-D")
        conv_case("1d (37)*(5)", (37,), (5,), allm, ("direct", "fft"))
        conv_case("1d (37)*(5)", (37,), (5,), allm, ("direct", "fft"), dtype=np.float32)
        conv_case("1d swap (6)*(41)", (6,), (41,), allm, ("direct", "fft"))
        conv_case("1d even (64)*(8)", (64,), (8,), allm, ("direct", "auto"))
        conv_case("1d multichare (300001)*(7)", (300001,), (7,), ("same", "valid"),
                  ("direct", "fft"), dtype=np.float32)
        conv_case("1d filter>tile (300001)*(270001)", (300001,), (270001,), ("same",),
                  ("fft",))

        print("convolve: 2-D")
        conv_case("2d (37,41)*(5,4)", (37, 41), (5, 4), allm, ("direct", "fft"))
        conv_case("2d (37,41)*(5,4)", (37, 41), (5, 4), allm, ("direct", "fft"),
                  dtype=np.float32)
        conv_case("2d (600,7)*(520,3)", (600, 7), (520, 3), allm, ("direct", "fft"))
        conv_case("2d (1030,9)*(3,3)", (1030, 9), (3, 3), ("same",), ("direct", "fft"),
                  dtype=np.float32)
        conv_case("2d valid swap (4,5)*(9,12)", (4, 5), (9, 12), ("valid", "same"),
                  ("direct",))
        conv_case("2d views", (41, 50), (7, 9), allm, ("direct", "fft"),
                  a_key=(slice(1, -1), slice(None, None, 2)), v_key=(slice(2, None), slice(1, 6)))

        print("convolve: 3-D")
        conv_case("3d rl (20,20,20)*(4,4,4)", (20, 20, 20), (4, 4, 4), ("same",),
                  ("direct", "fft", "auto"), dtype=np.float32)
        conv_case("3d (70,66,9)*(5,3,4)", (70, 66, 9), (5, 3, 4), allm, ("direct", "fft"))
        conv_case("3d (70,66,9)*(5,3,4)", (70, 66, 9), (5, 3, 4), ("same",), ("direct", "fft"),
                  dtype=np.float32)
        conv_case("3d filter>tile (70,3,5)*(66,3,3)", (70, 3, 5), (66, 3, 3), ("same", "full"),
                  ("direct", "fft"))
        conv_case("3d auto->fft (30,20,10)*(16,16,16)", (30, 20, 10), (16, 16, 16), ("same",),
                  ("auto",), dtype=np.float32)
        conv_case("3d view", (70, 20, 12), (3, 4, 5), ("same",), ("direct",),
                  a_key=(slice(3, 69, 1), slice(None, None, 3), slice(1, None)))

        print("convolve: integers (exact, direct)")
        for dt in (np.int32, np.int64):
            A = rng.integers(-9, 10, size=(600, 7), dtype=dt)
            V = rng.integers(-3, 4, size=(4, 3), dtype=dt)
            ha, hv = chk.fetch(A), chk.fetch(V)
            for mode in allm:
                ref = ref_convolve(ha, hv, mode, exact_int=True).astype(dt)
                chk.check(f"int {np.dtype(dt).name} (600,7)*(4,3) {mode}",
                          cnp.convolve(A, V, mode=mode, method="fft"), ref, dtype=dt)

        print("convolve: edge cases")
        A = rand((30, 7))
        S = rand((1, 1))
        ha, hs = chk.fetch(A), chk.fetch(S)
        chk.check("size-1 filter", cnp.convolve(A, S, mode="same"), ha * hs[0, 0], rtol=1e-15)
        X = rand((5, 6), np.float32)
        Y = rand((3, 2), np.float64)
        hx, hy = chk.fetch(X), chk.fetch(Y)
        ref = ref_convolve(hx.astype(np.float64), hy, "same")
        chk.check("mixed dtype -> result_type", cnp.convolve(X, Y, mode="same"), ref,
                  dtype=np.float64, **tol(np.float64, ref))
        I = cnp.asarray(np.arange(12, dtype=np.int8).reshape(3, 4) - 5)
        hi = chk.fetch(I)
        chk.check("int8 convolve wraps", cnp.convolve(I, I, mode="full"),
                  ref_convolve(hi.astype(np.int64), hi.astype(np.int64), "full", exact_int=True).astype(np.int8),
                  dtype=np.int8)
        Bm = cnp.asarray(np.array([[True, False, True], [False, False, True]]))
        hb = chk.fetch(Bm)
        chk.check("bool convolve", cnp.convolve(Bm, Bm[:, :2], mode="full"),
                  ref_convolve(hb.astype(np.int64), hb[:, :2].astype(np.int64), "full", exact_int=True) != 0,
                  dtype=np.bool_)
        chk.check("flip bool", cnp.flip(Bm, 1), hb[:, ::-1], dtype=np.bool_)
        F16 = cnp.asarray(np.arange(7, dtype=np.float16))
        chk.check("flip float16", cnp.flip(F16), np.arange(7, dtype=np.float16)[::-1],
                  dtype=np.float16)
        C16 = cnp.asarray(np.arange(5) + 1j * np.arange(5))
        chk.check("flip complex128", C16[::-1], (np.arange(5) + 1j * np.arange(5))[::-1],
                  dtype=np.complex128)
        hl = np.arange(10.0)
        chk.check("host operands", cnp.convolve(hl, [1.0, -1.0, 2.0]),
                  np.convolve(hl, [1.0, -1.0, 2.0]), rtol=1e-14)
        chk.raises("bad mode", ValueError, lambda: cnp.convolve(A, A, mode="nope"))
        chk.raises("bad method", ValueError, lambda: cnp.convolve(A, A, method="nope"))
        chk.raises("ndim mismatch", RuntimeError, lambda: cnp.convolve(A, rand((3,))))
        chk.raises("valid incompatible", ValueError,
                   lambda: cnp.convolve(rand((5, 2)), rand((2, 5)), mode="valid"))

    if on("flip"):
        print("flip")
        cases = [
            ((37,), [None, 0, -1]),
            ((300001,), [None]),
            ((600, 7), [None, 0, 1, (0, 1), (-1,)]),
            ((1030, 3), [0]),
            ((70, 3, 5), [None, 0, 1, 2, (0, 2), (1, 2)]),
            ((5, 70, 66), [(1, 2)]),
        ]
        for shape, axes_list in cases:
            for dt in (np.float64, np.float32):
                X = rand(shape, dt)
                hx = chk.fetch(X)
                for axes in axes_list:
                    chk.check(f"flip {shape} axis={axes} {np.dtype(dt).name}",
                              cnp.flip(X, axes), np.flip(hx, axes), dtype=dt)
        I = rng.integers(-100, 100, size=(600, 7), dtype=np.int32)
        hi = chk.fetch(I)
        chk.check("flip int32", cnp.flip(I), np.flip(hi), dtype=np.int32)
        X = rand((41, 50))
        hx = chk.fetch(X)
        key = (slice(1, -1), slice(None, None, 3))
        chk.check("flip of view", cnp.flip(X[key], 0), np.flip(hx[key], 0))
        chk.check("flipud", cnp.flipud(X), np.flipud(hx))
        chk.check("fliplr", cnp.fliplr(X), np.fliplr(hx))
        chk.raises("fliplr 1-D", ValueError, lambda: cnp.fliplr(rand((5,))))
        chk.raises("flip bad axis", np.exceptions.AxisError, lambda: cnp.flip(X, 2))

    if on("neg"):
        print("negative-step slicing")
        X1 = rand((300001,))
        X2 = rand((600, 7))
        X3 = rand((70, 3, 5))
        h1, h2, h3 = chk.fetch(X1), chk.fetch(X2), chk.fetch(X3)
        for label, X, h, key in [
            ("[::-1]", X1, h1, np.s_[::-1]),
            ("[::-7]", X1, h1, np.s_[::-7]),
            ("[1000:5:-3]", X1, h1, np.s_[1000:5:-3]),
            ("[::-1, ::-1]", X2, h2, np.s_[::-1, ::-1]),
            ("[:, ::-2]", X2, h2, np.s_[:, ::-2]),
            ("[550:10:-1, 1:]", X2, h2, np.s_[550:10:-1, 1:]),
            ("[::-1, :, ::-3]", X3, h3, np.s_[::-1, :, ::-3]),
            ("[..., ::-1]", X3, h3, np.s_[..., ::-1]),
        ]:
            chk.check(f"get {label}", X[key], h[key])
        V = X2[1:-1, :]
        chk.check("view[::-1]", V[::-1], h2[1:-1, :][::-1])
        Y = cnp.zeros((600, 7))
        Y[::-1] = X2
        chk.check("set [::-1] = array", Y, h2[::-1])
        Z = cnp.zeros((600, 7))
        Z[:, ::-2] = np.arange(4.0)
        hz = np.zeros((600, 7))
        hz[:, ::-2] = np.arange(4.0)
        chk.check("set [:, ::-2] = host row", Z, hz)
        Z[10:0:-1, 0] = 5.0
        hz[10:0:-1, 0] = 5.0
        chk.check("set [10:0:-1, 0] = scalar", Z, hz)

    if on("chain"):
        print("chains (one DAG)")
        img = rand((20, 20, 20), np.float32)
        psf = rand((4, 4, 4), np.float32)
        hi, hp = chk.fetch(img), chk.fetch(psf)
        est = cnp.full(img.shape, 0.5, dtype=np.float32)
        h_est = np.full(img.shape, 0.5)
        pm = cnp.flip(psf)
        for _ in range(2):
            conv = cnp.convolve(est, psf, mode="same", method="direct")
            rel = img / conv
            est = est * cnp.convolve(rel, pm, mode="same", method="fft")
            hconv = ref_convolve(h_est, hp.astype(np.float64), "same")
            h_est = h_est * ref_convolve(hi / hconv, np.flip(hp).astype(np.float64), "same")
        chk.check("richardson-lucy x2", est, h_est, rtol=1e-4, atol=1e-5)
        X = rand((600, 7))
        hx = chk.fetch(X)
        r = cnp.flip(cnp.convolve(cnp.flip(X), X[:5, :3], mode="full")) + 1.0
        ref = np.flip(ref_convolve(np.flip(hx), hx[:5, :3], "full")) + 1.0
        chk.check("flip(convolve(flip)) + 1", r, ref, **tol(np.float64, ref))

    chk.finish()


if __name__ == "__main__":
    main()
