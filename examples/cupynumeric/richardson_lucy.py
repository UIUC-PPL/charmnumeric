#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/richardson_lucy.py (3-D RL deconvolution).

Verdict: port with workarounds (fully native). Changes:
  * np.convolve(x, psf, mode="same") on 3-D arrays (as in cuPyNumeric and
    scipy.signal.convolve) is a direct convolution:
        same[i] = sum_k psf[k] * P[i + f - 1 - k]
    P is x zero-padded by f//2 (low) and (f-1)//2 (high) per dim. Each term
    is a native slice of P times a size-1 view of psf (a BROADCAST scalar
    operand). That is fx*fy*fz multiply-adds per convolution. Only the
    direct method is available; --conv-method fft is not.
  * np.flip(psf): charmnumeric has no negative strides. psf is input data,
    so psf_mirror is generated natively from the reversed linear index
    (U.uniform_from_index(total-1-L)). The check confirms it equals
    np.flip(psf) exactly.
  * np.random.rand is replaced by the native generator U.uniform.
Backend ops that would make it direct: convolve (or a stencil/correlate
primitive) and flip / negative-step slicing.
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

float_type = numpy.float32


def convolve_same(x, psf):
    """Native 3-D 'same' convolution (scipy.signal.convolve semantics)."""
    n = x.shape
    f = psf.shape
    lo = tuple(fd // 2 for fd in f)
    P = cnp.zeros(tuple(n[d] + f[d] - 1 for d in range(3)), dtype=x.dtype)
    P[lo[0] : lo[0] + n[0], lo[1] : lo[1] + n[1], lo[2] : lo[2] + n[2]] = x
    acc = None
    for kx in range(f[0]):
        sx = f[0] - 1 - kx
        for ky in range(f[1]):
            sy = f[1] - 1 - ky
            for kz in range(f[2]):
                sz = f[2] - 1 - kz
                w = psf[kx : kx + 1, ky : ky + 1, kz : kz + 1]  # size-1 -> broadcast scalar
                term = w * P[sx : sx + n[0], sy : sy + n[1], sz : sz + n[2]]
                acc = term if acc is None else acc + term
    return acc


def run_richardson_lucy(shape, filter_shape, num_iter, warmup):
    image = U.uniform(shape, seed=1, dtype=float_type)
    psf = U.uniform(filter_shape, seed=2, dtype=float_type)
    im_deconv = cnp.full(shape, 0.5, dtype=float_type)
    # psf_mirror = np.flip(psf): same generator on the reversed linear index
    total = int(numpy.prod(filter_shape))
    L = U.linear_index(filter_shape, dtype=float_type)
    psf_mirror = U.uniform_from_index((total - 1) - L, total, seed=2)

    for idx in range(num_iter + warmup):
        conv = convolve_same(im_deconv, psf)
        relative_blur = image / conv
        im_deconv *= convolve_same(relative_blur, psf_mirror)
    return image, psf, psf_mirror, im_deconv


# ---- NumPy reference ---------------------------------------------------------
def convolve_same_ref(x, v):
    n, f = x.shape, v.shape
    full_shape = tuple(n[d] + f[d] - 1 for d in range(3))
    full = numpy.zeros(full_shape, dtype=numpy.float64)
    for k in numpy.ndindex(*f):
        full[k[0] : k[0] + n[0], k[1] : k[1] + n[1], k[2] : k[2] + n[2]] += v[k] * x
    st = tuple((f[d] - 1) // 2 for d in range(3))
    return full[st[0] : st[0] + n[0], st[1] : st[1] + n[1], st[2] : st[2] + n[2]]


def reference(image, psf, num_iter, warmup):
    im_deconv = numpy.full(image.shape, 0.5)
    psf_mirror = numpy.flip(psf)
    for idx in range(num_iter + warmup):
        conv = convolve_same_ref(im_deconv, psf)
        relative_blur = image / conv
        im_deconv = im_deconv * convolve_same_ref(relative_blur, psf_mirror)
    return im_deconv


def main(chk):
    def tuple_of_ints(arg):
        return tuple(map(int, arg.strip("()").split(",")))

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iter", type=int, default=3, dest="I")
    parser.add_argument("-w", "--warmup", type=int, default=1, dest="warmup")
    parser.add_argument("--shape", type=tuple_of_ints, default=(20, 20, 20), dest="shape")
    parser.add_argument("--filter-shape", type=tuple_of_ints, default=(4, 4, 4), dest="filter_shape")
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("--conv-method", dest="conv_method", type=str, default="auto")
    args = parser.parse_args()
    if args.conv_method not in ("auto", "direct"):
        raise NotImplementedError("only direct convolution is available in charmnumeric")
    if len(args.shape) != 3 or len(args.filter_shape) != 3:
        raise NotImplementedError("the port implements the 3-D case of the benchmark")

    interface = get_interface()
    t0 = time.perf_counter()
    image, psf, psf_mirror, im_deconv = run_richardson_lucy(args.shape, args.filter_shape, args.I, args.warmup)
    got = im_deconv.get(interface)
    if args.timing:
        print("Elapsed Time: " + str((time.perf_counter() - t0) * 1e3) + " ms (incl. sync)")

    img_h, psf_h, psfm_h = image.get(interface), psf.get(interface), psf_mirror.get(interface)
    chk.close("generated psf_mirror == flip(psf)", psfm_h, numpy.flip(psf_h), rtol=0, atol=0)
    ref = reference(img_h.astype(numpy.float64), psf_h.astype(numpy.float64), args.I, args.warmup)
    chk.close("im_deconv", got, ref, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    U.run("richardson_lucy", main)
