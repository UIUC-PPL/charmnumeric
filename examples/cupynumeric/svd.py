#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/svd.py.

Verdict: needs backend features (gesvd/syevd). The port uses the
Gram-matrix (normal-equations) SVD. The O(m n^2) work is native and only
the small k x k eigenproblem (k = min(m, n)) runs on the host:
    m >= n:  G = A^T A (native SUMMA) -> eigh(G) = V diag(s^2) V^T  HOST-FALLBACK
             U = A @ (V / s)          (native SUMMA, V/s uploaded n x n)
             vh = V^T                 (host, it is the small factor)
    m <  n:  G = A A^T (native)  -> eigh -> U (host), vh = (U / s)^T @ A (native)
a = cnp.random.random((m, n)) as in the original; A^T comes from
cnp.random.transposed (same draw, permuted indices), since there is no
transpose. The Gram approach squares the condition number. That is
fine for the random test matrices, and the reconstruction (u*s) @ vh = A
holds regardless. full_matrices with m != n would need an orthogonal
complement (QR). The port returns the reduced factors, which is what the
original check uses. Integer dtypes are not supported (no astype). The
default is 32x32; the n x n factor V/s is a bulk upload.
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


def gram_svd(A, AT, interface):
    """Reduced SVD via the Gram matrix. Returns (u, s, vh) with u, vh as charmnumeric/host arrays."""
    m, n = A.shape
    if m >= n:
        G = AT @ A  # native
        # HOST-FALLBACK: k x k symmetric eigendecomposition (no syevd/gesvd in charmnumeric)
        w, V = numpy.linalg.eigh(G.get(interface))
        order = numpy.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        s = numpy.sqrt(numpy.clip(w, 0.0, None))
        VS = U.upload(V / s, dtype=A.dtype)
        u = A @ VS  # native: U = A V S^-1
        vh = V.T  # host (small factor)
        return u, s, vh
    G = A @ AT  # native, m x m
    w, Uh = numpy.linalg.eigh(G.get(interface))  # HOST-FALLBACK: k x k eigh
    order = numpy.argsort(w)[::-1]
    w, Uh = w[order], Uh[:, order]
    s = numpy.sqrt(numpy.clip(w, 0.0, None))
    US_T = U.upload((Uh / s).T, dtype=A.dtype)
    vh = US_T @ A  # native: V^T = S^-1 U^T A
    return Uh, s, vh


def check_result(chk, a, u, s, vh):
    print("Checking result...")
    m, n = a.shape
    k = min(m, n)
    u = u[:, :k] if k < m else u
    vh = vh[:k, :] if k < m else vh
    a2 = numpy.matmul(u * s, vh)
    chk.close("(u * s) @ vh vs a", a2, a, rtol=1e-5, atol=1e-8)
    chk.close("u^T u vs I", u.T @ u, numpy.eye(u.shape[1]), rtol=0, atol=1e-6)
    s_ref = numpy.linalg.svd(a, compute_uv=False)
    chk.close("singular values vs numpy", s, s_ref, rtol=1e-8, atol=1e-10)


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("-m", "--rows", type=int, default=32, dest="m")
    parser.add_argument("-n", "--cols", type=int, default=32, dest="n")
    parser.add_argument("--no-full-matrices", dest="full_matrices", action="store_false")
    parser.add_argument("-d", "--dtype", default="float64", choices=["float32", "float64"], dest="dtype")
    parser.add_argument("--check", dest="check", action="store_true")
    args = parser.parse_args()
    m, n = args.m, args.n
    dtype = numpy.dtype(args.dtype)
    if args.full_matrices and m != n:
        print("note: full_matrices with m != n would need a complement basis; returning reduced factors")

    interface = get_interface()
    a = cnp.random.random((m, n), dtype=dtype)
    aT = cnp.random.transposed(a)  # == a.T
    t0 = time.perf_counter()
    u, s, vh = gram_svd(a, aT, interface)
    u_h = u.get(interface) if not isinstance(u, numpy.ndarray) else u
    vh_h = vh.get(interface) if not isinstance(vh, numpy.ndarray) else vh
    if args.timing:
        print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3} ms (incl. sync)")

    a_h = a.get(interface)
    chk.close("generated a.T", aT.get(interface), a_h.T, rtol=0, atol=0)
    check_result(chk, a_h.astype(numpy.float64), u_h, s, vh_h)


if __name__ == "__main__":
    U.run("svd", main)
