#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/einsum.py.

Verdict: needs a backend feature (einsum/tensordot + transpose) for the
general case. This port lowers a fixed set of expressions to native
matmuls over the batch axis:
    kac,kbc->kab   (default)  C[k] = A[k] @ B[k].T
    kab,kbc->kac              C[k] = A[k] @ B[k]
    ab,bc->ac                 C    = A @ B
    ac,bc->ab                 C    = A @ B.T
The B.T operands are inputs that stay fixed across iterations: with the
original A <-> C swap, B never changes. They are generated natively as
transposed copies of B (same closed form, permuted indices; checked
against B.T). Any other expression raises NotImplementedError. A native
einsum (or transpose + batched matmul) would be needed.
Other changes: inputs are pseudo-random / N instead of ones, so the check
is meaningful (pass --ones for the original init). The out= form becomes
C[...] = ... / C[k] = ...
"""

import argparse
import os
import re
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from _cnum_connect import get_interface  # noqa: E402

SUPPORTED = {
    "kac,kbc->kab": ("batched", True),
    "kab,kbc->kac": ("batched", False),
    "ab,bc->ac": ("plain", False),
    "ac,bc->ab": ("plain", True),
}


def parse(expr):
    m = re.match(r"([a-zA-Z]*),([a-zA-Z]*)->([a-zA-Z]*)", expr)
    assert m is not None
    a_modes, b_modes, c_modes = list(m.group(1)), list(m.group(2)), list(m.group(3))
    assert len(a_modes) == len(set(a_modes))
    assert len(b_modes) == len(set(b_modes))
    assert len(c_modes) == len(set(c_modes))
    assert all(m in a_modes or m in b_modes for m in c_modes)
    assert len(a_modes) == len(c_modes) or len(b_modes) == len(c_modes)
    return a_modes, b_modes, c_modes


def contract(kind, A, Bop, out, N):
    """out = einsum(expr, A, B) using native matmuls (Bop is B or B.T)."""
    if kind == "plain":
        res = A @ Bop
        if out is None:
            return res
        out[...] = res
        return out
    if out is None:
        out = cnp.zeros(A.shape, dtype=A.dtype)
    for k in range(N):
        out[k] = A[k] @ Bop[k]  # (1,N,N) @ (1,N,N) -> (N,N) -> C[k]
    return out


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expr", default="kac,kbc->kab")
    parser.add_argument("-n", "--num", type=int, default=16, dest="N")
    parser.add_argument("-i", "--iters", type=int, default=3, dest="iters")
    parser.add_argument("-w", "--warmup", type=int, default=1, dest="warmup")
    parser.add_argument("-t", "--dtype", default="f32", choices=["f32", "f64"], dest="dtype")
    parser.add_argument("--cupy-compatibility", action="store_true", dest="cupy_compatibility")
    parser.add_argument("--ones", action="store_true", help="use the original ones() inputs")
    args = parser.parse_args()
    expr, N = args.expr, args.N
    dtype = {"f32": numpy.float32, "f64": numpy.float64}[args.dtype]

    a_modes, b_modes, c_modes = parse(expr)
    swap_a_c = len(a_modes) == len(c_modes)
    if expr not in SUPPORTED:
        raise NotImplementedError(
            f"einsum '{expr}' needs a native einsum/transpose op; supported: {sorted(SUPPORTED)}"
        )
    kind, transpose_b = SUPPORTED[expr]
    print(f"Expression: {expr}")
    print(f"Problem Size: {N}**{len(a_modes)} x {N}**{len(b_modes)} -> {N}**{len(c_modes)}")
    print(f"Total Iterations: {args.iters}")

    interface = get_interface()
    a_shape = (N,) * len(a_modes)
    b_shape = (N,) * len(b_modes)
    if args.ones:
        A = cnp.ones(a_shape, dtype=dtype)
        B = cnp.ones(b_shape, dtype=dtype)
        BT = cnp.ones(b_shape, dtype=dtype)
    else:
        A = U.uniform(a_shape, seed=1, dtype=dtype) * (1.0 / N)
        B = U.uniform(b_shape, seed=2, dtype=dtype) * (1.0 / N)
        perm = (0, 2, 1) if len(b_modes) == 3 else (1, 0)
        BT = U.uniform(b_shape, seed=2, dtype=dtype, perm=perm) * (1.0 / N)
    C = cnp.zeros((N,) * len(c_modes), dtype=dtype)
    Bop = BT if transpose_b else B
    Ah, Bh = A.get(interface), B.get(interface)
    if transpose_b:
        perm = (0, 2, 1) if len(b_modes) == 3 else (1, 0)
        chk.close("generated B^T == transpose(B)", BT.get(interface), numpy.transpose(Bh, perm), rtol=0, atol=0)

    t0 = time.perf_counter()
    for idx in range(args.iters + args.warmup):
        if args.cupy_compatibility:
            C = contract(kind, A, Bop, None, N)
        else:
            contract(kind, A, Bop, C, N)
        if swap_a_c:
            A, C = C, A
        else:
            B, C = C, B
            Bop = B  # (only reachable for non-transposed forms)
    A_out = A.get(interface)
    print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    # NumPy reference
    rA, rB, rC = Ah.copy(), Bh.copy(), numpy.zeros((N,) * len(c_modes), dtype=dtype)
    for idx in range(args.iters + args.warmup):
        numpy.einsum(expr, rA, rB, out=rC)
        if swap_a_c:
            rA, rC = rC, rA
        else:
            rB, rC = rC, rB
    rtol = 1e-4 if dtype == numpy.float32 else 1e-10
    chk.close("final A", A_out, rA, rtol=rtol, atol=rtol * numpy.abs(rA).max())


if __name__ == "__main__":
    U.run("einsum", main)
