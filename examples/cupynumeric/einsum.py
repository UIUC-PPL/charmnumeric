#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/einsum.py.

Verdict: needs a backend feature (einsum/tensordot) for the general case.
This port lowers every single-contraction matrix-product expression to
native transposes + matmuls:
    2-D: ij,jk->ik with any mode order in A, B and C, e.g.
         ab,bc->ac   C = A @ B          ac,bc->ab   C = A @ B.T
         ba,bc->ac   C = A.T @ B        ab,bc->ca   C = (A @ B).T
    3-D: the same with a leading batch mode k in all three operands,
         looped over k, e.g. kac,kbc->kab (default)  C[k] = A[k] @ B[k].T
Operand transposes use the native transpose (3-D: axes (0, 2, 1)). A
transposed operand is recomputed only when it changes (B never changes
under the original A <-> C swap). Any other expression raises
NotImplementedError.
Other changes: inputs are pseudo-random / N instead of ones, so the check
is meaningful (pass --ones for the original init). The out= form is native
(cnp.matmul(..., out=C)) for the 2-D expressions with an untransposed
output; a transposed output is written as C[...] = (A' @ B').T, and the
batched ones write C[k] = A[k] @ B[k] per batch (out=C[k] would be a (1,N,N) view, which NumPy
semantics reject for an (N,N) result).
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

def plan(a_modes, b_modes, c_modes):
    """Lower a matrix-product einsum to (kind, a_T, b_T, c_T), or None.

    kind is "plain" (2-D) or "batched" (3-D, batch mode first everywhere).
    x_T says whether operand x enters as its (last-two-axes) transpose:
    C' = op(A) @ op(B), C = C' or C'.T.
    """
    if len(a_modes) == len(b_modes) == len(c_modes) == 3:
        if not (a_modes[0] == b_modes[0] == c_modes[0]):
            return None
        kind = "batched"
        a_modes, b_modes, c_modes = a_modes[1:], b_modes[1:], c_modes[1:]
    elif len(a_modes) == len(b_modes) == len(c_modes) == 2:
        kind = "plain"
    else:
        return None
    shared = [m for m in a_modes if m in b_modes]
    if len(shared) != 1 or shared[0] in c_modes:
        return None
    s = shared[0]
    i = [m for m in a_modes if m != s][0]
    j = [m for m in b_modes if m != s][0]
    if i == j or sorted(c_modes) != sorted([i, j]):
        return None
    return kind, a_modes == [s, i], b_modes == [j, s], c_modes == [j, i]


def T(x, kind):
    """Native transpose of the last two axes."""
    return cnp.transpose(x, (0, 2, 1)) if kind == "batched" else x.T


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


def contract(kind, Aop, Bop, c_T, out, N):
    """out = einsum(expr, A, B) using native matmuls (Aop/Bop: A/B or transposes)."""
    if kind == "plain":
        if not c_T:
            return cnp.matmul(Aop, Bop, out=out)  # native out=
        res = (Aop @ Bop).T
        if out is None:
            return res
        out[...] = res
        return out
    if out is None:
        out = cnp.zeros(Aop.shape, dtype=Aop.dtype)
    for k in range(N):
        res = Aop[k] @ Bop[k]  # (1,N,N) @ (1,N,N) -> (N,N)
        out[k] = res.T if c_T else res
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
    lowered = plan(a_modes, b_modes, c_modes)
    if lowered is None:
        raise NotImplementedError(
            f"einsum '{expr}' needs a native einsum op; only single-contraction "
            "matrix products (2-D, or 3-D with a leading batch mode) are lowered"
        )
    kind, transpose_a, transpose_b, c_T = lowered
    print(f"Expression: {expr}")
    print(f"Problem Size: {N}**{len(a_modes)} x {N}**{len(b_modes)} -> {N}**{len(c_modes)}")
    print(f"Total Iterations: {args.iters}")

    interface = get_interface()
    a_shape = (N,) * len(a_modes)
    b_shape = (N,) * len(b_modes)
    if args.ones:
        A = cnp.ones(a_shape, dtype=dtype)
        B = cnp.ones(b_shape, dtype=dtype)
    else:
        A = U.uniform(a_shape, seed=1, dtype=dtype) * (1.0 / N)
        B = U.uniform(b_shape, seed=2, dtype=dtype) * (1.0 / N)
    C = cnp.zeros((N,) * len(c_modes), dtype=dtype)
    Ah, Bh = A.get(interface), B.get(interface)

    t0 = time.perf_counter()
    for idx in range(args.iters + args.warmup):
        # transposed operands: recomputed only when the operand changed
        if idx == 0 or swap_a_c:
            Aop = T(A, kind) if transpose_a else A
        if idx == 0 or not swap_a_c:
            Bop = T(B, kind) if transpose_b else B
        if args.cupy_compatibility:
            C = contract(kind, Aop, Bop, c_T, None, N)
        else:
            contract(kind, Aop, Bop, c_T, C, N)
        if swap_a_c:
            A, C = C, A
        else:
            B, C = C, B
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
