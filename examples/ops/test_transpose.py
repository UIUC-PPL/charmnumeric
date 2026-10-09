#!/usr/bin/env python
"""Tests for native transpose (TRANSPOSE executor) and transposed-operand matmul.

Covers a.T / cnp.transpose / a.transpose() / cnp.swapaxes on 2-D and 3-D
arrays, square / non-square / non-tile-aligned shapes, offset and strided
views, chains (A.T.T, (A.T @ B).T), elementwise consumers of transposes,
vecmat (x @ A) and A.T @ B, A @ B.T, A.T @ x, all checked against NumPy.

Inputs are generated natively (no per-element upload): linear-index grids,
whose values are all distinct, so any misplaced element is detected exactly.

Environment: CNUM_HOST, CNUM_PORT, CNUM_ODF (default 4). Run with e.g.
    harness/run_one.sh server.out 2 34001 examples/ops/test_transpose.py
Prints PASS test_transpose / FAIL test_transpose: ... (exit status 0 / 1).
On the CPU build the minimum 2-D tile is 512 and 3-D tile 64, so the larger
shapes here span several chares (more with +p2 and odf > 1).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cupynumeric"))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402


def lin(shape, dtype=np.float64, scale=1.0, shift=0.0):
    """Distinct values: shift + scale * row-major linear index (native)."""
    L = U.linear_index(shape, dtype=np.float64)
    if scale != 1.0 or shift != 0.0:
        L = L * scale + shift
    if np.dtype(dtype) == np.float64:
        return L
    # other dtypes: build directly from integer index grids
    shape = tuple(shape)
    out = cnp.zeros(shape, dtype=dtype)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        if shape[axis] > 1:
            out = out + U.index_grid(shape, axis, dtype=dtype) * (
                int(stride) if np.issubdtype(np.dtype(dtype), np.integer) else float(stride)
            )
        stride *= shape[axis]
    return out


def host_lin(shape, dtype=np.float64, scale=1.0, shift=0.0):
    L = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    if np.dtype(dtype) == np.float64:
        return L * scale + shift
    return L.astype(dtype)


_SECTIONS = os.environ.get("TT_SECTIONS")


def on(name):
    """Section filter: TT_SECTIONS=shapes,views,... runs only those sections."""
    return _SECTIONS is None or name in _SECTIONS.split(",")


def main(chk):
    host = os.environ.get("CNUM_HOST", "127.0.0.1")
    port = int(os.environ.get("CNUM_PORT", "1234"))
    odf = int(os.environ.get("CNUM_ODF", "4"))
    iface = cnp.CharmNumericInterface()
    iface.connect(host, port, odf)
    print(f"connected {host}:{port} odf={odf}")

    def eq(label, arr, ref, rtol=0.0, atol=0.0):
        got = arr.get(iface)
        ref = np.asarray(ref)
        if tuple(arr.shape) != ref.shape:
            chk.true(label + " shape", False, f"{arr.shape} != {ref.shape}")
            return
        chk.close(label, got, ref, rtol=rtol, atol=atol)

    # ---- 2-D shapes: square tile-aligned, non-square non-aligned, tiny ----
    if on("shapes"):
        for shape in [(1024, 1024), (1100, 700), (700, 1100), (37, 53)]:
            A = lin(shape)
            Ah = host_lin(shape)
            eq(f"{shape} A.T", A.T, Ah.T)
            eq(f"{shape} transpose(A)", cnp.transpose(A), Ah.T)
            eq(f"{shape} A.transpose()", A.transpose(), Ah.T)
            eq(f"{shape} A.transpose(1, 0)", A.transpose(1, 0), Ah.T)
            eq(f"{shape} swapaxes(A, 0, 1)", cnp.swapaxes(A, 0, 1), Ah.T)
            eq(f"{shape} A.T.T", A.T.T, Ah)

    # ---- dtypes ----
    if on("dtypes"):
        for dt in (np.float32, np.int32, np.int64):
            A = lin((600, 530), dtype=dt)
            eq(f"{np.dtype(dt).name} (600,530).T", A.T, host_lin((600, 530), dtype=dt).T)

    # shared input for the view / shifted / chain sections
    A = lin((1100, 700))
    Ah = host_lin((1100, 700))

    # ---- views / slices (offset, strided, row/column) ----
    if on("views"):
        views = [
            ((slice(100, 1050), slice(30, 690)), "offset view"),
            ((slice(513, 1100), slice(511, 700)), "unaligned offset view"),
            ((slice(1, 1100, 3), slice(5, 700, 2)), "strided view"),
            ((slice(0, 1100, 600), slice(0, 700, 7)), "stride > tile"),
            ((slice(5, 6), slice(None)), "row view"),
            ((slice(None), slice(3, 4)), "column view"),
        ]
        for key, label in views:
            eq(f"{label} A[...].T", A[key].T, Ah[key].T)
        # (1, n) / (n, 1) arrays (U.linear_index mis-builds singleton-dim grids,
        # so copy them out of A)
        eq("(1,700) array .T", A[5:6, :].copy().T, Ah[5:6, :].T)
        eq("(1100,1) array .T", A[:, 3:4].copy().T, Ah[:, 3:4].T)
        V = A[100:1050, 30:690]
        eq("view-of-view .T", V[10:900:2, 5:600].T, Ah[100:1050, 30:690][10:900:2, 5:600].T)

    # ---- source with a shifted (possibly offset) decomposition ----
    if on("shifted"):
        S = A[1:, :] + A[:-1, :]
        Sh = Ah[1:, :] + Ah[:-1, :]
        eq("shifted-sum result .T", S.T, Sh.T)
        eq("(shifted-sum).T + 1", S.T + 1.0, Sh.T + 1.0)

    # ---- elementwise consumers, chains, materialized semantics ----
    if on("chains"):
        B = lin((700, 1100), scale=0.5, shift=3.0)
        Bh = host_lin((700, 1100), scale=0.5, shift=3.0)
        eq("A.T + B", A.T + B, Ah.T + Bh)
        eq("A.T * 2 - B.T.T", A.T * 2.0 - B.T.T, Ah.T * 2.0 - Bh)
        C = A.T
        D = C.T
        eq("D = (A.T).T (separate DAG nodes)", D, Ah)

        E = lin((300, 200))
        Eh = host_lin((300, 200))
        ET = E.T
        E[0:1, 0:1] = -1.0  # writes to E must not reach the materialized ET
        Eh2 = Eh.copy()
        Eh2[0, 0] = -1.0
        eq("transpose is materialized (unaffected by later write)", ET, Eh.T)
        eq("transpose after write", E.T, Eh2.T)

        # 1-D and identity permutations return the array itself (NumPy view semantics)
        v = U.iota(10)
        chk.true("1-D .T is identity", v.T is v)
        chk.true("identity axes returns self", E.transpose(0, 1) is E)

    # ---- matmul with transposed operands ----
    if on("matmul"):
        rtol = 1e-12
        M = U.uniform((640, 600), seed=1)
        N_ = U.uniform((640, 560), seed=2)
        P = U.uniform((560, 600), seed=3)
        x = U.uniform((640,), seed=4)
        y = U.uniform((600,), seed=5)
        Mh, Nh, Ph, xh, yh = (t.get(iface) for t in (M, N_, P, x, y))
        eq("A.T @ B", M.T @ N_, Mh.T @ Nh, rtol=rtol)
        eq("A @ B.T", N_.T @ M, Nh.T @ Mh, rtol=rtol)
        eq("A @ B.T (rhs)", M @ P.T, Mh @ Ph.T, rtol=rtol)
        eq("(A.T @ B).T", (M.T @ N_).T, (Mh.T @ Nh).T, rtol=rtol)
        # operands that get different tile sizes at small +p/odf
        M2 = U.uniform((640, 520), seed=6)
        N2 = U.uniform((640, 300), seed=7)
        M2h, N2h = M2.get(iface), N2.get(iface)
        eq("A.T @ B (mixed tiles)", M2.T @ N2, M2h.T @ N2h, rtol=rtol)
        eq("x @ A (mixed tiles)", x @ M2, xh @ M2h, rtol=rtol)
        eq("A.T @ x", M.T @ x, Mh.T @ xh, rtol=rtol)
        eq("x @ A (vecmat)", x @ M, xh @ Mh, rtol=rtol)
        eq("x.dot(A)", x.dot(M), xh.dot(Mh), rtol=rtol)
        eq("cnp.dot(x, A)", cnp.dot(x, M), np.dot(xh, Mh), rtol=rtol)
        eq("cnp.matmul(x, A)", cnp.matmul(x, M), np.matmul(xh, Mh), rtol=rtol)
        eq("y @ A.T", y @ M.T, yh @ Mh.T, rtol=rtol)
        eq("x[0:300] @ view", x[0:300] @ M[100:400, 20:580], xh[0:300] @ Mh[100:400, 20:580],
           rtol=rtol)
        # vecmat caches the transpose internally; a write to A must invalidate it
        r1 = x @ M
        r1_ref = xh @ Mh
        M[0:1, :] = 2.0
        Mh[0, :] = 2.0
        eq("vecmat before write", r1, r1_ref, rtol=rtol)
        eq("vecmat after write to A", x @ M, xh @ Mh, rtol=rtol)
        # small non-tile-aligned vecmat
        Q = lin((37, 53))
        q = U.iota(37)
        eq("small vecmat", q @ Q, np.arange(37.0) @ host_lin((37, 53)), rtol=rtol)

    # ---- 3-D ----
    if on("3d"):
        for shape, axes in [
            ((130, 70, 20), (2, 0, 1)),
            ((130, 70, 20), None),
            ((20, 130, 70), (1, 0, 2)),
            ((5, 7, 9), (0, 2, 1)),
        ]:
            T3 = lin(shape)
            T3h = host_lin(shape)
            eq(f"3-D {shape} transpose{axes}", cnp.transpose(T3, axes), np.transpose(T3h, axes))
        T3 = lin((130, 70, 20))
        T3h = host_lin((130, 70, 20))
        eq("3-D swapaxes(0, 2)", T3.swapaxes(0, 2), T3h.swapaxes(0, 2))
        eq("3-D view .transpose(2,1,0)", T3[3:120:2, 10:70, 1:19].transpose(2, 1, 0),
           T3h[3:120:2, 10:70, 1:19].transpose(2, 1, 0))
        eq("3-D A.T.T", T3.T.T, T3h)

    # ---- errors ----
    if on("errors"):
        try:
            cnp.transpose(E, (0, 0))
            chk.true("repeated axes raise", False)
        except ValueError:
            chk.true("repeated axes raise", True)


if __name__ == "__main__":
    U.run("test_transpose", main)
