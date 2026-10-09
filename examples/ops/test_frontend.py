#!/usr/bin/env python
"""Tests for the frontend-only ops and fixes, checked against NumPy.

Covers: concatenate/hstack/vstack/stack, out= on elementwise functions and
matmul/dot, eye/identity through DIAG, take with host indices, newaxis,
1-D @ 2-D, (M,K) @ (K,1), cross-ndim assignment (offsets, views, size-1
dims), size-1 assignment sources, views fed to REDUCE/TILE/DIAG, the 1-3 dim
limit, and frontend names above the old 65536 cap.

Inputs are generated natively from index grids (no per-element upload),
fetched with .get(), and the NumPy reference is computed from the fetched
values. Prints PASS/FAIL per case and a final "PASS test_frontend" or
"FAIL test_frontend: ...".

Environment: CNUM_HOST / CNUM_PORT / CNUM_ODF select the server (the
harness's run_example.py also redirects connect()). CNUM_SIM=1 runs against
the NumPy simulator in ../cupynumeric/_cnum_sim.py instead (no backend).
"""

import itertools
import os
import sys
import traceback

import numpy as np

import charmtyles.core as ct_core
from charmtyles.ast import get_execution_state

# Start frontend names close to the int32 limit: every array in this test
# then has a name far above the old 1 << 16 cap, which exercises the wire
# format and the backend maps with large names.
# CNUM_TEST_NAME_BASE overrides the starting name (0 keeps the default).
_NAME_BASE = int(os.environ.get("CNUM_TEST_NAME_BASE", str((1 << 31) - 20_000_000)))
ct_core.next_name = max(ct_core.next_name, _NAME_BASE)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.environ.get("CNUM_SIM", "") not in ("", "0")
if _SIM:
    sys.path.insert(0, os.path.join(_HERE, "..", "cupynumeric"))
    import _cnum_sim  # noqa: F401,E402

import charmnumeric as cnp  # noqa: E402

iface = None
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def connect():
    if _SIM:
        from _cnum_sim import SimInterface

        return SimInterface()
    interface = cnp.CharmNumericInterface()
    interface.connect(
        os.environ.get("CNUM_HOST", "127.0.0.1"),
        int(os.environ.get("CNUM_PORT", "1234")),
        int(os.environ.get("CNUM_ODF", "4")),
    )
    return interface


def get(x):
    return np.asarray(x.get(iface))


def made(*arrays):
    """Fetch (and flush) inputs before the op under test is recorded."""
    hosts = [get(a) for a in arrays]
    return hosts[0] if len(hosts) == 1 else hosts


def check(label, got, exp, rtol=0.0, atol=0.0):
    got = np.asarray(got)
    exp = np.asarray(exp)
    if got.shape != exp.shape:
        raise AssertionError(f"{label}: shape {got.shape} != expected {exp.shape}")
    if got.dtype != exp.dtype:
        raise AssertionError(f"{label}: dtype {got.dtype} != expected {exp.dtype}")
    if not np.allclose(got, exp, rtol=rtol, atol=atol):
        err = np.max(np.abs(got.astype(np.float64) - exp.astype(np.float64)))
        raise AssertionError(f"{label}: max|err|={err:.3e}")


def raises(label, exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{label}: expected {exc.__name__}")


# ---------------------------------------------------------------------------
# native input generation
# ---------------------------------------------------------------------------
def index_grid(shape, axis, dtype):
    g = cnp.zeros(shape, dtype=dtype)
    n = shape[axis]
    k = 1
    while k < n:
        m = min(k, n - k)
        dst = [slice(None)] * len(shape)
        src = [slice(None)] * len(shape)
        dst[axis] = slice(k, k + m)
        src[axis] = slice(0, m)
        g[tuple(dst)] = g[tuple(src)] + k
        k *= 2
    return g


def data(shape, dtype=np.float64, seed=0):
    """Distinct, non-trivial values: a function of the row-major index."""
    shape = tuple(shape)
    dtype = np.dtype(dtype)
    if int(np.prod(shape)) == 1:
        # (the JIT cannot apply unary math to an all-broadcast kernel yet)
        return cnp.full(shape, 7 * seed + 1, dtype=dtype)
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    L = None
    for d in range(len(shape)):
        if shape[d] == 1:
            continue
        term = index_grid(shape, d, dtype) * (strides[d] if dtype.kind in "iu" else float(strides[d]))
        L = term if L is None else L + term
    if L is None:
        L = cnp.zeros(shape, dtype=dtype)
    if dtype.kind in "iu":
        return L * (3 + seed) + (7 * seed + 1)
    return cnp.tanh(L * (0.0137 + 0.001 * seed) + 0.1 * seed) * 3.0 + L * 0.001 + seed


# ---------------------------------------------------------------------------
# concatenate / hstack / vstack / stack
# ---------------------------------------------------------------------------
@case
def concatenate_1d():
    for sizes in [(7, 100, 3), (300000, 5), (1, 9)]:
        parts = [data((n,), seed=i) for i, n in enumerate(sizes)]
        hosts = [made(p) for p in parts]
        r = cnp.concatenate(parts)
        check(f"concatenate {sizes}", get(r), np.concatenate(hosts))


@case
def concatenate_2d_3d():
    for shapes, axis in [
        (((600, 4), (30, 4)), 0),
        (((5, 3), (5, 7), (5, 1)), 1),
        (((5, 3), (5, 7)), -1),
        (((70, 5, 3), (70, 5, 2)), 2),
        (((2, 5, 3), (70, 5, 3)), 0),
        (((4, 1, 3), (4, 6, 3)), 1),
    ]:
        parts = [data(s, seed=i) for i, s in enumerate(shapes)]
        hosts = [made(p) for p in parts]
        r = cnp.concatenate(parts, axis=axis)
        check(f"concatenate {shapes} axis={axis}", get(r), np.concatenate(hosts, axis=axis))


@case
def concatenate_views_int_host():
    A = data((20, 12), dtype=np.int64)
    B = data((9, 30), dtype=np.int64, seed=2)
    Ah, Bh = made(A, B)
    r = cnp.concatenate((A[1:5, ::2], B[3:7, 4:10], [[1, 2], [3, 4], [5, 6], [7, 8]]), axis=1)
    exp = np.concatenate((Ah[1:5, ::2], Bh[3:7, 4:10], [[1, 2], [3, 4], [5, 6], [7, 8]]), axis=1)
    check("concatenate views+host int64", get(r), exp)

    out = cnp.zeros((19, 6), dtype=np.int64)
    res = cnp.concatenate((A[0:10, 0:6], B[0:9, 0:6]), out=out)
    if res is not out:
        raise AssertionError("concatenate(out=) must return out")
    check("concatenate out=", get(out), np.concatenate((Ah[0:10, 0:6], Bh[0:9, 0:6])))

    big = cnp.zeros((12, 6), dtype=np.int64)
    cnp.concatenate((A[0:2, 0:6], A[5:9, 0:6]), out=big[3:9])
    exp = np.zeros((12, 6), dtype=np.int64)
    exp[3:9] = np.concatenate((Ah[0:2, 0:6], Ah[5:9, 0:6]))
    check("concatenate out=view", get(big), exp)

    # out aliases an input: must read before writing
    C = data((8,), seed=3)
    Ch = made(C)
    cnp.concatenate((C[4:8], C[0:4]), out=C)
    check("concatenate out aliasing input", get(C), np.concatenate((Ch[4:8], Ch[0:4])))

    raises("dtype mismatch", TypeError, lambda: cnp.concatenate((A, data((20, 12)))))
    raises("ndim mismatch", ValueError, lambda: cnp.concatenate((A, data((20,), dtype=np.int64))))
    raises("shape mismatch", ValueError, lambda: cnp.concatenate((A, B), axis=0))


@case
def hstack_vstack():
    N, F = 600, 5
    X = data((N, F))
    Xh = made(X)
    # linreg/logreg: np.hstack((intercept, features))
    f2 = cnp.hstack((cnp.ones((N, 1), dtype=np.float64), X))
    check("hstack intercept", get(f2), np.hstack((np.ones((N, 1)), Xh)))

    a, b = data((7,)), data((300,), seed=1)
    ah, bh = made(a, b)
    check("hstack 1-D", get(cnp.hstack((a, b))), np.hstack((ah, bh)))

    u, v, w = data((50,)), data((50,), seed=1), data((50,), seed=2)
    M = data((3, 50), seed=4)
    big = data((600, 20))
    col = data((600, 1), seed=5)
    uh, vh, wh, Mh, bigh, colh = made(u, v, w, M, big, col)
    check("vstack 1-D", get(cnp.vstack((u, v, w))), np.vstack((uh, vh, wh)))
    check("vstack mixed 1-D/2-D", get(cnp.vstack((M, u))), np.vstack((Mh, uh)))
    check("vstack 2-D multi-chare", get(cnp.vstack((big, big[0:3]))), np.vstack((bigh, bigh[0:3])))
    check("vstack rows of a 2-D array", get(cnp.vstack((big[5, :], big[9, :]))),
          np.vstack((bigh[5:6, :], bigh[9:10, :])))
    check("hstack column views", get(cnp.hstack((col, big[:, 3:4]))), np.hstack((colh, bigh[:, 3:4])))


@case
def stack_new_axis():
    u, v = data((40,)), data((40,), seed=1)
    uh, vh = made(u, v)
    check("stack 1-D axis 0", get(cnp.stack((u, v))), np.stack((uh, vh)))
    check("stack 1-D axis 1", get(cnp.stack((u, v), axis=1)), np.stack((uh, vh), axis=1))
    check("stack 1-D axis -1", get(cnp.stack((u, v, u), axis=-1)), np.stack((uh, vh, uh), axis=-1))
    P, Q = data((70, 6)), data((70, 6), seed=1)
    Ph, Qh = made(P, Q)
    for axis in (0, 1, 2):
        check(f"stack 2-D axis {axis}", get(cnp.stack((P, Q), axis=axis)), np.stack((Ph, Qh), axis=axis))
    check("stack views", get(cnp.stack((P[2:9, 1:4], Q[10:17, 0:3]))),
          np.stack((Ph[2:9, 1:4], Qh[10:17, 0:3])))
    R, S = data((1, 30)), data((1, 30), seed=2)
    Rh, Sh = made(R, S)
    check("stack (1, N) arrays", get(cnp.stack((R, S))), np.stack((Rh, Sh)))
    check("stack (1, N) arrays axis 2", get(cnp.stack((R, S), axis=2)), np.stack((Rh, Sh), axis=2))
    out = cnp.zeros((2, 40), dtype=np.float64)
    cnp.stack((u, v), out=out)
    check("stack out=", get(out), np.stack((uh, vh)))
    raises("stack to 4-D", NotImplementedError, lambda: cnp.stack((data((2, 3, 4)), data((2, 3, 4)))))
    raises("stack shape mismatch", ValueError, lambda: cnp.stack((u, data((41,)))))


# ---------------------------------------------------------------------------
# out=
# ---------------------------------------------------------------------------
@case
def elementwise_out():
    a, b = data((300,)), data((300,), seed=1)
    ah, bh = made(a, b)
    for fn, ref in [(cnp.add, np.add), (cnp.subtract, np.subtract),
                    (cnp.multiply, np.multiply), (cnp.divide, np.divide)]:
        out = cnp.zeros((300,), dtype=np.float64)
        res = fn(a, b, out=out)
        if res is not out:
            raise AssertionError(f"{fn.__name__}(out=) must return out")
        check(f"{fn.__name__} out=", get(out), ref(ah, bh), rtol=1e-14)
        out2 = cnp.zeros((300,), dtype=np.float64)
        fn(a, 2.5, out=(out2,))
        check(f"{fn.__name__} scalar out=(tuple)", get(out2), ref(ah, 2.5), rtol=1e-14)
    for fn, ref in [(cnp.tanh, np.tanh), (cnp.exp, np.exp)]:
        out = cnp.zeros((300,), dtype=np.float64)
        fn(a, out=out)
        check(f"{fn.__name__} out=", get(out), ref(ah), rtol=1e-12)

    # out is a view of a larger 2-D array
    M = data((6, 70))
    Mh = made(M)
    C = cnp.zeros((6, 70), dtype=np.float64)
    cnp.multiply(M[1:4, 10:40], M[2:5, 0:30], out=C[0:3, 5:35])
    exp = np.zeros((6, 70))
    exp[0:3, 5:35] = Mh[1:4, 10:40] * Mh[2:5, 0:30]
    check("multiply out=view", get(C), exp, rtol=1e-14)

    # in place, aliasing an input
    cnp.add(a, b, out=a)
    check("add out=input", get(a), ah + bh, rtol=1e-14)

    # scalar inputs broadcast into out
    s = cnp.zeros((5,), dtype=np.float64)
    cnp.add(1.0, 2.0, out=s)
    check("scalar+scalar out=", get(s), np.full(5, 3.0))

    raises("out shape mismatch", ValueError, lambda: cnp.add(a, b, out=cnp.zeros((299,))))
    raises("out dtype mismatch", TypeError,
           lambda: cnp.add(a, b, out=cnp.zeros((300,), dtype=np.float32)))
    raises("out not an array", TypeError, lambda: cnp.add(a, b, out=np.zeros(300)))


@case
def matmul_out():
    A, B = data((40, 30)), data((30, 20), seed=1)
    v = data((30,), seed=2)
    Ah, Bh, vh = made(A, B, v)
    C = cnp.zeros((40, 20), dtype=np.float64)
    res = cnp.matmul(A, B, out=C)
    if res is not C:
        raise AssertionError("matmul(out=) must return out")
    check("matmul out=", get(C), Ah @ Bh, rtol=1e-12)

    D = cnp.zeros((40, 20), dtype=np.float64)
    cnp.dot(A, B, out=D)
    check("dot out=", get(D), Ah @ Bh, rtol=1e-12)

    y = cnp.zeros((50,), dtype=np.float64)
    cnp.dot(A, v, out=y[5:45])
    exp = np.zeros(50)
    exp[5:45] = Ah @ vh
    check("dot matvec out=view", get(y), exp, rtol=1e-12)

    E = cnp.zeros((40, 20), dtype=np.float64)
    A.dot(B, out=E)
    check("Array.dot out=", get(E), Ah @ Bh, rtol=1e-12)
    raises("matmul out shape", ValueError, lambda: cnp.matmul(A, B, out=cnp.zeros((20, 40))))


# ---------------------------------------------------------------------------
# eye / identity
# ---------------------------------------------------------------------------
@case
def eye_identity():
    state = get_execution_state()
    before = state._pending_op_count()
    I = cnp.eye(500)
    ops = state._pending_op_count() - before
    if ops > 8:
        raise AssertionError(f"eye(500) recorded {ops} ops (expected a handful)")
    check("eye(500)", get(I), np.eye(500))
    for args, kw in [((5,), {}), ((4, 7), {"k": 2}), ((7, 4), {"k": -1}), ((6, 3), {}),
                     ((3, 6), {"k": -2}), ((3,), {"k": 5}), ((600,), {"k": 1})]:
        check(f"eye{args} {kw}", get(cnp.eye(*args, **kw)), np.eye(*args, **kw))
    check("identity int64", get(cnp.identity(9, dtype=np.int64)), np.identity(9, dtype=np.int64))
    check("eye float32", get(cnp.eye(4, 6, k=1, dtype=np.float32)), np.eye(4, 6, k=1, dtype=np.float32))


# ---------------------------------------------------------------------------
# take
# ---------------------------------------------------------------------------
@case
def take_1d():
    a = data((100,))
    ah = made(a)
    rng = np.random.default_rng(0)
    for idx in [
        [3, 4, 5, 6, 7, 50, 51, 52, 99],
        [0, 10, 20, 30, 5, 5, 5, -1, -2],
        rng.integers(0, 100, size=37),
        list(range(100)),
        [42],
    ]:
        check(f"take 1-D {list(idx)[:6]}...", get(cnp.take(a, idx)), np.take(ah, idx))
    check("take 2-D index on 1-D", get(cnp.take(a, [[1, 2, 3], [7, 7, 0]])), np.take(ah, [[1, 2, 3], [7, 7, 0]]))
    check("take scalar on 1-D (as (1,))", get(cnp.take(a, 17)), np.take(ah, [17]))
    check("take mode=wrap", get(cnp.take(a, [101, -105, 3], mode="wrap")), np.take(ah, [101, -105, 3], mode="wrap"))
    check("take mode=clip", get(cnp.take(a, [101, -5, 3], mode="clip")), np.take(ah, [101, -5, 3], mode="clip"))
    raises("take out of bounds", IndexError, lambda: cnp.take(a, [100]))
    raises("take device indices", NotImplementedError, lambda: cnp.take(a, data((3,), dtype=np.int64)))


@case
def take_2d_3d():
    A = data((600, 16), dtype=np.int32)
    T = data((70, 5, 4))
    Ah, Th = made(A, T)
    check("take axis 0", get(cnp.take(A, [0, 1, 2, 599, 300, 301], axis=0)),
          np.take(Ah, [0, 1, 2, 599, 300, 301], axis=0))
    check("take axis 1", get(A.take([15, 0, 2, 4, 6, 8, 3], axis=1)), Ah.take([15, 0, 2, 4, 6, 8, 3], axis=1))
    check("take axis -1 scalar", get(cnp.take(A, 5, axis=-1)), np.take(Ah, 5, axis=-1))
    # take_pairs: (M, N) take (P, 2) along axis 1 -> (M, P, 2)
    pairs = np.array(list(itertools.combinations(range(16), 2)))
    check("take pairs", get(A.take(pairs, axis=1)), Ah.take(pairs, axis=1))
    # on a view
    V = A[10:40:3, 2:14]
    check("take on view", get(cnp.take(V, [0, 1, 2, 9, 11], axis=1)), np.take(Ah[10:40:3, 2:14], [0, 1, 2, 9, 11], axis=1))
    check("take 3-D axis 1", get(cnp.take(T, [4, 0, 1], axis=1)), np.take(Th, [4, 0, 1], axis=1))
    out = cnp.zeros((70, 5, 2), dtype=np.float64)
    cnp.take(T, [3, 1], axis=2, out=out)
    check("take out=", get(out), np.take(Th, [3, 1], axis=2))
    raises("take to 4-D", NotImplementedError, lambda: cnp.take(T, [[0, 1]], axis=1))


# ---------------------------------------------------------------------------
# matmul shape fixes
# ---------------------------------------------------------------------------
@case
def vecmat_and_column():
    v, M = data((5,)), data((5, 7), seed=1)
    v2, M2 = data((600,)), data((600, 3), seed=2)
    A, B = data((40, 5)), data((5, 1), seed=3)
    X = data((600, 5), seed=4)
    vh, Mh, v2h, M2h, Ah, Bh, Xh = made(v, M, v2, M2, A, B, X)
    check("1-D @ 2-D", get(v @ M), vh @ Mh, rtol=1e-12)
    check("1-D @ 2-D multi-chare", get(cnp.matmul(v2, M2)), v2h @ M2h, rtol=1e-12)
    check("1-D view @ 2-D view", get(v2[0:30] @ M2[30:60, 1:3]), v2h[0:30] @ M2h[30:60, 1:3], rtol=1e-12)
    check("1-D @ (K,1)", get(v @ M[:, 2:3]), vh @ Mh[:, 2:3], rtol=1e-12)
    r = A @ B
    if r.shape != (40, 1):
        raise AssertionError(f"(40,5) @ (5,1) has shape {r.shape}")
    check("(M,K) @ (K,1)", get(r), Ah @ Bh, rtol=1e-12)
    check("(M,K) @ (K,1) view", get(X @ M[:, 3:4]), Xh @ Mh[:, 3:4], rtol=1e-12)
    check("dot of views", get(cnp.dot(v2[10:20], v2[20:30])), np.array([v2h[10:20] @ v2h[20:30]]), rtol=1e-12)


# ---------------------------------------------------------------------------
# newaxis, cross-ndim assignment, view materialization, dim limit
# ---------------------------------------------------------------------------
@case
def newaxis():
    v = data((50,))
    A = data((7, 600))
    vh, Ah = made(v, A)
    for key, label in [((slice(None), None), "v[:, None]"), ((None, slice(None)), "v[None, :]"),
                       ((slice(3, 9), cnp.newaxis), "v[3:9, newaxis]"), (None, "v[None]"),
                       ((None, Ellipsis, None), "v[None, ..., None]")]:
        check(label, get(v[key]), vh[key])
    check("A[:, None, :]", get(A[:, None, :]), Ah[:, None, :])
    check("A[2:4, None, ::7]", get(A[2:4, None, ::7]), Ah[2:4, None, ::7])
    check("A[None][...]", get(A[None][0:1, 2:5, :]), Ah[None][0:1, 2:5, :])
    c = cnp.zeros((50,), dtype=np.float64)
    c[:, None] = v[:, None]
    check("setitem with newaxis", get(c), vh)
    raises("newaxis to 4-D", NotImplementedError, lambda: A[None, :, None, :])


@case
def cross_ndim_assignment():
    Z = cnp.zeros((4, 70, 9), dtype=np.float64)
    M = data((70, 3))
    Mh = made(M)
    exp = np.zeros((4, 70, 9))
    Z[1, :, 2:5] = M            # offset in a non-singleton dim (README Q8)
    exp[1, :, 2:5] = Mh
    Z[2, 10:40, 6:9] = M[5:35]   # view source + offsets
    exp[2, 10:40, 6:9] = Mh[5:35]
    Z[3:4, 0:70, 0:1] = M[:, 0:1]  # source with a size-1 dim
    exp[3:4, 0:70, 0:1] = Mh[:, 0:1]
    check("cross-ndim set_region", get(Z), exp)
    w = cnp.zeros((70,), dtype=np.float64)
    w[...] = Z[1:2, :, 3:4]      # higher -> lower from a view
    check("3-D view -> 1-D", get(w), Mh[:, 1])
    raises("bad assignment shape", ValueError, lambda: Z.__setitem__((0, slice(None), slice(0, 3)), data((3, 70))))


@case
def size1_assignment():
    # whole size-1 arrays and size-1 views as assignment sources
    x = cnp.full((1,), 5.0, dtype=np.float64)
    r = data((4,))
    rh = made(r)
    u = cnp.zeros((4,), dtype=np.float64)
    u[...] = x
    check("u[...] = x (size-1 array)", get(u), np.full(4, 5.0))
    o = cnp.zeros((4,), dtype=np.float64)
    o[2:3] = x
    check("o[2:3] = x", get(o), np.array([0.0, 0.0, 5.0, 0.0]))
    w = cnp.zeros((3, 2), dtype=np.float64)
    w[1:3, 1:2] = r[2:3]
    exp = np.zeros((3, 2))
    exp[1:3, 1:2] = rh[2]
    check("w[1:3, 1:2] = r[2:3] (size-1 view)", get(w), exp)
    check("r[2:3] + 0", get(r[2:3] + 0.0), rh[2:3])


@case
def views_into_diag_tile():
    A = data((9, 9))
    v = data((20,))
    Ah, vh = made(A, v)
    check("diag of view", get(cnp.diag(A[1:6, 2:8])), np.diag(Ah[1:6, 2:8]))
    check("diag of 1-D view", get(cnp.diag(v[3:8], 1)), np.diag(vh[3:8], 1))
    check("tile of view", get(cnp.tile(v[2:6], 3)), np.tile(vh[2:6], 3))


@case
def dim_limit():
    raises("zeros 4-D", NotImplementedError, lambda: cnp.zeros((2, 2, 2, 2)))
    raises("tile to 4-D", NotImplementedError, lambda: cnp.tile(data((2, 2)), (2, 1, 1, 1)))
    if ct_core.max_name < (1 << 31) - 1:
        raise AssertionError(f"max_name is {ct_core.max_name}")
    a = cnp.ones((3,), dtype=np.float64)
    if a.name < (1 << 16):
        raise AssertionError("test expected large frontend names")
    check("large names round trip", get(a + 1.0), np.full(3, 2.0))


def main():
    global iface
    iface = connect()
    failures = []
    selected = set(sys.argv[1:])
    for fn in CASES:
        if selected and fn.__name__ not in selected:
            continue
        try:
            fn()
            print(f"  PASS {fn.__name__}", flush=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            print(f"  FAIL {fn.__name__}: {msg}", flush=True)
            failures.append(f"{fn.__name__}: {msg}")
    print(f"[test_frontend] names used up to {ct_core.next_name} (max {ct_core.max_name})")
    if failures:
        print("FAIL test_frontend: " + "; ".join(failures))
        sys.exit(1)
    print("PASS test_frontend")


if __name__ == "__main__":
    main()
