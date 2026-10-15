#!/usr/bin/env python
"""Frontend ops and fixes checked against NumPy: joins, out=, take, newaxis, assignment.

Inputs are generated natively and fetched with .get(); the NumPy reference uses the fetched values.
Env: CNUM_HOST / CNUM_PORT / CNUM_ODF select the server; CNUM_SIM=1 uses the simulator.
"""

import itertools
import os
import sys
import traceback

import numpy as np

import charmtyles.core as ct_core
from charmtyles.ast import get_execution_state

# Start names near the int32 limit to exercise large names on the wire.
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

    F = data((20, 12))
    Fh = made(F)
    mixed = cnp.concatenate((A, F))
    check("concatenate int64 + float64 -> float64", get(mixed), np.concatenate((Ah, Fh)))
    assert mixed.dtype == np.float64
    raises("ndim mismatch", ValueError, lambda: cnp.concatenate((A, data((20,), dtype=np.int64))))
    raises("shape mismatch", ValueError, lambda: cnp.concatenate((A, B), axis=0))


@case
def hstack_vstack():
    N, F = 600, 5
    X = data((N, F))
    Xh = made(X)
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
    s1, s2 = data((2, 3, 4)), data((2, 3, 4), seed=2)
    check("stack to 4-D", get(cnp.stack((s1, s2))), np.stack(made(s1, s2)))
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
    o32 = cnp.zeros((300,), dtype=np.float32)
    cnp.add(a, b, out=o32)  # same-kind cast, as NumPy's ufunc out=
    check("out dtype cast", get(o32), (get(a) + get(b)).astype(np.float32))
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
    check("take scalar on 1-D", get(cnp.take(a, 17)), np.take(ah, 17))
    check("take mode=wrap", get(cnp.take(a, [101, -105, 3], mode="wrap")), np.take(ah, [101, -105, 3], mode="wrap"))
    check("take mode=clip", get(cnp.take(a, [101, -5, 3], mode="clip")), np.take(ah, [101, -5, 3], mode="clip"))
    raises("take out of bounds", IndexError, lambda: cnp.take(a, [100]))
    check("take device indices", get(cnp.take(a, cnp.array([5, 0, 99, -1]))), np.take(ah, [5, 0, 99, -1]))


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
    check("take to 4-D", get(cnp.take(T, [[0, 1]], axis=1)), np.take(Th, [[0, 1]], axis=1))


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
    check("dot of views", get(cnp.dot(v2[10:20], v2[20:30])), np.asarray(v2h[10:20] @ v2h[20:30]),
          rtol=1e-12)


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
    check("newaxis to 4-D", get(A[None, :, None, :]), Ah[None, :, None, :])


@case
def cross_ndim_assignment():
    Z = cnp.zeros((4, 70, 9), dtype=np.float64)
    M = data((70, 3))
    Mh = made(M)
    exp = np.zeros((4, 70, 9))
    Z[1, :, 2:5] = M            # offset in a non-singleton dim
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
    check("zeros 4-D", get(cnp.zeros((2, 2, 2, 2))), np.zeros((2, 2, 2, 2)))
    raises("tile to 4-D", NotImplementedError, lambda: cnp.tile(data((2, 2)), (2, 1, 1, 1)))
    if ct_core.max_name < (1 << 31) - 1:
        raise AssertionError(f"max_name is {ct_core.max_name}")
    a = cnp.ones((3,), dtype=np.float64)
    if a.name < (1 << 16):
        raise AssertionError("test expected large frontend names")
    check("large names round trip", get(a + 1.0), np.full(3, 2.0))


# ---------------------------------------------------------------------------
# NumPy semantics used by the cuPyNumeric example programs
# ---------------------------------------------------------------------------
@case
def zero_d_results():
    x = data((300001,))
    A = data((37, 41))
    B = data((5, 7, 9))
    xh, Ah, Bh = made(x, A, B)
    s = cnp.sum(A)
    sh = Ah.sum()
    for label, got, exp in [("sum", s, sh), ("dot", x @ x, xh @ xh), ("max", B.max(), Bh.max()),
                            ("argmin", cnp.argmin(B), np.argmin(Bh)),
                            ("norm", cnp.linalg.norm(x), np.linalg.norm(xh)),
                            ("matrix inf-norm", cnp.linalg.norm(A, np.inf),
                             np.linalg.norm(Ah, np.inf))]:
        if got.shape != () or got.ndim != 0:
            raise AssertionError(f"{label}: shape {got.shape}")
        check(label, get(got), np.asarray(exp), rtol=1e-12)
    # 0-d with arrays of every rank, and with views (size-1 operand of another rank)
    check("A * s", get(A * s), Ah * sh, rtol=1e-12)
    check("s - B", get(s - B), sh - Bh, rtol=1e-12)
    check("B[1:3, 2:5] / s", get(B[1:3, 2:5] / s), Bh[1:3, 2:5] / sh, rtol=1e-12)
    check("x + s", get(x + s), xh + sh, rtol=1e-12)
    t = s / cnp.sum(x)
    check("0-d / 0-d", get(t), np.asarray(sh / xh.sum()), rtol=1e-12)
    for name in ("sqrt", "exp", "tanh", "log", "abs", "floor"):
        th = np.asarray(sh / xh.sum())
        check(f"{name} of 0-d", get(getattr(cnp, name)(t)), getattr(np, name)(th), rtol=1e-12)
    check("0-d into (1, 1) region", get(A[2:3, 4:5] * 0 + s), np.full((1, 1), sh), rtol=1e-12)
    # host conversions fetch the value
    if (abs(float(s) - sh) > 1e-9 * abs(sh) or not bool(s > sh - 1)
            or int(cnp.sum(cnp.ones((5,)))) != 5):
        raise AssertionError("float/bool/int of a 0-d array")
    if str(cnp.sum(cnp.ones((4,)))) != str(np.sum(np.ones(4))) or f"{s:.3f}" != f"{sh:.3f}":
        raise AssertionError("str/format of a 0-d array")
    raises("bool of a larger array", ValueError, lambda: bool(x > 0))
    if not np.allclose(np.asarray(A), Ah):
        raise AssertionError("__array__")
    z = cnp.array(2.5)
    if z.shape != () or float(z) != 2.5:
        raise AssertionError("array(scalar)")
    z[...] = 4.0
    z += 1
    check("0-d setitem / in-place", get(z), np.asarray(5.0))


@case
def integer_indices():
    A = data((37, 41))
    B = data((5, 7, 9))
    v = data((50,))
    Ah, Bh, vh = made(A, B, v)
    for key in [3, -1, (2, slice(1, 30, 3)), (slice(None), 7), (slice(2, 9), -4)]:
        check(f"A[{key}]", get(A[key]), Ah[key])
    for key in [1, (slice(None), 2), (2, slice(None), slice(1, 4)), (Ellipsis, 3), (1, 2),
                (slice(None, None, -1), 2), (1, None, slice(None)), (4, 6, 8)]:
        check(f"B[{key}]", get(B[key]), Bh[key])
    check("v[7]", get(v[7]), vh[7])
    check("A[2][5]", get(A[2][5]), Ah[2][5])
    check("take scalar along axis", get(cnp.take(B, 3, axis=1)), np.take(Bh, 3, axis=1))
    check("A[2:9, 3:30:4].T", get(A[2:9, 3:30:4].T), Ah[2:9, 3:30:4].T)
    check("B[1].T", get(B[1].T), Bh[1].T)
    check("B[:, 2:5, 1:8].transpose(2, 0, 1)", get(B[:, 2:5, 1:8].transpose(2, 0, 1)),
          Bh[:, 2:5, 1:8].transpose(2, 0, 1))
    # writes through an integer-indexed result reach the array
    r = A[4]
    r += 1.0
    Ah = Ah.copy()
    Ah[4] += 1.0
    check("A[4] += 1 via a row", get(A), Ah)
    B[2] += B[3]
    Bh = Bh.copy()
    Bh[2] += Bh[3]
    check("B[2] += B[3]", get(B), Bh)
    p = A[:, 5]
    p[1:3] = -7.0
    Ah[1:3, 5] = -7.0
    check("column write-through", get(A), Ah)
    e = v[3]
    e += 100.0  # a scalar result: no write back (NumPy returns a copy)
    check("v[3] scalar copy", get(v), vh)


@case
def shifted_assignment_results():
    # Write into a region that spans another chare.
    V = cnp.full((16, 8, 12), 0.1, dtype=np.float32)
    P = cnp.zeros((112, 8, 12))
    d = V * 2.0
    P[96:112] = d + 1.0
    Ph = np.zeros((112, 8, 12))
    Ph[96:112] = np.float32(0.1) * 2.0 + 1.0
    check("P[96:112] = d + 1", get(P), Ph)
    check("kept d", get(d), np.full((16, 8, 12), np.float32(0.1) * 2.0))
    check("d reused", get(d * 3.0), np.full((16, 8, 12), np.float32(0.1) * 6.0))
    F = cnp.zeros((7, 16, 8, 12), dtype=np.float32)  # 4-D: F[6] is a shifted storage view
    e = -1 * (V * cnp.sqrt(d))
    F[6][...] = e * d
    eh = -1 * (np.float32(0.1) * np.sqrt(np.float32(0.1) * 2.0))
    Fh = np.zeros((7, 16, 8, 12), dtype=np.float32)
    Fh[6] = eh * (np.float32(0.1) * 2.0)
    check("F[6][...] = e * d", get(F), Fh, rtol=1e-6)
    check("kept e", get(e), np.full((16, 8, 12), eh), rtol=1e-12)
    # the region is far larger than the result, at an offset off the tile grid
    W = cnp.full((365, 8, 12), 0.5)
    Q = cnp.zeros((2555, 8, 12))
    g = W * 3.0
    Q[2190:] = g - 1.0
    Qh = np.zeros((2555, 8, 12))
    Qh[2190:] = 0.5
    check("Q[2190:] = g - 1", get(Q), Qh)
    check("kept g", get(g), np.full((365, 8, 12), 1.5))


@case
def numpy_interop():
    x = data((40,))
    xh = made(x)
    y = np.sqrt(x)  # NumPy ufunc on a device array runs natively
    if not isinstance(y, cnp.Array):
        raise AssertionError(f"np.sqrt returned {type(y).__name__}")
    check("np.sqrt(x)", get(y), np.sqrt(xh), rtol=1e-14)
    check("np.float32 * x", get(np.float32(2.0) * x), np.float32(2.0) * xh)
    hx = np.ones(40)
    hx += x  # NumPy in-place with a device operand
    check("host += device", hx, 1.0 + xh)
    check("numpy.cumsum(x)", np.cumsum(x), np.cumsum(xh), rtol=1e-12)
    check("arange(4.0)", get(cnp.arange(4.0)), np.arange(4.0))
    check("arange(np.float32(1), 5)", get(cnp.arange(np.float32(1), 5)),
          np.arange(np.float32(1), 5))
    check("arange(0, 3, 0.5)", get(cnp.arange(0, 3, 0.5)), np.arange(0, 3, 0.5))
    hn = np.array([1.0, np.nan, np.inf, -np.inf, 1e-9, 5.0])
    hm = np.array([1.0 + 1e-6, np.nan, np.inf, np.inf, 0.0, 5.1])
    for kw in ({}, {"equal_nan": True}, {"rtol": 0.05}, {"atol": 1e-5}):
        check(f"isclose {kw}", get(cnp.isclose(cnp.array(hn), cnp.array(hm), **kw)),
              np.isclose(hn, hm, **kw))
        if bool(cnp.allclose(cnp.array(hn), hm, **kw)) != np.allclose(hn, hm, **kw):
            raise AssertionError(f"allclose {kw}")
    hi = np.arange(6)
    check("isclose int", get(cnp.isclose(cnp.array(hi), cnp.array(hi + 1), atol=1)),
          np.isclose(hi, hi + 1, atol=1))
    counts = cnp.array(np.array([0, 3, 1], dtype=np.int64))
    check("maximum(int64, uint64)", get(cnp.maximum(counts, cnp.ones((1,), dtype=np.uint64))),
          np.maximum(np.array([0, 3, 1]), np.ones((1,), dtype=np.uint64)))
    big = np.array([2**63 + 5, 7], dtype=np.uint64)
    check("uint64 above 2**63", get(cnp.array(big) + np.uint64(1)), big + np.uint64(1))
    cnp.synchronize(iface)


@case
def transpose_is_linked_view():
    # transpose results write through to the base, and follow later writes to it
    rng = np.random.default_rng(1)
    h = rng.integers(-4, 5, (33, 33)).astype(np.int64)
    base = cnp.array(h)
    x = base.T
    x -= x
    check("isub on transpose", get(base), np.zeros_like(h))
    h = rng.integers(-4, 5, (33, 513)).astype(np.int64)
    base = cnp.array(h)
    x = cnp.transpose(base, (1, 0))
    x **= 0
    check("ipow on transpose", get(base), np.ones_like(h))
    h = rng.uniform(-2, 2, (7, 2, 9))
    base = cnp.array(h)
    t = base.transpose(1, 2, 0)
    t[-2, 3:4] = -3.0
    e = h.copy()
    e.transpose(1, 2, 0)[-2, 3:4] = -3.0
    check("setitem on transpose", get(base), e)
    check("transpose after write", get(t), e.transpose(1, 2, 0))
    base[0] = 5.0
    e[0] = 5.0
    check("transpose refreshed", get(t), e.transpose(1, 2, 0))
    s = cnp.swapaxes(base, 0, 2)
    s[1:3, 0, 1] = 9.0
    e.swapaxes(0, 2)[1:3, 0, 1] = 9.0
    check("swapaxes write", get(base), e)


@case
def inplace_bitwise_size1():
    for shape in ((1, 1), (1, 1, 1), (37,), (5, 7)):
        h = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape) - 3
        for name in ("__ior__", "__ixor__", "__iand__"):
            x = cnp.array(h)
            getattr(x, name)(5)
            e = h.copy()
            getattr(e, name)(5)
            check(f"{name} {shape}", get(x), e)
        x = cnp.array(h)
        x ^= x
        check(f"x ^= x {shape}", get(x), np.zeros_like(h))
    m = cnp.array(np.array([True, False, True]))
    m |= cnp.array(np.array([False, False, True]))
    check("bool |=", get(m), np.array([True, False, True]))


@case
def shifts():
    rng = np.random.default_rng(2)
    for dt in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        info = np.iinfo(dt)
        bits = info.bits
        h = rng.integers(max(info.min, -(2**62)), min(info.max, 2**62), 100, dtype=np.int64).astype(dt)
        n = rng.integers(0, bits + 3, 100).astype(dt)  # includes counts >= width
        a, c = cnp.array(h), cnp.array(n)
        check(f"<< {np.dtype(dt)}", get(a << c), h << n)
        check(f">> {np.dtype(dt)}", get(a >> c), h >> n)
        check(f"<< scalar {np.dtype(dt)}", get(a << 3), h << 3)
        check(f">> scalar {np.dtype(dt)}", get(a >> 2), h >> 2)
        check(f"left_shift {np.dtype(dt)}", get(cnp.left_shift(a, c)), np.left_shift(h, n))
        check(f"right_shift {np.dtype(dt)}", get(cnp.right_shift(a, c)), np.right_shift(h, n))
    h = np.arange(-20, 20, dtype=np.int64)
    a = cnp.array(h)
    check("reflected <<", get(3 << cnp.array(np.arange(0, 70, dtype=np.int64))),
          3 << np.arange(0, 70, dtype=np.int64))
    check("reflected >>", get(-1000 >> cnp.array(np.arange(0, 70, dtype=np.int64))),
          -1000 >> np.arange(0, 70, dtype=np.int64))
    n = np.array([-1, -5, 64, 100, 63, 0])
    check("negative counts <<", get(cnp.array(np.ones(6, np.int64)) << cnp.array(n)), np.ones(6, np.int64) << n)
    check("negative counts >>", get(cnp.array(-np.ones(6, np.int64) * 8) >> cnp.array(n)),
          -np.ones(6, np.int64) * 8 >> n)
    x = cnp.array(h)
    x <<= 2
    y = h.copy()
    y <<= 2
    check("<<=", get(x), y)
    x >>= 1
    y >>= 1
    check(">>=", get(x), y)
    one = cnp.array(np.array([[-7]], dtype=np.int64))
    check("size-1 <<", get(one << 3), np.array([[-7]]) << 3)
    b = cnp.array(np.array([True, False]))
    check("bool << int", get(b << 2), np.array([True, False]) << 2)
    h4 = np.arange(24, dtype=np.int32).reshape(2, 3, 2, 2) - 7
    a4 = cnp.array(h4)
    check("4-D <<", get(a4 << 2), h4 << 2)
    check("4-D bitwise_right_shift", get(cnp.bitwise_right_shift(a4, a4 & 3)), np.right_shift(h4, h4 & 3))
    a4 |= 1
    check("4-D |=", get(a4), h4 | 1)
    raises("float <<", TypeError, lambda: cnp.array(np.ones(3)) << 1)
    raises("float >>", TypeError, lambda: cnp.right_shift(cnp.array(np.ones(3)), 1))


@case
def where_python_scalar_is_weak():
    c = np.arange(6.0).reshape(2, 3) - 2
    for shape in ((2, 3), (1, 2, 3, 1)):
        x = cnp.array(c.reshape(shape))
        r = cnp.where(x, -0.22, np.float32(0.45))
        check(f"where pyfloat/np32 {shape}", get(r), np.where(c.reshape(shape), -0.22, np.float32(0.45)))
        r = cnp.where(x, np.float32(0.45), -0.22)
        check(f"where np32/pyfloat {shape}", get(r), np.where(c.reshape(shape), np.float32(0.45), -0.22))


@case
def folded_take_and_dot():
    h = np.arange(9.0).reshape(9, 1, 1, 1)
    x = cnp.array(h)
    check("take folded", get(cnp.take(x, [3])), np.take(h, [3]))
    check("take folded idx", get(cnp.take(x, cnp.array(np.array([8, 0, 4])))), np.take(h, [8, 0, 4]))
    a = np.arange(4.0).reshape(2, 2, 1)
    b = np.array([[3]])
    check("dot 3-D x 2-D", get(cnp.dot(cnp.array(a), cnp.array(b))), np.dot(a, b))
    a = np.arange(60.0).reshape(3, 4, 5)
    b = np.arange(30.0).reshape(5, 6)
    check("dot 3-D x 2-D", get(cnp.array(a).dot(cnp.array(b))), np.dot(a, b))
    b = np.arange(2 * 5 * 3.0).reshape(2, 5, 3)
    check("dot 3-D x 3-D", get(cnp.dot(cnp.array(a), cnp.array(b))), np.dot(a, b))
    v = np.arange(5.0)
    check("dot 3-D x 1-D", get(cnp.dot(cnp.array(a), cnp.array(v))), np.dot(a, v))
    check("dot 1-D x 3-D", get(cnp.dot(cnp.array(v), cnp.array(b))), np.dot(v, b))
    a4 = np.arange(24.0).reshape(2, 3, 2, 2)
    b2 = np.arange(6.0).reshape(2, 3)
    check("dot 4-D x 2-D", get(cnp.dot(cnp.array(a4), cnp.array(b2))), np.dot(a4, b2))


@case
def negative_step_folded():
    h = np.arange(6 * 5 * 4 * 3).reshape(3, 2, 5, 4, 3).astype(np.int64)
    x = cnp.array(h)
    for key in ((slice(2, 0, -1),), (slice(None), slice(None, None, -1)),
                (slice(None), 1, slice(3, 0, -2)), (Ellipsis, slice(None, None, -1)),
                (slice(None, None, -1), slice(None, None, -1), 2, slice(None), slice(2, None, -1))):
        check(f"get {key}", get(x[key]), h[key])
    y = x[:][2:1:-1]
    check("chained", get(y), h[:][2:1:-1])
    e = h.copy()
    x[::-1, 1, 1:4] = 7
    e[::-1, 1, 1:4] = 7
    check("set negative step", get(x), e)
    x[:, :, ::-2] = cnp.array(np.arange(3 * 2 * 3 * 4 * 3).reshape(3, 2, 3, 4, 3).astype(np.int64))
    e[:, :, ::-2] = np.arange(3 * 2 * 3 * 4 * 3).reshape(3, 2, 3, 4, 3)
    check("set negative step array", get(x), e)
    z = cnp.array(np.arange(6).reshape(3, 2, 1, 1).astype(np.int64))
    check("chained 4-D", get(z[:][2:1:-1]), np.arange(6).reshape(3, 2, 1, 1)[:][2:1:-1])


@case
def zero_d_operands():
    v = np.arange(1000.0) / 7
    s = cnp.amax(cnp.array(v), axis=-1)
    check("inner 0-d x 1-D", get(cnp.inner(s, cnp.array(v))), np.inner(v.max(), v))
    check("inner 1-D x 0-d", get(cnp.inner(cnp.array(v), s)), np.inner(v, v.max()))
    check("inner 0-d x 0-d", get(cnp.inner(s, s)), np.inner(v.max(), v.max()))
    t = cnp.amin(cnp.array(np.arange(7)), axis=0)
    check("x[None]", get(t[None]), np.array(0)[None])
    check("x[None, None]", get(t[None, None]), np.array(0)[None, None])
    check("x[..., None]", get(t[..., None]), np.array(0)[..., None])
    t[None] = 4
    check("x[None] = v", get(t), np.array(4))
    check("transpose(0-d, ())", get(cnp.transpose(t, ())), np.array(4))
    check("0-d.transpose()", get(t.transpose()), np.array(4))
    check("0-d.T", get(t.T), np.array(4))
    w = np.arange(5.0)
    check("einsum ,e->e", get(cnp.einsum(",e->e", t.astype(np.float64), cnp.array(w))), np.einsum(",e->e", 4.0, w))
    a = np.arange(33 * 70.0).reshape(33, 1, 70)
    x = cnp.transpose(cnp.array(a), (1, 0, 2))
    check("einsum abc,ebc->ae", get(cnp.einsum("abc,ebc->ae", x, x)),
          np.einsum("abc,ebc->ae", a.transpose(1, 0, 2), a.transpose(1, 0, 2)), rtol=1e-9)
    a = np.arange(9 * 5 * 13.0).reshape(9, 5, 1, 13)
    b = np.arange(9 * 5 * 9.0).reshape(9, 5, 1, 9)
    check("einsum abcd,dfga->fca", get(cnp.einsum("abcd,dfga->fca", cnp.array(a[..., 2:11]), cnp.array(b))),
          np.einsum("abcd,dfga->fca", a[..., 2:11], b), rtol=1e-9)


@case
def mask_with_slices_and_newaxis():
    rng = np.random.default_rng(3)
    h = rng.integers(-5, 5, (6, 7, 5)).astype(np.int64)
    x = cnp.array(h)
    m = rng.integers(0, 2, 6).astype(bool)
    m2 = rng.integers(0, 2, 7).astype(bool)
    keys = [
        (m, slice(6, 1, -2), slice(0, 3, 2)),
        (slice(1, 5), m2, slice(None, None, -1)),
        (m, None),
        (None, m),
        (m, None, slice(1, 4)),
        (slice(None), m2, None),
        (m, slice(None), None, 2),
        (Ellipsis, None, slice(1, 3)),
    ]
    for key in keys:
        hk = tuple(k if not isinstance(k, np.ndarray) else k for k in key)
        ck = tuple(cnp.array(k) if isinstance(k, np.ndarray) else k for k in key)
        check(f"get {key}", get(x[ck]), h[hk])
    for key, val in (((m, None), 3), ((slice(None), m2, None), 6), ((m, slice(6, 1, -2), slice(0, 3, 2)), -1)):
        hk = key
        ck = tuple(cnp.array(k) if isinstance(k, np.ndarray) else k for k in key)
        y = cnp.array(h)
        e = h.copy()
        y[ck] = val
        e[hk] = val
        check(f"set {key}", get(y), e)
    # >3-D arrays: index arrays and masks with slices
    h4 = rng.integers(-5, 5, (7, 7, 3, 3)).astype(np.int64)
    x4 = cnp.array(h4)
    m4 = rng.integers(0, 2, 7).astype(bool)
    for key in ((m4, slice(6, 3, -2), slice(0, 3, 2)), (slice(None), m4, 1), (m4, None, slice(None), 2)):
        ck = tuple(cnp.array(k) if isinstance(k, np.ndarray) else k for k in key)
        check(f"get 4-D {key}", get(x4[ck]), h4[key])
    idx = np.array([5, -1, 0])
    check("get 4-D fancy", get(x4[cnp.array(idx), :, 1:]), h4[idx, :, 1:])
    y4 = cnp.array(h4)
    e4 = h4.copy()
    y4[m4, :, 1] = 9
    e4[m4, :, 1] = 9
    check("set 4-D mask", get(y4), e4)
    z = cnp.array(np.arange(4.0).reshape(1, 1, 2, 2))
    r = z[cnp.array(np.array([True])), slice(6, 3, -2), slice(0, 3, 2)]
    check("empty 4-D", get(r), np.arange(4.0).reshape(1, 1, 2, 2)[np.array([True]), 6:3:-2, 0:3:2])


@case
def dot_after_partition_growth():
    # a 1-D dot, then dots over larger arrays (the partition gains chares), then small ones again
    for n in (33, 262145, 700001, 33, 300000, 7):
        h = (np.arange(n) % 13 - 6).astype(np.float64)
        x = cnp.array(h)
        check(f"dot n={n}", get(cnp.dot(x, x)), np.dot(h, h))
        check(f"inner strided n={n}", get(cnp.inner(x[::2], x[::2])), np.inner(h[::2], h[::2]))
    h32 = (np.arange(524290) % 7 - 3).astype(np.float32)
    v = cnp.array(h32)[0:524289:2]
    check("inner float32 view", get(cnp.inner(v, v)), np.inner(h32[0:524289:2], h32[0:524289:2]))


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
