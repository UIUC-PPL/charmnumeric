"""Integer-index results (lazy linked copies: elementwise ops and assignments run on the
source view at storage rank) vs NumPy: values, write-back, later parent writes, mixing."""

import numpy as np

import charmnumeric as cnp
from _ops_common import Checker, connect

iface = connect()
C = Checker("test_lazy_index", iface)
TOL = dict(rtol=1e-12, atol=1e-12)
rng = np.random.default_rng(0)

for shape in [(7, 9), (600, 7), (37, 41)]:
    s = f" {shape}"
    A, B = rng.random(shape), rng.random(shape)
    a, b = cnp.array(A), cnp.array(B)
    C.check("col" + s, a[1:-1, 0], A[1:-1, 0])
    C.check("row" + s, a[2, :], A[2, :])
    C.check("expr" + s, (a[1:-1, 0] - a[1:-1, -2]) * 2 + b[1:-1, -1] / 3,
            (A[1:-1, 0] - A[1:-1, -2]) * 2 + B[1:-1, -1] / 3, **TOL)
    t, T = cnp.array(A), A.copy()
    t[1:-1, -1] = (t[1:-1, 0] + t[1:-1, -2]) * 0.5 - b[1:-1, 0]
    T[1:-1, -1] = (T[1:-1, 0] + T[1:-1, -2]) * 0.5 - B[1:-1, 0]
    C.check("assign" + s, t, T, **TOL)
    t[-1, :] = t[-2, :]
    T[-1, :] = T[-2, :]
    C.check("row copy" + s, t, T, **TOL)
    v = t[1:-1, 0]
    t[1:-1, 0] = 5.0
    T[1:-1, 0] = 5.0
    C.check("view after parent write" + s, v, T[1:-1, 0])
    w = t[3, :]
    w += 1
    T[3, :] += 1
    C.check("in-place write-back" + s, t, T, **TOL)
    C.check("in-place view" + s, w, T[3, :], **TOL)
    m = a[1:-1, 0] + 1
    C.check("mixed with 1-d" + s, m + cnp.array(np.arange(shape[0] - 2.0)),
            A[1:-1, 0] + 1 + np.arange(shape[0] - 2.0), **TOL)
    C.check("sum" + s, cnp.sum(a[2, :] * 2), np.sum(A[2, :] * 2), rtol=1e-12)
    r, R = a[1:-1, 0] * 3, A[1:-1, 0] * 3
    r2, R2 = r.copy(), R.copy()
    r[0] = 100.0
    R[0] = 100.0
    C.check("result write" + s, r, R, **TOL)
    C.check("result copy" + s, r2, R2, **TOL)
    d = a[:, 3]
    C.check("reshape" + s, d.reshape(shape[0], 1), A[:, 3].reshape(shape[0], 1))
    C.check("slice of copy" + s, d[2:5], A[2:5, 3])
    C.check("negative step" + s, a[::-1, 2], A[::-1, 2])
    C.check("newaxis" + s, a[1, None, :], A[1, None, :])
    k, K = cnp.array(A), A.copy()
    k[2, :] = k[3, :] * 2 + k[4, :]
    K[2, :] = K[3, :] * 2 + K[4, :]
    C.check("row expr" + s, k, K, **TOL)
    q = k[2, :]
    k[2, :] = q + 1
    K[2, :] = K[2, :] + 1
    C.check("self row" + s, k, K, **TOL)
    C.check("astype" + s, a[1, :].astype(np.float32), A[1, :].astype(np.float32))

X = rng.random((4, 5, 6))
x = cnp.array(X)
C.check("3d plane product", x[1, :, 2:5] * x[2, :, 2:5], X[1, :, 2:5] * X[2, :, 2:5], **TOL)
C.check("3d plane", x[0], X[0])
u, U = cnp.array(X), X.copy()
u[3, :, 0] = u[2, :, 1] - u[1, :, 2]
U[3, :, 0] = U[2, :, 1] - U[1, :, 2]
C.check("3d assign", u, U, **TOL)

# masks on lazy copies; overlapping self-assignment through a lazy rhs
M = rng.random((6, 10)) - 0.5
m = cnp.array(M)
y, Y = m[2], M[2]
C.check("mask of lazy copy", y[y > 0], Y[Y > 0])
y, Y = m[1], M[1].copy()
y[y > 0] = y[y > 0] * 2
Y[Y > 0] = Y[Y > 0] * 2
C.check("mask setitem on lazy copy", y, Y)
for shape in [(6, 10), (300, 400)]:
    T = rng.random(shape)
    t, s = cnp.array(T), f" {shape}"
    t[2, 1:] = t[2, :-1]
    T[2, 1:] = T[2, :-1]
    t[3, :-1] = t[3, 1:]
    T[3, :-1] = T[3, 1:]
    t[4, ::-1] = t[4, :]
    T[4, ::-1] = T[4, :]
    C.check("overlapping self-assign" + s, t, T)
C.finish()
