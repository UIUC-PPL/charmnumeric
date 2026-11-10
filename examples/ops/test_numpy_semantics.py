"""NumPy semantics the frontend used to get wrong: assignment broadcasting, linked-copy
caches, negative steps with newaxis, batched matmul, dtypes of power/sum/prod/choose,
>3-D random/indices/take/flip, 0-d device indices, zero-size reductions."""

import warnings

import numpy as np

import charmnumeric as cnp
from _ops_common import Checker, connect

iface = connect()
C = Checker("test_numpy_semantics", iface)


def host(x):
    return C.fetch(x)


# ---- assignment broadcasts against x[key]'s shape ----

x = cnp.zeros((4, 4, 3))
col = np.arange(4.0).reshape(4, 1)
x[:, :, 0] = cnp.array(col)
ref = np.zeros((4, 4, 3))
ref[:, :, 0] = col
C.check("x[:, :, 0] = (4, 1)", x, ref)

x = cnp.zeros((2, 3, 5))
x[..., 0] = cnp.arange(3.0)
ref = np.zeros((2, 3, 5))
ref[..., 0] = np.arange(3.0)
C.check("x[..., 0] = (3,)", x, ref)

x = cnp.zeros((3, 4))
x[None, :, 1] = cnp.arange(3.0)
ref = np.zeros((3, 4))
ref[None, :, 1] = np.arange(3.0)
C.check("x[None, :, 1] = (3,)", x, ref)

# ---- caches of linked copies follow writes to their array ----

hX = np.arange(24.0).reshape(2, 3, 4)
X = cnp.array(hX)
w = cnp.array(np.ones(3))
M = X[0]
cnp.sum(M)
C.check("w @ X[0]", w @ M, np.ones(3) @ hX[0])
X[0] *= 2
hX[0] *= 2
C.check("w @ X[0] after X[0] *= 2", w @ M, np.ones(3) @ hX[0])

hA = np.arange(12.0).reshape(3, 4)
A = cnp.array(hA)
v = cnp.array(np.ones(4))
B = A.T
C.check("v @ A.T", v @ B, np.ones(4) @ hA.T)
A[0, 1] = 100
hA[0, 1] = 100
C.check("v @ A.T after A[0, 1] = 100", v @ B, np.ones(4) @ hA.T)

# ---- negative steps next to newaxis ----

x = cnp.zeros(4)
x[::-1, None] = cnp.array(np.arange(4.0).reshape(4, 1))
C.check("x[::-1, None] = (4, 1)", x, np.arange(4.0)[::-1])

x = cnp.arange(4)
r = x[::-1, None]
r[0, 0] = 100
C.check("r = x[::-1, None]; r[0, 0] = 100", x, np.array([0, 1, 2, 100]))

# ---- matmul: 3-D operands are batches ----

for sa, sb in (((5, 1, 4), (4, 6)), ((3, 3, 1), (3, 1, 4)), ((5, 4, 1), (1, 6)),
               ((2, 3, 4), (4,)), ((4,), (2, 4, 3))):
    ha = np.arange(np.prod(sa), dtype=np.float64).reshape(sa)
    hb = np.arange(np.prod(sb), dtype=np.float64).reshape(sb) - 3
    C.check(f"{sa} @ {sb}", cnp.array(ha) @ cnp.array(hb), ha @ hb)
    C.check(f"matmul{sa}{sb}", cnp.matmul(cnp.array(ha), cnp.array(hb)), np.matmul(ha, hb))

# ---- power ----

hu = np.array([200, 3], np.uint8)
C.check("uint8 ** 2.0", cnp.array(hu) ** 2.0, hu ** 2.0)
C.check("uint8 ** 2", cnp.array(hu) ** 2, hu ** 2)
hf = np.array([0.1, 3.0], np.float32)
C.check("float32 ** np.float64(2)", cnp.array(hf) ** np.float64(2.0), hf ** np.float64(2.0))
C.check("int64 ** np.float32(2)", cnp.arange(4) ** np.float32(2.0),
        np.arange(4) ** np.float32(2.0))

# ---- NumPy scalars are strong in comparisons ----

C.check("float32 == np.float64(0.1)", cnp.array(hf) == np.float64(0.1), hf == np.float64(0.1))
C.check("float32 == 0.1", cnp.array(hf) == 0.1, hf == 0.1)
C.check("np.float64(0.1) < float32", np.float64(0.1) < cnp.array(hf), np.float64(0.1) < hf)
hi8 = np.array([-3, 100], np.int8)
C.check("int8 < np.int64(1000)", cnp.array(hi8) < np.int64(1000), hi8 < np.int64(1000))

# ---- random arrays with more than 3 dims ----

g1 = cnp.random.default_rng(7).random((2, 3, 4, 5))
g2 = cnp.random.default_rng(7).random(120)
h1 = host(g1)
C.check("rand(2,3,4,5) has 120 distinct values", np.array(len(np.unique(h1))), np.array(120))
C.check("rand(2,3,4,5) == rand(120).reshape", g1, host(g2).reshape(2, 3, 4, 5))
h3 = host(cnp.random.default_rng(9).random((6, 4, 5)))
C.check("rand(2,3,4,5) == rand(6,4,5).reshape",
        cnp.random.default_rng(9).random((2, 3, 4, 5)), h3.reshape(2, 3, 4, 5))

# ---- sum/prod accumulate in dtype= and out='s dtype ----

C.check("prod(arange(1, 26), dtype=float64)", cnp.prod(cnp.arange(1, 26), dtype=np.float64),
        np.prod(np.arange(1, 26), dtype=np.float64), rtol=1e-12)
C.check("sum([0.5, 0.5], dtype=int64)", cnp.sum(cnp.array([0.5, 0.5]), dtype=np.int64),
        np.sum(np.array([0.5, 0.5]), dtype=np.int64))
o = cnp.zeros((), dtype=np.float64)
cnp.prod(cnp.arange(1, 26), out=o)
C.check("prod(arange(1, 26), out=float64)", o,
        np.prod(np.arange(1, 26), out=np.zeros(())), rtol=1e-12)
hs = np.arange(-6, 6, dtype=np.int8)
C.check("sum(int8, axis=0, dtype=float32)", cnp.sum(cnp.array(hs), axis=0, dtype=np.float32),
        np.sum(hs, axis=0, dtype=np.float32))

# ---- zero-size reductions ----

e = cnp.arange(10.0)
e = e[e > 100]
for name in ("sum", "prod", "any", "all", "count_nonzero"):
    C.check(f"{name} of empty", getattr(cnp, name)(e), getattr(np, name)(np.zeros(0)))
C.check("sum(zeros((0, 3)), axis=0)", cnp.sum(cnp.zeros((0, 3)), axis=0),
        np.sum(np.zeros((0, 3)), axis=0))
C.check("sum(zeros((0, 3)), axis=1)", cnp.sum(cnp.zeros((0, 3)), axis=1),
        np.sum(np.zeros((0, 3)), axis=1))
C.check("prod(int8 empty)", cnp.prod(cnp.zeros(0, dtype=np.int8)),
        np.prod(np.zeros(0, dtype=np.int8)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    C.check("mean of empty", cnp.mean(e), np.mean(np.zeros(0)))
C.raises("max of empty", ValueError, lambda: cnp.max(e))

# ---- fractional ddof ----

hv = np.arange(10.0) ** 1.5
for ddof in (0.5, 1.5):
    C.check(f"var ddof={ddof}", cnp.var(cnp.array(hv), ddof=ddof), np.var(hv, ddof=ddof),
            rtol=1e-12)
    C.check(f"std ddof={ddof}", cnp.std(cnp.array(hv), ddof=ddof), np.std(hv, ddof=ddof),
            rtol=1e-12)

# ---- 0-d device index arrays ----

hy = np.array([3.0, 9.0, 1.0])
y = cnp.array(hy)
hx1 = np.arange(5.0) * 10
C.check("x1[argmax(y)]", cnp.array(hx1)[cnp.argmax(y)], hx1[np.argmax(hy)])
hx2 = np.arange(12.0).reshape(3, 4)
C.check("x2[argmax(y)]", cnp.array(hx2)[cnp.argmax(y)], hx2[np.argmax(hy)])
x2 = cnp.array(hx2)
x2[cnp.argmax(y)] = -1.0
ref = hx2.copy()
ref[np.argmax(hy)] = -1.0
C.check("x2[argmax(y)] = -1", x2, ref)
x2 = cnp.array(hx2)
x2[cnp.argmax(y)] = cnp.array([5.0, 6.0, 7.0, 8.0])
ref = hx2.copy()
ref[np.argmax(hy)] = [5.0, 6.0, 7.0, 8.0]
C.check("x2[argmax(y)] = (4,)", x2, ref)
C.check("x2[:, argmax(y)]", cnp.array(hx2)[:, cnp.argmax(y)], hx2[:, np.argmax(hy)])
C.check("x2[argmax(y), [0, 3]]", cnp.array(hx2)[cnp.argmax(y), cnp.array([0, 3])],
        hx2[np.argmax(hy), [0, 3]])

# ---- indices(sparse=True) with more than 3 dims ----

for dims in ((2, 3, 4, 5), (2, 1, 3, 2, 2)):
    got = cnp.indices(dims, sparse=True)
    for k, (gk, rk) in enumerate(zip(got, np.indices(dims, sparse=True))):
        C.check(f"indices({dims}, sparse)[{k}]", gk, rk)

# ---- indexing edge cases ----

hx = np.arange(12.0).reshape(3, 4)
C.raises("x[5, ::-1] out of range", IndexError, lambda: cnp.array(hx)[5, ::-1])
C.raises("x[-4, ::-1] out of range", IndexError, lambda: cnp.array(hx)[-4, ::-1])
C.check("x[-1, ::-1]", cnp.array(hx)[-1, ::-1], hx[-1, ::-1])
C.raises("x[True] (scalar bool index)", NotImplementedError, lambda: cnp.array(hx)[True])
C.check("x[2:2:-1]", cnp.arange(5.0)[2:2:-1], np.arange(5.0)[2:2:-1])
C.check("x[1:1:-1, ::-1]", cnp.array(hx)[1:1:-1, ::-1], hx[1:1:-1, ::-1])
x = cnp.array(hx)
x[1:1:-1] = 7.0
C.check("x[1:1:-1] = 7 (no-op)", x, hx)

# ---- choose: scalar choices take part in the result dtype ----

ha = np.array([0, 1, 1])
C.check("choose int32 / 2.5", cnp.choose(cnp.array(ha), [cnp.array(np.array([1, 2, 3], np.int32)),
                                                          2.5]),
        np.choose(ha, [np.array([1, 2, 3], np.int32), 2.5]))
C.check("choose int64 / 2.5", cnp.choose(cnp.array(ha), [cnp.array(np.array([1, 2, 3])), 2.5]),
        np.choose(ha, [np.array([1, 2, 3]), 2.5]))
C.check("choose 1 / 2.5", cnp.choose(cnp.array(ha), [1, 2.5]), np.choose(ha, [1, 2.5]))
C.check("choose int8 / np.int64(3)", cnp.choose(cnp.array(ha), [
    cnp.array(np.array([1, 2, 3], np.int8)), np.int64(3)]),
        np.choose(ha, [np.array([1, 2, 3], np.int8), np.int64(3)]))

# ---- 0-d inputs to atleast_*d and tile ----

z = cnp.array(np.float64(4.5))
hz = np.float64(4.5)
C.check("atleast_1d(0-d)", cnp.atleast_1d(z), np.atleast_1d(hz))
C.check("atleast_2d(0-d)", cnp.atleast_2d(z), np.atleast_2d(hz))
C.check("atleast_3d(0-d)", cnp.atleast_3d(z), np.atleast_3d(hz))
C.check("atleast_2d(1-d)", cnp.atleast_2d(cnp.arange(3.0)), np.atleast_2d(np.arange(3.0)))
C.check("atleast_3d(1-d)", cnp.atleast_3d(cnp.arange(3.0)), np.atleast_3d(np.arange(3.0)))
C.check("tile(0-d, 3)", cnp.tile(z, 3), np.tile(hz, 3))
C.check("tile(0-d, (2, 2))", cnp.tile(z, (2, 2)), np.tile(hz, (2, 2)))

# ---- take, take_along_axis and flip on >3-D arrays ----

h4 = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
a4 = cnp.array(h4)
for axis in range(4):
    idx = np.array([0, 1, 1])
    C.check(f"take 4-D axis={axis}", cnp.take(a4, idx, axis=axis), np.take(h4, idx, axis=axis))
    C.check(f"take 4-D axis={axis} scalar", cnp.take(a4, 1, axis=axis), np.take(h4, 1, axis=axis))
    C.check(f"flip 4-D axis={axis}", cnp.flip(a4, axis), np.flip(h4, axis))
    ishape = list(h4.shape)
    ishape[axis] = 2
    hidx = (np.arange(np.prod(ishape)).reshape(ishape) * 7) % h4.shape[axis]
    C.check(f"take_along_axis 4-D axis={axis}", cnp.take_along_axis(a4, hidx, axis),
            np.take_along_axis(h4, hidx, axis))
C.check("flip 4-D all axes", cnp.flip(a4), np.flip(h4))
C.check("flip 4-D axes (0, 3)", cnp.flip(a4, (0, 3)), np.flip(h4, (0, 3)))

# ---- flip returns a copy ----

x = cnp.arange(3.0).reshape(3, 1)
f = cnp.flip(x, 1)
f[0, 0] = 9.0
C.check("flip with nothing to flip is a copy", x, np.arange(3.0).reshape(3, 1))

# ---- lstsq rank is an integer ----

C.check("lstsq rank", cnp.linalg.lstsq(cnp.array(np.eye(3)), cnp.ones(3))[2],
        np.array(np.linalg.lstsq(np.eye(3), np.ones(3))[2]))

# ---- a failed SVD returns all-NaN factors (NumPy raises LinAlgError) ----

hn = np.arange(12.0).reshape(4, 3)
hn[1, 1] = np.nan
for label, res in (("svd", cnp.linalg.svd(cnp.array(hn))),
                   ("svd thin", cnp.linalg.svd(cnp.array(hn), full_matrices=False)),
                   ("svd values", (cnp.linalg.svd(cnp.array(hn), compute_uv=False),)),
                   ("lstsq", cnp.linalg.lstsq(cnp.array(hn), cnp.ones(4))[::3])):
    for i, part in enumerate(res):
        hp = host(part)
        C.check(f"{label} of NaN input: factor {i} all NaN", np.isnan(hp), np.ones(hp.shape, bool))

# ---- fused temporaries read twice ----

L = cnp.arange(1000, dtype=np.int64).reshape(10, 100)
l2 = L % 1009
h = (l2 * l2) * 13 + l2 * 7 + 3
hL = np.arange(1000, dtype=np.int64).reshape(10, 100)
hl2 = hL % 1009
C.check("fused temporary read twice", h % 100 - 50, ((hl2 * hl2) * 13 + hl2 * 7 + 3) % 100 - 50)

C.finish()
