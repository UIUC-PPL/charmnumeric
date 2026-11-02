"""GATHER / SCATTER paths vs NumPy: slices next to index arrays (both step signs), arrays
split over several chares along indexed and sliced dims, last-write-wins with array
values, values aliasing the target, every element size, and many destination chares."""

import itertools

import numpy as np

import charmnumeric as cnp
from _ops_common import Checker, connect, ints

iface = connect()
C = Checker("test_index_paths", iface)
rng = np.random.default_rng(7)


def dev(host):
    return cnp.array(np.ascontiguousarray(host))


# ---- take_pairs: a tall array, a small 2-D index along axis 1 ----
for m, n in [(9000, 32), (2500, 7)]:
    pairs = np.array(list(itertools.combinations(range(n), 2)))
    a = cnp.arange(m * n, dtype=np.int32).reshape((m, n))
    ah = np.arange(m * n, dtype=np.int32).reshape((m, n))
    C.check(f"take pairs axis=1 ({m}, {n})", a.take(pairs, axis=1), ah.take(pairs, axis=1))
    C.check(f"take pairs sum ({m}, {n})", np.array(int(a.take(pairs, axis=1).sum())),
            np.array(int(ah.take(pairs, axis=1).sum())))
    rows = rng.integers(-m, m, size=(40, 3))
    C.check(f"take rows axis=0 ({m}, {n})", a.take(rows, axis=0), ah.take(rows, axis=0))
    C.check(f"take dev rows ({m}, {n})", a.take(dev(rows), axis=0, mode="wrap"),
            ah.take(rows, axis=0, mode="wrap"))
    big = rng.integers(-3 * m, 3 * m, size=1000)
    C.check(f"take clip axis=0 ({m}, {n})", a.take(dev(big), axis=0, mode="clip"),
            ah.take(big, axis=0, mode="clip"))

# ---- slices with both step signs next to index arrays ----
x = ints((700, 9), -1000, 1000, dtype=np.int64)
xh = C.fetch(x)
cols = rng.integers(-9, 9, size=5)
rows = rng.integers(-700, 700, size=33)
C.check("x[::-1, cols]", x[::-1, cols], xh[::-1, cols])
C.check("x[650:3:-7, dev cols]", x[650:3:-7, dev(cols)], xh[650:3:-7, cols])
C.check("x[rows, ::-2]", x[rows, ::-2], xh[rows, ::-2])
C.check("x[rows, 7:1:-3]", x[rows, 7:1:-3], xh[rows, 7:1:-3])
C.check("x[1:600:4][dev rows % 150]", x[1:600:4][dev(rows % 150)], xh[1:600:4][rows % 150])
t, th = x.copy(), xh.copy()
t[::-3, cols] = -5
th[::-3, cols] = -5
C.check("t[::-3, cols] = s", t, th)
vals = rng.integers(-9, 9, size=(len(range(699, -1, -2)), 5))
t[::-2, dev(cols)] = dev(vals)
th[::-2, cols] = vals
C.check("t[::-2, dev cols] = array (dup cols)", t, th)
t[dev(rows), 8:0:-2] = dev(rng.integers(0, 5, size=(33, 4)))
th[rows, 8:0:-2] = C.fetch(t)[rows, 8:0:-2]
C.check("t[dev rows, 8:0:-2] = array (dup rows)", t, th)

# ---- 3-D: index arrays around a slice, slices around an index ----
y = ints((60, 13, 41), -99, 99, dtype=np.int32)
yh = C.fetch(y)
i0 = rng.integers(0, 41, size=60)
C.check("y[:, 2:11:3, i]", y[:, 2:11:3, i0[:7]], yh[:, 2:11:3, i0[:7]])
C.check("y[i, :, i]", y[i0 % 60, :, i0], yh[i0 % 60, :, i0])
C.check("y[::-1, i, ::2]", y[::-1, i0[:5] % 13, ::2], yh[::-1, i0[:5] % 13, ::2])
u, uh = y.copy(), yh.copy()
u[i0 % 60, :, i0] = 7
uh[i0 % 60, :, i0] = 7
C.check("u[i, :, i] = s", u, uh)
w = rng.integers(-5, 5, size=(60, 13)).astype(np.int32)
u[i0 % 60, :, i0 % 3] = dev(w)  # repeated (row, col) pairs: the last value wins
uh[i0 % 60, :, i0 % 3] = w
C.check("u[i, :, i % 3] = array (last wins)", u, uh)
u[:, dev(i0[:9] % 13), 5:40:5] = 1
uh[:, i0[:9] % 13, 5:40:5] = 1
C.check("u[:, i, 5:40:5] = s", u, uh)

# ---- scatter: duplicates across chares, last write wins ----
for n in (37, 300001):
    z = cnp.zeros((n,), dtype=np.float64)
    zh = np.zeros(n)
    idx = rng.integers(-n, n, size=5000)
    v = rng.random(5000)
    z[dev(idx)] = dev(v)
    zh[idx] = v
    C.check(f"z[dup] = v ({n})", z, zh)
    z[idx[::-1]] = dev(v)
    zh[idx[::-1]] = v
    C.check(f"z[dup reversed] = v ({n})", z, zh)
    cnp.put(z, dev(idx[:300]), dev(v[:300]))
    np.put(zh, idx[:300], v[:300])
    C.check(f"put dup ({n})", z, zh)

# ---- values that alias the target (evaluated before the write, as in NumPy) ----
q = cnp.arange(1000, dtype=np.int64)
qh = np.arange(1000)
perm = rng.permutation(1000)[:200]
q[perm] = q[300:500]
qh[perm] = qh[300:500].copy()
C.check("q[perm] = q[300:500]", q, qh)
q[dev(perm[::-1])] = q[::5]
qh[perm[::-1]] = qh[::5].copy()
C.check("q[perm] = q[::5]", q, qh)
r2 = ints((300, 40), -50, 50, dtype=np.int64)
r2h = C.fetch(r2)
ci = rng.integers(0, 40, size=40)
r2[:, ci] = r2
r2h[:, ci] = r2h.copy()
C.check("r2[:, ci] = r2", r2, r2h)

# ---- every element size, incl. 16-byte complex ----
for dt in (np.bool_, np.int8, np.uint16, np.float16, np.int32, np.float32, np.int64,
           np.complex64, np.complex128):
    h = (rng.integers(-9, 10, size=(300, 11)) + (1j if np.dtype(dt).kind == "c" else 0)).astype(dt)
    a = cnp.asarray(h)
    name = np.dtype(dt).name
    ri = rng.integers(-300, 300, size=50)
    C.check(f"{name} rows", a[ri], h[ri], dtype=dt)
    C.check(f"{name} take axis 1", cnp.take(a, [10, 0, 0, -1], axis=1),
            np.take(h, [10, 0, 0, -1], axis=1), dtype=dt)
    hc = h.copy()
    a[ri[:20], 3:9] = a[ri[20:40], 0:6]
    hc[ri[:20], 3:9] = hc[ri[20:40], 0:6].copy()
    C.check(f"{name} rows = gathered", a, hc, dtype=dt)
    if np.dtype(dt).itemsize <= 8:
        a[::7, [2, 4]] = h.flat[1]
        hc[::7, [2, 4]] = h.flat[1]
        C.check(f"{name} [::7, [2, 4]] = scalar", a, hc, dtype=dt)

# ---- indexing_routines patterns at a size with several chares ----
N = 1200
A2 = cnp.ones((N, N), dtype=int)
B = cnp.arange(N, dtype=int)
indx, indx_bool = B % 10, (B % 2).astype(bool)
A2[indx_bool, indx_bool] = 11
A2[:, indx] = 12
r = np.ones((N, N), dtype=int)
Bh = np.arange(N)
r[(Bh % 2).astype(bool), (Bh % 2).astype(bool)] = 11
r[:, Bh % 10] = 12
C.check("ai2 at 1200", A2, r)
R = cnp.repeat(1, N)
C.check("repeat(A2, R, axis=1)", cnp.repeat(A2, R, axis=1), np.repeat(r, np.ones(N, int), axis=1))

# ---- a long 1-D array: many chares, so many destination buckets ----
nl = 10_000_000
big = cnp.arange(nl, dtype=np.int64)
gi = rng.integers(-nl, nl, size=200_000)
C.check("take from 10M", cnp.take(big, dev(gi)), np.arange(nl)[gi])
gv = rng.integers(0, 1000, size=200_000)
big[dev(gi)] = dev(gv)
bh = np.arange(nl)
bh[gi] = gv
C.check("10M[dup] = v", big, bh)

# ---- out-of-bounds device indices are dropped, the rest is written ----
o = cnp.zeros((50,), dtype=np.int64)
o[dev(np.array([3, 60, 7]))] = 1
oh = np.zeros(50, dtype=np.int64)
oh[[3, 7]] = 1
C.check("dropped out-of-bounds device index", o, oh)

C.finish()
