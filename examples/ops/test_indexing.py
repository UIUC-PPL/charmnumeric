"""Advanced indexing vs NumPy: integer and boolean-mask reads/writes, take/put, nonzero, choose."""

import numpy as np

import charmnumeric as cnp
from _ops_common import SHAPES, Checker, connect, floats, ints

iface = connect()
C = Checker("test_indexing", iface)
rng = np.random.default_rng(0)


def dev(host):
    """Upload a host index array (device-resident from then on)."""
    return cnp.array(np.ascontiguousarray(host))


def index_cases(n):
    """Host index arrays into an axis of length n: repeats, negatives, unsorted."""
    k = max(1, min(n, 50))
    return {
        "perm": rng.permutation(n)[:k],
        "rep": rng.integers(0, n, size=2 * k),
        "neg": rng.integers(-n, n, size=k),
        "2d": rng.integers(-n, n, size=(3, 4)),
    }


# Size-1 operands first: later chare-grid growth replays these DAGs.
a = cnp.arange(37, dtype=np.int64)
ah = np.arange(37)
C.check("a[[3]] (size-1 index)", a[[3]], ah[[3]])
C.check("a[dev [5]]", a[dev(np.array([5]))], ah[[5]])
C.check("size-1 source", cnp.full((1,), 7.5)[[0, 0, 0]], np.full(3, 7.5))
t, th = a.copy(), ah.copy()
t[[4]] = cnp.full((1,), -1, dtype=np.int64)
th[[4]] = -1
C.check("t[[4]] = size-1 array", t, th)
t[[1, 2, 3, 4, 5]] = cnp.arange(5, dtype=np.int64)[::-1]
th[[1, 2, 3, 4, 5]] = np.arange(5)[::-1]
C.check("t[idx] = negative-step view", t, th)
t[2:12][[0, 9]] = 100
th[2:12][[0, 9]] = 100
C.check("view[idx] = s", t, th)
x, xh = cnp.arange(10, dtype=np.float64), np.arange(10.0)
x[x > 4] = x[:5] * 10
xh[xh > 4] = xh[:5] * 10
C.check("x[mask] = x[:5] * 10 (aliasing)", x, xh)
x[[0, 1]] = x[[1, 0]]
xh[[0, 1]] = xh[[1, 0]]
C.check("swap x[[0, 1]] = x[[1, 0]]", x, xh)
m2, m2h = cnp.arange(12, dtype=np.int64).reshape(3, 4), np.arange(12).reshape(3, 4)
C.check("reshape", m2, m2h)
C.check("reshape(2, -1, 2)", m2.reshape(2, -1, 2), m2h.reshape(2, -1, 2))
C.check("reshape of a view", cnp.reshape(m2[:, 1:3], -1), m2h[:, 1:3].reshape(-1))
C.check("argwhere 2-D", cnp.argwhere(m2 % 5 == 0), np.argwhere(m2h % 5 == 0))
C.check("float(), int()", np.array([float(cnp.sum(a)), int(cnp.amax(a))]),
        np.array([float(ah.sum()), 36.0]))
C.check("choose clip, host index", cnp.choose([0, 5, -1], [cnp.full((3,), 1), cnp.full((3,), 2)],
                                               mode="clip"),
        np.choose([0, 5, -1], [np.full(3, 1), np.full(3, 2)], mode="clip"))
C.check("repeat 0", cnp.repeat(a[:4], 0), np.repeat(ah[:4], 0))
C.check("compress all false", cnp.compress([False, False], a), np.compress([False, False], ah))
C.check("take(out=)", cnp.take(a, [1, 2], out=cnp.zeros((2,), dtype=np.int64)), ah.take([1, 2]))

for shape in SHAPES:
    s = f" {shape}"
    nd = len(shape)
    a = ints(shape, -50, 50, dtype=np.int64)
    f = floats(shape, seed=1)
    ah, fh = C.fetch(a), C.fetch(f)

    # ---- integer-array reads: first axis, last axis, host and device indices ----
    for name, idx in index_cases(shape[0]).items():
        if name == "2d" and nd == 3:
            continue  # 4-D result
        C.check(f"a[{name}]" + s, a[idx], ah[idx])
        C.check(f"f[dev {name}]" + s, f[dev(idx)], fh[idx])
    if nd >= 2:
        for name, idx in index_cases(shape[-1]).items():
            if name == "2d" and nd == 3:
                continue  # 4-D result
            key = (Ellipsis, idx)
            C.check(f"a[..., {name}]" + s, a[key], ah[key])
            C.check(f"f[..., dev {name}]" + s, f[(Ellipsis, dev(idx))], fh[key])
        i0 = rng.integers(-shape[0], shape[0], size=17)
        i1 = rng.integers(0, shape[-1], size=17)
        C.check("a[i, j] zipped" + s, a[(i0,) + (slice(None),) * (nd - 2) + (i1,)],
                ah[(i0,) + (slice(None),) * (nd - 2) + (i1,)])
        C.check("a[dev i, dev j]" + s, a[(dev(i0),) + (slice(None),) * (nd - 2) + (dev(i1),)],
                ah[(i0,) + (slice(None),) * (nd - 2) + (i1,)])
        C.check("a[i, 1]" + s, a[(i0, 1)], ah[(i0, 1)])
    if nd == 3:
        i1 = rng.integers(0, shape[1], size=6)
        C.check("a[:, i, :]" + s, a[:, i1, :], ah[:, i1, :])
        C.check("a[2, :, i]" + s, a[2, :, i1 % shape[2]], ah[2, :, i1 % shape[2]])

    # device indices computed on the backend (data-dependent)
    n0 = shape[0]
    di = (cnp.arange(n0, dtype=np.int64) * 7 + 3) % n0 - n0 // 2
    dih = (np.arange(n0) * 7 + 3) % n0 - n0 // 2
    C.check("a[computed idx]" + s, a[di], ah[dih])

    # views as source and as index
    if nd == 1:
        v, vh = a[3:-2:2], ah[3:-2:2]
    elif nd == 2:
        v, vh = a[1:-1, ::2], ah[1:-1, ::2]
    else:
        v, vh = a[1:, :, ::2], ah[1:, :, ::2]
    idx = rng.integers(-v.shape[0], v.shape[0], size=9)
    C.check("view[idx]" + s, v[idx], vh[idx])
    C.check("view[dev idx view]" + s, v[dev(np.concatenate([idx, idx]))[9:18]], vh[idx])

    # empty selections
    C.check("a[[]]" + s, a[[]], ah[[]])
    C.check("a[mask all false]" + s, a[a > 1000], ah[ah > 1000])

    # ---- boolean masks ----
    m = (a % 3).astype(bool)
    mh = (ah % 3).astype(bool)
    C.check("a[a % 3 != 0]" + s, a[m], ah[mh])
    C.check("f[f > 0.5]" + s, f[f > 0.5], fh[fh > 0.5])
    C.check("f[host mask]" + s, f[fh > 0.5], fh[fh > 0.5])
    if nd >= 2:
        lead = (cnp.arange(shape[0], dtype=np.int64) % 2).astype(bool)
        leadh = (np.arange(shape[0]) % 2).astype(bool)
        C.check("a[lead mask]" + s, a[lead], ah[leadh])
        c = min(shape[0], shape[1])
        m0, m1 = cnp.arange(shape[0], dtype=np.int64) < c, cnp.arange(shape[1], dtype=np.int64) < c
        C.check("a[mask0, mask1] zipped" + s, a[m0, m1],
                ah[np.arange(shape[0]) < c, np.arange(shape[1]) < c])

    # ---- writes ----
    t, th = a.copy(), ah.copy()
    idx = index_cases(shape[0])["rep"]
    t[idx] = 7
    th[idx] = 7
    C.check("a[rep] = 7" + s, t, th)
    t[dev(idx)] = -7
    th[idx] = -7
    C.check("a[dev rep] = -7" + s, t, th)
    vals_h = rng.integers(-100, 100, size=(len(idx),) + shape[1:])
    t[idx] = dev(vals_h)  # repeated indices: the last value wins, as in NumPy
    th[idx] = vals_h
    C.check("a[rep] = array (last wins)" + s, t, th)
    t[m] = 99
    th[mh] = 99
    C.check("a[mask] = 99" + s, t, th)
    g, gh = f.copy(), fh.copy()
    g[(a > 10).astype(bool)] = 0.25  # mask of the int dtype into a float array
    gh[ah > 10] = 0.25
    C.check("f[int mask] = 0.25" + s, g, gh)
    cnt = int(np.count_nonzero(fh < 0))
    g[f < 0] = cnp.arange(cnt, dtype=np.float64)
    gh[fh < 0] = np.arange(cnt, dtype=np.float64)
    C.check("f[mask] = array" + s, g, gh)
    if nd >= 2:
        j = rng.integers(0, shape[1], size=5)
        t[:, j] = 1234
        th[:, j] = 1234
        C.check("a[:, j] = s" + s, t, th)
        w, wh = t[1:, ::2], th[1:, ::2]
        jj = rng.integers(0, w.shape[0], size=4)
        w[jj] = -1
        wh[jj] = -1
        C.check("view[jj] = -1 (writes through)" + s, t, th)
    if nd == 3:
        k = rng.integers(0, min(shape[0], shape[2]), size=6)
        t[k, :, k] = 15
        th[k, :, k] = 15
        C.check("a[k, :, k] = 15" + s, t, th)

    # ---- negative steps and flips ----
    C.check("a[::-1]" + s, a[::-1], ah[::-1])
    C.check("a[-2:1:-3]" + s, a[-2:1:-3], ah[-2:1:-3])
    C.check("flip" + s, cnp.flip(f), np.flip(fh))
    C.check("flip(axis=-1)" + s, cnp.flip(a, -1), np.flip(ah, -1))
    if nd >= 2:
        C.check("fliplr" + s, cnp.fliplr(a), np.fliplr(ah))
        C.check("a[::2, ::-1]" + s, a[::2, ::-1], ah[::2, ::-1])
    C.check("flipud" + s, cnp.flipud(a), np.flipud(ah))
    t, th = a.copy(), ah.copy()
    t[::-2] = 5
    th[::-2] = 5
    C.check("a[::-2] = 5" + s, t, th)
    t[::-1] = a
    th[::-1] = ah
    C.check("a[::-1] = a" + s, t, th)

    # ---- take / put / *_along_axis ----
    for ax in range(nd):
        idx = rng.integers(-shape[ax], shape[ax], size=(2, 3)) if nd < 3 else \
            rng.integers(-shape[ax], shape[ax], size=4)
        C.check(f"take axis={ax}" + s, a.take(idx, axis=ax), ah.take(idx, axis=ax))
    big = rng.integers(-3 * ah.size, 3 * ah.size, size=20)
    C.check("take wrap" + s, cnp.take(a, big, mode="wrap"), np.take(ah, big, mode="wrap"))
    C.check("take clip dev" + s, cnp.take(f, dev(big), mode="clip"), np.take(fh, big, mode="clip"))
    t, th = a.copy(), ah.copy()
    pi = rng.integers(0, ah.size, size=11)
    cnp.put(t, pi, [1, 2, 3])
    np.put(th, pi, [1, 2, 3])
    C.check("put cyclic" + s, t, th)
    t.put(dev(pi), -5)
    th.put(pi, -5)
    C.check("put dev" + s, t, th)
    ax = nd - 1
    ia = rng.integers(0, shape[ax], size=shape[:-1] + (3,))
    C.check("take_along_axis" + s, cnp.take_along_axis(a, ia, ax), np.take_along_axis(ah, ia, ax))
    C.check("take_along_axis dev" + s, cnp.take_along_axis(f, dev(ia), ax),
            np.take_along_axis(fh, ia, ax))
    ip = np.argsort(rng.random(shape), axis=0)[:2]
    t, th = a.copy(), ah.copy()
    cnp.put_along_axis(t, dev(ip), -3, 0)
    np.put_along_axis(th, ip, -3, 0)
    C.check("put_along_axis" + s, t, th)

    # ---- nonzero / argwhere / compress ----
    C.check("len(nonzero)" + s, np.array(cnp.nonzero(m)[0].shape),
            np.array(np.nonzero(mh)[0].shape))
    for d, (got, exp) in enumerate(zip(cnp.nonzero(a > 20), np.nonzero(ah > 20))):
        C.check(f"nonzero[{d}]" + s, got, exp)
    C.check("flatnonzero" + s, cnp.flatnonzero(a % 7), np.flatnonzero(ah % 7))
    C.check("argwhere" + s, cnp.argwhere(a < -40), np.argwhere(ah < -40))
    cond = (np.arange(shape[0]) % 3) == 1
    C.check("compress host" + s, cnp.compress(cond, a, axis=0), np.compress(cond, ah, axis=0))
    C.check("compress dev" + s, cnp.compress(dev(cond.astype(np.int64)) > 0, a, axis=0),
            np.compress(cond, ah, axis=0))

    # ---- repeat ----
    C.check("repeat 3 flat" + s, cnp.repeat(a, 3) if ah.size * 3 < 2_000_000 else a[:1],
            np.repeat(ah, 3) if ah.size * 3 < 2_000_000 else ah[:1])
    rep = rng.integers(0, 4, size=shape[-1])
    C.check("repeat host counts" + s, cnp.repeat(a, rep, axis=nd - 1),
            np.repeat(ah, rep, axis=nd - 1))
    C.check("repeat dev counts" + s, cnp.repeat(f, dev(rep), axis=nd - 1),
            np.repeat(fh, rep, axis=nd - 1))

# ---- choose ----
for shape in [(37,), (300001,), (600, 7)]:
    s = f" {shape}"
    c = ints(shape, -4, 12, dtype=np.int64)
    ch = C.fetch(c)
    ch_list = [ints(shape, k, k + 3, dtype=np.int64, mult=k + 1) for k in range(4)]
    chh = [C.fetch(x) for x in ch_list]
    C.check("choose wrap" + s, cnp.choose(c, ch_list, mode="wrap"), np.choose(ch, chh, mode="wrap"))
    C.check("choose clip" + s, cnp.choose(c, ch_list, mode="clip"), np.choose(ch, chh, mode="clip"))
    C.check("choose scalars" + s, cnp.choose(c % 3, [10, 20, 30]), np.choose(ch % 3, [10, 20, 30]))
    fl = [floats(shape, seed=k) for k in range(3)]
    flh = [C.fetch(x) for x in fl]
    C.check("choose float choices" + s, cnp.choose(c, fl, mode="wrap"),
            np.choose(ch, flh, mode="wrap"))
    C.check("a.choose" + s, (c % 2).choose(ch_list[:2]), (ch % 2).choose(chh[:2]))

# ---- int32 / float32 and the indexing_routines patterns ----
x32 = ints((600, 7), -9, 9, dtype=np.int32)
x32h = C.fetch(x32)
i32 = cnp.arange(600, dtype=np.int32)[::-3]
C.check("int32 data, int32 index", x32[i32], x32h[np.arange(600, dtype=np.int32)[::-3]])
f32 = floats((70, 3, 5), seed=4, dtype=np.float32)
f32h = C.fetch(f32)
C.check("float32 mask read", f32[f32 > 0], f32h[f32h > 0])
N = 100
A2 = cnp.ones((N, N), dtype=int)
B = cnp.arange(N, dtype=int)
indx, indx_bool = B % 10, (B % 2).astype(bool)
A2[indx_bool, indx_bool] = 11
A2[:, indx] = 12
A2[(A2 % 2).astype(bool)] = 13
r = np.ones((N, N), dtype=int)
Bh = np.arange(N)
r[(Bh % 2).astype(bool), (Bh % 2).astype(bool)] = 11
r[:, Bh % 10] = 12
r[(r % 2).astype(bool)] = 13
C.check("indexing_routines ai2", A2, r)
A3 = cnp.ones((300, 3, 100), dtype=int)
B = cnp.arange(300, dtype=int)
A3[B % 10, :, B % 10] = 15
r3 = np.ones((300, 3, 100), dtype=int)
r3[np.arange(300) % 10, :, np.arange(300) % 10] = 15
C.check("indexing_routines ai3", A3, r3)
d = cnp.arange(320000, dtype=np.int64).take(np.array([[0, 1], [319999, 262144]]), axis=0)
C.check("take across 1-D tiles", d, np.array([[0, 1], [319999, 262144]]))

# ---- dtypes: masks are bool arrays; data of every dtype up to 8 bytes moves ----
hm = rng.integers(0, 2, size=(45, 9)).astype(bool)
m = cnp.asarray(hm)
C.check("mask dtype from comparison", f32 > 0, f32h > 0, dtype=np.bool_)
for dt in (np.bool_, np.int8, np.uint16, np.float16, np.int32, np.int64, np.float32,
           np.float64, np.complex64):
    h = (rng.integers(-9, 10, size=(45, 9)) + (1j if np.dtype(dt).kind == "c" else 0)).astype(dt)
    x = cnp.asarray(h)
    name = np.dtype(dt).name
    C.check(f"{name}[bool mask]", x[m], h[hm], dtype=dt)
    C.check(f"{name}[index]", x[[3, 1, 44, -1]], h[[3, 1, 44, -1]], dtype=dt)
    C.check(f"{name} take axis 1", cnp.take(x, [8, 0, 0], axis=1), np.take(h, [8, 0, 0], axis=1))
    C.check(f"nonzero {name}", cnp.flatnonzero(x), np.flatnonzero(h))
    x[m] = h.flat[0]
    hc = h.copy()
    hc[hm] = h.flat[0]
    C.check(f"{name}[mask] = scalar", x, hc)
    x[[0, 2]] = x[[5, 6]]
    hc[[0, 2]] = hc[[5, 6]]
    C.check(f"{name}[index] = gathered", x, hc)
bh = np.arange(20) % 3 == 0
b = cnp.asarray(bh)
C.check("bool[bool]", b[b], bh[bh], dtype=np.bool_)
C.check("where(bool mask)", cnp.where(m, 1, 2.5), np.where(hm, 1, 2.5))
C.check("int8 mask value", cnp.arange(10)[cnp.asarray(np.arange(10, dtype=np.int8)) > 4],
        np.arange(10)[np.arange(10) > 4])
C.raises("complex128 gather", NotImplementedError,
         lambda: cnp.asarray(np.ones(4, np.complex128))[[1, 2]])
C.raises("float index rejected", IndexError, lambda: a[cnp.zeros((3,))])
C.raises("out of bounds host index", IndexError, lambda: a[[10**6]])

C.finish()
