"""Index data movement vs NumPy: unaligned masks (moved on the device), the row offset scan
of COMPRESS / MASK_SCATTER / nonzero over multi-chare 2-D and 3-D grids, repeat (scalar
counts without an index array, host and device count arrays) and index arrays on >3-D
arrays (offsets built on the device)."""

import numpy as np

import charmnumeric as cnp
from _ops_common import Checker, connect

iface = connect()
C = Checker("test_index_scale", iface)
rng = np.random.default_rng(11)


def dev(h):
    return cnp.asarray(np.ascontiguousarray(h))


def rnd(shape, dt=np.float64):
    if dt == np.bool_:
        return rng.integers(0, 2, size=shape).astype(bool)
    if np.issubdtype(dt, np.complexfloating):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
    if np.issubdtype(dt, np.floating):
        return rng.standard_normal(shape).astype(dt)
    return rng.integers(-100, 100, size=shape).astype(dt)


# ---- masks laid out differently from x (unaligned) and multi-chare row scans ----
# 2-D and 3-D shapes split over several chares along every dim at +p4 (CPU and GPU tiles)

BIG = [(300001,), (1100, 1030), (130, 140, 135)]
for shape in BIG:
    nd = len(shape)
    s = f" {shape}"
    Xh = rnd(shape)
    X = dev(Xh)
    views = [(slice(None),) * nd, (slice(3, None),) * nd,
             tuple(slice(1, None, 2) if d == nd - 1 else slice(2, -1) for d in range(nd))]
    for key in views:
        vh = Xh[key]
        mh = rng.random(vh.shape) < 0.4
        k = f"{s} view {key}"
        # fresh mask vs a view of x: unaligned when the view does not start on a tile boundary
        C.check("x[m]" + k, X[key][dev(mh)], vh[mh])
        C.check("x[host m]" + k, X[key][mh], vh[mh])
        # mask that is itself a view of a larger array
        Mh = rng.random(shape) < 0.3
        M = dev(Mh)
        mv = M[key]
        C.check("x[m view]" + k, X[key][mv], vh[Mh[key]])
        if key == views[0]:
            sub = tuple(slice(1, None) for _ in range(nd))
            mh2 = Mh[sub]
            C.check("x[1:][m[1:]]" + s, X[sub][M[sub]], Xh[sub][mh2])
    # MASK_SCATTER with unaligned masks (x a view, mask fresh; x fresh, mask a view)
    key = views[1]
    xh = Xh.copy()
    x = dev(xh)
    mh = rng.random(xh[key].shape) < 0.5
    vh = rnd((int(mh.sum()),))
    xv = x[key]
    xv[dev(mh)] = dev(vh)
    xh[key][mh] = vh
    C.check("x[view][m] = v" + s, x, xh)
    Mh = rng.random(shape) < 0.5
    M = dev(Mh)
    sub = tuple(slice(1, None) for _ in range(nd))
    y = dev(np.zeros(Mh[sub].shape))
    yh = np.zeros(Mh[sub].shape)
    vh = rnd((int(Mh[sub].sum()),))
    y[M[sub]] = dev(vh)
    yh[Mh[sub]] = vh
    C.check("y[m view] = v" + s, y, yh)
    # nonzero / argwhere (EXPAND) on the same grids, views included
    for key in views[:2]:
        bh = rng.random(Xh[key].shape) < 0.01
        b = dev(bh)
        got = cnp.nonzero(b)
        want = np.nonzero(bh)
        for d in range(nd):
            C.check(f"nonzero[{d}]" + s + f" {key}", got[d], want[d])
        C.check("flatnonzero" + s, cnp.flatnonzero(X[key] > 1.5), np.flatnonzero(Xh[key] > 1.5))

for dt in (np.bool_, np.int8, np.int32, np.float32, np.complex64, np.complex128):
    xh = rnd((1100, 1030), dt)
    x = dev(xh)
    mh = rng.random((1097, 1027)) < 0.5
    C.check(f"{np.dtype(dt).name} x[3:,3:][m]", x[3:, 3:][dev(mh)], xh[3:, 3:][mh], dtype=dt)
    vh = rnd((int(mh.sum()),), dt)
    xv = x[3:, 3:]
    xv[dev(mh)] = dev(vh)
    xh[3:, 3:][mh] = vh
    C.check(f"{np.dtype(dt).name} x[3:,3:][m] = v", x, xh, dtype=dt)

# an empty and a full selection
zh = np.zeros((1100, 1030), bool)
Xh = rnd((1100, 1030))
X = dev(Xh)
C.check("x[all false]", X[1:, 1:][dev(zh[1:, 1:])], Xh[1:, 1:][zh[1:, 1:]])
C.check("x[all true]", X[1:, 1:][dev(~zh[1:, 1:])], Xh[1:, 1:][~zh[1:, 1:]])

# ---- repeat ----

for dt in (np.bool_, np.int8, np.int64, np.float32, np.complex128):
    ah = rnd((300001,), dt)
    a = dev(ah)
    name = np.dtype(dt).name
    for r in (0, 1, 3):
        C.check(f"repeat {name} 1-D r={r}", cnp.repeat(a, r), np.repeat(ah, r), dtype=dt)
    C.check(f"repeat {name} 1-D view r=2", cnp.repeat(a[5::3], 2), np.repeat(ah[5::3], 2), dtype=dt)
    C.check(f"{name} a.repeat(4, axis=0)", a.repeat(4, axis=0), ah.repeat(4, axis=0), dtype=dt)
for shape in [(1100, 1030), (130, 140, 135), (37, 41)]:
    ah = rnd(shape)
    a = dev(ah)
    s = f" {shape}"
    C.check("repeat axis=None r=2" + s, cnp.repeat(a, 2), np.repeat(ah, 2))
    sub = tuple(slice(1, None, 2) for _ in shape)
    C.check("repeat view axis=None r=3" + s, cnp.repeat(a[sub], 3), np.repeat(ah[sub], 3))
    for ax in range(len(shape)):
        C.check(f"repeat axis={ax} r=2" + s, cnp.repeat(a, 2, axis=ax), np.repeat(ah, 2, axis=ax))
    n0 = shape[0]
    rep = rng.integers(0, 4, size=n0)
    C.check("repeat host counts axis=0" + s, cnp.repeat(a, rep, axis=0), np.repeat(ah, rep, axis=0))
    C.check("repeat host list axis=0" + s, cnp.repeat(a, list(rep), axis=0),
            np.repeat(ah, list(rep), axis=0))
    C.check("repeat device counts axis=0" + s, cnp.repeat(a, dev(rep), axis=0),
            np.repeat(ah, rep, axis=0))
ah = rnd((300001,))
rep = rng.integers(0, 5, size=300001)
C.check("repeat 1-D host counts", cnp.repeat(dev(ah), rep), np.repeat(ah, rep))
C.check("repeat 1-D device counts", cnp.repeat(dev(ah), dev(rep)), np.repeat(ah, rep))
C.check("repeat zero counts", cnp.repeat(dev(ah[:10]), np.zeros(10, int)), np.repeat(ah[:10], 0))
C.check("repeat count 1 of 1", cnp.repeat(dev(ah[:1]), [5]), np.repeat(ah[:1], [5]))
C.check("repeat host scalar", cnp.repeat(3.5, 4), np.repeat(3.5, 4))
C.check("repeat host list a", cnp.repeat([1, 2, 3], 2), np.repeat([1, 2, 3], 2))
fh = rnd((3, 4, 5, 6))
C.check("repeat folded axis=None", cnp.repeat(dev(fh), 2), np.repeat(fh, 2))
C.raises("repeat negative host counts", ValueError, lambda: cnp.repeat(dev(ah[:3]), [1, -1, 2]))
C.raises("repeat float host counts", TypeError, lambda: cnp.repeat(dev(ah[:3]), [1.5, 1, 2]))
C.raises("repeat negative scalar", ValueError, lambda: cnp.repeat(dev(ah[:3]), -1))

# ---- index arrays on >3-D arrays ----

for shape in [(6, 7, 8, 9), (3, 4, 5, 6, 7), (40, 30, 70, 60)]:
    xh = rnd(shape)
    x = dev(xh)
    s = f" {shape}"
    n = len(shape)
    i0 = rng.integers(-shape[0], shape[0], size=5)
    il = rng.integers(0, shape[-1], size=4)
    i2 = rng.integers(0, shape[2], size=(2, 3))
    j0 = rng.integers(0, shape[0], size=4)
    j2 = rng.integers(0, shape[2], size=4)
    keys = [
        ("[i]", (i0,)),
        ("[:, i]", (slice(None), rng.integers(0, shape[1], size=3))),
        ("[..., i]", (Ellipsis, il)),
        ("[i, :, 2]", (i0, slice(None), 2)),
        ("[:, :, i2d]", (slice(None), slice(None), i2)),
        ("[1:, i, ::2]", (slice(1, None), rng.integers(0, shape[1], size=3), slice(None, None, 2))),
        ("[i, ..., ::-2]", (i0, Ellipsis, slice(None, None, -2))),
        ("[j, :, j]", (j0, slice(None), j2)),
        ("[None, i]", (None, i0)),
        ("[i, None]", (i0, None)),
        ("[::2, None, i]", (slice(None, None, 2), None, rng.integers(0, shape[1], size=3))),
        ("[2, i]", (2, rng.integers(0, shape[1], size=3))),
    ]
    for name, key in keys:
        want = xh[key]
        dkey = tuple(dev(k) if isinstance(k, np.ndarray) else k for k in key)
        C.check("folded host" + name + s, x[key], want)
        C.check("folded device" + name + s, x[dkey], want)
    # mixed host and device arrays
    C.check("folded mixed [j, :, dev j]" + s, x[j0, :, dev(j2)], xh[j0, :, j2])
    # a mask over leading dims (device and host)
    mh = rng.random(shape[:2]) < 0.5
    C.check("folded [mask2d]" + s, x[dev(mh)], xh[mh])
    C.check("folded [host mask2d]" + s, x[mh], xh[mh])
    C.check("folded [:, mask1d]" + s, x[:, dev(mh[0])], xh[:, mh[0]])
    # a >3-D index array
    ih = rng.integers(0, shape[0], size=(2, 2, 3, 2))
    C.check("folded [i4d]" + s, x[dev(ih)], xh[ih])
    # assignment
    yh = xh.copy()
    y = dev(yh)
    y[i0 % shape[0], :, 1] = 7.0
    yh[i0 % shape[0], :, 1] = 7.0
    C.check("folded [i, :, 1] = scalar" + s, y, yh)
    i1 = rng.integers(0, shape[1], size=4)
    vh = rnd(yh[:, i1].shape)
    y[:, dev(i1)] = dev(vh)
    yh[:, i1] = vh
    C.check("folded [:, i] = array" + s, y, yh)
    vh = rnd(yh[..., il].shape[-1:])
    y[..., il] = vh
    yh[..., il] = vh
    C.check("folded [..., i] = broadcast host" + s, y, yh)
    y[dev(mh)] = -1.5
    yh[mh] = -1.5
    C.check("folded [mask2d] = scalar" + s, y, yh)
C.raises("folded index out of bounds", IndexError, lambda: dev(rnd((2, 3, 4, 5)))[[0, 2]])

# keys without index arrays stay on the basic (view / block) path
fh = rnd((3, 4, 5, 6))
f = dev(fh)
for name, key in [("[1:, 2, None, ::2]", (slice(1, None), 2, None, slice(None, None, 2))),
                  ("[..., 3]", (Ellipsis, 3)), ("[None, 0, :, 1:4]", (None, 0, slice(None), slice(1, 4))),
                  ("[np.int64(1)]", (np.int64(1),))]:
    C.check("folded basic get" + name, f[key], fh[key])
f[1:, 2, None, ::2] = 4.5
fh[1:, 2, None, ::2] = 4.5
C.check("folded basic set [1:, 2, None, ::2] = s", f, fh)
f[None, 0, :, 1:4] = 0.25
fh[None, 0, :, 1:4] = 0.25
C.check("folded basic set [None, 0, :, 1:4] = s", f, fh)
vh = rnd((4, 5))
f[2, ..., 5] = dev(vh)
fh[2, ..., 5] = vh
C.check("folded basic set [2, ..., 5] = array", f, fh)

C.finish()
