"""Boolean masks over leading dims (COMPRESS / MASK_SCATTER) vs NumPy: reads and writes
for every dtype, views, unaligned masks, host masks and >3-D arrays."""

import numpy as np

import charmnumeric as cnp
from _ops_common import Checker, connect

iface = connect()
C = Checker("test_mask", iface)
rng = np.random.default_rng(7)

DTYPES = (np.bool_, np.int8, np.uint16, np.float16, np.int32, np.int64, np.float32, np.float64,
          np.complex64, np.complex128)
# above the CPU minimum tiles, so +p2 splits them over several chares
SHAPES = [(37,), (300001,), (41, 9), (1100, 530), (5, 7, 9), (130, 70, 3)]


def host(shape, dt):
    if dt == np.bool_:
        return rng.integers(0, 2, size=shape).astype(bool)
    if np.issubdtype(dt, np.complexfloating):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
    if np.issubdtype(dt, np.floating):
        return rng.standard_normal(shape).astype(dt)
    info = np.iinfo(dt)
    return rng.integers(max(info.min, -1000), min(info.max, 1000), size=shape).astype(dt)


def dev(h):
    return cnp.asarray(np.ascontiguousarray(h))


def hmask(shape, p=0.5):
    return rng.random(shape) < p


for shape in SHAPES:
    s = f" {shape}"
    big = int(np.prod(shape)) > 100000
    for dt in (DTYPES if not big else (np.float32, np.int64, np.complex128, np.bool_)):
        name = np.dtype(dt).name
        xh = host(shape, dt)
        x = dev(xh)
        mh = hmask(shape)
        m = dev(mh)
        C.check(f"{name}[mask]{s}", x[m], xh[mh], dtype=dt)
        vh = host((int(mh.sum()),), dt)
        x[m] = dev(vh)
        xh[mh] = vh
        C.check(f"{name}[mask] = array{s}", x, xh, dtype=dt)
    xh = host(shape, np.float64)
    x = dev(xh)
    mh = hmask(shape, 0.3)
    C.check("f8[host mask]" + s, x[mh], xh[mh])
    vh = host((int(mh.sum()),), np.float64)
    x[mh] = vh
    xh[mh] = vh
    C.check("f8[host mask] = host array" + s, x, xh)
    x[dev(mh)] = 2.5
    xh[mh] = 2.5
    C.check("f8[mask] = scalar" + s, x, xh)
    C.check("f8[empty mask]" + s, x[dev(np.zeros(shape, bool))], xh[np.zeros(shape, bool)])
    C.check("f8[full mask]" + s, x[dev(np.ones(shape, bool))], xh.reshape(-1))
    # comparison masks (laid out like x) and int -> float assignment casts
    C.check("x[x > 0]" + s, x[x > 0], xh[xh > 0])
    ih = host(shape, np.int64)
    i = dev(ih)
    i[i > 0] = x[i > 0]
    ih[ih > 0] = xh[ih > 0]
    C.check("int[mask] = float values (cast)" + s, i, ih)

    # views: offset and stepped views of x and of the mask (unaligned layouts)
    sub = tuple(slice(1, None) for _ in shape)
    stp = tuple(slice(None, None, 2) for _ in shape)
    other = tuple(slice(None, -1) for _ in shape)
    xh = host(shape, np.float32)
    x = dev(xh)
    mh = hmask(shape)
    m = dev(mh)
    C.check("x[1:][m[:-1]]" + s, x[sub][m[other]], xh[sub][mh[other]])
    C.check("x[::2][m[::2]]" + s, x[stp][m[stp]], xh[stp][mh[stp]])
    C.check("x[1:][fresh mask]" + s, x[sub][x[sub] > 0.2], xh[sub][xh[sub] > 0.2])
    vh = host((int(mh[other].sum()),), np.float32)
    xv = x[sub]
    xv[m[other]] = dev(vh)
    xh[sub][mh[other]] = vh
    C.check("x[1:][m[:-1]] = array" + s, x, xh)
    vh = host((int(mh[stp].sum()) * 2,), np.float32)
    xv = x[stp]
    xv[m[stp]] = dev(vh)[::2]
    xh[stp][mh[stp]] = vh[::2]
    C.check("x[::2][m[::2]] = stepped view" + s, x, xh)

    # masks over leading dims
    if len(shape) > 1:
        lh = hmask(shape[:1])
        l = dev(lh)
        C.check("x[lead mask]" + s, x[l], xh[lh])
        rest = shape[1:]
        vh = host((int(lh.sum()),) + rest, np.float32)
        x[l] = dev(vh)
        xh[lh] = vh
        C.check("x[lead mask] = array" + s, x, xh)
        rh = host(rest, np.float32)
        x[l] = dev(rh)
        xh[lh] = rh
        C.check("x[lead mask] = broadcast row" + s, x, xh)
        x[lh] = -1.0
        xh[lh] = -1.0
        C.check("x[host lead mask] = scalar" + s, x, xh)
        if len(shape) == 3:
            l2h = hmask(shape[:2])
            C.check("x[2-D lead mask]" + s, x[dev(l2h)], xh[l2h])

# aliasing
bh = hmask((1000,))
b = dev(bh)
C.check("b[b]", b[b], bh[bh], dtype=np.bool_)
xh = host((5000,), np.int64)
x = dev(xh)
mh = hmask((5000,), 0.1)
n = int(mh.sum())
x[dev(mh)] = x[:n]
xh[mh] = xh[:n].copy()
C.check("x[m] = x[:n] (value aliases x)", x, xh)
x[x > 0] = -x[x > 0]
xh[xh > 0] = -xh[xh > 0]
C.check("x[x > 0] = -x[x > 0]", x, xh)
bh = hmask((3000,))
b = dev(bh)
b[b] = dev(np.zeros(int(bh.sum()), bool))
bh[bh] = False
C.check("b[b] = False array (mask is the target)", b, bh, dtype=np.bool_)

# >3-D arrays (folded storage)
for shape in [(3, 4, 5, 6), (2, 3, 4, 5, 6)]:
    s = f" {shape}"
    xh = host(shape, np.float64)
    x = dev(xh)
    mh = hmask(shape)
    m = dev(mh)
    C.check("folded x[mask]" + s, x[m], xh[mh])
    C.check("folded x[host mask]" + s, x[mh], xh[mh])
    C.check("folded x[x > 0]" + s, x[x > 0], xh[xh > 0])
    vh = host((int(mh.sum()),), np.float64)
    x[m] = dev(vh)
    xh[mh] = vh
    C.check("folded x[mask] = array" + s, x, xh)
    x[m] = 3.0
    xh[mh] = 3.0
    C.check("folded x[mask] = scalar" + s, x, xh)

# reused masks: the cached nonzero count must follow every write to the mask
import charmnumeric.indexing as _ix  # noqa: E402

fetches = [0]
_fetch = _ix._fetch_int


def counting_fetch(x):
    fetches[0] += 1
    return _fetch(x)


_ix._fetch_int = counting_fetch


def reuse(label, mutate, mh, m):
    """x[m], x[m] = v and y[m, m] = v before and after ``mutate`` changes mh/m."""
    for step in ("", " after " + label):
        if step:
            mutate()
        xh = host((600,), np.float64)
        x = dev(xh)
        C.check("reuse x[m]" + step, x[m], xh[mh])
        vh = host((int(mh.sum()),), np.float64)
        x[m] = dev(vh)
        xh[mh] = vh
        C.check("reuse x[m] = array" + step, x, xh)
        yh = host((600, 600), np.int64)
        y = dev(yh)
        y[m, m] = 7
        yh[mh, mh] = 7
        C.check("reuse y[m, m] = 7" + step, y, yh)
        C.check("reuse nonzero(m)" + step, cnp.nonzero(m)[0], np.nonzero(mh)[0])


mh = hmask((600,))
m = dev(mh)
reuse("no write", lambda: None, mh, m)
before = fetches[0]
xh = host((600,), np.float64)
x = dev(xh)
for _ in range(3):
    C.check("unchanged mask x[m]", x[m], xh[mh])
C.check("unchanged mask: count is not fetched again", np.array(fetches[0]), np.array(before))


def write(fn_dev, fn_host):
    def go():
        fn_dev()
        fn_host()
    return go


nh = hmask((600,))
reuse("m[...] = other", write(lambda: m.__setitem__(Ellipsis, dev(nh)),
                              lambda: mh.__setitem__(Ellipsis, nh)), mh, m)
reuse("m[10:300] = True", write(lambda: m.__setitem__(slice(10, 300), True),
                                lambda: mh.__setitem__(slice(10, 300), True)), mh, m)
nh = hmask((600,))
reuse("m ^= other", write(lambda: m.__ixor__(dev(nh)), lambda: mh.__ixor__(nh)), mh, m)
idx = rng.integers(0, 600, 50)
reuse("m[idx] = False", write(lambda: m.__setitem__(dev(idx), False),
                              lambda: mh.__setitem__(idx, False)), mh, m)
sel = hmask((600,))
reuse("m[sel] = array", write(lambda: m.__setitem__(dev(sel), dev(~mh[sel])),
                              lambda: mh.__setitem__(sel, ~mh[sel])), mh, m)
reuse("m.fill(True)", write(lambda: m.fill(True), lambda: mh.fill(True)), mh, m)
nh = hmask((600,))
reuse("copyto(m, other)", write(lambda: cnp.copyto(m, dev(nh)),
                                lambda: np.copyto(mh, nh)), mh, m)
nh = hmask((600,))
reuse("logical_not(other, out=m)", write(lambda: cnp.logical_not(dev(nh), out=m),
                                         lambda: np.logical_not(nh, out=mh)), mh, m)
# a view of the mask, written through its root
mv = m[100:]
reuse("root m[...] = other (view mask)", write(lambda: m.__setitem__(Ellipsis, dev(~mh)),
                                                lambda: mh.__setitem__(Ellipsis, ~mh)),
      mh, m)
xh = host((500,), np.float64)
x = dev(xh)
C.check("view mask x[m[100:]]", x[mv], xh[mh[100:]])
m[150:250] = False
mh[150:250] = False
C.check("view mask after root write", x[mv], xh[mh[100:]])
# linked copies (integer index, transpose) refresh from their array
Mh = hmask((4, 600))
M = dev(Mh)
row = M[2]
C.check("linked row mask", x[row[:500]], xh[Mh[2, :500]])
M[2] = dev(~Mh[2])
Mh[2] = ~Mh[2]
C.check("linked row mask after array write", x[row[:500]], xh[Mh[2, :500]])
Th = hmask((600, 4))
T = dev(Th)
col = T.T
C.check("transposed mask", cnp.nonzero(col[1])[0], np.nonzero(Th.T[1])[0])
T[:, 1] = dev(~Th[:, 1])
Th[:, 1] = ~Th[:, 1]
C.check("transposed mask after array write", cnp.nonzero(col[1])[0], np.nonzero(Th.T[1])[0])
# a mask computed from an array keeps its value when that array changes
bh = host((600,), np.float64)
b = dev(bh)
cm = b > 0
cmh = bh > 0
C.check("computed mask", x[cm[:500]], xh[cmh[:500]])
b[...] = -b
C.check("computed mask after its input changes", x[cm[:500]], xh[cmh[:500]])
# repeat with device counts caches their sum the same way
Rh = rng.integers(0, 4, 300)
R = dev(Rh)
xh = host((300,), np.float64)
x = dev(xh)
C.check("repeat(x, R)", cnp.repeat(x, R), np.repeat(xh, Rh))
before = fetches[0]
C.check("repeat(x, R) again", cnp.repeat(x, R), np.repeat(xh, Rh))
C.check("repeat: sum is not fetched again", np.array(fetches[0]), np.array(before))
R[5:50] = 3
Rh[5:50] = 3
C.check("repeat(x, R) after R[5:50] = 3", cnp.repeat(x, R), np.repeat(xh, Rh))
R += 1
Rh += 1
C.check("repeat(x, R) after R += 1", cnp.repeat(x, R), np.repeat(xh, Rh))
_ix._fetch_int = _fetch

C.raises("mask shape mismatch", IndexError, lambda: dev(np.ones(5))[dev(np.ones(4, bool))])
C.raises("value count mismatch", ValueError,
         lambda: dev(np.ones(5)).__setitem__(dev(np.ones(5, bool)), np.ones(3)))

C.finish()
