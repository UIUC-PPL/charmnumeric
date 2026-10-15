"""JIT edge cases vs NumPy: signed zeros (sign, maximum/minimum, clip, complex
sign), empty arrays, GPU launch shapes (extents over 65535 off the
contiguous dim, tile boundaries). CNUM_BIG_SHAPES=1 adds 8e7-element shapes."""

import os

import numpy as np

import charmnumeric as cnp
from _ops_common import Checker, connect

iface = connect()
C = Checker("test_jit_edges", iface)


def check_bits(label, got, expected):
    """Values and the sign bits of both parts."""
    g = np.asarray(got.get(iface))
    e = np.asarray(expected)
    ok = g.shape == e.shape and g.dtype == e.dtype and np.array_equal(g, e, equal_nan=True)
    for part in (np.real, np.imag):
        ok = ok and np.array_equal(np.signbit(part(g)), np.signbit(part(e)))
    C.count += 1
    if ok:
        print(f"  PASS {label}")
    else:
        print(f"  FAIL {label}: got {g!r} expected {e!r}")
        C.failures.append(label)


with np.errstate(all="ignore"):
    for dt in (np.float64, np.float32):
        xs = np.array([0.0, -0.0, 0.0, -0.0, np.nan, 1.0, -0.0, np.nan], dt)
        ys = np.array([-0.0, 0.0, 0.0, -0.0, -0.0, np.nan, np.nan, np.nan], dt)
        x, y = cnp.array(xs), cnp.array(ys)
        s = f" {np.dtype(dt)}"
        # fmax/fmin ties follow NumPy's SIMD loops (which differ from its scalar tail)
        for n in ("maximum", "minimum", "fmax", "fmin"):
            check_bits(n + s, getattr(cnp, n)(x, y), getattr(np, n)(xs, ys))
        check_bits("sign" + s, cnp.sign(x), np.sign(xs))
        check_bits("clip(x, 0, 1)" + s, cnp.clip(x, 0.0, 1.0), np.clip(xs, 0.0, 1.0))
        check_bits("clip(x, -1, -0)" + s, cnp.clip(x, -1.0, -0.0), np.clip(xs, -1.0, -0.0))

    inf, nan = np.inf, np.nan
    cs = np.array([complex(-0.0, -0.0), complex(0.0, -0.0), complex(3, -4), complex(nan, 0),
                   complex(0, nan), complex(inf, 1), complex(-inf, nan), complex(inf, inf),
                   complex(2, -inf), complex(1e-310, 0)])
    for dt in (np.complex128, np.complex64):
        check_bits(f"sign {np.dtype(dt)}", cnp.sign(cnp.array(cs.astype(dt))), np.sign(cs.astype(dt)))

for shape in [(0,), (3, 0), (0, 4, 5)]:
    a, ah = cnp.zeros(shape), np.zeros(shape)
    b, bh = cnp.zeros(shape) + 1.0, np.zeros(shape) + 1.0
    C.check(f"a + 2b {shape}", a + b * 2.0, ah + bh * 2.0)
    C.check(f"sin {shape}", cnp.sin(a), np.sin(ah))
    C.check(f"maximum {shape}", cnp.maximum(a, b), np.maximum(ah, bh))
    C.check(f"sign {shape}", cnp.sign(b - 2.0), np.sign(bh - 2.0))
    C.check(f"where {shape}", cnp.where(a < b, a, b), np.where(ah < bh, ah, bh))


def ramp(shape, dtype=np.float64):
    """Distinct values along every axis (broadcast aranges)."""
    out = cnp.zeros(shape, dtype=dtype)
    host = np.zeros(shape, dtype=dtype)
    scale = 1.0
    for ax in range(len(shape) - 1, -1, -1):
        idx = [1] * len(shape)
        idx[ax] = shape[ax]
        r = np.arange(shape[ax], dtype=dtype).reshape(idx) * scale
        out = out + cnp.array(r)
        host = host + r
        scale *= 0.5
    return out, host


# Large extents off the contiguous dim (were gridDim.y/z > 65535 on GPU) and rank 4
SHAPES = [(3, 70000, 2), (2, 70001, 3), (70000, 3), (2, 3, 70000, 2), (2, 3, 4, 5, 6)]
if os.environ.get("CNUM_BIG_SHAPES", "") not in ("", "0"):
    SHAPES += [(2, 40_000_000), (70000, 300), (3, 70000, 70), (2, 2, 17_000_000)]
for shape in SHAPES:
    a, ah = ramp(shape)
    C.check(f"a * 3 + 1 {shape}", a * 3.0 + 1.0, ah * 3.0 + 1.0)
    C.check(f"maximum(a, a[..., ::-1]) {shape}", cnp.maximum(a, a[..., ::-1]),
            np.maximum(ah, ah[..., ::-1]))
    del a

# Tile boundaries: in-place updates on prefixes must not touch the rest
for n in (1, 255, 256, 257, 1023, 1025, 65537):
    a = cnp.zeros(n + 300)
    a[:n] += 1.0
    h = np.zeros(n + 300)
    h[:n] += 1.0
    C.check(f"a[:{n}] += 1", a, h)
    b = cnp.zeros((3, n + 40))
    b[:, 1:n + 1] = b[:, 1:n + 1] * 2.0 + 5.0
    hb = np.zeros((3, n + 40))
    hb[:, 1:n + 1] = hb[:, 1:n + 1] * 2.0 + 5.0
    C.check(f"b[:, 1:{n + 1}] = 2b + 5", b, hb)

C.finish()
