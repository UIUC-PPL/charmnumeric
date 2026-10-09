"""Shared helpers for the elementwise-op tests in this directory.

Environment (same as examples/cupynumeric):
    CNUM_HOST, CNUM_PORT, CNUM_ODF   backend to connect to
    CNUM_SIM=1                       run against the NumPy simulator in
                                     ../cupynumeric/_cnum_sim.py instead

Inputs are generated on the backend from index grids (no per-element
upload), fetched with .get(), and the NumPy reference is computed from the
fetched values. Every check prints PASS/FAIL; the last line is
``PASS <test>`` or ``FAIL <test>: <n> failures``.

"""

import os
import sys

import numpy as np

import charmnumeric as cnp
from charmtyles.core import sync

_HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.environ.get("CNUM_SIM", "") not in ("", "0")
if SIM:
    sys.path.insert(0, os.path.join(_HERE, "..", "cupynumeric"))
    import _cnum_sim  # noqa: F401  (installs the simulator hooks)


def connect():
    if SIM:
        from _cnum_sim import SimInterface

        return SimInterface()
    iface = cnp.CharmNumericInterface()
    iface.connect(
        os.environ.get("CNUM_HOST", "127.0.0.1"),
        int(os.environ.get("CNUM_PORT", "1234")),
        int(os.environ.get("CNUM_ODF", "4")),
    )
    return iface


# Shapes: 1-D/2-D/3-D, non-tile-aligned sizes, and sizes above the CPU
# minimum tiles (1-D 262144, 2-D 512, 3-D 64) so that +p2 runs split arrays
# over several chares.
SHAPES = [(37,), (1000,), (300001,), (37, 41), (600, 7), (5, 7, 9), (70, 3, 5)]


def _axis_index(shape, axis, dtype):
    """g[...] = index along `axis`, as a Hillis-Steele prefix sum of ones.

    Every slice used has length >= 2 unless the axis itself is short, which
    keeps clear of size-1 (broadcast) operand views.
    """
    nd = len(shape)
    n = shape[axis]

    def sl(s):
        return tuple(s if d == axis else slice(None) for d in range(nd))

    g = cnp.zeros(shape, dtype=dtype)
    if n == 1:
        return g
    g[sl(slice(1, None))] = g[sl(slice(1, None))] + 1
    k = 1
    while k < n:
        h = g.copy()
        h[sl(slice(k, None))] = g[sl(slice(k, None))] + g[sl(slice(0, n - k))]
        g = h
        k *= 2
    return g


def linear_index(shape, dtype=np.float64):
    shape = tuple(shape)
    stride = 1
    L = None
    for axis in range(len(shape) - 1, -1, -1):
        if shape[axis] > 1:
            term = _axis_index(shape, axis, dtype) * stride
            L = term if L is None else L + term
        stride *= shape[axis]
    return cnp.zeros(shape, dtype=dtype) if L is None else L


def floats(shape, lo=-3.0, hi=3.0, seed=0, dtype=np.float64):
    """Pseudo-random floats in [lo, hi) (logistic map on the linear index)."""
    shape = tuple(shape)
    total = float(np.prod(shape))
    L = linear_index(shape, dtype)
    c = 0.6 * ((seed * 0.6180339887498949) % 1.0) - 0.3
    x = 0.05 + 0.4 * (0.5 + 0.5 * cnp.tanh(2.0 * ((L + 0.5) / total - 0.5) + c))
    for _ in range(16):
        x = 3.99 * x * (1.0 - x)
    return lo + (hi - lo) * x


def halves(shape, dtype=np.float64):
    """Exact multiples of 0.5 around zero (round-half-even cases)."""
    total = int(np.prod(shape))
    return (linear_index(shape, dtype) - float(total // 2)) * 0.5


def ints(shape, lo=-5, hi=5, dtype=np.int64, mult=1):
    """Integers cycling through [lo, hi] (truncating int division on L >= 0)."""
    L = linear_index(shape, dtype)
    if mult != 1:
        L = L * mult
    m = hi - lo + 1
    return L - (L / m) * m + lo


def with_specials(x, values=(np.nan, np.inf, -np.inf, -0.0, 0.0)):
    """Set the first elements of a 1-D float array to special values."""
    for i, v in enumerate(values):
        if i < x.shape[0]:
            x[i] = float(v)
    return x


class Checker:
    def __init__(self, name, iface):
        self.name = name
        self.iface = iface
        self.failures = []
        self.count = 0

    def fetch(self, x):
        sync(self.iface)  # inputs are flushed before any op reads them
        return np.asarray(x.get(self.iface))

    def check(self, label, got, expected, rtol=0.0, atol=0.0, dtype=None):
        self.count += 1
        if hasattr(got, "get"):
            want_dtype = got.dtype if dtype is None else dtype
            got = np.asarray(got.get(self.iface))
        else:
            want_dtype = dtype
        expected = np.asarray(expected)
        msg = None
        if got.shape != expected.shape:
            msg = f"shape {got.shape} != {expected.shape}"
        elif want_dtype is not None and got.dtype != np.dtype(want_dtype):
            msg = f"dtype {got.dtype} != {np.dtype(want_dtype)}"
        else:
            exp = expected.astype(got.dtype) if expected.dtype != got.dtype else expected
            if np.issubdtype(got.dtype, np.floating):
                ok = np.allclose(got, exp, rtol=rtol, atol=atol, equal_nan=True)
                if rtol == 0.0 and atol == 0.0:
                    ok = np.array_equal(got, exp, equal_nan=True)
            else:
                ok = np.array_equal(got, exp)
            if not ok:
                bad = np.argwhere(~np.isclose(got, exp, rtol=rtol, atol=atol, equal_nan=True))
                idx = tuple(bad[0]) if len(bad) else ()
                msg = f"{len(bad)} mismatches, first at {idx}: got {got[idx]!r} expected {exp[idx]!r}"
        if msg is None:
            print(f"  PASS {label}")
        else:
            print(f"  FAIL {label}: {msg}")
            self.failures.append(label)

    def raises(self, label, exc, fn):
        self.count += 1
        try:
            fn()
        except exc:
            print(f"  PASS {label} (raises {exc.__name__})")
            return
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {label}: raised {type(e).__name__}: {e}")
            self.failures.append(label)
            return
        print(f"  FAIL {label}: did not raise {exc.__name__}")
        self.failures.append(label)

    def finish(self):
        if self.failures:
            print(f"FAIL {self.name}: {len(self.failures)}/{self.count} failures: "
                  + ", ".join(self.failures[:20]))
            sys.exit(1)
        print(f"PASS {self.name} ({self.count} checks)")
        sys.exit(0)


def views_of(x):
    """A few non-trivial views of x (slices, strides, interior)."""
    nd = x.ndim
    out = []
    if nd == 1:
        out.append(("[1:-1]", (slice(1, -1),)))
        out.append(("[::3]", (slice(None, None, 3),)))
    elif nd == 2:
        out.append(("[1:-1,::2]", (slice(1, -1), slice(None, None, 2))))
        out.append(("[2:,1:]", (slice(2, None), slice(1, None))))
    else:
        out.append(("[1:,:,::2]", (slice(1, None), slice(None), slice(None, None, 2))))
    return out
