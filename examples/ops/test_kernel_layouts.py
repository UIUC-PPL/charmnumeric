"""Fused kernels whose results cannot all share the kernel's chare layout, vs NumPy.

A fused kernel writes its kept results and set_region buffers in one chare layout. When
an array the kernel writes is pinned elsewhere (an earlier DAG's array, a matmul operand
at offset 0), the backend writes it in the kernel's layout and redistributes it in a
follow-up node. Arrays are sized past the minimum tiles (CPU 2^18 / 512 / 64, GPU
2^20 / 1024 / 128) so the layouts differ inside the views.
"""

import numpy as np

import charmnumeric as cnp
from charmtyles import ast as cta
from _ops_common import Checker, connect, sync

iface = connect()
C = Checker("test_kernel_layouts", iface)
TOL = dict(rtol=1e-12, atol=0.0)
PAD = 12
default_max = cta.max_fused_ops


def view(a, k, shape):
    return a[tuple(slice(k, k + n) for n in shape)]


def make_base(shape):
    full = tuple(n + PAD for n in shape)
    b = cnp.arange(int(np.prod(full)), dtype=np.float64)
    if len(full) > 1:
        b = b.reshape(full)
    return b, np.arange(np.prod(full), dtype=np.float64).reshape(full)


for shape in ((1100001,), (1100, 1037), (133, 131, 9)):
    base, bh = make_base(shape)
    sync(iface)
    small = tuple(n - 1 for n in shape)

    # an earlier DAG's array at a view offset written from a matmul operand
    if len(shape) == 1:
        u = view(base, 3, shape) * 1.0
        sync(iface)
        t = view(base, 0, shape) * 2.0
        u[...] = t + 1.0
        m = cnp.arange(2 * shape[0], dtype=np.float64).reshape(2, shape[0]) * 1e-6
        y = m @ t
        th = view(bh, 0, shape) * 2.0
        mh = np.arange(2 * shape[0], dtype=np.float64).reshape(2, shape[0]) * 1e-6
        C.check(f"set_region target {shape}", u, th + 1.0, **TOL)
        C.check(f"matmul operand {shape}", t, th, **TOL)
        C.check(f"matmul {shape}", y, mh @ th, rtol=1e-9)

    # one kernel writes two earlier arrays at different offsets
    u = view(base, 0, shape) * 1.0
    w = view(base, 5, shape) * 1.0
    sync(iface)
    t = view(base, 1, shape) * 0.5
    u[...] = t + 1.0
    w[...] = t * 2.0
    th = view(bh, 1, shape) * 0.5
    C.check(f"two fixed targets: first {shape}", u, th + 1.0, **TOL)
    C.check(f"two fixed targets: second {shape}", w, th * 2.0, **TOL)

    # a whole-array set_region and a shifted partial one in one kernel
    u = cnp.zeros(shape)
    w = cnp.zeros(tuple(n + 2 for n in shape))
    sync(iface)
    t = view(base, 0, shape) * 1.0
    u[...] = t
    w[tuple(slice(1, 1 + n) for n in shape)] = t + 1.0
    th = view(bh, 0, shape) * 1.0
    wh = np.zeros(tuple(n + 2 for n in shape))
    wh[tuple(slice(1, 1 + n) for n in shape)] = th + 1.0
    C.check(f"two set_regions: whole {shape}", u, th, **TOL)
    C.check(f"two set_regions: shifted {shape}", w, wh, **TOL)

    # results of different shapes
    a = view(base, 1, shape) * 2.0
    b = view(base, 0, small) * 3.0
    c = view(a, 0, small) + b
    ah, bh2 = view(bh, 1, shape) * 2.0, view(bh, 0, small) * 3.0
    C.check(f"mixed shapes: larger {shape}", a, ah, **TOL)
    C.check(f"mixed shapes: smaller {shape}", b, bh2, **TOL)
    C.check(f"mixed shapes: sum {shape}", c, view(ah, 0, small) + bh2, **TOL)

    # kernel 1 keeps a and b; kernel 2 keeps c and writes a
    cta.set_max_fused_ops(2)
    a = view(base, 0, shape) * 1.0
    b = view(base, 2, shape) * 0.5
    c = view(base, 3, shape) * 2.0
    cta.set_max_fused_ops(3)
    a[...] = c + 1.0
    cta.set_max_fused_ops(default_max)
    r = a + b + c
    ch = view(bh, 3, shape) * 2.0
    bh3 = view(bh, 2, shape) * 0.5
    C.check(f"across kernels: written {shape}", a, ch + 1.0, **TOL)
    C.check(f"across kernels: kept {shape}", b, bh3, **TOL)
    C.check(f"across kernels: sum {shape}", r, ch + 1.0 + bh3 + ch, **TOL)

    # copy() of a shifted result fused with its producer
    t = view(base, 2, shape) * 0.5
    y = t.copy()
    z = view(base, 1, shape) + t
    th = view(bh, 2, shape) * 0.5
    C.check(f"copy {shape}", y, th, **TOL)
    C.check(f"copy source reader {shape}", z, view(bh, 1, shape) + th, **TOL)

C.finish()
