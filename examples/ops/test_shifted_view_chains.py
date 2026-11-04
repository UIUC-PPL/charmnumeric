"""Chains of shifted views split over several fused kernels in one DAG, vs NumPy.

A kernel that keeps several results (e.g. s1 = s0 + x[1:] and t2 = x[2:] * 0.5, both
read by the next kernel) writes them all in one chare layout; arrays sized past the
minimum tiles (CPU 2^18 / 512 / 64, GPU 2^20 / 1024 / 128) put tile boundaries inside
the views.
"""

import numpy as np

import charmnumeric as cnp
from charmtyles import ast as cta
from _ops_common import Checker, connect, sync

iface = connect()
C = Checker("test_shifted_view_chains", iface)
TOL = dict(rtol=1e-12, atol=0.0)
PAD = 12


def view(a, k, shape):
    return a[tuple(slice(k, k + n) for n in shape)]


def make_base(shape):
    full = tuple(n + PAD for n in shape)
    h = np.arange(np.prod(full), dtype=np.float64).reshape(full)
    b = cnp.arange(int(np.prod(full)), dtype=np.float64)
    if len(full) > 1:
        b = b.reshape(full)
    return b, h


default_max = cta.max_fused_ops
for shape in ((1100001,), (1100, 1037), (133, 131, 9)):
    base, bh = make_base(shape)
    C.check(f"base {shape}", base, bh)
    for max_ops in (default_max, 2, 3, 4, 5):
        cta.set_max_fused_ops(max_ops)
        for K in (4, 10):
            s, sh = view(base, 0, shape) * 1.0, view(bh, 0, shape) * 1.0
            for k in range(1, K):
                s = s + view(base, k, shape) * (1.0 / k)
                sh = sh + view(bh, k, shape) * (1.0 / k)
            C.check(f"chain {shape} K={K} max_fused_ops={max_ops}", s, sh, **TOL)

    # one kernel keeps an aligned and a shifted result; the next kernel reads both
    cta.set_max_fused_ops(2)
    a = view(base, 0, shape) * 1.0
    t = view(base, 2, shape) * 0.5
    r = a + t
    r = r + view(base, 3, shape)
    rh = view(bh, 0, shape) + view(bh, 2, shape) * 0.5 + view(bh, 3, shape)
    C.check(f"two kept results {shape}", r, rh, **TOL)

    # a whole-array set_region fused with a kept shifted result
    cta.set_max_fused_ops(default_max)
    u = cnp.zeros(shape)
    sync(iface)
    t = view(base, 2, shape) * 0.5
    u[...] = t + 1.0
    cta.set_max_fused_ops(1)
    r = view(base, 1, shape) + t
    cta.set_max_fused_ops(default_max)
    th = view(bh, 2, shape) * 0.5
    C.check(f"set_region + kept shifted result: target {shape}", u, th + 1.0, **TOL)
    C.check(f"set_region + kept shifted result: reader {shape}", r, view(bh, 1, shape) + th,
            **TOL)
cta.set_max_fused_ops(default_max)

C.finish()
