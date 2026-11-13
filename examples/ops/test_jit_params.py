"""Fused kernels with more arrays and scalars than one GPU launch can pass (ptxas allows
4352 bytes of parameters below PTX ISA 8.1) vs NumPy: long unsynchronized chains, many
live outputs, many distinct inputs and many views of one array. The JIT splits such
kernels; it used to abort the server."""

import numpy as np

import charmnumeric as cnp
from charmtyles import ast as cta
from _ops_common import Checker, connect, sync

iface = connect()
C = Checker("test_jit_params", iface)
N = 300001


def chain(iters, n=N):
    """x = x * c; y = x - 1 without synchronizing (lat.py's batch loop)."""
    x, xh = cnp.ones(n, dtype=np.float32), np.ones(n, dtype=np.float32)
    for _ in range(iters):
        x = x * 1.0000001
        y = x - 1.0
        xh = xh * np.float32(1.0000001)
        yh = xh - np.float32(1.0)
    sync(iface)
    C.check(f"chain {iters} x", x, xh, rtol=1e-6)
    C.check(f"chain {iters} y", y, yh, rtol=1e-6, atol=1e-6)


default_max = cta.max_fused_ops
chain(200)
cta.set_max_fused_ops(0)
try:
    chain(300)
finally:
    cta.set_max_fused_ops(default_max)

# every intermediate stays live, so each one is a kernel output
for shape in ((N,), (700, 31), (40, 9, 23)):
    a = cnp.ones(shape) * 0.5
    sync(iface)
    ah = np.full(shape, 0.5)
    outs, outs_h = [], []
    v, vh = a, ah
    for k in range(120):
        v = v * 0.99 + (k % 7) * 0.01
        vh = vh * 0.99 + (k % 7) * 0.01
        outs.append(v)
        outs_h.append(vh)
    sync(iface)
    for k in (0, 1, 59, 118, 119):
        C.check(f"live outputs {shape} [{k}]", outs[k], outs_h[k], rtol=1e-12)

# many distinct input arrays, mixed dtypes
ins = [cnp.ones(N, dtype=np.float32 if k % 3 else np.float64) * float(k + 1) for k in range(110)]
sync(iface)
acc, acc_h = ins[0] * 1.0, np.full(N, 1.0)
for k in range(1, 110):
    acc = acc + ins[k] * 0.5
    acc_h = acc_h + (k + 1) * 0.5
C.check("110 inputs", acc, acc_h, rtol=1e-12)

# many views of one array (each a separate kernel argument)
M = 200001
base = cnp.arange(M + 160, dtype=np.float64)
base_h = np.arange(M + 160, dtype=np.float64)
s, s_h = base[0:M] * 1.0, base_h[0:M].copy()
for k in range(1, 150):
    s = s + base[k:k + M] * (1.0 / k)
    s_h = s_h + base_h[k:k + M] * (1.0 / k)
C.check("150 views", s, s_h, rtol=1e-12)

C.finish()
