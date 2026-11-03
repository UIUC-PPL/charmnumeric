"""In-place stencil updates over several chares vs NumPy: every iteration reads the halo
of the array it overwrites, so a halo send must see the values from before the write."""

import numpy as np

import charmnumeric as cnp
from charmtyles import ast as cta
from _ops_common import Checker, connect, floats, sync

iface = connect()
C = Checker("test_inplace_halo", iface)
TOL = dict(rtol=1e-12, atol=1e-12)


def step1(u):
    u[1:-1] = 0.5 * (u[:-2] + u[2:])


def step2(u):
    u[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])


def step3(u):
    u[1:-1, 1:-1, 1:-1] = (u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]
                           + u[1:-1, 2:, 1:-1] + u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]) / 6.0


# shapes past the GPU minimum tiles (1D 2^20, 2D 1024, 3D 128) give several chares
for shape, step in (((3000001,), step1), ((40001,), step1), ((2600, 37), step2),
                    ((37, 2600), step2), ((300, 9, 7), step3)):
    for synced in (False, True):
        u = floats(shape, -1.0, 1.0, seed=len(shape))
        uh = C.fetch(u)
        for _ in range(12):
            step(u)
            step(uh)
            if synced:
                sync(iface)
        C.check(f"{step.__name__} {shape} synced={synced}", u, uh, **TOL)



def sum_shift(u):
    u[1:-1] = u[:-2] + u[2:]


def shift_down(a):
    a[1:] = a[:-1]


def shift_right(a):
    a[:, 1:] = a[:, :-1]


def add_reversed(a):
    a += a[::-1]


# overlapping in-place updates, fused (default) and split into small kernels
default_max = cta.max_fused_ops
for max_ops in (default_max, 2):
    cta.set_max_fused_ops(max_ops)
    tag = f"max_fused_ops={max_ops}"
    for shape, steps in (((3000001,), (sum_shift, shift_down, add_reversed)),
                         ((2600, 37), (sum_shift, shift_down, shift_right, add_reversed)),
                         ((1500, 1100), (sum_shift, shift_down, shift_right))):
        for step in steps:
            u = floats(shape, -1.0, 1.0, seed=len(shape) + 7)
            uh = C.fetch(u)
            for _ in range(3):
                step(u)
                step(uh)
            C.check(f"{step.__name__} {shape} {tag}", u, uh, **TOL)
    for shape in ((3000001,), (2600, 37)):
        a = floats(shape, -1.0, 1.0, seed=3)
        b = cnp.zeros(shape)
        ah, bh = C.fetch(a), np.zeros(shape)
        for x, y in ((a, b), (ah, bh)):
            y[...] = 2.0 * x + 1.0
            x[...] = x - 3.0
        C.check(f"B=f(A); A=g(A) A {shape} {tag}", a, ah, **TOL)
        C.check(f"B=f(A); A=g(A) B {shape} {tag}", b, bh, **TOL)
cta.set_max_fused_ops(default_max)

C.finish()
