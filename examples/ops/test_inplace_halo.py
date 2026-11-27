"""In-place stencil updates over several chares vs NumPy: every iteration reads the halo
of the array it overwrites, so a halo send must see the values from before the write."""

import numpy as np

import charmnumeric as cnp
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

C.finish()
