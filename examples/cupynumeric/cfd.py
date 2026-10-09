#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/cfd.py (2-D channel flow, periodic BCs).

Verdict: direct port. The solver functions take the array module as `np`
(as in the original) and are run with charmnumeric and with numpy. Changes:
  * np.sum(u) is the native full reduction (cnp.sum, a (1,) array).
  * stencil_hint calls are dropped (Legate-specific).
  * Integer column/row indices such as u[1:-1, 0] or p[-1, :] are written
    as length-1 slices (u[1:-1, 0:1], p[-1:, :]). In charmnumeric an integer
    index already keeps the dim, giving an (n, 1) view, but NumPy would drop
    it, and the same function has to run under both.
"""

import argparse
import math
import os
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from _cnum_connect import get_interface  # noqa: E402


def build_up_b(np, rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = rho * (
        1
        / dt
        * (
            (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
            + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
        )
        - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2
        - 2
        * (
            (u[2:, 1:-1] - u[0:-2, 1:-1])
            / (2 * dy)
            * (v[1:-1, 2:] - v[1:-1, 0:-2])
            / (2 * dx)
        )
        - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2
    )

    # Periodic BC Pressure @ x = 2
    b[1:-1, -1:] = rho * (
        1
        / dt
        * (
            (u[1:-1, 0:1] - u[1:-1, -2:-1]) / (2 * dx)
            + (v[2:, -1:] - v[0:-2, -1:]) / (2 * dy)
        )
        - ((u[1:-1, 0:1] - u[1:-1, -2:-1]) / (2 * dx)) ** 2
        - 2
        * (
            (u[2:, -1:] - u[0:-2, -1:])
            / (2 * dy)
            * (v[1:-1, 0:1] - v[1:-1, -2:-1])
            / (2 * dx)
        )
        - ((v[2:, -1:] - v[0:-2, -1:]) / (2 * dy)) ** 2
    )

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0:1] = rho * (
        1
        / dt
        * (
            (u[1:-1, 1:2] - u[1:-1, -1:]) / (2 * dx)
            + (v[2:, 0:1] - v[0:-2, 0:1]) / (2 * dy)
        )
        - ((u[1:-1, 1:2] - u[1:-1, -1:]) / (2 * dx)) ** 2
        - 2
        * (
            (u[2:, 0:1] - u[0:-2, 0:1])
            / (2 * dy)
            * (v[1:-1, 1:2] - v[1:-1, -1:])
            / (2 * dx)
        )
        - ((v[2:, 0:1] - v[0:-2, 0:1]) / (2 * dy)) ** 2
    )

    return b


def pressure_poisson_periodic(np, b, nit, p, dx, dy):
    pn = np.empty_like(p)

    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2
            + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2
        ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1:] = (
            (pn[1:-1, 0:1] + pn[1:-1, -2:-1]) * dy**2
            + (pn[2:, -1:] + pn[0:-2, -1:]) * dx**2
        ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1:]

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0:1] = (
            (pn[1:-1, 1:2] + pn[1:-1, -1:]) * dy**2
            + (pn[2:, 0:1] + pn[0:-2, 0:1]) * dx**2
        ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0:1]

        # Wall boundary conditions, pressure
        p[-1:, :] = p[-2:-1, :]  # dp/dy = 0 at y = 2
        p[0:1, :] = p[1:2, :]  # dp/dy = 0 at y = 0

    return p


def channel_flow(np, iteration, warmup, nit, nx, ny, dt, dx, dy, rho, nu, F):
    # initial conditions
    u = np.zeros((ny, nx))
    un = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    vn = np.zeros((ny, nx))
    p = np.ones((ny, nx))
    b = np.zeros((ny, nx))

    stepcount = 0
    for _ in range(iteration + warmup):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(np, rho, dt, dx, dy, u, v)
        p = pressure_poisson_periodic(np, b, nit, p, dx, dy)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
            - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
            + nu
            * (
                dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            )
            + F * dt
        )
        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
            - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
            + nu
            * (
                dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                + dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
            )
        )
        # Periodic BC u @ x = 2
        u[1:-1, -1:] = (
            un[1:-1, -1:]
            - un[1:-1, -1:] * dt / dx * (un[1:-1, -1:] - un[1:-1, -2:-1])
            - vn[1:-1, -1:] * dt / dy * (un[1:-1, -1:] - un[0:-2, -1:])
            - dt / (2 * rho * dx) * (p[1:-1, 0:1] - p[1:-1, -2:-1])
            + nu
            * (
                dt / dx**2 * (un[1:-1, 0:1] - 2 * un[1:-1, -1:] + un[1:-1, -2:-1])
                + dt / dy**2 * (un[2:, -1:] - 2 * un[1:-1, -1:] + un[0:-2, -1:])
            )
            + F * dt
        )
        # Periodic BC u @ x = 0
        u[1:-1, 0:1] = (
            un[1:-1, 0:1]
            - un[1:-1, 0:1] * dt / dx * (un[1:-1, 0:1] - un[1:-1, -1:])
            - vn[1:-1, 0:1] * dt / dy * (un[1:-1, 0:1] - un[0:-2, 0:1])
            - dt / (2 * rho * dx) * (p[1:-1, 1:2] - p[1:-1, -1:])
            + nu
            * (
                dt / dx**2 * (un[1:-1, 1:2] - 2 * un[1:-1, 0:1] + un[1:-1, -1:])
                + dt / dy**2 * (un[2:, 0:1] - 2 * un[1:-1, 0:1] + un[0:-2, 0:1])
            )
            + F * dt
        )
        # Periodic BC v @ x = 2
        v[1:-1, -1:] = (
            vn[1:-1, -1:]
            - un[1:-1, -1:] * dt / dx * (vn[1:-1, -1:] - vn[1:-1, -2:-1])
            - vn[1:-1, -1:] * dt / dy * (vn[1:-1, -1:] - vn[0:-2, -1:])
            - dt / (2 * rho * dy) * (p[2:, -1:] - p[0:-2, -1:])
            + nu
            * (
                dt / dx**2 * (vn[1:-1, 0:1] - 2 * vn[1:-1, -1:] + vn[1:-1, -2:-1])
                + dt / dy**2 * (vn[2:, -1:] - 2 * vn[1:-1, -1:] + vn[0:-2, -1:])
            )
        )
        # Periodic BC v @ x = 0
        v[1:-1, 0:1] = (
            vn[1:-1, 0:1]
            - un[1:-1, 0:1] * dt / dx * (vn[1:-1, 0:1] - vn[1:-1, -1:])
            - vn[1:-1, 0:1] * dt / dy * (vn[1:-1, 0:1] - vn[0:-2, 0:1])
            - dt / (2 * rho * dy) * (p[2:, 0:1] - p[0:-2, 0:1])
            + nu
            * (
                dt / dx**2 * (vn[1:-1, 1:2] - 2 * vn[1:-1, 0:1] + vn[1:-1, -1:])
                + dt / dy**2 * (vn[2:, 0:1] - 2 * vn[1:-1, 0:1] + vn[0:-2, 0:1])
            )
        )

        # Wall BC: u,v = 0 @ y = 0,2
        u[0:1, :] = 0
        u[-1:, :] = 0
        v[0:1, :] = 0
        v[-1:, :] = 0

        stepcount += 1

    sum_u = np.sum(u)
    sum_un = np.sum(un)
    return u, v, p, sum_u, sum_un, stepcount


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--nx", type=int, default=41, dest="nx")
    parser.add_argument("-y", "--ny", type=int, default=41, dest="ny")
    parser.add_argument("-n", "--np", type=int, default=10, dest="np")
    parser.add_argument("-i", "--iter", type=int, default=5, dest="iter")
    parser.add_argument("--dt", dest="dt", type=float, default=None)
    parser.add_argument("--warmup", dest="warmup", type=int, default=1)
    args = parser.parse_args()

    nx, ny, nit = args.nx, args.ny, args.np
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    rho, nu, F = 1, 0.1, 1
    if args.dt is None:
        sigma = 0.0009
        dt = sigma * dx * dy / nu
    else:
        dt = args.dt

    interface = get_interface()
    print("Running channel flow...")
    t0 = time.perf_counter()
    u, v, p, sum_u_a, sum_un_a, steps = channel_flow(
        cnp, args.iter, args.warmup, nit, nx, ny, dt, dx, dy, rho, nu, F
    )
    sum_u = U.to_float(sum_u_a, interface)
    sum_un = U.to_float(sum_un_a, interface)
    total = (time.perf_counter() - t0) * 1e3
    udiff = (sum_u - sum_un) / sum_u
    print(
        f"Elapsed Time: {total:.3f} ms (incl. sync). Steps: {steps - args.warmup}. "
        f"Warmups: {args.warmup}. dt: {dt}. udiff: {udiff}. sum_u: {sum_u}. sum_un: {sum_un}."
    )
    u_h, v_h, p_h = u.get(interface), v.get(interface), p.get(interface)

    u_r, v_r, p_r, su_r, sun_r, _ = channel_flow(
        numpy, args.iter, args.warmup, nit, nx, ny, dt, dx, dy, rho, nu, F
    )
    chk.true("udiff not NaN", not math.isnan(udiff))
    chk.close("u", u_h, u_r, rtol=1e-9, atol=1e-12)
    chk.close("v", v_h, v_r, rtol=1e-9, atol=1e-12)
    chk.close("p", p_h, p_r, rtol=1e-9, atol=1e-12)
    chk.close("sum(u)", sum_u, su_r, rtol=1e-9)
    chk.close("sum(un)", sum_un, sun_r, rtol=1e-9)


if __name__ == "__main__":
    U.run("cfd", main)
