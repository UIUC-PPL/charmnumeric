#!/usr/bin/env python
"""Chares past an array's end. A small result aligned with the far end of a long view gets
the minimum tile at a large offset, so the 1-D chare grid grows to about n / 2**18 chares;
the long array, with a tile near n / (odf * PEs), then has chares c with c * tile >= 2**31.
Their start used to be computed in int and wrapped, so such a chare took a piece of the
array from 0 again: sums came out too large or the server hung, with n far below 2**31.

Env: CNUM_HOST, CNUM_PORT, CNUM_ODF; GRID_N (default 2**28 int8 elements, enough for
odf * PEs up to 64 on the CPU build).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cupynumeric"))

import charmnumeric as cnp  # noqa: E402
import _cnum_connect  # noqa: E402
import _cnum_util as U  # noqa: E402


def total(chk, iface, label, a, want):
    got = int(np.asarray(cnp.sum(a, dtype=np.int64).get(iface)).reshape(-1)[0])
    chk.true(label, got == want, "" if got == want else f"got {got}, expected {want}")


def main(chk):
    iface = _cnum_connect.get_interface()
    n = int(float(os.environ.get("GRID_N", 2**28)))
    k = 1000
    v = cnp.ones((n,), dtype=np.int8)
    total(chk, iface, "sum(v)", v, n)
    x = v[n - k:] * 2
    total(chk, iface, f"sum(v[n - {k}:] * 2)", x, 2 * k)
    del x
    total(chk, iface, "sum(v) after", v, n)
    total(chk, iface, "sum(v * 3 - 1)", v * 3 - 1, 2 * n)
    total(chk, iface, "sum(v[1:] - v[:-1])", v[1:] - v[:-1], 0)
    c = int(np.asarray(cnp.count_nonzero(v).get(iface)).reshape(-1)[0])
    chk.true("count_nonzero(v)", c == n, f"got {c}")
    t = np.asarray(v[n - 16:].get(iface))
    chk.true("v[n - 16:]", t.shape == (16,) and bool((t == 1).all()))


if __name__ == "__main__":
    U.run("test_grid_past_end", main)
