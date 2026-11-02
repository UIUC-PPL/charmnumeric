#!/usr/bin/env python
"""Fetches over the CCS reply limit (pulled in chunks) and arrays over 2 GiB.

Chunked fetches are first exercised with a tiny chunk (CHARMTYLES_FETCH_CHUNK), then with
real arrays just over 2 GiB (about 7 GiB of client and server memory). CNUM_BIGIO_HEAVY=1
adds a cholesky whose blocks are 2 GiB messages (minutes on a CPU build).
Running the server with CNUM_MSG_MAX_BYTES=<small> makes every executor fragment its
messages (receive_frag), for testing that path with any example.

Run as e.g. harness/run_one.sh server.out 2 33001 examples/ops/test_bigio.py
"""

import os

import numpy as np

import charmnumeric as cnp
from _ops_common import SIM, Checker, connect


def small_chunks(chk, iface):
    os.environ["CHARMTYLES_FETCH_CHUNK"] = "1000"
    try:
        for shape, dt in [((100003,), np.float64), ((301, 77), np.int32), ((5, 63, 41), np.int8),
                          ((997,), np.complex128), ((40, 30), np.bool_)]:
            h = np.arange(np.prod(shape)).reshape(shape)
            h = (h % 3 == 0) if dt is np.bool_ else (h * 7 - 11).astype(dt)
            x = cnp.array(h)
            chk.check(f"chunked fetch {shape} {np.dtype(dt).name}", x, h)
            chk.check(f"chunked fetch of a view {shape}", x[1:, ...], h[1:, ...])
            chk.check("scalar fetch between chunked ones", cnp.sum(x[:1]), np.sum(h[:1]))
        x = cnp.arange(300000.0).reshape(1000, 300)
        y = x * 2.0 + 1.0
        chk.check("chunked fetch after ops", y, np.arange(300000.0).reshape(1000, 300) * 2.0 + 1.0)
        chk.check("ops after chunked fetches", cnp.sum(y), np.sum(np.arange(300000.0) * 2.0 + 1.0))
    finally:
        del os.environ["CHARMTYLES_FETCH_CHUNK"]


def over_2gib(chk, iface):
    n = (1 << 28) + 3  # 2 GiB + 24 bytes of float64
    h = chk.fetch(cnp.arange(n, dtype=np.float64))
    ok = h.shape == (n,) and np.array_equal(h, np.arange(n, dtype=np.float64))
    del h
    chk.check("fetch 1-D float64 over 2 GiB", np.array(ok), np.array(True))
    shape = ((1 << 15) + 1, 1 << 13)
    h = chk.fetch(cnp.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape) - 5.0)
    ok = h.shape == shape and np.array_equal(h.reshape(-1), np.arange(h.size) - 5.0)
    del h
    chk.check("fetch 2-D float64 over 2 GiB", np.array(ok), np.array(True))
    n = (1 << 29) + 7  # 4 GiB: 1-D tiles of 2 GiB (split gather messages)
    x = cnp.arange(n, dtype=np.float64)
    h = chk.fetch(x)
    ok = h.shape == (n,) and h[-1] == n - 1 and np.array_equal(h[::4097], np.arange(0, n, 4097))
    del h
    chk.check("fetch 1-D float64 of 4 GiB", np.array(ok), np.array(True))
    chk.check("server alive after big fetches", cnp.sum(x[:1000]), np.sum(np.arange(1000)))


def heavy(chk, iface):
    n = 16385  # 2 x 2 tiles of 16384: a 2 GiB linalg block message
    L = cnp.linalg.cholesky(cnp.eye(n) * 4.0)
    chk.check("cholesky with 2 GiB blocks", cnp.sum(L), np.float64(2 * n))


def main():
    iface = connect()
    chk = Checker("test_bigio", iface)
    if not SIM:
        small_chunks(chk, iface)
        over_2gib(chk, iface)
        if os.environ.get("CNUM_BIGIO_HEAVY", "") not in ("", "0"):
            heavy(chk, iface)
    chk.finish()


if __name__ == "__main__":
    main()
