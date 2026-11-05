#!/usr/bin/env python
"""get() of arrays with a short last dimension (many rows per chare), e.g. take_pairs.

Each chare sends its tile to PE 0 in a few large messages, whatever the row length (it used
to send one message per row: 248M messages for take_pairs at 500000 x 32, which never
finished). The default cases are small. CNUM_BIGGET=1 adds take_pairs at 500000 x 32
(1.98 GB) and arrays just over 2 GiB with 8-byte rows, about 10 GB of client memory and
4-5 GB on the server's PE 0. Running the server with CNUM_MSG_MAX_BYTES=<small> also tests
tiles split over several messages.

Run as e.g. harness/run_one.sh server.out 4 33001 examples/ops/test_bigget.py
"""

import itertools
import os
import time

import numpy as np

import charmnumeric as cnp
from _ops_common import Checker, connect

HEAVY = os.environ.get("CNUM_BIGGET", "") not in ("", "0")


def take_pairs(chk, m, n, timed=False):
    idx = np.array(list(itertools.combinations(range(n), 2)))
    data = cnp.arange(m * n, dtype=np.int32).reshape((m, n))
    x = data.take(cnp.array(idx), axis=1)
    chk.fetch(data)
    t = time.perf_counter()
    got = x.get(chk.iface)
    dt = time.perf_counter() - t
    want = np.arange(m * n, dtype=np.int32).reshape((m, n)).take(idx, axis=1)
    label = f"take_pairs get {got.shape} ({got.nbytes / 2**20:.0f} MiB)"
    if timed:
        print(f"  {label}: {dt:.2f} s")
    chk.check(label, got, want)


def reshaped(chk, shape, dtype):
    n = int(np.prod(shape))
    x = cnp.arange(n, dtype=dtype).reshape(shape)
    chk.check(f"get {shape} {np.dtype(dtype).name}", x, np.arange(n, dtype=dtype).reshape(shape))


def small(chk):
    for shape, dt in [((20001, 37, 2), np.int32), ((3001, 2, 1), np.float64), ((7, 5, 3), np.int8),
                      ((200001, 3), np.int16), ((1001, 1), np.float32), ((400, 300, 1), np.int64)]:
        reshaped(chk, shape, dt)
    x = cnp.arange(6000 * 40, dtype=np.float32).reshape((6000, 40))
    h = np.arange(6000 * 40, dtype=np.float32).reshape((6000, 40))
    chk.check("get of a column slice", x[:, 3:5], h[:, 3:5])
    chk.check("get of a strided view", x[1::3, ::7], h[1::3, ::7])
    take_pairs(chk, 20000, 32, timed=True)


def heavy(chk):
    take_pairs(chk, 500000, 32, timed=True)
    n = (1 << 29) + 13  # 2 GiB + 52 bytes of int32
    h = chk.fetch(cnp.arange(n, dtype=np.int32))
    ok = h.shape == (n,) and h[-1] == n - 1 and np.array_equal(h, np.arange(n, dtype=np.int32))
    del h
    chk.check("get 1-D int32 over 2 GiB", np.array(ok), np.array(True))
    take_pairs(chk, 541201, 32, timed=True)  # 2 GiB + 1920 bytes


def main():
    iface = connect()
    chk = Checker("test_bigget", iface)
    small(chk)
    if HEAVY:
        heavy(chk)
    chk.check("server alive", cnp.sum(cnp.arange(1000.0)), np.sum(np.arange(1000.0)))
    chk.finish()


if __name__ == "__main__":
    main()
