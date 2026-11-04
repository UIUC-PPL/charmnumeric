#!/usr/bin/env python
"""Fetches over the CCS reply limit (pulled in chunks) and arrays over 2 GiB.

Chunked fetches are first exercised with a tiny chunk (CHARMTYLES_FETCH_CHUNK), then with
real arrays just over 2 GiB (about 7 GiB of client and server memory). CNUM_BIGIO_HEAVY=1
adds a 4 GiB fetch and a cholesky whose blocks are 2 GiB messages (minutes on a CPU build).
Running the server with CNUM_MSG_MAX_BYTES=<small> makes every executor fragment its
messages (receive_frag), for testing that path with any example.

Run as e.g. harness/run_one.sh server.out 2 33001 examples/ops/test_bigio.py
"""

import os

import struct

import numpy as np

import charmnumeric as cnp
from charmtyles.interface import Handlers, to_bytes
from charmnumeric.interface import _BARRIER_NAME
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


def _long_get(iface, ndims, name, size, wire_dtype, chunk):
    """Send a chunked-format get; returns (epoch, total bytes)."""
    epoch = iface.epoch
    cmd = (to_bytes(ndims, 'i') + to_bytes(epoch, 'i') + to_bytes(name, 'i') + to_bytes(-1, 'i')
           + to_bytes(wire_dtype, 'i') + to_bytes(size, 'q') + to_bytes(chunk, 'q'))
    reply = iface.send_command_raw_var(Handlers.get_nd_handler, cmd)
    return epoch, (struct.unpack('q', reply)[0] if len(reply) == 8 else -len(reply))


def _pull(iface, epoch, offset, n):
    out = np.zeros(n, dtype=np.uint8)
    iface._send(Handlers.get_chunk_handler, to_bytes(epoch, 'i') + to_bytes(offset, 'q'),
                advance=False)
    return out[:iface._receive_into(out)]


def fetch_protocol(chk, iface):
    """Abandoned pulls, chunk retries and non-data replies to a chunked-format request."""
    h = np.arange(10007, dtype=np.float64) * 3.0
    x = cnp.array(h)
    chk.check("protocol input", x, h)
    raw = h.view(np.uint8)
    epoch, total = _long_get(iface, 1, x.name, h.size, 1, 4096)
    chk.check("chunked get announces its size", np.array(total), np.array(h.nbytes))
    first = _pull(iface, epoch, 0, 4096)
    chk.check("first chunk", first, raw[:4096])
    # abandoned: no further pulls, no release; the next gets must still work
    chk.check("get after an abandoned pull", cnp.sum(x), np.sum(h))
    os.environ["CHARMTYLES_FETCH_CHUNK"] = "4096"
    try:
        chk.check("chunked get after an abandoned pull", x, h)
    finally:
        del os.environ["CHARMTYLES_FETCH_CHUNK"]
    epoch, total = _long_get(iface, 1, x.name, h.size, 1, 4096)
    last = (total - 1) // 4096 * 4096
    a = _pull(iface, epoch, last, 4096)
    b = _pull(iface, epoch, last, 4096)
    chk.check("final chunk", a, raw[last:])
    chk.check("final chunk retried", b, raw[last:])
    chk.check("middle chunk after the final one", _pull(iface, epoch, 4096, 4096),
              raw[4096:8192])
    iface._send(Handlers.get_release_handler, to_bytes(epoch, 'i'), advance=False)
    iface._receive(1)
    chk.check("released fetch replies empty", np.array(len(_pull(iface, epoch, 0, 4096))),
              np.array(0))
    # a barrier (non-data reply) in the chunked format, then chunked gets on later epochs
    cnp.sum(x).get(iface)
    epoch, total = _long_get(iface, 1, _BARRIER_NAME, 0, 4, 4096)
    chk.check("barrier in chunked format replies one byte", np.array(total), np.array(-1))
    os.environ["CHARMTYLES_FETCH_CHUNK"] = "4096"
    try:
        y = x * 2.0
        chk.check("chunked get after a non-data reply", y, h * 2.0)
        chk.check("second chunked get", x, h)
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
    chk.check("server alive after big fetches", cnp.sum(cnp.arange(1000.0)),
              np.sum(np.arange(1000.0)))


def heavy(chk, iface):
    n = (1 << 29) + 7  # 4 GiB: 1-D tiles of 2 GiB (split gather messages)
    x = cnp.arange(n, dtype=np.float64)
    h = chk.fetch(x)
    ok = h.shape == (n,) and h[-1] == n - 1 and np.array_equal(h[::4097], np.arange(0, n, 4097))
    del h
    chk.check("fetch 1-D float64 of 4 GiB", np.array(ok), np.array(True))
    n = 16385  # 2 x 2 tiles of 16384: a 2 GiB linalg block message
    L = cnp.linalg.cholesky(cnp.eye(n) * 4.0)
    chk.check("cholesky with 2 GiB blocks", cnp.sum(L), np.float64(2 * n))


def main():
    iface = connect()
    chk = Checker("test_bigio", iface)
    if not SIM:
        small_chunks(chk, iface)
        fetch_protocol(chk, iface)
        over_2gib(chk, iface)
        if os.environ.get("CNUM_BIGIO_HEAVY", "") not in ("", "0"):
            heavy(chk, iface)
    chk.finish()


if __name__ == "__main__":
    main()
