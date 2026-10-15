#!/usr/bin/env python
"""Uploads above the per-message inline limit (lowered here to 64 KiB) are split into
UPLOAD + set_region pieces, one message each; oversized messages fail with a clear error.

Run as e.g. harness/run_one.sh server.out 4 33001 examples/ops/test_upload_split.py
"""

import numpy as np

import charmnumeric as cnp
import charmnumeric.charmnumeric as cm
import charmtyles.ast as ct_ast
import charmtyles.interface as ct_interface
from charmtyles.core import execute
from _ops_common import SIM, Checker, connect

LIMIT = 64 << 10


def record_messages(iface):
    sizes = []
    send = iface.run_async

    def run_async(cmd, bulk=()):
        sizes.append(sum(len(b) for b in bulk))
        return send(cmd, bulk)

    iface.run_async = run_async
    return sizes


def split_uploads(chk, iface, sizes, rng):
    cases = [
        ((100003,), np.float64),
        ((300, 1000), np.float64),
        ((3, 20001), np.float64),  # rows above the limit: split along axis 1 too
        ((4, 50, 1001), np.float32),
        ((2, 3, 40, 500), np.float64),  # folded 4-D
        ((5000, 7), np.complex128),
        ((70001,), np.int8),
        ((300, 401), np.bool_),
    ]
    for shape, dtype in cases:
        h = rng.standard_normal(shape) * 50
        if np.dtype(dtype).kind == "c":
            h = h + 1j * rng.standard_normal(shape)
        h = (h > 0) if dtype is np.bool_ else h.astype(dtype)
        del sizes[:]
        X = cnp.array(h)
        execute(iface)
        if not SIM:
            chk.check(f"{shape} {np.dtype(dtype).name} pieces <= limit",
                      np.array(max(sizes) <= LIMIT and sum(sizes) == h.nbytes), np.array(True))
        chk.check(f"upload {shape} {np.dtype(dtype).name}", X, h)
    h = rng.standard_normal((400, 300))
    X = cnp.array(h)
    chk.check("split upload then op", X * 2.0 + cnp.array(h[::-1].copy()), 2.0 * h + h[::-1],
              rtol=1e-15)


def many_uploads(chk, iface, sizes, rng):
    hs = [rng.standard_normal(3000) for _ in range(8)]  # 24 KB each: several per message
    del sizes[:]
    xs = [cnp.array(h) for h in hs]
    total = xs[0]
    for x in xs[1:]:
        total = total + x
    chk.check("many uploads summed", total, sum(hs), rtol=1e-12)
    if not SIM:
        chk.check("many uploads batched", np.array(max(sizes) <= LIMIT and len(sizes) < 8),
                  np.array(True))


def assign_from_host(chk, iface, rng):
    h = rng.standard_normal((500, 70))
    T = cnp.zeros((520, 90))
    T[10:510, 5:75] = h
    e = np.zeros((520, 90))
    e[10:510, 5:75] = h
    chk.check("T[...] = big ndarray", T, e)


def limits(chk, iface):
    pieces = list(cm._upload_pieces((3 << 27,), 8, 7 << 26))  # 3 GiB float64 in 448 MiB pieces
    chk.check("3 GiB pieces", np.array([k[0].stop - k[0].start for k in pieces]),
              np.array([7 << 23] * 6 + [(3 << 27) - 6 * (7 << 23)]))
    pieces = list(cm._upload_pieces((2, 3 << 27), 4, 7 << 26))
    chk.check("rows above the limit", np.array(len(pieces)), np.array(8))
    old = ct_ast.CCS_MAX_MESSAGE_BYTES
    try:
        ct_ast.CCS_MAX_MESSAGE_BYTES = 1000
        chk.raises("oversized InlineData", ValueError, lambda: ct_ast.InlineData(b"x" * 1001))
    finally:
        ct_ast.CCS_MAX_MESSAGE_BYTES = old
    if not SIM:
        old = ct_interface.CCS_MAX_MESSAGE_BYTES
        try:
            ct_interface.CCS_MAX_MESSAGE_BYTES = 1000
            chk.raises("oversized DAG message", ValueError,
                       lambda: iface._dag_message(b"", [b"x" * 1000]))
        finally:
            ct_interface.CCS_MAX_MESSAGE_BYTES = old


def main():
    iface = connect()
    chk = Checker("test_upload_split", iface)
    rng = np.random.default_rng(11)
    ct_ast.set_max_inline_bytes(LIMIT)
    sizes = record_messages(iface)
    split_uploads(chk, iface, sizes, rng)
    many_uploads(chk, iface, sizes, rng)
    assign_from_host(chk, iface, rng)
    limits(chk, iface)
    chk.finish()


if __name__ == "__main__":
    main()
