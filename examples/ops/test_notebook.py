#!/usr/bin/env python
"""Interactive-use behavior: bounded display fetches, repr without server traffic,
host conversions, Ctrl-C during a long get, and reconnecting.

Run as e.g. harness/run_one.sh server.out 2 33001 examples/ops/test_notebook.py
"""

import os
import signal
import subprocess
import sys
import time

import numpy as np

import charmnumeric as cnp
from charmtyles.interface import CCSInterface
from _ops_common import Checker, connect

_fetched = []
_get = CCSInterface.get


def _counting_get(self, ndims, name, size, dtype=None):
    _fetched.append(size)
    return _get(self, ndims, name, size, dtype=dtype)


CCSInterface.get = _counting_get


def fetched(fn):
    """(result of fn(), elements fetched while running it)."""
    _fetched.clear()
    result = fn()
    return result, sum(_fetched)


def display(chk):
    host = np.arange(1_000_000, dtype=np.float64) * 0.5
    x = cnp.arange(1_000_000, dtype=cnp.float64) * 0.5
    chk.check("value", x, host)
    text, n = fetched(lambda: repr(x))
    if n or "shape=(1000000,)" not in text:
        chk.failures.append(f"repr fetched {n} elements: {text}")
    cases = [
        ("1-D large", x, host),
        ("1-D small", x[:7], host[:7]),
        ("2-D", cnp.arange(60000, dtype=cnp.float32).reshape((200, 300)),
         np.arange(60000, dtype=np.float32).reshape(200, 300)),
        ("3-D int", cnp.arange(50 * 60 * 70, dtype=cnp.int32).reshape((50, 60, 70)),
         np.arange(50 * 60 * 70, dtype=np.int32).reshape(50, 60, 70)),
        ("view", x[10:500000:3], host[10:500000:3]),
        ("transpose", cnp.arange(60000.0).reshape((200, 300)).T,
         np.arange(60000.0).reshape(200, 300).T),
        ("4-D", cnp.arange(3 * 4 * 5 * 60.0).reshape((3, 4, 5, 60)),
         np.arange(3 * 4 * 5 * 60.0).reshape(3, 4, 5, 60)),
        ("0-d", cnp.sum(x[:10]), np.sum(host[:10])),
        ("bool", x > 1000.0, host > 1000.0),
        ("one long axis", cnp.ones((2, 5000)), np.ones((2, 5000))),
    ]
    for label, a, ref in cases:
        ref = np.asarray(ref)
        chk.count += 1
        text, n = fetched(lambda: str(a))
        limit = ref.size if ref.size <= 1000 else 6 ** ref.ndim * 2
        if text != str(ref):
            chk.failures.append(f"str {label}")
            print(f"  FAIL str {label}:\n{text}\n!=\n{ref}")
        elif n > limit:
            chk.failures.append(f"str {label} fetched {n}")
            print(f"  FAIL str {label}: fetched {n} > {limit} elements")
        else:
            print(f"  PASS str {label} ({n} elements fetched)")
        bundle, n2 = fetched(lambda: a._repr_mimebundle_())
        chk.count += 1
        plain, page = bundle["text/plain"], bundle["text/html"]
        kind = type(a).__name__
        tail = f", shape={ref.shape}" if ref.size > 1000 else ""
        want = (f"{kind}({np.array2string(ref, separator=', ', prefix=kind + '(')}{tail}, "
                f"dtype={ref.dtype.name})")
        if n2 > limit or plain != want or "<pre" not in page:
            chk.failures.append(f"display {label}")
            print(f"  FAIL display {label} ({n2} fetched):\n{plain}\n!=\n{want}")
        else:
            print(f"  PASS display {label}")
    previous = cnp.set_display(preview=False)
    y = x + 1
    bundle, n = fetched(lambda: y._repr_mimebundle_())
    cnp.set_display(**previous)
    chk.count += 1
    if n or "pending" not in bundle["text/plain"]:
        chk.failures.append("preview=False")
        print(f"  FAIL preview=False ({n} fetched): {bundle['text/plain']}")
    else:
        print("  PASS preview=False shows metadata only")
    chk.check("format spec", np.asarray(f"{cnp.sum(x[:4]):.2f}"), np.asarray("3.00"))
    got, n = fetched(lambda: x.to_numpy())
    chk.check("to_numpy is one fetch", np.asarray(n), np.asarray(x.size))
    chk.check("to_numpy", got, host)
    chk.check("block_until_ready", np.asarray(y.block_until_ready() is y), np.asarray(True))


def slow_product():
    a = cnp.ones((2000, 2000)) * 1e-3
    for _ in range(12):
        a = (a @ a) * 1e-3
    return cnp.sum(a)


def interrupt(chk, iface):
    total = slow_product()
    signal.signal(signal.SIGALRM, lambda *_: os.kill(os.getpid(), signal.SIGINT))
    start = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        total.get(iface)
        print("  SKIP interrupt: the product finished before the interrupt")
    except KeyboardInterrupt:
        waited = time.monotonic() - start
        chk.check("Ctrl-C stops the wait", np.asarray(waited < 4.0), np.asarray(True))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    chk.check("usable after Ctrl-C", cnp.arange(5) + 1, np.arange(5) + 1)
    chk.check("interrupted result", total, np.asarray(0.0))


def reconnect(chk, iface):
    keep = cnp.arange(10.0) * 3
    chk.check("before reconnect", keep, np.arange(10.0) * 3)
    again = connect()  # e.g. a re-run notebook cell
    chk.check("old array after reconnect", keep + 1, np.arange(10.0) * 3 + 1)
    chk.check("new array after reconnect", cnp.ones(4) * 2, np.full(4, 2.0))
    code = ("from _ops_common import connect\n"
            "try:\n    connect()\nexcept ConnectionError as e:\n    print(e)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__)), timeout=120)
    chk.count += 1
    if "cannot be reused" in out.stdout:
        print("  PASS another process is refused")
    else:
        chk.failures.append("another process")
        print(f"  FAIL another process: {out.stdout} {out.stderr}")
    chk.check("server still serves us", cnp.arange(3.0), np.arange(3.0))
    return again


def main():
    iface = connect()
    chk = Checker("test_notebook", iface)
    display(chk)
    interrupt(chk, iface)
    reconnect(chk, iface)
    chk.finish()


if __name__ == "__main__":
    main()
