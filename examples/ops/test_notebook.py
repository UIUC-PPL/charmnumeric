#!/usr/bin/env python
"""Interactive-use behavior: bounded display fetches, repr without server traffic,
host conversions, Ctrl-C during a long get, reconnecting and fork.

Run as e.g. harness/run_one.sh server.out 2 33001 examples/ops/test_notebook.py
"""

import os
import signal
import subprocess
import sys
import time

import numpy as np

import charmnumeric as cnp
from charmnumeric import _display
from charmtyles.interface import Handlers
from _ops_common import Checker, connect

_fetched = []


def count_fetches(iface):
    """Record the element count of every get on iface."""
    get = iface.get

    def counting(ndims, name, size, dtype=None):
        _fetched.append(size)
        return get(ndims, name, size, dtype=dtype)

    iface.get = counting


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
        want = _display._repr_text(type(a).__name__, ref, ref.size > 1000, ref.shape, ref.dtype)
        if type(a).__name__ == "Array" and ref.dtype in (np.float32, np.int32):
            want = np.array_repr(ref).replace("array(", "Array(", 1)  # NumPy shows these dtypes
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
    a = cnp.ones((1500, 1500)) * 1e-3
    for _ in range(10):
        a = (a @ a) * 1e-3
    return cnp.sum(a)


def interrupt(chk, iface):
    total = slow_product()
    signal.signal(signal.SIGALRM, lambda *_: os.kill(os.getpid(), signal.SIGINT))
    start = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, 0.3)
    waited = None
    try:
        total.get(iface)
    except KeyboardInterrupt:
        waited = time.monotonic() - start
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    chk.check("usable after Ctrl-C", cnp.arange(5) + 1, np.arange(5) + 1)
    ready = time.monotonic() - start
    chk.check("interrupted result", total, np.asarray(0.0))
    chk.count += 1
    if waited is None:
        print("  SKIP Ctrl-C stops the wait: the product finished before the interrupt")
    elif waited < ready - 0.5:
        print(f"  PASS Ctrl-C stops the wait ({waited:.1f} s; result ready at {ready:.1f} s)")
    else:
        chk.failures.append("Ctrl-C")
        print(f"  FAIL Ctrl-C returned at {waited:.1f} s; result ready at {ready:.1f} s")


def reconnect(chk, iface):
    keep = cnp.arange(10.0) * 3
    chk.check("before reconnect", keep, np.arange(10.0) * 3)
    again = connect()  # e.g. a re-run notebook cell
    lost = cnp.CharmNumericInterface()  # a connect whose reply never arrives
    lost._open(os.environ.get("CNUM_HOST", "127.0.0.1"), int(os.environ.get("CNUM_PORT", "1234")))
    lost._send(Handlers.connection_handler, lost._connect_request(4))
    again = connect()
    chk.check("old array after reconnect", keep + 1, np.arange(10.0) * 3 + 1)
    chk.check("new array after reconnect", cnp.ones(4) * 2, np.full(4, 2.0))
    code = ("from _ops_common import connect\n"
            "try:\n    connect()\nexcept ConnectionError as e:\n    print(e)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__)), timeout=120)
    chk.count += 1
    if "another client process" in out.stdout:
        print("  PASS another process is refused")
    else:
        chk.failures.append("another process")
        print(f"  FAIL another process: {out.stdout} {out.stderr}")
    chk.check("server still serves us", cnp.arange(3.0), np.arange(3.0))
    pid = os.fork()
    if pid == 0:
        status = 0
        try:
            cnp.arange(3.0).get(again)
            status |= 1
        except ConnectionError as e:
            status |= 0 if "stale" in str(e) else 2
        try:
            connect()
            status |= 4
        except ConnectionError as e:
            status |= 0 if "another client process" in str(e) else 8
        os._exit(status)
    _, status = os.waitpid(pid, 0)
    chk.check("fork: inherited connection is stale, new connect refused",
              np.asarray(os.waitstatus_to_exitcode(status)), np.asarray(0))
    chk.check("parent still served after fork", cnp.arange(4.0) * 2, np.arange(4.0) * 2)


def main():
    iface = connect()
    count_fetches(iface)
    chk = Checker("test_notebook", iface)
    display(chk)
    interrupt(chk, iface)
    reconnect(chk, iface)
    chk.finish()


if __name__ == "__main__":
    main()
