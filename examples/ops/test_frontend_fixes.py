"""Frontend regressions: kernel sharing across scalar constants, exact big ints in
writes, stale arange/random recipes, size-1 broadcast set_region, CCS failures."""

import os
import socket
import struct
import threading
import time
import types
import warnings

import numpy as np

import charmnumeric as cnp
from _ops_common import SIM, Checker, connect, floats

iface = connect()
C = Checker("test_frontend_fixes", iface)


def ok(label, cond, detail=""):
    C.count += 1
    if cond:
        print(f"  PASS {label}")
    else:
        print(f"  FAIL {label}{': ' + detail if detail else ''}")
        C.failures.append(label)


# ---- scalar constants are kernel arguments, not part of the kernel identity ----

if not SIM:
    from charmtyles.ast import get_execution_state

    def prefix_id():
        return get_execution_state().open_prefix.identifier

    x = floats((1000,))
    hx = C.fetch(x)
    ids = set()
    for i in range(20):
        y = x * i
        ids.add(prefix_id())
        C.check(f"x * {i}", y, hx * i)
    for v in (0.5, -3.25, 1e300):
        y = x * v
        ids.add(prefix_id())
        C.check(f"x * {v}", y, hx * v)
    ok("x * i shares one kernel identifier", len(ids) == 1, f"{len(ids)} identifiers")

    z = cnp.zeros((300,), dtype=np.int64)
    C.fetch(z)
    ids = set()
    for i in range(5):
        z[...] = i * 7
        ids.add(prefix_id())
        C.check(f"fill {i * 7}", z, np.full(300, i * 7))
    ok("scalar fill shares one kernel identifier", len(ids) == 1, f"{len(ids)} identifiers")

# ---- ints beyond 2**53 are written exactly ----

BIG = 2**60 + 1
for dt, v in ((np.int64, BIG), (np.int64, -BIG), (np.uint64, 2**64 - 3)):
    name = np.dtype(dt).name
    C.check(f"full {name} {v}", cnp.full((5,), v, dtype=dt), np.full(5, v, dtype=dt))
    a = cnp.zeros((3, 4), dtype=dt)
    a[...] = v
    a[1, 2] = v - 2
    a[2].fill(v - 4)
    r = np.full((3, 4), v, dtype=dt)
    r[1, 2] = v - 2
    r[2] = v - 4
    C.check(f"setitem/fill {name} {v}", a, r)
    a = cnp.zeros((6,), dtype=dt)
    a[[0, 2]] = v
    cnp.put(a, [4], v - 2)
    r = np.zeros(6, dtype=dt)
    r[[0, 2]] = v
    r[4] = v - 2
    C.check(f"fancy setitem/put {name} {v}", a, r)
    a = cnp.zeros((2, 3), dtype=dt)
    cnp.put_along_axis(a, np.array([[0], [2]]), v, axis=1)
    r = np.zeros((2, 3), dtype=dt)
    np.put_along_axis(r, np.array([[0], [2]]), v, axis=1)
    C.check(f"put_along_axis {name} {v}", a, r)
a = cnp.zeros((2, 2, 2, 2), dtype=np.int64)
a[[0, 1], 0, 0, 1] = BIG
r = np.zeros((2, 2, 2, 2), dtype=np.int64)
r[[0, 1], 0, 0, 1] = BIG
C.check("folded fancy setitem big int", a, r)

# ---- arange/linspace/random recipes are dropped when the array is written ----

C.check("meshgrid of arange", cnp.meshgrid(cnp.arange(4), cnp.arange(3))[0],
        np.meshgrid(np.arange(4), np.arange(3))[0])
writes = {
    "setitem": lambda t: t.__setitem__(0, 99),
    "fill": lambda t: t.fill(7),
    "inplace": lambda t: t.__iadd__(1),
    "view": lambda t: t[1:3].__setitem__(0, 5),
    "fancy": lambda t: t.__setitem__([1], 6),
    "out": lambda t: cnp.add(t, 1, out=t),
}
for label, write in writes.items():
    for make in (lambda: cnp.arange(5), lambda: cnp.linspace(0.0, 1.0, 5)):
        t = make()
        write(t)
        C.raises(f"meshgrid after {label}", NotImplementedError, lambda: cnp.meshgrid(t))
rnd = cnp.random.random((3, 4))
C.check("transposed random", cnp.random.transposed(rnd), C.fetch(rnd).T)
for label, write in writes.items():
    if label in ("view", "fancy"):
        write = {"view": lambda t: t[1:3, 1:2].__setitem__(0, 5),
                 "fancy": lambda t: t.__setitem__(([1], [1]), 6)}[label]
    t = cnp.random.random((3, 4))
    write(t)
    C.raises(f"transposed after {label}", ValueError, lambda: cnp.random.transposed(t))

# ---- a whole size-1 rhs of another rank is a broadcast, not a cross-rank copy ----

if not SIM:
    from charmtyles.ast import ASTNode, _cross_rank_set_region
    from charmtyles.operation import Operation
    from charmnumeric.charmnumeric import _full_region

    target = cnp.zeros((4, 5))
    region = _full_region((4, 5))

    def set_region_of(rhs):
        return types.SimpleNamespace(
            _active_ast=ASTNode(Operation.set_region, [target, region, rhs], -1))

    one = cnp.sum(cnp.arange(5.0))
    pair = cnp.zeros((2,))
    ok("size-1 rhs of another rank is not cross-rank", not _cross_rank_set_region(set_region_of(one)))
    ok("size-1 view of a larger array is cross-rank", _cross_rank_set_region(set_region_of(pair[0:1])))
    ok("rank-1 rhs is cross-rank", _cross_rank_set_region(set_region_of(cnp.zeros((5,)))))

# ---- CCS: a malformed timeout falls back; a dead server fails fast ----

if not SIM:
    import charmtyles.interface as ccs

    def timeout_from(raw):
        old = os.environ.get("CHARMTYLES_CCS_TIMEOUT")
        os.environ["CHARMTYLES_CCS_TIMEOUT"] = raw
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = ccs.CCSInterface().reply_timeout
            return value, len(caught)
        finally:
            if old is None:
                del os.environ["CHARMTYLES_CCS_TIMEOUT"]
            else:
                os.environ["CHARMTYLES_CCS_TIMEOUT"] = old

    ok("CHARMTYLES_CCS_TIMEOUT=abc falls back", timeout_from("abc") == (3600, 1))
    ok("CHARMTYLES_CCS_TIMEOUT=-5 falls back", timeout_from("-5") == (3600, 1))
    ok("CHARMTYLES_CCS_TIMEOUT=1.5", timeout_from("1.5") == (2, 0))

    def fake_server(listener, replies):
        """Answer one request per connection; None closes without a reply."""
        for reply in replies:
            conn, _ = listener.accept()
            with conn:
                head = conn.recv(40, socket.MSG_WAITALL)
                size = struct.unpack(">i", head[:4])[0]
                if size:
                    conn.recv(size, socket.MSG_WAITALL)
                if reply is not None:
                    conn.sendall(struct.pack(">i", len(reply)) + reply)
                    time.sleep(0.2)
        listener.close()

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]
    getinfo = struct.pack(">ii", 1, 1)
    server = threading.Thread(target=fake_server, args=(listener, [getinfo, b"\0", None]),
                              daemon=True)
    server.start()
    dead = ccs.CCSInterface()
    dead._open("127.0.0.1", port)  # not connect(): test harnesses redirect it to the real server
    dead.send_command_raw(ccs.Handlers.connection_handler, ccs.to_bytes(1, "i"), 1)

    def timed_error(fn):
        start = time.monotonic()
        try:
            fn()
        except ConnectionError as e:
            return time.monotonic() - start, str(e)
        return None, "no ConnectionError"

    elapsed, msg = timed_error(lambda: dead.get(1, 0, 4, dtype=np.float64))
    ok("server closing the connection fails the get", elapsed is not None and elapsed < 30, msg)
    server.join(10)
    ccs._CONNECT_TIMEOUT = 2
    elapsed, msg = timed_error(lambda: dead.get(1, 0, 4, dtype=np.float64))
    ok("unreachable server fails the get", elapsed is not None and elapsed < 30, msg)

C.finish()
