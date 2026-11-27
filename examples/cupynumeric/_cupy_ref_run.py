"""cuPyNumeric side of CNUM_REFERENCE=cupynumeric; started by _cupy_reference.py via `legate`.

usage: legate [flags] _cupy_ref_run.py <workdir> <skip> <upstream program.py> [program args]
Runs the unmodified upstream program with upstream's _benchmark. Its random draws
and linalg factors are replaced by the charmnumeric run's (in.npz/in.json), and its
program-function locals and reported time are written to out.npz/out.json. The
first <skip> run_timed calls are skipped; the run exits after the next one.
"""

import functools
import json
import os
import sys
import time

import numpy

work, skip, prog = sys.argv[1], int(sys.argv[2]), os.path.abspath(sys.argv[3])
sys.argv = [prog] + sys.argv[4:]
sys.path.insert(0, os.path.dirname(prog))

import cupynumeric as cn  # noqa: E402
import _benchmark  # noqa: E402  (upstream's)
import _benchmark.harness as H  # noqa: E402

_TOOL = 4
_FACTORS = {"linalg": ("qr", "svd", "tssvd", "eigh"), "": ("argsort",)}
IN = json.load(open(os.path.join(work, "in.json")))
DATA = numpy.load(os.path.join(work, "in.npz"))
OUT_ARR = {}


class Cur:
    """Replay state of the run being executed."""

    def next(self, kind, name):
        items = self.rec[kind]
        if self.cursor[kind] >= len(items):
            raise RuntimeError(f"cuPyNumeric run made more {kind} than the charmnumeric run ({name})")
        item = items[self.cursor[kind]]
        self.cursor[kind] += 1
        if item[0] != name:
            raise RuntimeError(f"cuPyNumeric run called {name}, charmnumeric run {item[0]}")
        return item

    def start(self, rec):
        self.rec, self.cursor = rec, {"draws": 0, "factors": 0}
        self.failures, self.factors = [], []


CUR = Cur()


def replay_draw(name, want):
    _, kind, key = CUR.next("draws", name)
    got = DATA[key]
    wshape = getattr(want, "shape", numpy.shape(want))
    wdtype = getattr(want, "dtype", numpy.asarray(want).dtype)
    if tuple(got.shape) != tuple(wshape) or got.dtype != wdtype:
        CUR.failures.append(f"np.{name}: charmnumeric drew {got.dtype}{got.shape}, "
                            f"cuPyNumeric {wdtype}{tuple(wshape)}")
    if kind != "array":
        return type(want)(got.item())
    return cn.array(got)


class Gen:
    def __init__(self, gen):
        self._gen = gen

    def __getattr__(self, name):
        fn = getattr(self._gen, name)
        return lambda *a, **k: replay_draw("rng." + name, fn(*a, **k))


def as_np(x):
    return numpy.asarray(x)


def factor(name, fn):
    def call(*args, **kwargs):
        out = fn(*args, **kwargs)
        keys = []
        for o in (list(out) if isinstance(out, (tuple, list)) else [out]):
            keys.append(f"c{len(CUR.factors)}_{len(keys)}")
            OUT_ARR[keys[-1]] = as_np(o)
        akey = f"a{len(CUR.factors)}"
        OUT_ARR[akey] = as_np(args[0])
        CUR.factors.append(dict(name=name, a=akey, result=keys,
                                args=[x if isinstance(x, (int, str, bool, type(None))) else str(x)
                                      for x in args[1:]],
                                kwargs={k: v if isinstance(v, (int, str, bool, type(None))) else str(v)
                                        for k, v in kwargs.items()}))
        _, rkeys = CUR.next("factors", name)
        res = [cn.array(DATA[k]) for k in rkeys]
        return res[0] if name == "argsort" else tuple(res)

    return call


class Mod:
    """Attribute proxy of cupynumeric (and random/linalg) that replays draws and factors."""

    def __init__(self, module, path=""):
        self._module, self._path = module, path
        self.__name__ = module.__name__

    def __getattr__(self, name):
        attr = getattr(self._module, name)
        if self._path == "" and name in ("random", "linalg"):
            return Mod(attr, name)
        if self._path == "random" and callable(attr) and not isinstance(attr, type):
            if name == "default_rng":
                return lambda *a, **k: Gen(attr(*a, **k))
            if name == "seed":
                return attr
            return lambda *a, **k: replay_draw(name, attr(*a, **k))
        if name in _FACTORS.get(self._path, ()):
            return factor(name, attr)
        return attr


# ---- locals of every program function at return ----


def program_codes():
    main = sys.modules["__main__"]
    todo = [o.__code__ for o in vars(main).values()
            if getattr(o, "__code__", None) is not None and o.__code__.co_filename == prog]
    codes = []
    while todo:
        code = todo.pop()
        if code not in codes:
            codes.append(code)
            todo.extend(c for c in code.co_consts if hasattr(c, "co_code"))
    return codes


def conv(x):
    if isinstance(x, (cn.ndarray, numpy.ndarray, numpy.generic, int)):
        return as_np(x)
    if isinstance(x, (tuple, list)):
        items = [conv(i) for i in x]
        return None if any(i is None for i in items) else items
    return None


class Recording:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        mon = sys.monitoring

        def on_return(code, offset, retval):
            rec = dict(sys._getframe(1).f_locals)
            rec["<return>"] = retval
            self.store[code.co_qualname] = rec

        self.codes = program_codes()
        mon.use_tool_id(_TOOL, "cnum-ref")
        mon.register_callback(_TOOL, mon.events.PY_RETURN, on_return)
        for c in self.codes:
            mon.set_local_events(_TOOL, c, mon.events.PY_RETURN)

    def __exit__(self, *exc):
        mon = sys.monitoring
        for c in self.codes:
            mon.set_local_events(_TOOL, c, 0)
        mon.register_callback(_TOOL, mon.events.PY_RETURN, None)
        mon.free_tool_id(_TOOL)


def ext_time_ms(f, a, k):
    from legate.timing import time as ltime

    t0 = int(ltime("us"))
    r = f(*a, **k)
    return r, (int(ltime("us")) - t0) / 1000.0


RESULTS = []
STATE = {"run": 0}
orig_run = H.BenchmarkHarness._run


def patched_run(self, info, f, kwargs, plan, mode):
    n = STATE["run"]
    STATE["run"] += 1
    if n < skip:
        return
    calls = {"i": 0}

    @functools.wraps(f)
    def wrapped(*a, **k):
        rec = IN[calls["i"]]
        calls["i"] += 1
        CUR.start(rec)
        locs = {}
        with Recording(locs):
            if info.returns_time < 0:
                ret, ms = ext_time_ms(f, a, k)
            else:
                ret, ms = f(*a, **k), None
        if info.returns_time >= 0:
            ms = float(ret) if isinstance(info.output_names, str) else float(ret[info.returns_time])
        if CUR.cursor["draws"] != len(rec["draws"]):
            CUR.failures.append("cuPyNumeric run made fewer random draws")
        wants = {}
        for qual, loc in locs.items():
            ent = {}
            for var, val in loc.items():
                v = conv(val)
                if v is None:
                    continue
                if isinstance(v, list):
                    ks = []
                    for x in v:
                        ks.append(f"w{len(OUT_ARR)}")
                        OUT_ARR[ks[-1]] = x
                    ent[var] = ["t", ks]
                else:
                    OUT_ARR[f"w{len(OUT_ARR)}"] = v
                    ent[var] = ["a", f"w{len(OUT_ARR) - 1}"]
            wants[qual] = ent
        RESULTS.append(dict(wants=wants, factors=CUR.factors, failures=CUR.failures,
                            time_ms=ms))
        return ret

    orig_run(self, info, wrapped, kwargs, plan, mode)
    numpy.savez(os.path.join(work, "out.npz"), **OUT_ARR)
    with open(os.path.join(work, "out.json"), "w") as fh:
        json.dump(dict(runs=RESULTS), fh)
    sys.exit(0)


H.BenchmarkHarness._run = patched_run
_orig_parse = H.parse_with_harness


def parse_with_harness(parser):
    args, harness = _orig_parse(parser)
    harness.np = Mod(cn)
    return args, harness


H.parse_with_harness = _benchmark.parse_with_harness = parse_with_harness

import runpy  # noqa: E402

runpy.run_path(prog, run_name="__main__")
