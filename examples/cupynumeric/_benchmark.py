"""Stand-in for cuPyNumeric's examples/_benchmark, running programs with np = charmnumeric.

Each run_timed* call runs the program with charmnumeric, then with NumPy, and compares:

- np.random draws of the charmnumeric run are replayed in the NumPy run;
- QR/SVD/TSSVD/eigh factors and argsort are checked by their defining properties
  (signs and tie order are not unique), then replayed;
- the last recorded local of each function (sys.monitoring) is compared against
  NumPy; floats use a tolerance relative to the largest magnitude in the record.

The last line is ``PASS <program>`` or ``FAIL <program>: <reasons>``.
With --package numpy the program runs on NumPy only.
CNUM_REFERENCE=cupynumeric swaps the NumPy rerun for the upstream program on
cuPyNumeric in a subprocess (see _cupy_reference.py); the default is unchanged.
"""

import contextlib
import io
import itertools
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, replace

import numpy

import charmnumeric as cnp
from charmnumeric.folded import FoldedArray

_DEVICE = (cnp.Array, cnp.ArrayView, FoldedArray)
_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]
_TOOL = 4  # sys.monitoring tool id
_REF = "cuPyNumeric" if os.environ.get("CNUM_REFERENCE") == "cupynumeric" else "NumPy"
_RTOL = {"e": 1e-2, "f": 2e-5, "d": 1e-6, "F": 2e-5, "D": 1e-6}  # by dtype char


# ---- benchmark info (as in _benchmark/info.py) ----


@dataclass
class BenchmarkInfo:
    name: str
    input_names: dict
    output_names: object
    formats: dict
    returns_time: int

    def replace(self, /, **changes):
        return replace(self, **changes)


def benchmark_info(*, name=None, input_names={}, output_names=None, formats={},
                   returns_time=True):
    rt = (0 if returns_time else -1) if isinstance(returns_time, bool) else returns_time

    def inner(f):
        f._benchmark_info = BenchmarkInfo(name or f.__name__, dict(input_names),
                                          output_names or "time (milliseconds)",
                                          dict(formats), rt)
        return f

    return inner


def get_benchmark_info(f):
    info = getattr(f, "_benchmark_info", None)
    return info if info is not None else benchmark_info()(f)._benchmark_info


def format_dtype(dtype):
    return numpy.dtype(dtype).name


def get_numpy(np, a):
    """Convert an array of the package ``np`` to NumPy."""
    return numpy.array(a)


def nthroot(x, p, lower_bound=1):
    root = round(x ** (1 / p))
    return max(lower_bound, root if root**p == x else math.floor(x ** (1 / p)))


# ---- timers ----


class _Timer:
    def __init__(self, interface=None):
        self._interface = interface
        self._t0 = None

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        """Milliseconds since start(), after the backend has finished."""
        assert self._t0 is not None, "Timer.stop() called without preceding Timer.start()"
        if self._interface is not None:
            cnp.synchronize(self._interface)
        return (time.perf_counter() - self._t0) * 1e3

    def sync(self, sync_mode="fence"):
        if sync_mode == "block" and self._interface is not None:
            cnp.synchronize(self._interface)


def get_timer(np, *, blocking=False):
    return _Timer(_HARNESS.interface if np is not numpy and _HARNESS else None)


def timed_loop(f, timer, runs, warmup=0, *, sync_mode="fence"):
    if runs <= 0:
        return 0.0
    for i in range(runs + warmup):
        if i == warmup:
            timer.start()
        f()
        if i + 1 < runs + warmup:
            timer.sync(sync_mode)
    return timer.stop()


# ---- module proxies: record (charmnumeric) / replay (NumPy) ----


class _Session:
    """Values recorded by the charmnumeric run and replayed by the NumPy run."""

    def __init__(self):
        self.draws = []      # (name, device array copy or host value)
        self.factors = []    # (name, args, kwargs, result)
        self.cursor = {"draws": 0, "factors": 0}
        self.failures = []

    def next(self, kind, name):
        items = getattr(self, kind)
        i = self.cursor[kind]
        if i >= len(items):
            raise RuntimeError(f"NumPy run made more {kind} than the charmnumeric run ({name})")
        self.cursor[kind] = i + 1
        rec = items[i]
        if rec[0] != name:
            raise RuntimeError(f"NumPy run called {name}, charmnumeric run {rec[0]}")
        return rec


# Results NumPy leaves unspecified (factor signs, tie order), checked then replayed.
_FACTORS = {"linalg": ("qr", "svd", "tssvd", "eigh"), "": ("argsort",)}


def _record(value):
    return value.copy() if isinstance(value, _DEVICE) else value


class _RecordingGenerator:
    def __init__(self, gen, session):
        self._gen, self._session = gen, session

    def __getattr__(self, name):
        fn = getattr(self._gen, name)

        def draw(*args, **kwargs):
            out = fn(*args, **kwargs)
            self._session.draws.append(("rng." + name, _record(out)))
            return out

        return draw


class _ReplayGenerator:
    def __init__(self, gen, session):
        self._gen, self._session = gen, session

    def __getattr__(self, name):
        fn = getattr(self._gen, name)

        def draw(*args, **kwargs):
            return _replay_draw(self._session, "rng." + name, fn(*args, **kwargs))

        return draw


def _host(x):
    return numpy.asarray(x.get(_HARNESS.interface)) if isinstance(x, _DEVICE) else x


def _replay_draw(session, name, want):
    _, got = session.next("draws", name)
    got = _host(got)
    if (numpy.shape(got) != numpy.shape(want)
            or numpy.asarray(got).dtype != numpy.asarray(want).dtype):
        session.failures.append(
            f"np.{name}: charmnumeric drew {numpy.asarray(got).dtype}{numpy.shape(got)}, "
            f"NumPy {numpy.asarray(want).dtype}{numpy.shape(want)}")
    if numpy.ndim(want) == 0 and not isinstance(want, numpy.ndarray):
        return type(want)(numpy.asarray(got).item())
    return numpy.array(got)


class _Module:
    """Attribute proxy of an array module (and its random/linalg submodules)."""

    def __init__(self, module, session, replay, path=""):
        self._module, self._session, self._replay, self._path = module, session, replay, path
        self.__name__ = module.__name__

    def __getattr__(self, name):
        if self._replay and self._path == "linalg" and name == "tssvd":
            return self._factor(name, None)  # cuPyNumeric only: checked as a thin SVD
        attr = getattr(self._module, name)
        if self._path == "" and name in ("random", "linalg"):
            return _Module(attr, self._session, self._replay, name)
        if self._path == "random" and callable(attr) and not isinstance(attr, type):
            return self._random(name, attr)
        if name in _FACTORS.get(self._path, ()):
            return self._factor(name, attr)
        if self._replay and self._path == "" and name == "convolve":
            return _np_convolve
        return attr

    def _random(self, name, fn):
        session = self._session
        if name == "default_rng":
            wrap = _ReplayGenerator if self._replay else _RecordingGenerator
            return lambda *a, **k: wrap(fn(*a, **k), session)
        if name == "seed":
            return fn
        if self._replay:
            return lambda *a, **k: _replay_draw(session, name, fn(*a, **k))

        def draw(*args, **kwargs):
            out = fn(*args, **kwargs)
            session.draws.append((name, _record(out)))
            return out

        return draw

    def _factor(self, name, fn):
        session = self._session
        if self._replay:
            def check(*args, **kwargs):
                result = session.next("factors", name)[3]
                return _check_factors(session, name, args, kwargs, result)

            return check

        def factor(*args, **kwargs):
            out = fn(*args, **kwargs)
            session.factors.append((name, args, kwargs, out))
            return out

        return factor


def _np_convolve(a, v, mode="full", method="auto"):
    """cuPyNumeric's N-D convolve (scipy.signal.convolve semantics) in NumPy."""
    a, v = numpy.asarray(a), numpy.asarray(v)
    if a.ndim == 1:
        return numpy.convolve(a, v, mode)
    full_shape = tuple(n + f - 1 for n, f in zip(a.shape, v.shape))
    full = numpy.zeros(full_shape, dtype=numpy.result_type(a, v))
    for k in numpy.ndindex(*v.shape):
        full[tuple(slice(k[d], k[d] + a.shape[d]) for d in range(a.ndim))] += v[k] * a
    if mode == "full":
        return full
    if mode == "same":
        start = [(f - 1) // 2 for f in v.shape]
        return full[tuple(slice(s, s + n) for s, n in zip(start, a.shape))]
    return full[tuple(slice(f - 1, n) for f, n in zip(v.shape, a.shape))]


def _check_argsort(session, a, idx, axis=-1, kind=None, order=None):
    want = numpy.argsort(a, axis, kind=kind)
    if idx.shape != want.shape or idx.dtype != want.dtype:
        session.failures.append(
            f"np.argsort: {idx.dtype}{idx.shape}, NumPy {want.dtype}{want.shape}")
    elif kind in ("stable", "mergesort"):
        if not numpy.array_equal(idx, want):
            session.failures.append("np.argsort(kind='stable') differs from NumPy")
    else:
        flat = a.reshape(-1) if axis is None else a
        ax = 0 if axis is None else axis
        perm = numpy.sort(idx, axis=ax)
        ramp = numpy.broadcast_to(numpy.arange(flat.shape[ax]).reshape(
            [-1 if d == ax % flat.ndim else 1 for d in range(flat.ndim)]), flat.shape)
        if not numpy.array_equal(perm, ramp) or not numpy.array_equal(
                numpy.take_along_axis(flat, idx, ax), numpy.sort(flat, axis=ax), equal_nan=True):
            session.failures.append("np.argsort does not sort its input")
    return idx


def _check_factors(session, name, args, kwargs, result):
    """Validate charmnumeric factors of the NumPy-run input ``args[0]``; return them."""
    a = numpy.asarray(args[0])
    if name == "argsort":
        return _check_argsort(session, a, numpy.asarray(_host(result)), *args[1:], **kwargs)
    got = [numpy.asarray(_host(r)) for r in result]
    rtol = _RTOL.get(numpy.result_type(a.dtype, numpy.float32).char, 1e-9) * 10
    scale = max(1.0, float(numpy.max(numpy.abs(a))) if a.size else 1.0) * max(a.shape)

    def close(label, x, y):
        if numpy.shape(x) != numpy.shape(y) or not numpy.allclose(x, y, rtol=rtol,
                                                                   atol=rtol * scale):
            session.failures.append(f"np.linalg.{name}: {label}")

    def unitary(label, q, cols=True):
        q = q if cols else q.conj().T
        close(label, q.conj().T @ q, numpy.eye(q.shape[1]))

    if name == "tssvd":
        want = numpy.linalg.svd(a, full_matrices=False)
    else:
        want = getattr(numpy.linalg, name)(a, *args[1:], **kwargs)
    want = [numpy.asarray(w) for w in want]
    for g, w, part in zip(got, want, "abc"):
        if g.shape != w.shape or g.dtype != w.dtype:
            session.failures.append(f"np.linalg.{name}: factor {part} is {g.dtype}{g.shape}, "
                                    f"NumPy {w.dtype}{w.shape}")
            return tuple(got)
    if name == "qr":
        q, r = got
        close("q @ r != a", q @ r, a)
        unitary("q not orthonormal", q)
        close("r not upper triangular", numpy.tril(r, -1), numpy.zeros_like(r))
    elif name == "eigh":
        w, v = got
        close("eigenvalues", w, want[0])
        unitary("eigenvectors not orthonormal", v)
        close("a v != v w", a @ v, v * w)
    else:
        u, s, vh = got
        k = s.shape[0]
        close("singular values", s, want[1])
        unitary("u not orthonormal", u)
        unitary("vh not orthonormal", vh, cols=False)
        close("u s vh != a", (u[:, :k] * s) @ vh[:k], a)
    return tuple(got)


# ---- locals of every function of the program, recorded at return ----


def _program_codes():
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    codes, todo = [], []
    for obj in vars(main).values() if main else ():
        code = getattr(obj, "__code__", None)
        if code is not None and code.co_filename == path:
            todo.append(code)
    while todo:
        code = todo.pop()
        if code in codes:
            continue
        codes.append(code)
        todo.extend(c for c in code.co_consts if hasattr(c, "co_code"))
    return codes


@contextlib.contextmanager
def _recording(store):
    mon = sys.monitoring
    codes = _program_codes()

    def on_return(code, offset, retval):
        frame = sys._getframe(1)
        record = dict(frame.f_locals)
        record["<return>"] = retval
        store[code.co_qualname] = record

    mon.use_tool_id(_TOOL, "cnum-benchmark")
    mon.register_callback(_TOOL, mon.events.PY_RETURN, on_return)
    for code in codes:
        mon.set_local_events(_TOOL, code, mon.events.PY_RETURN)
    try:
        yield
    finally:
        for code in codes:
            mon.set_local_events(_TOOL, code, 0)
        mon.register_callback(_TOOL, mon.events.PY_RETURN, None)
        mon.free_tool_id(_TOOL)


def _comparable(x):
    return isinstance(x, _DEVICE + (numpy.ndarray, numpy.generic, int))


def _magnitude(x):
    if isinstance(x, (numpy.ndarray, numpy.generic)) and x.dtype.kind in "fc" and x.size:
        finite = numpy.abs(x[numpy.isfinite(x)]) if x.ndim else numpy.abs(x)
        return float(numpy.max(finite)) if numpy.size(finite) else 0.0
    return 0.0


def _compare_records(where, got_rec, want_rec, failures, stats=None):
    pairs = []
    for var, want in want_rec.items():
        if var not in got_rec:
            continue
        got = got_rec[var]
        if isinstance(want, (tuple, list)) and isinstance(got, (tuple, list)) \
                and len(want) == len(got):
            pairs.extend((f"{var}[{i}]", g, w) for i, (g, w) in enumerate(zip(got, want))
                         if _comparable(g))
        elif _comparable(got):
            pairs.append((var, got, want))
    host = []
    for var, g, w in pairs:
        if os.environ.get("CNUM_BENCH_TRACE"):
            print(f"[trace] {where}.{var} {type(g).__name__} {getattr(g, 'shape', '')}",
                  file=sys.stderr, flush=True)
        host.append((var, numpy.asarray(_host(g)), w))
    scale = max([_magnitude(numpy.asarray(w)) for _, _, w in host] + [0.0])
    for var, g, w in host:
        label = f"{where}: {var}"
        w = numpy.asarray(w)
        if g.shape != w.shape:
            failures.append(f"{label} has shape {g.shape}, {_REF} {w.shape}")
            continue
        if g.dtype != w.dtype:
            failures.append(f"{label} has dtype {g.dtype}, {_REF} {w.dtype}")
            continue
        if stats is not None and w.dtype.kind in "fciub" and w.size:
            with numpy.errstate(invalid="ignore"):
                d = numpy.abs(g.astype(complex) - w)
            d = d[numpy.isfinite(d)]
            err = float(d.max()) if d.size else 0.0
            stats["abs"] = max(stats["abs"], err)
            stats["rel"] = max(stats["rel"], err / max(scale, 1e-300))
        if w.dtype.kind in "fc":
            rtol = _RTOL.get(w.dtype.char, 1e-9)
            ok = numpy.allclose(g, w, rtol=rtol, atol=rtol * scale, equal_nan=True)
        else:
            ok = numpy.array_equal(g, w)
        if not ok:
            if w.dtype.kind in "fc":
                err = float(numpy.nanmax(numpy.abs(g.astype(complex) - w))) if w.size else 0.0
                detail = f"max|err| {err:.3e} (scale {scale:.3e})"
            else:
                detail = f"{int(numpy.sum(g != w))} of {w.size} differ"
            failures.append(f"{label} differs from {_REF}: {detail}")


# ---- harness ----


class _Tee(io.TextIOBase):
    """stdout copy that notices the program's own FAIL lines."""

    def __init__(self, out):
        self.out, self.failed, self._line = out, False, ""

    def write(self, text):
        self._line += text
        *lines, self._line = self._line.split("\n")
        self.failed |= any(line.startswith("FAIL") for line in lines)
        return self.out.write(text)

    def flush(self):
        self.out.flush()


class BenchmarkHarness:
    """The part of cuPyNumeric's BenchmarkHarness the example programs use."""

    name = "benchmark harness"

    def __init__(self, args):
        vargs = vars(args)
        self.repeat = vargs.get("__cpn_repeat", 0)
        self._numpy_only = vargs.get("__cpn_package") == "numpy"
        self.package = "numpy" if self._numpy_only else "charmnumeric"
        self.interface = None if self._numpy_only else _connect()
        self.session = _Session()
        self.np = numpy if self._numpy_only else _Module(cnp, self.session, False)
        self.timer = self.execution_timer = self.wall_timer = _Timer(self.interface)
        self.failures = []
        self.cupy = os.environ.get("CNUM_REFERENCE", "numpy") == "cupynumeric"

    def run_timed_with_info(self, info, f, *args, **kwargs):
        self._run(info, f, args, kwargs)

    run_with_info = run_timed_with_info

    def run_timed(self, f, *args, **kwargs):
        self._run(get_benchmark_info(f), f, args, kwargs)

    run = run_timed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def _run(self, info, f, args, kwargs):
        runs = []
        for combo in itertools.product(*[a if isinstance(a, list) else [a] for a in args]):
            label = "" if all(not isinstance(a, list) for a in args) else \
                " [" + ", ".join(str(c) for c, a in zip(combo, args) if isinstance(a, list)) + "]"
            if self._numpy_only:
                f(*combo, **kwargs)
                continue
            try:
                self._compare(f, combo, kwargs, label, info, runs)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                msg = str(exc).splitlines()[0] if str(exc) else ""
                self.failures.append(f"{type(exc).__name__}{label}: {msg}")
        if self.cupy and runs:
            try:
                import _cupy_reference

                _cupy_reference.compare(self, info, f, runs, kwargs)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.failures.append(f"cuPyNumeric reference: {type(exc).__name__}: {exc}")
        if self.failures:
            print(f"FAIL {_NAME}: " + "; ".join(self.failures[:8]), flush=True)
            sys.exit(1)
        print(f"PASS {_NAME}", flush=True)

    def _compare(self, f, combo, kwargs, label, info=None, runs=None):
        session = self.session = _Session()
        self.np._session = session
        got, want = {}, {}
        tee = _Tee(sys.stdout)
        t0 = time.perf_counter()
        with _recording(got), contextlib.redirect_stdout(tee):
            f(*[self.np if a is self.np else a for a in combo], **kwargs)
        wall_ms = (time.perf_counter() - t0) * 1e3
        if tee.failed:
            self.failures.append(f"program printed FAIL{label}")
        if self.cupy:
            runs.append(dict(label=label, combo=combo, got=got, session=session,
                             wall_ms=wall_ms))
            return

        ref = _Module(numpy, session, True)
        ref_args = [ref if a is self.np else a for a in combo]
        ref_kwargs = {k: (_Timer() if v is self.timer else v) for k, v in kwargs.items()}
        with _recording(want), contextlib.redirect_stdout(io.StringIO()):
            try:
                f(*ref_args, **ref_kwargs)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"NumPy rerun failed: {type(exc).__name__}: {exc}") from exc
        if session.cursor["draws"] != len(session.draws):
            session.failures.append("NumPy run made fewer random draws")
        failures = list(session.failures)
        for where in want:
            if where in got:
                _compare_records(where, got[where], want[where], failures)
        n = sum(len(r) for r in want.values())
        print(f"[benchmark] compared {len(want)} functions ({n} locals) with NumPy{label}: "
              f"{'ok' if not failures else f'{len(failures)} differences'}", flush=True)
        for msg in failures:
            print(f"  {msg}", flush=True)
        self.failures.extend(m + label for m in failures)


def _connect():
    from _cnum_connect import get_interface

    return get_interface()


_HARNESS = None


def _add_harness_flags(parser):
    group = parser.add_argument_group(BenchmarkHarness.name)
    group.add_argument("-b", "--benchmark", dest="__cpn_repeat", metavar="N", type=int,
                       default=0)
    group.add_argument("--package", dest="__cpn_package", default="charmnumeric",
                       choices=["charmnumeric", "legate", "numpy"])
    group.add_argument("--short", dest="__cpn_short", action="store_true")
    group.add_argument("--summarize", dest="__cpn_summarize", action="store_true")
    group.add_argument("--summarize-flush", dest="__cpn_summarize_flush", default="run")
    group.add_argument("--cupy-allocator", dest="__cpn_cupy_allocator", default="default")
    group.add_argument("--log-conda-list", dest="__cpn_log_conda_list", action="store_true")
    group.add_argument("--log-metadata-extra", dest="__cpn_metadata_extra", nargs="+",
                       default=[])
    group.add_argument("--timer-mode", dest="__cpn_timer_mode", default="execution")
    return group


def parse_with_harness(parser):
    global _HARNESS
    _add_harness_flags(parser)
    args = parser.parse_known_args()[0]
    _HARNESS = BenchmarkHarness(args)
    return args, _HARNESS


__all__ = [
    "BenchmarkHarness",
    "BenchmarkInfo",
    "benchmark_info",
    "format_dtype",
    "get_benchmark_info",
    "get_numpy",
    "get_timer",
    "nthroot",
    "parse_with_harness",
    "timed_loop",
]
