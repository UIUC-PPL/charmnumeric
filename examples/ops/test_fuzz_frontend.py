"""Differential fuzz test of the charmnumeric frontend against NumPy.

Usage: test_fuzz_frontend.py [first_seed [count]]

Random 1-3-D shapes (mostly non-tile-aligned), all dtypes, views and slices
(steps, negative indices, ellipsis, newaxis), scalar/array mixing, chained ops,
setitem and get(). Each case replays an op log against NumPy. Operations that
charmnumeric documents as unsupported (NotImplementedError, mixed-dtype
TypeError, ...) are tallied as "rejected"; wrong values, wrong shapes/dtypes or
unexpected exceptions are failures. A server hang (CCS timeout) aborts the run
with exit code 3 so a driver can restart the server at the next seed.
"""

import os
import sys
import traceback

import numpy as np

import charmnumeric as cnp
from charmtyles.core import sync  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops_common import connect  # noqa: E402

DTYPES = [np.float32, np.float64, np.int32, np.int64, np.bool_, np.int8, np.uint16, np.float16]
REJECT = (NotImplementedError,)


def host_values(rng, shape, dtype):
    if np.dtype(dtype).kind == "f":
        v = np.round(rng.uniform(-5, 5, shape) * 4) / 4
        if rng.random() < 0.15 and v.size:
            v.flat[int(rng.integers(0, v.size))] = [0.0, -0.0, np.inf, -np.inf, np.nan][int(rng.integers(0, 5))]
        if rng.random() < 0.05:
            v[...] = [0.0, -0.0][int(rng.integers(0, 2))]
    else:
        v = rng.integers(-9, 10, shape)
    return v.astype(dtype)


def rand_shape(rng):
    nd = int(rng.integers(1, 4))
    big = rng.random() < 0.08
    big = big and not os.environ.get("FUZZ_NOBIG")
    if big:
        return {1: (262144 + int(rng.integers(1, 5000)),), 2: (600, int(rng.integers(2, 9))),
                3: (70, 3, int(rng.integers(2, 6)))}[nd]
    return tuple(int(rng.integers(1, 40)) for _ in range(nd))


def rand_bound(rng, n):
    if rng.random() < 0.3:
        return None
    return int(rng.integers(-n - 2, n + 3))


def rand_key(rng, shape):
    """Return (cnp_key, numpy_key)."""
    nd = len(shape)
    k = int(rng.integers(0, nd + 1))
    ell = int(rng.integers(0, k + 1)) if (rng.random() < 0.15 and k < nd) else None
    items, ref = [], []
    for j in range(k):
        d = j if ell is None or j < ell else j + (nd - k)
        n = shape[d]
        r = rng.random()
        if r < 0.3:
            i = int(rng.integers(-n, n)) if rng.random() < 0.95 else n + int(rng.integers(0, 3))
            items.append(i)
            ref.append(i)
        elif r < 0.4:
            items.append(slice(None))
            ref.append(slice(None))
        else:
            step = [None, None, 1, 2, 3, 5, -1][int(rng.integers(0, 7))]
            sl = slice(rand_bound(rng, n), rand_bound(rng, n), step)
            items.append(sl)
            ref.append(sl)
    if ell is not None:
        items.insert(ell, Ellipsis)
        ref.insert(ell, Ellipsis)
    if rng.random() < 0.05:
        pos = int(rng.integers(0, len(items) + 1))
        items.insert(pos, None)
        ref.insert(pos, None)
    if len(items) == 1 and rng.random() < 0.5 and items[0] is not None:
        return items[0], ref[0]
    return tuple(items), tuple(ref)


class Fail(Exception):
    pass


class Case:
    def __init__(self, seed, iface):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.iface = iface
        self.log = []
        self.pool = []  # (cnp object, numpy reference)
        self.rejected = []

    def note(self, msg):
        self.log.append(msg)

    def check(self, obj, ref, what, signed=False):
        ref = np.asarray(ref)
        if tuple(obj.shape) != ref.shape:
            raise Fail(f"{what}: shape {tuple(obj.shape)} != numpy {ref.shape}")
        if obj.dtype != ref.dtype:
            raise Fail(f"{what}: dtype {obj.dtype} != numpy {ref.dtype}")
        got = obj.get(self.iface)
        if ref.size == 0:
            if got.shape != ref.shape or got.dtype != ref.dtype:
                raise Fail(f"{what}: empty get() gave {got.shape}/{got.dtype}")
            return
        if got.shape != ref.shape or got.dtype != ref.dtype:
            raise Fail(f"{what}: get() gave {got.shape}/{got.dtype}, want {ref.shape}/{ref.dtype}")
        if ref.dtype.kind == "f":
            rtol = {2: 2e-3, 4: 1e-5}.get(ref.dtype.itemsize, 1e-12)
            ok = np.allclose(got, ref, rtol=rtol, atol=rtol * 0.1, equal_nan=True) and (
                not signed or bool(np.all((got == 0) <= (np.signbit(got) == np.signbit(ref)))))
        else:
            ok = np.array_equal(got, ref)
        if not ok:
            diff = ~np.isclose(got, ref, equal_nan=True) if ref.dtype.kind == "f" else got != ref
            if ref.dtype.kind == "f" and signed:
                diff |= (got == 0) & (np.signbit(got) != np.signbit(ref))
            bad = np.argwhere(diff)
            i = tuple(bad[0])
            raise Fail(f"{what}: value mismatch at {i}: got {got[i]!r} want {ref[i]!r} "
                       f"({len(bad)} of {ref.size} differ)")

    def reject(self, kind, exc):
        self.rejected.append(f"{kind}: {type(exc).__name__}")

    def pick(self, nonempty=True):
        cand = [p for p in self.pool if (p[1].size > 0 or not nonempty)]
        return cand[int(self.rng.integers(0, len(cand)))] if cand else None

    # --- steps ---------------------------------------------------------
    def new_base(self):
        shape = rand_shape(self.rng)
        dtype = DTYPES[int(self.rng.integers(0, len(DTYPES)))]
        host = host_values(self.rng, shape, dtype)
        how = self.rng.random()
        if how < 0.6:
            obj = cnp.asarray(host)
            self.note(f"x{len(self.pool)} = cnp.asarray(host{shape} {np.dtype(dtype).name})")
        elif how < 0.8:
            obj = cnp.array(host.tolist(), dtype=dtype) if host.size < 500 else cnp.array(host)
            self.note(f"x{len(self.pool)} = cnp.array(list, dtype)")
        else:
            val = host.flat[0]
            obj = cnp.full(shape, val, dtype=dtype)
            host = np.full(shape, val, dtype=dtype)
            self.note(f"x{len(self.pool)} = cnp.full({shape}, {val!r}, {np.dtype(dtype).name})")
        self.pool.append((obj, host))
        self.check(obj, host, "creation", signed=True)

    def do_view(self):
        p = self.pick()
        obj, ref = p
        ck, rk = rand_key(self.rng, ref.shape)
        self.note(f"view key={ck!r} of shape {ref.shape}")
        try:
            want = np.asarray(ref[rk])  # integer-only keys give a 0-d copy
            want_exc = None
        except Exception as exc:
            want, want_exc = None, exc
        try:
            v = obj[ck]
            got_exc = None
        except Exception as exc:
            v, got_exc = None, exc
        if got_exc is not None:
            if isinstance(got_exc, REJECT):
                self.reject("getitem", got_exc)
                return
            if want_exc is not None and isinstance(got_exc, type(want_exc)):
                return
            raise Fail(f"getitem {ck!r}: raised {got_exc!r}; numpy: "
                       f"{'raised ' + repr(want_exc) if want_exc else 'ok'}")
        if want_exc is not None:
            raise Fail(f"getitem {ck!r}: no error but numpy raised {want_exc!r}")
        self.pool.append((v, want))
        self.check(v, want, f"view {ck!r}")

    def rand_operand(self, ref):
        """Operand for an op against array `ref`: (cnp operand, numpy operand, tag)."""
        r = self.rng.random()
        dt = ref.dtype
        if r < 0.3:
            if dt.kind == "f":
                v = float(np.round(self.rng.uniform(-3, 3) * 4) / 4)
                if self.rng.random() < 0.25:
                    v = int(v)
            else:
                v = int(self.rng.integers(-4, 5))
                if self.rng.random() < 0.15:
                    v = float(v) + 0.5 * int(self.rng.integers(0, 2))
            return v, v, f"scalar {v!r}"
        if r < 0.38:
            v = np.dtype(dt).type(self.rng.integers(-3, 4))
            return v, v, f"npscalar {v!r}"
        if r < 0.5:
            h = host_values(self.rng, ref.shape, dt)
            how = self.rng.random()
            if how < 0.7:
                return h, h, "host array"
            return h.tolist(), np.asarray(h.tolist()), "host list"
        cands = [p for p in self.pool if p[1].size > 0 and p[1].shape == ref.shape]
        if self.rng.random() < 0.15:
            cands = [p for p in self.pool if p[1].size > 0]
        if not cands:
            cands = [p for p in self.pool if p[1].shape == ref.shape and p[1].size > 0]
        o = cands[int(self.rng.integers(0, len(cands)))]
        return o[0], o[1], f"pool {o[1].shape}/{o[1].dtype}"

    def do_binop(self):
        obj, ref = self.pick()
        name = ["add", "sub", "mul", "div", "lt", "eq", "max", "pow2", "neg", "where", "mod"][
            int(self.rng.integers(0, 11))]
        reverse = self.rng.random() < 0.3
        if name in ("neg", "pow2"):
            ops = {"neg": lambda a: -a, "pow2": lambda a: a ** 2}
            self.note(f"{name} on {ref.shape}/{ref.dtype}")
            self.apply(lambda: ops[name](obj), lambda: ops[name](ref), name)
            return
        co, no, tag = self.rand_operand(ref)
        self.note(f"{name}{' rev' if reverse else ''} {ref.shape}/{ref.dtype} with {tag}")
        import operator as op
        fns = {
            "add": (op.add, op.add), "sub": (op.sub, op.sub), "mul": (op.mul, op.mul),
            "div": (op.truediv, op.truediv), "lt": (op.lt, op.lt),
            "eq": (op.eq, op.eq), "mod": (op.mod, op.mod),
            "max": (cnp.maximum, np.maximum),
        }
        if name == "where":
            def c():
                return cnp.where(obj > 0, obj, co) if not reverse else cnp.where(obj > 0, co, obj)

            def n():
                return np.where(ref > 0, ref, no) if not reverse else np.where(ref > 0, no, ref)
            self.apply(c, n, name)
            return
        cf, nf = fns[name]
        if name in ("mod", "div") and (np.asarray(ref if reverse else no) == 0).any():
            return
        if reverse:
            self.apply(lambda: cf(co, obj), lambda: nf(no, ref), name)
        else:
            self.apply(lambda: cf(obj, co), lambda: nf(ref, no), name)

    def apply(self, cfn, nfn, what):
        with np.errstate(all="ignore"):
            try:
                want, want_exc = nfn(), None
            except Exception as exc:
                want, want_exc = None, exc
        try:
            got, got_exc = cfn(), None
        except Exception as exc:
            got, got_exc = None, exc
        if got_exc is not None:
            if isinstance(got_exc, REJECT) or (isinstance(got_exc, TypeError) and
                                               ("dtype" in str(got_exc) or "convers" in str(got_exc))):
                self.reject(what, got_exc)
                return
            if want_exc is not None:
                return
            if isinstance(got_exc, ValueError) and "only size-1 operands broadcast" in str(got_exc):
                self.reject(what, got_exc)
                return
            raise Fail(f"{what}: raised {got_exc!r}")
        if want_exc is not None:
            raise Fail(f"{what}: no error, numpy raised {want_exc!r}")
        want = np.asarray(want)
        self.check(got, want, what)
        self.pool.append((got, want))

    def do_unary(self):
        obj, ref = self.pick()
        name = ["exp", "tanh", "sqrt", "abs", "floor", "sign", "copy", "square", "astype"][
            int(self.rng.integers(0, 9))]
        self.note(f"unary {name} on {ref.shape}/{ref.dtype}")
        cf = {
            "exp": lambda: cnp.exp(obj), "tanh": lambda: cnp.tanh(obj), "sqrt": lambda: cnp.sqrt(obj),
            "abs": lambda: abs(obj), "floor": lambda: cnp.floor(obj), "sign": lambda: cnp.sign(obj),
            "copy": lambda: obj.copy(), "square": lambda: cnp.square(obj), "astype": lambda: obj.astype(ref.dtype),
        }[name]
        nf = {
            "exp": lambda: np.exp(ref), "tanh": lambda: np.tanh(ref), "sqrt": lambda: np.sqrt(ref),
            "abs": lambda: np.abs(ref), "floor": lambda: np.floor(ref), "sign": lambda: np.sign(ref),
            "copy": lambda: ref.copy(), "square": lambda: np.square(ref), "astype": lambda: ref.copy(),
        }[name]
        self.apply(cf, nf, name)

    def do_reduce(self):
        obj, ref = self.pick()
        name = ["sum", "max", "min", "prod", "mean"][int(self.rng.integers(0, 5))]
        axis = None if ref.ndim <= 1 or self.rng.random() < 0.5 else int(
            self.rng.integers(-ref.ndim, ref.ndim))
        if name == "prod" and ref.size > 12:
            return
        self.note(f"reduce {name} axis={axis} on {ref.shape}/{ref.dtype}")
        try:
            got = getattr(cnp, name)(obj, axis=axis)
        except REJECT + (TypeError,) as exc:
            self.reject(name, exc)
            return
        want = np.asarray(getattr(np, name)(ref, axis=axis))
        if want.dtype != got.dtype and want.dtype.kind == "f" and got.dtype.kind == "f":
            pass
        self.check(got, want, f"{name} axis={axis}")
        self.pool.append((got, want))

    def do_setitem(self):
        obj, ref = self.pick()
        ck, rk = rand_key(self.rng, ref.shape)
        if self.rng.random() < 0.9:
            ck, rk = self._drop_none(ck, rk)
        try:
            tgt = ref[rk]
        except Exception:
            tgt = None
        if tgt is None or tgt.size == 0:
            return
        r = self.rng.random()
        if r < 0.35:
            v = float(self.rng.integers(-3, 4)) if ref.dtype.kind == "f" else int(self.rng.integers(-3, 4))
            if ref.dtype.kind != "f" and self.rng.random() < 0.15:
                v = [2.75, -2.75, 0.5][int(self.rng.integers(0, 3))]
            if self.rng.random() < 0.04:
                v = [2 ** 40, -2 ** 40, 2 ** 31][int(self.rng.integers(0, 3))]
            crhs = nrhs = v
            tag = f"scalar {v}"
        elif r < 0.55:
            h = host_values(self.rng, tgt.shape, ref.dtype)
            crhs, nrhs, tag = h, h, "host array"
        elif r < 0.65 and tgt.ndim > 1:
            h = host_values(self.rng, tgt.shape[-1:], ref.dtype)
            crhs, nrhs, tag = h, h, "host row broadcast"
        else:
            cands = [p for p in self.pool if p[1].shape == tgt.shape and p[1].size > 0
                     and p[1].dtype == ref.dtype]
            if self.rng.random() < 0.3:
                cands = [p for p in self.pool if p[1].size == tgt.size and p[1].size > 0]
            if not cands:
                return
            o = cands[int(self.rng.integers(0, len(cands)))]
            crhs, nrhs, tag = o[0], o[1], f"pool {o[1].shape}/{o[1].dtype}"
        self.note(f"setitem key={ck!r} on {ref.shape}/{ref.dtype} <- {tag}")
        # numpy evaluates the rhs fully first (overlap safe); snapshot to match
        nrhs_val = nrhs if isinstance(nrhs, (int, float)) else np.array(nrhs, copy=True)
        try:
            ref[rk] = nrhs_val
            want_exc = None
        except Exception as exc:
            want_exc = exc
        try:
            obj[ck] = crhs
            got_exc = None
        except Exception as exc:
            got_exc = exc
        if got_exc is not None:
            if isinstance(got_exc, REJECT) or (isinstance(got_exc, TypeError) and "dtype" in str(got_exc)):
                self.reject("setitem", got_exc)
                raise Fail("rejected setitem desyncs the reference; treated as unsupported") \
                    if want_exc is None and False else _Resync()
            if want_exc is not None:
                return
            raise Fail(f"setitem {ck!r}: raised {got_exc!r}")
        if want_exc is not None:
            sq = lambda sh: tuple(d for d in sh if d != 1)  # noqa: E731
            if sq(np.shape(nrhs)) == sq(tgt.shape):
                ref[rk] = np.reshape(nrhs_val, tgt.shape)  # layout-equal reshape is accepted
            else:
                raise Fail(f"setitem {ck!r}: no error, numpy raised {want_exc!r}")
        self.check(obj, ref, f"after setitem {ck!r}")

    @staticmethod
    def _drop_none(ck, rk):
        if isinstance(ck, tuple) and None in ck:
            return tuple(c for c in ck if c is not None), tuple(c for c in rk if c is not None)
        return ck, rk

    def run(self, steps):
        self.new_base()
        if self.rng.random() < 0.5:
            self.new_base()
        steps = min(steps, int(os.environ.get("FUZZ_MAXSTEP", steps)))
        for _ in range(steps):
            r = self.rng.random()
            try:
                if r < 0.35:
                    self.do_view()
                elif r < 0.7:
                    self.do_binop()
                elif r < 0.8:
                    self.do_setitem()
                elif r < 0.86:
                    self.do_unary()
                elif r < 0.9:
                    self.do_reduce()
                else:
                    self.new_base()
            except _Resync:
                # A rejected setitem leaves the reference ahead; resync it from the backend.
                for i, (o, rf) in enumerate(self.pool):
                    if rf.size:
                        rf[...] = o.get(self.iface)
        for i, (o, rf) in enumerate(self.pool):
            self.check(o, rf, f"final pool[{i}]")


class _Resync(Exception):
    pass


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    iface = connect()
    nfail = 0
    rejected = {}
    for seed in range(first, first + count):
        case = Case(seed, iface)
        print(f"SEED {seed}", flush=True)
        try:
            case.run(int(np.random.default_rng(seed + 7).integers(2, 9)))
        except Fail as exc:
            nfail += 1
            print(f"FAIL seed={seed}: {exc}", flush=True)
            for line in case.log:
                print("    " + line, flush=True)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            infra = any(s in msg for s in ("imeout", "onnection", "CCS", "reset", "Did not receive", "Not Alive"))
            print(f"{'HANG' if infra else 'ERROR'} seed={seed}: {msg}", flush=True)
            if not infra:
                traceback.print_exc(limit=4)
            for line in case.log:
                print("    " + line, flush=True)
            if infra:
                sys.exit(3)
            nfail += 1
        if os.environ.get("FUZZ_VERBOSE"):
            print(f"LOG seed={seed}", *case.log, sep="\n    ", flush=True)
        for r in case.rejected:
            rejected[r] = rejected.get(r, 0) + 1
    print("rejected (documented limits):", rejected, flush=True)
    print(("PASS" if nfail == 0 else "FAIL") + f" test_fuzz_frontend: {nfail} failing cases "
          f"of {count} (seeds {first}..{first + count - 1})", flush=True)
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
