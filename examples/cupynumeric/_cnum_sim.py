"""Host-side NumPy simulator for charmnumeric programs (static check aid).

Ops are evaluated eagerly as they are recorded; SimInterface.get() returns the
simulated values, so a port runs end to end without a backend. Checks the
Python side only.

Flags patterns the frontend accepts but the backend (as of reading src/) does
not support:
  ERROR  general broadcasting reaching an elementwise kernel
  ERROR  REDUCE / TILE / DIAG / scan / sort on a view (backend reads the whole array)
  ERROR  set_region shape mismatch
  WARN   cross-ndim set_region from a view, or into a nonzero offset in a
         non-singleton dim (cross_set_region_send ignores the target offset)
  WARN   set_region between arrays of different dtype
CNUM_SIM_STRICT=1 (default) raises SimError at the first ERROR.
"""

import atexit
import os
import sys
import weakref

import numpy as np

import charmtyles.ast as ct_ast
import charmtyles.core as ct_core
from charmtyles.core import FrontendObject, FrontendObjectView
from charmtyles.operation import Operation

from charmnumeric._rng_reference import creation_values
from charmnumeric.charmnumeric import ArrayOperation as AO
from charmnumeric.interface import CharmNumericInterface
from charmtyles.ast import InlineData


class SimError(RuntimeError):
    pass


STRICT = os.environ.get("CNUM_SIM_STRICT", "1") not in ("", "0")

_store = {}
_stats = {"ops": 0, "flushes": 0, "bytes": 0, "gets": 0}
_errors = []
_warnings = {}


def _error(msg):
    _errors.append(msg)
    if STRICT:
        raise SimError(msg)


def _warn(msg):
    _warnings[msg] = _warnings.get(msg, 0) + 1


def _slices(region):
    return tuple(slice(s, e, st) for s, e, st in zip(region.start, region.stop, region.step))


def _is_view(op):
    if not isinstance(op, FrontendObjectView):
        return False
    root = op._root_parent
    return tuple(op.shape) != tuple(root.shape) or any(s != 0 for s in op.region.start) or any(
        st != 1 for st in op.region.step
    )


def _value(op):
    if isinstance(op, FrontendObjectView):
        return _store[op._root_parent.name][_slices(op.region)]
    if isinstance(op, FrontendObject):
        return _store[op.name]
    return op


def _operand(op):
    """Value of an elementwise operand: size-1 arrays become scalars (BROADCAST)."""
    v = _value(op)
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.reshape(-1)[0]
    return v


def _desc(op):
    if isinstance(op, (FrontendObject, FrontendObjectView)):
        kind = "view" if isinstance(op, FrontendObjectView) else "array"
        return f"{kind}{tuple(op.shape)}"
    return repr(op)


def _int_power(a, b):
    """Backend integer power: wraps; negative exponents give 1 / +-1 / 0."""
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    neg = b < 0
    res = np.power(a, np.where(neg, 0, b))
    res = np.where(neg, np.where(a == 1, 1, np.where(a == -1, np.where(b % 2 == 0, 1, -1), 0)), res)
    return res


def _mask(fn):
    """Wrap a comparison or predicate so it returns bool arrays."""
    return lambda *args: np.asarray(fn(*args), dtype=bool)


_BINARY = {
    AO.add: np.add,
    AO.sub: np.subtract,
    AO.mul: np.multiply,
    AO.div: np.divide,
    AO.power: np.power,
    AO.maximum: np.maximum,
    AO.minimum: np.minimum,
    AO.fmax: np.fmax,
    AO.fmin: np.fmin,
    AO.mod: np.remainder,
    AO.floor_divide: np.floor_divide,
    AO.less: _mask(np.less),
    AO.less_equal: _mask(np.less_equal),
    AO.equal: _mask(np.equal),
    AO.not_equal: _mask(np.not_equal),
    AO.bitwise_and: np.bitwise_and,
    AO.bitwise_or: np.bitwise_or,
    AO.bitwise_xor: np.bitwise_xor,
    AO.left_shift: np.left_shift,
    AO.right_shift: np.right_shift,
}
_UNARY = {
    AO.tanh: np.tanh,
    AO.exp: np.exp,
    AO.sqrt: np.sqrt,
    AO.log: np.log,
    AO.log2: np.log2,
    AO.log10: np.log10,
    AO.log1p: np.log1p,
    AO.exp2: np.exp2,
    AO.expm1: np.expm1,
    AO.abs: np.abs,
    AO.floor: np.floor,
    AO.ceil: np.ceil,
    AO.rint: np.rint,
    AO.sin: np.sin,
    AO.cos: np.cos,
    AO.sign: np.sign,
    AO.isnan: _mask(np.isnan),
    AO.isinf: _mask(np.isinf),
    AO.isfinite: _mask(np.isfinite),
    AO.convert: lambda x: x,
    AO.real: np.real,
    AO.imag: np.imag,
    AO.conj: np.conjugate,
    AO.invert: np.invert,
}
# Compute dtype per emitOp in src/jit.cpp.
_COMPARISON_OPS = (AO.less, AO.less_equal, AO.equal, AO.not_equal)
_OPERAND_TYPED = (AO.isnan, AO.isinf, AO.isfinite, AO.abs, AO.real, AO.imag, AO.convert)


def _compute_dtype(op, ops, vals, result_dtype):
    arrays = [np.asarray(v).dtype for x, v in zip(ops, vals)
              if isinstance(x, (FrontendObject, FrontendObjectView))]
    if op in _COMPARISON_OPS:
        return np.result_type(*arrays) if arrays else np.dtype(np.float64)
    if op in _OPERAND_TYPED:
        return arrays[0] if arrays else np.dtype(np.float64)
    return np.dtype(result_dtype)
_TERNARY = {
    AO.where: lambda c, x, y: np.where(np.asarray(c) != 0, x, y),
    AO.clip: lambda x, lo, hi: np.minimum(np.maximum(x, lo), hi),
}
_NAMES = {AO.add: "add", AO.sub: "sub", AO.mul: "mul", AO.div: "div"}
_NAMES.update({op: name for name, op in vars(AO).items() if isinstance(op, int) and op >= 20})


def _squeeze_mat(v, what):
    if v.ndim == 3:
        singles = [i for i in range(3) if v.shape[i] == 1]
        if len(singles) != 1:
            _error(f"{what}: 3D operand needs exactly one singleton dim, got {v.shape}")
        return np.squeeze(v, axis=singles[0])
    return v


def _fit(result, obj):
    shape = tuple(obj.shape) or (1,)  # 0-d arrays are stored as (1,)
    result = np.asarray(result)
    if result.shape != shape:
        if result.size == 1:
            result = np.full(shape, result.reshape(-1)[0])
        elif result.size == int(np.prod(shape)):
            result = result.reshape(shape)
        else:
            _error(f"result shape {result.shape} != declared {shape}")
            result = np.zeros(shape)
    return result.astype(obj.dtype, copy=False)


def _eval(obj):
    ast = obj._active_ast
    op = ast.operation
    ops = ast.operands
    _stats["ops"] += 1

    if op == Operation.noop:
        return
    if op == Operation.create:
        _store_result(obj, np.zeros(tuple(obj.shape) or (1,), dtype=obj.dtype))
        return
    if op == Operation.set_region:
        target, region, rhs = ops
        tgt = _store[target.name]
        sl = _slices(region)
        dst = tgt[sl]
        if isinstance(rhs, (FrontendObject, FrontendObjectView)):
            rv = _value(rhs)
            if rv.dtype != tgt.dtype:
                _warn(f"set_region dtype {rv.dtype} -> {tgt.dtype}")
            if rv.size == 1:
                tgt[sl] = rv.reshape(-1)[0]
            elif rv.shape == dst.shape:
                tgt[sl] = rv
            elif rhs.ndims != target.ndims and tuple(s for s in rv.shape if s != 1) == tuple(
                s for s in dst.shape if s != 1
            ):
                if _is_view(rhs):
                    _warn(
                        f"cross-ndim set_region from a VIEW {_desc(rhs)} into "
                        f"{target.ndims}D region {dst.shape}"
                    )
                offs = [
                    region.start[d]
                    for d in range(len(region.start))
                    if dst.shape[d] > 1 and region.start[d] != 0
                ]
                if offs:
                    _warn(
                        f"cross-ndim set_region {rhs.ndims}D->{target.ndims}D with nonzero "
                        f"target offset in a non-singleton dim (region start {region.start})"
                    )
                tgt[sl] = rv.reshape(dst.shape)
            else:
                _error(
                    f"set_region shape mismatch: rhs {_desc(rhs)} into region {dst.shape} "
                    f"of {target.ndims}D array"
                )
        else:
            tgt[sl] = rhs
        return

    if op in _BINARY or op in _TERNARY:
        vals = [_operand(x) for x in ops]
        arrs = [v for v in vals if isinstance(v, np.ndarray)]
        if any(v.shape != arrs[0].shape for v in arrs[1:]):
            _error(
                f"elementwise {_NAMES[op]} with non-trivial broadcasting "
                f"{' vs '.join(_desc(x) for x in ops)} (backend only broadcasts size-1 operands)"
            )
        # operands are converted to the op's compute dtype in-kernel
        cdt = _compute_dtype(op, ops, vals, obj.dtype)
        with np.errstate(all="ignore"):
            vals = [np.asarray(v).astype(cdt) if i or op != AO.where else np.asarray(v)
                    for i, v in enumerate(vals)]
            if op == AO.power and np.issubdtype(cdt, np.integer):
                res = _int_power(*vals)
            else:
                res = (_BINARY.get(op) or _TERNARY.get(op))(*vals)
        _store_result(obj, _fit(res, obj))
        return

    if op in _UNARY:
        (a,) = ops
        v = _operand(a)
        cdt = _compute_dtype(op, ops, [v], obj.dtype)
        with np.errstate(all="ignore"):
            res = _UNARY[op](np.asarray(v).astype(cdt))
        _store_result(obj, _fit(res, obj))
        return

    if op == AO.reduce:
        a, b = ops
        for x in (a, b):
            if _is_view(x):
                _error(f"dot/REDUCE on a view {_desc(x)}: backend reduces the whole array")
        va, vb = _value(a), _value(b)
        _store_result(obj, _fit(np.array([np.dot(va.reshape(-1), vb.reshape(-1))]), obj))
        return

    if op == AO.matmul:
        m, v = ops
        mv = _squeeze_mat(_value(m), "matvec")
        vv = _value(v).reshape(-1)
        _store_result(obj, _fit(mv @ vv, obj))
        return

    if op == AO.matmatmul:
        a, b = ops
        av = _squeeze_mat(_value(a), "matmatmul lhs")
        bv = _squeeze_mat(_value(b), "matmatmul rhs")
        _store_result(obj, _fit(av @ bv, obj))
        return

    if op == AO.tile:
        x = ops[0]
        reps = tuple(int(r) for r in ops[1:])
        if _is_view(x):
            _error(f"tile of a view {_desc(x)}: backend tiles the whole array")
        _store_result(obj, _fit(np.tile(_value(x), reps), obj))
        return

    if op == AO.diag:
        v, k = ops
        if _is_view(v):
            _error(f"diag of a view {_desc(v)}: backend reads the whole array")
        _store_result(obj, _fit(np.diag(_value(v), int(k)), obj))
        return

    if op == AO.reduction:
        x, kind, mask, keepdims, ddof, _in_dtype = ops
        xv = _value(x)
        axes = tuple(d for d in range(xv.ndim) if (int(mask) >> d) & 1)
        _store_result(obj, _fit(_reduction(int(kind), xv, axes, bool(keepdims), int(ddof)), obj))
        return

    if op == AO.bincount:
        x = ops[0]
        weights = ops[1] if isinstance(ops[1], (FrontendObject, FrontendObjectView)) else None
        length = int(obj.shape[0])
        xv = _value(x).reshape(-1)
        keep = (xv >= 0) & (xv < length)
        wv = None if weights is None else _value(weights).reshape(-1)[keep]
        _store_result(obj, _fit(np.bincount(xv[keep], weights=wv, minlength=length), obj))
        return

    if op == AO.linalg:
        from charmnumeric import linalg as LA

        arrays = [x for x in ops if isinstance(x, (FrontendObject, FrontendObjectView))]
        kind, _nd, _s0, _s1, p0, p1, p2 = ops[len(arrays):]
        values = [_value(x) for x in arrays]
        res = LA._reference(int(kind), values, tuple(obj.shape), (p0, p1, p2))
        _store_result(obj, _fit(np.asarray(res).reshape(obj.shape), obj))
        return

    if op in (AO.upload, AO.random, AO.iota):
        # creation nodes: [region, (hi, lo) scalar word pairs..., InlineData?]
        scalars = [x for x in ops if isinstance(x, (int, float))]
        payload = next((x.data for x in ops if isinstance(x, InlineData)), None)
        _store_result(obj, creation_values(op, scalars, payload, tuple(obj.shape), obj.dtype))
        return

    if op in (AO.convolve, AO.flip):
        from charmnumeric import signal as SIG

        arrays = [x for x in ops if isinstance(x, (FrontendObject, FrontendObjectView))]
        scalars = ops[len(arrays):]
        res = SIG.reference(op, [_value(x) for x in arrays], scalars, tuple(obj.shape))
        _store_result(obj, _fit(np.asarray(res), obj))
        return

    if op in (AO.reshape, AO.broadcast):
        x = ops[0]
        shape = tuple(int(n) for n in ops[1:])
        fn = np.reshape if op == AO.reshape else np.broadcast_to
        _store_result(obj, _fit(np.array(fn(_value(x), shape)), obj))
        return

    if op == AO.transpose:
        x = ops[0]
        perm = tuple(int(p) for p in ops[1:])
        _store_result(obj, _fit(np.transpose(_value(x), perm), obj))
        return

    if op == AO.scan:
        x, axis, kind = ops
        if _is_view(x):
            _error(f"scan of a view {_desc(x)}: backend reads the whole array")
        fn = (np.cumsum, np.cumprod, np.nancumsum, np.nancumprod)[int(kind)]
        with np.errstate(all="ignore"):
            _store_result(obj, _fit(fn(_value(x), axis=int(axis)), obj))
        return

    if op in (AO.sort, AO.argsort):
        x, axis = ops
        if _is_view(x):
            _error(f"sort of a view {_desc(x)}: backend reads the whole array")
        fn = np.sort if op == AO.sort else np.argsort
        _store_result(obj, _fit(fn(_value(x), axis=int(axis), kind="stable"), obj))
        return

    if op in (AO.gather, AO.scatter):
        root, L = ops[0], ops[1]
        p = 3 if op == AO.scatter else 2
        npre, npost, nde = (int(v) for v in ops[p:p + 3])
        E = tuple(int(v) for v in ops[p + 3:p + 3 + nde])
        offs = _index_offsets(np.asarray(_value(L)).astype(np.int64), E, npre, npost,
                              [int(v) for v in ops[p + 3 + nde:]])
        tgt = _store[root.name]
        if offs.size and (offs.min() < 0 or offs.max() >= tgt.size):
            _error(f"index offsets out of bounds for array of size {tgt.size}")
            return
        if op == AO.gather:
            _store_result(obj, _fit(tgt.reshape(-1)[offs], obj))
        else:
            v = ops[2]
            vals = _value(v) if isinstance(v, (FrontendObject, FrontendObjectView)) else v
            tgt.flat[offs.reshape(-1)] = np.broadcast_to(np.asarray(vals), E).reshape(-1)
        return

    if op == AO.expand:
        x, mode, _total = ops
        xv = np.asarray(_value(x)).reshape(-1)
        counts = (xv != 0).astype(np.int64) if int(mode) == 0 else np.maximum(xv.astype(np.int64), 0)
        _store_result(obj, _fit(np.repeat(np.arange(xv.size), counts), obj))
        return

    _error(f"unknown opcode {op}")


def _index_offsets(Lv, E, npre, npost, strides):
    """Flat source/target offset of every element of a GATHER/SCATTER iteration space."""
    nb = Lv.ndim
    e = np.indices(E, dtype=np.int64)
    offs = Lv[tuple(e[npre:npre + nb])]
    slice_dims = list(range(npre)) + list(range(npre + nb, npre + nb + npost))
    for k, d in enumerate(slice_dims):
        offs = offs + e[d] * strides[k]
    return offs


def _reduction(kind, xv, axes, keepdims, ddof):
    """NumPy model of the REDUCTION executor."""
    from charmnumeric import reductions as R

    kw = dict(axis=axes, keepdims=keepdims)
    with np.errstate(all="ignore"):
        if kind == R._RED_SUM:
            return np.sum(xv, **kw)
        if kind == R._RED_PROD:
            return np.prod(xv, **kw)
        if kind == R._RED_MAX:
            return np.max(xv, **kw)
        if kind == R._RED_MIN:
            return np.min(xv, **kw)
        if kind in (R._RED_ARGMAX, R._RED_ARGMIN):
            fn = np.argmax if kind == R._RED_ARGMAX else np.argmin
            axis = axes[0] if len(axes) == 1 and xv.ndim > 1 else None
            return fn(xv, axis=axis, keepdims=keepdims)
        if kind == R._RED_COUNT_NONZERO:
            return np.count_nonzero(xv, **kw)
        if kind == R._RED_ALL:
            return np.all(xv, **kw)
        if kind == R._RED_ANY:
            return np.any(xv, **kw)
        if kind == R._RED_MEAN:
            return np.mean(xv, **kw)
        if kind == R._RED_VAR:
            return np.var(xv, ddof=ddof, **kw)
        if kind == R._RED_STD:
            return np.std(xv, ddof=ddof, **kw)
        if kind == R._RED_NORM2:
            return np.sqrt(np.sum(xv.astype(np.float64) * xv, **kw))
        if kind == R._RED_NORM1:
            return np.sum(np.abs(xv), **kw)
        if kind == R._RED_NORM_INF:
            return np.max(np.abs(xv), **kw)
        if kind == R._RED_NORM_NEGINF:
            return np.min(np.abs(xv), **kw)
    _error(f"unknown reduction kind {kind}")


def _store_result(obj, value):
    name = obj.name
    _store[name] = value
    weakref.finalize(obj, _store.pop, name, None)


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------
_orig_add_fusible = ct_ast.ExecutionState.add_fusible_obj
_orig_add_node = ct_ast.ExecutionState.add_node


def _add_fusible_obj(self, obj):
    _eval(obj)
    return _orig_add_fusible(self, obj)


def _add_node(self, node):
    for obj in node.objs:
        _eval(obj)
    return _orig_add_node(self, node)


if not getattr(ct_ast.ExecutionState, "_cnum_sim_hooked", False):
    ct_ast.ExecutionState.add_fusible_obj = _add_fusible_obj
    ct_ast.ExecutionState.add_node = _add_node
    ct_ast.ExecutionState._cnum_sim_hooked = True


class SimInterface(CharmNumericInterface):
    """CharmNumericInterface stand-in that never talks to a backend."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.odf = 4

    def connect(self, *args, **kwargs):
        pass

    def run_async(self, cmd):
        _stats["flushes"] += 1
        _stats["bytes"] += len(cmd)

    def run_sync(self, cmd, reply_type="i", reply_size=4):
        self.run_async(cmd)
        return 0

    def get(self, ndims, name, size, dtype=None):
        _stats["gets"] += 1
        if name == -1:  # synchronize() barrier
            return np.empty(0, dtype=dtype)
        arr = _store[name]
        if arr.size != size:
            _error(f"get size mismatch for name {name}: {arr.size} != {size}")
        return np.array(arr.reshape(-1), dtype=dtype if dtype is not None else arr.dtype)


def _summary():
    lines = [
        f"[cnum-sim] ops={_stats['ops']} names_used={ct_core.next_name}/{ct_core.max_name} "
        f"flushes={_stats['flushes']} dag_bytes={_stats['bytes']} gets={_stats['gets']} "
        f"errors={len(_errors)} warnings={sum(_warnings.values())}"
    ]
    for msg in _errors[:10]:
        lines.append(f"[cnum-sim] ERROR {msg}")
    for msg, n in list(_warnings.items())[:10]:
        lines.append(f"[cnum-sim] WARN x{n} {msg}")
    print("\n".join(lines), file=sys.stderr)


atexit.register(_summary)
