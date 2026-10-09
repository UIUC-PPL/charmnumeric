import builtins

import numpy as np

from charmtyles.operation import fusible_op, nonfusible_op

from .charmnumeric import (
    Array,
    ArrayOperation,
    ArrayView,
    _clip,
    _coerce_scalar,
    _compare,
    _ew_binary,
    _ew_unary,
    _is_scalar_like,
    _log,
    _logical_and,
    _logical_not,
    _logical_or,
    _logical_xor,
    _power,
    _reciprocal,
    _round,
    _trunc,
    _where,
    arange,
    array,
    asarray,
    copy,
    copyto,
    create_array,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    identity,
    ones,
    ones_like,
    zeros,
    zeros_like,
)


def _coerce_binary_operand(value):
    if isinstance(value, (Array, ArrayView)):
        return value
    if _is_scalar_like(value):
        return _coerce_scalar(value)
    return asarray(value)


def tanh(x):
    if _is_scalar_like(x):
        return np.tanh(_coerce_scalar(x))
    x = asarray(x)
    return fusible_op(x, operation=ArrayOperation.tanh, shape=x.shape, dtype=x.dtype)


def exp(x):
    if _is_scalar_like(x):
        return np.exp(_coerce_scalar(x))
    x = asarray(x)
    return fusible_op(x, operation=ArrayOperation.exp, shape=x.shape, dtype=x.dtype)


def tile(x, reps):
    if _is_scalar_like(x):
        return array(np.tile(_coerce_scalar(x), reps))

    x = asarray(x)
    if isinstance(reps, int):
        reps = (reps,)
    reps = tuple(int(rep) for rep in reps)

    out_ndims = max(x.ndims, len(reps))
    padded_shape = (1,) * (out_ndims - x.ndims) + x.shape
    padded_reps = (1,) * (out_ndims - len(reps)) + reps

    out_shape = tuple(s * r for s, r in zip(padded_shape, padded_reps))
    return nonfusible_op(
        x,
        *padded_reps,
        operation=ArrayOperation.tile,
        shape=out_shape,
        dtype=x.dtype,
    )


def add(x1, x2):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _coerce_scalar(x1) + _coerce_scalar(x2)
    return _coerce_binary_operand(x1) + _coerce_binary_operand(x2)


def subtract(x1, x2):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _coerce_scalar(x1) - _coerce_scalar(x2)
    return _coerce_binary_operand(x1) - _coerce_binary_operand(x2)


def multiply(x1, x2):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _coerce_scalar(x1) * _coerce_scalar(x2)
    return _coerce_binary_operand(x1) * _coerce_binary_operand(x2)


def divide(x1, x2):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _coerce_scalar(x1) / _coerce_scalar(x2)
    return _coerce_binary_operand(x1) / _coerce_binary_operand(x2)


def matmul(x1, x2):
    if _is_scalar_like(x1) or _is_scalar_like(x2):
        raise ValueError("matmul does not support scalar operands")
    return asarray(x1) @ asarray(x2)


def dot(x1, x2):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _coerce_scalar(x1) * _coerce_scalar(x2)

    lhs = _coerce_binary_operand(x1)
    rhs = _coerce_binary_operand(x2)
    if _is_scalar_like(lhs) or _is_scalar_like(rhs):
        return lhs * rhs
    return lhs.dot(rhs)


def norm2(x):
    """Compute the squared L2 norm of a 1D array."""

    x = asarray(x)
    if x.ndims != 1:
        raise ValueError(f"norm2 requires a 1D array, got {x.ndims}D")
    return x.dot(x)


def diag(v, k=0):
    """Construct a diagonal matrix or extract a diagonal."""

    v = asarray(v)

    if v.ndims == 1:
        n = v.shape[0] + builtins.abs(k)
        return nonfusible_op(v, k, operation=ArrayOperation.diag, shape=(n, n), dtype=v.dtype)
    if v.ndims == 2:
        rows, cols = v.shape
        if k >= 0:
            diag_len = max(0, min(rows, cols - k))
        else:
            diag_len = max(0, min(rows + k, cols))
        if diag_len == 0:
            raise ValueError(f"k={k} is out of range for shape ({rows}, {cols})")
        return nonfusible_op(
            v,
            k,
            operation=ArrayOperation.diag,
            shape=(diag_len,),
            dtype=v.dtype,
        )
    raise ValueError(f"diag requires a 1D or 2D array, got {v.ndims}D")


# ---------------------------------------------------------------------------
# Elementwise math, comparisons and selection (see charmnumeric.py, opcodes
# 20-49). Comparisons/predicates return 0/1 masks in the operands' dtype
# (no bool dtype). With only scalar arguments they fall back to NumPy.
# ---------------------------------------------------------------------------


def _all_scalars(*values):
    return all(_is_scalar_like(v) for v in values)


def _with_out(result, out):
    if out is None:
        return result
    out[...] = result
    return out


def _unary_function(name, operation, float_only=False, np_name=None):
    np_func = getattr(np, np_name or name)

    def func(x, out=None):
        if _is_scalar_like(x):
            return np_func(_coerce_scalar(x))
        return _with_out(_ew_unary(x, operation, name, float_only=float_only), out)

    func.__name__ = name
    func.__qualname__ = name
    func.__doc__ = f"Elementwise ``numpy.{np_name or name}``."
    return func


sqrt = _unary_function("sqrt", ArrayOperation.sqrt, float_only=True)
log2 = _unary_function("log2", ArrayOperation.log2, float_only=True)
log10 = _unary_function("log10", ArrayOperation.log10, float_only=True)
log1p = _unary_function("log1p", ArrayOperation.log1p, float_only=True)
exp2 = _unary_function("exp2", ArrayOperation.exp2, float_only=True)
expm1 = _unary_function("expm1", ArrayOperation.expm1, float_only=True)
sin = _unary_function("sin", ArrayOperation.sin, float_only=True)
cos = _unary_function("cos", ArrayOperation.cos, float_only=True)
fabs = _unary_function("fabs", ArrayOperation.abs, float_only=True)
absolute = _unary_function("absolute", ArrayOperation.abs)
abs = absolute
floor = _unary_function("floor", ArrayOperation.floor)
ceil = _unary_function("ceil", ArrayOperation.ceil)
rint = _unary_function("rint", ArrayOperation.rint)
sign = _unary_function("sign", ArrayOperation.sign)
isnan = _unary_function("isnan", ArrayOperation.isnan)
isinf = _unary_function("isinf", ArrayOperation.isinf)
isfinite = _unary_function("isfinite", ArrayOperation.isfinite)


def log(x, out=None, base=None):
    """Elementwise natural log (``base`` as in the old charmnumeric ``Array.log``)."""

    if _is_scalar_like(x):
        value = np.log(_coerce_scalar(x))
        return value if base is None else value / np.log(base)
    return _with_out(_log(x, base), out)


def trunc(x, out=None):
    if _is_scalar_like(x):
        return np.trunc(_coerce_scalar(x))
    return _with_out(_trunc(x), out)


fix = trunc


def round(a, decimals=0, out=None):
    """Round half to even (``numpy.round``)."""

    if _is_scalar_like(a):
        return np.round(_coerce_scalar(a), decimals)
    return _with_out(_round(a, decimals), out)


around = round


def square(x, out=None):
    if _is_scalar_like(x):
        return np.square(_coerce_scalar(x))
    x = asarray(x)
    return _with_out(x * x, out)


def reciprocal(x, out=None):
    if _is_scalar_like(x):
        return np.reciprocal(_coerce_scalar(x))
    return _with_out(_reciprocal(x), out)


def negative(x, out=None):
    if _is_scalar_like(x):
        return -_coerce_scalar(x)
    return _with_out(-asarray(x), out)


def positive(x, out=None):
    if _is_scalar_like(x):
        return +_coerce_scalar(x)
    return _with_out(asarray(x).copy(), out)


def _binary_function(name, operation, np_name=None):
    np_func = getattr(np, np_name or name)

    def func(x1, x2, out=None):
        if _all_scalars(x1, x2):
            return np_func(_coerce_scalar(x1), _coerce_scalar(x2))
        return _with_out(_ew_binary(x1, x2, operation, name), out)

    func.__name__ = name
    func.__qualname__ = name
    func.__doc__ = f"Elementwise ``numpy.{np_name or name}``."
    return func


maximum = _binary_function("maximum", ArrayOperation.maximum)
minimum = _binary_function("minimum", ArrayOperation.minimum)
fmax = _binary_function("fmax", ArrayOperation.fmax)
fmin = _binary_function("fmin", ArrayOperation.fmin)
remainder = _binary_function("remainder", ArrayOperation.mod)
mod = remainder
floor_divide = _binary_function("floor_divide", ArrayOperation.floor_divide)


def power(x1, x2, out=None):
    if _all_scalars(x1, x2):
        return np.power(_coerce_scalar(x1), _coerce_scalar(x2))
    return _with_out(_power(x1, x2), out)


pow = power


def divmod(x1, x2):
    return floor_divide(x1, x2), remainder(x1, x2)


def _comparison_function(name):
    np_func = getattr(np, name)

    def func(x1, x2, out=None):
        if _all_scalars(x1, x2):
            return np_func(_coerce_scalar(x1), _coerce_scalar(x2))
        return _with_out(_compare(x1, x2, name), out)

    func.__name__ = name
    func.__qualname__ = name
    func.__doc__ = (
        f"Elementwise ``numpy.{name}`` as a 0/1 mask in the operands' dtype."
    )
    return func


less = _comparison_function("less")
less_equal = _comparison_function("less_equal")
greater = _comparison_function("greater")
greater_equal = _comparison_function("greater_equal")
equal = _comparison_function("equal")
not_equal = _comparison_function("not_equal")


def logical_not(x, out=None):
    return _with_out(_logical_not(x), out)


def logical_and(x1, x2, out=None):
    return _with_out(_logical_and(x1, x2), out)


def logical_or(x1, x2, out=None):
    return _with_out(_logical_or(x1, x2), out)


def logical_xor(x1, x2, out=None):
    return _with_out(_logical_xor(x1, x2), out)


def where(condition, x=None, y=None):
    """``numpy.where(condition, x, y)``: nonzero (or NaN) condition selects x.

    The condition is usually a 0/1 mask from a comparison and must have the
    same dtype as array operands x/y (one element type per kernel).
    """

    if x is None and y is None:
        raise NotImplementedError("where(condition) without x and y (nonzero) is not supported")
    if x is None or y is None:
        raise ValueError("either both or neither of x and y should be given")
    if _all_scalars(condition, x, y):
        return np.where(_coerce_scalar(condition), _coerce_scalar(x), _coerce_scalar(y))
    return _where(condition, x, y)


def clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None):
    """``numpy.clip``: ``minimum(maximum(a, a_min), a_max)`` (NaN-propagating)."""

    if min is not None:
        a_min = min
    if max is not None:
        a_max = max
    if _is_scalar_like(a) and _is_scalar_like(a_min) and _is_scalar_like(a_max):
        return np.clip(_coerce_scalar(a), a_min, a_max)
    return _with_out(_clip(a, a_min, a_max), out)


__all__ = [
    "abs",
    "absolute",
    "add",
    "arange",
    "around",
    "array",
    "asarray",
    "ceil",
    "clip",
    "copy",
    "copyto",
    "cos",
    "create_array",
    "diag",
    "divide",
    "divmod",
    "dot",
    "empty",
    "empty_like",
    "equal",
    "exp",
    "exp2",
    "expm1",
    "eye",
    "fabs",
    "fix",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "full",
    "full_like",
    "greater",
    "greater_equal",
    "identity",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "negative",
    "norm2",
    "not_equal",
    "ones",
    "ones_like",
    "positive",
    "pow",
    "power",
    "reciprocal",
    "remainder",
    "rint",
    "round",
    "sign",
    "sin",
    "sqrt",
    "square",
    "subtract",
    "tanh",
    "tile",
    "trunc",
    "where",
    "zeros",
    "zeros_like",
]
