import builtins

import numpy as np

from charmtyles.operation import fusible_op, nonfusible_op

from .charmnumeric import (
    Array,
    ArrayOperation,
    ArrayView,
    _astype,
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
    _materialize,
    _normalize_dtype,
    _power,
    _reciprocal,
    _reshape_copy,
    _round,
    _scan,
    _sort,
    _swapaxes,
    _transpose,
    _trunc,
    _where,
    _write_out,
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


def tanh(x, out=None):
    if _is_scalar_like(x):
        return _write_out(np.tanh(_coerce_scalar(x)), out)
    x = asarray(x)
    return _write_out(
        fusible_op(x, operation=ArrayOperation.tanh, shape=x.shape, dtype=x.dtype), out
    )


def exp(x, out=None):
    if _is_scalar_like(x):
        return _write_out(np.exp(_coerce_scalar(x)), out)
    x = asarray(x)
    return _write_out(
        fusible_op(x, operation=ArrayOperation.exp, shape=x.shape, dtype=x.dtype), out
    )


def tile(x, reps):
    if _is_scalar_like(x):
        return array(np.tile(_coerce_scalar(x), reps))

    x = _materialize(asarray(x))
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


def add(x1, x2, out=None):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _write_out(_coerce_scalar(x1) + _coerce_scalar(x2), out)
    return _write_out(_coerce_binary_operand(x1) + _coerce_binary_operand(x2), out)


def subtract(x1, x2, out=None):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _write_out(_coerce_scalar(x1) - _coerce_scalar(x2), out)
    return _write_out(_coerce_binary_operand(x1) - _coerce_binary_operand(x2), out)


def multiply(x1, x2, out=None):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _write_out(_coerce_scalar(x1) * _coerce_scalar(x2), out)
    return _write_out(_coerce_binary_operand(x1) * _coerce_binary_operand(x2), out)


def divide(x1, x2, out=None):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _write_out(_coerce_scalar(x1) / _coerce_scalar(x2), out)
    return _write_out(_coerce_binary_operand(x1) / _coerce_binary_operand(x2), out)


def matmul(x1, x2, out=None):
    if _is_scalar_like(x1) or _is_scalar_like(x2):
        raise ValueError("matmul does not support scalar operands")
    return _write_out(asarray(x1) @ asarray(x2), out)


def dot(x1, x2, out=None):
    if _is_scalar_like(x1) and _is_scalar_like(x2):
        return _write_out(_coerce_scalar(x1) * _coerce_scalar(x2), out)

    lhs = _coerce_binary_operand(x1)
    rhs = _coerce_binary_operand(x2)
    if _is_scalar_like(lhs) or _is_scalar_like(rhs):
        return _write_out(lhs * rhs, out)
    return lhs.dot(rhs, out=out)


def norm2(x):
    """Compute the squared L2 norm of a 1D array."""

    x = asarray(x)
    if x.ndims != 1:
        raise ValueError(f"norm2 requires a 1D array, got {x.ndims}D")
    return x.dot(x)


def diag(v, k=0):
    """Construct a diagonal matrix or extract a diagonal."""

    v = _materialize(asarray(v))

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


def transpose(a, axes=None):
    """``numpy.transpose`` for 1-3D arrays; returns a new array, not a view."""

    return _transpose(asarray(a), axes)


def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array (``numpy.swapaxes``)."""

    return _swapaxes(asarray(a), axis1, axis2)


def _normalize_axis(axis, ndim):
    axis = int(axis)
    if not -ndim <= axis < ndim:
        raise np.exceptions.AxisError(axis, ndim)
    return axis % ndim


def _axis_key(ndim, axis, item):
    key = [slice(None)] * ndim
    key[axis] = item
    return tuple(key)


def _join_inputs(arrays, dtype):
    """Inputs as device arrays / host arrays of one dtype (device dtype wins)."""

    if isinstance(arrays, (Array, ArrayView)):
        raise TypeError("expected a sequence of arrays, not a single charmnumeric array")
    pieces = [
        a if isinstance(a, (Array, ArrayView)) else np.asarray(a) for a in arrays
    ]
    if not pieces:
        raise ValueError("need at least one array to concatenate")

    if dtype is None:
        device = [a.dtype for a in pieces if isinstance(a, (Array, ArrayView))]
        dtype = np.result_type(*(device or [a.dtype for a in pieces]))
    dtype = _normalize_dtype(dtype)

    result = []
    for a in pieces:
        if isinstance(a, (Array, ArrayView)):
            if a.dtype != dtype:
                raise TypeError(
                    f"cannot join a {a.dtype} array into a {dtype} result "
                    "(dtype conversion is not implemented yet)"
                )
            result.append(a)
        else:
            result.append(a.astype(dtype, copy=False))
    return result, dtype


def _join(pieces, shapes, axis, dtype, out):
    """One slice assignment per piece; ``shapes`` may add size-1 dims (vstack, stack)."""

    ndim = len(shapes[0])
    if any(len(shape) != ndim for shape in shapes):
        raise ValueError(
            "all the input arrays must have same number of dimensions, but got "
            + ", ".join(str(len(shape)) for shape in shapes)
        )
    axis = _normalize_axis(axis, ndim)
    for shape in shapes[1:]:
        for d in range(ndim):
            if d != axis and shape[d] != shapes[0][d]:
                raise ValueError(
                    "all the input array dimensions except for the concatenation axis "
                    f"must match exactly, but along dimension {d} the arrays have sizes "
                    f"{shapes[0][d]} and {shape[d]}"
                )

    total = sum(shape[axis] for shape in shapes)
    result_shape = shapes[0][:axis] + (total,) + shapes[0][axis + 1:]
    if len(result_shape) > 3:
        raise NotImplementedError(
            f"charmnumeric arrays must have 1 to 3 dimensions, got shape {result_shape}"
        )

    # Write straight into out when no input aliases it.
    direct = (
        out is not None
        and isinstance(out, (Array, ArrayView))
        and tuple(out.shape) == result_shape
        and out.dtype == dtype
        and not any(
            isinstance(p, (Array, ArrayView)) and p.name == out.name for p in pieces
        )
    )
    result = out if direct else empty(result_shape, dtype=dtype)

    offset = 0
    for piece, shape in zip(pieces, shapes):
        n = shape[axis]
        if n == 0:
            continue
        result[_axis_key(ndim, axis, slice(offset, offset + n))] = piece
        offset += n
    if direct:
        return out
    return _write_out(result, out)


def _atleast_shape(a, ndmin):
    shape = tuple(a.shape)
    if len(shape) >= ndmin:
        return shape
    if ndmin == 2 and len(shape) == 1:
        return (1,) + shape
    return (1,) * (ndmin - len(shape)) + shape


def concatenate(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    """Join arrays along an existing axis (slice assignment into a new array)."""

    if out is not None and dtype is not None:
        raise TypeError("concatenate() only takes `out` or `dtype` as an argument, but both were provided.")
    if dtype is None and isinstance(out, (Array, ArrayView)):
        dtype = out.dtype
    pieces, dtype = _join_inputs(arrays, dtype)
    if axis is None:
        if any(np.ndim(p) != 1 for p in pieces):
            raise NotImplementedError("concatenate(axis=None) of N-D arrays needs reshape (flatten)")
        axis = 0
    if any(np.ndim(p) == 0 for p in pieces):
        raise ValueError("zero-dimensional arrays cannot be concatenated")
    return _join(pieces, [tuple(p.shape) for p in pieces], axis, dtype, out)


def hstack(tup, *, dtype=None, casting="same_kind"):
    """Stack arrays column-wise (axis 0 for 1-D inputs, axis 1 otherwise)."""

    pieces, dtype = _join_inputs(tup, dtype)
    shapes = [_atleast_shape(p, 1) for p in pieces]
    axis = 0 if all(len(shape) == 1 for shape in shapes) else 1
    return _join(pieces, shapes, axis, dtype, None)


def vstack(tup, *, dtype=None, casting="same_kind"):
    """Stack arrays row-wise; 1-D inputs of length N become (1, N) rows."""

    pieces, dtype = _join_inputs(tup, dtype)
    shapes = [_atleast_shape(p, 2) for p in pieces]
    return _join(pieces, shapes, 0, dtype, None)


def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    """Join same-shape arrays along a new axis (result must be at most 3-D)."""

    if dtype is None and isinstance(out, (Array, ArrayView)):
        dtype = out.dtype
    pieces, dtype = _join_inputs(arrays, dtype)
    shape = tuple(pieces[0].shape)
    if any(tuple(p.shape) != shape for p in pieces):
        raise ValueError("all input arrays must have the same shape")
    axis = _normalize_axis(axis, len(shape) + 1)
    logical = shape[:axis] + (1,) + shape[axis:]
    return _join(pieces, [logical] * len(pieces), axis, dtype, out)


def _index_runs(flat):
    """(dst, src, length, step) slice copies: step-1 runs, or constant-step runs of >= 3."""

    runs = []
    count = len(flat)
    j = 0
    while j < count:
        if j + 1 < count:
            step = int(flat[j + 1] - flat[j])
            if step >= 1:
                k = j + 1
                while k + 1 < count and int(flat[k + 1] - flat[k]) == step:
                    k += 1
                length = k - j + 1
                if step == 1 or length >= 3:
                    runs.append((j, int(flat[j]), length, step))
                    j = k + 1
                    continue
        runs.append((j, int(flat[j]), 1, 1))
        j += 1
    return runs


def _take_1d(a, flat, axis):
    """a.take(flat, axis) for 1-D host indices: one slice copy per index run."""

    ndim = a.ndim
    result = empty(a.shape[:axis] + (len(flat),) + a.shape[axis + 1:], dtype=a.dtype)
    for dst, src, length, step in _index_runs(flat):
        stop = src + (length - 1) * step + 1
        result[_axis_key(ndim, axis, slice(dst, dst + length))] = a[
            _axis_key(ndim, axis, slice(src, stop, step))
        ]
    return result


def take(a, indices, axis=None, out=None, mode="raise"):
    """take() with host indices, lowered to slice copies (one per index run).

    A scalar index on a 1-D array gives a (1,) array (no 0-D arrays).
    """

    if isinstance(indices, (Array, ArrayView)):
        raise NotImplementedError(
            "take with a device index array needs a native gather op; "
            "fetch the indices with .get() first"
        )
    if not isinstance(a, (Array, ArrayView)):
        a = asarray(a)

    idx = np.asarray(indices)
    if idx.size == 0:
        raise NotImplementedError("take with an empty index list (empty result) is not supported")
    if idx.dtype.kind not in "iu":
        raise IndexError("arrays used as indices must be of integer type")
    idx = idx.astype(np.int64)

    if axis is None:
        if a.ndim != 1:
            raise NotImplementedError("take with axis=None on an N-D array needs reshape (flatten)")
        axis = 0
    axis = _normalize_axis(axis, a.ndim)
    n = a.shape[axis]
    if mode == "raise":
        bad = idx[(idx < -n) | (idx >= n)]
        if bad.size:
            raise IndexError(f"index {int(bad[0])} is out of bounds for axis {axis} with size {n}")
        idx = np.where(idx < 0, idx + n, idx)
    elif mode == "wrap":
        idx = np.mod(idx, n)
    elif mode == "clip":
        idx = np.clip(idx, 0, n - 1)
    else:
        raise ValueError(f"clipmode must be one of 'clip', 'raise', or 'wrap' (got {mode!r})")

    out_shape = a.shape[:axis] + idx.shape + a.shape[axis + 1:]
    if len(out_shape) > 3:
        raise NotImplementedError(
            f"take result would have shape {out_shape}; charmnumeric supports at most 3 dims"
        )

    if idx.ndim <= 1:
        result = _take_1d(a, idx.reshape(-1), axis)
        if idx.ndim == 0 and len(out_shape) > 0:
            # drop the size-1 axis left by the scalar index
            result = _reshape_copy(result, out_shape)
        return _write_out(result, out)

    # multi-dim indices: one 1-D take per line along the longest index dim
    free = int(np.argmax(idx.shape))
    result = empty(out_shape, dtype=a.dtype)
    lead_shape = tuple(1 if d == free else idx.shape[d] for d in range(idx.ndim))
    for lead in np.ndindex(*lead_shape):
        sel = list(lead)
        sel[free] = slice(None)
        part = _take_1d(a, idx[tuple(sel)], axis)
        key = [slice(None)] * len(out_shape)
        for d in range(idx.ndim):
            if d != free:
                key[axis + d] = slice(lead[d], lead[d] + 1)
        result[tuple(key)] = part
    return _write_out(result, out)


def astype(x, dtype, /, *, copy=True):
    """Copy of ``x`` converted to ``dtype``; float -> int truncates."""
    return asarray(x).astype(dtype, copy=copy)


def cumsum(a, axis=None, dtype=None, out=None):
    """Inclusive sum along ``axis``. axis=None only for 1-D input (no reshape)."""
    return _scan("cumsum", a, axis, dtype, out)


def cumprod(a, axis=None, dtype=None, out=None):
    """Inclusive product along ``axis``. axis=None only for 1-D input."""
    return _scan("cumprod", a, axis, dtype, out)


def nancumsum(a, axis=None, dtype=None, out=None):
    """cumsum treating NaNs as zero."""
    return _scan("nancumsum", a, axis, dtype, out)


def nancumprod(a, axis=None, dtype=None, out=None):
    """cumprod treating NaNs as one."""
    return _scan("nancumprod", a, axis, dtype, out)


def sort(a, axis=-1, kind=None, order=None):
    """Sorted copy along ``axis`` (NaNs last, stable; gathers to one chare)."""
    return _sort(a, axis, kind, order)


def argsort(a, axis=-1, kind=None, order=None):
    """int64 indices that sort ``a`` along ``axis`` (stable, NaNs last)."""
    return _sort(a, axis, kind, order, arg=True)


__all__ = [
    "abs",
    "absolute",
    "add",
    "arange",
    "argsort",
    "around",
    "array",
    "asarray",
    "astype",
    "ceil",
    "clip",
    "concatenate",
    "copy",
    "copyto",
    "cos",
    "create_array",
    "cumprod",
    "cumsum",
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
    "hstack",
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
    "nancumprod",
    "nancumsum",
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
    "sort",
    "sqrt",
    "square",
    "stack",
    "subtract",
    "swapaxes",
    "take",
    "tanh",
    "tile",
    "transpose",
    "trunc",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]
