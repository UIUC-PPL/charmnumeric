import builtins

import numpy as np

from .charmnumeric import (
    Array,
    ArrayOperation,
    ArrayView,
    _astype,
    _axis_index,
    _clip,
    _coerce_scalar,
    _compare,
    _ew,
    _ew_binary,
    _ew_unary,
    _imag,
    _is_array_nd,
    _real,
    _is_scalar_like,
    _log,
    _logical_and,
    _logical_not,
    _logical_or,
    _logical_xor,
    _materialize,
    _nd_aware,
    _normalize_dtype,
    _power,
    _reciprocal,
    _round,
    _scan,
    _sort,
    _square,
    _swapaxes,
    _transpose_view,
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
    nonfusible_op,
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
    return _write_out(_ew_unary(asarray(x), ArrayOperation.tanh, "tanh"), out)


def exp(x, out=None):
    if _is_scalar_like(x):
        return _write_out(np.exp(_coerce_scalar(x)), out)
    return _write_out(_ew_unary(asarray(x), ArrayOperation.exp, "exp"), out)


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


def _arith(operation, name, ufunc):
    def func(x1, x2, out=None):
        if _is_scalar_like(x1) and _is_scalar_like(x2):
            return _write_out(ufunc(x1, x2), out)
        # NumPy scalars stay typed (NEP 50), as with the operators
        return _write_out(_ew(operation, x1, x2, name=name, ufunc=ufunc), out)

    func.__name__ = func.__qualname__ = name
    func.__doc__ = f"Elementwise ``numpy.{name}``."
    return func


add = _arith(ArrayOperation.add, "add", np.add)
subtract = _arith(ArrayOperation.sub, "subtract", np.subtract)
multiply = _arith(ArrayOperation.mul, "multiply", np.multiply)
divide = _arith(ArrayOperation.div, "divide", np.true_divide)


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
# Elementwise math, comparisons and selection. Dtypes follow NumPy; scalar-only
# calls use NumPy directly.
# ---------------------------------------------------------------------------


def _all_scalars(*values):
    return all(_is_scalar_like(v) for v in values)


def _with_out(result, out):
    if out is None:
        return result
    out[...] = result
    return out


def _unary_function(name, operation, np_name=None):
    np_func = getattr(np, np_name or name)

    def func(x, out=None):
        if _is_scalar_like(x):
            return np_func(_coerce_scalar(x))
        return _with_out(_ew_unary(x, operation, name), out)

    func.__name__ = name
    func.__qualname__ = name
    func.__doc__ = f"Elementwise ``numpy.{np_name or name}``."
    return func


sqrt = _unary_function("sqrt", ArrayOperation.sqrt)
log2 = _unary_function("log2", ArrayOperation.log2)
log10 = _unary_function("log10", ArrayOperation.log10)
log1p = _unary_function("log1p", ArrayOperation.log1p)
exp2 = _unary_function("exp2", ArrayOperation.exp2)
expm1 = _unary_function("expm1", ArrayOperation.expm1)
sin = _unary_function("sin", ArrayOperation.sin)
cos = _unary_function("cos", ArrayOperation.cos)


def fabs(x, out=None):
    if _is_scalar_like(x):
        return np.fabs(_coerce_scalar(x))
    x = asarray(x)
    np.fabs.resolve_dtypes((x.dtype, None))  # TypeError for complex
    dtype = np.fabs(np.zeros(1, x.dtype)).dtype
    return _with_out(_ew_unary(x, ArrayOperation.abs, "fabs").astype(dtype, copy=False), out)


absolute = _unary_function("absolute", ArrayOperation.abs)
abs = absolute
floor = _unary_function("floor", ArrayOperation.floor)
ceil = _unary_function("ceil", ArrayOperation.ceil)
rint = _unary_function("rint", ArrayOperation.rint)
sign = _unary_function("sign", ArrayOperation.sign)
isnan = _unary_function("isnan", ArrayOperation.isnan)
isinf = _unary_function("isinf", ArrayOperation.isinf)
isfinite = _unary_function("isfinite", ArrayOperation.isfinite)
conjugate = _unary_function("conjugate", ArrayOperation.conj)
conj = conjugate
invert = _unary_function("invert", ArrayOperation.invert)
bitwise_not = invert


def real(val):
    """Real part (``numpy.real``)."""
    if _is_scalar_like(val):
        return np.real(_coerce_scalar(val))
    return _real(asarray(val))


def imag(val):
    """Imaginary part (``numpy.imag``)."""
    if _is_scalar_like(val):
        return np.imag(_coerce_scalar(val))
    return _imag(asarray(val))


def iscomplexobj(x):
    return np.dtype(getattr(x, "dtype", np.asarray(x).dtype)).kind == "c"


def isrealobj(x):
    return not iscomplexobj(x)


def log(x, out=None, base=None):
    """Elementwise natural log; ``base`` sets the logarithm base."""

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
    return _with_out(_square(asarray(x)), out)


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
bitwise_and = _binary_function("bitwise_and", ArrayOperation.bitwise_and)
bitwise_or = _binary_function("bitwise_or", ArrayOperation.bitwise_or)
bitwise_xor = _binary_function("bitwise_xor", ArrayOperation.bitwise_xor)
left_shift = _binary_function("left_shift", ArrayOperation.left_shift)
right_shift = _binary_function("right_shift", ArrayOperation.right_shift)
bitwise_left_shift = left_shift
bitwise_right_shift = right_shift


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
    func.__doc__ = f"Elementwise ``numpy.{name}`` (bool result)."
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
    """``numpy.where(condition, x, y)``: nonzero (or NaN) condition selects x."""

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


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """``numpy.isclose``: ``|a - b| <= atol + rtol * |b|`` (bool array)."""
    if _all_scalars(a, b):
        return np.isclose(_coerce_scalar(a), _coerce_scalar(b), rtol=rtol, atol=atol,
                          equal_nan=equal_nan)
    x, y = _coerce_binary_operand(a), _coerce_binary_operand(b)
    # as NumPy: b in its inexact type, so integer differences cannot overflow
    if isinstance(y, (Array, ArrayView)):
        y = _astype(y, np.result_type(y.dtype, 1.0))
    elif isinstance(y, int):
        y = float(y)
    result = (_compare(absolute(x - y), atol + rtol * absolute(y), "less_equal")
              & isfinite(y)) | _compare(x, y, "equal")
    if equal_nan:
        result = result | (isnan(x) & isnan(y))
    return result


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """``numpy.allclose``, as a 0-d bool array (``bool()`` fetches it)."""
    from .reductions import all as _all

    result = isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if _is_scalar_like(result):
        return bool(result)
    return _all(result)


def transpose(a, axes=None):
    """``numpy.transpose``; a copy linked to ``a`` like a view."""

    return _transpose_view(asarray(a), axes)


def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array (``numpy.swapaxes``)."""

    return _swapaxes(asarray(a), axis1, axis2)


def _axis_key(ndim, axis, item):
    key = [slice(None)] * ndim
    key[axis] = item
    return tuple(key)


def _join_inputs(arrays, dtype):
    """Inputs cast to their common dtype (``numpy.result_type``)."""

    if _is_array_nd(arrays):
        raise TypeError("expected a sequence of arrays, not a single charmnumeric array")
    pieces = [
        a if _is_array_nd(a) else np.asarray(a) for a in arrays
    ]
    if not pieces:
        raise ValueError("need at least one array to concatenate")

    if dtype is None:
        dtype = np.result_type(*[a.dtype for a in pieces])
    dtype = _normalize_dtype(dtype)

    return [a.astype(dtype, copy=False) for a in pieces], dtype


def _join(pieces, shapes, axis, dtype, out):
    """One slice assignment per piece; ``shapes`` may add size-1 dims (vstack, stack)."""

    ndim = len(shapes[0])
    if any(len(shape) != ndim for shape in shapes):
        raise ValueError(
            "all the input arrays must have same number of dimensions, but got "
            + ", ".join(str(len(shape)) for shape in shapes)
        )
    axis = _axis_index(axis, ndim)
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
        pieces = [p.ravel() for p in pieces]
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
    axis = _axis_index(axis, len(shape) + 1)
    logical = shape[:axis] + (1,) + shape[axis:]
    return _join(pieces, [logical] * len(pieces), axis, dtype, out)


def astype(x, dtype, /, *, copy=True):
    """Copy of ``x`` converted to ``dtype`` (NumPy astype semantics)."""
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


# >3-D (folded) operands for every elementwise function
for _name in (
    "tanh", "exp", "add", "subtract", "multiply", "divide", "sqrt", "log2", "log10", "log1p",
    "exp2", "expm1", "sin", "cos", "fabs", "absolute", "abs", "floor", "ceil", "rint", "sign",
    "isnan", "isinf", "isfinite", "log", "trunc", "fix", "round", "around", "square",
    "reciprocal", "negative", "positive", "maximum", "minimum", "fmax", "fmin", "remainder",
    "mod", "floor_divide", "power", "pow", "divmod", "less", "less_equal", "greater",
    "greater_equal", "equal", "not_equal", "logical_not", "logical_and", "logical_or",
    "logical_xor", "where", "clip", "left_shift", "right_shift", "bitwise_left_shift",
    "bitwise_right_shift",
):
    globals()[_name] = _nd_aware(globals()[_name])
del _name
true_divide = divide


__all__ = [
    "allclose",
    "isclose",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "left_shift",
    "right_shift",
    "conj",
    "conjugate",
    "imag",
    "invert",
    "iscomplexobj",
    "isrealobj",
    "real",
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
    "tanh",
    "tile",
    "transpose",
    "true_divide",
    "trunc",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]
