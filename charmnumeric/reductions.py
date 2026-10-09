"""Native reductions: sum, prod, mean, var, std, max, min, argmax, argmin,
count_nonzero, all, any, bincount and norm.

Every reduction is one REDUCTION (or BINCOUNT) node executed by
``src/executor_reduction.cpp``. A full reduction (``axis=None``) returns a
``(1,)`` array, like ``dot``, so it can feed later expressions as a broadcast
operand; ``keepdims=True`` returns ``(1,) * ndim`` instead. Axis reductions
return an ``(ndim - len(axis))``-D array (``keepdims`` keeps the reduced axes
with size 1). Views and slices reduce only their region.

Deviations from NumPy:

- full reductions return a ``(1,)`` array instead of a scalar;
- ``all``/``any`` return 0/1 in the input dtype (there is no bool dtype);
- ``bincount`` needs ``minlength`` (or ``interface=`` to compute ``max(x) + 1``
  with a native max reduction first); values outside ``[0, length)`` are
  ignored.
"""

import builtins
import numbers

import numpy as np

from charmtyles.operation import nonfusible_op

from .charmnumeric import (
    Array,
    ArrayOperation,
    ArrayView,
    _is_scalar_like,
    _normalize_dtype,
    asarray,
)

# Reduction kinds. Must match enum RedOp in src/executor_reduction.cpp.
_RED_SUM = 0
_RED_PROD = 1
_RED_MAX = 2
_RED_MIN = 3
_RED_ARGMAX = 4
_RED_ARGMIN = 5
_RED_COUNT_NONZERO = 6
_RED_ALL = 7
_RED_ANY = 8
_RED_MEAN = 9
_RED_VAR = 10
_RED_STD = 11
_RED_NORM2 = 12
_RED_NORM1 = 13
_RED_NORM_INF = 14
_RED_NORM_NEGINF = 15

_INT_DTYPES = (np.dtype("int32"), np.dtype("int64"))
_WIRE_DTYPE = {
    np.dtype("float32"): 0,
    np.dtype("float64"): 1,
    np.dtype("int32"): 2,
    np.dtype("int64"): 3,
}


def _normalize_axes(axis, ndim, single=False):
    """Return the sorted tuple of reduced axes (all axes for ``None``)."""

    if axis is None:
        return tuple(range(ndim))
    if single:
        if not isinstance(axis, numbers.Integral):
            raise TypeError(f"'{type(axis).__name__}' object cannot be interpreted as an integer")
        axes = (axis,)
    elif isinstance(axis, numbers.Integral):
        axes = (axis,)
    else:
        axes = tuple(axis)

    normalized = []
    for ax in axes:
        ax = int(ax)
        if ax < -ndim or ax >= ndim:
            raise np.exceptions.AxisError(ax, ndim)
        ax %= ndim
        if ax in normalized:
            raise ValueError("duplicate value in 'axis'")
        normalized.append(ax)
    return tuple(sorted(normalized))


def _result_shape(shape, axes, keepdims):
    out = []
    for d, n in enumerate(shape):
        if d in axes:
            if keepdims:
                out.append(1)
        else:
            out.append(n)
    return tuple(out) if out else (1,)


def _float_dtype(dtype):
    return np.dtype("float64") if dtype in _INT_DTYPES else dtype


def _default_out_dtype(op, dtype):
    if op in (_RED_SUM, _RED_PROD):
        # NumPy accumulates int32 in the platform int (int64).
        return np.dtype("int64") if dtype in _INT_DTYPES else dtype
    if op in (_RED_ARGMAX, _RED_ARGMIN, _RED_COUNT_NONZERO):
        return np.dtype("int64")
    if op in (_RED_MEAN, _RED_VAR, _RED_STD, _RED_NORM2, _RED_NORM1, _RED_NORM_INF,
              _RED_NORM_NEGINF):
        return _float_dtype(dtype)
    return dtype  # max, min, all, any


def _write_out(result, out):
    if out is None:
        return result
    if not isinstance(out, (Array, ArrayView)):
        raise TypeError("out must be a charmnumeric Array or ArrayView")
    if tuple(out.shape) != tuple(result.shape):
        raise ValueError(f"out has shape {out.shape}, expected {result.shape}")
    out[...] = result
    return out


def _reduce(a, op, axis=None, keepdims=False, dtype=None, out=None, ddof=0, single=False):
    a = asarray(a)
    axes = _normalize_axes(axis, a.ndims, single=single)
    if a.size == 0:
        raise ValueError("zero-size arrays are not supported by charmnumeric reductions")
    if op in (_RED_ARGMAX, _RED_ARGMIN) and len(axes) not in (1, a.ndims):
        raise ValueError("argmax/argmin reduce over one axis or all axes")

    out_dtype = _default_out_dtype(op, a.dtype)
    if dtype is not None:
        out_dtype = _normalize_dtype(dtype)

    mask = 0
    for ax in axes:
        mask |= 1 << ax
    result = nonfusible_op(
        a,
        op,
        mask,
        1 if keepdims else 0,
        int(ddof),
        _WIRE_DTYPE[a.dtype],
        operation=ArrayOperation.reduction,
        shape=_result_shape(a.shape, axes, keepdims),
        dtype=out_dtype,
    )
    return _write_out(result, out)


def _scalar_reduce(np_fn, a, *args, **kwargs):
    return np_fn(np.asarray(a), *args, **kwargs)


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if _is_scalar_like(a):
        return _scalar_reduce(np.sum, a, axis=axis, dtype=dtype, keepdims=keepdims)
    return _reduce(a, _RED_SUM, axis, keepdims, dtype, out)


def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    if _is_scalar_like(a):
        return _scalar_reduce(np.prod, a, axis=axis, dtype=dtype, keepdims=keepdims)
    return _reduce(a, _RED_PROD, axis, keepdims, dtype, out)


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    if _is_scalar_like(a):
        return _scalar_reduce(np.mean, a, axis=axis, dtype=dtype, keepdims=keepdims)
    return _reduce(a, _RED_MEAN, axis, keepdims, dtype, out)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if _is_scalar_like(a):
        return _scalar_reduce(np.var, a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return _reduce(a, _RED_VAR, axis, keepdims, dtype, out, ddof=ddof)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if _is_scalar_like(a):
        return _scalar_reduce(np.std, a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return _reduce(a, _RED_STD, axis, keepdims, dtype, out, ddof=ddof)


def amax(a, axis=None, out=None, keepdims=False):
    if _is_scalar_like(a):
        return _scalar_reduce(np.amax, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_MAX, axis, keepdims, None, out)


def amin(a, axis=None, out=None, keepdims=False):
    if _is_scalar_like(a):
        return _scalar_reduce(np.amin, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_MIN, axis, keepdims, None, out)


max = amax
min = amin


def argmax(a, axis=None, out=None, keepdims=False, *, dtype=None):
    """Index of the first maximum (flat index into ``a`` when ``axis=None``).

    ``dtype`` is a charmnumeric extension (there is no astype yet): the
    indices are produced directly in that dtype instead of int64.
    """

    if _is_scalar_like(a):
        return _scalar_reduce(np.argmax, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_ARGMAX, axis, keepdims, dtype, out, single=True)


def argmin(a, axis=None, out=None, keepdims=False, *, dtype=None):
    """Index of the first minimum (flat index into ``a`` when ``axis=None``).

    ``dtype`` is a charmnumeric extension (there is no astype yet): the
    indices are produced directly in that dtype instead of int64.
    """

    if _is_scalar_like(a):
        return _scalar_reduce(np.argmin, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_ARGMIN, axis, keepdims, dtype, out, single=True)


def count_nonzero(a, axis=None, keepdims=False):
    if _is_scalar_like(a):
        return _scalar_reduce(np.count_nonzero, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_COUNT_NONZERO, axis, keepdims)


def all(a, axis=None, out=None, keepdims=False):
    """1 where every element is nonzero, else 0, in the dtype of ``a``."""

    if _is_scalar_like(a):
        return _scalar_reduce(np.all, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_ALL, axis, keepdims, None, out)


def any(a, axis=None, out=None, keepdims=False):
    """1 where some element is nonzero, else 0, in the dtype of ``a``."""

    if _is_scalar_like(a):
        return _scalar_reduce(np.any, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_ANY, axis, keepdims, None, out)


def bincount(x, weights=None, minlength=0, interface=None):
    """Count occurrences of each value in a non-negative int vector.

    The output length must be known when the graph is built: it is
    ``minlength``, or ``max(minlength, max(x) + 1)`` when ``interface`` is
    given (that runs a native max reduction and fetches one value). Values
    outside ``[0, length)`` are ignored.
    """

    x = asarray(x)
    if x.ndims != 1:
        raise ValueError("object too deep for desired array: bincount requires a 1-D array")
    if x.dtype not in _INT_DTYPES:
        raise TypeError(f"Cannot cast array data from dtype('{x.dtype}') to dtype('int64')")
    minlength = int(minlength)
    if minlength < 0:
        raise ValueError("'minlength' must not be negative")
    length = minlength
    if interface is not None:
        length = builtins.max(length, int(amax(x).get(interface)[0]) + 1)
    if length <= 0:
        raise ValueError(
            "charmnumeric bincount needs minlength > 0 (or interface= to compute max(x) + 1)"
        )

    if weights is None:
        operands = (x, length, _WIRE_DTYPE[x.dtype], -1)
        out_dtype = np.dtype("int64")
    else:
        weights = asarray(weights)
        if tuple(weights.shape) != tuple(x.shape):
            raise ValueError("The weights and list don't have the same length.")
        # The executor pairs x and weights tile by tile, so give both the
        # same layout (fresh arrays are decomposed identically).
        if isinstance(x, ArrayView) or isinstance(weights, ArrayView):
            x = x.copy()
            weights = weights.copy()
        operands = (x, weights, length, _WIRE_DTYPE[x.dtype], _WIRE_DTYPE[weights.dtype])
        out_dtype = np.dtype("float64")

    return nonfusible_op(
        *operands,
        operation=ArrayOperation.bincount,
        shape=(length,),
        dtype=out_dtype,
    )


def norm(x, ord=None, axis=None, keepdims=False):
    """Vector and matrix norms (``numpy.linalg.norm`` subset).

    Vector norms: ord in {None, 2, 1, inf, -inf, 0}. Matrix norms (2-D, or a
    2-tuple axis): ord in {None, 'fro', inf, -inf, 1, -1}.
    """

    x = asarray(x)
    if axis is None:
        if ord is None:
            return _reduce(x, _RED_NORM2, None, keepdims)
        axes = tuple(range(x.ndims))
    else:
        axes = _normalize_axes(axis, x.ndims)

    if len(axes) == 1:
        if ord is None or ord == 2:
            op = _RED_NORM2
        elif ord == 1:
            op = _RED_NORM1
        elif ord == np.inf:
            op = _RED_NORM_INF
        elif ord == -np.inf:
            op = _RED_NORM_NEGINF
        elif ord == 0:
            return _reduce(x, _RED_COUNT_NONZERO, axes, keepdims, dtype=_float_dtype(x.dtype))
        else:
            raise NotImplementedError(f"vector norm ord={ord!r} is not supported")
        return _reduce(x, op, axes, keepdims)

    if len(axes) != 2:
        raise ValueError("Improper number of dimensions to norm.")
    if ord is None or ord == "fro":
        return _reduce(x, _RED_NORM2, axes, keepdims)
    row_ax, col_ax = axes if axis is None else _normalize_axes_ordered(axis, x.ndims)
    if ord in (np.inf, -np.inf):
        sums = _reduce(x, _RED_NORM1, (col_ax,), True)  # row sums of |x|
    elif ord in (1, -1):
        sums = _reduce(x, _RED_NORM1, (row_ax,), True)  # column sums of |x|
    else:
        raise NotImplementedError(f"matrix norm ord={ord!r} is not supported")
    op = _RED_MAX if ord in (np.inf, 1) else _RED_MIN
    return _reduce(sums, op, axes, keepdims)


def _normalize_axes_ordered(axis, ndim):
    axes = tuple(int(ax) % ndim for ax in axis)
    _normalize_axes(axes, ndim)  # range and duplicate checks
    return axes


# ---- Array / ArrayView methods ----


def _method_sum(self, axis=None, dtype=None, out=None, keepdims=False):
    return sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def _method_prod(self, axis=None, dtype=None, out=None, keepdims=False):
    return prod(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def _method_mean(self, axis=None, dtype=None, out=None, keepdims=False):
    return mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def _method_var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return var(self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def _method_std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return std(self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def _is_elementwise_other(axis):
    # The older charmnumeric method API (branch fix-temporaries) used
    # a.max(other) / a.min(other) for the elementwise maximum/minimum. An array
    # or a non-integral scalar cannot be an axis, so treat those as that form;
    # anything else follows NumPy's ndarray.max(axis=...) reduction.
    return isinstance(axis, (Array, ArrayView)) or (
        isinstance(axis, numbers.Real) and not isinstance(axis, numbers.Integral))


def _method_max(self, axis=None, out=None, keepdims=False):
    if _is_elementwise_other(axis):
        from .operations import maximum
        return maximum(self, axis)
    return amax(self, axis=axis, out=out, keepdims=keepdims)


def _method_min(self, axis=None, out=None, keepdims=False):
    if _is_elementwise_other(axis):
        from .operations import minimum
        return minimum(self, axis)
    return amin(self, axis=axis, out=out, keepdims=keepdims)


def _method_argmax(self, axis=None, out=None, keepdims=False, *, dtype=None):
    return argmax(self, axis=axis, out=out, keepdims=keepdims, dtype=dtype)


def _method_argmin(self, axis=None, out=None, keepdims=False, *, dtype=None):
    return argmin(self, axis=axis, out=out, keepdims=keepdims, dtype=dtype)


def _method_all(self, axis=None, out=None, keepdims=False):
    return all(self, axis=axis, out=out, keepdims=keepdims)


def _method_any(self, axis=None, out=None, keepdims=False):
    return any(self, axis=axis, out=out, keepdims=keepdims)


for _cls in (Array, ArrayView):
    _cls.sum = _method_sum
    _cls.prod = _method_prod
    _cls.mean = _method_mean
    _cls.var = _method_var
    _cls.std = _method_std
    _cls.max = _method_max
    _cls.min = _method_min
    _cls.argmax = _method_argmax
    _cls.argmin = _method_argmin
    _cls.all = _method_all
    _cls.any = _method_any
del _cls


__all__ = [
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "bincount",
    "count_nonzero",
    "max",
    "mean",
    "min",
    "norm",
    "prod",
    "std",
    "sum",
    "var",
]
