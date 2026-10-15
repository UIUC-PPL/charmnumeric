"""Native reductions: sum, prod, mean, var, std, max, min, argmax, argmin,
count_nonzero, all, any, bincount and norm.

Each reduction is one REDUCTION (or BINCOUNT) node run by
``src/executor_reduction.cpp``. ``axis=None`` returns a 0-d device array (NumPy
returns a host scalar), so it can feed later expressions without a sync.
Views reduce only their region.

Deviations from NumPy: ``bincount`` needs ``minlength`` or ``interface=``
(which runs a native max first); values outside ``[0, length)`` are ignored.
"""

import builtins
import numbers

import numpy as np


from .charmnumeric import (
    Array,
    ArrayOperation,
    ArrayView,
    DType,
    _array_abs,
    _NDBase,
    _imag,
    _is_array_nd,
    _squeeze_shape,
    _is_scalar_like,
    _normalize_dtype,
    _real,
    asarray,
    nonfusible_op,
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

_NORMS = (_RED_NORM2, _RED_NORM1, _RED_NORM_INF, _RED_NORM_NEGINF)


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
    return tuple(out)


def _float_dtype(dtype):
    """Result dtype of a norm (float64 for ints, real for complex)."""
    return np.linalg.norm(np.zeros(1, dtype)).dtype


def _default_out_dtype(op, dtype):
    """NumPy's result dtype of the reduction (sum of int8 is int64, etc.)."""
    z = np.zeros(1, dtype)
    if op == _RED_SUM:
        return np.sum(z).dtype
    if op == _RED_PROD:
        return np.prod(z).dtype
    if op in (_RED_ARGMAX, _RED_ARGMIN, _RED_COUNT_NONZERO):
        return np.dtype("int64")
    if op == _RED_MEAN:
        return np.mean(z).dtype
    if op in (_RED_VAR, _RED_STD):
        return np.var(z).dtype
    if op in _NORMS:
        return _float_dtype(dtype)
    if op in (_RED_ALL, _RED_ANY):
        return np.dtype(bool)
    return dtype  # max, min


def _write_out(result, out):
    if out is None:
        return result
    if not _is_array_nd(out):
        raise TypeError("out must be a charmnumeric Array or ArrayView")
    if tuple(out.shape) != tuple(result.shape):
        raise ValueError(f"out has shape {out.shape}, expected {result.shape}")
    out[...] = result
    return out


def _reduce(a, op, axis=None, keepdims=False, dtype=None, out=None, ddof=0, single=False):
    a = asarray(a)
    if isinstance(a, _NDBase):
        from .folded import reduce_nd

        return _write_out(reduce_nd(a, op, axis, keepdims, dtype, ddof, single), out)
    axes = _normalize_axes(axis, a.ndims, single=single)
    if a.size == 0:
        raise ValueError("zero-size arrays are not supported by charmnumeric reductions")
    if op in (_RED_ARGMAX, _RED_ARGMIN) and len(axes) not in (1, a.ndims):
        raise ValueError("argmax/argmin reduce over one axis or all axes")

    if a.dtype.kind == "c" and (op in _NORMS or op in (_RED_VAR, _RED_STD)):
        # complex: norms of |a|; var(a) = var(a.real) + var(a.imag)
        if op in _NORMS:
            result = _reduce(_array_abs(a), op, axis, keepdims, dtype, None, ddof, single)
        else:
            result = (_reduce(_real(a), _RED_VAR, axis, keepdims, dtype, None, ddof, single)
                      + _reduce(_imag(a), _RED_VAR, axis, keepdims, dtype, None, ddof, single))
            if op == _RED_STD:
                result = result.sqrt()
        return _write_out(result, out)

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
        DType.from_numpy(a.dtype),
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
    """Index of the first maximum (flat index when ``axis=None``).

    ``dtype`` is a charmnumeric extension giving the index dtype (default int64).
    """

    if _is_scalar_like(a):
        return _scalar_reduce(np.argmax, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_ARGMAX, axis, keepdims, dtype, out, single=True)


def argmin(a, axis=None, out=None, keepdims=False, *, dtype=None):
    """Index of the first minimum (flat index when ``axis=None``).

    ``dtype`` is a charmnumeric extension giving the index dtype (default int64).
    """

    if _is_scalar_like(a):
        return _scalar_reduce(np.argmin, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_ARGMIN, axis, keepdims, dtype, out, single=True)


def count_nonzero(a, axis=None, keepdims=False):
    if _is_scalar_like(a):
        return _scalar_reduce(np.count_nonzero, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_COUNT_NONZERO, axis, keepdims)


def all(a, axis=None, out=None, keepdims=False):
    """True where every element is nonzero."""

    if _is_scalar_like(a):
        return _scalar_reduce(np.all, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_ALL, axis, keepdims, None, out)


def any(a, axis=None, out=None, keepdims=False):
    """True where some element is nonzero."""

    if _is_scalar_like(a):
        return _scalar_reduce(np.any, a, axis=axis, keepdims=keepdims)
    return _reduce(a, _RED_ANY, axis, keepdims, None, out)


def bincount(x, weights=None, minlength=0, interface=None):
    """Count occurrences of each value in a non-negative int vector.

    Length is ``minlength``, or ``max(minlength, max(x) + 1)`` when ``interface``
    is given. Values outside ``[0, length)`` are ignored.
    """

    x = asarray(x)
    if x.ndims != 1:
        raise ValueError("object too deep for desired array: bincount requires a 1-D array")
    if x.dtype.kind not in "biu":
        raise TypeError(f"Cannot cast array data from dtype('{x.dtype}') to dtype('int64')")
    minlength = int(minlength)
    if minlength < 0:
        raise ValueError("'minlength' must not be negative")
    length = minlength
    if interface is not None:
        length = builtins.max(length, int(amax(x).get(interface)) + 1)
    if length <= 0:
        raise ValueError(
            "charmnumeric bincount needs minlength > 0 (or interface= to compute max(x) + 1)"
        )

    if weights is None:
        operands = (x, length, DType.from_numpy(x.dtype), -1)
        out_dtype = np.dtype("int64")
    else:
        weights = asarray(weights)
        if tuple(weights.shape) != tuple(x.shape):
            if _squeeze_shape(weights.shape) != _squeeze_shape(x.shape):
                raise ValueError("The weights and list don't have the same length.")
            weights = weights.reshape(x.shape)
        # The executor pairs x and weights tile by tile, so both need one layout.
        if isinstance(x, ArrayView) or isinstance(weights, ArrayView):
            x = x.copy()
            weights = weights.copy()
        if weights.dtype.kind == "c":
            raise TypeError("bincount: complex weights are not supported")
        operands = (x, weights, length, DType.from_numpy(x.dtype),
                    DType.from_numpy(weights.dtype))
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


def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear",
             keepdims=False, *, weights=None, interpolation=None):
    """``numpy.quantile``: native sort, then NumPy's interpolation of order statistics.

    ``q`` is host data (device arrays are fetched); ``axis`` is None or an int.
    """
    from numpy.lib import _function_base_impl as npq

    from .charmnumeric import _reshape, _sort, _transpose, _where, _write_out
    from .indexing import take

    if weights is not None:
        raise NotImplementedError("quantile(weights=...) is not supported")
    if interpolation is not None:
        method = interpolation
    a = asarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")
    weak_q = isinstance(q, (int, float)) and not isinstance(q, bool)
    q = np.asanyarray(q)
    if q.dtype.kind not in "biuf":
        q = q.astype(np.float64)
    if not npq._quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    if q.ndim > 2:
        raise ValueError("q must be a scalar or 1d")
    if method not in npq._QuantileMethods:
        raise ValueError(f"{method!r} is not a valid method. Use one of: "
                         f"{npq._QuantileMethods.keys()}")
    in_shape = tuple(a.shape)
    if axis is None:
        a, ax = _reshape(a, (-1,)), 0
    elif isinstance(axis, numbers.Integral):
        ax = int(axis)
        if not -a.ndim <= ax < a.ndim:
            raise np.exceptions.AxisError(ax, a.ndim)
        ax %= a.ndim
    else:
        raise NotImplementedError("quantile: axis must be None or an int")

    n = a.shape[ax]
    rest = tuple(a.shape[:ax]) + tuple(a.shape[ax + 1:])
    props = npq._QuantileMethods[method]
    vi = np.asanyarray(props["get_virtual_index"](n, q))
    s = _sort(a, ax)

    def at(idx):
        # s[idx] along ax, with the q dims first
        flat = np.asarray(idx, dtype=np.int64).reshape(-1) % n
        t = take(s, flat, axis=ax)
        if ax != 0:
            perm = (ax,) + tuple(d for d in range(t.ndim) if d != ax)
            t = _transpose(t, perm)
        return _reshape(t, np.shape(idx) + rest)

    if props["fix_gamma"] is None or (method == "linear" and vi.dtype.kind in "iu"):
        result = at(vi)
    else:
        prev, nxt = npq._get_indexes(np.empty(0, a.dtype), vi, n)
        gamma = npq._get_gamma(vi, prev, props)
        if weak_q:
            gamma = float(gamma)
        else:
            gamma = gamma.reshape(vi.shape + (1,) * len(rest))
        lo, hi = at(prev), at(nxt)
        diff = hi - lo
        lerp = lo + diff * gamma
        alt = hi - diff * (1 - gamma)
        if weak_q:
            result = alt if gamma >= 0.5 else lerp
        else:
            result = _where(asarray(gamma >= 0.5), alt, lerp)
        if result.dtype != lerp.dtype:
            result = result.astype(lerp.dtype)
    if a.dtype.kind == "f":
        # slices holding a NaN (sorted last) give NaN
        last = at(np.full(vi.shape, -1))
        result = _where(last != last, last, result)
    if keepdims:
        kept = ((1,) * len(in_shape) if axis is None
                else tuple(1 if d == ax else m for d, m in enumerate(in_shape)))
        result = _reshape(result, q.shape + kept)
    return _write_out(result, out)


def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear",
               keepdims=False, *, weights=None, interpolation=None):
    """``numpy.percentile`` (``quantile`` of ``q / 100``)."""
    weak_q = isinstance(q, (int, float)) and not isinstance(q, bool)
    q = np.true_divide(q, 100)
    q = q.item() if weak_q else np.asanyarray(q)
    return quantile(a, q, axis, out, overwrite_input, method, keepdims, weights=weights,
                    interpolation=interpolation)


def _normalize_axes_ordered(axis, ndim):
    axes = tuple(int(ax) % ndim for ax in axis)
    _normalize_axes(axes, ndim)
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
    # Legacy a.max(other) form: an array or non-integral scalar cannot be an axis.
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
    "percentile",
    "prod",
    "quantile",
    "std",
    "sum",
    "var",
]
