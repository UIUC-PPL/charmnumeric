"""Convolution and flips: CONVOLVE (opcode 150) and FLIP (151), ``src/executor_signal.cpp``.

Results are new arrays, not views; each tile gathers the source boxes it reads.
"""

import builtins
import numbers

import numpy as np

from .charmnumeric import Array, ArrayOperation, ArrayView, asarray, nonfusible_op

__all__ = ["convolve", "flip", "flipud", "fliplr"]

_CONV_DIRECT = 0
_CONV_FFT = 1
# method="auto" switches to the FFT once the filter has this many taps.
_AUTO_FFT_TAPS = 4096


def _shape_size(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _as_array(x):
    return x if isinstance(x, (Array, ArrayView)) else asarray(x)


def _conv_geometry(a_shape, v_shape, mode):
    """Result shape and per-axis shift s: out[j] = sum_k v[k] a[j + s - k]."""
    if mode == "full":
        return tuple(n + f - 1 for n, f in zip(a_shape, v_shape)), (0,) * len(a_shape)
    if mode == "same":
        return tuple(a_shape), tuple((f - 1) // 2 for f in v_shape)
    return tuple(n - f + 1 for n, f in zip(a_shape, v_shape)), tuple(f - 1 for f in v_shape)


def convolve(a, v, mode="full", method="auto"):
    """N-D linear convolution (cuPyNumeric ``convolve``; 1-D follows ``numpy.convolve``)."""
    if mode not in ("full", "same", "valid"):
        raise ValueError(f"mode must be one of 'full', 'same', 'valid' (got {mode!r})")
    if method not in ("auto", "direct", "fft"):
        raise ValueError("Acceptable method flags are 'auto', 'direct', or 'fft'.")
    a = _as_array(a)
    v = _as_array(v)
    if a.ndim != v.ndim:
        raise RuntimeError("Arrays should have the same dimensions")
    if a.ndim > 3:
        raise NotImplementedError(f"{a.ndim}-D arrays are not yet supported")
    if a.ndim == 1 and int(a.size) < int(v.size):
        a, v = v, a
    if mode == "valid" and not all(n >= f for n, f in zip(a.shape, v.shape)):
        # convolution commutes; NumPy/SciPy swap when v is larger everywhere
        if all(f >= n for n, f in zip(a.shape, v.shape)):
            a, v = v, a
        else:
            raise ValueError(
                "For 'valid' mode, one must be at least as large as the other in every dimension"
            )

    shape, shift = _conv_geometry(a.shape, v.shape, mode)
    if int(v.size) == 1:
        # size-1 operands travel as broadcast scalars: a plain multiply
        return a * v
    if int(a.size) == 1:
        if mode == "full":
            return v * a
        c = tuple(slice(s, s + 1) for s in shift)
        return v[c] * a

    result_dtype = np.result_type(a.dtype, v.dtype)
    if result_dtype.kind == "c":
        raise NotImplementedError("convolve of complex arrays is not supported")
    if result_dtype.kind == "f":
        dtype = np.dtype(np.float64 if result_dtype == np.float64 else np.float32)
    elif result_dtype in (np.int32, np.int64):
        dtype = result_dtype
    else:
        dtype = np.dtype(np.int64)  # bool and small/unsigned ints: exact mod 2**k
    a, v = a.astype(dtype, copy=False), v.astype(dtype, copy=False)
    if method == "auto":
        floating = dtype.kind == "f"
        method = "fft" if floating and _shape_size(v.shape) >= _AUTO_FFT_TAPS else "direct"
    code = _CONV_FFT if method == "fft" else _CONV_DIRECT
    out = nonfusible_op(
        a,
        v,
        code,
        *shift,
        *shape,
        operation=ArrayOperation.convolve,
        shape=shape,
        dtype=dtype,
    )
    return out.astype(result_dtype, copy=False)


def _flip_axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, numbers.Integral) else tuple(axis)
    out = []
    for ax in axes:
        ax = int(ax)
        if ax < -ndim or ax >= ndim:
            raise np.exceptions.AxisError(ax, ndim)
        out.append(ax % ndim)
    if len(set(out)) != len(out):
        raise ValueError("repeated axis")
    return tuple(out)


def flip(m, axis=None):
    """Reverse the order of elements along the given axes (a copy, not a view)."""
    m = _as_array(m)
    axes = [ax for ax in _flip_axes(axis, m.ndim) if m.shape[ax] > 1]
    if not axes or int(m.size) == 1:
        return m
    mask = builtins.sum(1 << ax for ax in axes)
    return nonfusible_op(m, mask, operation=ArrayOperation.flip, shape=m.shape, dtype=m.dtype)


def flipud(m):
    """Reverse the order of elements along axis 0."""
    m = _as_array(m)
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return flip(m, 0)


def fliplr(m):
    """Reverse the order of elements along axis 1."""
    m = _as_array(m)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return flip(m, 1)


def split_negative_steps(key, shape):
    """(positive key, flipped axes) for a key with negative-step slices, else None."""
    if not isinstance(key, tuple):
        key = (key,)
    if not any(isinstance(k, slice) and k.step is not None and k.step < 0 for k in key):
        return None
    from .charmnumeric import _expand_key

    key = _expand_key(key, len(shape))
    pos, axes = [], []
    for ax, item in enumerate(key):
        if isinstance(item, slice) and item.step is not None and item.step < 0:
            start, stop, step = item.indices(shape[ax])
            count = len(range(start, stop, step))
            if count == 0:
                raise NotImplementedError("empty slices are not supported")
            first = start + (count - 1) * step
            pos.append(slice(first, start + 1, -step))
            axes.append(ax)
        else:
            pos.append(item)
    return tuple(pos), tuple(axes)


def reference(op, values, scalars, shape):
    """NumPy model of a CONVOLVE/FLIP node (used by the CNUM_SIM simulator)."""
    if op == ArrayOperation.flip:
        (x,) = values
        mask = int(scalars[0])
        return np.flip(x, tuple(d for d in range(x.ndim) if (mask >> d) & 1))
    a, v = values
    nd = a.ndim
    shift = [int(s) for s in scalars[1 : 1 + nd]]
    full = np.zeros(tuple(n + f - 1 for n, f in zip(a.shape, v.shape)), dtype=np.result_type(a, v))
    for k in np.ndindex(*v.shape):
        full[tuple(slice(k[d], k[d] + a.shape[d]) for d in range(nd))] += v[k] * a
    return full[tuple(slice(shift[d], shift[d] + shape[d]) for d in range(nd))]
