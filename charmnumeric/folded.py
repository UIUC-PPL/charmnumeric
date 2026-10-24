"""Arrays with more than 3 dims.

The backend stores 1-3-D arrays. A FoldedArray of shape (s0, ..., s[n-1]),
n > 3, is stored as 3-D (s0 * ... * s[n-3], s[n-2], s[n-1]). Folding keeps C
order, so a reshape moves data only when the 3-D storage shapes differ.

Supported: creation, get, elementwise ops (NumPy broadcasting), basic indexing
and assignment, reductions, reshape/ravel/flatten/squeeze/expand_dims, astype,
copy, fill, cumsum/cumprod and sort along one of the last two axes or
axis=None, transpose, matmul, einsum/tensordot. Limits:

- An index is a view when it selects a rectangular block of the 3-D storage.
  A view keeps NumPy's shape only if its rank is unchanged or still > 3;
  otherwise (and for other selections, up to 4096 blocks) indexing returns a
  copy. Assignment works for every basic index.
- Broadcasts whose pattern differs across folded dims, and reductions over
  leading dims, reshape operands first (extra data movement).
- Transposes run as 3-D transposes of grouped reshapes (einsum.permute).
- matmul (`@`) is batched through einsum.
- Index arrays and masks are applied on the host (offsets computed with NumPy,
  then one GATHER/SCATTER).
- var/std/argmax/argmin over axes that split into more than 3 groups of
  adjacent dims, take along an axis, and linalg raise NotImplementedError.
"""

import numbers

import numpy as np

from charmtyles.core import _shape_size

from .charmnumeric import (
    Array,
    ArrayView,
    _NDBase,
    _broadcast_to,
    _broadcasts_to,
    _coerce_operand,
    _expand_key,
    _is_array_nd,
    _inplace,
    _is_scalar_like,
    _nd_aware,
    _reshape,
    _reshape3,
    _squeeze,
    _squeeze_shape,
    _trim_leading_ones,
    _write_out,
    empty,
)

_MAX_BLOCKS = 4096


def fold_shape(shape):
    shape = tuple(shape)
    if len(shape) <= 3:
        return shape
    return (_shape_size(shape[:-2]), shape[-2], shape[-1])


def _phys(x):
    return x._base if isinstance(x, FoldedArray) else x


def _wrap(r, shape):
    shape = tuple(shape)
    if len(shape) <= 3:
        return r
    return FoldedArray(r, shape)


def reshape_nd(x, shape):
    return _wrap(_reshape3(_phys(x), fold_shape(shape)), shape)


def _kinds(xs, shape):
    """Per dim: 'K' (kept), 'B' (broadcast) or None (size 1 in both)."""
    return ["B" if a == 1 and b != 1 else ("K" if b != 1 else None) for a, b in zip(xs, shape)]


def broadcast_nd(x, shape):
    """broadcast_to for a FoldedArray source or a >3-D target."""
    shape = tuple(shape)
    xs = (1,) * (len(shape) - len(x.shape)) + tuple(x.shape)
    if int(x.size) == 1 or xs == shape:
        return reshape_nd(x, shape) if xs == shape else _fill_from(x, shape)
    n = len(shape)
    kinds = _kinds(xs, shape)
    if len({k for k in kinds[: n - 2] if k is not None}) <= 1:
        # one pattern over the folded dims: broadcast the 3-D storage directly
        src = _phys(_reshape(x, fold_shape(xs)))
        return _wrap(_broadcast_to(src, fold_shape(shape)), shape)
    groups = []  # [kind, source size, target size, first dim, end dim]
    for d in range(n):
        k = kinds[d] or (groups[-1][0] if groups else "K")
        if groups and groups[-1][0] == k:
            groups[-1][1] *= xs[d]
            groups[-1][2] *= shape[d]
            groups[-1][4] = d + 1
        else:
            groups.append([k, xs[d], shape[d], d, d + 1])
    if len(groups) <= 3:
        src = _reshape(x, tuple(g[1] for g in groups))
        return _reshape(_broadcast_to(src, tuple(g[2] for g in groups)), shape)
    # expand the first broadcast group alone, then the rest
    lo, hi = next((g[3], g[4]) for g in groups if g[0] == "B")
    return broadcast_nd(broadcast_nd(x, xs[:lo] + shape[lo:hi] + xs[hi:]), shape)


def _fill_from(x, shape):
    result = empty(shape, dtype=x.dtype)
    result[...] = x
    return result


def to_shape(value, shape, phys=None):
    """<=3-D array of shape ``phys`` (default fold(shape)) holding value broadcast to shape."""
    shape = tuple(shape)
    phys = fold_shape(shape) if phys is None else tuple(phys)
    if int(value.size) == 1:
        return _phys(value)
    vs = _trim_leading_ones(tuple(value.shape), len(shape))
    if not _broadcasts_to(vs, shape) and _squeeze_shape(vs) == _squeeze_shape(shape):
        vs = shape  # only size-1 dims differ (as for <=3-D assignment)
    if vs != tuple(value.shape):
        value = _reshape(value, vs)
    return _reshape3(_phys(_broadcast_to(value, shape)), phys)


def _coerce(value):
    if _is_array_nd(value) or isinstance(value, (str, bytes, bool, np.generic)):
        return value  # NumPy scalars stay typed (NEP 50)
    return _coerce_operand(value)


def apply_elementwise(fn, args, kwargs):
    """Run an elementwise function on the 3-D storage of folded operands."""
    out = kwargs.pop("out", None)
    args = [_coerce(a) for a in args]
    kwargs = {k: _coerce(v) for k, v in kwargs.items()}
    arrays = [a for a in list(args) + list(kwargs.values()) if _is_array_nd(a)]
    shape = tuple(np.broadcast_shapes(*[tuple(a.shape) for a in arrays]))

    def operand(a):
        if not _is_array_nd(a) or int(a.size) == 1:
            return _phys(a) if isinstance(a, FoldedArray) else a
        return _phys(_broadcast_to(a, shape))

    result = fn(*[operand(a) for a in args], **{k: operand(v) for k, v in kwargs.items()})

    def wrap(r):
        if isinstance(r, (Array, ArrayView)) and tuple(r.shape) == fold_shape(shape):
            return _wrap(r, shape)
        return r

    result = tuple(wrap(r) for r in result) if isinstance(result, tuple) else wrap(result)
    if out is not None:
        return _write_out(result, out)
    return result


# ---- indexing ----


def _select(shape, key):
    """Per-dim (start, step, count, is_int) and the result shape (with newaxis)."""
    items = _expand_key(key, len(shape), keep_newaxis=True)
    sel, out_shape, d = [], [], 0
    for item in items:
        if item is None:
            out_shape.append(1)
            continue
        size = shape[d]
        if isinstance(item, slice):
            start, stop, step = item.indices(size)
            count = len(range(start, stop, step))
            sel.append((start, step, count, False))
            out_shape.append(count)
        elif isinstance(item, numbers.Integral):
            i = int(item)
            if i < 0:
                i += size
            if not 0 <= i < size:
                raise IndexError(f"index {item} is out of bounds for axis {d} with size {size}")
            sel.append((i, 1, 1, True))
        else:
            raise TypeError("only integers, slices, newaxis and ellipsis are valid charmnumeric indices")
        d += 1
    return sel, tuple(out_shape)


def _dim0_range(shape, sel):
    """(start, step, count) of the selection along storage dim 0, or None."""
    m = len(shape) - 2
    f = [1] * m
    for j in range(m - 2, -1, -1):
        f[j] = f[j + 1] * shape[j + 1]
    base = sum(sel[j][0] * f[j] for j in range(m))
    varying = [j for j in range(m) if sel[j][2] != 1]
    if not varying:
        return base, 1, 1
    if len(varying) == 1:
        j = varying[0]
        return base, sel[j][1] * f[j], sel[j][2]
    p = varying[0]
    if sel[p][1] == 1 and all(sel[j][:3] == (0, 1, shape[j]) for j in range(p + 1, m)):
        return base, 1, sel[p][2] * f[p]
    return None


def _sel_slice(start, step, count):
    stop = start + (count - 1) * step + (1 if step > 0 else -1)
    return slice(start, stop if stop >= 0 else None, step)


def _storage_view(x, sel, r):
    sl = _sel_slice

    n = len(x.shape)
    key = (sl(*r), sl(*sel[n - 2][:3]), sl(*sel[n - 1][:3]))
    return x._base[key]


def _blocks(shape, sel):
    """Leading dims to iterate so that each fixed choice is rectangular."""
    m = len(shape) - 2
    for q in range(m + 1):
        test = [(s[0], 1, 1, True) if j < q else s for j, s in enumerate(sel)]
        if _dim0_range(shape, test) is not None:
            break
    dims = [j for j in range(q) if sel[j][2] != 1]
    if _shape_size([sel[j][2] for j in dims]) > _MAX_BLOCKS:
        raise NotImplementedError("index selects too many non-contiguous blocks of a >3-D array")
    return dims


def _block_keys(sel, dims):
    """(key into x, key into the selection) for every block."""
    res_pos = {}
    k = 0
    for j, s in enumerate(sel):
        if not s[3]:
            res_pos[j] = k
            k += 1
    nres = k
    for combo in np.ndindex(*[sel[j][2] for j in dims]):
        xkey, rkey = [], [slice(None)] * nres
        fixed = dict(zip(dims, combo))
        for j, s in enumerate(sel):
            if j in fixed:
                xkey.append(s[0] + fixed[j] * s[1])
                rkey[res_pos[j]] = fixed[j]
            elif s[3]:
                xkey.append(s[0])
            else:
                xkey.append(_sel_slice(*s[:3]))
        yield tuple(xkey), tuple(rkey)


def _has_index_arrays(key):
    from .charmnumeric import _is_indexing_key

    items = key if isinstance(key, tuple) else (key,)
    return _is_indexing_key(key) or any(isinstance(k, _NDBase) for k in items)


def _detached(r):
    """Mark a copy returned for a basic index (writes to it do not reach the source)."""
    if isinstance(r, FoldedArray):
        r._copy_of_view = True
    return r


def getitem(x, key):
    if _has_index_arrays(key):
        from .indexing import folded_getitem

        return folded_getitem(x, key)
    sel, out_shape = _select(x.shape, key)
    sel_shape = tuple(s[2] for s in sel if not s[3])
    if 0 in sel_shape:
        return empty(out_shape, dtype=x.dtype)
    r = _dim0_range(x.shape, sel)
    if r is not None:
        view = _storage_view(x, sel, r)
        if out_shape == tuple(view.shape):
            return view
        if len(out_shape) > 3 and fold_shape(out_shape) == tuple(view.shape):
            return FoldedArray(view, out_shape)
        return _detached(_reshape(view, out_shape))
    result = empty(sel_shape, dtype=x.dtype)
    for xkey, rkey in _block_keys(sel, _blocks(x.shape, sel)):
        result[rkey] = getitem(x, xkey)
    return _detached(_reshape(result, out_shape))


def setitem(x, key, value):
    if _has_index_arrays(key):
        from .indexing import folded_setitem

        return folded_setitem(x, key, value)
    sel, _ = _select(x.shape, key)
    sel_shape = tuple(s[2] for s in sel if not s[3])
    if 0 in sel_shape:
        return
    value = _coerce(value)
    if not _is_array_nd(value) and not _is_scalar_like(value):
        raise TypeError(f"cannot assign a {type(value).__name__} to a charmnumeric array")
    if _is_array_nd(value) and value.dtype != x.dtype:
        value = value.astype(x.dtype)
    r = _dim0_range(x.shape, sel)
    if r is not None:
        view = _storage_view(x, sel, r)
        if _is_array_nd(value):
            value = to_shape(value, sel_shape or (1,), tuple(view.shape))
        view[...] = value
        return
    if _is_array_nd(value) and int(value.size) != 1:
        value = _broadcast_to(value, sel_shape)
    for xkey, rkey in _block_keys(sel, _blocks(x.shape, sel)):
        setitem(x, xkey, value[rkey] if _is_array_nd(value) and int(value.size) != 1 else value)


# ---- reductions, scans, sorts ----


def reduce_nd(a, op, axis, keepdims, dtype, ddof, single):
    from . import reductions as red

    if not isinstance(a, FoldedArray):
        return red._reduce(a, op, axis, keepdims, dtype, None, ddof, single)
    shape = tuple(a.shape)
    n = len(shape)
    axes = red._normalize_axes(axis, n, single=single)
    if op in (red._RED_ARGMAX, red._RED_ARGMIN) and len(axes) not in (1, n):
        raise ValueError("argmax/argmin reduce over one axis or all axes")
    out_shape = red._result_shape(shape, axes, keepdims)
    if len(axes) == n:
        return _reshape(red._reduce(a._base, op, None, False, dtype, None, ddof), out_shape)
    if min(axes) >= n - 2:
        # only the last two dims: reduce the storage directly
        kept = red._reduce(a._base, op, tuple(ax - (n - 3) for ax in axes), True, dtype, None, ddof)
        return _reshape(_wrap(kept, shape[:-2] + tuple(kept.shape[1:])), out_shape)
    groups = []
    for d in range(n):
        if groups and groups[-1][0] == (d in axes):
            groups[-1][1] *= shape[d]
            groups[-1][2].append(d)
        else:
            groups.append([d in axes, shape[d], [d]])
    if len(groups) <= 3:
        x3 = _reshape(a, tuple(g[1] for g in groups))
        gaxes = tuple(i for i, g in enumerate(groups) if g[0])
        return _reshape(red._reduce(x3, op, gaxes, False, dtype, None, ddof), out_shape)
    # more than 3 groups: the last reduced group first, then the others
    chained = {red._RED_SUM: red._RED_SUM, red._RED_PROD: red._RED_PROD, red._RED_MAX: red._RED_MAX,
               red._RED_MIN: red._RED_MIN, red._RED_ALL: red._RED_ALL, red._RED_ANY: red._RED_ANY,
               red._RED_MEAN: red._RED_MEAN, red._RED_COUNT_NONZERO: red._RED_SUM}
    if op not in chained:
        raise NotImplementedError("this reduction over these axes of a >3-D array is not supported")
    last = [g for g in groups if g[0]][-1][2]
    part = reduce_nd(a, op, tuple(last), False, dtype, ddof, False)
    rest = tuple(ax for ax in axes if ax not in last)
    return _reshape(reduce_nd(part, chained[op], rest, False, dtype, ddof, False), out_shape)


def along_last_dims(a, axis, fn):
    """fn(storage, storage_axis) for an op along one of the last two dims."""
    n = len(a.shape)
    if axis < n - 2:
        raise NotImplementedError(
            "along a leading axis of a >3-D array: only the last two axes or axis=None"
        )
    return _wrap(fn(a._base, axis - (n - 3)), a.shape)


# ---- the array type ----


class FoldedArray(_NDBase):
    """Array with more than 3 dims, stored as 3-D (see module docstring)."""

    __array_priority__ = 1000

    def __init__(self, base, shape):
        shape = tuple(int(n) for n in shape)
        if fold_shape(shape) != tuple(base.shape):
            raise ValueError(f"storage shape {tuple(base.shape)} does not fold {shape}")
        self._base = base
        self._shape = shape
        self.dtype = base.dtype

    shape = property(lambda self: self._shape)
    ndim = property(lambda self: len(self._shape))
    ndims = ndim
    size = property(lambda self: self._base.size)
    itemsize = property(lambda self: self.dtype.itemsize)
    nbytes = property(lambda self: int(self.size) * self.dtype.itemsize)

    def __len__(self):
        return self._shape[0]

    def __bool__(self):
        raise ValueError("The truth value of a charmnumeric array is ambiguous. "
                         "Use .get(interface) to materialize it first.")

    def __array__(self, dtype=None, copy=None):
        from .interface import default_interface

        host = self.get(default_interface())
        return host if dtype is None else host.astype(dtype, copy=False)

    def to_numpy(self):
        return self.__array__()

    def __repr__(self):
        return f"FoldedArray(shape={self._shape}, dtype={self.dtype.name})"

    def __str__(self):
        from ._display import values_str

        return values_str(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        from ._display import mimebundle

        return mimebundle(self)

    __hash__ = object.__hash__

    def get(self, interface):
        return self._base.get(interface).reshape(self._shape)

    __getitem__ = getitem
    __setitem__ = setitem

    def reshape(self, *shape, order="C"):
        if len(shape) == 1 and not isinstance(shape[0], numbers.Integral):
            shape = shape[0]
        return _reshape(self, shape, order)

    def ravel(self, order="C"):
        return _reshape(self, (-1,), order)

    def flatten(self, order="C"):
        return _reshape(self, (-1,), order).copy()

    def squeeze(self, axis=None):
        return _squeeze(self, axis)

    def copy(self):
        return FoldedArray(self._base.copy(), self._shape)

    def fill(self, value):
        self._base[...] = value

    def astype(self, dtype, copy=True):
        if np.dtype(dtype) == self.dtype:
            return self.copy() if copy else self
        return FoldedArray(self._base.astype(dtype), self._shape)

    def _not_supported(self, *args, **kwargs):
        raise NotImplementedError("dot/take of arrays with more than 3 dims (use einsum/tensordot)")

    take = _not_supported

    def dot(self, other, out=None):
        from .charmnumeric import _dot
        return _write_out(_dot(self, other), out)

    def __matmul__(self, other):
        from .charmnumeric import _array_matmul
        return _array_matmul(self, other)

    def __rmatmul__(self, other):
        from .charmnumeric import _array_matmul, _coerce_operand
        return _array_matmul(_coerce_operand(other), self)

    def transpose(self, *axes):
        from .charmnumeric import _transpose_method
        return _transpose_method(self, *axes)

    def swapaxes(self, axis1, axis2):
        from .charmnumeric import _swapaxes
        return _swapaxes(self, axis1, axis2)

    T = property(lambda self: self.transpose())

    def cumsum(self, axis=None, dtype=None, out=None):
        from .charmnumeric import _scan
        return _scan("cumsum", self, axis, dtype, out)

    def cumprod(self, axis=None, dtype=None, out=None):
        from .charmnumeric import _scan
        return _scan("cumprod", self, axis, dtype, out)

    def sort(self, axis=-1, kind=None, order=None):
        from .charmnumeric import _sort
        self[...] = _sort(self, axis, kind, order)

    def argsort(self, axis=-1, kind=None, order=None):
        from .charmnumeric import _sort
        return _sort(self, axis, kind, order, arg=True)


def _install():
    from . import charmnumeric as cn
    from . import reductions  # noqa: F401  (installs the reduction methods on Array)

    for name in cn._ND_DUNDERS:
        setattr(FoldedArray, name, getattr(Array, name))
    for name in ("__neg__", "__pos__", "__abs__", "__invert__", "round", "clip"):
        setattr(FoldedArray, name, _nd_aware(getattr(Array, name)))
    for name, fn in (("__iadd__", "__add__"), ("__isub__", "__sub__"), ("__imul__", "__mul__"),
                     ("__itruediv__", "__truediv__"), ("__ifloordiv__", "__floordiv__"),
                     ("__imod__", "__mod__"), ("__ipow__", "__pow__"), ("__iand__", "__and__"),
                     ("__ior__", "__or__"), ("__ixor__", "__xor__"),
                     ("__ilshift__", "__lshift__"), ("__irshift__", "__rshift__")):
        setattr(FoldedArray, name, _inplace(getattr(FoldedArray, fn)))
    for name in ("sum", "prod", "mean", "var", "std", "max", "min", "argmax", "argmin", "all",
                 "any"):
        setattr(FoldedArray, name, getattr(Array, name))


_install()
