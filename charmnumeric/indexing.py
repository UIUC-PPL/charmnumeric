"""Indexing with index arrays and masks: advanced indexing, take/put, nonzero,
compress, repeat and choose.

Each index expression is linearized into one array of flat offsets into the
indexed array's root, then executed by one GATHER or SCATTER node
(src/executor_indexing.cpp) over ``E = pre slice dims + broadcast(index shapes)
+ post slice dims``. Masks become index arrays via EXPAND (nonzero); the output
size is fetched first with a ``count_nonzero`` get, as are repeat with array
counts and compress with a device condition.

Differences from NumPy:
- Negative-step slices next to an index array are part of the gather; keys of
  only slices and integers go through views and FLIP (charmnumeric/signal.py).
- Device index arrays are not bounds-checked (out-of-range offsets are dropped
  by the backend); host index lists are checked.
- Device index arrays must have equal shapes (no broadcasting); results have at
  most 3 dimensions (more for >3-D sources, see folded.py).
- Gathered and scattered elements are at most 8 bytes (no complex128).
"""

import numbers

import numpy as np

from charmtyles.ast import NonFusibleNode, get_execution_state
from charmtyles.core import create_result

from .charmnumeric import (
    Array,
    ArrayOperation,
    ArrayView,
    _NDBase,
    _axis_index,
    _axis_range,
    _broadcast_to,
    _cast_operand,
    _clip,
    _coerce_scalar,
    _compare,
    _fresh_operands,
    _is_array,
    _is_array_nd,
    _reshape,
    _full_region,
    _is_scalar_like,
    _reshape_copy,
    _where,
    _wrote,
    arange,
    asarray,
    create_array,
    empty,
    full,
    nonfusible_op,
)
from .interface import default_interface

_INT64 = np.dtype("int64")
_BOOL = np.dtype(bool)


def _is_mask(x):
    return x.dtype == _BOOL


def _check_itemsize(x):
    if x.dtype.itemsize > 8:
        raise NotImplementedError(f"indexing {x.dtype} arrays is not supported")


def _fetch_int(x):
    """Value of a size-1 backend array (a blocking get)."""
    return int(x.get(default_interface()).reshape(-1)[0])


def _geometry(x):
    """(root, base, mult): x[i] is root.flat[base + sum(i_d * mult_d)]."""
    root = x._root_parent if isinstance(x, ArrayView) else x
    start, step = x.region.start, x.region.step
    rs = [1] * root.ndims
    for d in range(root.ndims - 2, -1, -1):
        rs[d] = rs[d + 1] * root.shape[d + 1]
    base = sum(int(start[d]) * rs[d] for d in range(x.ndims))
    mult = [int(step[d]) * rs[d] for d in range(x.ndims)]
    return root, base, mult


# ---- backend nodes ----


def _gather(root, L, E, npre, npost, strides):
    _check_itemsize(root)
    E = tuple(int(e) for e in E)
    if int(np.prod(E)) == 0:
        return create_array(E, dtype=root.dtype)
    return nonfusible_op(
        root, L, npre, npost, len(E), *E, *[int(s) for s in strides],
        operation=ArrayOperation.gather, shape=E, dtype=root.dtype,
    )


def _scatter(root, L, value, E, npre, npost, strides):
    _check_itemsize(root)
    E = tuple(int(e) for e in E)
    if int(np.prod(E)) == 0:
        return
    _fresh_operands((root, L, value))
    result = create_result(
        Array, ArrayOperation.scatter,
        [root, L, value, npre, npost, len(E), *E, *[int(s) for s in strides]],
        fusible=False, region=_full_region(E), dtype=root.dtype, shape=E,
    )
    result.writes = ((root.name, root.region),)  # in place: the result name is unused
    get_execution_state().add_node(NonFusibleNode(obj=result))
    _wrote(root)


def _expand(x, count_mode, total):
    """Flat positions of x's elements, each repeated (x != 0) or x times."""
    _check_itemsize(x)
    return nonfusible_op(
        x, 1 if count_mode else 0, int(total),
        operation=ArrayOperation.expand, shape=(int(total),), dtype=_INT64,
    )


# ---- index normalization ----


def _wrap_host(idx, n, mode, axis=0):
    idx = np.asarray(idx, dtype=np.int64)
    if mode == "wrap":
        return idx % n if n else idx
    if mode == "clip":
        return np.clip(idx, 0, max(n - 1, 0))
    bad = (idx < -n) | (idx >= n)
    if bad.any():
        raise IndexError(
            f"index {int(idx[bad].reshape(-1)[0])} is out of bounds for axis {axis} with size {n}"
        )
    return np.where(idx < 0, idx + n, idx)


def _wrap_device(idx, n, mode):
    if mode == "wrap":
        return idx % n
    if mode == "clip":
        return _clip(idx, 0, n - 1)
    return _where(_compare(idx, 0, "less"), idx + n, idx)


def _nonzero_flat(x):
    """Flat (row-major) positions of the nonzero elements of x, or None if none."""
    from .reductions import count_nonzero

    if x.dtype.itemsize > 8:
        x = _compare(x, 0, "not_equal")
    cnt = _fetch_int(count_nonzero(x))
    return _expand(x, False, cnt) if cnt else None


def _unravel(F, shape):
    """Coordinates of flat positions F (a backend int array) in an array of `shape`."""
    coords = []
    rest = F
    for d in range(len(shape) - 1, 0, -1):
        coords.append(rest % shape[d])
        rest = rest // shape[d]
    coords.append(rest)
    return tuple(reversed(coords))


def _items(x, key, newaxes=None):
    """One entry per dim of x: ("s", start, step, count), ("i", index) or
    ("a", index array, trusted). Masks become their nonzero coordinates; newaxis
    positions (dims of x before them) go to ``newaxes``."""
    key = key if isinstance(key, tuple) else (key,)
    raw = []
    for k in key:
        if k is None:
            if newaxes is None:
                raise NotImplementedError("np.newaxis is not supported here")
            raw.append(("n", None, 0))
        elif k is Ellipsis:
            raw.append(("...", None, 0))
        elif isinstance(k, slice):
            raw.append(("s", k, 1))
        elif _is_array(k):
            if _is_mask(k):
                raw.append(("m", k, k.ndims))
            elif np.issubdtype(k.dtype, np.integer):
                raw.append(("a", k, 1))
            else:
                raise IndexError("arrays used as indices must be of integer (or boolean) type")
        elif isinstance(k, (bool, np.bool_)):
            raise NotImplementedError("scalar boolean indices are not supported")
        elif isinstance(k, numbers.Integral):
            raw.append(("i", int(k), 1))
        else:
            h = np.asarray(k)
            if h.dtype == bool:
                if h.ndim == 0:
                    raise NotImplementedError("scalar boolean indices are not supported")
                raw.append(("hm", h, h.ndim))
            elif h.ndim == 0 and np.issubdtype(h.dtype, np.integer):
                raw.append(("i", int(h), 1))
            elif np.issubdtype(h.dtype, np.integer) or h.size == 0:
                raw.append(("a", h.astype(np.int64), 1))
            else:
                raise IndexError("arrays used as indices must be of integer (or boolean) type")
    if sum(1 for r in raw if r[0] == "...") > 1:
        raise IndexError("an index can only have a single ellipsis")
    used = sum(r[2] for r in raw)
    if used > x.ndims:
        raise IndexError(
            f"too many indices for array: array is {x.ndims}-dimensional, "
            f"but {used} were indexed"
        )
    if not any(r[0] == "..." for r in raw):
        raw.append(("...", None, 0))

    items = []
    seen = {}
    for kind, k, nd in raw:
        d = len(items)
        if kind == "...":
            items.extend(("s", slice(None)) for _ in range(x.ndims - used))
        elif kind == "n":
            newaxes.append(d)
        elif kind in ("m", "hm"):
            if tuple(k.shape) != tuple(x.shape[d:d + nd]):
                raise IndexError(
                    f"boolean index of shape {tuple(k.shape)} does not match the indexed "
                    f"dimensions {tuple(x.shape[d:d + nd])}"
                )
            if kind == "hm":
                items.extend(("a", c, True) for c in np.nonzero(k))
                continue
            if id(k) not in seen:  # e.g. a[b, b]: one nonzero for both
                F = _nonzero_flat(k)
                seen[id(k)] = None if F is None else _unravel(F, tuple(k.shape))
            coords = seen[id(k)]
            if coords is None:
                items.extend(("a", np.zeros(0, np.int64), True) for _ in range(nd))
            else:
                items.extend(("a", c, True) for c in coords)
        elif kind == "a":
            items.append(("a", k, False))
        else:
            items.append((kind, k))

    out = []
    for d, it in enumerate(items):
        n = x.shape[d]
        if it[0] == "s":
            start, stop, step = it[1].indices(n)
            out.append(("s", start, step, len(range(start, stop, step))))
        elif it[0] == "i":
            i = it[1] + n if it[1] < 0 else it[1]
            if not 0 <= i < n:
                raise IndexError(f"index {it[1]} is out of bounds for axis {d} with size {n}")
            out.append(("i", i))
        else:
            out.append(it)
    return out


class _Plan:
    """Index linearization: element e of E reads root.flat[L[b] + sum(s_k * strides[k])]."""

    __slots__ = ("root", "L", "E", "final", "npre", "npost", "strides")


def _plan(x, items, mode="raise", newaxes=()):
    """Linearize per-dim items over x; None if no index array is present.
    ``final`` is the result shape (``E`` plus a 1 per newaxis)."""
    arrays = [d for d, it in enumerate(items) if it[0] == "a"]
    if not arrays:
        return None
    adv = [d for d, it in enumerate(items) if it[0] in ("a", "i")]
    root, base, mult = _geometry(x)

    B = tuple(np.broadcast_shapes(*[tuple(items[d][1].shape) for d in arrays]))
    # key positions: one per dim of x, plus the newaxes (before dim n[i])
    toks = []
    for d in range(len(items) + 1):
        toks.extend(("n", None) for n in newaxes if n == d)
        if d < len(items):
            toks.append(("x", d))
    tadv = [i for i, t in enumerate(toks) if t[0] == "x" and t[1] in adv]
    contiguous = tadv == list(range(tadv[0], tadv[-1] + 1))
    slices = [d for d, it in enumerate(items) if it[0] == "s"]
    first = tadv[0]
    pos = {t[1]: i for i, t in enumerate(toks) if t[0] == "x"}
    pre = [d for d in slices if pos[d] < first] if contiguous else []
    post = [d for d in slices if d not in pre]
    E = tuple(items[d][3] for d in pre) + B + tuple(items[d][3] for d in post)

    def size(i):
        return 1 if toks[i][0] == "n" else items[toks[i][1]][3]

    keep = [i for i, t in enumerate(toks) if t[0] == "n" or items[t[1]][0] == "s"]
    lead = [i for i in keep if contiguous and i < first]
    rest = [i for i in keep if i not in lead]
    plan_final = tuple(size(i) for i in lead) + B + tuple(size(i) for i in rest)
    if len(E) > 3:
        raise NotImplementedError(f"indexing result would have {len(E)} > 3 dimensions")
    for d in slices:
        base += items[d][1] * mult[d]
    for d in adv:
        if items[d][0] == "i":
            base += items[d][1] * mult[d]

    plan = _Plan()
    plan.root = root
    plan.E = E
    plan.final = plan_final
    plan.npre, plan.npost = len(pre), len(post)
    plan.strides = [items[d][2] * mult[d] for d in pre + post]
    plan.L = None
    if int(np.prod(E)) == 0:
        return plan

    host = np.zeros(B, dtype=np.int64) + base
    devs = []
    for d in arrays:
        idx, trusted = items[d][1], items[d][2]
        if _is_array(idx):
            if tuple(idx.shape) != B:
                raise NotImplementedError(
                    f"device index arrays must all have the same shape (got {tuple(idx.shape)} "
                    f"and {B})"
                )
            devs.append((idx if trusted else _wrap_device(idx, x.shape[d], mode), mult[d]))
        else:
            host = host + _wrap_host(idx, x.shape[d], mode, d) * mult[d]
    if not devs:
        plan.L = asarray(np.ascontiguousarray(host))
        return plan
    dt = devs[0][0].dtype
    if any(a.dtype != dt for a, _ in devs):
        dt = _INT64
        devs = [(_cast_operand(a, dt), m) for a, m in devs]
    L = None
    for a, m in devs:
        t = a * m if m != 1 else a
        L = t if L is None else L + t
    if len(devs) < len(arrays):
        L = L + asarray(host.astype(dt))
    elif base:
        L = L + base
    plan.L = L
    return plan


def _key_plan(x, key, mode="raise"):
    newaxes = []
    items = _items(x, key, newaxes)
    return _plan(x, items, mode, newaxes)


# ---- __getitem__ / __setitem__ ----


def getitem(x, key):
    plan = _key_plan(x, key)
    if plan is None:  # e.g. a[[...]] parsed to plain slices
        raise IndexError(f"unsupported index {key!r}")
    result = _gather(plan.root, plan.L, plan.E, plan.npre, plan.npost, plan.strides)
    return result if plan.final == plan.E else _reshape(result, plan.final)


def _scatter_value(x, value, E):
    """A scalar, or a backend array of shape E with x's dtype (cast on assignment)."""
    if _is_array(value):
        value = _cast_operand(value, x.dtype)
        return value if tuple(value.shape) == tuple(E) else _broadcast_to(value, E)
    if _is_scalar_like(value):
        return _coerce_scalar(value)
    host = np.asarray(value, dtype=x.dtype)
    if host.ndim == 0:
        return _coerce_scalar(host.item())
    try:
        host = np.broadcast_to(host, E)
    except ValueError as exc:
        raise ValueError(
            f"could not broadcast input from shape {host.shape} into shape {E}"
        ) from exc
    return asarray(np.ascontiguousarray(host))


def setitem(x, key, value):
    k = key[0] if isinstance(key, tuple) and len(key) == 1 else key
    if (_is_array(k) and _is_mask(k) and tuple(k.shape) == tuple(x.shape)
            and _is_scalar_like(value)):
        # x[mask] = v without a sync: an elementwise select
        x[...] = _where(k, _coerce_scalar(value), x)
        return
    plan = _key_plan(x, key)
    if plan is None:
        raise IndexError(f"unsupported index {key!r}")
    if int(np.prod(plan.E)) == 0:
        return
    if plan.final != plan.E and _is_array(value):
        # newaxis dims of the target are size 1; drop them from the value too
        vs = tuple(value.shape)
        squeezed = lambda sh: tuple(n for n in sh if n != 1)
        if vs != plan.E and squeezed(vs) == squeezed(plan.E):
            value = _reshape(value, plan.E)
    value = _scatter_value(x, value, plan.E)
    _scatter(plan.root, plan.L, value, plan.E, plan.npre, plan.npost, plan.strides)


# ---- >3-D (folded) arrays: offsets computed on the host ----


def _host_key(key):
    key = key if isinstance(key, tuple) else (key,)
    return tuple(
        np.asarray(k.get(default_interface())) if _is_array_nd(k)
        else np.asarray(k) if isinstance(k, list) else k
        for k in key
    )


def _folded_offsets(shape, key):
    """Flat row-major offsets of x[key], shaped like the result (NumPy applies the key)."""
    n = len(shape)
    total = 0
    stride = 1
    for d in range(n - 1, -1, -1):
        ar = np.arange(shape[d], dtype=np.int64).reshape((1,) * d + (-1,) + (1,) * (n - d - 1))
        total = total + np.broadcast_to(ar, shape)[key] * stride
        stride *= shape[d]
    return total


def folded_getitem(x, key):
    offs = _folded_offsets(tuple(x.shape), _host_key(key))
    if offs.size == 0:
        return empty(offs.shape, dtype=x.dtype)
    L = asarray(np.ascontiguousarray(offs.reshape(-1)))
    result = _gather(_flat_source(x), L, (offs.size,), 0, 0, [])
    return _reshape(result, offs.shape)


def folded_setitem(x, key, value):
    offs = _folded_offsets(tuple(x.shape), _host_key(key))
    if offs.size == 0:
        return
    if getattr(x, "_copy_of_view", False):
        raise NotImplementedError("index arrays on a copy of a view of a >3-D array are not supported")
    root = x._base
    if isinstance(root, ArrayView) and tuple(root.shape) != tuple(root._root_parent.shape):
        raise NotImplementedError("index arrays on a view of a >3-D array are not supported")
    root = root._root_parent if isinstance(root, ArrayView) else root
    n = (int(offs.size),)
    if _is_array_nd(value):
        value = _reshape(_broadcast_to(_cast_operand(value, x.dtype), offs.shape), n)
    elif not _is_scalar_like(value):
        value = np.broadcast_to(np.asarray(value, dtype=x.dtype), offs.shape).reshape(-1)
    L = asarray(np.ascontiguousarray(offs.reshape(-1)))
    _scatter(root, L, _scatter_value(x, value, n), n, 0, 0, [])


# ---- NumPy functions ----


def nonzero(a):
    """numpy.nonzero: a tuple of int64 index arrays (blocking: fetches the count)."""
    a = asarray(a)
    F = _nonzero_flat(a)
    if F is None:
        return tuple(create_array((0,), dtype=_INT64) for _ in range(a.ndims))
    return _unravel(F, tuple(a.shape))


def flatnonzero(a):
    """numpy.flatnonzero (blocking: fetches the count)."""
    F = _nonzero_flat(asarray(a))
    return create_array((0,), dtype=_INT64) if F is None else F


def argwhere(a):
    """numpy.argwhere: (count, ndim) int64 array (blocking: fetches the count)."""
    a = asarray(a)
    coords = nonzero(a)
    cnt = coords[0].shape[0]
    out = empty((cnt, a.ndims), dtype=_INT64)
    if cnt:
        for d, c in enumerate(coords):
            out[:, d] = c
    return out


def _flat_source(a):
    """A root array whose flat order is a's row-major order."""
    if isinstance(a, _NDBase):
        a = a._base  # folded storage keeps C order
    if isinstance(a, ArrayView):
        root = a._root_parent
        if tuple(a.shape) != tuple(root.shape):
            return a.copy()
        return root
    return a


def _flat_gather(a, idx, mode):
    """a.flat[idx] (idx: host or backend int array)."""
    a = _flat_source(a)
    n = a.size
    if _is_array(idx):
        L = _wrap_device(idx, n, mode)
        return _gather(a, L, tuple(idx.shape), 0, 0, [])
    idx = _wrap_host(idx, n, mode)
    if idx.size == 0:
        return create_array(idx.shape, dtype=a.dtype)
    return _gather(a, asarray(np.ascontiguousarray(idx)), idx.shape, 0, 0, [])


def take(a, indices, axis=None, out=None, mode="raise"):
    """numpy.take (mode 'raise' does not check device indices)."""
    a = asarray(a)
    if mode not in ("raise", "wrap", "clip"):
        raise ValueError(f"clipmode must be one of 'clip', 'raise', or 'wrap' (got {mode!r})")
    if not _is_array(indices):
        h = np.asarray(indices)
        if h.size and not np.issubdtype(h.dtype, np.integer):
            raise IndexError("arrays used as indices must be of integer type")
    if not _is_array(indices) and np.ndim(indices) == 0:
        # scalar index: the axis is dropped
        if axis is None:
            result = _reshape_copy(_flat_gather(a, np.asarray([indices]), mode), ())
        else:
            axis = _axis_index(axis, a.ndims)
            i = int(_wrap_host(np.asarray([indices]), a.shape[axis], mode, axis)[0])
            result = a[(slice(None),) * axis + (slice(i, i + 1),)]
            result = _reshape_copy(result, tuple(a.shape[:axis]) + tuple(a.shape[axis + 1:]))
    elif axis is None:
        result = _flat_gather(a, indices, mode)
    elif a.ndims - 1 + np.ndim(indices) > 3 and np.ndim(indices) > 1:
        # a >3-D result: take along the flattened indices, then a native reshape
        axis = _axis_index(axis, a.ndims)
        ishape = tuple(indices.shape) if _is_array(indices) else np.shape(indices)
        flat = indices.reshape(-1) if _is_array(indices) else np.asarray(indices).reshape(-1)
        result = take(a, flat, axis=axis, mode=mode).reshape(
            tuple(a.shape[:axis]) + ishape + tuple(a.shape[axis + 1:]))
    else:
        axis = _axis_index(axis, a.ndims)
        items = _items(a, (slice(None),) * axis + (indices,))
        plan = _plan(a, items, mode)
        result = _gather(plan.root, plan.L, plan.E, plan.npre, plan.npost, plan.strides)
    if out is not None:
        out[...] = result
        return out
    return result


def put(a, ind, v, mode="raise"):
    """numpy.put: a.flat[ind] = v, v repeated as needed (in place)."""
    if not _is_array(a):
        raise TypeError("put requires a charmnumeric array")
    if isinstance(a, ArrayView) and tuple(a.shape) != tuple(a._root_parent.shape):
        raise NotImplementedError("put into a view; use the root array")
    a = a._root_parent if isinstance(a, ArrayView) else a
    n = a.size
    if _is_array(ind):
        L = _wrap_device(ind, n, mode)
        E = tuple(ind.shape)
    else:
        idx = _wrap_host(np.asarray(ind).reshape(-1), n, mode)
        if idx.size == 0:
            return
        L = asarray(np.ascontiguousarray(idx))
        E = idx.shape
    size = int(np.prod(E))
    if _is_array(v):
        v = _cast_operand(v, a.dtype)
        if tuple(v.shape) != E:
            if v.ndims != 1 or len(E) != 1:
                raise NotImplementedError("put with a device value array of another shape")
            v = _flat_gather(v, arange(size, dtype=_INT64) % v.size, "raise")
    elif not _is_scalar_like(v):
        host = np.asarray(v, dtype=a.dtype).reshape(-1)
        v = asarray(np.ascontiguousarray(np.resize(host, size).reshape(E)))
    else:
        v = _coerce_scalar(v)
    _scatter(a, L, v, E, 0, 0, [])


def _along_axis(arr, indices, axis):
    """(root, L, E) for take_along_axis / put_along_axis."""
    if axis is None:
        return None
    nd = arr.ndims
    axis = _axis_index(axis, nd)
    ishape = tuple(indices.shape) if _is_array(indices) else np.shape(indices)
    if len(ishape) != nd:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    for d in range(nd):
        if d != axis and ishape[d] != arr.shape[d]:
            raise NotImplementedError("take/put_along_axis with broadcast index shapes")
    root, base, mult = _geometry(arr)
    n = arr.shape[axis]
    if not _is_array(indices):
        idx = _wrap_host(indices, n, "raise", axis)
        grid = np.indices(ishape, dtype=np.int64)
        L = base + idx * mult[axis]
        for d in range(nd):
            if d != axis:
                L = L + grid[d] * mult[d]
        return root, (asarray(np.ascontiguousarray(L)) if L.size else None), ishape
    if not np.issubdtype(indices.dtype, np.integer):
        raise IndexError("arrays used as indices must be of integer type")
    L = _wrap_device(indices, n, "raise") * mult[axis]
    for d in range(nd):
        if d != axis and ishape[d] > 1:
            L = L + _axis_range(ishape, d, ishape[d], indices.dtype) * mult[d]
    if base:
        L = L + base
    return root, L, ishape


def take_along_axis(arr, indices, axis):
    """numpy.take_along_axis."""
    arr = asarray(arr)
    if axis is None:
        return _flat_gather(arr, indices, "raise")
    root, L, E = _along_axis(arr, indices, axis)
    return _gather(root, L, E, 0, 0, [])


def put_along_axis(arr, indices, values, axis):
    """numpy.put_along_axis (in place)."""
    if axis is None:
        return put(arr, indices, values)
    root, L, E = _along_axis(arr, indices, axis)
    if int(np.prod(E)) == 0:
        return
    _scatter(root, L, _scatter_value(arr, values, E), E, 0, 0, [])


def compress(condition, a, axis=None, out=None):
    """numpy.compress (blocking for a device condition: fetches the count)."""
    a = asarray(a)
    if _is_array(condition):
        if condition.ndims != 1:
            raise ValueError("condition must be a 1-d array")
        F = _nonzero_flat(condition)
        idx = np.zeros(0, np.int64) if F is None else F
    else:
        idx = np.nonzero(np.asarray(condition).astype(bool).reshape(-1))[0]
    return take(a, idx, axis=axis, out=out)


def repeat(a, repeats, axis=None):
    """numpy.repeat (blocking for device repeat counts: fetches their sum)."""
    if not _is_array(a):
        host = np.asarray(a)
        if host.ndim == 0 and not _is_array(repeats):
            r = np.asarray(repeats)
            n = int(r) if r.ndim == 0 else int(r.sum())
            return full((n,), host.item(), dtype=host.dtype)
        a = asarray(host)
    if axis is None:
        a = _flat_source(a)
        n = a.size
    else:
        axis = _axis_index(axis, a.ndims)
        n = a.shape[axis]

    if _is_array(repeats):
        if repeats.ndims != 1 or repeats.shape[0] not in (1, n):
            raise ValueError(f"operands could not be broadcast together with shape ({n},)")
        if repeats.shape[0] == 1 and n != 1:
            raise NotImplementedError("repeat with a size-1 device repeats array")
        from .reductions import sum as _sum

        total = _fetch_int(_sum(repeats))
        idx = _expand(repeats, True, total) if total else np.zeros(0, np.int64)
    elif np.ndim(repeats) == 0:
        r = int(repeats)
        if r < 0:
            raise ValueError("negative dimensions are not allowed")
        if r == 1:
            idx = arange(n, dtype=_INT64)
        else:
            idx = arange(n * r, dtype=_INT64) // r if n * r else np.zeros(0, np.int64)
    else:
        idx = np.repeat(np.arange(n), np.broadcast_to(np.asarray(repeats), (n,)))

    if axis is None:
        return _flat_gather(a, idx, "raise")
    return take(a, idx, axis=axis)


def choose(a, choices, out=None, mode="raise"):
    """numpy.choose (mode 'raise' does not check device indices)."""
    if mode not in ("raise", "wrap", "clip"):
        raise ValueError(f"clipmode must be one of 'clip', 'raise', or 'wrap' (got {mode!r})")
    n = len(choices)
    if n == 0:
        raise ValueError("choices must not be empty")
    a = asarray(a)
    if mode == "wrap":
        idx = a % n
    elif mode == "clip":
        idx = _clip(a, 0, n - 1)
    else:
        idx = a
    choices = [c if _is_array(c) else (_coerce_scalar(c) if _is_scalar_like(c) else asarray(c))
               for c in choices]
    arrays = [c for c in choices if _is_array(c)]
    if all(c.dtype == idx.dtype for c in arrays):
        # a chain of selects, fused into one kernel
        result = choices[n - 1]
        for k in range(n - 2, -1, -1):
            result = _where(_compare(idx, k, "equal"), choices[k], result)
        if not _is_array(result):
            result = full(tuple(idx.shape), result, dtype=idx.dtype)
    else:
        result = _choose_gather(idx, choices, arrays)
    if out is not None:
        out[...] = result
        return out
    return result


def _choose_gather(idx, choices, arrays):
    """choose for choices of another dtype than the index: gather from the stacked choices."""
    shape = tuple(idx.shape)
    if len(shape) > 2:
        raise NotImplementedError("choose with mixed dtypes on 3-D arrays")
    dtype = np.result_type(*[c.dtype for c in arrays])
    stacked = empty((len(choices),) + shape, dtype=dtype)
    for k, c in enumerate(choices):
        stacked[k] = c
    size = int(np.prod(shape))
    L = idx * size
    stride = 1
    for d in range(len(shape) - 1, -1, -1):
        if shape[d] > 1:
            L = L + _axis_range(shape, d, shape[d], idx.dtype) * stride
        stride *= shape[d]
    return _gather(stacked, L, shape, 0, 0, [])


def _method_put(self, indices, values, mode="raise"):
    return put(self, indices, values, mode=mode)


def _method_repeat(self, repeats, axis=None):
    return repeat(self, repeats, axis=axis)


def _method_compress(self, condition, axis=None, out=None):
    return compress(condition, self, axis=axis, out=out)


def _method_choose(self, choices, out=None, mode="raise"):
    return choose(self, choices, out=out, mode=mode)


def _method_nonzero(self):
    return nonzero(self)


for _cls in (Array, ArrayView):
    _cls.put = _method_put
    _cls.repeat = _method_repeat
    _cls.compress = _method_compress
    _cls.choose = _method_choose
    _cls.nonzero = _method_nonzero
del _cls


__all__ = [
    "argwhere",
    "choose",
    "compress",
    "flatnonzero",
    "nonzero",
    "put",
    "put_along_axis",
    "repeat",
    "take",
    "take_along_axis",
]
