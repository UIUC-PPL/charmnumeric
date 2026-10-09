import math
import numbers
import struct

import numpy as np

from charmtyles.ast import InlineData, NonFusibleNode, get_execution_state
from charmtyles.core import (
    OverlapType,
    Region,
    FrontendObject,
    FrontendObjectView,
    create_object,
    create_result,
    execute,
    _shape_size,
)
from charmtyles.operation import Operation, nonfusible_op
from charmtyles.operation import fusible_op as _fusible_op
from charmtyles.interface import to_bytes

from ._native_region import NativeArrayRegion


_FULL_REGION_CACHE = {}
_DEFAULT_DTYPE = np.dtype("float64")


class _CallableInt(int):
    """Int-like object that still supports legacy ``obj.size()`` calls."""

    def __new__(cls, value):
        return int.__new__(cls, int(value))

    def __call__(self):
        return int(self)


class DType(object):
    """Element type encoding shared with the C++ wire protocol."""

    FLOAT32 = 0
    FLOAT64 = 1
    INT32 = 2
    INT64 = 3

    _from_numpy = {
        np.dtype("float32"): FLOAT32,
        np.dtype("float64"): FLOAT64,
        np.dtype("int32"): INT32,
        np.dtype("int64"): INT64,
    }

    @staticmethod
    def from_numpy(np_dtype):
        """Convert a supported numpy dtype to the wire encoding."""

        dtype = _normalize_dtype(np_dtype)
        return DType._from_numpy[dtype]

    _to_numpy = {
        FLOAT32: np.dtype("float32"),
        FLOAT64: np.dtype("float64"),
        INT32: np.dtype("int32"),
        INT64: np.dtype("int64"),
    }

    @staticmethod
    def to_numpy(wire_dtype):
        """Convert a wire dtype encoding back to a numpy dtype."""

        if wire_dtype not in DType._to_numpy:
            raise TypeError(f"Unsupported wire dtype {wire_dtype!r}")
        return DType._to_numpy[wire_dtype]

    @staticmethod
    def promote(a, b):
        """Return the promoted wire dtype for operands *a* and *b*."""

        return DType.from_numpy(np.result_type(DType.to_numpy(a), DType.to_numpy(b)))


def _normalize_dtype(dtype, *, allow_none=False):
    if dtype is None:
        if allow_none:
            return None
        return _DEFAULT_DTYPE

    np_dtype = np.dtype(dtype)
    if np_dtype not in DType._from_numpy:
        raise TypeError(
            "charmnumeric only supports float32, float64, int32, and int64; "
            f"got {np_dtype}"
        )
    return np_dtype


def _normalize_shape(shape):
    if isinstance(shape, numbers.Integral):
        normalized = (int(shape),)
    else:
        normalized = tuple(int(dim) for dim in shape)

    for dim in normalized:
        if dim < 0:
            raise ValueError(f"negative dimensions are not allowed: {normalized}")
    return normalized


def _coerce_scalar(value):
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return value
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def _is_scalar_like(value):
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    if isinstance(value, np.generic):
        return True
    return isinstance(value, (bool, numbers.Real))


def _apply_ndmin(shape, ndmin):
    ndmin = int(ndmin)
    if ndmin < 0:
        raise ValueError("ndmin must be non-negative")
    if len(shape) >= ndmin:
        return tuple(shape)
    return (1,) * (ndmin - len(shape)) + tuple(shape)


class ArrayOperation(Operation):
    add = 0
    sub = 1
    mul = 2
    div = 3
    matmul = 4
    tanh = 5
    exp = 6
    tile = 7
    reduce = 8
    matmatmul = 9
    diag = 10
    # elementwise math, comparisons and selection (20-49, see src/opcodes.hpp)
    sqrt = 20
    log = 21
    log2 = 22
    log10 = 23
    log1p = 24
    exp2 = 25
    expm1 = 26
    abs = 27
    floor = 28
    ceil = 29
    rint = 30
    isfinite = 31
    sin = 32
    cos = 33
    sign = 34
    isnan = 35
    isinf = 36
    power = 37
    maximum = 38
    minimum = 39
    fmax = 40
    fmin = 41
    mod = 42
    floor_divide = 43
    less = 44
    less_equal = 45
    equal = 46
    not_equal = 47
    where = 48
    clip = 49
    # Array creation (opcodes 50-59): each node creates and fills a new array.
    upload = 50  # bulk host data carried in the DAG (INLINE_DATA operand)
    random = 51  # counter-based RNG (charmnumeric/random.py)
    iota = 52    # arange / linspace / indices / eye
    reduction = 60
    bincount = 61
    transpose = 70
    # 80-89: non-fused cast / scan / sort executors (src/executor_castscansort.cpp)
    astype = 80
    scan = 81
    sort = 82
    argsort = 83
    linalg = 100  # factorizations and solves (charmnumeric/linalg.py)


class ArrayRegion(Region):
    __slots__ = (
        "start",
        "stop",
        "step",
        "_shape",
        "_hash",
        "_serialized",
        "_native",
    )

    def __init__(self, start, stop, step):
        super().__init__()
        self.start = tuple(start)
        self.stop = tuple(stop)
        self.step = tuple(step)
        self._shape = tuple(
            (self.stop[i] - self.start[i] + self.step[i] - 1) // self.step[i]
            for i in range(len(self.start))
        )
        self._hash = hash((self.start, self.stop, self.step))
        self._serialized = None
        self._native = NativeArrayRegion(self.start, self.stop, self.step, False)

    def __str__(self):
        return f"({self.start}, {self.stop}, {self.step})"

    def __hash__(self):
        return self._hash

    def __eq__(self, value):
        if not isinstance(value, Region):
            return False
        if self.is_global or value.is_global:
            return True
        return (
            self.start == value.start
            and self.stop == value.stop
            and self.step == value.step
        )

    def serialize(self):
        if self._serialized is not None:
            return self._serialized

        payload = bytearray()
        payload.extend(to_bytes(1 if self.is_global else 0, "i"))
        if not self.is_global:
            payload.extend(to_bytes(len(self.start), "i"))
            for i in range(len(self.start)):
                payload.extend(to_bytes(self.start[i], "i"))
                payload.extend(to_bytes(self.stop[i], "i"))
                payload.extend(to_bytes(self.step[i], "i"))
        self._serialized = bytes(payload)
        return self._serialized

    def shape(self):
        return self._shape

    def compose(self, inner):
        """Compose an inner sub-region within this outer region."""

        if (
            inner.start == (0,) * len(inner.start)
            and inner.step == (1,) * len(inner.step)
            and inner.stop == self._shape
        ):
            return self

        ndims = len(inner.start)
        new_start = [0] * ndims
        new_stop = [0] * ndims
        new_step = [0] * ndims
        for d in range(ndims):
            new_start[d] = self.start[d] + inner.start[d] * self.step[d]
            new_stop[d] = self.start[d] + inner.stop[d] * self.step[d]
            new_step[d] = self.step[d] * inner.step[d]
        return ArrayRegion(new_start, new_stop, new_step)

    def overlaps(self, other):
        if not isinstance(other, ArrayRegion):
            raise TypeError(
                f"ArrayRegion overlap requires ArrayRegion, got {type(other).__name__}"
            )
        return self._native.overlaps(other._native)

    def covers(self, other):
        if not isinstance(other, ArrayRegion):
            raise TypeError(
                f"ArrayRegion cover check requires ArrayRegion, got {type(other).__name__}"
            )
        return self._native.covers(other._native)

    def intersect(self, other):
        if not isinstance(other, ArrayRegion):
            raise TypeError(
                f"ArrayRegion intersection requires ArrayRegion, got {type(other).__name__}"
            )
        result = self._native.intersect(other._native)
        if result is None:
            return None
        start, stop, step = result
        return ArrayRegion(start, stop, step)


def _full_region(shape):
    key = tuple(shape)
    region = _FULL_REGION_CACHE.get(key)
    if region is None:
        ndims = len(key)
        region = ArrayRegion(
            start=(0,) * ndims,
            stop=key,
            step=(1,) * ndims,
        )
        _FULL_REGION_CACHE[key] = region
    return region


def _expand_key(key, ndims, keep_newaxis=False):
    """Expand ellipsis and pad with full slices to ``ndims`` entries.

    ``None`` (np.newaxis) entries are dropped unless ``keep_newaxis`` is set,
    in which case they stay in place and do not count towards ``ndims``.
    """
    if key is Ellipsis:
        key = (Ellipsis,)
    elif not isinstance(key, tuple):
        key = (key,)

    normalized = []
    saw_ellipsis = False
    explicit_dims = sum(1 for item in key if item is not Ellipsis and item is not None)

    for item in key:
        if item is Ellipsis:
            if saw_ellipsis:
                raise IndexError("an index can only have a single ellipsis")
            saw_ellipsis = True
            fill = ndims - explicit_dims
            if fill < 0:
                raise IndexError("too many indices for array")
            normalized.extend(slice(None) for _ in range(fill))
        elif item is None:
            if keep_newaxis:
                normalized.append(None)
        else:
            normalized.append(item)

    indexed = sum(1 for item in normalized if item is not None)
    if indexed > ndims:
        raise IndexError(
            f"too many indices for array: array is {ndims}-dimensional, "
            f"but {indexed} were indexed"
        )

    if indexed < ndims:
        normalized.extend(slice(None) for _ in range(ndims - indexed))

    return tuple(normalized)


def _has_newaxis(key):
    if key is None:
        return True
    return isinstance(key, tuple) and any(item is None for item in key)


def _newaxis_shape(key, view_shape, ndims):
    """Shape of ``x[key]`` where ``view_shape`` is the shape without newaxes."""

    dims = iter(view_shape)
    return tuple(
        1 if item is None else next(dims) for item in _expand_key(key, ndims, keep_newaxis=True)
    )


def _parse_key(key, shape, ndims):
    """Normalize a ``__getitem__`` or ``__setitem__`` key into an ArrayRegion."""

    key = _expand_key(key, ndims)
    start = [0] * ndims
    stop = [0] * ndims
    step = [1] * ndims

    for i, item in enumerate(key):
        if isinstance(item, slice):
            step_i = 1 if item.step is None else int(item.step)
            if step_i == 0:
                raise ValueError("slice step cannot be zero")
            if step_i < 0:
                raise NotImplementedError("negative slicing is not supported yet")
            start_i, stop_i, step_i = item.indices(shape[i])
        elif isinstance(item, numbers.Integral):
            start_i = int(item)
            if start_i < 0:
                start_i += shape[i]
            if start_i < 0 or start_i >= shape[i]:
                raise IndexError(
                    f"index {item} is out of bounds for axis {i} with size {shape[i]}"
                )
            stop_i = start_i + 1
            step_i = 1
        else:
            raise TypeError(
                "only integers, slices, and ellipsis are valid charmnumeric indices"
            )

        start[i] = start_i
        stop[i] = stop_i
        step[i] = step_i

    return ArrayRegion(start, stop, step)


def _scalar_view_operand(value):
    """Copy a size-1 view of a larger array into its own size-1 array.

    Only whole size-1 arrays are broadcast as scalars by the backend (it reads
    element 0 of the operand's array), so x[k:k+1] has to be materialized
    before it is broadcast against a larger operand or assignment target.
    """
    if (isinstance(value, ArrayView) and value.size() == 1
            and value._root_parent.size() > 1):
        return value.copy()
    return value


def fusible_op(*operands, operation, **kwargs):
    shape = kwargs.get("shape")
    if shape is not None and _shape_size(shape) > 1:
        operands = tuple(_scalar_view_operand(op) for op in operands)
    return _fusible_op(*operands, operation=operation, **kwargs)


def _broadcast_shape(lhs_shape, rhs):
    if isinstance(rhs, (Array, ArrayView)):
        rhs_shape = rhs.shape
    else:
        return tuple(lhs_shape)

    try:
        return tuple(np.broadcast_shapes(lhs_shape, rhs_shape))
    except ValueError as exc:
        raise ValueError(
            f"operands could not be broadcast together with shapes "
            f"{lhs_shape} and {rhs_shape}"
        ) from exc


def _operand_dtype(value):
    if isinstance(value, (Array, ArrayView)):
        return value.dtype
    # Scalars do not promote the array dtype (matches numpy value-based casting:
    # a Python float does not upcast a float32 array to float64).
    return None


def _result_dtype(lhs, rhs):
    other_dtype = _operand_dtype(rhs)
    if other_dtype is None:
        # A Python float promotes an int array to float64 (NEP 50).
        if (
            isinstance(rhs, float)
            and lhs.dtype is not None
            and np.issubdtype(lhs.dtype, np.integer)
        ):
            return _DEFAULT_DTYPE
        return lhs.dtype
    return _normalize_dtype(np.result_type(lhs.dtype, other_dtype))


def _astype(x, dtype):
    """Non-fused CAST node: a new array of ``dtype`` with x's shape."""
    dtype = _normalize_dtype(dtype)
    x = _materialize(x)
    if x.dtype == dtype:
        return x
    return nonfusible_op(x, operation=ArrayOperation.astype, shape=x.shape, dtype=dtype)


def _cast_operand(value, dtype):
    if isinstance(value, (Array, ArrayView)) and value.dtype != dtype:
        return _astype(value, dtype)
    return value


def _promote_operands(lhs, rhs):
    """Cast array operands to the result dtype (kernels use one element type).

    Returns ``(lhs, rhs, dtype)``; ``lhs`` must be an Array or ArrayView.
    """
    dtype = _result_dtype(lhs, rhs)
    return _cast_operand(lhs, dtype), _cast_operand(rhs, dtype), dtype


def _normalize_axis(axis, ndims):
    axis = int(axis)
    if axis < -ndims or axis >= ndims:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndims}")
    return axis % ndims


_SCAN_KINDS = {"cumsum": 0, "cumprod": 1, "nancumsum": 2, "nancumprod": 3}


def _scan(name, a, axis=None, dtype=None, out=None):
    """cumsum / cumprod / nancumsum / nancumprod along one axis (SCAN node)."""
    a = asarray(a)
    if axis is None:
        if a.ndims != 1:
            raise NotImplementedError(
                f"{name}(axis=None) on a {a.ndims}-D array needs reshape (not supported)"
            )
        axis = 0
    axis = _normalize_axis(axis, a.ndims)
    if dtype is None:
        # NumPy accumulates small integers in the platform integer (int64).
        dtype = a.dtype
        if np.issubdtype(dtype, np.integer) and dtype.itemsize < np.dtype(np.int_).itemsize:
            dtype = np.dtype(np.int_)
    dtype = _normalize_dtype(dtype)
    src = _astype(a, dtype)
    result = nonfusible_op(
        src,
        axis,
        _SCAN_KINDS[name],
        operation=ArrayOperation.scan,
        shape=src.shape,
        dtype=dtype,
    )
    if out is not None:
        out[...] = result
        return out
    return result


def _sort(a, axis=-1, kind=None, order=None, arg=False):
    """sort / argsort along one axis (SORT / ARGSORT node).

    Gathers the array on one chare (not scalable). NaNs last, always stable.
    """
    if order is not None:
        raise NotImplementedError("sort(order=...) needs structured dtypes")
    a = asarray(a)
    if axis is None:
        if a.ndims != 1:
            raise NotImplementedError(
                f"sort(axis=None) on a {a.ndims}-D array needs reshape (not supported)"
            )
        axis = 0
    axis = _normalize_axis(axis, a.ndims)
    src = _materialize(a)
    return nonfusible_op(
        src,
        axis,
        operation=ArrayOperation.argsort if arg else ArrayOperation.sort,
        shape=src.shape,
        dtype=np.dtype(np.int64) if arg else src.dtype,
    )


def _coerce_operand(value):
    if isinstance(value, (Array, ArrayView)):
        return value
    if _is_scalar_like(value):
        return _coerce_scalar(value)
    if isinstance(value, (str, bytes)):
        return value

    try:
        host = np.asarray(value)
    except Exception:
        return value

    if host.ndim == 0:
        return _coerce_scalar(host.item())
    return asarray(host)


def _array_add(self, other):
    other = _coerce_operand(other)
    self, other, dtype = _promote_operands(self, other)
    return fusible_op(
        self,
        other,
        operation=ArrayOperation.add,
        shape=_broadcast_shape(self.shape, other),
        dtype=dtype,
    )


def _array_radd(self, other):
    other = _coerce_operand(other)
    self, other, dtype = _promote_operands(self, other)
    return fusible_op(
        other,
        self,
        operation=ArrayOperation.add,
        shape=_broadcast_shape(self.shape, other),
        dtype=dtype,
    )


def _array_sub(self, other):
    other = _coerce_operand(other)
    self, other, dtype = _promote_operands(self, other)
    return fusible_op(
        self,
        other,
        operation=ArrayOperation.sub,
        shape=_broadcast_shape(self.shape, other),
        dtype=dtype,
    )


def _array_neg(self):
    return fusible_op(
        self,
        -1,
        operation=ArrayOperation.mul,
        shape=self.shape,
        dtype=self.dtype,
    )


def _array_rsub(self, other):
    other = _coerce_operand(other)
    self, other, dtype = _promote_operands(self, other)
    return fusible_op(
        other,
        self,
        operation=ArrayOperation.sub,
        shape=_broadcast_shape(self.shape, other),
        dtype=dtype,
    )


def _array_mul(self, other):
    other = _coerce_operand(other)
    self, other, dtype = _promote_operands(self, other)
    return fusible_op(
        self,
        other,
        operation=ArrayOperation.mul,
        shape=_broadcast_shape(self.shape, other),
        dtype=dtype,
    )


def _array_rmul(self, other):
    other = _coerce_operand(other)
    self, other, dtype = _promote_operands(self, other)
    return fusible_op(
        other,
        self,
        operation=ArrayOperation.mul,
        shape=_broadcast_shape(self.shape, other),
        dtype=dtype,
    )


def _array_truediv(self, other):
    other = _coerce_operand(other)
    self, other, dtype = _promote_operands(self, other)
    return fusible_op(
        self,
        other,
        operation=ArrayOperation.div,
        shape=_broadcast_shape(self.shape, other),
        dtype=dtype,
    )


def _array_rtruediv(self, other):
    other = _coerce_operand(other)
    self, other, dtype = _promote_operands(self, other)
    return fusible_op(
        other,
        self,
        operation=ArrayOperation.div,
        shape=_broadcast_shape(self.shape, other),
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Elementwise math, comparisons and selection (opcodes 20-49).
#
# Comparisons and predicates (less, equal, isnan, logical_and, ...) return
# 0/1 *masks in the operands' dtype*, not bool arrays: a fused kernel has a
# single element type and there is no bool dtype. A mask works as the
# condition of where(), in arithmetic (mask * x) and in further logical ops.
#
# One element type per kernel also means: array operands of these ops must
# share a dtype (no int/float promotion), and a Python float scalar that is
# not integral cannot be combined with an integer array (NumPy would promote
# to float64; there is no in-kernel conversion). Float-only math (sqrt, log,
# sin, ...) rejects integer arrays for the same reason. Integer arrays keep
# their dtype through abs/floor/ceil/rint/sign/round.
#
# Broadcasting is limited to what the backend supports: every array operand
# has the result shape or has size 1.
# ---------------------------------------------------------------------------


def _is_array(value):
    return isinstance(value, (Array, ArrayView))


def _is_int_dtype(dtype):
    return np.issubdtype(dtype, np.integer)


def _ew_operands(name, *operands):
    """Coerce elementwise operands; return ``(operands, shape, dtype)``."""

    operands = tuple(_coerce_operand(value) for value in operands)
    arrays = [value for value in operands if _is_array(value)]
    if not arrays:
        raise TypeError(f"{name}: at least one operand must be a charmnumeric array")

    # Kernels use one element type: promote array operands like NumPy.
    dtype = np.result_type(*[value.dtype for value in arrays])
    if any(value.dtype != dtype for value in arrays):
        operands = tuple(_cast_operand(v, dtype) if _is_array(v) else v for v in operands)
        arrays = [value for value in operands if _is_array(value)]

    full = [value for value in arrays if value.size != 1]
    shape = full[0].shape if full else arrays[0].shape
    for value in full[1:]:
        if value.shape != shape:
            raise ValueError(
                f"{name}: operands could not be broadcast together with shapes "
                f"{shape} and {value.shape} (only size-1 operands broadcast)"
            )

    for value in operands:
        if _is_array(value):
            continue
        if not isinstance(value, numbers.Real):
            raise TypeError(f"{name}: unsupported operand type {type(value).__name__}")
        if _is_int_dtype(dtype) and isinstance(value, float) and not value.is_integer():
            raise TypeError(
                f"{name}: float scalar {value!r} with an {dtype} array would promote "
                "to float64; charmnumeric has no int->float conversion"
            )
    return operands, shape, dtype


def _ew_unary(x, operation, name, float_only=False):
    (x,), shape, dtype = _ew_operands(name, x)
    if float_only and _is_int_dtype(dtype):
        raise TypeError(
            f"{name}: integer input would give float64 in NumPy; charmnumeric has no "
            "int->float conversion, so pass a float array"
        )
    return fusible_op(x, operation=operation, shape=shape, dtype=dtype)


def _ew_binary(lhs, rhs, operation, name):
    (lhs, rhs), shape, dtype = _ew_operands(name, lhs, rhs)
    return fusible_op(lhs, rhs, operation=operation, shape=shape, dtype=dtype)


def _ew_ternary(a, b, c, operation, name):
    (a, b, c), shape, dtype = _ew_operands(name, a, b, c)
    return fusible_op(a, b, c, operation=operation, shape=shape, dtype=dtype)


# name -> (opcode, swap operands); greater* are less* with swapped operands
_COMPARISONS = {
    "less": (ArrayOperation.less, False),
    "less_equal": (ArrayOperation.less_equal, False),
    "greater": (ArrayOperation.less, True),
    "greater_equal": (ArrayOperation.less_equal, True),
    "equal": (ArrayOperation.equal, False),
    "not_equal": (ArrayOperation.not_equal, False),
}
_MIRRORED = {
    "less": "greater",
    "less_equal": "greater_equal",
    "greater": "less",
    "greater_equal": "less_equal",
    "equal": "equal",
    "not_equal": "not_equal",
}


def _const_mask(x, value):
    """All-ones (value=1) or all-zeros (value=0) mask shaped like integer array x."""

    operation = ArrayOperation.equal if value else ArrayOperation.not_equal
    return _ew_binary(x, x, operation, "mask")


def _compare_int_float(x, value, name):
    """``x <name> value`` for an integer array and a float scalar, exactly.

    The kernel would truncate the scalar to the integer type, so rewrite the
    comparison with an integer bound (or a constant mask) instead.
    """

    info = np.iinfo(x.dtype)
    if np.isnan(value):
        return _const_mask(x, name == "not_equal")
    if value > info.max or value < info.min:
        above = value > info.max  # the scalar exceeds every element
        result = {
            "less": above,
            "less_equal": above,
            "greater": not above,
            "greater_equal": not above,
            "equal": False,
            "not_equal": True,
        }[name]
        return _const_mask(x, result)
    if value.is_integer():
        return _compare(x, int(value), name)
    if name in ("equal", "not_equal"):
        return _const_mask(x, name == "not_equal")
    bound = {
        "less": int(np.ceil(value)),
        "less_equal": int(np.floor(value)),
        "greater": int(np.floor(value)),
        "greater_equal": int(np.ceil(value)),
    }[name]
    return _compare(x, bound, name)


def _compare(lhs, rhs, name):
    lhs = _coerce_operand(lhs)
    rhs = _coerce_operand(rhs)
    if not _is_array(lhs) and _is_array(rhs):
        lhs, rhs, name = rhs, lhs, _MIRRORED[name]
    if _is_array(lhs) and _is_int_dtype(lhs.dtype) and isinstance(rhs, float):
        return _compare_int_float(lhs, rhs, name)
    operation, swap = _COMPARISONS[name]
    if swap:
        lhs, rhs = rhs, lhs
    return _ew_binary(lhs, rhs, operation, name)


def _truth(value):
    """0/1 mask (array) or 0/1 int (scalar) for value != 0."""

    value = _coerce_operand(value)
    if _is_array(value):
        return _compare(value, 0, "not_equal")
    return int(bool(value))


def _logical_not(x):
    x = _coerce_operand(x)
    if not _is_array(x):
        return not x
    return _compare(x, 0, "equal")


def _logical_and(a, b):
    a, b = _truth(a), _truth(b)
    if not _is_array(a) and not _is_array(b):
        return bool(a and b)
    return _array_mul(a, b) if _is_array(a) else _array_rmul(b, a)


def _logical_or(a, b):
    a, b = _truth(a), _truth(b)
    if not _is_array(a) and not _is_array(b):
        return bool(a or b)
    return _ew_binary(a, b, ArrayOperation.maximum, "logical_or")


def _logical_xor(a, b):
    a, b = _truth(a), _truth(b)
    if not _is_array(a) and not _is_array(b):
        return bool(a) != bool(b)
    return _compare(a, b, "not_equal")


def _power(base, exponent):
    base = _coerce_operand(base)
    exponent = _coerce_operand(exponent)
    if _is_array(base) and not _is_array(exponent) and isinstance(exponent, numbers.Real):
        if _is_int_dtype(base.dtype) and isinstance(exponent, int) and exponent < 0:
            raise ValueError("Integers to negative integer powers are not allowed.")
        if exponent == 2:
            # same fast path as NumPy (x**2 -> square); pow(x, 2) == x*x exactly
            return _array_mul(base, base)
    return _ew_binary(base, exponent, ArrayOperation.power, "power")


def _reciprocal(x):
    x = _coerce_operand(x)
    if _is_array(x) and _is_int_dtype(x.dtype):
        # integer 1/x truncates: 1 for 1, -1 for -1, else 0 (0 also for x == 0)
        return _compare(x, 1, "equal") - _compare(x, -1, "equal")
    return _ew_binary(1, x, ArrayOperation.div, "reciprocal")


def _round(x, decimals=0):
    """Round half to even, with NumPy's algorithm for ``decimals != 0``."""

    (x,), _shape, dtype = _ew_operands("round", x)
    decimals = int(decimals)
    if _is_int_dtype(dtype):
        if decimals >= 0:
            return x.copy()
        raise NotImplementedError("round of integer arrays with decimals < 0")
    if decimals == 0:
        return _ew_unary(x, ArrayOperation.rint, "round")
    if decimals > 0:
        factor = float(10.0**decimals)
        return _ew_unary(x * factor, ArrayOperation.rint, "round") / factor
    factor = float(10.0 ** (-decimals))
    return _ew_unary(x / factor, ArrayOperation.rint, "round") * factor


def _trunc(x):
    (x,), _shape, dtype = _ew_operands("trunc", x)
    if _is_int_dtype(dtype):
        return x.copy()
    return _where(
        _compare(x, 0, "less"),
        _ew_unary(x, ArrayOperation.ceil, "trunc"),
        _ew_unary(x, ArrayOperation.floor, "trunc"),
    )


def _where(condition, x, y):
    condition = _coerce_operand(condition)
    if not _is_array(condition):
        condition = int(bool(condition))
    else:
        # The condition takes no part in dtype promotion (NumPy); pass it as an
        # exact 0/1 mask in the values' dtype.
        values = [v for v in (_coerce_operand(x), _coerce_operand(y)) if _is_array(v)]
        if values:
            dtype = np.result_type(*[v.dtype for v in values])
            if condition.dtype != dtype:
                condition = _cast_operand(_compare(condition, 0, "not_equal"), dtype)
    return _ew_ternary(condition, x, y, ArrayOperation.where, "where")


def _clip(a, a_min=None, a_max=None):
    if a_min is None and a_max is None:
        raise ValueError("One of max or min must be given")
    if a_min is None:
        return _ew_binary(a, a_max, ArrayOperation.minimum, "clip")
    if a_max is None:
        return _ew_binary(a, a_min, ArrayOperation.maximum, "clip")
    return _ew_ternary(a, a_min, a_max, ArrayOperation.clip, "clip")


def _log(x, base=None):
    if base is None or base == np.e:
        return _ew_unary(x, ArrayOperation.log, "log", float_only=True)
    if base == 2:
        return _ew_unary(x, ArrayOperation.log2, "log2", float_only=True)
    if base == 10:
        return _ew_unary(x, ArrayOperation.log10, "log10", float_only=True)
    return _ew_unary(x, ArrayOperation.log, "log", float_only=True) * float(1.0 / np.log(base))


def _array_pow(self, other):
    return _power(self, other)


def _array_rpow(self, other):
    return _power(other, self)


def _array_mod(self, other):
    return _ew_binary(self, other, ArrayOperation.mod, "remainder")


def _array_rmod(self, other):
    return _ew_binary(other, self, ArrayOperation.mod, "remainder")


def _array_floordiv(self, other):
    return _ew_binary(self, other, ArrayOperation.floor_divide, "floor_divide")


def _array_rfloordiv(self, other):
    return _ew_binary(other, self, ArrayOperation.floor_divide, "floor_divide")


def _array_divmod(self, other):
    return _array_floordiv(self, other), _array_mod(self, other)


def _array_rdivmod(self, other):
    return _array_rfloordiv(self, other), _array_rmod(self, other)


def _array_lt(self, other):
    return _compare(self, other, "less")


def _array_le(self, other):
    return _compare(self, other, "less_equal")


def _array_gt(self, other):
    return _compare(self, other, "greater")


def _array_ge(self, other):
    return _compare(self, other, "greater_equal")


def _array_eq(self, other):
    if other is None:
        return False
    return _compare(self, other, "equal")


def _array_ne(self, other):
    if other is None:
        return True
    return _compare(self, other, "not_equal")


def _array_abs(self):
    return _ew_unary(self, ArrayOperation.abs, "absolute")


def _array_pos(self):
    return self.copy()


class _ElementwiseMethods(object):
    """Elementwise methods shared by Array and ArrayView.

    NumPy's ndarray methods (round, clip) plus the method API of the original
    charmnumeric backend (branch fix-temporaries: sqrt, log(base), log2,
    log10, abs, where, power, modulo, max/min(other), ...).
    """

    __slots__ = ()

    # NumPy ndarray methods
    def round(self, decimals=0):
        return _round(self, decimals)

    def clip(self, min=None, max=None):
        return _clip(self, min, max)

    # Methods of the fix-temporaries backend
    def exp(self):
        return fusible_op(self, operation=ArrayOperation.exp, shape=self.shape, dtype=self.dtype)

    def tanh(self):
        return fusible_op(self, operation=ArrayOperation.tanh, shape=self.shape, dtype=self.dtype)

    def log(self, base=np.e):
        return _log(self, base)

    def log2(self):
        return _log(self, 2)

    def log10(self):
        return _log(self, 10)

    def sqrt(self):
        return _ew_unary(self, ArrayOperation.sqrt, "sqrt", float_only=True)

    def abs(self):
        return _array_abs(self)

    def absolute(self):
        return _array_abs(self)

    def negate(self):
        return -self

    def square(self):
        return _array_mul(self, self)

    def reciprocal(self):
        return _reciprocal(self)

    def sin(self):
        return _ew_unary(self, ArrayOperation.sin, "sin", float_only=True)

    def cos(self):
        return _ew_unary(self, ArrayOperation.cos, "cos", float_only=True)

    def relu(self):
        return _ew_binary(self, 0, ArrayOperation.maximum, "relu")

    def scale(self, scalar):
        return self * scalar

    def add_constant(self, constant):
        return self + constant

    def add(self, other):
        return self + other

    def subtract(self, other):
        return self - other

    def multiply(self, other):
        return self * other

    def divide(self, other):
        return self / other

    def modulo(self, other):
        return _array_mod(self, other)

    def power(self, other):
        return _power(self, other)

    # max/min (NumPy reductions, plus the old elementwise a.max(other) form) are
    # installed on Array/ArrayView by reductions.py.

    def greater_than(self, other):
        return _compare(self, other, "greater")

    def less_than(self, other):
        return _compare(self, other, "less")

    def equal(self, other, epsilon=None):
        """Exact equality mask, or ``|self - other| <= epsilon`` when epsilon is given."""

        if epsilon is None:
            return _compare(self, other, "equal")
        return _compare(_array_abs(self - other), epsilon, "less_equal")

    def where(self, x, y):
        """``where(self, x, y)``: self is the condition (old backend API)."""

        return _where(self, x, y)

    def weighted_average(self, other, w1, w2):
        return self * w1 + other * w2


def _array_matmul(self, other):
    other = _coerce_operand(other)
    if not isinstance(other, (Array, ArrayView)):
        raise ValueError("matmul does not support scalar operands")
    self, other, _ = _promote_operands(self, other)

    if self.ndims == 1 and other.ndims == 1:
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                f"Shape mismatch for dot product: ({self.shape[0]},) @ ({other.shape[0]},)"
            )
        return nonfusible_op(
            _materialize(self),
            _materialize(other),
            operation=ArrayOperation.reduce,
            shape=(1,),
            dtype=_result_dtype(self, other),
        )

    if self.ndims == 3:
        dropped = [i for i in range(3) if self.shape[i] == 1]
        if len(dropped) == 1:
            if other.ndims == 1:
                return _matvec(self, other)
            if other.ndims >= 2:
                return _matmatmul(self, other)
        raise ValueError(
            f"3D matmul requires exactly one singleton dim, got shape {self.shape}"
        )

    if self.ndims == 1 and other.ndims == 2:
        # vecmat: x @ A == A.T @ x
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                f"Shape mismatch for vecmat: ({self.shape[0]},) @ {other.shape}"
            )
        return _matvec(_cached_transpose(other), self)

    if self.ndims != 2:
        raise ValueError(f"matmul requires a 2D matrix, got {self.ndims}D")

    if other.ndims == 1:
        return _matvec(self, other)
    if other.ndims >= 2:
        if other.ndims == 2 and other.shape[1] == 1:
            # (M, K) @ (K, 1) -> (M, 1): run it as a matvec, keep the 2-D shape
            temp = create_array((other.shape[0],), dtype=other.dtype)
            temp[:] = other
            return _reshape_copy(_matvec(self, temp), (self.shape[0], 1))
        return _matmatmul(self, other)

    raise ValueError(f"matmul requires a 2D operand, got {other.ndims}D")




def _matvec(mat_obj, vec):
    """Matrix-vector multiply."""

    if not isinstance(vec, (Array, ArrayView)):
        raise ValueError("matvec requires an Array vector")

    if vec.ndims == 2 and vec.shape[1] == 1:
        temp = create_array((vec.shape[0],), dtype=vec.dtype)
        temp[:] = vec
        return _matvec(mat_obj, temp)

    if vec.ndims != 1:
        raise ValueError("matvec requires a 1D vector (N,) or 2D column vector (N, 1)")

    if mat_obj.ndims == 2:
        mat_rows, mat_cols = mat_obj.shape[0], mat_obj.shape[1]
    elif mat_obj.ndims == 3:
        dropped = [i for i in range(3) if mat_obj.shape[i] == 1]
        if len(dropped) != 1:
            raise ValueError(
                "3D matvec requires exactly one singleton dimension, "
                f"got shape {mat_obj.shape}"
            )
        remaining = [i for i in range(3) if i not in dropped]
        mat_rows = mat_obj.shape[remaining[0]]
        mat_cols = mat_obj.shape[remaining[1]]
    else:
        raise ValueError(f"matvec requires a 2D or 3D matrix, got {mat_obj.ndims}D")

    if mat_cols != vec.shape[0]:
        raise ValueError(
            f"Shape mismatch: matrix cols {mat_cols} != vector size {vec.shape[0]}"
        )

    return nonfusible_op(
        mat_obj,
        vec,
        operation=ArrayOperation.matmul,
        shape=(mat_rows,),
        dtype=_result_dtype(mat_obj, vec),
    )


def _matmatmul_shape(obj):
    """Return ``(rows, cols)`` for a matmatmul operand."""

    if obj.ndims == 2:
        return obj.shape[0], obj.shape[1]
    if obj.ndims == 3:
        dropped = [i for i in range(3) if obj.shape[i] == 1]
        if len(dropped) != 1:
            raise ValueError(
                f"3D matmatmul requires exactly one singleton dim, got shape {obj.shape}"
            )
        remaining = [i for i in range(3) if i not in dropped]
        return obj.shape[remaining[0]], obj.shape[remaining[1]]
    raise ValueError(f"matmatmul requires 2D or 3D operand, got {obj.ndims}D")


def _matmatmul(lhs, rhs):
    """Matrix-matrix multiply with inline region serialization."""

    if not isinstance(rhs, (Array, ArrayView)):
        raise ValueError("matmatmul requires Array operands")

    lhs_rows, lhs_cols = _matmatmul_shape(lhs)
    rhs_rows, rhs_cols = _matmatmul_shape(rhs)

    if lhs_cols != rhs_rows:
        raise ValueError(
            f"Shape mismatch: ({lhs_rows},{lhs_cols}) @ ({rhs_rows},{rhs_cols})"
        )

    return nonfusible_op(
        lhs,
        rhs,
        operation=ArrayOperation.matmatmul,
        shape=(lhs_rows, rhs_cols),
        dtype=_result_dtype(lhs, rhs),
    )


def _normalize_axes(axes, ndim):
    """Return a full axis permutation tuple for ``transpose(axes)``."""

    if axes is None:
        return tuple(range(ndim - 1, -1, -1))
    if isinstance(axes, numbers.Integral):
        axes = (axes,)
    axes = tuple(int(axis) for axis in axes)
    if len(axes) != ndim:
        raise ValueError("axes don't match array")
    perm = []
    for axis in axes:
        if axis < -ndim or axis >= ndim:
            raise np.exceptions.AxisError(axis, ndim)
        perm.append(axis % ndim)
    if len(set(perm)) != ndim:
        raise ValueError("repeated axis in transpose")
    return tuple(perm)


def _transpose(x, axes=None):
    """Permute the axes of ``x``. Returns a new array, not a view (1-D or
    identity permutations return ``x``)."""

    perm = _normalize_axes(axes, x.ndims)
    if perm == tuple(range(x.ndims)):
        return x
    if x.ndims > 3:
        raise NotImplementedError(
            f"transpose of a {x.ndims}D array is not supported (max 3 dims)"
        )

    shape = tuple(x.shape[axis] for axis in perm)
    if _shape_size(shape) <= 1:
        # size-1 operands are sent as broadcast scalars
        result = empty(shape, dtype=x.dtype)
        if _shape_size(shape) == 1:
            result[...] = x
        return result

    return nonfusible_op(
        x,
        *perm,
        operation=ArrayOperation.transpose,
        shape=shape,
        dtype=x.dtype,
    )


def _swapaxes(x, axis1, axis2):
    ndim = x.ndims
    for axis in (axis1, axis2):
        if axis < -ndim or axis >= ndim:
            raise np.exceptions.AxisError(axis, ndim)
    perm = list(range(ndim))
    a, b = int(axis1) % ndim, int(axis2) % ndim
    perm[a], perm[b] = perm[b], perm[a]
    return _transpose(x, perm)


def _transpose_method(self, *axes):
    if len(axes) == 0 or (len(axes) == 1 and axes[0] is None):
        return _transpose(self)
    if len(axes) == 1 and not isinstance(axes[0], numbers.Integral):
        return _transpose(self, axes[0])
    return _transpose(self, axes)


_TRANSPOSE_CACHE_MAX = 4


def _cached_transpose(x):
    """Internal transpose for vecmat; cached until the root array is written."""

    root = x._root_parent if isinstance(x, ArrayView) else x
    cache = root.__dict__.setdefault("_transpose_cache", {})
    region = x.region
    key = (tuple(region.start), tuple(region.stop), tuple(region.step))
    hit = cache.get(key)
    if hit is None:
        hit = _transpose(x)
        if len(cache) >= _TRANSPOSE_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = hit
    return hit


class Array(FrontendObject, _ElementwiseMethods):
    __array_priority__ = 1000
    # __eq__ is elementwise; keep FrontendObject's name-based hash
    __hash__ = FrontendObject.__hash__

    def __init__(self, **kwargs):
        dtype = _normalize_dtype(kwargs.pop("dtype", None), allow_none=True)
        self.dtype = dtype
        self._wire_dtype = DType.from_numpy(self.dtype) if self.dtype is not None else 0

        shape = kwargs.pop("shape", None)
        if shape is not None:
            shape = _normalize_shape(shape)

        super().__init__(**kwargs)

        self._shape = tuple(shape) if shape is not None else tuple(self.region.shape())
        self.ndims = len(self._shape)
        if not 1 <= self.ndims <= 3:
            raise NotImplementedError(
                f"charmnumeric arrays must have 1 to 3 dimensions, got shape {self._shape}"
            )
        self._size = _shape_size(self._shape)

        if self.region.is_global and self._shape is not None:
            self.region = _full_region(self._shape)

    def wire_dtype(self):
        return self._wire_dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self.ndims

    @property
    def size(self):
        return _CallableInt(self._size)

    @property
    def itemsize(self):
        return 0 if self.dtype is None else self.dtype.itemsize

    @property
    def nbytes(self):
        return int(self.size) * self.itemsize

    def __len__(self):
        if self.ndims == 0:
            raise TypeError("len() of unsized object")
        return self._shape[0]

    def __bool__(self):
        raise ValueError(
            "The truth value of a charmnumeric array is ambiguous. "
            "Use .get(interface) to materialize it first."
        )

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "charmnumeric arrays are lazy frontend objects. "
            "Call .get(interface) before converting to numpy."
        )

    def __repr__(self):
        dtype_name = None if self.dtype is None else self.dtype.name
        return f"Array(shape={self.shape}, dtype={dtype_name}, name={self.name})"

    def get(self, interface):
        execute(interface)
        data = interface.get(self.ndims, self.name, self.size(), dtype=self.dtype)
        if self.ndims > 1:
            return data.reshape(self._shape)
        return data

    def get_region(self, region, **kwargs):
        kwargs.setdefault("shape", region.shape())
        kwargs.setdefault("dtype", self.dtype)
        kwargs.setdefault("wire_dtype", self._wire_dtype)
        kwargs.setdefault("size", _shape_size(kwargs["shape"]))
        return ArrayView(self, region, **kwargs)

    def __getitem__(self, key):
        if _has_newaxis(key):
            # a view cannot change ndims: return a copy
            view = self[_expand_key(key, self.ndims)]
            return _reshape_copy(view, _newaxis_shape(key, view.shape, self.ndims))
        region = _parse_key(key, self._shape, self.ndims)
        region_shape = region.shape()
        return self.get_region(
            region,
            shape=region_shape,
            dtype=self.dtype,
            wire_dtype=self._wire_dtype,
            size=_shape_size(region_shape),
        )

    def __setitem__(self, key, rhs):
        _setitem(self, key, rhs)

    def set_region(self, region, rhs, **kwargs):
        # all writes (incl. via views) land here
        self.__dict__.pop("_transpose_cache", None)
        return super().set_region(region, rhs, **kwargs)

    @property
    def T(self):
        return _transpose(self)

    transpose = _transpose_method

    def swapaxes(self, axis1, axis2):
        return _swapaxes(self, axis1, axis2)

    __add__ = _array_add
    __radd__ = _array_radd
    __sub__ = _array_sub
    __neg__ = _array_neg
    __rsub__ = _array_rsub
    __mul__ = _array_mul
    __rmul__ = _array_rmul
    __truediv__ = _array_truediv
    __rtruediv__ = _array_rtruediv
    __matmul__ = _array_matmul
    __pow__ = _array_pow
    __rpow__ = _array_rpow
    __mod__ = _array_mod
    __rmod__ = _array_rmod
    __floordiv__ = _array_floordiv
    __rfloordiv__ = _array_rfloordiv
    __divmod__ = _array_divmod
    __rdivmod__ = _array_rdivmod
    __lt__ = _array_lt
    __le__ = _array_le
    __gt__ = _array_gt
    __ge__ = _array_ge
    __eq__ = _array_eq
    __ne__ = _array_ne
    __abs__ = _array_abs
    __pos__ = _array_pos
    # logical on 0/1 masks (there is no bool dtype; not bitwise on integers)
    __and__ = __rand__ = _logical_and
    __or__ = __ror__ = _logical_or
    __xor__ = __rxor__ = _logical_xor
    __invert__ = _logical_not

    def dot(self, other, out=None):
        return _write_out(_array_matmul(self, other), out)

    def take(self, indices, axis=None, out=None, mode="raise"):
        from .operations import take

        return take(self, indices, axis=axis, out=out, mode=mode)

    def matvec(self, vec):
        return _matvec(self, vec)

    def copy(self):
        result = empty(self.shape, dtype=self.dtype)
        result[...] = self
        return result

    def fill(self, value):
        self[...] = value

    def astype(self, dtype, copy=True):
        dtype = _normalize_dtype(dtype)
        if dtype != self.dtype:
            return _astype(self, dtype)
        return self.copy() if copy else self

    def cumsum(self, axis=None, dtype=None, out=None):
        return _scan("cumsum", self, axis, dtype, out)

    def cumprod(self, axis=None, dtype=None, out=None):
        return _scan("cumprod", self, axis, dtype, out)

    def sort(self, axis=-1, kind=None, order=None):
        """Sort in place along ``axis`` (like ``ndarray.sort``)."""
        self[...] = _sort(self, axis, kind, order)

    def argsort(self, axis=-1, kind=None, order=None):
        return _sort(self, axis, kind, order, arg=True)


class ArrayView(FrontendObjectView, _ElementwiseMethods):
    """Array-specific view with arithmetic operators and slicing."""

    __array_priority__ = 1000
    # __eq__ is elementwise; keep identity hashing
    __hash__ = object.__hash__

    @property
    def ndim(self):
        return self.ndims

    @property
    def size(self):
        return _CallableInt(self._size)

    @property
    def itemsize(self):
        return 0 if self.dtype is None else self.dtype.itemsize

    @property
    def nbytes(self):
        return int(self.size) * self.itemsize

    def __len__(self):
        if self.ndims == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __bool__(self):
        raise ValueError(
            "The truth value of a charmnumeric array is ambiguous. "
            "Use .get(interface) to materialize it first."
        )

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "charmnumeric arrays are lazy frontend objects. "
            "Call .get(interface) before converting to numpy."
        )

    def __repr__(self):
        dtype_name = None if self.dtype is None else self.dtype.name
        return f"ArrayView(shape={self.shape}, dtype={dtype_name}, name={self.name})"

    def get_region(self, region, **kwargs):
        kwargs.setdefault("shape", region.shape())
        kwargs.setdefault("dtype", self.dtype)
        kwargs.setdefault("wire_dtype", self.wire_dtype())
        kwargs.setdefault("size", _shape_size(kwargs["shape"]))
        return ArrayView(self, region, **kwargs)

    def get(self, interface):
        materialized = empty(self.shape, dtype=self.dtype)
        materialized[...] = self
        return materialized.get(interface)

    def __getitem__(self, key):
        if _has_newaxis(key):
            view = self[_expand_key(key, self.ndims)]
            return _reshape_copy(view, _newaxis_shape(key, view.shape, self.ndims))
        region = _parse_key(key, self.shape, self.ndims)
        region_shape = region.shape()
        return self.get_region(
            region,
            shape=region_shape,
            dtype=self.dtype,
            wire_dtype=self.wire_dtype(),
            size=_shape_size(region_shape),
        )

    def __setitem__(self, key, rhs):
        _setitem(self, key, rhs)

    @property
    def T(self):
        return _transpose(self)

    transpose = _transpose_method

    def swapaxes(self, axis1, axis2):
        return _swapaxes(self, axis1, axis2)

    __add__ = _array_add
    __radd__ = _array_radd
    __sub__ = _array_sub
    __neg__ = _array_neg
    __rsub__ = _array_rsub
    __mul__ = _array_mul
    __rmul__ = _array_rmul
    __truediv__ = _array_truediv
    __rtruediv__ = _array_rtruediv
    __matmul__ = _array_matmul
    __pow__ = _array_pow
    __rpow__ = _array_rpow
    __mod__ = _array_mod
    __rmod__ = _array_rmod
    __floordiv__ = _array_floordiv
    __rfloordiv__ = _array_rfloordiv
    __divmod__ = _array_divmod
    __rdivmod__ = _array_rdivmod
    __lt__ = _array_lt
    __le__ = _array_le
    __gt__ = _array_gt
    __ge__ = _array_ge
    __eq__ = _array_eq
    __ne__ = _array_ne
    __abs__ = _array_abs
    __pos__ = _array_pos
    # logical on 0/1 masks (there is no bool dtype; not bitwise on integers)
    __and__ = __rand__ = _logical_and
    __or__ = __ror__ = _logical_or
    __xor__ = __rxor__ = _logical_xor
    __invert__ = _logical_not

    def dot(self, other, out=None):
        return _write_out(_array_matmul(self, other), out)

    def take(self, indices, axis=None, out=None, mode="raise"):
        from .operations import take

        return take(self, indices, axis=axis, out=out, mode=mode)

    def matvec(self, vec):
        return _matvec(self, vec)

    def copy(self):
        result = empty(self.shape, dtype=self.dtype)
        result[...] = self
        return result

    def fill(self, value):
        self[...] = value

    def astype(self, dtype, copy=True):
        dtype = _normalize_dtype(dtype)
        if dtype != self.dtype:
            return _astype(self, dtype)
        return self.copy() if copy else self

    def cumsum(self, axis=None, dtype=None, out=None):
        return _scan("cumsum", self, axis, dtype, out)

    def cumprod(self, axis=None, dtype=None, out=None):
        return _scan("cumprod", self, axis, dtype, out)

    def sort(self, axis=-1, kind=None, order=None):
        """Sort in place along ``axis`` (like ``ndarray.sort``)."""
        self[...] = _sort(self, axis, kind, order)

    def argsort(self, axis=-1, kind=None, order=None):
        return _sort(self, axis, kind, order, arg=True)


def create_array(shape, dtype=_DEFAULT_DTYPE, **kwargs):
    shape = _normalize_shape(shape)
    dtype = _normalize_dtype(dtype)
    return create_object(
        Array,
        region=ArrayRegion(start=(0,) * len(shape), stop=shape, step=(1,) * len(shape)),
        dtype=dtype,
        shape=shape,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Array-creation nodes (UPLOAD / RANDOM / IOTA)
#
# Each creation op is a single non-fusible DAG node that creates a new array
# and fills it on the backend, chare by chare, with no communication. Its
# parameters travel as 64-bit words, each sent as two exact SCALAR operands
# (high and low 32 bits), so int64 bounds and double bit patterns arrive
# unchanged. UPLOAD carries the row-major host data as an INLINE_DATA operand.
# ---------------------------------------------------------------------------
_U64_MASK = (1 << 64) - 1


def _u64_from_int(value):
    return int(value) & _U64_MASK


def _u64_from_double(value):
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _u64_from_value(value, dtype):
    """64-bit parameter word for a value of `dtype` (int64 or double bits)."""
    if np.dtype(dtype).kind == "f":
        return _u64_from_double(value)
    return _u64_from_int(value)


def _creation_op(operation, shape, dtype, words=(), payload=None):
    shape = _normalize_shape(shape)
    dtype = _normalize_dtype(dtype)
    if len(shape) == 0:
        raise NotImplementedError("0D charmnumeric arrays are not supported yet")
    if len(shape) > 3:
        raise NotImplementedError(
            f"charmnumeric arrays have at most 3 dimensions, got shape {shape}"
        )
    region = _full_region(shape)
    operands = [region]
    for word in words:
        word = int(word) & _U64_MASK
        operands.append(float(word >> 32))
        operands.append(float(word & 0xFFFFFFFF))
    if payload is not None:
        operands.append(InlineData(payload))
    result = create_result(
        Array, operation, operands, fusible=False, region=region, dtype=dtype, shape=shape
    )
    get_execution_state().add_node(NonFusibleNode(obj=result))
    return result


def _upload(host):
    """Create an array holding the host ndarray `host` (bulk, one DAG node)."""
    host = np.ascontiguousarray(host)
    if host.ndim == 0:
        raise NotImplementedError("0D charmnumeric arrays are not supported yet")
    if host.size == 0:
        return create_array(host.shape, dtype=host.dtype)
    return _creation_op(ArrayOperation.upload, host.shape, host.dtype, payload=host.tobytes())


def _array_from_host_data(values, dtype=None, copy=True, ndmin=0):
    requested_dtype = _normalize_dtype(dtype, allow_none=True)

    if copy:
        host = np.array(values, dtype=requested_dtype, copy=True)
    else:
        host = np.asarray(values, dtype=requested_dtype)

    while host.ndim < ndmin:
        host = np.expand_dims(host, axis=0)

    if host.ndim == 0:
        raise NotImplementedError("0D charmnumeric arrays are not supported yet")

    host_dtype = _normalize_dtype(host.dtype)
    host = host.astype(host_dtype, copy=False)
    if host.size == 0:
        return create_array(host.shape, dtype=host_dtype)
    # Constant data is cheaper as a scalar fill than as an upload, but scalars
    # travel as doubles: int64 values beyond 2**53 must be uploaded.
    first = host.reshape(-1)[0]
    if (host.dtype.kind != "f" and int(float(first)) != int(first)) or (
        host.size > 1 and not np.all(host == first)
    ):
        return _upload(host)
    result = create_array(host.shape, dtype=host_dtype)
    result[...] = _coerce_scalar(first)
    return result


def _shape_dtype_from_like(obj):
    if isinstance(obj, (Array, ArrayView)):
        return obj.shape, obj.dtype

    host = np.asarray(obj)
    if host.ndim == 0:
        raise NotImplementedError("0D charmnumeric arrays are not supported yet")
    return host.shape, _normalize_dtype(host.dtype)


def _coerce_assignment_operand(rhs, shape, dtype):
    if isinstance(rhs, (Array, ArrayView)):
        if _shape_size(shape) > 1:
            return _scalar_view_operand(rhs)
        return rhs

    if _is_scalar_like(rhs):
        return _coerce_scalar(rhs)

    host = np.asarray(rhs, dtype=_normalize_dtype(dtype))
    if host.ndim == 0:
        return _coerce_scalar(host.item())

    try:
        host = np.broadcast_to(host, shape)
    except ValueError as exc:
        raise ValueError(
            f"could not broadcast input from shape {host.shape} into shape {shape}"
        ) from exc

    return _array_from_host_data(host, dtype=host.dtype, copy=False)


def _squeeze_shape(shape):
    return tuple(dim for dim in shape if dim != 1)


def _is_whole(x):
    """True if ``x`` is an Array, or a view covering its whole root array."""

    if isinstance(x, Array):
        return True
    root = x._root_parent
    region = x.region
    return (
        tuple(x.shape) == tuple(root.shape)
        and all(s == 0 for s in region.start)
        and all(st == 1 for st in region.step)
    )


def _materialize(x):
    """Copy a partial view (REDUCE/TILE/DIAG ignore view regions)."""

    if isinstance(x, ArrayView) and not _is_whole(x):
        return x.copy()
    return x


def _expand_scalar_array(value, shape):
    """Non-broadcast set_region source with a size-1 array's value.

    Cross-rank broadcasts hang the backend, so the rank changes via a
    one-element regular copy first.
    """

    def ones_view(arr):
        # one-element view of a 2-element array: a regular (non-broadcast) operand
        nd = arr.ndims
        pair = zeros((1,) * (nd - 1) + (2,), dtype=arr.dtype) + arr
        return pair[(slice(None),) * (nd - 1) + (slice(0, 1),)]

    if value.ndims != len(shape):
        moved = empty((1,) * len(shape), dtype=value.dtype)
        moved.set_region(_full_region(moved.shape), ones_view(value),
                         shape=moved.shape, dtype=value.dtype)
        value = moved
    if _shape_size(shape) > 1:
        return zeros(shape, dtype=value.dtype) + value
    return ones_view(value)


def _setitem(self, key, rhs):
    region = _parse_key(key, self.shape, self.ndims)
    shape = tuple(region.shape())
    rhs = _coerce_assignment_operand(rhs, shape, self.dtype)
    rhs = _cast_operand(rhs, self.dtype)  # NumPy casts on assignment
    if isinstance(rhs, (Array, ArrayView)) and _shape_size(
        (rhs._root_parent if isinstance(rhs, ArrayView) else rhs).shape
    ) == 1:
        if rhs.ndims == self.ndims:
            result = self.set_region(region, rhs, shape=shape, dtype=self.dtype)
            prefix = get_execution_state().open_prefix
            owner = rhs._root_parent if isinstance(rhs, ArrayView) else rhs
            if prefix is not None and any(o is result for o in prefix.objs) and any(
                o is owner for o in prefix.objs
            ):
                return  # fused with rhs's producer: the kernel has array inputs
        # a set_region whose only input is a broadcast writes nothing
        rhs = _expand_scalar_array(rhs, shape)
    if isinstance(rhs, (Array, ArrayView)) and int(rhs.size) != 1 and tuple(rhs.shape) != shape:
        # shapes differing only in size-1 dims: cross-ndim set_region
        if _squeeze_shape(rhs.shape) != _squeeze_shape(shape):
            raise ValueError(
                f"could not broadcast input array from shape {tuple(rhs.shape)} "
                f"into shape {shape}"
            )
    self.set_region(region, rhs, shape=shape, dtype=self.dtype)


def _reshape_copy(x, shape):
    """New Array of ``shape`` holding ``x`` (shapes differ only in size-1 dims)."""

    result = empty(shape, dtype=x.dtype)
    result[...] = x
    return result


def _write_out(result, out):
    """NumPy ``out=``: ``out[...] = result`` (out may be a view); returns out."""

    if out is None:
        return result
    if isinstance(out, tuple):
        if len(out) != 1:
            raise ValueError("out must be a single array or a 1-tuple")
        out = out[0]
    if not isinstance(out, (Array, ArrayView)):
        raise TypeError("out must be a charmnumeric Array or ArrayView")

    if isinstance(result, (Array, ArrayView)):
        if int(result.size) != 1 and tuple(result.shape) != tuple(out.shape):
            raise ValueError(
                f"non-broadcastable output operand with shape {tuple(out.shape)} "
                f"doesn't match the broadcast shape {tuple(result.shape)}"
            )
        if result.dtype != out.dtype:
            raise TypeError(
                f"cannot write a {result.dtype} result into a {out.dtype} out array "
                "(dtype conversion is not implemented yet)"
            )
    out[...] = result
    return out


def empty(shape, dtype=_DEFAULT_DTYPE):
    return create_array(shape, dtype=dtype)


def zeros(shape, dtype=_DEFAULT_DTYPE):
    return empty(shape, dtype=dtype)


def full(shape, fill_value, dtype=None):
    shape = _normalize_shape(shape)

    if _is_scalar_like(fill_value):
        target_dtype = _normalize_dtype(dtype, allow_none=True)
        if target_dtype is None:
            target_dtype = _normalize_dtype(np.asarray(fill_value).dtype)
        result = empty(shape, dtype=target_dtype)
        result[...] = _coerce_scalar(fill_value)
        return result

    host = np.asarray(fill_value, dtype=_normalize_dtype(dtype, allow_none=True))
    if host.ndim == 0:
        return full(shape, host.item(), dtype=dtype)

    target_dtype = _normalize_dtype(host.dtype)
    try:
        host = np.broadcast_to(host, shape)
    except ValueError as exc:
        raise ValueError(
            f"could not broadcast fill_value from shape {host.shape} into shape {shape}"
        ) from exc

    return _array_from_host_data(host, dtype=target_dtype, copy=False)


def ones(shape, dtype=_DEFAULT_DTYPE):
    return full(shape, 1, dtype=dtype)


def empty_like(a, dtype=None, shape=None):
    base_shape, base_dtype = _shape_dtype_from_like(a)
    target_shape = base_shape if shape is None else _normalize_shape(shape)
    target_dtype = base_dtype if dtype is None else _normalize_dtype(dtype)
    return empty(target_shape, dtype=target_dtype)


def zeros_like(a, dtype=None, shape=None):
    base_shape, base_dtype = _shape_dtype_from_like(a)
    target_shape = base_shape if shape is None else _normalize_shape(shape)
    target_dtype = base_dtype if dtype is None else _normalize_dtype(dtype)
    return zeros(target_shape, dtype=target_dtype)


def ones_like(a, dtype=None, shape=None):
    base_shape, base_dtype = _shape_dtype_from_like(a)
    target_shape = base_shape if shape is None else _normalize_shape(shape)
    target_dtype = base_dtype if dtype is None else _normalize_dtype(dtype)
    return ones(target_shape, dtype=target_dtype)


def full_like(a, fill_value, dtype=None, shape=None):
    base_shape, base_dtype = _shape_dtype_from_like(a)
    target_shape = base_shape if shape is None else _normalize_shape(shape)
    target_dtype = base_dtype if dtype is None else _normalize_dtype(dtype)
    return full(target_shape, fill_value, dtype=target_dtype)


def asarray(a, dtype=None):
    requested_dtype = _normalize_dtype(dtype, allow_none=True)

    if isinstance(a, (Array, ArrayView)):
        if requested_dtype is None or requested_dtype == a.dtype:
            return a
        return _astype(a, requested_dtype)

    return _array_from_host_data(a, dtype=requested_dtype, copy=False)


def array(a, dtype=None, copy=True, ndmin=0):
    requested_dtype = _normalize_dtype(dtype, allow_none=True)

    if isinstance(a, (Array, ArrayView)):
        target_dtype = a.dtype if requested_dtype is None else requested_dtype
        target_shape = _apply_ndmin(a.shape, ndmin)
        if target_dtype != a.dtype and target_shape == a.shape:
            return _astype(a, target_dtype)
        if not copy and target_shape == a.shape:
            return a

        result = empty(target_shape, dtype=target_dtype)
        result[...] = a
        return result

    return _array_from_host_data(a, dtype=requested_dtype, copy=copy, ndmin=ndmin)


def copy(a, order="K"):
    return array(a, copy=True)


def copyto(dst, src):
    if not isinstance(dst, (Array, ArrayView)):
        raise TypeError("copyto destination must be a charmnumeric Array or ArrayView")
    dst[...] = src
    return dst


# IOTA modes and LINSPACE flags (src/creation_kernels.hpp)
_IOTA_ARANGE = 0
_IOTA_LINSPACE = 1
_IOTA_INDICES = 2
_IOTA_EYE = 3
_LINSPACE_ENDPOINT = 1
_LINSPACE_STEP_ZERO = 2


def _iota(shape, dtype, mode, axis=0, flags=0, num=0, div=0, v=(0, 0, 0, 0)):
    words = [mode, axis, flags, _u64_from_int(num), _u64_from_int(div)] + list(v)
    return _creation_op(ArrayOperation.iota, shape, dtype, words)


def _arange_recipe(start, stop, step, dtype):
    """(dtype, length, first, second) exactly as numpy.arange computes them."""
    if step == 0:
        raise ZeroDivisionError("arange: step must not be zero")
    probe = np.arange(start, start, step, dtype=dtype)  # validates, infers dtype
    dt = _normalize_dtype(probe.dtype)
    length = max(0, int(math.ceil((stop - start) / step)))
    first = dt.type(start)
    second = dt.type(start + step) if length > 1 else first
    return dt, length, first, second


def arange(start, stop=None, step=1, dtype=None):
    """numpy.arange, generated on the backend (bit-identical to NumPy).

    NumPy fills ``out[i] = out[0] + i * (out[1] - out[0])`` in the output
    dtype, with ``out[1] = start + step``; the backend uses the same formula.
    """
    if stop is None:
        start, stop = 0, start
    requested_dtype = _normalize_dtype(dtype, allow_none=True)
    dt, length, first, second = _arange_recipe(start, stop, step, requested_dtype)
    if length == 0:
        return create_array((0,), dtype=dt)
    result = _iota(
        (length,), dt, _IOTA_ARANGE,
        v=(_u64_from_value(first, dt), _u64_from_value(second, dt), 0, 0),
    )
    result._iota_recipe = ("arange", dt, length, first, second)
    return result


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """numpy.linspace for scalar start/stop, generated on the backend.

    Python (or float64) start/stop are evaluated in float64 exactly as NumPy
    does (``i * step + start``, last sample set to ``stop``, floor for integer
    dtypes). Other inputs (float32 scalars, arrays) are computed with NumPy on
    the host and uploaded.
    """
    num = int(num)
    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")
    out_dtype = _normalize_dtype(dtype, allow_none=True)
    native = (
        isinstance(start, (int, float, np.integer, np.float64))
        and isinstance(stop, (int, float, np.integer, np.float64))
        and not isinstance(start, bool)
        and not isinstance(stop, bool)
        and axis == 0
    )
    if not native or num == 0:
        host, step = np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True,
                                 dtype=out_dtype, axis=axis)
        result = array(host) if host.size else create_array(host.shape, dtype=host.dtype)
        return (result, step) if retstep else result

    if out_dtype is None:
        out_dtype = np.dtype(np.float64)
    div = (num - 1) if endpoint else num
    start64 = np.float64(start)
    stop64 = np.float64(stop)
    delta = np.subtract(stop, start, dtype=np.float64)
    flags = _LINSPACE_ENDPOINT if endpoint else 0
    if div > 0:
        step = delta / div
        if step == 0:
            flags |= _LINSPACE_STEP_ZERO
    else:
        step = np.nan
    result = _iota(
        (num,), out_dtype, _IOTA_LINSPACE, flags=flags, num=num, div=div,
        v=(_u64_from_double(start64), _u64_from_double(stop64),
           _u64_from_double(step), _u64_from_double(delta)),
    )
    result._iota_recipe = ("linspace", flags, num, div, start64, stop64, step, delta)
    return (result, float(step)) if retstep else result


def indices(dimensions, dtype=int, sparse=False):
    """numpy.indices for 1-D/2-D grids (the dense result is at most 3-D)."""
    dimensions = _normalize_shape(dimensions)
    dtype = _normalize_dtype(dtype)
    nd = len(dimensions)
    if sparse:
        out = []
        for k in range(nd):
            shape = tuple(dimensions[k] if d == k else 1 for d in range(nd))
            out.append(_axis_range(shape, k, dimensions[k], dtype))
        return tuple(out)
    if nd + 1 > 3:
        raise NotImplementedError("indices: the dense result would have more than 3 dims")
    shape = (nd,) + dimensions
    if _shape_size(shape) == 0:
        return create_array(shape, dtype=dtype)
    return _iota(shape, dtype, _IOTA_INDICES)


def _axis_range(shape, axis, n, dtype, recipe=None):
    """out[..., i_axis, ...] = recipe(i_axis): an arange/linspace laid along `axis`."""
    if _shape_size(shape) == 0:
        return create_array(shape, dtype=dtype)
    if recipe is None:
        recipe = ("arange", dtype, n, dtype.type(0), dtype.type(1))
    if recipe[0] == "arange":
        _, rdt, _, first, second = recipe
        if np.dtype(rdt) != np.dtype(dtype):
            raise TypeError("meshgrid: dtype changes need astype")
        return _iota(shape, dtype, _IOTA_ARANGE, axis=axis,
                     v=(_u64_from_value(first, dtype), _u64_from_value(second, dtype), 0, 0))
    _, flags, num, div, start64, stop64, step, delta = recipe
    return _iota(shape, dtype, _IOTA_LINSPACE, axis=axis, flags=flags, num=num, div=div,
                 v=(_u64_from_double(start64), _u64_from_double(stop64),
                    _u64_from_double(step), _u64_from_double(delta)))


def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    """numpy.meshgrid for up to three 1-D inputs.

    Inputs may be host sequences / NumPy arrays (uploaded) or charmnumeric
    arrays made by arange/linspace (generated natively along each output
    axis). Other device arrays would need broadcasting and are not supported.
    """
    if indexing not in ("xy", "ij"):
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
    ndim = len(xi)
    if ndim == 0:
        return []
    if ndim > 3:
        raise NotImplementedError("meshgrid: at most 3 inputs (3-D outputs)")
    lengths = []
    for x in xi:
        if isinstance(x, (Array, ArrayView)):
            if getattr(x, "_iota_recipe", None) is None or x.ndims != 1:
                raise NotImplementedError(
                    "meshgrid of a device array needs general broadcasting; pass host "
                    "data or an array made by cnp.arange/cnp.linspace"
                )
            lengths.append(x.shape[0])
        else:
            lengths.append(int(np.asarray(x).size))
    axes = list(range(ndim))
    if indexing == "xy" and ndim > 1:
        axes[0], axes[1] = 1, 0
    full_shape = [0] * ndim
    for k in range(ndim):
        full_shape[axes[k]] = lengths[k]
    out = []
    for k, x in enumerate(xi):
        shape = tuple(full_shape) if not sparse else tuple(
            lengths[k] if d == axes[k] else 1 for d in range(ndim))
        if isinstance(x, (Array, ArrayView)):
            out.append(_axis_range(shape, axes[k], lengths[k], x.dtype, x._iota_recipe))
        else:
            host = np.asarray(x).reshape(-1)
            view_shape = [1] * ndim
            view_shape[axes[k]] = lengths[k]
            out.append(array(np.broadcast_to(host.reshape(view_shape), shape)))
    return out


def eye(N, M=None, k=0, dtype=_DEFAULT_DTYPE):
    rows = int(N)
    cols = rows if M is None else int(M)
    dtype = _normalize_dtype(dtype)
    if rows == 0 or cols == 0:
        return zeros((rows, cols), dtype=dtype)
    return _iota((rows, cols), dtype, _IOTA_EYE, v=(_u64_from_int(k), 0, 0, 0))


def identity(n, dtype=_DEFAULT_DTYPE):
    return eye(n, dtype=dtype)
