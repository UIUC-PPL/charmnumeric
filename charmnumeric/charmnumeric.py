import math
import numbers
import struct

import numpy as np

from charmtyles.ast import ASTNode, InlineData, NonFusibleNode, get_execution_state
import charmtyles.ast as _ct_ast
from charmtyles.core import (
    Region,
    FrontendObject,
    FrontendObjectView,
    create_object,
    create_result,
    execute,
    _PENDING_AST,
    _shape_size,
)
from charmtyles.operation import Operation
from charmtyles.operation import fusible_op as _fusible_op
from charmtyles.operation import nonfusible_op as _nonfusible_op
from charmtyles.interface import to_bytes

from ._native_region import NativeArrayRegion


_FULL_REGION_CACHE = {}
_DEFAULT_DTYPE = np.dtype("float64")
_FULL_SLICE = slice(None)


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
    BOOL = 4
    INT8 = 5
    INT16 = 6
    UINT8 = 7
    UINT16 = 8
    UINT32 = 9
    UINT64 = 10
    FLOAT16 = 11
    COMPLEX64 = 12
    COMPLEX128 = 13

    _to_numpy = {
        FLOAT32: np.dtype("float32"),
        FLOAT64: np.dtype("float64"),
        INT32: np.dtype("int32"),
        INT64: np.dtype("int64"),
        BOOL: np.dtype("bool"),
        INT8: np.dtype("int8"),
        INT16: np.dtype("int16"),
        UINT8: np.dtype("uint8"),
        UINT16: np.dtype("uint16"),
        UINT32: np.dtype("uint32"),
        UINT64: np.dtype("uint64"),
        FLOAT16: np.dtype("float16"),
        COMPLEX64: np.dtype("complex64"),
        COMPLEX128: np.dtype("complex128"),
    }
    _from_numpy = {v: k for k, v in _to_numpy.items()}

    @staticmethod
    def from_numpy(np_dtype):
        """Convert a supported numpy dtype to the wire encoding."""

        dtype = _normalize_dtype(np_dtype)
        return DType._from_numpy[dtype]

    @staticmethod
    def to_numpy(wire_dtype):
        """Convert a wire dtype encoding back to a numpy dtype."""

        if wire_dtype not in DType._to_numpy:
            raise TypeError(f"Unsupported wire dtype {wire_dtype!r}")
        return DType._to_numpy[wire_dtype]


def _normalize_dtype(dtype, *, allow_none=False):
    if dtype is None:
        if allow_none:
            return None
        return _DEFAULT_DTYPE

    np_dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
    if np_dtype not in DType._from_numpy:
        raise TypeError(
            "charmnumeric supports bool, int8/16/32/64, uint8/16/32/64, float16/32/64 "
            f"and complex64/128; got {np_dtype}"
        )
    return np_dtype


def _normalize_shape(shape):
    if shape.__class__ is tuple:
        for dim in shape:
            if dim.__class__ is not int or dim < 0:
                break
        else:
            return shape
    if isinstance(shape, numbers.Integral):
        normalized = (int(shape),)
    else:
        normalized = tuple(int(dim) for dim in shape)

    for dim in normalized:
        if dim < 0:
            raise ValueError(f"negative dimensions are not allowed: {normalized}")
    return normalized


def _coerce_scalar(value):
    cls = value.__class__
    if cls is float or cls is int:
        return value
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
    if isinstance(value, numbers.Complex):
        return complex(value)
    return value


def _is_scalar_like(value):
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    if isinstance(value, np.generic):
        return True
    return isinstance(value, (bool, numbers.Number))


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
    # 80-89: scan / sort executors (src/executor_castscansort.cpp)
    scan = 81
    sort = 82
    argsort = 83
    linalg = 100  # factorizations and solves (charmnumeric/linalg.py)
    # Data-dependent indexing (120-139, src/executor_indexing.cpp)
    gather = 120
    scatter = 121
    expand = 122
    compress = 123
    mask_scatter = 124
    # 140-149: shape ops (src/executor_shape.cpp)
    reshape = 140
    broadcast = 141
    convolve = 150  # charmnumeric/signal.py
    flip = 151
    # 160-169: fused astype, complex parts, bitwise ops
    convert = 160
    real = 161
    imag = 162
    conj = 163
    bitwise_and = 164
    bitwise_or = 165
    bitwise_xor = 166
    invert = 167
    left_shift = 168
    right_shift = 169


class _NDBase(object):
    """Marker base of FoldedArray (folded.py): arrays with more than 3 dims."""

    __slots__ = ()


def _has_folded(args, kwargs):
    for value in args:
        if isinstance(value, _NDBase):
            return True
    for value in kwargs.values():
        if isinstance(value, _NDBase):
            return True
    return False


def _nd_aware(fn):
    """Let an elementwise function take >3-D (folded) operands."""

    def wrapper(*args, **kwargs):
        if not _has_folded(args, kwargs):
            return fn(*args, **kwargs)
        from .folded import apply_elementwise

        return apply_elementwise(fn, args, kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "elementwise")
    wrapper.__qualname__ = getattr(fn, "__qualname__", wrapper.__name__)
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn
    return wrapper


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
            [(b - a + c - 1) // c for a, b, c in zip(self.start, self.stop, self.step)]
        )
        self._hash = hash((self.start, self.stop, self.step))
        self._serialized = None
        self._native = None

    def _native_region(self):
        native = self._native
        if native is None:
            native = self._native = NativeArrayRegion(self.start, self.stop, self.step, False)
        return native

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
        return (self._native or self._native_region()).overlaps(other._native or other._native_region())

    def covers(self, other):
        if not isinstance(other, ArrayRegion):
            raise TypeError(
                f"ArrayRegion cover check requires ArrayRegion, got {type(other).__name__}"
            )
        return (self._native or self._native_region()).covers(other._native or other._native_region())

    def intersect(self, other):
        if not isinstance(other, ArrayRegion):
            raise TypeError(
                f"ArrayRegion intersection requires ArrayRegion, got {type(other).__name__}"
            )
        result = self._native_region().intersect(other._native_region())
        if result is None:
            return None
        start, stop, step = result
        return ArrayRegion(start, stop, step)


def _storage_shape(shape):
    """Backend shape of an array of ``shape`` (0-d arrays are stored as (1,))."""
    return tuple(shape) or (1,)


# storage ranks of all arrays created so far (synchronize() waits on each)
_RANKS_USED = set()


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
    """Expand ellipsis and pad with full slices to ``ndims``; ``None`` entries are dropped unless ``keep_newaxis``."""
    if key is Ellipsis:
        key = (Ellipsis,)
    elif not isinstance(key, tuple):
        key = (key,)

    for item in key:
        if item is Ellipsis or item is None:
            break
    else:
        if len(key) > ndims:
            raise IndexError(
                f"too many indices for array: array is {ndims}-dimensional, "
                f"but {len(key)} were indexed"
            )
        return key + (_FULL_SLICE,) * (ndims - len(key))

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


def _parse_key(key, shape, ndims):
    """Normalize a ``__getitem__`` or ``__setitem__`` key into an ArrayRegion."""

    if key.__class__ is not tuple:
        key = (key,)
    if len(key) != ndims:
        key = _expand_key(key, ndims)
    else:
        for item in key:
            if item is Ellipsis or item is None:
                key = _expand_key(key, ndims)
                break
    start = [0] * ndims
    stop = [0] * ndims
    step = [1] * ndims

    for i, item in enumerate(key):
        if item.__class__ is slice:
            start_i, stop_i, step_i = item.indices(shape[i])
            if step_i < 0:
                raise NotImplementedError("negative slicing is not supported yet")
            if stop_i < start_i:
                stop_i = start_i
        elif item.__class__ is int or isinstance(item, numbers.Integral):
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


def _is_indexing_key(key):
    """True if a key needs the gather/scatter path (index arrays or masks)."""
    for item in key if isinstance(key, tuple) else (key,):
        if isinstance(item, (Array, ArrayView, list)):
            return True
        if isinstance(item, np.ndarray) and (item.ndim > 0 or item.dtype == bool):
            return True
    return False


def _size1_value(x):
    """The value of a size-1 array, fetched with a blocking get."""
    if x.size != 1:
        raise TypeError("only length-1 arrays can be converted to Python scalars")
    from .interface import default_interface
    return x.get(default_interface()).reshape(-1)[0]


class _HostConversions(object):
    """NumPy's conversions to host values (each fetches the array)."""

    __slots__ = ()

    def __array__(self, dtype=None, copy=None):
        from .interface import default_interface

        host = self.get(default_interface())
        return host if dtype is None else host.astype(dtype, copy=False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        import charmnumeric

        fn = getattr(charmnumeric, ufunc.__name__, None)
        if method != "__call__" or fn is None or not callable(fn) or set(kwargs) - {"out"}:
            return NotImplemented
        out = kwargs.pop("out", None)
        if isinstance(out, tuple):
            if len(out) != 1:
                return NotImplemented
            out = out[0]
        if isinstance(out, np.ndarray):  # a host output: compute, then fetch into it
            out[...] = np.asarray(fn(*inputs, **kwargs))
            return out
        return fn(*inputs, **kwargs) if out is None else fn(*inputs, out=out, **kwargs)

    def __bool__(self):
        if int(self.size) != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )
        return bool(_size1_value(self))

    def __int__(self):
        return int(_size1_value(self))

    def __float__(self):
        return float(_size1_value(self))

    def __complex__(self):
        return complex(_size1_value(self))

    def __index__(self):
        if self.ndim != 0 or self.dtype.kind not in "iu":
            raise TypeError("only integer scalar arrays can be converted to a scalar index")
        return int(_size1_value(self))

    def __str__(self):
        from ._display import values_str

        return values_str(self)

    def __format__(self, spec):
        return str(self) if not spec else format(self.__array__(), spec)

    def _repr_mimebundle_(self, include=None, exclude=None):
        from ._display import mimebundle

        return mimebundle(self)

    def to_numpy(self):
        """A host copy; matplotlib converts through this (one fetch, not one per element)."""
        return self.__array__()

    def block_until_ready(self):
        """Wait until the server has computed everything recorded so far; returns self."""
        from .interface import default_interface, synchronize

        synchronize(default_interface())
        return self


def _scalar_view_operand(value):
    """Copy a size-1 view of a larger array; the backend only broadcasts whole size-1 arrays."""
    if (isinstance(value, ArrayView) and value.size() == 1
            and value._root_parent.size() > 1):
        return value.copy()
    return value


def _size1_to_rank(value, nd):
    """A whole size-1 array of storage rank ``nd`` holding ``value`` (the backend hangs on a rank-mismatched size-1 operand)."""
    root = value._root_parent if isinstance(value, ArrayView) else value
    cache = root.__dict__.setdefault("_rank_cache", {}) if root.size() == 1 else {}
    hit = cache.get(nd)
    if hit is None:
        pair = zeros((1,) * (value.ndims - 1) + (2,), dtype=value.dtype) + value
        hit = empty((1,) * nd, dtype=value.dtype)
        hit.set_region(_full_region(hit.shape), pair[(_FULL_SLICE,) * (value.ndims - 1) +
                                                    (slice(0, 1),)],
                       shape=hit.shape, dtype=value.dtype)
        cache[nd] = hit
    return hit


def _kernel_operand(op, nd, size):
    """Operand of a fused kernel of storage rank ``nd`` and ``size`` elements."""
    if op.__class__ is ArrayView and size > 1:
        op = _scalar_view_operand(op)
    if op.size() == 1 and op.ndims != nd and isinstance(op, (Array, ArrayView)):
        root = op._root_parent if isinstance(op, ArrayView) else op
        if root.size() == 1:
            return _size1_to_rank(op, nd)
    return op


def fusible_op(*operands, operation, **kwargs):
    if any(isinstance(op, _NDBase) for op in operands):
        raise NotImplementedError("this operation does not support arrays with more than 3 dims")
    _fresh_operands(operands)
    shape = kwargs.get("shape")
    if shape is not None:
        nd, size = len(shape) or 1, _shape_size(shape)
        operands = tuple([_kernel_operand(op, nd, size)
                          if op.__class__ is Array or op.__class__ is ArrayView else op
                          for op in operands])
    return _fusible_op(*operands, operation=operation, **kwargs)


def _broadcast_operands(operands, name=None):
    """NumPy broadcasting; size-1 arrays stay as they are (read as scalars). Returns ``(operands, shape)``."""
    shapes = [tuple(v.shape) for v in operands if isinstance(v, (Array, ArrayView))]
    if not shapes:
        return tuple(operands), ()
    try:
        shape = tuple(np.broadcast_shapes(*shapes))
    except ValueError as exc:
        prefix = f"{name}: " if name else ""
        raise ValueError(
            prefix + "operands could not be broadcast together with shapes "
            + " ".join(str(sh) for sh in shapes)
        ) from exc
    out = []
    for v in operands:
        if (isinstance(v, (Array, ArrayView)) and int(v.size) != 1
                and tuple(v.shape) != shape):
            v = _broadcast_to(v, shape)
        out.append(v)
    return tuple(out), shape


# ---------------------------------------------------------------------------
# Elementwise typing: result dtypes follow NumPy (NEP 50 weak Python scalars).
# The JIT converts operands in-kernel, so no casts are emitted here. Python
# scalars travel as doubles; complex scalars and ints beyond 2**53 become
# size-1 arrays.
# ---------------------------------------------------------------------------


def _ew_operand(value):
    """_coerce_operand, keeping bool and NumPy scalars typed (they are strong in NEP 50)."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, (bool, np.generic)):
        return value
    return _coerce_operand(value)


def _np_operand(value):
    """Operand stand-in for NumPy type resolution (weak type for Python scalars)."""
    if isinstance(value, (Array, ArrayView, np.generic)):
        return value.dtype
    if isinstance(value, bool):
        return np.dtype(bool)
    if isinstance(value, numbers.Integral):
        return int
    if isinstance(value, numbers.Real):
        return float
    if isinstance(value, numbers.Complex):
        return complex
    raise TypeError(f"unsupported operand type {type(value).__name__}")


def _ufunc_dtypes(ufunc, *operands):
    """``(loop dtypes, result dtype)`` NumPy uses for ``ufunc(*operands)``."""
    res = ufunc.resolve_dtypes(tuple(_np_operand(v) for v in operands) + (None,))
    return res[:-1], _normalize_dtype(res[-1])


def _result_dtype(*operands):
    """numpy.result_type with weak Python scalars (NEP 50)."""
    return _normalize_dtype(np.result_type(*[
        v.dtype if isinstance(v, (Array, ArrayView)) else v for v in operands]))


def _scalar_wire(value, dtype):
    """Python scalar as a wire operand for a kernel computing in ``dtype``."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        if value.imag == 0 and dtype.kind != "c":
            return value.real
        if dtype.kind != "c":
            dtype = np.dtype(np.complex128)
        return _upload(np.array([value], dtype=dtype))
    if isinstance(value, int) and dtype.kind in "iu":
        dtype.type(value)  # OverflowError like NumPy
        if abs(value) > 2**53:
            return _upload(np.array([value], dtype=dtype))
    return value


def _check_shapes(name, operands):
    """``(operands, shape)``: operands broadcast to their common shape (NumPy rules)."""
    arrays = [v for v in operands if isinstance(v, (Array, ArrayView))]
    if not arrays:
        raise TypeError(f"{name}: at least one operand must be a charmnumeric array")
    full_ = [v for v in arrays if v.size != 1]
    shape = full_[0].shape if full_ else max((v.shape for v in arrays), key=len)
    if all(v.shape == shape for v in full_) and all(v.ndims <= len(shape) for v in arrays):
        return operands, tuple(shape)
    return _broadcast_operands(operands, name)


def _ew(operation, *operands, name, ufunc=None, dtype=None):
    """Fused elementwise node; result dtype from ``ufunc`` unless ``dtype`` is given."""
    operands = tuple(_ew_operand(v) for v in operands)
    for v in operands:
        if not isinstance(v, (Array, ArrayView, numbers.Number, np.bool_)):
            raise TypeError(f"{name}: unsupported operand type {type(v).__name__}")
    operands, shape = _check_shapes(name, operands)
    if dtype is None:
        loops, dtype = _ufunc_dtypes(ufunc, *operands)
    else:
        dtype = _normalize_dtype(dtype)
        loops = (dtype,) * len(operands)
    operands = tuple(
        v if isinstance(v, (Array, ArrayView)) else _scalar_wire(v, np.dtype(loop))
        for v, loop in zip(operands, loops))
    drop = _drop_operands(operands, shape)
    if drop is not None:
        ops, key = drop
        r = fusible_op(*ops, operation=operation, shape=ops[_first_array(ops)].shape, dtype=dtype)
        return _lazy_copy(r, key, lambda: r, shape, r, key)
    return fusible_op(*operands, operation=operation, shape=shape, dtype=dtype)


def _first_array(ops):
    return next(i for i, v in enumerate(ops) if v.__class__ is ArrayView or v.__class__ is Array)


def _drop_operands(operands, shape):
    """Source views and key if every array operand is a lazy int-index copy of ``shape``
    with the same storage layout, so the op runs at storage rank (no copies)."""
    ops, key, sshape = [], None, None
    for v in operands:
        if v.__class__ is not Array and v.__class__ is not ArrayView:
            ops.append(v)
            continue
        d = v.__dict__.get("_drop") if v.__class__ is Array else None
        if d is None or v._shape != shape:
            return None
        view, k = d
        if key is None:
            key, sshape = k, view.shape
        elif k != key or view.shape != sshape:
            return None
        ops.append(view)
    return (tuple(ops), key) if key is not None else None


def _lazy_copy(parent, key, source, shape, view, vkey):
    """Linked copy (see _basic_getitem) of ``parent[key]`` that is emitted on first use;
    until then elementwise ops and assignments read ``view`` (== ``view[vkey]`` at storage rank)."""
    result = Array(region=_full_region(_storage_shape(shape)), ast=_PENDING_AST,
                   dtype=view.dtype, shape=shape)
    result._pending_create = True
    result._view_of = (parent, key, source)
    result._seen = None
    result._drop = (view, vkey)
    return result


def _astype(x, dtype):
    """Fused CONVERT node: x converted to ``dtype``."""
    dtype = _normalize_dtype(dtype)
    if x.dtype == dtype:
        return x
    return fusible_op(x, operation=ArrayOperation.convert, shape=x.shape, dtype=dtype)


def _cast_operand(value, dtype):
    if isinstance(value, (Array, ArrayView)) and value.dtype != dtype:
        return _astype(value, dtype)
    return value


# Dtypes the BLAS-backed executors take; others compute in one of these.
_BLAS_DTYPES = {np.dtype(t) for t in ("float32", "float64", "int32", "int64",
                                      "complex64", "complex128")}


def _blas_dtype(dtype):
    if dtype in _BLAS_DTYPES:
        return dtype
    if dtype == np.dtype("float16"):
        return np.dtype("float32")
    return np.dtype("int64")  # bool and small/unsigned ints: exact mod 2**k


def _promote_operands(lhs, rhs):
    """Cast both operands of a BLAS op; returns ``(lhs, rhs, compute, result)``."""
    result = _normalize_dtype(np.matmul.resolve_dtypes((lhs.dtype, rhs.dtype, None))[-1])
    compute = _blas_dtype(result)
    return _cast_operand(lhs, compute), _cast_operand(rhs, compute), compute, result


def _normalize_axis(axis, ndims):
    axis = int(axis)
    if axis < -ndims or axis >= ndims:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndims}")
    return axis % ndims


def _axis_index(axis, ndim):
    """``axis`` as a non-negative index; numpy.exceptions.AxisError if out of range."""
    axis = int(axis)
    if not -ndim <= axis < ndim:
        raise np.exceptions.AxisError(axis, ndim)
    return axis % ndim


_SCAN_KINDS = {"cumsum": 0, "cumprod": 1, "nancumsum": 2, "nancumprod": 3}


def _scan(name, a, axis=None, dtype=None, out=None):
    """cumsum / cumprod / nancumsum / nancumprod along one axis (SCAN node)."""
    a = asarray(a)
    if axis is None:
        a = _reshape(a, (-1,))
        axis = 0
    axis = _normalize_axis(axis, a.ndims)
    if isinstance(a, _NDBase):
        from .folded import along_last_dims

        result = along_last_dims(a, axis, lambda b, ax: _scan(name, b, ax, dtype))
        if out is not None:
            out[...] = result
            return out
        return result
    if dtype is None:
        # NumPy's accumulation dtype (int64/uint64 for bool and small ints)
        dtype = getattr(np, name)(np.zeros(1, a.dtype)).dtype
    dtype = _normalize_dtype(dtype)
    src = _materialize(_astype(a, dtype))
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
    """sort / argsort along one axis; gathers on one chare (not scalable). NaNs last, stable."""
    if order is not None:
        raise NotImplementedError("sort(order=...) needs structured dtypes")
    a = asarray(a)
    if axis is None:
        a = _reshape(a, (-1,))
        axis = 0
    axis = _normalize_axis(axis, a.ndims)
    if isinstance(a, _NDBase):
        from .folded import along_last_dims

        return along_last_dims(a, axis, lambda b, ax: _sort(b, ax, kind, order, arg))
    src = _materialize(a)
    return nonfusible_op(
        src,
        axis,
        operation=ArrayOperation.argsort if arg else ArrayOperation.sort,
        shape=src.shape,
        dtype=np.dtype(np.int64) if arg else src.dtype,
    )


def _coerce_operand(value):
    if _is_array_nd(value):
        return value
    if value.__class__ is float or value.__class__ is int:
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
    return _ew(ArrayOperation.add, self, other, name="add", ufunc=np.add)


def _array_radd(self, other):
    return _ew(ArrayOperation.add, other, self, name="add", ufunc=np.add)


def _array_sub(self, other):
    return _ew(ArrayOperation.sub, self, other, name="subtract", ufunc=np.subtract)


def _array_rsub(self, other):
    return _ew(ArrayOperation.sub, other, self, name="subtract", ufunc=np.subtract)


def _array_neg(self):
    np.negative.resolve_dtypes((self.dtype, None))  # TypeError for bool
    if self.dtype.kind == "u":
        return _ew(ArrayOperation.sub, 0, self, name="negative", dtype=self.dtype)
    return _ew(ArrayOperation.mul, self, -1, name="negative", dtype=self.dtype)


def _array_mul(self, other):
    return _ew(ArrayOperation.mul, self, other, name="multiply", ufunc=np.multiply)


def _array_rmul(self, other):
    return _ew(ArrayOperation.mul, other, self, name="multiply", ufunc=np.multiply)


def _array_truediv(self, other):
    return _ew(ArrayOperation.div, self, other, name="divide", ufunc=np.true_divide)


def _array_rtruediv(self, other):
    return _ew(ArrayOperation.div, other, self, name="divide", ufunc=np.true_divide)


# ---------------------------------------------------------------------------
# Elementwise math, comparisons and selection (opcodes 20-49, 160-169).
# Comparisons return bool arrays; the backend converts operands in-kernel.
# ---------------------------------------------------------------------------


def _is_array(value):
    return isinstance(value, (Array, ArrayView))


def _is_array_nd(value):
    """Array, ArrayView or FoldedArray."""
    return isinstance(value, (Array, ArrayView, _NDBase))


def _is_int_dtype(dtype):
    return np.issubdtype(dtype, np.integer)


# opcode -> NumPy ufunc defining its dtype rules
_UFUNC = {
    ArrayOperation.tanh: np.tanh,
    ArrayOperation.exp: np.exp,
    ArrayOperation.sqrt: np.sqrt,
    ArrayOperation.log: np.log,
    ArrayOperation.log2: np.log2,
    ArrayOperation.log10: np.log10,
    ArrayOperation.log1p: np.log1p,
    ArrayOperation.exp2: np.exp2,
    ArrayOperation.expm1: np.expm1,
    ArrayOperation.abs: np.absolute,
    ArrayOperation.floor: np.floor,
    ArrayOperation.ceil: np.ceil,
    ArrayOperation.rint: np.rint,
    ArrayOperation.isfinite: np.isfinite,
    ArrayOperation.sin: np.sin,
    ArrayOperation.cos: np.cos,
    ArrayOperation.sign: np.sign,
    ArrayOperation.isnan: np.isnan,
    ArrayOperation.isinf: np.isinf,
    ArrayOperation.power: np.power,
    ArrayOperation.maximum: np.maximum,
    ArrayOperation.minimum: np.minimum,
    ArrayOperation.fmax: np.fmax,
    ArrayOperation.fmin: np.fmin,
    ArrayOperation.mod: np.remainder,
    ArrayOperation.floor_divide: np.floor_divide,
    ArrayOperation.conj: np.conjugate,
    ArrayOperation.bitwise_and: np.bitwise_and,
    ArrayOperation.bitwise_or: np.bitwise_or,
    ArrayOperation.bitwise_xor: np.bitwise_xor,
    ArrayOperation.invert: np.invert,
    ArrayOperation.left_shift: np.left_shift,
    ArrayOperation.right_shift: np.right_shift,
}


def _ew_unary(x, operation, name):
    return _ew(operation, x, name=name, ufunc=_UFUNC[operation])


def _ew_binary(lhs, rhs, operation, name):
    return _ew(operation, lhs, rhs, name=name, ufunc=_UFUNC[operation])


def _ew_operands(name, *operands):
    """Coerce elementwise operands; return ``(operands, shape, result_type)``."""
    operands = tuple(_coerce_operand(value) for value in operands)
    operands, shape = _check_shapes(name, operands)
    return operands, shape, _result_dtype(*operands)


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
_BOOL = np.dtype(bool)


def _const_mask(x, value):
    """All-True (value=1) or all-False (value=0) bool array shaped like integer array x."""

    operation = ArrayOperation.equal if value else ArrayOperation.not_equal
    return _ew(operation, x, x, name="mask", dtype=_BOOL)


def _compare_scalar_out_of_range(x, value, name):
    """``x <name> value`` for an out-of-range or non-integral scalar, computed exactly."""

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
    if float(value).is_integer():
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
    if _is_array(lhs) and not _is_array(rhs) and isinstance(rhs, numbers.Real):
        if lhs.dtype == _BOOL and rhs not in (0, 1):
            lhs = _astype(lhs, np.dtype(np.int64))
        if _is_int_dtype(lhs.dtype):
            info = np.iinfo(lhs.dtype)
            if isinstance(rhs, float) or not info.min <= rhs <= info.max:
                return _compare_scalar_out_of_range(lhs, rhs, name)
    operation, swap = _COMPARISONS[name]
    if swap:
        lhs, rhs = rhs, lhs
    return _ew(operation, lhs, rhs, name=name, ufunc=getattr(np, name))


def _truth(value):
    """bool array (array) or bool (scalar) for value != 0."""

    value = _coerce_operand(value)
    if _is_array(value):
        return _astype(value, _BOOL)
    return bool(value)


def _logical(operation, a, b, name, py):
    a, b = _truth(a), _truth(b)
    if not _is_array(a) and not _is_array(b):
        return py(a, b)
    return _ew(operation, a, b, name=name, dtype=_BOOL)


def _logical_not(x):
    x = _coerce_operand(x)
    if not _is_array(x):
        return not x
    return _ew(ArrayOperation.invert, x, name="logical_not", dtype=_BOOL)


def _logical_and(a, b):
    return _logical(ArrayOperation.bitwise_and, a, b, "logical_and", lambda p, q: p and q)


def _logical_or(a, b):
    return _logical(ArrayOperation.bitwise_or, a, b, "logical_or", lambda p, q: p or q)


def _logical_xor(a, b):
    return _logical(ArrayOperation.bitwise_xor, a, b, "logical_xor", lambda p, q: p != q)


def _bitwise(operation, ufunc, name):
    def op(self, other):
        return _ew(operation, self, other, name=name, ufunc=ufunc)

    def rop(self, other):
        return _ew(operation, other, self, name=name, ufunc=ufunc)

    return op, rop


_array_and, _array_rand = _bitwise(ArrayOperation.bitwise_and, np.bitwise_and, "bitwise_and")
_array_or, _array_ror = _bitwise(ArrayOperation.bitwise_or, np.bitwise_or, "bitwise_or")
_array_xor, _array_rxor = _bitwise(ArrayOperation.bitwise_xor, np.bitwise_xor, "bitwise_xor")
_array_lshift, _array_rlshift = _bitwise(ArrayOperation.left_shift, np.left_shift, "left_shift")
_array_rshift, _array_rrshift = _bitwise(ArrayOperation.right_shift, np.right_shift, "right_shift")


def _array_invert(self):
    return _ew(ArrayOperation.invert, self, name="invert", ufunc=np.invert)


def _real(x):
    """Real part (a copy for real arrays)."""
    x = _coerce_operand(x)
    dtype = np.real(np.zeros(1, x.dtype)).dtype
    return _ew(ArrayOperation.real, x, name="real", dtype=dtype)


def _imag(x):
    """Imaginary part (zeros for real arrays)."""
    x = _coerce_operand(x)
    dtype = np.imag(np.zeros(1, x.dtype)).dtype
    return _ew(ArrayOperation.imag, x, name="imag", dtype=dtype)


def _power(base, exponent):
    base = _coerce_operand(base)
    exponent = _coerce_operand(exponent)
    if _is_array(base) and not _is_array(exponent) and isinstance(exponent, numbers.Real):
        if _is_int_dtype(base.dtype) and isinstance(exponent, int) and exponent < 0:
            raise ValueError("Integers to negative integer powers are not allowed.")
        if exponent == 2 and base.dtype != _BOOL:
            # same fast path as NumPy (x**2 -> square); pow(x, 2) == x*x exactly
            return _array_mul(base, base)
    return _ew_binary(base, exponent, ArrayOperation.power, "power")


def _reciprocal(x):
    x = _coerce_operand(x)
    if _is_array(x) and (_is_int_dtype(x.dtype) or x.dtype == _BOOL):
        # integer 1/x truncates: 1 for 1, -1 for -1, else 0 (0 also for x == 0)
        np.reciprocal.resolve_dtypes((x.dtype, None))
        return _astype(_compare(x, 1, "equal"), x.dtype) - _astype(_compare(x, -1, "equal"),
                                                                   x.dtype)
    return _ew(ArrayOperation.div, 1, x, name="reciprocal", dtype=np.reciprocal.resolve_dtypes(
        (_np_operand(x), None))[-1])


def _round(x, decimals=0):
    """Round half to even, with NumPy's algorithm for ``decimals != 0``."""

    (x,), _shape, dtype = _ew_operands("round", x)
    decimals = int(decimals)
    if _is_int_dtype(dtype) or dtype == _BOOL:
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
    np.trunc.resolve_dtypes((dtype, None))  # TypeError for complex
    if _is_int_dtype(dtype) or dtype == _BOOL:
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
    x, y = _ew_operand(x), _ew_operand(y)
    return _ew(ArrayOperation.where, condition, x, y, name="where", dtype=_result_dtype(x, y))


def _clip(a, a_min=None, a_max=None):
    if a_min is None and a_max is None:
        raise ValueError("One of max or min must be given")
    if a_min is None:
        return _ew_binary(a, a_max, ArrayOperation.minimum, "clip")
    if a_max is None:
        return _ew_binary(a, a_min, ArrayOperation.maximum, "clip")
    a, a_min, a_max = (_ew_operand(v) for v in (a, a_min, a_max))
    return _ew(ArrayOperation.clip, a, a_min, a_max, name="clip",
               dtype=_result_dtype(a, a_min, a_max))


def _log(x, base=None):
    if base is None or base == np.e:
        return _ew_unary(x, ArrayOperation.log, "log")
    if base == 2:
        return _ew_unary(x, ArrayOperation.log2, "log2")
    if base == 10:
        return _ew_unary(x, ArrayOperation.log10, "log10")
    return _ew_unary(x, ArrayOperation.log, "log") * float(1.0 / np.log(base))


def _square(x):
    """numpy.square (bool squares in int8, as NumPy has no bool loop)."""
    if x.dtype == _BOOL:
        x = _astype(x, np.dtype(np.int8))
    return _array_mul(x, x)


def _array_pow(self, other):
    if other.__class__ is int and other == 2:
        return _square(self)  # NumPy's x**2 fast path
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
    """Elementwise methods shared by Array and ArrayView."""

    __slots__ = ()

    # NumPy ndarray methods
    def round(self, decimals=0):
        return _round(self, decimals)

    @property
    def real(self):
        return _real(self)

    @property
    def imag(self):
        return _imag(self)

    def conj(self):
        return _ew_unary(self, ArrayOperation.conj, "conjugate")

    conjugate = conj

    def clip(self, min=None, max=None):
        return _clip(self, min, max)

    # Legacy method API
    def exp(self):
        return _ew_unary(self, ArrayOperation.exp, "exp")

    def tanh(self):
        return _ew_unary(self, ArrayOperation.tanh, "tanh")

    def log(self, base=np.e):
        return _log(self, base)

    def log2(self):
        return _log(self, 2)

    def log10(self):
        return _log(self, 10)

    def sqrt(self):
        return _ew_unary(self, ArrayOperation.sqrt, "sqrt")

    def abs(self):
        return _array_abs(self)

    def absolute(self):
        return _array_abs(self)

    def negate(self):
        return -self

    def square(self):
        return _square(self)

    def reciprocal(self):
        return _reciprocal(self)

    def sin(self):
        return _ew_unary(self, ArrayOperation.sin, "sin")

    def cos(self):
        return _ew_unary(self, ArrayOperation.cos, "cos")

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

    # max/min are installed on Array/ArrayView by reductions.py.

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


def _batched_matmul(a, b):
    """NumPy matmul through einsum, for operands the 2-D/3-D-view path cannot take."""
    from .einsum import einsum

    la = "...ij" if a.ndims >= 2 else "j"
    lb = "...jk" if b.ndims >= 2 else "j"
    out = "..." + ("i" if a.ndims >= 2 else "") + ("k" if b.ndims >= 2 else "")
    return einsum(f"{la},{lb}->{out}", a, b)


def _needs_batched(x):
    # 3-D operands with exactly one size-1 dim are matrices to SUMMA/matvec
    return isinstance(x, _NDBase) or (x.ndims == 3 and sum(d == 1 for d in x.shape) != 1)


def _array_matmul(self, other):
    other = _coerce_operand(other)
    if not _is_array_nd(other):
        raise ValueError("matmul does not support scalar operands")
    if _needs_batched(self) or _needs_batched(other):
        return _batched_matmul(self, other)
    self, other, compute, result = _promote_operands(self, other)
    out = _matmul_blas(self, other)
    return out if compute == result else _astype(out, result)


def _dot(self, other):
    """numpy.dot: sums self's last axis with other's second-to-last (matmul only up to 2-D)."""
    other = _coerce_operand(other)
    if _is_scalar_like(other) or not self.shape or not other.shape:
        return self * other
    na, nb = len(self.shape), len(other.shape)
    if na <= 2 and nb <= 2:
        return _array_matmul(self, other)
    from .einsum import tensordot

    return tensordot(self, other, axes=([na - 1], [max(nb - 2, 0)]))


def _matmul_blas(self, other):
    """matmul of two operands of the same BLAS dtype."""

    if self.ndims == 1 and other.ndims == 1:
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                f"Shape mismatch for dot product: ({self.shape[0]},) @ ({other.shape[0]},)"
            )
        return nonfusible_op(
            _materialize(self),
            _materialize(other),
            operation=ArrayOperation.reduce,
            shape=(),
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


def _unstrided(x):
    """Copy a view with a non-unit step (the matmul executors ignore view steps)."""

    if isinstance(x, ArrayView) and any(st != 1 for st in x.region.step):
        return x.copy()
    return x


def _matvec(mat_obj, vec):
    """Matrix-vector multiply."""

    if not isinstance(vec, (Array, ArrayView)):
        raise ValueError("matvec requires an Array vector")
    mat_obj, vec = _unstrided(mat_obj), _unstrided(vec)

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
    lhs, rhs = _unstrided(lhs), _unstrided(rhs)

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
    """Permute the axes of ``x``; identity permutations return ``x``. >3-D goes through einsum."""

    if not x.shape:  # 0-d (stored as one element)
        _normalize_axes(axes, 0)
        return x
    perm = _normalize_axes(axes, x.ndims)
    if perm == tuple(range(x.ndims)):
        return x
    if x.ndims > 3:
        from .einsum import permute

        return permute(x, perm)

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


def _transpose_view(x, axes=None):
    """User-facing transpose: a copy linked to ``x`` like a NumPy view (see _basic_getitem)."""
    result = _transpose(x, axes)
    if result is x or isinstance(x, _NDBase) or x.ndims > 3:
        return result
    perm = _normalize_axes(axes, x.ndims)
    inv = tuple(int(i) for i in np.argsort(perm))

    def back(root):
        x[...] = _transpose(root, inv)

    result._view_of = (x, back, lambda: _transpose(x, perm))
    result._seen = _version(x)
    return result


def _swapaxes(x, axis1, axis2):
    ndim = x.ndims
    for axis in (axis1, axis2):
        if axis < -ndim or axis >= ndim:
            raise np.exceptions.AxisError(axis, ndim)
    perm = list(range(ndim))
    a, b = int(axis1) % ndim, int(axis2) % ndim
    perm[a], perm[b] = perm[b], perm[a]
    return _transpose_view(x, perm)


def _transpose_method(self, *axes):
    if len(axes) == 0 or (len(axes) == 1 and axes[0] is None):
        return _transpose_view(self)
    if len(axes) == 1 and not isinstance(axes[0], numbers.Integral):
        return _transpose_view(self, axes[0])
    return _transpose_view(self, axes)


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


class _ArrayMethods(object):
    """Members shared by Array and ArrayView."""

    __slots__ = ()

    __array_priority__ = 1000

    @property
    def itemsize(self):
        return 0 if self.dtype is None else self.dtype.itemsize

    @property
    def nbytes(self):
        return int(self.size) * self.itemsize

    def __getitem__(self, key):
        if _is_indexing_key(key):
            from .indexing import getitem

            return getitem(self, key)
        return _basic_getitem(self, key)

    def __setitem__(self, key, rhs):
        if _is_indexing_key(key):
            from .indexing import setitem

            return setitem(self, key, rhs)
        _setitem(self, key, rhs)

    @property
    def T(self):
        return _transpose_view(self)

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

    __and__ = _array_and
    __rand__ = _array_rand

    __or__ = _array_or
    __ror__ = _array_ror

    __xor__ = _array_xor
    __rxor__ = _array_rxor

    __lshift__ = _array_lshift
    __rlshift__ = _array_rlshift

    __rshift__ = _array_rshift
    __rrshift__ = _array_rrshift

    __invert__ = _array_invert

    def dot(self, other, out=None):
        return _write_out(_dot(self, other), out)

    def take(self, indices, axis=None, out=None, mode="raise"):
        from .indexing import take

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


class Array(FrontendObject, _ArrayMethods, _ElementwiseMethods, _HostConversions):
    # __eq__ is elementwise; keep FrontendObject's name-based hash
    __hash__ = FrontendObject.__hash__

    def __init__(self, **kwargs):
        dtype = _normalize_dtype(kwargs.pop("dtype", None), allow_none=True)
        self.dtype = dtype
        self._wire_dtype = DType._from_numpy[self.dtype] if self.dtype is not None else 0

        shape = kwargs.pop("shape", None)
        if shape is not None:
            shape = _normalize_shape(shape)

        super().__init__(**kwargs)

        self._shape = tuple(shape) if shape is not None else tuple(self.region.shape())
        # 0-d arrays are stored as (1,) arrays; ndims is the storage rank
        self.ndims = len(self._shape) or 1
        if self.ndims > 3:
            raise NotImplementedError(
                f"charmnumeric arrays must have 0 to 3 dimensions, got shape {self._shape}"
            )
        _RANKS_USED.add(self.ndims)
        self._size = _shape_size(self._shape)

        if self.region.is_global and self._shape is not None:
            self.region = _full_region(_storage_shape(self._shape))

    def wire_dtype(self):
        return self._wire_dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return _CallableInt(self._size)

    def __len__(self):
        if not self._shape:
            raise TypeError("len() of unsized object")
        return self._shape[0]

    def __repr__(self):
        dtype_name = None if self.dtype is None else self.dtype.name
        return f"Array(shape={self.shape}, dtype={dtype_name}, name={self.name})"

    def get(self, interface):
        if self._size == 0:
            return np.empty(self._shape, dtype=self.dtype)
        _fresh(self)
        execute(interface)
        data = interface.get(self.ndims, self.name, self.size(), dtype=self.dtype)
        return data.reshape(self._shape)

    def get_region(self, region, **kwargs):
        kwargs.setdefault("shape", region.shape())
        kwargs.setdefault("dtype", self.dtype)
        kwargs.setdefault("wire_dtype", self._wire_dtype)
        kwargs.setdefault("size", _shape_size(kwargs["shape"]))
        return ArrayView(self, region, **kwargs)

    def set_region(self, region, rhs, **kwargs):
        # all writes (incl. via views) land here, except SCATTER (indexing.py)
        _fresh(self)
        if rhs.__class__ is Array or rhs.__class__ is ArrayView:
            _fresh(rhs)
        result = super().set_region(region, rhs, **kwargs)
        _wrote(self)
        return result


class ArrayView(FrontendObjectView, _ArrayMethods, _ElementwiseMethods, _HostConversions):
    """Array-specific view with arithmetic operators and slicing."""

    # __eq__ is elementwise; keep identity hashing
    __hash__ = object.__hash__

    @property
    def ndim(self):
        return self.ndims

    @property
    def size(self):
        return _CallableInt(self._size)

    def __len__(self):
        if self.ndims == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

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
        if self.size == 0:
            return np.empty(self.shape, dtype=self.dtype)
        materialized = empty(self.shape, dtype=self.dtype)
        materialized[...] = self
        return materialized.get(interface)


def create_array(shape, dtype=_DEFAULT_DTYPE, **kwargs):
    shape = _normalize_shape(shape)
    dtype = _normalize_dtype(dtype)
    if len(shape) > 3:
        from .folded import FoldedArray, fold_shape

        return FoldedArray(create_array(fold_shape(shape), dtype=dtype, **kwargs), shape)
    return create_object(
        Array,
        region=_full_region(_storage_shape(shape)),
        dtype=dtype,
        shape=shape,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Array-creation nodes (UPLOAD / RANDOM / IOTA)
#
# One non-fusible node per array, filled chare by chare. Parameters travel as
# 64-bit words, each split into two exact SCALAR operands (high and low 32 bits).
# ---------------------------------------------------------------------------
_U64_MASK = (1 << 64) - 1


def _u64_from_int(value):
    return int(value) & _U64_MASK


def _u64_from_double(value):
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _u64_from_value(value, dtype):
    """64-bit parameter word for a value of `dtype` (int64, or double bits of the real part)."""
    if np.dtype(dtype).kind in "fc":
        return _u64_from_double(np.real(value))
    return _u64_from_int(value)


def _creation_op(operation, shape, dtype, words=(), payload=None):
    shape = _normalize_shape(shape)
    dtype = _normalize_dtype(dtype)
    if len(shape) > 3:
        # C order is the same in the folded 3-D storage (element-indexed generators)
        from .folded import FoldedArray, fold_shape

        return FoldedArray(_creation_op(operation, fold_shape(shape), dtype, words, payload), shape)
    region = _full_region(_storage_shape(shape))
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


def _upload_pieces(shape, itemsize, limit):
    """Slice keys cutting a C-order array of `shape` into pieces of at most `limit` bytes."""
    inner = itemsize * _shape_size(shape[1:])
    if inner <= limit or len(shape) == 1:
        step = max(1, limit // inner)
        for i in range(0, shape[0], step):
            yield (slice(i, min(i + step, shape[0])),)
    else:
        for i in range(shape[0]):
            for key in _upload_pieces(shape[1:], itemsize, limit):
                yield (slice(i, i + 1),) + key


def _upload(host):
    """Create an array holding the host ndarray `host` (one UPLOAD node per message-sized piece)."""
    host = np.ascontiguousarray(host)
    if host.size == 0:
        return create_array(host.shape, dtype=host.dtype)
    if host.ndim > 3:
        from .folded import FoldedArray, fold_shape

        return FoldedArray(_upload(host.reshape(fold_shape(host.shape))), host.shape)
    limit = _ct_ast.max_inline_bytes
    if host.nbytes <= limit:
        return _creation_op(ArrayOperation.upload, host.shape, host.dtype, payload=host.tobytes())
    result = create_array(host.shape, dtype=host.dtype)
    for key in _upload_pieces(host.shape, host.itemsize, limit):
        piece = host[key]
        result[key] = _creation_op(ArrayOperation.upload, piece.shape, host.dtype,
                                   payload=piece.tobytes())
    return result


def _array_from_host_data(values, dtype=None, copy=True, ndmin=0):
    requested_dtype = _normalize_dtype(dtype, allow_none=True)

    # No host copy needed for copy=True: uploads snapshot the bytes before returning.
    host = np.asarray(values, dtype=requested_dtype)

    while host.ndim < ndmin:
        host = np.expand_dims(host, axis=0)

    host_dtype = _normalize_dtype(host.dtype)
    host = host.astype(host_dtype, copy=False)
    if host.size == 0:
        return create_array(host.shape, dtype=host_dtype)
    # Constant data is cheaper as a scalar fill than as an upload, but scalars
    # travel as doubles: complex values and ints beyond 2**53 must be uploaded.
    first = host.reshape(-1)[0]
    if (host.dtype.kind == "c" or (host.dtype.kind in "iu" and abs(int(first)) > 2**53)
            or (host.size > 1 and not np.all(host == first))):
        return _upload(host)
    if host.dtype.kind == "f" and first == 0 and np.any(np.signbit(host) != np.signbit(first)):
        return _upload(host)
    result = create_array(host.shape, dtype=host_dtype)
    result[...] = _coerce_scalar(first)
    return result


def _shape_dtype_from_like(obj):
    if _is_array_nd(obj):
        return obj.shape, obj.dtype

    host = np.asarray(obj)
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

    if isinstance(x, _NDBase):
        raise NotImplementedError("this operation does not support arrays with more than 3 dims")
    if isinstance(x, ArrayView) and not _is_whole(x):
        return x.copy()
    return x


def _expand_scalar_array(value, shape):
    """Non-broadcast set_region source holding a size-1 array's value."""

    def ones_view(arr):
        # one-element view of a 2-element array: a regular (non-broadcast) operand
        nd = arr.ndims
        pair = zeros((1,) * (nd - 1) + (2,), dtype=arr.dtype) + arr
        return pair[(slice(None),) * (nd - 1) + (slice(0, 1),)]

    if value.ndims != len(shape):
        value = _size1_to_rank(value, len(shape))
    if _shape_size(shape) > 1:
        return zeros(shape, dtype=value.dtype) + value
    return ones_view(value)


def _has_negative_step(key):
    items = key if isinstance(key, tuple) else (key,)
    return any(isinstance(k, slice) and k.step is not None and k.step < 0 for k in items)


def _view(x, region):
    shape = region.shape()
    return x.get_region(region, shape=shape, dtype=x.dtype, wire_dtype=x.wire_dtype(),
                        size=_shape_size(shape))


def _check_0d_key(key):
    items = key if isinstance(key, tuple) else (key,)
    if all(k is None or k is Ellipsis for k in items) and sum(k is Ellipsis for k in items) <= 1:
        return
    n = len(key) if isinstance(key, tuple) else 1
    raise IndexError(f"too many indices for array: array is 0-dimensional, but {n} were indexed")


def _basic_getitem(x, key):
    """x[key] for ints, slices, ellipsis and newaxis (NumPy basic indexing).

    Positive-step slices give views. Integer, newaxis and negative-step results
    are copies linked to ``x``, so writes reach ``x`` and reads refresh from it.
    """
    shape = x.shape
    if not shape:
        _check_0d_key(key)
        items = key if key.__class__ is tuple else (key,)
        nnew = sum(k is None for k in items)
        if nnew:
            shape = (1,) * nnew
            result = _reshape3(x, shape)
            result._view_of = (x, lambda r: x.__setitem__(Ellipsis, r), lambda: _reshape3(x, shape))
            result._seen = _version(x)
            return result
        return x.copy() if key == () else x  # x[()] is a scalar copy, x[...] a view
    items = key if key.__class__ is tuple else (key,)
    for item in items:
        if item.__class__ is not slice and item is not Ellipsis:
            break
    else:
        if not _has_negative_step(key):
            return _view(x, _parse_key(key, shape, x.ndims))
    items = _expand_key(key, x.ndims, keep_newaxis=True)
    plain = tuple(item for item in items if item is not None)
    if _has_negative_step(plain):
        from .signal import flip, split_negative_steps

        pos, axes = split_negative_steps(plain, shape)
        kept = tuple(slice(int(k) % n, int(k) % n + 1) if k.__class__ is not slice else k
                     for k, n in zip(pos, shape))

        def source():
            return flip(_view(x, _parse_key(kept, shape, x.ndims)), axes)
    else:
        region = _parse_key(plain, shape, x.ndims)

        def source():
            return _view(x, region)
    view = source()
    dims = iter(view.shape)
    out = []
    for item in items:
        if item is None:
            out.append(1)
        elif item.__class__ is slice:
            out.append(next(dims))
        else:
            next(dims)
    out = tuple(out)
    if _shape_size(out) > 1 and len(items) == len(plain) and not _has_negative_step(plain):
        vkey = tuple(_FULL_SLICE if k.__class__ is slice else 0 for k in plain)
        return _lazy_copy(x, plain, source, out, view, vkey)
    result = _reshape_copy(view, out)
    if out:
        result._view_of = (x, plain, source)
        result._seen = _version(x)
    return result


def _version(x):
    root = x._root_parent if x.__class__ is ArrayView else x
    return root.__dict__.get("_version", 0)


def _fresh(x):
    """Refresh a linked copy (see _basic_getitem), or the root of a view of one."""
    root = x._root_parent if x.__class__ is ArrayView else x
    link = root.__dict__.get("_view_of") if root.__class__ is Array else None
    if link is None:
        return
    if root.__dict__.get("_pending_create"):
        root._pending_create = False
        root.set_active_ast(ASTNode(Operation.create, [root, root.region], root.name))
        state = get_execution_state()
        state.track_object(root)
        state.add_node(NonFusibleNode(obj=root))
    parent, key, source = link
    _fresh(parent)
    version = _version(parent)
    if root._seen != version:
        drop = root.__dict__.get("_drop")
        root._view_of = None  # neither refresh nor write back while copying
        root[...] = source()
        root._view_of = link
        root._seen = version
        if drop is not None:
            root._drop = drop


def _fresh_operands(operands):
    for op in operands:
        if op.__class__ is Array or op.__class__ is ArrayView:
            _fresh(op)


def _wrote(root):
    """Bookkeeping after an in-place write of the Array ``root``."""
    d = root.__dict__
    d.pop("_transpose_cache", None)
    d.pop("_rank_cache", None)
    d.pop("_drop", None)
    d.pop("_iota_recipe", None)
    d.pop("_rng_recipe", None)
    d["_version"] = d.get("_version", 0) + 1
    link = d.get("_view_of")
    if link is not None:
        # a linked copy: write it back to its array (see _basic_getitem)
        d["_view_of"] = None
        if callable(link[1]):
            link[1](root)
        else:
            link[0][link[1]] = root
        d["_view_of"] = link
        d["_seen"] = _version(link[0])


def nonfusible_op(*operands, operation, **kwargs):
    _fresh_operands(operands)
    return _nonfusible_op(*operands, operation=operation, **kwargs)


def _is_same_view(self, region, rhs):
    """True if rhs is the view self[region] (x[k] op= y assigns x[k] to itself)."""
    if not isinstance(rhs, ArrayView) or rhs.dtype != self.dtype:
        return False
    root = self._root_parent if isinstance(self, ArrayView) else self
    if rhs._root_parent is not root:
        return False
    target = self._view_region.compose(region) if isinstance(self, ArrayView) else region
    return rhs.region == target


def _trim_leading_ones(shape, ndim):
    """``shape`` without leading size-1 dims beyond ``ndim`` dims."""
    while len(shape) > ndim and shape[0] == 1:
        shape = shape[1:]
    return shape


def _broadcasts_to(src, shape):
    """True if shape ``src`` broadcasts to exactly ``shape``."""
    try:
        return len(src) <= len(shape) and tuple(np.broadcast_shapes(src, shape)) == shape
    except ValueError:
        return False


def _fit_assignment(rhs, shape):
    """Bring an assigned array to the target region shape (NumPy assignment rules)."""
    rshape = tuple(rhs.shape)
    if len(rshape) != len(shape) and _squeeze_shape(rshape) == _squeeze_shape(shape):
        return rhs  # only size-1 dims differ: cross-ndim set_region
    trimmed = _trim_leading_ones(rshape, len(shape))
    if not _broadcasts_to(trimmed, shape):
        if _squeeze_shape(rshape) == _squeeze_shape(shape):
            return _reshape(rhs, shape)
        raise ValueError(
            f"could not broadcast input array from shape {rshape} into shape {shape}"
        )
    if trimmed != rshape:
        rhs = _reshape(rhs, trimmed)
    return _broadcast_to(rhs, shape)


def _setitem(self, key, rhs):
    pshape = self.shape
    if not pshape:
        _check_0d_key(key)
        key, pshape = (_FULL_SLICE,), (1,)
    items = _expand_key(key, self.ndims, keep_newaxis=True)
    key = tuple(item for item in items if item is not None)
    if _has_negative_step(key):
        from .signal import flip, split_negative_steps

        key, axes = split_negative_steps(key, pshape)
        if isinstance(rhs, (Array, ArrayView, np.ndarray, list, tuple)):
            if not isinstance(rhs, (Array, ArrayView)):
                rhs = np.asarray(rhs)
            # flipped dims of the target x[key] (ints dropped), aligned to the right of rhs
            tdims = [d for d, k in enumerate(key) if k.__class__ is slice]
            lead = len(tdims) - rhs.ndim
            raxes = tuple(i - lead for i, d in enumerate(tdims)
                          if d in axes and i >= lead and rhs.shape[i - lead] > 1)
            if raxes:
                rhs = np.flip(rhs, raxes) if isinstance(rhs, np.ndarray) else flip(rhs, raxes)
    region = _parse_key(key, pshape, self.ndims)
    shape = tuple(region.shape())
    # NumPy's shape of x[key]: ints drop their dim, newaxis adds one
    dims = iter(shape)
    vshape = []
    for item in items:
        if item is None:
            vshape.append(1)
        elif item.__class__ is slice:
            vshape.append(next(dims))
        else:
            next(dims)
    vshape = tuple(vshape) if self.shape else ()
    if isinstance(rhs, _NDBase):
        from .folded import to_shape

        rhs = to_shape(rhs, shape)
    if _is_same_view(self, region, rhs):
        return
    rhs = _coerce_assignment_operand(rhs, vshape, self.dtype)
    if isinstance(rhs, complex) and self.dtype.kind != "c":
        raise TypeError(f"can't convert complex to {self.dtype}")
    if isinstance(rhs, int) and self.dtype.kind in "iu":
        self.dtype.type(rhs)  # OverflowError like NumPy
    if isinstance(rhs, complex) or (isinstance(rhs, int) and abs(rhs) > 2**53
                                    and self.dtype.kind in "iu"):
        rhs = _upload(np.array([rhs], dtype=self.dtype))  # not exact as a double
    if _shape_size(shape) == 0:
        return
    drop = rhs.__dict__.get("_drop") if rhs.__class__ is Array else None
    if drop is not None and _squeeze_shape(drop[0].shape) == _squeeze_shape(shape):
        rhs = drop[0]  # lazy int-index copy: assign from its source view
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
        rhs = _fit_assignment(rhs, shape)
    self.set_region(region, rhs, shape=shape, dtype=self.dtype)


def _reshape_copy(x, shape):
    """New Array of ``shape`` holding ``x`` (shapes differ only in size-1 dims)."""

    if len(shape) > 3 or isinstance(x, _NDBase):
        return _reshape(x, shape)
    result = empty(shape, dtype=x.dtype)
    result[...] = x
    return result


# ---------------------------------------------------------------------------
# Shape manipulation (RESHAPE / BROADCAST nodes, src/executor_shape.cpp).
# Results are new arrays, never views.
# ---------------------------------------------------------------------------


def _infer_shape(newshape, size):
    """Target shape with -1 resolved, checked against ``size``."""
    if isinstance(newshape, numbers.Integral):
        newshape = (newshape,)
    newshape = tuple(int(d) for d in newshape)
    if any(d < -1 for d in newshape):
        raise ValueError("negative dimensions not allowed")
    unknown = [i for i, d in enumerate(newshape) if d == -1]
    if len(unknown) > 1:
        raise ValueError("can only specify one unknown dimension")
    if unknown:
        known = _shape_size([d for d in newshape if d != -1])
        if known == 0 or size % known:
            raise ValueError(f"cannot reshape array of size {size} into shape {newshape}")
        newshape = tuple(size // known if d == -1 else d for d in newshape)
    if _shape_size(newshape) != size:
        raise ValueError(f"cannot reshape array of size {size} into shape {newshape}")
    return newshape


def _reshape3(x, shape):
    """Reshape a <=3-D array or view to a <=3-D shape (x itself if unchanged)."""
    shape = tuple(shape)
    if shape == tuple(x.shape):
        return x
    size = _shape_size(shape)
    if size == 0:
        return create_array(shape, dtype=x.dtype)
    if size == 1 or (len(shape) != x.ndims and _squeeze_shape(shape) == _squeeze_shape(x.shape)):
        # only size-1 dims change: cross-ndim set_region, no RESHAPE node
        result = empty(shape, dtype=x.dtype)
        result[...] = x
        return result
    return nonfusible_op(x, *shape, operation=ArrayOperation.reshape, shape=shape, dtype=x.dtype)


def _reshape(x, newshape, order="C"):
    """C-order reshape into a new array; returns ``x`` if the shape is unchanged."""
    if order not in ("C", "A", "K"):
        raise NotImplementedError(f"reshape(order={order!r}): only C order is supported")
    shape = _infer_shape(newshape, int(x.size))
    if shape == tuple(x.shape):
        return x
    if len(shape) > 3 or isinstance(x, _NDBase):
        from .folded import reshape_nd

        return reshape_nd(x, shape)
    return _reshape3(x, shape)


def _broadcast_to(x, shape):
    """numpy.broadcast_to as a new array (``x`` itself if the shape is unchanged)."""
    shape = _normalize_shape(shape)
    xs = tuple(x.shape)
    if not _broadcasts_to(xs, shape):
        raise ValueError(
            f"operands could not be broadcast together with remapped shapes "
            f"[original->remapped]: {xs} and requested shape {shape}"
        )
    if xs == shape:
        return x
    if len(shape) > 3 or isinstance(x, _NDBase):
        from .folded import broadcast_nd

        return broadcast_nd(x, shape)
    if _shape_size(shape) == 0:
        return create_array(shape, dtype=x.dtype)
    if int(x.size) == 1:
        result = empty(shape, dtype=x.dtype)
        result[...] = x
        return result
    return nonfusible_op(x, *shape, operation=ArrayOperation.broadcast, shape=shape, dtype=x.dtype)


def _shape_arg(shape):
    if len(shape) == 1 and not isinstance(shape[0], numbers.Integral):
        return shape[0]
    return shape


def _squeeze(a, axis=None):
    shape = tuple(a.shape)
    if axis is None:
        axes = {d for d, n in enumerate(shape) if n == 1}
    else:
        if isinstance(axis, numbers.Integral):
            axis = (axis,)
        axes = set()
        for ax in axis:
            ax = int(ax)
            if not -len(shape) <= ax < len(shape):
                raise np.exceptions.AxisError(ax, len(shape))
            ax %= len(shape)
            if shape[ax] != 1:
                raise ValueError("cannot select an axis to squeeze out which has size not equal to one")
            axes.add(ax)
    new = tuple(n for d, n in enumerate(shape) if d not in axes) or (1,)
    return _reshape(a, new)


def _array_reshape(self, *shape, order="C"):
    return _reshape(self, _shape_arg(shape), order)


def _array_ravel(self, order="C"):
    return _reshape(self, (-1,), order)


def _array_flatten(self, order="C"):
    result = _reshape(self, (-1,), order)
    return result.copy() if result is self else result


def _array_squeeze(self, axis=None):
    return _squeeze(self, axis)


def _inplace(fn):
    """In-place operator: writes into self (NumPy's same_kind output casting)."""

    def op(self, other):
        result = fn(self, other)
        if not np.can_cast(result.dtype, self.dtype, "same_kind"):
            if not self.shape:
                return result  # a 0-d result stands for a NumPy scalar, which rebinds
            raise TypeError(
                f"Cannot cast ufunc output from {result.dtype!r} to {self.dtype!r} "
                "with casting rule 'same_kind'"
            )
        self[...] = result
        return self

    return op


def reshape(a, newshape=None, order="C", *, shape=None):
    """numpy.reshape (C order); a new array unless the shape is unchanged."""
    if shape is not None:
        newshape = shape
    if newshape is None:
        raise TypeError("reshape() missing required argument 'shape'")
    return _reshape(asarray(a), newshape, order)


def ravel(a, order="C"):
    return _reshape(asarray(a), (-1,), order)


def squeeze(a, axis=None):
    return _squeeze(asarray(a), axis)


def expand_dims(a, axis):
    a = asarray(a)
    if isinstance(axis, numbers.Integral):
        axis = (axis,)
    nd = a.ndim + len(axis)
    axes = set()
    for ax in axis:
        ax = int(ax)
        if not -nd <= ax < nd:
            raise np.exceptions.AxisError(ax, nd)
        ax %= nd
        if ax in axes:
            raise ValueError("repeated axis")
        axes.add(ax)
    dims = iter(a.shape)
    return _reshape(a, tuple(1 if d in axes else next(dims) for d in range(nd)))


def _atleast(arys, fn):
    res = [fn(asarray(a)) for a in arys]
    return res[0] if len(res) == 1 else tuple(res)


def atleast_1d(*arys):
    return _atleast(arys, lambda a: a)


def atleast_2d(*arys):
    return _atleast(arys, lambda a: _reshape(a, (1,) + tuple(a.shape)) if a.ndim == 1 else a)


def atleast_3d(*arys):
    def fn(a):
        if a.ndim == 1:
            return _reshape(a, (1,) + tuple(a.shape) + (1,))
        if a.ndim == 2:
            return _reshape(a, tuple(a.shape) + (1,))
        return a

    return _atleast(arys, fn)


def broadcast_to(array, shape, subok=False):
    """numpy.broadcast_to; returns a new (materialized) array, not a read-only view."""
    return _broadcast_to(asarray(array), shape)


def broadcast_arrays(*args, subok=False):
    args = [asarray(a) for a in args]
    shape = tuple(np.broadcast_shapes(*[tuple(a.shape) for a in args]))
    return tuple(_broadcast_to(a, shape) for a in args)


broadcast_shapes = np.broadcast_shapes


_ND_DUNDERS = (
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__",
    "__rtruediv__", "__pow__", "__rpow__", "__mod__", "__rmod__", "__floordiv__",
    "__rfloordiv__", "__divmod__", "__rdivmod__", "__lt__", "__le__", "__gt__", "__ge__",
    "__eq__", "__ne__", "__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__", "__lshift__",
    "__rlshift__", "__rshift__", "__rrshift__",
)

for _cls in (Array, ArrayView):
    for _name in _ND_DUNDERS:
        setattr(_cls, _name, _nd_aware(getattr(_cls, _name)))
    _cls.reshape = _array_reshape
    _cls.ravel = _array_ravel
    _cls.flatten = _array_flatten
    _cls.squeeze = _array_squeeze
    _cls.__iadd__ = _inplace(_cls.__add__)
    _cls.__isub__ = _inplace(_cls.__sub__)
    _cls.__imul__ = _inplace(_cls.__mul__)
    _cls.__itruediv__ = _inplace(_cls.__truediv__)
    _cls.__ifloordiv__ = _inplace(_cls.__floordiv__)
    _cls.__imod__ = _inplace(_cls.__mod__)
    _cls.__ipow__ = _inplace(_cls.__pow__)
    _cls.__iand__ = _inplace(_cls.__and__)
    _cls.__ior__ = _inplace(_cls.__or__)
    _cls.__ixor__ = _inplace(_cls.__xor__)
    _cls.__ilshift__ = _inplace(_cls.__lshift__)
    _cls.__irshift__ = _inplace(_cls.__rshift__)
del _cls, _name


def _write_out(result, out):
    """NumPy ``out=``: ``out[...] = result`` (out may be a view); returns out."""

    if out is None:
        return result
    if isinstance(out, tuple):
        if len(out) != 1:
            raise ValueError("out must be a single array or a 1-tuple")
        out = out[0]
    if not _is_array_nd(out):
        raise TypeError("out must be a charmnumeric Array or ArrayView")

    if _is_array_nd(result):
        if int(result.size) != 1 and tuple(result.shape) != tuple(out.shape):
            raise ValueError(
                f"non-broadcastable output operand with shape {tuple(out.shape)} "
                f"doesn't match the broadcast shape {tuple(result.shape)}"
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

    if _is_array_nd(a):
        if requested_dtype is None or requested_dtype == a.dtype:
            return a
        return a.astype(requested_dtype)

    return _array_from_host_data(a, dtype=requested_dtype, copy=False)


def array(a, dtype=None, copy=True, ndmin=0):
    requested_dtype = _normalize_dtype(dtype, allow_none=True)

    if _is_array_nd(a):
        target_dtype = a.dtype if requested_dtype is None else requested_dtype
        target_shape = _apply_ndmin(a.shape, ndmin)
        if isinstance(a, _NDBase) or len(target_shape) > 3:
            result = _reshape(a, target_shape)
            if target_dtype != a.dtype:
                return result.astype(target_dtype)
            return result.copy() if (copy and result is a) else result
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
    if not _is_array_nd(dst):
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
    if dtype is None:
        dtype = np.result_type(*[np.asarray(v) for v in (start, stop, step)])
    probe = np.arange(start, start, step, dtype=dtype)  # validates
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
    """numpy.linspace; scalar float64-compatible inputs are generated on the backend, others on the host."""
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
    """numpy.meshgrid for up to three 1-D inputs; device inputs must come from arange/linspace."""
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
