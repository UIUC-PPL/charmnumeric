"""Random arrays generated on the backend (``numpy.random`` subset).

Every draw is one RANDOM DAG node. Element ``g`` (global row-major index) of the
``stream``-th draw after ``seed(s)`` is a pure function of ``(s, stream, g)``
(Philox4x32-10, ``src/creation_kernels.hpp``), so results do not depend on the
decomposition or PE count, and CPU and GPU builds match.

The values are NOT NumPy's MT19937 / PCG64 streams: ``seed(1)`` then ``rand(4)``
does not reproduce ``np.random``. Use ``.get()`` to fetch inputs for a NumPy
reference; ``charmnumeric._rng_reference`` reproduces the stream on the host.

Without ``seed`` the generator starts from seed 0; ``seed(None)`` draws from the OS.

Extensions: float functions take ``dtype=`` (float32 or float64), and
``transposed(x)`` regenerates the transpose of a random array.
"""

import operator
import os

import numpy as np

from . import _rng_reference as _ref
from .charmnumeric import (
    Array,
    ArrayOperation,
    ArrayView,
    _creation_op,
    _normalize_dtype,
    _normalize_shape,
    _u64_from_double,
    _u64_from_int,
    create_array,
)

__all__ = [
    "Generator",
    "default_rng",
    "normal",
    "rand",
    "randint",
    "randn",
    "random",
    "random_sample",
    "ranf",
    "sample",
    "seed",
    "standard_normal",
    "transposed",
    "uniform",
]

_DEFAULT_SEED = 0
_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _seed_value(seed):
    if seed is None:
        return int.from_bytes(os.urandom(8), "little")
    seed = operator.index(seed)
    if seed < 0 or seed >= (1 << 64):
        raise ValueError("Seed must be between 0 and 2**64 - 1")
    return int(seed)


def _size_shape(size):
    if size is None:
        return None
    return _normalize_shape(size)


class _Recipe(object):
    """How a random array was drawn; lets transposed() regenerate it."""

    __slots__ = ("kind", "seed", "stream", "a", "b", "shape", "strides", "dtype")

    def __init__(self, kind, seed, stream, a, b, shape, strides, dtype):
        self.kind = kind
        self.seed = seed
        self.stream = stream
        self.a = a
        self.b = b
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.dtype = dtype


def _emit(recipe):
    shape = recipe.shape
    if int(np.prod(shape)) == 0:
        return create_array(shape, dtype=recipe.dtype)
    strides = list(recipe.strides) + [0] * (3 - len(recipe.strides))
    words = [recipe.kind, recipe.seed, recipe.stream, recipe.a, recipe.b] + [
        _u64_from_int(s) for s in strides
    ]
    result = _creation_op(ArrayOperation.random, shape, recipe.dtype, words)
    result._rng_recipe = recipe
    return result


class Generator(object):
    """A seeded stream of draws (subset of ``numpy.random.Generator``).

    Draw ``k`` uses Philox key ``s`` and stream counter ``k``.
    """

    def __init__(self, seed=_DEFAULT_SEED):
        self._seed = _seed_value(seed)
        self._stream = 0

    def _draw(self, kind, size, dtype, a=0, b=0):
        shape = _size_shape(size)
        stream = self._stream
        self._stream += 1
        if shape is None:
            # Python scalar result: draw it on the host from the same stream.
            value = _ref.generate(kind, np.zeros(1, dtype=np.int64), self._seed, stream, dtype,
                                  _scalar_param(a, kind, 0), _scalar_param(b, kind, 1))
            return value[0].item()
        recipe = _Recipe(kind, self._seed, stream, a, b, shape,
                         _ref.row_major_strides(shape), dtype)
        return _emit(recipe)

    # ---- floats ----
    def random(self, size=None, dtype=np.float64):
        dtype = _float_dtype(dtype)
        return self._draw(_ref.KIND_RANDOM, size, dtype)

    def uniform(self, low=0.0, high=1.0, size=None, dtype=np.float64):
        dtype = _float_dtype(dtype)
        low, high = _float_scalar(low, "low"), _float_scalar(high, "high")
        return self._draw(_ref.KIND_UNIFORM, size, dtype,
                          _u64_from_double(low), _u64_from_double(high - low))

    def standard_normal(self, size=None, dtype=np.float64):
        return self.normal(0.0, 1.0, size=size, dtype=dtype)

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=np.float64):
        dtype = _float_dtype(dtype)
        loc, scale = _float_scalar(loc, "loc"), _float_scalar(scale, "scale")
        if scale < 0:
            raise ValueError("scale < 0")
        return self._draw(_ref.KIND_NORMAL, size, dtype,
                          _u64_from_double(loc), _u64_from_double(scale))

    # ---- integers ----
    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        if high is None:
            low, high = 0, low
        low, high = _int_scalar(low, "low"), _int_scalar(high, "high")
        if endpoint:
            high += 1
        return self._randint(low, high, size, dtype)

    def _randint(self, low, high, size, dtype):
        dtype = _normalize_dtype(dtype)
        if dtype.kind not in "biu":
            raise TypeError(f"randint: unsupported dtype {dtype}")
        if low >= high:
            raise ValueError("low >= high")
        lo, hi = (0, 1) if dtype.kind == "b" else (np.iinfo(dtype).min, np.iinfo(dtype).max)
        if low < lo:
            raise ValueError(f"low is out of bounds for {dtype}")
        if high > hi + 1:
            raise ValueError(f"high is out of bounds for {dtype}")
        return self._draw(_ref.KIND_RANDINT, size, dtype, _u64_from_int(low), _u64_from_int(high))


def _scalar_param(word, kind, which):
    """Decode a parameter word for the host-side (size=None) draw."""
    if kind == _ref.KIND_RANDINT:
        word = int(word)
        return word - (1 << 64) if word >= (1 << 63) else word
    if kind == _ref.KIND_RANDOM:
        return 0.0
    return float(np.array([word], dtype=np.uint64).view(np.float64)[0])


def _float_dtype(dtype):
    dtype = _normalize_dtype(dtype)
    if dtype not in _FLOAT_DTYPES:
        raise TypeError(f"Unsupported dtype {dtype} for floating-point draws")
    return dtype


def _check_scalar(value, name):
    if isinstance(value, (Array, ArrayView)) or np.ndim(value) != 0:
        raise NotImplementedError(f"array-valued {name} is not supported")


def _float_scalar(value, name):
    _check_scalar(value, name)
    return float(value)


def _int_scalar(value, name):
    _check_scalar(value, name)
    return operator.index(value)


def default_rng(seed=None):
    """A new Generator. Unlike NumPy, ``seed=None`` means the fixed default seed."""
    return Generator(_DEFAULT_SEED if seed is None else seed)


# ---- legacy module-level API (numpy.random.*) on one global Generator ----

_global = Generator(_DEFAULT_SEED)


def seed(seed=None):
    """Reseed the global generator (and restart its stream counter)."""
    global _global
    _global = Generator(_seed_value(seed))


def random(size=None, dtype=np.float64):
    return _global.random(size, dtype=dtype)


random_sample = random
ranf = random
sample = random


def rand(*shape, dtype=np.float64):
    return _global.random(shape if shape else None, dtype=dtype)


def randn(*shape, dtype=np.float64):
    return _global.standard_normal(shape if shape else None, dtype=dtype)


def standard_normal(size=None, dtype=np.float64):
    return _global.standard_normal(size, dtype=dtype)


def uniform(low=0.0, high=1.0, size=None, dtype=np.float64):
    return _global.uniform(low, high, size, dtype=dtype)


def normal(loc=0.0, scale=1.0, size=None, dtype=np.float64):
    return _global.normal(loc, scale, size, dtype=dtype)


def randint(low, high=None, size=None, dtype=np.int64):
    if high is None:
        low, high = 0, low
    return _global._randint(_int_scalar(low, "low"), _int_scalar(high, "high"), size, dtype)


def transposed(x, axes=None):
    """``x.transpose(axes)`` for an array produced directly by this module.

    Regenerates the draw with permuted element indices (no transpose op yet),
    so the result equals the transpose of ``x`` bit for bit.
    """
    recipe = getattr(x, "_rng_recipe", None)
    if recipe is None:
        raise ValueError("transposed() needs an array returned by charmnumeric.random")
    nd = len(recipe.shape)
    axes = tuple(range(nd))[::-1] if axes is None else tuple(int(a) % nd for a in axes)
    if sorted(axes) != list(range(nd)):
        raise ValueError(f"invalid axes {axes} for a {nd}-D array")
    shape = tuple(recipe.shape[a] for a in axes)
    strides = tuple(recipe.strides[a] for a in axes)
    return _emit(_Recipe(recipe.kind, recipe.seed, recipe.stream, recipe.a, recipe.b, shape,
                         strides, recipe.dtype))
