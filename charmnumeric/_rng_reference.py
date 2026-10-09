"""NumPy reference for charmnumeric's counter-based generators.

This mirrors ``src/creation_kernels.hpp`` operation for operation, so it
produces bit-identical values. It is used by the host simulator
(``examples/cupynumeric/_cnum_sim.py``) and by the tests to check backend
results exactly.

Generator: Philox4x32-10 (Salmon et al., SC'11). For element ``g`` (its global
row-major linear index, or the permuted index for ``random.transposed``) of
the ``stream``-th draw after ``seed(seed)``, the counter is
``(g_lo, g_hi, stream, attempt)`` and the key is ``(seed_lo, seed_hi)``.
Every element is a pure function of (seed, stream, g), so the values do not
depend on the decomposition, the over-decomposition factor or the PE count.
The streams differ from NumPy's MT19937.

Floating point follows a fixed sequence of IEEE operations (no FMA), with
software ``log`` and ``cos(2*pi*u)``, so CPU and GPU builds agree bitwise.
"""

import numpy as np

_U32 = np.uint64(0xFFFFFFFF)
_M0 = np.uint64(0xD2511F53)
_M1 = np.uint64(0xCD9E8D57)
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85

KIND_RANDOM = 0   # uniform [0, 1)
KIND_UNIFORM = 1  # low + (high - low) * u
KIND_NORMAL = 2   # loc + scale * z  (Box-Muller)
KIND_RANDINT = 3  # integers in [low, high)


def philox4x32(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 on uint64 arrays holding 32-bit words."""
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint64) & _U32 for c in (c0, c1, c2, c3))
    k0 = int(k0) & 0xFFFFFFFF
    k1 = int(k1) & 0xFFFFFFFF
    for r in range(10):
        if r > 0:
            k0 = (k0 + _W0) & 0xFFFFFFFF
            k1 = (k1 + _W1) & 0xFFFFFFFF
        p0 = _M0 * c0
        p1 = _M1 * c2
        hi0, lo0 = p0 >> np.uint64(32), p0 & _U32
        hi1, lo1 = p1 >> np.uint64(32), p1 & _U32
        c0, c1, c2, c3 = (
            hi1 ^ c1 ^ np.uint64(k0),
            lo1,
            hi0 ^ c3 ^ np.uint64(k1),
            lo0,
        )
    return c0, c1, c2, c3


def _block(g, seed, stream, attempt):
    g = np.asarray(g, dtype=np.uint64)
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    return philox4x32(
        g & _U32,
        g >> np.uint64(32),
        np.full(g.shape, int(stream) & 0xFFFFFFFF, dtype=np.uint64),
        np.full(g.shape, int(attempt) & 0xFFFFFFFF, dtype=np.uint64),
        seed & 0xFFFFFFFF,
        seed >> 32,
    )


def _u01(word64):
    return (word64 >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


# ---- software log / cos(2*pi*u), same operation sequence as the C++ ----

_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_SQRT2 = 1.4142135623730951
_PIO2 = 1.5707963267948966


def ct_log(x):
    x = np.asarray(x, dtype=np.float64)
    bits = x.view(np.uint64)
    e = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64) - 1023
    m = ((bits & np.uint64(0x000FFFFFFFFFFFFF)) | np.uint64(0x3FF0000000000000)).view(np.float64)
    big = m > _SQRT2
    m = np.where(big, m * 0.5, m)
    e = np.where(big, e + 1, e).astype(np.float64)
    f = m - 1.0
    s = f / (2.0 + f)
    s2 = s * s
    p = np.full(s.shape, 1.0 / 23.0)
    for k in range(10, -1, -1):
        p = p * s2 + 1.0 / (2 * k + 1)
    return e * _LN2_HI + (e * _LN2_LO + (2.0 * s) * p)


_COS_C = [1.0, -1.0 / 2.0, 1.0 / 24.0, -1.0 / 720.0, 1.0 / 40320.0, -1.0 / 3628800.0,
          1.0 / 479001600.0, -1.0 / 87178291200.0, 1.0 / 20922789888000.0,
          -1.0 / 6402373705728000.0]
_SIN_C = [1.0, -1.0 / 6.0, 1.0 / 120.0, -1.0 / 5040.0, 1.0 / 362880.0, -1.0 / 39916800.0,
          1.0 / 6227020800.0, -1.0 / 1307674368000.0, 1.0 / 355687428096000.0,
          -1.0 / 121645100408832000.0]


def ct_cos2pi(u):
    u = np.asarray(u, dtype=np.float64)
    t = u * 4.0
    q = t.astype(np.int64)
    f = t - q.astype(np.float64)
    flip = f > 0.5
    g = np.where(flip, 1.0 - f, f)
    r = g * _PIO2
    r2 = r * r
    c = np.full(r.shape, _COS_C[9])
    s = np.full(r.shape, _SIN_C[9])
    for k in range(8, -1, -1):
        c = c * r2 + _COS_C[k]
        s = s * r2 + _SIN_C[k]
    s = s * r
    cos_t = np.where(flip, s, c)
    sin_t = np.where(flip, c, s)
    return np.select([q == 0, q == 1, q == 2], [cos_t, -sin_t, -cos_t], sin_t)


def _mul64(x, r):
    """Full 64x64 -> 128-bit product of uint64 arrays, as (hi, lo)."""
    xl, xh = x & _U32, x >> np.uint64(32)
    rl, rh = r & _U32, r >> np.uint64(32)
    ll, lh, hl, hh = xl * rl, xl * rh, xh * rl, xh * rh
    mid = (ll >> np.uint64(32)) + (lh & _U32) + (hl & _U32)
    hi = hh + (lh >> np.uint64(32)) + (hl >> np.uint64(32)) + (mid >> np.uint64(32))
    lo = (mid << np.uint64(32)) | (ll & _U32)
    return hi, lo


def generate(kind, g, seed, stream, dtype, a=0.0, b=1.0):
    """Values of one draw at global indices g (any shape of int64/uint64).

    kind/a/b follow the backend: RANDOM (a, b unused), UNIFORM (a=low,
    b=high-low), NORMAL (a=loc, b=scale), RANDINT (a=low, b=high, ints).
    """
    dtype = np.dtype(dtype)
    g = np.asarray(g, dtype=np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        if kind == KIND_RANDINT:
            low = int(a)
            rng = np.uint64((int(b) - low) & 0xFFFFFFFFFFFFFFFF)
            x0, x1, x2, x3 = _block(g, seed, stream, 0)
            words = [(x1 << np.uint64(32)) | x0, (x3 << np.uint64(32)) | x2]
            if int(rng) == 0:  # the full 2**64 range: every word is a valid draw
                return (words[0] + np.uint64(low & 0xFFFFFFFFFFFFFFFF)).view(np.int64).astype(dtype)
            thresh = np.uint64(((1 << 64) - int(rng)) % int(rng))
            hi, lo = _mul64(words[0], np.full(g.shape, rng, dtype=np.uint64))
            draw = 1
            todo = lo < thresh
            while np.any(todo):
                if draw % 2 == 0:
                    x0, x1, x2, x3 = _block(g, seed, stream, draw // 2)
                    words = [(x1 << np.uint64(32)) | x0, (x3 << np.uint64(32)) | x2]
                h2, l2 = _mul64(words[draw % 2], np.full(g.shape, rng, dtype=np.uint64))
                hi = np.where(todo, h2, hi)
                lo = np.where(todo, l2, lo)
                todo = todo & (lo < thresh)
                draw += 1
            val = (hi + np.uint64(low & 0xFFFFFFFFFFFFFFFF)).view(np.int64)
            return val.astype(dtype)

        x0, x1, x2, x3 = _block(g, seed, stream, 0)
        wa = (x1 << np.uint64(32)) | x0
        if kind == KIND_RANDOM:
            if dtype == np.float32:
                return ((x0 >> np.uint64(8)).astype(np.float32) * np.float32(1.0 / 16777216.0))
            return _u01(wa).astype(dtype)
        if kind == KIND_UNIFORM:
            v = float(a) + float(b) * _u01(wa)
            return v.astype(dtype)
        if kind == KIND_NORMAL:
            wb = (x3 << np.uint64(32)) | x2
            u1 = 1.0 - _u01(wa)
            u2 = _u01(wb)
            z = np.sqrt(-2.0 * ct_log(u1)) * ct_cos2pi(u2)
            v = float(a) + float(b) * z
            return v.astype(dtype)
    raise ValueError(f"unknown generator kind {kind}")


def global_index(shape, strides=None):
    """Global index g of every element of `shape` (row-major unless strides)."""
    shape = tuple(int(s) for s in shape)
    if strides is None:
        strides = row_major_strides(shape)
    g = np.zeros(shape, dtype=np.int64)
    for d, n in enumerate(shape):
        ix = np.arange(n, dtype=np.int64).reshape((-1,) + (1,) * (len(shape) - d - 1))
        g = g + ix * int(strides[d])
    return g


def row_major_strides(shape):
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * int(shape[d + 1])
    return strides


# ---- whole creation nodes (used by the host simulator) ----

def _words_from_scalars(scalars):
    vals = [int(v) for v in scalars]
    return [(vals[2 * i] << 32) | vals[2 * i + 1] for i in range(len(vals) // 2)]


def _as_int64(word):
    return word - (1 << 64) if word >= (1 << 63) else word


def _as_double(word):
    return float(np.array([word], dtype=np.uint64).view(np.float64)[0])


def iota_values(words, shape, dtype):
    """Values of an IOTA node (mirrors ct_create::iota_value)."""
    dtype = np.dtype(dtype)
    mode, axis, flags, num, div, v0, v1, v2, v3 = words
    num, div = _as_int64(num), _as_int64(div)
    nd = len(shape)
    if mode == 2:  # INDICES
        return np.indices(shape[1:], dtype=dtype)
    if mode == 3:  # EYE
        return np.eye(shape[0], shape[1], k=_as_int64(v0), dtype=dtype)
    i = np.arange(shape[axis], dtype=np.int64)
    if mode == 0:  # ARANGE in the output dtype
        if dtype.kind == "f":
            b0, b1 = dtype.type(_as_double(v0)), dtype.type(_as_double(v1))
            line = b0 + i.astype(dtype) * dtype.type(b1 - b0)
        else:
            b0, b1 = _as_int64(v0), _as_int64(v1)
            with np.errstate(over="ignore"):
                line = (np.int64(b0) + i * np.int64(b1 - b0)).astype(dtype)
        if line.size > 0:
            line[0] = b0
        if line.size > 1:
            line[1] = b1
    elif mode == 1:  # LINSPACE in float64
        start, stop, step, delta = (_as_double(w) for w in (v0, v1, v2, v3))
        y = i.astype(np.float64)
        if div > 0:
            y = (y / float(div)) * delta if (flags & 2) else y * step
        else:
            y = y * delta
        y = y + start
        if (flags & 1) and num > 1:
            y[num - 1] = stop
        line = np.floor(y).astype(dtype) if dtype.kind != "f" else y.astype(dtype)
    else:
        raise ValueError(f"unknown iota mode {mode}")
    view = [1] * nd
    view[axis] = shape[axis]
    return np.broadcast_to(line.reshape(view), shape).copy()


def creation_values(operation, scalars, payload, shape, dtype):
    """Host evaluation of an UPLOAD (50) / RANDOM (51) / IOTA (52) node."""
    shape = tuple(shape)
    dtype = np.dtype(dtype)
    words = _words_from_scalars(scalars)
    if operation == 50:
        return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    if operation == 51:
        kind, seed, stream, a, b = words[:5]
        strides = [_as_int64(w) for w in words[5:5 + len(shape)]]
        if kind == KIND_RANDINT:
            a, b = _as_int64(a), _as_int64(b)
        else:
            a, b = _as_double(a), _as_double(b)
        return generate(kind, global_index(shape, strides), seed, stream, dtype, a, b)
    if operation == 52:
        return iota_values(words, shape, dtype)
    raise ValueError(f"not a creation opcode: {operation}")
