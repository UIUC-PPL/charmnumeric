"""``numpy.linalg`` subset (plus cuPyNumeric's ``solve_triangular`` and ``tssvd``).

Factorizations and solves are LINALG nodes (opcode 100, ``src/executor_linalg.cpp``):
operands are gathered to one chare, factored there (Eigen on CPU, cuSOLVER/cuBLAS on
NVIDIA), and scattered back. The problem must fit on one chare.

Deviations from NumPy:

- 2-D operands only; float32/float64/complex64/complex128 results (integer and bool
  inputs are factored in float64; float16 is rejected); lstsq, slogdet and tssvd are
  real-only;
- failures do not raise: a non-positive-definite ``cholesky`` returns NaNs, a
  singular ``solve``/``inv`` returns inf/NaN;
- ``lstsq`` returns ``rank`` as a 0-d array, and residuals even when ``rank < N``.
"""

import builtins
from collections import namedtuple

import numpy as np
from numpy.linalg import LinAlgError

from .charmnumeric import ArrayOperation, asarray, nonfusible_op
from .reductions import norm, sum as _sum

__all__ = [
    "LinAlgError",
    "cholesky",
    "det",
    "eigh",
    "eigvalsh",
    "inv",
    "lstsq",
    "norm",
    "qr",
    "slogdet",
    "solve",
    "solve_triangular",
    "svd",
    "svdvals",
    "tssvd",
]

# Kinds. Must match enum LaOp in src/executor_linalg.cpp.
_LA_CHOLESKY = 0
_LA_QR = 1
_LA_SOLVE = 2
_LA_INV = 3
_LA_EIGH = 4
_LA_SVD = 5
_LA_SOLVE_TRIANGULAR = 6
_LA_DET = 7
_LA_SLOGDET = 8
_LA_LSTSQ = 9
_LA_RESHAPE = 10
_LA_TSSVD = 11

QRResult = namedtuple("QRResult", ["Q", "R"])
EighResult = namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
SVDResult = namedtuple("SVDResult", ["U", "S", "Vh"])
SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])


def _result_dtype(*arrays, real_only=False):
    dt = np.result_type(*[a.dtype for a in arrays])
    if dt == np.float16:
        raise TypeError("array type float16 is unsupported in linalg")
    if dt.kind == "c":
        if real_only:
            raise NotImplementedError("charmnumeric: this linalg function is real-only")
        return dt
    return np.dtype("float32") if dt == np.float32 else np.dtype("float64")


def _real_part(x):
    """Real part of a packed complex result (eigenvalues, singular values)."""
    return x.real if x.dtype.kind == "c" else x


def _la(kind, arrays, shape, dtype, params=()):
    shape = tuple(int(s) for s in shape)
    st = shape or (1,)  # 0-d results are stored as (1,)
    p = tuple(float(x) for x in params) + (0.0,) * (3 - len(params))
    return nonfusible_op(
        *arrays,
        kind,
        len(st),
        st[0],
        st[1] if len(st) > 1 else 0,
        *p,
        operation=ArrayOperation.linalg,
        shape=shape,
        dtype=dtype,
    )


def _matrix(a, square=False):
    a = asarray(a)
    if a.ndim < 2:
        raise LinAlgError(
            f"{a.ndim}-dimensional array given. Array must be at least two-dimensional")
    if a.ndim > 2:
        raise NotImplementedError("charmnumeric linalg does not support stacked matrices")
    if square and a.shape[0] != a.shape[1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    if a.size == 0:
        raise ValueError("zero-size arrays are not supported by charmnumeric linalg")
    return a


def _rhs(a, b):
    b = asarray(b)
    if b.ndim not in (1, 2):
        raise NotImplementedError("charmnumeric linalg does not support stacked right-hand sides")
    if b.shape[0] != a.shape[0]:
        sig = "(m,m),(m)->(m)" if b.ndim == 1 else "(m,m),(m,n)->(m,n)"
        raise ValueError(
            f"Input operand 1 has a mismatch in its core dimension 0, with gufunc signature "
            f"{sig} (size {b.shape[0]} is different from {a.shape[0]})")
    return b


def _vec(P, row, c0, c1):
    """Copy P[row, c0:c1] of a packed result into a 1-D array."""
    return _la(_LA_RESHAPE, [P[row:row + 1, c0:c1]], (c1 - c0,), P.dtype)


def _block(P, r0, r1, c0, c1):
    """Copy P[r0:r1, c0:c1] of a packed result into its own array."""
    if (r1 - r0) * (c1 - c0) == 1:
        return _la(_LA_RESHAPE, [P[r0:r1, c0:c1]], (1, 1), P.dtype)
    return P[r0:r1, c0:c1].copy()


def cholesky(a, /, *, upper=False):
    """Cholesky factor L (A = L L^T) or, with upper=True, U (A = U^T U)."""
    a = _matrix(a, square=True)
    n = a.shape[0]
    return _la(_LA_CHOLESKY, [a], (n, n), _result_dtype(a), (bool(upper),))


def qr(a, mode="reduced"):
    """QR factorization; mode 'reduced', 'complete' or 'r'."""
    a = _matrix(a)
    m, n = a.shape
    k = builtins.min(m, n)
    dt = _result_dtype(a)
    if mode == "r":
        return _la(_LA_QR, [a], (k, n), dt, (1,))
    if mode == "reduced":
        P = _la(_LA_QR, [a], (m + k, n), dt, (0,))
        return QRResult(_block(P, 0, m, 0, k), _block(P, m, m + k, 0, n))
    if mode == "complete":
        P = _la(_LA_QR, [a], (2 * m, builtins.max(m, n)), dt, (2,))
        return QRResult(_block(P, 0, m, 0, m), _block(P, m, 2 * m, 0, n))
    if mode in ("raw", "full", "f", "economic", "e"):
        raise NotImplementedError(f"qr mode {mode!r} is not supported")
    raise ValueError(f"Unrecognized mode '{mode}'")


def solve(a, b):
    """Solve a x = b (b of shape (M,) or (M, K)) by partial-pivoted LU."""
    a = _matrix(a, square=True)
    b = _rhs(a, b)
    return _la(_LA_SOLVE, [a, b], b.shape, _result_dtype(a, b))


def inv(a):
    a = _matrix(a, square=True)
    n = a.shape[0]
    return _la(_LA_INV, [a], (n, n), _result_dtype(a))


def det(a):
    """Determinant, as a 0-d array."""
    a = _matrix(a, square=True)
    return _la(_LA_DET, [a], (), _result_dtype(a))


def slogdet(a):
    """(sign, logabsdet), each a 0-d array."""
    a = _matrix(a, square=True)
    P = _la(_LA_SLOGDET, [a], (2,), _result_dtype(a, real_only=True))
    sign = _la(_LA_RESHAPE, [P[0:1]], (), P.dtype)
    logabsdet = _la(_LA_RESHAPE, [P[1:2]], (), P.dtype)
    return SlogdetResult(sign, logabsdet)


def _uplo(UPLO):
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    return 1 if UPLO == "U" else 0


def eigh(a, UPLO="L"):
    """Eigenvalues (ascending) and eigenvectors of a symmetric matrix."""
    a = _matrix(a, square=True)
    n = a.shape[0]
    P = _la(_LA_EIGH, [a], (n + 1, n), _result_dtype(a), (_uplo(UPLO), 1))
    return EighResult(_real_part(_vec(P, 0, 0, n)), _block(P, 1, n + 1, 0, n))


def eigvalsh(a, UPLO="L"):
    a = _matrix(a, square=True)
    n = a.shape[0]
    return _real_part(_la(_LA_EIGH, [a], (n,), _result_dtype(a), (_uplo(UPLO), 0)))


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """Singular value decomposition (s descending). ``hermitian`` is accepted and ignored."""
    a = _matrix(a)
    m, n = a.shape
    k = builtins.min(m, n)
    dt = _result_dtype(a)
    if not compute_uv:
        return _real_part(_la(_LA_SVD, [a], (k,), dt, (bool(full_matrices), 0)))
    uc = m if full_matrices else k
    vr = n if full_matrices else k
    P = _la(_LA_SVD, [a], (m + 1 + vr, builtins.max(uc, n)), dt, (bool(full_matrices), 1))
    return SVDResult(_block(P, 0, m, 0, uc), _real_part(_vec(P, m, 0, k)),
                     _block(P, m + 1, m + 1 + vr, 0, n))


def svdvals(x, /):
    return svd(x, compute_uv=False)


def lstsq(a, b, rcond=None):
    """Least-squares solution (x, residuals, rank, s), SVD-based like NumPy's gelsd."""
    a = _matrix(a)
    b = asarray(b)
    if b.ndim not in (1, 2) or b.shape[0] != a.shape[0]:
        raise LinAlgError("Incompatible dimensions")
    m, n = a.shape
    k = builtins.min(m, n)
    dt = _result_dtype(a, b, real_only=True)
    eps = np.finfo(dt).eps
    if rcond is None:
        rcond = eps * builtins.max(m, n)
    elif rcond < 0:
        rcond = eps
    nrhs = 1 if b.ndim == 1 else b.shape[1]
    P = _la(_LA_LSTSQ, [a, b], (n + 2, builtins.max(nrhs, k)), dt, (float(rcond),))
    if b.ndim == 1:
        x = _la(_LA_RESHAPE, [P[0:n, 0:1]], (n,), dt)
    else:
        x = _block(P, 0, n, 0, nrhs)
    s = _vec(P, n, 0, k)
    rank = _la(_LA_RESHAPE, [P[n + 1:n + 2, 0:1]], (), dt)
    if m <= n:
        residuals = np.empty((0,), dtype=dt)
    else:
        r = b - a @ x
        residuals = _sum(r * r, axis=0) if b.ndim == 2 else _sum(r * r).reshape((1,))
    return x, residuals, rank, s


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False,
                     check_finite=True):
    """Solve a x = b for triangular a (scipy/cuPyNumeric signature; trans in 0/1/2/'N'/'T'/'C')."""
    a = _matrix(a, square=True)
    b = _rhs(a, b)
    trans = {"N": 0, "T": 1, "C": 2}.get(trans, trans)
    if trans not in (0, 1, 2):
        raise ValueError("trans must be one of 0, 1, 2, 'N', 'T', 'C'")
    if trans == 2 and np.result_type(a.dtype, b.dtype).kind == "c":
        raise NotImplementedError("solve_triangular: trans='C' with complex input")
    return _la(_LA_SOLVE_TRIANGULAR, [a, b], b.shape, _result_dtype(a, b),
               (trans != 0, bool(lower), bool(unit_diagonal)))


def tssvd(a):
    """Tall-skinny SVD (cuPyNumeric): eigh of A^T A on one chare, then U = A @ (V S^-1)
    as a distributed matmul. Returns u (M, N), s (N,), vh (N, N)."""
    a = asarray(a)
    if a.ndim != 2 or a.size <= 1:
        raise ValueError(f"Invalid input shape for tssvd: {a.shape}")
    n = a.shape[1]
    dt = _result_dtype(a, real_only=True)
    P = _la(_LA_TSSVD, [a], (1 + 2 * n, n), dt)
    s = _vec(P, 0, 0, n)
    vh = _block(P, 1, n + 1, 0, n)
    u = a @ P[n + 1:2 * n + 1, :]
    return u, s, vh


def _reference(kind, values, shape, params):
    """NumPy model of a LINALG node (used by the CNUM_SIM simulator)."""
    try:
        return _reference_impl(kind, values, shape, params)
    except LinAlgError:
        return np.full(shape, np.nan)  # the backend returns NaNs instead of raising


def _reference_impl(kind, values, shape, params):
    a = values[0].astype(np.float64)
    b = values[1].astype(np.float64) if len(values) > 1 else None
    out = np.zeros(shape)
    p0, p1, p2 = params
    if kind == _LA_RESHAPE:
        return a.reshape(shape)
    m, n = a.shape
    k = builtins.min(m, n)
    if kind == _LA_CHOLESKY:
        return np.linalg.cholesky(a, upper=bool(p0))
    if kind == _LA_QR:
        if int(p0) == 1:
            return np.linalg.qr(a, mode="r")
        q, r = np.linalg.qr(a, mode="complete" if int(p0) == 2 else "reduced")
        out[:m, :q.shape[1]] = q
        out[m:, :n] = r
        return out
    if kind == _LA_SOLVE:
        return np.linalg.solve(a, b)
    if kind == _LA_INV:
        return np.linalg.inv(a)
    if kind == _LA_DET:
        return np.array([np.linalg.det(a)])
    if kind == _LA_SLOGDET:
        return np.array(np.linalg.slogdet(a))
    if kind == _LA_EIGH:
        uplo = "U" if p0 else "L"
        if not p1:
            return np.linalg.eigvalsh(a, uplo)
        w, v = np.linalg.eigh(a, uplo)
        out[0] = w
        out[1:] = v
        return out
    if kind == _LA_SVD:
        if not p1:
            return np.linalg.svd(a, compute_uv=False)
        u, s, vh = np.linalg.svd(a, full_matrices=bool(p0))
        out[:m, :u.shape[1]] = u
        out[m, :k] = s
        out[m + 1:, :n] = vh
        return out
    if kind == _LA_SOLVE_TRIANGULAR:
        t = a.T if p0 else a
        lower = bool(p1) != bool(p0)
        t = np.tril(t) if lower else np.triu(t)
        if p2:
            np.fill_diagonal(t, 1.0)
        return np.linalg.solve(t, b)
    if kind == _LA_LSTSQ:
        x, _, rank, s = np.linalg.lstsq(a, b, rcond=p0)
        out[:n, :(1 if x.ndim == 1 else x.shape[1])] = x.reshape(n, -1)
        out[n, :k] = s
        out[n + 1, 0] = rank
        return out
    if kind == _LA_TSSVD:
        w, v = np.linalg.eigh(a.T @ a)
        w, v = w[::-1], v[:, ::-1]
        s = np.sqrt(w)
        out[0] = s
        out[1:n + 1] = v.T
        out[n + 1:] = v / s
        return out
    raise ValueError(f"unknown linalg kind {kind}")
