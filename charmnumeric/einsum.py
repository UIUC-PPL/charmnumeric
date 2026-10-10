"""einsum, tensordot, inner and outer, lowered to existing native ops.

Each operand is first cleaned up: broadcast size-1 dims are dropped, repeated
labels become diagonals, and labels found nowhere else are summed. Operands are
then contracted pairwise in ``numpy.einsum_path`` order. A pair becomes a
broadcast multiply when nothing is contracted, a side has no free labels, or
the product is small; otherwise it becomes SUMMA (one per batch element on 3-D
views). ``permute`` restores the output order.

Deviations from NumPy: results are never views, ``order`` is ignored, and
``optimize=False`` contracts left to right.
"""

import numbers
import string

import numpy as np

from charmtyles.core import _shape_size

from .charmnumeric import (
    _is_array_nd,
    _is_scalar_like,
    _normalize_dtype,
    _reshape,
    _transpose,
    _write_out,
    asarray,
    empty,
)

_LETTERS = string.ascii_letters

# Limits for using one broadcast product instead of a matmul per batch element.
_BCAST_IO_RATIO = 4
_BCAST_MAX = 1 << 24
_LOOP_MIN_BATCH = 64


def _prod(values):
    return int(_shape_size(tuple(values))) if len(values) else 1


# ---- parsing ----


def _sublist_term(sub):
    term = []
    for s in sub:
        if s is Ellipsis:
            term.append("...")
        elif isinstance(s, numbers.Integral) and 0 <= int(s) < len(_LETTERS):
            term.append(_LETTERS[int(s)])
        else:
            raise ValueError(f"subscript is not within the valid range [0, {len(_LETTERS)})")
    return "".join(term)


def _from_sublists(args):
    """einsum(op0, sublist0, op1, sublist1, ..., [sublistout]) -> subscripts, operands."""
    args = list(args)
    out = None
    if len(args) % 2 == 1:
        out = args.pop()
    operands = args[0::2]
    terms = [_sublist_term(sub) for sub in args[1::2]]
    subscripts = ",".join(terms)
    if out is not None:
        subscripts += "->" + _sublist_term(out)
    return subscripts, operands


def _parse(subscripts, shapes):
    """Expand ellipses: returns (input label lists, output label list)."""
    subscripts = subscripts.replace(" ", "")
    if "->" in subscripts:
        lhs, rhs = subscripts.split("->")
    else:
        lhs, rhs = subscripts, None
    terms = lhs.split(",")
    used = set(c for c in subscripts if c in _LETTERS)
    spare = [c for c in _LETTERS if c not in used]
    n_ell = 0
    for term, shape in zip(terms, shapes):
        if "..." in term:
            n_ell = max(n_ell, len(shape) - (len(term) - 3))
    ell = spare[:n_ell]
    inputs = []
    for term, shape in zip(terms, shapes):
        if "..." in term:
            pre, post = term.split("...")
            k = len(shape) - len(pre) - len(post)
            inputs.append(list(pre) + ell[n_ell - k:] + list(post))
        else:
            inputs.append(list(term))
    if rhs is None:
        counts = {}
        for c in lhs:
            if c in _LETTERS:
                counts[c] = counts.get(c, 0) + 1
        output = list(ell) + sorted(c for c, n in counts.items() if n == 1)
    elif "..." in rhs:
        pre, post = rhs.split("...")
        output = list(pre) + ell + list(post)
    elif n_ell:
        raise ValueError(
            "output has more dimensions than subscripts given in einstein sum, but no "
            "'...' ellipsis provided to broadcast the extra dimensions."
        )
    else:
        output = list(rhs)
    return inputs, output


# ---- axis permutation (any rank) ----


def permute(x, perm):
    """transpose(x, perm) for any rank (>3-D via grouped 3-D transposes)."""
    perm = tuple(int(p) for p in perm)
    n = len(perm)
    if perm == tuple(range(n)):
        return x
    shape = tuple(x.shape)
    final = tuple(shape[p] for p in perm)
    big = [p for p in perm if shape[p] != 1]
    if len(big) < n:
        # size-1 axes can go anywhere: drop them, permute the rest
        rank = {p: i for i, p in enumerate(sorted(big))}
        sq = tuple(shape[p] for p in sorted(big)) or (1,)
        y = _reshape(x, sq)
        if len(big) > 1:
            y = permute(y, [rank[p] for p in big])
        return _reshape(y, final)
    if n <= 3:
        return _transpose(x, perm)
    # coalesce source axes that stay adjacent
    runs = []
    for p in perm:
        if runs and runs[-1][-1] + 1 == p:
            runs[-1].append(p)
        else:
            runs.append([p])
    size = [_prod([shape[p] for p in run]) for run in runs]
    cur = sorted(range(len(runs)), key=lambda r: runs[r][0])
    y = _reshape(x, tuple(size[r] for r in cur))
    if len(runs) <= 3:
        y = _transpose(y, [cur.index(t) for t in range(len(runs))])
        return _reshape(y, final)
    for t in range(len(runs)):
        j = cur.index(t)
        if j == t:
            continue
        sizes = [size[r] for r in cur]
        pre, mid, post = _prod(sizes[:t]), _prod(sizes[t:j]), _prod(sizes[j:])
        if pre == 1:
            y = _transpose(_reshape(y, (mid, post)), (1, 0))
        else:
            y = _transpose(_reshape(y, (pre, mid, post)), (0, 2, 1))
        cur = cur[:t] + cur[j:] + cur[t:j]
    return _reshape(y, final)


# ---- labelled operands ----


def _arrange(x, labels, groups):
    """x with its axes ordered as the concatenated groups, one dim per non-empty group."""
    order = [l for g in groups for l in g]
    if len(order) != len(labels):  # size-1 labels left out of every group
        kept = [i for i, l in enumerate(labels) if l in order]
        x = _reshape(x, tuple(x.shape[i] for i in kept) or (1,))
        labels = [labels[i] for i in kept]
    x = permute(x, [labels.index(l) for l in order])
    shape = tuple(_prod([x.shape[order.index(l)] for l in g]) for g in groups if g)
    return _reshape(x, shape or (1,))


def _layout_cost(x, labels, groups):
    return 0 if labels == [l for g in groups for l in g] else int(x.size)


def _sum_out(x, labels, keep, dtype):
    axes = tuple(i for i, l in enumerate(labels) if l not in keep)
    if not axes:
        return x, labels
    from .reductions import sum as _sum

    return _sum(x, axis=axes, dtype=dtype), [l for l in labels if l in keep]


def _diagonals(x, labels):
    """Replace repeated labels by their diagonal (kept at the first occurrence)."""
    while len(set(labels)) < len(labels):
        l = next(m for m in labels if labels.count(m) > 1)
        if len(labels) == 2:
            from .operations import diag

            x, labels = diag(x), [l]
            continue
        axes = [i for i, m in enumerate(labels) if m == l]
        # merge repeated axes; the diagonal is every (n^r - 1) / (n - 1)-th element
        p, r, n = axes[0], len(axes), int(x.shape[axes[0]])
        order = list(range(p)) + axes + [i for i in range(p, len(labels)) if i not in axes]
        x = permute(x, order)
        labels = [labels[i] for i in order]
        shape = tuple(int(d) for d in x.shape)
        x = _reshape(x, shape[:p] + (n**r,) + shape[p + r:])
        step = (n**r - 1) // (n - 1) if n > 1 else 1
        x = x[(slice(None),) * p + (slice(None, None, step),)].copy()
        labels = labels[:p + 1] + labels[p + r:]
    return x, labels


def _drop_broadcast(x, labels, sizes):
    """Drop size-1 dims whose label is larger elsewhere (NumPy broadcasting)."""
    keep = [i for i, l in enumerate(labels) if not (x.shape[i] == 1 and sizes[l] != 1)]
    if len(keep) == len(labels):
        return x, labels
    shape = tuple(int(x.shape[i]) for i in keep) or (1,)
    return _reshape(x, shape), [labels[i] for i in keep]


def _shaped(x, labels, sizes):
    """Reshape a contraction result to one dim per label ((1,) for none)."""
    return _reshape(x, tuple(sizes[l] for l in labels) or (1,))


def _compact(*shapes):
    """Drop the positions where every shape has size 1 (keeps at least one dim)."""
    keep = [i for i in range(len(shapes[0])) if any(s[i] != 1 for s in shapes)] or [0]
    return [tuple(s[i] for i in keep) for s in shapes], keep


def _bcast_contract(a, shape_a, b, shape_b, sum_pos, dtype):
    """Broadcast multiply of the reshaped operands, then sum over sum_pos."""
    (sa, sb), keep = _compact(shape_a, shape_b)
    r = _reshape(a, sa) * _reshape(b, sb)
    if sum_pos is None or sum_pos not in keep:
        return r
    from .reductions import sum as _sum

    return _sum(r, axis=keep.index(sum_pos), dtype=dtype)


def _label_product(a, la, b, lb, keep, sizes, dtype, final=None):
    """Broadcast multiply in label space, then sum the labels not kept."""
    union = la + [l for l in lb if l not in la]
    cands = [union, lb + [l for l in la if l not in lb]]
    if final is not None:
        cands.append([l for l in final if l in union] + [l for l in union if l not in final])

    def cost(order):
        c = 0
        for x, lx in ((a, la), (b, lb)):
            if [l for l in order if l in lx] != lx:
                c += int(x.size)
        kept = [l for l in order if l in keep]
        if final is not None and kept != [l for l in final if l in kept]:
            c += _prod([sizes[l] for l in kept])
        return c

    labels = min(cands, key=cost)
    xs = []
    for x, lx in ((a, la), (b, lb)):
        x = permute(x, [lx.index(l) for l in labels if l in lx])
        xs.append(_reshape(x, tuple(sizes[l] if l in lx else 1 for l in labels)))
    return _sum_out(xs[0] * xs[1], labels, keep, dtype)


def _grouped_product(a, la, b, lb, batch, con, fa, fb, sizes, dtype):
    """(batch, fa, 1, con) * (batch, 1, fb, con) summed over con, for > 3 labels."""
    Bt, M, N, K = (_prod([sizes[l] for l in g]) for g in (batch, fa, fb, con))
    out = batch + fa + fb
    if K == 1:
        pa = _arrange(a, la, [batch, fa, con])
        pb = _arrange(b, lb, [batch, con, fb])
        return _shaped(_bcast_contract(pa, (Bt, M, 1), pb, (Bt, 1, N), None, dtype), out, sizes), out
    if M == 1 or N == 1:
        # vector side (batch, con); matrix side in whichever order needs no transpose
        if M != 1:
            a, la, b, lb, fb, N = b, lb, a, la, fa, M
        pa = _arrange(a, la, [batch, con])
        if N == 1:
            pb = _arrange(b, lb, [batch, con])
            r = _bcast_contract(pa, (Bt, K), pb, (Bt, K), 1, dtype)
        elif _layout_cost(b, lb, [batch, con, fb]) == 0:
            pb = _arrange(b, lb, [batch, con, fb])
            r = _bcast_contract(pa, (Bt, K, 1), pb, (Bt, K, N), 1, dtype)
        else:
            pb = _arrange(b, lb, [batch, fb, con])
            r = _bcast_contract(pa, (Bt, 1, K), pb, (Bt, N, K), 2, dtype)
        return _shaped(r, out, sizes), out
    from .reductions import sum as _sum

    pa = _reshape(_arrange(a, la, [batch, fa, con]), (Bt, M, 1, K))
    pb = _reshape(_arrange(b, lb, [batch, fb, con]), (Bt, 1, N, K))
    return _shaped(_sum(pa * pb, axis=3, dtype=dtype), out, sizes), out


def _contract_pair(a, la, b, lb, keep, sizes, dtype, final=None):
    """Contract two labelled operands into (array, labels).

    ``keep``: labels still needed afterwards; ``final``: output order if last.
    """
    a, la = _sum_out(a, la, set(lb) | keep, dtype)
    b, lb = _sum_out(b, lb, set(la) | keep, dtype)
    if not la or not lb:
        return a * b, (la or lb)
    sa, sb = set(la), set(lb)
    big = la if int(a.size) >= int(b.size) else lb
    batch = [l for l in big if l in sa and l in sb and l in keep]
    con = [l for l in big if l in sa and l in sb and l not in keep]
    fa = [l for l in la if l not in sb]
    fb = [l for l in lb if l not in sa]
    Bt = _prod([sizes[l] for l in batch])
    M = _prod([sizes[l] for l in fa])
    N = _prod([sizes[l] for l in fb])
    K = _prod([sizes[l] for l in con])

    inter = Bt * M * N * K
    io = Bt * (M * K + K * N + M * N)
    vector = M == 1 or N == 1
    small = Bt > 1 and (inter <= _BCAST_IO_RATIO * io or (Bt > _LOOP_MIN_BATCH and inter <= _BCAST_MAX))
    if K == 1 or vector or small:
        if K > 1 and Bt == 1 and vector:
            if M == 1 and N == 1 and len(la) == len(lb) == 1:
                return a @ b, []  # native dot
            vec, mat, lm, free = (a, b, lb, fb) if M == 1 else (b, a, la, fa)
            if len(vec.shape) == 1 and lm == free + con:
                out = free
                return _shaped(_arrange(mat, lm, [free, con]) @ vec, out, sizes), out
        if len(sa | sb) <= 3 or sa == sb:
            return _label_product(a, la, b, lb, keep, sizes, dtype, final)
        return _grouped_product(a, la, b, lb, batch, con, fa, fb, sizes, dtype)

    # matmul: pick the orientation and batch position needing fewest transposes
    def layout(x, lx, free, left):
        mat = [free, con] if left else [con, free]
        cands = [[batch] + mat, [mat[0], batch, mat[1]], mat + [batch]] if Bt > 1 else [mat]
        return min(((_layout_cost(x, lx, g), g) for g in cands), key=lambda c: c[0])

    plans = []
    for (x, lx, fx), (y, ly, fy) in (((a, la, fa), (b, lb, fb)), ((b, lb, fb), (a, la, fa))):
        cx, gx = layout(x, lx, fx, True)
        cy, gy = layout(y, ly, fy, False)
        out = batch + fx + fy
        cr = 0
        if final is not None and [l for l in final if l in out] != out:
            cr = Bt * M * N
        plans.append((cx + cy + cr, x, lx, gx, fx, y, ly, gy, fy, out))
    _, x, lx, gx, fx, y, ly, gy, fy, out = min(plans, key=lambda p: p[0])
    px = _arrange(x, lx, gx)
    py = _arrange(y, ly, gy)
    if Bt == 1:
        r = px @ py
    else:
        bx = next(i for i, g in enumerate(gx) if g is batch)
        by = next(i for i, g in enumerate(gy) if g is batch)
        Mx = _prod([sizes[l] for l in fx])
        Ny = _prod([sizes[l] for l in fy])
        r = empty((Bt, Mx, Ny), dtype=dtype)
        for i in range(Bt):
            r[i] = px[(slice(None),) * bx + (slice(i, i + 1),)] @ py[(slice(None),) * by + (slice(i, i + 1),)]
    return _shaped(r, out, sizes), out


# ---- driver ----


def _fold(ops, keep, sizes, dtype, final):
    """Contract labelled operands to one, smallest intermediate first."""
    ops = list(ops)
    while len(ops) > 1:
        best = None
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                rest = set(keep).union(*[o[1] for k, o in enumerate(ops) if k not in (i, j)])
                labels = (set(ops[i][1]) | set(ops[j][1])) & rest
                size = _prod([sizes[l] for l in labels])
                if best is None or size < best[0]:
                    best = (size, i, j, rest)
        _, i, j, rest = best
        (x, lx), (y, ly) = ops[i], ops[j]
        del ops[j], ops[i]
        ops.append(_contract_pair(x, lx, y, ly, rest, sizes, dtype, final if not ops else None))
    x, lx = ops[0]
    return _sum_out(x, lx, keep, dtype)



def _dummies(shapes, dtypes):
    return [np.broadcast_to(np.zeros((), dtype=d), s) for s, d in zip(shapes, dtypes)]


def _path(subscripts, shapes, dtypes, optimize):
    n = len(shapes)
    if n == 1:
        return [(0,)]
    if optimize is False:
        return [(0, 1)] + [(0, m - 1) for m in range(n - 1, 1, -1)]
    if isinstance(optimize, (list, tuple)) and len(optimize) and optimize[0] == "einsum_path":
        return [tuple(c) for c in optimize[1:]]
    if optimize is True:
        optimize = "greedy"
    path = np.einsum_path(subscripts, *_dummies(shapes, dtypes), optimize=optimize)[0]
    return [tuple(c) for c in path[1:]]


def einsum(*operands, out=None, dtype=None, order="K", casting="safe", optimize=True):
    """numpy.einsum (string or sublist form) lowered to native ops."""
    if not operands:
        raise ValueError("No input operands")
    if isinstance(operands[0], str):
        subscripts, ops = operands[0], list(operands[1:])
    else:
        subscripts, ops = _from_sublists(operands)
    scalars = [op for op in ops if _is_scalar_like(op)]
    arrays = [op if _is_array_nd(op) else asarray(op) for op in ops if not _is_scalar_like(op)]
    shapes = []
    dtypes = []
    for op in ops:
        if _is_scalar_like(op):
            shapes.append(())
            dtypes.append(np.asarray(op).dtype)
        else:
            shapes.append(tuple(int(s) for s in (op.shape if _is_array_nd(op) else np.shape(op))))
            dtypes.append(np.dtype(op.dtype) if hasattr(op, "dtype") else np.asarray(op).dtype)
    np.einsum_path(subscripts, *_dummies(shapes, dtypes), optimize=False)  # NumPy's checks

    res_dtype = np.result_type(*[a.dtype for a in arrays], *scalars) if arrays else None
    if dtype is not None:
        dtype = _normalize_dtype(dtype)
        for a in arrays:
            if not np.can_cast(a.dtype, dtype, casting=casting):
                raise TypeError(
                    f"Cannot cast einsum operand from {a.dtype} to {dtype} "
                    f"according to the rule '{casting}'"
                )
        res_dtype = dtype
    if not arrays:
        return _write_out(np.einsum(subscripts, *ops, dtype=dtype), out)
    res_dtype = _normalize_dtype(res_dtype)

    inputs, output = _parse(subscripts, shapes)
    sizes = {}
    for labels, shape in zip(inputs, shapes):
        for l, s in zip(labels, shape):
            sizes[l] = max(sizes.get(l, 1), int(s))

    path = _path(subscripts, shapes, dtypes, optimize)
    work = []
    factor = None
    it = iter(arrays)
    for op, labels in zip(ops, inputs):
        if _is_scalar_like(op):
            factor = op if factor is None else factor * op
            work.append(None)
            continue
        x = next(it)
        if x.dtype != res_dtype:
            x = x.astype(res_dtype)
        x, labels = _drop_broadcast(x, list(labels), sizes)
        x, labels = _diagonals(x, labels)
        work.append((x, labels))

    outset = set(output)
    for step in path:
        picked = [work[i] for i in step if work[i] is not None]
        work = [w for i, w in enumerate(work) if i not in step]
        others = [w for w in work if w is not None]
        keep = outset.union(*[w[1] for w in others])
        if picked:
            picked = _fold(picked, keep, sizes, res_dtype, None if others else output)
        work.append(picked or None)

    x, lx = _fold([w for w in work if w is not None], outset, sizes, res_dtype, output)
    x, lx = _sum_out(x, lx, outset, res_dtype)
    if lx:
        x = permute(x, [lx.index(l) for l in output])
    x = _reshape(x, tuple(sizes[l] for l in output))
    if factor is not None:
        x = x * np.asarray(factor).astype(res_dtype).item()
    if x.dtype != res_dtype:
        x = x.astype(res_dtype)
    if out is not None and out.dtype != x.dtype:
        if not np.can_cast(x.dtype, out.dtype, casting=casting):
            raise TypeError(f"Cannot cast einsum output from {x.dtype} to {out.dtype}")
        x = x.astype(out.dtype)
    return _write_out(x, out)


def einsum_path(*operands, optimize="greedy"):
    """numpy.einsum_path, computed from the operand shapes."""
    if isinstance(operands[0], str):
        subscripts, ops = operands[0], list(operands[1:])
    else:
        subscripts, ops = _from_sublists(operands)
    dummies = [
        np.asarray(op) if _is_scalar_like(op)
        else np.broadcast_to(np.zeros((), dtype=op.dtype), tuple(op.shape))
        for op in ops
    ]
    return np.einsum_path(subscripts, *dummies, optimize=optimize)


# ---- tensordot / inner / outer ----


def _axes_list(axes, ndim):
    if isinstance(axes, numbers.Integral):
        axes = [axes]
    return [int(ax) % ndim if -ndim <= int(ax) < ndim else _bad_axis(ax, ndim) for ax in axes]


def _bad_axis(ax, ndim):
    raise np.exceptions.AxisError(ax, ndim)


def tensordot(a, b, axes=2):
    """numpy.tensordot."""
    a = a if _is_array_nd(a) else asarray(a)
    b = b if _is_array_nd(b) else asarray(b)
    na, nb = len(a.shape), len(b.shape)
    if isinstance(axes, numbers.Integral):
        n = int(axes)
        if n < 0 or n > na or n > nb:
            raise ValueError("shape-mismatch for sum")
        ax_a, ax_b = list(range(na - n, na)), list(range(n))
    else:
        ax_a, ax_b = axes
        ax_a, ax_b = _axes_list(ax_a, na), _axes_list(ax_b, nb)
    if len(ax_a) != len(ax_b):
        raise ValueError("shape-mismatch for sum")
    for i, j in zip(ax_a, ax_b):
        if a.shape[i] != b.shape[j]:
            raise ValueError("shape-mismatch for sum")
    la = list(_LETTERS[:na])
    lb = list(_LETTERS[na:na + nb])
    for i, j in zip(ax_a, ax_b):
        lb[j] = la[i]
    out = [l for i, l in enumerate(la) if i not in ax_a] + [l for j, l in enumerate(lb) if j not in ax_b]
    return einsum("".join(la) + "," + "".join(lb) + "->" + "".join(out), a, b)


def inner(a, b):
    """numpy.inner: sum over the last axes."""
    if _is_scalar_like(a) or _is_scalar_like(b):
        return a * b
    a = a if _is_array_nd(a) else asarray(a)
    b = b if _is_array_nd(b) else asarray(b)
    na, nb = len(a.shape), len(b.shape)
    if not na or not nb:
        return a * b
    la = _LETTERS[:na]
    lb = _LETTERS[na:na + nb - 1] + la[-1]
    return einsum(la + "," + lb + "->" + la[:-1] + lb[:-1], a, b)


def outer(a, b, out=None):
    """numpy.outer: (a.size, b.size) products of the flattened inputs."""
    a = a if _is_array_nd(a) else asarray(a)
    b = b if _is_array_nd(b) else asarray(b)
    a = _reshape(a, (int(a.size),))
    b = _reshape(b, (int(b.size),))
    return einsum("i,j->ij", a, b, out=out)
