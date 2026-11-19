#!/usr/bin/env python
"""1-D arrays up to the 2**31-1 element limit at any process count: creation, elementwise
kernels, reductions, set_region, sort, quantile and get, each checked by value.

A layout computed with a wrapped int (n + tile - 1 over 2**31) used to give an array no
chares at all: every operation on it then "finished" at once without computing anything,
so nothing here trusts completion.

Env: CNUM_HOST, CNUM_PORT, CNUM_ODF. The default sizes are small; LIMIT1D_BIG=1 runs
1.2e9, 2**31 - 2**29 + 1, 2e9 and 2**31 - 1 elements (int8/bool arrays of n bytes and one
int32 index array; f64 holds up to four float64 arrays, about 32 n bytes over all
processes, 69 GB at 2**31 - 1). LIMIT1D_SIZES=n1,n2,... overrides the sizes; LIMIT1D_TESTS selects from
create,get,elem,reduce,setreg,sort,quantile,f64,fullget.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cupynumeric"))

import charmnumeric as cnp  # noqa: E402
import _cnum_connect  # noqa: E402
import _cnum_util as U  # noqa: E402

M = 7
BIG = os.environ.get("LIMIT1D_BIG", "0") not in ("", "0")
ALL = "create,get,elem,reduce,setreg,sort,quantile,f64,fullget"
TESTS = [t for t in os.environ.get("LIMIT1D_TESTS", ALL).split(",") if t]


def sizes():
    env = os.environ.get("LIMIT1D_SIZES", "")
    if env:
        return [int(float(s)) for s in env.split(",") if s]
    small = [300001, 2500000]
    return small + [1200000000, 2**31 - 2**29 + 1, 2000000000, 2**31 - 1] if BIG else small


def counts(n):
    """Number of i < n with i % M == k, for each k."""
    return np.array([(n - k + M - 1) // M for k in range(M)], dtype=np.int64)


def windows(n, extra=()):
    """Starts of 16-element windows: both ends and around likely tile boundaries."""
    w = {0, n - 16, n // 2 - 8, n // 4 - 8, 3 * (n // 4) - 8}
    for p in (2**18, 2**20, 2**29, 2**30, 2**31 - 2**29, *extra):
        if 8 <= p <= n - 8:
            w.add(p - 8)
    return sorted(s for s in w if 0 <= s <= n - 16)


def exact(chk, label, got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    ok = got.shape == expected.shape and got.dtype == expected.dtype and np.array_equal(got, expected)
    return chk.true(label, ok, "" if ok else f"got {got.dtype}{got.shape} {got.ravel()[:8]}, "
                    f"expected {expected.dtype}{expected.shape} {expected.ravel()[:8]}")


def scalar(chk, label, got, expected):
    got = np.asarray(got)
    ok = got.size == 1 and got.reshape(-1)[0] == expected
    return chk.true(label, ok, "" if ok else f"got {got.ravel()[:4]}, expected {expected}")


def run_size(chk, iface, n):
    print(f"-- n = {n} ({n / 2**31:.4f} * 2**31)")
    cnt = counts(n)
    total = int(np.dot(cnt, np.arange(M)))
    idx = cnp.arange(n, dtype=np.int32)
    v = (idx % M).astype(np.int8)

    def ref(s, e):
        return (np.arange(s, e, dtype=np.int64) % M).astype(np.int8)

    if "create" in TESTS:
        one = cnp.ones((n,), dtype=np.int8)
        scalar(chk, f"[{n}] sum(ones int8)", cnp.sum(one, dtype=np.int64).get(iface), n)
        del one
        z = cnp.zeros((n,), dtype=np.bool_)
        scalar(chk, f"[{n}] count_nonzero(zeros bool)", cnp.count_nonzero(z).get(iface), 0)
        scalar(chk, f"[{n}] count_nonzero(~zeros)", cnp.count_nonzero(~z).get(iface), n)
        del z
        f = cnp.full((n,), 3, dtype=np.int8)
        scalar(chk, f"[{n}] sum(full 3)", cnp.sum(f, dtype=np.int64).get(iface), 3 * n)
        del f
        exact(chk, f"[{n}] arange int32 tail", idx[n - 4:].get(iface),
              np.arange(n - 4, n, dtype=np.int32))
        scalar(chk, f"[{n}] max(arange)", cnp.max(idx).get(iface), n - 1)
        scalar(chk, f"[{n}] sum(arange)", cnp.sum(idx, dtype=np.int64).get(iface),
               n * (n - 1) // 2)
    if "get" in TESTS:
        for s in windows(n):
            exact(chk, f"[{n}] v[{s}:{s + 16}]", v[s:s + 16].get(iface), ref(s, s + 16))
        scalar(chk, f"[{n}] v[n - 1]", v[n - 1].get(iface), (n - 1) % M)
        step = max(1, n // 13)
        exact(chk, f"[{n}] v[5::{step}]", v[5::step].get(iface),
              (np.arange(5, n, step, dtype=np.int64) % M).astype(np.int8))
    if "reduce" in TESTS:
        scalar(chk, f"[{n}] sum(v)", cnp.sum(v, dtype=np.int64).get(iface), total)
        scalar(chk, f"[{n}] max(v)", cnp.max(v).get(iface), M - 1)
        scalar(chk, f"[{n}] min(v + 1)", cnp.min(v + 1).get(iface), 1)
        scalar(chk, f"[{n}] count_nonzero(v)", cnp.count_nonzero(v).get(iface), n - int(cnt[0]))
        h = n // 2 + 3
        scalar(chk, f"[{n}] sum(v[{h}:])", cnp.sum(v[h:], dtype=np.int64).get(iface),
               total - int(np.dot(counts(h), np.arange(M))))
    if "elem" in TESTS:
        w = v + v * 2 - 1
        chk.true(f"[{n}] (v + v * 2 - 1) dtype/shape", w.dtype == np.int8 and w.shape == (n,))
        scalar(chk, f"[{n}] sum(v + v * 2 - 1)", cnp.sum(w, dtype=np.int64).get(iface),
               3 * total - n)
        for s in windows(n)[::2]:
            exact(chk, f"[{n}] w[{s}:{s + 16}]", w[s:s + 16].get(iface), 3 * ref(s, s + 16) - 1)
        m = v > 3
        scalar(chk, f"[{n}] count_nonzero(v > 3)", cnp.count_nonzero(m).get(iface),
               int(cnt[4:].sum()))
        sh = v[1:] - v[:-1]
        scalar(chk, f"[{n}] sum(v[1:] - v[:-1])", cnp.sum(sh, dtype=np.int64).get(iface),
               (n - 1) % M)
        del w, m, sh
        # operands of one DAG that the solver may lay out shifted against each other
        a, b = cnp.ones((n,), dtype=np.int8), cnp.full((n,), 2, dtype=np.int8)
        c = a[1:] + b[:-1]
        d = a[:-2] * v[2:]
        e = cnp.zeros((n,), dtype=np.int8)
        e[1:-1] = v[:-2] + v[2:]
        scalar(chk, f"[{n}] sum(a[1:] + b[:-1])", cnp.sum(c, dtype=np.int64).get(iface),
               3 * (n - 1))
        scalar(chk, f"[{n}] sum(a[:-2] * v[2:])", cnp.sum(d, dtype=np.int64).get(iface), total - 1)
        scalar(chk, f"[{n}] sum of e[1:-1] = v[:-2] + v[2:]", cnp.sum(e, dtype=np.int64).get(iface),
               2 * total - 1 - (n - 2) % M - (n - 1) % M)
        exact(chk, f"[{n}] e[n - 6:]", e[n - 6:].get(iface),
              np.concatenate([ref(n - 7, n - 2) + ref(n - 5, n), [0]]).astype(np.int8))
        del a, b, c, d, e
    if "setreg" in TESTS:
        z = cnp.zeros((n,), dtype=np.int8)
        h = n // 2 + 1
        z[h:] = 1
        z[n - 5] = 9
        z[3] = 4
        z[7:n // 3] = v[n - n // 3 + 7:] + 1
        lo, hi = n - n // 3 + 7, n
        want = (n - h) + 8 + 4 + (total - int(np.dot(counts(lo), np.arange(M)))) + (hi - lo)
        scalar(chk, f"[{n}] sum after slice writes", cnp.sum(z, dtype=np.int64).get(iface), want)
        exact(chk, f"[{n}] z[n - 8:]", z[n - 8:].get(iface),
              np.array([1, 1, 1, 9, 1, 1, 1, 1], dtype=np.int8))
        exact(chk, f"[{n}] z[0:12]", z[0:12].get(iface),
              np.concatenate([[0, 0, 0, 4, 0, 0, 0], ref(lo, lo + 5) + 1]).astype(np.int8))
        exact(chk, f"[{n}] z[h - 4:h + 4]", z[h - 4:h + 4].get(iface),
              np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
        del z
    cum = np.concatenate([[0], np.cumsum(cnt)])

    def sorted_ref(s, e):
        return (np.searchsorted(cum, np.arange(s, e, dtype=np.int64), side="right") - 1).astype(np.int8)

    if "sort" in TESTS:
        s_ = cnp.sort(v)
        chk.true(f"[{n}] sort dtype/shape", s_.dtype == np.int8 and s_.shape == (n,))
        scalar(chk, f"[{n}] sum(sort(v))", cnp.sum(s_, dtype=np.int64).get(iface), total)
        for s in windows(n, extra=[int(c) for c in cum[1:-1]]):
            exact(chk, f"[{n}] sort(v)[{s}:{s + 16}]", s_[s:s + 16].get(iface),
                  sorted_ref(s, s + 16))
        scalar(chk, f"[{n}] descents in sort(v)", cnp.count_nonzero(s_[1:] < s_[:-1]).get(iface), 0)
        del s_
    if "quantile" in TESTS:
        q = np.array([0.0, 0.1, 0.25, 1.0 / 3, 0.5, 0.75, 0.9, 1.0])
        got = np.asarray(cnp.quantile(v, q).get(iface))
        pos = q * (n - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, n - 1)
        a = (np.searchsorted(cum, lo, side="right") - 1).astype(np.float64)
        b = (np.searchsorted(cum, hi, side="right") - 1).astype(np.float64)
        want = a + (b - a) * (pos - lo)
        ok = got.shape == want.shape and np.allclose(got, want, rtol=0, atol=1e-6)
        chk.true(f"[{n}] quantile(v, q)", ok, "" if ok else f"got {got}, expected {want}")
        got = np.asarray(cnp.quantile(v, 0.5).get(iface))
        ok = got.size == 1 and abs(float(got.reshape(-1)[0]) - want[4]) <= 1e-6
        chk.true(f"[{n}] quantile(v, 0.5)", ok, "" if ok else f"got {got}, expected {want[4]}")
    if "fullget" in TESTS:
        got = v.get(iface)
        ok = got.shape == (n,) and got.dtype == np.int8
        if ok:
            r = np.arange(n, dtype=np.int32)
            r %= M
            ok = bool(np.array_equal(got, r.astype(np.int8)))
            del r
        chk.true(f"[{n}] get of all of v", ok)
        del got
    del v, idx
    if "f64" in TESTS:
        x = cnp.arange(n, dtype=np.float64)
        t = cnp.sum(cnp.exp(x * (-1.0 / n))).get(iface)
        chk.close(f"[{n}] sum(exp(-i / n))", t, -np.expm1(-1.0) / -np.expm1(-1.0 / n), rtol=1e-9)
        y = cnp.sqrt(x) * 2.0 + cnp.exp(x * (-1.0 / n))
        del x

        def yref(i):
            i = np.asarray(i, dtype=np.float64)
            return np.sqrt(i) * 2.0 + np.exp(i * (-1.0 / n))

        for s in windows(n)[::2]:
            chk.close(f"[{n}] y[{s}:{s + 16}]", y[s:s + 16].get(iface), yref(np.arange(s, s + 16)),
                      rtol=1e-12)
        y *= -1.0
        r = cnp.sort(y)
        del y
        chk.close(f"[{n}] sort(-y)[:4]", r[:4].get(iface), -yref(np.arange(n - 1, n - 5, -1)),
                  rtol=1e-12)
        h = n // 2 + 5
        chk.close(f"[{n}] sort(-y)[{h}:{h + 4}]", r[h:h + 4].get(iface),
                  -yref(np.arange(n - 1 - h, n - 5 - h, -1)), rtol=1e-12)
        chk.close(f"[{n}] sort(-y)[n - 1]", r[n - 1].get(iface), -1.0, rtol=1e-12)
        del r


def main(chk):
    iface = _cnum_connect.get_interface()
    for n in sizes():
        run_size(chk, iface, n)
    for n in (2**31, 2**31 + 7):
        try:
            cnp.zeros((n,), dtype=np.int8)
            chk.true(f"zeros(({n},)) raises", False, "no exception")
        except (ValueError, OverflowError) as e:
            chk.true(f"zeros(({n},)) raises", True, f"({type(e).__name__})")
    chk.true("server alive", int(cnp.sum(cnp.arange(1000)).get(iface)) == 499500)


if __name__ == "__main__":
    U.run("test_1d_limit", main)
