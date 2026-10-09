#!/usr/bin/env python
"""Tests for the native array-creation ops: bulk upload, RNG, arange & co.

Covers
  * cnp.array / cnp.asarray / slice assignment from host data (UPLOAD):
    1-D/2-D/3-D, all four dtypes, non-tile-aligned shapes, MB-sized arrays,
    ordering against other writes of the same array, uploads into views;
  * cnp.random: rand, random, randn, uniform, normal, randint (int32 and the
    full int64 range), seed, default_rng, transposed. Values are checked
    bit for bit against charmnumeric._rng_reference (the host model of the
    backend generator) and statistically (mean/var/range). A digest of the
    draws is printed so runs at +p1/+p2 and different odf can be compared:
    it must not change;
  * arange (int/float/negative/fractional steps), linspace, eye/identity,
    indices, meshgrid: exact equality with NumPy.

Environment: CNUM_HOST, CNUM_PORT, CNUM_ODF (default 4); CNUM_SIM=1 runs the
host simulator; CREATION_TESTS=upload,arange,random,size1 selects sections. Run with e.g.
    harness/run_one.sh server.out 2 32901 examples/ops/test_creation.py
Prints PASS test_creation / FAIL test_creation: ... (exit status 0 / 1).
On the CPU build the minimum tiles are 262144 (1-D), 512 (2-D) and 64 (3-D),
so the larger shapes here span several chares.
"""

import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cupynumeric"))

import charmnumeric as cnp  # noqa: E402
from charmnumeric import _rng_reference as R  # noqa: E402
import _cnum_connect  # noqa: E402
import _cnum_util as U  # noqa: E402

DTYPES = (np.float32, np.float64, np.int32, np.int64)
SHAPES = [(7,), (1000,), (300007,), (37, 53), (600, 700), (3, 4, 5), (70, 65, 66)]
# Size-1 arrays go last: with a size-1 1-D array alive, a later 1-D array that
# needs a bigger chare grid hangs the backend (pre-existing, also with plain
# cnp.zeros((1,)) + 1; not related to the creation ops).
SIZE1_SHAPES = [(1,), (1, 1), (1, 1, 1)]


def host_data(shape, dtype, seed):
    g = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return (g.standard_normal(shape) * 1e3).astype(dtype)
    info = np.iinfo(dtype)
    return g.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)


def exact(chk, label, got, expected):
    got = np.asarray(got)
    expected = np.asarray(expected)
    ok = (got.dtype == expected.dtype and got.shape == expected.shape
          and np.array_equal(got, expected, equal_nan=got.dtype.kind == "f"))
    detail = ""
    if not ok:
        if got.shape == expected.shape:
            nbad = int(np.sum(got != expected))
            detail = f"({nbad} elements differ, dtype {got.dtype} vs {expected.dtype})"
        else:
            detail = f"(shape {got.shape} vs {expected.shape})"
    return chk.true(label, ok, detail)


def digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def test_upload(chk, iface):
    print("-- upload")
    for shape in SHAPES:
        for k, dt in enumerate(DTYPES):
            h = host_data(shape, dt, seed=hash((shape, k)) & 0xFFFF)
            exact(chk, f"array {shape} {np.dtype(dt).name}", cnp.array(h).get(iface), h)
    # constant data takes the fill path, mixed data the upload path
    exact(chk, "asarray constant", cnp.asarray(np.full((40, 30), 2.5)).get(iface),
          np.full((40, 30), 2.5))
    # non-contiguous / ndmin / list inputs
    h = host_data((50, 60), np.float64, 1)
    exact(chk, "array of a transposed host view", cnp.array(h.T).get(iface), h.T.copy())
    exact(chk, "array(list, ndmin=2)", cnp.array([1, 2, 3], ndmin=2).get(iface),
          np.array([1, 2, 3], ndmin=2))

    # MB-sized uploads: one DAG node each
    for shape, dt in (((2_000_000,), np.float64), ((1500, 1500), np.float64),
                      ((128, 128, 128), np.float32)):
        h = host_data(shape, dt, 5)
        t0 = time.perf_counter()
        a = cnp.array(h)
        got = a.get(iface)
        dt_s = time.perf_counter() - t0
        exact(chk, f"array {shape} {np.dtype(dt).name} ({h.nbytes / 2**20:.1f} MiB, "
              f"upload+get {dt_s:.2f}s)", got, h)

    # ordering: uploads interleaved with other writes of the same array
    h1 = host_data((300, 200), np.float64, 11)
    h2 = host_data((300, 200), np.float64, 12)
    ref = np.zeros((300, 200))
    a = cnp.zeros((300, 200))
    a[...] = h1
    ref[...] = h1
    b = a + 1.0
    refb = ref + 1.0
    a[...] = h2
    ref[...] = h2
    a[10:20, 5:150:3] = h1[:10, :49]
    ref[10:20, 5:150:3] = h1[:10, :49]
    c = a * 2.0 + cnp.array(h1)
    refc = ref * 2.0 + h1
    a[0:5] = 7.0
    ref[0:5] = 7.0
    exact(chk, "ordering: snapshot before reupload", b.get(iface), refb)
    exact(chk, "ordering: strided upload into a view", c.get(iface), refc)
    exact(chk, "ordering: final state", a.get(iface), ref)

    # 3-D upload into an interior view, and upload consumed by a view
    h3 = host_data((70, 65, 66), np.int64, 3)
    a3 = cnp.zeros((70, 65, 66), dtype=np.int64)
    r3 = np.zeros((70, 65, 66), dtype=np.int64)
    a3[3:68, 1:64:2, 60:] = h3[:65, :32, :6]
    r3[3:68, 1:64:2, 60:] = h3[:65, :32, :6]
    exact(chk, "3-D upload into a strided view", a3.get(iface), r3)
    up = cnp.array(h3)
    exact(chk, "view of an uploaded array", up[5:60, 2:63, 1:65].get(iface), h3[5:60, 2:63, 1:65])


def test_arange_family(chk, iface):
    print("-- arange / linspace / eye / indices / meshgrid")
    cases = [
        (10,), (3, 17), (0, 1, 0.1), (0.1, 3.0, 0.37), (5, -3, -0.7), (10, 0, -1),
        (1, 100, 3), (-7.5, 12.25, 0.25), (0, 400_003),
    ]  # zero-length results are left out: gets of 0-size arrays hang the backend
    for args in cases:
        for dt in (None, np.float32, np.float64, np.int32, np.int64):
            try:
                ref = np.arange(*args, dtype=dt)
            except Exception:  # noqa: BLE001
                continue
            exact(chk, f"arange{args} dtype={None if dt is None else np.dtype(dt).name}",
                  cnp.arange(*args, dtype=dt).get(iface), ref)
    exact(chk, "arange(0.5, 5, 1, dtype=int)", cnp.arange(0.5, 5, 1, dtype=np.int64).get(iface),
          np.arange(0.5, 5, 1, dtype=np.int64))
    exact(chk, "arange float32 long", cnp.arange(-3.3, 20000.0, 0.071, dtype=np.float32)
          .get(iface), np.arange(-3.3, 20000.0, 0.071, dtype=np.float32))

    for args, kw in [((0, 1, 11), {}), ((-1, 2.5, 13), {}), ((2, -5, 400_001), {}),
                     ((0, 10, 7), {"endpoint": False}), ((3, 3, 5), {}),
                     ((-1, 20, 13), {"dtype": np.int32}), ((0, 1, 9), {"dtype": np.float32}),
                     ((-7, 7, 100), {"dtype": np.int64, "endpoint": False}),
                     ((1e-310, 2e-310, 5), {})]:
        exact(chk, f"linspace{args} {kw}", cnp.linspace(*args, **kw).get(iface),
              np.linspace(*args, **kw))
    got, step = cnp.linspace(0, 3, 7, retstep=True)
    ref, rstep = np.linspace(0, 3, 7, retstep=True)
    exact(chk, "linspace retstep values", got.get(iface), ref)
    chk.true("linspace retstep step", step == rstep, f"{step} vs {rstep}")
    exact(chk, "linspace float32 bounds (host path)",
          cnp.linspace(np.float32(0.1), np.float32(0.7), 9).get(iface),
          np.linspace(np.float32(0.1), np.float32(0.7), 9))

    for args in [(5,), (4, 7), (7, 4, 2), (600, 700, -3), (3, 3, 5)]:
        exact(chk, f"eye{args}", cnp.eye(*args).get(iface), np.eye(*args))
    exact(chk, "eye int32", cnp.eye(9, dtype=np.int32).get(iface), np.eye(9, dtype=np.int32))
    exact(chk, "identity(6)", cnp.identity(6).get(iface), np.identity(6))

    exact(chk, "indices((7,))", cnp.indices((7,)).get(iface), np.indices((7,)))
    exact(chk, "indices((37, 53))", cnp.indices((37, 53)).get(iface), np.indices((37, 53)))
    exact(chk, "indices((65, 70), float)", cnp.indices((65, 70), dtype=np.float64).get(iface),
          np.indices((65, 70), dtype=np.float64))
    for got, ref in zip(cnp.indices((6, 9), sparse=True), np.indices((6, 9), sparse=True)):
        exact(chk, f"indices sparse {ref.shape}", got.get(iface), ref)

    x, y, z = np.arange(5), np.linspace(0, 1, 3), np.arange(-2.0, 2.0, 0.5)
    for indexing in ("xy", "ij"):
        for sparse in (False, True):
            refs = np.meshgrid(x, y, z, indexing=indexing, sparse=sparse)
            got = cnp.meshgrid(cnp.arange(5), cnp.linspace(0, 1, 3), cnp.arange(-2.0, 2.0, 0.5),
                               indexing=indexing, sparse=sparse)
            for k, (g, r) in enumerate(zip(got, refs)):
                exact(chk, f"meshgrid 3 inputs {indexing} sparse={sparse} [{k}]", g.get(iface), r)
    hx, hy = np.array([3.0, -1.0, 4.5]), [2.0, 7.0]
    for k, (g, r) in enumerate(zip(cnp.meshgrid(hx, hy), np.meshgrid(hx, hy))):
        exact(chk, f"meshgrid host inputs [{k}]", g.get(iface), r)


def test_random(chk, iface):
    print("-- random")
    rec = []

    def check_ref(label, arr, kind, seed, stream, dtype, a=0.0, b=0.0, strides=None):
        got = arr.get(iface)
        shape = got.shape
        ref = R.generate(kind, R.global_index(shape, strides), seed, stream, dtype, a, b)
        exact(chk, label + " == reference generator", got, ref.reshape(shape))
        rec.append(got)
        return got

    cnp.random.seed(1234)
    stream = 0
    for shape in [(9,), (300007,), (37, 53), (600, 700), (5, 6, 7), (70, 65, 66)]:
        for dt in (np.float64, np.float32):
            check_ref(f"rand{shape} {np.dtype(dt).name}", cnp.random.rand(*shape, dtype=dt),
                      R.KIND_RANDOM, 1234, stream, dt)
            stream += 1
    u = check_ref("uniform(-3, 5, (1000, 1000))", cnp.random.uniform(-3, 5, (1000, 1000)),
                  R.KIND_UNIFORM, 1234, stream, np.float64, -3.0, 8.0)
    stream += 1
    n = check_ref("normal(2, 3, (1000, 1000))", cnp.random.normal(2, 3, (1000, 1000)),
                  R.KIND_NORMAL, 1234, stream, np.float64, 2.0, 3.0)
    stream += 1
    z = check_ref("randn(100, 100, 100)", cnp.random.randn(100, 100, 100),
                  R.KIND_NORMAL, 1234, stream, np.float64, 0.0, 1.0)
    stream += 1
    z32 = check_ref("standard_normal float32", cnp.random.standard_normal((513, 700),
                    dtype=np.float32), R.KIND_NORMAL, 1234, stream, np.float32, 0.0, 1.0)
    stream += 1
    ri = check_ref("randint(-5, 17, 300007, int32)", cnp.random.randint(-5, 17, 300007,
                   dtype=np.int32), R.KIND_RANDINT, 1234, stream, np.int32, -5, 17)
    stream += 1
    big = check_ref("randint full int64 range", cnp.random.randint(
        np.iinfo(np.int64).min, np.iinfo(np.int64).max, (600, 700)),
        R.KIND_RANDINT, 1234, stream, np.int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max)
    stream += 1
    full = check_ref("randint [-2**63, 2**63)", cnp.random.randint(-2**63, 2**63, (70, 65, 66)),
                     R.KIND_RANDINT, 1234, stream, np.int64, -2**63, 2**63)
    stream += 1
    odd = check_ref("randint(0, 3*2**61) (rejection)", cnp.random.randint(0, 3 * 2**61, 100000),
                    R.KIND_RANDINT, 1234, stream, np.int64, 0, 3 * 2**61)
    stream += 1

    # statistics
    big_u = cnp.random.random((4_000_000,)).get(iface)
    stream += 1
    rec.append(big_u)
    chk.true("random: range [0, 1)", big_u.min() >= 0.0 and big_u.max() < 1.0,
             f"min {big_u.min()} max {big_u.max()}")
    chk.true("random: mean ~ 1/2", abs(big_u.mean() - 0.5) < 1e-3, f"{big_u.mean():.6f}")
    chk.true("random: var ~ 1/12", abs(big_u.var() - 1 / 12) < 1e-3, f"{big_u.var():.6f}")
    hist = np.bincount((big_u * 64).astype(np.int64), minlength=64)
    chi2 = float(np.sum((hist - big_u.size / 64) ** 2 / (big_u.size / 64)))
    chk.true("random: chi2 of 64 bins < 120", chi2 < 120, f"chi2={chi2:.1f}")
    lag = float(np.corrcoef(big_u[:-1], big_u[1:])[0, 1])
    chk.true("random: lag-1 correlation ~ 0", abs(lag) < 3e-3, f"{lag:.2e}")
    chk.true("uniform: range [-3, 5)", u.min() >= -3 and u.max() < 5, f"{u.min()} {u.max()}")
    chk.true("uniform: mean ~ 1, var ~ 64/12", abs(u.mean() - 1) < 0.01 and
             abs(u.var() - 64 / 12) < 0.02, f"{u.mean():.4f} {u.var():.4f}")
    chk.true("normal(2,3): mean, std", abs(n.mean() - 2) < 0.01 and abs(n.std() - 3) < 0.01,
             f"{n.mean():.4f} {n.std():.4f}")
    for label, s in (("randn", z), ("standard_normal float32", z32)):
        m, v = float(s.mean()), float(s.var())
        sk = float(np.mean(((s - m) / np.sqrt(v)) ** 3))
        ku = float(np.mean(((s - m) / np.sqrt(v)) ** 4))
        chk.true(f"{label}: mean ~ 0, var ~ 1, skew ~ 0, kurtosis ~ 3",
                 abs(m) < 0.01 and abs(v - 1) < 0.01 and abs(sk) < 0.02 and abs(ku - 3) < 0.05,
                 f"{m:.4f} {v:.4f} {sk:.4f} {ku:.4f}")
    frac = float(np.mean(np.abs(z) < 1.0))
    chk.true("randn: P(|z|<1) ~ 0.6827", abs(frac - 0.682689) < 0.003, f"{frac:.4f}")
    counts = np.bincount(ri - (-5), minlength=22)
    chk.true("randint int32: range and all values hit", ri.min() == -5 and ri.max() == 16
             and np.all(counts > 0.9 * ri.size / 22), f"{ri.min()} {ri.max()} {counts.min()}")
    chk.true("randint int64 full range: spread", big.min() < -9.0e18 and big.max() > 9.0e18 and
             abs(float(np.mean(big.astype(np.float64))) / 2**63) < 0.01,
             f"{big.min()} {big.max()}")
    chk.true("randint [-2**63, 2**63): spread", full.min() < -9.1e18 and full.max() > 9.1e18)
    third = float(np.mean(odd < 2**61))
    chk.true("randint(0, 3*2**61): unbiased thirds", abs(third - 1 / 3) < 0.01, f"{third:.4f}")

    # seeds and streams
    cnp.random.seed(99)
    a = cnp.random.rand(600, 700).get(iface)
    b = cnp.random.rand(600, 700).get(iface)
    cnp.random.seed(99)
    a2 = cnp.random.rand(600, 700).get(iface)
    exact(chk, "seed(99) repeats the first draw", a2, a)
    chk.true("successive draws differ", not np.array_equal(a, b))
    cnp.random.seed(100)
    c = cnp.random.rand(600, 700).get(iface)
    chk.true("different seeds differ", float(np.mean(a == c)) < 1e-4)
    cnp.random.seed(2**64 - 1)
    exact(chk, "seed 2**64-1", cnp.random.random((1000,)).get(iface),
          R.generate(R.KIND_RANDOM, np.arange(1000), 2**64 - 1, 0, np.float64))
    rec += [a, b, c]

    g1, g2 = cnp.random.default_rng(7), cnp.random.default_rng(7)
    x1 = g1.integers(0, 10, size=(64, 64, 64), dtype=np.int32).get(iface)
    x2 = g2.integers(0, 10, size=(64, 64, 64), dtype=np.int32).get(iface)
    exact(chk, "default_rng(7) twice: same integers", x1, x2)
    exact(chk, "Generator.integers endpoint=True", g1.integers(1, 6, 50000, endpoint=True)
          .get(iface), R.generate(R.KIND_RANDINT, np.arange(50000), 7, 1, np.int64, 1, 7))
    exact(chk, "Generator.random float32", g1.random((300, 301), dtype=np.float32).get(iface),
          R.generate(R.KIND_RANDOM, R.global_index((300, 301)), 7, 2, np.float32))
    s = cnp.random.random()
    chk.true("random() returns a Python float", isinstance(s, float) and 0 <= s < 1, repr(s))

    # transposed draws and views of draws
    cnp.random.seed(5)
    A = cnp.random.rand(600, 530)
    AT = cnp.random.transposed(A)
    hA = A.get(iface)
    exact(chk, "transposed(rand(600, 530)) == .T", AT.get(iface), hA.T.copy())
    B = cnp.random.randn(30, 70, 65)
    hB = B.get(iface)
    exact(chk, "transposed(randn 3-D, axes=(2,0,1))",
          cnp.random.transposed(B, axes=(2, 0, 1)).get(iface), np.transpose(hB, (2, 0, 1)).copy())
    exact(chk, "view of a random array", A[100:550:3, 7:500].get(iface), hA[100:550:3, 7:500])
    C = A * 2.0 + cnp.random.transposed(cnp.random.transposed(A))
    exact(chk, "random arrays in elementwise ops", C.get(iface), hA * 2.0 + hA)

    print(f"  random digest {digest(rec)} (must be identical for every PE count / odf)")


def test_size1(chk, iface):
    print("-- size-1 arrays")
    for shape in SIZE1_SHAPES:
        for k, dt in enumerate(DTYPES):
            h = host_data(shape, dt, seed=k + 100)
            exact(chk, f"array {shape} {np.dtype(dt).name}", cnp.array(h).get(iface), h)
    exact(chk, "arange(1)", cnp.arange(1).get(iface), np.arange(1))
    exact(chk, "linspace(2, 3, 1)", cnp.linspace(2, 3, 1).get(iface), np.linspace(2, 3, 1))
    cnp.random.seed(3)
    exact(chk, "rand(1, 1)", cnp.random.rand(1, 1).get(iface),
          R.generate(R.KIND_RANDOM, np.zeros((1, 1), dtype=np.int64), 3, 0, np.float64))


SECTIONS = {"upload": test_upload, "arange": test_arange_family, "random": test_random,
            "size1": test_size1}


def main(chk):
    iface = _cnum_connect.get_interface()
    # CREATION_TESTS=upload,random runs a subset (size1 always runs last)
    names = os.environ.get("CREATION_TESTS", "upload,arange,random,size1").split(",")
    for name in names:
        SECTIONS[name](chk, iface)


if __name__ == "__main__":
    U.run("test_creation", main)
