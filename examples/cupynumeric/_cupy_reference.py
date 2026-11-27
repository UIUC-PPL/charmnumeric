"""CNUM_REFERENCE=cupynumeric: reference run of the upstream program on cuPyNumeric.

The charmnumeric run's random draws and factors go to a temp dir as .npz/.json;
_cupy_ref_run.py replays them in the cupynumeric environment and returns the
locals and timing the same way. Environment:
  CNUM_CUPY_ENV    conda prefix with cupynumeric (default /work/hdd/mzu/atahilyani/cnum-deps/cupy/env)
  CNUM_CUPY_GPUS   GPUs for legate (default 1)
  CNUM_CUPY_ARGS   extra legate launcher flags
  CNUM_REF_JSON    append one result record per run_timed call to this JSON-lines file
  CNUM_REF_TMP     parent dir for the exchange files (default $TMPDIR)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy

import _benchmark as B

HERE = os.path.dirname(os.path.abspath(__file__))
_ENV = os.environ.get("CNUM_CUPY_ENV", "/work/hdd/mzu/atahilyani/cnum-deps/cupy/env")


def reported_ms(info, ret):
    """Time the program reports (its return value), as upstream's harness reads it."""
    if info.returns_time < 0:
        return None
    if isinstance(info.output_names, str):
        return float(ret)
    return float(ret[info.returns_time])


def _save_inputs(runs, work):
    arrays, meta = {}, []
    for i, run in enumerate(runs):
        sess, rec = run["session"], {"draws": [], "factors": []}
        for j, (name, val) in enumerate(sess.draws):
            host = B._host(val)
            key = f"d{i}_{j}"
            kind = "array" if isinstance(host, numpy.ndarray) else type(host).__name__
            arrays[key] = numpy.asarray(host)
            rec["draws"].append([name, kind, key])
        for j, (name, _, _, out) in enumerate(sess.factors):
            outs = list(out) if isinstance(out, (tuple, list)) else [out]
            keys = []
            for k, o in enumerate(outs):
                keys.append(f"f{i}_{j}_{k}")
                arrays[keys[-1]] = numpy.asarray(B._host(o))
            rec["factors"].append([name, keys])
        meta.append(rec)
    numpy.savez(os.path.join(work, "in.npz"), **arrays)
    with open(os.path.join(work, "in.json"), "w") as fh:
        json.dump(meta, fh)


def _launch(work, skip, argv):
    prog = os.path.join(HERE, "upstream", B._NAME + ".py")
    gpus = os.environ.get("CNUM_CUPY_GPUS", "1")
    cmd = [os.path.join(_ENV, "bin", "legate"), "--gpus", gpus,
           *os.environ.get("CNUM_CUPY_ARGS", "").split(),
           os.path.join(HERE, "_cupy_ref_run.py"), work, str(skip), prog, *argv, "--short"]
    env = {k: v for k, v in os.environ.items()
           if k not in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
                        "LD_PRELOAD", "CUDA_HOME", "CUDA_DIR", "CHARM_HOME")
           and not k.startswith(("CNUM_", "LEGATE", "OMP_", "KOKKOS"))}
    env["PATH"] = os.path.join(_ENV, "bin") + ":/usr/bin:/bin"
    print(f"[reference] cuPyNumeric: {' '.join(cmd)}", flush=True)
    log = os.path.join(work, "cupy.log")
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=work)
    text = open(log, errors="replace").read()
    for line in text.splitlines():
        print(f"[cupynumeric] {line}", flush=True)
    return rc


def compare(harness, info, f, runs, kwargs):
    work = tempfile.mkdtemp(prefix="cnum_ref_", dir=os.environ.get("CNUM_REF_TMP"))
    try:
        _compare(harness, info, f, runs, kwargs, work)
    finally:
        if not os.environ.get("CNUM_REF_KEEP"):
            shutil.rmtree(work, ignore_errors=True)


def _compare(harness, info, f, runs, kwargs, work):
    _save_inputs(runs, work)
    harness.ref_calls = getattr(harness, "ref_calls", 0)
    rc = _launch(work, harness.ref_calls, sys.argv[1:])
    harness.ref_calls += 1
    out_json = os.path.join(work, "out.json")
    if rc != 0 or not os.path.exists(out_json):
        raise RuntimeError(f"cuPyNumeric run failed (exit {rc}); see [cupynumeric] output above")
    with open(out_json) as fh:
        out = json.load(fh)
    data = numpy.load(os.path.join(work, "out.npz"))
    for run, res in zip(runs, out["runs"]):
        label = run["label"]
        stats = {"abs": 0.0, "rel": 0.0}
        failures = [f"cuPyNumeric: {m}" for m in res["failures"]]
        want = {}
        for qual, rec in res["wants"].items():
            want[qual] = {v: (data[e[1]] if e[0] == "a" else [data[k] for k in e[1]])
                          for v, e in rec.items()}
        # charmnumeric factors against the cuPyNumeric run's inputs, and cuPyNumeric's own
        for k, fac in enumerate(res["factors"]):
            a, rest = data[fac["a"]], fac["args"]
            kw = fac["kwargs"]
            charm = run["session"].factors[k][3]
            for who, result in (("charmnumeric", charm),
                                ("cuPyNumeric", [data[x] for x in fac["result"]]
                                 if fac["name"] != "argsort" else data[fac["result"][0]])):
                sess = B._Session()
                B._check_factors(sess, fac["name"], (a, *rest), kw, result)
                failures += [f"{who}: {m}" for m in sess.failures]
        for where in want:
            if where in run["got"]:
                B._compare_records(where, run["got"][where], want[where], failures, stats)
        n = sum(len(r) for r in want.values())
        charm_ms = None
        got_ret = run["got"].get(f.__qualname__, {}).get("<return>")
        if info.returns_time >= 0 and got_ret is not None:
            charm_ms = reported_ms(info, got_ret)
        elif info.returns_time < 0:
            charm_ms = run["wall_ms"]
        cupy_ms = res["time_ms"]
        print(f"[benchmark] compared {len(want)} functions ({n} locals) with cuPyNumeric{label}: "
              f"{'ok' if not failures else f'{len(failures)} differences'}", flush=True)
        for msg in failures:
            print(f"  {msg}", flush=True)
        ratio = (charm_ms / cupy_ms) if charm_ms and cupy_ms else None
        print(f"[reference] {B._NAME}{label}: max_abs={stats['abs']:.3e} max_rel={stats['rel']:.3e} "
              f"charmnumeric_ms={charm_ms} cupynumeric_ms={cupy_ms} ratio={ratio}", flush=True)
        path = os.environ.get("CNUM_REF_JSON")
        if path:
            with open(path, "a") as fh:
                fh.write(json.dumps(dict(program=B._NAME, label=label, match=not failures,
                                         max_abs=stats["abs"], max_rel=stats["rel"],
                                         charm_ms=charm_ms, cupy_ms=cupy_ms, ratio=ratio,
                                         failures=failures[:8])) + "\n")
        harness.failures.extend(m + label for m in failures)
