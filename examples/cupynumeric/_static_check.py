#!/usr/bin/env python
"""Run every port under the host-side simulator (CNUM_SIM=1) and tabulate results.

Usage:  python _static_check.py [--quick] [name ...]

Covers the Python side only (graph, shapes, fusion, DAG serialization, NumPy
checks) and flags unsupported patterns; the C++ backend is not exercised.
"""

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    ("black_scholes", []),
    ("black_scholes", ["--precision", "64", "-n", "3", "7"]),
    ("black_scholes", ["--precision", "16"]),
    ("black_scholes_greeks", ["-v", "8", "-t", "16", "-m", "12"]),
    ("cfd", ["-x", "33", "-y", "21", "-n", "10", "-i", "10"]),
    ("cg", ["--check"]),
    ("cg", ["--pre", "--check"]),
    ("cg", ["-k", "-n", "6", "8", "-v"]),
    ("cholesky", ["--check", "-t"]),
    ("cholesky", ["-d", "complex64", "-n", "8", "16"]),
    ("einsum", []),
    ("einsum", ["-e", "kab,kbc->kac", "-t", "f64", "-n", "20"]),
    ("einsum", ["-e", "kab,kbc->kac", "-t", "f16", "-n", "20"]),
    ("einsum", ["-e", "ab,bc->ac", "-t", "c128", "-n", "40"]),
    ("einsum", ["-e", "ab,bc->ac", "--cupy-compatibility", "-n", "64", "200"]),
    ("einsum", ["-e", "kab,kb->ka", "-n", "70"]),
    ("gemm", []),
    ("gemm", ["--precision", "64", "-i", "10"]),
    ("gemm", ["--precision", "16", "-n", "40"]),
    ("indexing_routines", []),
    ("indexing_routines", ["-r", "ai3", "-t", "-n", "200", "-i", "3", "-w", "1"]),
    ("jacobi", ["--check", "-t"]),
    ("kmeans", ["-n", "2"]),
    ("kmeans", ["--precision", "64", "-d", "3", "--centers", "5", "-m", "30", "-n", "2"]),
    ("kmeans_slow", ["-n", "2"]),
    ("linreg", []),
    ("linreg", ["-B", "--precision", "64", "-i", "300"]),
    ("logreg", []),
    ("logreg", ["-B", "-i", "300"]),
    ("lstm_backward", ["-t"]),
    ("lstm_forward", ["-t"]),
    ("lstm_forward", ["-B", "8", "16", "-s", "3"]),
    ("qr", ["-m", "40", "-n", "30", "--check"]),
    ("qr", ["-m", "48", "-n", "48", "-d", "complex128", "--check"]),
    ("quantiles", ["--check"]),
    ("quantiles", ["-s", "30,20", "-a", "1", "-m", "midpoint", "--check"]),
    ("quantiles", ["-s", "10,12,14", "-d", "int32", "-a", "2", "--check"]),
    ("quantiles", ["-s", "300", "-d", "float32", "-m", "hazen", "--check"]),
    ("richardson_lucy", []),
    ("richardson_lucy", ["--conv-method", "fft", "--shape", "30,20,10", "--filter-shape", "5,4,3"]),
    ("scan", ["--check"]),
    ("scan", ["-o", "cumprod", "-s", "40", "30", "--check"]),
    ("scan", ["-o", "nancumsum", "-a", "0", "--check"]),
    ("solve", ["--check", "-t"]),
    ("solve", ["-s", "3", "-n", "20", "-d", "complex64", "--check"]),
    ("sort", []),
    ("sort", ["-g"]),
    ("sort", ["-d", "int64", "-g", "-u", "50", "-s", "1000"]),
    ("sort", ["-d", "float32", "-s", "20,30", "-a", "0"]),
    ("stencil", ["-t"]),
    ("svd", ["--check"]),
    ("svd", ["-m", "40", "-n", "12", "--no-full-matrices", "--check"]),
    ("svd", ["-m", "20", "-n", "12", "-d", "float32", "--check"]),
    ("svd", ["-d", "int32", "--check"]),
    ("take_pairs", ["-m", "2000", "-n", "12"]),
    ("tssvd", ["--check", "-t"]),
    ("tssvd", ["-m", "60", "-n", "7", "--check"]),
]


def run_case(name, extra, timeout=1800):
    env = dict(os.environ, CNUM_SIM="1")
    cmd = [sys.executable, os.path.join(HERE, name + ".py")] + extra
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, cwd=HERE, capture_output=True, text=True, timeout=timeout)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return {"rc": -1, "verdict": "TIMEOUT", "sim": "", "secs": timeout}
    lines = [l for l in out.strip().splitlines() if l.startswith(("PASS ", "FAIL "))]
    verdict = lines[-1] if lines else "NO-VERDICT"
    sim = [l for l in err.splitlines() if l.startswith("[cnum-sim]")]
    return {"rc": rc, "verdict": verdict, "sim": sim, "secs": time.time() - t0, "err": err}


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    cases = [c for c in CASES if not names or c[0] in names]
    if quick:
        seen, q = set(), []
        for c in cases:
            if c[0] not in seen:
                seen.add(c[0])
                q.append(c)
        cases = q
    ok = 0
    print(f"{'case':58s} {'rc':>3s} {'secs':>6s}  verdict / simulator summary")
    for name, extra in cases:
        r = run_case(name, extra)
        label = " ".join([name] + extra)
        summary = ""
        if r["sim"]:
            m = re.search(r"names_used=(\d+)/\d+ .*errors=(\d+) warnings=(\d+)", r["sim"][0])
            if m:
                summary = f"[names={m.group(1)} errors={m.group(2)} warnings={m.group(3)}]"
        good = r["rc"] == 0 and r["verdict"].startswith("PASS")
        ok += good
        print(f"{label:58s} {r['rc']:>3d} {r['secs']:6.1f}  {r['verdict']}  {summary}")
        for line in r["sim"][1:]:
            print(f"{'':70s}{line}")
        if not good and r.get("err"):
            print("\n".join("    | " + l for l in r["err"].strip().splitlines()[-8:]))
    print(f"\n{ok}/{len(cases)} cases PASS under the simulator")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
