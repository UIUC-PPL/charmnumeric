#!/usr/bin/env python
"""Run every port under the host-side simulator (CNUM_SIM=1) and tabulate results.

Usage:  python _static_check.py [--quick] [name ...]

Each port runs in a subprocess with CNUM_SIM=1, so _cnum_connect.get_interface()
returns _cnum_sim.SimInterface. This exercises the whole Python side:
graph construction, shape and dtype logic, fusion, DAG serialization, and
the port's own PASS/FAIL verification against NumPy. It also flags
patterns the backend does not support (see _cnum_sim.py). It does NOT
exercise the C++ backend.
"""

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    ("black_scholes", []),
    ("black_scholes", ["--precision", "64"]),
    ("black_scholes_greeks", []),
    ("cfd", []),
    ("cg", []),
    ("cg", ["--pre"]),
    ("cg", ["-k"]),
    ("cholesky", []),
    ("cholesky", ["--input", "eye", "-n", "64"]),
    ("einsum", []),
    ("einsum", ["-e", "kab,kbc->kac", "-t", "f64"]),
    ("einsum", ["-e", "ab,bc->ac"]),
    ("gemm", []),
    ("gemm", ["--precision", "64"]),
    ("indexing_routines", []),
    ("jacobi", []),
    ("kmeans", []),
    ("kmeans", ["--precision", "32", "-d", "3", "--centers", "5", "-m", "30"]),
    ("kmeans_slow", []),
    ("linreg", []),
    ("linreg", ["-B", "--precision", "64"]),
    ("logreg", []),
    ("logreg", ["-B"]),
    ("lstm_backward", []),
    ("lstm_forward", []),
    ("qr", []),
    ("qr", ["-m", "48", "-n", "48"]),
    ("quantiles", []),
    ("quantiles", ["-s", "30,20", "-a", "1", "-m", "midpoint"]),
    ("richardson_lucy", []),
    ("scan", []),
    ("scan", ["-o", "cumprod"]),
    ("scan", ["-t", "float", "-o", "nancumsum"]),
    ("scan", ["-t", "float", "-s", "40", "30", "-a", "1", "-o", "nancumprod"]),
    ("scan", ["-s", "40", "30"]),
    ("solve", []),
    ("solve", ["-s", "3"]),
    ("sort", []),
    ("sort", ["--argsort"]),
    ("sort", ["-d", "int64", "--argsort", "-u", "50"]),
    ("sort", ["-d", "int32", "-s", "20,30", "-a", "0"]),
    ("stencil", []),
    ("svd", []),
    ("svd", ["-m", "40", "-n", "12"]),
    ("take_pairs", []),
    ("tssvd", []),
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
