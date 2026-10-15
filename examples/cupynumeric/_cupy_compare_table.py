"""Markdown table from run_cupy_compare.sh results: _cupy_compare_table.py p1/results.jsonl p4/results.jsonl"""
import json
import sys


def load(path):
    rows = {}
    for line in open(path):
        d = json.loads(line)
        rows[d["program"]] = d
    return rows


def cell(d):
    if d is None:
        return "-", "-", "-", "-"
    ok = "yes" if d["match"] else "NO"
    err = f'{d["max_abs"]:.1e} / {d["max_rel"]:.1e}'
    ms = lambda v: "-" if v is None else f"{v:.0f}"
    ratio = "-" if d["ratio"] is None else f'{d["ratio"]:.2f}'
    return f"{ok} ({err})", ms(d["charm_ms"]), ms(d["cupy_ms"]), ratio


one, four = (load(p) for p in sys.argv[1:3])
print("| program | match 1 GPU (max abs / rel) | charm ms | cupy ms | charm/cupy "
      "| match 4 GPU (max abs / rel) | charm ms | cupy ms | charm/cupy |")
print("|---|---|---|---|---|---|---|---|---|")
for name in sorted(set(one) | set(four)):
    print(f"| {name} | " + " | ".join(cell(one.get(name)) + cell(four.get(name))) + " |")
