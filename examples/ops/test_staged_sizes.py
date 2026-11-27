#!/usr/bin/env python
"""Messages of many distinct sizes (one new pinned block size class after another), then many
of one new, larger size: results must stay exact.

On a GPU build this also checks that zero-copy staging survives its registration cap. Run the
server with every cross-process message staged and sent zero-copy over RDMA and a small cap,
and point CNUM_STAGESTAT_LOG at its log, e.g. (ofi build, one host, 4 processes):
    CNUM_GPU_IPC=0 CNUM_GPU_ZC_MIN=1 CNUM_GPU_ZC_REGS=8 CNUM_STAGE_STATS=1 \\
    SERVER_ARGS=+noCMAForZC LOCAL= CNUM_STAGESTAT_LOG=logs/test_staged_sizes.server.log \\
    harness/run_one.sh server.out 4 34001 examples/ops/test_staged_sizes.py
The first phase registers far more distinct blocks than the cap; the last phase moves most of
the bytes in blocks of a size not seen before, so it is zero-copy only if registrations of
cached blocks make room for new ones. Env: CNUM_HOST, CNUM_PORT, CNUM_ODF.
"""

import os
import re
import time

import numpy as np

import charmnumeric as cnp
from _ops_common import SIM, Checker, connect, linear_index, sync

SIZES = [40, 56, 80, 112, 160, 224, 320, 448, 640, 900, 1280]  # message bytes double per step
LAST, LAST_REPS = 2896, 6


def transposes(chk, n, reps=1):
    h = np.arange(n * n, dtype=np.float64).reshape(n, n)
    x = linear_index((n, n))
    for r in range(reps):
        y = cnp.transpose(x) * 2.0 + float(r)
        chk.check(f"transpose {n}x{n} rep {r}", y, h.T * 2.0 + float(r))


def stagestat_share(path):
    """Zero-copy share of the bytes each sending process sent (last CNUM_STAGE_STATS line)."""
    pat = re.compile(r"stagestat pid=(\d+) .*?send n=\d+ bytes=(\d+) zc=(\d+)")
    last = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                last[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return {pid: zc / b for pid, (b, zc) in last.items() if b > 0}


def main():
    iface = connect()
    chk = Checker("test_staged_sizes", iface)
    for n in SIZES:
        transposes(chk, n)
    transposes(chk, LAST, LAST_REPS)
    log = os.environ.get("CNUM_STAGESTAT_LOG")
    if log and not SIM:
        sync(iface)
        time.sleep(0.5)  # stats print at most every 0.25 s: one more message flushes them
        transposes(chk, SIZES[0])
        time.sleep(0.5)
        share = stagestat_share(log)
        print("zero-copy share of sent bytes per process:",
              {k: round(v, 3) for k, v in sorted(share.items())})
        chk.check("staged sends in several processes", np.array(len(share) > 1), np.array(True))
        chk.check("zero-copy past the registration cap",
                  np.array(min(share.values(), default=0.0) >= 0.5), np.array(True))
    else:
        print("  SKIP zero-copy check (needs CNUM_STAGESTAT_LOG and a GPU server)")
    chk.finish()


if __name__ == "__main__":
    main()
