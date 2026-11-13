#!/usr/bin/env python
"""Sustained multi-tile traffic (several MB per message) through gathers and panel exchanges.

On 2+ nodes the Charm++ netlrts default (9000-byte datagrams) loses messages after enough
traffic and the server hangs; launch with +eth (see README). Env: CNUM_HOST, CNUM_PORT.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_linalg as T  # noqa: E402
from _ops_common import Checker, connect  # noqa: E402


def main():
    T.IFACE = connect()
    chk = Checker("test_multinode_traffic", T.IFACE)
    for _ in range(4):
        T.check_cholesky(chk, 1100, np.float64)
        T.check_solve(chk, 777, np.float64)
        T.check_qr(chk, 1030, 600, np.float64)
        T.check_svd(chk, 600, 1030, np.float64)
        T.check_eigh(chk, 700, np.float64)
    chk.finish()


if __name__ == "__main__":
    main()
