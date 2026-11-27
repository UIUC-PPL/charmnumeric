#!/usr/bin/env python
"""np.tile with an input or a result at a nonzero decomposition offset. The tile path
assumed both at offset 0: a result unioned with a view's far end (v[n - k:] + tile(a))
sent to chares counted from 0 and hung, one at a small offset came out shifted, and an
input at an offset (tile(v[n - k:] * 2)) was read from the wrong positions or hung.

Env: CNUM_HOST, CNUM_PORT, CNUM_ODF.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cupynumeric"))

import charmnumeric as cnp  # noqa: E402
import _cnum_connect  # noqa: E402
import _cnum_util as U  # noqa: E402


def main(chk):
    iface = _cnum_connect.get_interface()

    def get(x):
        return np.asarray(x.get(iface))

    n = 2**21
    k = 1000
    vh = np.arange(n, dtype=np.float64)
    ah = np.arange(100, dtype=np.float64)
    v = cnp.arange(n, dtype=np.float64)
    a = cnp.arange(100, dtype=np.float64)

    y = v[n - k:] + cnp.tile(a, 10)
    chk.close("v[n - k:] + tile(a, 10)", get(y), vh[n - k:] + np.tile(ah, 10))
    y = v[7:7 + k] + cnp.tile(a, 10)
    chk.close("v[7:7 + k] + tile(a, 10)", get(y), vh[7:7 + k] + np.tile(ah, 10))
    m = 3 * n // 4
    y = v[n // 4:] + cnp.tile(a[:64], m // 64)
    chk.close("v[n / 4:] + tile(a[:64])", get(y), vh[n // 4:] + np.tile(ah[:64], m // 64))

    x = v[n - 100:] * 2
    chk.close("tile(v[n - 100:] * 2, 30)", get(cnp.tile(x, 30)), np.tile(2 * vh[n - 100:], 30))
    x = v[n - 300000:] * 2
    chk.close("tile(v[n - 3e5:] * 2, 3)", get(cnp.tile(x, 3)), np.tile(2 * vh[n - 300000:], 3))
    chk.close("tile(v[n - 100:], (2, 3))", get(cnp.tile(v[n - 100:], (2, 3))),
              np.tile(vh[n - 100:], (2, 3)))

    s = 2600
    bh = np.ones((s, s), dtype=np.float32)
    b = cnp.ones((s, s), dtype=np.float32)
    ch = np.arange(40, dtype=np.float32).reshape(5, 8)
    c = cnp.asarray(ch)
    y = b[s - 200:, s - 200:] + cnp.tile(c, (40, 25))
    chk.close("b[s - 200:, s - 200:] + tile(c, (40, 25))", get(y),
              bh[s - 200:, s - 200:] + np.tile(ch, (40, 25)))


if __name__ == "__main__":
    U.run("test_tile_offsets", main)
