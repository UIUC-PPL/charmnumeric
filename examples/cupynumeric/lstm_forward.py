#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/lstm_forward.py.

Verdict: direct port. The forward pass (lstm_forward(np, ...)) is shared
with the NumPy reference. Changes:
  * np.random.randn is replaced by the native approximately-normal
    generator U.randn.
  * np.tile(h0, (b, 1)) becomes np.tile(h0, (1, b, 1)), a 3-D (1, b, d)
    array, so that Hin[t, :, word_size:] = prev is a same-ndim set_region.
    A 2-D -> 3-D set_region at a nonzero column offset goes through the
    backend cross-ndim path, which does not appear to apply the target
    offset (see README). NumPy accepts both forms identically.
  * In charmnumeric, A[t] keeps the leading dim, giving a (1, b, k) view,
    and Hin[t].dot(W) is a 3-D (one singleton dim) x 2-D SUMMA.
"""

import argparse
import os
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from _cnum_connect import get_interface  # noqa: E402


def lstm_forward(np, X, h0, WLSTM, n, b, d, word_size):
    xphpb = WLSTM.shape[0]
    Hin = np.zeros((n, b, xphpb))
    Hout = np.zeros((n, b, d))
    IFOG = np.zeros((n, b, d * 4))
    IFOGf = np.zeros((n, b, d * 4))
    C = np.zeros((n, b, d))
    Ct = np.zeros((n, b, d))

    for t in range(0, n):
        if t == 0:
            prev = np.tile(h0, (1, b, 1))
        else:
            prev = Hout[t - 1]

        Hin[t, :, :word_size] = X[t]
        Hin[t, :, word_size:] = prev
        # compute all gate activations. dots:
        IFOG[t] = Hin[t].dot(WLSTM)
        # non-linearities
        IFOGf[t, :, : 3 * d] = 1.0 / (1.0 + np.exp(-IFOG[t, :, : 3 * d]))  # sigmoids
        IFOGf[t, :, 3 * d :] = np.tanh(IFOG[t, :, 3 * d :])  # tanh
        # compute the cell activation
        C[t] = IFOGf[t, :, :d] * IFOGf[t, :, 3 * d :]
        if t > 0:
            C[t] += IFOGf[t, :, d : 2 * d] * C[t - 1]
        Ct[t] = np.tanh(C[t])
        Hout[t] = IFOGf[t, :, 2 * d : 3 * d] * Ct[t]
    return Hout, C, Ct, IFOGf


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch", type=int, default=32, dest="batch")
    parser.add_argument("--hidden", type=int, default=10, dest="hidden")
    parser.add_argument("-s", "--sentence", type=int, default=4, dest="sentence")
    parser.add_argument("-w", "--word", type=int, default=10, dest="word")
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    args = parser.parse_args()
    b, d, n, word_size = args.batch, args.hidden, args.sentence, args.word
    if word_size != d:
        # same constraint as the original: X has hidden_size columns but fills word_size
        raise ValueError("the original example requires --word == --hidden")

    interface = get_interface()
    t0 = time.perf_counter()
    X = U.randn((n, b, d), seed=1)
    h0 = U.randn((1, d), seed=2)
    WLSTM = U.randn((word_size + d, 4 * d), seed=3) / numpy.sqrt(word_size + d)

    outs = lstm_forward(cnp, X, h0, WLSTM, n, b, d, word_size)
    got = [o.get(interface) for o in outs]
    if args.timing:
        print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    Xh, h0h, Wh = X.get(interface), h0.get(interface), WLSTM.get(interface)
    ref = lstm_forward(numpy, Xh, h0h, Wh, n, b, d, word_size)
    for name, g, r in zip(("Hout", "C", "Ct", "IFOGf"), got, ref):
        chk.close(name, g, r, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    U.run("lstm_forward", main)
