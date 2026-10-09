#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/lstm_backward.py.

Verdict: port with workarounds (fully native). Changes:
  * WLSTM.transpose(): charmnumeric has no transpose. WLSTM is input data,
    so its transpose WT is generated natively from the same closed form with
    permuted indices (U.randn(..., perm=(1, 0))). The port checks that the
    fetched WT equals WLSTM.T exactly. A native transpose op would remove
    this workaround.
  * np.sum(dHin[t, :, word_size:], 0) is the native axis reduction
    cnp.sum(..., axis=1): in charmnumeric the integer index keeps its dim, so
    the view is (1, b, d) and the batch axis is axis 1; the result is the
    (1, d) row that dh0[0] expects.
  * np.random.randn is replaced by the native generator U.randn.
The backward pass (lstm_backward(np, ...)) is shared with the NumPy
reference.
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


def lstm_backward(np, WLSTM, WT, dHout, IFOGf, C, Ct, Hin, n, b, d, word_size, batch_sum):
    dIFOG = np.zeros((n, b, d * 4))
    dIFOGf = np.zeros(IFOGf.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dh0 = np.zeros((1, d))

    for t in reversed(range(n)):
        tanhCt = Ct[t]
        dIFOGf[t, :, 2 * d : 3 * d] = tanhCt * dHout[t]
        # backprop tanh non-linearity first then continue backprop
        dC[t] += (1 - tanhCt**2) * (IFOGf[t, :, 2 * d : 3 * d] * dHout[t])

        if t > 0:
            dIFOGf[t, :, d : 2 * d] = C[t - 1] * dC[t]
            dC[t - 1] += IFOGf[t, :, d : 2 * d] * dC[t]

        dIFOGf[t, :, :d] = IFOGf[t, :, 3 * d :] * dC[t]
        dIFOGf[t, :, 3 * d :] = IFOGf[t, :, :d] * dC[t]

        # backprop activation functions
        dIFOG[t, :, 3 * d :] = (1 - IFOGf[t, :, 3 * d :] ** 2) * dIFOGf[t, :, 3 * d :]
        y = IFOGf[t, :, : 3 * d]
        dIFOG[t, :, : 3 * d] = (y * (1.0 - y)) * dIFOGf[t, :, : 3 * d]

        # backprop matrix multiply (WT == WLSTM.transpose())
        dHin[t] = dIFOG[t].dot(WT)

        # backprop the identity transforms into Hin
        if t > 0:
            dHout[t - 1, :] += dHin[t, :, word_size:]
        else:
            dh0[0] += batch_sum(dHin[t, :, word_size:])
    return dIFOG, dIFOGf, dHin, dC, dh0, dHout


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch", type=int, default=32, dest="batch")
    parser.add_argument("--hidden", type=int, default=10, dest="hidden")
    parser.add_argument("-s", "--sentence", type=int, default=4, dest="sentence")
    parser.add_argument("-w", "--word", type=int, default=10, dest="word")
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    args = parser.parse_args()
    b, d, n, word_size = args.batch, args.hidden, args.sentence, args.word

    interface = get_interface()
    t0 = time.perf_counter()
    scale = 1.0 / numpy.sqrt(word_size + d)
    WLSTM = U.randn((word_size + d, 4 * d), seed=1) * scale
    WT = U.randn((word_size + d, 4 * d), seed=1, perm=(1, 0)) * scale  # == WLSTM.T
    xphpb = WLSTM.shape[0]
    dHout = U.randn((n, b, d), seed=2)
    IFOGf = U.randn((n, b, d * 4), seed=3)
    C = U.randn((n, b, d), seed=4)
    Ct = U.randn((n, b, d), seed=5)
    Hin = U.randn((n, b, xphpb), seed=6)
    host_in = [x.get(interface) for x in (WLSTM, WT, dHout, IFOGf, C, Ct, Hin)]

    outs = lstm_backward(
        cnp, WLSTM, WT, dHout, IFOGf, C, Ct, Hin, n, b, d, word_size,
        batch_sum=lambda x: cnp.sum(x, axis=1),
    )
    got = [o.get(interface) for o in outs]
    if args.timing:
        print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    Wh, WTh, dHouth, IFOGfh, Ch, Cth, Hinh = host_in
    chk.close("generated WT == WLSTM.T", WTh, Wh.T, rtol=0, atol=0)
    ref = lstm_backward(
        numpy, Wh, Wh.T, dHouth.copy(), IFOGfh, Ch, Cth, Hinh, n, b, d, word_size,
        batch_sum=lambda x: numpy.sum(x, 0),
    )
    for name, g, r in zip(("dIFOG", "dIFOGf", "dHin", "dC", "dh0", "dHout"), got, ref):
        chk.close(name, g, r, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    U.run("lstm_backward", main)
