#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/cg.py (CG / Jacobi-preconditioned CG).

Verdict: direct port. Changes from the original:
  * The 2-D adjacency matrix is built natively with cnp.diag(v, k) and
    np.eye(n) becomes cnp.diag(cnp.ones(n)).
  * b uses the native generator U.uniform instead of np.random.rand.
  * The convergence test needs the residual on the host (a scalar fetch, as
    in any lazy runtime). sqrt(rs) < thr is evaluated on that host scalar.
  * The final check A.dot(x) ~ b is computed natively. In addition, x is
    compared against numpy.linalg.solve on the fetched A and b.
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


def generate_2D(N, corners):
    n2 = N**2
    eye = cnp.diag(cnp.ones((n2,)))
    if corners:
        print("Generating %dx%d 2-D adjacency system with corners..." % (n2, n2))
        A = cnp.zeros((n2, n2)) + 8 * eye
    else:
        print("Generating %dx%d 2-D adjacency system without corners..." % (n2, n2))
        A = cnp.zeros((n2, n2)) + 4 * eye
    # These are the same for both cases
    off_one = cnp.full((n2 - 1,), -1, dtype=cnp.float64)
    A += cnp.diag(off_one, k=1)
    A += cnp.diag(off_one, k=-1)
    off_N = cnp.full((N * (N - 1),), -1, dtype=cnp.float64)
    A += cnp.diag(off_N, k=N)
    A += cnp.diag(off_N, k=-N)
    # If we have corners then we have four more cases
    if corners:
        off_N_plus = cnp.full((N * (N - 1) - 1,), -1, dtype=cnp.float64)
        A += cnp.diag(off_N_plus, k=N + 1)
        A += cnp.diag(off_N_plus, k=-(N + 1))
        off_N_minus = cnp.full((N * (N - 1) + 1,), -1, dtype=cnp.float64)
        A += cnp.diag(off_N_minus, k=N - 1)
        A += cnp.diag(off_N_minus, k=-(N - 1))
    # Then we can generate a random b vector (natively)
    b = U.uniform((n2,), seed=7)
    return A, b


def run_cg(interface, N, corners, conv_iters, warmup, max_iters, conv_threshold, verbose):
    A, b = generate_2D(N, corners)

    print("Solving system...")
    x = cnp.zeros((A.shape[1],))
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    converged = -1
    max_iters = min(max_iters, b.shape[0]) if max_iters is not None else b.shape[0]

    r_start = -1.0
    r_end = -1.0
    for i in range(-warmup, max_iters):
        if i == 0:
            r_start = numpy.sqrt(U.to_float(rsold, interface))
        Ap = A.dot(p)
        alpha = rsold / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if verbose:
            print(f"Residual {i + 1}: {U.to_float(rsnew, interface)}")
        # We only do the convergence test every conv_iters or on the last iteration
        if (
            i >= 0
            and (i % conv_iters == 0 or i == (max_iters - 1))
            and numpy.sqrt(U.to_float(rsnew, interface)) < conv_threshold
        ):
            converged = i
            r_end = numpy.sqrt(U.to_float(rsnew, interface))
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    if converged < 0:
        r_end = numpy.sqrt(U.to_float(rsold, interface))
        num_iters = max_iters
        print("Convergence FAILURE!")
    else:
        num_iters = converged + 1
        print("Converged in %d iterations" % (num_iters))
    return A, b, x, converged, num_iters, r_start, r_end


def precondition(N, corners):
    d = 8 * (N**2) if corners else 4 * (N**2)
    M = cnp.diag(cnp.full((N**2,), 1.0 / d))
    return M


def run_preconditioned_cg(interface, N, corners, conv_iters, warmup, max_iters, conv_threshold, verbose):
    print("Solving system with preconditioner...")
    A, b = generate_2D(N, corners)
    M = precondition(N, corners)

    x = cnp.zeros((A.shape[1],))
    r = b - A.dot(x)
    z = M.dot(r)
    p = z
    rzold = r.dot(z)
    converged = -1
    max_iters = min(max_iters, b.shape[0]) if max_iters is not None else b.shape[0]

    r_start = -1.0
    r_end = -1.0
    for i in range(-warmup, max_iters):
        if i == 0:
            r_start = numpy.sqrt(U.to_float(rzold, interface))
        Ap = A.dot(p)
        alpha = rzold / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rznew = r.dot(r)
        if (
            i >= 0
            and (i % conv_iters == 0 or i == (max_iters - 1))
            and numpy.sqrt(U.to_float(rznew, interface)) < conv_threshold
        ):
            converged = i
            break
        if verbose:
            print("Residual: " + str(U.to_float(rznew, interface)))
        z = M.dot(r)
        rznew = r.dot(z)
        beta = rznew / rzold
        p = z + beta * p
        rzold = rznew

    if converged < 0:
        r_end = numpy.sqrt(U.to_float(rzold, interface))
        num_iters = max_iters
        print("Convergence FAILURE!")
    else:
        num_iters = converged + 1
        print("Converged in %d iterations" % (num_iters))
    return A, b, x, converged, num_iters, r_start, r_end


def main(chk):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", dest="check", action="store_true")
    parser.add_argument("-k", "--corners", dest="corners", action="store_true")
    parser.add_argument("-i", "--iters", type=int, default=25, dest="conv_iters")
    parser.add_argument("--pre", dest="precondition", action="store_true")
    parser.add_argument("-m", "--max", type=int, default=None, dest="max_iters")
    parser.add_argument("-w", "--warmup", type=int, default=5, dest="warmup")
    parser.add_argument("-n", "--num", type=int, default=10, dest="N")
    parser.add_argument("-t", "--time", dest="timing", action="store_true")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("--threshold", type=float, default=1e-10, dest="conv_threshold")
    args = parser.parse_args()

    interface = get_interface()
    solver = run_preconditioned_cg if args.precondition else run_cg
    t0 = time.perf_counter()
    A, b, x, converged, num_iters, r_start, r_end = solver(
        interface, args.N, args.corners, args.conv_iters, args.warmup,
        args.max_iters, args.conv_threshold, args.verbose,
    )
    x_val = x.get(interface)
    if args.timing:
        print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")
    print(f"iterations={num_iters} r_start={r_start:.3e} r_end={r_end:.3e}")

    print("Checking result...")
    Ax = A.dot(x).get(interface)
    A_h, b_h = A.get(interface), b.get(interface)
    # the matrix itself must match the NumPy construction from the original
    n2 = args.N**2
    A_np = numpy.zeros((n2, n2)) + (8 if args.corners else 4) * numpy.eye(n2)
    A_np += numpy.diag(numpy.full(n2 - 1, -1.0), k=1) + numpy.diag(numpy.full(n2 - 1, -1.0), k=-1)
    off_N = numpy.full(args.N * (args.N - 1), -1.0)
    A_np += numpy.diag(off_N, k=args.N) + numpy.diag(off_N, k=-args.N)
    if args.corners:
        a = numpy.full(args.N * (args.N - 1) - 1, -1.0)
        c = numpy.full(args.N * (args.N - 1) + 1, -1.0)
        A_np += numpy.diag(a, k=args.N + 1) + numpy.diag(a, k=-(args.N + 1))
        A_np += numpy.diag(c, k=args.N - 1) + numpy.diag(c, k=-(args.N - 1))
    chk.close("A vs numpy construction", A_h, A_np, rtol=0, atol=0)
    if args.max_iters is None:
        chk.true("converged", converged >= 0)
    chk.close("A.dot(x) vs b", Ax, b_h, rtol=1e-5, atol=1e-8)
    if converged >= 0:
        chk.close("x vs numpy.linalg.solve", x_val, numpy.linalg.solve(A_h, b_h), rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    U.run("cg", main)
