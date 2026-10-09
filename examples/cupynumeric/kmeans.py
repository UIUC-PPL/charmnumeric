#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/kmeans.py (bincount variant).

Verdict: port with workarounds (fully native).
The original needs argmin, bincount, amin, linalg.norm(axis=1), newaxis
broadcasting and centroids.T. The port maps each one to native ops:
  * labels = argmin(pairwise, axis=1) is the native axis reduction, produced
    directly in the working float dtype (cnp.argmin(..., dtype=T), standing
    in for astype) as an (N, 1) column. It is expanded to a one-hot matrix
    M (N, C) with M[:, c] = (labels == c), a native comparison (0/1 mask).
  * bincount(labels) = sum(M, axis=0) (native axis reduction).
    bincount(labels, weights=data[:, d]) for all d at once = dataT @ M,
    giving the (D, C) centroid sums (one SUMMA instead of D bincounts whose
    1-D results would need 1-D -> 2-D row placement).
  * sum(amin(pairwise, axis=1)) is native (cnp.amin + cnp.sum).
  * data_dots[:, newaxis] + centroid_dots[newaxis, :] is built with two
    SUMMAs against ones: (data*data) @ ones(D, C) + ones(N, D) @ (cT*cT).
  * centroids.T: the port keeps centroids in transposed form cT (D, C)
    throughout. dataT is generated natively from the same closed form as
    data (checked against data.T).
  * maximum(counts, 1) is native. counts[:, newaxis] is
    tile(counts_row, (D, 1)).
  * np.not_equal(labels, new_labels).sum() = count_nonzero(labels - new_labels)
    (native).
Defaults to float64. With float32, a near-tie can make argmin differ from
the NumPy reference. Backend ops that would make it direct: astype,
transpose, general broadcasting.
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


def initialize(N, D, C, T):
    data = U.uniform((N, D), seed=1, dtype=T)
    dataT = U.uniform((N, D), seed=1, dtype=T, perm=(1, 0))  # == data.T
    # random centers, held transposed: centroidsT == centroids.T
    centroidsT = U.uniform((C, D), seed=2, dtype=T, perm=(1, 0))
    return data, dataT, centroidsT


def calculate_distances(data, centroidsT, DD, ones_ND):
    # centroid_dots[newaxis, :] broadcast to (N, C): ones(N, D) @ (cT * cT)
    CD = ones_ND @ (centroidsT * centroidsT)
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
    return DD + CD - 2.0 * (data @ centroidsT)


def relabel(pairwise, N, C, T):
    """labels = argmin(pairwise, axis=1) (native) and its one-hot (N, C) form."""
    labels = cnp.argmin(pairwise, axis=1, keepdims=True, dtype=T)  # (N, 1), exact ints
    M = cnp.zeros((N, C), dtype=T)
    for c in range(C):
        M[:, c : c + 1] = cnp.equal(labels, float(c))  # [labels == c], 0/1 mask
    return labels, M


def find_centroids(dataT, M, pairwise, N, D, C, T):
    # Get the number of points associated with each centroid: bincount(labels)
    counts = cnp.sum(M, axis=0, keepdims=True)  # (1, C)
    # weighted bincounts for every dimension at once: (D, N) @ (N, C)
    centroidsT = dataT @ M
    distance_sum = cnp.sum(cnp.amin(pairwise, axis=1))
    counts = cnp.maximum(counts, 1.0)
    centroidsT = centroidsT / cnp.tile(U.materialize(counts), (D, 1))
    return centroidsT, distance_sum


def run_kmeans(interface, C, D, T, I, N, S, benchmarking=False):  # noqa: E741
    print("Running kmeans...")
    print("Number of data points: " + str(N))
    print("Number of dimensions: " + str(D))
    print("Number of centroids: " + str(C))
    print("Max iterations: " + str(I))
    data, dataT, centroidsT = initialize(N, D, C, T)
    init = (data, dataT, centroidsT)
    init_host = [x.get(interface) for x in init]

    # data_dots[:, newaxis] broadcast to (N, C): (data*data) @ ones(D, C)
    DD = (data * data) @ cnp.ones((D, C), dtype=T)
    ones_ND = cnp.ones((N, D), dtype=T)

    labels = None
    iteration = 0
    prior_distance_sum = None
    while iteration < I:
        pairwise = calculate_distances(data, centroidsT, DD, ones_ND)
        new_labels, M = relabel(pairwise, N, C, T)
        centroidsT, distance_sum = find_centroids(dataT, M, pairwise, N, D, C, T)

        if iteration > 0 and iteration % S == 0:
            # np.sum(np.not_equal(labels, new_labels))
            total_changes = U.to_float(cnp.count_nonzero(labels - new_labels), interface)
            ds = U.to_float(distance_sum, interface)
            delta = ds / U.to_float(prior_distance_sum, interface)
            print(f"Iteration {iteration} produced {total_changes:.0f} changes, and total distance is {ds}")
            # We ignore the threshold test when benchmarking
            if delta > 1 - 0.000001 and not benchmarking:
                print("Threshold triggered, terminating iterations early")
                break
        prior_distance_sum = distance_sum
        labels = new_labels
        iteration += 1
    prior_distance_sum = U.to_float(prior_distance_sum, interface)
    print(f"Final distance sum at iteration {iteration}: {prior_distance_sum}")
    return centroidsT, iteration, prior_distance_sum, init_host


# ---- original NumPy algorithm (reference) -------------------------------------
def reference(np, data, centroids, C, I, S):  # noqa: E741
    data_dots = np.square(np.linalg.norm(data, ord=2, axis=1))
    labels = None
    iteration = 0
    prior_distance_sum = None
    while iteration < I:
        centroid_dots = np.square(np.linalg.norm(centroids, ord=2, axis=1))
        pairwise = data_dots[:, np.newaxis] + centroid_dots[np.newaxis, :]
        pairwise -= 2.0 * np.dot(data, centroids.T)
        new_labels = np.argmin(pairwise, axis=1)
        counts = np.bincount(new_labels, minlength=C)
        for idx in range(data.shape[1]):
            centroids[:, idx] = np.bincount(new_labels, weights=data[:, idx], minlength=C)
        distance_sum = np.sum(np.amin(pairwise, axis=1))
        counts = np.maximum(counts, np.ones((1,), dtype=np.uint64))
        centroids /= counts[:, np.newaxis]
        if iteration > 0 and iteration % S == 0:
            delta = distance_sum / prior_distance_sum
            if delta > 1 - 0.000001:
                break
        prior_distance_sum = distance_sum
        labels = new_labels  # noqa: F841
        iteration += 1
    return centroids, iteration, prior_distance_sum


def parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", type=int, default=10, dest="C")
    parser.add_argument("-d", "--dims", type=int, default=2, dest="D")
    parser.add_argument("-m", "--max-iters", type=int, default=20, dest="I")
    parser.add_argument("-n", "--num", type=int, default=1, dest="N",
                        help="number of elements in the data set in thousands")
    parser.add_argument("--precision", type=int, default=64, dest="P")
    parser.add_argument("-s", "--sample", type=int, default=5, dest="S")
    return parser.parse_args(argv)


def check(chk, centroidsT, iters, dist, init_host, args, T, interface):
    data_h, dataT_h, cT_h = init_host
    chk.close("generated dataT == data.T", dataT_h, data_h.T, rtol=0, atol=0)
    c_ref, it_ref, d_ref = reference(numpy, data_h.copy(), cT_h.T.copy(), args.C, args.I, args.S)
    rtol = 1e-4 if T == numpy.float32 else 1e-9
    chk.true("iteration count", iters == it_ref, f"(port {iters}, numpy {it_ref})")
    chk.close("centroids", centroidsT.get(interface).T, c_ref, rtol=rtol, atol=rtol)
    chk.close("final distance sum", dist, d_ref, rtol=rtol, atol=0)


def main(chk):
    args = parse()
    T = {32: numpy.float32, 64: numpy.float64}.get(args.P)
    if T is None:
        raise TypeError("Precision must be one of 32 or 64")
    interface = get_interface()
    t0 = time.perf_counter()
    centroidsT, iters, dist, init_host = run_kmeans(interface, args.C, args.D, T, args.I, args.N * 1000, args.S)
    print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")
    check(chk, centroidsT, iters, dist, init_host, args, T, interface)


if __name__ == "__main__":
    U.run("kmeans", main)
