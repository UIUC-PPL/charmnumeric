#!/usr/bin/env python
"""charmnumeric port of cupynumeric/examples/kmeans_slow.py (mask/where variant).

Verdict: port with workarounds (fully native).
Same as kmeans.py (distances, native argmin -> labels and one-hot M,
transposed centroids), but find_centroids keeps the original per-centroid loop:
    counts = bincount(labels)      ->  sum(M, axis=0)              (native)
    centroid_mask = labels == idx  ->  mask = M @ e_idx            (1-D, native)
    np.sum(np.where(mask[..., newaxis], data, zero_point), axis=0)
                                   ->  dataT @ mask   (no (N,1) x (N,D) broadcast;
                                                       where(m,x,0) == m*x)
    distance_sum += np.sum(np.where(mask, pairwise[:, idx], 0.0))
                                   ->  sum(mask * (pairwise @ e_idx))   (native sum)
centroidsT[:, idx:idx+1] = dataT @ mask is a 1-D -> 2-D set_region whose
offset lies only in the singleton column dim. The backend cross-ndim path
handles that case.
"""

import os
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charmnumeric as cnp  # noqa: E402
import _cnum_util as U  # noqa: E402
from _cnum_connect import get_interface  # noqa: E402
from kmeans import calculate_distances, initialize, parse, relabel  # noqa: E402


def find_centroids(centroidsT, dataT, M, pairwise, N, D, C, T):
    # Get the number of points associated with each centroid
    counts = cnp.sum(M, axis=0, keepdims=True)  # bincount(labels, minlength=C) as (1, C)
    # Build label masks for each centroid and sum across all the
    # points assocated with each new centroid
    distance_sum = None
    for idx in range(C):
        e = U.unit_vector(C, idx, dtype=T)
        # Boolean mask indicating where the points are for this center
        centroid_mask = labels == idx
        centroidsT[:, idx : idx + 1] = dataT @ centroid_mask
        term = cnp.sum(centroid_mask * (pairwise @ e))
        distance_sum = term if distance_sum is None else distance_sum + term

    # To avoid introducing divide by zero errors
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
    init_host = [x.get(interface) for x in (data, dataT, centroidsT)]

    DD = (data * data) @ cnp.ones((D, C), dtype=T)
    ones_ND = cnp.ones((N, D), dtype=T)

    labels = None
    iteration = 0
    prior_distance_sum = None
    while iteration < I:
        pairwise = calculate_distances(data, centroidsT, DD, ones_ND)
        new_labels, M = relabel(pairwise, N, C, T)
        centroidsT, distance_sum = find_centroids(
            centroidsT, dataT, M, pairwise, N, D, C, T
        )
        if iteration > 0 and iteration % S == 0:
            total_changes = U.to_float(cnp.count_nonzero(labels - new_labels), interface)
            ds = U.to_float(distance_sum, interface)
            delta = ds / U.to_float(prior_distance_sum, interface)
            print(f"Iteration {iteration} produced {total_changes:.0f} changes, and total distance is {ds}")
            if delta > 1 - 0.000001 and not benchmarking:
                print("Threshold triggered, terminating iterations early")
                break
        prior_distance_sum = distance_sum
        labels = new_labels
        iteration += 1
    prior_distance_sum = U.to_float(prior_distance_sum, interface)
    print(f"Final distance sum at iteration {iteration}: {prior_distance_sum}")
    return centroidsT, iteration, prior_distance_sum, init_host


# ---- original NumPy algorithm (slow variant, reference) ----------------------
def reference(np, data, centroids, C, D, I, S):  # noqa: E741
    data_dots = np.square(np.linalg.norm(data, ord=2, axis=1))
    zero_point = np.zeros((1, data.shape[1]), dtype=data.dtype)
    iteration = 0
    prior_distance_sum = None
    while iteration < I:
        centroid_dots = np.square(np.linalg.norm(centroids, ord=2, axis=1))
        pairwise = data_dots[:, np.newaxis] + centroid_dots[np.newaxis, :]
        pairwise -= 2.0 * np.dot(data, centroids.T)
        labels = np.argmin(pairwise, axis=1)
        counts = np.bincount(labels, minlength=C)
        distance_sum = 0.0
        for idx in range(C):
            centroid_mask = labels == idx
            centroids[idx, :] = np.sum(np.where(centroid_mask[..., np.newaxis], data, zero_point), axis=0)
            distance_sum += np.sum(np.where(centroid_mask, pairwise[:, idx], 0.0))
        counts = np.maximum(counts, np.ones((1,), dtype=np.uint64))
        centroids /= counts[:, np.newaxis]
        if iteration > 0 and iteration % S == 0:
            if distance_sum / prior_distance_sum > 1 - 0.000001:
                break
        prior_distance_sum = distance_sum
        iteration += 1
    return centroids, iteration, prior_distance_sum


def main(chk):
    args = parse()
    T = {32: numpy.float32, 64: numpy.float64}.get(args.P)
    if T is None:
        raise TypeError("Precision must be one of 32 or 64")
    interface = get_interface()
    t0 = time.perf_counter()
    centroidsT, iters, dist, init_host = run_kmeans(interface, args.C, args.D, T, args.I, args.N * 1000, args.S)
    print(f"Elapsed Time: {(time.perf_counter() - t0) * 1e3:.3f} ms (incl. sync)")

    data_h, dataT_h, cT_h = init_host
    chk.close("generated dataT == data.T", dataT_h, data_h.T, rtol=0, atol=0)
    c_ref, it_ref, d_ref = reference(numpy, data_h.copy(), cT_h.T.copy(), args.C, args.D, args.I, args.S)
    rtol = 1e-4 if T == numpy.float32 else 1e-9
    chk.true("iteration count", iters == it_ref, f"(port {iters}, numpy {it_ref})")
    chk.close("centroids", centroidsT.get(interface).T, c_ref, rtol=rtol, atol=rtol)
    chk.close("final distance sum", dist, d_ref, rtol=rtol, atol=0)


if __name__ == "__main__":
    U.run("kmeans_slow", main)
