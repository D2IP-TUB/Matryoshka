"""
Optimal partition interval computation for LSH Ensemble.

This module creates optimal partition intervals:
    [l1, u1], [l2, u2], ...
with all boundaries inclusive, using set size distribution.

Partitions help LSH Ensemble handle variable set sizes efficiently
by grouping similar-sized sets together for more accurate containment queries.
"""

import numpy as np
from typing import List, Tuple


def _compute_nfp_uniform(l: int, u: int, cum_counts: np.ndarray, sizes: np.ndarray) -> float:
    """
    Computes the expected number of false positives caused by using
    u to approximate set sizes in the interval [l, u], assuming uniform
    distribution of set sizes within the interval.

    Args:
        l: The lower bound index on set sizes.
        u: The upper bound index on set sizes.
        cum_counts: The complete cumulative distribution of set sizes.
        sizes: The complete domain of set sizes.

    Returns:
        float: The expected number of false positives.
    """
    if l > u:
        raise ValueError("l must be less or equal to u")
    if l == 0:
        n = cum_counts[u]
    else:
        n = cum_counts[u] - cum_counts[l - 1]
    if sizes[u] == 0:
        return 0.0
    return n * float(sizes[u] - sizes[l]) / float(2 * sizes[u])


def _compute_nfps_uniform(cum_counts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """
    Computes the matrix of expected false positives for all possible
    sub-intervals of the complete domain of set sizes, assuming uniform
    distribution of set sizes within each sub-interval.

    Args:
        cum_counts: The complete cumulative distribution of set sizes.
        sizes: The complete domain of set sizes.

    Returns:
        np.ndarray: The 2-D array of expected number of false positives
            for every pair of [l, u] interval, where l is axis-0 and u is axis-1.
    """
    nfps = np.zeros((len(sizes), len(sizes)))
    for l in range(len(sizes)):
        for u in range(l, len(sizes)):
            nfps[l, u] = _compute_nfp_uniform(l, u, cum_counts, sizes)
    return nfps


def _compute_nfp_real(l: int, u: int, counts: np.ndarray, sizes: np.ndarray) -> float:
    """
    Computes the expected number of false positives caused by using
    u to approximate set sizes in the interval [l, u], using the real
    set size distribution.

    Args:
        l: The lower bound index on set sizes.
        u: The upper bound index on set sizes.
        counts: The complete distribution of set sizes.
        sizes: The complete domain of set sizes.

    Returns:
        float: The expected number of false positives.
    """
    if l > u:
        raise ValueError("l must be less or equal to u")
    if sizes[u] == 0:
        return 0.0
    return np.sum((float(sizes[u]) - sizes[l:u + 1]) / float(sizes[u]) * counts[l:u + 1])


def _compute_nfps_real(counts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """
    Computes the matrix of expected false positives for all possible
    sub-intervals of the complete domain of set sizes.

    Args:
        counts: The complete distribution of set sizes.
        sizes: The complete domain of set sizes.

    Returns:
        np.ndarray: The 2-D array of expected number of false positives
            for every pair of [l, u] interval, where l is axis-0 and u is axis-1.
    """
    nfps = np.zeros((len(sizes), len(sizes)))
    for l in range(len(sizes)):
        for u in range(l, len(sizes)):
            nfps[l, u] = _compute_nfp_real(l, u, counts, sizes)
    return nfps


def _compute_best_partitions(num_part: int, sizes: np.ndarray, nfps: np.ndarray):
    """
    Computes the optimal partitions given the size distributions
    and computed number of expected false positives for all sub-intervals.

    Args:
        num_part: The number of partitions to create.
        sizes: The complete domain of set sizes in sorted order.
        nfps: The computed number of expected false positives
            for all sub-intervals; axis-0 is for the indexes of lower bounds and
            axis-1 is for the indexes of upper bounds.

    Returns:
        tuple: (partitions, total_nfps, cost)
            - partitions: List of lower and upper bounds of set sizes for all partitions.
            - total_nfps: Total number of expected false positives from all partitions.
            - cost: A N x (p-1) matrix of the computed optimal NFPs for
                all sub-problems given upper bound set size and number of partitions.
    """
    if num_part < 2:
        raise ValueError("num_part cannot be less than 2")
    if num_part > len(sizes):
        raise ValueError("num_part cannot be greater than the domain size of all set sizes")

    # If number of partitions is 2, simply find the upper bound of the first partition
    if num_part == 2:
        total_nfps, u = min(
            (nfps[0, u1] + nfps[u1 + 1, len(sizes) - 1], u1)
            for u1 in range(0, len(sizes) - 1)
        )
        return [(sizes[0], sizes[u]), (sizes[u + 1], sizes[-1])], total_nfps, None

    # Initialize subproblem total NFPs
    cost = np.zeros((len(sizes), num_part - 2))

    # p is the number of partitions in the subproblem
    # p2i translates the number of partition into the index in the matrix
    def p2i(p):
        return p - 2

    # Compute p >= 2 until before p = num_part
    for p in range(2, num_part):
        # Compute best partition for subproblems with increasing max index u,
        # starting from the smallest possible u given the p
        for u in range(p - 1, len(sizes)):
            if p == 2:
                cost[u, p2i(p)] = min(
                    nfps[0, u1] + nfps[u1 + 1, u]
                    for u1 in range(u)
                )
            else:
                cost[u, p2i(p)] = min(
                    cost[u1, p2i(p - 1)] + nfps[u1 + 1, u]
                    for u1 in range((p - 1) - 1, u)
                )

    p = num_part
    # Find the optimal upper bound index of the 2nd right-most partition
    total_nfps, u = min(
        (cost[u1, p2i(p - 1)] + nfps[u1 + 1, len(sizes) - 1], u1)
        for u1 in range((p - 1) - 1, len(sizes) - 1)
    )
    partitions = [(sizes[u + 1], sizes[-1])]
    p -= 1

    # Back track to find the best partitions
    while p > 1:
        _, u1_best = min(
            (cost[u1, p2i(p)] + nfps[u1 + 1, u], u1)
            for u1 in range((p - 1) - 1, u)
        )
        partitions.insert(0, (sizes[u1_best + 1], sizes[u]))
        u = u1_best
        p -= 1

    partitions.insert(0, (sizes[0], sizes[u]))
    return partitions, total_nfps, cost


def optimal_partitions(sizes: np.ndarray, counts: np.ndarray, num_part: int) -> List[Tuple[int, int]]:
    """
    Compute the optimal partitions given a distribution of set sizes.

    Args:
        sizes: The complete domain of set sizes in ascending order.
        counts: The frequencies of all set sizes in the same order as `sizes`.
        num_part: The number of partitions to create.

    Returns:
        List[Tuple[int, int]]: A list of partitions in the form of `(lower, upper)` tuples,
            where `lower` and `upper` are lower and upper bound (inclusive)
            set sizes of each partition.
    """
    if num_part < 2:
        return [(sizes[0], sizes[-1])]
    if num_part >= len(sizes):
        partitions = [(x, x) for x in sizes]
        return partitions
    nfps = _compute_nfps_real(counts, sizes)
    partitions, _, _ = _compute_best_partitions(num_part, sizes, nfps)
    return partitions
