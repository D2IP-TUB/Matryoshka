from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.sparse as sp


class PrioritySample(NamedTuple):
    indices: np.ndarray
    threshold: float
    row_norms_sq: np.ndarray


def priority_sampling(A: sp.csr_matrix | np.ndarray, k: int, seed: int = 0) -> PrioritySample:
    """Select k row indices from A via Priority Sampling (Algorithm 1).

    Works with both dense numpy arrays and scipy CSR sparse matrices.

    Parameters
    ----------
    A : (n, d) csr_matrix or ndarray
    k : number of rows to keep
    seed : shared random seed (must match across matrices)

    Returns
    -------
    PrioritySample with (indices, threshold, row_norms_sq)
    """
    n = A.shape[0]
    rng = np.random.RandomState(seed)
    h = rng.uniform(0, 1, size=n)

    # Squared row norms — sparse-friendly
    if sp.issparse(A):
        row_norms_sq = np.asarray(A.multiply(A).sum(axis=1)).ravel()
    else:
        row_norms_sq = np.sum(A ** 2, axis=1)

    # Rank R_i = h(i) / ||A_i||^2  (inf for zero rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        ranks = np.where(row_norms_sq > 0, h / row_norms_sq, np.inf)

    num_finite = int(np.sum(np.isfinite(ranks)))
    if k >= num_finite:
        selected = np.where(np.isfinite(ranks))[0]
    else:
        selected = np.argpartition(ranks, k)[:k]

    # τ = (k+1)-th smallest rank (or inf if ≤k non-zero rows)
    if len(selected) < num_finite:
        remaining = np.setdiff1d(np.arange(n), selected)
        remaining = remaining[np.isfinite(ranks[remaining])]
        tau = float(np.min(ranks[remaining]))
    else:
        tau = np.inf

    return PrioritySample(
        indices=selected,
        threshold=tau,
        row_norms_sq=row_norms_sq[selected],
    )


def reweight_numpy(X: np.ndarray, sample: PrioritySample) -> np.ndarray:
    """Compute reweighted X from a priority sample.

    Each sampled row i contributes x_i^T x_i / p_i where
    p_i = min(1, ||A_i||^2 * τ).
    """
    tau = sample.threshold
    X_s = X[sample.indices]
    weights = np.minimum(1.0, sample.row_norms_sq * tau)
    # Scale rows by 1/sqrt(p_i), then standard X^T X
    scale = 1.0 / np.sqrt(weights)
    X_scaled = X_s * scale[:, None]
    return X_scaled


def reweight(X: pl.DataFrame, sample: PrioritySample, sample_mapping: pl.DataFrame, key_column: str) -> pl.DataFrame:
    """Compute reweighted X from a priority sample, for Polars DataFrames."""
    tau = sample.threshold
    X_joined = X.join(sample_mapping, on=key_column, how="inner")

    scale_expr = (
        pl.when(pl.col("row_norms_sq") * tau < 1.0)
        .then(1.0 / (pl.col("row_norms_sq") * tau).sqrt())
        .otherwise(1.0)
    )

    float_cols = [c for c in X_joined.columns if c not in (key_column, "row_norms_sq") and X_joined[c].dtype in pl.FLOAT_DTYPES]

    return X_joined.with_columns(
        [pl.col(c) * scale_expr for c in float_cols]
    ).drop("row_norms_sq")
