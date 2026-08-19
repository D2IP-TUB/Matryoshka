"""Preparation of a query table for augmentation.

The discovery pipeline consumes a query table in a specific shape, because the
Gram matrix sketch of the query table is built directly from its columns:

* column 0 is the join key, of type ``String``, normalised with
  :func:`matryoshka.utils.common.process_key` so that it matches the keys the
  offline indexer wrote into the inverted index;
* the remaining columns are numeric and free of nulls;
* the target column is numeric, ordinal-encoded for classification tasks.

:class:`matryoshka.join_selection.JoinSelection` rejects any other shape with
:class:`matryoshka.exceptions.UserTableNotProcessed`. :func:`prepare_query_table`
produces that shape from an arbitrary table using ordinal encoding for
categorical columns and constant-free imputation, with scikit-learn as the only
dependency. It is deliberately simple: the paper's experiments use a heavier
AutoGluon-based pipeline, and any encoder that yields numeric, null-free
columns is a valid substitute.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .utils.common import process_key

# Columns whose magnitude exceeds this bound destabilise the Cholesky and QR
# factorisations of the proxy models, so they are dropped rather than scaled:
# rescaling would change the feature semantics recorded in the augmentation plan.
_MAX_ABS_VALUE = 1e17

_NUMERIC_DTYPES = (
    pl.Float32, pl.Float64,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Boolean,
)


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype in _NUMERIC_DTYPES


def prepare_query_table(
    table: pl.DataFrame,
    key: str,
    target: str,
    *,
    features: list[str] | None = None,
    task: str = 'classification',
    n_bins: int | None = None,
) -> pl.DataFrame:
    """Return ``table`` in the layout expected by ``JoinSelection.find_best_joins``.

    Parameters
    ----------
    table
        Query table. Any column types are accepted.
    key
        Name of the join column. Cast to ``String`` and normalised.
    target
        Name of the prediction target.
    features
        Feature columns to retain. Defaults to every column other than ``key``
        and ``target``.
    task
        ``'classification'`` ordinal-encodes the target; ``'regression'`` casts
        it to ``Float64``.
    n_bins
        If given, numeric features are quantile-binned into ``n_bins`` buckets
        and one-hot encoded with the first level dropped. This reproduces the
        ``Binned Features`` regime of Section 7.2.

    Returns
    -------
    polars.DataFrame
        Key column first, features next, target last.
    """
    if task not in ('classification', 'regression'):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    for name in (key, target):
        if name not in table.columns:
            raise KeyError(f'column {name!r} is not in the query table')

    if features is None:
        features = [c for c in table.columns if c not in (key, target)]
    missing = [c for c in features if c not in table.columns]
    if missing:
        raise KeyError(f'feature columns not in the query table: {missing}')

    df = table.drop_nulls(subset=[key, target])
    if df.height == 0:
        raise ValueError('no rows left after dropping null keys and targets')

    # --- target -----------------------------------------------------------
    if task == 'classification':
        categories = df.get_column(target).unique(maintain_order=True).to_list()
        codes = {value: float(i) for i, value in enumerate(categories)}
        y = df.get_column(target).replace_strict(codes, return_dtype=pl.Float64)
    else:
        y = df.get_column(target).cast(pl.Float64)
    y = y.rename(target)

    # --- features ---------------------------------------------------------
    encoded: list[pl.Series] = []
    for column in features:
        series = df.get_column(column)
        if _is_numeric(series.dtype):
            values = series.cast(pl.Float64)
            fill = values.drop_nulls().drop_nans().median()
            values = values.fill_null(fill if fill is not None else 0.0)
            values = values.fill_nan(fill if fill is not None else 0.0)
        else:
            # Ordinal encoding by first appearance; nulls collapse into the
            # most frequent level so that no column carries a sentinel value.
            levels = series.cast(pl.String).unique(maintain_order=True).drop_nulls().to_list()
            mapping = {level: float(i) for i, level in enumerate(levels)}
            values = series.cast(pl.String).replace_strict(
                mapping, default=None, return_dtype=pl.Float64
            )
            mode = values.drop_nulls().mode()
            values = values.fill_null(float(mode[0]) if len(mode) else 0.0)
        if values.abs().max() is not None and values.abs().max() > _MAX_ABS_VALUE:
            continue
        encoded.append(values.rename(column))

    if not encoded:
        raise ValueError('no usable feature columns remain after encoding')

    features_df = pl.DataFrame(encoded)
    if n_bins is not None:
        if n_bins < 2:
            raise ValueError(f'n_bins must be at least 2, got {n_bins}')
        binned = features_df.select([
            pl.col(c).qcut(
                n_bins, labels=[str(i) for i in range(n_bins)], allow_duplicates=True
            ).alias(c)
            for c in features_df.columns
        ])
        features_df = binned.to_dummies(drop_first=True)

    keys = df.get_column(key).cast(pl.String).map_elements(
        process_key, return_dtype=pl.String
    ).rename(key)

    out = pl.DataFrame([keys]).hstack(features_df).hstack(pl.DataFrame([y]))
    assert out.null_count().to_numpy().sum() == 0, 'prepared query table contains nulls'
    return out


def train_test_split_by_key(
    table: pl.DataFrame, test_fraction: float = 0.2, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split ``table`` into train and test frames by row, with a fixed seed.

    Provided so that examples and the paper's downstream evaluation share one
    deterministic split routine.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f'test_fraction must lie in (0, 1), got {test_fraction}')
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(table.height)
    cut = int(round(table.height * (1.0 - test_fraction)))
    train_idx, test_idx = permutation[:cut], permutation[cut:]
    return table[train_idx], table[test_idx]
