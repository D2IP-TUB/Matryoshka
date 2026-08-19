import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from collections import defaultdict
from copy import deepcopy

import adbc_driver_postgresql.dbapi as adbc_dbapi
import numpy as np
import polars as pl
import ray

from matryoshka.utils.f32_numpy import is_patched

from ..exceptions import EmptyAugmentation
from ..utils.common import compute_outer_products_adaptive as compute_outer_products
from ..utils.common import extract_cofactors
from .base.sketch import AugSketch, BaseSketch


def _build_group_aug_sketch(sums: np.ndarray, table_feature_index: str,
                            column_names: list[str] | None = None,
                            base_column: str | None = None) -> AugSketch:
    """Build an AugSketch for a group of (possibly synthetic) polynomial/interaction
    columns. Mirrors the semantics of `SketchProcessor._split_aug_table_sketch`:
    each row has count=1, `sum_` holds per-row cell values, `diag` is the
    elementwise square of `sum_`, and `cofactors` are per-row outer products
    for multi-column groups (object-null for single-column groups, matching the
    existing split code path so `_join_aug_table` takes its object-dtype branch).

    Args:
        sums: array of shape (n_rows, n_cols) — already-expanded per-row values.
        table_feature_index: unique cache key for the resulting group.
        column_names: optional list of length n_cols with synthetic feature
            names; if provided, populated into `aug_feature_indices`.
        base_column: optional identifier of the source column (post-aggregation
            stripping) used for sibling dedup at selection time.
    """
    if sums.ndim != 2:
        raise ValueError(f'_build_group_aug_sketch expects 2D sums, got shape {sums.shape}')
    n_rows, n_cols = sums.shape
    counts = np.repeat([[1]], n_rows, axis=0)
    diag = sums * sums

    if n_cols == 1:
        cofactors = np.repeat([[[None]]], n_rows, axis=0)
    else:
        # Per-row outer products → drop diagonal → shape (n_rows, n_cols-1, n_cols),
        # matching the layout produced by `_join_aug_sketches` so that
        # `_join_aug_table` consumes them via its numeric-dtype branch.
        outer = np.einsum('bi,bj->bij', sums, sums)
        mask = ~np.eye(n_cols, dtype=bool)
        cofactors = outer[:, mask].reshape(n_rows, n_cols, n_cols - 1)
        cofactors = cofactors.transpose(0, 2, 1)

    aug_feature_indices = None
    if column_names is not None:
        aug_feature_indices = {i: name for i, name in enumerate(column_names)}

    return AugSketch(
        table_feature_index=table_feature_index,
        count=counts,
        sum_=sums,
        diag=diag,
        cofactors=cofactors,
        aug_feature_indices=aug_feature_indices,
        base_column=base_column,
    )


def expand_with_polynomials(base_sum_col: np.ndarray, base_name: str,
                            degree: int, table_feature_index: str,
                            base_column: str | None = None) -> AugSketch:
    """Expand a single base column into a grouped AugSketch containing
    `[x, x^2, ..., x^degree]`. Powers are computed elementwise from per-row
    `sum_` values (which carry raw cell values after the upstream split). The
    base column and its powers form one atomic group so forward selection
    evaluates and emits them together (a power is never selected without its
    base).

    Args:
        base_sum_col: shape (n_rows, 1) — per-row values of the base column.
        base_name: identifier of the base column (used for synthetic names).
        degree: highest power to include (>= 2). Group size will be `degree`.
        table_feature_index: unique cache key for the resulting group sketch.
        base_column: qualified source-column identifier for sibling dedup.

    Returns:
        AugSketch with n_cols == degree. First column is the original base.
    """
    if degree < 2:
        raise ValueError(f'expand_with_polynomials requires degree >= 2, got {degree}')
    if base_sum_col.ndim != 2 or base_sum_col.shape[1] != 1:
        raise ValueError(f'expand_with_polynomials expects shape (n_rows, 1), got {base_sum_col.shape}')

    x = base_sum_col[:, 0]
    cols = [x ** k for k in range(1, degree + 1)]
    sums = np.stack(cols, axis=1)
    names = [base_name] + [f'{base_name}^{k}' for k in range(2, degree + 1)]
    return _build_group_aug_sketch(sums, table_feature_index, column_names=names,
                                   base_column=base_column)


def quantile_bin_thresholds(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute ``n_bins - 1`` interior bin edges from quantiles of ``values``
    so that each bin holds approximately the same number of observations.
    Duplicates produced when many values collapse onto the same quantile are
    removed so the bin count may shrink below ``n_bins``. Returned edges are
    strictly increasing.
    """
    if n_bins < 2:
        raise ValueError(f'quantile_bin_thresholds requires n_bins >= 2, got {n_bins}')
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # interior edges
    edges = np.quantile(values, quantiles)
    # Drop duplicates so collapsed bins do not produce zero-width buckets.
    edges = np.unique(edges)
    return edges


def expand_with_quantile_bins(base_sum_col: np.ndarray, base_name: str,
                              n_bins: int, table_feature_index: str,
                              base_column: str | None = None
                              ) -> tuple[AugSketch | None, np.ndarray]:
    """Dummify a per-key aggregated column into one indicator per quantile bin.

    The forward selection proxy operates on per-key aggregated statistics, so
    the levels we dummify are bins of the per-key SUM rather than raw row
    values. We compute ``n_bins`` quantile bins of ``base_sum_col`` and emit
    one indicator column per bin. The indicators are mutually exclusive per
    key by construction, so the outer-product cofactors produced inside
    ``_build_group_aug_sketch`` are zero off the diagonal (which is the
    correct linear-in-dummies algebra for this approximation).

    Args:
        base_sum_col: shape ``(K, 1)`` per-key aggregated values of the
            source candidate.
        base_name: identifier of the source candidate; used as the prefix of
            the emitted bin names.
        n_bins: target number of quantile bins. Collapsed quantiles (when
            many keys share the same value) reduce the effective bin count.
        table_feature_index: unique cache key for the resulting group.
        base_column: qualified source-column identifier for sibling dedup.

    Returns:
        ``(sketch, edges)`` where ``sketch`` is the AugSketch group (one
        column per effective bin) or ``None`` if the column collapses to a
        single bin (e.g. all values equal). ``edges`` is the (n_bins - 1,)
        array of interior bin edges retained for materialization.
    """
    if base_sum_col.ndim != 2 or base_sum_col.shape[1] != 1:
        raise ValueError(f'expand_with_quantile_bins expects shape (K, 1), got {base_sum_col.shape}')
    x = base_sum_col[:, 0]
    edges = quantile_bin_thresholds(x, n_bins)
    if len(edges) == 0:
        # All values identical or quantiles collapsed; no useful split.
        return None, edges
    # Right-edge inclusive bucket assignment so the highest value lands in
    # the top bin instead of an out-of-range bucket.
    bin_idx = np.searchsorted(edges, x, side='right')
    effective_bins = int(bin_idx.max()) + 1
    # One indicator column per occupied bin. Empty bins (no keys assigned)
    # are dropped so the proxy does not waste a candidate slot.
    occupied = np.unique(bin_idx)
    sums = np.zeros((x.shape[0], len(occupied)), dtype=np.float64)
    for col, b in enumerate(occupied):
        sums[:, col] = (bin_idx == b).astype(np.float64)
    names = [f'{base_name}=bin_{int(b)}' for b in occupied]
    sketch = _build_group_aug_sketch(sums, table_feature_index,
                                     column_names=names, base_column=base_column)
    return sketch, edges


def build_interaction_sketch(sum_a: np.ndarray, sum_b: np.ndarray,
                             name_a: str, name_b: str,
                             table_feature_index: str) -> AugSketch:
    """Build a singleton-group AugSketch for the pairwise interaction `a * b`.
    Treated as a stand-alone candidate; not coupled to either parent column.
    """
    if sum_a.shape != sum_b.shape or sum_a.ndim != 2 or sum_a.shape[1] != 1:
        raise ValueError('build_interaction_sketch requires matching (n_rows, 1) inputs')
    interaction = sum_a * sum_b
    name = f'{name_a}_mul_{name_b}'
    return _build_group_aug_sketch(interaction, table_feature_index, column_names=[name])


def _pad_linreg_statistics_pure(cofactor_matrix, feature_target_vector, m, shifted=False, offset=None):
    """Pure (non-method) version of _pad_linreg_statistics for use outside the actor."""
    padded_matrix = np.pad(cofactor_matrix, pad_width=((0, m), (0, m)), mode='constant', constant_values=0)
    padded_vector = np.pad(feature_target_vector, pad_width=((0, m), (0, 0)), mode='constant', constant_values=0)
    if shifted:
        new_feature_range = list(range(-m, 0))
        test_feature_range = list(range(-offset, -m))
        current_indexing = test_feature_range + new_feature_range
        new_indexing = new_feature_range + test_feature_range
        padded_matrix[current_indexing, :] = padded_matrix[new_indexing, :]
        padded_matrix[:, current_indexing] = padded_matrix[:, new_indexing]
        padded_vector[current_indexing, :] = padded_vector[new_indexing, :]
    return padded_matrix, padded_vector


def _gram_from_cofactor_upper(user_table_agg: pl.DataFrame) -> np.ndarray:
    """Reconstruct the per-key Gram tensor from the flat upper-triangular
    cofactor column produced by ``JoinSelection._sketch_user_table``.

    Each row of ``cofactor_upper`` stores the P*(P+1)/2 upper-triangular
    entries of sum_{i in key k} v_i v_i^T, where v_i is the per-row vector
    of (target, base features) used as input to the per-key aggregation.
    The output has shape (K, P, P) and is symmetric per slice.
    """
    flat = np.asarray(
        user_table_agg.select('cofactor_upper').to_series().to_list(),
        dtype=np.float64,
    )
    K, n_tri = flat.shape
    P = int(round((-1 + np.sqrt(1 + 8 * n_tri)) / 2))
    if P * (P + 1) // 2 != n_tri:
        raise ValueError(
            f'cofactor_upper has {n_tri} entries per key, which is not a '
            f'triangular number; cannot infer a square Gram dimension.'
        )
    iu = np.triu_indices(P)
    gram = np.zeros((K, P, P), dtype=np.float64)
    gram[:, iu[0], iu[1]] = flat
    off = iu[0] != iu[1]
    gram[:, iu[1][off], iu[0][off]] = flat[:, off]
    return gram


def _standardize_gram_inputs_pure(cofactor_matrix, feature_target_vector, features_sum, target_sum, joint_count):
    """Pure (non-method) version of _standardize_gram_inputs for use outside the actor."""
    feature_target_vector = feature_target_vector.reshape(-1)
    target_mean = target_sum / joint_count
    cofactor_centered = cofactor_matrix - np.outer(features_sum, features_sum) / joint_count
    feature_target_centered = feature_target_vector - (features_sum * target_mean)
    feature_vars = np.diag(cofactor_centered) / joint_count
    feature_vars = np.clip(feature_vars, a_min=0.0, a_max=None)
    feature_stds = np.sqrt(feature_vars)
    feature_stds[feature_stds == 0] = 1.0
    D_inv = 1.0 / feature_stds
    cofactor_std = (D_inv[:, None] * cofactor_centered) * D_inv[None, :]
    feature_target_std = (D_inv * feature_target_centered).reshape(-1, 1)
    return np.array(cofactor_std), np.array(feature_target_std)


def _compute_feature_update(shared_state, feature_entry, task):
    """Pure function: compute the cache update + fs_output for one feature.
    
    Args:
        shared_state: dict with keys: new_feature, joint_cofactor_matrix,
            joint_feature_target_vector, joint_count, y_t_y, joint_sum_vec
        feature_entry: dict (one entry from self.cache[idx])
        task: 'regression' or 'classification'
    
    Returns:
        (cache_update_dict, fs_output_dict)
    """
    base_sum = shared_state['new_feature']
    joint_count = shared_state['joint_count']
    y_t_y = shared_state['y_t_y']
    aug_sum = feature_entry['initial_feature']

    joint_cofactor_matrix = shared_state['joint_cofactor_matrix']
    init_rows, init_cols = joint_cofactor_matrix.shape
    joint_feature_target_vector = shared_state['joint_feature_target_vector']
    new_features_count = aug_sum.shape[1]
    joint_cofactor_matrix, joint_feature_target_vector = _pad_linreg_statistics_pure(
        joint_cofactor_matrix, joint_feature_target_vector, new_features_count
    )

    joint_cofactors = base_sum[:, :, np.newaxis] * aug_sum[:, np.newaxis, :]
    joint_cofactors_vec = joint_cofactors.sum(axis=0)
    aug_cofactor_matrix = feature_entry['joint_cofactor_matrix'].copy()
    aug_init_rows, aug_init_cols = aug_cofactor_matrix.shape
    aug_feature_target_vector = feature_entry['joint_feature_target_vector'][-new_features_count:, :].copy()
    aug_cofactor_matrix[:aug_init_rows - new_features_count, :aug_init_cols - new_features_count] = 0
    feature_diff = joint_cofactor_matrix.shape[1] - aug_cofactor_matrix.shape[1]
    aug_cofactor_matrix, _ = _pad_linreg_statistics_pure(
        aug_cofactor_matrix, aug_feature_target_vector,
        m=feature_diff, shifted=True, offset=feature_diff + aug_sum.shape[1]
    )

    joint_cofactor_matrix = joint_cofactor_matrix + aug_cofactor_matrix

    joint_cofactor_matrix_covar_idx = init_cols - base_sum.shape[1]
    joint_cofactor_matrix_covar_part = joint_cofactor_matrix[joint_cofactor_matrix_covar_idx:, joint_cofactor_matrix_covar_idx:]
    pad_rows = joint_cofactor_matrix_covar_part.shape[0] - joint_cofactors_vec.shape[0]
    pad_cols = joint_cofactor_matrix_covar_part.shape[1] - joint_cofactors_vec.shape[1]
    joint_cofactors_vec = np.pad(joint_cofactors_vec, pad_width=((0, pad_rows), (pad_cols, 0)), mode='constant', constant_values=0)
    joint_cofactors_vec = joint_cofactors_vec + joint_cofactors_vec.T
    joint_cofactor_matrix[joint_cofactor_matrix_covar_idx:, joint_cofactor_matrix_covar_idx:] = joint_cofactor_matrix_covar_part + joint_cofactors_vec

    aug_sum_scaled = feature_entry['new_feature']
    aug_sum_vec = aug_sum_scaled.sum(axis=0)
    joint_cofactor_matrix[init_rows:, 0] = aug_sum_vec
    joint_cofactor_matrix[0, init_cols:] = aug_sum_vec

    joint_feature_target_vector[init_rows:, :] = aug_feature_target_vector

    joint_sum_vec = np.hstack([shared_state['joint_sum_vec'], aug_sum_vec])
    base_target_sum_vec = feature_entry['base_target_sum_vec']

    intercept_idx = 0
    all_indices = np.arange(joint_cofactor_matrix.shape[1])
    complement = np.setdiff1d(all_indices, [intercept_idx])
    joint_cofactor_matrix_std, joint_feature_target_vector_std = _standardize_gram_inputs_pure(
        joint_cofactor_matrix[np.ix_(complement, complement)],
        joint_feature_target_vector[complement],
        joint_sum_vec, base_target_sum_vec, joint_count
    )

    cache_update = {
        'new_feature': aug_sum_scaled,
        'joint_count': joint_count,
        'joint_cofactor_matrix': joint_cofactor_matrix.copy(),
        'joint_feature_target_vector': joint_feature_target_vector.copy(),
        'y_t_y': y_t_y,
        'joint_sum_vec': joint_sum_vec,
        'base_target_sum_vec': base_target_sum_vec
    }

    if task == 'regression':
        fs_output = {
            'joint_cofactor_matrix': joint_cofactor_matrix_std,
            'joint_feature_target_vector': joint_feature_target_vector_std,
            'joint_count': joint_count,
            'y_t_y': y_t_y,
            'base_target_sum_vec': base_target_sum_vec,
            # Candidate block size (trailing n_Z columns of the joint Gram),
            # needed by the conditional-GCV proxy to split the joint into the
            # current (X) and candidate (Z) blocks.
            'new_features_count': new_features_count,
        }
    elif task == 'classification':
        features_mean = joint_sum_vec / joint_count
        fs_output = {
            'joint_cofactor_matrix': joint_cofactor_matrix_std,
            'features_mean': features_mean,
            'joint_count': joint_count,
            'new_features_count': new_features_count
        }

    return cache_update, fs_output


def _filter_sums_diags_pure(table: pl.DataFrame) -> pl.DataFrame:
    """Stateless version of SketchProcessor._filter_sums_diags."""
    table = (
        table
            .with_columns(
                [
                    (
                        (pl.col('sum') / pl.col('drop_feature'))
                            .alias('sum')
                            .list.filter(
                                (pl.element().eq(float('inf')).not_())
                                &
                                (pl.element().eq(float('-inf')).not_())
                                &
                                (pl.element().is_nan().not_())
                            )
                    ),
                    (
                        (pl.col('diag') / pl.col('drop_feature'))
                            .alias('diag')
                            .list.filter(
                                (pl.element().eq(float('inf')).not_())
                                &
                                (pl.element().is_nan().not_())
                            )
                    )
                ]
            )
    )
    return table


def _filter_cofactors_pure(cofactors, indexes_to_keep):
    """Stateless version of SketchProcessor._filter_cofactors."""
    cofactors = np.array(cofactors)
    n = cofactors.shape[2]
    square = np.zeros((cofactors.shape[0], n, n), dtype=cofactors.dtype)
    square[:, :cofactors.shape[1], :cofactors.shape[2]] = cofactors
    i, j = np.triu_indices(square.shape[-1], k=1)
    square[..., j, i] = square[..., i, j]
    square = np.triu(square, k=1) + np.tril(square, k=-1)
    indexes_to_keep = np.where(np.array(indexes_to_keep) == 1)[0].tolist()
    cofactors, _ = extract_cofactors(square, indexes_to_keep)
    return cofactors


def _fetch_cofactors_pure(table, conninfo, feature_selection_table_name):
    """Stateless version of SketchProcessor._fetch_cofactors."""
    row = table.row(0)
    table_index = row[2]
    key_col_index = row[3]
    row_indices = table.select('row_index').to_series().to_list()
    query = f'SELECT cofactors, shape FROM {feature_selection_table_name} WHERE table_index = {table_index} AND key_col_index = {key_col_index} AND row_index IN ({", ".join(map(str, row_indices))})'
    conn = adbc_dbapi.connect(conninfo)
    with conn.cursor() as cursor:
        cursor.execute(query)
        cofactors_df = cursor.fetch_polars()
    return cofactors_df


def _deserialize_cofactors_pure(table, conninfo, feature_selection_table_name):
    """Stateless version of SketchProcessor._deserialize_cofactors."""
    cofactors_df = _fetch_cofactors_pure(table, conninfo, feature_selection_table_name).to_numpy()
    decoded_cofactors = [
        np.frombuffer(cofactor_bytes, dtype=np.float64).reshape(shape)
        for cofactor_bytes, shape in cofactors_df
    ]
    decoded_cofactors = np.vstack(np.array(decoded_cofactors)[np.newaxis, ...])
    return decoded_cofactors


def _impute_sketch_table_pure(user_table_agg, sketch_table, query_column_name, target_cofactor_shape=None):
    """Stateless version of SketchProcessor._impute_sketch_table."""
    all_ids = set(user_table_agg[query_column_name].to_list())
    group_ids = set(sketch_table['key'].to_list())
    missing_ids = all_ids - group_ids
    padding_data = {
        col: [None] * len(missing_ids) for col in sketch_table.columns
    }
    padding_data.update({'key': list(missing_ids)})
    padded_rows = pl.DataFrame(padding_data)
    list_len = sketch_table.select(pl.col('sum').list.len()).row(0)[0]
    sums = (
        sketch_table.select(
            pl.concat_list(
                pl.col('sum')
                .list.to_struct(upper_bound=list_len)
                .struct.unnest()
                .mean()
            )
        )
        .to_numpy()
        .item()
        .reshape(1, -1)
    )

    idx = np.arange(sums.shape[1])
    outer_products = np.outer(sums, sums)[np.newaxis, ...]
    cofactors, diag = extract_cofactors(outer_products, idx)

    # Normalize to shape (1, n-1, n) so that `np.repeat(..., axis=0)` yields
    # (padded_rows.height, n-1, n), matching the per-row shape of decoded_cofactors.
    if cofactors.ndim == 2:
        row_cofactors = cofactors[np.newaxis, ...]
    else:
        row_cofactors = cofactors

    # The indexer may have stored a per-row shape different from what extract_cofactors
    # produces here (notably for single-column groups, where it stores a (1, 1) NaN
    # placeholder instead of a (0, 1) empty matrix). Match the target shape so that
    # padded_cofactors[i] aligns with decoded_cofactors[i].
    if target_cofactor_shape is not None and tuple(row_cofactors.shape[1:]) != tuple(target_cofactor_shape):
        row_cofactors = np.full((1, *target_cofactor_shape), np.nan, dtype=np.float64)
    elif 'shape' in sketch_table.columns and sketch_table.height > 0:
        stored_shape = tuple(sketch_table.select(pl.col('shape')).row(0)[0])
        if tuple(row_cofactors.shape[1:]) != stored_shape:
            row_cofactors = np.full((1, *stored_shape), np.nan, dtype=np.float64)

    sums_series = pl.Series(sums, dtype=pl.List(pl.Float64))
    diag_series = pl.Series(diag, dtype=pl.List(pl.Float64))
    # row_cofactors has shape (1, n-1, n) and must match the per-row shape of decoded_cofactors.
    # Broadcast it to (padded_rows.height, n-1, n) so padded_cofactors[idx] is (n-1, n).
    cofactors = np.repeat(row_cofactors, repeats=padded_rows.height, axis=0)
    padded_rows = padded_rows.with_columns(
        pl.col('count').fill_null(1).cast(pl.Int32),
        pl.col('sum').fill_null(sums_series),
        pl.col('diag').fill_null(diag_series),
        pl.col('drop_feature').fill_null(sketch_table.select(pl.col('drop_feature')).row(0)[0]) if 'drop_feature' in sketch_table.columns else pl.lit(None)
    )

    return sketch_table, padded_rows, cofactors


@ray.remote
def _create_aug_table_sketch_task(group, user_table_agg, query_column_name,
                                  conninfo, feature_selection_table_name,
                                  var_threshold=None, corr_threshold=None):
    """Stateless remote function: builds an AugSketch for one candidate group.
    Runs as an independent Ray task (not on the actor), enabling true parallelism."""
    table = group[1]
    # Capture the per-column headers BEFORE any filtering so we can populate
    # `aug_feature_indices` on the joint AugSketch. Headers carry the raw
    # column names like `<base>_<aggregation>` which downstream code uses to
    # dedup candidates that are different aggregations of the same column.
    raw_column_headers = None
    if 'column_headers' in table.columns and table.height > 0:
        raw_column_headers = table['column_headers'][0].to_list()
    # Fetch cofactors first so we know the actual per-row shape stored in the DB; this
    # shape must be propagated to padding to avoid ragged arrays downstream.
    decoded_cofactors = _deserialize_cofactors_pure(table, conninfo, feature_selection_table_name)
    target_cofactor_shape = decoded_cofactors.shape[1:] if decoded_cofactors.ndim >= 2 else None
    if table.height < user_table_agg.height:
        table, padding, padded_cofactors = _impute_sketch_table_pure(
            user_table_agg, table, query_column_name, target_cofactor_shape=target_cofactor_shape
        )
        if var_threshold is not None or corr_threshold is not None:
            table = _filter_sums_diags_pure(table)
            padding = _filter_sums_diags_pure(padding)
    else:
        padding = pl.DataFrame({'key': [None]})
        if var_threshold is not None or corr_threshold is not None:
            table = _filter_sums_diags_pure(table)
    table_rows = table.with_row_index().partition_by('key', as_dict=True)
    padding_rows = padding.with_row_index().partition_by('key', as_dict=True)
    counts = []
    sums = []
    diags = []
    cofactors = []
    for row in user_table_agg.iter_rows(named=True):
        key = row[query_column_name]
        cand_table = table_rows.get((key,), None)
        cand_padding = padding_rows.get((key,), None)
        if cand_table is not None:
            idx = cand_table[0]['index'][0]
            counts.append(table.row(idx)[5])
            sums.append(table.row(idx)[6])
            diags.append(table.row(idx)[7])
            cofactors.append(decoded_cofactors[idx])
        elif cand_padding is not None:
            idx = cand_padding[0]['index'][0]
            counts.append(padding.row(idx)[5])
            sums.append(padding.row(idx)[6])
            diags.append(padding.row(idx)[7])
            cofactors.append(padded_cofactors[idx])

    key_column_index = group[0][2]
    feature_index = f'{key_column_index}_{group[0][1]}'

    sums = np.array(sums)
    counts = np.array(counts).reshape(-1, 1)
    diags = np.array(diags)
    if var_threshold is not None or corr_threshold is not None:
        indexes_to_keep = table[0].select(pl.col('drop_feature')).to_series().to_list()[0]
        cofactors = _filter_cofactors_pure(cofactors, indexes_to_keep)
        if raw_column_headers is not None and len(raw_column_headers) == len(indexes_to_keep):
            raw_column_headers = [h for h, k in zip(raw_column_headers, indexes_to_keep) if k == 1]
    aug_feature_indices = (
        {i: h for i, h in enumerate(raw_column_headers)}
        if raw_column_headers is not None and len(raw_column_headers) == sums.shape[1]
        else None
    )
    aug_sketch = AugSketch(feature_index, counts, sums, diags, cofactors,
                           aug_feature_indices=aug_feature_indices)

    return aug_sketch


@ray.remote
def _create_aug_table_sketch_clf_task(group, user_table_agg, query_column_name,
                                      target_column_name, conninfo,
                                      feature_selection_table_name,
                                      var_threshold=None, corr_threshold=None):
    """Stateless remote function: builds per-class AugSketches for one candidate group.
    Runs as an independent Ray task, enabling true parallelism."""
    table = group[1]
    # Capture the per-column headers BEFORE any filtering so we can populate
    # `aug_feature_indices` on the per-class joint AugSketches.
    raw_column_headers = None
    if 'column_headers' in table.columns and table.height > 0:
        raw_column_headers = table['column_headers'][0].to_list()
    # Fetch cofactors first so we know the actual per-row shape stored in the DB; this
    # shape must be propagated to padding to avoid ragged arrays downstream.
    decoded_cofactors = _deserialize_cofactors_pure(table, conninfo, feature_selection_table_name)
    target_cofactor_shape = decoded_cofactors.shape[1:] if decoded_cofactors.ndim >= 2 else None
    if table.height < user_table_agg.height:
        table, padding, padded_cofactors = _impute_sketch_table_pure(
            user_table_agg, table, query_column_name, target_cofactor_shape=target_cofactor_shape
        )
        if var_threshold is not None or corr_threshold is not None:
            table = _filter_sums_diags_pure(table)
            padding = _filter_sums_diags_pure(padding)
    else:
        # No missing keys: empty padding with the same key-column schema so that
        # pl.concat([table.select('key'), padding.select('key')]) has height == user_table_agg.height.
        padding = pl.DataFrame({'key': []}, schema={'key': table.schema['key']})
        padded_cofactors = None
        if var_threshold is not None or corr_threshold is not None:
            table = _filter_sums_diags_pure(table)
    key_column_index = group[0][2]
    feature_index = f'{key_column_index}_{group[0][1]}'

    table_rows = table.with_row_index().partition_by('key', as_dict=True)
    padding_rows = padding.with_row_index().partition_by('key', as_dict=True)

    local_dict = defaultdict(list)
    # Join target values by key (robust to any height mismatch / order differences
    # between `table`, `padding`, and `user_table_agg`).
    targets_per_key = (
        user_table_agg
            .group_by(query_column_name, maintain_order=True)
            .agg(target_column_name)
            .rename({query_column_name: 'key'})
            .select(['key', target_column_name])
    )
    keyed = (
        pl.concat([table.select('key'), padding.select('key')], how='vertical')
            .unique(subset=['key'], maintain_order=True)
            .join(targets_per_key, on='key', how='inner')
            .explode(target_column_name)
            .sort([target_column_name, 'key'])
            .group_by(target_column_name, maintain_order=True)
    )
    for target_class, sub_table in keyed:
        counts = []
        sums = []
        diags = []
        cofactors = []
        for row in sub_table.iter_rows(named=True):
            key = row['key']
            cand_table = table_rows.get((key,), None)
            cand_padding = padding_rows.get((key,), None)
            if cand_table is not None:
                idx = cand_table[0]['index'][0]
                counts.append(table.row(idx)[5])
                sums.append(table.row(idx)[6])
                diags.append(table.row(idx)[7])
                cofactors.append(decoded_cofactors[idx])
            elif cand_padding is not None:
                idx = cand_padding[0]['index'][0]
                counts.append(padding.row(idx)[5])
                sums.append(padding.row(idx)[6])
                diags.append(padding.row(idx)[7])
                cofactors.append(padded_cofactors[idx])

        sums = np.array(sums)
        counts = np.array(counts).reshape(-1, 1)
        diags = np.array(diags)
        cofactors = np.array(cofactors)
        if var_threshold is not None or corr_threshold is not None:
            indexes_to_keep = table[0].select(pl.col('drop_feature')).to_series().to_list()[0]
            cofactors = _filter_cofactors_pure(cofactors, indexes_to_keep)
            class_headers = (
                [h for h, k in zip(raw_column_headers, indexes_to_keep) if k == 1]
                if raw_column_headers is not None and len(raw_column_headers) == len(indexes_to_keep)
                else None
            )
        else:
            class_headers = raw_column_headers
        aug_feature_indices = (
            {i: h for i, h in enumerate(class_headers)}
            if class_headers is not None and sums.ndim == 2 and len(class_headers) == sums.shape[1]
            else None
        )
        aug_sketch = AugSketch(feature_index, counts, sums, diags, cofactors,
                               aug_feature_indices=aug_feature_indices)
        local_dict[target_class[0]].append(aug_sketch)

    return local_dict


@ray.remote
def _compute_feature_update_batch(shared_state, feature_entries, task):
    """Process a batch of features in a single remote call.

    Args:
        shared_state: dict with shared actor state (via ray.put)
        feature_entries: list of (table_feature_index, cache_entry) pairs
        task: 'regression' or 'classification'

    Returns:
        (list of (table_feature_index, cache_update, fs_output), compute_s)
    """
    import time as _time
    t0 = _time.perf_counter()
    results = []
    for idx, entry in feature_entries:
        cache_update, fs_output = _compute_feature_update(shared_state, entry, task)
        results.append((idx, cache_update, fs_output))
    return results, _time.perf_counter() - t0


@ray.remote(num_cpus=0)
class SketchProcessor:
    def __init__(self, conninfo: str = None, feature_selection_table_name: str = None, corr_threshold: float = None, var_threshold: float = None):
        self.conninfo = conninfo
        self.feature_selection_table_name = feature_selection_table_name
        self.cache = {}
        self.fs_output = {}
        self.init_features_counter: int = None
        self.new_feature: np.ndarray = None
        self.joint_cofactor_matrix: np.ndarray = None
        self.joint_feature_target_vector: np.ndarray = None
        self.joint_count: int = None
        self.y_t_y: np.ndarray = None
        self.M1: np.ndarray = None
        self.M2: np.ndarray = None
        self.d: np.ndarray = None
        is_patched()


    def _create_base_table_sketch(self, user_table_agg: pl.DataFrame, target_as_feature: bool = False) -> BaseSketch:
        base_cofactors, _ = compute_outer_products(np.array(user_table_agg.select('sum').to_series().to_list()))
        base_sum = np.array(user_table_agg.select('sum').to_series().to_list())
        base_count = np.array(user_table_agg.select('count').to_series().to_list()).reshape(-1, 1)

        # True per-key Gram block sum_{i in k} v_i v_i^T over v_i = [y_i, x_i],
        # reconstructed from the upper-triangular cofactor list materialised by
        # `_sketch_user_table`. Stored alongside the legacy `s_k s_k^T` block
        # so that `_evaluate_base_only_score` can fit an OLS proxy on the
        # actual base table while the joint reconstruction path keeps using
        # `features_cofactors` / `target_cofactors` with their legacy semantics.
        per_key_gram = _gram_from_cofactor_upper(user_table_agg)
        if target_as_feature:
            features_gram = per_key_gram
        else:
            features_gram = per_key_gram[:, 1:, 1:]
        target_gram_row = per_key_gram[:, :, 0].reshape(per_key_gram.shape[0], per_key_gram.shape[1], 1)

        if target_as_feature:
            base_table_sketch = BaseSketch(
                base_count,
                features_sum=base_sum,
                target_sum=base_sum[:, 0].reshape(-1, 1),
                features_cofactors=base_cofactors,
                target_cofactors=base_cofactors[:, :, 0].reshape(-1, base_cofactors.shape[1], 1),
                initial_frame=user_table_agg,
                features_gram=features_gram,
                target_gram_row=target_gram_row,
            )
        else:
            base_table_sketch = BaseSketch(
                base_count,
                features_sum=base_sum[:, 1:],
                target_sum=base_sum[:, 0].reshape(-1, 1),
                features_cofactors=base_cofactors[:, 1:, 1:],
                target_cofactors=base_cofactors[:, :, 0].reshape(-1, base_cofactors.shape[1], 1),
                initial_frame=user_table_agg,
                features_gram=features_gram,
                target_gram_row=target_gram_row,
            )

        return base_table_sketch


    def _check_square_matrix(self, mul_: list):
        mat = np.array(mul_)
        return len(mat.shape) == 2


    def _create_base_table_sketch_clf(self, sub_table: pl.DataFrame, query_column_name: str, target_column_name: str, label: tuple[int]) -> BaseSketch:
        local_dict = defaultdict(list)
        sub_table = sub_table.sort(query_column_name)

        base_cofactors, _ = compute_outer_products(np.array(sub_table.select('sum').to_series().to_list()))
        base_sum = np.array(sub_table.select('sum').to_series().to_list())
        base_count = np.array(sub_table.select('count').to_series().to_list()).reshape(-1, 1)

        base_table_sketch = BaseSketch(
            base_count,
            features_sum=base_sum,
            target_sum=base_sum[:, 0].reshape(-1, 1),
            features_cofactors=base_cofactors,
            target_cofactors=base_cofactors[:, :, 0].reshape(-1, base_cofactors.shape[1], 1)
        )
        local_dict[label[0]] = base_table_sketch

        return local_dict


    def _join_aug_sketches(self, aug_tables_sketches: list[AugSketch], n_init_features: int, label: int = None):
        feature_index = {}
        aug_sums = []
        for aug_table_sketch in aug_tables_sketches:
            aug_sums.append(aug_table_sketch.sum_)
            vals = np.arange(aug_table_sketch.sum_.shape[1])
            keys = vals + len(feature_index) + n_init_features
            vals = [f'{"_".join(aug_table_sketch.table_feature_index.split("_")[:-1] + [str(v)])}' for v in vals]
            # map index of joint cofactor matrix column to inverted index
            feature_index.update(dict(zip(keys, vals)))
        try:
            if len(aug_sums) == 0:
                raise EmptyAugmentation()
        except EmptyAugmentation as e:
            raise RuntimeError(str(e))

        joint_sum = np.concatenate(aug_sums, axis=1)
        joint_count = aug_tables_sketches[0].count
        outer_prod, _ = compute_outer_products(joint_sum)
        joint_diag = np.vstack([np.diag(outer_prod[i]) for i in range(outer_prod.shape[0])])

        triu = np.triu(outer_prod, k=1)
        tril = np.tril(outer_prod, k=-1)
        cofactors = triu+tril
        batch, n, _ = cofactors.shape
        diag_mask = ~np.eye(n, dtype=bool)
        diag_mask = np.broadcast_to(diag_mask, (batch, n, n))
        cofactors = cofactors[diag_mask].reshape(batch, n, -1)
        cofactors = cofactors.transpose(0, 2, 1)
        cofactors = cofactors.squeeze()
        if np.isnan(cofactors).all():
            joint_cofactors = np.expand_dims(cofactors, 1)
        else:
            if len(cofactors.shape) == 2:
                joint_cofactors = np.expand_dims(cofactors, 2)
            else:
                joint_cofactors = cofactors

        joint_sketch = AugSketch('joint_feature', joint_count, joint_sum, joint_diag, joint_cofactors, feature_index)

        if label is not None:
            return {label: joint_sketch}

        return joint_sketch


    def _split_aug_table_sketch(self, sums, diags, feature_indices):
        counts = np.repeat([[1]], sums.shape[0], axis=0)
        cofactors = np.repeat([[[None]]], sums.shape[0], axis=0)

        sketch = AugSketch(
            sum_=sums,
            diag=diags,
            table_feature_index=feature_indices,
            count=counts,
            cofactors=cofactors
        )

        return sketch


    def _prepare_split_inputs(self, aug_table_sketch: AugSketch):
        split_sums = np.split(aug_table_sketch.sum_, aug_table_sketch.sum_.shape[1], axis=1)
        split_diags = np.split(aug_table_sketch.diag, aug_table_sketch.diag.shape[1], axis=1)
        split_feature_indices = [
            '_'.join(aug_table_sketch.table_feature_index.split('_')[:-1] + [str(i)])
            for i in range(len(split_sums))
        ]
        # Per-column raw header names (e.g. `<base>_<aggregation>`) carried
        # from the DB through the joint AugSketch's `aug_feature_indices`.
        # ``None`` slots are tolerated downstream (no dedup for that column).
        afi = aug_table_sketch.aug_feature_indices or {}
        split_column_names = [afi.get(i) for i in range(len(split_sums))]
        return split_sums, split_diags, split_feature_indices, split_column_names


    def _filter_sums_diags(self, table: pl.DataFrame):
        table = (
            table
                .with_columns(
                    [
                        (
                            (pl.col('sum') / pl.col('drop_feature'))
                                .alias('sum')
                                .list.filter(
                                    (pl.element().eq(float('inf')).not_())
                                    &
                                    (pl.element().eq(float('-inf')).not_())
                                    &
                                    (pl.element().is_nan().not_())
                                )
                        ),
                        (
                            (pl.col('diag') / pl.col('drop_feature'))
                                .alias('diag')
                                .list.filter(
                                    (pl.element().eq(float('inf')).not_())
                                    &
                                    (pl.element().is_nan().not_())
                                )

                        )
                    ]
                )
        )
        return table


    def _filter_cofactors(self, cofactors, indexes_to_keep):
        cofactors = np.array(cofactors)
        n = cofactors.shape[2]
        square = np.zeros((cofactors.shape[0], n, n), dtype=cofactors.dtype)
        square[:, :cofactors.shape[1], :cofactors.shape[2]] = cofactors
        i, j = np.triu_indices(square.shape[-1], k=1)
        square[..., j, i] = square[..., i, j]
        square = np.triu(square, k=1)+np.tril(square, k=-1)
        indexes_to_keep = np.where(np.array(indexes_to_keep) == 1)[0].tolist()
        cofactors, _ = extract_cofactors(square, indexes_to_keep)

        return cofactors


    def _deserialize_cofactors(self, table: pl.DataFrame) -> pl.DataFrame:
        cofactors_df = self._fetch_cofactors(table).to_numpy()
        decoded_cofactors = [
            np.frombuffer(cofactor_bytes, dtype=np.float64).reshape(shape)
            for cofactor_bytes, shape in cofactors_df
        ]
        decoded_cofactors = np.vstack(np.array(decoded_cofactors)[np.newaxis, ...])

        return decoded_cofactors


    def _fetch_cofactors(self, table):
        row = table.row(0)
        table_index = row[2]
        key_col_index = row[3]
        row_indices = table.select('row_index').to_series().to_list()
        query = f'SELECT cofactors, shape FROM {self.feature_selection_table_name} WHERE table_index = {table_index} AND key_col_index = {key_col_index} AND row_index IN ({", ".join(map(str, row_indices))})'
        conn = adbc_dbapi.connect(self.conninfo)
        with conn.cursor() as cursor:
            cursor.execute(query)
            cofactors_df = cursor.fetch_polars()
        return cofactors_df


    def _create_aug_table_sketch(self, group: pl.DataFrame, user_table_agg: pl.DataFrame, query_column_name: str, var_threshold: float = None, corr_threshold: float = None) -> AugSketch:
        table = group[1]
        if table.height < user_table_agg.height:
            table, padding, padded_cofactors = self._impute_sketch_table(user_table_agg, table, query_column_name)
            if var_threshold is not None or corr_threshold is not None:
                table = self._filter_sums_diags(table)
                padding = self._filter_sums_diags(padding)
        else:
            padding = pl.DataFrame({'key': [None]})
            if var_threshold is not None or corr_threshold is not None:
                table = self._filter_sums_diags(table)

        decoded_cofactors = self._deserialize_cofactors(table)
        table_rows = table.with_row_index().partition_by('key', as_dict=True)
        padding_rows = padding.with_row_index().partition_by('key', as_dict=True)
        counts = []
        sums = []
        diags = []
        cofactors = []
        for row in user_table_agg.iter_rows(named=True):
            key = row[query_column_name]
            cand_table = table_rows.get((key,), None)
            cand_padding = padding_rows.get((key,), None)
            if cand_table is not None:
                idx = cand_table[0]['index'][0]
                counts.append(table.row(idx)[5])
                sums.append(table.row(idx)[6])
                diags.append(table.row(idx)[7])
                cofactors.append(decoded_cofactors[idx])
            elif cand_padding is not None:
                idx = cand_padding[0]['index'][0]
                counts.append(padding.row(idx)[5])
                sums.append(padding.row(idx)[6])
                diags.append(padding.row(idx)[7])
                cofactors.append(padded_cofactors[idx])

        key_column_index = group[0][2]
        feature_index = f'{key_column_index}_{group[0][1]}'

        sums = np.array(sums)
        counts = np.array(counts).reshape(-1, 1)
        diags = np.array(diags)
        if var_threshold is not None or corr_threshold is not None:
            indexes_to_keep = table[0].select(pl.col('drop_feature')).to_series().to_list()[0]
            cofactors = self._filter_cofactors(cofactors, indexes_to_keep)
        aug_sketch = AugSketch(feature_index, counts, sums, diags, cofactors)

        return aug_sketch


    def _create_aug_table_sketch_clf(self, group: pl.DataFrame, user_table_agg: pl.DataFrame, query_column_name: str, target_column_name: str, var_threshold: float = None, corr_threshold: float = None) -> AugSketch:
        table = group[1]
        if table.height < user_table_agg.height:
            table, padding, padded_cofactors = self._impute_sketch_table(user_table_agg, table, query_column_name)
            if var_threshold is not None or corr_threshold is not None:
                table = self._filter_sums_diags(table)
                padding = self._filter_sums_diags(padding)
        else:
            # No missing keys: empty padding with matching key-column schema so the
            # concat target-series alignment below keeps height == user_table_agg.height.
            padding = pl.DataFrame({'key': []}, schema={'key': table.schema['key']})
            padded_cofactors = None
            if var_threshold is not None or corr_threshold is not None:
                table = self._filter_sums_diags(table)
        key_column_index = group[0][2]
        feature_index = f'{key_column_index}_{group[0][1]}'

        decoded_cofactors = self._deserialize_cofactors(table)
        table_rows = table.with_row_index().partition_by('key', as_dict=True)
        padding_rows = padding.with_row_index().partition_by('key', as_dict=True)

        local_dict = defaultdict(list)
        targets_per_key = (
            user_table_agg
                .group_by(query_column_name, maintain_order=True)
                .agg(target_column_name)
                .rename({query_column_name: 'key'})
                .select(['key', target_column_name])
        )
        keyed = (
            pl.concat([table.select('key'), padding.select('key')], how='vertical')
                .unique(subset=['key'], maintain_order=True)
                .join(targets_per_key, on='key', how='inner')
                .explode(target_column_name)
                .sort([target_column_name, 'key'])
                .group_by(target_column_name, maintain_order=True)
        )
        for target_class, sub_table in keyed:
            counts = []
            sums = []
            diags = []
            cofactors = []
            for row in sub_table.iter_rows(named=True):
                key = row['key']
                cand_table = table_rows.get((key,), None)
                cand_padding = padding_rows.get((key,), None)
                if cand_table is not None:
                    idx = cand_table[0]['index'][0]
                    counts.append(table.row(idx)[5])
                    sums.append(table.row(idx)[6])
                    diags.append(table.row(idx)[7])
                    cofactors.append(decoded_cofactors[idx])
                elif cand_padding is not None:
                    idx = cand_padding[0]['index'][0]
                    counts.append(padding.row(idx)[5])
                    sums.append(padding.row(idx)[6])
                    diags.append(padding.row(idx)[7])
                    cofactors.append(padded_cofactors[idx])

            sums = np.array(sums)
            counts = np.array(counts).reshape(-1, 1)
            diags = np.array(diags)
            cofactors = np.array(cofactors)
            if var_threshold is not None or corr_threshold is not None:
                indexes_to_keep = table[0].select(pl.col('drop_feature')).to_series().to_list()[0]
                cofactors = self._filter_cofactors(cofactors, indexes_to_keep)
            aug_sketch = AugSketch(feature_index, counts, sums, diags, cofactors)
            local_dict[target_class[0]].append(aug_sketch)

        return local_dict



    def _impute_sketch_table(self, user_table_agg: pl.DataFrame, sketch_table: pl.DataFrame, query_column_name: str) -> pl.DataFrame:
        all_ids = set(user_table_agg[query_column_name].to_list())
        group_ids = set(sketch_table['key'].to_list())
        missing_ids = all_ids - group_ids
        padding_data = {
            col: [None] * len(missing_ids) for col in sketch_table.columns
        }
        padding_data.update({'key': list(missing_ids)})
        padded_rows = pl.DataFrame(padding_data)
        list_len = sketch_table.select(pl.col('sum').list.len()).row(0)[0]
        sums = (
            sketch_table.select(
                pl.concat_list(
                    pl.col('sum')
                    .list.to_struct(upper_bound=list_len)
                    .struct.unnest()
                    .mean()
                )
            )
            .to_numpy()
            .item()
            .reshape(1, -1)
        )

        idx = np.arange(sums.shape[1])
        outer_products = np.outer(sums, sums)[np.newaxis, ...]
        cofactors, diag = extract_cofactors(outer_products, idx)

        # Normalize to shape (1, n-1, n) so that `np.repeat(..., axis=0)` yields
        # (padded_rows.height, n-1, n), matching the per-row shape of decoded_cofactors.
        if cofactors.ndim == 2:
            row_cofactors = cofactors[np.newaxis, ...]
        else:
            row_cofactors = cofactors

        # Match the shape the indexer actually stored for this group (handles
        # edge cases such as single-column groups that store (1, 1) placeholders).
        if 'shape' in sketch_table.columns and sketch_table.height > 0:
            stored_shape = tuple(sketch_table.select(pl.col('shape')).row(0)[0])
            if tuple(row_cofactors.shape[1:]) != stored_shape:
                row_cofactors = np.full((1, *stored_shape), np.nan, dtype=np.float64)

        sums_series = pl.Series(sums, dtype=pl.List(pl.Float64))
        diag_series = pl.Series(diag, dtype=pl.List(pl.Float64))
        # row_cofactors has shape (1, n-1, n) and must match the per-row shape of decoded_cofactors.
        # Broadcast it to (padded_rows.height, n-1, n) so padded_cofactors[idx] is (n-1, n).
        cofactors = np.repeat(row_cofactors, repeats=padded_rows.height, axis=0)
        padded_rows = padded_rows.with_columns(
            pl.col('count').fill_null(1).cast(pl.Int32),
            pl.col('sum').fill_null(sums_series),
            pl.col('diag').fill_null(diag_series),
            pl.col('drop_feature').fill_null(sketch_table.select(pl.col('drop_feature')).row(0)[0]) if 'drop_feature' in sketch_table.columns else pl.lit(None)
        )

        return sketch_table, padded_rows, cofactors


    def _join_aug_table(self, table_feature_index: str, base_table_sketch: BaseSketch, aug_table_sketch: AugSketch, task: str, label: int = None, standardize: bool = True) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        base_count = base_table_sketch.count
        base_features_sum = base_table_sketch.features_sum
        base_features_cofactors = base_table_sketch.features_cofactors
        base_target_sum = base_table_sketch.target_sum
        base_target_cofactors = base_table_sketch.target_cofactors
        aug_sum = aug_table_sketch.sum_
        aug_count = aug_table_sketch.count
        aug_cofactors = aug_table_sketch.cofactors
        aug_diag = aug_table_sketch.diag

        self.init_features_counter = base_features_sum.shape[1]+1 # +1 for intercept
        new_features_count = aug_sum.shape[1]

        base_rows, base_cols = base_features_cofactors[0].shape
        M = base_cols + aug_diag.shape[1]
        joint_cofactor_matrix = np.array(np.zeros((M, M)))

        base_features_cofactors = base_features_cofactors * aug_count[..., np.newaxis]
        base_features_cofactors_agg = base_features_cofactors.sum(axis=0)
        joint_cofactor_matrix[:base_rows, :base_cols] = base_features_cofactors_agg

        self.cache[table_feature_index] = {'initial_diag': aug_diag}
        aug_diag = (aug_diag * base_count).sum(axis=0)
        joint_count_vec = base_count * aug_count
        joint_count = joint_count_vec.sum()
        diag = joint_cofactor_matrix.diagonal().copy()
        zeros = [i for i in range(base_cols, M)]
        diag[zeros] = aug_diag
        joint_cofactor_matrix[np.diag_indices(M)] = diag

        joint_cofactors = base_features_sum[:, :, np.newaxis] * aug_sum[:, np.newaxis, :]
        joint_cofactors_vec = joint_cofactors.sum(axis=0)
        new_features_indexes = list(range(base_cols, M))
        joint_cofactor_matrix[:new_features_indexes[0], new_features_indexes] = joint_cofactors_vec
        joint_cofactor_matrix[new_features_indexes, :new_features_indexes[0]] = joint_cofactors_vec.T

        self.cache[table_feature_index].update({'initial_cofactors': aug_cofactors})
        if aug_cofactors.dtype != object:
            joint_aug_cofactors = aug_cofactors * base_count[..., np.newaxis]
            joint_aug_cofactors_vec = joint_aug_cofactors.sum(axis=0)
            n = joint_aug_cofactors_vec.shape[1]
            joint_aug_cofactors_mask = np.array(np.zeros((n, n)))
            diff = joint_aug_cofactors_mask.shape[0] - joint_aug_cofactors_vec.shape[0]

            upper = np.triu(joint_aug_cofactors_vec, k=diff)
            min_rows = min(upper.shape[0], joint_aug_cofactors_mask.shape[0])
            min_cols = min(upper.shape[1], joint_aug_cofactors_mask.shape[1])
            joint_aug_cofactors_mask[:min_rows, :min_cols] += upper[:min_rows, :min_cols]

            lower = np.tril(joint_aug_cofactors_vec, k=diff-1)
            min_rows = min(lower.shape[0], joint_aug_cofactors_mask.shape[0])
            min_cols = min(lower.shape[1], joint_aug_cofactors_mask.shape[1])
            joint_aug_cofactors_mask[-min_rows:, -min_cols:] += lower[-min_rows:, -min_cols:]

            if np.all(np.diag(joint_aug_cofactors_mask) != 0):
                joint_aug_cofactors_mask = np.flipud(joint_aug_cofactors_mask)

            joint_cofactor_matrix[new_features_indexes, new_features_indexes[0]:new_features_indexes[-1]+1] += joint_aug_cofactors_mask

        joint_feature_target_vector = np.array(np.zeros((M, 1)))
        base_target_cofactors = base_target_cofactors * aug_count[..., np.newaxis]
        y_t_y = base_target_cofactors[:, 0, :]
        y_t_y = y_t_y.sum(axis=0)
        base_target_cofactors = base_target_cofactors[:, 1:, :]
        joint_feature_target_vector[:base_target_cofactors.shape[1], :] = base_target_cofactors.sum(axis=0)
        joint_target_cofactors = base_target_sum[:, :, np.newaxis] * aug_sum[:, np.newaxis, :]
        joint_target_cofactors_vec = joint_target_cofactors.sum(axis=0).reshape(-1, 1)
        joint_feature_target_vector[new_features_indexes[0]:new_features_indexes[-1]+1, :] = joint_target_cofactors_vec

        base_features_sum = base_features_sum * aug_count
        self.cache[table_feature_index].update({'initial_feature': aug_sum})
        aug_sum = aug_sum * base_count
        joint_sum = np.hstack([base_features_sum, aug_sum])
        joint_sum_vec = joint_sum.sum(axis=0)
        joint_cofactor_matrix = np.pad(joint_cofactor_matrix, pad_width=((1, 0), (1, 0)), mode='constant', constant_values=0)
        joint_cofactor_matrix[0, 0] = joint_count
        joint_cofactor_matrix[1:, 0] = joint_sum_vec
        joint_cofactor_matrix[0, 1:] = joint_sum_vec
        base_target_sum = base_target_sum * aug_count
        base_target_sum_vec = base_target_sum.sum(axis=0)
        joint_feature_target_vector = np.pad(joint_feature_target_vector, pad_width=((1, 0), (0, 0)), mode='constant', constant_values=0)
        joint_feature_target_vector[0, :] = base_target_sum.sum()

        intercept_idx = 0
        all_indices = np.arange(joint_cofactor_matrix.shape[1])
        complement = np.setdiff1d(all_indices, [intercept_idx])
        joint_cofactor_matrix_std, joint_feature_target_vector_std = self._standardize_gram_inputs(
            joint_cofactor_matrix[np.ix_(complement, complement)],
            joint_feature_target_vector[complement],
            joint_sum_vec,
            base_target_sum_vec,
            joint_count
        )
        self.cache[table_feature_index].update(
            {
                'new_feature': aug_sum,
                'joint_count': joint_count,
                'joint_cofactor_matrix': deepcopy(joint_cofactor_matrix),
                'joint_feature_target_vector': deepcopy(joint_feature_target_vector),
                'y_t_y': y_t_y,
                'joint_sum_vec': joint_sum_vec,
                'base_target_sum_vec': base_target_sum_vec
            }
        )

        if task == 'regression':
            self.fs_output[table_feature_index] = {
                'joint_cofactor_matrix': joint_cofactor_matrix_std,
                'joint_feature_target_vector': joint_feature_target_vector_std,
                'joint_count': joint_count,
                'y_t_y': y_t_y,
                'base_target_sum_vec': base_target_sum_vec
            }
        elif task == 'classification':
            features_mean = joint_sum_vec / joint_count
            self.fs_output[table_feature_index] = {
                'joint_cofactor_matrix': joint_cofactor_matrix_std,
                'features_mean': features_mean,
                'joint_count': joint_count,
                'new_features_count': new_features_count
            }

        if task == 'regression' and standardize:
            return joint_cofactor_matrix_std, joint_feature_target_vector_std, joint_count, y_t_y, new_features_count, base_target_sum_vec
        elif task == 'regression' and not standardize:
            return joint_cofactor_matrix[np.ix_(complement, complement)], joint_feature_target_vector[complement], joint_count, y_t_y, new_features_count, base_target_sum_vec
        elif task == 'classification':
            return {label: [joint_cofactor_matrix_std, features_mean, joint_count, new_features_count]}


    def _update_from_cache(self, table_feature_index: str, task: str) -> None:
        base_sum = self.new_feature  # best feature which was added from previous iteration
        joint_count = self.joint_count
        y_t_y = self.y_t_y
        aug_sum = self.cache[table_feature_index]['initial_feature'] # new feature which is being added in current iteration, not scaled by `joint_count`

        joint_cofactor_matrix = self.joint_cofactor_matrix
        init_rows, init_cols = joint_cofactor_matrix.shape
        joint_feature_target_vector = self.joint_feature_target_vector
        new_features_count = aug_sum.shape[1]
        joint_cofactor_matrix, joint_feature_target_vector = self._pad_linreg_statistics(joint_cofactor_matrix, joint_feature_target_vector, new_features_count)

        joint_cofactors = base_sum[:, :, np.newaxis] * aug_sum[:, np.newaxis, :]
        joint_cofactors_vec = joint_cofactors.sum(axis=0)
        # Cached arrays come from Ray's plasma store and are read-only; copy
        # before zeroing the upper-left block so we can mutate in place.
        aug_cofactor_matrix = np.array(self.cache[table_feature_index]['joint_cofactor_matrix'], copy=True)
        aug_init_rows, aug_init_cols = aug_cofactor_matrix.shape
        aug_feature_target_vector = self.cache[table_feature_index]['joint_feature_target_vector'][-new_features_count:, :]
        aug_cofactor_matrix[:aug_init_rows-new_features_count, :aug_init_cols-new_features_count] = 0
        feature_diff = joint_cofactor_matrix.shape[1] - aug_cofactor_matrix.shape[1]
        aug_cofactor_matrix, _ = self._pad_linreg_statistics(
            aug_cofactor_matrix,
            aug_feature_target_vector,
            m=feature_diff,
            shifted=True,
            offset=feature_diff+aug_sum.shape[1]
        )

        joint_cofactor_matrix = joint_cofactor_matrix + aug_cofactor_matrix

        joint_cofactor_matrix_covar_idx = init_cols-base_sum.shape[1]
        joint_cofactor_matrix_covar_part = joint_cofactor_matrix[joint_cofactor_matrix_covar_idx:, joint_cofactor_matrix_covar_idx:]
        pad_rows = joint_cofactor_matrix_covar_part.shape[0] - joint_cofactors_vec.shape[0]
        pad_cols = joint_cofactor_matrix_covar_part.shape[1] - joint_cofactors_vec.shape[1]
        joint_cofactors_vec = np.pad(joint_cofactors_vec, pad_width=((0, pad_rows), (pad_cols, 0)), mode='constant', constant_values=0)
        joint_cofactors_vec = joint_cofactors_vec + joint_cofactors_vec.T
        joint_cofactor_matrix[joint_cofactor_matrix_covar_idx:, joint_cofactor_matrix_covar_idx:] = joint_cofactor_matrix_covar_part + joint_cofactors_vec

        aug_sum = self.cache[table_feature_index]['new_feature'] # new feature which is being added in current iteration, scaled by `joint_count`
        aug_sum_vec = aug_sum.sum(axis=0)
        joint_cofactor_matrix[init_rows:, 0] = aug_sum_vec
        joint_cofactor_matrix[0, init_cols:] = aug_sum_vec

        joint_feature_target_vector[init_rows:, :] = aug_feature_target_vector

        joint_sum_vec = np.hstack([self.joint_sum_vec, aug_sum_vec])
        base_target_sum_vec = self.cache[table_feature_index]['base_target_sum_vec']

        intercept_idx = 0
        all_indices = np.arange(joint_cofactor_matrix.shape[1])
        complement = np.setdiff1d(all_indices, [intercept_idx])
        joint_cofactor_matrix_std, joint_feature_target_vector_std = self._standardize_gram_inputs(
            joint_cofactor_matrix[np.ix_(complement, complement)],
            joint_feature_target_vector[complement],
            joint_sum_vec,
            base_target_sum_vec,
            joint_count
        )

        self.cache[table_feature_index].update(
            {
                'new_feature': aug_sum,
                'joint_count': joint_count,
                'joint_cofactor_matrix': deepcopy(joint_cofactor_matrix),
                'joint_feature_target_vector': deepcopy(joint_feature_target_vector),
                'y_t_y': y_t_y,
                'joint_sum_vec': joint_sum_vec,
                'base_target_sum_vec': base_target_sum_vec
            }
        )

        joint_cofactor_matrix[np.ix_(complement, complement)] = joint_cofactor_matrix_std
        joint_feature_target_vector[complement] = joint_feature_target_vector_std

        if task == 'regression':
            self.fs_output[table_feature_index] = {
                'joint_cofactor_matrix': joint_cofactor_matrix_std,
                'joint_feature_target_vector': joint_feature_target_vector_std,
                'joint_count': joint_count,
                'y_t_y': y_t_y,
                'base_target_sum_vec': base_target_sum_vec
            }
        elif task == 'classification':
            features_mean = joint_sum_vec / joint_count
            self.fs_output[table_feature_index] = {
                'joint_cofactor_matrix': joint_cofactor_matrix_std,
                'features_mean': features_mean,
                'joint_count': joint_count,
                'new_features_count': new_features_count
            }

        return joint_cofactor_matrix_std, joint_feature_target_vector_std, joint_count, y_t_y, new_features_count


    def _update_from_cache_batch(self, table_feature_indices: list[str], task: str) -> None:
        '''Batch version of _update_from_cache: processes all features in a single
        actor call instead of one remote call per feature, avoiding per-call
        scheduling overhead.'''
        for table_feature_index in table_feature_indices:
            self._update_from_cache(table_feature_index, task)


    def _update_cache_and_get_stats(self, task: str, n_jobs: int = 1):
        '''Combined operation: updates all cached features and returns
        (remaining_feature_keys, fs_output, timing_dict) in a single actor round-trip.
        When n_jobs > 1, fans out computation to parallel workers.
        timing_dict keys: put_s, dispatch_s, get_s, compute_s.'''
        import time as _time
        _ZERO_TIMING = {'put_s': 0.0, 'dispatch_s': 0.0, 'get_s': 0.0, 'compute_s': 0.0}
        remaining = list(self.cache.keys())
        if n_jobs <= 1 or len(remaining) < 4:
            # Sequential path — original behavior
            for table_feature_index in remaining:
                self._update_from_cache(table_feature_index, task)
            return remaining, self.fs_output, _ZERO_TIMING

        # Parallel path: extract shared state, fan out, collect & apply
        shared_state = {
            'new_feature': self.new_feature,
            'joint_cofactor_matrix': self.joint_cofactor_matrix,
            'joint_feature_target_vector': self.joint_feature_target_vector,
            'joint_count': self.joint_count,
            'y_t_y': self.y_t_y,
            'joint_sum_vec': self.joint_sum_vec
        }
        t_put = _time.perf_counter()
        shared_ref = ray.put(shared_state)
        t_put_end = _time.perf_counter()

        # Build list of (idx, cache_entry) — each entry is already a dict of numpy arrays
        feature_entries = [(idx, self.cache[idx]) for idx in remaining]

        # Split into n_jobs batches
        batch_size = max(1, (len(feature_entries) + n_jobs - 1) // n_jobs)
        batches = [feature_entries[i:i + batch_size] for i in range(0, len(feature_entries), batch_size)]

        t_dispatch = _time.perf_counter()
        futures = [
            _compute_feature_update_batch.remote(shared_ref, batch, task)
            for batch in batches
        ]
        t_dispatch_end = _time.perf_counter()

        t_get = _time.perf_counter()
        batch_results_raw = ray.get(futures)
        t_get_end = _time.perf_counter()

        # Apply results back into actor state; collect per-task compute times
        compute_times = []
        for batch_result, compute_s in batch_results_raw:
            compute_times.append(compute_s)
            for idx, cache_update, fs_output in batch_result:
                self.cache[idx].update(cache_update)
                self.fs_output[idx] = fs_output

        timing = {
            'put_s': t_put_end - t_put,
            'dispatch_s': t_dispatch_end - t_dispatch,
            'get_s': t_get_end - t_get,
            'compute_s': max(compute_times) if compute_times else 0.0,
        }
        return remaining, self.fs_output, timing

    def accept_best_feature(self, best_feature: str):
        '''Combined operation: reads best feature from cache, updates internal
        state with it, removes it from the pool, and returns remaining count.
        Replaces 2 blocking ray.get round-trips per iteration with 1.'''
        best = self.cache[best_feature]
        self.new_feature = best['new_feature']
        self.joint_cofactor_matrix = best['joint_cofactor_matrix']
        self.joint_feature_target_vector = best['joint_feature_target_vector']
        self.joint_count = best['joint_count']
        self.y_t_y = best['y_t_y']
        self.joint_sum_vec = best['joint_sum_vec']
        del self.cache[best_feature]
        return len(self.cache)


    def _standardize_gram_inputs(self, cofactor_matrix, feature_target_vector, features_sum, target_sum, joint_count):
        feature_target_vector = feature_target_vector.reshape(-1)

        target_mean = target_sum / joint_count

        cofactor_centered = cofactor_matrix - np.outer(features_sum, features_sum) / joint_count
        feature_target_centered = feature_target_vector - (features_sum * target_mean)

        feature_vars = np.diag(cofactor_centered) / joint_count
        feature_vars = np.clip(feature_vars, a_min=0.0, a_max=None)
        feature_stds = np.sqrt(feature_vars)

        # avoid division by zero
        feature_stds[feature_stds == 0] = 1.0

        # construct diagonal scaling matrix D^{-1} (as a vector)
        D_inv = 1.0 / feature_stds

        cofactor_std = (D_inv[:, None] * cofactor_centered) * D_inv[None, :]
        feature_target_std = (D_inv * feature_target_centered).reshape(-1, 1)
        cofactor_std = np.array(cofactor_std)
        feature_target_std = np.array(feature_target_std)

        return cofactor_std, feature_target_std


    def update_with_best_feature(
        self,
        new_feature: np.ndarray,
        joint_cofactor_matrix: np.ndarray,
        joint_feature_target_vector: np.ndarray,
        joint_count: int,
        y_t_y: np.ndarray,
        joint_sum_vec: np.ndarray,
        M1: np.ndarray = None,
        M2: np.ndarray = None,
        d: np.ndarray = None
    ) -> None:
        '''
        Written as a separate method since Ray Actors cannot update attribute values directly.
        '''
        self.new_feature = new_feature
        self.joint_cofactor_matrix = joint_cofactor_matrix
        self.joint_feature_target_vector = joint_feature_target_vector
        self.joint_count = joint_count
        self.y_t_y = y_t_y
        self.joint_sum_vec = joint_sum_vec
        self.M1 = M1
        self.M2 = M2
        self.d = d


    def remove_from_features_pool(self, table_feature_index: str) -> None:
        '''
        Written as a separate method since Ray Actors cannot update attribute values directly.
        '''
        del self.cache[table_feature_index]


    def retrieve(self, attr_name):
        '''
        Written as a separate method since Ray Actors cannot return attribute values directly.
        '''
        return getattr(self, attr_name, None)


    def _pad_linreg_statistics(self, cofactor_matrix: np.ndarray, feature_target_vector: np.ndarray, m: int, shifted: bool = False, offset: int = None) -> tuple[np.ndarray, np.ndarray]:
        padded_matrix = np.pad(cofactor_matrix, pad_width=((0, m), (0, m)), mode='constant', constant_values=0)
        padded_vector = np.pad(feature_target_vector, pad_width=((0, m), (0, 0)), mode='constant', constant_values=0)

        if shifted:
            new_feature_range = list(range(-m, 0))
            test_feature_range = list(range(-offset, -m))
            current_indexing = test_feature_range + new_feature_range
            new_indexing = new_feature_range + test_feature_range

            padded_matrix[current_indexing, :] = padded_matrix[new_indexing, :]
            padded_matrix[:, current_indexing] = padded_matrix[:, new_indexing]

            padded_vector[current_indexing, :] = padded_vector[new_indexing, :]

        return padded_matrix, padded_vector
