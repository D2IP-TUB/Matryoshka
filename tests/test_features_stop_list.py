"""The features stop-list merged into the ``drop_feature`` mask of pruning.

Positions of the sketch vectors are named by ``column_headers`` and ordered
differently for every key column, so the tests pin that stop-listed features are
located by name within each (table, key column, feature group), and that the
stop-list composes with the mask built by correlation pruning.
"""
import polars as pl
import pytest

from matryoshka.retrieval import apply_features_stop_list, normalize_features_stop_list


def retrieval_results(drop_feature=None) -> pl.DataFrame:
    # Two keys for each of three (table, key column) pairs; the headers of the
    # same table are ordered differently for its two key columns.
    pairs = [
        (0, 0, 'wdc-1438042988250.59-6.parquet', ['federalcontribution_mean', 'region_mean', 'federalcontribution_max']),
        (0, 1, 'wdc-1438042988250.59-6.parquet', ['region_mean', 'federalcontribution_mean', 'id3v2_nunique']),
        (1, 0, 'other.parquet', ['federalcontribution_mean', 'region_mean']),
    ]
    rows = []
    for table_index, key_col_index, table_name, headers in pairs:
        for key in ('a', 'b'):
            rows.append({
                'key': key,
                'feature_index': key_col_index,
                'table_index': table_index,
                'key_col_index': key_col_index,
                'sum': [float(i) for i in range(len(headers))],
                'table_name': table_name,
                'column_headers': headers,
            })
    schema = {
        'key': pl.String, 'feature_index': pl.Int32, 'table_index': pl.Int32, 'key_col_index': pl.Int32,
        'sum': pl.List(pl.Float64), 'table_name': pl.String, 'column_headers': pl.List(pl.String),
    }
    results = pl.DataFrame(rows, schema=schema)
    if drop_feature is not None:
        results = results.with_columns(pl.Series('drop_feature', drop_feature, dtype=pl.List(pl.Int16)))
    return results


def masks(results: pl.DataFrame) -> dict[tuple[int, int], list[int]]:
    return {
        (row['table_index'], row['key_col_index']): row['drop_feature']
        for row in results.iter_rows(named=True)
    }


def test_masks_every_aggregate_by_name_for_each_key_column():
    stop_list = normalize_features_stop_list({'wdc-1438042988250.59-6': ['federalContribution']})
    out, masked = apply_features_stop_list(retrieval_results(), stop_list, has_mask=False)
    assert masked
    assert masks(out) == {(0, 0): [0, 1, 0], (0, 1): [1, 0, 1], (1, 0): [1, 1]}
    assert out.height == 6


def test_normalized_names_and_table_name_variants_match():
    for table_name in ('wdc-1438042988250.59-6.parquet', '/lake/wdc-1438042988250.59-6.parquet', 'wdc-1438042988250.59-6'):
        stop_list = normalize_features_stop_list({table_name: ['id3v2']})
        out, masked = apply_features_stop_list(retrieval_results(), stop_list, has_mask=False)
        assert masked, table_name
        assert masks(out)[(0, 1)] == [1, 1, 0]


def test_merges_with_the_correlation_mask():
    correlation_mask = [[1, 1, 0]] * 2 + [[0, 1, 1]] * 2 + [[1, 0]] * 2
    stop_list = normalize_features_stop_list({'wdc-1438042988250.59-6.parquet': ['region']})
    out, masked = apply_features_stop_list(retrieval_results(correlation_mask), stop_list, has_mask=True)
    assert masked
    assert masks(out) == {(0, 0): [1, 0, 0], (0, 1): [0, 1, 1], (1, 0): [1, 0]}


def test_drops_feature_groups_left_without_features():
    stop_list = normalize_features_stop_list({'other.parquet': ['federalContribution', 'region']})
    out, masked = apply_features_stop_list(retrieval_results(), stop_list, has_mask=False)
    assert masked
    assert set(masks(out)) == {(0, 0), (0, 1)}
    assert out.height == 4


def test_no_match_leaves_the_results_unchanged():
    results = retrieval_results()
    for stop_list in ({}, {'absent.parquet': ['region']}, {'other.parquet': ['absent']}):
        out, masked = apply_features_stop_list(results, normalize_features_stop_list(stop_list), has_mask=False)
        assert not masked
        assert out.equals(results)


def test_single_column_name_is_not_split_into_characters():
    assert normalize_features_stop_list({'t.csv': 'Region'}) == {'t.csv': {'region'}}


def test_missing_headers_fail_loudly():
    results = retrieval_results().drop('column_headers')
    with pytest.raises(ValueError, match='column_headers'):
        apply_features_stop_list(results, normalize_features_stop_list({'other.parquet': ['region']}), has_mask=False)
