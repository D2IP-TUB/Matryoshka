"""Per-key value shares of text columns (``ExhaustiveIndex(category_indicators=True)``).

The count features of text columns are constant when every key occurs once, so without the shares the values of text
columns (e.g., a passenger's sex) never reach selection. These tests pin which columns and values are encoded, the
shares for unique and repeated keys, and that distinct share columns survive the duplicate-column filter.
"""
import polars as pl
import pytest

from matryoshka.index import ExhaustiveIndex
from matryoshka.selection.algorithms.greedy import _strip_aggregation_suffix


def make_index(**kwargs) -> ExhaustiveIndex:
    # Bypass __init__ to avoid logging and database setup; set only what the queries read.
    index = ExhaustiveIndex.__new__(ExhaustiveIndex)
    index.feature_extraction = False
    index.category_indicators = True
    index.max_category_values = kwargs.get('max_category_values', 10)
    index.min_category_coverage = kwargs.get('min_category_coverage', 0.5)
    return index


def passengers() -> pl.DataFrame:
    # One row per key; `cabin` is mostly missing and `port` has three values.
    return pl.DataFrame({
        'name': ['Abbott, Mr. A', 'Abelson, Mr. S', 'Abelson, Mrs. S', 'Adahl, Mr. M', 'Allen, Miss E', 'Baxter, Mr. Q'],
        'sex': ['male', 'male', 'female', 'male', 'female', 'male'],
        'cabin': [None, None, 'B5', None, None, 'C22'],
        'port': ['S', 'C', 'C', 'S', 'S', 'Q'],
    })


def shares(index, table, key, columns):
    out = index._category_indicator_query(table.lazy(), key, columns)
    return out.sort(key)


def test_unique_keys_hold_the_value_and_leave_the_most_frequent_value_out():
    out = shares(make_index(), passengers(), 'name', ['sex', 'port'])
    assert set(out.columns) == {'name', 'sex_female_share', 'port_c_share', 'port_q_share'}
    by_key = {r['name']: r for r in out.iter_rows(named=True)}
    assert by_key['abelsonmrss']['sex_female_share'] == 1.0
    assert by_key['abelsonmrs']['port_c_share'] == 1.0
    assert by_key['abbottmra']['sex_female_share'] == 0.0 and by_key['abbottmra']['port_c_share'] == 0.0


def test_text_columns_called_count_are_encoded():
    # Counting the values of a column called 'count' used to produce two columns of that name and fail the table
    table = passengers().rename({'sex': 'count'})
    out = shares(make_index(), table, 'name', ['count'])
    assert set(out.columns) == {'name', 'count_female_share'}


def test_mostly_missing_and_identifier_columns_are_not_encoded():
    table = passengers().with_columns(pl.col('name').alias('ticket'))
    out = shares(make_index(max_category_values=2), table, 'sex', ['cabin', 'ticket'])
    assert out.columns == ['sex']


def test_repeated_keys_hold_the_share_of_their_rows():
    beers = pl.DataFrame({
        'state': ['MI', 'MI', 'MI', 'MI', 'IN', 'IN'],
        'style': ['American IPA', 'American IPA', 'Stout', None, 'Rauchbier', 'Stout'],
    })
    out = shares(make_index(max_category_values=2, min_category_coverage=0.5), beers, 'state', ['style'])
    # The two most frequent values cover 4 of 6 rows, so neither is left out; the missing value holds none of them.
    by_key = {r['state']: r for r in out.iter_rows(named=True)}
    assert by_key['mi']['style_americanipa_share'] == pytest.approx(0.5)
    assert by_key['mi']['style_stout_share'] == pytest.approx(0.25)
    assert by_key['in']['style_americanipa_share'] == 0.0
    assert by_key['in']['style_stout_share'] == pytest.approx(0.5)


def test_values_normalizing_to_the_same_name_stay_distinct():
    # The missing value keeps every value (they do not cover all rows); 'a-b' and 'ab' both normalize to 'ab', and '--'
    # to an empty name
    table = pl.DataFrame({'id': [1, 2, 3, 4, 5], 'kind': ['a-b', 'ab', 'a-b', '--', None]})
    out = shares(make_index(), table, 'id', ['kind'])
    assert set(out.columns) == {'id', 'kind_ab_share', 'kind_value1_share', 'kind_ab2_share'}


def test_share_columns_with_as_many_ones_are_not_dropped_as_duplicates():
    index = make_index()
    table = pl.DataFrame({
        'id': [1, 2, 3, 4],
        'port': ['C', 'Q', 'S', 'S'],
        'amount': [1.0, 2.0, 3.0, 4.0],
    })
    out = index._combined_query(table.lazy(), 'id', ['amount'], ['port'], ['amount', 'port'], False)
    # port_c_share and port_q_share each hold a single one, in different rows
    assert {'port_c_share', 'port_q_share'} <= set(out.columns)
    # exact duplicates are still dropped: amount mean, median, max and min are identical for unique keys
    assert sum(c.startswith('amount_') for c in out.columns) == 1


def test_shares_are_disabled_by_default():
    index = make_index()
    index.category_indicators = False
    out = index._non_numeric_query(passengers().lazy(), 'name', [], ['sex', 'port'], ['sex', 'port'])
    assert not any(c.endswith('_share') for c in out.columns)


def test_key_columns_with_only_share_features_are_indexed(tmp_path):
    # A table with a key and a text column only: the key's features are all shares, which have no plain aggregate to
    # lead the feature group
    index = ExhaustiveIndex(
        feature_selection_table_name='unused', overlap_table_name='unused_overlap', max_workers=2,
        category_indicators=True, settings='postgresql://user:password@localhost:5432/unused',
        log_dir=str(tmp_path), silent=True,
    )
    table = passengers().select('name', 'sex')
    rows = index.index_table(table.lazy(), table_index=0, table_name='passengers.parquet')
    headers = {h for r in rows if r.row(0, named=True)['key_col_index'] == 0 for h in r.row(0, named=True)['column_headers']}
    assert headers == {'sex_female_share'}


def test_every_value_is_its_own_candidate():
    assert _strip_aggregation_suffix('sex_female_share') == 'sex_female'
    assert _strip_aggregation_suffix('fare_mean') == 'fare'


def test_unsupported_combinations_are_rejected():
    with pytest.raises(ValueError, match='update_support'):
        ExhaustiveIndex(category_indicators=True, update_support=True)
    with pytest.raises(ValueError, match='feature_extraction'):
        ExhaustiveIndex(category_indicators=True, feature_extraction=True)
