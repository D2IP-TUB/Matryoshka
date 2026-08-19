"""The query table contract enforced by ``prepare_query_table``.

``JoinSelection`` builds the query table's Gram matrix sketch straight from its
columns, so it accepts exactly one layout: normalised string key first, numeric
null-free features, target last. These tests pin that layout, and check that a
prepared table is in fact accepted by the checkup that guards discovery.
"""
import polars as pl
import pytest

from matryoshka.exceptions import UserTableNotProcessed
from matryoshka.preprocessing import prepare_query_table, train_test_split_by_key
from matryoshka.utils.common import process_key


def raw_table() -> pl.DataFrame:
    return pl.DataFrame({
        'zip': ['  12 345 ', 'ABC-678', '12345', None, 'xyz'],
        'age': [30, 41, None, 55, 22],
        'city': ['Berlin', 'Vienna', 'Berlin', 'Berlin', None],
        'label': ['yes', 'no', 'yes', 'no', 'yes'],
    })


def test_column_order_key_features_target():
    out = prepare_query_table(raw_table(), key='zip', target='label')
    assert out.columns[0] == 'zip'
    assert out.columns[-1] == 'label'
    assert set(out.columns[1:-1]) == {'age', 'city'}


def test_key_is_normalised_string():
    out = prepare_query_table(raw_table(), key='zip', target='label')
    assert out.schema['zip'] == pl.String
    # '  12 345 ' and '12345' normalise to the same token, which is what makes
    # them join against the same inverted-index entry.
    assert out.get_column('zip')[0] == process_key('  12 345 ') == '12345'


def test_null_keys_and_targets_are_dropped():
    out = prepare_query_table(raw_table(), key='zip', target='label')
    assert out.height == 4  # the row with a null key is gone


def test_no_nulls_and_all_features_numeric():
    out = prepare_query_table(raw_table(), key='zip', target='label')
    assert out.null_count().to_numpy().sum() == 0
    assert all(dtype == pl.Float64 for dtype in out.schema.values() if dtype != pl.String)


def test_classification_target_is_ordinal_encoded():
    out = prepare_query_table(raw_table(), key='zip', target='label',
                              task='classification')
    assert sorted(out.get_column('label').unique().to_list()) == [0.0, 1.0]


def test_regression_target_is_cast_not_encoded():
    raw = pl.DataFrame({'k': ['a', 'b', 'c'], 'x': [1, 2, 3], 'y': [1.5, 2.5, 3.5]})
    out = prepare_query_table(raw, key='k', target='y', task='regression')
    assert out.get_column('y').to_list() == [1.5, 2.5, 3.5]


def test_feature_subset_is_honoured():
    out = prepare_query_table(raw_table(), key='zip', target='label', features=['age'])
    assert out.columns == ['zip', 'age', 'label']


def test_binning_expands_into_indicator_columns():
    raw = pl.DataFrame({
        'k': [f'k{i}' for i in range(40)],
        'x': list(range(40)),
        'y': [i % 2 for i in range(40)],
    })
    out = prepare_query_table(raw, key='k', target='y', n_bins=4)
    # Four quantile bins, first level dropped, leaves three indicators.
    assert out.width == 1 + 3 + 1


def test_unknown_columns_raise():
    with pytest.raises(KeyError):
        prepare_query_table(raw_table(), key='absent', target='label')
    with pytest.raises(KeyError):
        prepare_query_table(raw_table(), key='zip', target='label', features=['absent'])


def test_invalid_task_rejected():
    with pytest.raises(ValueError, match='classification'):
        prepare_query_table(raw_table(), key='zip', target='label', task='ranking')


def test_prepared_table_passes_the_discovery_checkup():
    """The contract this function exists to satisfy."""
    from matryoshka.join_selection import JoinSelection

    out = prepare_query_table(raw_table(), key='zip', target='label')
    checkup = JoinSelection._user_table_checkup
    assert checkup(None, out, 'zip') is out
    # A raw table with string feature columns is rejected, which is the failure
    # mode prepare_query_table removes.
    with pytest.raises(UserTableNotProcessed):
        checkup(None, raw_table(), 'zip')


def test_split_is_deterministic_and_partitions_the_rows():
    table = pl.DataFrame({'k': [f'k{i}' for i in range(100)], 'y': list(range(100))})
    train, test = train_test_split_by_key(table, test_fraction=0.2, seed=7)
    again, _ = train_test_split_by_key(table, test_fraction=0.2, seed=7)
    assert train.height == 80 and test.height == 20
    assert train.equals(again)
    assert set(train['y'].to_list()) | set(test['y'].to_list()) == set(range(100))


def test_split_rejects_degenerate_fractions():
    table = pl.DataFrame({'k': ['a'], 'y': [1]})
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            train_test_split_by_key(table, test_fraction=bad)
