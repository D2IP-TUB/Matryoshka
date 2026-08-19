"""Unit test for ExhaustiveIndex.find_union_peer.

We bypass Postgres by stubbing `_seek_key_overlaps`, `_seek_cat_overlaps`, and
`_table_meta_by_index` with synthetic results. The test pins the bipartite
combiner / matching / threshold logic of `find_union_peer`.

Run:
    PYTHONPATH=. python3 augmentation/testing/test_find_union_peer.py
"""
from __future__ import annotations

import os
import sys

import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from matryoshka.index import ExhaustiveIndex


def _make_index() -> ExhaustiveIndex:
    idx = ExhaustiveIndex.__new__(ExhaustiveIndex)
    return idx


def _stub(idx, key_rows, cat_rows, metas):
    idx._seek_key_overlaps = lambda value_sets, exclude_table_index, top_k: pl.DataFrame(
        key_rows,
        schema={'q_col': pl.String, 'r_kind': pl.String, 'table_index': pl.Int64,
                'r_col': pl.String, 'key_col_index': pl.Int64, 'overlap': pl.Int64},
    )
    idx._seek_cat_overlaps = lambda value_sets, exclude_table_index, top_k: pl.DataFrame(
        cat_rows,
        schema={'q_col': pl.String, 'r_kind': pl.String, 'table_index': pl.Int64,
                'r_col': pl.String, 'overlap': pl.Int64},
    )
    idx._table_meta_by_index = lambda table_indices: {
        ti: m for ti, m in metas.items() if ti in set(table_indices)
    }


def test_simple_peer_via_key_and_cat():
    """Query has 2 cols; peer table_index=7 has matching key + cat columns."""
    idx = _make_index()
    metas = {
        7: {'table_index': 7, 'table_name': 'peer_table',
            'numeric_cols': [], 'cat_cols': ['city'], 'key_columns': ['user_id'],
            'useful_columns': [], 'n_rows': 100},
        9: {'table_index': 9, 'table_name': 'noisy_table',
            'numeric_cols': [], 'cat_cols': ['xxx'], 'key_columns': ['k'],
            'useful_columns': [], 'n_rows': 1000},
    }
    key_rows = [
        {'q_col': 'uid', 'r_kind': 'key', 'table_index': 7,
         'r_col': '__key_col_0__', 'key_col_index': 0, 'overlap': 50},
        {'q_col': 'uid', 'r_kind': 'key', 'table_index': 9,
         'r_col': '__key_col_0__', 'key_col_index': 0, 'overlap': 5},
    ]
    cat_rows = [
        {'q_col': 'town', 'r_kind': 'cat', 'table_index': 7,
         'r_col': 'city', 'overlap': 30},
    ]
    _stub(idx, key_rows, cat_rows, metas)

    df = pl.DataFrame({'uid': ['a', 'b', 'c'], 'town': ['NY', 'LA', 'SF']})
    res = idx.find_union_peer(df, threshold=0.0, min_cols=0.0)
    assert res is not None, 'Expected a peer'
    assert res['peer_table_index'] == 7, res
    assert res['peer_table_name'] == 'peer_table'
    assert res['mapping'] == {'uid': 'user_id', 'town': 'city'}, res['mapping']
    assert res['matched_cols'] == 2
    assert res['score'] > 0
    print('OK test_simple_peer_via_key_and_cat:', res)


def test_no_peer_when_seekers_empty():
    idx = _make_index()
    _stub(idx, [], [], {})
    df = pl.DataFrame({'a': ['x', 'y'], 'b': ['m', 'n']})
    res = idx.find_union_peer(df)
    assert res is None
    print('OK test_no_peer_when_seekers_empty')


def test_min_cols_filter_blocks_low_coverage():
    """Query has 4 cols; peer covers only 1. min_cols=0.5 should reject."""
    idx = _make_index()
    metas = {
        7: {'table_index': 7, 'table_name': 'peer_table',
            'numeric_cols': [], 'cat_cols': ['city'], 'key_columns': ['user_id'],
            'useful_columns': [], 'n_rows': 100},
    }
    cat_rows = [{'q_col': 'town', 'r_kind': 'cat', 'table_index': 7,
                 'r_col': 'city', 'overlap': 30}]
    _stub(idx, [], cat_rows, metas)

    df = pl.DataFrame({'a': ['1'], 'b': ['1'], 'c': ['1'], 'town': ['NY']})
    res = idx.find_union_peer(df, threshold=0.0, min_cols=0.5)
    # 2*1 / (4 + 2) = 0.33 < 0.5 → rejected
    assert res is None, res
    print('OK test_min_cols_filter_blocks_low_coverage')


def test_threshold_filter_blocks_low_overlap():
    """Tiny overlap relative to row counts → cell-overlap threshold rejects."""
    idx = _make_index()
    metas = {
        7: {'table_index': 7, 'table_name': 'peer_table',
            'numeric_cols': [], 'cat_cols': ['city'], 'key_columns': ['user_id'],
            'useful_columns': [], 'n_rows': 1_000_000},
    }
    cat_rows = [{'q_col': 'town', 'r_kind': 'cat', 'table_index': 7,
                 'r_col': 'city', 'overlap': 1}]
    key_rows = [{'q_col': 'uid', 'r_kind': 'key', 'table_index': 7,
                 'r_col': '__key_col_0__', 'key_col_index': 0, 'overlap': 1}]
    _stub(idx, key_rows, cat_rows, metas)

    df = pl.DataFrame({'uid': ['a'], 'town': ['NY']})
    res = idx.find_union_peer(df, threshold=0.5, min_cols=0.0)
    assert res is None, res
    print('OK test_threshold_filter_blocks_low_overlap')


def test_picks_best_among_candidates():
    """Two viable peers: pick the one with higher matched-overlap score."""
    idx = _make_index()
    metas = {
        7: {'table_index': 7, 'table_name': 'better',
            'numeric_cols': [], 'cat_cols': ['city'], 'key_columns': ['user_id'],
            'useful_columns': [], 'n_rows': 100},
        8: {'table_index': 8, 'table_name': 'weaker',
            'numeric_cols': [], 'cat_cols': ['city'], 'key_columns': ['user_id'],
            'useful_columns': [], 'n_rows': 100},
    }
    key_rows = [
        {'q_col': 'uid', 'r_kind': 'key', 'table_index': 7,
         'r_col': '__key_col_0__', 'key_col_index': 0, 'overlap': 80},
        {'q_col': 'uid', 'r_kind': 'key', 'table_index': 8,
         'r_col': '__key_col_0__', 'key_col_index': 0, 'overlap': 20},
    ]
    cat_rows = [
        {'q_col': 'town', 'r_kind': 'cat', 'table_index': 7, 'r_col': 'city', 'overlap': 50},
        {'q_col': 'town', 'r_kind': 'cat', 'table_index': 8, 'r_col': 'city', 'overlap': 10},
    ]
    _stub(idx, key_rows, cat_rows, metas)

    df = pl.DataFrame({'uid': ['a'] * 10, 'town': ['NY'] * 10})
    res = idx.find_union_peer(df, threshold=0.0, min_cols=0.0)
    assert res is not None
    assert res['peer_table_index'] == 7, res
    print('OK test_picks_best_among_candidates:', res['peer_table_index'], res['score'])


if __name__ == '__main__':
    test_simple_peer_via_key_and_cat()
    test_no_peer_when_seekers_empty()
    test_min_cols_filter_blocks_low_coverage()
    test_threshold_filter_blocks_low_overlap()
    test_picks_best_among_candidates()
    print('\nAll find_union_peer tests passed.')
