"""Equivalence test for the production update helpers, without touching Postgres.

Validates that for the production code paths:
    derive(merge(state(T1), state(T2))) == derive(state(T1 ∪ T2))

This is the property `ExhaustiveIndex.update_table` relies on. We exercise:
    - `_build_num_state_records` / `_build_cat_state_records`
    - `_merge_num_dicts` / `_merge_cat_dicts`
    - `_derive_features_from_state` / `_postprocess_feature_df`

…and compare the result to the same pipeline applied to the full table (no split).

Run:
    python -m pytest tests/test_index_update_helpers.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from matryoshka.index import ExactMedianBackend, ExhaustiveIndex


def _records_to_num_dict(records: pl.DataFrame | None) -> dict:
    out: dict = {}
    if records is None:
        return out
    for r in records.iter_rows(named=True):
        arr = np.frombuffer(r['values_'], dtype=np.float64) if r['values_'] else np.empty(0, dtype=np.float64)
        out[(r['key'], r['col_name'])] = {
            'count': int(r['count_']),
            'sum':   float(r['sum_']),
            'min':   float(r['min_']),
            'max':   float(r['max_']),
            'values': arr,
        }
    return out


def _records_to_cat_dict(records: pl.DataFrame | None) -> dict:
    out: dict = {}
    if records is None:
        return out
    for r in records.iter_rows(named=True):
        out.setdefault(r['col_name'], {}).setdefault(r['key'], {})[r['category']] = int(r['raw_count'])
    return out


def _make_index() -> ExhaustiveIndex:
    # We don't need a real DB, just the helper methods. Provide harmless table names.
    idx = ExhaustiveIndex.__new__(ExhaustiveIndex)
    # Bypass __init__ entirely to avoid logging/SSH/DB setup. Only the two
    # attributes the state helpers read need to be set by hand.
    idx.feature_extraction = False
    idx._median_backend = ExactMedianBackend()
    return idx


def _state_dicts(idx: ExhaustiveIndex, df: pl.DataFrame, group_col: str, num_cols: list[str], cat_cols: list[str]) -> tuple[dict, dict]:
    num_records = idx._build_num_state_records(df, group_col, 0, num_cols, table_index=0)
    cat_records = idx._build_cat_state_records(df, group_col, 0, cat_cols, table_index=0)
    return _records_to_num_dict(num_records), _records_to_cat_dict(cat_records)


def _derive(idx: ExhaustiveIndex, num_d: dict, cat_d: dict, group_col: str, num_cols: list[str], cat_cols: list[str]) -> pl.DataFrame:
    df = idx._derive_features_from_state(num_d, cat_d, group_col, num_cols, cat_cols)
    return idx._postprocess_feature_df(df, group_col)


def _assert_equal(a: pl.DataFrame, b: pl.DataFrame, label: str = '') -> None:
    a = a.sort(a.columns[0]).select(sorted(a.columns))
    b = b.sort(b.columns[0]).select(sorted(b.columns))
    assert a.columns == b.columns, f'{label}: columns differ\n  inc:  {a.columns}\n  full: {b.columns}'
    assert a.height == b.height,   f'{label}: heights differ {a.height} vs {b.height}'
    for col in a.columns:
        if a[col].dtype.is_numeric() and b[col].dtype.is_numeric():
            av = np.asarray(a[col].cast(pl.Float64).fill_null(np.nan).to_list())
            bv = np.asarray(b[col].cast(pl.Float64).fill_null(np.nan).to_list())
            mask = ~(np.isnan(av) & np.isnan(bv))
            if not np.allclose(av[mask], bv[mask], atol=1e-9, equal_nan=False):
                raise AssertionError(f'{label}: column {col} mismatch\n  inc ={av}\n  full={bv}')
        else:
            if a[col].to_list() != b[col].to_list():
                raise AssertionError(f'{label}: column {col} mismatch')


def _synth(n_rows: int, n_keys: int, n_num: int, n_cat: int, n_cats_per_col: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {'k': rng.integers(0, n_keys, size=n_rows).astype(str)}
    for i in range(n_num):
        data[f'n{i}'] = rng.normal(loc=i, scale=2.0, size=n_rows)
    for i in range(n_cat):
        data[f'c{i}'] = rng.integers(0, n_cats_per_col, size=n_rows).astype(str)
    return pl.DataFrame(data)


def test_full_equivalence():
    idx = _make_index()
    T = _synth(2000, 30, n_num=3, n_cat=2, n_cats_per_col=8, seed=1)
    num_cols, cat_cols = ['n0', 'n1', 'n2'], ['c0', 'c1']
    nd, cd = _state_dicts(idx, T, 'k', num_cols, cat_cols)
    full = _derive(idx, nd, cd, 'k', num_cols, cat_cols)

    # No split — just verify state→derive matches itself (sanity).
    nd2, cd2 = _state_dicts(idx, T, 'k', num_cols, cat_cols)
    again = _derive(idx, nd2, cd2, 'k', num_cols, cat_cols)
    _assert_equal(again, full, 'full_equivalence')
    print('PASS  test_full_equivalence')


def test_split_then_merge():
    idx = _make_index()
    T = _synth(3000, 40, n_num=3, n_cat=2, n_cats_per_col=8, seed=2)
    num_cols, cat_cols = ['n0', 'n1', 'n2'], ['c0', 'c1']

    half = T.height // 2
    T1, T2 = T.slice(0, half), T.slice(half, T.height - half)

    full_nd, full_cd = _state_dicts(idx, T, 'k', num_cols, cat_cols)
    full = _derive(idx, full_nd, full_cd, 'k', num_cols, cat_cols)

    # Build state from T1, then merge T2 delta.
    nd1, cd1 = _state_dicts(idx, T1, 'k', num_cols, cat_cols)
    delta_num = idx._build_num_state_records(T2, 'k', 0, num_cols, table_index=0)
    delta_cat = idx._build_cat_state_records(T2, 'k', 0, cat_cols, table_index=0)
    merged_nd = idx._merge_num_dicts(nd1, delta_num)
    merged_cd = idx._merge_cat_dicts(cd1, delta_cat)
    merged = _derive(idx, merged_nd, merged_cd, 'k', num_cols, cat_cols)

    _assert_equal(merged, full, 'split_then_merge')
    print('PASS  test_split_then_merge')


def test_three_chunk_merge():
    idx = _make_index()
    T = _synth(2400, 25, n_num=2, n_cat=3, n_cats_per_col=10, seed=4)
    num_cols, cat_cols = ['n0', 'n1'], ['c0', 'c1', 'c2']
    a, b = T.height // 3, 2 * T.height // 3
    T1, T2, T3 = T.slice(0, a), T.slice(a, b - a), T.slice(b, T.height - b)

    full_nd, full_cd = _state_dicts(idx, T, 'k', num_cols, cat_cols)
    full = _derive(idx, full_nd, full_cd, 'k', num_cols, cat_cols)

    nd, cd = _state_dicts(idx, T1, 'k', num_cols, cat_cols)
    for chunk in (T2, T3):
        dn = idx._build_num_state_records(chunk, 'k', 0, num_cols, table_index=0)
        dc = idx._build_cat_state_records(chunk, 'k', 0, cat_cols, table_index=0)
        nd = idx._merge_num_dicts(nd, dn)
        cd = idx._merge_cat_dicts(cd, dc)
    merged = _derive(idx, nd, cd, 'k', num_cols, cat_cols)
    _assert_equal(merged, full, 'three_chunk_merge')
    print('PASS  test_three_chunk_merge')


def test_new_categories_grow_N_c():
    idx = _make_index()
    rng = np.random.default_rng(3)
    n = 1500
    keys = rng.integers(0, 20, size=n).astype(str)
    nums = rng.normal(size=n)
    cats_t1 = rng.integers(0, 5, size=n // 2).astype(str)
    cats_t2 = rng.integers(5, 9, size=n - n // 2).astype(str)  # disjoint vocab
    T1 = pl.DataFrame({'k': keys[: n // 2], 'n0': nums[: n // 2], 'c0': cats_t1})
    T2 = pl.DataFrame({'k': keys[n // 2:],  'n0': nums[n // 2:],  'c0': cats_t2})
    T = pl.concat([T1, T2])
    num_cols, cat_cols = ['n0'], ['c0']

    full_nd, full_cd = _state_dicts(idx, T, 'k', num_cols, cat_cols)
    full = _derive(idx, full_nd, full_cd, 'k', num_cols, cat_cols)

    nd1, cd1 = _state_dicts(idx, T1, 'k', num_cols, cat_cols)
    dn = idx._build_num_state_records(T2, 'k', 0, num_cols, table_index=0)
    dc = idx._build_cat_state_records(T2, 'k', 0, cat_cols, table_index=0)
    merged_nd = idx._merge_num_dicts(nd1, dn)
    merged_cd = idx._merge_cat_dicts(cd1, dc)
    merged = _derive(idx, merged_nd, merged_cd, 'k', num_cols, cat_cols)

    # N_c must reflect all 9 distinct categories across T1∪T2.
    seen = set()
    for per_cat in merged_cd['c0'].values():
        seen.update(per_cat.keys())
    assert len(seen) == 9, f'expected N_c=9, got {len(seen)}'

    _assert_equal(merged, full, 'new_categories_grow_N_c')
    print('PASS  test_new_categories_grow_N_c')


def test_round_trip_dict_to_df():
    idx = _make_index()
    T = _synth(800, 15, n_num=2, n_cat=2, n_cats_per_col=5, seed=7)
    num_cols, cat_cols = ['n0', 'n1'], ['c0', 'c1']

    nd, cd = _state_dicts(idx, T, 'k', num_cols, cat_cols)

    # Roundtrip dict -> DataFrame -> records -> dict and compare.
    num_df = idx._dict_to_num_state_df(nd, table_index=0, group_col_index=0)
    cat_df = idx._dict_to_cat_state_df(cd, table_index=0, group_col_index=0)
    nd_rt = _records_to_num_dict(num_df)
    cd_rt = _records_to_cat_dict(cat_df)

    assert set(nd.keys()) == set(nd_rt.keys()), 'num_state key set differs after roundtrip'
    for k in nd:
        a, b = nd[k], nd_rt[k]
        assert a['count'] == b['count'] and a['sum'] == b['sum'] and a['min'] == b['min'] and a['max'] == b['max']
        assert np.array_equal(np.sort(a['values']), np.sort(b['values']))
    assert cd == cd_rt, 'cat_state differs after roundtrip'
    print('PASS  test_round_trip_dict_to_df')


if __name__ == '__main__':
    test_full_equivalence()
    test_split_then_merge()
    test_three_chunk_merge()
    test_new_categories_grow_N_c()
    test_round_trip_dict_to_df()
    print('\nAll update-helper tests passed.')
