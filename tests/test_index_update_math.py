"""Validate the math of the index update module without touching the DB.

Mirrors the aggregations performed by ExhaustiveIndex._numeric_query and
_non_numeric_query (no feature_extraction path).

Validated:
  * derive(state(T))            == aggregate_full(T)
  * derive(merge(state(T1), state(T2))) == aggregate_full(T1 ∪ T2)
  * Updates that introduce new categorical values (N_c grows) — the derived
    row for an UNCHANGED key still matches the full rebuild because raw
    counts are kept in state and re-divided by the new N_c.

Out of scope (deliberately):
  * Null imputation. Production fills numeric nulls with the GLOBAL mean and
    categorical nulls with the GLOBAL mode before group-by. The fill values
    can shift after an update, so any naive row-storage of imputed values
    would drift. The update module must persist *raw* values (or sum+count
    of raw values) and impute at derive time. Tested data has no nulls.
  * Median approximation. We keep exact value lists here as ground truth;
    in production a t-digest replaces the list and introduces a bounded error.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import polars as pl

# ---------- ground truth: full-rebuild aggregates ----------

def aggregate_numeric_full(df: pl.DataFrame, group_col: str, num_cols: list[str]) -> pl.DataFrame:
    if not num_cols:
        return df.select(group_col).unique()
    return (
        df.group_by(group_col)
        .agg(
            [pl.col(c).mean().alias(f'{c}_mean')   for c in num_cols]
          + [pl.col(c).median().alias(f'{c}_median') for c in num_cols]
          + [pl.col(c).max().alias(f'{c}_max')     for c in num_cols]
          + [pl.col(c).min().alias(f'{c}_min')     for c in num_cols]
        )
    )


def aggregate_categorical_full(df: pl.DataFrame, group_col: str, cat_cols: list[str]) -> pl.DataFrame:
    """Mirror _non_numeric_query exactly:
      counts_per_key[c] = pl.col(c).unique_counts()        # vector r_k
      r_k_div = r_k / N_c                                  # element-wise
      c_mean    = sum(r_k_div) / N_c   == sum(r_k) / N_c**2
      c_max     = max(r_k_div)         == max(r_k) / N_c
      c_nunique = n_unique(r_k_div)    == n_unique(r_k)
    """
    if not cat_cols:
        return df.select(group_col).unique()
    n_unique = {c: df[c].n_unique() for c in cat_cols}
    g = df.group_by(group_col).agg([pl.col(c).unique_counts() for c in cat_cols])
    return g.select(
        [group_col]
      + [(pl.col(c).list.sum().cast(pl.Float64) / (n_unique[c] ** 2)).alias(f'{c}_mean')   for c in cat_cols]
      + [(pl.col(c).list.max().cast(pl.Float64) /  n_unique[c]      ).alias(f'{c}_max')    for c in cat_cols]
      + [pl.col(c).list.n_unique().cast(pl.Int64).alias(f'{c}_nunique')                     for c in cat_cols]
    )


def aggregate_full(df: pl.DataFrame, group_col: str, num_cols: list[str], cat_cols: list[str]) -> pl.DataFrame:
    n = aggregate_numeric_full(df, group_col, num_cols)
    c = aggregate_categorical_full(df, group_col, cat_cols)
    return n.join(c, on=group_col, how='inner').sort(group_col)


# ---------- incremental state ----------

@dataclass
class NumericKeyState:
    count: int = 0
    sum_: float = 0.0
    min_: float = math.inf
    max_: float = -math.inf
    values: list = field(default_factory=list)  # exact ground truth for median


@dataclass
class State:
    group_col: str
    num_cols: list
    cat_cols: list
    num_state: dict = field(default_factory=dict)
    cat_state: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    cat_dict: dict = field(default_factory=lambda: defaultdict(set))

    def update(self, df: pl.DataFrame) -> None:
        for c in self.num_cols:
            for key, val in df.select(self.group_col, c).iter_rows():
                if val is None:
                    continue
                st = self.num_state.setdefault((key, c), NumericKeyState())
                st.count += 1
                fv = float(val)
                st.sum_ += fv
                st.min_ = min(st.min_, fv)
                st.max_ = max(st.max_, fv)
                st.values.append(fv)
        for c in self.cat_cols:
            for key, val in df.select(self.group_col, c).iter_rows():
                if val is None:
                    continue
                self.cat_state[c][key][val] += 1
                self.cat_dict[c].add(val)

    def derive(self) -> pl.DataFrame:
        all_keys = set()
        for (k, _c) in self.num_state:
            all_keys.add(k)
        for c in self.cat_cols:
            all_keys.update(self.cat_state[c].keys())

        def has_all_num(k):
            return all((k, c) in self.num_state for c in self.num_cols)
        def has_all_cat(k):
            return all(k in self.cat_state[c] for c in self.cat_cols)
        keys = sorted(k for k in all_keys if has_all_num(k) and has_all_cat(k))

        rows = []
        for k in keys:
            row = {self.group_col: k}
            for c in self.num_cols:
                st = self.num_state[(k, c)]
                row[f'{c}_mean']   = st.sum_ / st.count
                vs = sorted(st.values)
                n = len(vs)
                row[f'{c}_median'] = vs[n // 2] if n % 2 == 1 else 0.5 * (vs[n // 2 - 1] + vs[n // 2])
                row[f'{c}_max']    = st.max_
                row[f'{c}_min']    = st.min_
            for c in self.cat_cols:
                N_c = len(self.cat_dict[c])
                counts = list(self.cat_state[c][k].values())
                row[f'{c}_mean']    = sum(counts) / (N_c ** 2)
                row[f'{c}_max']     = max(counts) / N_c
                row[f'{c}_nunique'] = len(set(counts))
            rows.append(row)
        return pl.DataFrame(rows).sort(self.group_col)


def assert_frames_equal(a: pl.DataFrame, b: pl.DataFrame, *, atol: float = 1e-9, label: str = '') -> None:
    a = a.sort(a.columns[0]).select(sorted(a.columns))
    b = b.sort(b.columns[0]).select(sorted(b.columns))
    assert a.columns == b.columns, f'{label}: column mismatch {a.columns} vs {b.columns}'
    assert a.height == b.height, f'{label}: height mismatch {a.height} vs {b.height}'
    for col in a.columns:
        if a[col].dtype.is_numeric() and b[col].dtype.is_numeric():
            av = np.asarray(a[col].cast(pl.Float64).fill_null(np.nan).to_list())
            bv = np.asarray(b[col].cast(pl.Float64).fill_null(np.nan).to_list())
            mask_both_nan = np.isnan(av) & np.isnan(bv)
            if not np.allclose(av[~mask_both_nan], bv[~mask_both_nan], atol=atol, equal_nan=False):
                raise AssertionError(f'{label}: column {col} differs\n  inc ={av}\n  full={bv}')
        else:
            if a[col].to_list() != b[col].to_list():
                raise AssertionError(f'{label}: column {col} differs')


def synth_table(n_rows, n_keys, n_num, n_cat, n_cats_per_col, seed=0, int_numerics=False):
    rng = np.random.default_rng(seed)
    data = {'k': rng.integers(0, n_keys, size=n_rows)}
    for i in range(n_num):
        if int_numerics:
            data[f'n{i}'] = rng.integers(-50, 50, size=n_rows)
        else:
            data[f'n{i}'] = rng.normal(loc=i, scale=2.0, size=n_rows)
    for i in range(n_cat):
        data[f'c{i}'] = rng.integers(0, n_cats_per_col, size=n_rows).astype(str)
    return pl.DataFrame(data)


def test_full_equivalence():
    T = synth_table(2000, 40, n_num=3, n_cat=2, n_cats_per_col=8, seed=1)
    num_cols = [c for c in T.columns if c.startswith('n')]
    cat_cols = [c for c in T.columns if c.startswith('c')]
    full = aggregate_full(T, 'k', num_cols, cat_cols)
    state = State('k', num_cols, cat_cols); state.update(T)
    assert_frames_equal(state.derive(), full, label='full_equivalence')
    print('PASS  test_full_equivalence')


def test_split_update():
    T = synth_table(3000, 50, n_num=3, n_cat=2, n_cats_per_col=8, seed=2)
    num_cols = [c for c in T.columns if c.startswith('n')]
    cat_cols = [c for c in T.columns if c.startswith('c')]
    half = T.height // 2
    T1, T2 = T.slice(0, half), T.slice(half, T.height - half)
    full = aggregate_full(T, 'k', num_cols, cat_cols)
    state = State('k', num_cols, cat_cols)
    state.update(T1); state.update(T2)
    assert_frames_equal(state.derive(), full, label='split_update')
    print('PASS  test_split_update')


def test_three_chunk_update():
    T = synth_table(2400, 35, n_num=2, n_cat=3, n_cats_per_col=10, seed=4)
    num_cols = [c for c in T.columns if c.startswith('n')]
    cat_cols = [c for c in T.columns if c.startswith('c')]
    a, b = T.height // 3, 2 * T.height // 3
    T1, T2, T3 = T.slice(0, a), T.slice(a, b - a), T.slice(b, T.height - b)
    full = aggregate_full(T, 'k', num_cols, cat_cols)
    state = State('k', num_cols, cat_cols)
    for chunk in (T1, T2, T3):
        state.update(chunk)
    assert_frames_equal(state.derive(), full, label='three_chunk')
    print('PASS  test_three_chunk_update')


def test_new_category_grows_N_c():
    rng = np.random.default_rng(3)
    n = 1500
    keys = rng.integers(0, 30, size=n)
    nums = rng.normal(size=n)
    cats_t1 = rng.integers(0, 5, size=n // 2).astype(str)
    cats_t2 = rng.integers(5, 9, size=n - n // 2).astype(str)
    T1 = pl.DataFrame({'k': keys[: n // 2], 'n0': nums[: n // 2], 'c0': cats_t1})
    T2 = pl.DataFrame({'k': keys[n // 2:],  'n0': nums[n // 2:],  'c0': cats_t2})
    T = pl.concat([T1, T2])
    full = aggregate_full(T, 'k', ['n0'], ['c0'])
    state = State('k', ['n0'], ['c0']); state.update(T1); state.update(T2)
    assert_frames_equal(state.derive(), full, label='new_category_grows_N_c')
    assert len(state.cat_dict['c0']) == 9, f"expected N_c=9, got {len(state.cat_dict['c0'])}"
    print('PASS  test_new_category_grows_N_c')


def test_integer_numerics_and_singletons():
    rng = np.random.default_rng(5)
    keys = list(range(20)) + list(rng.integers(0, 20, size=200))
    nums = rng.integers(-50, 50, size=len(keys))
    cats = rng.integers(0, 4, size=len(keys)).astype(str)
    T = pl.DataFrame({'k': keys, 'n0': nums, 'c0': cats})
    full = aggregate_full(T, 'k', ['n0'], ['c0'])
    state = State('k', ['n0'], ['c0']); state.update(T)
    assert_frames_equal(state.derive(), full, label='integers_singletons')
    print('PASS  test_integer_numerics_and_singletons')


def test_useful_columns_projection():
    T = synth_table(1500, 25, n_num=4, n_cat=3, n_cats_per_col=6, seed=6)
    useful_num = ['n0', 'n2']
    useful_cat = ['c0']
    full = aggregate_full(T, 'k', useful_num, useful_cat)
    state = State('k', useful_num, useful_cat)
    state.update(T.select(['k', *useful_num, *useful_cat]))
    assert_frames_equal(state.derive(), full, label='useful_columns_projection')
    print('PASS  test_useful_columns_projection')


if __name__ == '__main__':
    test_full_equivalence()
    test_split_update()
    test_three_chunk_update()
    test_new_category_grows_N_c()
    test_integer_numerics_and_singletons()
    test_useful_columns_projection()
    print('\nAll math validations passed.')
