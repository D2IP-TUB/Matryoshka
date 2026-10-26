"""The paper's evaluation protocol: leakage filter, scoring and preparation.

The leakage filter decides which lake columns may reach the downstream model,
so its two criteria are pinned on frames where the answer is known. Tests that
need the ``paper`` extra are skipped without it.
"""
import numpy as np
import polars as pl
import pytest

pd = pytest.importorskip('pandas')

from matryoshka.evaluation import (  # noqa: E402
    PAPER_SETUPS,
    detect_target_leakage,
    prepare_query_table_paper,
    score_random_forest,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_drops_a_column_correlated_above_the_threshold(rng):
    y = rng.normal(size=500)
    X = pd.DataFrame({'copy': y + rng.normal(scale=1e-3, size=500), 'noise': rng.normal(size=500), 'y': y})
    assert detect_target_leakage(X, 'y', 'regression') == ['copy']


def test_drops_every_component_of_an_additive_identity(rng):
    # Global_Sales = NA + EU + JP + Other: no single component crosses the
    # correlation threshold, but together they reproduce the target exactly.
    parts = {name: rng.gamma(2.0, size=1000) for name in ('na', 'eu', 'jp', 'other')}
    X = pd.DataFrame({**parts, 'year': rng.integers(1990, 2020, size=1000).astype(float)})
    X['global'] = sum(parts.values())
    leaky = detect_target_leakage(X, 'global', 'regression')
    assert sorted(leaky) == ['eu', 'jp', 'na', 'other']


def test_keeps_informative_but_not_leaking_columns(rng):
    x = rng.normal(size=500)
    X = pd.DataFrame({'signal': x, 'y': 0.8 * x + rng.normal(scale=0.6, size=500)})
    assert detect_target_leakage(X, 'y', 'regression') == []


def test_classification_targets_are_compared_by_class_code(rng):
    labels = rng.choice(['a', 'b', 'c'], size=600)
    codes = pd.Categorical(labels).codes.astype(float)
    X = pd.DataFrame({'encoded_target': codes, 'noise': rng.normal(size=600), 'label': labels})
    assert detect_target_leakage(X, 'label', 'classification') == ['encoded_target']


def test_constant_targets_and_non_numeric_columns_are_ignored(rng):
    X = pd.DataFrame({'text': ['x'] * 50, 'num': rng.normal(size=50), 'y': np.ones(50)})
    assert detect_target_leakage(X, 'y', 'regression') == []


def test_scores_regression_and_multiclass(rng):
    x = rng.normal(size=(400, 3))
    reg = pd.DataFrame(x, columns=['a', 'b', 'c']).assign(y=2 * x[:, 0] + rng.normal(scale=0.1, size=400))
    scores = score_random_forest(reg, 'y', 'regression', n_jobs=1)
    assert set(scores) == {'r2', 'rmse'} and scores['r2'] > 0.5

    cls = pd.DataFrame(x, columns=['a', 'b', 'c']).assign(y=np.digitize(x[:, 0], [-0.5, 0.5]))
    scores = score_random_forest(cls, 'y', 'classification', n_jobs=1)
    assert set(scores) == {'accuracy', 'f1_weighted'} and scores['accuracy'] > 0.7


def test_scores_binary_classification(rng):
    x = rng.normal(size=(300, 2))
    X = pd.DataFrame(x, columns=['a', 'b']).assign(y=(x[:, 0] > 0).astype(int))
    assert set(score_random_forest(X, 'y', 'classification', n_jobs=1)) == {'accuracy', 'f1', 'roc_auc'}


@pytest.fixture
def query_table(rng):
    n = 300
    return pl.DataFrame({
        'street': rng.choice(['Main St', 'Oak Ave', 'Elm St.', None], size=n).tolist(),
        'site': rng.choice(['front', 'back', None], size=n).tolist(),
        'diameter': [None if i % 17 == 0 else float(v) for i, v in enumerate(rng.gamma(3.0, size=n))],
        'condition': rng.choice(['good', 'fair', 'poor'], size=n).tolist(),
    }).filter(pl.col('street').is_not_null())


@pytest.mark.parametrize('setup', PAPER_SETUPS)
def test_prepares_the_layout_discovery_expects(query_table, setup):
    pytest.importorskip('autogluon.features')
    out = prepare_query_table_paper(query_table, 'street', 'condition', categorical_features=['street', 'site'],
                                    numeric_features=['diameter'], task='classification', setup=setup)
    assert out.columns[0] == 'street' and out.columns[-1] == 'condition'
    assert out.schema['street'] == pl.String
    assert set(out['street'].unique()) <= {'mainst', 'oakave', 'elmst'}
    assert all(out.schema[c].is_numeric() for c in out.columns[1:])
    assert out.null_count().sum_horizontal().item() == 0
    assert sorted(out['condition'].unique().to_list()) == [0.0, 1.0, 2.0]
    feature_columns = out.columns[1:-1]
    if setup == 'categorical':
        assert not any('diameter' in c for c in feature_columns)
    if setup == 'join_key':
        assert feature_columns == ['cat__street']
    if setup == 'binned':
        assert any(c.startswith('cat__diameter_') for c in feature_columns)


def test_rejects_an_unknown_setup(query_table):
    with pytest.raises(ValueError, match='setup'):
        prepare_query_table_paper(query_table, 'street', 'condition', categorical_features=['site'], setup='raw')
