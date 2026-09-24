"""Units of the class statistics that the classification proxies score.

Every class is sketched separately: its Gram matrix is standardized by the class's own feature standard deviations,
whereas its feature means stay in the features' units. Scoring the two together made class separability grow with the
squared units of a feature, so that, e.g., a distance in kilometers outscored a far more informative 0/1 feature.
These tests build per-class statistics as the sketch processors do and pin that the scores no longer depend on units.
"""
import numpy as np
import pytest

from matryoshka.selection.models import conditional_fisher_score, standardized_class_statistics


def class_statistics(X, y, with_stds=True):
    '''Per-class statistics as emitted by the sketch processors (see `SketchProcessor._join_aug_table`).'''
    statistics = []
    for label in np.unique(y):
        Xk = X[y == label]
        n = len(Xk)
        centered = (Xk - Xk.mean(axis=0)).T @ (Xk - Xk.mean(axis=0))
        stds = np.sqrt(np.diag(centered) / n)
        stats = {
            'joint_cofactor_matrix': centered / np.outer(stds, stds),
            'features_mean': Xk.mean(axis=0),
            'joint_count': n,
        }
        if with_stds:
            stats['features_std'] = stds
        statistics.append(stats)
    return statistics


def data(seed=0, n=4000, distance_scale=1000.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # weakly informative, large units (like a flight distance in km)
    distance = distance_scale * (1.0 + 0.2 * y + rng.normal(0.0, 1.0, n))
    # strongly informative 0/1 feature (like a travel type)
    indicator = (rng.random(n) < np.where(y == 1, 0.8, 0.2)).astype(float)
    return np.column_stack([distance, indicator]), y


def standalone_scores(statistics):
    means, covariances, counts = statistics
    return [
        conditional_fisher_score(means[:, [j]], covariances[:, [j]][:, :, [j]], counts)['raw_score']
        for j in range(means.shape[1])
    ]


def test_scores_do_not_depend_on_units():
    X, y = data()
    scores = standalone_scores(standardized_class_statistics(class_statistics(X, y)))
    X_scaled = X * np.array([1e-3, 1e3])
    scores_scaled = standalone_scores(standardized_class_statistics(class_statistics(X_scaled, y)))
    assert scores == pytest.approx(scores_scaled, rel=1e-6)


def test_the_more_informative_feature_scores_higher_whatever_its_units():
    X, y = data()
    distance_score, indicator_score = standalone_scores(standardized_class_statistics(class_statistics(X, y)))
    assert indicator_score > distance_score
    # Before the fix (per-class standardized covariances, raw means) the ranking was inverted by the units
    distance_score, indicator_score = standalone_scores(standardized_class_statistics(class_statistics(X, y, with_stds=False)))
    assert distance_score > 1000 * indicator_score


def test_constant_features_do_not_look_separating():
    # A feature equal to 3 on every key (e.g., the level of every country in a location table), with the per-class
    # statistics the sketch processors emit for it: no variance, so a unit standard deviation, and equal means
    statistics = [
        {'joint_cofactor_matrix': np.zeros((1, 1)), 'features_mean': np.array([3.0]), 'features_std': np.array([1.0]), 'joint_count': 151},
        {'joint_cofactor_matrix': np.zeros((1, 1)), 'features_mean': np.array([3.0]), 'features_std': np.array([1.0]), 'joint_count': 32},
    ]
    means, covariances, counts = standardized_class_statistics(statistics)
    assert conditional_fisher_score(means, covariances, counts)['raw_score'] < 1e-12


def test_covariances_are_the_class_covariances_on_the_total_scale():
    X, y = data()
    means, covariances, counts = standardized_class_statistics(class_statistics(X, y))
    total_std = X.std(axis=0)
    for k, label in enumerate(np.unique(y)):
        Xk = X[y == label]
        assert means[k] == pytest.approx(Xk.mean(axis=0) / total_std)
        assert covariances[k] == pytest.approx(np.cov(Xk.T, bias=True) / np.outer(total_std, total_std))
    assert counts.tolist() == [np.sum(y == label) for label in np.unique(y)]
