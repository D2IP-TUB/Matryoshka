"""Centering of the target sum of squares that the regression proxies score.

The sketch processors hand the proxies standardized, centered cross-products X'y; the target sum of squares y'y must be
centered too. Uncentered, every residual sum of squares carried the constant n * mean(y)^2 (10 to 23 times the target's
variation on the web tables), which shrank the relative improvements of the candidates and let the GCV penalty reject
any second feature.
"""
import numpy as np
import pytest

from matryoshka.selection.models import conditional_regression_score
from matryoshka.selection.sketch_processing import _centered_target_ss, _standardize_gram_inputs_pure


def sketch_inputs(X, y):
    '''Standardized Gram matrix and cross-products, as the sketch processors compute them.'''
    n = len(y)
    return _standardize_gram_inputs_pure(X.T @ X, X.T @ y, X.sum(axis=0), y.sum(), n)


def test_centered_target_sum_of_squares():
    y = np.array([4.0, 6.0, 5.0, 9.0])
    assert _centered_target_ss(np.sum(y ** 2), np.array([y.sum()]), len(y)) == pytest.approx(np.sum((y - y.mean()) ** 2))


def test_residual_sum_of_squares_matches_least_squares_with_intercept():
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(5.0, 2.0, (n, 3))
    y = 6.0 + X @ np.array([0.5, -0.2, 0.0]) + rng.normal(0.0, 1.0, n)
    xtx, xty = sketch_inputs(X, y)
    yty = _centered_target_ss(np.sum(y ** 2), np.array([y.sum()]), n)

    result = conditional_regression_score(
        current_xtx=None, current_xty=None, candidate_ztz=xtx, candidate_zty=xty.reshape(-1), yty=yty, n=n,
        lambda_x=1e-9, lambda_z=1e-9,
    )
    design = np.column_stack([np.ones(n), X])
    residuals = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    assert result['rss_new'] == pytest.approx(np.sum(residuals ** 2), rel=1e-6)
    assert result['rss_current'] == pytest.approx(np.sum((y - y.mean()) ** 2), rel=1e-9)
