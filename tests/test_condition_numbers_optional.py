"""The condition numbers in the conditional scorers' diagnostics are each a full
SVD, computed once per candidate per iteration by forward selection, where they
dominated the runtime. They are reported only when asked for, and asking changes
nothing else."""
import numpy as np

from matryoshka.selection.models import (
    conditional_fisher_score,
    conditional_regression_score,
)


def _classification_inputs(p=40, q=3, k=2, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(k, p + q))
    covs = np.stack([
        np.eye(p + q) + 0.1 * (lambda a: a @ a.T)(rng.normal(size=(p + q, p + q)))
        for _ in range(k)
    ])
    counts = np.array([120, 80])
    return dict(
        candidate_class_means=means[:, p:],
        candidate_class_covs=covs[:, p:, p:],
        class_counts=counts,
        current_class_means=means[:, :p],
        current_class_covs=covs[:, :p, :p],
        cross_class_covs=covs[:, p:, :p],
    )


def _regression_inputs(n=200, p=40, q=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    z = rng.normal(size=(n, q))
    y = rng.normal(size=n)
    return dict(
        current_xtx=x.T @ x,
        current_xty=x.T @ y,
        candidate_ztz=z.T @ z,
        candidate_zty=z.T @ y,
        yty=float(y @ y),
        n=n,
        cross_ztx=z.T @ x,
        p_current=p,
        q_candidate=q,
    )


def test_classification_condition_numbers_are_opt_in():
    inputs = _classification_inputs()
    off = conditional_fisher_score(**inputs)
    on = conditional_fisher_score(**inputs, with_diagnostics=True)

    assert off['diagnostics']['cond_S_XX_reg'] is None
    assert off['diagnostics']['cond_S_Z_given_X_reg'] is None
    assert on['diagnostics']['cond_S_XX_reg'] > 0
    assert on['diagnostics']['cond_S_Z_given_X_reg'] > 0

    # the score is what forward selection ranks on: it must not depend on this
    assert off['incremental_score'] == on['incremental_score']
    assert off['raw_score'] == on['raw_score']


def test_regression_condition_numbers_are_opt_in():
    inputs = _regression_inputs()
    off = conditional_regression_score(**inputs)
    on = conditional_regression_score(**inputs, with_diagnostics=True)

    assert off['diagnostics']['cond_A_reg'] is None
    assert off['diagnostics']['cond_G_Z_given_X_reg'] is None
    assert on['diagnostics']['cond_A_reg'] > 0
    assert on['diagnostics']['cond_G_Z_given_X_reg'] > 0

    for key, value in off.items():
        if key == 'diagnostics' or isinstance(value, np.ndarray):
            continue
        assert value == on[key], key


def test_no_current_set_leaves_the_current_set_condition_number_unset():
    """With no selected features there is no A to condition, asked for or not."""
    rng = np.random.default_rng(1)
    means = rng.normal(size=(2, 2))
    covs = np.stack([np.eye(2), np.eye(2)])
    for flag in (False, True):
        out = conditional_fisher_score(means, covs, np.array([10, 10]),
                                       with_diagnostics=flag)
        assert out['diagnostics']['cond_S_XX_reg'] is None
