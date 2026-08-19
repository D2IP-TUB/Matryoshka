"""Lazy resolution of proxy models, search strategies and baseline augmenters.

These registries replaced the ``globals()[name]`` lookups the original code used,
which required every optional dependency to be importable before any name could
be resolved. The tests pin the two properties that motivated the change: an
unknown name fails loudly, and a name whose dependency is absent fails with a
message that says which dependency.
"""
import pytest

import matryoshka as mk
from matryoshka.selection import registry


def test_core_models_resolve():
    for name in ('RegressionQR', 'RegressionCholesky', 'IncrementalRegressionFGS',
                 'ClassificationCholesky'):
        assert isinstance(registry.resolve_model(name), type)


def test_core_strategies_resolve():
    # The greedy strategies are `@ray.remote` actor classes, so they resolve to
    # a Ray ActorClass rather than a plain type; `JoinSelection` calls
    # `.remote(...)` on them.
    for name in ('ForwardSelection', 'BackwardElimination'):
        resolved = registry.resolve_strategy(name)
        assert hasattr(resolved, 'remote'), f'{name} is not callable as a Ray actor'


def test_unknown_model_lists_the_registered_ones():
    with pytest.raises(KeyError, match='ForwardSelection|registered'):
        registry.resolve_strategy('NoSuchStrategy')


def test_available_names_are_sorted_and_non_empty():
    assert registry.available_models() == sorted(registry.available_models())
    assert registry.available_strategies() == sorted(registry.available_strategies())
    assert mk.available_baselines() == sorted(mk.available_baselines())


def test_registering_a_model_by_class():
    class Dummy:
        pass

    registry.register_model('DummyProxy', Dummy)
    try:
        assert registry.resolve_model('DummyProxy') is Dummy
        assert 'DummyProxy' in registry.available_models()
    finally:
        registry._MODELS.pop('DummyProxy', None)
        registry._CACHE.pop('model:DummyProxy', None)


def test_missing_dependency_is_named_in_the_error():
    registry.register_model('Phantom', 'no_such_module_xyz:Thing')
    try:
        with pytest.raises(ImportError, match='no_such_module_xyz'):
            registry.resolve_model('Phantom')
    finally:
        registry._MODELS.pop('Phantom', None)


def test_unknown_baseline_lists_the_registered_ones():
    with pytest.raises(KeyError, match='ArdaAugmenter|registered'):
        mk.resolve_baseline('NoSuchAugmenter')


def test_registering_a_baseline_by_target_string():
    mk.register_baseline('PhantomAugmenter', 'no_such_module_xyz:Augmenter')
    with pytest.raises(ImportError, match='baselines'):
        mk.resolve_baseline('PhantomAugmenter')


def test_every_configurable_strategy_is_registered():
    """DiscoveryConfig must not advertise a strategy nothing can resolve."""
    config = mk.DiscoveryConfig(
        task='classification', ranking='passthrough', strategy='ForwardSelection',
        model='ClassificationCholesky',
        params={'metric': 'conditional_mahalanobis', 'tol': 0.05},
    )
    advertised = set()
    for rules in config._task_rules.values():
        for entry in rules.get('ranking_strategy_rules', {}).values():
            advertised.update(s for s in entry.get('allowed_strategy', []) if s)
    # The imputation task is documented as unimplemented and is excluded here.
    advertised.discard('IterativeImputer')
    assert advertised <= set(registry.available_strategies()), (
        f'advertised but unregistered: {advertised - set(registry.available_strategies())}'
    )
