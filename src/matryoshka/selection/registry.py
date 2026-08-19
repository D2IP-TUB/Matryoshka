"""Name-to-class resolution for proxy models and selection strategies.

:class:`matryoshka.config.DiscoveryConfig` names the proxy model and the search
strategy as strings, and the planner passes those strings down to
:class:`matryoshka.join_selection.JoinSelection`. Resolution used to rely on
star imports populating the module namespace, which forced every optional
dependency of every model to be imported before any of them could be used.

This registry replaces that. Entries are ``"module:attribute"`` strings
imported on demand, so a configuration that never selects the L1 models never
imports ``skglm`` or ``scikit-optimize``.
"""
from __future__ import annotations

import importlib
from typing import Callable

_MODELS: dict[str, str] = {
    # Regression proxies.
    'RegressionQR':             'matryoshka.selection.models:RegressionQR',
    'RegressionCholesky':       'matryoshka.selection.models:RegressionCholesky',
    'IncrementalRegressionFGS': 'matryoshka.selection.models:IncrementalRegressionFGS',
    # Classification proxy (linear discriminant analysis).
    'ClassificationCholesky':   'matryoshka.selection.models:ClassificationCholesky',
    # L1 proxies. Require the optional `l1` extra.
    'Lasso':                    'matryoshka.selection.models:Lasso',
    'LinRegL1':                 'skglm:Lasso',
    'LogRegL1':                 'skglm:SparseLogisticRegression',
}

_STRATEGIES: dict[str, str] = {
    'ForwardSelection':      'matryoshka.selection.algorithms.greedy:ForwardSelection',
    'BackwardElimination':   'matryoshka.selection.algorithms.greedy:BackwardElimination',
    'IncrementalSelection':  'matryoshka.selection.algorithms.stepwise:IncrementalSelection',
    'LassoFeatureSelector':  'matryoshka.selection.algorithms.lasso:LassoFeatureSelector',
}

_CACHE: dict[str, Callable] = {}


def _resolve(table: dict[str, str], kind: str, name: str) -> Callable:
    cache_key = f'{kind}:{name}'
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    try:
        target = table[name]
    except KeyError:
        raise KeyError(
            f'unknown {kind} {name!r}; registered: {sorted(table)}'
        ) from None
    module_name, _, attr = target.partition(':')
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f'{kind} {name!r} requires {module_name!r}, which is not installed '
            f'({exc}). Install the optional dependency, for example '
            f'`pip install "matryoshka[l1]"`.'
        ) from exc
    cls = getattr(module, attr)
    _CACHE[cache_key] = cls
    return cls


def resolve_model(name: str) -> Callable:
    """Return the proxy-model class registered under ``name``."""
    return _resolve(_MODELS, 'model', name)


def resolve_strategy(name: str) -> Callable:
    """Return the selection-strategy class registered under ``name``."""
    return _resolve(_STRATEGIES, 'strategy', name)


def available_models() -> list[str]:
    """Names accepted by :func:`resolve_model`."""
    return sorted(_MODELS)


def available_strategies() -> list[str]:
    """Names accepted by :func:`resolve_strategy`."""
    return sorted(_STRATEGIES)


def register_model(name: str, target: str | type) -> None:
    """Register an additional proxy model, by class or ``"module:attribute"``."""
    _register(_MODELS, 'model', name, target)


def register_strategy(name: str, target: str | type) -> None:
    """Register an additional selection strategy."""
    _register(_STRATEGIES, 'strategy', name, target)


def _register(table: dict[str, str], kind: str, name: str, target: str | type) -> None:
    if isinstance(target, str):
        table[name] = target
    else:
        table[name] = f'{target.__module__}:{target.__qualname__}'
        _CACHE[f'{kind}:{name}'] = target
