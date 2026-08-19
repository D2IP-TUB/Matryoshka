"""Lazy resolution of baseline augmenters.

The core package implements one augmentation strategy, the greedy forward
selection of Section 6 of the paper. The competing augmenters evaluated in
Section 7 (ARDA, AutoFeat, CAAFE, Kitana, QCR, COCOA, Metam) wrap third-party
research code with heavy and partly incompatible dependencies: Neo4j, PyTorch,
AutoGluon, an OpenAI client, a running Aurum service. Importing them eagerly
would make ``import matryoshka`` fail on a machine that only wants to run
Matryoshka itself.

The baselines therefore live outside the installed package, under the
top-level ``baselines/`` directory of the repository, and are resolved here by
name at the moment ``JoinSelection.run_baseline`` needs them. A configuration
with ``baseline=True`` and ``strategy='ArdaAugmenter'`` works whenever that
directory is importable; otherwise the failure is a single actionable error
instead of an import-time crash.

Register additional augmenters with :func:`register_baseline`.
"""
from __future__ import annotations

import importlib
from typing import Callable

# Strategy name -> "module path:class name". Modules are imported on demand.
_REGISTRY: dict[str, str] = {
    'ArdaAugmenter':       'baselines.augmenters.arda:ArdaAugmenter',
    'ArdaOgAugmenter':     'baselines.augmenters.arda_og:ArdaOgAugmenter',
    'AutofeatAugmenter':   'baselines.augmenters.autofeat:AutofeatAugmenter',
    'AutofeatOgAugmenter': 'baselines.augmenters.autofeat_og:AutofeatOgAugmenter',
    'CaafeAugmenter':      'baselines.augmenters.caafe:CaafeAugmenter',
    'CocoaAugmenter':      'baselines.augmenters.cocoa:CocoaAugmenter',
    'KitanaAugmenter':     'baselines.augmenters.kitana:KitanaAugmenter',
    'MetamAugmenter':      'baselines.augmenters.metam:MetamAugmenter',
    'QcrAugmenter':        'baselines.augmenters.qcr:QcrAugmenter',
}


def register_baseline(name: str, target: str | type) -> None:
    """Register a baseline augmenter under ``name``.

    ``target`` is either the class itself or a ``"module:attribute"`` string
    resolved on first use.
    """
    if isinstance(target, str):
        _REGISTRY[name] = target
    else:
        _RESOLVED[name] = target
        _REGISTRY.setdefault(name, f'{target.__module__}:{target.__qualname__}')


def available_baselines() -> list[str]:
    """Names accepted by :func:`resolve_baseline`, whether importable or not."""
    return sorted(_REGISTRY)


_RESOLVED: dict[str, Callable] = {}


def resolve_baseline(name: str) -> Callable:
    """Import and return the augmenter class registered under ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    ImportError
        If the module is registered but cannot be imported, with the missing
        dependency named in the message.
    """
    if name in _RESOLVED:
        return _RESOLVED[name]
    try:
        target = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f'unknown baseline strategy {name!r}; '
            f'registered: {available_baselines()}'
        ) from None
    module_name, _, attr = target.partition(':')
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f'baseline {name!r} maps to {target!r}, which is not importable '
            f'({exc}). Baselines are not part of the installed package: run '
            f'from the repository root so that "baselines/" is on sys.path, '
            f'and install its extra dependencies (see baselines/README.md).'
        ) from exc
    cls = getattr(module, attr)
    _RESOLVED[name] = cls
    return cls
