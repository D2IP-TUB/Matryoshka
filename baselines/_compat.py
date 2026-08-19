"""Bridges to the paper's experiment harness.

Two of the baseline adapters (CAAFE and Kitana) re-preprocess the query table
themselves, and the AutoFeat reference implementation scores with the
harness's own trainer. Both facilities belong to the evaluation harness under
``experiments/``, not to the library, and neither is needed to run Matryoshka.

The names below are proxies: importing them always succeeds, and the harness
is only imported when one is instantiated. An adapter therefore remains
importable on a machine without the harness, and a run that genuinely needs it
fails with one actionable message instead of an opaque ``ModuleNotFoundError``
raised deep inside the adapter.
"""
from __future__ import annotations

import importlib
from typing import Any

_MESSAGE = (
    '{name} belongs to the experiment harness under experiments/, which is not '
    'part of the installed library. It is required only to reproduce the '
    "paper's baseline numbers. Make experiments/ importable (run from the "
    'repository root with the harness present), or use '
    'matryoshka.preprocessing.prepare_query_table instead.'
)


class _HarnessProxy:
    """Resolves ``module:attribute`` from the harness on first instantiation."""

    _module: str = ''
    _attribute: str = ''

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls._resolve()(*args, **kwargs)

    @classmethod
    def _resolve(cls) -> Any:
        try:
            return getattr(importlib.import_module(cls._module), cls._attribute)
        except (ImportError, AttributeError) as exc:
            raise ImportError(_MESSAGE.format(name=cls._attribute)) from exc


class PreProcessor(_HarnessProxy):
    """The harness query-table preprocessor, resolved on instantiation."""

    _module = 'experiments.base_tables.base_table_preprocessing'
    _attribute = 'PreProcessor'


class AutoFeatOGTrainer(_HarnessProxy):
    """The harness trainer used by the original AutoFeat implementation."""

    _module = 'experiments.base_tables.base_scoring'
    _attribute = 'AutoFeatOGTrainer'
