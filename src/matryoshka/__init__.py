"""Matryoshka: relevant-feature discovery over data lakes for machine learning.

Matryoshka augments a query table with features drawn from a data lake. It
replaces join materialisation with compact Gram matrix sketches and selects
features by greedily fitting linear proxy models over those sketches, so the
cost of evaluating a candidate feature set is independent of the number of
rows in the joined relations.

The system has two phases.

**Offline.** Each lake table is scanned once. Columns unlikely to serve as join
keys or features are pruned, per-entity aggregates are computed by ``GROUP BY``
over each candidate key, and the resulting Gram matrix sketch is written to
PostgreSQL together with an inverted index from cell value to table, column and
row. See :class:`matryoshka.index.ExhaustiveIndex`, wrapped by
:class:`matryoshka.LakeIndex`.

**Online.** For a query table with a join key and a prediction target, the
inverted index yields the top-*k* joinable lake tables, their sketches are
joined against the query table sketch, correlation-based pruning removes
low-signal candidates, and greedy forward selection incrementally fits a linear
proxy until no candidate improves it. See
:class:`matryoshka.join_selection.JoinSelection`, wrapped by
:class:`matryoshka.Augmenter`.

Minimal use::

    import matryoshka as mk

    index = mk.LakeIndex('covertype', settings='postgresql://user@localhost/matryoshka')
    index.build('data/covertype')

    query = mk.prepare_query_table(raw, key='Key_0_0', target='class')
    result = mk.Augmenter(index, task='classification').augment(
        query, key='Key_0_0', target='class'
    )
    print(result.n_selected, result.table.shape)

Reference: F. Turchenko et al., "Matryoshka: Uncovering Relevant Features in
Data Lakes to Enhance Machine Learning Applications", PVLDB 19, 2026.
"""
from .api import AugmentationResult, Augmenter, IndexStats, LakeIndex
from .baselines import available_baselines, register_baseline, resolve_baseline
from .config import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    ConfigValidationError,
    DiscoveryConfig,
)
from .db.settings import DBSettings, SSHSettings, resolve_settings
from .exceptions import (
    EmptyAugmentation,
    KeyNotFoundError,
    LinAlgError,
    UserTableNotProcessed,
)
from .preprocessing import prepare_query_table, train_test_split_by_key

__version__ = '0.1.0'

__all__ = [
    # high-level API
    'LakeIndex',
    'Augmenter',
    'AugmentationResult',
    'IndexStats',
    'prepare_query_table',
    'train_test_split_by_key',
    # configuration
    'DiscoveryConfig',
    'ConfigValidationError',
    'REGRESSION_METRICS',
    'CLASSIFICATION_METRICS',
    'DBSettings',
    'SSHSettings',
    'resolve_settings',
    # errors
    'EmptyAugmentation',
    'KeyNotFoundError',
    'LinAlgError',
    'UserTableNotProcessed',
    # baseline registry
    'available_baselines',
    'register_baseline',
    'resolve_baseline',
    '__version__',
]


def __getattr__(name: str):
    # The two engine classes pull in Ray, pgpq and the sketch stack, which
    # costs seconds of import time. Expose them lazily so that
    # `import matryoshka` stays cheap for callers that only need the facade.
    if name == 'ExhaustiveIndex':
        from .index import ExhaustiveIndex

        return ExhaustiveIndex
    if name == 'JoinSelection':
        from .join_selection import JoinSelection

        return JoinSelection
    if name == 'JoinDiscovery':
        from .retrieval import JoinDiscovery

        return JoinDiscovery
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
