"""High-level interface to the two phases of Matryoshka.

:class:`LakeIndex` covers the offline phase: it indexes a data lake into the
PostgreSQL tables described in :mod:`matryoshka.db.handler`, and reports what
that index contains. :class:`Augmenter` covers the online phase: given a query
table, a join key and a target, it retrieves joinable lake tables, selects a
feature set with the greedy forward procedure, and materialises the augmented
query table.

Both classes are thin facades over :class:`matryoshka.index.ExhaustiveIndex`
and :class:`matryoshka.join_selection.JoinSelection`, which remain available
for callers who need the full parameter surface.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl

from .config import DiscoveryConfig
from .db.settings import DBSettings, resolve_settings

# Suffixes appended to the lake name to derive table names. They match the
# naming used by the paper's experiment configuration, so an index built by
# this API is readable by the reproduction scripts and vice versa.
FS_INDEX_SUFFIX = '_matryoshka_fs_index'
OVERLAP_INDEX_SUFFIX = '_matryoshka_overlap_index'

# Default proxy-model configuration per task. `model` names a class in
# `matryoshka.selection.models`; `metric` names a scoring criterion in
# `matryoshka.config.REGRESSION_METRICS` / `CLASSIFICATION_METRICS`.
#
# The defaults are the conditional proxies: the conditional Fisher score for
# classification and the conditional GCV score for regression. Both score a
# candidate feature *given* the features already selected, which suppresses
# redundant additions more aggressively than their marginal counterparts. The
# marginal criteria used for the published numbers are `average_mahalanobis`
# (tol 0.5) and `mse` (tol 0.001).
_TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    'classification': {
        'model': 'ClassificationCholesky',
        'metric': 'conditional_mahalanobis',
        'tol': 0.05,
        'corr_threshold': 0.3,
    },
    'regression': {
        'model': 'IncrementalRegressionFGS',
        'metric': 'conditional_gcv',
        'tol': 0.05,
        'corr_threshold': 0.1,
    },
}

_ARCHIVE_SUFFIXES = ('.tar.gz', '.tgz', '.tar', '.zip')


@dataclass
class IndexStats:
    """Row and size statistics of one lake index."""

    feature_table: str
    overlap_table: str
    rows: int
    tables_indexed: int
    total_bytes: int

    @property
    def total_size(self) -> str:
        """Human-readable form of :attr:`total_bytes`."""
        size = float(self.total_bytes)
        for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
            if size < 1024 or unit == 'TiB':
                return f'{size:.1f} {unit}'
            size /= 1024
        return f'{size:.1f} TiB'


@dataclass
class AugmentationResult:
    """Outcome of one augmentation query."""

    table: pl.DataFrame
    plan: list
    runtime_seconds: float

    @property
    def n_selected(self) -> int:
        """Number of lake features appended to the query table."""
        return len(self.plan)

    def __iter__(self):
        # Allows `augmented, plan = augmenter.augment(...)`, matching the
        # tuple returned by `JoinSelection.find_best_joins`.
        return iter((self.table, self.plan))


class LakeIndex:
    """A Matryoshka index over one data lake.

    Parameters
    ----------
    name
        Lake name. Table names default to ``<name>_matryoshka_fs_index`` and
        ``<name>_matryoshka_overlap_index``.
    settings
        Connection settings: a :class:`~matryoshka.db.settings.DBSettings`, a
        ``postgresql://`` DSN, or ``None`` to resolve from the environment.
    feature_table, overlap_table
        Explicit table names, overriding the defaults derived from ``name``.
    """

    def __init__(
        self,
        name: str,
        *,
        settings: DBSettings | str | None = None,
        feature_table: str | None = None,
        overlap_table: str | None = None,
    ) -> None:
        self.name = name
        self.feature_table = feature_table or f'{name}{FS_INDEX_SUFFIX}'
        self.overlap_table = overlap_table or f'{name}{OVERLAP_INDEX_SUFFIX}'
        self.settings = resolve_settings(settings)

    def __repr__(self) -> str:
        return (
            f'LakeIndex(name={self.name!r}, feature_table={self.feature_table!r}, '
            f'settings={self.settings!r})'
        )

    # -- offline phase ----------------------------------------------------

    def build(
        self,
        path: str | os.PathLike,
        *,
        max_workers: int = 1,
        batch_size: int | None = 1,
        max_cols: int = 100,
        feature_extraction: bool = False,
        update_support: bool = False,
        start_table_index: int = 0,
        create_btree_indexes: bool = True,
        **index_kwargs: Any,
    ) -> int:
        """Index the lake at ``path`` and return the number of tables processed.

        ``path`` is either a directory of tables, which is indexed in one pass,
        or a directory of archives (``.tar.gz``, ``.tar``, ``.zip``), each of
        which is indexed in turn with a continuing table index.

        Parameters
        ----------
        max_workers
            Number of Ray workers. Values above 1 enable the sliding-window
            scheduler in :meth:`matryoshka.index.ExhaustiveIndex.index_lake`.
        batch_size
            Tables completed before flushing to PostgreSQL. ``None`` flushes
            once per ``max_workers`` tables.
        max_cols
            Tables wider than this are skipped, as in the paper's setup.
        feature_extraction
            Enables the extended per-key aggregate set.
        update_support
            Also writes the ``_meta`` / ``_num_state`` / ``_cat_state`` side
            tables required by ``ExhaustiveIndex.update_table`` and
            ``find_union_peer``. Roughly doubles the index size.
        start_table_index
            First table index to assign. Set this to resume an interrupted
            build at the table after the last one logged.
        create_btree_indexes
            Builds the btree indexes on the finished tables. Retrieval is
            unusably slow without them; disable only to defer the cost when
            appending more archives afterwards.
        """
        from .index import ExhaustiveIndex

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'lake path does not exist: {path}')

        sources = self._resolve_sources(path)
        table_index = start_table_index
        worker: ExhaustiveIndex | None = None
        for source in sources:
            worker = ExhaustiveIndex(
                data_dir=str(source),
                feature_selection_table_name=self.feature_table,
                overlap_table_name=self.overlap_table,
                batch_size=batch_size,
                max_workers=max_workers,
                feature_extraction=feature_extraction,
                update_support=update_support,
                max_cols=max_cols,
                settings=self.settings,
                **index_kwargs,
            )
            from_tar = str(source).endswith(('.tar.gz', '.tgz', '.tar'))
            # `index_lake` returns the next free table index, not the last one
            # used, so the following source continues from it directly. The
            # original offline/main.py added one here as well, which left a gap
            # in `table_index` for every archive after the first.
            table_index = worker.index_lake(table_index, from_tar_archive=from_tar)

        if worker is not None and create_btree_indexes:
            conn = worker.db_connect()
            try:
                worker.create_db_table_index(conn)
            finally:
                conn.close()
        return table_index - start_table_index

    @staticmethod
    def _resolve_sources(path: Path) -> list[Path]:
        """Return the units passed to ``index_lake``, one call each."""
        if path.is_file():
            return [path]
        archives = sorted(
            child for child in path.iterdir()
            if child.is_file() and str(child).endswith(_ARCHIVE_SUFFIXES)
        )
        # A directory holding archives is indexed archive by archive; a
        # directory holding tables is indexed in a single pass.
        return archives if archives else [path]

    # -- introspection ----------------------------------------------------

    def exists(self) -> bool:
        """Whether the feature-selection index table is present."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute('SELECT to_regclass(%s) IS NOT NULL', (self.feature_table,))
            return bool(cur.fetchone()[0])

    def stats(self) -> IndexStats:
        """Exact row count, distinct indexed tables, and on-disk size."""
        if not self.exists():
            raise RuntimeError(
                f'index table {self.feature_table!r} does not exist in '
                f'database {self.settings.dbname!r}; build it first'
            )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f'SELECT count(*), count(DISTINCT table_index) FROM {self.feature_table}'
            )
            rows, tables = cur.fetchone()
            cur.execute(
                'SELECT coalesce(pg_total_relation_size(%s), 0) '
                '     + coalesce(pg_total_relation_size(%s), 0)',
                (self.feature_table, self.overlap_table),
            )
            total_bytes = cur.fetchone()[0]
        return IndexStats(
            feature_table=self.feature_table,
            overlap_table=self.overlap_table,
            rows=int(rows),
            tables_indexed=int(tables),
            total_bytes=int(total_bytes),
        )

    def indexed_tables(self) -> pl.DataFrame:
        """Table index and name of every lake table with index entries."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f'SELECT DISTINCT table_index, table_name FROM {self.feature_table} '
                'ORDER BY table_index'
            )
            rows = cur.fetchall()
        return pl.DataFrame(
            {'table_index': [r[0] for r in rows], 'table_name': [r[1] for r in rows]},
            schema={'table_index': pl.Int32, 'table_name': pl.String},
        )

    def drop(self, *, missing_ok: bool = True) -> None:
        """Drop every table belonging to this index."""
        suffixes = ('', '_meta', '_num_state', '_cat_state')
        names = [f'{self.feature_table}{s}' for s in suffixes] + [self.overlap_table]
        clause = ' IF EXISTS' if missing_ok else ''
        with self._connect() as conn, conn.cursor() as cur:
            for name in names:
                cur.execute(f'DROP TABLE{clause} {name} CASCADE')
            conn.commit()

    def _connect(self):
        import psycopg2

        return psycopg2.connect(
            dbname=self.settings.dbname,
            user=self.settings.user,
            password=self.settings.password,
            host=self.settings.host,
            port=self.settings.port,
        )


class Augmenter:
    """Discovery and feature selection against one :class:`LakeIndex`.

    Parameters
    ----------
    index
        The lake index to query.
    task
        ``'classification'`` or ``'regression'``. Selects the proxy model and
        the default scoring criterion.
    top_k
        Number of joinable lake tables retrieved per query.
    metric, tol, model
        Overrides of the per-task defaults. ``metric`` must be listed in
        :data:`matryoshka.config.REGRESSION_METRICS` or
        :data:`matryoshka.config.CLASSIFICATION_METRICS`; ``tol`` is the
        minimum relative score improvement a candidate must yield to be
        accepted.
    corr_threshold
        Correlation-based pruning threshold applied before selection. ``None``
        disables pruning, which the paper does for the AutoFeat lakes.
    n_jobs
        Ray parallelism for candidate evaluation.
    budget_seconds
        Wall-clock cap on the discovery phase. ``None`` runs to convergence.
    exclude_tables
        Lake table names never retrieved. Set this when the query table is
        itself a member of the lake, so that it cannot retrieve itself and
        leak its own target back in as a candidate feature.
    features_stop_list
        Lake columns never used as features, as ``{lake table name: [column
        names]}``. Table names are file names, with or without the extension;
        column names can be given as in the lake table or normalized as in the
        index. Set this when some lake columns are known to leak the target, so
        that they are dropped before selection while the rest of their table
        remains a candidate. The stop-listed features are removed through the
        ``drop_feature`` mask of correlation pruning, also when pruning is
        disabled.
    params
        Extra entries merged into ``DiscoveryConfig.params``, for example
        ``{'polynomial_features': {'enabled': True, 'degree': 2}}``.
    verbose, log_dir
        Structured JSON logs are written to ``log_dir``, defaulting to
        ``$MATRYOSHKA_LOG_DIR`` or ``./.matryoshka/logs``, and echoed to stderr
        when ``verbose``.
    """

    def __init__(
        self,
        index: LakeIndex,
        *,
        task: str = 'classification',
        top_k: int = 20,
        metric: str | None = None,
        tol: float | None = None,
        model: str | None = None,
        corr_threshold: float | None = ...,  # type: ignore[assignment]
        n_jobs: int = 1,
        budget_seconds: float | None = None,
        exclude_tables: Iterable[str] | None = None,
        features_stop_list: Mapping[str, Iterable[str]] | None = None,
        params: dict[str, Any] | None = None,
        verbose: bool = False,
        log_dir: str | os.PathLike | None = None,
    ) -> None:
        if task not in _TASK_DEFAULTS:
            raise ValueError(
                f'task must be one of {sorted(_TASK_DEFAULTS)}, got {task!r}'
            )
        defaults = _TASK_DEFAULTS[task]
        self.index = index
        self.task = task
        self.top_k = top_k
        self.metric = metric or defaults['metric']
        self.tol = defaults['tol'] if tol is None else tol
        self.model = model or defaults['model']
        # `...` distinguishes "not supplied" from an explicit None, which
        # disables pruning.
        self.corr_threshold = (
            defaults['corr_threshold'] if corr_threshold is ... else corr_threshold
        )
        self.n_jobs = n_jobs
        self.budget_seconds = budget_seconds
        self.exclude_tables = list(exclude_tables or ())
        self.features_stop_list = {
            table_name: [column_names] if isinstance(column_names, str) else list(column_names)
            for table_name, column_names in (features_stop_list or {}).items()
        }
        self.extra_params = dict(params or {})
        self.verbose = verbose
        self.log_dir = str(log_dir) if log_dir is not None else None

    def __repr__(self) -> str:
        return (
            f'Augmenter(index={self.index.name!r}, task={self.task!r}, '
            f'metric={self.metric!r}, tol={self.tol}, top_k={self.top_k})'
        )

    def discovery_config(self) -> DiscoveryConfig:
        """The validated :class:`DiscoveryConfig` this augmenter will run."""
        params: dict[str, Any] = {'metric': self.metric, 'tol': float(self.tol)}
        if self.budget_seconds is not None:
            params['budget_seconds'] = float(self.budget_seconds)
        params.update(self.extra_params)
        return DiscoveryConfig(
            task=self.task,
            ranking='passthrough',
            strategy='ForwardSelection',
            model=self.model,
            params=params,
        )

    def augment(
        self,
        query_table: pl.DataFrame,
        *,
        key: str,
        target: str,
        debug: bool = False,
    ) -> AugmentationResult:
        """Augment ``query_table`` with features selected from the lake.

        ``query_table`` must already be in the layout produced by
        :func:`matryoshka.preprocessing.prepare_query_table`: the join key
        first as a normalised string column, numeric null-free features next,
        and the target last.

        With ``debug=False`` a failure during discovery is logged and the
        unmodified query table is returned with an empty plan, matching the
        behaviour the experiment harness relies on. ``debug=True`` re-raises.
        """
        import time

        from .join_selection import JoinSelection

        for name in (key, target):
            if name not in query_table.columns:
                raise KeyError(f'column {name!r} is not in the query table')

        worker = JoinSelection(
            feature_selection_table_name=self.index.feature_table,
            overlap_table_name=self.index.overlap_table,
            verbose=self.verbose,
            log_dir=self.log_dir,
            log_file_name=f'{self.index.name}_{target}.log',
            settings=self.index.settings,
            exclude_tables=self.exclude_tables,
            features_stop_list=self.features_stop_list,
        )
        start = time.perf_counter()
        table, plan = worker.find_best_joins(
            user_table_processed=query_table,
            query_column_name=key,
            target_column_name=target,
            top_k=self.top_k,
            config=self.discovery_config(),
            corr_threshold=self.corr_threshold,
            n_jobs=self.n_jobs,
            debug=debug,
        )
        return AugmentationResult(
            table=table, plan=list(plan), runtime_seconds=time.perf_counter() - start
        )
