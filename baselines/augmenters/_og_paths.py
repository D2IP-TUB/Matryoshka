"""Helpers for aligning our experiment paths with what the OG
``feature_discovery`` package expects.

OG ingests each dataset as a folder under a single ``DATA_FOLDER`` and
stores Neo4j node IDs of the form ``<dataset>/<table>.csv``. The base
table is renamed to ``table_0_0.csv`` in the graph even though on disk
the file may be named ``<dataset>.csv``.

Our experiment config points ``data_lake_path`` at ``<...>/autofeat/<dataset>``
and ``base_node_id`` at ``<dataset>.csv``. This module bridges the two:

* ``DATA_FOLDER`` becomes the parent of ``data_lake_path``.
* ``base_node_id`` becomes ``<dataset>/table_0_0.csv``.
* A transient ``<dataset>/table_0_0.csv`` file is created on disk
  (from the preprocessed base table) so OG's ``DATA_FOLDER / node_id``
  reads succeed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import polars as pl


def resolve_og_paths(
    data_lake_path: str,
    base_node_id: str,
    query_table_path: Optional[str] = None,
) -> Tuple[Path, str, Optional[str]]:
    """Translate experiment-style paths into OG-style paths.

    Returns
    -------
    (data_folder, og_node_id, transient_path)
        ``data_folder`` to assign to ``feature_discovery.config.DATA_FOLDER``;
        ``og_node_id`` to pass to OG functions as the base table id;
        ``transient_path`` is the on-disk file we created (or ``None``) so
        the caller can clean it up.
    """
    data_lake = Path(data_lake_path).resolve()
    dataset = data_lake.name
    parent = data_lake.parent

    # Look up the OG-registered base table name in datasets.csv (they use
    # 'table_0_0.csv' for most datasets but 'base.csv' for school). Also capture
    # the target column so we can strip it from any duplicate-base alias.
    og_base_name = "table_0_0.csv"
    target_column: Optional[str] = None
    datasets_csv = parent / "datasets.csv"
    if datasets_csv.exists():
        try:
            import csv as _csv
            with open(datasets_csv, newline="") as fh:
                for row in _csv.DictReader(fh):
                    if row.get("base_table_label") == dataset:
                        og_base_name = row.get("base_table_name") or og_base_name
                        target_column = row.get("target_column")
                        break
        except Exception:
            pass

    og_node_id = f"{dataset}/{og_base_name}"
    transient_path = parent / og_node_id

    # Resolve the source base table once (preprocessed query table preferred,
    # else the raw base file under the dataset folder).
    source: Optional[Path] = None
    if query_table_path and os.path.exists(query_table_path):
        source = Path(query_table_path)
    else:
        candidate = data_lake / base_node_id
        if candidate.exists():
            source = candidate

    def _materialise(target: Path, drop_target: bool = False) -> Optional[str]:
        """Write the base table to ``target`` if absent; return it if created.

        ``drop_target`` removes the target column -- used for the duplicate-base
        alias, which is reached as a *join candidate* (not the base). Keeping the
        target there would let the join re-introduce it as a feature and leak the
        label into the downstream model."""
        if target.exists():
            return None
        if source is None:
            raise FileNotFoundError(
                f"Cannot materialise OG base table at {target}: "
                f"neither {query_table_path!r} nor {data_lake / base_node_id!s} exist"
            )
        df = pl.read_csv(str(source))
        if drop_target and target_column and target_column in df.columns:
            df = df.drop(target_column)
        target.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(str(target))
        return str(target)

    created = []
    primary = _materialise(transient_path)
    if primary:
        created.append(primary)
    # Some graphs (e.g. school) register the base under its real name
    # (``base.csv``) AND under the OG-standard ``table_0_0.csv`` alias, both
    # wired to the same join tables. The BFS can reach the alias, so when it is
    # missing on disk we materialise it from the same source -- otherwise the
    # OG read crashes on the absent file. The alias is a join candidate, so we
    # strip the target column from it to avoid leaking the label. Datasets whose
    # base already is ``table_0_0.csv`` are unaffected.
    if og_base_name != "table_0_0.csv":
        alias = _materialise(parent / f"{dataset}/table_0_0.csv", drop_target=True)
        if alias:
            created.append(alias)

    return parent, og_node_id, created


def cleanup_transient(paths) -> None:
    """Remove transient base file(s). Accepts a path, a list of paths, or None."""
    if not paths:
        return
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass
