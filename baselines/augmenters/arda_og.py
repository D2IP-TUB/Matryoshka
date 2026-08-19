import gc
import os
import sys
import time
import warnings
from pathlib import Path

import polars as pl


def _ensure_og_on_path():
    """Make the original feature_discovery package importable."""
    og_src = os.path.join(os.path.dirname(__file__), 'autofeat_og_utils', 'src')
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


class ArdaOgAugmenter:
    """Wrapper around the original ARDA implementation.

    Unlike :class:`ArdaAugmenter`, this version does not use Aurum for join
    discovery. The original package relies on a Neo4j-based dataset relation
    graph, which is expected to already be populated (see
    ``autofeat_og_utils/src/feature_discovery/dataset_relation_graph``).
    """

    def run(
        self,
        query_table_path: str,
        data_lake_path: str,
        base_node_id: str,
        target_column_name: str,
        sample_size: int,
        regression: bool,
        **kwargs,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _ensure_og_on_path()

            from baselines.augmenters._og_paths import (
                resolve_og_paths,
                cleanup_transient,
            )

            data_folder, og_node_id, transient_path = resolve_og_paths(
                data_lake_path, base_node_id, query_table_path
            )

            # Point the original package's DATA_FOLDER at the parent of
            # the dataset folder (OG layout). Modules import DATA_FOLDER
            # by name, so we must rebind it on every consumer module.
            import feature_discovery.config as fd_config  # type: ignore
            from feature_discovery.helpers import read_data as fd_read_data  # type: ignore
            from feature_discovery.baselines import arda as fd_arda  # type: ignore
            fd_config.DATA_FOLDER = data_folder
            fd_read_data.DATA_FOLDER = data_folder
            fd_arda.DATA_FOLDER = data_folder

            from feature_discovery.baselines.arda import (  # type: ignore
                select_arda_features_budget_join,
            )

            try:
                start = time.perf_counter()
                (
                    joined_df,
                    base_table_columns,
                    final_selected_features,
                    join_name,
                ) = select_arda_features_budget_join(
                    base_node_id=og_node_id,
                    target_column=target_column_name,
                    sample_size=sample_size,
                    regression=regression,
                )
                fetch_time = time.perf_counter() - start
            finally:
                cleanup_transient(transient_path)

            start = time.perf_counter()
            augplan = list(final_selected_features)
            kept = list(base_table_columns) + [target_column_name] + augplan
            kept = [c for c in kept if c in joined_df.columns]
            # Drop duplicate columns introduced by joins (keep first occurrence).
            joined_df = joined_df.loc[:, ~joined_df.columns.duplicated()]
            joined_df = joined_df[kept]
            joined_df.reset_index(drop=True, inplace=True)
            left_table = pl.from_pandas(joined_df)
            augmentation_time = time.perf_counter() - start

            gc.collect()

        return left_table, fetch_time, augmentation_time, augplan
