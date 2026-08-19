import gc
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

# Set PYTHONHASHSEED for deterministic hashing
os.environ['PYTHONHASHSEED'] = '42'


def _ensure_og_on_path():
    """Make the original feature_discovery package importable."""
    og_src = os.path.join(os.path.dirname(__file__), 'autofeat_og_utils', 'src')
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


class AutofeatOgAugmenter:
    """Wrapper around the original AutoFeat implementation.

    Unlike :class:`AutofeatAugmenter`, this version does not use Aurum for
    join discovery. The original package relies on a Neo4j-based dataset
    relation graph, which is expected to already be populated (see
    ``autofeat_og_utils/src/feature_discovery/dataset_relation_graph``).
    """

    def run(
        self,
        query_table_path: str,
        data_lake_path: str,
        problem_type: str,
        base_node_id: str,
        target_column_name: str,
        save_joins_to_disk: bool = True,
        use_polars: bool = True,
        value_ratio: float = 0.65,
        top_k: int = 15,
        sample_size: int = 3000,
        pearson: bool = False,
        jmi: bool = False,
        no_relevance: bool = False,
        no_redundancy: bool = False,
        base_table_label: str = 'base_table',
        **kwargs,
    ):
        random.seed(42)
        np.random.seed(42)
        os.environ['PYTHONHASHSEED'] = '42'

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

            # The OG repo runs against a single Neo4j database (see
            # feature_discovery.config.NEO4J_DATABASE, default 'lake');
            # rely on the env var the user has configured in .env.
            import feature_discovery.config as fd_config  # type: ignore
            from feature_discovery.helpers import read_data as fd_read_data  # type: ignore
            fd_config.DATA_FOLDER = data_folder
            # Modules import DATA_FOLDER by name; rebind on every consumer
            # module that has already imported it.
            fd_read_data.DATA_FOLDER = data_folder

            base_node_id = og_node_id

            from feature_discovery.autofeat_pipeline.autofeat import AutoFeat as AutoFeatBase  # type: ignore
            from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length  # type: ignore
            from feature_discovery.experiments.evaluate_join_paths import (  # type: ignore
                create_join_tree,
                join_from_path,
            )
            from feature_discovery.experiments.dataset_object import REGRESSION  # type: ignore
            # The vendored OG evaluate_all_algorithms cannot run under the
            # installed AutoGluon (it calls TabularPredictor.get_model_names,
            # since removed). Use the repo's faithful re-implementation of the
            # OG downstream evaluation to score paths instead.
            from baselines._compat import AutoFeatOGTrainer

            start = time.perf_counter()
            autofeat = AutoFeatBase(
                base_table_id=str(base_node_id),
                base_table_label=base_table_label,
                save_joins_to_disk=save_joins_to_disk,
                use_polars=use_polars,
                target_column=target_column_name,
                task=problem_type,
                value_ratio=value_ratio,
                top_k=top_k,
                sample_size=sample_size,
                pearson=pearson,
                jmi=jmi,
                no_relevance=no_relevance,
                no_redundancy=no_redundancy,
            )
            # Snapshot the base-table features before traversal. The original
            # streaming_feature_selection accumulates selected features in
            # place into partial_join_selected_features[base_table_id] (the
            # list is extended by reference at each join), so after the call
            # that list also contains every selected join feature. Capturing it
            # now preserves the true original base columns to exclude later.
            original_base_features = list(
                autofeat.partial_join_selected_features.get(
                    autofeat.base_table_id, []
                )
            )
            autofeat.streaming_feature_selection(queue={str(base_node_id)})
            fetch_time = time.perf_counter() - start

            start = time.perf_counter()
            sorted_paths = sorted(
                autofeat.ranking.items(),
                key=lambda r: (r[1], -get_path_length(r[0])),
                reverse=True,
            )
            top_k_paths = sorted_paths[:top_k] if len(sorted_paths) > top_k else sorted_paths

            # ``base_features`` aliases the in-place mutated list and by now
            # holds every selected join feature. The original evaluate_paths
            # uses it verbatim and relies on the per-path column intersection
            # below to recover the features actually present in each path.
            base_features = autofeat.partial_join_selected_features.get(
                autofeat.base_table_id, []
            )
            is_regression = problem_type == REGRESSION

            # Follow the original evaluate_paths: build each top-k path, restrict
            # to its selected features, evaluate with the OG RF/GBM/XT/XGB
            # procedure, and keep the best-scoring path. The reported per-dataset
            # number is the best across both paths and models, so path selection
            # must happen here; the downstream trainer only takes the best across
            # models for the single returned table.
            best_df = None
            best_features: list = []
            best_score = None
            eval_root = tempfile.mkdtemp(prefix="autofeat_og_paths_")
            for path_idx, (join_name, _rank) in enumerate(top_k_paths):
                if join_name == autofeat.base_table_id:
                    continue

                features = list(autofeat.partial_join_selected_features[join_name])
                features.append(autofeat.target_column)
                features.extend(base_features)

                features_tables = sorted({f"{f.split('.csv')[0]}.csv" for f in features})

                path_tables = {}
                for p in join_name.split("--"):
                    aux = p.split("-")
                    if len(aux) == 4:
                        path_tables[aux[3]] = (aux[0], aux[1], aux[2], aux[3])

                path_list = []
                for table in features_tables:
                    if table in path_list:
                        continue
                    path_aux = create_join_tree(table, path_tables)
                    # Skip entries that are neither a resolved join tree (a
                    # list) nor a known path table. The target column (e.g.
                    # 'class') is turned into a bogus 'class.csv' above; without
                    # this guard join_from_path tries to read it as a table and
                    # raises. Mirrors the original evaluate_paths.
                    if not (type(path_aux) is list) and (path_aux not in path_tables.keys()):
                        continue
                    path_list.append(path_aux)

                try:
                    dataframe = join_from_path(
                        path_list, autofeat.target_column, autofeat.base_table_id
                    )
                except Exception:
                    continue
                if dataframe is None:
                    continue

                # Recover the per-path features by intersecting the (aliased)
                # accumulated feature set with the columns present in this
                # path's join. Fall back to the base columns when fewer than two
                # features survive, as the original does.
                path_features = list(set(features).intersection(set(dataframe.columns)))
                if len(path_features) < 2:
                    path_features = list(base_features)
                    path_features.append(autofeat.target_column)
                    path_features = list(
                        set(path_features).intersection(set(dataframe.columns))
                    )

                # Score this path with the repo's working OG scorer
                # (RF/GBM/XT/XGB, best across models on a fixed 80/20 split),
                # mirroring the original choice of best path by downstream
                # accuracy.
                try:
                    trainer = AutoFeatOGTrainer(
                        problem_type=problem_type,
                        target_column=autofeat.target_column,
                        output_dir=os.path.join(eval_root, f"path_{path_idx}"),
                    )
                    trainer.train(dataframe[path_features])
                    perf = trainer.evaluate(dataframe[path_features])
                except Exception:
                    continue

                # 'accuracy' for classification (maximise), 'rmse' for
                # regression (minimise).
                path_score = perf.get("rmse") if is_regression else perf.get("accuracy")
                if path_score is None or (
                    isinstance(path_score, float) and np.isnan(path_score)
                ):
                    continue

                better = (
                    best_score is None
                    or (path_score < best_score if is_regression else path_score > best_score)
                )
                if better:
                    best_score = path_score
                    best_df = dataframe[path_features]
                    best_features = path_features

            shutil.rmtree(eval_root, ignore_errors=True)

            # The augmentation plan is the best path's selected features with the
            # target and the original base columns removed.
            augplan = [
                f for f in best_features
                if f != autofeat.target_column and f not in original_base_features
            ]

            if best_df is None:
                # No path produced a usable join; fall back to the base table.
                from feature_discovery.helpers.read_data import get_df_with_prefix  # type: ignore
                best_df, _ = get_df_with_prefix(
                    str(base_node_id), autofeat.target_column
                )
                augplan = []

            best_df = best_df.loc[:, ~best_df.columns.duplicated()]
            best_df.reset_index(drop=True, inplace=True)
            left_table = pl.from_pandas(best_df)
            augmentation_time = time.perf_counter() - start

            cleanup_transient(transient_path)

            gc.collect()

        return left_table, fetch_time, augmentation_time, augplan
