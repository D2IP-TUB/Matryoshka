import logging
import os
import random
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from matryoshka.utils.common import process_key
from .join_data import join_and_save
from .join_path_feature_selection import RelevanceRedundancy
from .join_path_utils import compute_join_name
from .neo4j_transactions import (
    get_adjacent_nodes,
    get_relation_properties_node_name,
    get_df_with_prefix
)

# Set PYTHONHASHSEED for deterministic hashing
os.environ['PYTHONHASHSEED'] = '42'

logging.getLogger().setLevel(logging.INFO)


class AutoFeat:
    def __init__(
        self,
        join_paths_df: pd.DataFrame,
        lake_data_folder: str,
        base_table_sep: str,
        base_table_label: str,
        base_table_id: str,
        target_column: str,
        save_joins_to_disk: bool,
        use_polars: bool,
        task: str,
        value_ratio: float = 0.65,
        top_k: int = 15,
        sample_size: int = 3000,
        pearson: bool = False,
        jmi: bool = False,
        no_relevance: bool = False,
        no_redundancy: bool = False
    ):
        """
        :param base_table_label: The name (label) of the base table to be used for saving data.
        :param target_column: Target column containing the class labels for training.
        :param value_ratio: Pruning threshold. It represents the ration between the number of non-null values in a column and the total number of values.
        """
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        os.environ['PYTHONHASHSEED'] = '42'
        
        self.join_paths_df: pd.DataFrame = join_paths_df
        self.lake_data_folder: str = lake_data_folder
        self.base_table_sep: str = base_table_sep
        self.base_table_label: str = base_table_label
        self.target_column: str = target_column
        self.value_ratio: float = value_ratio
        self.top_k: int = top_k
        self.sample_size: int = sample_size
        self.base_table_id: str = base_table_id
        self.task: str = task
        # Mapping with the name of the join and the corresponding name of the file containing the join result.
        self.join_name_mapping: Dict[str, str] = {}
        # Set used to track the visited nodes.
        self.discovered: Set[str] = set()
        # Save the selected features of the previous join path (used for conditional redundancy)
        self.partial_join_selected_features: Dict[str, List] = {}

        self.ranking: Dict[str, float] = {}
        self.join_keys: Dict[str, list] = {}
        self.rel_red = RelevanceRedundancy(target_column, jmi=jmi, pearson=pearson)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.use_polars = use_polars
        self.partial_join = self.initialisation()

        # Ablation study parameters
        self.sample_data_step = True
        self.no_relevance = no_relevance
        self.no_redundancy = no_redundancy

        # Whether to save the joins to disk or not
        self.save_joins_to_disk = save_joins_to_disk
        self.iters = 0
        if self.save_joins_to_disk is not True:
            self.joins_to_df: Dict[str, pd.DataFrame] = {}
        else:
            logging.warn(f"Saving intermediate joins to disk: {self.temp_dir.name}")

    def initialisation(self):
        from sklearn.model_selection import train_test_split

        # Read dataframe
        base_table_df, partial_join_name = get_df_with_prefix(
            self.join_paths_df, self.lake_data_folder, self.base_table_id, target_column=self.target_column, table_sep=self.base_table_sep, use_polars=self.use_polars
        )

        # Stratified sampling
        if self.sample_size < base_table_df.shape[0]:
            if self.task == 'classification':
                X_train, X_test = train_test_split(
                    base_table_df,
                    train_size=self.sample_size,
                    stratify=base_table_df[self.target_column],
                    random_state=42,
                )
            else:
                X_train, X_test = train_test_split(base_table_df, train_size=self.sample_size, random_state=42)
        else:
            X_train = base_table_df

        # Base table features are the selected features
        features = list(X_train.columns)
        if self.target_column in features:
            features.remove(self.target_column)

        self.partial_join_selected_features[partial_join_name] = features
        self.ranking[partial_join_name] = 0
        self.join_keys[partial_join_name] = []

        return X_train

    def _resolve_column_name(self, column_name: str, df: pd.DataFrame, table_prefix: str) -> str:
        """
        Resolve placeholder column names (like 'col_4') to actual column names.
        
        Args:
            column_name: The column name from join_paths (might be placeholder like 'col_4')
            df: The dataframe containing the actual columns
            table_prefix: The table prefix used in the dataframe columns
            
        Returns:
            The actual column name (without prefix) or original if not a placeholder
        """
        if column_name.startswith('col_'):
            try:
                col_index = int(column_name.split('_')[1])
                # Get column names without prefix
                actual_columns = [col.replace(f"{table_prefix}.", "", 1) for col in df.columns if col.startswith(f"{table_prefix}.")]
                if col_index < len(actual_columns):
                    return actual_columns[col_index]
            except (ValueError, IndexError):
                pass
        return column_name

    def streaming_feature_selection(self, join_paths_df: pd.DataFrame, lake_data_folder: str, lake_table_sep: str, queue: set, previous_queue: set = None,
                                    budget_clock=None, trajectory_emitter=None):
        if len(queue) == 0:
            return

        if previous_queue is None:
            previous_queue = queue.copy()

        # Anytime hook for AutoFeat. Each BFS queue-pop corresponds to one
        # outer iteration. We emit the cumulative set of features collected
        # across all explored join paths up to that point and break when the
        # budget expires. The final ``evaluate_paths`` step is not run here,
        # so the trajectory rows record AutoFeat's BFS-discovered candidate
        # set rather than its final top-k. The caller appends one extra
        # trajectory point post-``evaluate_paths`` if it wants the natural
        # endpoint to appear on the plot.
        def _emit_progress(iter_idx):
            if trajectory_emitter is None:
                return
            cumulative = []
            seen = set()
            for feats in self.partial_join_selected_features.values():
                for f in feats:
                    if f not in seen:
                        seen.add(f)
                        cumulative.append(f)
            t = budget_clock.elapsed_s if budget_clock is not None else 0.0
            trajectory_emitter.emit(iter_idx, t, cumulative)

        _emit_progress(0)
        _iter_idx = 0

        # Iterate through all the elements of the queue:
        # 1) in the first iteration: queue = base_node_id
        # 2) in all the other iterations: queue = neighbours of the previous node
        all_neighbours = set()
        while len(queue) > 0:
            if budget_clock is not None and budget_clock.expired:
                break
            # Get the current/base node
            base_node_id = queue.pop()
            self.discovered.add(base_node_id)
            logging.debug(f"New iteration with base node: {base_node_id}")

            # Determine the neighbours (unvisited)
            neighbours = sorted(set(get_adjacent_nodes(join_paths_df, base_node_id)) - set(self.discovered))
            if len(neighbours) == 0:
                continue

            all_neighbours.update(neighbours)

            # Process every neighbour - join, determine quality, get features
            for node in neighbours:
                self.discovered.add(node)
                logging.debug(f"Adjacent node: {node}")

                # Get the join keys with the highest score
                join_keys = get_relation_properties_node_name(join_paths_df, from_id=base_node_id, to_id=node)
                logging.debug(f"\tJoin keys found: {join_keys}")
                left_table_join_key = join_keys[0][0]['from_column']
                left_table_join_key = f"{base_node_id}.{left_table_join_key}"

                if len(join_keys) == 1:
                    highest_ranked_join_keys = join_keys
                else:
                    highest_ranked_join_keys = []
                    for jk in join_keys:
                        if str(jk[0]['weight']) == str(join_keys[0][0]['weight']):
                            highest_ranked_join_keys.append(jk)
                        else:
                            break

                # Read the neighbour node
                right_df, right_label = get_df_with_prefix(join_paths_df, lake_data_folder, node, table_sep=lake_table_sep, use_polars=self.use_polars)
                logging.debug(f"\tRight table shape: {right_df.shape}")

                current_queue = set()
                logging.debug(f"\tPrevious queue: {previous_queue}")
                while len(previous_queue) > 0:
                    previous_join_name = previous_queue.pop()

                    previous_join = None
                    if previous_join_name == self.base_table_id:
                        previous_join_name = self.base_table_id
                        previous_join = self.partial_join.copy()
                    else:
                        filename_key = self.join_name_mapping[previous_join_name]
                        if self.save_joins_to_disk:
                            previous_join = pd.read_parquet(
                                Path(self.temp_dir.name) / filename_key,
                            )
                        else:
                            previous_join = self.joins_to_df[filename_key]
                            # del self.joins_to_df[filename_key]

                    # self.iters += 1
                    # print(self.iters)

                    # The current node can only be joined through the base node.
                    # If the base node doesn't exist in the previous join path, the join can't be performed

                    if base_node_id not in previous_join_name:
                        logging.debug(f"\tBase node {base_node_id} not in partial join {previous_join_name}")
                        continue

                    for prop in highest_ranked_join_keys:
                        join_prop, from_table, to_table = prop
                        if join_prop['from_label'] != from_table:
                            continue

                        if join_prop['from_column'] == self.target_column:
                            current_queue.add(previous_join_name)
                            continue

                        logging.debug(f"\t\tJoin properties: {join_prop}")

                        # Step - Explore all possible join paths based on the join keys - Compute the name of the join
                        join_name = compute_join_name(join_key_property=prop, partial_join_name=previous_join_name)
                        logging.debug(f"\tJoin name: {join_name}")

                        # Step - Join  
                        # Resolve placeholder column name if needed
                        from_column_resolved = self._resolve_column_name(join_prop['from_column'], previous_join, from_table)
                        left_join_key_resolved = f"{from_table}.{from_column_resolved}"
                        previous_join[left_join_key_resolved] = previous_join[left_join_key_resolved].apply(process_key)
                        joined_df, join_filename, join_columns = self.step_join(
                            join_key_properties=prop, left_df=previous_join, right_df=right_df, right_label=right_label
                        )

                        if joined_df is None:
                            current_queue.add(previous_join_name)
                            if not self.save_joins_to_disk:
                                self.joins_to_df[join_filename] = joined_df
                            continue

                        data_quality = self.step_data_quality(join_key_properties=prop, joined_df=joined_df)
                        if not data_quality:
                            current_queue.add(previous_join_name)
                            if not self.save_joins_to_disk:
                                self.joins_to_df[join_filename] = joined_df
                            continue

                        result = self.streaming_relevance_redundancy(
                            dataframe=joined_df.copy(),
                            new_features=list(right_df.columns),
                            selected_features=self.partial_join_selected_features[previous_join_name],
                        )
                        if result is not None:
                            self.ranking[join_name] = result[0]
                            all_selected_features = self.partial_join_selected_features[previous_join_name]
                            all_selected_features.extend(result[1])
                            self.partial_join_selected_features[join_name] = all_selected_features
                        else:
                            self.partial_join_selected_features[join_name] = self.partial_join_selected_features[
                                previous_join_name
                            ]

                        join_columns.extend(self.join_keys[previous_join_name])
                        self.join_keys[join_name] = join_columns
                        self.join_name_mapping[join_name] = join_filename

                        current_queue.add(join_name)
                        if not self.save_joins_to_disk:
                            self.joins_to_df[join_filename] = joined_df
                # Initialise the queue with the new paths (current_queue)
                previous_queue.update(current_queue)

            _iter_idx += 1
            _emit_progress(_iter_idx)

        if not all_neighbours - self.discovered:
            return
        if budget_clock is not None and budget_clock.expired:
            return
        self.streaming_feature_selection(
            join_paths_df, lake_data_folder, lake_table_sep,
            all_neighbours, previous_queue,
            budget_clock=budget_clock, trajectory_emitter=trajectory_emitter,
        )

    def streaming_relevance_redundancy(
        self, dataframe: pd.DataFrame, new_features: List[str], selected_features: List[str]
    ) -> Optional[Tuple[float, List[dict]]]:
        df = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False
        ).fit_transform(X=dataframe, random_state=42, random_seed=42)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        features = list(set(X.columns).intersection(set(new_features)))
        top_feat = len(features) if len(features) < self.top_k else self.top_k

        relevant_features = new_features
        sum_m = 0
        m = 1
        if not self.no_relevance:
            feature_score_relevance = self.rel_red.measure_relevance(
                dataframe=X, new_features=features, target_column=y
            )[:top_feat]
            if len(feature_score_relevance) == 0:
                return None
            relevant_features = list(dict(feature_score_relevance).keys())
            m = len(feature_score_relevance) if len(feature_score_relevance) > 0 else m
            sum_m = sum(list(map(lambda x: x[1], feature_score_relevance)))

        final_features = relevant_features
        sum_o = 0
        o = 1
        if not self.no_redundancy:
            feature_score_redundancy = self.rel_red.measure_redundancy(
                dataframe=X, selected_features=selected_features, relevant_features=relevant_features, target_column=y
            )

            if len(feature_score_redundancy) == 0:
                return None

            o = len(feature_score_redundancy) if feature_score_redundancy else o
            sum_o = sum(list(map(lambda x: x[1], feature_score_redundancy)))
            final_features = list(dict(feature_score_redundancy).keys())

        score = (o * sum_m + m * sum_o) / (m * o)

        return score, final_features

    def step_join(
        self,
        join_key_properties: tuple,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        right_label: str,
    ) -> Tuple[pd.DataFrame, str, list]:
        logging.debug("\tSTEP Join ... ")
        join_prop, from_table, to_table = join_key_properties
        
        # Resolve placeholder column names to actual column names
        from_column = self._resolve_column_name(join_prop['from_column'], left_df, from_table)
        to_column = self._resolve_column_name(join_prop['to_column'], right_df, right_label)
        
        right_df_fk = f"{right_label}.{to_column}"
        right_df[right_df_fk] = right_df[right_df_fk].apply(process_key)
        # Step - Sample neighbour data - Transform to 1:1 or M:1
        sampled_right_df = right_df
        if self.sample_data_step:
            if self.use_polars:
                right_df_pl = pl.from_pandas(right_df)
                sampled_right_df = right_df_pl.filter(
                    pl.int_range(0, pl.count()).shuffle(seed=42).over(right_df_fk) < 1
                )
            else:
                sampled_right_df = right_df.groupby(right_df_fk).sample(
                    n=1, random_state=42
                )

        # File naming convention as the filename can be gigantic
        join_filename = f"{self.base_table_label}_join_BFS_{self.value_ratio}_{str(uuid.uuid4())}.parquet"

        # Join
        left_on = f"{from_table}.{from_column}"
        right_on = f"{to_table}.{to_column}"
        joined_df = join_and_save(
            left_df=pl.from_pandas(left_df) if self.use_polars else left_df,
            right_df=sampled_right_df,
            left_column_name=left_on,
            right_column_name=right_on,
            join_path=Path(self.temp_dir.name) / join_filename,
            csv=False,
            save_to_disk=self.save_joins_to_disk,
        )
        if joined_df is None:
            return None, join_filename, []

        # Check that the join produced at least some non-NaN values in the new columns
        # New columns are those that were in right_df but not in left_df
        left_cols = set(left_df.columns)
        right_table_columns = [col for col in joined_df.columns if col not in left_cols]
        
        if right_table_columns:
            # Check if at least one new column has at least one non-NaN value
            has_valid_data = False
            for col in right_table_columns:
                non_na_count = joined_df[col].notna().sum()
                if non_na_count > 0:
                    has_valid_data = True
                    break
            
            if not has_valid_data:
                logging.debug(f"\t\tJoin produced only NaN values in all {len(right_table_columns)} new columns from {to_table}.\nSKIPPED Join")
                return None, join_filename, []
            
            # Log statistics about the join quality
            total_rows = len(joined_df)
            valid_cols = sum(1 for col in right_table_columns if joined_df[col].notna().sum() > 0)
            logging.debug(f"\t\tJoin quality: {valid_cols}/{len(right_table_columns)} columns have data")

        return joined_df, join_filename, [left_on, right_on]

    def step_data_quality(self, join_key_properties: tuple, joined_df: pd.DataFrame) -> bool:
        logging.debug("\tSTEP data quality ...")
        join_prop, _, to_table = join_key_properties

        # Resolve placeholder column name
        to_column = self._resolve_column_name(join_prop['to_column'], joined_df, to_table)
        
        # Data Quality check - Prune the joins with high null values ratio
        if joined_df[f"{to_table}.{to_column}"].count() / joined_df.shape[0] < self.value_ratio:
            logging.debug(f"\t\tRight column value ration below {self.value_ratio}.\nSKIPPED Join")
            return False

        return True
