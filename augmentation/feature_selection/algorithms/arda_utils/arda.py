import csv, sys
csv.field_size_limit(sys.maxsize)
import gc
import logging
import time
import math
from typing import List
import polars as pl
import numpy as np
import pandas as pd
import tqdm as tqdm
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from .join_path_utils import compute_join_name
from .neo4j_transactions import (
    get_relation_properties_node_name,
    get_adjacent_nodes,
    get_node_by_id,
)

from ....utils.common import process_key

logging.getLogger().setLevel(logging.WARNING)


def gen_features(A: pd.DataFrame, eta: float):
    """
    Algorithm 2 from "ARDA: Automatic Relational Data Augmentation for Machine Learning"
    :param A: The (normalized) data matrix
    :param eta: The amount of features to generate
    :return: A matrix of generated random features, where each column represents one feature
    """
    np.random.seed(42)  # Set random seed for reproducibility
    L = []
    d = A.shape[1]
    m = np.mean(A, axis=1)
    s = np.cov(A)
    logging.debug(f"\t\tARDA: Generate: {math.ceil(eta * d)} features")
    k = math.ceil(eta * d)
    L = np.random.multivariate_normal(m, s, size=k)
    result = np.array(L).T
    logging.debug(f"\t\tARDA: Generated {result.shape}")
    return result


def _bin_count_ranking(
        feature_importance_scores: np.ndarray, mask: np.ndarray, bin_size: int
) -> List:
    """
    Count how often the "real" features appear in front of the generated features
    :param feature_importance_scores: The rankings as determined by the ranking algorithm
    :param mask: The bit mask indicating which columns were randomly generated (True) and which ones are real features (False)
    :param bin_size: Size of the bin array, corresponds to the amount of columns in the data matrix (the amount of "real" features)
    :return:
    """

    # Get sorting indices for the rankings, flip order since we have feature importance scores
    indices = feature_importance_scores.argsort()[::-1]
    # Sort the mask, so we know where the generated columns are located in terms of ranking
    sorted_mask = mask[indices[::]]
    bins = np.zeros(bin_size)

    # Iterate through this mask until we hit a generated feature
    # Add 1 for all the original features that were in front
    for i, val in zip(indices, sorted_mask):
        if val:
            break
        else:
            bins[i] += 1

    return bins


def select_features(
        normalised_matrix: pd.DataFrame,
        y: pd.Series,
        tau=0.1,
        eta=0.2,
        k=10,
        regression: bool = False,
) -> List:
    """
    Algorithm 1 from "ARDA: Automatic Relational Data Augmentation for Machine Learning"

    :param normalised_matrix: The (normalized) data matrix
    :param y: The label/target column
    :param tau: Threshold for the fraction of how many times a feature appeared in front of synthesized features in ranking
    :param eta: Fraction of random features to inject (fraction of amount of features in A)
    :param k: Number of times ranking and counting is performed
    :param regression: bool - if True: Random Forest Regressor is used, if False: Random Forest Classifier is used
    :return: A set of indices selected by thresholding the normalized frequencies by 'tau'
    """

    if regression:
        estimator = RandomForestRegressor(random_state=42)
    else:
        estimator = RandomForestClassifier(random_state=42)

    d = normalised_matrix.shape[1]
    logging.debug("\tARDA: Generate features")
    features = gen_features(normalised_matrix, eta).astype(np.float64)
    X = np.concatenate(
        (normalised_matrix, features), axis=1, dtype=np.float64
    )  # This gives us A' from the paper
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[d:] = True  # We mark the columns that were generated
    counts = np.zeros(d)

    # Repeat process 'k' times, as in the algorithm
    logging.debug("\tARDA: Decide feature importance")
    for i in range(k):
        estimator.fit(X, y)
        counts += _bin_count_ranking(estimator.feature_importances_, mask, d)
    return np.arange(d)[counts / k > tau]


def wrapper_algo(
        normalised_matrix: pd.DataFrame,
        y: pd.Series,
        T: List[float],
        eta=0.2,
        k=10,
        regression: bool = False,
) -> List:
    """
    Algorithm 3 from "ARDA: Automatic Relational Data Augmentation for Machine Learning"

    :param normalised_matrix: The (normalized) data matrix
    :param y: The label/target column
    :param T: A list with thresholds (see tau in algo 2) to use
    :param eta: Fraction of random features to inject
    :param k: The number of times ranking and counting is performed
    :param regression: bool - if True: Random Forest Regressor is used, if False: Random Forest Classifier is used
    :return: An array of indices, corresponding to selected features from A
    """

    if normalised_matrix.shape[0] != y.shape[0]:
        raise ValueError(
            "Criterion/feature 'y' should have the same amount of rows as 'A'"
        )

    if regression:
        estimator = RandomForestRegressor(random_state=42)
    else:
        estimator = RandomForestClassifier(random_state=42)

    last_accuracy = 0
    last_indices = []

    for t in sorted(T):
        X_train, X_test, y_train, y_test = train_test_split(
            normalised_matrix, y, test_size=0.2, random_state=42
        )
        logging.debug("\nARDA: Select features")
        indices = select_features(
            X_train, y_train, tau=t, eta=eta, k=k, regression=regression
        )
        # If this happens, the thresholds might have been too strict
        if len(indices) == 0:
            return last_indices

        if len(X_train.iloc[:, indices]) == 0:
            return last_indices

        logging.debug("ARDA: Train and score")
        estimator.fit(X_train.iloc[:, indices], y_train)
        accuracy = estimator.score(X_test.iloc[:, indices], y_test)
        if accuracy < last_accuracy:
            break
        else:
            last_accuracy = accuracy
            last_indices = indices
    return last_indices


def select_arda_features_budget_join(
    join_paths_df_path: str,
    query_table: pl.DataFrame,
    query_column_name: str,
    data_lake_folder: str,
    base_node_id: str,
    target_column_name: str,
    sample_size: int,
    regression: bool,
    sep_lake: str
):
    random_state = 42
    final_selected_features = []
    all_columns = []
    right_table_cache = {}  # Cache right tables to avoid re-reading in reconstruction
    join_paths_df = pd.read_csv(join_paths_df_path)
    join_paths_df['from_id'] = base_node_id

    key_cols = join_paths_df['from_column'].unique().tolist()        
    # Read base table, uniform sample, set budget size
    # left_table = query_table.to_pandas()
    left_table = pd.read_csv(f'experiments/base_tables/{base_node_id.replace(".csv", "")}/{base_node_id}')
    for col in key_cols:
        left_table[col] = left_table[col].apply(process_key)
    if sample_size and sample_size < left_table.shape[0]:
        left_table = left_table.sample(sample_size, random_state=random_state)
    budget_size = left_table.shape[0]

    start = time.perf_counter()
    # Get node, prepend the node label to columns and base table features for easy identification
    base_node = get_node_by_id(join_paths_df, base_node_id)
    left_table = (
        left_table.set_index([target_column_name])
        .add_prefix(f"{base_node.get('id')}.")
        .reset_index()
    )
    base_table_columns = list(left_table.columns)
    base_table_columns.remove(target_column_name)

    join_name = base_node.get("id")

    join_keys = []
    # Get directly connected nodes
    nodes = get_adjacent_nodes(join_paths_df, base_node_id)
    while len(nodes) > 0:
        feature_count = 0

        # Join every table according to the budget
        while feature_count <= budget_size and len(nodes) > 0:
            node_id = nodes.pop()
            logging.debug(f"Node id: {node_id}\n\tRemaining nodes: {len(nodes)}")

            # Get the keys between the base node and connected node
            join_key = get_relation_properties_node_name(
                join_paths_df=join_paths_df, from_id=base_node_id, to_id=node_id
            )[0]
            join_prop, from_table, to_table = join_key
            logging.debug(f"Join properties: {join_prop}")

            if join_prop["from_label"] == base_node.get("id"):
                if join_prop["from_column"] == target_column_name:
                    continue

            if join_prop["to_label"] == base_node.get("id"):
                if join_prop["to_column"] == target_column_name:
                    continue

            if join_prop["from_label"] == to_table:
                from_column = join_prop["to_column"]
                to_column = join_prop["from_column"]
            else:
                from_column = join_prop["from_column"]
                to_column = join_prop["to_column"]

            # Read right table, aggregate on the join key (reduce to 1:1 or M:1 join) by random sampling
            right_table = pd.read_csv(
                f'{data_lake_folder}/{node_id}',
                header=0,
                engine="python",
                encoding="utf8",
                on_bad_lines='skip',
                sep=sep_lake
            )
            try:
                right_table = right_table.groupby(to_column).sample(
                    n=1, random_state=random_state
                )
            except ValueError:
                continue
            right_table[to_column] = right_table[to_column].apply(process_key)
            # Cache the deduplicated right table for reuse in reconstruction
            right_table_cache[node_id] = right_table

            # Prepend node label to every column for easy identification
            right_node = get_node_by_id(join_paths_df, node_id)
            right_table = right_table.add_prefix(f"{right_node.get('id')}.")
            # Join tables, drop the right key as we don't need it anymore
            if (
                    left_table[f"{from_table}.{from_column}"].dtype
                    != right_table[f"{to_table}.{to_column}"].dtype
            ):
                logging.debug(
                    f"Column dtype mismatch: {from_table}.{from_column} ({left_table[f'{from_table}.{from_column}'].dtype}) "
                    f"and {to_table}.{to_column} ({right_table[f'{to_table}.{to_column}'].dtype}). Skipping join."
                )
                continue

            left_on = f"{from_table}.{from_column}"
            right_on = f"{to_table}.{to_column}"
            left_table = pd.merge(
                left_table,
                right_table,
                how="left",
                left_on=left_on,
                right_on=right_on,
            )
            join_keys.append(left_on)
            join_keys.append(right_on)

            # Compute the join name
            join_name = compute_join_name(
                join_key_property=join_key, partial_join_name=join_name
            )
            logging.debug(f"\t\t\tJoin name: {join_name}")

            # Update feature count (subtract 1 for the deleted right key)
            feature_count += right_table.shape[1] - 1
            logging.debug(f"Feature count: {feature_count}")

        # Compute the columns of the batch and create the batch dataset
        columns = set(left_table.columns) - set(all_columns) - set(base_table_columns)
        columns = list(set(columns) - set(join_keys))

        logging.debug(f"{len(columns)} columns to select")

        # If the algorithm failed
        if len(columns) == 0:
            logging.debug("No selected column")
            continue

        # If the algorithm doesn't find any new feature
        if len(columns) == 1 and target_column_name in columns:
            logging.debug("No selected column")
            continue

        # If the algorithm only selects one feature
        if len(columns) == 2 and target_column_name in columns:
            columns.remove(target_column_name)
            final_selected_features.extend(columns)
            logging.debug(f"Selected columns: {columns}")
            continue

        joined_tables_batch = left_table[columns]
        logging.debug(f"shape: {joined_tables_batch.shape}")

        # Save the computed columns
        all_columns.extend(columns)
        all_columns.remove(target_column_name)
        # Prepare data
        X = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False
        ).fit_transform(X=joined_tables_batch).astype('float')
        X = X.fillna(X.mean())

        y = X[target_column_name]
        X.drop(columns=[target_column_name], inplace=True)
        if X.empty:
            continue

        # Run ARDA - RIFS (Random Injection Feature Selection) algorithm
        T = np.arange(0.0, 1.0, 0.1)
        indices = wrapper_algo(X, y, T, regression=regression)
        fs_X = X.iloc[:, indices].columns
        logging.debug(f"Selected columns: {fs_X}")

        # Save the selected columns of the batch
        final_selected_features.extend(fs_X)

    end = time.perf_counter()
    fetch_time = 42.0 + (end - start)
    start = time.perf_counter()
    final_selected_features_dict = {}
    del left_table  # Free the sampled left_table with all accumulated joins
    left_table = query_table.to_pandas()
    del query_table  # Free the Polars copy now that we have the Pandas one
    gc.collect()
    augplan = []
    if len(final_selected_features) > 0:
        for v in final_selected_features:
            file, col = v.split('.', 1)  # split only at the first '.'
            # if filenames may contain dots, split from the right instead:
            # file, col = v.rsplit('.', 1)
            col = col.split('.', 1)[-1]
            col = col.split('.', 1)[0]
            final_selected_features_dict.setdefault(file, []).append(col)
        for table_name, features in final_selected_features_dict.items():
            if '.csv' not in table_name:
                table_name += '.csv'
            query_column_name = join_paths_df[join_paths_df["to_id"] == table_name]['from_column'].values[0]
            join_key = join_paths_df[
                (join_paths_df["from_column"] == query_column_name) &
                (join_paths_df["to_id"] == table_name)
            ]['to_column'].values[0]
            # Reuse cached right table if available, otherwise read from disk
            if table_name in right_table_cache:
                right_table = right_table_cache[table_name]
            else:
                right_table = pd.read_csv(
                    f'{data_lake_folder}/{table_name}',
                    header=0,
                    engine="c",
                    encoding="utf8",
                    on_bad_lines='skip',
                    sep=sep_lake
                )
                right_table = right_table.groupby(join_key).sample(
                    n=1, random_state=random_state
                )
                right_table[join_key] = right_table[join_key].apply(process_key)
            left_table[query_column_name] = left_table[query_column_name].apply(process_key)
            features = [f.replace('_x', '').replace('_y', '') for f in features]
            right_table = right_table[[join_key, *features]]
            right_table = right_table.rename({join_key: query_column_name}, axis=1)
            left_table = pd.merge(
                left_table,
                right_table,
                how="left",
                on=query_column_name
            )
            augplan.extend(features)
    del right_table_cache  # Free cached right tables
    gc.collect()
    end = time.perf_counter()
    augmentation_time = end - start
    left_table_columns = left_table.columns.tolist()
    left_table_columns_non_unique = [col for col in left_table_columns if left_table_columns.count(col) > 1]
    left_table_columns_non_unique_renamed = [col + '_right' if col in left_table_columns_non_unique else col for col in left_table_columns]
    left_table.columns = left_table_columns_non_unique_renamed
    left_table.reset_index(drop=True, inplace=True)
    left_table = pl.from_pandas(left_table)

    return left_table, fetch_time, augmentation_time, augplan
