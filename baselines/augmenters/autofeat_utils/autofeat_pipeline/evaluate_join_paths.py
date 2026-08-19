import logging
import os
import pickle
import random
from typing import Tuple, List

import numpy as np
import pandas as pd
import tqdm
from sklearnex import patch_sklearn
logging.getLogger("sklearnex").disabled = True
patch_sklearn()
from .autofeat import AutoFeat
from .join_path_utils import get_path_length
from .evaluation_algorithms import evaluate_all_algorithms
from .neo4j_transactions import get_df_with_prefix


def evaluate_paths(bfs_result: AutoFeat, problem_type: str, algorithm: str, join_paths_df: pd.DataFrame, lake_data_folder: str, lake_table_sep: str, base_table_sep: str, top_k_paths: int = 15, budget_clock=None, partial_state_path: str | None = None) -> Tuple[List, List[Tuple]]:
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    
    logging.debug(f"Evaluate top-{top_k_paths} paths ... ")
    # Sort with explicit ordering for deterministic results:
    # 1. Higher rank first (-rank)
    # 2. Shorter paths first (length, not negated)
    # 3. Alphabetically by path name for stability
    sorted_paths = sorted(
        bfs_result.ranking.items(), 
        key=lambda r: (-float(r[1]), get_path_length(r[0]), r[0])
    )
    top_k_path_list = sorted_paths if len(sorted_paths) < top_k_paths else sorted_paths[:top_k_paths]
    base_features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]

    all_results = []
    selected_features = set()
    print(f"Evaluating {top_k_path_list} paths...")
    for path in tqdm.tqdm(top_k_path_list):
        # Cooperative budget check at iteration boundary. Once the wall-clock
        # deadline has been reached the loop exits with whatever results were
        # accumulated so far. Catches the common case where the SIGALRM hard
        # cap is delayed by a long C-extension call inside the previous
        # iteration's ``evaluate_all_algorithms``.
        if budget_clock is not None and getattr(budget_clock, 'expired', False):
            print(f"[evaluate_paths] budget exhausted; stopping after {len(all_results)} paths", flush=True)
            break
        join_name, rank = path
        if join_name == bfs_result.base_table_id:
            continue

        features = bfs_result.partial_join_selected_features[join_name]
        features.append(bfs_result.target_column)
        features.extend(base_features)
        logging.debug(f"Feature before join_key removal:\n{features}")
        # features = list((set(features) - set(bfs_result.join_keys[join_name])).intersection(set(dataframe.columns)))
        # logging.debug(f"Feature after join_key removal:\n{features}")

        features_tables = [f"{feat.split('.csv')[0]}.csv" for feat in features]
        features_tables.sort()
        features_tables = set(features_tables)

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
            # if not (type(path_aux) is list) and (path_aux not in path_tables.keys()):
            #     continue
            path_list.append(path_aux)

        # dataframe = join_from_path(path_list, bfs_result.target_column, bfs_result.base_table_id)
        dataframe = join_from_path(path_list, join_paths_df, lake_data_folder, lake_table_sep, base_table_sep, bfs_result.base_table_id, bfs_result.target_column)
        
        # Skip if join failed to produce a valid dataframe
        if dataframe is None:
            continue
            
        features = list(set(features).intersection(set(dataframe.columns)))

        if len(features) < 2:
            features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]
            features.append(bfs_result.target_column)

        results, _ = evaluate_all_algorithms(dataframe=dataframe[features],
                                             target_column=bfs_result.target_column,
                                             problem_type=problem_type,
                                             algorithm=algorithm)
        for result in results:
            result.rank = rank
            result.data_path = path_list
            for f in result.join_path_features:
                selected_features.add(f)
        print('Results for path:', results)
        all_results.extend(results)

        # Persist after each iteration so an external killer (SIGALRM, SIGTERM,
        # OOM, or a hard-deadline watchdog) leaves usable partial state behind.
        if partial_state_path is not None:
            try:
                with open(partial_state_path, 'wb') as _f:
                    pickle.dump({
                        'all_results': all_results,
                        'selected_features': list(selected_features),
                        'top_k_path_list': top_k_path_list,
                    }, _f)
            except Exception:
                pass

        dataframe = None

    selected_features = list(selected_features)

    return all_results, top_k_path_list, selected_features


def join_from_path(path: list[str], join_paths_df, lake_data_folder: str, lake_table_sep: str, base_table_sep: str, base_node: str, target: str = None):
    step = 3
    joined_df = None
    path.remove(base_node)
    left_table, _ = get_df_with_prefix(
        join_paths_df,
        lake_data_folder,
        base_node,
        base_table_sep,
        target
    )
    for p in path:
        try:
            right_table, _ = get_df_with_prefix(join_paths_df, lake_data_folder, p, lake_table_sep)
            to_column = join_paths_df[
                join_paths_df['to_id'] == p
            ]['to_column'].values[0]
            left_on = join_paths_df[
                join_paths_df['from_id'] == base_node
            ]['from_column'].values[0]
            
            # Resolve placeholder column names (col_0, col_1, etc.) to actual column names
            if to_column.startswith('col_'):
                try:
                    col_index = int(to_column.split('_')[1])
                    actual_columns = [col.replace(f"{p}.", "", 1) for col in right_table.columns if col.startswith(f"{p}.")]
                    if col_index < len(actual_columns):
                        to_column = actual_columns[col_index]
                except (ValueError, IndexError):
                    pass
                    
            if left_on.startswith('col_'):
                try:
                    col_index = int(left_on.split('_')[1])
                    actual_columns = [col.replace(f"{base_node}.", "", 1) for col in left_table.columns if col.startswith(f"{base_node}.")]
                    if col_index < len(actual_columns):
                        left_on = actual_columns[col_index]
                except (ValueError, IndexError):
                    pass
            
            right_table = right_table.groupby(f'{p}.{to_column}').sample(n=1, random_state=42)

            joined_df = pd.merge(
                left_table,
                right_table,
                how="left",
                left_on=f'{base_node}.{left_on}',
                right_on=f'{p}.{to_column}'
            )
        except Exception as e:
            continue

    return joined_df


def create_join_tree(table, path_tables):
    if table in path_tables.keys():
        value = path_tables[table]
        result = create_join_tree(value[0], path_tables)
        if type(result) is not list:
            result = [result]

        result.append(value[1])
        result.append(value[2])
        result.append(value[3])
        return result

    return table