import logging
import time
import os
import tempfile
from typing import Optional, List
import random
import numpy as np

import pandas as pd
import typer as typer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split

# Set PYTHONHASHSEED for deterministic hashing
os.environ['PYTHONHASHSEED'] = '42'

from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class Result:
    TFD_PATH = "TFD_PATH"
    TFD = "AutoFeat"
    TFD_REL = "AutoFeat_Rel"
    TFD_RED = "AutoFeat_Red"
    TFD_Pearson = "AutoFeat-Pearson-MRMR"
    TFD_Pearson_JMI = "AutoFeat-Pearson-JMI"
    TFD_JMI = "AutoFeat-Spearman-JMI"
    ARDA = "ARDA"
    JOIN_ALL_BFS = "Join_All_BFS"
    JOIN_ALL_BFS_F = "Join_All_BFS_Filter"
    JOIN_ALL_BFS_W = "Join_All_BFS_Wrapper"
    JOIN_ALL_DFS = "Join_All_DFS"
    JOIN_ALL_DFS_F = "Join_All_DFS_Filter"
    JOIN_ALL_DFS_W = "Join_All_DFS_Wrapper"
    BASE = "BASE"

    algorithm: str
    data_path: str = None
    approach: str = None
    data_label: str = None
    join_time: Optional[float] = None
    total_time: float = 0.0
    feature_selection_time: Optional[float] = None
    depth: Optional[int] = None
    accuracy: Optional[float] = None
    train_time: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    join_path_features: List[str] = None
    cutoff_threshold: Optional[float] = None
    redundancy_threshold: Optional[float] = None
    rank: Optional[int] = None
    top_k: int = None

    def __post_init__(self):
        if self.join_time is not None:
            self.total_time += self.join_time

        if self.train_time is not None:
            self.total_time += self.train_time

        if self.feature_selection_time is not None:
            self.total_time += self.feature_selection_time

hyper_parameters = [
    # {"LR": {'penalty': 'L1'}},
    {"RF": {}},
    # {"GBM": {}},
    # {"XT": {}},
    # {"XGB": {}},
    # {'KNN': {}},
]


def get_hyperparameters(algorithm: Optional[str] = None) -> List[dict]:
    if algorithm is None:
        return hyper_parameters

    if algorithm == 'LR':
        return [{'LR': {'penalty': 'L1'}}]

    model = {algorithm: {}}
    if model in hyper_parameters:
        return [model]
    else:
        raise typer.BadParameter(
            "Unsupported algorithm. Choose one from the list: [RF, GBM, XT, XGB, KNN, LR]."
        )


def run_auto_gluon(dataframe: pd.DataFrame, target_column: str, problem_type: str, algorithms_to_run: dict, automl_results_folder: str | None = None):
    from autogluon.tabular import TabularPredictor
    import hashlib

    # The dataset splits label a regression target as 'continuous', but AutoGluon
    # only recognises 'binary'/'multiclass'/'regression'/'quantile' -- passing
    # 'continuous' raises NotImplementedError in Cleaner.construct and also breaks
    # the regression score-type selection below. Normalise here.
    if problem_type == 'continuous':
        problem_type = 'regression'

    # Default AutoGluon model path: prefer a writable project-local tmp dir,
    # falling back to the system tmp. Avoid hard-coded /mnt paths that may be
    # read-only on some hosts.
    if automl_results_folder is None:
        env_dir = os.environ.get('AUTOFEAT_AUTOML_DIR')
        if env_dir:
            automl_results_folder = env_dir
        else:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')
            )
            candidate = os.path.join(project_root, 'tmp', 'autogluon')
            try:
                os.makedirs(candidate, exist_ok=True)
                automl_results_folder = candidate
            except (OSError, PermissionError):
                automl_results_folder = os.path.join(tempfile.gettempdir(), 'autogluon')
    os.makedirs(automl_results_folder, exist_ok=True)

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    start = time.time()

    logging.debug(f"Train algorithms: {list(algorithms_to_run.keys())} with AutoGluon ...")
    # Replace all inf/-inf with NaN across entire dataframe, then drop rows with NaN target
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    dataframe = dataframe.dropna(subset=[target_column])
    # Final safety: force-coerce target to numeric and drop any remaining non-finite
    dataframe[target_column] = pd.to_numeric(dataframe[target_column], errors='coerce')
    dataframe = dataframe.dropna(subset=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=[target_column]),
        dataframe[[target_column]],
        test_size=0.2,
        random_state=10,
    )
    join_path_features = list(X_train.columns)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[target_column] = y_train.values
    X_test[target_column] = y_test.values

    # Use deterministic path based on data hash
    data_hash = hashlib.md5(pd.util.hash_pandas_object(X_train, index=True).values).hexdigest()[:8]
    model_path = f'{automl_results_folder}/models_{data_hash}'
    
    predictor = TabularPredictor(label=target_column,
                                 problem_type=problem_type,
                                 verbosity=0,
                                 path=model_path).fit(train_data=X_train, hyperparameters=algorithms_to_run)
    score_type = 'accuracy'
    if problem_type == 'regression':
        score_type = 'root_mean_squared_error'

    results = []

    model_names = sorted(predictor.model_names())  # Sort for deterministic order
    # Use all models except ensemble (which is typically last when sorted)
    models_to_eval = [m for m in model_names if 'Ensemble' not in m and 'WeightedEnsemble' not in m]
    for model in models_to_eval:
        result = predictor.evaluate(data=X_test, model=model)
        accuracy = abs(result[score_type])
        ft_imp = predictor.feature_importance(
            data=X_test, model=model, feature_stage="original"
        )
        # Sort feature importance for deterministic ordering
        ft_imp_sorted = ft_imp.sort_index()
        entry = Result(
            algorithm=model,
            accuracy=accuracy,
            feature_importance=dict(zip(list(ft_imp_sorted.index), ft_imp_sorted["importance"])),
            join_path_features=join_path_features,
        )

        results.append(entry)

    end = time.time()

    return end - start, results


def evaluate_all_algorithms(dataframe: pd.DataFrame, target_column: str, algorithm: str, problem_type: str = 'binary'):
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    all_results = []
    
    try:
        df = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False
        ).fit_transform(X=dataframe, random_state=42, random_seed=42)
    except TypeError:
        # Fallback if the version doesn't support these parameters
        df = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False
        ).fit_transform(X=dataframe)

    # Drop rows with non-finite target values (NaN from left joins)
    df = df.dropna(subset=[target_column])

    logging.debug(f"Training AutoGluon ... ")
    runtime, results = run_auto_gluon(
        dataframe=df,
        target_column=target_column,
        algorithms_to_run={"LR": {}},
        problem_type=problem_type,
    )

    for res in results:
        res.train_time = runtime
        res.total_time += res.train_time
    all_results.extend(results)

    return all_results, df