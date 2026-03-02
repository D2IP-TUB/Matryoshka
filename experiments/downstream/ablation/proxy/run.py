"""
Proxy Fidelity Experiment
=========================
Measures how well the linear proxy model's train score tracks
the downstream Random Forest CV score as features are added in the order
chosen by forward selection.

For each experiment log directory:
  1. Load the augmented CSV and the ordered augmentation plan.
  2. Identify base features (pre-augmentation columns).
  3. Progressively add features in forward-selection order.
  4. At each step, evaluate:
       - Proxy model train score (LinearRegression / LDA fit on full data)
       - Downstream model CV score (RandomForest via 5-fold CV)
  5. Store all step-wise scores in a single CSV for plotting.
"""
import subprocess
import polars as pl
from experiments.downstream.experiment_executor import ExperimentExecutor
from augmentation.utils.config import DiscoveryConfig
from augmentation.join_selection import JoinSelection
from experiments.base_tables.base_table_preprocessing import PreProcessor
import os
import yaml
import gc
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
)

from experiments.base_tables.base_scoring import SimpleTrainer

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
BASE_TABLES_DIR = "experiments/base_tables"

CV_FOLDS = 5


def load_experiment(log_path: str, dir_name: str):
    """Return (augmented_df, augmentation_plan, splits_config, dir_name)."""
    # Find augmented CSV and plan pickle (filenames vary slightly)
    csv_file = None
    plan_file = None
    for f in os.listdir(log_path):
        if f.startswith("augmented_") and f.endswith(".csv"):
            csv_file = os.path.join(log_path, f)
        if f.startswith("augmentation_plan_") and f.endswith(".pkl"):
            plan_file = os.path.join(log_path, f)
    if csv_file is None or plan_file is None:
        return None

    df = pd.read_csv(csv_file)
    with open(plan_file, "rb") as f:
        plan = pickle.load(f)

    # Derive table name: dir pattern is  <lake>_<table>_[<model>_]<strategy>
    # E.g. nyc_energy_forward  or  cuk_arrest_ClassificationCholesky_forward
    parts = dir_name.split("_")
    table_name = parts[1]

    splits_path = os.path.join(BASE_TABLES_DIR, table_name, "splits.json")
    with open(splits_path) as f:
        splits = json.load(f)[0]

    return df, plan, splits, table_name


def identify_base_features(df: pd.DataFrame, plan: list[str], target: str, query_col: str) -> list[str]:
    """Return the list of base-table feature columns (everything that is not
    an augmentation-plan column, target, or query column)."""
    plan_set = set(plan)
    exclude = {target, query_col}
    return [c for c in df.columns
            if c not in exclude and c not in plan_set and not c.startswith("Unnamed")]


def evaluate_step(
    X_raw: pd.DataFrame,
    X_imputed: pd.DataFrame,
    y: np.ndarray,
    problem_type: str,
    target_col: str,
) -> dict:
    """Evaluate proxy (train score) + downstream (CV score) and return a dict.

    Parameters
    ----------
    X_raw : pd.DataFrame
        Original (non-imputed) feature slice – used for the proxy model
        with mean imputation, matching the main pipeline.
    X_imputed : pd.DataFrame
        Pre-imputed (IterativeImputer) feature slice – used for the
        downstream RF evaluation (no additional imputation needed).
    problem_type : str
        "regression" or "classification".
    target_col : str
        Name of the target column (needed by SimpleTrainer).
    """
    results = {}

    # ── Proxy model (mean-imputed, train score on full data) ─────
    imputer = SimpleImputer(strategy="mean")
    X_proxy = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)

    try:
        if problem_type == "regression":
            proxy = LinearRegression()
            proxy.fit(X_proxy, y)
            proxy_preds = proxy.predict(X_proxy)
            results["proxy_mse"] = mean_squared_error(y, proxy_preds)
        else:
            proxy = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            proxy.fit(X_proxy, y)
            proxy_preds = proxy.predict(X_proxy)
            results["proxy_f1"] = f1_score(y, proxy_preds, average="weighted")
    except Exception as e:
        print(f"    [proxy failed: {e}]")
        results["proxy_mse" if problem_type == "regression" else "proxy_f1"] = np.nan

    # ── Downstream model (SimpleTrainer: grid search + K-fold CV) ─
    try:
        train_data = X_imputed.copy()
        train_data[target_col] = y

        # SimpleTrainer expects "regression" / "binary" / "multiclass"
        st_problem_type = problem_type if problem_type == "regression" else "multiclass"

        trainer = SimpleTrainer(
            problem_type=st_problem_type,
            target_column=target_col,
            output_dir=os.path.join(LOGS_DIR, "_tmp_model"),
        )
        trainer.train(train_data, cv_folds=CV_FOLDS, verbosity=0)
        perf = trainer.evaluate(train_data, cv_folds=CV_FOLDS)

        if problem_type == "regression":
            results["downstream_mse"] = perf["rmse"] ** 2
        else:
            results["downstream_f1"] = perf.get("f1_weighted", perf.get("f1", np.nan))
    except Exception as e:
        print(f"    [downstream failed: {e}]")
        results["downstream_mse" if problem_type == "regression" else "downstream_f1"] = np.nan

    return results


def run_experiment(log_path: str, dir_name: str) -> pd.DataFrame | None:
    """Run the full progressive evaluation for one experiment."""
    loaded = load_experiment(log_path, dir_name)
    if loaded is None:
        print(f"  Skipping {dir_name}: missing files")
        return None
    df, plan, splits, table_name = loaded

    target = splits["target"]
    query_col = splits["query_col"]
    problem_type = "regression" if splits["target_type"] == "continuous" else "classification"

    base_features = identify_base_features(df, plan, target, query_col)
    # Filter plan to features actually present in the CSV
    plan = [f for f in plan if f in df.columns]

    if len(plan) == 0:
        print(f"  Skipping {dir_name}: no augmentation features found in CSV")
        return None

    print(f"  {dir_name}: {problem_type}, {len(base_features)} base features, {len(plan)} augmented features")

    y = df[target].values
    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    df.drop(columns=[query_col], inplace=True)

    # ── Pre-impute full augmented dataset for downstream RF ──────
    all_features = base_features + plan
    print(f"  Running IterativeImputer on {len(all_features)} features ...")
    iter_imputer = IterativeImputer(
        estimator=BayesianRidge(),
        sample_posterior=False,
        random_state=42,
        n_nearest_features=None,
    )
    df_full_imputed = pd.DataFrame(
        iter_imputer.fit_transform(df[all_features]),
        columns=all_features,
        index=df.index,
    )
    print(f"  IterativeImputer done.")

    # Evaluate every step (no subsampling)
    n_plan = len(plan)
    eval_indices = list(range(n_plan + 1))  # 0 = base only, 1..n_plan = adding features

    rows = []
    for i in eval_indices:
        cols = base_features + plan[:i]
        X_raw_step = df[cols]
        X_imp_step = df_full_imputed[cols]
        step_results = evaluate_step(X_raw_step, X_imp_step, y, problem_type, target)
        step_results["step"] = i
        step_results["n_augmented_features"] = i
        step_results["n_total_features"] = len(cols)
        step_results["table"] = table_name
        step_results["task"] = problem_type
        step_results["experiment"] = dir_name
        rows.append(step_results)
        # Progress
        metric_key = "proxy_mse" if problem_type == "regression" else "proxy_f1"
        ds_key = "downstream_mse" if problem_type == "regression" else "downstream_f1"
        print(f"    step {i:3d}/{n_plan}: {metric_key}={step_results.get(metric_key, 'N/A'):.4f}  "
              f"{ds_key}={step_results.get(ds_key, 'N/A'):.4f}")

    return pd.DataFrame(rows)


def main():
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # subprocess.run(["bash", os.path.join(script_dir, "generate_config.sh")])
    # config_df = pl.read_csv(os.path.join(script_dir, "experiments.csv"))
    # config_df = config_df.filter(
    #     (pl.col('strategy').str.contains('forward')) &
    #     (pl.col('table').is_in(['energy', 'arrest', 'vgsales', 'pageviews']))
    # )
    # base_tables_dir = 'experiments/base_tables/'
    # for lake, config in config_df.group_by('lake'):
    #     headers = config.columns
    #     for row in config.iter_rows():
    #         exp_config = {k: row[i] for i, k in enumerate(headers)}
    #         lake = exp_config.pop('lake')
    #         base_table_name = exp_config.pop('table')
    #         strategy = exp_config.pop('algorithm')
    #         baseline = exp_config.pop('baseline')
    #         strat_index = headers.index('strategy')
    #         strat = row[strat_index]

    #         base_table_path = os.path.join(base_tables_dir, base_table_name, f'{base_table_name}.csv')
    #         base_table_splits_path = os.path.join(base_tables_dir, base_table_name, 'splits.json')
    #         with open(base_table_splits_path, 'r') as f:
    #             splits = json.load(f)
    #         features = splits[0]['features']
    #         preprocessor = PreProcessor(base_table_path, base_table_splits_path, 0)
    #         X, query_col, target, nan_mask = preprocessor.run()
    #         X.write_csv(
    #             os.path.join(
    #                 base_tables_dir,
    #                 base_table_name,
    #                 f'{base_table_name}_preprocessed.csv'
    #             )
    #         )

    #         execution_data = ExperimentExecutor.from_params(exp_config, strat)
    #         worker_init_kwargs = execution_data.init_args
    #         combination_hash = f'{lake}_{base_table_name}_{strategy}'
    #         log_dir = os.path.join(script_dir, 'logs', combination_hash)
    #         if not os.path.exists(os.path.join(script_dir, 'logs', combination_hash)):
    #             os.makedirs(log_dir)
    #         log_file_name = f'{combination_hash}.log'
    #         worker_init_kwargs.update({
    #             'verbose': True,
    #             'log_dir': log_dir,
    #             'log_file_name': log_file_name
    #         })
    #         worker = JoinSelection(**worker_init_kwargs)
    #         discovery_config_kwargs = {}
    #         discovery_config_kwargs['baseline'] = baseline
    #         params = execution_data.discovery_config_args['params']
    #         discovery_config_kwargs['params'] = params
    #         if baseline:
    #             for key in execution_data.discovery_config_args:
    #                 if key == 'strategy':
    #                     discovery_config_kwargs['strategy'] = execution_data.discovery_config_args['strategy']
    #                 elif key == 'params':
    #                     pass
    #                 else:
    #                     discovery_config_kwargs['params'].update({key: execution_data.discovery_config_args[key]})
    #             for key in execution_data.unknown_args:
    #                 discovery_config_kwargs['params'].update({key: execution_data.unknown_args[key]})
    #             for key in execution_data.run_args:
    #                 discovery_config_kwargs['params'].update({key: execution_data.run_args[key]})

    #             if strategy == 'arda':
    #                 discovery_config_kwargs['params']['query_table'] = X
    #                 discovery_config_kwargs['params']['features'] = features
    #             elif strategy == 'kitana':
    #                 discovery_config_kwargs['params']['query_table_path'] = base_table_path
    #                 discovery_config_kwargs['params']['features'] = features
    #             elif strategy == 'qcr':
    #                 discovery_config_kwargs['params']['query_table'] = X
    #             elif strategy == 'autofeat':
    #                 discovery_config_kwargs['params']['base_table_sep'] = ','
    #                 discovery_config_kwargs['params']['problem_type'] = splits[0]['target_type']
    #                 discovery_config_kwargs['params']['features'] = features
    #         else:
    #             execution_data.discovery_config_args.pop('data_lake_path')
    #             execution_data.discovery_config_args.pop('lake_table_sep')
    #             discovery_config_kwargs = execution_data.discovery_config_args
    #         config = DiscoveryConfig(**discovery_config_kwargs)

    #         find_best_joins_kwargs = {'user_table_processed': X, 'top_k': 20, 'n_jobs': 16, 'config': config}
    #         with open('experiments/downstream/config.yml', 'r') as f:
    #             config = yaml.safe_load(f)
    #         base_tables = config['lakes'][lake]['base_tables']
    #         for base_table in base_tables:
    #             for key in base_table:
    #                 if key == base_table_name:
    #                     task = base_table[key][0]['task']
            
    #         if task == 'regression':
    #             find_best_joins_kwargs['corr_threshold'] = 0.55
    #         elif task == 'classification':
    #             find_best_joins_kwargs['corr_threshold'] = 0.01
    #         find_best_joins_kwargs.update(execution_data.run_args)

    #         errors_log_file = os.path.join(script_dir, 'errors.log')
    #         try:
    #             df_aug, augplan = worker.find_best_joins(**find_best_joins_kwargs, debug=True)
    #             # validate_kwargs(worker.find_best_joins, find_best_joins_kwargs)
    #             # print(execution_data.discovery_config_args)
    #             # print(find_best_joins_kwargs)
    #             df_aug.write_csv(
    #                 os.path.join(
    #                     log_dir,
    #                     f'augmented_{combination_hash}.csv'
    #                 )
    #             )
    #             with open(os.path.join(log_dir, f'augmentation_plan_{combination_hash}.pkl'), 'wb') as f:
    #                 pickle.dump(augplan, f)
    #         except Exception as e:
    #             df_aug = X
    #             df_aug.write_csv(
    #                 os.path.join(
    #                     log_dir,
    #                     f'augmented_{combination_hash}.csv'
    #                 )
    #             )
    #             augplan = []
    #             with open(os.path.join(log_dir, f'augmentation_plan_{combination_hash}.pkl'), 'wb') as f:
    #                 pickle.dump(augplan, f)
    #             with open(errors_log_file, 'a') as f:
    #                 f.write(f'Error for combination {combination_hash}: {str(e)}\n')
    #         finally:
    #             # Free memory between experiments to prevent OOM
    #             worker = df_aug = augplan = config = X = find_best_joins_kwargs = None
    #             gc.collect()

    all_results = []
    experiment_dirs = sorted(os.listdir(LOGS_DIR))

    for dir_name in experiment_dirs:
        log_path = os.path.join(LOGS_DIR, dir_name)
        if not os.path.isdir(log_path):
            continue
        print(f"\n{'='*60}")
        print(f"Processing: {dir_name}")
        print(f"{'='*60}")
        result_df = run_experiment(log_path, dir_name)
        if result_df is not None:
            all_results.append(result_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(LOGS_DIR, "proxy_fidelity_results.csv")
        combined.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")
        print(f"  Total rows: {len(combined)}")
        print(combined.groupby("experiment")[["step"]].max().to_string())
    else:
        print("\nNo results produced.")


if __name__ == "__main__":
    main()
