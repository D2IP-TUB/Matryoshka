import inspect
import json
import os
import pickle
import subprocess
import pandas as pd
import polars as pl
import yaml

os.environ['PYTHONPATH'] = '.'
from augmentation.join_selection import JoinSelection
from augmentation.utils.config import DiscoveryConfig
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from experiments.base_tables.base_scoring import SimpleTrainer
from experiments.base_tables.base_table_preprocessing import PreProcessor
from experiments.downstream.experiment_executor import ExperimentExecutor
from sklearnex import patch_sklearn
patch_sklearn()
import logging
logging.getLogger('sklearnex').setLevel(logging.WARNING)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from res import save_results


def validate_kwargs(method, kwargs):
    sig = inspect.signature(method)
    try:
        sig.bind(**kwargs)
        return True
    except TypeError as e:
        raise e

if __name__ == "__main__":
    script_dir = 'experiments/downstream/'
    subprocess.run(["bash", os.path.join(script_dir, "generate_config.sh")])
    config_df = pl.read_csv(os.path.join(script_dir, "experiments.csv"))
    config_df = config_df.filter(pl.col('strategy')=='ForwardSelection')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_tables_dir = 'experiments/base_tables/'
    top_k_choices = [5, 10, 20, 50]
    for top_k in top_k_choices:
        for lake, config in config_df.group_by('lake'):
            headers = config.columns
            for row in config.iter_rows():
                exp_config = {k: row[i] for i, k in enumerate(headers)}
                lake = exp_config.pop('lake')
                base_table_name = exp_config.pop('table')
                strategy = exp_config.pop('algorithm')
                baseline = exp_config.pop('baseline')
                strat_index = headers.index('strategy')
                strat = row[strat_index]

                base_table_path = os.path.join(base_tables_dir, base_table_name, f'{base_table_name}.csv')
                base_table_splits_path = os.path.join(base_tables_dir, base_table_name, 'splits.json')
                with open(base_table_splits_path, 'r') as f:
                    splits = json.load(f)
                features = splits[0]['features']
                preprocessor = PreProcessor(base_table_path, base_table_splits_path, 0)
                X, query_col, target, nan_mask = preprocessor.run(skip_num_features=True, binning=False)
                X.write_csv(
                    os.path.join(
                        base_tables_dir,
                        base_table_name,
                        f'{base_table_name}_preprocessed.csv'
                    )
                )

                execution_data = ExperimentExecutor.from_params(exp_config, strat)
                worker_init_kwargs = execution_data.init_args
                combination_hash = f'{lake}_{base_table_name}_{top_k}'
                log_dir = os.path.join(script_dir, 'logs', combination_hash)
                if not os.path.exists(os.path.join(script_dir, 'logs', combination_hash)):
                    os.makedirs(log_dir)
                log_file_name = f'{combination_hash}.log'
                worker_init_kwargs.update({
                    'verbose': True,
                    'log_dir': log_dir,
                    'log_file_name': log_file_name
                })
                worker = JoinSelection(**worker_init_kwargs)
                discovery_config_kwargs = {}
                discovery_config_kwargs['baseline'] = baseline
                params = execution_data.discovery_config_args['params']
                discovery_config_kwargs['params'] = params
                if baseline:
                    for key in execution_data.discovery_config_args:
                        if key == 'strategy':
                            discovery_config_kwargs['strategy'] = execution_data.discovery_config_args['strategy']
                        elif key == 'params':
                            pass
                        else:
                            discovery_config_kwargs['params'].update({key: execution_data.discovery_config_args[key]})
                    for key in execution_data.unknown_args:
                        discovery_config_kwargs['params'].update({key: execution_data.unknown_args[key]})
                    for key in execution_data.run_args:
                        discovery_config_kwargs['params'].update({key: execution_data.run_args[key]})

                    if strategy == 'arda':
                        discovery_config_kwargs['params']['query_table'] = X
                        discovery_config_kwargs['params']['features'] = features
                    elif strategy == 'kitana':
                        discovery_config_kwargs['params']['query_table_path'] = base_table_path
                        discovery_config_kwargs['params']['features'] = features
                    elif strategy == 'qcr':
                        discovery_config_kwargs['params']['query_table'] = X
                    elif strategy == 'autofeat':
                        discovery_config_kwargs['params']['base_table_sep'] = ','
                        discovery_config_kwargs['params']['problem_type'] = splits[0]['target_type']
                        discovery_config_kwargs['params']['features'] = features
                else:
                    execution_data.discovery_config_args.pop('data_lake_path')
                    execution_data.discovery_config_args.pop('lake_table_sep')
                    discovery_config_kwargs = execution_data.discovery_config_args
                config = DiscoveryConfig(**discovery_config_kwargs)

                find_best_joins_kwargs = {'user_table_processed': X, 'top_k': top_k, 'n_jobs': 16, 'config': config}
                with open('experiments/downstream/config.yml', 'r') as f:
                    config = yaml.safe_load(f)
                base_tables = config['lakes'][lake]['base_tables']
                for base_table in base_tables:
                    for key in base_table:
                        if key == base_table_name:
                            task = base_table[key][0]['task']
                
                if strategy != 'LassoFeatureSelector':
                    if task == 'regression':
                        find_best_joins_kwargs['corr_threshold'] = 0.1
                    elif task == 'classification':
                        find_best_joins_kwargs['corr_threshold'] = 0.3
                else:
                    find_best_joins_kwargs['corr_threshold'] = None
                find_best_joins_kwargs.update(execution_data.run_args)

                errors_log_file = os.path.join(script_dir, 'errors.log')
                try:
                    df_aug, augplan = worker.find_best_joins(**find_best_joins_kwargs, debug=True)
                    # validate_kwargs(worker.find_best_joins, find_best_joins_kwargs)
                    # print(execution_data.discovery_config_args)
                    # print(find_best_joins_kwargs)
                    df_aug.write_csv(
                        os.path.join(
                            log_dir,
                            f'augmented_{combination_hash}.csv'
                        )
                    )
                    with open(os.path.join(log_dir, f'augmentation_plan_{combination_hash}.pkl'), 'wb') as f:
                        pickle.dump(augplan, f)
                except Exception as e:
                    df_aug = X
                    df_aug.write_csv(
                        os.path.join(
                            log_dir,
                            f'augmented_{combination_hash}.csv'
                        )
                    )
                    augplan = []
                    with open(os.path.join(log_dir, f'augmentation_plan_{combination_hash}.pkl'), 'wb') as f:
                        pickle.dump(augplan, f)
                    with open(errors_log_file, 'a') as f:
                        f.write(f'Error for combination {combination_hash}: {str(e)}\n')
    
    experiment_dir = os.listdir('experiments/downstream/ablation/top_k/logs/')
    trainers = {
        'simple': SimpleTrainer,
        # 'autogluon': AutoGluonTrainer
    }
    for trainer_name in trainers:
        trainer_resuts = pd.read_csv(f'experiments/base_tables/base_{trainer_name}_scores.csv', usecols=['Unnamed: 0', 'r2','rmse','accuracy','f1_weighted'])
        trainer_resuts = trainer_resuts.rename({'Unnamed: 0': 'table'}, axis=1)
        trainer_class = trainers[trainer_name]
        all_results = pd.DataFrame()
        for dir in experiment_dir:
            if not os.path.isdir(os.path.join('experiments/downstream/ablation/top_k/logs/', dir)):
                continue
            log_path = os.path.join('experiments/downstream/ablation/top_k/logs/', dir)
            print(dir)
            lake, table_name, strategy = dir.split('_')
            try:
                X = pd.read_csv(os.path.join(log_path, f'augmented_{dir}.csv'))
                with open(os.path.join(log_path, f'augmentation_plan_{dir}.pkl'), 'rb') as f:
                    augplan = pickle.load(f)
                if len(augplan) == 0:
                    experiment_col = pd.DataFrame([dir], columns=[''])
                    df = trainer_resuts[trainer_resuts['table'] == table_name].reset_index(drop=True)
                    df = pd.concat([experiment_col, df], axis=1)
                    all_results = pd.concat([all_results, df], ignore_index=False)
                    continue
            except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                experiment_col = pd.DataFrame([dir], columns=[''])
                df = trainer_resuts[trainer_resuts['table'] == table_name].reset_index(drop=True)
                df = pd.concat([experiment_col, df], axis=1)
                all_results = pd.concat([all_results, df], ignore_index=False)
            base_table_splits_path = os.path.join('experiments/base_tables/', table_name, 'splits.json')
            with open(base_table_splits_path, 'r') as f:
                splits = json.load(f)
            problem_type = splits[0]['target_type']
            if problem_type == 'continuous':
                problem_type = 'regression'
            target = splits[0]['target']
            query_col = splits[0]['query_col']
            numeric_dtypes = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
            X = pl.from_pandas(X)
            preprocessed_flag = all([X[col].dtype in numeric_dtypes for col in X.columns if col not in [query_col, target]])
            numeric_cols = []
            categorical_cols = []
            if not preprocessed_flag:
                try:
                    X.select(pl.col(target))
                except Exception as e:
                    target = target+'_x'
                if augplan is not None:
                    for col in augplan:
                        if col in X.columns:
                            dtype = X[col].dtype
                            if dtype in numeric_dtypes:
                                if col not in numeric_cols and col != target:
                                    numeric_cols.append(col)
                            else:
                                if col not in categorical_cols and col != query_col:
                                    categorical_cols.append(col)
                X = X.to_pandas()
                X = X.replace([np.inf, -np.inf], np.nan)
                simple_imputer = SimpleImputer(strategy='most_frequent').set_output(transform="pandas")
                if len(categorical_cols) > 0:
                    X[categorical_cols] = simple_imputer.fit_transform(X[categorical_cols])
                imputer = IterativeImputer(
                    estimator=HistGradientBoostingRegressor(
                        max_iter=100,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=42
                    ),
                    sample_posterior=False,
                    random_state=42,
                    n_nearest_features=10,
                    max_iter=5,
                    tol=1e-2
                ).set_output(transform="pandas")
                if len(numeric_cols) > 0:
                    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

                feat_gen = AutoMLPipelineFeatureGenerator(
                    enable_text_ngram_features=False,
                    enable_text_special_features=False
                )
                X = feat_gen.fit_transform(X)
            else:
                X = X.to_pandas()
                imputer = IterativeImputer(
                    estimator=HistGradientBoostingRegressor(
                        max_iter=100,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=42
                    ),
                    sample_posterior=False,
                    random_state=42,
                    n_nearest_features=10,
                    max_iter=5,
                    tol=1e-2
                ).set_output(transform="pandas")
                X = X.drop(columns=query_col, errors='ignore')
                X = X.replace([np.inf, -np.inf], np.nan)
                try:
                    X = imputer.fit_transform(X)
                except Exception as e:
                    print(f"Error during imputation for {dir}: {e}")
                    X.to_csv('tmp.csv')
                    raise e
            X = X.drop(columns=query_col, errors='ignore')
            X = X.drop(columns='Unnamed: 0', errors='ignore')
            X = X.dropna()
            model_dir = f'{log_path}/{trainer_name}_model'
            trainer = trainer_class(
                problem_type=problem_type,
                target_column=target,
                time_limit=1800,
                presets="good_quality",
                output_dir=model_dir
            )
            
            if os.path.exists(model_dir) and os.listdir(model_dir):
                print(f"Model directory {model_dir} already exists and is not empty, skipping training...")
                trainer.load_model(model_dir)
                performance = trainer.evaluate(X)
                df = pd.DataFrame([performance], index=[dir])
                all_results = pd.concat([all_results, df], ignore_index=False)
                continue
            else:
                try:
                    results = trainer.train(X)
                    performance = trainer.evaluate(X)
                    df = pd.DataFrame([performance], index=[dir])
                    all_results = pd.concat([all_results, df], ignore_index=False)
                except Exception as e:
                    print(f"Error during training or evaluation for {dir}: {e}")
                    experiment_col = pd.DataFrame([dir], columns=[''])
                    df = trainer_resuts[trainer_resuts['table'] == table_name].reset_index(drop=True)
                    df = pd.concat([experiment_col, df], axis=1)
                    all_results = pd.concat([all_results, df], ignore_index=False)

        all_results.to_csv(f'experiments/downstream/ablation/top_k/logs/{trainer_name}_results.csv')