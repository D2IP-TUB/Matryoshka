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
from experiments.base_tables.base_scoring import AutoGluonTrainer, SimpleTrainer
from experiments.base_tables.base_table_preprocessing import PreProcessor
from experiments.downstream.experiment_executor import ExperimentExecutor


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
    config_df = config_df.filter(
        (pl.col('strategy')=='ForwardSelection')
    )
    print(config_df)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_tables_dir = 'experiments/base_tables/'
    n_jobs_choices = [1, 2, 4, 16, 32, 64]
    for n_jobs in n_jobs_choices:
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
                X, query_col, target, nan_mask = preprocessor.run()
                X.write_csv(
                    os.path.join(
                        base_tables_dir,
                        base_table_name,
                        f'{base_table_name}_preprocessed.csv'
                    )
                )

                execution_data = ExperimentExecutor.from_params(exp_config, strat)
                worker_init_kwargs = execution_data.init_args
                combination_hash = f'{lake}_{base_table_name}_{n_jobs}'
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

                find_best_joins_kwargs = {'user_table_processed': X, 'top_k': 50, 'n_jobs': n_jobs, 'config': config}
                with open('experiments/downstream/config.yml', 'r') as f:
                    config = yaml.safe_load(f)
                base_tables = config['lakes'][lake]['base_tables']
                for base_table in base_tables:
                    for key in base_table:
                        if key == base_table_name:
                            task = base_table[key][0]['task']
                
                if strategy != 'LassoFeatureSelector':
                    if task == 'regression':
                        find_best_joins_kwargs['corr_threshold'] = 0.01
                    elif task == 'classification':
                        find_best_joins_kwargs['corr_threshold'] = 0.01
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