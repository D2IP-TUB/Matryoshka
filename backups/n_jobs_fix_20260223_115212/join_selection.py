from collections import namedtuple
import warnings
from polars.exceptions import MapWithoutReturnDtypeWarning
warnings.filterwarnings("ignore", category=MapWithoutReturnDtypeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import hashlib
import inspect
import os
import augmentation.utils.f32_numpy
import numpy as np
import polars as pl
import ray
import time
from .feature_selection.algorithms.arda import *
from .feature_selection.algorithms.autofeat import *
from .feature_selection.algorithms.greedy import *
from .feature_selection.algorithms.kitana import *
from .feature_selection.algorithms.lasso import *
from .feature_selection.algorithms.stepwise import *
from .feature_selection.algorithms.qcr import *
from .feature_selection.base.user_table import BaseTable
from .feature_selection.collinearity import CollinearityAnalysis
from .feature_selection.ranking import OverlapRanking
from .planner import DiscoveryPlanner, StepStatus
from .retrieval import JoinDiscovery
from .utils.config import DiscoveryConfig
from .utils.common import process_key
from .utils.exceptions import EmptyAugmentation, UserTableNotProcessed
from .utils.logging.logger_config import setup_logger


class JoinSelection(JoinDiscovery):
    def __init__(self, feature_selection_table_name: str, overlap_table_name: str, verbose: bool = False, log_file_name: str = 'log', log_dir: str = '.'):
        super().__init__(feature_selection_table_name, overlap_table_name, verbose=False)

        self.feature_selection_table_name = feature_selection_table_name
        self.overlap_table_name = overlap_table_name
        self.verbose = verbose
        if not log_dir:
            script_path = os.path.abspath(__file__)
            script_dir = os.path.dirname(script_path)
            log_dir = os.path.join(script_dir, 'utils', 'logging', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if verbose:
            silent = False
        else:
            silent = True    
        self.logger = setup_logger(name='join_selection', log_dir=log_dir, log_file=log_file_name, json_logging=True, silent=silent)

        self.join_selection_query = 'SELECT * ' \
                                   f'FROM {feature_selection_table_name} ' \
                                    'WHERE (table_index, key_col_index, row_index) IN ' \
                                    '(' \
                                    '   SELECT table_index, key_col_index, row_index ' \
                                    f'  FROM temp_valid_combinations ' \
                                    ');'

        self.temp_table_query = 'CREATE TEMPORARY TABLE temp_valid_combinations ( ' \
                                'table_index INTEGER, ' \
                                'key_col_index INTEGER, ' \
                                'row_index INTEGER, ' \
                                ');'

        self.augmentation_query = 'SELECT key, table_index, key_col_index, feature_index ' \
                                  f'FROM {feature_selection_table_name} ' \
                                  'WHERE (table_index, key_col_index, row_index) IN ' \
                                  '(' \
                                  '   SELECT table_index, key_col_index, row_index ' \
                                  f'  FROM temp_valid_combinations ' \
                                  ');'


    def find_best_joins(
        self,
        user_table_processed: pl.DataFrame,
        query_column_name: str,
        target_column_name: str,
        top_k: int,
        config: DiscoveryConfig,
        corr_threshold: float = 0.55,
        n_jobs: int = 1,
        debug: bool = False
    ) -> pl.DataFrame:
        self.corr_threshold = corr_threshold
        query_col_cardinality = user_table_processed.select(pl.col(query_column_name).n_unique()).row(0)[0]
        run_config = {
            'corr_threshold': corr_threshold,
            'top_k': top_k,
            'query_col_cardinality': query_col_cardinality
        }
        self.logger.info('Join Selection started', extra={'config': hashlib.md5(str(run_config).encode('utf-8')).hexdigest()})
        try:
            if config.baseline:
                start_error = time.perf_counter()
                function_registry = {'run_baseline': self.run_baseline}
                planner = DiscoveryPlanner(function_registry)
                physical_plan = planner.create_plan(config)
                known_vars = locals()
                for step, func in zip(physical_plan, function_registry.values()):
                    sig = inspect.signature(func)
                    kwargs = {}
                    for name in sig.parameters:
                        if name == "self":
                            continue
                        if name in known_vars:
                            kwargs[name] = known_vars[name]
                    step.params.update(kwargs)
                start = time.perf_counter()
                context = planner.execute_plan(physical_plan, config)
                augmented_table, t1, t2, augplan = context.get_result('baseline_execution')
                end = time.perf_counter()
                self.logger.info('Augmentation', extra={'runtime': t1+t2, 'augmentation_plan': augplan})
            else:
                user_table_processed, query_column = self._preprocess_user_dataset(user_table_processed, query_column_name, target_column_name)
                user_table_agg = self._sketch_user_table(user_table_processed, query_column_name, target_column_name, config)
                planner = DiscoveryPlanner()
                logical_plan = planner.create_plan(config)
                
                function_registry = {f'run_{step.name}': getattr(self, f'run_{step.name}') for step in logical_plan}
                planner = DiscoveryPlanner(function_registry)
                physical_plan = planner.create_plan(config)
                globals()['physical_plan'] = physical_plan

                base_table = BaseTable(
                    feature_selection_table_name=self.feature_selection_table_name,
                    overlap_table_name=self.overlap_table_name,
                    table=user_table_processed,
                    table_agg=user_table_agg,
                    rows_map=None,
                    conninfo=self.conninfo,
                    query_column_name=query_column_name,
                    target_column_name=target_column_name,
                    baseline=config.baseline
                )

                known_vars = locals()
                for step, func in zip(physical_plan, function_registry.values()):
                    sig = inspect.signature(func)
                    kwargs = {}
                    for name in sig.parameters:
                        if name == "self":
                            continue
                        if name in known_vars:
                            kwargs[name] = known_vars[name]
                    step.params.update(kwargs)

                context = planner.execute_plan(physical_plan, config)
                augmented_table, augplan = context.get_result('augmentation')
        except Exception as e:
            augplan = []
            if config.baseline:
                end_error = time.perf_counter()
                runtime = end_error - start_error
                self.logger.info('Augmentation', extra={'runtime': runtime, 'augmentation_plan': augplan})
            try:
                ray.shutdown()
            except Exception as shutdown_exception:
                self.logger.error(f"Error occurred during Ray shutdown: {shutdown_exception}", exc_info=True)
            self.logger.error(f"Error occurred during join selection: {e}", exc_info=True)
            augmented_table = user_table_processed
            if debug:
                raise e
        finally:
            self.logger.handlers.clear()

        return augmented_table, augplan


    def run_find_joinable_tables(self, context, query_column, top_k, user_table_processed, corr_threshold, **kwargs):
        start = time.perf_counter()
        token_query_results, join_selection_query_results, overlap_ratio = self.find_joinable_tables(query_column, top_k=top_k, user_table_processed=user_table_processed, join_selection=True)
        end = time.perf_counter()
        runtime_sec = end - start
        total_features = join_selection_query_results.group_by('table_column_index').agg(pl.col('sum').list.len().max()).sum().row(0)[1]
        self.logger.info('Retrieval', extra={'runtime': runtime_sec, 'joinability': float(overlap_ratio.mean()), 'total_features': total_features})
        query_column_name = query_column.columns[0]
        target_column_name = user_table_processed.columns[0]
        start = time.perf_counter()
        if self.corr_threshold is not None:
            join_selection_query_results, status = self._prune_features(
                join_selection_query_results, query_column_name, target_column_name, user_table_processed, context.config.task, self.corr_threshold
            )
            if status == False:
                self.corr_threshold = None
                fitted_features = total_features
            else:
                fitted_features = join_selection_query_results.group_by('table_column_index').agg(pl.col('drop_feature').list.eval((pl.element() != 0).sum()).list.sum().first()).sum().row(0)[1]
        else:
            fitted_features = total_features
        end = time.perf_counter()
        runtime_sec = end - start
        self.logger.info('Pruning', extra={'runtime': runtime_sec, 'fitted_features': fitted_features})
        result = namedtuple('Result', ['token_query_results', 'join_selection_query_results', 'overlap_ratio'])
        return result(token_query_results, join_selection_query_results, overlap_ratio)


    def run_loop_entry(
        self,
        context,
        logical_plan,
        user_table_agg: pl.DataFrame,
        base_table: BaseTable,
        query_column_name: str,
        target_column_name: str,
        n_jobs: int,
        **kwargs
    ):
        result = context.get_result('find_joinable_tables')
        token_query_results, join_selection_query_results, overlap_ratio = result.token_query_results, result.join_selection_query_results, result.overlap_ratio
        table_batch_size = kwargs['table_batch_size']

        total_tables = len(join_selection_query_results.group_by(['table_index', 'key_col_index']).len()) if join_selection_query_results is not None else None
        if table_batch_size > 0:
            total_batches = (total_tables + table_batch_size - 1) // table_batch_size
        else:
            total_batches = 1
        n_init_features = len(user_table_agg.row(0)[2]) if context.config.task == 'regression' else len(user_table_agg.row(0)[3])
        augmentation_plan = set()
        start = time.perf_counter()
        for batch_iter in range(1, total_batches + 1):
            join_selection_query_batch = pl.DataFrame()
            batches_processed = 0
            n_iter = 0
            for group in join_selection_query_results.group_by(['table_index', 'key_col_index']):
                join_selection_query_batch = pl.concat(
                    [join_selection_query_batch, group[1]],
                    how='vertical'
                )
                join_selection_query_results = join_selection_query_results.filter(
                    ~(
                        (pl.col('table_index') == group[0][0]) &
                        (pl.col('key_col_index') == group[0][1])
                    )
                )
                batches_processed += 1

                context.results['find_joinable_tables'] = namedtuple('Result', ['token_query_results', 'join_selection_query_results', 'overlap_ratio'])(
                    None, join_selection_query_results, overlap_ratio
                )
                try:
                    joint_tuples = self.run_ranking(
                        context,
                        user_table_agg=user_table_agg,
                        query_column_name=query_column_name,
                        target_column_name=target_column_name,
                        n_jobs=n_jobs,
                        **kwargs
                    )
                    kwargs.update({'joint_tuples': joint_tuples})
                    joint_tuples = self.run_collinearity_analysis(context, **kwargs)
                except (EmptyAugmentation, RuntimeError) as e:
                    self.logger.info(f'No augmentation possible in this batch. Skipping to the next batch. Error: {str(e)}')
                    continue
                kwargs.update({'joint_tuples': joint_tuples})
                iter_result = self.run_strategy_with_model(context, logical_plan, base_table, **kwargs)

                iter_score, iter_augmentation_plan, aug_sums = iter_result.score, iter_result.augmentation_plan, iter_result.aug_sums
                print(f'Batch {batch_iter}/{total_batches}, Iteration Score: {iter_score}')
                if n_iter == 0:
                    base_score = iter_score
                    iter_score = None

                stopping_condition = self.run_loop_exit(context, iter_score, base_score, kwargs['tol'])
                if stopping_condition:
                    break
                else:
                    n_iter += 1
                    user_table_agg = self._update_user_table_sketch(user_table_agg, query_column_name, target_column_name, n_init_features, aug_sums, context.config.task)
                    augmentation_plan.update(iter_augmentation_plan)

        augmentation_plan = list(augmentation_plan)
        aug_feature_indices = {k: v for k, v in zip(range(len(augmentation_plan)), augmentation_plan)}
        Result = namedtuple('JointTuple', 'aug_feature_indices')
        self.run_loop_exit(context, None, None, kwargs['tol'], **{'output': Result(aug_feature_indices)})
        end = time.perf_counter()
        runtime_sec = end - start
        self.logger.info('Selection', extra={'runtime': runtime_sec})

        return Result(aug_feature_indices)


    def run_loop_exit(self, context, current_score: float, previous_score: float | None, tol: float, **kwargs):
        if current_score is None and previous_score is not None:
            # first iteration, do nothing
            return False
        elif previous_score is None and current_score is None:
            # skip subsequent steps before augmentation
            physical_plan = globals()['physical_plan']
            steps = [i for i in range(len(physical_plan) - 1)] # -1 to exclude the last augmentation step
            for i in steps:
                globals()['physical_plan'][i].status = StepStatus.COMPLETED
            context.results['loop_exit'] = kwargs.get('output')
        elif current_score is not None and previous_score is not None:
            improvement = (current_score - previous_score) / abs(previous_score)
            stopping_condition = improvement < tol

            return stopping_condition


    def run_ranking(
        self,
        context,
        user_table_agg: pl.DataFrame,
        query_column_name: str,
        target_column_name: str,
        n_jobs: int,
        corr_threshold: float = None,
        var_threshold: float = None,
        **kwargs
    ):
        result = context.get_result('find_joinable_tables')
        _, join_selection_query_results, overlap_ratio = result.token_query_results, result.join_selection_query_results, result.overlap_ratio
        ranker = OverlapRanking(self.feature_selection_table_name, self.conninfo, self.corr_threshold, var_threshold)
        params = {
            'task': context.config.task,
            'join_selection_query_results': join_selection_query_results,
            'user_table_agg': user_table_agg,
            'query_column_name': query_column_name,
            'target_column_name': target_column_name,
            'overlap_ratio': overlap_ratio,
            'n_jobs': n_jobs
        }
        params.update(kwargs)
        start = time.perf_counter()
        joint_tuples = ranker.rank(**params)
        end = time.perf_counter()
        runtime_sec = end - start
        self.logger.info('Ranking', extra={'runtime': runtime_sec})
        
        return joint_tuples


    def run_collinearity_analysis(self, context, **kwargs):
        analyzer = CollinearityAnalysis()
        params = {
            'joint_tuples': context.get_result('ranking')
        }
        params.update(kwargs)
        params['model'] = globals()[params.get('model')]
        start = time.perf_counter()
        joint_tuples = analyzer.eliminate(**params)
        end = time.perf_counter()
        runtime_sec = end - start
        self.logger.info('Collinearity', extra={'runtime': runtime_sec})

        return joint_tuples


    def run_imputation(self, context):
        pass


    def run_strategy_with_model(self, context, logical_plan, base_table, **kwargs):
        res = context.get_result('find_joinable_tables')
        token_query_results = res.token_query_results
        join_selection_query_results = res.join_selection_query_results
        base_table.rows_map = None
        steps = [s.name for s in logical_plan]
        prev_step_index = steps.index('strategy_with_model') - 1
        strategy_cls = globals()[kwargs.get('strategy')]
        strategy = strategy_cls.remote(kwargs.get('metric'))
        model_cls = globals()[kwargs.get('model')]

        params = {}
        params.update(kwargs)
        params.update({
            'model': model_cls,
            'task': context.config.task,
            'joint_tuples': kwargs.get('joint_tuples', context.get_result(steps[prev_step_index])),
            'tol': kwargs.get('tol'),
            'token_query_results': token_query_results,
            'join_selection_query_results': join_selection_query_results,
            'base_table': base_table
        })
        start = time.perf_counter()
        joint_tuples = ray.get(strategy.run.remote(**params))
        end = time.perf_counter()
        runtime_sec = end - start
        self.logger.info('Selection', extra={'runtime': runtime_sec})

        return joint_tuples
    

    def run_baseline(self, context, **kwargs):
        params = context.config.params
        strategy = context.config.strategy
        baseline_cls = globals()[strategy]
        baseline = baseline_cls()
        result = baseline.run(**params)
        return result


    def run_augmentation(self, context, logical_plan, query_column_name: str, user_table_processed: pl.DataFrame, **kwargs):
        second_last_step = logical_plan[-2].name
        join_selection_query_results = context.get_result('find_joinable_tables').join_selection_query_results
        top_features = list(context.get_result(second_last_step).aug_feature_indices.values())
        start = time.perf_counter()
        if len(top_features) > 0:
            augmented_table = self._create_augmentation_table(user_table_processed, join_selection_query_results, top_features, query_column_name)
        else:
            augmented_table = user_table_processed
        end = time.perf_counter()
        runtime_sec = end - start
        self.logger.info('Augmentation', extra={'runtime': runtime_sec, 'augmentation_plan': top_features})

        return augmented_table, top_features
    

    def _sketch_user_table(self, user_table_processed: pl.DataFrame, query_column_name: str, target_column_name: str, config) -> pl.DataFrame:
        '''
        Compute the count, sum and dot product sketches for the user table.

        Parameters:
        ----------
        user_table_processed: pl.DataFrame
            User table to find joinable tables and perform feature selection for. \n
            All rows must be unique and all columns except for the foreign key column must be numeric.

        query_column_name: str
            Name of the query column in the user table
            
        Returns:
        --------
        pl.DataFrame: User table with count, sum and dot product aggregations, i.e., cofactor matrix sketches
        '''
        task = config.task
        match task:
            case 'regression':
                numeric_cols = user_table_processed.select(pl.all().exclude(pl.String)).columns
                user_table_agg = user_table_processed.group_by(query_column_name).agg(
                    [
                        pl.len().alias('count'),
                        pl.concat_list(pl.col(numeric_cols).sum()).alias('sum'),
                        # pl.concat_list(numeric_cols).alias('dot_product_upper_triangular_values')
                    ]
                )
                # user_table_agg = user_table_agg.group_by(query_column_name).agg(
                #     pl.col('count'),
                #     pl.col('sum'),
                #     pl.struct(
                #         [
                #             pl.col('dot_product_upper_triangular_values').alias('first'),
                #             pl.col('dot_product_upper_triangular_values').alias('second')
                #         ]
                #     )
                #         .map_batches(
                #             lambda x: pl.Series(
                #                 (
                #                     np.vstack(x.struct.field('first').list.explode().to_numpy()).T\
                #                         .dot(np.vstack(x.struct.field('second').list.explode().to_numpy()))
                #                 )
                #             ),
                #             return_dtype=pl.Array(pl.Float64, shape=user_table_processed.shape[1]-1)
                #         )
                #         .alias('dot_product_upper_triangular_values')
                # )
                user_table_agg = user_table_agg.sort(query_column_name, maintain_order=True)
            case 'classification':
                numeric_cols = user_table_processed.select(pl.all().exclude(pl.String).exclude(target_column_name)).columns
                user_table_agg = user_table_processed.group_by([query_column_name, target_column_name]).agg(
                    [
                        pl.len().alias('count'),
                        pl.concat_list(pl.col(numeric_cols).sum()).alias('sum'),
                        # pl.concat_list(numeric_cols).alias('dot_product_upper_triangular_values')
                    ]
                )
                # user_table_agg = user_table_agg.group_by([query_column_name, target_column_name]).agg(
                #     pl.col('count'),
                #     pl.col('sum'),
                #     pl.struct(
                #         [
                #             pl.col('dot_product_upper_triangular_values').alias('first'),
                #             pl.col('dot_product_upper_triangular_values').alias('second')
                #         ]
                #     )
                #         .map_batches(
                #             lambda x: pl.Series(
                #                 (
                #                     np.vstack(x.struct.field('first').list.explode().to_numpy()).T\
                #                         .dot(np.vstack(x.struct.field('second').list.explode().to_numpy()))
                #                 )
                #             ),
                #             return_dtype=pl.Array(pl.Float64, shape=user_table_processed.shape[1]-2)
                #         )
                #         .alias('dot_product_upper_triangular_values')
                # )
                user_table_agg = user_table_agg.sort(query_column_name, maintain_order=True)

        return user_table_agg


    def _update_user_table_sketch(self, user_table_agg: pl.DataFrame, query_column_name: str, target_column_name: str, n_init_features: int, aug_sums: np.ndarray[float], task: str):
        n_features = aug_sums.shape[1] + n_init_features

        def dot_product_agg(group, count, n_init_features):
            dot = group.T.dot(group)
            normalization_slice = dot[np.ix_(range(n_init_features), range(n_init_features))]
            rows, cols = np.triu_indices(normalization_slice.shape[0])
            dot[rows, cols] /= count
            n_new_features = dot.shape[0] - n_init_features
            new_feat_rows, new_feat_cols = np.ix_(range(n_init_features, n_init_features + n_new_features), range(n_init_features, n_init_features + n_new_features))
            dot[new_feat_rows, new_feat_cols] *= count
            index = np.triu_indices_from(dot)
            values = dot[index]
            return values.tolist()
        
        match task:
            case 'regression':
                user_table_agg = user_table_agg.with_columns(new_features = aug_sums).with_columns(pl.col('new_features').cast(pl.List(pl.Float64)))
                user_table_agg = (
                    user_table_agg
                        .with_columns(
                            pl.concat_list('sum', 'new_features').alias('sum')
                            .cast(pl.List(pl.Float64))
                            .list.to_struct(upper_bound=n_features)
                            .struct.unnest()
                        )
                        .drop('new_features')
                )
                user_table_agg = (
                    user_table_agg
                        .group_by(query_column_name, maintain_order=True)
                        .agg(
                            [
                                pl.col('count').sum().alias('count'),
                                pl.concat_list(r'^field_\d+$').flatten().alias('sum'),
                                pl.concat_list(r'^field_\d+$').alias('concat')
                            ]
                        )
                        .with_columns(
                            pl.col('sum').list.head(n_init_features).alias('sum'),
                            (pl.col('sum').list.gather(range(n_init_features, n_features)) * pl.col('count')).alias('updated_sum'),
                            pl.struct([pl.col('count'), pl.col('concat')])
                            .map_elements(lambda x: dot_product_agg(np.vstack(x['concat']), x['count'], n_init_features), return_dtype=pl.List(pl.Float64))
                            .alias('dot_product_upper_triangular_values')
                        )
                        .with_columns(
                            pl.concat_list('sum', 'updated_sum').alias('sum')
                        )
                        .drop('concat', 'updated_sum')
                )
            case 'classification':
                user_table_agg = user_table_agg.sort([target_column_name, query_column_name])
                user_table_agg = user_table_agg.with_columns(new_features = aug_sums).with_columns(pl.col('new_features').cast(pl.List(pl.Float64)))
                user_table_agg = (
                    user_table_agg
                        .with_columns(
                            pl.concat_list('sum', 'new_features').alias('sum')
                            .cast(pl.List(pl.Float64))
                            .list.to_struct(upper_bound=n_features)
                            .struct.unnest()
                        )
                        .drop('new_features')
                )
                user_table_agg = (
                    user_table_agg
                        .group_by([query_column_name, target_column_name], maintain_order=True)
                        .agg(
                            [
                                pl.col('count').sum().alias('count'),
                                pl.concat_list(r'^field_\d+$').flatten().alias('sum'),
                                pl.concat_list(r'^field_\d+$').alias('concat')
                            ]
                        )
                        .with_columns(
                            pl.col('sum').list.head(n_init_features).alias('sum'),
                            (pl.col('sum').list.gather(range(n_init_features, n_features)) * pl.col('count')).alias('updated_sum'),
                            pl.struct([pl.col('count'), pl.col('concat')])
                            .map_elements(lambda x: dot_product_agg(np.vstack(x['concat']), x['count'], n_init_features))
                            .alias('dot_product_upper_triangular_values')
                        )
                        .with_columns(
                            pl.concat_list('sum', 'updated_sum').alias('sum')
                        )
                        .drop('concat', 'updated_sum')
                )

        return user_table_agg


    def _create_augmentation_table(
        self,
        user_table_processed: pl.DataFrame,
        join_selection_query_results: pl.DataFrame,
        top_features: list[str],
        query_column_name: str
    ) -> pl.DataFrame:
        ray.shutdown() # shut down ray cluster which is always initialized at ranking step

        table_indices = []
        key_col_indices = []
        table_features = {}
        for s in top_features:
            feature_split = s.split('_')
            table_id = int(feature_split[0])
            table_indices.append(table_id)
            key_col_id = int(feature_split[1])
            key_col_indices.append(key_col_id)
            if table_id not in table_features:
                table_features[table_id] = {}
            if key_col_id not in table_features.get(table_id, {}):
                table_features[table_id].update({key_col_id: []})
            table_features[table_id][key_col_id].append(s)

        join_selection_query_results = join_selection_query_results.filter(
            (
                (pl.col('table_index').is_in(table_indices))
                    &
                (pl.col('key_col_index').is_in(key_col_indices))
            )
        )
        aug_df = join_selection_query_results.select(pl.col('key').unique())
        for idx, group in join_selection_query_results.group_by(['table_index', 'key_col_index'], maintain_order=True):
            full_features = table_features[idx[0]].get(idx[1], [])
            if len(full_features) == 0:
                continue
            features = [int(f.split('_')[-1]) for f in full_features]
            group = group.with_columns(
                pl.col('sum')
                    .list.gather(features)
                    .list.to_struct(fields=[f'{idx[0]}_{idx[1]}_{f}' for f in features])
                    .struct.unnest()
            )
            aug_df = aug_df.join(group.select(pl.col(['key'] + full_features)), on='key', how='left')

        aug_df = aug_df.rename({'key': query_column_name})
        user_table_processed = user_table_processed.join(aug_df, on=query_column_name, how='left')

        return user_table_processed
    

    def _preprocess_user_dataset(self, user_table_processed, query_column_name, target_column_name):
        # check that user_table_processed has only numeric columns
        # raises error and breaks execution otherwise
        user_table_processed = self._user_table_checkup(user_table_processed, query_column_name)
        # target feature should be the first column in the user table for future consistency in sketch operations
        target_feature = user_table_processed.get_column(target_column_name)
        user_table_processed = user_table_processed.drop(target_column_name)
        user_table_processed.insert_column(0, target_feature)
        query_column = user_table_processed.select(query_column_name)

        return user_table_processed, query_column


    def _user_table_checkup(self, user_table: pl.DataFrame, query_column_name: str) -> pl.DataFrame:
        '''
        Check that the user table has only numeric columns.

        Parameters:
        ----------
        user_table: pl.DataFrame
            User table to find joinable tables and perform feature selection for.

        query_column_name: str
            Name of the query column in the user table
        
        Returns:
        --------
        pl.DataFrame: Unchanged user table
        '''
        numeric_check = user_table.select(pl.all().exclude(query_column_name)).dtypes
        if pl.String in numeric_check:
            raise UserTableNotProcessed()
        
        return user_table