import ast
import warnings
from collections import namedtuple

from polars.exceptions import MapWithoutReturnDtypeWarning

warnings.filterwarnings("ignore", category=MapWithoutReturnDtypeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import hashlib
import inspect
import os
import time

import numpy as np
import polars as pl
import ray

from .baselines import resolve_baseline
from .config import DiscoveryConfig
from .exceptions import UserTableNotProcessed
from .planner import DiscoveryPlanner
from .retrieval import JoinDiscovery
from .selection.base.user_table import BaseTable
from .selection.collinearity import CollinearityAnalysis
from .selection.ranking import OverlapRanking
from .selection.registry import resolve_model, resolve_strategy
from .selection.sketch_processing import quantile_bin_thresholds
from .utils.logging import default_log_dir, setup_logger


class JoinSelection(JoinDiscovery):
    def __init__(self, feature_selection_table_name: str, overlap_table_name: str, verbose: bool = False,
                 log_file_name: str = 'log', log_dir: str = None, settings=None, exclude_tables=None,
                 features_stop_list=None):
        log_dir = str(log_dir) if log_dir else str(default_log_dir())
        super().__init__(feature_selection_table_name, overlap_table_name, verbose=False,
                         settings=settings, exclude_tables=exclude_tables, log_dir=log_dir,
                         features_stop_list=features_stop_list)

        self.feature_selection_table_name = feature_selection_table_name
        self.overlap_table_name = overlap_table_name
        self.verbose = verbose
        os.makedirs(log_dir, exist_ok=True)
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
        self._last_friendly_plan: list[str] = []
        # Whether pruning built a `drop_feature` mask, from correlation or from the features stop-list
        self.features_masked = False


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
            if not config.baseline:
                ray.init(
                    include_dashboard=False,
                    logging_level="ERROR",
                    ignore_reinit_error=True,
                    num_cpus=n_jobs,
                    _metrics_export_port=None,
                    _system_config={
                        "task_events_report_interval_ms": 0
                    }
                )
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
            self.logger.error(f"Error occurred during join selection: {e}", exc_info=True)
            augmented_table = user_table_processed
            if debug:
                raise e
        finally:
            try:
                ray.shutdown()
            except Exception as shutdown_exception:
                self.logger.error(f"Error occurred during Ray shutdown: {shutdown_exception}", exc_info=True)
            self.logger.handlers.clear()

        return augmented_table, augplan


    def run_find_joinable_tables(self, context, query_column, top_k, user_table_processed, corr_threshold, **kwargs):
        start = time.perf_counter()
        # Allow non-baseline strategies (e.g. forward) to traverse multi-hop
        # join graphs by reading ``n_hops``/``joinability_threshold`` from the
        # config params block. Defaults preserve single-hop behaviour.
        cfg_params = (context.config.params or {}) if context and context.config else {}
        n_hops = int(cfg_params.get('n_hops', kwargs.get('n_hops', 1)))
        joinability_threshold = float(cfg_params.get('joinability_threshold', kwargs.get('joinability_threshold', 0.5)))
        token_query_results, join_selection_query_results, overlap_ratio = self.find_joinable_tables(
            query_column,
            top_k=top_k,
            user_table_processed=user_table_processed,
            join_selection=True,
            n_hops=n_hops,
            joinability_threshold=joinability_threshold,
        )
        end = time.perf_counter()
        runtime_sec = end - start
        total_features = join_selection_query_results.group_by('table_column_index').agg(pl.col('sum').list.len().max()).sum().row(0)[1]
        self.logger.info('Retrieval', extra={'runtime': runtime_sec, 'joinability': float(overlap_ratio.mean()), 'total_features': total_features})
        query_column_name = query_column.columns[0]
        target_column_name = user_table_processed.columns[0]
        start = time.perf_counter()
        self.features_masked = False
        if self.corr_threshold is not None or self.features_stop_list:
            join_selection_query_results, status = self._prune_features(
                join_selection_query_results, query_column_name, target_column_name, user_table_processed, context.config.task, self.corr_threshold
            )
            self.features_masked = status
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
        # The ranking uses the correlation threshold only to decide whether to apply the `drop_feature` mask, so a
        # mask built from the features stop-list alone (without correlation pruning) needs a threshold as well
        mask_threshold = self.corr_threshold
        if mask_threshold is None and self.features_masked:
            mask_threshold = 0.0
        ranker = OverlapRanking(self.feature_selection_table_name, self.conninfo, mask_threshold, var_threshold)
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
        params['model'] = resolve_model(params.get('model'))
        start = time.perf_counter()
        joint_tuples = analyzer.eliminate(**params)
        end = time.perf_counter()
        runtime_sec = end - start
        self.logger.info('Collinearity', extra={'runtime': runtime_sec})

        return joint_tuples


    def run_imputation(self, context):
        pass


    def run_strategy_with_model(self, context, logical_plan, base_table, n_jobs: int = 1, **kwargs):
        res = context.get_result('find_joinable_tables')
        token_query_results = res.token_query_results
        join_selection_query_results = res.join_selection_query_results
        base_table.rows_map = None
        steps = [s.name for s in logical_plan]
        prev_step_index = steps.index('strategy_with_model') - 1
        strategy_cls = resolve_strategy(kwargs.get('strategy'))
        strategy = strategy_cls.remote(kwargs.get('metric'))
        model_cls = resolve_model(kwargs.get('model'))

        params = {}
        params.update(kwargs)
        params.update({
            'model': model_cls,
            'task': context.config.task,
            'joint_tuples': kwargs.get('joint_tuples', context.get_result(steps[prev_step_index])),
            'tol': kwargs.get('tol'),
            'token_query_results': token_query_results,
            'join_selection_query_results': join_selection_query_results,
            'base_table': base_table,
            'n_jobs': n_jobs
        })
        start = time.perf_counter()
        joint_tuples = ray.get(strategy.run.remote(**params))
        end = time.perf_counter()
        runtime_sec = end - start
        timing = getattr(joint_tuples, 'timing_breakdown', {})
        self.logger.info('Selection', extra={
            'runtime': runtime_sec,
            'n_iters': timing.get('n_iters'),
            'computation_s': timing.get('computation_s'),
            'ray_overhead_s': timing.get('ray_overhead_s'),
        })

        return joint_tuples


    def run_baseline(self, context, **kwargs):
        params = context.config.params
        strategy = context.config.strategy
        baseline_cls = resolve_baseline(strategy)
        baseline = baseline_cls()
        result = baseline.run(**params)
        return result


    def run_augmentation(self, context, logical_plan, query_column_name: str, user_table_processed: pl.DataFrame, **kwargs):
        second_last_step = logical_plan[-2].name
        join_selection_query_results = context.get_result('find_joinable_tables').join_selection_query_results
        top_features = list(context.get_result(second_last_step).aug_feature_indices.values())
        self._last_friendly_plan = []
        # Dummification n_bins is required at materialization to reproduce the
        # quantile-binned indicator columns selected by forward. Pulled from
        # config.params if present, otherwise the default 4 bins.
        config_params = kwargs.get('config_params') or {}
        dummify_features = config_params.get('dummify_features') or {}
        dummify_n_bins = int(dummify_features.get('n_bins', 4)) if dummify_features.get('enabled') else None
        start = time.perf_counter()
        if len(top_features) > 0:
            augmented_table = self._create_augmentation_table(
                user_table_processed, join_selection_query_results, top_features,
                query_column_name, dummify_n_bins=dummify_n_bins,
            )
        else:
            augmented_table = user_table_processed
        end = time.perf_counter()
        runtime_sec = end - start
        # `top_features` holds internal `<table>_<key_col>_<feature>` triples;
        # `_last_friendly_plan` holds the `<lake table>.<column>_<aggregate>`
        # names that were actually appended. Return the readable form when it
        # is available, so the plan matches the columns of the augmented table.
        plan = self._last_friendly_plan if self._last_friendly_plan else top_features
        self.logger.info('Augmentation', extra={'runtime': runtime_sec, 'augmentation_plan': plan})

        return augmented_table, plan


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
                cofactor_upper_exprs = [
                    (pl.col(numeric_cols[i]) * pl.col(numeric_cols[j])).sum().cast(pl.Float64)
                    for i in range(len(numeric_cols))
                    for j in range(i, len(numeric_cols))
                ]
                user_table_agg = user_table_processed.group_by(query_column_name).agg(
                    [
                        pl.len().alias('count'),
                        pl.concat_list(pl.col(numeric_cols).sum()).alias('sum'),
                        pl.concat_list(cofactor_upper_exprs).alias('cofactor_upper'),
                    ]
                )
                user_table_agg = user_table_agg.sort(query_column_name, maintain_order=True)
            case 'classification':
                numeric_cols = user_table_processed.select(pl.all().exclude(pl.String).exclude(target_column_name)).columns
                cofactor_upper_exprs = [
                    (pl.col(numeric_cols[i]) * pl.col(numeric_cols[j])).sum().cast(pl.Float64)
                    for i in range(len(numeric_cols))
                    for j in range(i, len(numeric_cols))
                ]
                user_table_agg = user_table_processed.group_by([query_column_name, target_column_name]).agg(
                    [
                        pl.len().alias('count'),
                        pl.concat_list(pl.col(numeric_cols).sum()).alias('sum'),
                        pl.concat_list(cofactor_upper_exprs).alias('cofactor_upper'),
                    ]
                )
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
        query_column_name: str,
        dummify_n_bins: int | None = None,
    ) -> pl.DataFrame:
        # Parse each top feature name. Synthetic polynomial features use the
        # `<base>^<k>` convention (e.g. "6_42_5^2"); their base column is the
        # portion before `^`. Quantile-binned dummies use the `<base>=bin_<i>`
        # convention (e.g. "6_42_5=bin_2"). We fetch each base column once,
        # then materialize polynomial variants and bin indicators via Polars
        # expressions after the join.
        def _parse_feature(s: str) -> tuple[str, int, int | None]:
            '''Return (base_name, power, bin_index). For synthetic columns:
            polynomial powers set power > 1; quantile-binned dummies set
            bin_index to the requested bin. power and bin_index are mutually
            exclusive; both are None / 1 for plain candidates.'''
            if '=bin_' in s:
                base, bin_str = s.rsplit('=bin_', 1)
                return base, 1, int(bin_str)
            if '^' in s:
                base, power_str = s.rsplit('^', 1)
                return base, int(power_str), None
            return s, 1, None

        def _normalize_headers(raw_headers) -> list[str]:
            '''Normalize DB metadata payload to a python list of header names.'''
            if raw_headers is None:
                return []
            if isinstance(raw_headers, np.ndarray):
                return [str(x) for x in raw_headers.tolist()]
            if isinstance(raw_headers, (list, tuple)):
                return [str(x) for x in raw_headers]
            if isinstance(raw_headers, str):
                text = raw_headers.strip()
                if not text:
                    return []
                # Accept python-like list/tuple strings.
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, (list, tuple)):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
                # Accept Postgres array string format: {"a","b"}.
                if text.startswith('{') and text.endswith('}'):
                    inner = text[1:-1]
                    if not inner:
                        return []
                    return [part.strip().strip('"') for part in inner.split(',')]
            return []

        table_indices = []
        key_col_indices = []
        table_features: dict[int, dict[int, list[str]]] = {}
        # Map (table_id, key_col_id) -> {base_feature_name: set[power]} so the
        # same base column is fetched only once even if multiple powers select it.
        synthetic_powers: dict[tuple[int, int], dict[str, set[int]]] = {}
        # Map (table_id, key_col_id) -> {base_feature_name: set[bin_index]} for
        # quantile-binned dummies, mirroring synthetic_powers.
        synthetic_bins: dict[tuple[int, int], dict[str, set[int]]] = {}
        for s in top_features:
            base_name, power, bin_index = _parse_feature(s)
            feature_split = base_name.split('_')
            table_id = int(feature_split[0])
            key_col_id = int(feature_split[1])
            table_indices.append(table_id)
            key_col_indices.append(key_col_id)
            table_features.setdefault(table_id, {}).setdefault(key_col_id, [])
            if base_name not in table_features[table_id][key_col_id]:
                table_features[table_id][key_col_id].append(base_name)
            if bin_index is None:
                synthetic_powers.setdefault((table_id, key_col_id), {}).setdefault(base_name, set()).add(power)
            else:
                synthetic_bins.setdefault((table_id, key_col_id), {}).setdefault(base_name, set()).add(bin_index)

        join_selection_query_results = join_selection_query_results.filter(
            (
                (pl.col('table_index').is_in(table_indices))
                    &
                (pl.col('key_col_index').is_in(key_col_indices))
            )
        )
        # Map synthetic feature ids (e.g. "6_42_5" or "6_42_5^2") to a more
        # human-friendly "<table_name>.<column_header>" string when both pieces
        # of metadata are present in the DB rows. Falls back silently to the
        # synthetic id otherwise.
        has_table_name = 'table_name' in join_selection_query_results.columns
        has_headers = 'column_headers' in join_selection_query_results.columns
        has_feature_index = 'feature_index' in join_selection_query_results.columns
        friendly_renames: dict[str, str] = {}
        unresolved_features: list[str] = []
        aug_df = join_selection_query_results.select(pl.col('key').unique())
        for idx, group in join_selection_query_results.group_by(['table_index', 'key_col_index'], maintain_order=True):
            full_features = table_features[idx[0]].get(idx[1], [])
            if len(full_features) == 0:
                continue
            features = [int(f.split('_')[-1]) for f in full_features]
            # Keep reconstruction in the same feature space used by ranking/selection.
            # When pruning is enabled, selected feature indices refer to the masked
            # (drop_feature == 1) coordinates, not the raw DB column positions.
            drop_mask = None
            if 'drop_feature' in group.columns and group.height > 0:
                first_drop = group.row(0, named=True).get('drop_feature')
                if first_drop is not None:
                    drop_mask = [int(x) for x in first_drop]
                    group = group.with_columns(
                        pl.col('sum').map_elements(
                            lambda arr: [v for v, keep in zip(arr, drop_mask) if keep == 1],
                            return_dtype=pl.List(pl.Float64)
                        )
                    )

            if group.height == 0:
                continue

            # Guard against out-of-bounds gathers: keep only indices that exist in
            # the post-pruning vector space so augmentation never crashes and we can
            # still rename whatever is valid.
            vector_width = len(group.row(0, named=True).get('sum') or [])
            valid_pairs = [(name, feat) for name, feat in zip(full_features, features) if 0 <= feat < vector_width]
            if not valid_pairs:
                unresolved_features.extend(full_features)
                continue
            full_features = [name for name, _ in valid_pairs]
            features = [feat for _, feat in valid_pairs]

            group = group.with_columns(
                pl.col('sum')
                    .list.gather(features)
                    .list.to_struct(fields=[f'{idx[0]}_{idx[1]}_{f}' for f in features])
                    .struct.unnest()
            )
            # Resolve friendly names from the first row of this group when the
            # required metadata columns are available.
            base_friendly: dict[str, str] = {}
            if has_table_name and has_headers and group.height > 0:
                first = group.row(0, named=True)
                table_name = first.get('table_name')
                column_headers_raw = _normalize_headers(first.get('column_headers'))
                column_headers = column_headers_raw
                if drop_mask is not None and len(column_headers_raw) == len(drop_mask):
                    column_headers = [h for h, keep in zip(column_headers_raw, drop_mask) if int(keep) == 1]
                if table_name is not None and column_headers is not None:
                    # Build the best available per-position header list for the
                    # current (possibly pruned) vector space.
                    effective_headers: list[str | None] = []
                    if len(column_headers) == vector_width:
                        effective_headers = list(column_headers)

                    # Fallback: derive position mapping from feature_index metadata
                    # when header width does not match the vector width.
                    if has_feature_index:
                        try:
                            feature_indices = (
                                group
                                .select(pl.col('feature_index').drop_nulls().unique().sort())
                                .to_series()
                                .to_list()
                            )
                            encoded_to_header: dict[int, str] = {}
                            for pos, encoded_idx in enumerate(feature_indices):
                                eidx = int(encoded_idx)
                                if 0 <= eidx < len(column_headers_raw):
                                    encoded_to_header[eidx] = column_headers_raw[eidx]
                                elif pos < len(column_headers):
                                    encoded_to_header[eidx] = column_headers[pos]
                            if len(feature_indices) == vector_width:
                                effective_headers = [encoded_to_header.get(int(eidx)) for eidx in feature_indices]
                        except Exception:
                            pass

                    for base_name, feat_idx in zip(full_features, features):
                        header = None
                        if 0 <= feat_idx < len(effective_headers):
                            header = effective_headers[feat_idx]
                        elif 0 <= feat_idx < len(column_headers):
                            header = column_headers[feat_idx]
                        if header is not None and str(header).strip() != '':
                            base_friendly[base_name] = f'{table_name}.{header}'
            # Materialize polynomial variants for each selected base column.
            poly_exprs = []
            final_names: list[str] = []
            for base_name in full_features:
                powers = sorted(synthetic_powers.get((idx[0], idx[1]), {}).get(base_name, set()))
                # Include the base p=1 column ONLY if it was requested directly
                # or if a polynomial power was requested (a power requires its
                # base). Pure-dummy selections do not request the raw column.
                requested_bins = synthetic_bins.get((idx[0], idx[1]), {}).get(base_name, set())
                if not powers and not requested_bins:
                    # Backwards-compat: legacy callers never populate either.
                    powers = [1]
                for p in powers:
                    if p == 1:
                        final_names.append(base_name)
                        if base_name in base_friendly:
                            friendly_renames[base_name] = base_friendly[base_name]
                        else:
                            unresolved_features.append(base_name)
                    else:
                        syn = f'{base_name}^{p}'
                        poly_exprs.append((pl.col(base_name) ** p).alias(syn))
                        final_names.append(syn)
                        if base_name in base_friendly:
                            friendly_renames[syn] = f'{base_friendly[base_name]}^{p}'
                        else:
                            unresolved_features.append(syn)
            if poly_exprs:
                group = group.with_columns(poly_exprs)

            # Materialize quantile-binned indicators for requested dummies.
            # Reuses the same quantile algorithm as ``expand_with_quantile_bins``
            # (sketch_processing.py) on the same per-key sum values, so the
            # bin assignments match what forward saw at selection time.
            bin_exprs = []
            bin_temp_drops: list[str] = []
            for base_name in full_features:
                requested_bins = sorted(synthetic_bins.get((idx[0], idx[1]), {}).get(base_name, set()))
                if not requested_bins:
                    continue
                col_values = group[base_name].to_numpy()
                n_bins_eff = dummify_n_bins if dummify_n_bins is not None else 4
                edges = quantile_bin_thresholds(col_values, n_bins_eff)
                bin_idx_array = np.searchsorted(edges, col_values, side='right') if len(edges) else np.zeros_like(col_values, dtype=int)
                for b in requested_bins:
                    ind_name = f'{base_name}=bin_{b}'
                    indicator_values = (bin_idx_array == b).astype(np.int8)
                    group = group.with_columns(pl.Series(ind_name, indicator_values))
                    final_names.append(ind_name)
                    if base_name in base_friendly:
                        friendly_renames[ind_name] = f'{base_friendly[base_name]}=bin_{b}'
                    else:
                        unresolved_features.append(ind_name)
                # If the base column was NOT explicitly requested by forward,
                # we still pulled it to compute bin edges; drop it from the
                # final selection so it does not leak into the augmented frame.
                base_requested = base_name in final_names and base_name not in bin_temp_drops and \
                    not synthetic_powers.get((idx[0], idx[1]), {}).get(base_name, set())
                if base_requested:
                    bin_temp_drops.append(base_name)
            if bin_temp_drops:
                final_names = [n for n in final_names if n not in bin_temp_drops]

            aug_df = aug_df.join(group.select(pl.col(['key'] + final_names)), on='key', how='left')

        aug_df = aug_df.rename({'key': query_column_name})
        # Apply friendly renames last, deduplicating any collisions by appending
        # the synthetic id as a discriminator. A collision is not a bug: the
        # friendly name is `<lake table>.<column>_<aggregate>`, which omits the
        # key column, so the same column aggregated over two different key
        # columns of the same table maps to one name. Those are genuinely
        # distinct candidates -- `_drop_aggregation_siblings` qualifies a base
        # column by `<table_index>_<key_col_index>` and so does not treat them
        # as siblings -- and the `__<table>_<keycol>_<feature>` suffix keeps
        # them apart in both the augmented frame and the returned plan.
        if friendly_renames:
            seen: dict[str, int] = {}
            unique_renames: dict[str, str] = {}
            existing_cols = set(aug_df.columns)
            for synth, friendly in friendly_renames.items():
                if synth not in existing_cols:
                    continue
                count = seen.get(friendly, 0)
                seen[friendly] = count + 1
                unique_renames[synth] = friendly if count == 0 else f'{friendly}__{synth}'
            if unique_renames:
                aug_df = aug_df.rename(unique_renames)
        else:
            unique_renames = {}

        # Persist a human-friendly augmentation plan for logging/reporting.
        # Keep the raw ID when no mapping could be resolved.
        self._last_friendly_plan = [unique_renames.get(f, f) for f in top_features]
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
