import ray
import numpy as np
import polars as pl
from collections import defaultdict, namedtuple
from kneefinder import KneeFinder
from typing import Any, Literal

from tabulate import tabulate
from .sketch_processing import SketchProcessor


class OverlapRanking:
    def __init__(self, feature_selection_table_name: str, conninfo: str, corr_threshold: float = None, var_threshold: float = None):
        self.join_selection_query = 'COPY ( ' \
                                   f'   SELECT * ' \
                                   f'   FROM {feature_selection_table_name} ' \
                                    '   WHERE (table_index, key_col_index, row_index) IN ' \
                                    '   (' \
                                    '       SELECT table_index, key_col_index, row_index ' \
                                    f'      FROM temp_valid_combinations ' \
                                    '   )' \
                                    ' ) TO STDOUT WITH (FORMAT BINARY);'
        self.conninfo = conninfo
        self.feature_selection_table_name = feature_selection_table_name
        self.corr_threshold = corr_threshold
        self.var_threshold = var_threshold


    def rank(self, method: Literal['passthrough', 'correlation', 'eta', 'joinability'], n_jobs: int, **kwargs) -> Any:
        ray.init(
            include_dashboard=False,  # disables the web dashboard
            logging_level="ERROR",    # suppress most logs
            ignore_reinit_error=True, # avoids warnings on reinit
            num_cpus=n_jobs,               # limits CPU usage if needed
            _metrics_export_port=None, # disables metrics export
            _system_config={
                "task_events_report_interval_ms": 0
            }
        )
        match method:
            case 'passthrough':
                joint_tuples = self._passthrough(**kwargs)

            case 'correlation':
                joint_tuples = self._rank_by_correlation(**kwargs)

            case 'eta':
                joint_tuples = self._rank_by_eta(**kwargs)

            case 'joinability':
                joint_tuples = self._rank_by_joinability(**kwargs)

        return joint_tuples


    def _passthrough(self, **kwargs):
        task = kwargs['task']
        user_table_agg = kwargs['user_table_agg']
        query_column_name = kwargs['query_column_name']
        target_column_name = kwargs['target_column_name']
        join_selection_query_results = kwargs['join_selection_query_results']
        sketch_proc = SketchProcessor.remote(
            conninfo=self.conninfo,
            feature_selection_table_name=self.feature_selection_table_name,
            corr_threshold=self.corr_threshold,
            var_threshold=self.var_threshold
        )
        if 'regression' in task:
            aug_table_sketches = ray.get(
                [
                    sketch_proc._create_aug_table_sketch.remote(group, user_table_agg, query_column_name, self.corr_threshold, self.var_threshold)
                    for group in join_selection_query_results.group_by(['table_index', 'feature_index', 'table_column_index'])
                ]
            )
            aug_table_sketches = [sketch for sketch in aug_table_sketches if sketch is not None]
            base_table_sketch = ray.get(sketch_proc._create_base_table_sketch.remote(user_table_agg, False))
            Result = namedtuple('JointTuple', ['base_table_sketch', 'aug_table_sketches'])

            return Result(base_table_sketch, aug_table_sketches)

        elif 'classification' in task:
            base_table_sketches = ray.get(
                [
                    sketch_proc._create_base_table_sketch_clf.remote(sub_table, query_column_name, target_column_name, label)
                    for label, sub_table in (
                        user_table_agg
                            .group_by(target_column_name)
                    )
                ]
            )
            base_table_sketches_per_class = {k: v for d in base_table_sketches for k, v in d.items()}
            base_table_sketches_per_class = dict(sorted(base_table_sketches_per_class.items(), key=lambda item: item[0]))

            aug_table_sketches = ray.get(
                [
                    sketch_proc._create_aug_table_sketch_clf.remote(group, user_table_agg, query_column_name, target_column_name)
                    for group in join_selection_query_results.group_by(['table_index', 'feature_index', 'table_column_index'])
                ]
            )
            aug_table_sketches = [sketch for sketch in aug_table_sketches if sketch is not None]
            aug_table_sketches_per_class = defaultdict(list)
            for d in aug_table_sketches:
                for key, values in d.items():
                    aug_table_sketches_per_class[key].extend(values)
            Result = namedtuple('JointTuple', ['base_table_sketches_per_class', 'aug_table_sketches_per_class'])
            
            return Result(base_table_sketches_per_class, aug_table_sketches_per_class)


    def _rank_by_joinability(self, **kwargs):
        task = kwargs['task']
        user_table_agg = kwargs['user_table_agg']
        query_column_name = kwargs['query_column_name']
        target_column_name = kwargs['target_column_name']
        join_selection_query_results = kwargs['join_selection_query_results']
        sketch_proc = SketchProcessor.remote(
            conninfo=self.conninfo,
            feature_selection_table_name=self.feature_selection_table_name,
            corr_threshold=self.corr_threshold,
            var_threshold=self.var_threshold
        )

        if 'regression' in task:
            aug_table_sketches = ray.get(
                [
                    sketch_proc._create_aug_table_sketch.remote(group, user_table_agg, query_column_name, self.corr_threshold, self.var_threshold)
                    for group in join_selection_query_results.group_by(['table_index', 'feature_index', 'table_column_index'])
                ]
            )
            aug_table_sketches = [sketch for sketch in aug_table_sketches if sketch is not None]
            n_init_features = len(user_table_agg.select('sum').row(0)[0]) - 1 # -1 for the target column which will not be passed to feature selection
            aug_table_sketch = ray.get(sketch_proc._join_aug_sketches.remote(aug_table_sketches, n_init_features))
            table_feature_index = aug_table_sketch.table_feature_index

            base_table_sketch = ray.get(sketch_proc._create_base_table_sketch.remote(user_table_agg, False))
            joint_cofactor_matrix, joint_feature_target_vector, joint_count, y_t_y, new_features_count, sum_y = ray.get(sketch_proc._join_aug_table.remote(table_feature_index, base_table_sketch, aug_table_sketch, task))
            Result = namedtuple('JointTuple', ['joint_cofactor_matrix', 'joint_feature_target_vector', 'joint_count', 'y_t_y', 'new_features_count', 'aug_feature_indices', 'sum_y', 'aug_table_sketches', 'base_table_sketch'])

            return Result(joint_cofactor_matrix, joint_feature_target_vector, joint_count, y_t_y, new_features_count, aug_table_sketch.aug_feature_indices, sum_y, aug_table_sketches, base_table_sketch)

        elif 'classification' in task:
            base_table_sketches = ray.get(
                [
                    sketch_proc._create_base_table_sketch_clf.remote(sub_table, query_column_name, target_column_name, label)
                    for label, sub_table in (
                        user_table_agg
                            .group_by(target_column_name)
                    )
                ]
            )
            base_table_sketches_per_class = {k: v for d in base_table_sketches for k, v in d.items()}
            base_table_sketches_per_class = dict(sorted(base_table_sketches_per_class.items(), key=lambda item: item[0]))

            aug_table_sketches = ray.get(
                [
                    sketch_proc._create_aug_table_sketch_clf.remote(group, user_table_agg, query_column_name, target_column_name)
                    for group in join_selection_query_results.group_by(['table_index', 'feature_index', 'table_column_index'])
                ]
            )
            aug_table_sketches = [sketch for sketch in aug_table_sketches if sketch is not None]
            aug_table_sketches_per_class = defaultdict(list)
            for d in aug_table_sketches:
                for key, values in d.items():
                    aug_table_sketches_per_class[key].extend(values)

            n_init_features = len(user_table_agg.select('sum').row(0)[0])
            joint_aug_table_sketches_per_class = ray.get([sketch_proc._join_aug_sketches.remote(aug_table_sketches_per_class[key], n_init_features, key) for key in aug_table_sketches_per_class.keys()])
            joint_aug_table_sketches_per_class = {k: v for d in joint_aug_table_sketches_per_class for k, v in d.items()}
            joint_aug_table_sketches_per_class = dict(sorted(joint_aug_table_sketches_per_class.items(), key=lambda item: item[0]))

            table_feature_index = joint_aug_table_sketches_per_class[0].table_feature_index
            joint_tuples = [
                ray.get(sketch_proc._join_aug_table.remote(table_feature_index, base_table, aug_table, task, label))
                for label, base_table, aug_table in zip(
                    base_table_sketches_per_class.keys(), base_table_sketches_per_class.values(), joint_aug_table_sketches_per_class.values()
                )
            ]
            joint_tuples = {k: v for d in joint_tuples for k, v in d.items()}

            Result = namedtuple('JointTuple', ['class_dict', 'new_features_count', 'aug_feature_indices', 'aug_table_sketches_per_class'])
            class_dict = {}
            for label, (joint_cofactor_matrix_std, features_mean, joint_count, _) in joint_tuples.items():
                class_dict[label] = {
                    'joint_cofactor_matrix': joint_cofactor_matrix_std,
                    'features_mean': features_mean,
                    'joint_count': joint_count
                }
            new_features_count = len(features_mean) - n_init_features
            aug_feature_indices = joint_aug_table_sketches_per_class[0].aug_feature_indices

            return Result(class_dict, new_features_count, aug_feature_indices, aug_table_sketches_per_class)


    def _rank_by_correlation(
        self,
        user_table_agg: pl.DataFrame,
        query_column_name: str,
        **kwargs
    ):
        task = kwargs['task']
        join_selection_query_results = kwargs['join_selection_query_results']

        sketch_proc = SketchProcessor.remote(
            conninfo=self.conninfo,
            feature_selection_table_name=self.feature_selection_table_name,
            corr_threshold=self.corr_threshold,
            var_threshold=self.var_threshold
        )
        aug_table_sketches = ray.get(
            [
                sketch_proc._create_aug_table_sketch.remote(group, user_table_agg, query_column_name, self.corr_threshold, self.var_threshold)
                for group in join_selection_query_results.group_by(['table_index', 'feature_index', 'table_column_index'])
            ]
        )
        aug_table_sketches = np.array(aug_table_sketches)
        aug_table_sketches = aug_table_sketches[aug_table_sketches != None].tolist()
        n_init_features = len(user_table_agg.select('sum').row(0)[0]) - 1 # -1 for the target column which will not be passed to feature selection
        aug_table_sketch = ray.get(sketch_proc._join_aug_sketches.remote(aug_table_sketches, n_init_features))
        table_feature_index = aug_table_sketch.table_feature_index
        base_table_sketch = ray.get(sketch_proc._create_base_table_sketch.remote(user_table_agg, True))

        joint_cofactor_matrix, _, joint_count, y_t_y, new_features_count, sum_y = ray.get(sketch_proc._join_aug_table.remote(table_feature_index, base_table_sketch, aug_table_sketch, task, standardize=False))
        joint_feature_target_vector = joint_cofactor_matrix[:, 0].reshape(-1, 1)
        cache = ray.get(sketch_proc.retrieve.remote('cache'))[table_feature_index]
        features_sum, target_sum = cache['joint_sum_vec'], cache['base_target_sum_vec']
        joint_cofactor_matrix, joint_feature_target_vector = ray.get(
            sketch_proc._standardize_gram_inputs.remote(
                joint_cofactor_matrix,
                joint_feature_target_vector,
                features_sum,
                target_sum,
                joint_count
            )
        )

        corr_dict = self._pearson_corr(joint_cofactor_matrix, joint_count, new_features_count)
        elbow_idx = self._find_elbow(corr_dict)
        keys_to_keep = list(corr_dict.keys())[:elbow_idx+1]
        new_features_count_corr = len(keys_to_keep)

        init_features_idx = (np.arange(joint_cofactor_matrix.shape[1] - new_features_count)).tolist()[1:]
        keys_to_keep_all = init_features_idx + keys_to_keep

        joint_cofactor_matrix = joint_cofactor_matrix[np.ix_(keys_to_keep_all, keys_to_keep_all)]
        joint_feature_target_vector = joint_feature_target_vector[keys_to_keep_all]

        aug_feature_indices = aug_table_sketch.aug_feature_indices
        aug_feature_indices = {k: v for k, v in aug_feature_indices.items() if k in keys_to_keep}
        new_keys = [k for k in range(init_features_idx[-1], len(keys_to_keep_all))]
        vals = list(aug_feature_indices.values())
        aug_feature_indices = dict(zip(new_keys, vals))

        Result = namedtuple('JointTuple', ['joint_cofactor_matrix', 'joint_feature_target_vector', 'joint_count', 'y_t_y', 'new_features_count', 'aug_feature_indices', 'sum_y'])

        return Result(joint_cofactor_matrix, joint_feature_target_vector, joint_count, y_t_y, new_features_count_corr, aug_feature_indices, sum_y)


    def _pearson_corr(self, joint_cofactor_matrix: np.ndarray, joint_count: int, new_features_count: int) -> tuple[np.ndarray]:
        covar_matrix = joint_cofactor_matrix / joint_count
        new_features_first_index = covar_matrix.shape[0] - new_features_count
        target_corr = np.abs(covar_matrix[new_features_first_index:, 0])
        indexes = np.arange(target_corr.shape[0])+new_features_first_index
        corr_dict = {i: c for i, c in zip(indexes, target_corr)}
        corr_dict = dict(sorted(corr_dict.items(), key=lambda item: item[1], reverse=True))

        return corr_dict


    def _rank_by_eta(
        self,
        user_table_agg: pl.DataFrame,
        query_column_name: str,
        target_column_name: str,
        **kwargs
    ):
        task = kwargs['task']
        join_selection_query_results = kwargs['join_selection_query_results']

        sketch_proc = SketchProcessor.remote(
            conninfo=self.conninfo,
            feature_selection_table_name=self.feature_selection_table_name,
            corr_threshold=self.corr_threshold,
            var_threshold=self.var_threshold
        )

        base_table_sketches = ray.get(
            [
                sketch_proc._create_base_table_sketch_clf.remote(sub_table, query_column_name, target_column_name, label)
                for label, sub_table in (
                    user_table_agg
                        .group_by(target_column_name)
                )
            ]
        )
        base_table_sketches_per_class = {k: v for d in base_table_sketches for k, v in d.items()}
        base_table_sketches_per_class = dict(sorted(base_table_sketches_per_class.items(), key=lambda item: item[0]))

        aug_table_sketches = ray.get(
            [
                sketch_proc._create_aug_table_sketch_clf.remote(group, user_table_agg, query_column_name, target_column_name, var_threshold=self.var_threshold, corr_threshold=self.corr_threshold)
                for group in join_selection_query_results.group_by(['table_index', 'feature_index', 'table_column_index'])
            ]
        )
        aug_table_sketches = [sketch for sketch in aug_table_sketches if sketch is not None]
        aug_table_sketches_per_class = defaultdict(list)
        for d in aug_table_sketches:
            for key, values in d.items():
                aug_table_sketches_per_class[key].extend(values)

        n_init_features = len(user_table_agg.select('sum').row(0)[0])
        joint_aug_table_sketches_per_class = ray.get([sketch_proc._join_aug_sketches.remote(aug_table_sketches_per_class[key], n_init_features, key) for key in aug_table_sketches_per_class.keys()])
        joint_aug_table_sketches_per_class = {k: v for d in joint_aug_table_sketches_per_class for k, v in d.items()}
        joint_aug_table_sketches_per_class = dict(sorted(joint_aug_table_sketches_per_class.items(), key=lambda item: item[0]))

        table_feature_index = joint_aug_table_sketches_per_class[0].table_feature_index
        joint_tuples = [
            ray.get(sketch_proc._join_aug_table.remote(table_feature_index, base_table, aug_table, task, label))
            for label, base_table, aug_table in zip(
                base_table_sketches_per_class.keys(), base_table_sketches_per_class.values(), joint_aug_table_sketches_per_class.values()
            )
        ]
        joint_tuples = {k: v for d in joint_tuples for k, v in d.items()}
        # eta_dict = self._correlation_ratio(joint_tuples)

        # elbow_idx = self._find_elbow(eta_dict)
        # keys_to_keep = list(eta_dict.keys())[:elbow_idx+1]
        # new_features_count_eta = len(keys_to_keep)

        new_features_count = joint_tuples[0][-1]


        new_features_count_eta = new_features_count

        # init_features_idx = (np.arange(joint_tuples[0][0].shape[1] - new_features_count)).tolist()
        # keys_to_keep_all = init_features_idx + keys_to_keep

        aug_feature_indices = joint_aug_table_sketches_per_class[0].aug_feature_indices
        # aug_feature_indices = {k: v for k, v in aug_feature_indices.items() if k in keys_to_keep}
        # new_keys = [k for k in range(init_features_idx[-1]+1, len(keys_to_keep_all)+1)]
        # vals = list(aug_feature_indices.values())
        # aug_feature_indices = dict(zip(new_keys, vals))

        Result = namedtuple('JointTuple', ['class_dict', 'new_features_count', 'aug_feature_indices', 'aug_table_sketches_per_class'])
        class_dict = {}
        for label, (joint_cofactor_matrix_std, features_mean, joint_count, _) in joint_tuples.items():
            class_dict[label] = {
                # 'joint_cofactor_matrix': joint_cofactor_matrix_std[np.ix_(keys_to_keep_all, keys_to_keep_all)],
                'joint_cofactor_matrix': joint_cofactor_matrix_std,
                # 'features_mean': features_mean[keys_to_keep_all],
                'features_mean': features_mean,
                'joint_count': joint_count
            }

        return Result(class_dict, new_features_count_eta, aug_feature_indices, aug_table_sketches_per_class)


    def _correlation_ratio(self, joint_tuples: dict):
        class_means = []
        class_counts = []
        class_covars = []
        for joint_cofactor_matrix_std, features_mean, joint_count, new_features_count in joint_tuples.values():
            class_means.append(features_mean)
            class_counts.append(joint_count)
            class_covars.append(joint_cofactor_matrix_std / joint_count)
        class_means = np.array(class_means)
        class_counts = np.array(class_counts)
        class_covars = np.array(class_covars)
        grand_mean = np.average(class_means, axis=0, weights=class_counts)

        ss_between = np.sum(class_counts[:, None] * (class_means - grand_mean) ** 2, axis=0)
        class_vars = np.stack([np.diag(cov) for cov in class_covars])
        ss_within = np.sum((class_counts - 1)[:, None] * class_vars, axis=0)

        ss_total = ss_between + ss_within
        eta = np.sqrt(np.divide(ss_between, ss_total, out=np.zeros_like(ss_between), where=ss_total>0))

        new_features_first_index = class_covars[0].shape[0] - new_features_count
        indexes = np.arange(eta.shape[0])+new_features_first_index
        eta_dict = {i: c for i, c in zip(indexes, eta)}
        eta_dict = dict(sorted(eta_dict.items(), key=lambda item: item[1], reverse=True))

        return eta_dict
    

    def _find_elbow(self, corr_dict: dict[int, float]) -> int:
        y = np.array(list(corr_dict.values()))
        nan_indexes = np.isnan(y)
        y[nan_indexes] = 0.0
        elbow_idx = int(np.percentile(np.arange(len(y)), 90))

        return elbow_idx