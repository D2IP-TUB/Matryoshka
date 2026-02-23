import adbc_driver_postgresql.dbapi as adbc_dbapi
import psycopg
import ray
import re
import numpy as np
import polars as pl
from collections import namedtuple
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from ..base.model import FeatureSelectionModel


@ray.remote(num_cpus=0)
class LassoFeatureSelector:
    def __init__(self, metric, random_state: int = 42):
        self.metric = metric
        self.random_state = random_state

        
    def run(self, model: FeatureSelectionModel, task: str, **kwargs):
        token_query_results = kwargs['token_query_results']
        join_selection_query_results = kwargs['join_selection_query_results']
        base_table = kwargs['base_table']
        conninfo = base_table.conninfo
        feature_selection_table_name = base_table.feature_selection_table_name
        user_table_processed = base_table.table
        query_column_name = base_table.query_column_name
        target_column_name = base_table.target_column_name

        augmentation_plan = self._create_augmentation_plan(token_query_results, feature_selection_table_name, conninfo)
        aug_df = self._create_augmentation_table(
            user_table_processed,
            join_selection_query_results,
            augmentation_plan,
            query_column_name
        )
        X_aug = aug_df.drop([query_column_name, target_column_name])
        imp = SimpleImputer(strategy='mean')
        X = imp.fit_transform(X_aug.to_numpy())
        y = aug_df[target_column_name].to_numpy().reshape(-1)

        alphas = kwargs['alphas']
        cv = kwargs['cv']
        match task:
            case 'regression':
                model_instance = model()
                param_grid = {'alpha': alphas}
                grid_search = GridSearchCV(model_instance, param_grid, cv=cv, scoring='neg_mean_squared_error')
                grid_search.fit(X, y)
                coefs = grid_search.best_estimator_.coef_
                nonzero_coefs = np.where(coefs != 0)[0]
            case 'classification':
                model_instance = model()
                param_grid = {'C': [1.0 / a for a in alphas]}  # skglm uses C = 1/alpha
                grid_search = GridSearchCV(model_instance, param_grid, cv=cv, scoring=self.metric)
                grid_search.fit(X, y)
                coefs = grid_search.best_estimator_.coef_
                nonzero_mask = np.any(coefs != 0, axis=0)
                nonzero_coefs = np.where(nonzero_mask)[0]
        nonzero_coefs = nonzero_coefs.tolist()
        if len(nonzero_coefs) > 0:
            nonzero_features = X_aug.select(pl.nth(*nonzero_coefs)).columns
            augmentation_plan = [f for f in nonzero_features if f not in user_table_processed.columns]
        else:
            augmentation_plan = []
        aug_feature_indices = {i: augmentation_plan[i] for i in range(len(augmentation_plan))}
        Result = namedtuple('Result', ['aug_feature_indices'])

        return Result(aug_feature_indices)


    def _create_augmentation_plan(self, token_query_results: pl.DataFrame, feature_selection_table_name: str, conninfo: str) -> pl.DataFrame:
        join_selection_query_results = self._run_join_selection_query(token_query_results, feature_selection_table_name, conninfo)
        join_selection_query_results = join_selection_query_results.with_columns(full_index = pl.col('table_column_index')+'_0')
        full_indices = join_selection_query_results.group_by('full_index').agg(pl.col('sum').list.len().first()).to_numpy()
        augmentation_plan = []
        for i in full_indices:
            base_str, els = i[0], i[1]
            augmentation_plan.extend([re.sub(r'_0$', f'_{str(j)}', base_str) for j in range(els)])

        return augmentation_plan


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
    

    def _run_join_selection_query(self, token_query_results: pl.DataFrame, feature_selection_table_name: pl.DataFrame, conninfo: str) -> pl.DataFrame:
        token_query_results_arrow = token_query_results.to_arrow()

        conn = adbc_dbapi.connect(conninfo)
        try:
            with conn.cursor(adbc_stmt_kwargs={"adbc.postgresql.batch_size_hint_bytes": 512 * 1024 * 1024}) as cur:
                cur.adbc_ingest("temp_valid_combinations", token_query_results_arrow, mode="create", temporary=True)
                join_sql = f"""
                SELECT f.key, f.feature_index, f.table_index, f.key_col_index, f.row_index, f.count, f.sum, f.diag, f.qcr_term_positive, f.qcr_term_negative
                FROM {feature_selection_table_name} AS f
                JOIN temp_valid_combinations AS v
                USING (table_index, key_col_index, row_index)
                """
                cur.execute(join_sql)
                join_selection_query_results = cur.fetch_polars()
        finally:
            conn.close()

        join_selection_query_results = join_selection_query_results.with_columns(pl.col(['sum', 'diag', 'qcr_term_positive', 'qcr_term_negative']).cast(pl.Utf8).str.json_decode(dtype=pl.List(pl.Float64)))
        join_selection_query_results = join_selection_query_results.with_columns(
            pl.concat_str(pl.col('table_index'), pl.col('key_col_index'), separator='_').alias('table_column_index'),
            pl.concat_str(pl.col('table_index'), pl.col('row_index'), separator='_').alias('table_row_index')
        )

        return join_selection_query_results