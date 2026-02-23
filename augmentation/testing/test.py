import polars as pl
from augmentation.join_selection import JoinSelection
from augmentation.feature_selection.meta_algo import FullTrueJoin
from augmentation.feature_selection.models import Lasso
from augmentation.testing.data import DataGenerator

# offline
table = pl.read_csv('experiments/base_tables/airbnb/AB_US_2023.csv').filter(pl.col('city') == 'New York City')
datagen = DataGenerator(
    feature_selection_table_name='gt_fs_index_mm',
    overlap_table_name='gt_overlap_mm',
    token_index_table_name='gt_token_index_mm'
)
datagen.index_lake(table, query_col_name='id', target_col_name='price', n_rows=50, table_index=0)

# online
r1 = pl.read_csv('augmentation/testing/base_table.csv')
worker = JoinSelection(
    feature_selection_table_name='gt_fs_index_mm',
    overlap_table_name='gt_overlap_mm',
    token_index_table_name='gt_token_index_mm'
)
r1 = r1.with_columns(pl.col('id').cast(pl.String))
print(
    worker.find_best_joins(
        user_table_processed=r1,
        query_column_name='id',
        target_feature_name='price',
        strategy=FullTrueJoin,
        model=Lasso,
        metric='mse',
        tol=0.000001,
        top_k=2,
        min_overlap_share=0.000001,
        n_jobs=16,
        alphas=[0.0001, 100.0],
        n_trials=100
    )
)