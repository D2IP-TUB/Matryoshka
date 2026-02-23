import json
import time
from augmentation.join_selection import JoinSelection
from augmentation.utils.config import DiscoveryConfig
from experiments.base_tables.base_table_preprocessing import PreProcessor

TABLE = 'energy'
LAKE = 'nyc'
base_table_path = f'experiments/base_tables/{TABLE}/{TABLE}.csv'
base_table_splits_path = f'experiments/base_tables/{TABLE}/splits.json'
with open(base_table_splits_path, 'r') as f:
    splits = json.load(f)
features = splits[0]['features']
preprocessor = PreProcessor(base_table_path, base_table_splits_path, 0)
X, query_col, target, nan_mask = preprocessor.run()

config = DiscoveryConfig(
    task='regression',
    ranking='passthrough',
    model='RegressionQR',
    strategy='ForwardSelection',
    params={
        'tol': 0.001,
        'metric': 'mse',
    }
)
# config = DiscoveryConfig(
#     task='classification',
#     ranking='passthrough',
#     model='ClassificationCholesky',
#     strategy='ForwardSelection',
#     params={
#         'tol': 0.3,
#         'metric': 'average_mahalanobis',
#     }
# )
# config = DiscoveryConfig(
#     task='regression',
#     model='LinRegL1',
#     strategy='LassoFeatureSelector',
#     params={
#         'alphas': [100, 10, 0.1, 0.01, 0.001],
#         'cv': 5,
#         'metric': 'mse'
#     }
# )
# config = DiscoveryConfig(
#     strategy='ArdaAugmenter',
#     baseline=True,
#     params=dict(
#         join_paths_df_path=f'experiments/base_tables/{TABLE}/join_paths.csv',
#         query_table=X,
#         query_table_path=base_table_path,
#         features=features,
#         query_column_name=query_col,
#         data_lake_path=f'/mnt/data1/lakes/{LAKE}/extracted',
#         base_node_id=f'{TABLE}.csv',
#         target_column_name=target,
#         sample_size=3000,
#         regression=False,
#         lake_table_sep=','
#     )
# )
# config = DiscoveryConfig(
#     strategy='KitanaAugmenter',
#     baseline=True,
#     params=dict(
#         join_paths_df_path=f'experiments/base_tables/{TABLE}/join_paths.csv',
#         base_node_id=f'{TABLE}.csv',
#         query_column_name=query_col,
#         query_table_path=base_table_path,
#         target_column_name=target,
#         data_lake_path=f'/mnt/data1/lakes/{LAKE}/extracted',
#         buyer_sep=',',
#         lake_table_sep=',',
#         n_iter=100,
#         features=features
#     )
# )
# config = DiscoveryConfig(
#     strategy='AutofeatAugmenter',
#     baseline=True,
#     params=dict(
#         join_paths_df_path=f'experiments/base_tables/{TABLE}/join_paths.csv',
#         base_node_id=f'{TABLE}.csv',
#         query_column_name=query_col,
#         target_column_name=target,
#         data_lake_path=f'/mnt/data1/lakes/{LAKE}/extracted',
#         base_table_sep=',',
#         lake_table_sep='\t',
#         problem_type='multiclass',
#         base_table_label=target,
#         features=features
#     )
# )
# config = DiscoveryConfig(
#     strategy='QcrAugmenter',
#     baseline=True,
#     params=dict(
#         top_k=50,
#         query_column_name=query_col,
#         target_column_name=target,
#         qcr_table_name='gittables_qcr_index',
#         data_lake_path='/mnt/data1/lakes/gittables/extracted',
#         query_table=X,
#         lake_table_sep=','
#     )
# )
worker = JoinSelection(
    feature_selection_table_name=f'{LAKE}_matryoshka_fs_index',
    overlap_table_name=f'{LAKE}_matryoshka_overlap_index',
    verbose=True,
    log_dir='experiments/base_tables',
    log_file_name='backward.log'
)
runtimes = {}
for n_jobs in [1, 4, 16]:
    start = time.perf_counter()
    df_aug, _ = worker.find_best_joins(
        config=config,
        user_table_processed=X,
        query_column_name=query_col,
        target_column_name=target,
        top_k=20,
        n_jobs=n_jobs,
        debug=True,
        corr_threshold=0.55
    )
    end = time.perf_counter()
    runtimes[n_jobs] = end - start
print(runtimes)
# print(f'Time taken: {end - start} seconds')
# print(df_aug.schema)
# print(df_aug.shape)
# if config.baseline:
#     df_aug[0].write_csv('experiments/base_tables/housing/augmented/backward.csv')
# else:
#     df_aug.write_csv('experiments/base_tables/housing/augmented/backward.csv')