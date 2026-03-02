import csv
import os
import time
import torch
import warnings
import pandas as pd
import polars as pl
import numpy as np
import random
from augmentation.retrieval import AurumJoinDiscovery
from experiments.base_tables.base_table_preprocessing import PreProcessor
from .kitana_utils.data_provider import PrepareBuyerSellers
from .kitana_utils.new_search_gpu import DataMarket, SearchEngine
from ...utils.common import process_key


class KitanaAugmenter:
    def run(
        self,
        join_paths_df_path: str,
        base_node_id: str,
        query_column_name: str,
        query_table_path: str,
        features: list[str],
        target_column_name: str,
        data_lake_path: str,
        buyer_sep: str = ",",
        lake_table_sep: str = ",",
        n_iter: int = 100,
        **kwargs
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            lake = data_lake_path.split('/')[-2]
            _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            aurum_index_file = os.path.join(_project_root, 'augmentation', 'Aurum', 'graphs', f'{lake}.pkl')
            aurum = AurumJoinDiscovery(aurum_index_file, separator=lake_table_sep)
            base_path = join_paths_df_path.split('/')[:-1]
            join_paths = aurum.find_joinable_tables(
                query_table_path=f'{"/".join(base_path)}/{base_node_id}',
                query_col=query_column_name,
                output_path=join_paths_df_path,
                features=features
            )
            join_paths_df = pd.read_csv(join_paths_df_path)
            join_keys = sorted(join_paths_df['from_column'].unique().tolist())
            join_keys = [[key] for key in join_keys]
            buyer_features = features
            splits_path = f'experiments/base_tables/{base_node_id.split(".")[0]}/splits.json'
            preprocessor = PreProcessor(query_table_path, splits_path, 0)
            buyer_df, query_col, target, nan_mask = preprocessor.run()
            buyer_df = buyer_df.to_pandas()
            target_feature_dtype = buyer_df[target_column_name].dtype
            if target_feature_dtype == 'object':
                unique_categories = buyer_df[target_column_name].unique()
                category_mapping = {category: idx for idx, category in enumerate(unique_categories)}
                buyer_df[target_column_name] = buyer_df[target_column_name].map(category_mapping)
            query_table_path = f'{"/".join(base_path)}/tmp_{base_node_id}'
            buyer_df.to_csv(query_table_path, sep=buyer_sep, index=False)
            prepare_data = PrepareBuyerSellers(buyer_sep=buyer_sep, seller_sep=lake_table_sep)
            prepare_data.add_buyer_by_path(query_table_path, join_keys, buyer_features=buyer_features, target_feature=target_column_name, sep=buyer_sep)

            lake_tables = sorted(join_paths_df['to_id'].unique().tolist())
            seller_data_paths = [f"{data_lake_path}/{table}" for table in lake_tables]

            items = join_paths_df[['from_column', 'to_column']].drop_duplicates()
            schema_mapping = dict(zip(items['to_column'], items['from_column']))
            for seller_path in seller_data_paths:
                seller_features = self._read_csv_header(seller_path, join_keys, lake_table_sep)
                prepare_data.add_seller_by_path(seller_path, join_keys, seller_features, schema_mapping=schema_mapping)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Set seeds for reproducibility
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                # Make CUDA operations deterministic
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            market = DataMarket(device)
            print(prepare_data.get_join_key_domains().keys())
            print(prepare_data.get_buyer_join_keys())
            available_domains = prepare_data.get_join_key_domains()
            buyer_join_keys = [k for k in prepare_data.get_buyer_join_keys() if k in available_domains]
            market.register_buyer(
                buyer_df=prepare_data.get_buyer_data(),
                join_keys=buyer_join_keys,
                join_key_domains=available_domains,
                target_feature=target_column_name,
                fit_by_residual=True
            )

            sellers = prepare_data.get_seller_data()
            for name in sorted(sellers.keys()):
                seller = sellers[name]
                market.register_seller(
                    seller_df=seller.data,
                    seller_name=name,
                    join_keys=seller.join_keys,
                    join_key_domains=seller.join_key_domains
                )

            engine = SearchEngine(data_market=market, fit_by_residual=True)
            augplan, _, _ = engine.start(iter=n_iter)
            augplan_set = set(augplan)
            if (len(augplan_set) == 0) or (len(augplan_set) == 1 and augplan[0] == None):
                end = time.perf_counter()
                fetch_time = end - start
                return pl.from_pandas(buyer_df), fetch_time, 0.0, []

            augplan_initial_features = []
            seller_names = []
            for tup in augplan:
                seller_name = tup[-2]
                seller_names.append(seller_name)
                colname = tup[-1]
                search_str = f'_{seller_name}_'
                initial_colname = colname[colname.find(search_str)+len(search_str):]
                augplan_initial_features.append(initial_colname)
            end = time.perf_counter()
            fetch_time = end - start

            start = time.perf_counter()
            query_df = pd.read_csv(query_table_path, sep=buyer_sep)
            augmented_df = query_df.copy()
            join_keys = [key[i] for key in join_keys for i in range(len(key))]
            for jk in join_keys:
                augmented_df[jk] = augmented_df[jk].apply(process_key)
            print(f'Augmentation plan: {list(zip(augplan_initial_features, seller_names))}')
            augplan_actual = []
            for seller_features, aug_df in zip(augplan_initial_features, seller_names):
                path = f"{data_lake_path}/{aug_df}.csv"
                print(f"Augmenting with {path} ...")
                seller_cols = self._read_csv_header(path, [], lake_table_sep)
                seller_join_col = None
                for col in seller_cols:
                    if col in schema_mapping.keys():
                        seller_join_col = col
                        break
                seller_df = pd.read_csv(path, sep=lake_table_sep, usecols=[seller_features, seller_join_col])
                join_col_dtype = augmented_df[schema_mapping[seller_join_col]].dtype
                # print(f"Original join column dtype in query df: {join_col_dtype}")
                # if join_col_dtype == 'float64' or join_col_dtype == 'object':
                #     try:
                #         seller_df[seller_join_col] = seller_df[seller_join_col].astype('Int64')
                #     except ValueError:
                #         seller_df[seller_join_col] = seller_df[seller_join_col].astype('object')
                # print(f"Join column in seller df before processing: {seller_df[seller_join_col]}")
                seller_df = seller_df.groupby(seller_join_col).sample(
                    n=1, random_state=42
                )
                seller_df[seller_join_col] = seller_df[seller_join_col].apply(process_key)
                # print(f"Join column in seller df after processing: {seller_df[seller_join_col]}")
                query_df_join_col = schema_mapping[seller_join_col]
                # print(f"Joining on {query_df_join_col} and {seller_join_col} ...")
                # print(f"Data types check before merge: {augmented_df[query_df_join_col].dtype}, {seller_df[seller_join_col].dtype}")
                # print(f"Set intersection size: {len(set(augmented_df[query_df_join_col]).intersection(set(seller_df[seller_join_col])))}")
                # print(f"Examples of join keys in query df: {augmented_df[query_df_join_col].sort_values().unique()[:5]}")
                # print(f"Examples of join keys in seller df: {seller_df[seller_join_col].sort_values().unique()[:5]}")
                augmented_df = augmented_df.merge(
                    seller_df,
                    how='left',
                    left_on=query_df_join_col,
                    right_on=seller_join_col,
                    suffixes=('', f'_{aug_df}')
                )
                try:
                    augmented_df.drop(columns=[seller_join_col], inplace=True)
                except KeyError:
                    pass
                augplan_actual.append(seller_features)
            end = time.perf_counter()
            augmentation_time = end-start
        augmented_df = pl.from_pandas(augmented_df)

        return augmented_df, fetch_time, augmentation_time, augplan_actual


    def _read_csv_header(self, csv_file: str, join_keys: list, lake_table_sep: str) -> list:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=str(lake_table_sep))
            first_row = next(reader)
            join_keys_flat = [i[j] for i in join_keys for j in range(len(i))]
            join_keys_set = set(join_keys_flat)
            # Preserve order while removing join keys
            first_row = [col for col in first_row if col not in join_keys_set]
        return sorted(first_row)  # Sort for deterministic ordering