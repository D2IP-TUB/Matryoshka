import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../augmentation/')))

import numpy as np
import polars as pl
from augmentation.index import ExhaustiveIndex
from itertools import zip_longest
from typing import Callable


class DataGenerator(ExhaustiveIndex):
    def __init__(self, token_index_table_name: str = None, feature_selection_table_name: str = None, overlap_table_name: str = None, n_threads: int = 128) -> None:
        super().__init__(token_index_table_name=token_index_table_name, feature_selection_table_name=feature_selection_table_name, overlap_table_name=overlap_table_name, data_dir=None, n_threads=n_threads)
        self.token_index_table_name = token_index_table_name
        self.feature_selection_table_name = feature_selection_table_name
        self.overlap_table_name = overlap_table_name


    def index_lake(self, table: pl.DataFrame, query_col_name: str, target_col_name: str, n_rows: int, table_index: int) -> list[str]:
        conn = self.db_connect_pool()
        self.create_tables(conn)
        conn.closeall()

        processed_tables_path = self._compute_abspath('processed_aug_tables')
        if not os.path.exists(processed_tables_path):
            os.mkdir(processed_tables_path)

        tables = self._generate_base_dim_tables(table, query_col_name, target_col_name, n_rows)
        for t in tables:
            self.index_table(t, table_index, key_columns=[query_col_name])
            table_index += 1

        conn = self.db_connect()
        self.create_overlap_table(conn)
        conn.close()
        conn = self.db_connect()
        self.create_db_table_index(conn)
        conn.close()

        generated_tables = os.listdir(processed_tables_path)

        return generated_tables

    def _index_features(self, table: pl.LazyFrame, query_func: Callable, group_col: str, group_col_index: int, numeric_cols: list[str], non_numeric_cols: list[str], table_index: int, max_feature_index: int, feature_extraction: bool = True) -> list[tuple]:
        result = query_func(table, group_col, numeric_cols, non_numeric_cols, feature_extraction)
        grouped_columns = self._grouped_columns_mapping(result, group_col)

        feature_selection_index = []
        for col in list(grouped_columns.keys()):
            try:
                mapped_cols = grouped_columns[col]
            except KeyError:
                continue

            feature_slice = result.select([group_col] + mapped_cols)
            # Find the base column
            for feature_slice_col in feature_slice.columns:
                if feature_slice_col.count('_') == 1:
                    base_col = feature_slice_col
                    break

            combs = self._combinations_with_element(mapped_cols, base_col)
            features_col_indices = [i for i in range(max_feature_index, len(combs) + max_feature_index)]
            max_feature_index = max(features_col_indices) + 1
            
            key_col = feature_slice.select(group_col).to_series()
            feature_slice = feature_slice.select(mapped_cols)
            
            processed_tables_path = self._compute_abspath('processed_aug_tables')
            for feature_index, c in zip(features_col_indices, combs):
                table_name = os.path.join(processed_tables_path, f'{table_index}_{feature_index}.csv')
                (
                    pl.concat(
                        [
                            pl.DataFrame(key_col),
                            feature_slice.select(pl.nth(*c))
                        ],
                        how='horizontal'
                    )
                    .write_csv(table_name)
                )

            feature_slice = feature_slice.to_numpy()
            feature_selection_index.extend(self._inverted_index_from_feature_matrix(feature_slice, key_col, table_index, group_col_index, combs, features_col_indices))

            if len(mapped_cols) > 1:
                mapped_cols.remove(base_col)
                for k in mapped_cols:
                    try:
                        del grouped_columns[k]
                    except KeyError:
                        pass
        
        new_max_feature_index = max_feature_index

        return feature_selection_index, new_max_feature_index

    
    def _generate_base_dim_tables(self, table: pl.DataFrame, query_col_name: str, target_col_name: str, n_rows: int):
        base_features = ['calculated_host_listings_count', 'availability_365']
        numeric_cols, non_numeric_cols = self._split_columns_by_type_internal(table, query_col_name, target_col_name, base_features)
        
        blended_cols = [col for pair in zip_longest(numeric_cols, non_numeric_cols) for col in pair if col is not None]
        half = len(blended_cols) // 2
        dim1_features, dim2_features = blended_cols[:half], blended_cols[half:]

        base_table = table.select([query_col_name, target_col_name] + base_features)[:n_rows].with_columns(pl.col(query_col_name).cast(pl.String))
        half_rows = n_rows // 2
        base_table = pl.concat((
            base_table[:half_rows].select(pl.all().repeat_by(2).flatten()),
            base_table[half_rows:].select(pl.all().repeat_by(3).flatten())
        ))
        
        dim1_table = table.select([query_col_name] + dim1_features)[:n_rows].with_columns(pl.col(query_col_name).cast(pl.String))
        dim1_table = pl.concat((
            dim1_table[:half_rows].select(pl.all().repeat_by(2).flatten()),
            dim1_table[half_rows:].select(pl.all().repeat_by(3).flatten())
        ))

        dim2_table = table.select([query_col_name] + dim2_features)[:n_rows].with_columns(pl.col(query_col_name).cast(pl.String))
        dim2_table = pl.concat((
            dim2_table[:half_rows].select(pl.all().repeat_by(3).flatten()),
            dim2_table[half_rows:].select(pl.all().repeat_by(2).flatten())
        ))

        assert (set(dim1_table.select(pl.col(query_col_name)).to_numpy().squeeze()) == set(dim2_table.select(pl.col(query_col_name)).to_numpy().squeeze())), "Query columns in dim1 and dim2 tables must match."

        base_table_path = self._compute_abspath('base_table.csv')
        base_table.write_csv(base_table_path)

        dim1_table_path = self._compute_abspath('dim1_table.csv')
        dim1_table.write_csv(dim1_table_path)
        dim2_table_path = self._compute_abspath('dim2_table.csv')
        dim2_table.write_csv(dim2_table_path)

        return dim1_table.lazy(), dim2_table.lazy()



    def _split_columns_by_type_internal(self, table: pl.DataFrame, query_col_name: str, target_col_name: str, base_features: list[str]) -> tuple[list[str], list[str]]:
        to_exclude = [query_col_name, target_col_name] + base_features
        numeric_cols = table.select(pl.all().exclude(pl.String).exclude(to_exclude)).collect_schema().names()
        non_numeric_cols = table.select(pl.all().exclude(numeric_cols + to_exclude)).collect_schema().names()

        return numeric_cols, non_numeric_cols
    

    def _compute_abspath(self, filename: str) -> str:
        script_path = os.path.realpath(__file__)
        script_dir = os.path.dirname(script_path)
        return os.path.abspath(os.path.join(script_dir, filename))