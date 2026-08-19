import time
from typing import Any, Dict, Tuple

import polars as pl
import ray

from .archives import open_source


class TableProcessor:
    def __init__(self, index_table_func = None, cocoa_index_table_func = None, qcr_index_table_func = None, max_cols: int = 100):
        self.index_table_func = index_table_func
        self.cocoa_index_table_func = cocoa_index_table_func
        self.qcr_index_table_func = qcr_index_table_func
        self.max_cols = max_cols


    def process_table(self, table_data: Tuple[str, pl.LazyFrame, int]) -> Tuple[int, bool]:
        table_name, table, table_index = table_data
        if not isinstance(table, pl.LazyFrame):
            # parallel processing with Ray requires DataFrame but further processing is done with LazyFrame
            table = table.lazy()

        log_messages = []
        log_messages.append(f'Processing table {table_index} - {table_name}')

        try:
            n_rows = table.select(pl.len()).collect().item()
            n_cols = len(table.collect_schema())
            if n_rows > 1_000_000:
                log_messages.append(f'Table {table_name} has {n_rows} rows, sampling 1_000_000 rows')
                table = self._null_aware_sample(table, 1_000_000)
            if n_cols > self.max_cols:
                log_messages.append(
                    f'Skipping table {table_index} - {table_name}: n_cols={n_cols} > max_cols={self.max_cols}'
                )
                index_data = []
                return index_data, table_index, False, '\n'.join(log_messages)
            log_messages.append(f'Table {table_name} has {n_rows} rows and {n_cols} columns')

            compute_start = time.perf_counter()
            index_data = self.index_table_func(table, table_index, table_name)
            compute_elapsed = time.perf_counter() - compute_start
            log_messages.append(f'Table {table_index} compute_time_sec={compute_elapsed:.3f}')
            log_messages.append(f'Processed table {table_index} - {table_name}')
            return index_data, table_index, True, '\n'.join(log_messages)

        except Exception as e:
            index_data = []
            error_msg = f'Error processing table {table_index} - {table_name}: {e}'
            log_messages.append(error_msg)
            table_name_clean = table_name.split('/')[-1].split('.')[0] # save failed table for debugging
            table.collect().write_csv(f'/app/data/{table_name_clean}.tsv', separator='\t')
            with open('/app/data/last_table_index.txt', 'w') as f:
                f.write(str(table_index))
            return index_data, table_index, False, '\n'.join(log_messages)


    def _null_aware_sample(self, table: pl.LazyFrame, n: int) -> pl.LazyFrame:
        null_counts = table.with_columns(pl.sum_horizontal(pl.all().is_null()).alias('null_count')).collect()
        null_counts = null_counts.sort('null_count')
        top_k = null_counts.head(n * 2)
        sample = top_k.sample(n, with_replacement=False, seed=42).lazy()
        return sample


    def process_table_cocoa(self, table_data: Tuple[str, pl.LazyFrame, int]) -> Tuple[int, bool]:
        table_name, table, table_index = table_data
        if not isinstance(table, pl.LazyFrame):
            # parallel processing with Ray requires DataFrame but further processing is done with LazyFrame
            table = table.lazy()

        log_messages = []
        log_messages.append(f'Processing table {table_index} - {table_name}')
        try:
            index_data = self.cocoa_index_table_func(table, table_index)
            log_messages.append(f'Processed table {table_index} - {table_name}')
            return index_data, table_index, True, '\n'.join(log_messages)

        except Exception as e:
            index_data = []
            error_msg = f'Error processing table {table_index} - {table_name}: {e}'
            log_messages.append(error_msg)
            table_name_clean = table_name.split('/')[-1].split('.')[0] # save failed table for debugging
            table.collect().write_csv(f'/app/data/{table_name_clean}.tsv', separator='\t')
            with open('/app/data/last_table_index.txt', 'w') as f:
                f.write(str(table_index))
            return index_data, table_index, False, '\n'.join(log_messages)


    def process_table_qcr(self, table_data: Tuple[str, pl.LazyFrame, int]) -> Tuple[int, bool]:
        table_name, table, table_index = table_data
        if not isinstance(table, pl.LazyFrame):
            # parallel processing with Ray requires DataFrame but further processing is done with LazyFrame
            table = table.lazy()

        log_messages = []
        log_messages.append(f'Processing table {table_index} - {table_name}')
        try:
            index_data = self.qcr_index_table_func(table, table_name)
            log_messages.append(f'Processed table {table_index} - {table_name}')
            return index_data, table_index, True, '\n'.join(log_messages)

        except Exception as e:
            index_data = []
            error_msg = f'Error processing table {table_index} - {table_name}: {e}'
            log_messages.append(error_msg)
            table_name_clean = table_name.split('/')[-1].split('.')[0] # save failed table for debugging
            table.collect().write_csv(f'/app/data/{table_name_clean}.tsv', separator='\t')
            with open('/app/data/last_table_index.txt', 'w') as f:
                f.write(str(table_index))
            return index_data, table_index, False, '\n'.join(log_messages)


@ray.remote
class RayTableProcessor:
    def __init__(self, index_table_func, max_cols: int = 100):
        self.processor = TableProcessor(index_table_func, max_cols=max_cols)

    def process_table(self, table_data: Tuple[str, pl.LazyFrame, int]) -> Tuple[int, bool]:
        return self.processor.process_table(table_data)

    def process_source(self, source_data: Tuple[str, Dict[str, Any], int]) -> Tuple[int, bool]:
        table_name, source_spec, table_index = source_data
        try:
            table = open_source(source_spec)
        except Exception as e:
            log = f'Error opening source {table_name}: {e}'
            return [], table_index, False, log
        return self.processor.process_table((table_name, table, table_index))

    def process_table_cocoa(self, table_data: Tuple[str, pl.LazyFrame, int]) -> Tuple[int, bool]:
        return self.processor.process_table_cocoa(table_data)
