import os
import pgpq
import re
import time
import numpy as np
import polars as pl
import pyarrow as pa
import ray
from augmentation.utils.common import process_key
from augmentation.utils.database.query_processing import DBHandlerCocoa
from augmentation.utils.lakes.archives import TarArchiveLoader, ZipArchiveLoader
from augmentation.utils.lakes.multiprocessing_utils import TableProcessor, RayTableProcessor
from augmentation.utils.logging.logger_config import setup_logger
from cocoa_system.DataAugmentation import create_index
from psycopg_pool import ConnectionPool
from ray.experimental.tqdm_ray import tqdm as tqdm_ray


class CocoaIndex(DBHandlerCocoa):
    def __init__(self, data_dir: str = None, main_tokenized_table_name: str = None, ordered_index_table_name: str = None, distinct_tokens_table_name: str = None, max_column_table_name: str = None, tunnel: bool = False, max_workers: int = 1, batch_size: int = 1) -> None:
        self.data_dir = data_dir
        super().__init__(main_tokenized_table_name=main_tokenized_table_name, ordered_index_table_name=ordered_index_table_name, distinct_tokens_table_name=distinct_tokens_table_name, max_column_table_name=max_column_table_name, tunnel=tunnel)
        timestamp = time.asctime(time.localtime()).replace(' ', '_').replace(':', '_')
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        log_dir = os.path.join(script_dir, 'utils', 'logging', 'logs')
        self.logger = setup_logger(name='exhaustive_index', log_dir=log_dir, log_file=f'exhaustive_index_{timestamp}.log', silent=False)
        self.max_workers = max_workers
        self.batch_size = batch_size


    def index_lake(self, table_index: int = None, from_tar_archive: bool = False, from_checkpoint: bool = False, checkpoint: str = None) -> None:
        '''
        Indexes all the tables in the data lake and stores the inverted index in the database. \n
        Wrapper for the `index_table` method.
        '''
        self.logger.info('Starting process')
        conn = self.db_connect_pool()
        self.create_tables(conn)
        conn.closeall()

        if from_tar_archive:
            loader = TarArchiveLoader()
        else:
            loader = ZipArchiveLoader()

        checkpoint_name = checkpoint if from_checkpoint else None
        all_tables = list(loader.load_tables(self.data_dir, self.logger, checkpoint_name))
      
        batch_size = self.max_workers if self.batch_size is None else self.batch_size
        batches = [all_tables[i:i + batch_size] for i in range(0, len(all_tables), batch_size)]
        for tables in batches:
            if self.max_workers > 1:
                # materialize tables to DataFrames for parallel processing
                self.logger.info("Materializing tables for parallel processing...")
                table_data = []
                for i, (name, lazy_table) in enumerate(tables):
                    df = lazy_table.collect()
                    table_data.append((name, df, table_index + i))
            else:
                # keep as lazy frames for sequential processing
                table_data = [(name, table, table_index + i) for i, (name, table) in enumerate(tables)]

            if self.max_workers > 1:
                if not ray.is_initialized():
                    ray.init()
                processors = [RayTableProcessor.remote(cocoa_index_table_func=self.index_table) for _ in range(self.max_workers)]
                futures = []
                for i, data in enumerate(table_data):
                    processor = processors[i % len(processors)]
                    future = processor.process_table_cocoa.remote(data)
                    futures.append(future)

                results = []
                index_data = []
                for future in tqdm_ray(futures, desc='Processing tables'):
                    result = ray.get(future)
                    results.append(result[1:])

                    data, _, _, log_message = result
                    for line in log_message.split('\n'):
                        if 'Error' in line:
                            self.logger.error(line)
                        else:
                            self.logger.info(line)
                    index_data.extend(data)
                results = [(idx, success) for idx, success, _ in results]
            else:
                processor = TableProcessor(cocoa_index_table_func=self.index_table)
                results = []
                index_data = []
                for data in table_data:
                    result = processor.process_table_cocoa(data)
                    d, table_index, success, log_message = result
                    for line in log_message.split('\n'):
                        if 'Error' in line:
                            self.logger.error(line)
                        else:
                            self.logger.info(line)
                    results.append((table_index, success))
                    index_data.extend(d)

            if len(index_data) == 0:
                table_index += 1
                continue

            self.write_to_db(index_data, self.main_tokenized_table_name)
            self.write_to_db(index_data, self.ordered_index_table_name)

            successful = sum(1 for _, success in results if success)
            failed = len(results) - successful
            self.logger.info(f'Finished indexing tables: {successful} successful, {failed} failed')
            table_index = max(idx for idx, _ in results) + 1

        self.logger.info('Finished indexing tables in the folder')
        self.logger.handlers.clear()
        if ray.is_initialized():
            ray.shutdown()

        return table_index


    def index_table(self, table: pl.LazyFrame, table_index: int, key_columns: list[str] = None) -> None:
        # normalize column names so that further prefixes and suffixes logic works
        new_colnames = [process_key(name) for name in table.collect_schema().keys()]
        duplicate_cols_indices = [i for i, name in enumerate(new_colnames) if new_colnames.count(name) > 1]
        for i in duplicate_cols_indices:
            new_colnames[i] += f'{i}v2'
        table = table.rename({name: new_col for name, new_col in zip(table.collect_schema().keys(), new_colnames)}).collect().lazy()
        if not key_columns:
            key_columns = self._prune_features(table)
            if len(key_columns) == 0:
                return [()]
            table = table.select(key_columns).collect().lazy()

        group_col_index = 0
        index_data = []
        for group_col in key_columns:
            feature_selection_index = self._index_features(table, group_col, group_col_index, table_index)
            group_col_index += 1
            if len(feature_selection_index) > 0:
                index_data.append(feature_selection_index)

        return index_data
    

    def write_to_db(self, index_data: list[tuple], pg_table_name: str) -> None:
        match pg_table_name:
            case self.ordered_index_table_name:
                table_col_index = []
                is_numeric = []
                min_index = []
                final_order_list = []
                final_binary_list = []
                for item in index_data:
                    if len(item) != 0:
                        table_col_index.append(item[0])
                        is_numeric.append(item[1])
                        min_index.append(item[2])
                        final_order_list.append(item[3])
                        final_binary_list.append(item[4])

                df = pl.DataFrame(
                    {
                        'table_col_index': table_col_index,
                        'is_numeric': is_numeric,
                        'min_index': min_index,
                        'final_order_list': final_order_list,
                        'final_binary_list': final_binary_list
                    }
                )
                if df.height > 0:
                    conn_pool = ConnectionPool(self.conninfo, min_size=8, max_size=32, timeout=1800)
                    conn_pool.open()
                    try:
                        for batch in self._preprocess_dataframe(df, pa.schema([
                            pa.field('table_col_index', pa.string()),
                            pa.field('is_numeric', pa.bool_()),
                            pa.field('min_index', pa.int32()),
                            pa.field('final_order_list', pa.string()),
                            pa.field('final_binary_list', pa.string())
                        ])):
                            self._write_batch_to_pg(batch, conn_pool, pg_table_name)
                    finally:
                        conn_pool.close()
            case self.main_tokenized_table_name:
                data = [item[5] for item in index_data if len(item) != 0]
                if len(data) > 0:
                    all_tokenized = pl.concat(data, how='vertical')
                    conn_pool = ConnectionPool(self.conninfo, min_size=8, max_size=32, timeout=1800)
                    conn_pool.open()
                    try:
                        for batch in self._preprocess_dataframe(all_tokenized, pa.schema([
                            pa.field('tokenized', pa.string()),
                            pa.field('tableid', pa.int32()),
                            pa.field('rowid', pa.int32()),
                            pa.field('table_col_id', pa.string())
                        ])):
                            self._write_batch_to_pg(batch, conn_pool, pg_table_name)
                    finally:
                        conn_pool.close()

    
    def _preprocess_dataframe(self, df: pl.DataFrame, target_schema: pa.Schema) -> pa.RecordBatch:
        batch = df.to_arrow().to_batches()[0]
        yield batch.cast(target_schema)


    def _write_batch_to_pg(self, batch: pa.RecordBatch, pool: ConnectionPool, pg_table_name: str) -> None:
        all_cols = batch.schema.names
        field_encoders = {
            col: pgpq.ArrowToPostgresBinaryEncoder.infer_encoder(batch.field(col))
            for col in all_cols
        }
        encoder = pgpq.ArrowToPostgresBinaryEncoder.new_with_encoders(batch.schema, field_encoders)
        batches = pa.Table.from_batches([batch]).combine_chunks().to_batches()

        with pool.connection() as conn:
            with conn.cursor() as cursor:
                with cursor.copy(f'COPY {pg_table_name} FROM STDIN WITH (FORMAT BINARY)') as copy:
                    copy.write(encoder.write_header())
                    for b in batches:
                        copy.write(encoder.write_batch(b))
                    copy.write(encoder.finish())

    
    def _index_features(self, table: pl.LazyFrame, group_col: str, group_col_index: int, table_index: int, n: int = 10_000):
        df = table.select(group_col).drop_nulls().collect()
        try:
            df = df.drop_nans()
        except Exception:
            pass
        if df.height <= n:
            arr = df.to_numpy()
        else:
            value_counts = df.select(pl.col(group_col).value_counts(normalize=True).struct.unnest())
            df = df.join(value_counts, on=group_col)
            df = df.with_row_index("idx")
            df_sampled = (
                df
                    .with_columns(p=pl.col("proportion") / pl.col("proportion").sum())
                    .group_by(group_col)
                    .agg([
                        pl.col("idx").sample(
                            fraction=pl.col("p").first() * n,
                            with_replacement=True, seed=1
                        )
                    ])
                    .explode("idx")
                    .drop_nulls("idx")
            )
            samples = pl.concat(
                [df_sampled, df.select(pl.all().exclude([group_col, "idx"]).gather(df_sampled["idx"]))],
                how="horizontal"
            )
            df = samples.select(group_col)
            arr = df.to_numpy()
        try:
            min_index, final_order_list, final_binary_list = create_index(arr)
            is_numeric = np.issubdtype(np.array(final_order_list[1]).dtype, np.number)
            table_col_index = f'{table_index}_{group_col_index}'
            final_order_list_str = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", str(final_order_list)).replace(' ', '')
            final_binary_list_str = re.sub(r"np\.str_\('([^']*)'\)", r"'\1'", str(final_binary_list)).replace(' ', '')
        except Exception as e:
            return tuple()

        main_tokenized = df.select(
            pl.col(group_col)
            .cast(pl.String)
            .alias("tokenized")
        ).with_row_index("rowid").with_columns(pl.col("rowid").cast(pl.Int32))
        main_tokenized = main_tokenized.with_columns([
            pl.lit(table_index).alias('tableid').cast(pl.Int32),
            pl.lit(table_col_index).alias('table_col_id').cast(pl.Utf8)
        ])
        main_tokenized = main_tokenized.select([
            "tokenized",
            "tableid",
            "rowid",
            "table_col_id"
        ])
        
        return table_col_index, is_numeric, min_index, final_order_list_str, final_binary_list_str, main_tokenized
    

    def _prune_features(
        self,
        df: pl.LazyFrame,
        sample_size: int = 1000,
        null_threshold: float = 0.8,
        high_cardinality_threshold: float = 0.8,
        constant_threshold: float = 0.99,
        boolean_skew_threshold: float = 0.01,
        max_text_avg_len: int = 100,
    ) -> list[str]:
        n_rows = df.select(pl.len()).collect().item()
        if n_rows > sample_size:
            df = df.collect().sample(n=sample_size)
        else:
            df = df.collect()
        
        total = df.height
        if total == 0:
            return []

        # --- build expressions for stats ---
        exprs = []
        for c in df.columns:
            s = df[c]
            if s.dtype in (pl.Binary, pl.Object):
                continue  # skip early
            exprs.extend([
                pl.col(c).null_count().alias(f"{c}_nulls"),
                pl.col(c).n_unique().alias(f"{c}_uniq"),
                pl.col(c).drop_nulls().value_counts().struct[1].max().alias(f"{c}_top")
            ])
            # text avg len
            if s.dtype == pl.Utf8:
                exprs.append(
                    pl.col(c).drop_nulls().str.len_chars().mean().alias(f"{c}_avglen")
                )
            # boolean counts
            if s.dtype == pl.Boolean:
                exprs.append(
                    pl.col(c).drop_nulls().value_counts().struct[1].min().alias(f"{c}_boolmin")
                )

        try:
            stats = df.select(exprs)
        except Exception as e:
            return []

        # --- evaluate heuristics ---
        promising = []
        for c in df.columns:
            s = df[c]
            if s.dtype in (pl.Binary, pl.Object):
                continue

            nulls = stats[f"{c}_nulls"][0] if f"{c}_nulls" in stats.columns else 0
            non_null = total - nulls
            if non_null == 0 or nulls / total > null_threshold:
                continue

            uniq = stats[f"{c}_uniq"][0]
            if uniq / non_null > high_cardinality_threshold:
                continue

            top = stats[f"{c}_top"][0]
            if top / non_null > constant_threshold:
                continue

            if s.dtype == pl.Utf8 and f"{c}_avglen" in stats.columns:
                avglen = stats[f"{c}_avglen"][0]
                if avglen is not None and avglen > max_text_avg_len:
                    continue

            if s.dtype == pl.Boolean and f"{c}_boolmin" in stats.columns:
                boolmin = stats[f"{c}_boolmin"][0]
                if boolmin / non_null < boolean_skew_threshold:
                    continue

            promising.append(c)

        return promising


if __name__ == '__main__':
    import sys
    data_folder = sys.argv[1]
    lake_name = sys.argv[2]
    from_tar = sys.argv[3].lower() == 'true'
    data_dirs = [os.path.join(data_folder, d) for d in os.listdir(data_folder)]
    table_index = 0
    for data_dir in data_dirs:
        worker = CocoaIndex(
            data_dir=data_dir,
            main_tokenized_table_name=f'{lake_name}_cocoa_main_tokenized',
            ordered_index_table_name=f'{lake_name}_cocoa_ordered_index',
            distinct_tokens_table_name=f'{lake_name}_cocoa_distinct_tokens',
            max_column_table_name=f'{lake_name}_cocoa_max_column',
            tunnel=False,
            max_workers=1,
            batch_size=1
        )
        cur_table_index = worker.index_lake(table_index, from_tar_archive=from_tar, from_checkpoint=False, checkpoint=None)
        table_index = cur_table_index + 1