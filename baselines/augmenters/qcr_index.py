import os
import pgpq
import time
import polars as pl
import pyarrow as pa
import ray
from matryoshka.utils.common import process_key
from matryoshka.lakes.archives import TarArchiveLoader, ZipArchiveLoader, DirectoryLoader
from matryoshka.lakes.parallel import TableProcessor, RayTableProcessor
from matryoshka.utils.logging import setup_logger
from psycopg_pool import ConnectionPool
from ray.experimental.tqdm_ray import tqdm as tqdm_ray
import hashlib
import heapq
import pandas as pd
from collections import defaultdict
from typing import List, Callable, Tuple, Union, DefaultDict
from unicodedata import numeric
from baselines.augmenters.db_handlers import DBHandlerQCR


class QcrIndex(DBHandlerQCR):
    def __init__(self, data_dir: str = None, qcr_table_name: str = None, tunnel: bool = False, max_workers: int = 1, batch_size: int = 1) -> None:
        super().__init__(qcr_table_name=qcr_table_name, tunnel=tunnel)
        self.data_dir = data_dir
        self.max_workers = max_workers
        self.batch_size = batch_size
        timestamp = time.asctime(time.localtime()).replace(' ', '_').replace(':', '_')
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        log_dir = os.path.join(script_dir, 'logs')
        self.logger = setup_logger(name='exhaustive_index', log_dir=log_dir, log_file=f'exhaustive_index_{timestamp}.log', silent=False)


    def index_lake(self, table_index: int = None, from_tar_archive: bool = False, from_checkpoint: bool = False, checkpoint: str = None) -> None:
        '''
        Indexes all the tables in the data lake and stores the inverted index in the database. \n
        Wrapper for the `index_table` method.
        '''
        self.logger.info('Starting process')
        conn = self.db_connect_pool()
        self.create_tables(conn)
        conn.closeall()

        if os.path.isdir(self.data_dir):
            loader = DirectoryLoader()
        elif from_tar_archive:
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
                processors = [RayTableProcessor.remote(qcr_index_table_func=self.index_table) for _ in range(self.max_workers)]
                futures = []
                for i, data in enumerate(table_data):
                    processor = processors[i % len(processors)]
                    future = processor.process_table_qcr.remote(data)
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
                processor = TableProcessor(qcr_index_table_func=self.index_table)
                results = []
                index_data = []
                for data in table_data:
                    result = processor.process_table_qcr(data)
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

            self.write_to_db(index_data, self.qcr_table_name)

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
        group_col = ''
        group_col_index = 0
        index_data = []
        feature_selection_index = self._index_features(table, group_col, group_col_index, table_index)
        group_col_index += 1
        if len(feature_selection_index) > 0:
            index_data.append(feature_selection_index)

        return index_data


    def write_to_db(self, index_data: list[pl.DataFrame], pg_table: str = '') -> None:
        if len(index_data) > 0:
            all_tokenized = pl.concat(index_data, how='vertical')
            conn_pool = ConnectionPool(self.conninfo, min_size=8, max_size=32, timeout=1800)
            conn_pool.open()
            try:
                for batch in self._preprocess_dataframe(all_tokenized, pa.schema([
                    pa.field('term', pa.string()),
                    pa.field('tableid_catcol_numcol', pa.string())
                ])):
                    self._write_batch_to_pg(batch, conn_pool, self.qcr_table_name)
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
        df_qcr = self.callback_qcr(table.collect().to_pandas(), table_index)
        df_qcr = pl.from_pandas(df_qcr)

        return df_qcr


    def callback_qcr(self, df_in: pd.DataFrame, table_name: str) -> pd.DataFrame:
        def hash_md5(obj: object) -> int:
            """
            Hashes an object to an integer value
            :param obj: Object that needs to be hashed
            :return: hashed value
            """
            return int.from_bytes(
                hashlib.md5(str(obj).encode("utf-8")).digest(), "big", signed=False
            )
        df_in.columns.name = table_name
        c_col = self.get_kc(df_in)
        n_col = self.get_c(df_in)
        cross_product_tables_list = self.cross_product_tables(c_col, n_col, df_in.columns.name)
        list1, list2 = [], []
        for i in cross_product_tables_list:
            sketch = self.create_sketch(i.iloc[:, 0], i.iloc[:, 1], hash_md5, n=512)
            labels = self.key_labeling(sketch, hash_md5, inner_hash=False)
            list1.extend(labels)
            list2.extend([i.columns.name] * len(labels))
        df_out = pd.DataFrame(zip(list1, list2), columns=['term', 'tableid_catcol_numcol'])
        
        return df_out


    def key_labeling(self, sketch: List[Tuple[str, numeric]], h: Callable[[str], int] = lambda x: int.from_bytes(str(x).encode("utf-8"), "big", signed=False), inner_hash: bool = False) \
            -> List[Union[int, str]]:
        """
        labels keys according to their values' distribution. +key or -key
        :param sketch: table with keys and their values
        :param h: hash function if hashed keys shall be labeled, nothing if literal keys shall be labeled
        :return: returns a two col table of labeled keys and values
        """
        if not sketch:
            return []

        mue = sum([value for key, value in sketch]) / len(sketch)
        return [format(h(f'{f"{h(key):032x}" if inner_hash else key}{"+1" if value > mue else "-1"}'), "032x") for key, value in sketch]


    def create_sketch(
            self,
            kc: List[str],
            c: List[numeric],
            hash_funct: Callable[[str], int],
            n=100
    ) -> List[Tuple[str, numeric]]:
        """
        This function creates a sketch of size n from two columns (one with numeric values, one with categorical values).
        It hashes the categorical column and builds a table (list of tuples) with the hashes and the corresponding values
        from the numerical column. This table is sorted by the hash-column and the rows with the n-smallest hash-values are
        kept for form the sketch
        :param kc: list of categorical keys (key column)
        :param c: list of numeric values (value column)
        :param hash_funct: collision free hash function string -> int/float
        :param n: size of sketch, default 100
        :return: sketch of size n for given columns
        """
        grouped = pd.DataFrame({'kc': kc, 'c': c}).groupby('kc').mean(numeric_only=True).reset_index()
        grouped = grouped.dropna()

        sketch = heapq.nsmallest(n, zip(grouped["kc"], grouped["c"]), key=lambda x: hash_funct(x[0]))
        return sketch


    def cross_product_tables(self, cat_col: DefaultDict[str, List[str]], num_col: DefaultDict[str, List[numeric]],
                            table_id: str) -> List[pd.DataFrame]:
        """
        combines all numerical and categorical columns like a cross-product.
        eg: c1, c2 x n1, n2, n3 -> ['c1_n1', c1_n2', 'c1_n3','c2_n1', 'c2_n2', c2_n3']
        :param cat_col: default dict with column name as key and list of categorical-column-values as value.
        :param num_col: default dict with column name as key and list of numerical-column-values as value.
        :param table_id: name of table, that is to be split
        :return: list of named tables.
        """
        tables = []
        for cat_header in cat_col:
            for num_header in num_col:
                table = pd.DataFrame(list(zip(cat_col[cat_header], num_col[num_header])), columns=[cat_header, num_header])
                table.columns.name = f"{table_id}_|_{cat_header}_|_{num_header}"  # here we use the column names as name for the new table
                tables.append(table)
        return tables


    def get_kc(self, table: pd.DataFrame) -> DefaultDict[str, List[str]]:
        """
        extract categorical columns from dataframe
        :param table: input table
        :return: dict of categorical columns by column name
        """
        kc_column_name = table.select_dtypes(include=["object"]).columns
        columns = defaultdict(List[str])
        for col in kc_column_name:
            columns[col] = (table[col].astype(str).apply(lambda x: x.strip().lower())).values.tolist()
        return columns


    def get_c(self, table: pd.DataFrame) -> DefaultDict[str, List[numeric]]:
        """
        extract numerical columns from dataframe
        :param table: input table
        :return: dict of numerical columns by column name
        """
        c_column_name = table.select_dtypes(include=["float64", "int64"]).columns
        columns = defaultdict(List[str])
        for col in c_column_name:
            columns[col] = (table[col].values.tolist())
        return columns


    def get_table_id(self, table: pd.DataFrame) -> str:
        """
        extract name from pandas dataFrame
        :param table: pandas dataFrame
        :return: name (string)
        """
        return table.columns.name


if __name__ == '__main__':
    import sys
    data_folder = sys.argv[1]
    lake_name = sys.argv[2]
    from_tar = sys.argv[3].lower() == 'true'
    data_dirs = [os.path.join(data_folder, d) for d in os.listdir(data_folder)]
    table_index = 0
    for data_dir in data_dirs:
        worker = QcrIndex(
            data_dir=data_dir,
            qcr_table_name=f'{lake_name}_qcr_index',
            tunnel=False,
            max_workers=1,
            batch_size=1
        )
        cur_table_index = worker.index_lake(table_index, from_tar_archive=from_tar, from_checkpoint=False, checkpoint=None)
        table_index = cur_table_index + 1