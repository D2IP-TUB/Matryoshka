import psycopg as pg
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import time


class DB_handler:
    def __init__(self, preferred_db = 'duckdb', main_table = 'main_tokenized', conn_dict = None):
        # index_path = 'baseline_gittables_object.duckdb'
        # self.duckdb_con = duckdb.connect(index_path, read_only=True).cursor()
        self.preferred_db = preferred_db  # vertica / duckdb / postgres
        if preferred_db == 'postgres':
            CONN_INFO_postgres = conn_dict
            self.postgres_con = pg.connect(**CONN_INFO_postgres).cursor()

        # self.preferred_db = 'postgres' # vertica / duckdb / postgres
        self.main_table = main_table
        # if self.preferred_db == 'postgres':
        #     self.main_table = 'gittables_quadrants'
        # elif self.preferred_db == 'vertica':
        #     self.main_table = 'main_tokenized'

    def create_sql_where_condition_from_value_list(self, values):
        values = [str(x).replace('\'', '') for x in values]
        return "'{}'".format("' , '".join(set(values)))

    def clean_query(self, query):
        return query.replace('AllTables_quadrants', f'{self.main_table}').replace('AllTables', f'{self.main_table}')\
            .replace('TableId', 'tableid').replace('ColumnId', 'colid').replace('RowId', 'rowid')\
            .replace('CellValue', 'tokenized').replace('Quadrants', 'quadrants_webtables').replace('QDR', 'quadrant')\
            .replace('COCOA', 'order_index').replace('MATE', 'MATE_index')

    def execute_and_fetchall(self, query):
        """Returns results, execution time, and fetch time"""
        query = self.clean_query(query)
        # print(query)
        if self.preferred_db == 'vertica':
            start_time = time.time()
            self.vertica_con.execute(query)
            execution_time = time.time() - start_time
            start_time = time.time()
            results = self.vertica_con.fetchall()
            fetch_time = time.time() - start_time
            return results, execution_time, fetch_time
        elif self.preferred_db == 'duckdb':
            start_time = time.time()
            self.duckdb_con.execute(query)
            execution_time = time.time() - start_time
            start_time = time.time()
            results = self.duckdb_con.fetchall()
            fetch_time = time.time() - start_time
            return results, execution_time, fetch_time
        elif self.preferred_db == 'postgres':
            start_time = time.time()
            self.postgres_con.execute(query)
            execution_time = time.time() - start_time
            start_time = time.time()
            results = self.postgres_con.fetchall()
            fetch_time = time.time() - start_time
            return results, execution_time, fetch_time
        else:
            print('The DB index is not implemented')

    def execute_and_fetch_df(self, query, column_name_list):
        if self.preferred_db == 'vertica':
            query = self.clean_query(query)
            return pd.DataFrame(self.vertica_con.execute(query).fetchall(), columns=column_name_list)
        elif self.preferred_db == 'duckdb':
            query = self.clean_query(query)
            return pd.DataFrame(self.duckdb_con.execute(query).fetchall(), columns=column_name_list)
        elif self.preferred_db == 'postgres':
            query = self.clean_query(query)
            return pd.DataFrame(self.postgres_con.execute(query).fetchall(), columns=column_name_list)
        else:
            print('The DB index is not implemented')

    def execute(self, query):
        if self.preferred_db == 'vertica':
            query = self.clean_query(query)
            return self.vertica_con.execute(query)
        elif self.preferred_db == 'duckdb':
            query = self.clean_query(query)
            return self.duckdb_con.execute(query)
        elif self.preferred_db == 'postgres':
            query = self.clean_query(query)
            return self.postgres_con.execute(query)
        else:
            print('The DB index is not implemented')

    def get_table_dataframe(self, tbl_id, db_name = 'vertica', corpus = 'main_tokenized'):
        tables_fetch_query = f'SELECT tableid, colid, rowid, tokenized FROM {corpus} WHERE tableid = {int(tbl_id)} order by tableid, colid, rowid;'
        if db_name == 'vertica':
            external_tables = pd.DataFrame(self.vertica_con.execute(tables_fetch_query).fetchall(), columns=['tableid', 'colid', 'rowid', 'tokenized'])
        else:
            external_tables = pd.DataFrame(self.postgres_con.execute(tables_fetch_query).fetchall(), columns=['tableid', 'colid', 'rowid', 'tokenized'])
        grouped_tables = external_tables.sort_values(by=['tableid', 'rowid', 'colid']).groupby(
            ['tableid', 'rowid']).tokenized.apply(list).reset_index()
        tbl_condition = grouped_tables['tableid'] == int(tbl_id)
        tbl_dict = grouped_tables[tbl_condition].drop(['tableid'], axis=1).set_index('rowid').to_dict()['tokenized']
        df_tbl = pd.DataFrame.from_dict(tbl_dict, orient='index')
        df_tbl.columns = [str(x) for x in np.arange(len(df_tbl.columns.values))]
        return df_tbl

    def create_benchmark_dataset(self, corpus, number_of_tables):
        max_table_id = int(self.postgres_con.execute(f'SELECT MAX(tableid) FROM {corpus}').fetchall()[0][0])
        for i in tqdm(np.arange(number_of_tables)):
            rand_table_id = random.randint(0, max_table_id)
            self.get_table_dataframe(rand_table_id, corpus).to_csv(f'../dataset/benchmark/{corpus}/{i}.csv', index=False)

    def create_benchmark_columns(self, corpus, number_of_columns):
        if corpus in ['main_tokenized']:
            max_table_id = int(self.vertica_con.execute(f'SELECT MAX(tableid) FROM open_data_main_tokenized').fetchall()[0][0])
            for i in tqdm(np.arange(number_of_columns)):
                rand_table = random.randint(0, max_table_id)
                try:
                    rand_col = int(self.vertica_con.execute(f'select colid from main_tokenized where tableid = {rand_table} order by RANDOM() LIMIT 1;').fetchone()[0])
                except:
                    continue
                col_tokens = [x[0] for x in self.vertica_con.execute(f'select distinct tokenized from open_data_main_tokenized where tableid = {rand_table} and colid = {rand_col} and tokenized != \'\' and tokenized != \' \';').fetchall()]
                joint_tokens = ','.join(col_tokens)
                if len(col_tokens) > 0:
                    with open(f"../dataset/benchmark/{corpus}/query.inp", "a") as f:
                        f.write(f'{len(col_tokens)}\t{joint_tokens}' + "\n")
        elif corpus in ['gittables_main_tokenized']:
            max_table_id = int(self.postgres_con.execute(f'SELECT MAX(tableid) FROM open_data_main_tokenized').fetchall()[0][0])
            for i in tqdm(np.arange(number_of_columns)):
                rand_table = random.randint(0, max_table_id)
                try:
                    rand_col = int(self.postgres_con.execute(f'select colid from main_tokenized where tableid = {rand_table} order by RANDOM() LIMIT 1;').fetchone()[0])
                except:
                    continue
                col_tokens = [x[0] for x in self.postgres_con.execute(f'select distinct tokenized from open_data_main_tokenized where tableid = {rand_table} and colid = {rand_col} and tokenized != \'\' and tokenized != \' \';').fetchall()]
                joint_tokens = ','.join(col_tokens)
                if len(col_tokens) > 0:
                    with open(f"../dataset/benchmark/{corpus}/query.inp", "a") as f:
                        f.write(f'{len(col_tokens)}\t{joint_tokens}' + "\n")

    def create_benchmark_numerical_columns(self, corpus, number_of_columns):
        if corpus in ['main_tokenized']:
            table_col_ids = random.sample(self.vertica_con.execute(f'SELECT distinct tableid, colid FROM main_tokenized_quadrants where quadrant is not null ORDER BY colid limit {number_of_columns*100};').fetchall(), number_of_columns)
            for i in tqdm(table_col_ids):
                table = i[0]
                col = i[1]
                col_tokens = [x[0] for x in self.vertica_con.execute(f'select distinct tokenized from main_tokenized_quadrants where tableid = {table} and colid = {col} and tokenized != \'\' and tokenized != \' \';').fetchall()]
                joint_tokens = ','.join(col_tokens)
                if len(col_tokens) > 0:
                    with open(f"../dataset/benchmark/{corpus}/numerical_query.inp", "a") as f:
                        f.write(f'{len(col_tokens)}\t{joint_tokens}' + "\n")
        elif corpus in ['gittables_main_tokenized']:
            table_col_ids = random.sample(self.postgres_con.execute(
                f'SELECT distinct tableid, colid FROM gittables_quadrants where quadrant is not null ORDER BY colid limit {number_of_columns * 100};').fetchall(),
                                          number_of_columns)
            for i in tqdm(table_col_ids):
                table = i[0]
                col = i[1]
                col_tokens = [x[0] for x in self.postgres_con.execute(
                    f'select distinct tokenized from gittables_quadrants where tableid = {table} and colid = {col} and tokenized != \'\' and tokenized != \' \';').fetchall()]
                joint_tokens = ','.join(col_tokens)
                if len(col_tokens) > 0:
                    with open(f"../dataset/benchmark/{corpus}/numerical_query.inp", "a") as f:
                        f.write(f'{len(col_tokens)}\t{joint_tokens}' + "\n")

    def findValidTable_dxf(self, left, right):
        left_example_joint = self.create_sql_where_condition_from_value_list(left)
        right_example_joint = self.create_sql_where_condition_from_value_list(right)

        qString = 'SELECT colX.tableid, colX.colid, colX.rowid, colY.tableid, colY.colid, colY.rowid, colX.tokenized, colY.tokenized \
                      FROM \
                      (SELECT tableid, colid, rowid, tokenized FROM main_tokenized WHERE tokenized in ({})) AS colX JOIN \
                      (SELECT tableid, colid, rowid, tokenized FROM main_tokenized WHERE tokenized in ({})) AS colY \
                       ON colX.tableid = colY.tableid AND colX.rowid = colY.rowid\
                      WHERE \
                      colX.colid <> colY.colid'

        qString = qString.format(left_example_joint, right_example_joint)

        if self.preferred_db == 'postgres':
            cur = self.postgres_con
        elif self.preferred_db == 'vertica':
            cur = self.vertica_con
        elif self.preferred_db == 'duckdb':
            cur = self.duckdb_con
        start_time = time.time()
        cur.execute(qString)
        execution_time = time.time() - start_time
        res = list()
        start_time = time.time()
        fetched_results = cur.fetchall()
        fetch_time = time.time() - start_time
        for i in fetched_results:
            res.append(i)
        return res, execution_time, fetch_time

    def get_Column_Content_dxf(self, tableid, colid):
        qString = f'SELECT tokenized FROM main_tokenized WHERE tableid = {tableid} AND colid = {colid}';

        if self.preferred_db == 'postgres':
            cur = self.postgres_con
        elif self.preferred_db == 'vertica':
            cur = self.vertica_con
        elif self.preferred_db == 'duckdb':
            cur = self.duckdb_con
        start_time = time.time()
        cur.execute(qString)
        execution_time = time.time() - start_time
        res = list()
        start_time = time.time()
        fetched_results = cur.fetchall()
        fetch_time = time.time() - start_time
        for i in fetched_results:
            res.append(i)
        return res, execution_time, fetch_time

    def find_dxf_query(self, left_example, right_example):

        left_example_joint = self.create_sql_where_condition_from_value_list(left_example)
        right_example_joint = self.create_sql_where_condition_from_value_list(right_example)
        qString = 'SELECT colX.tableid, colX.colid, colY.tableid, colY.colid \
                                      FROM \
                                      (SELECT tableid, colid FROM main_tokenized WHERE tokenized in ({}) GROUP BY tableid, colid HAVING COUNT(DISTINCT tokenized) >= 2) AS colX, \
                                      (SELECT tableid, colid FROM main_tokenized WHERE tokenized in ({}) GROUP BY tableid, colid HAVING COUNT(DISTINCT tokenized) >= 2) AS colY \
                                      WHERE \
                                      colX.tableid = colY.tableid AND colX.colid <> colY.colid'

        qString = qString.format(left_example_joint, right_example_joint)
        start_time = time.time()
        if self.preferred_db == 'postgres':
            self.postgres_con.execute(qString)
        elif self.preferred_db == 'vertica':
            self.vertica_con.execute(qString)
        elif self.preferred_db == 'duckdb':
            self.duckdb_con.execute(qString)
        execution_time = time.time()-start_time
        start_time = time.time()
        if self.preferred_db == 'postgres':
            results = self.postgres_con.fetchall()
        elif self.preferred_db == 'vertica':
            results = self.vertica_con.fetchall()
        elif self.preferred_db == 'duckdb':
            results = self.duckdb_con.fetchall()
        fetch_time = time.time() - start_time
        return results, execution_time, fetch_time

    def find_multi_column_dxf_query(self, left_examples, right_example):
        where_condition_list = []
        for col_name in left_examples.columns.values:
            left_example_joint = self.create_sql_where_condition_from_value_list(left_examples[col_name])
            where_condition_list += [left_example_joint]
        right_example_joint = self.create_sql_where_condition_from_value_list(right_example)

        qString = 'SELECT colX1.tableid, colX1.colid, OTHER_SELECT colY.tableid, colY.colid ' \
                  'FROM ' \
                  '(SELECT tableid, colid FROM main_tokenized WHERE tokenized in ({}) GROUP BY tableid, colid HAVING COUNT(DISTINCT tokenized) >= 2) AS colX1, ' \
                  'OTHER_JOIN ' \
                  '(SELECT tableid, colid FROM main_tokenized WHERE tokenized in ({}) GROUP BY tableid, colid HAVING COUNT(DISTINCT tokenized) >= 2) AS colY ' \
                  'WHERE ' \
                  'colX1.tableid = colY.tableid ' \
                  'OTHER_WHERE_tableid ' \
                  'AND colX1.colid <> colY.colid' \
                  'OTHER_WHERE_colid'

        column_counter = 2
        for X_column_name in left_examples.columns.values[1:]:
            qString.replace('OTHER_SELECT', f'colX{column_counter}.tableid, colX{column_counter}.colid, OTHER_SELECT')
            qString.replace('OTHER_JOIN', f'(SELECT tableid, colid FROM main_tokenized WHERE tokenized in {where_condition_list[column_counter-1]} GROUP BY tableid, colid HAVING COUNT(DISTINCT tokenized) >= 2) AS colX{column_counter},  OTHER_JOIN')
            qString.replace('OTHER_WHERE_tableid', f'AND colX1.tableid = colX{column_counter}.tableid OTHER_WHERE_tableid')
            qString.replace('OTHER_WHERE_colid', f'colX1.colid <> colX{column_counter}.colid AND colX{column_counter}.colid <> colY.colid OTHER_WHERE_colid')
            column_counter += 1
        qString = qString.replace('OTHER_SELECT', '').replace('OTHER_JOIN', '').replace('OTHER_WHERE_tableid', '').replace('OTHER_WHERE_colid', '')

        qString = qString.format(where_condition_list[0], right_example_joint)
        start_time = time.time()
        self.postgres_con.execute(qString)
        execution_time = time.time()-start_time
        start_time = time.time()
        results = self.postgres_con.fetchall()
        fetch_time = time.time() - start_time
        return results, execution_time, fetch_time

    def findValidTable_multi_column_dxf(self, lefts, right):
        where_condition_list = []
        for col_name in lefts.columns.values:
            left_example_joint = self.create_sql_where_condition_from_value_list(lefts[col_name])
            where_condition_list += [left_example_joint]
        right_example_joint = self.create_sql_where_condition_from_value_list(right)

        qString = 'SELECT colX1.tableid, colX1.colid, colX1.rowid, colY.tableid, colY.colid, colY.rowid, colX1.tokenized, colY.tokenized, OTHER_SELECT ' \
                  'FROM ' \
                  '(SELECT tableid, colid, rowid, tokenized FROM main_tokenized WHERE tokenized in ({})) AS colX1 JOIN ' \
                  '(SELECT tableid, colid, rowid, tokenized FROM main_tokenized WHERE tokenized in ({})) AS colY ' \
                  'ON colX1.tableid = colY.tableid AND colX1.rowid = colY.rowid ' \
                  'OTHER_JOIN ' \
                  'WHERE ' \
                  'colX1.colid <> colY.colid ' \
                  'OTHER_WHERE'

        column_counter = 2
        for X_column_name in lefts.columns.values[1:]:
            qString.replace('OTHER_SELECT', f'colX{column_counter}.tableid, colX{column_counter}.colid, colX{column_counter}.rowid, colX{column_counter}.tokenized, OTHER_SELECT')
            qString.replace('OTHER_JOIN', f' JOIN (SELECT tableid, colid, rowid, tokenized FROM main_tokenized WHERE tokenized in {where_condition_list[column_counter-1]} AS colX{column_counter} ON colX{column_counter}.tableid = colY.tableid AND colX{column_counter}.rowid = colY.rowid ,  OTHER_JOIN')
            qString.replace('OTHER_WHERE', f'AND colX{column_counter}.colid <> colY.colid OTHER_WHERE')
            column_counter += 1
        qstring = qString.replace('OTHER_SELECT', '').replace('OTHER_JOIN', '').replace('OTHER_WHERE', '')

        qString = qString.format(where_condition_list[0], right_example_joint)

        cur = self.vertica_con.cursor()
        start_time = time.time()
        cur.execute(qString)
        execution_time = time.time() - start_time
        res = list()
        start_time = time.time()
        fetched_results = cur.fetchall()
        fetch_time = time.time() - start_time
        for i in fetched_results:
            res.append(i)
        return res, execution_time, fetch_time

    def get_table_from_index(self, table_id: int) -> pd.DataFrame:
        sql = f"""
        SELECT CellValue, ColumnId, RowId
        FROM AllTables
        WHERE TableId = {table_id}
        """

        results = self.execute_and_fetchall(sql)[0]
        df = pd.DataFrame(results, columns=['CellValue', 'ColumnId', 'RowId'])
        df = df.pivot(index='RowId', columns='ColumnId', values='CellValue')
        df.index.name = None
        df.columns.name = None

        return df
