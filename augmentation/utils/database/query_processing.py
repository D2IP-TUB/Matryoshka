import os
import psycopg2
import shutil
import sys
import yaml
from functools import partial
from paramiko import RSAKey
from psycopg2 import pool
from sshtunnel import SSHTunnelForwarder
from .string_iterator import StringIteratorIO, clean_csv_value


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
config_path = os.path.join(script_dir, 'db_config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

db_config = config['db']
ssh_config = config['ssh']


class DBHandler:
    def __init__(self, feature_selection_table_name: str, overlap_table_name: str, tunnel: bool = False, work_mem_size: str = '12GB'):
        self.create_fs_table_query =  f'CREATE TABLE {feature_selection_table_name} (' \
                                         'key TEXT,' \
                                         'feature_index INT4,' \
                                         'table_index INT4,' \
                                         'key_col_index INT4,' \
                                         'row_index INT8,' \
                                         'count INT4,' \
                                         'sum BYTEA,' \
                                         'diag BYTEA,' \
                                         'cofactors BYTEA,' \
                                         'shape INT4[],' \
                                         'qcr_term_positive BYTEA,' \
                                         'qcr_term_negative BYTEA' \
                                         ');'

        self.create_overlap_table_query = f'CREATE TABLE {overlap_table_name} (' \
                                         'key TEXT,' \
                                         'table_index INT4,' \
                                         'key_col_index INT4,' \
                                         'row_index INT8' \
                                         ');'

        self.create_index_query = 'CREATE INDEX index_name ON public.table_name USING method (columns);'

        self.btree_index_cols = {
            feature_selection_table_name: ['btree (table_index, key_col_index, row_index)', 'btree (key)']
        }

        self.dbname = db_config['dbname']
        self.user = db_config['user']
        self.password = db_config['password']
        self.host = db_config['host']
        self.port = db_config['port']

        self.ssh_host = ssh_config['host']
        self.ssh_user = ssh_config['user']
        self.ssh_port = ssh_config['port']
        self.ssh_key = ssh_config['key']
        self.ssh_key_pwd = ssh_config['key_password']

        self.tunnel = tunnel
        if tunnel:
            self.tun = SSHTunnelForwarder(
                (self.ssh_host, self.ssh_port),
                ssh_username=self.ssh_user,
                ssh_pkey=RSAKey.from_private_key_file(self.ssh_key, password=self.ssh_key_pwd),
                remote_bind_address=(self.host, self.port),
                local_bind_address=('localhost', 15432)
            )
            self.host = 'localhost'
            self.port = 15432
            self.tun.start()

        self.conninfo = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'


    def db_connect(self):
        conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        
        return conn


    def db_connect_pool(self):
        try:
            conn_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
        except psycopg2.Error as e:
            raise e
        
        return conn_pool


    def create_tables(self, conn_pool):
        try:
            conn = conn_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(self.create_fs_table_query)
                cur.execute(self.create_overlap_table_query)
                conn.commit()
        except psycopg2.errors.DuplicateTable:
            conn.rollback()
        finally:
            conn_pool.putconn(conn)


    def create_overlap_table(self, conn):
        try:
            with conn.cursor() as cur:
                cur.execute(self.create_overlap_table_query)
                conn.commit()
        except psycopg2.errors.DuplicateTable:
            conn.rollback()

    
    def insert_data(self, conn_pool, data: tuple, table_name: str, sep: str = '|', buffer_size: int = 1024):
        try:
            conn = conn_pool.getconn()
            with conn.cursor() as cur:
                if self._check_disk_space():
                    data = iter(data)
                    data_str_iterator = StringIteratorIO((
                        sep.join(map(partial(clean_csv_value, separator=sep), tup)) + '\n'
                        for tup in data
                    ))
                    cur.copy_from(data_str_iterator, table_name, sep='|', size=buffer_size)
                    conn.commit()
        except psycopg2.Error as e:
            raise e
        finally:
            conn_pool.putconn(conn)


    def create_index(self, conn, query: str):
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                conn.commit()
        except psycopg2.errors.DuplicateTable:
            self.conn.rollback()


    def _check_disk_space(self, threshold_gb=10):
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        if free_gb < threshold_gb:
            sys.exit('Stopping script due to low disk space.')
        else:
            return True
        

class DBHandlerCocoa(DBHandler):
    def __init__(self, main_tokenized_table_name: str, ordered_index_table_name: str, distinct_tokens_table_name: str, max_column_table_name: str, tunnel: bool = False):
        self.main_tokenized_table_name = main_tokenized_table_name
        self.ordered_index_table_name = ordered_index_table_name
        self.distinct_tokens_table_name = distinct_tokens_table_name
        self.max_column_table_name = max_column_table_name
        self.tunnel = tunnel
        self.create_main_tokenized_table_query = f'CREATE TABLE {main_tokenized_table_name} (' \
                                                  f'tokenized TEXT, ' \
                                                  f'tableid INT NOT NULL, ' \
                                                  f'rowid INT NOT NULL, ' \
                                                  f'table_col_id TEXT NOT NULL' \
                                                  f');'
        self.create_ordered_index_table_query = f'CREATE TABLE {ordered_index_table_name} (' \
                                                  f'table_col_id TEXT NOT NULL, ' \
                                                  f'is_numeric BOOLEAN, ' \
                                                  f'min_index INT NOT NULL, ' \
                                                  f'order_list TEXT, ' \
                                                  f'binary_list TEXT' \
                                                  f');'
        self.create_distinct_tokens_table_query = f'CREATE TABLE {distinct_tokens_table_name} (' \
                                                  f'tokenized TEXT, ' \
                                                  f'table_col_id TEXT NOT NULL' \
                                                  f');'
        self.create_max_column_table_query = f'CREATE TABLE {max_column_table_name} (' \
                                                  f'tableid INT NOT NULL,' \
                                                  f'max_col_id INT NOT NULL,' \
                                                  f'PRIMARY KEY (tableid)' \
                                                  f');'

        self.dbname = db_config['dbname']
        self.user = db_config['user']
        self.password = db_config['password']
        self.host = db_config['host']
        self.port = db_config['port']

        self.ssh_host = ssh_config['host']
        self.ssh_user = ssh_config['user']
        self.ssh_port = ssh_config['port']
        self.ssh_key = ssh_config['key']
        self.ssh_key_pwd = ssh_config['key_password']

        self.tunnel = tunnel
        if tunnel:
            self.tun = SSHTunnelForwarder(
                (self.ssh_host, self.ssh_port),
                ssh_username=self.ssh_user,
                ssh_pkey=RSAKey.from_private_key_file(self.ssh_key, password=self.ssh_key_pwd),
                remote_bind_address=(self.host, self.port),
                local_bind_address=('localhost', 15432)
            )
            self.host = 'localhost'
            self.port = 15432
            self.tun.start()
        
        self.conninfo = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'
        
    
    def create_tables(self, conn_pool):
        conn = conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(self.create_main_tokenized_table_query)
                cur.execute(self.create_ordered_index_table_query)
                cur.execute(self.create_distinct_tokens_table_query)
                cur.execute(self.create_max_column_table_query)
                conn.commit()
        except psycopg2.errors.DuplicateTable:
            conn.rollback()
        finally:
            conn_pool.putconn(conn)


class DBHandlerQCR(DBHandler):
    def __init__(self, qcr_table_name: str, tunnel: bool = False):
        self.qcr_table_name = qcr_table_name
        self.tunnel = tunnel
        self.create_qcr_table_query = f'CREATE TABLE {qcr_table_name} (' \
                                      f'term TEXT, ' \
                                      f'tableid_catcol_numcol TEXT' \
                                      f');'

        self.dbname = db_config['dbname']
        self.user = db_config['user']
        self.password = db_config['password']
        self.host = db_config['host']
        self.port = db_config['port']

        self.ssh_host = ssh_config['host']
        self.ssh_user = ssh_config['user']
        self.ssh_port = ssh_config['port']
        self.ssh_key = ssh_config['key']
        self.ssh_key_pwd = ssh_config['key_password']

        self.tunnel = tunnel
        if tunnel:
            self.tun = SSHTunnelForwarder(
                (self.ssh_host, self.ssh_port),
                ssh_username=self.ssh_user,
                ssh_pkey=RSAKey.from_private_key_file(self.ssh_key, password=self.ssh_key_pwd),
                remote_bind_address=(self.host, self.port),
                local_bind_address=('localhost', 15432)
            )
            self.host = 'localhost'
            self.port = 15432
            self.tun.start()
        
        self.conninfo = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'
        
    
    def create_tables(self, conn_pool):
        conn = conn_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(self.create_qcr_table_query)
                conn.commit()
        except psycopg2.errors.DuplicateTable:
            conn.rollback()
        finally:
            conn_pool.putconn(conn)