"""Schema definition and connection management for the Matryoshka index.

``DBHandler`` owns the DDL for the four tables that make up an index of one
data lake, plus connection helpers shared by the offline indexer
(:class:`matryoshka.index.ExhaustiveIndex`) and the online query side
(:class:`matryoshka.retrieval.JoinDiscovery`):

``<name>``
    the feature-selection index. One row per (lake table, key column, key)
    group, holding the Gram matrix sketch (``sum``, ``diag``, ``cofactors``)
    and the QCR pruning terms.
``<name>_overlap``
    the inverted index from cell value to (table, key column, row).
``<name>_meta``, ``<name>_num_state``, ``<name>_cat_state``
    per-table metadata and raw per-key aggregation state. Written only when
    the index is built with ``update_support=True``, and read by
    ``ExhaustiveIndex.update_table`` and ``find_union_peer``.

Connection parameters are resolved lazily through
:func:`matryoshka.db.settings.resolve_settings`, so importing this module
never touches the environment or the filesystem.
"""
import shutil
import sys
from functools import partial

import psycopg2
from psycopg2 import pool

from .settings import DBSettings, resolve_settings
from .string_iterator import StringIteratorIO, clean_csv_value


class DBHandler:
    def __init__(self, feature_selection_table_name: str, overlap_table_name: str, tunnel: bool = False,
                 work_mem_size: str = '12GB', settings: DBSettings | str | None = None):
        self.create_fs_table_query =  f'CREATE TABLE {feature_selection_table_name} (' \
                                         'key TEXT,' \
                                         'feature_index INT4,' \
                                         'table_index INT4,' \
                                         'table_name TEXT,' \
                                         'key_col_index INT4,' \
                                         'row_index INT8,' \
                                         'count INT4,' \
                                         'sum BYTEA,' \
                                         'diag BYTEA,' \
                                         'cofactors BYTEA,' \
                                         'shape INT4[],' \
                                         'qcr_term_positive BYTEA,' \
                                         'qcr_term_negative BYTEA,' \
                                         'column_headers TEXT[]' \
                                         ');'

        self.create_overlap_table_query = f'CREATE TABLE {overlap_table_name} (' \
                                         'key TEXT,' \
                                         'table_index INT4,' \
                                         'key_col_index INT4,' \
                                         'row_index INT8' \
                                         ');'

        # ---- Index update support: per-table state tables. -------------------------
        # Names are derived from the fs table name so we don't expand the constructor
        # signature; all three live alongside the fs/overlap tables.
        self.tables_meta_table_name = f'{feature_selection_table_name}_meta'
        self.num_state_table_name   = f'{feature_selection_table_name}_num_state'
        self.cat_state_table_name   = f'{feature_selection_table_name}_cat_state'

        self.create_tables_meta_query = (
            f'CREATE TABLE {self.tables_meta_table_name} ('
            'table_index INT4 PRIMARY KEY,'
            'table_name TEXT UNIQUE,'
            'numeric_cols TEXT[],'
            'cat_cols TEXT[],'
            'key_columns TEXT[],'
            'useful_columns TEXT[],'
            'n_rows INT8,'
            'last_updated TIMESTAMP DEFAULT NOW()'
            ');'
        )
        # num_state stores raw aggregates per (table, key_col, key, num_col):
        #   count_/sum_/min_/max_ are scalars used to derive mean/min/max features.
        #   values_ is a packed float64[] of the (sorted) raw values used to derive
        #   median exactly. For very large groups this can be replaced with a
        #   t-digest blob without changing the schema (BYTEA is opaque).
        self.create_num_state_query = (
            f'CREATE TABLE {self.num_state_table_name} ('
            'table_index INT4,'
            'key_col_index INT4,'
            'key TEXT,'
            'col_name TEXT,'
            'count_ INT8,'
            'sum_ FLOAT8,'
            'min_ FLOAT8,'
            'max_ FLOAT8,'
            'values_ BYTEA,'
            'PRIMARY KEY (table_index, key_col_index, key, col_name)'
            ');'
        )
        # cat_state: one row per (table, key_col, key, cat_col, category). N_c is
        # derived at read time as the count of distinct categories per (table,
        # key_col, cat_col).
        self.create_cat_state_query = (
            f'CREATE TABLE {self.cat_state_table_name} ('
            'table_index INT4,'
            'key_col_index INT4,'
            'key TEXT,'
            'col_name TEXT,'
            'category TEXT,'
            'raw_count INT8,'
            'PRIMARY KEY (table_index, key_col_index, key, col_name, category)'
            ');'
        )

        self.create_index_query = 'CREATE INDEX index_name ON public.table_name USING method (columns);'

        self.btree_index_cols = {
            feature_selection_table_name: ['btree (table_index, key_col_index, row_index)', 'btree (key)'],
            self.num_state_table_name:    ['btree (table_index, key_col_index)'],
            self.cat_state_table_name:    ['btree (table_index, key_col_index)', 'btree (table_index, key_col_index, col_name)', 'btree (category)'],
            self.tables_meta_table_name:  ['btree (table_name)'],
        }

        self.settings = resolve_settings(settings)
        self.dbname = self.settings.dbname
        self.user = self.settings.user
        self.password = self.settings.password
        self.host = self.settings.host
        self.port = self.settings.port

        self.tunnel = tunnel
        if tunnel:
            # paramiko and sshtunnel are optional; only the tunnelled path needs them.
            from paramiko import RSAKey
            from sshtunnel import SSHTunnelForwarder

            ssh = self.settings.ssh
            self.tun = SSHTunnelForwarder(
                (ssh.host, ssh.port),
                ssh_username=ssh.user,
                ssh_pkey=RSAKey.from_private_key_file(ssh.key, password=ssh.key_password),
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


    def create_tables(self, conn_pool, include_state_tables: bool = True):
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

        # Side tables for `update_table` / `find_union_peer`. Skipped when callers
        # explicitly opt out (e.g. plain offline indexing with no update support).
        if not include_state_tables:
            return

        # Best-effort creation of the state tables. They are independent of the
        # fs/overlap tables, so DuplicateTable on those should not block us here.
        for q in (self.create_tables_meta_query, self.create_num_state_query, self.create_cat_state_query):
            conn = conn_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(q)
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
