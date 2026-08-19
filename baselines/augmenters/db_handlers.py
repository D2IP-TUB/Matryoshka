"""PostgreSQL schema handlers for the COCOA and QCR baseline indexes.

Split out of ``matryoshka.db.handler``: neither table layout is part of the
Matryoshka index, and both are needed only when rebuilding the corresponding
baseline. Connection parameters resolve exactly as for the core handler, see
:mod:`matryoshka.db.settings`.
"""
import psycopg2
from paramiko import RSAKey
from sshtunnel import SSHTunnelForwarder

from matryoshka.db.handler import DBHandler
from matryoshka.db.settings import resolve_settings


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

        settings = resolve_settings()
        self.dbname = settings.dbname
        self.user = settings.user
        self.password = settings.password
        self.host = settings.host
        self.port = settings.port

        self.ssh_host = settings.ssh.host
        self.ssh_user = settings.ssh.user
        self.ssh_port = settings.ssh.port
        self.ssh_key = settings.ssh.key
        self.ssh_key_pwd = settings.ssh.key_password

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

        settings = resolve_settings()
        self.dbname = settings.dbname
        self.user = settings.user
        self.password = settings.password
        self.host = settings.host
        self.port = settings.port

        self.ssh_host = settings.ssh.host
        self.ssh_user = settings.ssh.user
        self.ssh_port = settings.ssh.port
        self.ssh_key = settings.ssh.key
        self.ssh_key_pwd = settings.ssh.key_password

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