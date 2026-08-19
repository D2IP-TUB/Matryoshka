import pandas as pd
import psycopg2

from matryoshka.db.settings import resolve_settings
from .cocoa_utils import DataAugmentation


class CocoaAugmenter:
    # Connection parameters resolve through the library settings layer
    # (environment, then db_config.yaml), lazily on first access.
    def __init__(self, settings=None) -> None:
        resolved = resolve_settings(settings)
        self.host = resolved.host
        self.dbname = resolved.dbname
        self.user = resolved.user
        self.password = resolved.password
        self.port = resolved.port

    def run(
        self,
        distinct_tokens_table: str,
        main_tokenized_table: str,
        max_column_table: str,
        order_index_table: str,
        query_table_path: str,
        query_column: str,
        target_column: str,
        top_k_joinable: int,
        top_k_correlated: int
    ):
        conn_info = {
            'host': self.host,
            'dbname': self.dbname,
            'user': self.user,
            'password': self.password,
        }

        db_tables = {
            'dt': distinct_tokens_table,
            'mt': main_tokenized_table,
            'mc': max_column_table,
            'oi': order_index_table,
        }

        dataset = pd.read_csv(query_table_path)

        conn = psycopg2.connect(**conn_info)
        cocoa = DataAugmentation.COCOAHandler(conn, db_tables)
        result = cocoa.enrich(
            dataset,
            top_k_joinable,
            top_k_correlated,
            query_column,
            target_column
        )

        return result