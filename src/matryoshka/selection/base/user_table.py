from dataclasses import dataclass

import polars as pl

from matryoshka.db.handler import DBHandler


@dataclass
class BaseTable(DBHandler):
    def __init__(
        self,
        feature_selection_table_name: str,
        overlap_table_name: str,
        table: pl.DataFrame,
        conninfo: str,
        query_column_name: str,
        target_column_name: str,
        baseline: bool = False,
        table_agg: pl.DataFrame = None,
        rows_map: dict[str, str] = None
    ):
        super().__init__(feature_selection_table_name=feature_selection_table_name, overlap_table_name=overlap_table_name)
        self.feature_selection_table_name = feature_selection_table_name
        self.overlap_table_name = overlap_table_name
        self.conninfo = conninfo
        self.table = table
        self.target_column_name = target_column_name
        self.query_column_name = query_column_name
        self.table_agg = table_agg
        self.rows_map = rows_map
        self.baseline = baseline
