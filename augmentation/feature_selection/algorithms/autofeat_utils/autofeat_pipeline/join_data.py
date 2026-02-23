from pathlib import Path

import pandas as pd
import polars as pl
from polars import col

from .neo4j_transactions import get_pk_fk_nodes

def transform_node_to_dict(node):
    dict_node = {'id': node.get('id'), 'label': node.get('label'), 'name': node.get('name'),
                 'source_name': node.get('source_name'), 'source_path': node.get('source_path')}
    return dict_node

def join_directly_connected(base_table_id: str):
    nodes = get_pk_fk_nodes(base_table_id)
    partial_join = None
    for pk, fk in nodes:
        pk_node = transform_node_to_dict(pk)
        fk_node = transform_node_to_dict(fk)

        left_table = pd.read_csv(pk_node["source_path"])
        right_table = pd.read_csv(fk_node["source_path"])
        if partial_join is not None:
            left_table = partial_join

        partial_join = pd.merge(
            left_table,
            right_table,
            how="left",
            left_on=pk_node["name"],
            right_on=fk_node["name"],
            suffixes=("", "_b"),
        )
        columns_to_drop = [c for c in list(partial_join.columns) if c.endswith("_b")]
        partial_join.drop(columns=columns_to_drop, inplace=True)

    return partial_join


def pl_outer_join(df1: pl.DataFrame, df2: pl.DataFrame, how, left_on, right_on):
    new_names = [f"{x}_tmp" for x in [left_on, right_on]]

    # replicate join columns with new names
    df1 = df1.with_columns(col(left_on).alias(new_names[0]))
    df2 = df2.with_columns(col(right_on).alias(new_names[1]))

    # perform join and drop columns
    return df1.join(df2, left_on=new_names[0], right_on=new_names[1], how=how).drop(new_names)


def join_and_save(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_column_name: str,
    right_column_name: str,
    join_path: Path,
    csv: bool = True,
    save_to_disk: bool = True,
) -> pd.DataFrame:
    """
    Join two dataframes and save the result on disk.

    :param left_df: Left side of the join
    :param right_df: Right side of the join
    :param left_column_name: The left join column
    :param right_column_name: The right join column
    :param join_path: The path to save the join result.
    :param csv: Whether to save as CSV or not.
    :return: The join result.
    """
    if left_df[left_column_name].dtype != right_df[right_column_name].dtype:
        print(f"Data type mismatch: {left_df[left_column_name].dtype} vs {right_df[right_column_name].dtype}")
        return None

    if isinstance(left_df, pl.DataFrame):
        partial_join = pl_outer_join(
            left_df,
            right_df,
            how="left",
            left_on=left_column_name,
            right_on=right_column_name,
        ).to_pandas()
    elif isinstance(left_df, pd.DataFrame):
        partial_join = pd.merge(
            left_df,
            right_df,
            how="left",
            left_on=left_column_name,
            right_on=right_column_name,
        )
    else:
        raise Exception("Unknown dataframe type")

    if save_to_disk:
        if csv:
            partial_join.to_csv(join_path, index=False)
        else:
            partial_join.to_parquet(join_path, index=False)

    return partial_join
