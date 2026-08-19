import pandas as pd
import polars as pl

# Module-level cache for raw (unprefixed) dataframes, keyed by (lake_data_folder, node_id)
_df_cache: dict[tuple, pd.DataFrame] = {}


def get_pk_fk_nodes(join_paths_df: pd.DataFrame, source_path: str):
    """Return pairs of nodes (n, m) connected by a relationship with weight == 1."""
    matches = join_paths_df[
        ((join_paths_df["from_id"] == source_path) | (join_paths_df["to_id"] == source_path))
        & (join_paths_df.get("weight", 0) == 1)
    ]

    results = []
    for _, row in matches.iterrows():
        n = {"id": row["from_id"]}
        m = {"id": row["to_id"]}
        results.append([n, m])
    return results


def get_adjacent_nodes(join_paths_df: pd.DataFrame, base_node_id: str):
    """Simulate Neo4j: return all node IDs adjacent to base_node_id."""
    adjacent = set()
    for _, row in join_paths_df.iterrows():
        if row["from_id"] == base_node_id:
            adjacent.add(row["to_id"])
        elif row["to_id"] == base_node_id:
            adjacent.add(row["from_id"])
    return list(adjacent)


def get_relation_properties_node_name(join_paths_df: pd.DataFrame, from_id: str, to_id: str):
    """Simulate Neo4j: return relationship properties between two nodes."""
    matches = join_paths_df[
        ((join_paths_df["from_id"] == from_id) & (join_paths_df["to_id"] == to_id))
        | ((join_paths_df["from_id"] == to_id) & (join_paths_df["to_id"] == from_id))
    ]

    results = []
    for _, row in matches.iterrows():
        props = {
            "from_label": row["from_id"],
            "to_label": row["to_id"],
            "from_column": row["from_column"],
            "to_column": row["to_column"],
        }
        if "weight" in row:
            props["weight"] = row["weight"]

        results.append([props, row["from_id"], row["to_id"]])
    return results

def get_node_by_id(join_paths_df: pd.DataFrame, node_id: str):
    """Simulate Neo4j _get_node_by_id() using own join paths CSV."""
    matches = join_paths_df[
        (join_paths_df["from_id"] == node_id) | (join_paths_df["to_id"] == node_id)
    ]

    if matches.empty:
        return None

    columns = set()
    for _, row in matches.iterrows():
        if row["from_id"] == node_id and "from_column" in row:
            columns.add(row["from_column"])
        if row["to_id"] == node_id and "to_column" in row:
            columns.add(row["to_column"])

    return {
        "id": node_id,
        "columns": sorted(list(columns)),
        "relationships": len(matches),  # optional metadata
    }

def clear_df_cache():
    """Clear the module-level dataframe cache."""
    _df_cache.clear()


def get_df_with_prefix(
    join_paths_df: pd.DataFrame,
    lake_data_folder: str,
    node_id: str,
    table_sep: str,
    target_column=None,
    use_polars: bool = False,
) -> tuple:
    """
    Get the node from the database, read the file identified by node_id and prefix the column names with the node label.
    Raw (unprefixed) dataframes are cached by default to avoid re-reading large CSVs.

    :param node_id: ID of the node - used to retrieve the corresponding node from the database
    :param target_column: Optional parameter. The name of the label/target column containing the classes,
            only needed when the dataset to read contains the class.
    :return: 0: A pandas dataframe whose columns are prefixed with the node label, 1: the node label
    """
    node = get_node_by_id(join_paths_df, node_id)
    if node is None:
        print(f"Node with id {node_id} not found in join paths dataframe.")
        raise ValueError(f"Node with id {node_id} not found in join paths dataframe.")
    
    node_label = node.get("id")

    # Check cache for raw dataframe
    cache_key = (lake_data_folder, node_id)
    if cache_key in _df_cache:
        raw_df = _df_cache[cache_key].copy()
    else:
        if use_polars:
            raw_df = pl.read_csv(f'{lake_data_folder}/{node_id}', encoding="utf8", quote_char='"', separator=table_sep, ignore_errors=True).to_pandas()
        else:
            raw_df = pd.read_csv(
                f'{lake_data_folder}/{node_id}', header=0, engine="python", encoding="utf8", sep=table_sep,
                on_bad_lines='skip'
            )
        _df_cache[cache_key] = raw_df.copy()

    # Apply prefixing
    if target_column:
        dataframe = raw_df.set_index([target_column]).add_prefix(f"{node_label}.").reset_index()
    else:
        dataframe = raw_df.add_prefix(f"{node_label}.")

    return dataframe, node_label