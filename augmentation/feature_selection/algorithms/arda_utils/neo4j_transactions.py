import pandas as pd


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

def get_node_by_id(join_paths_df: pd.DataFrame,node_id: str):
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