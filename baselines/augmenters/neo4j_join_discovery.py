"""Neo4j-backed multi-hop join discovery.

When the underlying lake has been ingested into a Neo4j graph (as done by the
original AutoFeat code base in ``autofeat_og_utils``), this helper traverses
the graph BFS-style from a base table up to ``max_depth`` hops and emits a
``join_paths_df``-shaped CSV that drop-in replaces Aurum's output.

Multi-hop chains (depth > 1) are materialised into single CSVs under
``<output_dir>/_neo4j_chains/`` so downstream augmenters (kitana, caafe, ...)
that expect single-hop ``buyer ↔ seller`` semantics keep working without
changes: each materialised chain looks like one big seller table keyed by the
original buyer-side join column.

Configuration is via the same environment variables the OG package uses:
``NEO4J_HOST``, ``NEO4J_USER``, ``NEO4J_PASS``, ``NEO4J_DATABASE``.
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable, List, Optional, Tuple

import pandas as pd

NEO4J_HOST_ENV = "NEO4J_HOST"
NEO4J_USER_ENV = "NEO4J_USER"
NEO4J_PASS_ENV = "NEO4J_PASS"
NEO4J_DB_ENV = "NEO4J_DATABASE"


def _driver():
    from neo4j import GraphDatabase  # type: ignore

    host = os.environ.get(NEO4J_HOST_ENV, "neo4j://localhost:7687")
    user = os.environ.get(NEO4J_USER_ENV, "neo4j")
    password = os.environ.get(NEO4J_PASS_ENV, "neo4j")
    return GraphDatabase.driver(host, auth=(user, password))


def _database() -> Optional[str]:
    return os.environ.get(NEO4J_DB_ENV) or None


def _bfs_paths(
    base_node_id: str, max_depth: int
) -> List[List[dict]]:
    """Return all simple paths from ``base_node_id`` up to ``max_depth`` hops.

    Each path is a list of edge dicts ``{from_id, from_column, to_id, to_column}``.
    Implemented client-side BFS to avoid relying on APOC.
    """
    driver = _driver()
    db = _database()
    paths: List[List[dict]] = []
    try:
        with driver.session(database=db) if db else driver.session() as session:
            frontier: List[Tuple[str, List[dict], set]] = [
                (base_node_id, [], {base_node_id})
            ]
            for _depth in range(max_depth):
                next_frontier: List[Tuple[str, List[dict], set]] = []
                for cur, edges, visited in frontier:
                    result = session.run(
                        "MATCH (n:Node {id: $cur})-[r:RELATED]-(m:Node) "
                        "RETURN properties(r) AS props, n.id AS from_id, "
                        "m.id AS to_id ORDER BY r.weight DESC",
                        cur=cur,
                    )
                    for record in result:
                        props = record["props"]
                        from_id = record["from_id"]
                        to_id = record["to_id"]
                        if to_id in visited:
                            continue
                        # Neo4j stores undirected RELATED edges; orient using
                        # ``from_label``/``to_label`` so columns line up.
                        if props.get("from_label") == from_id:
                            from_col = props["from_column"]
                            to_col = props["to_column"]
                        else:
                            from_col = props["to_column"]
                            to_col = props["from_column"]
                        edge = {
                            "from_id": from_id,
                            "from_column": from_col,
                            "to_id": to_id,
                            "to_column": to_col,
                        }
                        new_edges = edges + [edge]
                        paths.append(new_edges)
                        next_frontier.append(
                            (to_id, new_edges, visited | {to_id})
                        )
                frontier = next_frontier
                if not frontier:
                    break
    finally:
        driver.close()
    return paths


def _read_lake_csv(
    data_lake_path: str, table_id: str, sep: str, **kwargs
) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(data_lake_path, table_id),
        sep=sep,
        engine="c",
        encoding="utf8",
        on_bad_lines="skip",
        **kwargs,
    )


def _materialise_chain(
    edges: List[dict],
    data_lake_path: str,
    lake_table_sep: str,
    chain_dir: str,
) -> Optional[Tuple[str, str]]:
    """Materialise an N-hop chain into a single CSV in ``chain_dir``.

    Returns ``(chain_filename, base_join_column)`` or ``None`` if the chain
    could not be built (missing columns, IO errors, etc.). The base table is
    NOT included in the chain — only intermediate hops + the leaf table —
    keyed by the first-hop ``from_column`` so the buyer can merge on it.
    """
    if not edges:
        return None

    # Hash the chain identity so repeated runs reuse the same file.
    key = "|".join(
        f"{e['from_id']}.{e['from_column']}->{e['to_id']}.{e['to_column']}"
        for e in edges
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    leaf = edges[-1]["to_id"].replace(".csv", "")
    chain_name = f"_chain_{leaf}_{digest}.csv"
    chain_path = os.path.join(chain_dir, chain_name)
    os.makedirs(chain_dir, exist_ok=True)
    if os.path.exists(chain_path):
        return chain_name, edges[0]["from_column"]

    base_join_col = edges[0]["from_column"]

    # Hop 0: read the first neighbour, keep base_join_col aliased to itself.
    first = edges[0]
    try:
        df = _read_lake_csv(data_lake_path, first["to_id"], lake_table_sep)
    except Exception:
        return None
    if first["to_column"] not in df.columns:
        return None
    # Drop duplicate columns that some lake CSVs ship with.
    df = df.loc[:, ~df.columns.duplicated()]
    # Reduce to ≤1 row per join key (M:1 semantics, mirrors caafe/kitana).
    try:
        df = df.groupby(first["to_column"]).sample(n=1, random_state=42)
    except (KeyError, ValueError):
        df = df.drop_duplicates(subset=[first["to_column"]])
    # Rename the seller-side join key to the buyer-side name so the rest of
    # the chain can address it uniformly.
    if first["to_column"] != base_join_col:
        df = df.rename(columns={first["to_column"]: base_join_col})

    # Subsequent hops: each ``edge.from_column`` lives on the previous hop's
    # frame (and was retained because we never drop columns). ``edge.to_column``
    # lives on the new table being merged in.
    for edge in edges[1:]:
        from_col = edge["from_column"]
        to_col = edge["to_column"]
        if from_col not in df.columns:
            return None
        try:
            right = _read_lake_csv(data_lake_path, edge["to_id"], lake_table_sep)
        except Exception:
            return None
        if to_col not in right.columns:
            return None
        right = right.loc[:, ~right.columns.duplicated()]
        try:
            right = right.groupby(to_col).sample(n=1, random_state=42)
        except (KeyError, ValueError):
            right = right.drop_duplicates(subset=[to_col])
        # Prefix non-key columns to avoid clashes across hops.
        stem = edge["to_id"].replace(".csv", "")
        rename_map = {
            c: f"{stem}__{c}" for c in right.columns if c != to_col
        }
        right = right.rename(columns=rename_map)
        try:
            df = df.merge(
                right,
                how="left",
                left_on=from_col,
                right_on=to_col,
                suffixes=("", f"__{stem}"),
            )
        except Exception:
            return None
        if to_col != from_col:
            try:
                df = df.drop(columns=[to_col])
            except KeyError:
                pass
        df = df.loc[:, ~df.columns.duplicated()]

    if base_join_col not in df.columns:
        return None
    try:
        df.to_csv(chain_path, index=False)
    except OSError:
        return None
    return chain_name, base_join_col


def discover_neo4j_join_paths(
    base_node_id: str,
    data_lake_path: str,
    lake_table_sep: str,
    output_path: str,
    max_depth: int = 2,
    feature_filter: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Generate a Neo4j-backed ``join_paths_df`` and write it to ``output_path``.

    The returned dataframe has Aurum-compatible columns
    ``[from_id, from_column, to_id, to_column]`` plus a ``depth`` column.
    Multi-hop chains are materialised under
    ``<output_dir>/_neo4j_chains/`` and exposed as synthetic single-hop
    sellers keyed on the buyer's first-hop join column.

    ``feature_filter`` is accepted for API compatibility with
    :class:`AurumJoinDiscovery.find_joinable_tables` but currently unused
    (no feature-level filtering is applied client-side).
    """
    paths = _bfs_paths(base_node_id, max_depth=max_depth)

    output_dir = os.path.dirname(output_path) or "."
    chain_dir = os.path.join(output_dir, "_neo4j_chains")

    rows: List[dict] = []
    seen: set = set()  # (from_id, from_column, to_id, to_column) dedup
    for edges in paths:
        depth = len(edges)
        if depth == 1:
            e = edges[0]
            key = (base_node_id, e["from_column"], e["to_id"], e["to_column"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "from_id": base_node_id,
                    "from_column": e["from_column"],
                    "to_id": e["to_id"],
                    "to_column": e["to_column"],
                    "depth": 1,
                }
            )
        else:
            mat = _materialise_chain(
                edges, data_lake_path, lake_table_sep, chain_dir
            )
            if mat is None:
                continue
            chain_name, base_join_col = mat
            key = (
                base_node_id,
                base_join_col,
                chain_name,
                base_join_col,
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "from_id": base_node_id,
                    "from_column": base_join_col,
                    "to_id": chain_name,
                    "to_column": base_join_col,
                    "depth": depth,
                }
            )

    df = pd.DataFrame(
        rows, columns=["from_id", "from_column", "to_id", "to_column", "depth"]
    )
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def neo4j_chain_data_lake(output_path: str, data_lake_path: str) -> str:
    """Return a path that contains both lake CSVs and materialised chains.

    Augmenters typically read seller files with
    ``f"{data_lake_path}/{to_id}"``. We materialise chains under
    ``<output_dir>/_neo4j_chains/<chain>.csv``, so callers should pass this
    return value as ``data_lake_path`` after creating symlinks. To avoid
    leaking files into the real data lake folder, we simply create a symlink
    farm: ``<output_dir>/_neo4j_lake/`` pointing to each lake table plus the
    chain CSVs.
    """
    output_dir = os.path.dirname(output_path) or "."
    farm = os.path.join(output_dir, "_neo4j_lake")
    chain_dir = os.path.join(output_dir, "_neo4j_chains")
    os.makedirs(farm, exist_ok=True)
    # Symlink lake tables once.
    if not os.path.exists(os.path.join(farm, ".lake_symlinked")):
        try:
            for entry in os.listdir(data_lake_path):
                src = os.path.join(data_lake_path, entry)
                dst = os.path.join(farm, entry)
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        pass
            with open(os.path.join(farm, ".lake_symlinked"), "w") as f:
                f.write("ok\n")
        except OSError:
            pass
    # Symlink chain files (always re-check; new chains may have been created).
    if os.path.isdir(chain_dir):
        for entry in os.listdir(chain_dir):
            src = os.path.join(chain_dir, entry)
            dst = os.path.join(farm, entry)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    pass
    return farm
