"""
Multi-hop join graph (bridge-column expansion).

A graph rooted at a query column. Each non-root node represents a key
column ``(table_index, key_col_index)`` in the data lake reachable from
the root via a chain of bridge joins.

Bridge expansion
----------------
For a node ``X`` whose key column matched its parent on some rows, we
expand ``X`` by inspecting the *other* key columns of ``X``'s table at
those matched rows. The values found in those bridge columns are used as
input to an overlap query, which surfaces the next-hop children.

Root-token tracking
-------------------
Every row carried by a node is annotated with the set of root tokens that
reached it via the chain. This lets the join-selection step rewrite each
descendant's ``key`` back into the root key space (the ``{1,2,3} -> {a,b,c}``
substitution) so that downstream joins to the user table remain valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import polars as pl

from .exceptions import KeyNotFoundError


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class JoinNode:
    """A reachable key column in the multi-hop join graph.

    ``rows`` is the per-node payload: one entry per row of this node's
    key column that is reachable from the root, with the set of root
    tokens that reach it.
    """
    node_id: str
    table_index: Optional[int] = None
    key_col_index: Optional[int] = None
    # rows: pl.DataFrame with columns
    #   - row_index    : Int64   row index in this node's table
    #   - key          : String  value of this node's key column at that row
    #                            (== bridge value from parent at that row)
    #   - root_tokens  : List[String]  root tokens that reach this row
    rows: pl.DataFrame = field(default_factory=lambda: _empty_rows_df())
    # Joinability w.r.t. parent. None for root.
    joinability: Optional[float] = None
    # Bridge column on parent's table that produced this node (None for hop-1
    # children, since hop-1 expansion uses the root column directly).
    parent_bridge_col_index: Optional[int] = None
    depth: int = 0
    parent: Optional["JoinNode"] = field(default=None, repr=False)
    children: list["JoinNode"] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------- API
    def add_child(self, child: "JoinNode") -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def walk(self):
        """Pre-order traversal yielding every node in the subtree."""
        yield self
        for c in self.children:
            yield from c.walk()

    def path_to_root(self) -> list["JoinNode"]:
        path, cur = [], self
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        return list(reversed(path))


def _empty_rows_df() -> pl.DataFrame:
    return pl.DataFrame(schema={
        'row_index': pl.Int64,
        'key': pl.String,
        'root_tokens': pl.List(pl.String),
    })


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #
class JoinGraph:
    """
    Build a multi-hop join graph rooted at a query column.

    Parameters
    ----------
    worker
        A ``JoinDiscovery`` instance (provides ``conninfo``, the overlap
        query template, and ``_run_overlap_query`` /
        ``_process_overlap_query_results``).
    feature_selection_table_name
        Name of the feature-selection index table. Used to fetch bridge
        columns directly. If not provided, parsed from the worker's
        ``overlap_query`` template.
    top_k
        Top-k candidates per overlap query.
    n_hops
        Maximum hop count (root -> child -> grandchild = ``n_hops=2``).
    joinability_threshold
        After hop 1, only children whose overlap ratio with parent is
        ``>= joinability_threshold`` are expanded further (and kept).
    """

    def __init__(
        self,
        worker,
        feature_selection_table_name: Optional[str] = None,
        top_k: int = 10,
        n_hops: int = 2,
        joinability_threshold: float = 0.5,
    ) -> None:
        self.worker = worker
        self.top_k = top_k
        self.n_hops = n_hops
        self.joinability_threshold = joinability_threshold
        self.fs_table = (
            feature_selection_table_name
            or self._infer_fs_table_name(worker)
        )
        self.root: Optional[JoinNode] = None
        # Track (table_index, key_col_index) already attached to avoid
        # turning the underlying DAG into an exponentially-large tree.
        self._seen: set[tuple[int, int]] = set()
        # Total distinct root tokens; set in `build`. Used to gate hop>=2
        # children by their cumulative reach back to the root, not just
        # the local parent->child overlap.
        self._n_root_tokens: int = 0

    # ------------------------------------------------------------------ build
    def build(self, query_column: pl.DataFrame) -> JoinNode:
        """Construct the graph and return the root node."""
        query_column_name = query_column.columns[0]

        # Root: each token maps to itself.
        root_tokens = (
            query_column
            .select(pl.col(query_column_name).cast(pl.String))
            .to_series()
            .drop_nulls()
            .unique()
            .to_list()
        )
        root_rows = pl.DataFrame({
            'row_index': pl.Series(range(len(root_tokens)), dtype=pl.Int64),
            'key': pl.Series(root_tokens, dtype=pl.String),
            'root_tokens': pl.Series(
                [[t] for t in root_tokens],
                dtype=pl.List(pl.String),
            ),
        })
        self._n_root_tokens = len(root_tokens)
        self.root = JoinNode(
            node_id=query_column_name,
            rows=root_rows,
            depth=0,
        )

        # Hop 1: expand directly off the root column (no bridge needed).
        self._expand_hop1(self.root, query_column)

        # Hop 2..N: bridge expansion through other key columns of parent's table.
        frontier = list(self.root.children)
        for _ in range(1, self.n_hops):
            next_frontier: list[JoinNode] = []
            for node in frontier:
                self._expand_via_bridges(node)
                next_frontier.extend(node.children)
            if not next_frontier:
                break
            frontier = next_frontier

        return self.root

    # ----------------------------------------------------------- hop-1 (root)
    def _expand_hop1(self, root: JoinNode, query_column: pl.DataFrame) -> None:
        """Expand the root using the original overlap query."""
        try:
            overlap_results, distinct_tokens = self.worker._run_overlap_query(
                query_column, top_k=self.top_k
            )
            processed, _ratio = self.worker._process_overlap_query_results(
                overlap_results, query_column.columns[0], distinct_tokens
            )
        except KeyNotFoundError:
            return

        grouped = (
            processed
            .group_by(['table_index', 'key_col_index'], maintain_order=True)
            .agg([
                pl.col('key'),
                pl.col('row_index'),
                pl.col('number_of_tokens').first().alias('joinability'),
            ])
        )

        for r in grouped.iter_rows(named=True):
            tbl, kci = int(r['table_index']), int(r['key_col_index'])
            if (tbl, kci) in self._seen:
                continue
            joinability = float(r['joinability'])
            if joinability < self.joinability_threshold:
                continue
            keys = [str(k) for k in r['key']]
            row_indices = [int(x) for x in r['row_index']]
            # Hop-1 child: each matched key IS itself a root token.
            rows = pl.DataFrame({
                'row_index': pl.Series(row_indices, dtype=pl.Int64),
                'key': pl.Series(keys, dtype=pl.String),
                'root_tokens': pl.Series(
                    [[k] for k in keys],
                    dtype=pl.List(pl.String),
                ),
            })
            child = JoinNode(
                node_id=f'{tbl}_{kci}',
                table_index=tbl,
                key_col_index=kci,
                rows=rows,
                joinability=joinability,
                parent_bridge_col_index=None,
            )
            root.add_child(child)
            self._seen.add((tbl, kci))

    # ----------------------------------------------------- hop>=2 (bridging)
    def _expand_via_bridges(self, node: JoinNode) -> None:
        """Expand `node` by treating each non-key column of its table as a bridge."""
        if node.table_index is None or node.rows.height == 0:
            return

        bridge_df = self._fetch_bridge_columns(
            node.table_index,
            node.key_col_index,
            node.rows['row_index'].to_list(),
        )
        if bridge_df.is_empty():
            return

        # Attach root_tokens carried by `node` for each row, then collapse to
        # (bridge_col, bridge_value) -> aggregated root_tokens.
        bridge_with_roots = (
            bridge_df
            .with_columns(pl.col('key').cast(pl.String))
            .join(
                node.rows.select(['row_index', 'root_tokens']),
                on='row_index',
                how='inner',
            )
            .group_by(['key_col_index', 'key'])
            .agg(
                pl.col('root_tokens').flatten().unique().alias('root_tokens'),
            )
        )

        # One overlap query per bridge column.
        for (bridge_col_idx,), bridge_group in bridge_with_roots.group_by(
            ['key_col_index'], maintain_order=True
        ):
            self._expand_one_bridge(node, int(bridge_col_idx), bridge_group)

    def _expand_one_bridge(
        self,
        parent: JoinNode,
        bridge_col_idx: int,
        bridge_group: pl.DataFrame,
    ) -> None:
        """
        Run an overlap query using `bridge_group` as input tokens and attach
        children. `bridge_group` has columns (key, root_tokens) where each
        row gives a bridge value and the set of root tokens reaching it.
        """
        bridge_values = bridge_group['key'].to_list()
        if not bridge_values:
            return
        v2roots: dict[str, list[str]] = dict(
            zip(bridge_values, bridge_group['root_tokens'].to_list())
        )

        bridge_df = pl.DataFrame(
            {f'bridge_{parent.table_index}_{bridge_col_idx}': bridge_values}
        )
        try:
            overlap_results, distinct_tokens = self.worker._run_overlap_query(
                bridge_df, top_k=self.top_k
            )
            processed, _ratio = self.worker._process_overlap_query_results(
                overlap_results, bridge_df.columns[0], distinct_tokens
            )
        except KeyNotFoundError:
            return

        # Drop self-matches: any candidate within the same table as the
        # parent is not a real join — it is just another column of the
        # same row. The trivial (table, bridge_col) match is the obvious
        # case, but other key columns of the same table can also surface
        # via the overlap query (e.g. parent `13_0` bridging through
        # column 3 of table 13 will pull `(13, 3)` back as a "child").
        processed = processed.filter(
            pl.col('table_index') != parent.table_index
        )

        grouped = (
            processed
            .group_by(['table_index', 'key_col_index'], maintain_order=True)
            .agg([
                pl.col('key'),
                pl.col('row_index'),
                pl.col('number_of_tokens').first().alias('joinability'),
            ])
        )

        for r in grouped.iter_rows(named=True):
            tbl, kci = int(r['table_index']), int(r['key_col_index'])
            if (tbl, kci) in self._seen:
                continue
            joinability = float(r['joinability'])
            if joinability < self.joinability_threshold:
                continue

            keys = [str(k) for k in r['key']]
            row_indices = [int(x) for x in r['row_index']]
            # For each matched row of the candidate, root_tokens come from
            # the bridge value at that row.
            root_tokens_per_row = [
                sorted(set(v2roots.get(k, []))) for k in keys
            ]
            rows = pl.DataFrame({
                'row_index': pl.Series(row_indices, dtype=pl.Int64),
                'key': pl.Series(keys, dtype=pl.String),
                'root_tokens': pl.Series(
                    root_tokens_per_row, dtype=pl.List(pl.String)
                ),
            }).filter(pl.col('root_tokens').list.len() > 0)
            if rows.is_empty():
                continue

            # Cumulative-reach gate: fraction of root tokens that actually
            # reach this candidate via the full chain. Local parent->child
            # joinability is not enough because a high-overlap link off a
            # low-coverage parent still produces an essentially empty join.
            n_root_reached = (
                rows['root_tokens'].explode().drop_nulls().n_unique()
            )
            cumulative_joinability = (
                n_root_reached / self._n_root_tokens
                if self._n_root_tokens
                else 0.0
            )
            if cumulative_joinability < self.joinability_threshold:
                continue

            child = JoinNode(
                node_id=f'{tbl}_{kci}',
                table_index=tbl,
                key_col_index=kci,
                rows=rows,
                joinability=cumulative_joinability,
                parent_bridge_col_index=bridge_col_idx,
            )
            parent.add_child(child)
            self._seen.add((tbl, kci))

    # ------------------------------------------------------------- DB helper
    def _fetch_bridge_columns(
        self,
        table_index: int,
        exclude_key_col_index: int,
        row_indices: list[int],
    ) -> pl.DataFrame:
        """
        Fetch (key_col_index, row_index, key) for all key columns of
        ``table_index`` other than ``exclude_key_col_index``, restricted to
        ``row_indices``.
        """
        empty = pl.DataFrame(schema={
            'key_col_index': pl.Int64,
            'row_index': pl.Int64,
            'key': pl.String,
        })
        if not row_indices:
            return empty
        rows_csv = ','.join(str(int(r)) for r in row_indices)
        query = (
            'SELECT key_col_index, row_index, key '
            f'FROM {self.fs_table} '
            f'WHERE table_index = {int(table_index)} '
            f'  AND key_col_index <> {int(exclude_key_col_index)} '
            f'  AND row_index IN ({rows_csv});'
        )
        return pl.read_database_uri(query, self.worker.conninfo)

    @staticmethod
    def _infer_fs_table_name(worker) -> str:
        """Best-effort extraction of the fs table name from the worker's overlap_query."""
        q = getattr(worker, 'overlap_query', '')
        marker = 'FROM '
        idx = q.find(marker)
        if idx < 0:
            raise ValueError(
                'Could not infer feature_selection_table_name; pass it '
                'explicitly to JoinGraph(feature_selection_table_name=...).'
            )
        return q[idx + len(marker):].lstrip().split()[0]

    # ----------------------------------------------------------------- export
    def to_edges(self) -> pl.DataFrame:
        """Edge list for visualization/debugging."""
        rows = []
        if self.root is None:
            return pl.DataFrame(schema={
                'parent_id': pl.String,
                'child_id': pl.String,
                'depth': pl.Int64,
                'joinability': pl.Float64,
                'bridge_col': pl.Int64,
            })
        for node in self.root.walk():
            for child in node.children:
                rows.append({
                    'parent_id': node.node_id,
                    'child_id': child.node_id,
                    'depth': child.depth,
                    'joinability': child.joinability,
                    'bridge_col': child.parent_bridge_col_index,
                })
        return pl.DataFrame(rows, schema={
            'parent_id': pl.String,
            'child_id': pl.String,
            'depth': pl.Int64,
            'joinability': pl.Float64,
            'bridge_col': pl.Int64,
        })

    def to_join_selection_input(self) -> pl.DataFrame:
        """
        Flatten all non-root nodes into a frame ready for the
        join-selection query, with each row's ``key`` rewritten into the
        root token space.

        Schema (one row per (descendant_node, lake_row, root_token)):
            - table_index    : Int64
            - key_col_index  : Int64
            - row_index      : Int64    row in the candidate table
            - matched_key    : String   bridge value at that row
            - key            : String   root token (rewritten)
            - depth          : Int64
            - joinability    : Float64
        """
        empty_schema = {
            'table_index': pl.Int64,
            'key_col_index': pl.Int64,
            'row_index': pl.Int64,
            'matched_key': pl.String,
            'key': pl.String,
            'depth': pl.Int64,
            'joinability': pl.Float64,
        }
        if self.root is None:
            return pl.DataFrame(schema=empty_schema)

        frames = []
        for node in self.root.walk():
            if node is self.root or node.rows.is_empty():
                continue
            frames.append(
                node.rows
                .rename({'key': 'matched_key'})
                .with_columns([
                    pl.lit(node.table_index, dtype=pl.Int64).alias('table_index'),
                    pl.lit(node.key_col_index, dtype=pl.Int64).alias('key_col_index'),
                    pl.lit(node.depth, dtype=pl.Int64).alias('depth'),
                    pl.lit(node.joinability, dtype=pl.Float64).alias('joinability'),
                ])
                .explode('root_tokens')
                .rename({'root_tokens': 'key'})
                .select([
                    'table_index', 'key_col_index', 'row_index',
                    'matched_key', 'key', 'depth', 'joinability',
                ])
            )
        if not frames:
            return pl.DataFrame(schema=empty_schema)
        return pl.concat(frames, how='vertical_relaxed')
