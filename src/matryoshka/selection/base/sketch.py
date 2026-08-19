from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import polars as pl


@dataclass
class AugSketch:
    table_feature_index: str
    count: int
    sum_: List[float]
    diag: List[float]
    cofactors: List[float]
    aug_feature_indices: Dict[int, str] = None
    # Identifier of the source column underlying this sketch, stripped of any
    # aggregation suffix (e.g. `_min`, `_max`, `_median`, `_mean`, `_nunique`).
    # Used to dedup candidates that are merely different aggregations of the
    # same base column. ``None`` when not applicable (e.g. joint sketches).
    base_column: str = None

    def __repr__(self):
        return f"AugSketch(table_feature_index={self.table_feature_index}, feature_indices={self.aug_feature_indices}, base_column={self.base_column}, count={self.count.shape}), sum_=[{self.sum_.shape}, {self.sum_.dtype}], diag=[{self.diag.shape}, {self.diag.dtype}], cofactors=[{self.cofactors.shape}, {self.cofactors.dtype}])"


@dataclass
class BaseSketch:
    count: int
    features_sum: List[float]
    # Per-key outer product of the per-key sum vector, i.e. s_k s_k^T.
    # Required by the joint-cofactor reconstruction path in SketchProcessor,
    # which is why this slot keeps its legacy semantics.
    features_cofactors: List[float]
    target_sum: List[float]
    # Per-key first column of s_k s_k^T (target row of the legacy block).
    target_cofactors: List[float]
    initial_frame: pl.DataFrame = None
    # True per-key Gram of the base features, sum_{i in key k} x_i x_i^T,
    # shape (K, p_base, p_base). Used by _evaluate_base_only_score to
    # recover X_base^T X_base via .sum(axis=0). None if not materialised.
    features_gram: Optional[np.ndarray] = None
    # Per-key first column of the full (p_base+1, p_base+1) Gram including
    # the target as column 0: shape (K, p_base+1, 1). Row 0 holds y'y for
    # that key; rows 1..p_base hold x_j'y. None if not materialised.
    target_gram_row: Optional[np.ndarray] = None


    def __repr__(self):
        return f"BaseSketch(count={self.count.shape}, features_sum=[{self.features_sum.shape}, {self.features_sum.dtype}], features_cofactors=[{self.features_cofactors.shape}, {self.features_cofactors.dtype}], target_sum=[{self.target_sum.shape}, {self.target_sum.dtype}], target_cofactors=[{self.target_cofactors.shape}, {self.target_cofactors.dtype}])"
