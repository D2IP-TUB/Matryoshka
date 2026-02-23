from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AugSketch:
    table_feature_index: str
    count: int
    sum_: List[float]
    diag: List[float]
    cofactors: List[float]
    aug_feature_indices: Dict[int, str] = None

    def __repr__(self):
        return f"AugSketch(table_feature_index={self.table_feature_index}, feature_indices={self.aug_feature_indices}, count={self.count.shape}), sum_=[{self.sum_.shape}, {self.sum_.dtype}], diag=[{self.diag.shape}, {self.diag.dtype}], cofactors=[{self.cofactors.shape}, {self.cofactors.dtype}])"


@dataclass
class BaseSketch:
    count: int
    features_sum: List[float]
    features_cofactors: List[float]
    target_sum: List[float]
    target_cofactors: List[float]


    def __repr__(self):
        return f"BaseSketch(count={self.count.shape}, features_sum=[{self.features_sum.shape}, {self.features_sum.dtype}], features_cofactors=[{self.features_cofactors.shape}, {self.features_cofactors.dtype}], target_sum=[{self.target_sum.shape}, {self.target_sum.dtype}], target_cofactors=[{self.target_cofactors.shape}, {self.target_cofactors.dtype}])"