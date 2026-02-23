"""
MinHash LSH Ensemble implementation for containment queries.

LSH Ensemble is designed for set containment queries, unlike traditional
LSH which targets Jaccard similarity. It partitions sets by size and
uses separate LSH indexes for each partition.

Reference: LSH Ensemble: Internet-Scale Domain Search
"""

import pickle
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any, Iterator

import numpy as np

from .lsh import MinHashLSH
from .lshensemble_partition import optimal_partitions


def _false_positive_probability(threshold: float, b: int, r: int) -> float:
    """
    Compute probability of false positive for containment query.
    
    Integration over [0, threshold] of 1 - (1 - s^r)^b.
    """
    from scipy.integrate import quad
    
    def prob(s):
        return 1.0 - (1.0 - s ** float(r)) ** float(b)
    
    result, _ = quad(prob, 0.0, threshold)
    return result


def _false_negative_probability(threshold: float, b: int, r: int) -> float:
    """
    Compute probability of false negative for containment query.
    
    Integration over [threshold, 1.0] of 1 - (1 - (1 - s^r)^b).
    """
    from scipy.integrate import quad
    
    def prob(s):
        return 1.0 - (1.0 - (1.0 - s ** float(r)) ** float(b))
    
    result, _ = quad(prob, threshold, 1.0)
    return result


def _optimal_param(threshold: float, num_perm: int, max_r: int,
                   false_positive_weight: float, false_negative_weight: float) -> Tuple[int, int]:
    """
    Compute optimal LSH parameters minimizing weighted FP/FN error.
    
    Args:
        threshold: Containment threshold
        num_perm: Number of permutation functions
        max_r: Maximum rows per band
        false_positive_weight: Weight for false positives
        false_negative_weight: Weight for false negatives
        
    Returns:
        Tuple[int, int]: Optimal (b, r) parameters
    """
    min_error = float("inf")
    opt = (0, 0)
    
    for r in range(1, max_r + 1):
        max_b = int(num_perm / r)
        for b in range(1, max_b + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    
    return opt


class MinHashLSHEnsemble:
    """
    LSH Ensemble index for containment similarity queries.
    
    Unlike standard LSH for Jaccard similarity, LSH Ensemble handles
    containment queries where we want to find sets X such that
    |Q ∩ X| / |Q| >= threshold.
    
    Sets are partitioned by size, and each partition uses optimized
    LSH parameters for accurate containment estimation.

    Args:
        threshold (float): Containment threshold in [0.0, 1.0].
        num_perm (int): Number of MinHash permutation functions.
        num_part (int): Number of partitions to create.
        m (int): Memory multiplier - higher values give better accuracy
            at the cost of more memory.
        weights (tuple): (false_positive_weight, false_negative_weight)
            to balance precision vs recall.
        prepickle (bool): Whether to pickle keys before storage.

    Example:
        >>> lsh = MinHashLSHEnsemble(threshold=0.5, num_perm=256, num_part=32)
        >>> # Index data as (key, minhash, size) tuples
        >>> lsh.index([(key, mh, size) for key, mh, size in data])
        >>> # Query
        >>> results = lsh.query(query_minhash, query_size)
    """

    def __init__(self, threshold: float = 0.5, num_perm: int = 128,
                 num_part: int = 8, m: int = 4,
                 weights: Tuple[float, float] = (0.5, 0.5),
                 prepickle: bool = False):
        
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("num_perm must be at least 2")
        if num_part < 1:
            raise ValueError("num_part must be at least 1")
        if m < 1:
            raise ValueError("m must be at least 1")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("weights must be in [0.0, 1.0]")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1.0")
        
        self.threshold = threshold
        self.num_perm = num_perm
        self.num_part = num_part
        self.m = m
        self.weights = weights
        self.prepickle = prepickle
        
        # These are populated during indexing
        self.partitions: List[Tuple[int, int]] = []
        self.indexes: List[Optional[MinHashLSH]] = []
        self._initialized = False

    def index(self, entries: Iterator[Tuple[Any, Any, int]]) -> None:
        """
        Index a collection of sets.
        
        Args:
            entries: Iterator of (key, minhash, size) tuples where:
                - key: Unique identifier for the set
                - minhash: MinHash signature of the set
                - size: Cardinality of the set
        """
        # Collect entries and compute size distribution
        entry_list = list(entries)
        
        if len(entry_list) == 0:
            raise ValueError("Cannot index empty collection")
        
        # Build size distribution
        sizes = [e[2] for e in entry_list]
        size_counter = Counter(sizes)
        unique_sizes = np.array(sorted(size_counter.keys()))
        counts = np.array([size_counter[s] for s in unique_sizes])
        
        # Compute optimal partitions based on size distribution
        num_part = min(self.num_part, len(unique_sizes))
        if num_part < 2:
            self.partitions = [(unique_sizes[0], unique_sizes[-1])]
        else:
            self.partitions = optimal_partitions(unique_sizes, counts, num_part)
        
        # Compute optimal LSH parameters for each partition
        max_r = int(self.num_perm / 2)
        self.indexes = []
        
        for lower, upper in self.partitions:
            # Adjust threshold based on partition upper bound
            partition_threshold = self.threshold
            b, r = _optimal_param(partition_threshold, self.num_perm, max_r,
                                  self.weights[0], self.weights[1])
            
            if b == 0 or r == 0:
                # Fallback to reasonable defaults
                b, r = max(1, self.num_perm // 4), 4
            
            lsh = MinHashLSH(
                threshold=partition_threshold,
                num_perm=self.num_perm,
                params=(b, r),
                prepickle=self.prepickle
            )
            self.indexes.append(lsh)
        
        # Index entries into appropriate partitions
        for key, minhash, size in entry_list:
            # Find partition for this size
            for i, (lower, upper) in enumerate(self.partitions):
                if lower <= size <= upper:
                    self.indexes[i].insert(key, minhash)
                    break
        
        self._initialized = True

    def query(self, minhash, size: int) -> List:
        """
        Query for sets with high containment of the query set.
        
        Finds sets X such that |Q ∩ X| / |Q| >= threshold.

        Args:
            minhash: MinHash signature of the query set.
            size: Cardinality of the query set.

        Returns:
            List of keys for sets with estimated containment >= threshold.
        """
        if not self._initialized:
            raise ValueError("Index not initialized. Call index() first.")
        
        if size == 0:
            return []
        
        candidates: set = set()
        
        # Query each partition
        for lsh in self.indexes:
            if lsh is not None:
                results = lsh.query(minhash)
                candidates.update(results)
        
        return list(candidates)

    def __contains__(self, key) -> bool:
        """Check if a key exists in the index."""
        if not self._initialized:
            return False
        
        if self.prepickle:
            key = pickle.dumps(key)
        
        for lsh in self.indexes:
            if lsh is not None:
                for hashtable in lsh.hashtables:
                    for bucket in hashtable.keys():
                        if key in hashtable.get(bucket):
                            return True
        return False

    def is_empty(self) -> bool:
        """Check if the index is empty."""
        if not self._initialized:
            return True
        return all(lsh is None or lsh.is_empty() for lsh in self.indexes)

    def __getstate__(self) -> Dict:
        """Support pickling."""
        return {
            'threshold': self.threshold,
            'num_perm': self.num_perm,
            'num_part': self.num_part,
            'm': self.m,
            'weights': self.weights,
            'prepickle': self.prepickle,
            'partitions': self.partitions,
            'indexes': self.indexes,
            '_initialized': self._initialized,
        }

    def __setstate__(self, state: Dict) -> None:
        """Support unpickling."""
        self.threshold = state['threshold']
        self.num_perm = state['num_perm']
        self.num_part = state['num_part']
        self.m = state['m']
        self.weights = state['weights']
        self.prepickle = state['prepickle']
        self.partitions = state['partitions']
        self.indexes = state['indexes']
        self._initialized = state['_initialized']
