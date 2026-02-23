"""
MinHash LSH implementation for Jaccard similarity queries.

Uses banding technique to create hash tables for approximate
nearest neighbor search based on Jaccard similarity threshold.
"""

import pickle
import random
import string
import struct
from typing import List, Optional, Tuple, Set, Any

import numpy as np
from scipy.integrate import quad as integrate

from .storage import ordered_storage, unordered_storage


def _false_positive_probability(threshold: float, b: int, r: int) -> float:
    """
    Compute the probability of false positive using integration.
    
    Args:
        threshold: Jaccard similarity threshold
        b: Number of bands
        r: Number of rows per band
        
    Returns:
        float: Probability of false positive
    """
    def _probability(s):
        return 1 - (1 - s ** float(r)) ** float(b)
    
    a, _ = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold: float, b: int, r: int) -> float:
    """
    Compute the probability of false negative using integration.
    
    Args:
        threshold: Jaccard similarity threshold
        b: Number of bands
        r: Number of rows per band
        
    Returns:
        float: Probability of false negative
    """
    def _probability(s):
        return 1 - (1 - (1 - s ** float(r)) ** float(b))
    
    a, _ = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold: float, num_perm: int,
                   false_positive_weight: float, false_negative_weight: float) -> Tuple[int, int]:
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    
    Args:
        threshold: Jaccard similarity threshold
        num_perm: Number of permutation functions
        false_positive_weight: Weight for false positives
        false_negative_weight: Weight for false negatives
        
    Returns:
        Tuple[int, int]: Optimal (b, r) parameters
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def _random_name(length: int) -> bytes:
    """Generate a random name for storage identification."""
    return ''.join(random.choice(string.ascii_lowercase)
                   for _ in range(length)).encode('utf8')


class MinHashLSH:
    """
    The MinHash LSH index supporting Jaccard similarity threshold queries.
    
    Reference: Chapter 3, Mining of Massive Datasets (http://www.mmds.org/)

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and 1.0.
            The initialized MinHash LSH will be optimized for the threshold by
            minimizing the false positive and false negative.
        num_perm (int): The number of permutation functions used by the MinHash.
        weights (tuple): Used to adjust the relative importance of
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            Format: (false_positive_weight, false_negative_weight).
        params (tuple): The LSH parameters (b, r) - number of bands and rows per band.
            This bypasses parameter optimization; threshold and weights will be ignored.
        storage_config (dict): Type of storage service for hashtables.
        prepickle (bool): If True, all keys are pickled to bytes before insertion.
        hashfunc (function): Optional hash function to compress index keys.

    Note:
        `weights` must sum to 1.0.
        For example, if minimizing false negative is more important,
        assign more weight: weights=(0.4, 0.6).
    """

    def __init__(self, threshold: float = 0.9, num_perm: int = 128,
                 weights: Tuple[float, float] = (0.5, 0.5),
                 params: Optional[Tuple[int, int]] = None,
                 storage_config: Optional[dict] = None,
                 prepickle: Optional[bool] = None,
                 hashfunc=None):
        
        storage_config = {'type': 'dict'} if not storage_config else storage_config
        self._buffer_size = 50000
        
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.h = num_perm
        
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError(
                    f"The product of b and r in params is {self.b} * {self.r} = {self.b * self.r} "
                    f"-- it must be less than num_perm {num_perm}. Did you forget to specify num_perm?"
                )
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(threshold, num_perm,
                                            false_positive_weight, false_negative_weight)

        self.prepickle = storage_config['type'] == 'redis' if prepickle is None else prepickle
        self.hashfunc = hashfunc
        
        if hashfunc:
            self._H = self._hashed_byteswap
        else:
            self._H = self._byteswap

        basename = storage_config.get('basename', _random_name(11))
        self.hashtables = [
            unordered_storage(storage_config, name=b''.join([basename, b'_bucket_', struct.pack('>H', i)]))
            for i in range(self.b)
        ]
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int):
        for t in self.hashtables:
            t.buffer_size = value
        self._buffer_size = value

    def insert(self, key, minhash, check_duplication: bool = True):
        """
        Insert a key to the index, together with a MinHash of the set.

        Args:
            key: The identifier of the set.
            minhash: The MinHash of the set.
            check_duplication: To avoid duplicate keys in the storage.
        """
        self._insert(key, minhash, check_duplication=check_duplication, buffer=False)

    def _insert(self, key, minhash, check_duplication: bool = True, buffer: bool = False):
        if len(minhash) != self.h:
            raise ValueError(f"Expecting minhash with length {self.h}, got {len(minhash)}")
        
        if self.prepickle:
            key = pickle.dumps(key)
        
        Hs = [self._H(minhash.hashvalues[start:end]) for start, end in self.hashranges]
        
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, set([key]), buffer=buffer)

    def query(self, minhash) -> List:
        """
        Given a MinHash, retrieve keys with Jaccard similarities
        likely greater than the threshold.

        Results are approximate. For more accuracy, filter with minhash.jaccard().

        Args:
            minhash: The MinHash of the query set.

        Returns:
            List of unique keys.
        """
        if len(minhash) != self.h:
            raise ValueError(f"Expecting minhash with length {self.h}, got {len(minhash)}")
        
        candidates: Set = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                candidates.add(key)
        
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        else:
            return list(candidates)

    def add_to_query_buffer(self, minhash):
        """
        Buffer queries for batch execution.
        
        Use with collect_query_buffer() for faster batch queries.

        Args:
            minhash: The MinHash of the query set.
        """
        if len(minhash) != self.h:
            raise ValueError(f"Expecting minhash with length {self.h}, got {len(minhash)}")
        
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            hashtable.add_to_select_buffer([H])

    def collect_query_buffer(self) -> List:
        """
        Execute buffered queries and return results.

        If multiple MinHashes were added, returns the intersection of all results.

        Returns:
            List of unique keys.
        """
        collected_result_sets = [
            set(collected_result_lists)
            for hashtable in self.hashtables
            for collected_result_lists in hashtable.collect_select_buffer()
        ]
        
        if not collected_result_sets:
            return []
        
        if self.prepickle:
            return [pickle.loads(key) for key in set.intersection(*collected_result_sets)]
        return list(set.intersection(*collected_result_sets))

    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return any(t.size() == 0 for t in self.hashtables)

    def _byteswap(self, hs) -> bytes:
        return bytes(hs.byteswap().data)

    def _hashed_byteswap(self, hs) -> Any:
        return self.hashfunc(bytes(hs.byteswap().data))

    def _query_b(self, minhash, b: int) -> Set:
        """Query using only the first b hash tables."""
        if len(minhash) != self.h:
            raise ValueError(f"Expecting minhash with length {self.h}, got {len(minhash)}")
        if b > len(self.hashtables):
            raise ValueError("b must be less or equal to the number of hash tables")
        
        candidates: Set = set()
        for (start, end), hashtable in zip(self.hashranges[:b], self.hashtables[:b]):
            H = self._H(minhash.hashvalues[start:end])
            if H in hashtable:
                for key in hashtable[H]:
                    candidates.add(key)
        
        if self.prepickle:
            return {pickle.loads(key) for key in candidates}
        else:
            return candidates

    def get_counts(self) -> List:
        """
        Get bucket allocation counts for each hash table.
        
        Returns:
            List of dicts with bucket counts for each permutation.
        """
        return [hashtable.itemcounts() for hashtable in self.hashtables]


class MinHashLSHInsertionSession:
    """Context manager for batch insertion into MinHashLSH."""

    def __init__(self, lsh: MinHashLSH, buffer_size: int):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def insert(self, key, minhash, check_duplication: bool = True):
        """
        Insert a key with its MinHash into the index.

        Args:
            key: The unique identifier of the set.
            minhash: The MinHash of the set.
        """
        self.lsh._insert(key, minhash, check_duplication=check_duplication, buffer=True)
