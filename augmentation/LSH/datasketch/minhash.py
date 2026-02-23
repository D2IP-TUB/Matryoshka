"""
MinHash implementation for set similarity estimation.

Uses permutation-based MinHash with configurable number of hash functions.
Supports computing Jaccard similarity between MinHash signatures.
"""

import struct
import hashlib
import numpy as np

# Constants for hash function generation
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)


def _sha1_hash32(b):
    """Deterministic 32-bit hash using SHA-1 (consistent across processes)."""
    return struct.unpack('<I', hashlib.sha1(b).digest()[:4])[0]


class MinHash:
    """
    MinHash signature for estimating Jaccard similarity between sets.
    
    Uses random permutation functions to generate hash signatures.
    The signature can be used for approximate nearest neighbor search
    via Locality-Sensitive Hashing.
    
    Args:
        num_perm (int): Number of permutation functions (default: 128)
        seed (int): Random seed for reproducibility (default: 1)
        hashfunc: Optional custom hash function
        hashobj: Deprecated, use hashfunc
        hashvalues: Pre-computed hash values (for deserialization)
        permutations: Pre-computed permutation functions
    """
    
    __slots__ = ('permutations', 'hashvalues', 'seed', 'hashfunc')

    def __init__(self, num_perm: int = 128, seed: int = 1, hashfunc=None,
                 hashobj=None, hashvalues=None, permutations=None):
        if hashfunc is None:
            # Use built-in hash with FarmHash-like mixing
            self.hashfunc = self._default_hash
        else:
            self.hashfunc = hashfunc
        
        self.seed = seed
        
        if hashvalues is not None:
            # Restore from serialized values
            self.hashvalues = self._parse_hashvalues(hashvalues)
            self.permutations = permutations
        else:
            # Generate new permutation functions
            generator = np.random.RandomState(seed)
            # Generate random coefficients for linear hash functions: h(x) = (ax + b) mod p
            self.permutations = np.array([
                (generator.randint(1, _mersenne_prime, dtype=np.uint64),
                 generator.randint(0, _mersenne_prime, dtype=np.uint64))
                for _ in range(num_perm)
            ], dtype=np.uint64).T
            # Initialize hash values to maximum
            self.hashvalues = np.full(num_perm, _max_hash, dtype=np.uint64)

    def _default_hash(self, val):
        """Default hash function using SHA-1 (deterministic across processes)."""
        return _sha1_hash32(val)

    def _parse_hashvalues(self, values):
        """Parse hash values from various input types."""
        if isinstance(values, np.ndarray):
            return values.astype(np.uint64)
        return np.array(values, dtype=np.uint64)

    def update(self, b):
        """
        Update the MinHash with a new element.
        
        Args:
            b: A hashable element (will be converted to bytes if not already)
        """
        if isinstance(b, str):
            b = b.encode('utf-8')
        elif not isinstance(b, bytes):
            b = str(b).encode('utf-8')
        
        hv = self.hashfunc(b) & _max_hash  # Mask to 32 bits
        a, b_coef = self.permutations
        phv = np.bitwise_and((a * hv + b_coef) % _mersenne_prime, np.uint64(_max_hash))
        self.hashvalues = np.minimum(self.hashvalues, phv)

    def update_batch(self, items):
        """
        Update the MinHash with multiple elements efficiently using
        vectorized numpy operations.
        
        Args:
            items: Iterable of hashable elements (bytes or str)
        """
        # Hash all items on CPU
        hv_list = []
        for item in items:
            if isinstance(item, str):
                item = item.encode('utf-8')
            elif not isinstance(item, bytes):
                item = str(item).encode('utf-8')
            hv_list.append(self.hashfunc(item) & _max_hash)
        
        if not hv_list:
            return
        
        # Vectorized permutation application + min reduction
        hv = np.array(hv_list, dtype=np.uint64).reshape(-1, 1)  # (N, 1)
        a, b_coef = self.permutations  # each (num_perm,)
        phv = np.bitwise_and((hv * a + b_coef) % _mersenne_prime, np.uint64(_max_hash))  # (N, num_perm)
        self.hashvalues = np.minimum(self.hashvalues, phv.min(axis=0))

    def jaccard(self, other: 'MinHash') -> float:
        """
        Estimate Jaccard similarity with another MinHash.
        
        Args:
            other: Another MinHash object
            
        Returns:
            float: Estimated Jaccard similarity [0.0, 1.0]
        """
        if len(self) != len(other):
            raise ValueError("MinHash objects must have same number of permutations")
        return np.count_nonzero(self.hashvalues == other.hashvalues) / float(len(self))

    def count(self) -> int:
        """
        Estimate cardinality of the set using the MinHash signature.
        
        Uses the harmonic mean estimator.
        
        Returns:
            int: Estimated cardinality
        """
        # Normalize hash values to [0, 1]
        normalized = self.hashvalues.astype(np.float64) / float(_max_hash)
        # Use harmonic mean for estimation
        return int(len(self) / np.sum(normalized) - 1.0)

    def merge(self, other: 'MinHash') -> None:
        """
        Merge another MinHash into this one (union operation).
        
        Args:
            other: Another MinHash object to merge
        """
        if len(self) != len(other):
            raise ValueError("Cannot merge MinHash with different number of permutations")
        self.hashvalues = np.minimum(self.hashvalues, other.hashvalues)

    def copy(self) -> 'MinHash':
        """
        Create a copy of this MinHash.
        
        Returns:
            MinHash: A new MinHash with the same signature
        """
        return MinHash(
            num_perm=len(self),
            seed=self.seed,
            hashfunc=self.hashfunc,
            hashvalues=self.hashvalues.copy(),
            permutations=self.permutations
        )

    def clear(self) -> None:
        """Reset the MinHash to its initial state."""
        self.hashvalues = np.full(len(self), _max_hash, dtype=np.uint64)

    def is_empty(self) -> bool:
        """Check if the MinHash has been updated."""
        return np.all(self.hashvalues == _max_hash)

    def digest(self) -> np.ndarray:
        """Get the hash values as a numpy array."""
        return self.hashvalues

    def __len__(self) -> int:
        return len(self.hashvalues)

    def __eq__(self, other: 'MinHash') -> bool:
        return np.array_equal(self.hashvalues, other.hashvalues)

    def __hash__(self):
        return hash(tuple(self.hashvalues))

    def __getstate__(self):
        """Support pickling."""
        return {
            'hashvalues': self.hashvalues,
            'permutations': self.permutations,
            'seed': self.seed,
        }

    def __setstate__(self, state):
        """Support unpickling."""
        self.hashvalues = state['hashvalues']
        self.permutations = state['permutations']
        self.seed = state['seed']
        self.hashfunc = self._default_hash

    @classmethod
    def bulk(cls, b, **minhash_kwargs):
        """
        Compute MinHashes in bulk, reusing initialized state.
        
        Args:
            b: Iterable of lists of bytes/str, each list is
               hashed into one MinHash in the output.
            **minhash_kwargs: Keyword arguments for MinHash init.
            
        Returns:
            list[MinHash]: A list of computed MinHashes.
        """
        results = []
        # Create a template to reuse permutations
        template = cls(**minhash_kwargs)
        for item_list in b:
            mh = cls(
                num_perm=len(template),
                seed=template.seed,
                hashfunc=template.hashfunc,
                permutations=template.permutations
            )
            mh.update_batch(item_list)
            results.append(mh)
        return results

    @classmethod
    def union(cls, *minhashes) -> 'MinHash':
        """
        Create a new MinHash that is the union of multiple MinHashes.
        
        Args:
            *minhashes: MinHash objects to union
            
        Returns:
            MinHash: The union MinHash
        """
        if len(minhashes) == 0:
            raise ValueError("At least one MinHash is required")
        
        first = minhashes[0]
        result = first.copy()
        for mh in minhashes[1:]:
            result.merge(mh)
        return result


def jaccard_similarity(mh1: MinHash, mh2: MinHash) -> float:
    """
    Convenience function to compute Jaccard similarity.
    
    Args:
        mh1: First MinHash
        mh2: Second MinHash
        
    Returns:
        float: Estimated Jaccard similarity
    """
    return mh1.jaccard(mh2)


def containment(mh_query: MinHash, mh_target: MinHash, query_size: int) -> float:
    """
    Estimate containment of query set in target set.
    
    Uses the formula: containment(A, B) = |A ∩ B| / |A| ≈ jaccard(A, B) * |A ∪ B| / |A|
    
    For more accurate containment estimation, use LSH Ensemble.
    
    Args:
        mh_query: MinHash of query set
        mh_target: MinHash of target set
        query_size: Size of the query set
        
    Returns:
        float: Estimated containment [0.0, 1.0]
    """
    if query_size == 0:
        return 0.0
    jaccard = mh_query.jaccard(mh_target)
    # Estimate intersection size
    intersection = int(jaccard * (query_size + mh_target.count()) / (1 + jaccard))
    return min(1.0, intersection / query_size)
