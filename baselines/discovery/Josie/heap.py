"""
Heap data structures for top-k result management in Joise algorithm.
"""

import heapq
from typing import List, Tuple, Optional, Dict


class SearchResult:
    """
    Represents a single search result with set ID and overlap score.
    """
    
    def __init__(self, set_id: int, overlap: int):
        self.id = set_id
        self.overlap = overlap
    
    def __lt__(self, other):
        """Min-heap ordering: smaller overlap first, then larger ID (for determinism)."""
        if self.overlap != other.overlap:
            return self.overlap < other.overlap
        return self.id > other.id  # Larger ID = lower priority (removed first on tie)
    
    def __repr__(self):
        return f"SearchResult(id={self.id}, overlap={self.overlap})"


class SearchResultHeap:
    """
    Min-heap for maintaining top-k results.
    
    Uses a min-heap so that the smallest overlap is always at the root,
    making it easy to decide whether a new candidate should replace
    the current k-th best result.
    """
    
    def __init__(self):
        self.heap: List[SearchResult] = []
    
    def __len__(self):
        return len(self.heap)
    
    def push(self, result: SearchResult):
        """Add a result to the heap."""
        heapq.heappush(self.heap, result)
    
    def pop(self) -> SearchResult:
        """Remove and return the result with smallest overlap."""
        return heapq.heappop(self.heap)
    
    def peek(self) -> Optional[SearchResult]:
        """Return the result with smallest overlap without removing it."""
        if self.heap:
            return self.heap[0]
        return None
    
    def get_results(self) -> List[int]:
        """Return list of all set IDs in the heap (sorted by overlap desc, then ID asc)."""
        sorted_results = sorted(self.heap, key=lambda r: (-r.overlap, r.id))
        return [r.id for r in sorted_results]
    
    def get_results_with_scores(self) -> List[Tuple[int, int]]:
        """Return list of (set_id, overlap) tuples (sorted by overlap desc, then ID asc)."""
        sorted_results = sorted(self.heap, key=lambda r: (-r.overlap, r.id))
        return [(r.id, r.overlap) for r in sorted_results]
    
    def get_ordered_results(self) -> List[Tuple[int, int]]:
        """Return results sorted by overlap (descending)."""
        return sorted([(r.id, r.overlap) for r in self.heap], 
                      key=lambda x: -x[1])
    
    def show_heap(self, set_map: Dict):
        """Print heap contents with table/column names."""
        for result in self.heap:
            print(f"setID: {result.id}")
            if result.id + 1 in set_map:
                info = set_map[result.id + 1]
                print(f"  table: {info.get('table_name', 'unknown')}")
                print(f"  column: {info.get('column_name', 'unknown')}")
            print(f"  overlap: {result.overlap}")
    
    def order_heap(self):
        """Re-heapify after modifications."""
        heapq.heapify(self.heap)


def kth_overlap(heap: SearchResultHeap, k: int) -> int:
    """
    Get the k-th largest overlap value.
    
    If heap has fewer than k elements, returns 0.
    
    Args:
        heap: The search result heap
        k: Number of top results to consider
        
    Returns:
        The k-th largest overlap (or 0 if not enough results)
    """
    if len(heap) < k:
        return 0
    return heap.heap[0].overlap


def kth_overlap_after_push(heap: SearchResultHeap, k: int, overlap: int) -> int:
    """
    Estimate what the k-th overlap would be after pushing a new result.
    
    Used for cost-benefit analysis without actually modifying the heap.
    
    Args:
        heap: The search result heap
        k: Number of top results
        overlap: Overlap of potential new result
        
    Returns:
        Estimated k-th overlap after hypothetical push
    """
    h = heap.heap
    if len(h) < k - 1:
        return 0
    
    kth = h[0].overlap
    if overlap <= kth:
        return kth
    
    if k == 1:
        return overlap
    
    # Get the (k-1)th smallest element
    jth = h[1].overlap if k == 2 else min(h[1].overlap, h[2].overlap)
    return min(jth, overlap)


def push_candidate(heap: SearchResultHeap, k: int, set_id: int, overlap: int) -> bool:
    """
    Try to push a candidate into the top-k heap.
    
    If heap is full and candidate's overlap is not better than
    the current k-th best, the candidate is rejected.
    
    Args:
        heap: The search result heap
        k: Maximum number of results to maintain
        set_id: ID of the candidate set
        overlap: Overlap score of the candidate
        
    Returns:
        True if candidate was added, False otherwise
    """
    if len(heap) == k:
        if heap.heap[0].overlap >= overlap:
            return False
        heapq.heappop(heap.heap)
    
    heapq.heappush(heap.heap, SearchResult(set_id, overlap))
    heap.order_heap()
    return True


def copy_heap(heap: SearchResultHeap) -> SearchResultHeap:
    """Create a copy of the heap."""
    new_heap = SearchResultHeap()
    new_heap.heap = heap.heap.copy()
    return new_heap


def ordered_results(heap: SearchResultHeap) -> List[SearchResult]:
    """Return results sorted by overlap (descending)."""
    return sorted(heap.heap, key=lambda x: -x.overlap)
