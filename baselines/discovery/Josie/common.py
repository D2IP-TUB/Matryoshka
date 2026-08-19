"""
Common utility functions for Joise algorithm.
"""

import math
from typing import List, Tuple


# Global variable for total number of sets (used in pruning power calculation)
total_number_of_sets = 1.0


def init_total_sets(num_sets: int):
    """Initialize the total number of sets in the data lake."""
    global total_number_of_sets
    total_number_of_sets = float(num_sets)


def pruning_power_ub(freq: int, k: int) -> float:
    """
    Calculate upper bound on pruning power.
    
    Args:
        freq: Token frequency
        k: Number of top results
        
    Returns:
        Upper bound on pruning power (BM25-like score)
    """
    return math.log(((min(k, freq) + 0.5) * (total_number_of_sets - k - freq + min(k, freq) + 0.5)) /
                    (((max(0, k - freq) + 0.5) * (max(freq - k, 0) + 0.5))))


def inverse_set_frequency(freq: int) -> float:
    """
    Calculate inverse set frequency (ISF).
    
    Args:
        freq: Token frequency across sets
        
    Returns:
        ISF score
    """
    return math.log(total_number_of_sets / freq)


def next_distinct_list(tokens: List[int], gids: List[int], curr_list_index: int) -> Tuple[int, int]:
    """
    Find the next distinct posting list (skipping duplicates based on group ID).
    
    Args:
        tokens: List of token IDs
        gids: List of group IDs (for identifying duplicates)
        curr_list_index: Current position in the token list
        
    Returns:
        Tuple of (next_index, num_skipped)
    """
    if curr_list_index == len(tokens) - 1:
        return len(tokens), 0
    
    num_skipped = 0
    for i in range(curr_list_index + 1, len(tokens)):
        if i < len(tokens) - 1 and gids[i + 1] == gids[i]:
            num_skipped += 1
            continue
        return i, num_skipped
    return len(tokens), 0


def overlap(set_tokens: List[int], query_tokens: List[int]) -> int:
    """
    Calculate overlap between two sorted token lists.
    
    Uses merge-based intersection for O(n+m) complexity.
    
    Args:
        set_tokens: Sorted list of token IDs from candidate set
        query_tokens: Sorted list of token IDs from query
        
    Returns:
        Number of overlapping tokens
    """
    i, j = 0, 0
    overlap_count = 0
    
    while i < len(query_tokens) and j < len(set_tokens):
        d = query_tokens[i] - set_tokens[j]
        if d == 0:
            overlap_count += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:  # d > 0
            j += 1
    
    return overlap_count


def overlap_simple(list1: List, list2: List) -> int:
    """
    Calculate overlap using simple list comprehension.
    
    Useful when lists are not sorted or contain non-integer tokens.
    
    Args:
        list1: First list of tokens
        list2: Second list of tokens
        
    Returns:
        Number of overlapping elements
    """
    return len([v for v in list1 if v in list2])


def overlap_and_update_counts(set_tokens: List[int], query_tokens: List[int], counts: List[int]) -> int:
    """
    Calculate overlap while updating frequency counts.
    
    Args:
        set_tokens: Sorted list of token IDs from candidate set
        query_tokens: Sorted list of token IDs from query
        counts: Frequency counts to update
        
    Returns:
        Number of overlapping tokens
    """
    i, j = 0, 0
    overlap_count = 0
    
    while i < len(query_tokens) and j < len(set_tokens):
        d = query_tokens[i] - set_tokens[j]
        if d == 0:
            counts[i] -= 1
            overlap_count += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:  # d > 0
            j += 1
    
    return overlap_count
