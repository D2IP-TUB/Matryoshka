"""
Joise (Join Set Intersection Estimation) Algorithm.

This module implements the core Joise algorithm for finding joinable tables
in a data lake using inverted index and cost-based optimization.
"""

import sys
from typing import List, Dict, Tuple, Optional, Any

from josie_util import (
    CandidateEntry, upperbound_overlap_unknown_candidate,
    next_batch_distinct_lists, prefix_length, read_lists_benefit_for_candidate,
    process_candidates_init, process_candidates_update, read_set_benefit
)
from heap import SearchResultHeap, kth_overlap, kth_overlap_after_push, push_candidate
from cost import read_list_cost
from common import next_distinct_list, overlap


# Batch size for reading posting lists
BATCH_SIZE = 5

# Budget for expensive estimation (num_candidate * num_estimation upper bound)
# Set to 0 for fast estimation only, sys.maxsize for always expensive estimation
EXPENSIVE_ESTIMATION_BUDGET = sys.maxsize


class ListEntry:
    """Entry from a posting list."""
    
    def __init__(self, set_id: int, match_position: int, size: int):
        self.ID = set_id
        self.MatchPosition = match_position
        self.Size = size


def process_query(query: List[str], raw_dict: Dict[str, List]) -> Tuple[List[int], List[int], List[int]]:
    """
    Process query tokens to get token IDs, group IDs, and frequencies.
    
    Args:
        query: List of raw tokens from query column
        raw_dict: Dictionary mapping raw tokens to [tid, gid, freq]
        
    Returns:
        Tuple of (token_ids, group_ids, frequencies)
    """
    tids = []
    gids = []
    freqs = []
    
    for token in query:
        if token in raw_dict:
            tid = raw_dict[token][0]
            gid = raw_dict[token][1]
            freq = raw_dict[token][2]
            tids.append(tid)
            gids.append(gid)
            freqs.append(freq)
    
    return tids, freqs, gids


def get_entries(token: int, posting_lists: Dict[str, List]) -> List[ListEntry]:
    """
    Get posting list entries for a given token.
    
    Args:
        token: Token ID
        posting_lists: Dictionary of posting lists
        
    Returns:
        List of ListEntry objects
    """
    entries = []
    token_str = str(token)
    
    if token_str in posting_lists:
        pls = posting_lists[token_str]
        for pl in pls:
            entry = ListEntry(pl[0], pl[1], pl[2])
            entries.append(entry)
    
    return entries


def set_tokens_suffix(integer_set: Dict[str, List[int]], set_id: int, start_pos: int) -> List[int]:
    """
    Get suffix of a set's tokens starting from given position.
    
    Args:
        integer_set: Dictionary of integer sets
        set_id: Set ID
        start_pos: Starting position
        
    Returns:
        List of token IDs in the suffix
    """
    set_id_str = str(set_id)
    if set_id_str in integer_set:
        return integer_set[set_id_str][start_pos:]
    return []


def search_joise(integer_set: Dict[str, List[int]],
                 posting_lists: Dict[str, List],
                 query: List[str],
                 raw_dict: Dict[str, List],
                 set_map: Dict[int, Dict[str, str]],
                 k: int,
                 ignore_self: bool = False,
                 query_id: int = -1,
                 batch_size: int = BATCH_SIZE,
                 expensive_budget: int = EXPENSIVE_ESTIMATION_BUDGET) -> List[int]:
    """
    Main Joise search algorithm.
    
    Finds top-k candidate sets with largest overlap with the query.
    Uses cost-based optimization to balance between reading more posting lists
    and probing candidate sets.
    
    Args:
        integer_set: Dictionary mapping set IDs to lists of token IDs
        posting_lists: Inverted index (token ID -> posting list)
        query: List of raw tokens from query column
        raw_dict: Dictionary mapping raw tokens to [tid, gid, freq]
        set_map: Mapping from set ID to table/column info
        k: Number of top results to return
        ignore_self: Whether to ignore the query set itself
        query_id: ID of query set (for self-ignore)
        batch_size: Number of posting lists to read in each batch
        expensive_budget: Budget for expensive estimation
        
    Returns:
        List of set IDs with highest overlap
    """
    # Process query tokens
    tokens, freqs, gids = process_query(query, raw_dict)
    
    if len(tokens) == 0:
        return []
    
    # Precompute cumulative read list costs
    read_list_costs = [0.0] * len(freqs)
    for i in range(len(freqs)):
        if i == 0:
            read_list_costs[i] = read_list_cost(freqs[i] + 1)
        else:
            read_list_costs[i] = read_list_costs[i - 1] + read_list_cost(freqs[i] + 1)
    
    query_size = len(tokens)
    counter: Dict[int, CandidateEntry] = {}  # Candidates from posting lists
    ignores: Dict[int, bool] = {}  # Ignored candidates
    
    if ignore_self and query_id >= 0:
        ignores[query_id] = True
    
    heap = SearchResultHeap()
    curr_batch_lists = batch_size
    p = query_size  # Prefix length threshold
    
    for i in range(query_size):
        # Prefix filtering: stop if we've reached the prefix length
        if i + 1 >= p:
            break
        
        token = tokens[i]
        num_skipped_result = next_distinct_list(tokens, gids, i)
        skipped_overlap = num_skipped_result[0]
        
        max_overlap_unseen = upperbound_overlap_unknown_candidate(query_size, i, skipped_overlap)
        
        # Early termination: when k-th overlap >= max for unseen and no candidates left
        if kth_overlap(heap, k) >= max_overlap_unseen and len(counter) == 0:
            break
        
        # Read posting list for current token
        entries = get_entries(token, posting_lists)
        
        # Process entries from posting list
        for entry in entries:
            # Skip ignored candidates
            if entry.ID in ignores:
                continue
            
            # Update existing candidate
            if entry.ID in counter:
                ce = counter[entry.ID]
                ce.update(entry.MatchPosition, skipped_overlap)
                continue
            
            # Skip if k-th overlap >= max for unseen candidates
            if kth_overlap(heap, k) >= max_overlap_unseen:
                continue
            
            # Add new candidate
            counter[entry.ID] = CandidateEntry.new_candidate_entry(
                entry.ID, entry.Size, entry.MatchPosition, i, skipped_overlap
            )
        
        # Stop at last list
        if i == query_size - 1:
            break
        
        # Continue reading if no candidates, fewer than k candidates, or still in batch
        if (len(counter) == 0 or 
            (len(counter) < k and len(heap) < k) or
            curr_batch_lists > 0):
            curr_batch_lists -= 1
            continue
        
        # Reset batch counter
        curr_batch_lists = batch_size
        
        # Find next batch end index
        next_batch_end_index = next_batch_distinct_lists(tokens, gids, i, batch_size)
        
        # Compute cost of reading next batch of posting lists
        merge_lists_cost = read_list_costs[next_batch_end_index] - read_list_costs[i]
        
        # Process candidates to estimate benefits
        merge_lists_benefit, num_with_benefit, candidates = process_candidates_init(
            query_size, i + 1, next_batch_end_index, kth_overlap(heap, k),
            batch_size, counter, ignores
        )
        
        # If no qualified candidates or none with benefit, continue reading lists
        if num_with_benefit == 0 or len(candidates) == 0:
            continue
        
        # Sort candidates by estimated overlap (descending)
        candidates.sort(key=lambda c: c.estimated_overlap, reverse=True)
        
        # Track estimation budget
        prev_kth_overlap = kth_overlap(heap, k)
        num_candidate_expensive = 0
        fast_estimate = False
        fast_estimate_kth_overlap = 0
        
        # Greedily select candidates to probe
        for candidate in candidates:
            if candidate is None:
                continue
            
            kth = kth_overlap(heap, k)
            
            # Stop when estimated overlap <= k-th best
            if candidate.estimated_overlap <= kth:
                break
            
            # Always read when heap not full
            if len(heap) >= k:
                num_candidate_expensive += 1
                
                # Switch to fast estimation if budget exceeded
                if not fast_estimate and num_candidate_expensive * len(candidates) > expensive_budget:
                    fast_estimate = True
                    fast_estimate_kth_overlap = prev_kth_overlap
                
                # Update merge benefit if not using fast estimation
                if not fast_estimate:
                    merge_lists_benefit = process_candidates_update(
                        kth, candidates, counter, ignores
                    )
                
                # Estimate benefit of probing this set
                probe_set_benefit = read_set_benefit(
                    query_size, kth,
                    kth_overlap_after_push(heap, k, candidate.estimated_overlap),
                    candidates, read_list_costs, fast_estimate
                )
                probe_set_cost = candidate.estimated_cost
                
                # If probing is not better than reading more lists, stop
                if probe_set_benefit - probe_set_cost < merge_lists_benefit - merge_lists_cost:
                    break
            
            # Reduce merge benefit for fast estimation
            if fast_estimate or (num_candidate_expensive + 1) * len(candidates) > expensive_budget:
                merge_lists_benefit -= read_lists_benefit_for_candidate(
                    candidate, fast_estimate_kth_overlap
                )
            
            # Mark candidate as read
            candidate.read = True
            ignores[candidate.id] = True
            
            if candidate.id in counter:
                del counter[candidate.id]
            
            # Skip if maximum overlap is not better than k-th
            if candidate.maximum_overlap <= kth:
                continue
            
            # Compute actual overlap
            if candidate.suffix_length() > 0:
                suffix_tokens = set_tokens_suffix(
                    integer_set, candidate.id,
                    candidate.latest_match_position + 1
                )
                suffix_overlap = overlap(suffix_tokens, tokens[i + 1:])
                total_overlap = suffix_overlap + candidate.partial_overlap
            else:
                total_overlap = candidate.partial_overlap
            
            # Update state
            prev_kth_overlap = kth
            p = prefix_length(query_size, prev_kth_overlap)
            
            # Push to heap
            push_candidate(heap, k, candidate.id, total_overlap)
    
    # Process remaining candidates from counter
    for ce in counter.values():
        push_candidate(heap, k, ce.id, ce.partial_overlap)
    
    return heap.get_results()


def read_query_id(set_map: Dict[int, Dict[str, str]], 
                  table_name: str, column_name: str) -> int:
    """
    Get the set ID for a query column.
    
    Args:
        set_map: Mapping from set ID to table/column info
        table_name: Name of the table
        column_name: Name of the column
        
    Returns:
        Set ID - 1 (0-indexed), or -1 if not found
    """
    for key, value in set_map.items():
        if value.get('table_name') == table_name and value.get('column_name') == column_name:
            return key - 1
    return -1
