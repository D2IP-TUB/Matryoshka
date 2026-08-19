"""
Utility classes and functions for Joise candidate management.

This module contains the CandidateEntry class for tracking candidate sets
during the search process, and related utility functions.
"""

from typing import List, Dict, Tuple, Optional
from common import next_distinct_list
from cost import read_set_cost


class CandidateEntry:
    """
    Represents a candidate set being tracked during Joise search.
    
    Tracks partial overlap, position information, and various estimates
    used for cost-benefit analysis.
    """
    
    def __init__(self, set_id: int, size: int, 
                 candidate_current_position: int, 
                 query_current_position: int,
                 skipped_overlap: int):
        """
        Initialize a new candidate entry.
        
        Args:
            set_id: Unique identifier for the candidate set
            size: Total number of tokens in the candidate set
            candidate_current_position: Position of first match in candidate
            query_current_position: Position of first match in query
            skipped_overlap: Number of skipped overlapping tokens (from duplicates)
        """
        self.id = set_id
        self.size = size
        self.first_match_position = candidate_current_position
        self.latest_match_position = candidate_current_position
        self.query_first_match_position = query_current_position
        self.partial_overlap = skipped_overlap + 1
        
        # Estimated values (computed later)
        self.maximum_overlap = 0
        self.estimated_overlap = 0
        self.estimated_cost = 0.0
        self.estimated_next_upperbound = 0
        self.estimated_next_truncation = 0
        self.read = False
    
    @staticmethod
    def new_candidate_entry(set_id: int, size: int,
                            candidate_current_position: int,
                            query_current_position: int,
                            skipped_overlap: int) -> 'CandidateEntry':
        """Factory method to create a new candidate entry."""
        return CandidateEntry(set_id, size, candidate_current_position,
                              query_current_position, skipped_overlap)
    
    def update(self, candidate_current_position: int, skipped_overlap: int):
        """
        Update candidate when a new matching token is found.
        
        Args:
            candidate_current_position: Position of new match in candidate
            skipped_overlap: Number of skipped overlapping tokens
        """
        self.latest_match_position = candidate_current_position
        self.partial_overlap += skipped_overlap + 1
    
    def upperbound_overlap(self, query_size: int, query_current_position: int) -> int:
        """
        Calculate upper bound on total overlap (Formula 6 in paper).
        
        Args:
            query_size: Total size of query
            query_current_position: Current position in query processing
            
        Returns:
            Maximum possible overlap
        """
        self.maximum_overlap = self.partial_overlap + min(
            query_size - query_current_position - 1,
            self.size - self.latest_match_position - 1
        )
        return self.maximum_overlap
    
    def est_overlap(self, query_size: int, query_current_position: int) -> int:
        """
        Estimate total overlap based on partial overlap (Formula 4 in paper).
        
        Uses sampling-based estimation from observed prefix.
        
        Args:
            query_size: Total size of query
            query_current_position: Current position in query processing
            
        Returns:
            Estimated total overlap
        """
        prefix_length = query_current_position + 1 - self.query_first_match_position
        remaining_length = query_size - self.query_first_match_position
        
        self.estimated_overlap = int(
            float(self.partial_overlap) / float(prefix_length) * float(remaining_length)
        )
        
        # Bound by maximum possible overlap
        self.estimated_overlap = min(
            self.estimated_overlap, 
            self.upperbound_overlap(query_size, query_current_position)
        )
        return self.estimated_overlap
    
    def est_cost(self) -> float:
        """
        Estimate I/O cost of reading this candidate set.
        
        Returns:
            Estimated cost
        """
        self.estimated_cost = read_set_cost(self.suffix_length())
        return self.estimated_cost
    
    def est_truncation(self, query_size: int, query_current_position: int,
                       query_next_position: int) -> int:
        """
        Estimate how much of the set can be truncated (skipped).
        
        Args:
            query_size: Total size of query
            query_current_position: Current position
            query_next_position: Position after reading next batch
            
        Returns:
            Estimated number of tokens that can be truncated
        """
        self.estimated_next_truncation = int(
            float(query_next_position - query_current_position) /
            float(query_size - self.query_first_match_position) *
            float(self.size - self.first_match_position)
        )
        return self.estimated_next_truncation
    
    def est_next_overlap_upperbound(self, query_size: int,
                                    query_current_position: int,
                                    query_next_position: int) -> int:
        """
        Estimate upper bound on overlap after reading next batch (Formula 9-11).
        
        Args:
            query_size: Total size of query
            query_current_position: Current position
            query_next_position: Position after reading next batch
            
        Returns:
            Estimated upper bound on overlap
        """
        query_jump_length = query_next_position - query_current_position
        query_prefix_length = query_current_position + 1 - self.query_first_match_position
        
        # Estimate additional overlap from the jump
        additional_overlap = int(
            float(self.partial_overlap) / float(query_prefix_length) * float(query_jump_length)
        )
        
        # Estimate next matching position (Formula 10)
        next_latest_match_position = int(
            float(query_jump_length) /
            float(query_size - self.query_first_match_position) *
            float(self.size - self.first_match_position)
        ) + self.latest_match_position
        
        # Compute estimated upper bound (Formula 11)
        self.estimated_next_upperbound = self.partial_overlap + additional_overlap + min(
            query_size - query_next_position - 1,
            self.size - next_latest_match_position - 1
        )
        
        return self.estimated_next_upperbound
    
    def suffix_length(self) -> int:
        """Get the remaining length of the candidate set."""
        return self.size - self.latest_match_position - 1
    
    def check_min_sample_size(self, query_current_position: int, batch_size: int) -> bool:
        """
        Check if we have enough samples for reliable estimation.
        
        Args:
            query_current_position: Current position in query
            batch_size: Minimum sample size required
            
        Returns:
            True if sample size is sufficient
        """
        return (query_current_position - self.query_first_match_position + 1) > batch_size


def upperbound_overlap_unknown_candidate(query_size: int, query_current_position: int,
                                         prefix_overlap: int) -> int:
    """
    Calculate overlap upper bound for candidates not yet seen.
    
    Args:
        query_size: Total size of query
        query_current_position: Current position in query processing
        prefix_overlap: Number of overlapping tokens in prefix
        
    Returns:
        Upper bound on overlap for unknown candidates
    """
    return query_size - query_current_position + prefix_overlap


def next_batch_distinct_lists(tokens: List[int], gids: List[int],
                              curr_index: int, batch_size: int) -> int:
    """
    Find the end index of the next batch of distinct posting lists.
    
    Args:
        tokens: List of token IDs
        gids: List of group IDs
        curr_index: Current position
        batch_size: Number of distinct lists in a batch
        
    Returns:
        End index of the next batch
    """
    n = 0
    next_index = next_distinct_list(tokens, gids, curr_index)[0]
    
    while next_index < len(tokens):
        curr_index = next_index
        n += 1
        if n == batch_size:
            break
        next_index = next_distinct_list(tokens, gids, curr_index)[0]
    
    return curr_index


def prefix_length(query_size: int, kth_overlap: int) -> int:
    """
    Calculate required prefix length for given overlap threshold.
    
    Args:
        query_size: Total size of query
        kth_overlap: Current k-th best overlap
        
    Returns:
        Number of posting lists to read
    """
    if kth_overlap == 0:
        return query_size
    return query_size - kth_overlap + 1


def read_lists_benefit_for_candidate(ce: CandidateEntry, kth_overlap: int) -> float:
    """
    Estimate benefit of reading additional lists for a candidate (Formula 12).
    
    Args:
        ce: Candidate entry
        kth_overlap: Current k-th best overlap
        
    Returns:
        Estimated benefit
    """
    if kth_overlap >= ce.estimated_next_upperbound:
        return ce.estimated_cost
    return ce.estimated_cost - read_set_cost(ce.suffix_length() - ce.estimated_next_truncation)


def process_candidates_init(query_size: int, query_current_position: int,
                           next_batch_end_index: int, kth_overlap: int,
                           min_sample_size: int, candidates: Dict[int, CandidateEntry],
                           ignores: Dict[int, bool]) -> Tuple[float, int, List[CandidateEntry]]:
    """
    Process unread candidates to get qualified candidates and compute benefits.
    
    Args:
        query_size: Total size of query
        query_current_position: Current position (1-indexed for this function)
        next_batch_end_index: End index of next batch of posting lists
        kth_overlap: Current k-th best overlap
        min_sample_size: Minimum samples needed for estimation
        candidates: Dictionary of candidate entries
        ignores: Dictionary of ignored candidates
        
    Returns:
        Tuple of (read_lists_benefit, num_with_benefit, qualified_candidates)
    """
    read_lists_benefit = 0.0
    num_with_benefit = 0
    qualified = []
    to_be_removed = []
    
    for ce in candidates.values():
        # Compute upper bound overlap
        ce.upperbound_overlap(query_size, query_current_position)
        
        # Mark candidates with upper bound <= kth for removal
        if kth_overlap >= ce.maximum_overlap:
            to_be_removed.append(ce.id)
            ignores[ce.id] = True
            continue
        
        # Skip candidates without enough samples for reliable estimation
        if not ce.check_min_sample_size(query_current_position, min_sample_size):
            continue
        
        # Compute estimates
        ce.est_cost()
        ce.est_overlap(query_size, query_current_position)
        ce.est_truncation(query_size, query_current_position, next_batch_end_index)
        ce.est_next_overlap_upperbound(query_size, query_current_position, next_batch_end_index)
        
        # Compute read list benefit
        read_lists_benefit += read_lists_benefit_for_candidate(ce, kth_overlap)
        
        # Add qualified candidate
        qualified.append(ce)
        if ce.estimated_overlap > kth_overlap:
            num_with_benefit += 1
    
    # Remove disqualified candidates
    for set_id in to_be_removed:
        del candidates[set_id]
    
    return read_lists_benefit, num_with_benefit, qualified


def process_candidates_update(kth_overlap: int, candidates: List[Optional[CandidateEntry]],
                             counter: Dict[int, CandidateEntry],
                             ignores: Dict[int, bool]) -> float:
    """
    Update candidate processing after heap changes.
    
    Args:
        kth_overlap: Current k-th best overlap
        candidates: List of candidate entries (may contain None)
        counter: Dictionary of all candidates
        ignores: Dictionary of ignored candidates
        
    Returns:
        Updated read lists benefit
    """
    read_lists_benefit = 0.0
    
    for j, ce in enumerate(candidates):
        if ce is None or ce.read:
            continue
        
        if ce.maximum_overlap <= kth_overlap:
            # Mark as eliminated
            candidates[j] = None
            if ce.id in counter:
                del counter[ce.id]
            ignores[ce.id] = True
        
        # Compute read list benefit for qualified candidate
        read_lists_benefit += read_lists_benefit_for_candidate(ce, kth_overlap)
    
    return read_lists_benefit


def read_set_benefit(query_size: int, kth_overlap: int, kth_overlap_after_push: int,
                     candidates: List[Optional[CandidateEntry]],
                     read_list_costs: List[float], fast: bool) -> float:
    """
    Calculate benefit of reading a candidate set.
    
    Args:
        query_size: Total size of query
        kth_overlap: Current k-th best overlap
        kth_overlap_after_push: K-th overlap after adding candidate
        candidates: List of candidate entries
        read_list_costs: Cumulative costs of reading posting lists
        fast: Whether to use fast estimation
        
    Returns:
        Estimated benefit
    """
    benefit = 0.0
    
    if kth_overlap_after_push <= kth_overlap:
        return benefit
    
    p0 = prefix_length(query_size, kth_overlap)
    p1 = prefix_length(query_size, kth_overlap_after_push)
    
    benefit += read_list_costs[p0 - 1] - read_list_costs[p1 - 1]
    
    if fast:
        return benefit
    
    for ce in candidates:
        if ce is None or ce.read:
            continue
        if ce.maximum_overlap <= kth_overlap_after_push:
            # Add benefit from eliminating the candidate
            benefit += ce.estimated_cost
    
    return benefit
