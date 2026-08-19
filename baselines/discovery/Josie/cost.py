"""
Cost model functions for Joise algorithm.

These functions estimate I/O costs for reading posting lists and sets.
"""

import math

# Cost model parameters (calibrated from benchmarks)
MIN_READ_COST = 1000000.0
READ_SET_COST_SLOPE = 1253.19054300781
READ_SET_COST_INTERCEPT = -9423326.99507381
READ_LIST_COST_SLOPE = 1661.93366983753
READ_LIST_COST_INTERCEPT = 1007857.48225696


def read_list_cost(length: int) -> float:
    """
    Estimate the cost of reading a posting list of given length.
    
    Args:
        length: Number of entries in the posting list
        
    Returns:
        Estimated I/O cost (normalized to millions)
    """
    cost = READ_LIST_COST_SLOPE * length + READ_LIST_COST_INTERCEPT
    if cost < MIN_READ_COST:
        cost = MIN_READ_COST
    return cost / 1000000.0


def read_set_cost(size: int) -> float:
    """
    Estimate the cost of reading a candidate set of given size.
    
    Args:
        size: Number of tokens in the set
        
    Returns:
        Estimated I/O cost (normalized to millions)
    """
    cost = READ_SET_COST_SLOPE * size + READ_SET_COST_INTERCEPT
    if cost < MIN_READ_COST:
        cost = MIN_READ_COST
    return cost / 1000000.0


def read_set_cost_reduction(size: int, truncation: int) -> float:
    """
    Estimate reduction in cost when truncating a set read.
    
    Args:
        size: Original size of the set
        truncation: Number of tokens that can be skipped
        
    Returns:
        Cost reduction from truncation
    """
    return read_set_cost(size) - read_set_cost(size - truncation)
