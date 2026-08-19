"""Search procedures over the candidate feature pool.

``greedy`` implements the forward selection of Section 6 (``ForwardSelection``)
and its backward counterpart (``BackwardElimination``); ``stepwise`` implements
the batched incremental variant; ``lasso`` implements the L1 selectors used as
internal comparison points.
"""
