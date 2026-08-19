"""Feature selection over Gram matrix sketches.

``sketch_processing`` assembles candidate sketches from the index and joins
them against the query table sketch; ``models`` implements the linear proxy
models whose closed-form solutions score a candidate feature set;
``algorithms`` holds the greedy search procedures driven by those scores.
"""
