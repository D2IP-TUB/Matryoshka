"""Adapters that expose each competing augmentation system behind one interface.

Every class implements ``run(**params)`` and returns
``(augmented_table, discovery_seconds, selection_seconds, augmentation_plan)``.
They are reached through :func:`matryoshka.baselines.resolve_baseline`, which
imports them lazily by name.
"""
