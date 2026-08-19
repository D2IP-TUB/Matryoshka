"""Anytime-performance instrumentation for feature-selection algorithms.

Provides two small primitives that the five algorithms (forward / backward,
arda, autofeat, kitana, caafe) call from inside their outer iteration
loops:

  * :class:`BudgetClock` — wall-clock deadline tracker. ``expired`` flips
    to ``True`` once ``elapsed_s > seconds``. Algorithms poll it at safe
    iteration boundaries and break.
  * :class:`TrajectoryEmitter` — append-only CSV writer. One row per
    outer iteration with ``(iter, t_s, n_features, features_json,
    extra_json)``. ``flush()`` is called after every row so a SIGTERM from
    the outer subprocess wrapper at ``T_top + grace`` never loses more
    than the in-flight iteration.

Design pillars (from the Q1-Q5 design discussion):

  * Cooperative checkpointing inside the algorithm (this module) +
    subprocess SIGTERM net outside the algorithm (orchestrator).
  * One run per ``(algo, query)`` at the top-of-grid budget ``T_top``.
    Post-hoc indexing into multiple budgets reads the trajectory file
    and selects rows by ``t_s <= T_i``.
  * Truncate-and-restart on re-run. The :class:`TrajectoryEmitter`
    constructor opens the file in write mode (not append) so the
    previous trajectory is overwritten.

Algorithm integration contract:

  def run(..., budget_seconds: float | None = None,
              trajectory_dir: Path | None = None, **kw):
      clock = BudgetClock(budget_seconds).start()
      emitter = (TrajectoryEmitter(trajectory_dir, algo=self.__class__.__name__)
                 if trajectory_dir else None)
      if emitter:
          emitter.emit(0, 0.0, [])
      for iter_idx, ... in outer_loop:
          # ... compute one iteration, update `selected` ...
          if emitter:
              emitter.emit(iter_idx, clock.elapsed_s, list(selected))
          if clock.expired:
              break
      if emitter:
          emitter.close()

``budget_seconds=None`` disables the deadline check (algorithm runs to
completion). ``trajectory_dir=None`` disables emission. Both default
to ``None`` so the patched ``.run()`` keeps backwards-compatible
behaviour when neither is set.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

__all__ = ['BudgetClock', 'TrajectoryEmitter']


class BudgetClock:
    """Wall-clock deadline tracker with explicit ``start`` semantics."""

    __slots__ = ('seconds', '_t0')

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds
        self._t0: Optional[float] = None

    def start(self) -> 'BudgetClock':
        self._t0 = time.perf_counter()
        return self

    @property
    def elapsed_s(self) -> float:
        if self._t0 is None:
            return 0.0
        return time.perf_counter() - self._t0

    @property
    def expired(self) -> bool:
        if self.seconds is None:
            return False
        return self.elapsed_s > self.seconds


class TrajectoryEmitter:
    """Append-only CSV writer with per-row flush.

    Each call to :meth:`emit` appends one row and flushes the underlying
    file descriptor so the row is durable even if the process is killed
    by SIGTERM at ``T_top + grace``.

    The constructor truncates any existing trajectory file at
    ``<dir>/trajectory.csv`` to match the "truncate and restart"
    resume semantics agreed in Q4 of the design discussion.
    """

    HEADER = ('iter', 't_s', 'n_features', 'features_json', 'extra_json')

    def __init__(self, log_dir: Path | str, algo: Optional[str] = None,
                 filename: str = 'trajectory.csv') -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / filename
        # Truncate and write header. Subsequent emits open the file
        # again in append mode so we can flush per-row without holding a
        # long-lived descriptor.
        with self.path.open('w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(self.HEADER)
        self._algo = algo

    def emit(
        self,
        iter_idx: int,
        t_elapsed_s: float,
        features: Sequence[Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one row to the trajectory CSV.

        ``features`` may be any sequence of JSON-serialisable items. We
        do not enforce a particular type so each algorithm can record
        whatever uniquely identifies its selected subset (sketch keys,
        column names, join-path tuples, ...).
        """
        try:
            features_json = json.dumps(list(features), default=str)
        except TypeError:
            features_json = json.dumps([str(f) for f in features])
        extra_json = json.dumps(extra) if extra else ''
        with self.path.open('a', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([
                int(iter_idx),
                f'{float(t_elapsed_s):.6f}',
                len(features),
                features_json,
                extra_json,
            ])
            fh.flush()
            os.fsync(fh.fileno())

    def close(self) -> None:
        # No-op since we open per emit, but kept for symmetry with the
        # algorithm-side ``if emitter: emitter.close()`` pattern.
        pass
