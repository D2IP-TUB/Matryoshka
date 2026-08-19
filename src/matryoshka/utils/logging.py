"""Structured logging for the discovery pipeline.

Each phase of a run emits one JSON record naming the phase and its wall-clock
time, so a run can be parsed into the per-step runtime breakdown reported in
Section 7 without instrumenting the caller.

Log files are written under :func:`default_log_dir`, which resolves to
``$MATRYOSHKA_LOG_DIR`` if set and otherwise to ``.matryoshka/logs`` in the
working directory. Nothing is ever written inside the installed package.
"""
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

_LOG_DIR_ENV = 'MATRYOSHKA_LOG_DIR'
_DEFAULT_LOG_SUBDIR = Path('.matryoshka') / 'logs'


def default_log_dir() -> Path:
    """Directory for log files: ``$MATRYOSHKA_LOG_DIR`` or ``./.matryoshka/logs``."""
    override = os.environ.get(_LOG_DIR_ENV)
    return Path(override).expanduser() if override else Path.cwd() / _DEFAULT_LOG_SUBDIR


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'strategy': getattr(record, 'strategy', None),
            'model': getattr(record, 'model', None),
            'total_features': getattr(record, 'total_features', None),
            'fitted_features': getattr(record, 'fitted_features', None),
            'joinability': getattr(record, 'joinability', None),
            'score': getattr(record, 'score', None),
            'runtime': getattr(record, 'runtime', None),
            'augmentation_plan': getattr(record, 'augmentation_plan', None),
            'config': getattr(record, 'config', None),
            'n_iters': getattr(record, 'n_iters', None),
            'computation_s': getattr(record, 'computation_s', None),
            'ray_overhead_s': getattr(record, 'ray_overhead_s', None),
        }
        return json.dumps(log_record)


def setup_logger(name: str, log_dir: str, log_file: str, level: int = logging.DEBUG, silent: bool = False, json_logging: bool = False) -> logging.Logger:
    '''
    Sets up and returns a logger with the specified name and file.
    '''
    if json_logging:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(log_path / log_file))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    # ``setup_logger`` is called once per experiment combination, but the
    # logger object is shared across calls because it is keyed by name.
    # Without resetting we keep accumulating handlers, which is what
    # produces the duplicate (and triplicate, ...) lines in run.out.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    # Don't propagate to the root logger either; otherwise a parallel
    # ``logging.basicConfig`` somewhere else re-emits each record as
    # ``INFO:join_selection:...``.
    logger.propagate = False
    if silent:
        logger.addHandler(logging.NullHandler())
    else:
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def store_artifact(artifact: Any, path: str, verbose: bool = False) -> None:
    '''
    Stores the artifact at the specified path
    '''
    if verbose:
        directory = path[:path.rindex('/')]
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(artifact, f)
