import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any


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
            'config': getattr(record, 'config', None)
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
    try:
        log_path.mkdir()
    except FileExistsError:
        pass
    
    file_handler = logging.FileHandler(str(log_path / log_file))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
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