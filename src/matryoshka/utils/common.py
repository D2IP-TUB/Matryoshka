'''
Common utiltiy functions for different modules
'''
import cProfile
import gc
import gzip
import os
import tempfile
import time
import unicodedata
from io import BytesIO
from typing import Callable, Optional

import numpy as np
import polars as pl
from scipy.sparse import csc_matrix

from .system_monitoring import MemoryThresholdCalculator


def profile(func):
    def wrapper(*args, **kwargs):
        datafn = f'{func.__name__}.profile'
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval
    return wrapper


def measure_runtime(get_logger: Callable, condition=lambda self, *args, **kwargs: True):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not condition(self, *args, **kwargs):  # Check if timing should be skipped
                return func(self, *args, **kwargs)

            logger = get_logger(self)
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            end = time.perf_counter()
            execution_time = end - start

            logger.info(f'{func.__name__} executed in {execution_time:.6f} seconds')
            return result
        return wrapper
    return decorator


def read_tsv_from_archive(gz_file):
    with gzip.open(BytesIO(gz_file), 'rt') as f:
        return pl.scan_csv(f, separator='\t', ignore_errors=True, low_memory=True)


def semiring_aggregates(group: np.ndarray | csc_matrix, features_locs: list[list[int]]) -> tuple[int, list[float], list[float]]:
    '''
    Computes semi-ring aggregates - a sketch of cofactor matrix

    Parameters:
    ----------
    group: `np.ndarray`
        Group of rows from a table which belong to the key (`value` at iteration of `create_index` method)

    Returns:
    -------
    `tuple`: Count, sum, dot product upper-triangular values
    '''
    count_ = count_agg(group, features_locs)
    sum_ = sum_agg(group, features_locs)
    squares, cofactors = dot_product_agg(group, features_locs)

    return count_, sum_, squares, cofactors


def count_agg(group: np.ndarray | csc_matrix, features_locs: list[list[int]]) -> int:
    '''
    Computes the count of non-NaN values in a group

    Parameters:
    ----------
    group: `np.ndarray`
        Group of rows from a table which belong to the key (`value` at iteration of `create_index` method)
    
    Returns:
    -------
    `int`: Count of non-NaN values
    '''
    counts = [np.repeat([1], group[:, idx].shape[0], axis=0) for idx in features_locs]
    return counts


def sum_agg(group: np.ndarray | np.matrix, features_locs: list[list[int]]) -> list[float]:
    '''
    Computes the sum of numeric values in a group

    Parameters:
    ----------
    group: `np.ndarray`
        Group of rows from a table which belong to the key (`value` at iteration of `create_index` method)
    
    Returns:
    -------
    `np.ndarray`: Column-wise sum of numeric values
    '''
    sums = [group[:, idx] for idx in features_locs]

    return sums


def dot_product_agg(group: np.ndarray | np.matrix, features_locs: list[list[int]]) -> tuple[list[list[float]]]:
    '''
    Computes the self dot product of a group of rows

    Parameters:
    ----------
    group: `np.ndarray`
        Group of rows from a table which belong to the key (`value` at iteration of `create_index` method)
    
    Returns:
    -------
    `list[list[float]]`: Upper-triangular values of dot product matrix
    '''
    outer_products, memory_info = compute_outer_products_adaptive(group)
    diag_els = []
    cofactors_els = []
    for idx in features_locs:
        if len(idx) == 1:
            slice_ = outer_products[:, idx][:, :, idx]
            diag = np.diagonal(slice_, axis1=1, axis2=2)
            cofactors = np.repeat([[np.nan]], slice_.shape[0], axis=0)
        else:
            if memory_info['peak_memory_mb']*5 < memory_info['usable_memory_gb']*1024:
                cofactors, diag = extract_cofactors(outer_products, idx)
            elif memory_info['peak_memory_mb']*3 < memory_info['usable_memory_gb']*1024:
                cofactors, diag = np.array([]), np.array([])
            else:
                cofactors, diag = extract_cofactors_chunked(outer_products, idx, chunk_size=1000)

        if np.isnan(cofactors).all():
            row_cofactors = np.expand_dims(cofactors, 1)
        else:
            if len(cofactors.shape) == 2:
                row_cofactors = np.expand_dims(cofactors, 2)
            else:
                row_cofactors = cofactors
        del cofactors
        gc.collect()

        diag_els.append(diag)
        cofactors_els.append(row_cofactors)
    return diag_els, cofactors_els


def compute_outer_products_adaptive(
    group: np.ndarray,
    container_memory_gb: Optional[float] = None,
    safety_factor: float = 0.6,
    memmap_filename: Optional[str] = None,
    online: bool = False
) -> np.ndarray:
    B, D = group.shape
    # Determine processing method
    calculator = MemoryThresholdCalculator(container_memory_gb, safety_factor)
    use_memmap, memory_info = calculator.should_use_memmap(B, D)

    if online and use_memmap:
        raise RuntimeError("Online processing with memmap is not supported.")

    if use_memmap:
        # Use memory-mapped approach
        if memmap_filename is None:
            _tmp = os.environ.get('MATRYOSHKA_TMP_DIR', '/app/data/tmp')
            os.makedirs(_tmp, exist_ok=True)
            memmap_filename = f'{_tmp}/outer_products_{os.getpid()}.dat'

        result = np.memmap(memmap_filename, dtype=np.float64, mode='w+', shape=(B, D, D))

        # Process in chunks to avoid memory issues
        chunk_size = min(1000, max(1, int(calculator.available_memory_bytes * 0.1 / (D * D * 8))))

        for start_idx in range(0, B, chunk_size):
            end_idx = min(start_idx + chunk_size, B)

            for i in range(start_idx, end_idx):
                v = group[i]
                result[i] = np.outer(v, v)

            # Periodically flush to disk
            if start_idx % (chunk_size * 5) == 0:
                result.flush()

        result.flush()
        return result, memory_info
    else:
        # Use in-memory approach
        result = np.einsum('bi,bj->bij', group, group)
        return result, memory_info


def extract_cofactors(outer_products, idx):
    batch_idx = np.arange(outer_products.shape[0])
    slice_ = outer_products[np.ix_(batch_idx, idx, idx)]
    diag = np.diagonal(slice_, axis1=1, axis2=2)

    batch, n, _ = slice_.shape
    diag_mask = ~np.eye(n, dtype=bool)
    cofactors = slice_[:, diag_mask].reshape(batch, n, n-1)
    cofactors = cofactors.transpose(0, 2, 1)

    return cofactors, diag


def extract_cofactors_chunked(outer_products, idx, chunk_size=1000, output_dir=None, keep_temp=False):
    n_batches = outer_products.shape[0]
    n = len(idx)
    diag_mask = ~np.eye(n, dtype=bool)

    # Create temporary directory
    if output_dir is None:
        _tmp = os.environ.get('MATRYOSHKA_TMP_DIR', '/app/data/tmp')
        os.makedirs(_tmp, exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir=_tmp)
    else:
        temp_dir = output_dir
        os.makedirs(temp_dir, exist_ok=True)

    cofactors_file = os.path.join(temp_dir, 'cofactors_temp.npy')
    diag_file = os.path.join(temp_dir, 'diag_temp.npy')

    try:
        # First pass: determine output shapes and create memory-mapped files
        first_chunk = outer_products[0:1][:, idx][:, :, idx]
        cofactors_shape_per_batch = first_chunk[:, diag_mask].reshape(1, n, n-1).transpose(0, 2, 1).shape[1:]

        # Calculate final shapes
        if len(cofactors_shape_per_batch) == 2:
            final_cofactors_shape = (n_batches,) + cofactors_shape_per_batch
        else:
            final_cofactors_shape = (n_batches, n-1)

        final_diag_shape = (n_batches, n)

        # Create memory-mapped files
        cofactors_mmap = np.lib.format.open_memmap(
            cofactors_file, mode='w+',
            dtype=outer_products.dtype,
            shape=final_cofactors_shape
        )

        diag_mmap = np.lib.format.open_memmap(
            diag_file, mode='w+',
            dtype=outer_products.dtype,
            shape=final_diag_shape
        )

        # Process in chunks and write directly to disk
        for i in range(0, n_batches, chunk_size):
            end = min(i + chunk_size, n_batches)

            # Process chunk
            batch_idx = np.arange(i, end)
            chunk = outer_products[np.ix_(batch_idx, idx, idx)]

            # Extract diagonal and write to disk
            diag_chunk = np.diagonal(chunk, axis1=1, axis2=2)
            diag_mmap[i:end] = diag_chunk

            # Extract off-diagonal elements and write to disk
            cofactors_chunk = chunk[:, diag_mask].reshape(-1, n, n-1).transpose(0, 2, 1)
            if len(cofactors_shape_per_batch) == 1:
                cofactors_chunk = cofactors_chunk.squeeze(axis=1)
            cofactors_mmap[i:end] = cofactors_chunk

            # Explicitly delete chunk to free memory
            del chunk, diag_chunk, cofactors_chunk

        # Flush to disk
        cofactors_mmap.flush()
        diag_mmap.flush()

        del outer_products
        gc.collect()

        cofactors = np.array(cofactors_mmap)
        diag = np.array(diag_mmap)

        del cofactors_mmap, diag_mmap
        gc.collect()

        return cofactors, diag

    finally:
        # Cleanup temporary files unless keep_temp is True
        if not keep_temp:
            try:
                if os.path.exists(cofactors_file):
                    os.remove(cofactors_file)
                if os.path.exists(diag_file):
                    os.remove(diag_file)
                if output_dir is None:
                    os.rmdir(temp_dir)
            except:
                pass


def construct_dot_product_matrix(values: list[float], triu: bool = True) -> np.ndarray:
    '''
    Constructs the dot product matrix from its upper-triangular values

    Parameters:
    ----------
    values: `list[float]`
        Upper-triangular values of the dot product matrix

    triu: `bool`
        If True, the matrix is upper-triangular, otherwise it is complete

    Returns:
    -------
    `np.ndarray`: Upper-triangualr dot product matrix
    '''
    # compute the shape of the matrix given its upper-triangular values
    n = int((-1 + np.sqrt(1 + 8 * len(values))) / 2)
    matrix = np.zeros((n, n))
    index = np.triu_indices_from(matrix)
    # compute value-counts for each row to know how many values to assign
    rows_vc = np.unique(index[0], return_counts=True)

    # counter to slice the values list while assigning values to the upper-triangular matrix
    processed_rows = 0
    # starting non-zero column index for each row
    starting_col = 0
    for row_idx, row_count in zip(*rows_vc):
        matrix[row_idx, starting_col:] = values[processed_rows : (processed_rows + row_count)]
        processed_rows += row_count
        starting_col += 1

    if not triu:
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))

    return matrix


def process_key(key: str) -> str:
    '''
    Processes the foreign key value by normalizing and removing special characters

    Parameters:
    ----------
    key: `str`
        Key to be processed
    
    Returns:
    -------
    `str`: Processed key
    '''
    if not isinstance(key, str):
        key = str(key)
    key = key.replace('.0', '')
    key = key.lower()
    key = key.replace(u'\xa0', u' ')
    key = key.replace(r'\s+', ' ')
    key = key.replace(' ', '')
    key = unicodedata.normalize('NFKD', key).encode('ascii', 'ignore').decode('utf-8')
    key = ''.join(char for char in key if char.isalnum())

    return key
