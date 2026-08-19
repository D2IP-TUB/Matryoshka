import os
import tarfile
import zipfile
from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Dict, Iterator, Optional, Tuple

import polars as pl

from ..utils.common import read_tsv_from_archive


def open_source(source_spec: Dict[str, Any]) -> pl.LazyFrame:
    '''Open a source spec (produced by a loader's `load_table_sources`) into a LazyFrame.
    Safe to call inside a worker process.'''
    kind = source_spec['kind']
    fmt = source_spec.get('format', 'csv')
    if kind == 'file':
        path = source_spec['path']
        if fmt == 'parquet':
            return pl.scan_parquet(path)
        sep = '\t' if fmt == 'tsv' else ','
        return pl.scan_csv(
            path,
            separator=sep,
            ignore_errors=True,
            truncate_ragged_lines=True,
            quote_char=None,
        )
    if kind == 'zip':
        # Zip member scans cannot be lazy across processes safely; read bytes then scan.
        import io
        with zipfile.ZipFile(source_spec['archive'], 'r') as zf:
            data = zf.read(source_spec['member'])
        buf = io.BytesIO(data)
        if fmt == 'parquet':
            return pl.scan_parquet(buf)
        return pl.scan_csv(
            buf,
            ignore_errors=True,
            truncate_ragged_lines=True,
            quote_char=None,
        )
    raise ValueError(f'Unknown source kind: {kind}')


class ArchiveLoader(ABC):
    @abstractmethod
    def load_tables(self, archive_path: str, logger: Logger, checkpoint: Optional[str] = None) -> Iterator[Tuple[str, pl.LazyFrame]]:
        pass

    def load_table_sources(self, archive_path: str, logger: Logger, checkpoint: Optional[str] = None, n_tables: Optional[int] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
        '''Yield (name, source_spec) pairs. Workers can re-open each source via `open_source`.
        Default implementation falls back to `load_tables` and wraps the LazyFrame as an in-memory spec
        (sub-optimal; concrete loaders should override).'''
        raise NotImplementedError


class ZipArchiveLoader(ArchiveLoader):
    def load_tables(self, archive_path: str, logger: Logger, checkpoint: Optional[str] = None, n_tables: Optional[int] = None) -> Iterator[Tuple[str, pl.LazyFrame]]:
        start_processing = checkpoint is None

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for i, name in enumerate(zip_ref.namelist()[:n_tables]):
                if checkpoint and not start_processing:
                    if name == checkpoint:
                        start_processing = True

                if start_processing:
                    if name.endswith('.parquet'):
                        with zip_ref.open(name) as file:
                            table = pl.scan_parquet(file)
                        yield name, table
                    elif name.endswith('.csv'):
                        with zip_ref.open(name) as file:
                            table = pl.scan_csv(
                                file,
                                ignore_errors=True,
                                truncate_ragged_lines=True,
                                quote_char=None,
                            )
                        yield name, table

                    if i % 100 == 0:
                        logger.info(f'Read {i+1} table(s) from {archive_path}')

    def load_table_sources(self, archive_path: str, logger: Logger, checkpoint: Optional[str] = None, n_tables: Optional[int] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
        start_processing = checkpoint is None
        abs_archive = os.path.abspath(archive_path)
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            names = zip_ref.namelist()[:n_tables] if n_tables else zip_ref.namelist()
        for i, name in enumerate(names):
            if checkpoint and not start_processing:
                if name == checkpoint:
                    start_processing = True
                else:
                    continue
            if start_processing:
                if name.endswith('.parquet'):
                    yield name, {'kind': 'zip', 'archive': abs_archive, 'member': name, 'format': 'parquet'}
                elif name.endswith('.csv'):
                    yield name, {'kind': 'zip', 'archive': abs_archive, 'member': name, 'format': 'csv'}
                if i % 100 == 0:
                    logger.info(f'Enumerated {i+1} table(s) from {archive_path}')


class TarArchiveLoader(ArchiveLoader):
    def load_tables(self, archive_path: str, logger: Logger, checkpoint: Optional[str] = None, n_tables: Optional[int] = None) -> Iterator[Tuple[str, pl.LazyFrame]]:
        start_processing = checkpoint is None

        with tarfile.open(archive_path, 'r:gz') as tar:
            for i, member in enumerate(tar.getmembers()[:n_tables]):
                if member.name.endswith('.tsv.gz'):
                    if checkpoint and not start_processing:
                        if member.name == checkpoint:
                            start_processing = True
                        else:
                            continue

                    if start_processing:
                        gz_file = tar.extractfile(member).read()
                        table = read_tsv_from_archive(gz_file)
                        yield member.name, table

                    if i % 100 == 0:
                        logger.info(f'Read {i+1} table(s) from {archive_path}')


class DirectoryLoader(ArchiveLoader):
    def load_tables(self, archive_path: str, logger: Logger, checkpoint: Optional[str] = None, n_tables: Optional[int] = None) -> Iterator[Tuple[str, pl.LazyFrame]]:
        start_processing = checkpoint is None

        # List all files in directory, sorted for consistent ordering
        files = sorted([f for f in os.listdir(archive_path) if os.path.isfile(os.path.join(archive_path, f))])
        files = files[:n_tables] if n_tables else files

        for i, filename in enumerate(files):
            filepath = os.path.join(archive_path, filename)

            if checkpoint and not start_processing:
                if filename == checkpoint:
                    start_processing = True
                else:
                    continue

            if start_processing:
                if filename.endswith('.parquet'):
                    table = pl.scan_parquet(filepath)
                    yield filename, table
                elif filename.endswith('.csv') or filename.endswith('.tsv'):
                    table = pl.scan_csv(
                        filepath,
                        separator='\t' if filename.endswith('.tsv') else ',',
                        ignore_errors=True,
                        truncate_ragged_lines=True,
                        quote_char=None,
                    )
                    yield filename, table

                if i % 100 == 0:
                    logger.info(f'Read {i+1} table(s) from {archive_path}')

    def load_table_sources(self, archive_path: str, logger: Logger, checkpoint: Optional[str] = None, n_tables: Optional[int] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
        start_processing = checkpoint is None
        files = sorted([f for f in os.listdir(archive_path) if os.path.isfile(os.path.join(archive_path, f))])
        files = files[:n_tables] if n_tables else files
        for i, filename in enumerate(files):
            filepath = os.path.abspath(os.path.join(archive_path, filename))
            if checkpoint and not start_processing:
                if filename == checkpoint:
                    start_processing = True
                else:
                    continue
            if start_processing:
                if filename.endswith('.parquet'):
                    yield filename, {'kind': 'file', 'path': filepath, 'format': 'parquet'}
                elif filename.endswith('.csv'):
                    yield filename, {'kind': 'file', 'path': filepath, 'format': 'csv'}
                elif filename.endswith('.tsv'):
                    yield filename, {'kind': 'file', 'path': filepath, 'format': 'tsv'}
                if i % 100 == 0:
                    logger.info(f'Enumerated {i+1} table(s) from {archive_path}')
