import tarfile
import zipfile
import polars as pl
from abc import ABC, abstractmethod
from logging import Logger
from typing import Iterator, Tuple, Optional
from ..common import read_tsv_from_archive


class ArchiveLoader(ABC):    
    @abstractmethod
    def load_tables(self, archive_path: str, logger: Logger, checkpoint: Optional[str] = None) -> Iterator[Tuple[str, pl.LazyFrame]]:
        pass


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
                            table = pl.scan_csv(file, ignore_errors=True)
                        yield name, table
                    
                    if i % 100 == 0:
                        logger.info(f'Read {i+1} table(s) from {archive_path}')


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