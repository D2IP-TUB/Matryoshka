"""Readers for the physical layout of a data lake.

``archives`` provides loaders for directories, tar and zip archives;
``parallel`` provides the sequential and Ray-backed table processors used by
the offline indexer.
"""
from .archives import DirectoryLoader, TarArchiveLoader, ZipArchiveLoader

__all__ = ['DirectoryLoader', 'TarArchiveLoader', 'ZipArchiveLoader']
