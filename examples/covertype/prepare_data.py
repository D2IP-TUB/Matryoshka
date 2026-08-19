"""Fetch and prepare the AutoFeat ``covertype`` lake.

The lake is published on Zenodo as part of the AutoFeat benchmark:

    Ionescu, Andra-Denis. *Dataset for testing Feature Discovery.*
    Zenodo, 2024. https://doi.org/10.5281/zenodo.12755408 (CC BY 4.0)

It accompanies Ionescu et al., *AutoFeat: Transitive Feature Discovery over
Join Paths*, ICDE 2024, and derives from OpenML dataset 44159, split into a
base table and twelve auxiliary tables by synthetic foreign keys.

One transformation is applied on download. The published key columns hold bare
integers (``316150``); this script prefixes them with ``k`` (``k316150``), which
is what the paper's experiments indexed. Without it a key column is inferred as
numeric rather than as a join-key candidate, and the lake yields no joinable
tables. The transformation is confined to columns named ``Key_*``.

Usage::

    python examples/covertype/prepare_data.py               # 212 MB download
    python examples/covertype/prepare_data.py --keep-archive
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import polars as pl

ZENODO_RECORD = '12755408'
ARCHIVE_URL = (
    f'https://zenodo.org/records/{ZENODO_RECORD}/files/autofeat-data.tar?download=1'
)
ARCHIVE_MD5 = 'e92a3158bbf4bbda0307ff0b5a891222'
ARCHIVE_BYTES = 212_741_120
MEMBER_PREFIX = 'autofeat/covertype/'

# The 13 lake tables. The benchmark's own manifests go to _metadata/, see prepare().
EXPECTED_TABLES = [
    'table_0_0.csv',
    'table_1_1.csv', 'table_1_2.csv', 'table_1_3.csv',
    'table_2_4.csv', 'table_2_5.csv', 'table_2_6.csv', 'table_2_7.csv',
    'table_2_8.csv', 'table_2_9.csv', 'table_2_10.csv', 'table_2_11.csv',
    'table_2_12.csv',
]
EXPECTED_ROWS = 423_680


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(Path(__file__).parent / 'data'),
                        help='directory to write the prepared lake into')
    parser.add_argument('--archive', default=None,
                        help='use this local autofeat-data.tar instead of downloading')
    parser.add_argument('--keep-archive', action='store_true',
                        help='do not delete the downloaded archive afterwards')
    parser.add_argument('--force', action='store_true',
                        help='re-prepare even if the output directory looks complete')
    return parser.parse_args()


def md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def download(destination: Path) -> Path:
    print(f'downloading {ARCHIVE_BYTES / 1e6:.0f} MB from Zenodo record {ZENODO_RECORD}')

    def progress(count: int, block: int, total: int) -> None:
        if total > 0:
            done = min(count * block, total)
            print(f'\r  {done / 1e6:7.1f} / {total / 1e6:.0f} MB', end='', flush=True)

    urllib.request.urlretrieve(ARCHIVE_URL, destination, reporthook=progress)
    print()
    checksum = md5(destination)
    if checksum != ARCHIVE_MD5:
        destination.unlink(missing_ok=True)
        raise SystemExit(
            f'checksum mismatch: expected {ARCHIVE_MD5}, got {checksum}. '
            f'The Zenodo record may have changed; verify it before proceeding.'
        )
    print(f'  checksum ok ({checksum})')
    return destination


def prepare(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir.parent / '_extract'
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(MEMBER_PREFIX)]
        if not members:
            raise SystemExit(f'no {MEMBER_PREFIX} members in {archive}')
        # Guard against path traversal before extracting anything.
        for member in members:
            resolved = (staging / member.name).resolve()
            if not str(resolved).startswith(str(staging.resolve())):
                raise SystemExit(f'refusing unsafe archive member {member.name!r}')
        tar.extractall(staging, members=members)

    source = staging / MEMBER_PREFIX
    print(f'preparing {len(EXPECTED_TABLES)} tables into {out_dir}')
    for name in EXPECTED_TABLES:
        path = source / name
        if not path.is_file():
            raise SystemExit(f'{name} missing from the archive')
        table = pl.read_csv(path)
        if table.height != EXPECTED_ROWS:
            raise SystemExit(
                f'{name} has {table.height} rows, expected {EXPECTED_ROWS}'
            )
        key_columns = [c for c in table.columns if c.startswith('Key_')]
        if not key_columns:
            raise SystemExit(f'{name} has no Key_* column')
        table = table.with_columns(
            [('k' + pl.col(c).cast(pl.String)).alias(c) for c in key_columns]
        )
        table.write_csv(out_dir / name)
        print(f'  {name:16s} {table.height:>7,} rows, '
              f'{table.width} columns, keys {key_columns}')

    # The benchmark's own manifests describe the intended join graph. They go
    # into a subdirectory rather than alongside the tables: the indexer treats
    # every .csv in the lake directory as a lake table, and connections.csv
    # would otherwise be indexed as a fourteenth one.
    metadata_dir = out_dir / '_metadata'
    metadata_dir.mkdir(exist_ok=True)
    for extra in ('tables.json', 'connections.csv'):
        if (source / extra).is_file():
            shutil.copy2(source / extra, metadata_dir / extra)
    shutil.rmtree(staging)

    stray = sorted(
        p.name for p in out_dir.iterdir()
        if p.is_file() and p.suffix in {'.csv', '.tsv', '.parquet'}
        and p.name not in EXPECTED_TABLES
    )
    if stray:
        raise SystemExit(
            f'{out_dir} holds files the indexer would treat as lake tables but '
            f'which are not part of the benchmark: {stray}. Remove them first.'
        )


def is_complete(out_dir: Path) -> bool:
    return all((out_dir / name).is_file() for name in EXPECTED_TABLES)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    if is_complete(out_dir) and not args.force:
        print(f'{out_dir} already holds the {len(EXPECTED_TABLES)} lake tables; '
              f'pass --force to re-prepare')
        return 0

    if args.archive:
        archive = Path(args.archive)
        if not archive.is_file():
            raise SystemExit(f'{archive} not found')
        downloaded = False
    else:
        archive = out_dir.parent / 'autofeat-data.tar'
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.is_file() and md5(archive) == ARCHIVE_MD5:
            print(f'reusing {archive}')
        else:
            download(archive)
        downloaded = True

    prepare(archive, out_dir)
    if downloaded and not args.keep_archive:
        archive.unlink(missing_ok=True)
    print(f'\nready. Next: python examples/covertype/run_example.py --data {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
