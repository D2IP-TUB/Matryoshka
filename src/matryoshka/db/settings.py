"""Resolution of PostgreSQL connection settings.

Matryoshka stores its sketch index and inverted index in PostgreSQL, so every
entry point needs connection parameters. Resolution is deliberately lazy:
nothing is read at import time, so ``import matryoshka`` succeeds on a machine
with no database configured and fails only when a connection is actually
required.

Precedence, highest first:

1. a :class:`DBSettings` instance passed explicitly to the constructor;
2. ``MATRYOSHKA_DSN``, a full ``postgresql://`` URI;
3. the ``MATRYOSHKA_DB_*`` variables (``HOST``, ``PORT``, ``NAME``, ``USER``,
   ``PASSWORD``);
4. the standard libpq variables (``PGHOST``, ``PGPORT``, ``PGDATABASE``,
   ``PGUSER``, ``PGPASSWORD``);
5. a YAML file named by ``MATRYOSHKA_DB_CONFIG``, else ``./db_config.yaml``,
   else ``~/.config/matryoshka/db_config.yaml``;
6. the defaults below.

The YAML schema is the one used by the original research code, so an existing
``db_config.yaml`` keeps working:

.. code-block:: yaml

    db:
      dbname: matryoshka
      user: postgres
      password: postgres
      host: localhost
      port: 5432
    ssh:              # optional, only for tunnel=True
      host: bastion.example.org
      port: 22
      user: postgres
      key: /path/to/key
      key_password: null
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 5432
DEFAULT_DBNAME = 'matryoshka'
DEFAULT_USER = 'postgres'
DEFAULT_PASSWORD = 'postgres'

_CONFIG_ENV = 'MATRYOSHKA_DB_CONFIG'
_CONFIG_NAMES = ('db_config.yaml', 'db_config.yml')


@dataclass
class SSHSettings:
    """Parameters for an optional SSH tunnel to the database host."""

    host: str = 'localhost'
    port: int = 22
    user: str = 'postgres'
    key: str | None = None
    key_password: str | None = None


@dataclass
class DBSettings:
    """PostgreSQL connection parameters."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    dbname: str = DEFAULT_DBNAME
    user: str = DEFAULT_USER
    password: str = DEFAULT_PASSWORD
    ssh: SSHSettings = field(default_factory=SSHSettings)

    @property
    def dsn(self) -> str:
        """libpq connection URI for this configuration."""
        return (
            f'postgresql://{self.user}:{self.password}'
            f'@{self.host}:{self.port}/{self.dbname}'
        )

    @classmethod
    def from_dsn(cls, dsn: str, ssh: SSHSettings | None = None) -> 'DBSettings':
        """Parse a ``postgresql://user:password@host:port/dbname`` URI."""
        parsed = urlparse(dsn)
        if parsed.scheme not in ('postgres', 'postgresql'):
            raise ValueError(
                f'expected a postgresql:// URI, got scheme {parsed.scheme!r}'
            )
        return cls(
            host=parsed.hostname or DEFAULT_HOST,
            port=parsed.port or DEFAULT_PORT,
            dbname=(parsed.path or '/').lstrip('/') or DEFAULT_DBNAME,
            user=unquote(parsed.username) if parsed.username else DEFAULT_USER,
            password=unquote(parsed.password) if parsed.password else DEFAULT_PASSWORD,
            ssh=ssh or SSHSettings(),
        )

    def __repr__(self) -> str:  # never leak the password into logs
        return (
            f'DBSettings(host={self.host!r}, port={self.port}, '
            f'dbname={self.dbname!r}, user={self.user!r}, password=***)'
        )


def _find_config_file() -> Path | None:
    explicit = os.environ.get(_CONFIG_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f'{_CONFIG_ENV}={explicit!r} does not point at a readable file'
            )
        return path
    for name in _CONFIG_NAMES:
        for base in (Path.cwd(), Path.home() / '.config' / 'matryoshka'):
            candidate = base / name
            if candidate.is_file():
                return candidate
    return None


def _from_yaml(path: Path) -> DBSettings:
    import yaml  # imported lazily; only needed on the YAML path

    with path.open() as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    db = raw.get('db', {}) or {}
    ssh_raw = raw.get('ssh', {}) or {}
    return DBSettings(
        host=db.get('host', DEFAULT_HOST),
        port=int(db.get('port', DEFAULT_PORT)),
        dbname=db.get('dbname', DEFAULT_DBNAME),
        user=db.get('user', DEFAULT_USER),
        password=db.get('password', DEFAULT_PASSWORD),
        ssh=SSHSettings(
            host=ssh_raw.get('host', 'localhost'),
            port=int(ssh_raw.get('port', 22)),
            user=ssh_raw.get('user', 'postgres'),
            key=ssh_raw.get('key'),
            key_password=ssh_raw.get('key_password'),
        ),
    )


def _from_env() -> DBSettings | None:
    dsn = os.environ.get('MATRYOSHKA_DSN')
    if dsn:
        return DBSettings.from_dsn(dsn)
    keys = {
        'host':     ('MATRYOSHKA_DB_HOST', 'PGHOST'),
        'port':     ('MATRYOSHKA_DB_PORT', 'PGPORT'),
        'dbname':   ('MATRYOSHKA_DB_NAME', 'PGDATABASE'),
        'user':     ('MATRYOSHKA_DB_USER', 'PGUSER'),
        'password': ('MATRYOSHKA_DB_PASSWORD', 'PGPASSWORD'),
    }
    found = {}
    for field_name, names in keys.items():
        for name in names:
            if os.environ.get(name):
                found[field_name] = os.environ[name]
                break
    if not found:
        return None
    if 'port' in found:
        found['port'] = int(found['port'])
    return DBSettings(**found)


def resolve_settings(settings: DBSettings | str | None = None) -> DBSettings:
    """Return the effective connection settings.

    Parameters
    ----------
    settings
        A :class:`DBSettings`, a ``postgresql://`` DSN string, or ``None`` to
        resolve from the environment and configuration files.
    """
    if isinstance(settings, DBSettings):
        return settings
    if isinstance(settings, str):
        return DBSettings.from_dsn(settings)
    from_env = _from_env()
    if from_env is not None:
        return from_env
    config_file = _find_config_file()
    if config_file is not None:
        return _from_yaml(config_file)
    return DBSettings()
