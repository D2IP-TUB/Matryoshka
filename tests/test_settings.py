"""Resolution order of PostgreSQL connection settings.

The precedence chain is the one documented in ``matryoshka.db.settings``:
explicit argument, ``MATRYOSHKA_DSN``, ``MATRYOSHKA_DB_*``, ``PG*``, a YAML
file, defaults. Every layer is exercised here, plus the property that nothing
is read at import time.
"""
import pytest

from matryoshka.db.settings import (
    DEFAULT_DBNAME,
    DBSettings,
    SSHSettings,
    resolve_settings,
)

_ENV_KEYS = [
    'MATRYOSHKA_DSN', 'MATRYOSHKA_DB_CONFIG',
    'MATRYOSHKA_DB_HOST', 'MATRYOSHKA_DB_PORT', 'MATRYOSHKA_DB_NAME',
    'MATRYOSHKA_DB_USER', 'MATRYOSHKA_DB_PASSWORD',
    'PGHOST', 'PGPORT', 'PGDATABASE', 'PGUSER', 'PGPASSWORD',
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No connection variables set, and a working directory with no config file."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    # `_find_config_file` also probes ~/.config/matryoshka; point HOME at an
    # empty directory so a developer's own config cannot influence the result.
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    (tmp_path / 'home').mkdir()
    return tmp_path


def test_defaults_when_nothing_is_configured(clean_env):
    settings = resolve_settings()
    assert settings.host == 'localhost'
    assert settings.port == 5432
    assert settings.dbname == DEFAULT_DBNAME


def test_explicit_settings_win(clean_env, monkeypatch):
    monkeypatch.setenv('MATRYOSHKA_DSN', 'postgresql://env@envhost:1/envdb')
    explicit = DBSettings(host='given', port=9, dbname='givendb', user='u', password='p')
    assert resolve_settings(explicit) is explicit


def test_dsn_string_argument(clean_env):
    settings = resolve_settings('postgresql://alice:secret@db.example:6543/lake')
    assert (settings.host, settings.port, settings.dbname) == ('db.example', 6543, 'lake')
    assert (settings.user, settings.password) == ('alice', 'secret')


def test_dsn_env_beats_component_env(clean_env, monkeypatch):
    monkeypatch.setenv('MATRYOSHKA_DSN', 'postgresql://a:b@dsnhost:1234/dsndb')
    monkeypatch.setenv('MATRYOSHKA_DB_HOST', 'ignored')
    assert resolve_settings().host == 'dsnhost'


def test_component_env_beats_libpq_env(clean_env, monkeypatch):
    monkeypatch.setenv('MATRYOSHKA_DB_HOST', 'preferred')
    monkeypatch.setenv('PGHOST', 'fallback')
    assert resolve_settings().host == 'preferred'


def test_libpq_env(clean_env, monkeypatch):
    monkeypatch.setenv('PGHOST', 'pg.example')
    monkeypatch.setenv('PGPORT', '15432')
    monkeypatch.setenv('PGDATABASE', 'pgdb')
    settings = resolve_settings()
    assert (settings.host, settings.port, settings.dbname) == ('pg.example', 15432, 'pgdb')


def test_yaml_file_in_working_directory(clean_env):
    (clean_env / 'db_config.yaml').write_text(
        'db:\n'
        '  dbname: fromyaml\n'
        '  user: yamluser\n'
        '  password: yamlpass\n'
        '  host: yamlhost\n'
        '  port: 5433\n'
        'ssh:\n'
        '  host: bastion\n'
        '  port: 2222\n'
        '  user: tunnel\n'
        '  key: /tmp/key\n'
        '  key_password: null\n'
    )
    settings = resolve_settings()
    assert (settings.host, settings.port, settings.dbname) == ('yamlhost', 5433, 'fromyaml')
    assert (settings.ssh.host, settings.ssh.port, settings.ssh.user) == ('bastion', 2222, 'tunnel')


def test_env_beats_yaml(clean_env, monkeypatch):
    (clean_env / 'db_config.yaml').write_text('db:\n  host: yamlhost\n')
    monkeypatch.setenv('MATRYOSHKA_DSN', 'postgresql://u:p@envhost:5432/envdb')
    assert resolve_settings().host == 'envhost'


def test_named_config_file(clean_env, monkeypatch):
    path = clean_env / 'elsewhere.yaml'
    path.write_text('db:\n  host: namedhost\n  dbname: nameddb\n')
    monkeypatch.setenv('MATRYOSHKA_DB_CONFIG', str(path))
    assert resolve_settings().host == 'namedhost'


def test_missing_named_config_file_is_an_error(clean_env, monkeypatch):
    monkeypatch.setenv('MATRYOSHKA_DB_CONFIG', str(clean_env / 'absent.yaml'))
    with pytest.raises(FileNotFoundError):
        resolve_settings()


def test_dsn_round_trip():
    original = DBSettings(host='h', port=1234, dbname='d', user='u', password='p')
    assert DBSettings.from_dsn(original.dsn) == DBSettings(
        host='h', port=1234, dbname='d', user='u', password='p', ssh=SSHSettings()
    )


def test_non_postgres_scheme_is_rejected():
    with pytest.raises(ValueError, match='postgresql'):
        DBSettings.from_dsn('mysql://u:p@h:3306/d')


def test_repr_hides_the_password():
    rendered = repr(DBSettings(password='hunter2'))
    assert 'hunter2' not in rendered
    assert 'password=***' in rendered


def test_importing_the_package_reads_nothing(clean_env, monkeypatch):
    """`import matryoshka` must not require a configured database."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, '-c', 'import matryoshka; print(matryoshka.__version__)'],
        capture_output=True, text=True, cwd=str(clean_env),
    )
    assert result.returncode == 0, result.stderr
