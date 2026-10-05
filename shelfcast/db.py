"""One small interface over two warehouses.

* ``SnowflakeBackend`` is the real deployment.
* ``DuckDBBackend`` runs the same SQL on a local file, so the project works offline and in
  tests.

Every query returns a pandas DataFrame with lower-case column names.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, load_dotenv


def split_sql(text: str) -> list[str]:
    """Split a .sql file into statements (no semicolons inside strings in our files)."""
    text = re.sub(r"--[^\n]*", "", text)
    return [s.strip() for s in text.split(";") if s.strip()]


class Backend:
    name = "base"

    def execute(self, sql: str, params=None) -> None:
        raise NotImplementedError

    def query(self, sql: str, params=None) -> pd.DataFrame:
        raise NotImplementedError

    def write(self, df: pd.DataFrame, table: str, schema: str, append: bool = False) -> None:
        raise NotImplementedError

    def run_sql_file(self, path: str | Path) -> None:
        for statement in split_sql(Path(path).read_text(encoding="utf-8")):
            self.execute(statement)

    def table_exists(self, schema: str, table: str) -> bool:
        df = self.query(
            "SELECT COUNT(*) AS n FROM information_schema.tables "
            f"WHERE UPPER(table_schema) = '{schema.upper()}' AND UPPER(table_name) = '{table.upper()}'"
        )
        return int(df["n"].iloc[0]) > 0

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).lower() for c in df.columns]
    return df


class DuckDBBackend(Backend):
    name = "duckdb"

    def __init__(self, path: str | Path):
        import duckdb

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.con = duckdb.connect(self.path)

    def execute(self, sql: str, params=None) -> None:
        self.con.execute(sql, params)

    def query(self, sql: str, params=None) -> pd.DataFrame:
        return _lower(self.con.execute(sql, params).df())

    def write(self, df: pd.DataFrame, table: str, schema: str, append: bool = False) -> None:
        self.con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        self.con.register("_incoming", df)
        try:
            if append and self.table_exists(schema, table):
                self.con.execute(f"INSERT INTO {schema}.{table} BY NAME SELECT * FROM _incoming")
            else:
                self.con.execute(f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM _incoming")
        finally:
            self.con.unregister("_incoming")

    def close(self) -> None:
        self.con.close()


class SnowflakeBackend(Backend):
    name = "snowflake"

    def __init__(self, params: dict):
        import snowflake.connector

        self.params = params
        self.con = snowflake.connector.connect(**params)

    @classmethod
    def from_env(cls) -> "SnowflakeBackend":
        def need(key: str) -> str:
            value = os.environ.get(key)
            if not value:
                raise RuntimeError(f"Set {key} in your environment or .env file (see .env.example)")
            return value

        params = {
            "account": need("SNOWFLAKE_ACCOUNT"),
            "user": need("SNOWFLAKE_USER"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "SHELFCAST_WH"),
            "database": os.environ.get("SNOWFLAKE_DATABASE", "SHELFCAST"),
            "schema": "MART",
            "session_parameters": {"QUERY_TAG": "shelfcast"},
        }
        # A programmatic access token (PAT) goes in the password field.
        secret = os.environ.get("SNOWFLAKE_PAT") or os.environ.get("SNOWFLAKE_PASSWORD")
        if secret:
            params["password"] = secret
        for env_key, param in (("SNOWFLAKE_ROLE", "role"), ("SNOWFLAKE_AUTHENTICATOR", "authenticator")):
            if os.environ.get(env_key):
                params[param] = os.environ[env_key]
        return cls(params)

    def execute(self, sql: str, params=None) -> None:
        with self.con.cursor() as cur:
            cur.execute(sql, params)

    def query(self, sql: str, params=None) -> pd.DataFrame:
        with self.con.cursor() as cur:
            cur.execute(sql, params)
            df = cur.fetch_pandas_all()
        return _lower(df)

    def write(self, df: pd.DataFrame, table: str, schema: str, append: bool = False) -> None:
        from snowflake.connector.pandas_tools import write_pandas

        self.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        upper = df.copy()
        upper.columns = [c.upper() for c in upper.columns]
        write_pandas(
            self.con,
            upper,
            table_name=table.upper(),
            schema=schema.upper(),
            auto_create_table=True,
            overwrite=not append,
            quote_identifiers=False,
            use_logical_type=True,
        )

    def close(self) -> None:
        self.con.close()


def duckdb_path() -> str:
    return os.environ.get("SHELFCAST_DUCKDB", str(DATA_DIR / "shelfcast.duckdb"))


def backend_kind(kind: str = "auto") -> str:
    """``auto`` means Snowflake when SNOWFLAKE_ACCOUNT is set, otherwise local DuckDB."""
    load_dotenv()
    if kind == "auto":
        kind = os.environ.get("SHELFCAST_BACKEND") or ("snowflake" if os.environ.get("SNOWFLAKE_ACCOUNT") else "duckdb")
    if kind not in ("snowflake", "duckdb"):
        raise ValueError(f"Unknown backend {kind!r}; use auto, duckdb or snowflake")
    return kind


def get_backend(kind: str = "auto") -> Backend:
    kind = backend_kind(kind)
    if kind == "snowflake":
        return SnowflakeBackend.from_env()
    return DuckDBBackend(duckdb_path())
