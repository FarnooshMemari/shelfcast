import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def ignore_local_env_file(monkeypatch):
    """Never read the local .env during tests, so they can't touch a real Snowflake account."""
    import shelfcast.db

    monkeypatch.setattr(shelfcast.db, "load_dotenv", lambda *a, **k: None)


@pytest.fixture(scope="session")
def tiny_warehouse(tmp_path_factory):
    """A small DuckDB warehouse built end to end from synthetic data (a few seconds)."""
    from shelfcast import pipeline
    from shelfcast.data import make_synthetic_transactions
    from shelfcast.db import DuckDBBackend
    from shelfcast.forecast import TrainSettings

    home = tmp_path_factory.mktemp("shelfcast_home")
    db_path = home / "data" / "shelfcast.duckdb"
    tx = make_synthetic_transactions(n_products=8, start="2011-01-01", end="2011-12-09", seed=3)
    with DuckDBBackend(db_path) as db:
        pipeline.load(db, tx, pipeline.SYNTHETIC_SOURCE, log=lambda *_: None)
        out = pipeline.train(
            db,
            n_products=6,
            min_days=30,
            settings=TrainSettings(epochs=3, hidden=32),
            models_dir=home / "models",
            reports_dir=home / "reports",
            log=lambda *_: None,
        )
    return {"home": home, "db_path": db_path, "transactions": tx, "result": out}


@pytest.fixture()
def duckdb_env(tiny_warehouse, monkeypatch):
    """Point the app and CLI at the tiny warehouse."""
    monkeypatch.setenv("SHELFCAST_BACKEND", "duckdb")
    monkeypatch.setenv("SHELFCAST_DUCKDB", str(tiny_warehouse["db_path"]))
    for key in list(os.environ):
        if key.startswith("SNOWFLAKE_"):
            monkeypatch.delenv(key)
    return tiny_warehouse
