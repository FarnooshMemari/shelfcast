import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")


def _app():
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()
    return AppTest.from_file(APP, default_timeout=120)


def test_app_renders_and_reacts_to_settings(duckdb_env):
    at = _app().run()
    assert not at.exception
    assert at.title[0].value == "ShelfCast"
    labels = [m.label for m in at.metric]
    assert "Order now" in labels and "Forecast error (WAPE)" in labels
    assert [t.label for t in at.tabs][:2] == ["Restock list", "Product forecast"]
    assert any("synthetic" in w.value for w in at.warning)  # demo data is labelled as such

    value_before = next(m.value for m in at.metric if m.label == "Order value at cost")
    at.sidebar.slider[1].set_value(0).run()  # plan for the median instead of P90
    assert not at.exception
    value_after = next(m.value for m in at.metric if m.label == "Order value at cost")
    to_number = lambda s: float(s.replace("£", "").replace(",", ""))  # noqa: E731
    assert to_number(value_after) <= to_number(value_before)

    at.radio[0].set_value("All products").run()
    last_code = duckdb_env["result"]["recommendations"]["stock_code"].iloc[-1]
    at.selectbox[0].set_value(last_code).run()
    assert not at.exception
    assert last_code in at.selectbox[0].options[at.selectbox[0].index]


def test_app_shows_setup_help_when_warehouse_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELFCAST_BACKEND", "duckdb")
    monkeypatch.setenv("SHELFCAST_DUCKDB", str(tmp_path / "empty.duckdb"))
    monkeypatch.setenv("SHELFCAST_SNAPSHOT_DIR", str(tmp_path / "no_snapshot"))
    at = _app().run()
    assert not at.exception
    assert "No forecasts" in at.info[0].value
    assert any("synthetic" in b.label for b in at.button)


def test_app_falls_back_to_a_saved_snapshot(duckdb_env, tmp_path, monkeypatch):
    from shelfcast import pipeline
    from shelfcast.db import DuckDBBackend

    with DuckDBBackend(duckdb_env["db_path"]) as db:
        pipeline.write_snapshot(db, tmp_path / "snap", log=lambda *_: None)
    assert pipeline.has_snapshot(tmp_path / "snap")
    monkeypatch.setenv("SHELFCAST_DUCKDB", str(tmp_path / "empty.duckdb"))
    monkeypatch.setenv("SHELFCAST_SNAPSHOT_DIR", str(tmp_path / "snap"))
    at = _app().run()
    assert not at.exception
    assert any("snapshot" in i.value for i in at.info)
    assert "Order value at cost" in [m.label for m in at.metric]


def test_app_uses_the_snapshot_when_snowflake_is_unreachable(duckdb_env, tmp_path, monkeypatch):
    from shelfcast import pipeline
    from shelfcast.db import DuckDBBackend

    with DuckDBBackend(duckdb_env["db_path"]) as db:
        pipeline.write_snapshot(db, tmp_path / "snap", log=lambda *_: None)
    monkeypatch.setenv("SHELFCAST_BACKEND", "snowflake")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "")  # empty account, so connecting fails right away
    monkeypatch.setenv("SHELFCAST_SNAPSHOT_DIR", str(tmp_path / "snap"))
    at = _app().run()
    assert not at.exception
    assert any("snapshot" in m.value for m in at.sidebar.markdown)
    assert "Order value at cost" in [m.label for m in at.metric]


@pytest.mark.slow
def test_cli_runs_end_to_end(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SNOWFLAKE_")}
    env.update(SHELFCAST_HOME=str(tmp_path), SHELFCAST_BACKEND="duckdb", SHELFCAST_DUCKDB=str(tmp_path / "w.duckdb"))
    cmd = [sys.executable, "-m", "shelfcast", "all", "--synthetic", "--products", "5", "--epochs", "2"]
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert "WAPE" in out.stdout and "Restock brief" in out.stdout
    assert (tmp_path / "reports" / "synthetic" / "RESULTS.md").exists()
    check = subprocess.run([sys.executable, "-m", "shelfcast", "check"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert check.returncode == 0 and "MART.RESTOCK_PLAN: yes" in check.stdout
