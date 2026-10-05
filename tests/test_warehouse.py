"""End-to-end checks on a small DuckDB warehouse (same SQL as Snowflake)."""
import json

import pandas as pd

from shelfcast.db import DuckDBBackend, split_sql
from shelfcast.pipeline import read_plan_inputs
from shelfcast.restock import recommend


def test_split_sql_ignores_comments():
    sql = "-- a comment; with a semicolon\nSELECT 1;\n\nSELECT 2; -- trailing\n"
    assert split_sql(sql) == ["SELECT 1", "SELECT 2"]


def test_cleaning_removes_cancellations_postage_and_returns(tiny_warehouse):
    with DuckDBBackend(tiny_warehouse["db_path"]) as db:
        clean = db.query("SELECT invoice, stock_code, quantity, price FROM MART.CLEAN_SALES")
        raw = db.query("SELECT COUNT(*) AS n FROM RAW.TRANSACTIONS")["n"].item()
    assert 0 < len(clean) < raw
    assert not clean["invoice"].str.startswith("C").any()
    assert "POST" not in set(clean["stock_code"])
    assert (clean["quantity"] > 0).all() and (clean["price"] > 0).all()


def test_daily_sales_add_up(tiny_warehouse):
    with DuckDBBackend(tiny_warehouse["db_path"]) as db:
        total_daily = db.query("SELECT SUM(units) AS u FROM MART.DAILY_SALES")["u"].item()
        total_clean = db.query("SELECT SUM(quantity) AS u FROM MART.CLEAN_SALES")["u"].item()
        summary = db.query("SELECT * FROM MART.PRODUCT_SUMMARY")
    assert total_daily == total_clean
    assert {"stock_code", "description", "total_units", "selling_days", "median_price"} <= set(summary.columns)


def test_forecast_tables_are_complete(tiny_warehouse):
    with DuckDBBackend(tiny_warehouse["db_path"]) as db:
        fc = db.query("SELECT * FROM MART.FORECASTS")
        bt = db.query("SELECT * FROM MART.BACKTEST")
        runs = db.query("SELECT * FROM MART.MODEL_RUNS")
        inv = db.query("SELECT * FROM MART.INVENTORY")
    assert fc.groupby("stock_code")["h"].nunique().eq(14).all()
    assert fc["stock_code"].nunique() == 6
    assert (fc[["p10", "total_p10"]] >= 0).all().all()
    assert ((fc["p10"] <= fc["p50"]) & (fc["p50"] <= fc["p90"])).all()
    assert ((fc["total_p10"] <= fc["total_p50"]) & (fc["total_p50"] <= fc["total_p90"])).all()
    assert len(bt) == 6 * 14 and runs["run_id"].nunique() >= 1
    assert set(inv["stock_code"]) == set(fc["stock_code"])
    metrics = json.loads(runs["metrics_json"].iloc[-1])
    assert metrics["products"] == 6 and 0 <= metrics["coverage_p10_p90"] <= 1


def test_sql_restock_plan_matches_python(tiny_warehouse):
    with DuckDBBackend(tiny_warehouse["db_path"]) as db:
        inputs = read_plan_inputs(db)
        sql_plan = db.query("SELECT * FROM MART.RESTOCK_PLAN")
    py_plan = recommend(inputs["forecasts"], inputs["inventory"], inputs["products"])
    merged = py_plan.merge(sql_plan.astype({"stock_code": str}), on="stock_code", suffixes=("_py", "_sql"))
    assert len(merged) == len(py_plan) == len(sql_plan) == 6
    assert (merged["status_py"] == merged["status_sql"]).all()
    assert (merged["days_left_py"] == merged["days_left_sql"]).all()
    assert (merged["order_qty_py"] - merged["order_qty_sql"]).abs().max() <= 1  # rounding only


def test_retraining_keeps_existing_stock_levels(tiny_warehouse):
    from shelfcast import pipeline
    from shelfcast.forecast import TrainSettings

    home = tiny_warehouse["home"]
    with DuckDBBackend(tiny_warehouse["db_path"]) as db:
        db.execute("UPDATE MART.INVENTORY SET on_hand = 12345 WHERE stock_code = (SELECT MIN(stock_code) FROM MART.INVENTORY)")
        pipeline.train(db, 6, 30, TrainSettings(epochs=1, hidden=16), models_dir=home / "models",
                       reports_dir=home / "reports", log=lambda *_: None)
        inv = db.query("SELECT * FROM MART.INVENTORY")
        runs = db.query("SELECT * FROM MART.MODEL_RUNS")
    assert (inv["on_hand"] == 12345).sum() == 1
    assert len(runs) >= 2  # runs are appended, not replaced
    assert (home / "reports" / "synthetic" / "RESULTS.md").exists()  # demo runs never overwrite real results
    assert not (home / "reports" / "RESULTS.md").exists()
    assert (home / "models" / "shelfcast_quantile_mlp.pt").exists()


def test_report_mentions_the_baselines(tiny_warehouse):
    text = (tiny_warehouse["home"] / "reports" / "synthetic" / "RESULTS.md").read_text(encoding="utf-8")
    assert "Same weekday last week" in text and "28-day average" in text
    assert "Synthetic demo data" in text
    assert pd.notna(tiny_warehouse["result"]["metrics"]["wape_shelfcast"])
