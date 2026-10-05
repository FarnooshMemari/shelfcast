"""The steps behind ``python -m shelfcast ...``: load, train, report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from . import data as datamod
from .config import HORIZON, LOOKBACK, MODELS_DIR, REPORTS_DIR, SNAPSHOT_DIR, SQL_DIR, snapshot_dir
from .db import Backend, DuckDBBackend
from .restock import RestockSettings, brief, recommend, simulate_inventory, with_summed_daily_totals

if TYPE_CHECKING:  # torch is only imported when training, so the app can run without it
    from .forecast import TrainSettings

REAL_SOURCE = "UCI Online Retail II (real data)"
SYNTHETIC_SOURCE = "Synthetic demo data"
MODEL_FILE = "shelfcast_quantile_mlp.pt"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def load(backend: Backend, transactions: pd.DataFrame, source: str, log=print) -> dict:
    """Write raw transactions to RAW.TRANSACTIONS and build the cleaning SQL on top."""
    log(f"[load] writing {len(transactions):,} transactions to {backend.name} RAW.TRANSACTIONS")
    backend.write(transactions, "TRANSACTIONS", "RAW")
    info = pd.DataFrame([{"source": source, "rows_loaded": len(transactions), "loaded_at": _now()}])
    backend.write(info, "LOAD_INFO", "RAW")
    log("[load] running sql/10_views.sql (cleaning, daily sales, product summary)")
    backend.run_sql_file(SQL_DIR / "10_views.sql")
    counts = backend.query(
        "SELECT (SELECT COUNT(*) FROM RAW.TRANSACTIONS) AS raw_rows, "
        "(SELECT COUNT(*) FROM MART.CLEAN_SALES) AS clean_rows, "
        "(SELECT COUNT(*) FROM MART.PRODUCT_SUMMARY) AS products, "
        "(SELECT MIN(sale_date) FROM MART.DAILY_SALES) AS first_day, "
        "(SELECT MAX(sale_date) FROM MART.DAILY_SALES) AS last_day"
    ).iloc[0].to_dict()
    log(
        f"[load] {int(counts['raw_rows']):,} raw rows -> {int(counts['clean_rows']):,} clean sales lines, "
        f"{int(counts['products']):,} products, {pd.Timestamp(counts['first_day']).date()} to "
        f"{pd.Timestamp(counts['last_day']).date()}"
    )
    return counts


def data_source(backend: Backend) -> str:
    if backend.table_exists("RAW", "LOAD_INFO"):
        df = backend.query("SELECT source FROM RAW.LOAD_INFO")
        if len(df):
            return str(df["source"].iloc[0])
    return "unknown"


def ensure_inventory(backend: Backend, matrix: pd.DataFrame, prices: pd.Series, reset: bool = False) -> pd.DataFrame:
    """Keep stock levels that are already in MART.INVENTORY; simulate any that are missing."""
    fresh = simulate_inventory(matrix, prices)
    if not reset and backend.table_exists("MART", "INVENTORY"):
        current = backend.query("SELECT stock_code, on_hand, on_order, lead_time_days, unit_cost FROM MART.INVENTORY")
        current["stock_code"] = current["stock_code"].astype(str)
        missing = fresh[~fresh["stock_code"].isin(current["stock_code"])]
        inventory = pd.concat([current, missing], ignore_index=True)
    else:
        inventory = fresh
    for col in ("on_hand", "on_order", "lead_time_days"):
        inventory[col] = pd.to_numeric(inventory[col]).astype("int64")
    inventory["unit_cost"] = pd.to_numeric(inventory["unit_cost"]).astype(float)
    backend.write(inventory, "INVENTORY", "MART")
    return inventory


def train(
    backend: Backend,
    n_products: int = 200,
    min_days: int = 120,
    settings: TrainSettings | None = None,
    reset_inventory: bool = False,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    log=print,
) -> dict:
    """Pick products, backtest and train the model, then write forecasts and the plan."""
    import torch

    from .forecast import TrainSettings, backtest_and_forecast, model_summary

    settings = settings or TrainSettings()
    summary = backend.query("SELECT * FROM MART.PRODUCT_SUMMARY")
    summary["stock_code"] = summary["stock_code"].astype(str)
    products = datamod.select_products(summary, n_products, min_days)
    if len(products) < 2:
        raise RuntimeError(f"Only {len(products)} products sold on {min_days}+ days; lower --min-days")
    in_list = ", ".join("'" + p.replace("'", "''") + "'" for p in products)
    daily = backend.query(f"SELECT stock_code, sale_date, units FROM MART.DAILY_SALES WHERE stock_code IN ({in_list})")
    start = pd.to_datetime(summary["first_sale"]).min()
    end = pd.to_datetime(summary["last_sale"]).max()
    matrix = datamod.daily_matrix(daily, products, start, end)
    log(f"[train] {len(products)} products x {len(matrix)} days ({start.date()} to {end.date()})")

    result = backtest_and_forecast(matrix, settings)
    m = result.metrics
    log(
        f"[train] backtest {m['test_period']}: WAPE {m['wape_shelfcast']:.1%} vs "
        f"{m['wape_same_weekday_last_week']:.1%} (same weekday last week) and "
        f"{m['wape_28_day_average']:.1%} (28-day average); P10-P90 coverage {m['coverage_p10_p90']:.0%}"
    )

    run_id = _now().strftime("%Y%m%dT%H%M%S")
    backend.write(result.future.assign(run_id=run_id), "FORECASTS", "MART")
    backend.write(result.backtest.assign(run_id=run_id), "BACKTEST", "MART")
    prices = summary.set_index("stock_code")["median_price"].astype(float)
    inventory = ensure_inventory(backend, matrix, prices, reset_inventory)
    backend.run_sql_file(SQL_DIR / "20_restock_plan.sql")

    # the default plan, and what the "add up daily P90s" shortcut would have ordered instead
    recs = recommend(result.future, inventory, summary, RestockSettings())
    shortcut = recommend(with_summed_daily_totals(result.future), inventory, summary, RestockSettings())
    m["order_value_running_total_p90"] = round(float(recs["order_value"].sum()), 2)
    m["order_value_if_summing_daily_p90"] = round(float(shortcut["order_value"].sum()), 2)

    source = data_source(backend)
    model_info = model_summary(result.model)
    run_row = {
        "run_id": run_id,
        "created_at": _now(),
        "backend": backend.name,
        "data_source": source,
        "as_of": m["as_of"],
        "products": int(m["products"]),
        "wape_shelfcast": float(m["wape_shelfcast"]),
        "improvement_vs_best_baseline": float(m["improvement_vs_best_baseline"]),
        "metrics_json": json.dumps(m),
        "settings_json": json.dumps(settings.as_dict()),
        "model_json": json.dumps(model_info),
    }
    backend.write(pd.DataFrame([run_row]), "MODEL_RUNS", "MART", append=True)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "config": {"lookback": LOOKBACK, "horizon": HORIZON, "hidden": settings.hidden},
            "products": products,
            "metrics": m,
            "run_id": run_id,
        },
        models_dir / MODEL_FILE,
    )

    reports_dir = Path(reports_dir)
    if "synthetic" in source.lower():  # keep demo runs from overwriting the real results
        reports_dir = reports_dir / "synthetic"
    write_reports(reports_dir, run_id, source, m, settings, model_info, recs)
    log(f"[train] wrote MART.FORECASTS, MART.BACKTEST, MART.INVENTORY, MART.RESTOCK_PLAN and {reports_dir / 'RESULTS.md'}")
    return {"run_id": run_id, "metrics": m, "source": source, "recommendations": recs, "history": result.history}


def _pct(x: float) -> str:
    return f"{x:.1%}"


def write_reports(reports_dir: Path, run_id: str, source: str, m: dict, settings: TrainSettings, model_info: dict, recs: pd.DataFrame) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "data_source": source, "metrics": m, "settings": settings.as_dict(), "model": model_info}
    (reports_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    counts = recs["status"].value_counts() if len(recs) else pd.Series(dtype=int)
    lines = [
        "# ShelfCast results",
        "",
        f"_Generated by `python -m shelfcast train` (run {run_id}). Data: {source}; "
        f"{m['products']} products, {m['history_days']} days of history up to {m['as_of']}._",
        "",
        f"## Backtest: the last {HORIZON} days the model never saw ({m['test_period']})",
        "",
        "| Forecast | Daily error (WAPE) | Two-week total error (WAPE) |",
        "|---|---|---|",
        f"| **ShelfCast (PyTorch, P50)** | **{_pct(m['wape_shelfcast'])}** | **{_pct(m['wape_total_shelfcast'])}** |",
        f"| Same weekday last week | {_pct(m['wape_same_weekday_last_week'])} | {_pct(m['wape_total_same_weekday_last_week'])} |",
        f"| 28-day average | {_pct(m['wape_28_day_average'])} | {_pct(m['wape_total_28_day_average'])} |",
        "",
        f"- Daily error is {_pct(abs(m['improvement_vs_best_baseline']))} "
        f"{'lower' if m['improvement_vs_best_baseline'] >= 0 else 'higher'} than the best simple baseline; "
        f"two-week totals are {_pct(abs(m['total_improvement_vs_best_baseline']))} "
        f"{'lower' if m['total_improvement_vs_best_baseline'] >= 0 else 'higher'}.",
        f"- {_pct(m['coverage_p10_p90'])} of actual daily sales fell inside the daily P10-P90 range, and "
        f"{_pct(m['coverage_total_p10_p90'])} of two-week totals fell inside the running-total P10-P90 range (ideal: 80%).",
        f"- For comparison, adding up the daily ranges would give a two-week range that is "
        f"{_pct(1 / max(1 - m['total_band_narrower_than_summed_daily'], 1e-9) - 1)} wider and covers "
        f"{_pct(m['coverage_total_summed_daily'])} of totals.",
        "- WAPE = total absolute error / total units sold. Lower is better.",
        "",
        "## Restock plan (default settings: order every 7 days, plan for the P90 forecast)",
        "",
        f"Order now: {int(counts.get('Order now', 0))} products, order this week: {int(counts.get('Order this week', 0))}, "
        f"OK: {int(counts.get('OK', 0))}. Total order value at cost: £{recs['order_value'].sum():,.0f}." if len(recs) else "No products.",
        "",
        f"Sizing the same orders by adding up daily P90 forecasts would cost £{m['order_value_if_summing_daily_p90']:,.0f} "
        f"instead of £{m['order_value_running_total_p90']:,.0f} "
        f"({m['order_value_if_summing_daily_p90'] / max(m['order_value_running_total_p90'], 1e-9) - 1:.0%} more stock at cost).",
        "",
        "| Status | Product | On hand | Days left | Lead time | Order qty | Order value |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in recs.head(10).itertuples(index=False):
        name = (r.description or r.stock_code).replace("|", "/")
        lines.append(
            f"| {r.status} | {name} ({r.stock_code}) | {r.on_hand} | {r.days_left} | {r.lead_time_days} | "
            f"{r.order_qty} | £{r.order_value:,.2f} |"
        )
    lines += [
        "",
        "Stock levels are simulated because the public dataset has no inventory data.",
        "",
        "## Model",
        "",
        f"- {model_info['type']}, {model_info['parameters']:,} parameters",
        f"- Inputs: {model_info['inputs']}",
        f"- Outputs: {model_info['outputs']}",
        f"- {m['training_windows']:,} training windows, best epoch {m['best_epoch']}, "
        f"{m['train_seconds']} s for backtest + final training on CPU",
        "",
        "## Brief",
        "",
        "```",
        brief(recs, m["as_of"]),
        "```",
        "",
    ]
    (reports_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def read_plan_inputs(backend: Backend) -> dict:
    """Everything the app and the brief need, with tidy types."""
    out = {
        "forecasts": backend.query(
            "SELECT stock_code, h, forecast_date, p10, p50, p90, total_p10, total_p50, total_p90 FROM MART.FORECASTS"
        ),
        "inventory": backend.query("SELECT stock_code, on_hand, on_order, lead_time_days, unit_cost FROM MART.INVENTORY"),
        "products": backend.query(
            "SELECT stock_code, description, total_units, selling_days, median_price FROM MART.PRODUCT_SUMMARY "
            "WHERE stock_code IN (SELECT DISTINCT stock_code FROM MART.FORECASTS)"
        ),
    }
    for df in out.values():
        df["stock_code"] = df["stock_code"].astype(str)
    fc = out["forecasts"]
    fc["forecast_date"] = pd.to_datetime(fc["forecast_date"])
    fc["h"] = fc["h"].astype(int)
    for col in ("p10", "p50", "p90", "total_p10", "total_p50", "total_p90"):
        fc[col] = fc[col].astype(float)
    inv = out["inventory"]
    for col in ("on_hand", "on_order", "lead_time_days"):
        inv[col] = pd.to_numeric(inv[col]).astype("int64")
    inv["unit_cost"] = pd.to_numeric(inv["unit_cost"]).astype(float)
    out["products"]["description"] = out["products"]["description"].fillna("").astype(str)
    out["as_of"] = str((fc["forecast_date"].min() - pd.offsets.Day(1)).date()) if len(fc) else ""
    return out


SNAPSHOT_TABLES = {
    ("RAW", "LOAD_INFO"): "SELECT * FROM RAW.LOAD_INFO",
    ("MART", "FORECASTS"): "SELECT * FROM MART.FORECASTS",
    ("MART", "INVENTORY"): "SELECT * FROM MART.INVENTORY",
    ("MART", "BACKTEST"): "SELECT * FROM MART.BACKTEST",
    ("MART", "MODEL_RUNS"): "SELECT * FROM MART.MODEL_RUNS",
    ("MART", "PRODUCT_SUMMARY"): (
        "SELECT * FROM MART.PRODUCT_SUMMARY WHERE stock_code IN (SELECT DISTINCT stock_code FROM MART.FORECASTS)"
    ),
    ("MART", "DAILY_SALES"): (
        "SELECT * FROM MART.DAILY_SALES WHERE stock_code IN (SELECT DISTINCT stock_code FROM MART.FORECASTS)"
    ),
}


def write_snapshot(backend: Backend, out_dir: Path = SNAPSHOT_DIR, log=print) -> Path:
    """Save the tables the app reads as small Parquet files (from Snowflake or DuckDB)."""
    import duckdb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        for (schema, table), sql in SNAPSHOT_TABLES.items():
            df = backend.query(sql)
            target = out_dir / f"{schema.lower()}__{table.lower()}.parquet"
            con.register("snapshot_df", df)
            con.execute(f"COPY (SELECT * FROM snapshot_df) TO '{target.as_posix()}' (FORMAT PARQUET)")
            con.unregister("snapshot_df")
            log(f"[snapshot] {schema}.{table}: {len(df):,} rows -> {target}")
    finally:
        con.close()
    return out_dir


def has_snapshot(folder: Path | None = None) -> bool:
    return (Path(folder or snapshot_dir()) / "mart__forecasts.parquet").exists()


def snapshot_backend(folder: Path | None = None) -> DuckDBBackend:
    """An in-memory DuckDB warehouse built from the snapshot files (read-only use)."""
    db = DuckDBBackend(":memory:")
    for f in sorted(Path(folder or snapshot_dir()).glob("*__*.parquet")):
        schema, table = f.stem.split("__", 1)
        db.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        db.execute(f"CREATE TABLE {schema}.{table} AS SELECT * FROM read_parquet('{f.as_posix()}')")
    return db
