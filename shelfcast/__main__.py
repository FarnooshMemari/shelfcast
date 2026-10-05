"""Command line: ``python -m shelfcast --help``.

Typical runs:
    python -m shelfcast all --synthetic        # offline demo on local DuckDB (about a minute)
    python -m shelfcast all                    # real data; Snowflake if .env has credentials
    python -m shelfcast --backend snowflake check
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import data as datamod
from . import pipeline
from .db import duckdb_path, get_backend
from .forecast import TrainSettings
from .restock import RestockSettings, brief, cortex_brief, recommend


def _transactions(args):
    if getattr(args, "excel", None):
        print(f"[data] converting {args.excel}")
        datamod.convert_local_excel(Path(args.excel))
        return datamod.read_transactions(datamod.REAL_DATA), pipeline.REAL_SOURCE
    if args.synthetic:
        print("[data] using synthetic transactions (offline demo)")
        return datamod.get_transactions(synthetic=True), pipeline.SYNTHETIC_SOURCE
    return datamod.get_transactions(synthetic=False), pipeline.REAL_SOURCE


def _settings(args) -> TrainSettings:
    return TrainSettings(epochs=args.epochs, seed=args.seed)


def cmd_data(args) -> None:
    tx, source = _transactions(args)
    print(f"[data] {len(tx):,} rows ready ({source})")


def cmd_load(args) -> None:
    tx, source = _transactions(args)
    with get_backend(args.backend) as db:
        pipeline.load(db, tx, source)


def cmd_train(args) -> None:
    with get_backend(args.backend) as db:
        pipeline.train(db, args.products, args.min_days, _settings(args), args.reset_inventory)


def cmd_all(args) -> None:
    tx, source = _transactions(args)
    with get_backend(args.backend) as db:
        pipeline.load(db, tx, source)
        out = pipeline.train(db, args.products, args.min_days, _settings(args), args.reset_inventory)
    print()
    print(brief(out["recommendations"], out["metrics"]["as_of"]))
    print("\nNext: streamlit run app/streamlit_app.py")


def cmd_brief(args) -> None:
    with get_backend(args.backend) as db:
        inputs = pipeline.read_plan_inputs(db)
        recs = recommend(inputs["forecasts"], inputs["inventory"], inputs["products"], RestockSettings())
        if args.cortex:
            text, model = cortex_brief(db, recs, inputs["as_of"], os.environ.get("SHELFCAST_CORTEX_MODEL"))
            print(f"(written by {model} in Snowflake Cortex)\n{text}")
        else:
            print(brief(recs, inputs["as_of"]))


def cmd_snapshot(args) -> None:
    with get_backend(args.backend) as db:
        out = pipeline.write_snapshot(db, Path(args.out) if args.out else pipeline.SNAPSHOT_DIR)
    print(f"Saved. The app falls back to {out} when the warehouse has no forecasts.")


def cmd_check(args) -> None:
    with get_backend(args.backend) as db:
        if db.name == "snowflake":
            row = db.query(
                "SELECT CURRENT_USER() AS user_name, CURRENT_ROLE() AS role_name, "
                "CURRENT_WAREHOUSE() AS warehouse, CURRENT_DATABASE() AS database_name"
            ).iloc[0]
            print(f"Connected to Snowflake as {row['user_name']} (role {row['role_name']}, "
                  f"warehouse {row['warehouse']}, database {row['database_name']})")
        else:
            print(f"Using local DuckDB at {duckdb_path()}")
        for schema, table in (("RAW", "TRANSACTIONS"), ("MART", "DAILY_SALES"), ("MART", "FORECASTS"), ("MART", "RESTOCK_PLAN")):
            print(f"  {schema}.{table}: {'yes' if db.table_exists(schema, table) else 'not yet'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m shelfcast", description="ShelfCast: demand forecasts that turn into restock decisions.")
    parser.add_argument("--backend", choices=["auto", "duckdb", "snowflake"], default="auto",
                        help="auto = Snowflake when SNOWFLAKE_ACCOUNT is set (e.g. in .env), else local DuckDB")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p):
        p.add_argument("--synthetic", action="store_true", help="use generated data instead of downloading the real dataset")
        p.add_argument("--excel", help="path to online_retail_II.xlsx if you downloaded it yourself")

    def train_args(p):
        p.add_argument("--products", type=int, default=200, help="how many best-selling products to forecast")
        p.add_argument("--min-days", type=int, default=120, help="only products that sold on at least this many days")
        p.add_argument("--epochs", type=int, default=40, help="maximum training epochs (early stopping picks the best)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--reset-inventory", action="store_true", help="re-simulate stock levels instead of keeping them")

    p = sub.add_parser("data", help="download the real dataset (or make synthetic data)")
    data_args(p)
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("load", help="load transactions into the warehouse and build the SQL views")
    data_args(p)
    p.set_defaults(func=cmd_load)
    p = sub.add_parser("train", help="backtest, train, forecast and write the restock plan")
    train_args(p)
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("all", help="data + load + train in one go")
    data_args(p)
    train_args(p)
    p.set_defaults(func=cmd_all)
    p = sub.add_parser("brief", help="print the restock brief")
    p.add_argument("--cortex", action="store_true", help="have Snowflake Cortex write it")
    p.set_defaults(func=cmd_brief)
    p = sub.add_parser("snapshot", help="save the app's tables as small Parquet files in demo/")
    p.add_argument("--out", help="folder to write to (default: demo/)")
    p.set_defaults(func=cmd_snapshot)
    p = sub.add_parser("check", help="test the warehouse connection and list the tables")
    p.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
