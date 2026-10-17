"""Command line: ``python -m shelfcast --help``.

Typical runs:
    python -m shelfcast all --synthetic        # offline demo on local DuckDB (about a minute)
    python -m shelfcast all                    # real data; Snowflake if .env has credentials
    python -m shelfcast --backend snowflake check
    python -m shelfcast brief --ai             # the AI brief, checked against the order list
    python -m shelfcast ask "what if lead times go up to 10 days?"
    python -m shelfcast evals --models gpt-4o-mini,llama-3.1-8b-instant
"""
from __future__ import annotations

import argparse
import json
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


def _plan_inputs(backend: str) -> dict:
    """The app's tables from the warehouse, or from the saved snapshot when it has none."""
    with get_backend(backend) as db:
        if db.table_exists("MART", "FORECASTS") and db.table_exists("MART", "INVENTORY"):
            return pipeline.read_plan_inputs(db) | {"history": db.query("SELECT stock_code, sale_date, units FROM MART.DAILY_SALES")}
    if not pipeline.has_snapshot():
        raise SystemExit("No forecasts yet. Run `python -m shelfcast all` first (or `all --synthetic`).")
    with pipeline.snapshot_backend() as db:
        return pipeline.read_plan_inputs(db) | {"history": db.query("SELECT stock_code, sale_date, units FROM MART.DAILY_SALES")}


def cmd_brief(args) -> None:
    if args.cortex:
        with get_backend(args.backend) as db:
            inputs = pipeline.read_plan_inputs(db)
            recs = recommend(inputs["forecasts"], inputs["inventory"], inputs["products"], RestockSettings())
            text, model = cortex_brief(db, recs, inputs["as_of"], os.environ.get("SHELFCAST_CORTEX_MODEL"))
        print(f"(written by {model} in Snowflake Cortex)\n{text}")
        return
    inputs = _plan_inputs(args.backend)
    recs = recommend(inputs["forecasts"], inputs["inventory"], inputs["products"], RestockSettings())
    if not args.ai:
        print(brief(recs, inputs["as_of"]))
        return
    from .brief_ai import ai_brief
    from .llm import client_from_env

    client = client_from_env(args.model)
    if client is None:
        raise SystemExit("Set SHELFCAST_LLM_API_KEY (or OPENAI_API_KEY) and SHELFCAST_LLM_MODEL to use --ai; see .env.example")
    result = ai_brief(client, recs, inputs["as_of"])
    print(result.text)
    if result.source == "ai":
        print(f"\n(written by {result.model}; {result.report.summary()})")
    else:
        print(f"\n(built-in summary shown instead: {result.fallback_reason})")


def cmd_ask(args) -> None:
    from .assistant import Assistant, RestockTools
    from .llm import client_from_env

    client = client_from_env(args.model)
    if client is None:
        raise SystemExit("Set SHELFCAST_LLM_API_KEY (or OPENAI_API_KEY) and SHELFCAST_LLM_MODEL first; see .env.example")
    inputs = _plan_inputs(args.backend)
    tools = RestockTools(inputs["forecasts"], inputs["inventory"], inputs["products"],
                         RestockSettings(args.review_days, args.buffer / 100), inputs["history"], inputs["as_of"])
    answer = Assistant(client, tools).ask(" ".join(args.question))
    for run in answer.tool_runs:
        print(f"[tool] {run.name}({json.dumps(run.arguments)})")
    print()
    print(answer.text)
    if answer.report is not None:
        print(f"\n({answer.model}; {answer.report.summary()})")


def cmd_evals(args) -> None:
    from . import evals
    from .llm import client_from_env

    if args.make_cases or not evals.CASES_PATH.exists():
        inputs = _plan_inputs(args.backend)
        cases = evals.make_cases(inputs)
        print(f"[evals] wrote {len(cases)} cases to {evals.CASES_PATH}")
        if args.make_cases:
            return
    cases = evals.load_cases()
    clients: dict = {"template": "template"}
    names = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if not names:
        default = client_from_env()
        if default is not None:
            clients[default.model] = default
    for name in names:
        client = client_from_env(name)
        if client is None:
            raise SystemExit("Set SHELFCAST_LLM_API_KEY (or OPENAI_API_KEY) to evaluate a model; see .env.example")
        clients[name] = client
    if len(clients) == 1:
        print("[evals] no model configured, scoring the template baseline only (set SHELFCAST_LLM_API_KEY or pass --models)")
    scores = evals.run(cases, clients)
    path = evals.write_report(scores)
    print()
    for s in scores:
        print(f"{s.model:<28} grounded first try {s.grounded_first_try:>4.0%}   after retry {s.grounded:>4.0%}   fallback {s.fallback:>4.0%}   {s.mean_words:.0f} words   {s.mean_seconds:.1f} s")
    print(f"\nReport: {path}")


def cmd_feedback(args) -> None:
    from . import feedback

    entries = feedback.load()
    if not entries:
        print(f"No feedback yet ({feedback.feedback_path()}).")
        return
    for row in feedback.summary(entries):
        verified = "-" if row["verified_share"] is None else f"{row['verified_share']:.0%}"
        print(f"{row['feature']:<10} {row['model'] or '(no model)':<24} {row['ratings']:>4} ratings   {row['rated_up']:>4.0%} rated up   verified {verified}")


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
    p.add_argument("--ai", action="store_true", help="have the configured model write it, checked against the order list")
    p.add_argument("--model", help="model name for --ai (default: SHELFCAST_LLM_MODEL)")
    p.add_argument("--cortex", action="store_true", help="have Snowflake Cortex write it")
    p.set_defaults(func=cmd_brief)
    p = sub.add_parser("ask", help="ask the restock assistant a question (it answers with read-only tools)")
    p.add_argument("question", nargs="+")
    p.add_argument("--model", help="model name (default: SHELFCAST_LLM_MODEL)")
    p.add_argument("--review-days", type=int, default=7, help="days between orders")
    p.add_argument("--buffer", type=int, default=100, help="safety buffer in percent")
    p.set_defaults(func=cmd_ask)
    p = sub.add_parser("evals", help="score the AI brief on the saved order lists and write reports/brief_evals.md")
    p.add_argument("--models", help="comma-separated model names to compare (default: SHELFCAST_LLM_MODEL)")
    p.add_argument("--make-cases", action="store_true", help="rebuild evals/brief_cases.json from the warehouse and stop")
    p.set_defaults(func=cmd_evals)
    p = sub.add_parser("feedback", help="summarise the thumbs up / down given in the app")
    p.set_defaults(func=cmd_feedback)
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
