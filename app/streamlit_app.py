"""ShelfCast web app.

Run it with:  streamlit run app/streamlit_app.py

It reads the forecasts, stock levels and backtest from the warehouse (Snowflake, or the
local DuckDB file), then turns them into an order list you can tweak with the sliders.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # lets the app import shelfcast without installing it
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from shelfcast import data as datamod  # noqa: E402
from shelfcast import pipeline  # noqa: E402
from shelfcast.db import backend_kind, get_backend  # noqa: E402
from shelfcast.restock import RestockSettings, brief, cortex_brief, recommend  # noqa: E402

STATUS_COLOR = {"Order now": "#dc2626", "Order this week": "#d97706", "OK": "#16a34a"}
BLUE, GREY, RED = "#2563eb", "#111827", "#dc2626"

st.set_page_config(page_title="ShelfCast", page_icon=":material/inventory_2:", layout="wide")


def _has_forecasts(db) -> bool:
    return db.table_exists("MART", "FORECASTS") and db.table_exists("MART", "INVENTORY")


def _read_tables(db) -> dict:
    out = pipeline.read_plan_inputs(db)
    out["ready"] = True
    out["snapshot"] = False
    out["source"] = pipeline.data_source(db)
    out["backtest"] = db.query("SELECT * FROM MART.BACKTEST")
    out["runs"] = db.query(
        "SELECT run_id, created_at, backend, data_source, as_of, products, wape_shelfcast, "
        "improvement_vs_best_baseline, metrics_json, model_json FROM MART.MODEL_RUNS"
    )
    out["history"] = db.query(
        "SELECT stock_code, sale_date, units FROM MART.DAILY_SALES "
        "WHERE stock_code IN (SELECT DISTINCT stock_code FROM MART.FORECASTS)"
    )
    return out


@st.cache_data(ttl=900, show_spinner="Loading forecasts from the warehouse...")
def load_data(kind: str) -> dict:
    """``kind`` is snowflake, duckdb, or snapshot (the saved copy of a real run in demo/)."""
    out = None
    if kind != "snapshot":
        with get_backend(kind) as db:
            out = _read_tables(db) if _has_forecasts(db) else None
    if out is None and kind in ("duckdb", "snapshot") and pipeline.has_snapshot():
        with pipeline.snapshot_backend() as db:  # made with `python -m shelfcast snapshot`
            out = _read_tables(db)
        out["snapshot"] = True
    if out is None:
        return {"ready": False}
    bt, hist = out["backtest"], out["history"]
    bt["stock_code"] = bt["stock_code"].astype(str)
    bt["sale_date"] = pd.to_datetime(bt["sale_date"])
    hist["stock_code"] = hist["stock_code"].astype(str)
    hist["sale_date"] = pd.to_datetime(hist["sale_date"])
    hist["units"] = pd.to_numeric(hist["units"]).astype(float)
    out["runs"]["created_at"] = pd.to_datetime(out["runs"]["created_at"])
    return out


def latest_run(runs: pd.DataFrame) -> dict:
    row = runs.sort_values("created_at").iloc[-1]
    return {"run_id": row["run_id"], "metrics": json.loads(row["metrics_json"]), "model": json.loads(row["model_json"])}


def pretty_date(value) -> str:
    ts = pd.Timestamp(value)
    return f"{ts.day} {ts:%b %Y}"


def render_empty(kind: str) -> None:
    st.title("ShelfCast")
    st.info("No forecasts in the warehouse yet.")
    st.markdown(
        "Run the pipeline once from the project folder:\n\n"
        "```bash\n"
        "python -m shelfcast all --synthetic   # quick offline demo\n"
        "python -m shelfcast all               # real Online Retail II data\n"
        "```"
    )
    try:
        from shelfcast.forecast import TrainSettings  # needs PyTorch
    except ImportError:
        return
    if kind == "duckdb" and st.button("Build a demo with synthetic data now (about 30 seconds)", type="primary"):
        with st.spinner("Generating data, training the model and writing forecasts..."):
            with get_backend(kind) as db:
                tx = datamod.make_synthetic_transactions(n_products=40)
                pipeline.load(db, tx, pipeline.SYNTHETIC_SOURCE, log=lambda *_: None)
                pipeline.train(db, n_products=40, min_days=60, settings=TrainSettings(epochs=15), log=lambda *_: None)
        st.cache_data.clear()
        st.rerun()


def stock_editor(inventory: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    st.markdown(
        "The public sales data has no stock levels, so these are **simulated**. "
        "Type in your own numbers and every tab updates."
    )
    names = products.set_index("stock_code")["description"]
    table = inventory.assign(description=inventory["stock_code"].map(names).fillna(""))
    table = table[["stock_code", "description", "on_hand", "on_order", "lead_time_days", "unit_cost"]]
    edited = st.data_editor(
        table,
        key="stock_editor",
        hide_index=True,
        width="stretch",
        disabled=["stock_code", "description"],
        column_config={
            "stock_code": "Code",
            "description": "Product",
            "on_hand": st.column_config.NumberColumn("On hand", min_value=0, step=1, format="%d"),
            "on_order": st.column_config.NumberColumn("On order", min_value=0, step=1, format="%d"),
            "lead_time_days": st.column_config.NumberColumn("Lead time (days)", min_value=1, max_value=60, step=1),
            "unit_cost": st.column_config.NumberColumn("Unit cost (£)", min_value=0.0, format="%.2f"),
        },
    )
    edited = edited.drop(columns="description").copy()
    for col, low in (("on_hand", 0), ("on_order", 0), ("lead_time_days", 1)):
        edited[col] = pd.to_numeric(edited[col], errors="coerce").fillna(low).clip(lower=low).astype("int64")
    edited["unit_cost"] = pd.to_numeric(edited["unit_cost"], errors="coerce").fillna(0.0)
    return edited


def kpis(recs: pd.DataFrame, run: dict) -> None:
    m = run["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order now", int((recs["status"] == "Order now").sum()), help="Stock runs out before a new order could arrive.")
    c2.metric("Order this week", int((recs["status"] == "Order this week").sum()), help="An order is needed, but there is still time.")
    c3.metric("Order value at cost", f"£{recs['order_value'].sum():,.0f}")
    c4.metric(
        "Forecast error (WAPE)",
        f"{m['wape_shelfcast']:.0%}",
        delta=f"{-m['improvement_vs_best_baseline']:.0%} vs. best baseline",
        delta_color="inverse",
        help="Backtest on the last two weeks the model never saw. WAPE = total absolute error / total units sold.",
    )


def restock_list(recs: pd.DataFrame, as_of: str) -> None:
    show = st.radio("Show", ["Needs an order", "All products"], horizontal=True, label_visibility="collapsed")
    view = recs if show == "All products" else recs[recs["status"] != "OK"]
    styled = view.style.map(lambda s: f"color: {STATUS_COLOR.get(s, 'inherit')}; font-weight: 600", subset=["status"])
    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        height=520,
        column_order=[
            "status", "stock_code", "description", "on_hand", "days_left", "lead_time_days",
            "demand_mid", "order_qty", "order_value", "why",
        ],
        column_config={
            "status": "Status",
            "stock_code": "Code",
            "description": st.column_config.TextColumn("Product", width="medium"),
            "on_hand": st.column_config.NumberColumn("On hand", format="%d"),
            "days_left": st.column_config.NumberColumn("Days left", help="How long current stock lasts at the median forecast"),
            "lead_time_days": st.column_config.NumberColumn("Lead time"),
            "demand_mid": st.column_config.NumberColumn(
                "Expected demand", format="%.0f", help="Units expected to sell before the next delivery (median)"
            ),
            "order_qty": st.column_config.NumberColumn("Order qty", format="%d"),
            "order_value": st.column_config.NumberColumn("Order value", format="£%.2f"),
            "why": st.column_config.TextColumn("Why", width="large"),
        },
    )
    order = recs[recs["order_qty"] > 0][["stock_code", "description", "order_qty", "order_value", "status"]]
    st.download_button(
        "Download purchase order (CSV)",
        order.to_csv(index=False),
        file_name=f"shelfcast_order_after_{as_of}.csv",
        mime="text/csv",
    )


def daily_chart(hist: pd.DataFrame, fc: pd.DataFrame) -> alt.Chart:
    x = alt.X("date:T", title=None)
    actual = alt.Chart(hist).mark_line(color=GREY, point=alt.OverlayMarkDef(size=12, color=GREY)).encode(
        x=x, y=alt.Y("units:Q", title="Units per day"), tooltip=["date:T", alt.Tooltip("units:Q", format=".0f")]
    )
    band = alt.Chart(fc).mark_area(color=BLUE, opacity=0.18).encode(x=x, y="p10:Q", y2="p90:Q")
    median = alt.Chart(fc).mark_line(color=BLUE, strokeWidth=2.5).encode(
        x=x,
        y="p50:Q",
        tooltip=["date:T", alt.Tooltip("p10:Q", format=".0f"), alt.Tooltip("p50:Q", format=".0f"), alt.Tooltip("p90:Q", format=".0f")],
    )
    return (band + actual + median).properties(height=300)


def stock_chart(fc: pd.DataFrame, on_hand: int, cover: int) -> alt.Chart:
    x = alt.X("date:T", title=None)
    band = alt.Chart(fc).mark_area(color=BLUE, opacity=0.18).encode(x=x, y="total_p10:Q", y2="total_p90:Q")
    median = alt.Chart(fc).mark_line(color=BLUE, strokeWidth=2.5).encode(
        x=x,
        y=alt.Y("total_p50:Q", title="Units sold since today (running total)"),
        tooltip=["date:T", alt.Tooltip("total_p10:Q", format=".0f"), alt.Tooltip("total_p50:Q", format=".0f"), alt.Tooltip("total_p90:Q", format=".0f")],
    )
    stock = alt.Chart(pd.DataFrame({"y": [on_hand]})).mark_rule(color=RED, strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
    label = alt.Chart(pd.DataFrame({"date": [fc["date"].min()], "y": [on_hand], "text": [f"stock on hand: {on_hand}"]})).mark_text(
        color=RED, align="left", dy=-8
    ).encode(x="date:T", y="y:Q", text="text:N")
    cover_rule = alt.Chart(fc[fc["h"] == cover]).mark_rule(color=GREY, strokeDash=[2, 3]).encode(x="date:T")
    return (band + median + stock + label + cover_rule).properties(height=300)


def backtest_chart(bt: pd.DataFrame) -> alt.Chart:
    x = alt.X("sale_date:T", title=None)
    band = alt.Chart(bt).mark_area(color=BLUE, opacity=0.18).encode(x=x, y="p10:Q", y2="p90:Q")
    median = alt.Chart(bt).mark_line(color=BLUE, strokeWidth=2.5).encode(x=x, y=alt.Y("p50:Q", title="Units per day"))
    naive = alt.Chart(bt).mark_line(color="#9ca3af", strokeDash=[4, 3]).encode(x=x, y="naive:Q")
    actual = alt.Chart(bt).mark_line(color=GREY, point=alt.OverlayMarkDef(size=14, color=GREY)).encode(x=x, y="actual:Q")
    return (band + naive + median + actual).properties(height=260)


def product_view(recs: pd.DataFrame, data: dict, settings: RestockSettings, horizon: int) -> None:
    labels = {r.stock_code: f"{r.description or r.stock_code} ({r.stock_code}) - {r.status}" for r in recs.itertuples()}
    code = st.selectbox("Product", list(labels), format_func=labels.get, key="product")
    rec = recs.set_index("stock_code").loc[code]
    cover = int(min(rec["lead_time_days"] + settings.review_days, horizon))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("On hand", f"{rec['on_hand']:,}")
    c2.metric("Days of stock left", f"{rec['days_left']}" if rec["days_left"] < horizon else f"{horizon}+")
    c3.metric(f"Expected demand, next {cover} days", f"{rec['demand_mid']:,.0f}", help=f"80% range {rec['demand_low']:,.0f} to {rec['demand_high']:,.0f}")
    c4.metric("Suggested order", f"{rec['order_qty']:,}", help=f"£{rec['order_value']:,.2f} at cost")
    st.info(rec["why"])

    fc = data["forecasts"][data["forecasts"]["stock_code"] == code].rename(columns={"forecast_date": "date"})
    as_of = fc["date"].min() - pd.offsets.Day(1)
    days = pd.date_range(as_of - pd.offsets.Day(55), as_of, freq="D")
    hist = data["history"][data["history"]["stock_code"] == code].set_index("sale_date")["units"]
    hist = hist.reindex(days, fill_value=0.0).rename_axis("date").reset_index()

    left, right = st.columns(2)
    with left:
        st.markdown("**Daily sales and forecast**")
        st.altair_chart(daily_chart(hist, fc), width="stretch")
        st.caption("Black: actual sales, last 8 weeks. Blue: median forecast with its 80% range (P10 to P90).")
    with right:
        st.markdown("**Will the stock last?**")
        st.altair_chart(stock_chart(fc, int(rec["on_hand"]), cover), width="stretch")
        st.caption(
            "Blue: expected running total of sales with its 80% range. Red: stock on hand. "
            "Where blue crosses red, the shelf is empty. Dotted line: the end of the period this order has to cover "
            "(lead time + days between orders)."
        )

    bt = data["backtest"][data["backtest"]["stock_code"] == code]
    with st.expander("How did the model do on this product in the backtest?"):
        if len(bt):
            st.altair_chart(backtest_chart(bt), width="stretch")
            st.caption(
                f"The last {len(bt)} days of data, hidden from the model during training. Black: what really sold. "
                "Blue: ShelfCast forecast and 80% range. Grey dashes: 'same weekday last week' baseline."
            )


def model_view(run: dict, runs: pd.DataFrame) -> None:
    m, info = run["metrics"], run["model"]
    st.markdown(
        f"**Backtest:** the model was trained without the last 14 days of data ({m['test_period']}), "
        f"then asked to forecast them for {m['products']} products."
    )
    table = pd.DataFrame({
        "Forecast": ["ShelfCast (PyTorch)", "Same weekday last week", "28-day average"],
        "Daily error (WAPE)": [m["wape_shelfcast"], m["wape_same_weekday_last_week"], m["wape_28_day_average"]],
        "Two-week total error (WAPE)": [
            m["wape_total_shelfcast"], m["wape_total_same_weekday_last_week"], m["wape_total_28_day_average"]
        ],
    })
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Daily error (WAPE)": st.column_config.NumberColumn(format="percent"),
                "Two-week total error (WAPE)": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.markdown(
            f"- Daily error is **{m['improvement_vs_best_baseline']:.0%} lower** than the best simple baseline.\n"
            f"- **{m['coverage_total_p10_p90']:.0%}** of real two-week totals landed inside the model's 80% range "
            f"(a well-calibrated range catches about 80%).\n"
            f"- Adding up daily ranges instead would make that range much wider and catch "
            f"{m['coverage_total_summed_daily']:.0%}, which means ordering more safety stock than needed. "
            "That is why the model forecasts running totals directly.\n"
            "- WAPE = total absolute error / total units sold. Lower is better."
        )
    with right:
        bars = alt.Chart(table).mark_bar().encode(
            x=alt.X("Daily error (WAPE):Q", axis=alt.Axis(format="%"), title="Daily error (WAPE), lower is better"),
            y=alt.Y("Forecast:N", sort=None, title=None),
            color=alt.condition(alt.datum.Forecast == "ShelfCast (PyTorch)", alt.value(BLUE), alt.value("#9ca3af")),
        ).properties(height=160)
        st.altair_chart(bars, width="stretch")

    st.markdown("**The model**")
    st.markdown(
        f"- {info['type']} with {info['parameters']:,} parameters, one network for all products\n"
        f"- Inputs: {info['inputs']}\n"
        f"- Outputs: {info['outputs']}\n"
        f"- Trained with quantile (pinball) loss on {m['training_windows']:,} windows; "
        f"{m['train_seconds']} s on a CPU, no GPU needed"
    )
    st.caption(
        "Why is daily error so high for every method? Single products sell in lumpy wholesale orders "
        "(0 units one day, 400 the next), so day-level numbers are noisy. What matters is the comparison "
        "with the baselines, and the two-week totals that orders are based on."
    )
    with st.expander("Training runs"):
        st.dataframe(
            runs.sort_values("created_at", ascending=False)[
                ["run_id", "created_at", "backend", "data_source", "as_of", "products", "wape_shelfcast", "improvement_vs_best_baseline"]
            ],
            hide_index=True,
            width="stretch",
        )


def brief_view(kind: str, recs: pd.DataFrame, as_of: str, settings: RestockSettings) -> None:
    key = (settings.review_days, settings.buffer)
    if kind == "snowflake":
        st.markdown("Snowflake Cortex turns the order list into a short note for the store manager, inside Snowflake.")
        if st.button("Write the brief with Snowflake Cortex", type="primary"):
            with st.spinner("Asking Cortex..."):
                try:
                    with get_backend(kind) as db:
                        text, model = cortex_brief(db, recs, as_of, os.environ.get("SHELFCAST_CORTEX_MODEL") or None)
                    st.session_state["cortex"] = {"key": key, "text": text, "model": model}
                except Exception as exc:  # noqa: BLE001 - show the template instead
                    st.warning(f"Cortex didn't answer, so here is the built-in summary instead. ({exc})")
        saved = st.session_state.get("cortex")
        if saved and saved["key"] == key:
            st.success(saved["text"])
            st.caption(f"Written by {saved['model']} in Snowflake Cortex, using only the order list above.")
            return
    else:
        st.info("Running on local DuckDB, so this summary comes from a simple template. Connect Snowflake to have Cortex write it.")
    text = brief(recs, as_of)
    with st.container(border=True):
        st.markdown(text.replace("\n- ", "\n\n- ", 1))


def render_app(kind: str, data: dict) -> None:
    fc, inventory, products = data["forecasts"], data["inventory"], data["products"]
    horizon = int(fc["h"].max())
    as_of = data["as_of"]
    run = latest_run(data["runs"])

    with st.sidebar:
        st.header("Planning settings")
        review_days = st.slider(
            "Days between orders", 1, 14, 7,
            help="How often you place orders. Each order has to last until the next one arrives.",
        )
        buffer_pct = st.slider(
            "Safety buffer", 0, 100, 100, step=10, format="%d%%",
            help="0% plans for the expected (median) demand. 100% plans for the high (P90) forecast: "
            "about a 90% chance of not running out. More buffer means fewer stock-outs but more cash tied up in stock.",
        )
        st.divider()
        warehouse = "Snowflake" if kind == "snowflake" else "DuckDB (local)"
        if data["snapshot"]:
            warehouse = "DuckDB (saved snapshot in demo/)"
        st.markdown(f"**Warehouse:** {warehouse}")
        st.markdown(f"**Data:** {data['source']}")
        st.markdown(f"**Sales data up to:** {pretty_date(as_of)}")
        st.caption("Stock levels are simulated, because the public dataset has none. Edit them in the Stock levels tab.")
        if st.button("Reload from warehouse"):
            st.cache_data.clear()
            st.rerun()

    st.title("ShelfCast")
    st.markdown(f"##### What to restock and why: forecasts for the {horizon} days after {pretty_date(as_of)}")
    if "synthetic" in data["source"].lower():
        st.warning("Demo mode: these are synthetic sales. Run `python -m shelfcast all` to use the real Online Retail II data.")
    if data["snapshot"]:
        st.info(
            "You are looking at a saved snapshot of a real run (the demo/ folder). "
            "Run `python -m shelfcast all` to build your own, on DuckDB or Snowflake."
        )
    kpi_box = st.container()
    tab_list, tab_product, tab_stock, tab_model, tab_brief = st.tabs(
        ["Restock list", "Product forecast", "Stock levels", "Model quality", "AI brief"]
    )
    with tab_stock:
        edited = stock_editor(inventory, products)
    settings = RestockSettings(review_days=review_days, buffer=buffer_pct / 100)
    recs = recommend(fc, edited, products, settings)
    if recs.empty:
        st.error("No products to plan for: the forecasts and stock tables have no products in common.")
        return

    with kpi_box:
        kpis(recs, run)
    with tab_list:
        restock_list(recs, as_of)
    with tab_product:
        product_view(recs, data, settings, horizon)
    with tab_model:
        model_view(run, data["runs"])
    with tab_brief:
        brief_view(kind, recs, as_of, settings)


def secrets_to_env() -> None:
    """On Streamlit Community Cloud, credentials live in st.secrets; copy the top-level
    ones (SNOWFLAKE_ACCOUNT, ...) into the environment, where shelfcast.db looks for them."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float)):
                os.environ.setdefault(key, str(value))
    except Exception:  # noqa: BLE001 - no secrets file, which is fine locally
        pass


def main() -> None:
    secrets_to_env()
    try:
        kind = backend_kind()
    except ValueError as exc:
        st.error(str(exc))
        return
    try:
        data = load_data(kind)
    except Exception as exc:  # noqa: BLE001 - show a readable message instead of a stack trace
        print(f"ShelfCast could not read from {kind}: {exc!r}", file=sys.stderr)
        if not pipeline.has_snapshot():
            st.title("ShelfCast")
            st.error(f"Could not read from {kind} ({type(exc).__name__}). Details are in the terminal.")
            if kind == "snowflake":
                st.info("Check the SNOWFLAKE_* values in .env, then run `python -m shelfcast check`.")
            return
        # e.g. a deployed app whose Snowflake trial or token has expired: keep the demo working
        st.toast(f"Could not reach {kind}, so this is the saved snapshot of a real run.")
        kind, data = "snapshot", load_data("snapshot")
    if not data["ready"]:
        render_empty(kind)
        return
    render_app(kind, data)


main()
