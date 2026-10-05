"""Turn forecasts into an order list.

Same rules as sql/20_restock_plan.sql. An order has to cover demand until the next one
arrives (lead time + days between orders). Target stock = P50 + buffer * (P90 - P50) of
the running-total forecast for that period; order quantity = target - on hand - on order.
"Order now" means the shelf runs empty (at the P50 forecast) before a new order could arrive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

STATUS_ORDER = {"Order now": 0, "Order this week": 1, "OK": 2}
CORTEX_MODELS = ("llama3.3-70b", "mistral-large2", "llama3.1-70b")


@dataclass
class RestockSettings:
    review_days: int = 7  # how often orders are placed
    buffer: float = 1.0  # 0 = plan for the median forecast, 1 = plan for the high (P90) forecast


def simulate_inventory(matrix: pd.DataFrame, prices: pd.Series | None = None, seed: int = 7) -> pd.DataFrame:
    """The public dataset has no stock levels, so we simulate them for the demo: on-hand
    stock covers between 1 and 30 days of recent sales, lead times are 3-10 days."""
    rng = np.random.default_rng(seed)
    recent = matrix.tail(28).mean()
    rows = []
    for code in matrix.columns:
        days = rng.uniform(1, 30)
        price = float(prices.get(code, 2.5)) if prices is not None else 2.5
        rows.append({
            "stock_code": str(code),
            "on_hand": int(round(recent[code] * days)),
            "on_order": 0,
            "lead_time_days": int(rng.choice([3, 5, 7, 10])),
            "unit_cost": round(0.6 * price, 2),  # assumes a 40% margin on the selling price
        })
    return pd.DataFrame(rows)


def recommend(
    forecasts: pd.DataFrame,
    inventory: pd.DataFrame,
    products: pd.DataFrame | None = None,
    settings: RestockSettings | None = None,
) -> pd.DataFrame:
    """One row per product: how much to order and why.

    ``forecasts``: stock_code, h (1..horizon), total_p10, total_p50, total_p90 (running totals).
    ``inventory``: stock_code, on_hand, on_order, lead_time_days, unit_cost.
    """
    settings = settings or RestockSettings()
    names = {}
    if products is not None and len(products):
        names = dict(zip(products["stock_code"].astype(str), products["description"].fillna("").astype(str)))
    fc = forecasts.copy()
    fc["stock_code"] = fc["stock_code"].astype(str)
    horizon = int(fc["h"].max())
    groups = {code: g.sort_values("h") for code, g in fc.groupby("stock_code")}
    inv = inventory.copy()
    inv["stock_code"] = inv["stock_code"].astype(str)

    rows = []
    for item in inv.itertuples(index=False):
        f = groups.get(item.stock_code)
        if f is None or f.empty:
            continue
        on_hand, on_order, lead = int(item.on_hand), int(item.on_order), int(item.lead_time_days)
        cover = int(min(lead + settings.review_days, horizon))
        at_cover = f[f["h"] == cover].iloc[0]
        low, mid, high = (float(at_cover[c]) for c in ("total_p10", "total_p50", "total_p90"))
        target = mid + settings.buffer * (high - mid)
        order_qty = max(0, math.ceil(target - on_hand - on_order))

        runs_out = np.flatnonzero(f["total_p50"].to_numpy() > on_hand)
        days_left = int(f["h"].iloc[runs_out[0]] - 1) if len(runs_out) else horizon
        if days_left < lead:
            status = "Order now"
        elif order_qty > 0:
            status = "Order this week"
        else:
            status = "OK"

        low_r, mid_r, high_r = round(low, 1), round(mid, 1), round(high, 1)  # what the tables show
        lasts = f"lasts about {days_left} days" if days_left < horizon else f"lasts {horizon}+ days"
        why = (
            f"Next {cover} days: expect {mid_r:,.0f} units (80% range {low_r:,.0f}-{high_r:,.0f}). "
            f"{on_hand:,} on hand {lasts}; a new order takes {lead} days."
        )
        rows.append({
            "status": status,
            "stock_code": item.stock_code,
            "description": names.get(item.stock_code, ""),
            "on_hand": on_hand,
            "days_left": days_left,
            "lead_time_days": lead,
            "demand_low": low_r,
            "demand_mid": mid_r,
            "demand_high": high_r,
            "order_qty": int(order_qty),
            "order_value": round(order_qty * float(item.unit_cost), 2),
            "why": why,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_rank"] = out["status"].map(STATUS_ORDER)
    out = out.sort_values(["_rank", "order_value"], ascending=[True, False]).drop(columns="_rank")
    return out.reset_index(drop=True)


def with_summed_daily_totals(forecasts: pd.DataFrame) -> pd.DataFrame:
    """The common shortcut: running totals made by adding up daily quantiles. Used only to
    show how much more stock that would buy than the model's own running-total forecast."""
    out = forecasts.sort_values(["stock_code", "h"]).copy()
    for q in ("p10", "p50", "p90"):
        out[f"total_{q}"] = out.groupby("stock_code")[q].cumsum()
    return out


def brief(recs: pd.DataFrame, as_of: str, top: int = 5) -> str:
    """A plain-language summary of the order list (no LLM needed)."""
    if recs.empty:
        return "No products to review."
    urgent = recs[recs["status"] == "Order now"]
    soon = recs[recs["status"] == "Order this week"]
    total = recs["order_value"].sum()
    lines = [
        f"Restock brief for the two weeks after {as_of}.",
        f"{len(urgent)} products need an order today and {len(soon)} more this week, "
        f"about £{total:,.0f} in total at cost.",
    ]
    for r in recs.head(top).itertuples(index=False):
        if r.status == "OK":
            break
        lines.append(f"- {r.description or r.stock_code}: order {r.order_qty}. {r.why}")
    return "\n".join(lines)


def cortex_prompt(recs: pd.DataFrame, as_of: str, top: int = 8) -> str:
    cols = ["status", "description", "on_hand", "days_left", "lead_time_days", "demand_mid", "order_qty", "order_value"]
    table = recs.head(top)[cols]
    return (
        "You help a small online gift retailer plan restocking. Using only the table below, "
        "write a short, friendly brief for the store manager (under 120 words): what to order "
        "first and why, and one risk to watch. Do not invent numbers that are not in the table. "
        "days_left is how long current stock lasts; demand_mid is expected units until the next "
        f"delivery; order_value is in GBP. The forecast starts the day after {as_of}.\n\n"
        f"{table.to_csv(index=False)}"
    )


def cortex_brief(backend, recs: pd.DataFrame, as_of: str, model: str | None = None) -> tuple[str, str]:
    """Ask Snowflake Cortex to write the brief; returns (text, model used).

    Tries the newer AI_COMPLETE function first, then SNOWFLAKE.CORTEX.COMPLETE, with the
    given model or a few common ones. Raises RuntimeError if none of them work.
    """
    prompt = cortex_prompt(recs, as_of)
    models = [model] if model else list(CORTEX_MODELS)
    last_error = None
    for name in models:
        for sql in ("SELECT AI_COMPLETE(%s, %s) AS answer", "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS answer"):
            try:
                answer = backend.query(sql, (name, prompt))["answer"].iloc[0]
                return str(answer).strip().strip('"'), name
            except Exception as exc:  # noqa: BLE001 - try the next function / model
                last_error = exc
    raise RuntimeError(f"Cortex is not available here: {last_error}")
