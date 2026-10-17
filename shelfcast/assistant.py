"""A restock assistant that answers questions by calling read-only tools.

The model never sees the whole warehouse and never writes SQL. It can only call four
functions over the tables the app already holds in memory: the current order list, one
product's forecast and stock, a what-if that re-runs the restock rules with different
settings, and a product search. Every tool call and its result are kept, so the app can
show what the assistant looked up, and the final answer is checked against those results:
a number that is not in them is flagged.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from .brief_ai import GroundingReport, check_grounding
from .llm import LLMError
from .restock import RestockSettings, recommend

MAX_ROUNDS = 4
SYSTEM_PROMPT = (
    "You are ShelfCast's restock assistant for a small online gift shop. You answer questions "
    "about what to order, how much, when, and why, using only the tools provided: get_restock_plan, "
    "get_product, what_if and search_products. Always call a tool before giving numbers, and use "
    "only numbers and product names that appear in tool results (you may round to whole units). "
    "If a question is not about this shop's stock, forecasts or orders, say briefly that you can "
    "only help with restocking. Keep answers short and concrete: two to five sentences, or a short "
    "list. Quantities are units; order_value is in GBP; days_left is how long current stock lasts "
    "at the expected (median) forecast; demand_mid is the expected units until the next delivery, "
    "demand_low and demand_high are its 80% range. Settings the user can change: days between "
    "orders (review_days, 1-14), safety buffer (buffer_pct, 0-100, where 100 plans for the high "
    "forecast) and a product's lead time in days."
)

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_restock_plan",
            "description": "The current order list under the current settings: one row per product with status, stock, expected demand, order quantity and a one-line reason. Most urgent first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["needs order", "Order now", "Order this week", "OK", "all"], "description": "Which rows to return. Default: needs order."},
                    "top": {"type": "integer", "description": "How many rows (1-20). Default 10."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "One product's plan row, its forecast totals for the next 7 and 14 days (low/expected/high) and its sales over the last 28 days.",
            "parameters": {
                "type": "object",
                "properties": {"product": {"type": "string", "description": "Stock code or (part of) the product name."}},
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "what_if",
            "description": "Re-run the restock rules with different settings and return the resulting order list (and, for one product, the change versus the current plan). Use it for questions like 'what if lead time becomes 5 days' or 'what if we order every 3 days'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "review_days": {"type": "integer", "description": "Days between orders, 1-14."},
                    "buffer_pct": {"type": "integer", "description": "Safety buffer, 0-100."},
                    "lead_time_days": {"type": "integer", "description": "Lead time in days, 1-60. Applies to the given product, or to every product when none is given."},
                    "product": {"type": "string", "description": "Stock code or (part of) a product name, to focus on one product."},
                    "top": {"type": "integer", "description": "How many rows (1-20). Default 10."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Find products by words in their name (e.g. 'lantern', 'christmas'), with their current status and order quantity.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "top": {"type": "integer", "description": "How many matches (1-20). Default 5."}},
                "required": ["query"],
            },
        },
    },
]

PLAN_COLUMNS = ["status", "stock_code", "description", "on_hand", "days_left", "lead_time_days", "demand_low", "demand_mid", "demand_high", "order_qty", "order_value", "why"]


def _clamp(value, low: int, high: int, default: int) -> int:
    try:
        return int(min(max(int(value), low), high))
    except (TypeError, ValueError):
        return default


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _totals(recs: pd.DataFrame) -> dict:
    return {
        "order_now": int((recs["status"] == "Order now").sum()),
        "order_this_week": int((recs["status"] == "Order this week").sum()),
        "products": int(len(recs)),
        "order_value_gbp": round(float(recs["order_value"].sum()), 2),
    }


class RestockTools:
    """The read-only functions the assistant may call. Built from the app's in-memory tables."""

    def __init__(self, forecasts: pd.DataFrame, inventory: pd.DataFrame, products: pd.DataFrame,
                 settings: RestockSettings, history: pd.DataFrame | None = None, as_of: str = ""):
        self.forecasts = forecasts
        self.inventory = inventory
        self.products = products
        self.settings = settings
        self.history = history
        self.as_of = as_of
        self.recs = recommend(forecasts, inventory, products, settings)
        self.names = dict(zip(products["stock_code"].astype(str), products["description"].fillna("").astype(str)))

    # -- helpers ---------------------------------------------------------------------------
    def find_product(self, text: str) -> str | None:
        """Stock code, exact name, substring, then best word overlap. None when nothing fits."""
        if text is None:
            return None
        text = str(text).strip()
        if text in self.names:
            return text
        lowered = text.lower()
        exact = [c for c, n in self.names.items() if n.lower() == lowered]
        if exact:
            return exact[0]
        contains = [c for c, n in self.names.items() if lowered and lowered in n.lower()]
        if contains:
            return sorted(contains, key=lambda c: len(self.names[c]))[0]
        words = set(re.findall(r"[a-z0-9]+", lowered))
        if not words:
            return None
        scored = [(len(words & set(re.findall(r"[a-z0-9]+", n.lower()))), c) for c, n in self.names.items()]
        best = max(scored, default=(0, None))
        return best[1] if best[0] > 0 else None

    def _plan_rows(self, recs: pd.DataFrame, status: str, top: int) -> pd.DataFrame:
        if status == "needs order":
            view = recs[recs["status"] != "OK"]
        elif status == "all":
            view = recs
        else:
            view = recs[recs["status"] == status]
        return view.head(top)[PLAN_COLUMNS]

    # -- tools -----------------------------------------------------------------------------
    def get_restock_plan(self, status: str = "needs order", top: int = 10) -> dict:
        top = _clamp(top, 1, 20, 10)
        if status not in ("needs order", "Order now", "Order this week", "OK", "all"):
            status = "needs order"
        rows = self._plan_rows(self.recs, status, top)
        return {"settings": {"review_days": self.settings.review_days, "buffer_pct": int(self.settings.buffer * 100)},
                "rows": _records(rows), "totals": _totals(self.recs)}

    def get_product(self, product: str) -> dict:
        code = self.find_product(product)
        if code is None:
            return {"error": f"No product matches '{product}'. Try search_products."}
        row = self.recs[self.recs["stock_code"] == code]
        fc = self.forecasts[self.forecasts["stock_code"] == code].sort_values("h")
        out: dict = {"product": _records(row[PLAN_COLUMNS])[0] if len(row) else {"stock_code": code, "description": self.names[code]}}
        horizon = int(fc["h"].max()) if len(fc) else 0
        for days in (7, 14):
            if horizon >= days:
                at = fc[fc["h"] == days].iloc[0]
                out[f"forecast_next_{days}_days"] = {
                    "low": round(float(at["total_p10"]), 1), "expected": round(float(at["total_p50"]), 1), "high": round(float(at["total_p90"]), 1)
                }
        if self.history is not None and len(self.history):
            h = self.history[self.history["stock_code"] == code]
            if len(h):
                last = pd.to_datetime(h["sale_date"]).max()
                recent = h[pd.to_datetime(h["sale_date"]) > last - pd.Timedelta(days=28)]
                out["units_sold_last_28_days"] = int(round(float(recent["units"].sum())))
        return out

    def what_if(self, review_days=None, buffer_pct=None, lead_time_days=None, product=None, top: int = 10) -> dict:
        settings = RestockSettings(
            review_days=_clamp(review_days, 1, 14, self.settings.review_days) if review_days is not None else self.settings.review_days,
            buffer=_clamp(buffer_pct, 0, 100, int(self.settings.buffer * 100)) / 100 if buffer_pct is not None else self.settings.buffer,
        )
        inventory = self.inventory.copy()
        code = self.find_product(product) if product else None
        if product and code is None:
            return {"error": f"No product matches '{product}'. Try search_products."}
        if lead_time_days is not None:
            lead = _clamp(lead_time_days, 1, 60, 7)
            mask = inventory["stock_code"].astype(str) == code if code else slice(None)
            inventory.loc[mask, "lead_time_days"] = lead
        recs = recommend(self.forecasts, inventory, self.products, settings)
        out = {
            "settings": {"review_days": settings.review_days, "buffer_pct": int(settings.buffer * 100),
                         "lead_time_days": lead_time_days if lead_time_days is not None else "unchanged",
                         "product": self.names.get(code, code) if code else "all"},
            "totals": _totals(recs),
        }
        if code:
            before = self.recs[self.recs["stock_code"] == code]
            after = recs[recs["stock_code"] == code]
            out["rows"] = _records(after[PLAN_COLUMNS])
            if len(before) and len(after):
                out["change"] = {
                    "order_qty_before": int(before["order_qty"].iloc[0]), "order_qty_after": int(after["order_qty"].iloc[0]),
                    "status_before": str(before["status"].iloc[0]), "status_after": str(after["status"].iloc[0]),
                }
        else:
            out["rows"] = _records(self._plan_rows(recs, "needs order", _clamp(top, 1, 20, 10)))
            out["totals_before"] = _totals(self.recs)
        return out

    def search_products(self, query: str, top: int = 5) -> dict:
        top = _clamp(top, 1, 20, 5)
        words = set(re.findall(r"[a-z0-9]+", str(query).lower()))
        if not words:
            return {"rows": []}
        scored = []
        for code, name in self.names.items():
            name_words = set(re.findall(r"[a-z0-9]+", name.lower()))
            hit = len(words & name_words) + (1 if str(query).lower() in name.lower() else 0)
            if hit:
                scored.append((hit, code))
        codes = [c for _, c in sorted(scored, key=lambda t: (-t[0], t[1]))[:top]]
        rows = self.recs[self.recs["stock_code"].isin(codes)][["status", "stock_code", "description", "on_hand", "days_left", "order_qty", "order_value"]]
        return {"rows": _records(rows)}

    def call(self, name: str, arguments: dict) -> dict:
        fn = {"get_restock_plan": self.get_restock_plan, "get_product": self.get_product, "what_if": self.what_if, "search_products": self.search_products}.get(name)
        if fn is None:
            return {"error": f"Unknown tool {name}"}
        try:
            return fn(**(arguments or {}))
        except TypeError as exc:  # unexpected argument names from the model
            return {"error": f"Bad arguments for {name}: {exc}"}


@dataclass
class ToolRun:
    name: str
    arguments: dict
    result: dict
    seconds: float = 0.0


@dataclass
class Answer:
    text: str
    tool_runs: list[ToolRun] = field(default_factory=list)
    report: GroundingReport | None = None
    model: str = ""
    rounds: int = 0
    error: str = ""

    @property
    def verified(self) -> bool:
        return self.report is not None and self.report.ok


def grounding_context(tool_runs: list[ToolRun], as_of: str) -> dict:
    """Everything the tools returned, in the shape ``check_grounding`` expects."""
    rows: list[dict] = []
    extra: list[dict] = []
    for run in tool_runs:
        for key, value in run.result.items():
            if key == "rows" and isinstance(value, list):
                rows.extend(value)
            elif key == "product" and isinstance(value, dict):
                rows.append(value)
            else:
                extra.append({key: value})
        extra.append({"arguments": run.arguments})
    return {"as_of": as_of, "rows": rows, "totals": {}, "other": extra}


class Assistant:
    def __init__(self, client, tools: RestockTools, max_rounds: int = MAX_ROUNDS):
        self.client = client
        self.tools = tools
        self.max_rounds = max_rounds

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history or []:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})
        answer = Answer(text="", model=getattr(self.client, "model", ""))
        corrected = False
        for _ in range(self.max_rounds + 1):
            try:
                resp = self.client.complete(messages, tools=TOOL_SPECS)
            except LLMError as exc:
                answer.error = str(exc)
                answer.text = "The assistant could not reach the model. " + str(exc)
                return answer
            answer.rounds += 1
            answer.model = resp.model or answer.model
            if resp.tool_calls:
                messages.append(resp.raw_message or {"role": "assistant", "content": None})
                for call in resp.tool_calls:
                    started = time.perf_counter()
                    result = self.tools.call(call.name, call.arguments)
                    answer.tool_runs.append(ToolRun(call.name, call.arguments, result, round(time.perf_counter() - started, 3)))
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
                continue
            answer.text = resp.text.strip()
            context = grounding_context(answer.tool_runs, self.tools.as_of)
            answer.report = check_grounding(answer.text, context, require_product_mention=False)
            if answer.report.ok or corrected:
                return answer
            corrected = True
            messages.append({"role": "assistant", "content": answer.text})
            messages.append({
                "role": "user",
                "content": "Some numbers in your answer are not in the tool results: "
                + ", ".join(answer.report.unverified_numbers + answer.report.unknown_codes)
                + ". Call a tool if you need more data, and answer again using only numbers from the results.",
            })
        if not answer.text:
            answer.text = "The assistant stopped after too many lookups without an answer. Please ask again more specifically."
            answer.report = check_grounding(answer.text, grounding_context(answer.tool_runs, self.tools.as_of), require_product_mention=False)
        return answer
