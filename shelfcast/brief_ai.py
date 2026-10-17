"""The AI restock brief, with a grounding check.

The model gets the top rows of the order list and writes a short note for the store manager.
Before the note is shown, ``check_grounding`` verifies it against the same table: every
number in the note must appear in the table (or be one of its totals), no unknown stock
codes may appear, at least one of the most urgent products must be mentioned, and the note
must stay short. If the note fails, the model gets one chance to fix it; if it fails again,
the app shows the plain built-in summary instead and says why.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from .llm import LLMError
from .restock import brief as template_brief

TABLE_COLUMNS = [
    "status", "stock_code", "description", "on_hand", "days_left", "lead_time_days",
    "demand_low", "demand_mid", "demand_high", "order_qty", "order_value", "why",
]
MAX_WORDS = 200
NUMBER = re.compile(r"(?<![\w.])[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?(?![\w.]*\d)")
STOCK_CODE = re.compile(r"\b\d{5}[A-Za-z]?\b")

SYSTEM_PROMPT = (
    "You help a small online gift retailer plan restocking. You will get the top rows of "
    "today's order list as JSON, plus totals. Write a short, friendly brief for the store "
    "manager (under 120 words, plain text, no headings): what to order first and why, then "
    "one risk to watch. Rules: use only numbers that appear in the data you were given, "
    "exactly as given (you may round to whole units); name products as they are written in "
    "the data; do not mention products that are not in the data; order_value is in GBP; "
    "days_left is how long current stock lasts; demand_mid is the expected units until the "
    "next delivery; demand_low and demand_high are the 80% range."
)


@dataclass
class GroundingReport:
    numbers_checked: int = 0
    unverified_numbers: list[str] = field(default_factory=list)
    unknown_codes: list[str] = field(default_factory=list)
    mentions_top_product: bool = True
    words: int = 0

    @property
    def ok(self) -> bool:
        return not self.unverified_numbers and not self.unknown_codes and self.mentions_top_product and self.words <= MAX_WORDS

    @property
    def problems(self) -> list[str]:
        out = []
        if self.unverified_numbers:
            out.append("numbers not in the order list: " + ", ".join(self.unverified_numbers))
        if self.unknown_codes:
            out.append("unknown stock codes: " + ", ".join(self.unknown_codes))
        if not self.mentions_top_product:
            out.append("none of the most urgent products is mentioned")
        if self.words > MAX_WORDS:
            out.append(f"too long ({self.words} words, limit {MAX_WORDS})")
        return out

    def summary(self) -> str:
        if self.ok:
            return f"All {self.numbers_checked} numbers in this brief match the order list."
        return "; ".join(self.problems)


@dataclass
class BriefResult:
    text: str
    source: str  # "ai", "template"
    model: str = ""
    report: GroundingReport | None = None
    attempts: int = 0
    fallback_reason: str = ""


def brief_context(recs: pd.DataFrame, as_of: str, top: int = 8) -> dict:
    """What the model sees: the top rows of the order list and the totals."""
    rows = recs.head(top)[[c for c in TABLE_COLUMNS if c in recs.columns]].copy()
    records = json.loads(rows.to_json(orient="records"))
    return {
        "as_of": as_of,
        "rows": records,
        "totals": {
            "order_now": int((recs["status"] == "Order now").sum()),
            "order_this_week": int((recs["status"] == "Order this week").sum()),
            "products_reviewed": int(len(recs)),
            "order_value_gbp": round(float(recs["order_value"].sum()), 2),
            "products_shown": int(len(rows)),
        },
    }


def context_messages(context: dict) -> list[dict]:
    user = (
        f"The forecast starts the day after {context['as_of']}.\n\n"
        f"Order list (most urgent first):\n{json.dumps(context['rows'], indent=1)}\n\n"
        f"Totals for all {context['totals']['products_reviewed']} products reviewed: "
        f"{json.dumps(context['totals'])}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _as_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def allowed_numbers(context: dict) -> set[float]:
    """Every number in the data the model saw, plus rounded forms of each."""
    found: set[float] = set()
    text = json.dumps(context)
    for token in NUMBER.findall(text):
        value = _as_float(token)
        if value is None:
            continue
        found.update({value, float(round(value)), round(value, 1), round(value, 2)})
    totals = context.get("totals", {})
    for value in totals.values():
        found.update({float(value), float(round(value))})
    found.update({float(n) for n in range(0, max(1, totals.get("products_shown", 0)) + 1)})  # "the top 3"
    as_of = str(context.get("as_of", ""))
    for part in re.findall(r"\d+", as_of):
        found.add(float(part))
    return found


def check_grounding(text: str, context: dict, require_product_mention: bool = True) -> GroundingReport:
    """``require_product_mention`` is for the brief; the assistant answers one question at a time."""
    report = GroundingReport(words=len(text.split()))
    allowed = allowed_numbers(context)
    codes = {str(r.get("stock_code", "")) for r in context.get("rows", [])}

    code_like = []
    for token in set(STOCK_CODE.findall(text)):
        if token in codes:
            code_like.append(token)
        elif token.isdigit() and float(token) in allowed:
            continue  # a five-digit quantity from the table, e.g. "order 16800", not a code
        else:
            code_like.append(token)
            report.unknown_codes.append(token)
    report.unknown_codes.sort()

    stripped = text
    for token in code_like:  # codes are not numbers
        stripped = re.sub(rf"\b{re.escape(token)}\b", " ", stripped)
    for token in NUMBER.findall(stripped):
        value = _as_float(token)
        if value is None:
            continue
        report.numbers_checked += 1
        if value in allowed or float(round(value)) in allowed:
            continue
        if any(abs(value - a) <= 0.5 and a >= 100 for a in allowed):  # "about 1,235 units" for 1234.5
            continue
        report.unverified_numbers.append(token)

    urgent = [r for r in context.get("rows", []) if r.get("status") != "OK"][:3]
    if urgent and require_product_mention:
        lowered = text.lower()
        report.mentions_top_product = any(
            str(r.get("description") or r.get("stock_code")).strip().lower() in lowered
            or str(r.get("stock_code")) in text
            for r in urgent
        )
    return report


def ai_brief(client, recs: pd.DataFrame, as_of: str, top: int = 8, context: dict | None = None) -> BriefResult:
    """Write the brief with the model, check it, fix it once if needed, else fall back.

    ``context`` lets the evals replay a saved order list exactly; normally it is built here.
    """
    fallback = template_brief(recs, as_of)
    if client is None:
        return BriefResult(text=fallback, source="template", fallback_reason="no model configured")
    if recs.empty:
        return BriefResult(text=fallback, source="template", fallback_reason="nothing to brief")

    context = context or brief_context(recs, as_of, top)
    messages = context_messages(context)
    model = getattr(client, "model", "")
    report = GroundingReport()
    text = ""
    for attempt in (1, 2):
        try:
            response = client.complete(messages, max_tokens=400)
        except LLMError as exc:
            return BriefResult(text=fallback, source="template", model=model, attempts=attempt, fallback_reason=str(exc))
        model = response.model or model
        text = response.text.strip()
        report = check_grounding(text, context)
        if report.ok:
            return BriefResult(text=text, source="ai", model=model, report=report, attempts=attempt)
        messages = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "Your brief has problems: " + "; ".join(report.problems) + ". Rewrite it using only "
                "numbers and products from the data, under 120 words.",
            },
        ]
    return BriefResult(
        text=fallback, source="template", model=model, report=report, attempts=2,
        fallback_reason="the model's brief did not match the order list: " + "; ".join(report.problems),
    )
