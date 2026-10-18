"""Evaluate the AI brief: does a model stay inside the numbers it was given?

``make_cases`` builds a fixed set of order lists (different planning settings, plus a few
hand-made edge cases) and saves them to ``evals/brief_cases.json``. ``run`` asks each model
to write a brief for every case, scores every answer with the same grounding check the app
uses, and writes ``reports/brief_evals.md``. The built-in template is scored too, as a
baseline that should always pass; if it ever fails, the check itself is broken.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .brief_ai import BriefResult, ai_brief, brief_context, check_grounding
from .config import REPORTS_DIR, ROOT
from .restock import RestockSettings, brief as template_brief, recommend

CASES_PATH = ROOT / "evals" / "brief_cases.json"
SETTINGS_GRID = [(3, 0.0), (7, 0.0), (7, 0.5), (7, 1.0), (14, 1.0), (1, 1.0)]
TOP_GRID = [3, 8]


def _recs_from_context(context: dict) -> pd.DataFrame:
    """The order list a case holds (only the rows the model sees, which is all we need to score)."""
    rows = pd.DataFrame(context["rows"])
    if rows.empty:
        rows = pd.DataFrame(columns=["status", "stock_code", "description", "order_qty", "order_value", "why"])
    return rows


def edge_cases(as_of: str) -> list[dict]:
    """Small hand-made order lists that trip up careless briefs."""
    def ctx(name, recs, top=8):
        return {"id": name, "context": brief_context(recs, as_of, top)}

    base = {"on_hand": 10, "days_left": 2, "lead_time_days": 5, "demand_low": 40.0, "demand_mid": 60.0, "demand_high": 90.0}
    why = "Next 12 days: expect 60 units (80% range 40-90). 10 on hand lasts about 2 days; a new order takes 5 days."
    one = pd.DataFrame([{"status": "Order now", "stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER",
                         **base, "order_qty": 80, "order_value": 204.8, "why": why}])
    nothing = pd.DataFrame([{"status": "OK", "stock_code": "22423", "description": "REGENCY CAKESTAND 3 TIER", "on_hand": 900,
                             "days_left": 14, "lead_time_days": 3, "demand_low": 100.0, "demand_mid": 150.0, "demand_high": 200.0,
                             "order_qty": 0, "order_value": 0.0, "why": "Next 10 days: expect 150 units (80% range 100-200). 900 on hand lasts 14+ days; a new order takes 3 days."}])
    big = pd.DataFrame([
        {"status": "Order now", "stock_code": "23084", "description": "RABBIT NIGHT LIGHT", "on_hand": 1200, "days_left": 1,
         "lead_time_days": 10, "demand_low": 9000.0, "demand_mid": 12500.5, "demand_high": 18000.0, "order_qty": 16800,
         "order_value": 25872.0, "why": "Next 14 days: expect 12,500 units (80% range 9,000-18,000). 1,200 on hand lasts about 1 days; a new order takes 10 days."},
        {"status": "Order this week", "stock_code": "84879", "description": "ASSORTED COLOUR BIRD ORNAMENT", "on_hand": 300,
         "days_left": 9, "lead_time_days": 3, "demand_low": 200.0, "demand_mid": 310.0, "demand_high": 450.0, "order_qty": 150,
         "order_value": 199.5, "why": "Next 10 days: expect 310 units (80% range 200-450). 300 on hand lasts about 9 days; a new order takes 3 days."},
    ])
    return [ctx("edge_one_product", one), ctx("edge_nothing_to_order", nothing), ctx("edge_large_numbers", big)]


def make_cases(inputs: dict, out_path: Path = CASES_PATH) -> list[dict]:
    """Order lists from the warehouse under several planning settings, plus the edge cases."""
    cases = []
    for review_days, buffer in SETTINGS_GRID:
        recs = recommend(inputs["forecasts"], inputs["inventory"], inputs["products"], RestockSettings(review_days, buffer))
        for top in TOP_GRID:
            cases.append({
                "id": f"review{review_days}_buffer{int(buffer * 100)}_top{top}",
                "settings": {"review_days": review_days, "buffer": buffer, "top": top},
                "context": brief_context(recs, inputs["as_of"], top),
            })
    cases.extend(edge_cases(inputs["as_of"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, indent=1), encoding="utf-8")
    return cases


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class CaseScore:
    case_id: str
    model: str
    source: str
    grounded_first_try: bool
    grounded: bool
    attempts: int
    words: int
    seconds: float
    problems: list[str] = field(default_factory=list)
    text: str = ""


@dataclass
class ModelScore:
    model: str
    cases: int
    grounded_first_try: float
    grounded: float
    fallback: float
    mean_words: float
    mean_seconds: float
    failures: list[CaseScore]


class _TrackingClient:
    """Wraps a client to record how many calls a brief took and how long they lasted."""

    def __init__(self, client):
        self.client = client
        self.model = getattr(client, "model", "")
        self.first_report = None
        self.calls = 0
        self.seconds = 0.0

    def complete(self, messages, tools=None, max_tokens=1500):
        self.calls += 1
        started = time.perf_counter()
        resp = self.client.complete(messages, tools=tools, max_tokens=max_tokens)
        self.seconds += resp.seconds or (time.perf_counter() - started)
        return resp


def score_case(case: dict, client, model_name: str) -> CaseScore:
    context = case["context"]
    recs = _recs_from_context(context)
    as_of = context["as_of"]
    if client == "template":
        text = template_brief(recs, as_of)
        report = check_grounding(text, brief_context(recs, as_of, top=max(len(recs), 1)))  # the rows it was built from
        return CaseScore(case["id"], model_name, "template", report.ok, report.ok, 1, report.words, 0.0, report.problems, text)
    tracker = _TrackingClient(client)
    result: BriefResult = ai_brief(tracker, recs, as_of, context=context)
    first_ok = result.source == "ai" and result.attempts == 1
    report = result.report
    return CaseScore(
        case_id=case["id"], model=model_name, source=result.source, grounded_first_try=first_ok,
        grounded=result.source == "ai", attempts=max(result.attempts, 1),
        words=report.words if report else len(result.text.split()), seconds=round(tracker.seconds, 2),
        problems=(report.problems if report and not first_ok else []) + ([result.fallback_reason] if result.source == "template" and not report else []),
        text=result.text,
    )


def run(cases: list[dict], clients: dict, log=print) -> list[ModelScore]:
    """``clients`` maps a display name to a client (or the string "template")."""
    scores = []
    for name, client in clients.items():
        rows = []
        for case in cases:
            rows.append(score_case(case, client, name))
            log(f"[evals] {name:<24} {case['id']:<28} {'ok' if rows[-1].grounded_first_try else ('ok after retry' if rows[-1].grounded else 'FALLBACK')}")
        n = len(rows)
        scores.append(ModelScore(
            model=name, cases=n,
            grounded_first_try=sum(r.grounded_first_try for r in rows) / n,
            grounded=sum(r.grounded for r in rows) / n,
            fallback=sum(r.source == "template" for r in rows) / n if client != "template" else 0.0,
            mean_words=sum(r.words for r in rows) / n,
            mean_seconds=sum(r.seconds for r in rows) / n,
            failures=[r for r in rows if not r.grounded_first_try],
        ))
    return scores


def write_report(scores: list[ModelScore], reports_dir: Path = REPORTS_DIR, cases_path: Path = CASES_PATH) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AI brief evaluation",
        "",
        f"Each model wrote a restock brief for the {scores[0].cases} order lists in `{cases_path.name}` "
        "(different planning settings on the real run, plus hand-made edge cases). Every brief was checked "
        "against the order list it was given: all numbers must come from the list, no unknown stock codes, "
        "at least one of the most urgent products named, under 200 words. A brief that fails gets one retry; "
        "if it fails again the app shows the built-in summary (a *fallback*).",
        "",
        "| Model | Grounded on the first try | Grounded after one retry | Fell back to the template | Avg words | Avg seconds per brief |",
        "|---|---|---|---|---|---|",
    ]
    for s in scores:
        lines.append(
            f"| {s.model} | {s.grounded_first_try:.0%} | {s.grounded:.0%} | {s.fallback:.0%} | {s.mean_words:.0f} | {s.mean_seconds:.1f} |"
        )
    lines.append("")
    lines.append("The template row is the baseline: it is built from the numbers, so it should always score 100%.")
    failures = [(s.model, f) for s in scores if s.model != "template" for f in s.failures]
    if failures:
        lines += ["", "## Briefs that failed the check on the first try", ""]
        for model, f in failures:
            lines.append(f"- **{model}**, `{f.case_id}`: {'; '.join(f.problems) or 'fixed on retry'}")
    lines.append("")
    (reports_dir / "brief_evals.md").write_text("\n".join(lines), encoding="utf-8")
    (reports_dir / "brief_evals.json").write_text(
        json.dumps([{**asdict(s), "failures": [asdict(f) for f in s.failures]} for s in scores], indent=1), encoding="utf-8"
    )
    return reports_dir / "brief_evals.md"
