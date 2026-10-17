"""Thumbs up / down on AI answers, kept as one JSON line per rating in data/feedback.jsonl.

A file is enough for a single-shop app, and it keeps working when the warehouse is a
read-only snapshot. ``summary`` turns the log into the numbers that decide whether the AI
brief and the assistant are worth keeping: how often people rate them up, by feature and
by model.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR


def feedback_path() -> Path:
    return Path(os.environ.get("SHELFCAST_FEEDBACK", str(DATA_DIR / "feedback.jsonl")))


def record(feature: str, rating: str, question: str = "", answer: str = "", model: str = "",
           verified: bool | None = None, path: Path | None = None) -> dict:
    """``feature`` is "brief" or "assistant"; ``rating`` is "up" or "down"."""
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    entry = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature": feature,
        "rating": rating,
        "model": model,
        "verified": verified,
        "question": question[:500],
        "answer": answer[:2000],
    }
    path = Path(path or feedback_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load(path: Path | None = None) -> list[dict]:
    path = Path(path or feedback_path())
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(entries: list[dict]) -> list[dict]:
    """One row per feature and model: ratings, share rated up, and how many answers were verified."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        groups.setdefault((e.get("feature", ""), e.get("model", "")), []).append(e)
    rows = []
    for (feature, model), items in sorted(groups.items()):
        ups = sum(1 for e in items if e.get("rating") == "up")
        verified = [e for e in items if e.get("verified") is not None]
        rows.append({
            "feature": feature,
            "model": model,
            "ratings": len(items),
            "rated_up": round(ups / len(items), 2),
            "verified_share": round(sum(1 for e in verified if e["verified"]) / len(verified), 2) if verified else None,
        })
    return rows
