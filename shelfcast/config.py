"""Paths, model settings and a tiny .env reader (so no extra dependency is needed)."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"

# Where data, models and reports go. Tests point this somewhere temporary.
HOME = Path(os.environ.get("SHELFCAST_HOME", str(ROOT)))
DATA_DIR = HOME / "data"
MODELS_DIR = HOME / "models"
REPORTS_DIR = HOME / "reports"
# A small copy of the tables the app needs, saved with `python -m shelfcast snapshot`, so the
# app can be demoed (e.g. on Streamlit Community Cloud) without a warehouse.
SNAPSHOT_DIR = Path(os.environ.get("SHELFCAST_SNAPSHOT_DIR", str(ROOT / "demo")))


def snapshot_dir() -> Path:
    """Like SNAPSHOT_DIR, but reads the environment at call time (handy for the app and tests)."""
    return Path(os.environ.get("SHELFCAST_SNAPSHOT_DIR", str(ROOT / "demo")))

REAL_DATA = DATA_DIR / "online_retail_ii.csv.gz"
SYNTHETIC_DATA = DATA_DIR / "synthetic_transactions.csv.gz"
UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

LOOKBACK = 56  # days of history the model sees
HORIZON = 14  # days it forecasts
QUANTILES = (0.1, 0.5, 0.9)  # low / median / high


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (existing values win)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"'):
            value = value[1:].split(value[0], 1)[0]  # quoted: keep everything inside the quotes
        else:
            value = re.split(r"\s#", value, maxsplit=1)[0].strip()  # drop "  # inline comments"
            if value.startswith("#"):
                value = ""
        os.environ.setdefault(key.strip(), value)
