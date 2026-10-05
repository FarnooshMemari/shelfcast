"""Getting transactions (real or synthetic) and shaping daily sales for the model.

Real data: Online Retail II from the UCI Machine Learning Repository (CC BY 4.0), about a
million transactions from a UK online gift retailer, December 2009 to December 2011.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REAL_DATA, SYNTHETIC_DATA, UCI_URL

COLUMN_MAP = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "price",
    "unitprice": "price",
    "customerid": "customer_id",
    "country": "country",
}
COLUMNS = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"]


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the source columns to snake_case and fix their types."""
    renamed = {c: COLUMN_MAP.get(re.sub(r"[^a-z]", "", str(c).lower()), c) for c in df.columns}
    df = df.rename(columns=renamed)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; found {list(df.columns)}")
    df = df[COLUMNS].copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype("int64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).astype(float)
    df["customer_id"] = df["customer_id"].astype("string").str.replace(r"\.0$", "", regex=True).fillna("")
    df["country"] = df["country"].fillna("").astype(str)
    return df


def download_online_retail(dest: Path = REAL_DATA) -> Path:
    """Download the UCI zip, read both Excel sheets, save one compressed CSV."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {UCI_URL} ...")
    with urllib.request.urlopen(UCI_URL, timeout=120) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls")))
        print(f"reading {name} (takes a few minutes) ...")
        sheets = pd.read_excel(io.BytesIO(zf.read(name)), sheet_name=None)
    df = standardize(pd.concat(sheets.values(), ignore_index=True))
    df.to_csv(dest, index=False, compression="gzip")
    print(f"saved {len(df):,} rows to {dest}")
    return dest


def convert_local_excel(path: Path, dest: Path = REAL_DATA) -> Path:
    """Use this if you downloaded online_retail_II.xlsx by hand."""
    sheets = pd.read_excel(path, sheet_name=None)
    df = standardize(pd.concat(sheets.values(), ignore_index=True))
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False, compression="gzip")
    return dest


def read_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"invoice": str, "stock_code": str, "customer_id": str})
    return standardize(df)


def make_synthetic_transactions(
    n_products: int = 40, start: str = "2009-12-01", end: str = "2011-12-09", seed: int = 0
) -> pd.DataFrame:
    """Realistic-looking fake transactions in the Online Retail II schema.

    Demand has weekday and holiday-season patterns, promotions, noise, a closed
    Christmas week, plus a few cancellations and postage rows for the cleaning SQL.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    weekday = np.array([1.1, 1.15, 1.1, 1.25, 0.9, 0.12, 0.65])  # Mon..Sun
    closed = (dates.month == 12) & (dates.day >= 23) | (dates.month == 1) & (dates.day <= 3)
    words = ["HEART", "LANTERN", "MUG", "BAG", "CANDLE", "CLOCK", "TIN", "SIGN", "BUNTING", "CUSHION"]
    colours = ["RED", "WHITE", "PINK", "BLUE", "VINTAGE", "GREEN", "IVORY", "SPOTTY"]

    rows = []  # (stock_code, description, quantity, day index, price)
    for p in range(n_products):
        code = f"{84000 + p * 7}{'ABC'[p % 3]}"
        name = f"{colours[p % len(colours)]} {words[(p * 3) % len(words)]} {['SMALL', 'LARGE', 'SET OF 3'][p % 3]}"
        price = round(float(rng.lognormal(1.0, 0.5)), 2)
        base = float(rng.lognormal(2.3, 0.6))
        season = 1 + rng.uniform(0.3, 1.2) * np.exp(-(((doy - 325) / 35.0) ** 2))
        trend = 1 + rng.normal(0, 0.25) * np.linspace(0, 1, len(dates))
        lam = base * weekday[dow] * season * np.clip(trend, 0.3, None)
        promo = rng.random(len(dates)) < 0.02
        lam = np.where(promo, lam * 2.5, lam)
        demand = rng.poisson(lam * rng.gamma(4.0, 0.25, len(dates)))
        demand[closed] = 0
        for day, qty in enumerate(demand):
            remaining = int(qty)
            while remaining > 0:  # split each day's demand into order lines of 1-24 units
                q = int(min(remaining, rng.integers(1, 25)))
                rows.append((code, name, q, day, price))
                remaining -= q
    df = pd.DataFrame(rows, columns=["stock_code", "description", "quantity", "day", "price"])
    n = len(df)
    minutes = rng.integers(8 * 60, 18 * 60, n)
    df["invoice_date"] = dates[df["day"].to_numpy()] + pd.to_timedelta(minutes, unit="m")
    df["invoice"] = (489001 + np.arange(n)).astype(str)
    df["customer_id"] = rng.integers(12000, 18000, n).astype(str)
    df["country"] = "United Kingdom"
    df = df[COLUMNS]

    # noise the cleaning step must remove
    sample = df.sample(frac=0.01, random_state=seed)
    cancels = sample.assign(invoice="C" + sample["invoice"], quantity=-sample["quantity"])
    postage = df.sample(n=min(200, len(df)), random_state=seed + 1).assign(
        stock_code="POST", description="POSTAGE", quantity=1, price=18.0
    )
    return standardize(pd.concat([df, cancels, postage], ignore_index=True))


def get_transactions(synthetic: bool = False) -> pd.DataFrame:
    if synthetic:
        if not SYNTHETIC_DATA.exists():
            SYNTHETIC_DATA.parent.mkdir(parents=True, exist_ok=True)
            make_synthetic_transactions().to_csv(SYNTHETIC_DATA, index=False, compression="gzip")
        return read_transactions(SYNTHETIC_DATA)
    if not REAL_DATA.exists():
        download_online_retail(REAL_DATA)
    return read_transactions(REAL_DATA)


def select_products(summary: pd.DataFrame, n: int = 200, min_days: int = 120) -> list[str]:
    """The best sellers that sold on enough days to be worth forecasting."""
    s = summary[summary["selling_days"] >= min_days].assign(stock_code=lambda d: d["stock_code"].astype(str))
    # ties broken by code, so the same products come out in the same order on every run
    s = s.sort_values(["total_units", "stock_code"], ascending=[False, True], kind="mergesort")
    return s["stock_code"].head(n).tolist()


def daily_matrix(daily: pd.DataFrame, products: list[str], start=None, end=None) -> pd.DataFrame:
    """Rows = every calendar day from ``start`` to ``end``, columns = products,
    values = units sold (0 on days without sales)."""
    d = daily.copy()
    d["sale_date"] = pd.to_datetime(d["sale_date"])
    d["stock_code"] = d["stock_code"].astype(str)
    d["units"] = pd.to_numeric(d["units"], errors="coerce").astype(float)
    pivot = d.pivot_table(index="sale_date", columns="stock_code", values="units", aggfunc="sum")
    start = pd.Timestamp(start) if start is not None else pivot.index.min()
    end = pd.Timestamp(end) if end is not None else pivot.index.max()
    days = pd.date_range(start, end, freq="D")
    return pivot.reindex(days).reindex(columns=products).fillna(0.0).astype(float)
