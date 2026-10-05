"""Train, backtest against simple baselines, then forecast the next two weeks.

Backtest: the last ``HORIZON`` days are the test period and the ``HORIZON`` days before
them are for early stopping. The model never sees either during training. After the
backtest, a final model is trained on all data and forecasts the days after the data ends.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from .config import HORIZON, LOOKBACK
from .model import QuantileForecaster, build_windows, feature_size, forecast_loss


@dataclass
class TrainSettings:
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 256
    patience: int = 6
    clip_quantile: float = 0.995  # tame one-off bulk orders in the model's inputs
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _weekdays(dates: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    extended = pd.date_range(dates[0], periods=len(dates) + horizon, freq="D")
    return extended.dayofweek.to_numpy()


def _clip(values: np.ndarray, q: float, upto: int | None = None) -> np.ndarray:
    """Cap each product's daily sales at its ``q`` quantile (caps learned from rows before ``upto``)."""
    reference = values if upto is None else values[:upto]
    caps = np.quantile(reference, q, axis=0)
    return np.minimum(values, np.maximum(caps, 1.0))


def _fit(x_tr, y_tr, x_va, y_va, settings: TrainSettings, epochs: int | None = None):
    torch.manual_seed(settings.seed)
    model = QuantileForecaster(x_tr.shape[1], y_tr.shape[1], hidden=settings.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    xv = torch.from_numpy(x_va) if x_va is not None else None
    yv = torch.from_numpy(y_va) if y_va is not None else None
    n_epochs = epochs or settings.epochs
    best, best_epoch, best_state, history, waited = float("inf"), n_epochs, None, [], 0
    g = torch.Generator().manual_seed(settings.seed)
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(xt), generator=g)
        total, batches = 0.0, 0
        for i in range(0, len(order), settings.batch_size):
            idx = order[i : i + settings.batch_size]
            loss = forecast_loss(*model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, batches = total + loss.item(), batches + 1
        row = {"epoch": epoch, "train_loss": round(total / max(batches, 1), 5)}
        if xv is not None:
            model.eval()
            with torch.no_grad():
                row["val_loss"] = round(forecast_loss(*model(xv), yv).item(), 5)
            if row["val_loss"] < best - 1e-5:
                best, best_epoch, waited = row["val_loss"], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
        history.append(row)
        if xv is not None and waited >= settings.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def _predict(model, x: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily and running-total quantiles in units, each (n, horizon, 3)."""
    with torch.no_grad():
        daily, running = model(torch.from_numpy(x))
    s = scale.astype(np.float64)[:, None, None]
    return daily.numpy().astype(np.float64) * s, running.numpy().astype(np.float64) * s


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error / total actual sales."""
    total = float(np.abs(actual).sum())
    return float(np.abs(actual - forecast).sum() / total) if total else float("nan")


@dataclass
class ForecastResult:
    model: QuantileForecaster
    metrics: dict
    backtest: pd.DataFrame
    future: pd.DataFrame
    history: list


def backtest_and_forecast(
    matrix: pd.DataFrame,
    settings: TrainSettings | None = None,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> ForecastResult:
    """``matrix``: rows = consecutive days, columns = products, values = units sold."""
    settings = settings or TrainSettings()
    t0 = time.time()
    dates = pd.DatetimeIndex(matrix.index)
    products = [str(c) for c in matrix.columns]
    raw = matrix.to_numpy(dtype=float)
    weekdays = _weekdays(dates, horizon)
    n_days = len(dates)
    if n_days < lookback + 4 * horizon:
        raise ValueError(f"Need at least {lookback + 4 * horizon} days of history, got {n_days}")

    # backtest: train on the past, stop early on the next two weeks, test on the last two
    test_origin = n_days - horizon
    val_origin = test_origin - horizon
    values = _clip(raw, settings.clip_quantile, upto=test_origin)
    train_starts = np.arange(0, val_origin - lookback - horizon + 1)
    x_tr, y_tr, _ = build_windows(values, weekdays, train_starts, lookback, horizon)
    x_va, y_va, _ = build_windows(values, weekdays, np.array([val_origin - lookback]), lookback, horizon)
    x_te, _, s_te = build_windows(values, weekdays, np.array([test_origin - lookback]), lookback, horizon)

    model, history, best_epoch = _fit(x_tr, y_tr, x_va, y_va, settings)
    daily_q, total_q = _predict(model, x_te, s_te)  # (products, horizon, 3) each

    actual = raw[test_origin:].T  # (products, horizon), real (unclipped) sales
    last_week = raw[test_origin - 7 : test_origin].T
    naive = np.tile(last_week, (1, int(np.ceil(horizon / 7))))[:, :horizon]
    ma28 = np.repeat(raw[test_origin - 28 : test_origin].mean(0)[:, None], horizon, axis=1)
    p10, p50, p90 = daily_q[..., 0], daily_q[..., 1], daily_q[..., 2]
    totals = actual.sum(1)
    t10, t50, t90 = total_q[:, -1, 0], total_q[:, -1, 1], total_q[:, -1, 2]  # whole test period
    summed10, summed90 = p10.sum(1), p90.sum(1)

    metrics = {
        "as_of": str(dates[-1].date()),
        "test_period": f"{dates[test_origin].date()} to {dates[-1].date()}",
        "products": len(products),
        "history_days": n_days,
        "training_windows": int(len(x_tr)),
        "best_epoch": int(best_epoch),
        # day by day
        "wape_shelfcast": round(wape(actual, p50), 4),
        "wape_same_weekday_last_week": round(wape(actual, naive), 4),
        "wape_28_day_average": round(wape(actual, ma28), 4),
        "coverage_p10_p90": round(float(((actual >= p10) & (actual <= p90)).mean()), 4),
        "mae_shelfcast": round(float(np.abs(actual - p50).mean()), 3),
        # two-week totals per product (what a restock order has to cover)
        "wape_total_shelfcast": round(wape(totals, t50), 4),
        "wape_total_same_weekday_last_week": round(wape(totals, naive.sum(1)), 4),
        "wape_total_28_day_average": round(wape(totals, ma28.sum(1)), 4),
        "coverage_total_p10_p90": round(float(((totals >= t10) & (totals <= t90)).mean()), 4),
        "coverage_total_summed_daily": round(float(((totals >= summed10) & (totals <= summed90)).mean()), 4),
        "total_band_narrower_than_summed_daily": round(
            1 - float((t90 - t10).sum() / max((summed90 - summed10).sum(), 1e-9)), 4
        ),
    }
    best_baseline = min(metrics["wape_same_weekday_last_week"], metrics["wape_28_day_average"])
    metrics["improvement_vs_best_baseline"] = round(1 - metrics["wape_shelfcast"] / best_baseline, 4)
    best_total = min(metrics["wape_total_same_weekday_last_week"], metrics["wape_total_28_day_average"])
    metrics["total_improvement_vs_best_baseline"] = round(1 - metrics["wape_total_shelfcast"] / best_total, 4)

    test_dates = dates[test_origin:]
    backtest = pd.DataFrame({
        "stock_code": np.repeat(products, horizon),
        "sale_date": np.tile(test_dates, len(products)),
        "actual": actual.reshape(-1),
        "p10": p10.reshape(-1),
        "p50": p50.reshape(-1),
        "p90": p90.reshape(-1),
        "naive": naive.reshape(-1).astype(float),
        "actual_total": actual.cumsum(1).reshape(-1),
        "total_p10": total_q[..., 0].reshape(-1),
        "total_p50": total_q[..., 1].reshape(-1),
        "total_p90": total_q[..., 2].reshape(-1),
    })

    # final model on everything, trained for the number of epochs the backtest picked
    values_all = _clip(raw, settings.clip_quantile)
    all_starts = np.arange(0, n_days - lookback - horizon + 1)
    x_all, y_all, _ = build_windows(values_all, weekdays, all_starts, lookback, horizon)
    final, _, _ = _fit(x_all, y_all, None, None, settings, epochs=max(best_epoch, 3))
    x_fut, _, s_fut = build_windows(values_all, weekdays, np.array([n_days - lookback]), lookback, horizon)
    daily_f, total_f = _predict(final, x_fut, s_fut)
    future_dates = pd.date_range(dates[-1] + pd.offsets.Day(1), periods=horizon, freq="D")
    # p10/p50/p90 = sales on day h; total_p10/50/90 = running total from day 1 up to day h
    future = pd.DataFrame({
        "stock_code": np.repeat(products, horizon),
        "h": np.tile(np.arange(1, horizon + 1), len(products)),
        "forecast_date": np.tile(future_dates, len(products)),
        "p10": daily_f[..., 0].reshape(-1),
        "p50": daily_f[..., 1].reshape(-1),
        "p90": daily_f[..., 2].reshape(-1),
        "total_p10": total_f[..., 0].reshape(-1),
        "total_p50": total_f[..., 1].reshape(-1),
        "total_p90": total_f[..., 2].reshape(-1),
    })
    metrics["train_seconds"] = round(time.time() - t0, 1)
    return ForecastResult(final, metrics, backtest, future, history)


def model_summary(model: torch.nn.Module) -> dict:
    return {
        "type": "QuantileForecaster (MLP, PyTorch)",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "inputs": f"{LOOKBACK} days of scaled sales + weekdays of the {HORIZON} forecast days",
        "outputs": f"P10 / P50 / P90 of daily sales and of the running total for each of the next {HORIZON} days",
        "features": feature_size(),
    }
