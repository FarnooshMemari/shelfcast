"""Global quantile forecaster (PyTorch).

One network is trained on all products. Input: the last LOOKBACK days of sales, scaled by
their mean, plus the weekdays being forecast. Two heads: P10/P50/P90 of daily sales, and
P10/P50/P90 of the running total from day 1 to day h. Orders depend on total demand until
the next delivery, and the spread of a total is smaller than the sum of daily spreads, so
the totals get their own head. The output layer keeps quantiles ordered and non-negative,
and running totals non-decreasing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from .config import HORIZON, LOOKBACK, QUANTILES


def feature_size(lookback: int = LOOKBACK, horizon: int = HORIZON) -> int:
    return lookback + 2 + horizon * 7


def build_windows(
    values: np.ndarray, weekdays: np.ndarray, starts: np.ndarray, lookback: int = LOOKBACK, horizon: int = HORIZON
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut windows out of a (days x products) matrix.

    ``starts`` are the first day of each history window and ``weekdays`` must cover
    ``len(values) + horizon`` days. Returns features (n, feature_size), scaled daily
    targets (n, horizon) and scales (n,), ordered window by window, then product by product.
    Targets beyond the end of ``values`` are NaN (that is the real future we forecast).
    """
    total = lookback + horizon
    padded = np.vstack([values, np.full((horizon, values.shape[1]), np.nan)])
    windows = sliding_window_view(padded, total, axis=0)[starts]  # (len(starts), products, total)
    history = windows[..., :lookback]
    future = windows[..., lookback:]
    scale = np.nanmean(history, axis=-1) + 1.0
    hist_scaled = history / scale[..., None]
    mean7 = hist_scaled[..., -7:].mean(-1, keepdims=True)
    mean28 = hist_scaled[..., -28:].mean(-1, keepdims=True)

    day_idx = starts[:, None] + lookback + np.arange(horizon)[None, :]  # (len(starts), horizon)
    onehot = np.eye(7, dtype=np.float32)[weekdays[day_idx]].reshape(len(starts), 1, horizon * 7)
    onehot = np.broadcast_to(onehot, (len(starts), values.shape[1], horizon * 7))

    x = np.concatenate([hist_scaled, mean7, mean28, onehot], axis=-1).reshape(-1, feature_size(lookback, horizon))
    y = (future / scale[..., None]).reshape(-1, horizon)
    return x.astype(np.float32), y.astype(np.float32), scale.reshape(-1).astype(np.float32)


class QuantileForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int = HORIZON, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * 6),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (daily, running_total), each (batch, horizon, 3) = P10, P50, P90."""
        raw = self.net(x).view(-1, self.horizon, 6)
        median = F.softplus(raw[..., 1])
        low = torch.clamp(median - F.softplus(raw[..., 0]), min=0.0)
        high = median + F.softplus(raw[..., 2])
        daily = torch.stack([low, median, high], dim=-1)

        steps = torch.cumsum(F.softplus(raw[..., 3:]), dim=1)  # three non-decreasing curves
        total_low = steps[..., 0]
        total_mid = total_low + steps[..., 1]
        total_high = total_mid + steps[..., 2]
        running = torch.stack([total_low, total_mid, total_high], dim=-1)
        return daily, running


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles=QUANTILES) -> torch.Tensor:
    """Average quantile (pinball) loss over P10/P50/P90; ignores NaN targets."""
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    diff = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * diff, (q - 1) * diff)
    mask = ~torch.isnan(loss)
    return loss[mask].mean()


def forecast_loss(daily: torch.Tensor, running: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Daily loss + running-total loss. Totals are divided by h so long horizons don't dominate."""
    h = torch.arange(1, target.shape[1] + 1, dtype=target.dtype, device=target.device)
    totals = torch.cumsum(target, dim=1)
    return pinball_loss(daily, target) + pinball_loss(running / h[None, :, None], totals / h[None, :])
