import numpy as np
import pandas as pd
import torch

from shelfcast.forecast import TrainSettings, backtest_and_forecast, wape
from shelfcast.model import QuantileForecaster, build_windows, feature_size, forecast_loss, pinball_loss

L, H = 56, 14


def _ramp(days=200, products=3):
    values = np.arange(days, dtype=float)[:, None] * np.ones((1, products))
    weekdays = np.arange(days + H) % 7
    return values, weekdays


def test_windows_never_leak_the_future():
    values, weekdays = _ramp()
    starts = np.array([0, 10, 100])
    x, y, scale = build_windows(values, weekdays, starts, L, H)
    assert x.shape == (len(starts) * 3, feature_size(L, H))
    assert y.shape == (len(starts) * 3, H)
    for i, s in enumerate(np.repeat(starts, 3)):
        history = x[i, :L] * scale[i]
        target = y[i] * scale[i]
        np.testing.assert_allclose(history, np.arange(s, s + L), rtol=1e-5)
        np.testing.assert_allclose(target, np.arange(s + L, s + L + H), rtol=1e-5)
        assert history.max() < target.min()  # the model only ever sees days before the ones it predicts


def test_weekday_features_match_the_forecast_days():
    values, weekdays = _ramp()
    x, _, _ = build_windows(values, weekdays, np.array([5]), L, H)
    onehot = x[0, L + 2 :].reshape(H, 7)
    assert (onehot.sum(1) == 1).all()
    np.testing.assert_array_equal(onehot.argmax(1), (5 + L + np.arange(H)) % 7)


def test_future_window_targets_are_missing():
    values, weekdays = _ramp(days=100)
    _, y, _ = build_windows(values, weekdays, np.array([100 - L]), L, H)
    assert np.isnan(y).all()


def test_quantiles_are_ordered_non_negative_and_totals_grow():
    torch.manual_seed(0)
    model = QuantileForecaster(feature_size(L, H), H, hidden=32)
    daily, running = model(torch.randn(64, feature_size(L, H)) * 3)
    for q in (daily, running):
        assert q.shape == (64, H, 3)
        assert (q >= 0).all()
        assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()
    assert (running[:, 1:, :] >= running[:, :-1, :]).all()  # running totals never go down


def test_pinball_loss_values_and_nan_handling():
    pred = torch.zeros(1, 2, 3)
    target = torch.tensor([[1.0, float("nan")]])
    # target above every quantile: loss = q * 1 averaged over q = (0.1 + 0.5 + 0.9) / 3
    assert abs(pinball_loss(pred, target).item() - 0.5) < 1e-6
    daily, running = torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)
    assert torch.isfinite(forecast_loss(daily, running, target))


def test_backtest_beats_naive_on_a_clean_weekly_pattern():
    rng = np.random.default_rng(0)
    days = pd.date_range("2011-01-03", periods=260, freq="D")
    pattern = np.array([10, 12, 11, 14, 9, 1, 6], dtype=float)
    level = rng.uniform(0.5, 3, size=12)
    sales = rng.poisson(pattern[days.dayofweek][:, None] * level[None, :]).astype(float)
    matrix = pd.DataFrame(sales, index=days, columns=[f"P{i}" for i in range(12)])
    out = backtest_and_forecast(matrix, TrainSettings(epochs=25, hidden=64))
    m = out.metrics
    assert m["wape_shelfcast"] < m["wape_28_day_average"]
    assert len(out.future) == 12 * H and set(out.future["h"]) == set(range(1, H + 1))
    assert out.future["forecast_date"].min() == days[-1] + pd.offsets.Day(1)
    assert (out.future["total_p10"] <= out.future["total_p90"]).all()


def test_wape():
    assert wape(np.array([10.0, 10.0]), np.array([5.0, 15.0])) == 0.5
    assert np.isnan(wape(np.zeros(3), np.ones(3)))
