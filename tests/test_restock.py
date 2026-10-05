import numpy as np
import pandas as pd

from shelfcast.restock import RestockSettings, brief, cortex_brief, recommend, simulate_inventory

H = 14


def _forecast(code="A", per_day=10.0, spread=0.3):
    h = np.arange(1, H + 1)
    total = per_day * h
    return pd.DataFrame({
        "stock_code": code,
        "h": h,
        "p10": per_day * 0.5,
        "p50": per_day,
        "p90": per_day * 1.5,
        "total_p10": total * (1 - spread),
        "total_p50": total,
        "total_p90": total * (1 + spread),
    })


def _inventory(on_hand, lead=5, on_order=0, code="A", cost=2.0):
    return pd.DataFrame([{"stock_code": code, "on_hand": on_hand, "on_order": on_order, "lead_time_days": lead, "unit_cost": cost}])


def test_order_now_when_stock_runs_out_before_delivery():
    rec = recommend(_forecast(), _inventory(on_hand=30)).iloc[0]
    # cover = 5 + 7 = 12 days -> P50 120, P90 156; 30 units last 3 days (< 5-day lead time)
    assert rec["status"] == "Order now"
    assert rec["days_left"] == 3
    assert rec["demand_mid"] == 120 and rec["demand_high"] == 156
    assert rec["order_qty"] == 156 - 30
    assert rec["order_value"] == (156 - 30) * 2.0


def test_order_this_week_when_there_is_still_time():
    rec = recommend(_forecast(), _inventory(on_hand=100)).iloc[0]
    assert rec["status"] == "Order this week"
    assert rec["days_left"] == 10
    assert rec["order_qty"] == 56


def test_ok_when_stock_covers_the_period():
    rec = recommend(_forecast(), _inventory(on_hand=1000)).iloc[0]
    assert rec["status"] == "OK"
    assert rec["order_qty"] == 0
    assert rec["days_left"] == H


def test_buffer_and_stock_on_order():
    median_plan = recommend(_forecast(), _inventory(on_hand=30), settings=RestockSettings(buffer=0.0)).iloc[0]
    assert median_plan["order_qty"] == 120 - 30
    with_open_order = recommend(_forecast(), _inventory(on_hand=30, on_order=50)).iloc[0]
    assert with_open_order["order_qty"] == 156 - 30 - 50


def test_cover_period_is_capped_at_the_horizon():
    rec = recommend(_forecast(), _inventory(on_hand=0, lead=10), settings=RestockSettings(review_days=14)).iloc[0]
    assert rec["demand_mid"] == 10 * H


def test_sorting_names_and_missing_forecasts():
    fc = pd.concat([_forecast("A"), _forecast("B", per_day=1.0)])
    inv = pd.concat([_inventory(1000, code="A"), _inventory(0, code="B"), _inventory(5, code="NO_FORECAST")])
    products = pd.DataFrame({"stock_code": ["A", "B"], "description": ["Mug", "Lantern"]})
    recs = recommend(fc, inv, products)
    assert list(recs["stock_code"]) == ["B", "A"]  # urgent first
    assert list(recs["description"]) == ["Lantern", "Mug"]
    text = brief(recs, "2011-12-09")
    assert "Lantern" in text and "Mug" not in text  # OK products are not listed


def test_simulated_inventory_is_sane():
    days = pd.date_range("2011-01-01", periods=60, freq="D")
    matrix = pd.DataFrame({"A": np.full(60, 10.0), "B": np.zeros(60)}, index=days)
    inv = simulate_inventory(matrix, pd.Series({"A": 5.0}))
    assert list(inv.columns) == ["stock_code", "on_hand", "on_order", "lead_time_days", "unit_cost"]
    assert (inv["on_hand"] >= 0).all() and set(inv["lead_time_days"]) <= {3, 5, 7, 10}
    assert inv.loc[inv.stock_code == "A", "unit_cost"].item() == 3.0
    assert inv.loc[inv.stock_code == "B", "on_hand"].item() == 0


class _FakeCortex:
    """Pretends the first function name is unknown, like an older Snowflake account."""

    def __init__(self):
        self.calls = []

    def query(self, sql, params):
        self.calls.append((sql, params[0]))
        if "AI_COMPLETE" in sql:
            raise RuntimeError("Unknown function AI_COMPLETE")
        return pd.DataFrame({"answer": ['"Order the lanterns first."']})


def test_cortex_brief_falls_back_to_older_function():
    recs = recommend(_forecast(), _inventory(30), pd.DataFrame({"stock_code": ["A"], "description": ["Mug"]}))
    fake = _FakeCortex()
    text, model = cortex_brief(fake, recs, "2011-12-09", model="some-model")
    assert text == "Order the lanterns first."
    assert model == "some-model"
    assert [c[0].split("(")[0] for c in fake.calls] == ["SELECT AI_COMPLETE", "SELECT SNOWFLAKE.CORTEX.COMPLETE"]
