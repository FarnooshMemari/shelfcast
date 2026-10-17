import json

import pandas as pd

from shelfcast.assistant import TOOL_SPECS, Assistant, RestockTools, grounding_context
from shelfcast.llm import FakeClient, LLMError
from shelfcast.restock import RestockSettings

from test_restock import _forecast, _inventory

PRODUCTS = pd.DataFrame({
    "stock_code": ["85123A", "22423", "71053"],
    "description": ["WHITE HANGING HEART T-LIGHT HOLDER", "REGENCY CAKESTAND 3 TIER", "WHITE METAL LANTERN"],
})


def _tools(review_days=7, buffer=1.0):
    fc = pd.concat([_forecast("85123A", per_day=10.0), _forecast("22423", per_day=4.0), _forecast("71053", per_day=1.0)])
    inv = pd.concat([_inventory(30, code="85123A", cost=2.0), _inventory(20, code="22423", lead=7, cost=5.0), _inventory(500, code="71053")])
    history = pd.DataFrame({
        "stock_code": ["85123A"] * 40, "sale_date": pd.date_range("2011-10-31", periods=40, freq="D"), "units": [10.0] * 40,
    })
    return RestockTools(fc, inv, PRODUCTS, RestockSettings(review_days, buffer), history=history, as_of="2011-12-09")


def test_tool_specs_are_valid_function_schemas():
    names = [t["function"]["name"] for t in TOOL_SPECS]
    assert names == ["get_restock_plan", "get_product", "what_if", "search_products"]
    assert all(t["function"]["parameters"]["type"] == "object" for t in TOOL_SPECS)


def test_find_product_by_code_name_substring_and_words():
    t = _tools()
    assert t.find_product("22423") == "22423"
    assert t.find_product("white metal lantern") == "71053"
    assert t.find_product("lantern") == "71053"
    assert t.find_product("cakestand tier") == "22423"
    assert t.find_product("white") == "71053"  # shortest name wins among substring matches
    assert t.find_product("teapot") is None


def test_get_restock_plan_and_search():
    t = _tools()
    plan = t.get_restock_plan()
    assert [r["stock_code"] for r in plan["rows"]] == ["22423", "85123A"]  # urgent first, OK rows left out
    assert plan["totals"] == {"order_now": 2, "order_this_week": 0, "products": 3, "order_value_gbp": 517.0}
    assert len(t.get_restock_plan(status="all", top=1)["rows"]) == 1
    assert t.get_restock_plan(status="OK")["rows"][0]["stock_code"] == "71053"
    hits = t.search_products("white")
    assert {r["stock_code"] for r in hits["rows"]} == {"85123A", "71053"}
    assert t.search_products("") == {"rows": []}


def test_get_product_returns_plan_forecast_and_recent_sales():
    t = _tools()
    out = t.get_product("hanging heart")
    assert out["product"]["stock_code"] == "85123A" and out["product"]["order_qty"] == 126
    assert out["forecast_next_7_days"] == {"low": 49.0, "expected": 70.0, "high": 91.0}
    assert out["forecast_next_14_days"]["expected"] == 140.0
    assert out["units_sold_last_28_days"] == 280
    assert "error" in t.get_product("teapot")


def test_what_if_changes_one_product_or_the_settings():
    t = _tools()
    out = t.what_if(lead_time_days=10, product="22423")
    assert out["settings"]["product"] == "REGENCY CAKESTAND 3 TIER"
    # cover is capped at the 14-day horizon, so demand is unchanged and the order stays 53
    assert out["change"]["order_qty_before"] == 53 and out["change"]["order_qty_after"] == 53
    cheaper = t.what_if(buffer_pct=0)
    assert cheaper["totals"]["order_value_gbp"] < cheaper["totals_before"]["order_value_gbp"]
    assert cheaper["settings"]["buffer_pct"] == 0 and cheaper["settings"]["product"] == "all"
    short = t.what_if(lead_time_days=1, product="85123A")
    assert short["change"]["status_after"] == "Order this week"  # 30 units last 3 days, which now beats a 1-day lead time
    assert "error" in t.what_if(product="teapot")
    assert "error" in t.call("drop_table", {})
    assert "Bad arguments" in t.call("what_if", {"nope": 1})["error"]


def test_assistant_calls_tools_then_answers_with_verified_numbers():
    t = _tools()
    script = [
        [("what_if", {"lead_time_days": 1, "product": "white hanging heart"})],
        "If the lead time drops to 1 day, WHITE HANGING HEART T-LIGHT HOLDER moves from Order now to Order this week and the order falls from 126 to 74 units.",
    ]
    client = FakeClient(script)
    answer = Assistant(client, t).ask("What if the heart holder's lead time becomes 1 day?")
    assert answer.rounds == 2 and [r.name for r in answer.tool_runs] == ["what_if"]
    assert answer.tool_runs[0].result["change"]["status_after"] == "Order this week"
    assert answer.verified and answer.report.numbers_checked == 3
    # the tool result went back to the model as a tool message
    tool_messages = [m for m in client.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1 and json.loads(tool_messages[0]["content"])["change"]["order_qty_after"] == 74
    assert client.calls[0]["tools"] == TOOL_SPECS


def test_assistant_flags_numbers_the_tools_never_returned():
    t = _tools()
    client = FakeClient([[("get_restock_plan", {})], "Order 500 units of the cakestand today.", "Order 500 units of the cakestand today."])
    answer = Assistant(client, t).ask("What should I order?")
    assert not answer.verified
    assert answer.report.unverified_numbers == ["500"] and answer.rounds == 3
    correction = client.calls[2]["messages"][-1]
    assert correction["role"] == "user" and "500" in correction["content"]


def test_assistant_keeps_history_and_declines_without_numbers():
    t = _tools()
    client = FakeClient(["I can only help with restocking this shop's products."])
    answer = Assistant(client, t).ask("Write me a poem", history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert answer.verified and not answer.tool_runs
    assert [m["role"] for m in client.calls[0]["messages"]] == ["system", "user", "assistant", "user"]


def test_assistant_reports_model_errors_and_round_limits():
    t = _tools()

    class Down:
        model = "down"

        def complete(self, *a, **k):
            raise LLMError("timeout")

    answer = Assistant(Down(), t).ask("What should I order?")
    assert answer.error == "timeout" and "could not reach" in answer.text

    looping = FakeClient(lambda messages: [("get_restock_plan", {})])
    answer = Assistant(looping, t, max_rounds=2).ask("What should I order?")
    assert len(answer.tool_runs) == 3 and "too many lookups" in answer.text


def test_grounding_context_collects_rows_and_arguments():
    t = _tools()
    from shelfcast.assistant import ToolRun

    runs = [ToolRun("get_product", {"product": "lantern"}, t.get_product("lantern")), ToolRun("search_products", {"query": "white"}, t.search_products("white"))]
    ctx = grounding_context(runs, "2011-12-09")
    assert {r["stock_code"] for r in ctx["rows"]} == {"85123A", "71053"}
    assert ctx["as_of"] == "2011-12-09"
