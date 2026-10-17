import json

import pandas as pd
import pytest

from shelfcast.brief_ai import ai_brief, allowed_numbers, brief_context, check_grounding
from shelfcast.llm import FakeClient, LLMError, LLMSettings, parse_chat_completion
from shelfcast.restock import brief, recommend

from test_restock import _forecast, _inventory

PRODUCTS = pd.DataFrame({"stock_code": ["85123A", "22423", "71053"], "description": ["WHITE HANGING HEART", "REGENCY CAKESTAND", "WHITE METAL LANTERN"]})


def _recs():
    fc = pd.concat([_forecast("85123A", per_day=10.0), _forecast("22423", per_day=4.0), _forecast("71053", per_day=1.0)])
    inv = pd.concat([_inventory(30, code="85123A", cost=2.0), _inventory(20, code="22423", lead=7, cost=5.0), _inventory(500, code="71053")])
    return recommend(fc, inv, PRODUCTS)


def test_context_holds_rows_and_totals():
    recs = _recs()
    ctx = brief_context(recs, "2011-12-09", top=2)
    assert [r["stock_code"] for r in ctx["rows"]] == ["22423", "85123A"]  # both urgent, bigger order value first
    assert ctx["totals"]["products_reviewed"] == 3 and ctx["totals"]["products_shown"] == 2
    assert ctx["totals"]["order_value_gbp"] == round(float(recs["order_value"].sum()), 2)
    numbers = allowed_numbers(ctx)
    assert 126.0 in numbers and 2011.0 in numbers  # order_qty of the first product; the year


def test_grounding_accepts_numbers_from_the_table_and_rejects_others():
    recs = _recs()
    ctx = brief_context(recs, "2011-12-09")
    good = "Order WHITE HANGING HEART first: 126 units (£252), stock lasts 3 days. Then REGENCY CAKESTAND. Risk: 2 products need orders today."
    report = check_grounding(good, ctx)
    assert report.ok, report.problems
    assert report.numbers_checked == 4

    bad = "Order WHITE HANGING HEART first: 999 units for about £1,500. Also restock 12345B."
    report = check_grounding(bad, ctx)
    assert not report.ok
    assert report.unverified_numbers == ["999", "1,500"]
    assert report.unknown_codes == ["12345B"]

    vague = "Please order the usual items this week."
    report = check_grounding(vague, ctx)
    assert not report.mentions_top_product and not report.ok

    report = check_grounding("word " * 201, ctx)
    assert report.words == 201 and "too long" in report.problems[-1]


def test_grounding_allows_rounding_and_ignores_stock_codes_as_numbers():
    ctx = {"as_of": "2011-12-09", "rows": [{"status": "Order now", "stock_code": "85123A", "description": "MUG", "order_value": 1234.56}], "totals": {"products_shown": 1}}
    assert check_grounding("Order MUG (85123A) now: about £1,235.", ctx).ok
    assert check_grounding("Order MUG now: £1,234.56.", ctx).ok
    assert check_grounding("Order MUG now: £1,240.", ctx).unverified_numbers == ["1,240"]


def test_ai_brief_keeps_a_grounded_answer():
    recs = _recs()
    client = FakeClient(["Order WHITE HANGING HEART first: 126 units, stock lasts 3 days. Risk: REGENCY CAKESTAND runs low."])
    result = ai_brief(client, recs, "2011-12-09")
    assert result.source == "ai" and result.attempts == 1 and result.report.ok
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert "85123A" in client.calls[0]["messages"][1]["content"]


def test_ai_brief_retries_once_then_falls_back_to_the_template():
    recs = _recs()
    client = FakeClient(["Order 999 units of WHITE HANGING HEART.", "Order 777 units of WHITE HANGING HEART."])
    result = ai_brief(client, recs, "2011-12-09")
    assert result.source == "template" and result.attempts == 2
    assert result.text == brief(recs, "2011-12-09")
    assert "777" in result.fallback_reason
    second_call = client.calls[1]["messages"]
    assert second_call[-1]["role"] == "user" and "999" in second_call[-1]["content"]  # told what was wrong


def test_ai_brief_without_a_model_or_on_error_uses_the_template():
    recs = _recs()
    assert ai_brief(None, recs, "2011-12-09").source == "template"

    class Broken:
        model = "down"

        def complete(self, *a, **k):
            raise LLMError("HTTP 503")

    result = ai_brief(Broken(), recs, "2011-12-09")
    assert result.source == "template" and "503" in result.fallback_reason


def test_parse_chat_completion_reads_text_and_tool_calls():
    payload = {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "what_if", "arguments": json.dumps({"lead_time_days": 5})}}
        ]}}],
    }
    resp = parse_chat_completion(payload, "fallback")
    assert resp.text == "" and resp.tool_calls[0].name == "what_if" and resp.tool_calls[0].arguments == {"lead_time_days": 5}
    with pytest.raises(LLMError):
        parse_chat_completion({"error": "bad"}, "m")


def test_settings_come_from_the_environment(monkeypatch):
    for key in ("SHELFCAST_LLM_API_KEY", "OPENAI_API_KEY", "SHELFCAST_LLM_MODEL", "SHELFCAST_LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    assert LLMSettings.from_env() is None
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("SHELFCAST_LLM_BASE_URL", "http://localhost:11434/v1/")
    s = LLMSettings.from_env(model="llama3")
    assert s.api_key == "k" and s.model == "llama3" and s.base_url == "http://localhost:11434/v1"
