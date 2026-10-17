import json

from shelfcast.llm import LLMResponse, ToolCall

from test_app_and_cli import _app
from test_evals import careful_model


class _DemoModel:
    """Plays both roles: writes a grounded brief, and answers the assistant with one tool call."""

    model = "demo-model"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None, max_tokens=600):
        self.calls += 1
        if tools is None:  # the brief
            text = careful_model(messages)
            return LLMResponse(text=text, model=self.model, raw_message={"role": "assistant", "content": text})
        if messages[-1]["role"] != "tool":
            call = ToolCall(id="c1", name="get_restock_plan", arguments={"top": 3})
            raw = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_restock_plan", "arguments": "{\"top\": 3}"}}]}
            return LLMResponse(text="", tool_calls=[call], model=self.model, raw_message=raw)
        result = json.loads(messages[-1]["content"])
        rows = result["rows"]
        text = (f"Order {rows[0]['description']} first ({rows[0]['order_qty']} units). "
                f"{result['totals']['order_now']} products need an order today.") if rows else "Nothing needs an order today."
        return LLMResponse(text=text, model=self.model, raw_message={"role": "assistant", "content": text})


def test_app_without_a_model_explains_how_to_turn_ai_on(duckdb_env, monkeypatch):
    for key in ("SHELFCAST_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    at = _app().run()
    assert not at.exception
    assert [t.label for t in at.tabs][-2:] == ["AI brief", "Assistant"]
    assert any("SHELFCAST_LLM_API_KEY" in i.value for i in at.info)
    assert not at.chat_input  # the assistant stays off without a model


def test_app_ai_brief_and_assistant_with_a_fake_model(duckdb_env, monkeypatch, tmp_path):
    import shelfcast.llm

    model = _DemoModel()
    monkeypatch.setattr(shelfcast.llm, "client_from_env", lambda *a, **k: model)
    monkeypatch.setenv("SHELFCAST_FEEDBACK", str(tmp_path / "feedback.jsonl"))
    at = _app().run()
    assert not at.exception

    # AI brief: written by the fake model and verified against the order list
    brief_button = next(b for b in at.button if b.label == "Write the brief with AI")
    brief_button.click().run()
    assert not at.exception
    assert any("demo-model" in c.value and "numbers in this brief match" in c.value for c in at.caption)
    assert model.calls == 1

    # Assistant: one tool call, then an answer whose numbers come from the tool result
    at.chat_input[0].set_value("What should I order today?").run()
    assert not at.exception
    assistant_turns = [m for m in at.chat_message if m.name == "assistant"]
    assert assistant_turns and "first (" in assistant_turns[-1].markdown[0].value
    assert any("What the assistant looked up (1 tool call)" in e.label for e in at.expander)
    assert any("numbers in this brief match" in c.value or "numbers in this" in c.value for c in at.caption)
    assert model.calls == 3
    assert not (tmp_path / "feedback.jsonl").exists()  # nothing is logged until someone clicks a thumb


def test_feedback_summary_counts_by_feature_and_model(tmp_path):
    from shelfcast import feedback

    path = tmp_path / "f.jsonl"
    feedback.record("brief", "up", model="m", verified=True, path=path)
    feedback.record("brief", "down", model="m", verified=False, path=path)
    feedback.record("assistant", "up", model="m", path=path)
    rows = feedback.summary(feedback.load(path))
    assert rows == [
        {"feature": "assistant", "model": "m", "ratings": 1, "rated_up": 1.0, "verified_share": None},
        {"feature": "brief", "model": "m", "ratings": 2, "rated_up": 0.5, "verified_share": 0.5},
    ]
