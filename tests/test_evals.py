import json
import re

from shelfcast import evals
from shelfcast.llm import FakeClient


def _rows_from_messages(messages):
    """What a careful model would do: read the order list it was given."""
    user = messages[1]["content"]
    start, end = user.index("[", user.index("Order list")), user.rindex("]", 0, user.index("Totals"))
    return json.loads(user[start:end + 1])


def careful_model(messages):
    rows = _rows_from_messages(messages)
    urgent = [r for r in rows if r["status"] != "OK"]
    if not urgent:
        return "Nothing needs an order today; stock levels cover the next two weeks."
    r = urgent[0]
    return f"Order {r['description']} first: {r['order_qty']} units (stock lasts {r['days_left']} days). Risk: a late delivery."


def sloppy_model(messages):
    rows = _rows_from_messages(messages)
    text = careful_model(messages)
    if len(rows) % 2 == 0:  # every other case: invent a number
        text = re.sub(r"\d+ units", "999 units", text, count=1)
    return text


def test_make_cases_covers_settings_and_edge_cases(tiny_warehouse, tmp_path):
    from shelfcast import pipeline
    from shelfcast.db import DuckDBBackend

    with DuckDBBackend(tiny_warehouse["db_path"]) as db:
        inputs = pipeline.read_plan_inputs(db)
    cases = evals.make_cases(inputs, tmp_path / "cases.json")
    assert (tmp_path / "cases.json").exists()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(evals.SETTINGS_GRID) * len(evals.TOP_GRID) + 3
    assert "review7_buffer100_top8" in ids and "edge_nothing_to_order" in ids
    assert all(set(c["context"]) == {"as_of", "rows", "totals"} for c in cases)
    assert evals.load_cases(tmp_path / "cases.json") == cases


def test_run_scores_models_and_the_template_baseline(tmp_path):
    cases = evals.edge_cases("2011-12-09")
    scores = evals.run(cases, {"template": "template", "careful": FakeClient(careful_model), "sloppy": FakeClient(sloppy_model)}, log=lambda *_: None)
    by_name = {s.model: s for s in scores}
    assert by_name["template"].grounded == 1.0
    assert by_name["careful"].grounded_first_try == 1.0 and by_name["careful"].fallback == 0.0
    sloppy = by_name["sloppy"]
    assert 0 < sloppy.grounded_first_try < 1.0
    assert sloppy.fallback > 0  # the retry repeats the invented number, so those cases fall back
    assert all("999" in "; ".join(f.problems) for f in sloppy.failures)

    path = evals.write_report(scores, tmp_path, cases_path=tmp_path / "cases.json")
    text = path.read_text(encoding="utf-8")
    assert "| template | 100% |" in text and "| sloppy |" in text and "edge_large_numbers" in text
    assert json.loads((tmp_path / "brief_evals.json").read_text(encoding="utf-8"))[0]["model"] == "template"
