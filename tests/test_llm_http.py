"""The HTTP client against a tiny local server that speaks the chat-completions API."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from shelfcast.llm import LLMError, LLMSettings, OpenAICompatibleClient


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    responses: list = []  # (status, payload) in order

    def do_POST(self):  # noqa: N802 - http.server naming
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        status, payload = _Handler.responses.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture()
def server():
    _Handler.requests, _Handler.responses = [], []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def _client(base_url, model="test-model"):
    return OpenAICompatibleClient(LLMSettings(api_key="secret", model=model, base_url=base_url, timeout=5))


def test_client_posts_messages_and_tools_and_reads_the_answer(server):
    _Handler.responses = [(200, {"model": "test-model-2024", "choices": [{"message": {"role": "assistant", "content": "Order the lanterns."}}]})]
    resp = _client(server).complete([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}])
    assert resp.text == "Order the lanterns." and resp.model == "test-model-2024" and resp.seconds >= 0
    sent = _Handler.requests[0]
    assert sent["path"] == "/v1/chat/completions" and sent["auth"] == "Bearer secret"
    assert sent["body"]["model"] == "test-model" and sent["body"]["tool_choice"] == "auto" and sent["body"]["messages"][0]["content"] == "hi"


def test_client_retries_once_on_a_server_error_then_gives_up(server, monkeypatch):
    _Handler.responses = [(503, {"error": "busy"}), (200, {"choices": [{"message": {"content": "ok"}}]})]
    client = _client(server)
    import shelfcast.llm as llm

    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)  # don't wait 2 s in the test
    assert client.complete([{"role": "user", "content": "x"}]).text == "ok"
    assert len(_Handler.requests) == 2

    _Handler.responses = [(401, {"error": {"message": "bad key"}})]
    with pytest.raises(LLMError, match="HTTP 401"):
        client.complete([{"role": "user", "content": "x"}])


def test_client_reports_an_unreachable_endpoint():
    with pytest.raises(LLMError, match="Could not reach"):
        _client("http://127.0.0.1:9/v1").complete([{"role": "user", "content": "x"}])


def test_a_reply_cut_off_by_the_token_limit_is_an_error_not_a_fragment(server):
    _Handler.responses = [(200, {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "8, lead_time_days 10)"}}]})]
    with pytest.raises(LLMError, match="ran out of tokens"):
        _client(server).complete([{"role": "user", "content": "x"}])


def test_reasoning_effort_is_only_sent_when_set(server):
    _Handler.responses = [(200, {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})] * 2
    _client(server).complete([{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in _Handler.requests[0]["body"]
    settings = LLMSettings(api_key="k", model="m", base_url=server, timeout=5, reasoning_effort="low")
    OpenAICompatibleClient(settings).complete([{"role": "user", "content": "x"}])
    assert _Handler.requests[1]["body"]["reasoning_effort"] == "low" and _Handler.requests[1]["body"]["max_tokens"] == 1500
