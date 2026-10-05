"""Loopback-only tests for local Ollama answer generation."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ktem.local_qa_core import EvidenceCard
from ktem.local_qa_ollama import OllamaLocalClient


@pytest.fixture
def loopback_ollama():
    class OllamaTestServer(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False

        def __init__(self):
            super().__init__(("127.0.0.1", 0), RequestHandler)
            self.request_lock = threading.Lock()
            self.request_count = 0
            self.redirect_target_count = 0
            self.last_body = None
            self.redirect = False
            self.response_status = 200
            self.response_body = json.dumps(
                {"message": {"content": "Answer [1]"}}
            ).encode("utf-8")
            self.response_delay = 0.0
            self.stream_parts = None
            self.stream_first_sent = threading.Event()
            self.release_stream = threading.Event()
            self.stream_completed = threading.Event()
            self.stream_gate = False
            self.url = f"http://127.0.0.1:{self.server_address[1]}"

        @property
        def last_json(self):
            with self.request_lock:
                if self.last_body is None:
                    return None
                return json.loads(self.last_body)

    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/capture":
                with self.server.request_lock:
                    self.server.redirect_target_count += 1
                self._respond(200, b"captured")
                return

            if self.path != "/api/chat":
                self._respond(404, b"missing")
                return

            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with self.server.request_lock:
                self.server.request_count += 1
                self.server.last_body = body
                redirect = self.server.redirect
                status = self.server.response_status
                response_body = self.server.response_body
                response_delay = self.server.response_delay

            if redirect:
                self.send_response(302)
                self.send_header("Location", "/capture")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if response_delay:
                time.sleep(response_delay)
            if (
                json.loads(body).get("stream") is True
                and self.server.stream_parts is not None
            ):
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for index, part in enumerate(self.server.stream_parts):
                        self.wfile.write(part)
                        self.wfile.flush()
                        if index == 0:
                            self.server.stream_first_sent.set()
                            if self.server.stream_gate:
                                self.server.release_stream.wait(timeout=3)
                    self.server.stream_completed.set()
                except OSError:
                    # The streaming client may stop reading after a bounded
                    # response validation error.
                    pass
                return
            self._respond(status, response_body)

        def _respond(self, status, body):
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                # A client timeout may close the socket before this synthetic
                # server sends its response.
                pass

        def log_message(self, _format, *_args):
            pass

    server = OllamaTestServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def card(
    *,
    source_rank=1,
    chunk_rank=1,
    source_id="source-a",
    source_label="policy.pdf",
    locator=None,
    score=0.8,
    text="shown text",
):
    return EvidenceCard(
        source_rank=source_rank,
        chunk_rank=chunk_rank,
        source_id=source_id,
        source_label=source_label,
        locator=locator if locator is not None else {"page": 2},
        score=score,
        text=text,
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434",
        "http://localhost:11434",
        "http://user:pass@127.0.0.1:11434",
        "http://@127.0.0.1:11434",
        "http://127.0.0.1:11434?token=secret",
        "http://127.0.0.1:11434#fragment",
        "http://127.0.0.1:bad-port",
        "http://127.0.0.1:11434/api",
    ],
)
def test_rejects_unsupported_endpoint_before_connection(loopback_ollama, endpoint):
    with pytest.raises(ValueError):
        OllamaLocalClient(endpoint=endpoint)
    assert loopback_ollama.request_count == 0


def test_accepts_ipv6_loopback_endpoint_without_connecting():
    OllamaLocalClient(endpoint="http://[::1]:11434")


@pytest.mark.parametrize(
    "model",
    ["", "   ", "qwen2.5:7b-cloud", "QWEN2.5:CLOUD", "https://127.0.0.1/model"],
)
def test_rejects_invalid_or_cloud_model_before_connection(loopback_ollama, model):
    with pytest.raises(ValueError):
        OllamaLocalClient(endpoint=loopback_ollama.url, model=model)
    assert loopback_ollama.request_count == 0


def test_prompt_contains_only_displayed_cards_and_preserves_citation_mapping(
    loopback_ollama,
):
    shown = card(
        source_rank=2,
        chunk_rank=3,
        source_id="source-b",
        source_label="policy.pdf",
        locator={"page": 7},
        score=0.61,
        text="shown text",
    )
    answer = OllamaLocalClient(endpoint=loopback_ollama.url).generate(
        "What does the policy say?", [shown]
    )

    payload = loopback_ollama.last_json
    assert answer == "Answer [1]"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "30m"
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
    ]
    system_prompt, user_prompt = [message["content"] for message in payload["messages"]]
    assert "untrusted" in system_prompt.lower()
    assert "concise" in system_prompt.lower()
    assert "insufficient" in system_prompt.lower()
    assert "[rank]" in system_prompt.lower()
    user_payload = json.loads(user_prompt)
    assert user_payload == {
        "question": "What does the policy say?",
        "evidence": [
            {
                "source_rank": 2,
                "chunk_rank": 3,
                "source_id": "source-b",
                "source_label": "policy.pdf",
                "locator": {"page": 7},
                "score": 0.61,
                "text": "shown text",
            }
        ],
    }
    assert "undisplayed candidate" not in user_prompt


def test_ignores_proxy_environment_for_loopback(loopback_ollama, monkeypatch):
    # Port 1 is loopback too; an opener that honors these proxy settings fails
    # instead of reaching the synthetic Ollama server directly.
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")

    answer = OllamaLocalClient(endpoint=loopback_ollama.url).generate(
        "A local question", [card()]
    )

    assert answer == "Answer [1]"
    assert loopback_ollama.request_count == 1


def test_redirect_is_an_error_and_location_is_never_contacted(loopback_ollama):
    loopback_ollama.redirect = True
    client = OllamaLocalClient(endpoint=loopback_ollama.url)

    with pytest.raises((ValueError, urllib.error.HTTPError)):
        client.generate("secret question", [card(text="secret evidence")])

    assert loopback_ollama.request_count == 1
    assert loopback_ollama.redirect_target_count == 0


@pytest.mark.parametrize(
    "body",
    [
        b"not JSON",
        b"{}",
        b'{"message": {"content": ""}}',
        b'{"message": {"content": "   "}}',
        b'{"message": {"content": 3}}',
    ],
)
def test_rejects_malformed_or_empty_ollama_response(loopback_ollama, body):
    loopback_ollama.response_body = body
    with pytest.raises(ValueError):
        OllamaLocalClient(endpoint=loopback_ollama.url).generate(
            "secret question", [card(text="secret evidence")]
        )


def test_http_failure_does_not_expose_prompt_or_response_body(loopback_ollama):
    loopback_ollama.response_status = 500
    loopback_ollama.response_body = b"synthetic private server response"

    with pytest.raises(ValueError) as error:
        OllamaLocalClient(endpoint=loopback_ollama.url).generate(
            "secret question", [card(text="secret evidence")]
        )

    assert "secret question" not in str(error.value)
    assert "secret evidence" not in str(error.value)
    assert "synthetic private server response" not in str(error.value)


def test_uses_configured_finite_timeout(loopback_ollama):
    loopback_ollama.response_delay = 0.2

    with pytest.raises(ValueError):
        OllamaLocalClient(endpoint=loopback_ollama.url, timeout=0.05).generate(
            "timeout question", [card()]
        )

    assert loopback_ollama.request_count == 1


def _event(value):
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def test_generate_stream_sends_keep_alive_and_returns_content_chunks(loopback_ollama):
    loopback_ollama.stream_parts = [
        _event({"message": {"content": "Answer "}, "done": False}),
        _event({"message": {"content": ""}, "done": False}),
        _event({"message": {"content": "[1]"}, "done": False}),
        _event({"message": {"content": ""}, "done": True}),
    ]

    chunks = list(
        OllamaLocalClient(endpoint=loopback_ollama.url).generate_stream(
            "synthetic question", [card(text="synthetic evidence")]
        )
    )

    assert chunks == ["Answer ", "[1]"]
    payload = loopback_ollama.last_json
    assert payload["stream"] is True
    assert payload["keep_alive"] == "30m"
    assert loopback_ollama.request_count == 1


def test_generate_stream_yields_before_server_finishes(loopback_ollama):
    loopback_ollama.stream_gate = True
    loopback_ollama.stream_parts = [
        _event({"message": {"content": "first"}, "done": False}),
        _event({"message": {"content": "second"}, "done": True}),
    ]
    chunks = OllamaLocalClient(endpoint=loopback_ollama.url).generate_stream(
        "synthetic question", [card()]
    )

    try:
        assert next(chunks) == "first"
        assert loopback_ollama.stream_first_sent.wait(timeout=1)
        assert not loopback_ollama.stream_completed.wait(timeout=0.05)
        loopback_ollama.release_stream.set()
        assert list(chunks) == ["second"]
        assert loopback_ollama.stream_completed.wait(timeout=1)
    finally:
        loopback_ollama.release_stream.set()


@pytest.mark.parametrize(
    "parts",
    [
        [b"not JSON\n"],
        [b'{"message":{"content":"invalid utf8 \xff"},"done":true}\n'],
        [_event(["not", "an", "object"])],
        [_event({"message": [], "done": True})],
        [_event({"message": {"content": 3}, "done": True})],
        [_event({"error": "private server detail"})],
        [_event({"message": {"content": "partial"}, "done": False})],
        [
            _event({"message": {"content": ""}, "done": False}),
            _event({"message": {"content": ""}, "done": True}),
        ],
    ],
)
def test_generate_stream_rejects_malformed_or_incomplete_events_without_leaking(
    loopback_ollama, parts
):
    loopback_ollama.stream_parts = parts

    with pytest.raises(ValueError) as error:
        list(
            OllamaLocalClient(endpoint=loopback_ollama.url).generate_stream(
                "private question", [card(text="private evidence")]
            )
        )

    assert type(error.value) is ValueError
    assert "private question" not in str(error.value)
    assert "private evidence" not in str(error.value)
    assert "private server detail" not in str(error.value)


def test_generate_stream_rejects_a_line_over_the_line_limit(loopback_ollama):
    loopback_ollama.stream_parts = [
        _event({"message": {"content": "x" * (1024 * 1024)}, "done": True})
    ]

    with pytest.raises(ValueError):
        list(
            OllamaLocalClient(endpoint=loopback_ollama.url).generate_stream(
                "synthetic question", [card()]
            )
        )


def test_generate_stream_rejects_total_response_over_two_mib(loopback_ollama):
    loopback_ollama.stream_parts = [
        _event({"message": {"content": "x" * 100_000}, "done": False})
        for _ in range(22)
    ]
    loopback_ollama.stream_parts.append(
        _event({"message": {"content": "done"}, "done": True})
    )

    with pytest.raises(ValueError):
        list(
            OllamaLocalClient(endpoint=loopback_ollama.url).generate_stream(
                "synthetic question", [card()]
            )
        )


def test_generate_stream_hides_http_error_body(loopback_ollama):
    loopback_ollama.response_status = 500
    loopback_ollama.response_body = b"private response body"

    with pytest.raises(ValueError) as error:
        list(
            OllamaLocalClient(endpoint=loopback_ollama.url).generate_stream(
                "private question", [card(text="private evidence")]
            )
        )

    assert "private question" not in str(error.value)
    assert "private evidence" not in str(error.value)
    assert "private response body" not in str(error.value)


def test_generate_stream_does_not_follow_redirect(loopback_ollama):
    loopback_ollama.redirect = True
    loopback_ollama.stream_parts = [_event({"done": True})]

    with pytest.raises(ValueError):
        list(
            OllamaLocalClient(endpoint=loopback_ollama.url).generate_stream(
                "synthetic question", [card()]
            )
        )

    assert loopback_ollama.request_count == 1
    assert loopback_ollama.redirect_target_count == 0


def test_generate_stream_ignores_proxy_environment(loopback_ollama, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")
    loopback_ollama.stream_parts = [
        _event({"message": {"content": "local"}, "done": True})
    ]

    chunks = list(
        OllamaLocalClient(endpoint=loopback_ollama.url).generate_stream(
            "synthetic question", [card()]
        )
    )

    assert chunks == ["local"]
    assert loopback_ollama.request_count == 1


def test_generate_stream_timeout_is_a_value_error(loopback_ollama):
    loopback_ollama.response_delay = 0.2
    loopback_ollama.stream_parts = [_event({"done": True})]

    with pytest.raises(ValueError):
        list(
            OllamaLocalClient(
                endpoint=loopback_ollama.url, timeout=0.05
            ).generate_stream("synthetic question", [card()])
        )
