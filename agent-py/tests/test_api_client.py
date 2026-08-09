from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from neuronpedia_agent.api.client import NeuronpediaAPIError, NeuronpediaGraphClient


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: bytes = b'{}',
        json_data: Any = None,
        content_type: str = "application/json",
    ) -> None:
        self.status_code = status_code
        self._body = body
        self._json_data = json_data
        self.headers = {"Content-Type": content_type}
        self.closed = False

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    def iter_content(self, chunk_size: int) -> Iterable[bytes]:
        del chunk_size
        yield self._body

    def json(self) -> Any:
        if self._json_data is not None:
            return self._json_data
        raise ValueError("no JSON fixture")

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected request")
        return self.responses.pop(0)


def test_generate_graph_executes_request_and_writes_atomically(tmp_path: Path) -> None:
    response = FakeResponse(body=b'{"nodes": [], "edges": []}')
    session = FakeSession([response])
    client = NeuronpediaGraphClient(
        base_url="http://graph.test/",
        secret="secret-value",
        session=session,  # type: ignore[arg-type]
    )
    output = tmp_path / "graph.json"

    result = client.generate_graph(
        prompt="The capital of France is",
        model_id="google/gemma-2-2b",
        output_path=output,
    )

    assert result.path == output
    assert output.read_bytes() == b'{"nodes": [], "edges": []}'
    assert response.closed
    assert session.calls[0]["url"] == "http://graph.test/generate-graph"
    assert session.calls[0]["headers"]["x-secret-key"] == "secret-value"
    assert session.calls[0]["json"]["prompt"] == "The capital of France is"


def test_http_error_is_not_converted_to_fake_success() -> None:
    response = FakeResponse(status_code=401, body=b'{"detail":"unauthorized"}')
    client = NeuronpediaGraphClient(
        base_url="http://graph.test",
        secret="bad-secret",
        session=FakeSession([response]),  # type: ignore[arg-type]
    )

    with pytest.raises(NeuronpediaAPIError) as caught:
        client.steer({"prompt": "x", "model_id": "google/gemma-2-2b", "features": []})

    assert caught.value.status_code == 401
    assert "unauthorized" in caught.value.body


def test_steer_returns_structured_json() -> None:
    payload = {"DEFAULT_GENERATION": "Paris", "STEERED_GENERATION": "Lyon"}
    response = FakeResponse(json_data=payload)
    client = NeuronpediaGraphClient(
        base_url="http://graph.test",
        session=FakeSession([response]),  # type: ignore[arg-type]
    )

    assert client.steer({"prompt": "x", "features": []}) == payload
    assert response.closed


def test_retryable_status_can_retry_when_explicitly_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    first = FakeResponse(status_code=429, body=b"rate limited")
    second = FakeResponse(json_data={"ok": True})
    session = FakeSession([first, second])
    client = NeuronpediaGraphClient(
        base_url="http://graph.test",
        retries=1,
        session=session,  # type: ignore[arg-type]
    )
    monkeypatch.setattr("neuronpedia_agent.api.client.time.sleep", lambda _: None)

    assert client.steer({"prompt": "x", "features": []}) == {"ok": True}
    assert len(session.calls) == 2
    assert first.closed
