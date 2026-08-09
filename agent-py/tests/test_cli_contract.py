from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner

import main as agent_main
from neuronpedia_agent.api import GeneratedGraph


class FakeGraphClient:
    base_url = "http://graph.test"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_graph(self, **kwargs: Any) -> GeneratedGraph:
        self.calls.append(kwargs)
        output = Path(kwargs["output_path"])
        return GeneratedGraph(
            path=output,
            manifest_path=Path(f"{output}.manifest.json"),
            byte_count=27,
            sha256="a" * 64,
            content_type="application/json",
        )


def test_generate_only_executes_transport_instead_of_describing_it(monkeypatch) -> None:
    fake = FakeGraphClient()
    monkeypatch.setattr(agent_main, "_make_client", lambda *args, **kwargs: fake)
    runner = CliRunner()

    result = runner.invoke(
        agent_main.cli,
        [
            "generate-only",
            "--prompt",
            "The capital of France is",
            "--model",
            "google/gemma-2-2b",
            "--output",
            "graph.json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1
    assert fake.calls[0]["prompt"] == "The capital of France is"
    assert fake.calls[0]["model_id"] == "google/gemma-2-2b"
    assert fake.calls[0]["output_path"] == "graph.json"
    assert '"status": "ok"' in result.output
    assert "Would use" not in result.output
    assert "Requires API integration" not in result.output
    assert "[Makes API call" not in result.output
