from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from autocircuit import pipeline
from autocircuit.baseline import BaselineMetrics, make_example_result, write_example_results
from autocircuit.candidates import candidate_registry, frozen_config, select_candidate
from autocircuit.config import MVPConfig
from autocircuit.datasets.associative_recall import (
    VALUES,
    ExamplePair,
    normalized_query_position,
)
from autocircuit.diagnostics import correctness_quadrant, group_metrics
from autocircuit.pipeline import verify_resume


def example(index: int = 1) -> ExamplePair:
    return ExamplePair(
        f"example-{index}",
        f"family-{index}",
        "discovery",
        "Alice likes apples.\nAlice likes",
        "Alice likes tea.\nAlice likes",
        " apples",
        " tea",
        1,
        2,
        "queried_value_swap",
        42,
        "likes-v1",
        {
            "fact_count": 3,
            "query_entity": "Alice",
            "query_fact_index": 0,
            "normalized_query_position": "first",
            "prompt_token_length": 7,
        },
    )


def result(index: int = 1, clean: float = 2.0, corrupt: float = -1.0):
    return make_example_result(example(index), clean, 0.0, corrupt, 0.0)


def test_per_example_record_and_byte_stable_jsonl(tmp_path: Path) -> None:
    record = result()
    assert record.clean_correct and record.corrupt_correct
    assert record.clean_corrupt_recovery_span == 3.0
    first, second = tmp_path / "windows.jsonl", tmp_path / "linux.jsonl"
    assert write_example_results(first, [record]) == write_example_results(second, [record])
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()


def test_positions_and_quadrants() -> None:
    assert [normalized_query_position(i, 3) for i in range(3)] == ["first", "interior", "last"]
    assert correctness_quadrant(True, False) == "clean correct, corrupt incorrect"
    with pytest.raises(ValueError):
        normalized_query_position(3, 3)


def test_group_metrics_and_bootstrap_are_deterministic() -> None:
    records = [result(1, 2, -1), result(2, -1, 1)]
    first = group_metrics(records, bootstrap_seed=9, bootstrap_samples=100, minimum_count=3)
    assert first == group_metrics(records, bootstrap_seed=9, bootstrap_samples=100, minimum_count=3)
    template = first["template_id"][0]
    assert template["count"] == 2
    assert template["clean_accuracy"] == 0.5
    assert template["underpowered"]


def test_registry_and_selection_rule() -> None:
    assert candidate_registry("likes") == candidate_registry("likes")
    assert len(candidate_registry("likes")) == 7
    passing = BaselineMetrics(0.8, 1.0, 1.2, -1.0, 2.2, 10, 0)
    better = BaselineMetrics(0.9, 1.0, 1.0, -1.0, 2.0, 10, 0)
    failing = BaselineMetrics(0.79, 1.0, 5.0, -1.0, 6.0, 10, 0)
    assert select_candidate({"a": passing, "b": better, "c": failing}) == "b"
    assert select_candidate({"c": failing}) is None
    frozen = frozen_config(candidate_registry("likes")[0], passing, "pythia-70m", "main")
    assert "created_at" not in json.dumps(frozen)
    assert frozen["status"] == "discovery_selected_requires_held_out_validation"


def test_resume_hash_verification(tmp_path: Path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"stable")
    import hashlib

    verify_resume(path, hashlib.sha256(b"stable").hexdigest())
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash verification"):
        verify_resume(path, hashlib.sha256(b"stable").hexdigest())


def test_record_field_order_is_explicitly_stable() -> None:
    keys = list(asdict(result()))
    assert keys[0:4] == ["example_id", "family_id", "split", "template_id"]


class FakeTokenizer:
    name_or_path = "offline"

    def __init__(self) -> None:
        self.mapping = {
            value: index
            for index, value in enumerate(sum((list(group) for group in VALUES), []), 10)
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.mapping[text]] if text in self.mapping else list(range(len(text.split())))


class FakeAdapter:
    tokenizer = FakeTokenizer()
    model_id = "fake/pythia-70m"
    revision = "main"
    device = "cpu"
    dtype = "float32"

    def score(self, examples: list[ExamplePair], batch_size: int):
        del batch_size
        return [make_example_result(item, -1.0, 0.0, 1.0, 0.0) for item in examples]


def test_fake_offline_end_to_end_and_scientific_fail_is_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "load_config", lambda path: MVPConfig("pythia-70m", 2, 4, 2, 2))
    args = argparse.Namespace(
        model="pythia-70m",
        revision="main",
        device="cpu",
        batch_size=2,
        seed=2,
        output=tmp_path,
        resume=False,
        force=False,
    )
    final = pipeline.run_discovery(args, lambda model, revision, device: FakeAdapter())
    assert final["status"] == "NO_ELIGIBLE_CONFIGURATION"
    assert final["software_success"] is True
    assert not final["held_out_splits_opened"]
    assert not list(tmp_path.rglob("validation.jsonl"))
    assert not list(tmp_path.rglob("test.jsonl"))


def test_software_failure_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(args: argparse.Namespace) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "run_discovery", fail)
    assert pipeline.main(["discovery", "--device", "cpu"]) == 1
