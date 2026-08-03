from __future__ import annotations

import argparse
import builtins
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from autocircuit import pipeline
from autocircuit.baseline import (
    BaselineMetrics,
    make_example_result,
    make_failed_result,
    write_example_results,
)
from autocircuit.candidates import (
    candidate_is_eligible,
    candidate_registry,
    frozen_config,
    select_candidate,
)
from autocircuit.config import MVPConfig
from autocircuit.datasets.associative_recall import (
    VALUES,
    ExamplePair,
    normalized_query_position,
)
from autocircuit.diagnostics import correctness_quadrant, group_metrics
from autocircuit.pipeline import PythiaAdapter, verify_resume


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


def test_failed_record_retains_identity_and_prevents_eligibility() -> None:
    failed = make_failed_result(example(), RuntimeError(" deterministic  failure\n"))
    assert failed.example_id == "example-1"
    assert failed.clean_prompt == example().clean_prompt
    assert failed.processing_status == "error"
    assert failed.clean_logit_difference is None and failed.clean_correct is None
    assert failed.error == "RuntimeError: deterministic failure"
    metrics = BaselineMetrics(1.0, 1.0, 2.0, -2.0, 4.0, 3, 1)
    assert not candidate_is_eligible(metrics, 4)


def test_adapter_batches_by_length_and_preserves_order() -> None:
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def __call__(self, prompts: list[str], return_type: str):
            assert return_type == "logits"
            self.calls.append(prompts)
            values = torch.zeros((len(prompts), 1, 10))
            for row, prompt in enumerate(prompts):
                values[row, 0, 1] = float(prompt.removeprefix("prompt-"))
            return values

    items = [example(i) for i in (3, 1, 2)]
    items = [
        ExamplePair(
            **(
                asdict(item)
                | {
                    "clean_prompt": f"prompt-{item.example_id.removeprefix('example-')}",
                    "corrupt_prompt": "prompt-0",
                    "metadata": item.metadata
                    | {"prompt_token_length": 7 if item.example_id != "example-1" else 8},
                }
            )
        )
        for item in items
    ]
    adapter = object.__new__(PythiaAdapter)
    adapter.model = Model()
    records = adapter.score(items, batch_size=2)
    assert [record.example_id for record in records] == [item.example_id for item in items]
    assert [record.clean_target_logit for record in records] == [3.0, 1.0, 2.0]
    assert max(len(call) for call in adapter.model.calls) == 2

    class FailingModel:
        def __call__(self, prompts: list[str], return_type: str):
            raise RuntimeError("offline batch failure")

    adapter.model = FailingModel()
    failures = adapter.score(items, batch_size=2)
    assert [record.example_id for record in failures] == [item.example_id for item in items]
    assert len(failures) == len(items)
    assert all(record.processing_status == "error" for record in failures)
    assert all(record.error == "RuntimeError: offline batch failure" for record in failures)


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
    frozen = frozen_config(
        candidate_registry("likes")[0],
        passing,
        "pythia-70m",
        "fake-tokenizer",
        "main",
        None,
        42,
    )
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
    tokenizer_id = "offline"
    resolved_revision = None
    revision_resolution_error = "offline fake"

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

    resumed = argparse.Namespace(**(vars(args) | {"resume": True}))
    assert (
        pipeline.run_discovery(
            resumed,
            lambda model, revision, device: (_ for _ in ()).throw(AssertionError("model loaded")),
        )
        == final
    )


def test_resume_detects_tampered_candidate_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "load_config", lambda path: MVPConfig("pythia-70m", 2, 4, 2, 2))
    args = argparse.Namespace(
        model="pythia-70m",
        revision="main",
        device="cpu",
        batch_size=2,
        seed=3,
        output=tmp_path,
        resume=False,
        force=False,
    )
    pipeline.run_discovery(args, lambda model, revision, device: FakeAdapter())
    root = tmp_path / "seed-3-main"
    candidate_record = next((root / "candidates").glob("*/examples.jsonl"))
    original = candidate_record.read_bytes()
    candidate_record.write_bytes(original + b"tamper")
    with pytest.raises(RuntimeError, match="hash verification"):
        pipeline.run_discovery(
            argparse.Namespace(**(vars(args) | {"resume": True})), lambda *unused: FakeAdapter()
        )
    candidate_record.write_bytes(original)
    report = root / "v1" / "diagnostics.md"
    report.write_text("tamper", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash verification"):
        pipeline.run_discovery(
            argparse.Namespace(**(vars(args) | {"resume": True})), lambda *unused: FakeAdapter()
        )


def test_interrupted_resume_and_input_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "load_config", lambda path: MVPConfig("pythia-70m", 2, 4, 2, 2))
    args = argparse.Namespace(
        model="pythia-70m",
        revision="main",
        device="cpu",
        batch_size=2,
        seed=4,
        output=tmp_path,
        resume=False,
        force=False,
    )
    pipeline.run_discovery(args, lambda *unused: FakeAdapter())
    manifest_path = tmp_path / "seed-4-main" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    resumed = argparse.Namespace(**(vars(args) | {"resume": True}))
    assert pipeline.run_discovery(resumed, lambda *unused: FakeAdapter())["software_success"]
    mismatch = argparse.Namespace(**(vars(resumed) | {"batch_size": 3}))
    with pytest.raises(RuntimeError, match="inputs do not match"):
        pipeline.run_discovery(mismatch, lambda *unused: FakeAdapter())


def test_discovery_never_accesses_held_out_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("validation.jsonl", "test.jsonl"):
        (tmp_path / name).write_text("SENTINEL", encoding="utf-8")
    real_open = builtins.open
    real_path_open = Path.open
    real_read_text = Path.read_text
    real_read_bytes = Path.read_bytes
    real_stat = Path.stat

    def reject_held_out(path: Path) -> None:
        if path.name in {"validation.jsonl", "test.jsonl"}:
            raise AssertionError("held-out example content accessed")

    def guarded_open(file: object, *args: object, **kwargs: object):
        reject_held_out(Path(file))  # type: ignore[arg-type]
        return real_open(file, *args, **kwargs)

    def guarded_path_open(path: Path, *args: object, **kwargs: object):
        reject_held_out(path)
        return real_path_open(path, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_read_text(path: Path, *args: object, **kwargs: object):
        reject_held_out(path)
        return real_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_read_bytes(path: Path) -> bytes:
        reject_held_out(path)
        return real_read_bytes(path)

    def guarded_stat(path: Path, *args: object, **kwargs: object):
        reject_held_out(path)
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(pipeline, "load_config", lambda path: MVPConfig("pythia-70m", 2, 4, 2, 2))
    args = argparse.Namespace(
        model="pythia-70m",
        revision="main",
        device="cpu",
        batch_size=2,
        seed=5,
        output=tmp_path / "artifacts",
        resume=False,
        force=False,
    )
    assert (
        pipeline.run_discovery(args, lambda *unused: FakeAdapter())["held_out_splits_opened"]
        is False
    )


def test_software_failure_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(args: argparse.Namespace) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "run_discovery", fail)
    assert pipeline.main(["discovery", "--device", "cpu"]) == 1


def test_resume_and_force_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as exc:
        pipeline.build_parser().parse_args(["discovery", "--resume", "--force"])
    assert exc.value.code == 2
