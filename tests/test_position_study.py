from __future__ import annotations

import argparse
import builtins
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from autocircuit import pipeline
from autocircuit.baseline import make_example_result, make_failed_result
from autocircuit.datasets.associative_recall import ENTITIES, VALUES, ExamplePair
from autocircuit.position_study import (
    FAMILY_COUNT,
    decision_status,
    generate_matched_position_dataset,
    paired_position_effects,
    secondary_summaries,
    validate_matched_dataset,
    validate_scoring_identity,
    write_dataset,
)


class FakeTokenizer:
    name_or_path = "offline"

    def __init__(self) -> None:
        self.mapping = {value: index for index, value in enumerate(VALUES[0], 10)}

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
        values = {"first": 2.0, "interior": 1.0, "last": 0.0}
        return [
            make_example_result(
                item,
                values[str(item.metadata["normalized_query_position"])],
                0.0,
                -1.0,
                0.0,
            )
            for item in examples
        ]


def args(root: Path, **updates: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "command": "position-study",
        "model": "pythia-70m",
        "revision": "main",
        "device": "cpu",
        "batch_size": 8,
        "seed": 42,
        "output": root,
        "resume": False,
        "force": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_generation_is_byte_stable_balanced_and_matched(tmp_path: Path) -> None:
    first = generate_matched_position_dataset(FakeTokenizer())
    second = generate_matched_position_dataset(FakeTokenizer())
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_dataset(a, first)
    write_dataset(b, second)
    assert a.read_bytes() == b.read_bytes()
    assert len(first) == 360
    grouped: dict[str, list[ExamplePair]] = defaultdict(list)
    for item in first:
        grouped[item.family_id].append(item)
    assert len(grouped) == FAMILY_COUNT
    for variants in grouped.values():
        assert {item.metadata["normalized_query_position"] for item in variants} == {
            "first",
            "interior",
            "last",
        }
        invariant_keys = (
            "assignments",
            "corrupt_assignments",
            "matched_family_id",
            "query_entity",
            "fact_count",
            "generator_version",
        )
        invariant_values = {
            tuple(json.dumps(item.metadata[key]) for key in invariant_keys) for item in variants
        }
        assert len(invariant_values) == 1
        assert len({item.target_text for item in variants}) == 1
        assert len({item.distractor_text for item in variants}) == 1
    representatives = [variants[0] for variants in grouped.values()]
    assert set(Counter(item.target_text for item in representatives).values()) == {10}
    assert set(Counter(item.distractor_text for item in representatives).values()) == {10}
    query_counts = Counter(str(item.metadata["query_entity"]) for item in representatives)
    assert set(query_counts.values()) == {12}
    for position in ("first", "interior", "last"):
        marginal = Counter(
            (item.target_text, item.distractor_text, item.metadata["query_entity"])
            for item in first
            if item.metadata["normalized_query_position"] == position
        )
        assert marginal == Counter(
            (item.target_text, item.distractor_text, item.metadata["query_entity"])
            for item in first
            if item.metadata["normalized_query_position"] == "first"
        )
    report = validate_matched_dataset(first, FakeTokenizer())
    assert report["all_invariants_passed"]
    assert set(report["query_entity_counts"]) == set(ENTITIES[0])
    assert report["ordered_pair_observed_count"] == 120
    assert report["ordered_pair_missing_count"] == 12
    assert report["ordered_pair_minimum_including_zero"] == 0
    assert report["ordered_pair_maximum"] == 1


def test_seed_changes_population_but_preserves_all_invariants(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    populations = [
        generate_matched_position_dataset(tokenizer, seed)
        for seed in (42, 42, 43)
    ]
    paths = [tmp_path / name for name in ("42.jsonl", "same.jsonl", "43.jsonl")]
    for path, population in zip(paths, populations, strict=True):
        write_dataset(path, population)
        assert validate_matched_dataset(population, tokenizer)["all_invariants_passed"]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert paths[0].read_bytes() != paths[2].read_bytes()


def test_semantic_validation_rejects_prompt_and_invariant_tampering() -> None:
    tokenizer = FakeTokenizer()
    original = generate_matched_position_dataset(tokenizer)

    def rejected(index: int, **changes: object) -> None:
        tampered = original.copy()
        tampered[index] = replace(tampered[index], **changes)
        with pytest.raises(ValueError):
            validate_matched_dataset(tampered, tokenizer)

    lines = original[0].clean_prompt.split("\n")
    rejected(0, clean_prompt="\n".join([lines[1], lines[0], *lines[2:]]))
    rejected(1, metadata=original[1].metadata | {"query_fact_index": 0})
    rejected(2, corrupt_prompt=original[2].corrupt_prompt + " ")
    rejected(1, changed_factor="tampered")

    class PositionLengthTokenizer(FakeTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            tokens = super().encode(text, add_special_tokens)
            lines = text.split("\n")
            query = lines[-1].split()[0]
            if len(lines) == 4 and lines[2].startswith(query + " chooses"):
                tokens.append(999)
            return tokens

    changed = original.copy()
    for index, item in enumerate(changed):
        if item.metadata["normalized_query_position"] == "last":
            changed[index] = replace(
                item,
                metadata=item.metadata
                | {"prompt_token_length": item.metadata["prompt_token_length"] + 1},
            )
    with pytest.raises(ValueError, match="unequal prompt token lengths"):
        validate_matched_dataset(changed, PositionLengthTokenizer())


def test_paired_bootstrap_uses_complete_families_and_is_deterministic() -> None:
    examples = generate_matched_position_dataset(FakeTokenizer())[:6]
    records = FakeAdapter().score(examples, 8)
    result = paired_position_effects(records, seed=7, samples=100)
    assert result == paired_position_effects(records, seed=7, samples=100)
    assert result["complete_family_count"] == 2
    comparison = result["comparisons"]["first_minus_last_clean_logit_difference"]
    assert comparison["resampling_unit"] == "matched_family_id"
    assert comparison["family_count"] == 2
    transitions = result["paired_correctness_transitions"]["first_to_last"]
    assert set(transitions) == {
        "clean_correct",
        "corrupt_correct",
        "joint_clean_corrupt_state",
    }


def test_scoring_identity_rejects_duplicate_missing_and_wrong_identity() -> None:
    examples = generate_matched_position_dataset(FakeTokenizer())
    records = FakeAdapter().score(examples, 8)
    validate_scoring_identity(examples, records)
    with pytest.raises(ValueError, match="not unique"):
        validate_scoring_identity(examples, records[:-1] + [records[0]])
    with pytest.raises(ValueError, match="exactly match"):
        validate_scoring_identity(examples, records[:-1])
    wrong = records.copy()
    wrong[0] = replace(wrong[0], family_id="wrong")
    with pytest.raises(ValueError, match="family ID"):
        validate_scoring_identity(examples, wrong)


def test_decision_boundaries_and_processing_failure() -> None:
    metrics = {
        position: {
            "clean_pairwise_accuracy": 0.8,
            "mean_clean_logit_difference": 1.0,
            "example_count": 120,
            "failed_count": 0,
        }
        for position in ("first", "interior", "last")
    }
    effects = {
        "complete_family_count": 120,
        "comparisons": {
            "first_minus_last_clean_logit_difference": {
                "estimate": 0.1,
                "ci_95": [0.0001, 0.2],
            }
        }
    }
    balance = {"all_invariants_passed": True}
    assert decision_status(metrics, effects, balance).startswith("QUERY_FIRST")
    for change in (
        {"clean_pairwise_accuracy": 0.7999},
        {"mean_clean_logit_difference": 0.9999},
        {"example_count": 119},
        {"failed_count": 1},
    ):
        changed = json.loads(json.dumps(metrics))
        changed["first"].update(change)
        assert decision_status(changed, effects, balance) == "POSITION_EFFECT_UNCONFIRMED"
    zero_ci = json.loads(json.dumps(effects))
    zero_ci["comparisons"]["first_minus_last_clean_logit_difference"]["ci_95"][0] = 0
    assert decision_status(metrics, zero_ci, balance) == "POSITION_EFFECT_UNCONFIRMED"
    incomplete = json.loads(json.dumps(effects))
    incomplete["complete_family_count"] = 119
    assert decision_status(metrics, incomplete, balance) == "POSITION_EFFECT_UNCONFIRMED"
    for position in ("interior", "last"):
        failed = json.loads(json.dumps(metrics))
        failed[position].update({"example_count": 119, "failed_count": 1})
        assert decision_status(failed, effects, balance) == "POSITION_EFFECT_UNCONFIRMED"


def test_offline_run_outputs_resume_force_and_no_heldout_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = Path.open
    original_builtin_open = builtins.open
    original_read_text = Path.read_text
    original_read = Path.read_bytes
    original_stat = Path.stat

    def reject(path: Path) -> None:
        if path.name in {"validation.jsonl", "test.jsonl"}:
            raise AssertionError(f"held-out access: {path}")

    def guarded_open(path: Path, *items: object, **kwargs: object):
        reject(path)
        return original_open(path, *items, **kwargs)

    def guarded_builtin_open(file: object, *items: object, **kwargs: object):
        reject(Path(file))
        return original_builtin_open(file, *items, **kwargs)

    def guarded_read_text(path: Path, *items: object, **kwargs: object) -> str:
        reject(path)
        return original_read_text(path, *items, **kwargs)

    def guarded_read(path: Path) -> bytes:
        reject(path)
        return original_read(path)

    def guarded_stat(path: Path, *items: object, **kwargs: object):
        reject(path)
        return original_stat(path, *items, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    final = pipeline.run_position_study(args(tmp_path), lambda *unused: FakeAdapter())
    assert final["software_success"] and not final["held_out_splits_opened"]
    root = tmp_path / "seed-42-main"
    expected = {
        "run_manifest.json",
        "matched_dataset.jsonl",
        "examples.jsonl",
        "balance_report.json",
        "position_metrics.json",
        "paired_position_effects.json",
        "secondary_summaries.json",
        "position_study.md",
        "final_status.json",
    }
    assert expected <= {path.name for path in root.iterdir()}
    for path in root.glob("*.json"):
        assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()
    resumed = args(tmp_path, resume=True)
    assert pipeline.run_position_study(
        resumed, lambda *unused: (_ for _ in ()).throw(AssertionError("model reloaded"))
    ) == final
    dataset = root / "matched_dataset.jsonl"
    dataset.write_bytes(dataset.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="hash verification"):
        pipeline.run_position_study(resumed, lambda *unused: FakeAdapter())
    pipeline.run_position_study(args(tmp_path, force=True), lambda *unused: FakeAdapter())
    summaries = secondary_summaries(
        FakeAdapter().score(generate_matched_position_dataset(FakeTokenizer()), 8)
    )
    assert summaries["role"] == "secondary_descriptive_only"
    assert set(summaries) == {"role", "target_token", "distractor_token", "query_entity"}


@pytest.mark.parametrize("failed_position", ["first", "interior", "last"])
def test_failed_records_prevent_end_to_end_eligibility(
    tmp_path: Path, failed_position: str
) -> None:
    class FailingAdapter(FakeAdapter):
        def score(self, examples: list[ExamplePair], batch_size: int):
            records = super().score(examples, batch_size)
            index = next(
                i
                for i, item in enumerate(examples)
                if item.metadata["normalized_query_position"] == failed_position
            )
            records[index] = make_failed_result(examples[index], "failure")
            return records

    final = pipeline.run_position_study(args(tmp_path), lambda *unused: FailingAdapter())
    assert final["status"] == "POSITION_EFFECT_UNCONFIRMED"


def test_missing_last_result_is_software_integrity_failure(tmp_path: Path) -> None:
    class MissingAdapter(FakeAdapter):
        def score(self, examples: list[ExamplePair], batch_size: int):
            return [
                record
                for record in super().score(examples, batch_size)
                if record.normalized_query_position != "last"
                or record.family_id != examples[0].family_id
            ]

    with pytest.raises(ValueError, match="exactly match"):
        pipeline.run_position_study(args(tmp_path), lambda *unused: MissingAdapter())
