from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from autocircuit.config import MVPConfig
from autocircuit.datasets.associative_recall import (
    GENERATOR_VERSION,
    V2_GENERATOR_VERSION,
    VALUES,
    GenerationParameters,
    generate_dataset,
    generate_split,
    generation_seed_material,
    read_jsonl,
    write_jsonl,
)
from autocircuit.datasets.validation import (
    DatasetValidationError,
    validate_answer_tokens,
    validate_split_leakage,
)


class FakeTokenizer:
    name_or_path = "fake-pythia"

    def __init__(self, invalid: bool = False) -> None:
        self.invalid = invalid
        self.mapping = {
            value: index
            for index, value in enumerate(sum((list(group) for group in VALUES), []), 10)
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if "\n" in text:
            return list(range(len(text.split())))
        if self.invalid or text not in self.mapping:
            return [1, 2]
        return [self.mapping[text]]


def small_config(seed: int = 42) -> MVPConfig:
    return MVPConfig("pythia-70m", seed, 12, 7, 5)


def test_same_seed_is_byte_deterministic(tmp_path: Path) -> None:
    first, _ = generate_dataset(small_config(), FakeTokenizer())
    second, _ = generate_dataset(small_config(), FakeTokenizer())
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    assert write_jsonl(one, first) == write_jsonl(two, second)
    assert one.read_bytes() == two.read_bytes()


def test_legacy_v1_golden_hash_is_unchanged(tmp_path: Path) -> None:
    assert GENERATOR_VERSION == "1.0.0"
    examples, _ = generate_split("discovery", 12, 42, FakeTokenizer())
    assert write_jsonl(tmp_path / "golden.jsonl", examples) == (
        "ed2306d041b4cf31f528815d02c82ef082e1ad4c29a080ab2b4ca0a2f0e01675"
    )


def test_v2_seed_material_uses_v2_generator_version(monkeypatch: pytest.MonkeyPatch) -> None:
    assert V2_GENERATOR_VERSION == "2.0.0"
    original = generation_seed_material(42, "discovery", "candidate:control")
    assert original.endswith(":2.0.0")
    original_examples, _ = generate_split(
        "discovery", 4, 42, FakeTokenizer(), seed_namespace="candidate:control"
    )
    monkeypatch.setattr("autocircuit.datasets.associative_recall.V2_GENERATOR_VERSION", "2.0.1")
    assert generation_seed_material(42, "discovery", "candidate:control") != original
    changed_examples, _ = generate_split(
        "discovery", 4, 42, FakeTokenizer(), seed_namespace="candidate:control"
    )
    assert [item.clean_prompt for item in changed_examples] != [
        item.clean_prompt for item in original_examples
    ]


def test_different_seed_changes_generation() -> None:
    first, _ = generate_dataset(small_config(1), FakeTokenizer())
    second, _ = generate_dataset(small_config(2), FakeTokenizer())
    assert [item.clean_prompt for item in first] != [item.clean_prompt for item in second]


def test_sizes_and_no_split_leakage() -> None:
    examples, _ = generate_dataset(small_config(), FakeTokenizer())
    assert [
        sum(item.split == split for item in examples)
        for split in ("discovery", "validation", "test")
    ] == [12, 7, 5]
    assert validate_split_leakage(examples).passed


def test_clean_corrupt_are_position_matched_value_swap() -> None:
    examples, _ = generate_split("discovery", 8, 42, FakeTokenizer())
    for item in examples:
        clean_lines = item.clean_prompt.splitlines()
        corrupt_lines = item.corrupt_prompt.splitlines()
        assert len(clean_lines) == len(corrupt_lines)
        assert clean_lines[-1] == corrupt_lines[-1]
        assert sum(a != b for a, b in zip(clean_lines, corrupt_lines, strict=True)) == 2
        assert item.target_text != item.distractor_text
        assert item.target_token_id != item.distractor_token_id
        assert item.metadata["query_position"] == "final_next_token"


def test_single_token_validation() -> None:
    tokenizer = FakeTokenizer()
    target, distractor = VALUES[0][:2]
    assert validate_answer_tokens(tokenizer, target, distractor) == (10, 11)
    with pytest.raises(DatasetValidationError, match="expected one token"):
        validate_answer_tokens(FakeTokenizer(invalid=True), target, distractor)
    tokenizer.mapping[distractor] = tokenizer.mapping[target]
    with pytest.raises(DatasetValidationError, match="same token id"):
        validate_answer_tokens(tokenizer, target, distractor)


def test_insufficient_vocabulary_has_actionable_error() -> None:
    with pytest.raises(DatasetValidationError, match="could not generate"):
        generate_split("discovery", 1, 42, FakeTokenizer(invalid=True), max_attempts=3)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (GenerationParameters(allowed_templates=("unknown",)), "unknown allowed template"),
        (GenerationParameters(relation_mode="other"), "relation_mode"),
        (GenerationParameters(separator="spaces"), "separator"),
        (GenerationParameters(allowed_query_positions=("middle",)), "query position"),
        (GenerationParameters(minimum_fact_count=2), "at least 3"),
    ],
)
def test_v2_generation_parameters_are_strictly_validated(
    parameters: GenerationParameters, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_split("discovery", 1, 42, FakeTokenizer(), parameters=parameters)


def test_jsonl_round_trip_and_hash(tmp_path: Path) -> None:
    examples, _ = generate_split("test", 3, 9, FakeTokenizer())
    path = tmp_path / "test.jsonl"
    digest = write_jsonl(path, examples)
    assert read_jsonl(path) == examples
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    manifest = {"file_hashes": {path.name: digest}}
    encoded = json.dumps(manifest, sort_keys=True)
    assert json.loads(encoded)["file_hashes"][path.name] == digest


def test_leakage_validator_detects_copied_family() -> None:
    examples, _ = generate_split("discovery", 1, 42, FakeTokenizer())
    copied = replace(examples[0], split="test", example_id="copy")
    result = validate_split_leakage([examples[0], copied])
    assert not result.passed
    assert set(result.reasons) == {"assignment", "entity_value", "family_id", "prompt"}
