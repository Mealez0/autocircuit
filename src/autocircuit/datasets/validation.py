"""Tokenizer and split validation for paired datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class Tokenizer(Protocol):
    name_or_path: str

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class DatasetValidationError(ValueError):
    """Raised when an example violates the frozen data contract."""


def single_token_id(tokenizer: Tokenizer, text: str) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) != 1:
        raise DatasetValidationError(f"expected one token for {text!r}, got {len(tokens)}")
    return tokens[0]


def validate_answer_tokens(tokenizer: Tokenizer, target: str, distractor: str) -> tuple[int, int]:
    target_id = single_token_id(tokenizer, target)
    distractor_id = single_token_id(tokenizer, distractor)
    if target_id == distractor_id:
        raise DatasetValidationError("target and distractor have the same token id")
    return target_id, distractor_id


@dataclass(frozen=True)
class LeakageResult:
    passed: bool
    reasons: dict[str, int]


def validate_split_leakage(examples: Sequence[Any]) -> LeakageResult:
    """Check cross-split uniqueness using fields exposed by ExamplePair."""
    indexes: dict[str, dict[object, str]] = {
        "prompt": {},
        "assignment": {},
        "family_id": {},
        "entity_value": {},
    }
    failures: Counter[str] = Counter()
    for item in examples:
        split = item.split
        metadata = item.metadata
        checks = {
            "prompt": (item.clean_prompt, item.corrupt_prompt),
            "assignment": tuple(sorted(tuple(pair) for pair in metadata["assignments"])),
            "family_id": item.family_id,
        }
        for name, key in checks.items():
            previous = indexes[name].setdefault(key, split)
            if previous != split:
                failures[name] += 1
        for pair in metadata["assignments"]:
            key = tuple(pair)
            previous = indexes["entity_value"].setdefault(key, split)
            if previous != split:
                failures["entity_value"] += 1
    return LeakageResult(not failures, dict(sorted(failures.items())))
