"""Parametric, paired associative-recall data generation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autocircuit.config import MVPConfig
from autocircuit.datasets.validation import (
    DatasetValidationError,
    Tokenizer,
    validate_answer_tokens,
    validate_split_leakage,
)

GENERATOR_VERSION = "1.0.0"
V2_GENERATOR_VERSION = "2.0.0"
TEMPLATES = (("likes", "."), ("prefers", "."), ("chooses", "."))
ENTITIES = (
    ("Alice", "Aaron", "Abel", "Ada", "Aiden", "Alan", "Amy", "Anna", "April", "Ava"),
    ("Ben", "Bella", "Bill", "Blake", "Bob", "Brent", "Brian", "Brooke", "Bruce", "Bryan"),
    ("Cara", "Carl", "Carol", "Chad", "Chloe", "Chris", "Clara", "Clark", "Cody", "Cora"),
)
VALUES = (
    (
        " apples",
        " art",
        " birds",
        " bread",
        " chess",
        " coffee",
        " dance",
        " films",
        " fruit",
        " games",
        " hats",
        " jazz",
    ),
    (
        " books",
        " cake",
        " cats",
        " drums",
        " flowers",
        " gold",
        " maps",
        " music",
        " pasta",
        " poems",
        " tea",
        " trains",
    ),
    (
        " boats",
        " cards",
        " dogs",
        " eggs",
        " history",
        " lakes",
        " math",
        " pizza",
        " rocks",
        " songs",
        " stars",
        " tools",
    ),
)
SPLITS = ("discovery", "validation", "test")


@dataclass(frozen=True)
class ExamplePair:
    example_id: str
    family_id: str
    split: str
    clean_prompt: str
    corrupt_prompt: str
    target_text: str
    distractor_text: str
    target_token_id: int
    distractor_token_id: int
    changed_factor: str
    seed: int
    template_id: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExamplePair:
        return cls(**value)


@dataclass(frozen=True)
class GenerationParameters:
    """Population-level controls allowed in the preregistered v2 search."""

    allowed_templates: tuple[str, ...] = ("likes", "prefers", "chooses")
    minimum_fact_count: int = 3
    maximum_fact_count: int = 6
    allowed_query_positions: tuple[str, ...] = ("first", "interior", "last")
    relation_mode: str = "sampled"
    separator: str = "newline"

    def validate(self) -> None:
        template_names = {template for template, _ in TEMPLATES}
        unknown = set(self.allowed_templates) - template_names
        if unknown:
            raise ValueError(f"unknown allowed template(s): {sorted(unknown)}")
        if not self.allowed_templates:
            raise ValueError("allowed_templates cannot be empty")
        if self.relation_mode not in {"fixed", "sampled"}:
            raise ValueError("relation_mode must be 'fixed' or 'sampled'")
        if self.relation_mode == "fixed" and len(self.allowed_templates) != 1:
            raise ValueError("fixed relation_mode requires exactly one allowed template")
        if self.separator not in {"newline", "blank_line"}:
            raise ValueError("separator must be 'newline' or 'blank_line'")
        positions = {"first", "interior", "last"}
        if not self.allowed_query_positions:
            raise ValueError("allowed_query_positions cannot be empty")
        unknown_positions = set(self.allowed_query_positions) - positions
        if unknown_positions:
            raise ValueError(f"unknown normalized query position(s): {sorted(unknown_positions)}")
        if self.minimum_fact_count < 3:
            raise ValueError("minimum_fact_count must be at least 3")
        if self.maximum_fact_count < self.minimum_fact_count:
            raise ValueError("maximum_fact_count must be >= minimum_fact_count")
        if self.maximum_fact_count > min(len(group) for group in ENTITIES):
            raise ValueError("maximum_fact_count exceeds available entity population")


def _prompt(
    assignments: list[tuple[str, str]], query: str, relation: str, stop: str, separator: str
) -> str:
    joiner = {"newline": "\n", "blank_line": "\n\n"}[separator]
    facts = joiner.join(f"{key} {relation}{value}{stop}" for key, value in assignments)
    return f"{facts}{joiner}{query} {relation}"


def normalized_query_position(index: int, fact_count: int) -> str:
    if not 0 <= index < fact_count:
        raise ValueError("query index must be within the presented facts")
    if index == 0:
        return "first"
    if index == fact_count - 1:
        return "last"
    return "interior"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_split(
    split: str,
    count: int,
    seed: int,
    tokenizer: Tokenizer,
    *,
    max_attempts: int | None = None,
    parameters: GenerationParameters | None = None,
    seed_namespace: str = "legacy-v1",
) -> tuple[list[ExamplePair], dict[str, int]]:
    if split not in SPLITS or count <= 0:
        raise ValueError("unknown split or non-positive count")
    index = SPLITS.index(split)
    parameters = parameters or GenerationParameters()
    parameters.validate()
    seed_material = (
        f"{seed}:{split}:{GENERATOR_VERSION}"
        if seed_namespace == "legacy-v1"
        else f"{seed_namespace}:{seed}:{split}:{GENERATOR_VERSION}"
    )
    rng = random.Random(seed_material)
    rejected: dict[str, int] = {}
    results: list[ExamplePair] = []
    seen: set[str] = set()
    limit = max_attempts or count * 100
    for attempt in range(limit):
        fact_count = rng.randint(parameters.minimum_fact_count, parameters.maximum_fact_count)
        entities = rng.sample(ENTITIES[index], fact_count)
        values = rng.sample(VALUES[index], fact_count)
        assignments = list(zip(entities, values, strict=True))
        rng.shuffle(assignments)
        allowed_indices = [
            i
            for i in range(fact_count)
            if normalized_query_position(i, fact_count) in parameters.allowed_query_positions
        ]
        query_index = rng.choice(allowed_indices)
        distractor_index = rng.choice([i for i in range(fact_count) if i != query_index])
        query, target = assignments[query_index]
        distractor = assignments[distractor_index][1]
        choices = [item for item in TEMPLATES if item[0] in parameters.allowed_templates]
        relation, stop = choices[0] if parameters.relation_mode == "fixed" else rng.choice(choices)
        family_key = tuple(sorted(assignments))
        family_id = _digest(family_key)[:16]
        if family_id in seen:
            rejected["duplicate_family"] = rejected.get("duplicate_family", 0) + 1
            continue
        try:
            target_id, distractor_id = validate_answer_tokens(tokenizer, target, distractor)
        except DatasetValidationError as exc:
            reason = "token_validation:" + str(exc)
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        corrupt = assignments.copy()
        corrupt[query_index] = (query, distractor)
        other_key = corrupt[distractor_index][0]
        corrupt[distractor_index] = (other_key, target)
        clean_prompt = _prompt(assignments, query, relation, stop, parameters.separator)
        corrupt_prompt = _prompt(corrupt, query, relation, stop, parameters.separator)
        clean_length = len(tokenizer.encode(clean_prompt, add_special_tokens=False))
        corrupt_length = len(tokenizer.encode(corrupt_prompt, add_special_tokens=False))
        if clean_length != corrupt_length:
            rejected["prompt_token_length_mismatch"] = (
                rejected.get("prompt_token_length_mismatch", 0) + 1
            )
            continue
        seen.add(family_id)
        identity = _digest([seed, split, attempt, family_id])[:20]
        metadata: dict[str, Any] = {
            "assignments": [list(pair) for pair in assignments],
            "query_entity": query,
            "fact_count": fact_count,
            "query_position": "final_next_token",
            "prompt_token_length": clean_length,
        }
        if seed_namespace != "legacy-v1":
            metadata |= {
                "query_fact_index": query_index,
                "normalized_query_position": normalized_query_position(query_index, fact_count),
            }
        results.append(
            ExamplePair(
                example_id=f"ar-{identity}",
                family_id=family_id,
                split=split,
                clean_prompt=clean_prompt,
                corrupt_prompt=corrupt_prompt,
                target_text=target,
                distractor_text=distractor,
                target_token_id=target_id,
                distractor_token_id=distractor_id,
                changed_factor="queried_value_swap",
                seed=seed,
                template_id=f"{relation}-v1",
                metadata=metadata,
            )
        )
        if len(results) == count:
            return results, rejected
    raise DatasetValidationError(
        f"could not generate {count} valid {split} examples after {limit} attempts; "
        f"generated={len(results)}, rejected={rejected}"
    )


def generate_dataset(
    config: MVPConfig, tokenizer: Tokenizer
) -> tuple[list[ExamplePair], dict[str, int]]:
    sizes = (config.discovery_examples, config.validation_examples, config.test_examples)
    all_examples: list[ExamplePair] = []
    rejected: dict[str, int] = {}
    for split, size in zip(SPLITS, sizes, strict=True):
        examples, split_rejected = generate_split(split, size, config.seed, tokenizer)
        all_examples.extend(examples)
        for reason, number in split_rejected.items():
            rejected[reason] = rejected.get(reason, 0) + number
    leakage = validate_split_leakage(all_examples)
    if not leakage.passed:
        raise DatasetValidationError(f"split leakage detected: {leakage.reasons}")
    return all_examples, rejected


def write_jsonl(path: Path, examples: list[ExamplePair]) -> str:
    content = "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in examples)
    payload = content.encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[ExamplePair]:
    return [ExamplePair.from_dict(json.loads(line)) for line in path.read_text().splitlines()]
