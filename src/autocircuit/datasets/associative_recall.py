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


def _prompt(assignments: list[tuple[str, str]], query: str, relation: str, stop: str) -> str:
    facts = "\n".join(f"{key} {relation}{value}{stop}" for key, value in assignments)
    return f"{facts}\n{query} {relation}"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_split(
    split: str, count: int, seed: int, tokenizer: Tokenizer, *, max_attempts: int | None = None
) -> tuple[list[ExamplePair], dict[str, int]]:
    if split not in SPLITS or count <= 0:
        raise ValueError("unknown split or non-positive count")
    index = SPLITS.index(split)
    rng = random.Random(f"{seed}:{split}:{GENERATOR_VERSION}")
    rejected: dict[str, int] = {}
    results: list[ExamplePair] = []
    seen: set[str] = set()
    limit = max_attempts or count * 100
    for attempt in range(limit):
        fact_count = rng.randint(3, 6)
        entities = rng.sample(ENTITIES[index], fact_count)
        values = rng.sample(VALUES[index], fact_count)
        assignments = list(zip(entities, values, strict=True))
        rng.shuffle(assignments)
        query_index = rng.randrange(fact_count)
        distractor_index = rng.choice([i for i in range(fact_count) if i != query_index])
        query, target = assignments[query_index]
        distractor = assignments[distractor_index][1]
        relation, stop = rng.choice(TEMPLATES)
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
        clean_prompt = _prompt(assignments, query, relation, stop)
        corrupt_prompt = _prompt(corrupt, query, relation, stop)
        clean_length = len(tokenizer.encode(clean_prompt, add_special_tokens=False))
        corrupt_length = len(tokenizer.encode(corrupt_prompt, add_special_tokens=False))
        if clean_length != corrupt_length:
            rejected["prompt_token_length_mismatch"] = (
                rejected.get("prompt_token_length_mismatch", 0) + 1
            )
            continue
        seen.add(family_id)
        identity = _digest([seed, split, attempt, family_id])[:20]
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
                metadata={
                    "assignments": [list(pair) for pair in assignments],
                    "query_entity": query,
                    "fact_count": fact_count,
                    "query_position": "final_next_token",
                    "prompt_token_length": clean_length,
                },
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
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def read_jsonl(path: Path) -> list[ExamplePair]:
    return [ExamplePair.from_dict(json.loads(line)) for line in path.read_text().splitlines()]
