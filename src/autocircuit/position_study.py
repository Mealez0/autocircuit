"""Preregistered discovery-only matched query-position study."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from autocircuit.baseline import ExampleResult, successful_difference
from autocircuit.datasets.associative_recall import ENTITIES, VALUES, ExamplePair
from autocircuit.datasets.validation import Tokenizer, validate_answer_tokens

GENERATOR_VERSION = "position-study-1.0.0"
FAMILY_COUNT = 120
POSITIONS = ("first", "interior", "last")
BOOTSTRAP_SAMPLES = 10_000


def _prompt(assignments: list[tuple[str, str]], query: str) -> str:
    return "\n".join(f"{entity} chooses{value}." for entity, value in assignments) + (
        f"\n{query} chooses"
    )


def generate_matched_position_dataset(
    tokenizer: Tokenizer, seed: int = 42
) -> list[ExamplePair]:
    """Generate 120 balanced families, each presented at all three positions."""
    values = VALUES[0]
    entities = ENTITIES[0]
    examples: list[ExamplePair] = []
    for family_number in range(FAMILY_COUNT):
        target_index = family_number % len(values)
        query_index = (family_number // len(values)) % len(entities)
        # Ten distinct nonzero cyclic offsets give unique, balanced ordered pairs.
        distractor_index = (target_index + query_index + 1) % len(values)
        third_index = (target_index + query_index + 2) % len(values)
        if third_index == distractor_index:
            third_index = (third_index + 1) % len(values)
        query = entities[query_index]
        other_entities = [
            entities[(query_index + 1) % len(entities)],
            entities[(query_index + 2) % len(entities)],
        ]
        target, distractor = values[target_index], values[distractor_index]
        assignments = [
            (query, target),
            (other_entities[0], distractor),
            (other_entities[1], values[third_index]),
        ]
        corrupt = [
            (query, distractor),
            (other_entities[0], target),
            (other_entities[1], values[third_index]),
        ]
        target_id, distractor_id = validate_answer_tokens(tokenizer, target, distractor)
        matched_id = f"position-family-{family_number:03d}"
        orders = ((0, 1, 2), (1, 0, 2), (1, 2, 0))
        for position_index, (position, order) in enumerate(zip(POSITIONS, orders, strict=True)):
            clean_ordered = [assignments[index] for index in order]
            corrupt_ordered = [corrupt[index] for index in order]
            clean_prompt = _prompt(clean_ordered, query)
            corrupt_prompt = _prompt(corrupt_ordered, query)
            clean_length = len(tokenizer.encode(clean_prompt, add_special_tokens=False))
            corrupt_length = len(tokenizer.encode(corrupt_prompt, add_special_tokens=False))
            if clean_length != corrupt_length:
                raise ValueError(f"clean/corrupt token-length mismatch in {matched_id}/{position}")
            examples.append(
                ExamplePair(
                    example_id=f"{matched_id}-{position}",
                    family_id=matched_id,
                    split="discovery",
                    clean_prompt=clean_prompt,
                    corrupt_prompt=corrupt_prompt,
                    target_text=target,
                    distractor_text=distractor,
                    target_token_id=target_id,
                    distractor_token_id=distractor_id,
                    changed_factor="queried_value_swap",
                    seed=seed,
                    template_id="chooses-position-study-v1",
                    metadata={
                        "assignments": [list(item) for item in assignments],
                        "corrupt_assignments": [list(item) for item in corrupt],
                        "matched_family_id": matched_id,
                        "query_entity": query,
                        "fact_count": 3,
                        "query_fact_index": position_index,
                        "normalized_query_position": position,
                        "prompt_token_length": clean_length,
                        "generator_version": GENERATOR_VERSION,
                        "ordering": list(order),
                    },
                )
            )
    validate_matched_dataset(examples)
    return examples


def validate_matched_dataset(examples: list[ExamplePair]) -> dict[str, Any]:
    """Fail closed on every preregistered population and within-family invariant."""
    if len(examples) != FAMILY_COUNT * len(POSITIONS):
        raise ValueError("expected exactly 360 examples")
    families: dict[str, list[ExamplePair]] = defaultdict(list)
    for item in examples:
        if item.split != "discovery":
            raise ValueError("position study may only contain discovery examples")
        families[item.family_id].append(item)
    if len(families) != FAMILY_COUNT:
        raise ValueError("expected exactly 120 matched families")
    targets: Counter[str] = Counter()
    distractors: Counter[str] = Counter()
    queries: Counter[str] = Counter()
    position_marginals: dict[str, Counter[tuple[str, str, str]]] = {
        position: Counter() for position in POSITIONS
    }
    ordered_pairs: Counter[tuple[str, str]] = Counter()
    for family_id, variants in families.items():
        by_position = {str(item.metadata["normalized_query_position"]): item for item in variants}
        if set(by_position) != set(POSITIONS) or len(variants) != 3:
            raise ValueError(f"family {family_id} lacks exactly one variant per position")
        first = variants[0]
        invariant = (
            first.metadata["assignments"],
            first.metadata["corrupt_assignments"],
            first.metadata["query_entity"],
            first.target_text,
            first.distractor_text,
            first.target_token_id,
            first.distractor_token_id,
        )
        for position, item in by_position.items():
            candidate = (
                item.metadata["assignments"],
                item.metadata["corrupt_assignments"],
                item.metadata["query_entity"],
                item.target_text,
                item.distractor_text,
                item.target_token_id,
                item.distractor_token_id,
            )
            if candidate != invariant or item.metadata["matched_family_id"] != family_id:
                raise ValueError(f"family matching invariant failed for {family_id}")
            if item.target_text == item.distractor_text:
                raise ValueError("target equals distractor")
            if item.metadata["prompt_token_length"] != len(
                item.metadata.get("prompt_token_ids", [])
            ) and "prompt_token_ids" in item.metadata:
                raise ValueError("recorded prompt token length is invalid")
            position_marginals[position][
                (item.target_text, item.distractor_text, str(item.metadata["query_entity"]))
            ] += 1
        targets[first.target_text] += 1
        distractors[first.distractor_text] += 1
        queries[str(first.metadata["query_entity"])] += 1
        ordered_pairs[(first.target_text, first.distractor_text)] += 1
    if set(targets) != set(VALUES[0]) or len(set(targets.values())) != 1:
        raise ValueError("target balance failed")
    if set(distractors) != set(VALUES[0]) or len(set(distractors.values())) != 1:
        raise ValueError("distractor balance failed")
    if set(queries) != set(ENTITIES[0]) or len(set(queries.values())) != 1:
        raise ValueError("query entity balance failed")
    if len({tuple(sorted(counter.items())) for counter in position_marginals.values()}) != 1:
        raise ValueError("position lexical marginals differ")
    if max(ordered_pairs.values()) - min(ordered_pairs.values()) > 1:
        raise ValueError("ordered pairs are not balanced as possible")
    return {
        "all_invariants_passed": True,
        "generator_version": GENERATOR_VERSION,
        "example_count": len(examples),
        "matched_family_count": len(families),
        "target_counts": dict(sorted(targets.items())),
        "distractor_counts": dict(sorted(distractors.items())),
        "query_entity_counts": dict(sorted(queries.items())),
        "ordered_pair_minimum": min(ordered_pairs.values()),
        "ordered_pair_maximum": max(ordered_pairs.values()),
        "positions": list(POSITIONS),
    }


def position_metrics(records: list[ExampleResult]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for position in POSITIONS:
        selected = [r for r in records if r.normalized_query_position == position]
        valid = [r for r in selected if r.processing_status == "ok"]
        clean = [
            float(r.clean_logit_difference)
            for r in valid
            if r.clean_logit_difference is not None
        ]
        corrupt = [
            float(r.corrupt_logit_difference)
            for r in valid
            if r.corrupt_logit_difference is not None
        ]
        if valid:
            clean_mean, corrupt_mean = sum(clean) / len(clean), sum(corrupt) / len(corrupt)
            output[position] = {
                "clean_pairwise_accuracy": sum(value > 0 for value in clean) / len(clean),
                "corrupt_pairwise_accuracy": sum(value < 0 for value in corrupt) / len(corrupt),
                "mean_clean_logit_difference": clean_mean,
                "mean_corrupt_logit_difference": corrupt_mean,
                "clean_corrupt_contrast": clean_mean - corrupt_mean,
                "joint_success_rate": sum(
                    bool(r.clean_correct and r.corrupt_correct) for r in valid
                )
                / len(valid),
                "example_count": len(valid),
                "failed_count": len(selected) - len(valid),
            }
        else:
            output[position] = {"example_count": 0, "failed_count": len(selected)}
    return output


def paired_position_effects(
    records: list[ExampleResult], seed: int = 42, samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    """Bootstrap complete matched families, never individual variants."""
    grouped: dict[str, dict[str, ExampleResult]] = defaultdict(dict)
    for record in records:
        if record.processing_status == "ok":
            grouped[record.family_id][record.normalized_query_position] = record
    complete = {key: value for key, value in grouped.items() if set(value) == set(POSITIONS)}
    family_ids = sorted(complete)
    comparisons: dict[str, Any] = {}
    rng = random.Random(f"position-study-bootstrap:{seed}")
    specs = [("first", "interior"), ("first", "last"), ("interior", "last")]
    for metric_name in ("clean_logit_difference", "contrast"):
        for left, right in specs:
            differences = []
            for family_id in family_ids:
                a, b = complete[family_id][left], complete[family_id][right]
                if metric_name == "clean_logit_difference":
                    av = successful_difference(a.clean_logit_difference)
                    bv = successful_difference(b.clean_logit_difference)
                else:
                    av = successful_difference(
                        a.clean_logit_difference
                    ) - successful_difference(a.corrupt_logit_difference)
                    bv = successful_difference(
                        b.clean_logit_difference
                    ) - successful_difference(b.corrupt_logit_difference)
                differences.append(av - bv)
            key = f"{left}_minus_{right}_{metric_name}"
            if not differences:
                comparisons[key] = {"estimate": None, "ci_95": None, "family_count": 0}
                continue
            boot = sorted(
                sum(differences[rng.randrange(len(differences))] for _ in differences)
                / len(differences)
                for _ in range(samples)
            )
            comparisons[key] = {
                "estimate": sum(differences) / len(differences),
                "ci_95": [boot[int(samples * 0.025)], boot[int(samples * 0.975) - 1]],
                "family_count": len(differences),
                "resampling_unit": "matched_family_id",
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
            }
    transitions: dict[str, Counter[str]] = {}
    for left, right in specs:
        counter: Counter[str] = Counter()
        for variants in complete.values():
            a, b = variants[left], variants[right]
            counter[f"{bool(a.clean_correct)}->{bool(b.clean_correct)}"] += 1
        transitions[f"{left}_to_{right}_clean_correct"] = counter
    return {
        "complete_family_count": len(complete),
        "comparisons": comparisons,
        "paired_correctness_transitions": {
            key: dict(sorted(value.items())) for key, value in transitions.items()
        },
    }


def decision_status(
    metrics: dict[str, Any], effects: dict[str, Any], balance: dict[str, Any]
) -> str:
    first = metrics["first"]
    comparison = effects["comparisons"]["first_minus_last_clean_logit_difference"]
    ci = comparison["ci_95"]
    eligible = (
        first.get("clean_pairwise_accuracy", 0.0) >= 0.80
        and first.get("mean_clean_logit_difference", -math.inf) >= 1.0
        and first["failed_count"] == 0
        and first["example_count"] == FAMILY_COUNT
        and comparison["estimate"] is not None
        and comparison["estimate"] > 0
        and ci is not None
        and ci[0] > 0
        and balance.get("all_invariants_passed") is True
    )
    return (
        "QUERY_FIRST_DISCOVERY_ELIGIBLE_REQUIRES_HELD_OUT_VALIDATION"
        if eligible
        else "POSITION_EFFECT_UNCONFIRMED"
    )


def json_has_only_finite_numbers(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(json_has_only_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(json_has_only_finite_numbers(item) for item in value)
    return True


def write_dataset(path: Path, examples: list[ExamplePair]) -> None:
    payload = "".join(
        json.dumps(asdict(item), sort_keys=True, separators=(",", ":")) + "\n"
        for item in examples
    )
    path.write_bytes(payload.encode())
