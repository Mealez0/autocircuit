"""Preregistered discovery-only matched query-position study."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
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
    rng = random.Random(f"{GENERATOR_VERSION}:seed:{seed}")
    values = list(VALUES[0])
    entities = list(ENTITIES[0])
    offsets = list(range(1, len(values)))
    rng.shuffle(values)
    rng.shuffle(entities)
    rng.shuffle(offsets)
    offsets = offsets[: len(entities)]
    population = [
        (target, query)
        for query in range(len(entities))
        for target in range(len(values))
    ]
    rng.shuffle(population)
    examples: list[ExamplePair] = []
    for target_index, query_index in population:
        distractor_index = (target_index + offsets[query_index]) % len(values)
        third_index = next(
            index for index in range(len(values)) if index not in {target_index, distractor_index}
        )
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
        provenance = f"{GENERATOR_VERSION}:{seed}:{target_index}:{query_index}"
        matched_id = "position-family-" + hashlib.sha256(provenance.encode()).hexdigest()[:16]
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
    validate_matched_dataset(examples, tokenizer)
    return examples


def validate_matched_dataset(examples: list[ExamplePair], tokenizer: Tokenizer) -> dict[str, Any]:
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
        first = by_position["first"]
        if first.template_id != "chooses-position-study-v1":
            raise ValueError("relation template must be exactly chooses")
        assignments = [tuple(item) for item in first.metadata.get("assignments", [])]
        corrupt = [tuple(item) for item in first.metadata.get("corrupt_assignments", [])]
        if len(assignments) != 3 or int(first.metadata.get("fact_count", -1)) != 3:
            raise ValueError("fact count must be exactly 3")
        query = str(first.metadata.get("query_entity"))
        if assignments[0] != (query, first.target_text):
            raise ValueError("canonical query assignment does not match target")
        distractor_indices = [
            i for i, item in enumerate(assignments) if item[1] == first.distractor_text
        ]
        if len(distractor_indices) != 1 or distractor_indices[0] == 0:
            raise ValueError("canonical distractor assignment is invalid")
        swap_index = distractor_indices[0]
        expected_corrupt = assignments.copy()
        expected_corrupt[0] = (query, first.distractor_text)
        expected_corrupt[swap_index] = (assignments[swap_index][0], first.target_text)
        if corrupt != expected_corrupt:
            raise ValueError("corrupt assignments are not the exact target/distractor swap")
        invariant_pair = asdict(first)
        invariant_metadata = invariant_pair.pop("metadata")
        permitted_pair = {"example_id", "clean_prompt", "corrupt_prompt"}
        permitted_metadata = {
            "ordering",
            "query_fact_index",
            "normalized_query_position",
            "prompt_token_length",
        }
        lengths: set[int] = set()
        for position, item in by_position.items():
            candidate_pair = asdict(item)
            candidate_metadata = candidate_pair.pop("metadata")
            if any(
                candidate_pair[key] != invariant_pair[key]
                for key in invariant_pair
                if key not in permitted_pair
            ):
                raise ValueError(f"invariant ExamplePair field changed in {family_id}")
            if any(
                candidate_metadata.get(key) != invariant_metadata.get(key)
                for key in set(candidate_metadata) | set(invariant_metadata)
                if key not in permitted_metadata
            ):
                raise ValueError(f"invariant metadata field changed in {family_id}")
            if item.metadata.get("matched_family_id") != family_id:
                raise ValueError(f"matched family ID mismatch for {family_id}")
            ordering = item.metadata.get("ordering")
            if not isinstance(ordering, list) or sorted(ordering) != [0, 1, 2]:
                raise ValueError("ordering metadata is not a permutation")
            if [index for index in ordering if index != 0] != [1, 2]:
                raise ValueError("non-query facts changed relative order")
            actual_index = ordering.index(0)
            actual_position = POSITIONS[actual_index]
            if item.metadata.get("query_fact_index") != actual_index or position != actual_position:
                raise ValueError("query position metadata disagrees with ordering")
            clean_ordered = [assignments[index] for index in ordering]
            corrupt_ordered = [corrupt[index] for index in ordering]
            if item.clean_prompt != _prompt(clean_ordered, query):
                raise ValueError("clean prompt cannot be reconstructed exactly")
            if item.corrupt_prompt != _prompt(corrupt_ordered, query):
                raise ValueError("corrupt prompt cannot be reconstructed exactly")
            if item.target_text == item.distractor_text:
                raise ValueError("target equals distractor")
            target_id, distractor_id = validate_answer_tokens(
                tokenizer, item.target_text, item.distractor_text
            )
            if (target_id, distractor_id) != (item.target_token_id, item.distractor_token_id):
                raise ValueError("answer token IDs are invalid")
            clean_length = len(tokenizer.encode(item.clean_prompt, add_special_tokens=False))
            corrupt_length = len(tokenizer.encode(item.corrupt_prompt, add_special_tokens=False))
            if (
                clean_length != corrupt_length
                or item.metadata.get("prompt_token_length") != clean_length
            ):
                raise ValueError("clean/corrupt or recorded token lengths differ")
            lengths.add(clean_length)
            position_marginals[position][
                (item.target_text, item.distractor_text, str(item.metadata["query_entity"]))
            ] += 1
        if len(lengths) != 1:
            raise ValueError("position variants have unequal prompt token lengths")
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
    complete_pairs = {
        (target, distractor): ordered_pairs[(target, distractor)]
        for target in VALUES[0]
        for distractor in VALUES[0]
        if target != distractor
    }
    pair_values = list(complete_pairs.values())
    if Counter(pair_values) != Counter({0: 12, 1: 120}):
        raise ValueError("ordered pair population is not the exact balanced design")
    if max(pair_values) - min(pair_values) > 1:
        raise ValueError("ordered pairs are not balanced as possible")
    pair_mapping = {
        f"{target}|{distractor}": count
        for (target, distractor), count in complete_pairs.items()
    }
    return {
        "all_invariants_passed": True,
        "generator_version": GENERATOR_VERSION,
        "example_count": len(examples),
        "matched_family_count": len(families),
        "target_counts": dict(sorted(targets.items())),
        "distractor_counts": dict(sorted(distractors.items())),
        "query_entity_counts": dict(sorted(queries.items())),
        "ordered_pair_observed_count": sum(count > 0 for count in pair_values),
        "ordered_pair_missing_count": sum(count == 0 for count in pair_values),
        "ordered_pair_minimum_including_zero": min(pair_values),
        "ordered_pair_maximum": max(pair_values),
        "ordered_pair_counts": pair_mapping,
        "ordered_pair_counts_sha256": hashlib.sha256(
            json.dumps(pair_mapping, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
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


def validate_scoring_identity(examples: list[ExamplePair], records: list[ExampleResult]) -> None:
    """Require a one-to-one identity-preserving result for every expected example."""
    expected = {item.example_id: item for item in examples}
    if len(expected) != len(examples):
        raise ValueError("dataset example IDs are not unique")
    actual_ids = [record.example_id for record in records]
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("result example IDs are not unique")
    if set(actual_ids) != set(expected):
        raise ValueError("result example IDs do not exactly match the dataset")
    for record in records:
        item = expected[record.example_id]
        if record.family_id != item.family_id:
            raise ValueError(f"result family ID changed for {record.example_id}")
        if record.normalized_query_position != item.metadata["normalized_query_position"]:
            raise ValueError(f"result position changed for {record.example_id}")


def secondary_summaries(records: list[ExampleResult]) -> dict[str, Any]:
    """Compute descriptive lexical/entity performance; never used for eligibility."""
    dimensions: dict[str, Callable[[ExampleResult], str]] = {
        "target_token": lambda record: record.target_text,
        "distractor_token": lambda record: record.distractor_text,
        "query_entity": lambda record: record.query_entity,
    }
    output: dict[str, Any] = {"role": "secondary_descriptive_only"}
    for dimension, key_function in dimensions.items():
        grouped: dict[str, list[ExampleResult]] = defaultdict(list)
        for record in records:
            grouped[key_function(record)].append(record)
        summaries: dict[str, Any] = {}
        for key in sorted(grouped):
            selected = grouped[key]
            valid = [record for record in selected if record.processing_status == "ok"]
            if not valid:
                summaries[key] = {
                    "count": 0,
                    "failed_count": len(selected),
                    "clean_accuracy": None,
                    "corrupt_accuracy": None,
                    "mean_clean_logit_difference": None,
                    "mean_corrupt_logit_difference": None,
                    "clean_corrupt_contrast": None,
                }
                continue
            clean = [successful_difference(record.clean_logit_difference) for record in valid]
            corrupt = [successful_difference(record.corrupt_logit_difference) for record in valid]
            clean_mean = sum(clean) / len(clean)
            corrupt_mean = sum(corrupt) / len(corrupt)
            summaries[key] = {
                "count": len(valid),
                "failed_count": len(selected) - len(valid),
                "clean_accuracy": sum(value > 0 for value in clean) / len(clean),
                "corrupt_accuracy": sum(value < 0 for value in corrupt) / len(corrupt),
                "mean_clean_logit_difference": clean_mean,
                "mean_corrupt_logit_difference": corrupt_mean,
                "clean_corrupt_contrast": clean_mean - corrupt_mean,
            }
        output[dimension] = summaries
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
    transitions: dict[str, dict[str, Counter[str]]] = {}
    for left, right in specs:
        counters = {
            "clean_correct": Counter[str](),
            "corrupt_correct": Counter[str](),
            "joint_clean_corrupt_state": Counter[str](),
        }
        for variants in complete.values():
            a, b = variants[left], variants[right]
            counters["clean_correct"][f"{bool(a.clean_correct)}->{bool(b.clean_correct)}"] += 1
            counters["corrupt_correct"][
                f"{bool(a.corrupt_correct)}->{bool(b.corrupt_correct)}"
            ] += 1
            a_joint = f"clean={bool(a.clean_correct)},corrupt={bool(a.corrupt_correct)}"
            b_joint = f"clean={bool(b.clean_correct)},corrupt={bool(b.corrupt_correct)}"
            counters["joint_clean_corrupt_state"][f"{a_joint}->{b_joint}"] += 1
        transitions[f"{left}_to_{right}"] = counters
    return {
        "complete_family_count": len(complete),
        "comparisons": comparisons,
        "paired_correctness_transitions": {
            comparison: {
                kind: dict(sorted(counter.items())) for kind, counter in kinds.items()
            }
            for comparison, kinds in transitions.items()
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
        and all(
            metrics[position]["example_count"] == FAMILY_COUNT
            and metrics[position]["failed_count"] == 0
            for position in POSITIONS
        )
        and effects.get("complete_family_count") == FAMILY_COUNT
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
