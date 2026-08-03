"""One-shot held-out validation protocol for the matched position effect."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from autocircuit.datasets.associative_recall import ENTITIES, VALUES, ExamplePair
from autocircuit.datasets.validation import Tokenizer, validate_answer_tokens
from autocircuit.position_study import FAMILY_COUNT, POSITIONS, STUDY_VERSION

VALIDATION_PROTOCOL_VERSION = "position-validation-1.0.0"
VALIDATION_MANIFEST_SCHEMA_VERSION = "position-validation-manifest-1.0.0"
VALIDATION_POOL_ID = "associative-recall-validation-pool-v1"
VALIDATION_SEED_NAMESPACE = f"{VALIDATION_PROTOCOL_VERSION}:validation"
DISCOVERY_ELIGIBLE_STATUS = "QUERY_FIRST_DISCOVERY_ELIGIBLE_REQUIRES_HELD_OUT_VALIDATION"
VALIDATED_STATUS = "QUERY_FIRST_HELD_OUT_VALIDATED_READY_FOR_CAUSAL_LOCALIZATION"
NOT_CONFIRMED_STATUS = "HELD_OUT_VALIDATION_NOT_CONFIRMED"
FROZEN_PRIMARY_RULE = {
    "all_dataset_invariants_pass": True,
    "complete_matched_families": 120,
    "processed_examples_per_position": 120,
    "failed_count_per_position": 0,
    "first_clean_pairwise_accuracy_minimum": 0.80,
    "first_mean_clean_logit_difference_minimum": 1.0,
    "first_minus_last_clean_ld_estimate_positive": True,
    "first_minus_last_clean_ld_ci_95_lower_positive": True,
}


def _prompt(assignments: list[tuple[str, str]], query: str) -> str:
    return "\n".join(f"{entity} chooses{value}." for entity, value in assignments) + (
        f"\n{query} chooses"
    )


def validation_lexical_pools(split: str = "validation") -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve only the frozen validation pool; no generic split selection is allowed."""
    if split != "validation":
        raise ValueError("the position-validation protocol can only resolve validation pools")
    return ENTITIES[1], VALUES[1]


def _validation_design(
    discovery_seed: int,
) -> tuple[list[str], list[str], list[int], list[tuple[int, int]]]:
    seed_material = f"{VALIDATION_PROTOCOL_VERSION}:{discovery_seed}:validation"
    rng = random.Random(seed_material)
    entity_pool, value_pool = validation_lexical_pools()
    values = list(value_pool)
    entities = list(entity_pool)
    offsets = list(range(1, len(values)))
    rng.shuffle(values)
    rng.shuffle(entities)
    rng.shuffle(offsets)
    offsets = offsets[: len(entities)]
    population = [(target, query) for query in range(10) for target in range(12)]
    rng.shuffle(population)
    return entities, values, offsets, population


def generate_validation_dataset(tokenizer: Tokenizer, discovery_seed: int) -> list[ExamplePair]:
    entities, values, offsets, population = _validation_design(discovery_seed)
    examples: list[ExamplePair] = []
    for target_index, query_index in population:
        distractor_index = (target_index + offsets[query_index]) % 12
        third_index = next(i for i in range(12) if i not in {target_index, distractor_index})
        query = entities[query_index]
        other_entities = [entities[(query_index + 1) % 10], entities[(query_index + 2) % 10]]
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
        provenance = (
            f"{VALIDATION_PROTOCOL_VERSION}:validation:"
            f"{discovery_seed}:{target_index}:{query_index}"
        )
        family_id = "validation-family-" + hashlib.sha256(provenance.encode()).hexdigest()[:16]
        for position_index, (position, order) in enumerate(
            zip(POSITIONS, ((0, 1, 2), (1, 0, 2), (1, 2, 0)), strict=True)
        ):
            clean_prompt = _prompt([assignments[i] for i in order], query)
            corrupt_prompt = _prompt([corrupt[i] for i in order], query)
            clean_length = len(tokenizer.encode(clean_prompt, add_special_tokens=False))
            corrupt_length = len(tokenizer.encode(corrupt_prompt, add_special_tokens=False))
            if clean_length != corrupt_length:
                raise ValueError("validation clean/corrupt token lengths differ")
            examples.append(
                ExamplePair(
                    f"{family_id}-{position}",
                    family_id,
                    "validation",
                    clean_prompt,
                    corrupt_prompt,
                    target,
                    distractor,
                    target_id,
                    distractor_id,
                    "queried_value_swap",
                    discovery_seed,
                    "chooses-position-validation-v1",
                    {
                        "assignments": [list(item) for item in assignments],
                        "corrupt_assignments": [list(item) for item in corrupt],
                        "matched_family_id": family_id,
                        "query_entity": query,
                        "fact_count": 3,
                        "query_fact_index": position_index,
                        "normalized_query_position": position,
                        "prompt_token_length": clean_length,
                        "ordering": list(order),
                        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
                        "source_discovery_study_version": STUDY_VERSION,
                        "pool_identity": VALIDATION_POOL_ID,
                        "seed_namespace": VALIDATION_SEED_NAMESPACE,
                    },
                )
            )
    validate_validation_dataset(examples, tokenizer, discovery_seed)
    return examples


def validate_validation_dataset(
    examples: list[ExamplePair], tokenizer: Tokenizer, expected_discovery_seed: int
) -> dict[str, Any]:
    if len(examples) != 360:
        raise ValueError("expected exactly 360 validation examples")
    entities, values, offsets, population = _validation_design(expected_discovery_seed)
    expected_population = set(population)
    entity_pool, value_pool = validation_lexical_pools()
    families: dict[str, list[ExamplePair]] = defaultdict(list)
    for item in examples:
        if item.split != "validation":
            raise ValueError("validation dataset contains another split")
        if item.template_id != "chooses-position-validation-v1":
            raise ValueError("validation relation must be chooses")
        if item.metadata.get("validation_protocol_version") != VALIDATION_PROTOCOL_VERSION:
            raise ValueError("validation protocol version mismatch")
        if item.metadata.get("source_discovery_study_version") != STUDY_VERSION:
            raise ValueError("source discovery version mismatch")
        if item.metadata.get("pool_identity") != VALIDATION_POOL_ID:
            raise ValueError("validation lexical pool mismatch")
        if item.seed != expected_discovery_seed:
            raise ValueError("validation example seed mismatch")
        if item.metadata.get("matched_family_id") != item.family_id:
            raise ValueError("validation matched family ID mismatch")
        if item.metadata.get("seed_namespace") != VALIDATION_SEED_NAMESPACE:
            raise ValueError("validation seed namespace mismatch")
        families[item.family_id].append(item)
    if len(families) != FAMILY_COUNT:
        raise ValueError("expected exactly 120 validation families")
    targets: Counter[str] = Counter()
    distractors: Counter[str] = Counter()
    queries: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    marginals = {position: Counter[tuple[str, str, str]]() for position in POSITIONS}
    for variants in families.values():
        by_position = {str(item.metadata["normalized_query_position"]): item for item in variants}
        if len(variants) != 3 or set(by_position) != set(POSITIONS):
            raise ValueError("validation family position coverage failed")
        first = by_position["first"]
        assignments = [tuple(item) for item in first.metadata["assignments"]]
        corrupt = [tuple(item) for item in first.metadata["corrupt_assignments"]]
        query = str(first.metadata["query_entity"])
        if len(assignments) != 3 or first.metadata.get("fact_count") != 3:
            raise ValueError("validation fact count is not three")
        if first.target_text == first.distractor_text:
            raise ValueError("validation target equals distractor")
        all_assignments = assignments + corrupt
        if any(
            entity not in entity_pool or value not in value_pool
            for entity, value in all_assignments
        ):
            raise ValueError("validation assignments contain an out-of-pool entity or value")
        try:
            target_index = values.index(first.target_text)
            query_index = entities.index(query)
        except ValueError as exc:
            raise ValueError("validation family provenance is outside the frozen pool") from exc
        if (target_index, query_index) not in expected_population:
            raise ValueError("validation family provenance is not in the frozen population")
        distractor_index = (target_index + offsets[query_index]) % 12
        third_index = next(i for i in range(12) if i not in {target_index, distractor_index})
        expected_assignments = [
            (query, values[target_index]),
            (entities[(query_index + 1) % 10], values[distractor_index]),
            (entities[(query_index + 2) % 10], values[third_index]),
        ]
        if assignments != expected_assignments or first.distractor_text != values[distractor_index]:
            raise ValueError("validation canonical assignments are invalid")
        provenance = (
            f"{VALIDATION_PROTOCOL_VERSION}:validation:"
            f"{expected_discovery_seed}:{target_index}:{query_index}"
        )
        expected_family_id = (
            "validation-family-" + hashlib.sha256(provenance.encode()).hexdigest()[:16]
        )
        if first.family_id != expected_family_id:
            raise ValueError("validation family ID is not reconstructible")
        expected_corrupt = assignments.copy()
        expected_corrupt[0] = (query, first.distractor_text)
        expected_corrupt[1] = (assignments[1][0], first.target_text)
        if corrupt != expected_corrupt:
            raise ValueError("validation corrupt swap is invalid")
        base = asdict(first)
        base_metadata = base.pop("metadata")
        lengths: set[int] = set()
        for position, item in by_position.items():
            candidate = asdict(item)
            metadata = candidate.pop("metadata")
            if any(
                candidate[key] != base[key]
                for key in base
                if key not in {"example_id", "clean_prompt", "corrupt_prompt"}
            ):
                raise ValueError("validation invariant ExamplePair field changed")
            permitted = {
                "ordering",
                "query_fact_index",
                "normalized_query_position",
                "prompt_token_length",
            }
            if any(
                metadata.get(key) != base_metadata.get(key)
                for key in set(metadata) | set(base_metadata)
                if key not in permitted
            ):
                raise ValueError("validation invariant metadata changed")
            order = metadata.get("ordering")
            if not isinstance(order, list) or sorted(order) != [0, 1, 2]:
                raise ValueError("validation ordering is invalid")
            if [i for i in order if i != 0] != [1, 2]:
                raise ValueError("validation non-query ordering changed")
            query_index = order.index(0)
            if (
                query_index != metadata.get("query_fact_index")
                or POSITIONS[query_index] != position
            ):
                raise ValueError("validation position metadata is invalid")
            if item.clean_prompt != _prompt([assignments[i] for i in order], query):
                raise ValueError("validation clean prompt reconstruction failed")
            if item.corrupt_prompt != _prompt([corrupt[i] for i in order], query):
                raise ValueError("validation corrupt prompt reconstruction failed")
            if item.example_id != f"{expected_family_id}-{position}":
                raise ValueError("validation example ID is not reconstructible")
            target_id, distractor_id = validate_answer_tokens(
                tokenizer, item.target_text, item.distractor_text
            )
            if (target_id, distractor_id) != (item.target_token_id, item.distractor_token_id):
                raise ValueError("validation answer token IDs changed")
            clean_length = len(tokenizer.encode(item.clean_prompt, add_special_tokens=False))
            corrupt_length = len(tokenizer.encode(item.corrupt_prompt, add_special_tokens=False))
            if (
                clean_length != corrupt_length
                or metadata.get("prompt_token_length") != clean_length
            ):
                raise ValueError("validation token length mismatch")
            lengths.add(clean_length)
            marginals[position][(item.target_text, item.distractor_text, query)] += 1
        if len(lengths) != 1:
            raise ValueError("validation cross-position token lengths differ")
        targets[first.target_text] += 1
        distractors[first.distractor_text] += 1
        queries[query] += 1
        pairs[(first.target_text, first.distractor_text)] += 1
    if set(targets) != set(value_pool) or set(targets.values()) != {10}:
        raise ValueError("validation target balance failed")
    if set(distractors) != set(value_pool) or set(distractors.values()) != {10}:
        raise ValueError("validation distractor balance failed")
    if set(queries) != set(entity_pool) or set(queries.values()) != {12}:
        raise ValueError("validation entity balance failed")
    if len({tuple(sorted(value.items())) for value in marginals.values()}) != 1:
        raise ValueError("validation position marginals differ")
    complete_pairs = {
        f"{target}|{distractor}": pairs[(target, distractor)]
        for target in value_pool
        for distractor in value_pool
        if target != distractor
    }
    if Counter(complete_pairs.values()) != Counter({0: 12, 1: 120}):
        raise ValueError("validation ordered-pair accounting failed")
    return {
        "all_invariants_passed": True,
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "source_discovery_study_version": STUDY_VERSION,
        "pool_identity": VALIDATION_POOL_ID,
        "example_count": 360,
        "matched_family_count": 120,
        "target_counts": dict(sorted(targets.items())),
        "distractor_counts": dict(sorted(distractors.items())),
        "query_entity_counts": dict(sorted(queries.items())),
        "ordered_pair_observed_count": 120,
        "ordered_pair_missing_count": 12,
        "ordered_pair_minimum_including_zero": 0,
        "ordered_pair_maximum": 1,
        "ordered_pair_counts": complete_pairs,
        "ordered_pair_counts_sha256": hashlib.sha256(
            json.dumps(complete_pairs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def validation_decision(
    metrics: dict[str, Any], effects: dict[str, Any], balance: dict[str, Any]
) -> str:
    first = metrics["first"]
    comparison = effects["comparisons"]["first_minus_last_clean_logit_difference"]
    ci = comparison["ci_95"]
    confirmed = (
        balance.get("all_invariants_passed") is True
        and effects.get("complete_family_count") == 120
        and all(
            metrics[position]["example_count"] == 120 and metrics[position]["failed_count"] == 0
            for position in POSITIONS
        )
        and first.get("clean_pairwise_accuracy", 0.0) >= 0.80
        and first.get("mean_clean_logit_difference", float("-inf")) >= 1.0
        and comparison.get("estimate") is not None
        and comparison["estimate"] > 0
        and ci is not None
        and ci[0] > 0
    )
    return VALIDATED_STATUS if confirmed else NOT_CONFIRMED_STATUS
