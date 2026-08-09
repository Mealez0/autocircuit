"""Deterministic discovery-only counterfactuals for mechanism falsification.

These pairs specify which natural prompts provide base and donor states for an
interchange intervention. They never open validation or test data and never
upgrade a proposed mechanism into scientific evidence by themselves.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from autocircuit.datasets.associative_recall import ExamplePair, TEMPLATES
from autocircuit.datasets.validation import (
    DatasetValidationError,
    Tokenizer,
    validate_answer_tokens,
)

COUNTERFACTUAL_VERSION = "mechanism-counterfactuals-0.1.0"
INTERPRETATION_SCOPE = "exploratory_discovery_only"
KINDS = ("query_swap", "value_binding_swap")
_EXPERIMENT_KINDS: dict[str, str | None] = {
    "interchange_query_state_at_heads": "query_swap",
    "interchange_query_state_at_mlp": "query_swap",
    "interchange_value_state_at_mlp": "value_binding_swap",
    "suppress_secondary_head": None,
}


@dataclass(frozen=True)
class CounterfactualPair:
    """One shape-compatible base/donor prompt pair from the discovery population."""

    counterfactual_id: str
    source_example_id: str
    source_family_id: str
    split: str
    kind: str
    base_prompt: str
    donor_prompt: str
    base_query_entity: str
    donor_query_entity: str
    base_target_text: str
    donor_target_text: str
    base_target_token_id: int
    donor_target_token_id: int
    changed_variables: tuple[str, ...]
    prompt_token_length: int
    evidence_status: str = "proposal_only_not_evidence"

    def __post_init__(self) -> None:
        if self.split != "discovery":
            raise ValueError("mechanism counterfactuals must remain discovery-only")
        if self.kind not in KINDS:
            raise ValueError(f"unknown mechanism counterfactual kind: {self.kind}")
        for label, value in (
            ("counterfactual id", self.counterfactual_id),
            ("source example id", self.source_example_id),
            ("source family id", self.source_family_id),
            ("base prompt", self.base_prompt),
            ("donor prompt", self.donor_prompt),
            ("base query entity", self.base_query_entity),
            ("donor query entity", self.donor_query_entity),
            ("base target text", self.base_target_text),
            ("donor target text", self.donor_target_text),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be non-empty")
        if self.base_prompt == self.donor_prompt:
            raise ValueError("counterfactual donor prompt must differ from base prompt")
        if (
            not isinstance(self.base_target_token_id, int)
            or isinstance(self.base_target_token_id, bool)
            or not isinstance(self.donor_target_token_id, int)
            or isinstance(self.donor_target_token_id, bool)
            or self.base_target_token_id == self.donor_target_token_id
        ):
            raise ValueError("counterfactual target token ids must be distinct integers")
        if (
            not isinstance(self.prompt_token_length, int)
            or isinstance(self.prompt_token_length, bool)
            or self.prompt_token_length <= 0
        ):
            raise ValueError("counterfactual prompt token length must be positive")
        if not self.changed_variables or len(self.changed_variables) != len(
            set(self.changed_variables)
        ):
            raise ValueError("counterfactual changed variables must be unique and non-empty")
        if self.evidence_status != "proposal_only_not_evidence":
            raise ValueError("counterfactual pairs remain proposal-only")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def counterfactual_kind_for_experiment(experiment_id: str) -> str | None:
    """Return the natural donor/base contrast required by a mechanism experiment."""

    if experiment_id not in _EXPERIMENT_KINDS:
        raise ValueError(f"unknown mechanism experiment: {experiment_id}")
    return _EXPERIMENT_KINDS[experiment_id]


def _assignments(example: ExamplePair) -> list[tuple[str, str]]:
    raw = example.metadata.get("assignments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("source discovery metadata has no assignments")
    assignments: list[tuple[str, str]] = []
    for index, pair in enumerate(raw):
        if (
            not isinstance(pair, list | tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or not isinstance(pair[1], str)
            or not pair[1]
        ):
            raise ValueError(f"source discovery assignment {index} is malformed")
        assignments.append((pair[0], pair[1]))
    entities = [entity for entity, _ in assignments]
    if len(entities) != len(set(entities)):
        raise ValueError("source discovery assignments contain duplicate entities")
    fact_count = example.metadata.get("fact_count")
    if fact_count != len(assignments):
        raise ValueError("source discovery fact count disagrees with assignments")
    return assignments


def _template(example: ExamplePair) -> tuple[str, str]:
    relation = example.template_id.removesuffix("-v1")
    choices = {name: stop for name, stop in TEMPLATES}
    if relation not in choices:
        raise ValueError(f"unsupported source discovery template: {example.template_id}")
    return relation, choices[relation]


def _separator(example: ExamplePair) -> str:
    return "\n\n" if "\n\n" in example.clean_prompt else "\n"


def _render(
    assignments: Sequence[tuple[str, str]],
    query: str,
    relation: str,
    stop: str,
    separator: str,
) -> str:
    facts = separator.join(f"{entity} {relation}{value}{stop}" for entity, value in assignments)
    return f"{facts}{separator}{query} {relation}"


def _validated_source(
    example: ExamplePair, tokenizer: Tokenizer
) -> tuple[list[tuple[str, str]], str, str, str, int]:
    if example.split != "discovery":
        raise ValueError("mechanism counterfactual generation accepts discovery examples only")
    assignments = _assignments(example)
    query = example.metadata.get("query_entity")
    if not isinstance(query, str) or not query:
        raise ValueError("source discovery query entity is missing")
    relation, stop = _template(example)
    separator = _separator(example)
    reconstructed = _render(assignments, query, relation, stop, separator)
    if reconstructed != example.clean_prompt:
        raise ValueError("source clean prompt does not match reconstructed discovery metadata")

    matching = [value for entity, value in assignments if entity == query]
    if len(matching) != 1 or matching[0] != example.target_text:
        raise ValueError("source discovery target disagrees with query assignment")
    target_id, distractor_id = validate_answer_tokens(
        tokenizer, example.target_text, example.distractor_text
    )
    if target_id != example.target_token_id or distractor_id != example.distractor_token_id:
        raise ValueError("source discovery answer token ids disagree with tokenizer")

    base_length = len(tokenizer.encode(example.clean_prompt, add_special_tokens=False))
    corrupt_length = len(tokenizer.encode(example.corrupt_prompt, add_special_tokens=False))
    recorded_length = example.metadata.get("prompt_token_length")
    if recorded_length != base_length:
        raise ValueError("source discovery prompt token length disagrees with metadata")
    if corrupt_length != base_length:
        raise ValueError("source discovery clean/corrupt prompt lengths differ")
    return assignments, query, relation, stop, base_length


def _counterfactual_id(
    example: ExamplePair,
    kind: str,
    donor_query: str,
    donor_target: str,
) -> str:
    payload = json.dumps(
        [COUNTERFACTUAL_VERSION, example.example_id, example.family_id, kind, donor_query, donor_target],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"mcf-{hashlib.sha256(payload).hexdigest()[:20]}"


def _value_binding_pair(example: ExamplePair, prompt_length: int) -> CounterfactualPair:
    query = str(example.metadata["query_entity"])
    return CounterfactualPair(
        counterfactual_id=_counterfactual_id(
            example, "value_binding_swap", query, example.distractor_text
        ),
        source_example_id=example.example_id,
        source_family_id=example.family_id,
        split="discovery",
        kind="value_binding_swap",
        base_prompt=example.clean_prompt,
        donor_prompt=example.corrupt_prompt,
        base_query_entity=query,
        donor_query_entity=query,
        base_target_text=example.target_text,
        donor_target_text=example.distractor_text,
        base_target_token_id=example.target_token_id,
        donor_target_token_id=example.distractor_token_id,
        changed_variables=("value_binding", "retrieved_value"),
        prompt_token_length=prompt_length,
    )


def _query_swap_pair(
    example: ExamplePair,
    tokenizer: Tokenizer,
    assignments: Sequence[tuple[str, str]],
    base_query: str,
    relation: str,
    stop: str,
    prompt_length: int,
) -> tuple[CounterfactualPair | None, str | None]:
    separator = _separator(example)
    saw_length_match = False
    candidates = sorted(
        ((entity, value) for entity, value in assignments if entity != base_query),
        key=lambda item: (item[0], item[1]),
    )
    for donor_query, donor_target in candidates:
        donor_prompt = _render(assignments, donor_query, relation, stop, separator)
        donor_length = len(tokenizer.encode(donor_prompt, add_special_tokens=False))
        if donor_length != prompt_length:
            continue
        saw_length_match = True
        try:
            donor_id, base_id = validate_answer_tokens(
                tokenizer, donor_target, example.target_text
            )
        except DatasetValidationError:
            continue
        if base_id != example.target_token_id:
            raise ValueError("source discovery target token id changed during donor validation")
        return (
            CounterfactualPair(
                counterfactual_id=_counterfactual_id(
                    example, "query_swap", donor_query, donor_target
                ),
                source_example_id=example.example_id,
                source_family_id=example.family_id,
                split="discovery",
                kind="query_swap",
                base_prompt=example.clean_prompt,
                donor_prompt=donor_prompt,
                base_query_entity=base_query,
                donor_query_entity=donor_query,
                base_target_text=example.target_text,
                donor_target_text=donor_target,
                base_target_token_id=example.target_token_id,
                donor_target_token_id=donor_id,
                changed_variables=("query_key", "match_slot", "retrieved_value"),
                prompt_token_length=prompt_length,
            ),
            None,
        )
    reason = (
        "query_swap_no_single_token_donor"
        if saw_length_match
        else "query_swap_no_length_matched_donor"
    )
    return None, reason


def build_discovery_counterfactuals(
    examples: Sequence[ExamplePair], tokenizer: Tokenizer
) -> tuple[list[CounterfactualPair], dict[str, int]]:
    """Build deterministic shape-compatible mechanism contrasts from discovery only."""

    if any(example.split != "discovery" for example in examples):
        raise ValueError("mechanism counterfactual generation accepts discovery examples only")
    example_ids = [example.example_id for example in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("source discovery examples contain duplicate example ids")

    pairs: list[CounterfactualPair] = []
    rejected: dict[str, int] = {}
    for example in sorted(examples, key=lambda item: item.example_id):
        assignments, query, relation, stop, prompt_length = _validated_source(example, tokenizer)
        query_pair, rejection = _query_swap_pair(
            example,
            tokenizer,
            assignments,
            query,
            relation,
            stop,
            prompt_length,
        )
        if query_pair is not None:
            pairs.append(query_pair)
        elif rejection is not None:
            rejected[rejection] = rejected.get(rejection, 0) + 1
        pairs.append(_value_binding_pair(example, prompt_length))

    ids = [pair.counterfactual_id for pair in pairs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("mechanism counterfactual generator produced duplicate ids")
    return pairs, dict(sorted(rejected.items()))


def build_counterfactual_manifest(
    pairs: Sequence[CounterfactualPair],
    rejected: dict[str, int],
    *,
    source_example_count: int,
) -> dict[str, Any]:
    """Serialize a discovery-only counterfactual contract with explicit guardrails."""

    if any(pair.split != "discovery" for pair in pairs):
        raise ValueError("counterfactual manifest accepts discovery pairs only")
    if (
        not isinstance(source_example_count, int)
        or isinstance(source_example_count, bool)
        or source_example_count < 0
    ):
        raise ValueError("source example count must be a non-negative integer")
    if any(
        not isinstance(reason, str)
        or not reason
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for reason, count in rejected.items()
    ):
        raise ValueError("counterfactual rejection counts are malformed")
    return {
        "schema_version": 1,
        "generator_version": COUNTERFACTUAL_VERSION,
        "interpretation_scope": INTERPRETATION_SCOPE,
        "source_example_count": source_example_count,
        "counterfactual_count": len(pairs),
        "rejected": dict(sorted(rejected.items())),
        "counterfactuals": [pair.to_dict() for pair in pairs],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
