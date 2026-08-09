from __future__ import annotations

import hashlib

import pytest

from autocircuit.datasets.associative_recall import ExamplePair
from autocircuit.mechanism_counterfactuals import (
    build_counterfactual_manifest,
    build_discovery_counterfactuals,
    counterfactual_kind_for_experiment,
)


class WordTokenizer:
    name_or_path = "word-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens = text.replace("\n", " ").split()
        return [int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") for token in tokens]


class QueryLengthMismatchTokenizer(WordTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = super().encode(text, add_special_tokens=add_special_tokens)
        if text.endswith("Ben likes") or text.endswith("Cara likes"):
            return [*tokens, 999999]
        return tokens


def _example(*, split: str = "discovery") -> ExamplePair:
    tokenizer = WordTokenizer()
    target_id = tokenizer.encode(" apples")[0]
    distractor_id = tokenizer.encode(" books")[0]
    assignments = [["Alice", " apples"], ["Ben", " books"], ["Cara", " cats"]]
    return ExamplePair(
        example_id="ar-demo",
        family_id="family-demo",
        split=split,
        clean_prompt=(
            "Alice likes apples.\n"
            "Ben likes books.\n"
            "Cara likes cats.\n"
            "Alice likes"
        ),
        corrupt_prompt=(
            "Alice likes books.\n"
            "Ben likes apples.\n"
            "Cara likes cats.\n"
            "Alice likes"
        ),
        target_text=" apples",
        distractor_text=" books",
        target_token_id=target_id,
        distractor_token_id=distractor_id,
        changed_factor="queried_value_swap",
        seed=17,
        template_id="likes-v1",
        metadata={
            "assignments": assignments,
            "query_entity": "Alice",
            "fact_count": 3,
            "query_position": "final_next_token",
            "prompt_token_length": 11,
            "query_fact_index": 0,
            "normalized_query_position": "first",
        },
    )


def test_counterfactuals_isolate_query_and_value_changes_on_discovery_only() -> None:
    pairs, rejected = build_discovery_counterfactuals([_example()], WordTokenizer())

    assert rejected == {}
    assert [pair.kind for pair in pairs] == ["query_swap", "value_binding_swap"]
    query_pair, value_pair = pairs

    assert query_pair.base_prompt == _example().clean_prompt
    assert query_pair.donor_prompt.endswith("Ben likes")
    assert query_pair.base_query_entity == "Alice"
    assert query_pair.donor_query_entity == "Ben"
    assert query_pair.base_target_text == " apples"
    assert query_pair.donor_target_text == " books"
    assert query_pair.changed_variables == ("query_key", "match_slot", "retrieved_value")
    assert query_pair.prompt_token_length == 11

    assert value_pair.base_prompt == _example().clean_prompt
    assert value_pair.donor_prompt == _example().corrupt_prompt
    assert value_pair.base_query_entity == value_pair.donor_query_entity == "Alice"
    assert value_pair.base_target_text == " apples"
    assert value_pair.donor_target_text == " books"
    assert value_pair.changed_variables == ("value_binding", "retrieved_value")
    assert value_pair.prompt_token_length == 11


def test_counterfactual_ids_and_manifest_are_deterministic_and_fail_closed() -> None:
    tokenizer = WordTokenizer()
    first, first_rejected = build_discovery_counterfactuals([_example()], tokenizer)
    second, second_rejected = build_discovery_counterfactuals([_example()], tokenizer)

    assert [pair.counterfactual_id for pair in first] == [pair.counterfactual_id for pair in second]
    assert first_rejected == second_rejected == {}

    manifest = build_counterfactual_manifest(first, first_rejected, source_example_count=1)
    assert manifest["interpretation_scope"] == "exploratory_discovery_only"
    assert manifest["counterfactual_count"] == 2
    assert manifest["held_out_validation_reused"] is False
    assert manifest["held_out_test_opened"] is False
    assert manifest["scientific_confirmation"] is False
    assert manifest["circuit_found"] is False


def test_query_swap_is_skipped_when_no_equal_length_donor_exists() -> None:
    pairs, rejected = build_discovery_counterfactuals(
        [_example()], QueryLengthMismatchTokenizer()
    )

    assert [pair.kind for pair in pairs] == ["value_binding_swap"]
    assert rejected == {"query_swap_no_length_matched_donor": 1}


def test_non_discovery_examples_are_rejected_before_counterfactual_generation() -> None:
    with pytest.raises(ValueError, match="discovery examples only"):
        build_discovery_counterfactuals([_example(split="validation")], WordTokenizer())


def test_malformed_source_prompt_or_metadata_fails_closed() -> None:
    example = _example()
    example.metadata["query_entity"] = "Ben"

    with pytest.raises(ValueError, match="source clean prompt"):
        build_discovery_counterfactuals([example], WordTokenizer())


def test_experiments_have_explicit_counterfactual_requirements() -> None:
    assert counterfactual_kind_for_experiment("interchange_query_state_at_heads") == "query_swap"
    assert counterfactual_kind_for_experiment("interchange_query_state_at_mlp") == "query_swap"
    assert counterfactual_kind_for_experiment("interchange_value_state_at_mlp") == "value_binding_swap"
    assert counterfactual_kind_for_experiment("suppress_secondary_head") is None
    with pytest.raises(ValueError, match="unknown mechanism experiment"):
        counterfactual_kind_for_experiment("unregistered")
