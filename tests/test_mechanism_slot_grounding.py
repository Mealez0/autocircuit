from __future__ import annotations

import hashlib

from autocircuit.datasets.associative_recall import ExamplePair
from autocircuit.mechanism_counterfactuals import build_discovery_counterfactuals


class WordTokenizer:
    name_or_path = "word-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [
            int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little")
            for token in text.replace("\n", " ").split()
        ]


def _example() -> ExamplePair:
    tokenizer = WordTokenizer()
    assignments = [["Alice", " apples"], ["Ben", " books"], ["Cara", " cats"]]
    return ExamplePair(
        example_id="ar-slot-demo",
        family_id="family-slot-demo",
        split="discovery",
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
        target_token_id=tokenizer.encode(" apples")[0],
        distractor_token_id=tokenizer.encode(" books")[0],
        changed_factor="queried_value_swap",
        seed=23,
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


def test_query_counterfactual_is_grounded_in_fact_slot_not_entity_identity() -> None:
    pairs, rejected = build_discovery_counterfactuals([_example()], WordTokenizer())
    assert rejected == {}
    query_pair = next(pair for pair in pairs if pair.kind == "query_swap")

    assert query_pair.base_query_fact_index == 0
    assert query_pair.donor_query_fact_index == 1
    assert query_pair.base_query_entity == "Alice"
    assert query_pair.donor_query_entity == "Ben"


def test_value_binding_counterfactual_preserves_selected_fact_slot() -> None:
    pairs, _ = build_discovery_counterfactuals([_example()], WordTokenizer())
    value_pair = next(pair for pair in pairs if pair.kind == "value_binding_swap")

    assert value_pair.base_query_fact_index == 0
    assert value_pair.donor_query_fact_index == 0
