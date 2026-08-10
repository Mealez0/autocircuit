from __future__ import annotations

from autocircuit.mechanism_counterfactuals import CounterfactualPair
from autocircuit.mechanism_partition import (
    MechanismPartitionPolicy,
    build_mechanism_partition,
    select_partition_role,
)


def _pair(family: str, source: str, kind: str) -> CounterfactualPair:
    query = kind == "query_swap"
    return CounterfactualPair(
        counterfactual_id=f"mcf-{family}-{source}-{kind}",
        source_example_id=source,
        source_family_id=family,
        split="discovery",
        kind=kind,
        base_prompt=f"base-{source}",
        donor_prompt=f"donor-{source}-{kind}",
        base_query_entity="Alice",
        donor_query_entity="Ben" if query else "Alice",
        base_target_text=" apples",
        donor_target_text=" books",
        base_target_token_id=10,
        donor_target_token_id=11,
        changed_variables=(
            ("query_key", "match_slot", "retrieved_value")
            if query
            else ("value_binding", "retrieved_value")
        ),
        prompt_token_length=7,
        base_query_fact_index=0,
        donor_query_fact_index=1 if query else 0,
    )


def _pairs() -> list[CounterfactualPair]:
    rows: list[CounterfactualPair] = []
    for family_index in range(8):
        family = f"family-{family_index}"
        for source_index in range(2):
            source = f"source-{family_index}-{source_index}"
            rows.append(_pair(family, source, "query_swap"))
            rows.append(_pair(family, source, "value_binding_swap"))
    return rows


def test_partition_is_deterministic_order_invariant_and_family_atomic() -> None:
    policy = MechanismPartitionPolicy(fit_fraction=0.625)
    first = build_mechanism_partition(_pairs(), policy=policy)
    second = build_mechanism_partition(list(reversed(_pairs())), policy=policy)

    assert first == second
    assert first["fit_family_count"] == 5
    assert first["eval_family_count"] == 3
    role_by_family: dict[str, set[str]] = {}
    for pair in _pairs():
        role = first["role_by_counterfactual_id"][pair.counterfactual_id]
        role_by_family.setdefault(pair.source_family_id, set()).add(role)
    assert all(len(roles) == 1 for roles in role_by_family.values())


def test_partition_selects_only_requested_discovery_role() -> None:
    pairs = _pairs()
    partition = build_mechanism_partition(pairs)
    fit = select_partition_role(pairs, partition, "alignment_fit")
    evaluate = select_partition_role(pairs, partition, "mechanism_eval")

    assert fit
    assert evaluate
    assert {pair.counterfactual_id for pair in fit}.isdisjoint(
        pair.counterfactual_id for pair in evaluate
    )
    assert len(fit) + len(evaluate) == len(pairs)
    assert all(pair.split == "discovery" for pair in fit + evaluate)


def test_partition_refuses_single_family_or_duplicate_counterfactual_ids() -> None:
    single = [_pair("one-family", "source-1", "query_swap")]
    try:
        build_mechanism_partition(single)
    except ValueError as exc:
        assert "two discovery families" in str(exc)
    else:
        raise AssertionError("single-family partition should fail closed")

    duplicate = _pairs()[:2]
    duplicate.append(duplicate[0])
    try:
        build_mechanism_partition(duplicate)
    except ValueError as exc:
        assert "duplicate counterfactual ids" in str(exc)
    else:
        raise AssertionError("duplicate counterfactual ids should fail closed")


def test_partition_policy_rejects_extreme_or_nonfinite_fraction() -> None:
    for fraction in (0.0, 1.0, float("nan")):
        try:
            MechanismPartitionPolicy(fit_fraction=fraction)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid fit fraction accepted: {fraction}")
