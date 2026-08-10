from __future__ import annotations

import pytest
import torch

from autocircuit.causal_subspace import ActivationAlignment, CausalSubspace
from autocircuit.mechanism_artifacts import build_alignment_manifest
from autocircuit.mechanism_counterfactuals import CounterfactualPair
from autocircuit.mechanism_execution import prepare_execution_bundle
from autocircuit.mechanism_synthesis import build_associative_recall_campaign


def _search_plan() -> dict[str, object]:
    return {
        "interpretation_scope": "exploratory_discovery_only",
        "selected_layer": 5,
        "component_context": {
            "attention_family_specific_advantage": 0.15,
            "mlp_family_specific_advantage": 0.17,
        },
        "ranked_head_evidence": [
            {
                "head_index": 3,
                "family_specific_advantage": 0.08,
                "family_specific_advantage_ci_95": [0.04, 0.12],
                "stable_positive": True,
            },
            {
                "head_index": 6,
                "family_specific_advantage": 0.07,
                "family_specific_advantage_ci_95": [0.03, 0.11],
                "stable_positive": True,
            },
        ],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def _subspace(variable: str, feature_dim: int) -> CausalSubspace:
    basis = torch.zeros((feature_dim, 1), dtype=torch.float64)
    basis[0, 0] = 1.0
    return CausalSubspace(
        variable_name=variable,
        basis=basis,
        class_labels=("slot_0", "slot_1"),
        fit_sample_count=8,
        singular_values=(1.0,),
        between_class_energy_fraction=1.0,
    )


def _alignment_manifest() -> dict[str, object]:
    alignments = {
        "head_query_state": ActivationAlignment(
            "head-query-state",
            "query_slot",
            "blocks.5.attn.hook_z",
            -1,
            (3, 6),
            _subspace("query_slot", 4),
        ),
        "mlp_query_state": ActivationAlignment(
            "mlp-query-state",
            "query_slot",
            "blocks.5.hook_mlp_out",
            -1,
            (),
            _subspace("query_slot", 3),
        ),
        "mlp_value_state": ActivationAlignment(
            "mlp-value-state",
            "retrieved_value",
            "blocks.5.hook_mlp_out",
            -1,
            (),
            CausalSubspace(
                variable_name="retrieved_value",
                basis=torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64),
                class_labels=("token_10", "token_11"),
                fit_sample_count=8,
                singular_values=(1.0,),
                between_class_energy_fraction=1.0,
            ),
        ),
    }
    return build_alignment_manifest(
        alignments,
        source_artifacts={"counterfactuals": {"path": "cf.json", "sha256": "a" * 64}},
        model_identity={
            "model_id": "EleutherAI/pythia-70m",
            "requested_revision": "step143000",
            "resolved_revision": "deadbeef",
            "tokenizer_id": "EleutherAI/pythia-70m",
            "dtype": "torch.float32",
        },
    )


def _pair(kind: str, source: str) -> CounterfactualPair:
    is_query = kind == "query_swap"
    return CounterfactualPair(
        counterfactual_id=f"mcf-{kind}-{source}",
        source_example_id=source,
        source_family_id=f"family-{source}",
        split="discovery",
        kind=kind,
        base_prompt=f"base-{source}",
        donor_prompt=(
            f"alternative-slot-{source}" if is_query else f"value-donor-{source}"
        ),
        base_query_entity="Alice",
        donor_query_entity="Ben" if is_query else "Alice",
        base_target_text=" apples",
        donor_target_text=" books",
        base_target_token_id=10,
        donor_target_token_id=11,
        changed_variables=(
            ("query_key", "match_slot", "retrieved_value")
            if is_query
            else ("value_binding", "retrieved_value")
        ),
        prompt_token_length=7,
        base_query_fact_index=0,
        donor_query_fact_index=1 if is_query else 0,
    )


def _counterfactual_manifest(*, include_query: bool = True) -> dict[str, object]:
    pairs = [_pair("value_binding_swap", "one")]
    if include_query:
        pairs.insert(0, _pair("query_swap", "one"))
    return {
        "interpretation_scope": "exploratory_discovery_only",
        "counterfactuals": [pair.to_dict() for pair in pairs],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def _mechanism_plan(experiment_id: str) -> dict[str, object]:
    campaign = build_associative_recall_campaign(_search_plan())
    return {
        "interpretation_scope": "exploratory_discovery_only",
        "campaign": campaign.to_dict(),
        "state": {
            "surviving_hypotheses": [hypothesis.hypothesis_id for hypothesis in campaign.hypotheses]
        },
        "next_experiment": {"experiment_id": experiment_id},
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def test_value_interchange_uses_query_swap_only_as_alternative_slot_reference() -> None:
    prepared = prepare_execution_bundle(
        _mechanism_plan("interchange_value_state_at_mlp"),
        _counterfactual_manifest(),
        _alignment_manifest(),
    )

    assert [pair.kind for pair in prepared.counterfactuals] == ["value_binding_swap"]
    assert [pair.kind for pair in prepared.slot_references] == ["query_swap"]
    assert prepared.counterfactuals[0].donor_prompt == "value-donor-one"
    assert prepared.slot_references[0].donor_prompt == "alternative-slot-one"
    assert (
        prepared.counterfactuals[0].source_example_id
        == prepared.slot_references[0].source_example_id
    )


def test_query_interchange_uses_its_own_query_swap_as_slot_reference() -> None:
    prepared = prepare_execution_bundle(
        _mechanism_plan("interchange_query_state_at_heads"),
        _counterfactual_manifest(),
        _alignment_manifest(),
    )

    assert prepared.counterfactuals == prepared.slot_references
    assert prepared.slot_references[0].donor_query_fact_index == 1


def test_suppression_uses_query_swap_base_and_alternative_slot_reference() -> None:
    prepared = prepare_execution_bundle(
        _mechanism_plan("suppress_secondary_head"),
        _counterfactual_manifest(),
        _alignment_manifest(),
    )

    assert prepared.input_mode == "base_only"
    assert [pair.kind for pair in prepared.counterfactuals] == ["query_swap"]
    assert prepared.counterfactuals == prepared.slot_references


def test_value_interchange_refuses_to_fake_slot_reference_when_query_swap_missing() -> None:
    with pytest.raises(ValueError, match="query_swap slot reference"):
        prepare_execution_bundle(
            _mechanism_plan("interchange_value_state_at_mlp"),
            _counterfactual_manifest(include_query=False),
            _alignment_manifest(),
        )


def test_prepared_manifest_records_separate_slot_reference_ids() -> None:
    prepared = prepare_execution_bundle(
        _mechanism_plan("interchange_value_state_at_mlp"),
        _counterfactual_manifest(),
        _alignment_manifest(),
    )
    manifest = prepared.to_manifest()

    assert manifest["counterfactual_ids"] == ["mcf-value_binding_swap-one"]
    assert manifest["slot_reference_ids"] == ["mcf-query_swap-one"]
