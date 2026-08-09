from __future__ import annotations

import copy

import pytest
import torch

from autocircuit.causal_subspace import ActivationAlignment, CausalSubspace
from autocircuit.mechanism_artifacts import (
    build_alignment_manifest,
    load_alignment_manifest,
)
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


def _subspace(variable: str, feature_dim: int, coordinate: int = 0) -> CausalSubspace:
    basis = torch.zeros((feature_dim, 1), dtype=torch.float64)
    basis[coordinate, 0] = 1.0
    return CausalSubspace(
        variable_name=variable,
        basis=basis,
        class_labels=("base", "donor"),
        fit_sample_count=8,
        singular_values=(1.0,),
        between_class_energy_fraction=1.0,
    )


def _alignments() -> dict[str, ActivationAlignment]:
    return {
        "head_query_state": ActivationAlignment(
            alignment_id="head-query-state",
            variable_name="query_slot",
            hook_site="blocks.5.attn.hook_z",
            position_index=-1,
            selected_heads=(3, 6),
            subspace=_subspace("query_slot", 4),
        ),
        "mlp_query_state": ActivationAlignment(
            alignment_id="mlp-query-state",
            variable_name="query_slot",
            hook_site="blocks.5.hook_mlp_out",
            position_index=-1,
            selected_heads=(),
            subspace=_subspace("query_slot", 3),
        ),
        "mlp_value_state": ActivationAlignment(
            alignment_id="mlp-value-state",
            variable_name="retrieved_value",
            hook_site="blocks.5.hook_mlp_out",
            position_index=-1,
            selected_heads=(),
            subspace=_subspace("retrieved_value", 3, 1),
        ),
    }


def _alignment_manifest() -> dict[str, object]:
    return build_alignment_manifest(
        _alignments(),
        source_artifacts={
            "counterfactuals": {
                "path": "artifacts/mechanism_counterfactuals.json",
                "sha256": "a" * 64,
            }
        },
        model_identity={
            "model_id": "EleutherAI/pythia-70m",
            "requested_revision": "step143000",
            "resolved_revision": "deadbeef",
            "tokenizer_id": "EleutherAI/pythia-70m",
            "dtype": "torch.float32",
        },
    )


def _pair(kind: str, suffix: str = "1") -> CounterfactualPair:
    query = kind == "query_swap"
    return CounterfactualPair(
        counterfactual_id=f"mcf-{kind}-{suffix}",
        source_example_id=f"example-{suffix}",
        source_family_id=f"family-{suffix}",
        split="discovery",
        kind=kind,
        base_prompt="base prompt",
        donor_prompt="donor prompt",
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
    )


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


def _counterfactual_manifest() -> dict[str, object]:
    pairs = [_pair("query_swap", "1"), _pair("value_binding_swap", "1")]
    return {
        "interpretation_scope": "exploratory_discovery_only",
        "counterfactuals": [pair.to_dict() for pair in pairs],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def test_alignment_manifest_round_trips_exact_fingerprints() -> None:
    manifest = _alignment_manifest()
    loaded = load_alignment_manifest(manifest)

    assert set(loaded) == set(_alignments())
    assert {
        key: alignment.fingerprint() for key, alignment in loaded.items()
    } == {
        key: alignment.fingerprint() for key, alignment in _alignments().items()
    }
    assert manifest["held_out_test_opened"] is False
    assert manifest["scientific_confirmation"] is False


def test_alignment_manifest_rejects_tampered_basis_even_with_valid_shape() -> None:
    manifest = copy.deepcopy(_alignment_manifest())
    alignment = manifest["alignments"]["mlp_query_state"]
    alignment["subspace"]["basis"][0][0] = 0.0
    alignment["subspace"]["basis"][1][0] = 1.0

    with pytest.raises(ValueError, match="fingerprint"):
        load_alignment_manifest(manifest)


def test_execution_bundle_binds_query_experiment_to_query_counterfactuals() -> None:
    prepared = prepare_execution_bundle(
        _mechanism_plan("interchange_query_state_at_mlp"),
        _counterfactual_manifest(),
        _alignment_manifest(),
    )

    assert prepared.recipe.hook_site == "blocks.5.hook_mlp_out"
    assert prepared.recipe.operation == "interchange_subspace"
    assert prepared.input_mode == "base_donor"
    assert [pair.kind for pair in prepared.counterfactuals] == ["query_swap"]
    manifest = prepared.to_manifest()
    assert manifest["experiment_id"] == "interchange_query_state_at_mlp"
    assert manifest["counterfactual_kind"] == "query_swap"
    assert manifest["alignment_fingerprint"]
    assert manifest["held_out_validation_reused"] is False
    assert manifest["held_out_test_opened"] is False
    assert manifest["scientific_confirmation"] is False
    assert manifest["circuit_found"] is False


def test_execution_bundle_binds_value_experiment_to_value_counterfactuals() -> None:
    prepared = prepare_execution_bundle(
        _mechanism_plan("interchange_value_state_at_mlp"),
        _counterfactual_manifest(),
        _alignment_manifest(),
    )

    assert prepared.input_mode == "base_donor"
    assert [pair.kind for pair in prepared.counterfactuals] == ["value_binding_swap"]


def test_suppression_bundle_uses_one_base_record_per_source_example() -> None:
    prepared = prepare_execution_bundle(
        _mechanism_plan("suppress_secondary_head"),
        _counterfactual_manifest(),
        _alignment_manifest(),
    )

    assert prepared.recipe.operation == "zero_selected_heads"
    assert prepared.input_mode == "base_only"
    assert len(prepared.counterfactuals) == 1
    assert prepared.counterfactuals[0].source_example_id == "example-1"


def test_execution_bundle_rejects_missing_required_counterfactual_kind() -> None:
    manifest = _counterfactual_manifest()
    manifest["counterfactuals"] = [_pair("value_binding_swap").to_dict()]

    with pytest.raises(ValueError, match="query_swap"):
        prepare_execution_bundle(
            _mechanism_plan("interchange_query_state_at_heads"),
            manifest,
            _alignment_manifest(),
        )


def test_execution_bundle_rejects_any_opened_held_out_artifact() -> None:
    mechanism_plan = _mechanism_plan("interchange_query_state_at_mlp")
    mechanism_plan["held_out_test_opened"] = True

    with pytest.raises(ValueError, match="held-out"):
        prepare_execution_bundle(
            mechanism_plan,
            _counterfactual_manifest(),
            _alignment_manifest(),
        )
