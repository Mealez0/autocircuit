from __future__ import annotations

import pytest
import torch

from autocircuit.causal_subspace import ActivationAlignment, CausalSubspace
from autocircuit.mechanism_compile import build_hook_callback, compile_experiment
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


def _subspace(variable: str, feature_dim: int, selected_feature: int = 0) -> CausalSubspace:
    basis = torch.zeros((feature_dim, 1), dtype=torch.float64)
    basis[selected_feature, 0] = 1.0
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
            subspace=_subspace("retrieved_value", 3, selected_feature=1),
        ),
    }


def test_compiler_maps_abstract_experiments_to_exact_transformerlens_sites() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    alignments = _alignments()

    head_recipe = compile_experiment(campaign, "interchange_query_state_at_heads", alignments)
    mlp_recipe = compile_experiment(campaign, "interchange_query_state_at_mlp", alignments)
    suppress_recipe = compile_experiment(campaign, "suppress_secondary_head", alignments)

    assert head_recipe.hook_site == "blocks.5.attn.hook_z"
    assert head_recipe.selected_heads == (3, 6)
    assert head_recipe.operation == "interchange_subspace"
    assert head_recipe.position_index == -1
    assert mlp_recipe.hook_site == "blocks.5.hook_mlp_out"
    assert mlp_recipe.selected_heads == ()
    assert suppress_recipe.hook_site == "blocks.5.attn.hook_z"
    assert suppress_recipe.selected_heads == (6,)
    assert suppress_recipe.operation == "zero_selected_heads"
    assert suppress_recipe.alignment is None


def test_compiler_requires_alignment_matching_layer_site_and_heads() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    alignments = _alignments()
    wrong = alignments["head_query_state"]
    alignments["head_query_state"] = ActivationAlignment(
        alignment_id=wrong.alignment_id,
        variable_name=wrong.variable_name,
        hook_site="blocks.4.attn.hook_z",
        position_index=-1,
        selected_heads=(3, 6),
        subspace=wrong.subspace,
    )

    with pytest.raises(ValueError, match="hook site"):
        compile_experiment(campaign, "interchange_query_state_at_heads", alignments)


def test_head_subspace_callback_changes_only_selected_aligned_coordinates() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    recipe = compile_experiment(
        campaign,
        "interchange_query_state_at_heads",
        _alignments(),
    )
    base = torch.zeros((1, 2, 8, 2), dtype=torch.float64)
    donor = torch.zeros_like(base)
    donor[:, -1, 3, :] = torch.tensor([7.0, 8.0])
    donor[:, -1, 6, :] = torch.tensor([9.0, 10.0])

    patched = build_hook_callback(recipe, donor)(base, None)

    assert patched[0, -1, 3, 0].item() == pytest.approx(7.0)
    assert patched[0, -1, 3, 1].item() == pytest.approx(0.0)
    assert patched[0, -1, 6, :].tolist() == [0.0, 0.0]
    assert torch.equal(patched[:, 0, :, :], base[:, 0, :, :])
    untouched_heads = [head for head in range(8) if head not in (3, 6)]
    assert torch.equal(patched[:, -1, untouched_heads, :], base[:, -1, untouched_heads, :])


def test_mlp_subspace_callback_preserves_orthogonal_component() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    recipe = compile_experiment(campaign, "interchange_value_state_at_mlp", _alignments())
    base = torch.tensor([[[1.0, 10.0, 100.0]]], dtype=torch.float64)
    donor = torch.tensor([[[2.0, 30.0, 300.0]]], dtype=torch.float64)

    patched = build_hook_callback(recipe, donor)(base, None)

    assert patched.tolist() == [[[1.0, 30.0, 100.0]]]


def test_suppression_callback_zeros_only_secondary_head_at_query_position() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    recipe = compile_experiment(campaign, "suppress_secondary_head", _alignments())
    base = torch.ones((1, 2, 8, 2), dtype=torch.float64)

    patched = build_hook_callback(recipe)(base, None)

    assert patched[0, -1, 6, :].tolist() == [0.0, 0.0]
    assert patched[0, -1, 3, :].tolist() == [1.0, 1.0]
    assert torch.equal(patched[:, 0, :, :], base[:, 0, :, :])


def test_interchange_callback_rejects_missing_or_mismatched_donor() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    recipe = compile_experiment(
        campaign,
        "interchange_query_state_at_mlp",
        _alignments(),
    )
    with pytest.raises(ValueError, match="donor activation"):
        build_hook_callback(recipe)

    callback = build_hook_callback(recipe, torch.zeros((2, 1, 3), dtype=torch.float64))
    with pytest.raises(RuntimeError, match="shape"):
        callback(torch.zeros((1, 1, 3), dtype=torch.float64), None)
