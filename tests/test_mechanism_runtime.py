from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from autocircuit.causal_subspace import ActivationAlignment, CausalSubspace
from autocircuit.mechanism_counterfactuals import CounterfactualPair
from autocircuit.mechanism_execution import PreparedMechanismExperiment
from autocircuit.mechanism_outcomes import ObservationPolicy
from autocircuit.mechanism_runtime import (
    RuntimeForward,
    fit_runtime_alignments,
    run_prepared_experiment,
)
from autocircuit.mechanism_synthesis import build_associative_recall_campaign

Hook = tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]


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


def _pair(kind: str, index: int) -> CounterfactualPair:
    query = kind == "query_swap"
    return CounterfactualPair(
        counterfactual_id=f"mcf-{kind}-{index}",
        source_example_id=f"example-{index}",
        source_family_id=f"family-{index}",
        split="discovery",
        kind=kind,
        base_prompt=f"base-{kind}-{index}",
        donor_prompt=f"donor-{kind}-{index}",
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


class FitRuntime:
    model_id = "EleutherAI/pythia-70m"
    revision = "step143000"
    resolved_revision = "deadbeef"
    tokenizer_id = "EleutherAI/pythia-70m"
    dtype = "torch.float32"
    device = "cpu"

    def forward(
        self,
        prompt: str,
        *,
        cache_sites: tuple[str, ...],
        hooks: tuple[Hook, ...] = (),
    ) -> RuntimeForward:
        assert not hooks
        donor = prompt.startswith("donor-")
        slot_sign = -1.0 if donor else 1.0
        value_sign = -1.0 if donor else 1.0
        cache: dict[str, torch.Tensor] = {}
        for site in cache_sites:
            if site.endswith("attn.hook_z"):
                value = torch.zeros((1, 2, 8, 2), dtype=torch.float64)
                value[0, -1, 3, 0] = slot_sign
                value[0, -1, 6, 0] = slot_sign * 0.5
                cache[site] = value
            elif site.endswith("hook_mlp_out"):
                cache[site] = torch.tensor(
                    [[[0.0, 0.0, 0.0], [slot_sign, value_sign, 0.0]]],
                    dtype=torch.float64,
                )
            elif site.endswith("hook_resid_post"):
                cache[site] = torch.tensor(
                    [[[0.0, 0.0, 0.0], [slot_sign, 0.0, 0.0]]],
                    dtype=torch.float64,
                )
            else:
                raise AssertionError(site)
        return RuntimeForward(final_logits=torch.zeros(32, dtype=torch.float64), cache=cache)


def _axis_subspace(variable: str, dimension: int) -> CausalSubspace:
    basis = torch.zeros((dimension, 1), dtype=torch.float64)
    basis[0, 0] = 1.0
    return CausalSubspace(
        variable_name=variable,
        basis=basis,
        class_labels=("slot_0", "slot_1"),
        fit_sample_count=8,
        singular_values=(1.0,),
        between_class_energy_fraction=1.0,
    )


def test_runtime_alignment_fit_uses_fact_slots_and_adds_downstream_readout() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    pairs = (
        _pair("query_swap", 1),
        _pair("query_swap", 2),
        _pair("value_binding_swap", 1),
        _pair("value_binding_swap", 2),
    )

    alignments, report = fit_runtime_alignments(
        FitRuntime(), campaign, pairs, max_rank=2
    )

    assert set(alignments) == {
        "head_query_state",
        "mlp_query_state",
        "downstream_slot_state",
        "mlp_value_state",
    }
    assert alignments["head_query_state"].selected_heads == (3, 6)
    assert alignments["head_query_state"].hook_site == "blocks.5.attn.hook_z"
    assert alignments["downstream_slot_state"].hook_site == "blocks.5.hook_resid_post"
    assert alignments["downstream_slot_state"].subspace.class_labels == ("slot_0", "slot_1")
    assert alignments["mlp_value_state"].subspace.class_labels == ("token_10", "token_11")
    assert report["query_training_sample_count"] == 4
    assert report["value_training_sample_count"] == 4
    assert report["fit_scope"] == "exploratory_discovery_only"
    assert report["scientific_confirmation"] is False


class ExecutionRuntime:
    model_id = "EleutherAI/pythia-70m"
    revision = "step143000"
    resolved_revision = "deadbeef"
    tokenizer_id = "EleutherAI/pythia-70m"
    dtype = "torch.float32"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], int]] = []

    def forward(
        self,
        prompt: str,
        *,
        cache_sites: tuple[str, ...],
        hooks: tuple[Hook, ...] = (),
    ) -> RuntimeForward:
        self.calls.append((prompt, cache_sites, len(hooks)))
        is_donor = prompt.startswith("donor-")
        is_patched = bool(hooks)
        cache: dict[str, torch.Tensor] = {}
        for site in cache_sites:
            if site == "blocks.5.hook_mlp_out":
                source = torch.tensor(
                    [[[0.0, 0.0, 0.0], [-1.0 if is_donor else 1.0, 0.0, 0.0]]],
                    dtype=torch.float64,
                )
                cache[site] = source
            elif site == "blocks.5.hook_resid_post":
                slot = -0.9 if is_patched or is_donor else 1.0
                cache[site] = torch.tensor(
                    [[[0.0, 0.0, 0.0], [slot, 0.0, 0.0]]],
                    dtype=torch.float64,
                )
        logits = torch.zeros(32, dtype=torch.float64)
        if is_patched:
            logits[10] = 0.5
            logits[11] = 3.0
        elif is_donor:
            logits[10] = 0.0
            logits[11] = 3.0
        else:
            logits[10] = 3.0
            logits[11] = 0.0
        return RuntimeForward(final_logits=logits, cache=cache)


def _execution_alignments() -> dict[str, ActivationAlignment]:
    return {
        "mlp_query_state": ActivationAlignment(
            alignment_id="mlp-query-state",
            variable_name="query_slot",
            hook_site="blocks.5.hook_mlp_out",
            position_index=-1,
            selected_heads=(),
            subspace=_axis_subspace("query_slot", 3),
        ),
        "downstream_slot_state": ActivationAlignment(
            alignment_id="downstream-slot-state",
            variable_name="query_slot",
            hook_site="blocks.5.hook_resid_post",
            position_index=-1,
            selected_heads=(),
            subspace=_axis_subspace("query_slot", 3),
        ),
    }


def test_runtime_executes_selected_intervention_and_emits_measurable_observation() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    from autocircuit.mechanism_compile import compile_experiment

    alignments = _execution_alignments()
    recipe = compile_experiment(
        campaign,
        "interchange_query_state_at_mlp",
        alignments,
    )
    prepared = PreparedMechanismExperiment(
        experiment_id="interchange_query_state_at_mlp",
        recipe=recipe,
        input_mode="base_donor",
        counterfactual_kind="query_swap",
        counterfactuals=(_pair("query_swap", 1),),
    )
    runtime = ExecutionRuntime()

    result = run_prepared_experiment(
        runtime,
        prepared,
        alignments,
        policy=ObservationPolicy(min_evaluable_pairs=1, min_consensus_fraction=1.0),
    )

    assert result["aggregation"]["status"] == "observation_ready"
    assert result["aggregation"]["observation"] == "donor_slot_switch"
    assert result["measurements"][0]["outcome"] == "donor_slot_switch"
    assert result["measurements"][0]["base_margin"] == pytest.approx(3.0)
    assert result["measurements"][0]["patched_margin"] == pytest.approx(-2.5)
    assert result["held_out_validation_reused"] is False
    assert result["held_out_test_opened"] is False
    assert result["scientific_confirmation"] is False
    assert result["circuit_found"] is False
    assert any(call[2] == 1 for call in runtime.calls)


def test_runtime_refuses_downstream_slot_alignment_from_wrong_layer() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    from autocircuit.mechanism_compile import compile_experiment

    alignments = _execution_alignments()
    downstream = alignments["downstream_slot_state"]
    alignments["downstream_slot_state"] = ActivationAlignment(
        alignment_id=downstream.alignment_id,
        variable_name=downstream.variable_name,
        hook_site="blocks.4.hook_resid_post",
        position_index=-1,
        selected_heads=(),
        subspace=downstream.subspace,
    )
    recipe = compile_experiment(
        campaign,
        "interchange_query_state_at_mlp",
        alignments,
    )
    prepared = PreparedMechanismExperiment(
        experiment_id="interchange_query_state_at_mlp",
        recipe=recipe,
        input_mode="base_donor",
        counterfactual_kind="query_swap",
        counterfactuals=(_pair("query_swap", 1),),
    )

    with pytest.raises(ValueError, match="downstream slot alignment"):
        run_prepared_experiment(
            ExecutionRuntime(),
            prepared,
            alignments,
            policy=ObservationPolicy(min_evaluable_pairs=1),
        )
