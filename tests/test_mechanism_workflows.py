from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from autocircuit.causal_subspace import ActivationAlignment, CausalSubspace
from autocircuit.mechanism_artifacts import build_alignment_manifest
from autocircuit.mechanism_counterfactuals import CounterfactualPair
from autocircuit.mechanism_fit import fit_discovery_artifacts
from autocircuit.mechanism_outcomes import ObservationPolicy
from autocircuit.mechanism_run import execute_mechanism_artifacts
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


def _mechanism_plan(experiment_id: str = "interchange_query_state_at_mlp") -> dict[str, object]:
    campaign = build_associative_recall_campaign(_search_plan())
    return {
        "schema_version": 1,
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
    for index in range(4):
        family = f"family-{index}"
        source = f"source-{index}"
        rows.append(_pair(family, source, "query_swap"))
        rows.append(_pair(family, source, "value_binding_swap"))
    return rows


def _subspace(variable: str, feature_dim: int, coordinate: int = 0) -> CausalSubspace:
    basis = torch.zeros((feature_dim, 1), dtype=torch.float64)
    basis[coordinate, 0] = 1.0
    return CausalSubspace(
        variable_name=variable,
        basis=basis,
        class_labels=("slot_0", "slot_1"),
        fit_sample_count=8,
        singular_values=(1.0,),
        between_class_energy_fraction=1.0,
    )


def _alignments() -> dict[str, ActivationAlignment]:
    return {
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
        "downstream_slot_state": ActivationAlignment(
            "downstream-slot-state",
            "query_slot",
            "blocks.5.hook_resid_post",
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


class FakeRuntime:
    model_id = "EleutherAI/pythia-70m"
    revision = "step143000"
    resolved_revision = "deadbeef"
    tokenizer_id = "EleutherAI/pythia-70m"
    dtype = "torch.float32"
    device = "cpu"

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("workflow unit test should stub runtime execution")


def test_fit_workflow_publishes_disjoint_fit_and_eval_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autocircuit.mechanism_fit as module

    pairs = _pairs()
    monkeypatch.setattr(
        module,
        "build_discovery_counterfactuals",
        lambda examples, tokenizer: (pairs, {}),
    )
    monkeypatch.setattr(
        module,
        "fit_runtime_alignments",
        lambda runtime, campaign, fit_pairs, max_rank: (
            _alignments(),
            {
                "fit_scope": "exploratory_discovery_only",
                "fit_pair_count": len(fit_pairs),
                "max_rank": max_rank,
                "scientific_confirmation": False,
                "circuit_found": False,
            },
        ),
    )
    output = tmp_path / "fit"
    examples = [object(), object(), object(), object()]

    manifest = fit_discovery_artifacts(
        _mechanism_plan(),
        examples,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        FakeRuntime(),
        output,
        source_artifacts={"mechanism_plan": {"path": "plan.json", "sha256": "a" * 64}},
        requested_revision="step143000",
    )

    assert manifest["status"] == "complete"
    fit = json.loads((output / "alignment_fit_counterfactuals.json").read_text())
    evaluate = json.loads((output / "mechanism_eval_counterfactuals.json").read_text())
    assert fit["analysis_role"] == "alignment_fit"
    assert evaluate["analysis_role"] == "mechanism_eval"
    fit_families = {row["source_family_id"] for row in fit["counterfactuals"]}
    eval_families = {row["source_family_id"] for row in evaluate["counterfactuals"]}
    assert fit_families.isdisjoint(eval_families)
    assert fit_families | eval_families == {f"family-{index}" for index in range(4)}
    assert (output / "alignments.json").is_file()
    assert (output / "run_manifest.json").is_file()


def _alignment_manifest() -> dict[str, object]:
    return build_alignment_manifest(
        _alignments(),
        source_artifacts={
            "alignment_fit_counterfactuals": {
                "path": "fit.json",
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


def _eval_manifest() -> dict[str, object]:
    pairs = [_pair("family-eval", "source-eval", "query_swap")]
    return {
        "interpretation_scope": "exploratory_discovery_only",
        "analysis_role": "mechanism_eval",
        "counterfactuals": [pair.to_dict() for pair in pairs],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def test_run_workflow_emits_direct_replan_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autocircuit.mechanism_run as module

    monkeypatch.setattr(
        module,
        "run_prepared_experiment",
        lambda runtime, prepared, alignments, policy: {
            "schema_version": 1,
            "interpretation_scope": "exploratory_discovery_only",
            "experiment_id": prepared.experiment_id,
            "aggregation": {
                "status": "observation_ready",
                "observation": "donor_slot_switch",
            },
            "held_out_validation_reused": False,
            "held_out_test_opened": False,
            "scientific_confirmation": False,
            "circuit_found": False,
        },
    )
    output = tmp_path / "run"
    manifest = execute_mechanism_artifacts(
        _mechanism_plan(),
        _eval_manifest(),
        _alignment_manifest(),
        FakeRuntime(),
        output,
        source_artifacts={"alignments": {"path": "alignments.json", "sha256": "b" * 64}},
        policy=ObservationPolicy(min_evaluable_pairs=1),
    )

    accepted = json.loads((output / "accepted_observation.json").read_text())
    assert manifest["observation_status"] == "observation_ready"
    assert accepted == {"interchange_query_state_at_mlp": "donor_slot_switch"}
    assert (output / "experiment_report.json").is_file()


def test_run_workflow_rejects_fit_population_or_model_identity_mismatch(tmp_path: Path) -> None:
    fit_manifest = _eval_manifest()
    fit_manifest["analysis_role"] = "alignment_fit"
    with pytest.raises(ValueError, match="mechanism_eval"):
        execute_mechanism_artifacts(
            _mechanism_plan(),
            fit_manifest,
            _alignment_manifest(),
            FakeRuntime(),
            tmp_path / "wrong-role",
            source_artifacts={"x": {"path": "x", "sha256": "a" * 64}},
        )

    class WrongRuntime(FakeRuntime):
        resolved_revision = "different"

    with pytest.raises(RuntimeError, match="resolved revision mismatch"):
        execute_mechanism_artifacts(
            _mechanism_plan(),
            _eval_manifest(),
            _alignment_manifest(),
            WrongRuntime(),
            tmp_path / "wrong-model",
            source_artifacts={"x": {"path": "x", "sha256": "a" * 64}},
        )
