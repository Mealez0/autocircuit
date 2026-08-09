from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocircuit.mechanism_plan import generate_mechanism_plan
from autocircuit.mechanism_synthesis import build_associative_recall_campaign


def _search_plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "planner_version": "adaptive-search-controller-0.1.0",
        "interpretation_scope": "exploratory_discovery_only",
        "selected_layer": 5,
        "strategy": "causal_evidence_guided_ucb_with_coverage",
        "component_context": {
            "attention_family_specific_advantage": 0.153,
            "mlp_family_specific_advantage": 0.171,
        },
        "ranked_head_evidence": [
            {
                "head_index": 3,
                "family_specific_advantage": 0.083,
                "family_specific_advantage_ci_95": [0.050, 0.116],
                "stable_positive": True,
                "uncertainty_radius": 0.033,
                "upper_confidence_score": 0.0995,
            },
            {
                "head_index": 6,
                "family_specific_advantage": 0.076,
                "family_specific_advantage_ci_95": [0.041, 0.109],
                "stable_positive": True,
                "uncertainty_radius": 0.034,
                "upper_confidence_score": 0.093,
            },
            {
                "head_index": 1,
                "family_specific_advantage": 0.028,
                "family_specific_advantage_ci_95": [-0.004, 0.060],
                "stable_positive": False,
                "uncertainty_radius": 0.032,
                "upper_confidence_score": 0.044,
            },
        ],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_associative_recall_campaign_uses_discovery_carriers_without_claiming_roles() -> None:
    campaign = build_associative_recall_campaign(_search_plan())
    rendered = campaign.to_dict()

    assert rendered["selected_layer"] == 5
    assert rendered["candidate_carriers"]["attention_heads"] == [3, 6]
    assert len(rendered["hypotheses"]) == 3
    assert {hypothesis["hypothesis_id"] for hypothesis in rendered["hypotheses"]} == {
        "heads_match_mlp_value",
        "mlp_match_heads_route",
        "split_head_match_route",
    }
    assert all(
        hypothesis["evidence_status"] == "proposal_only_not_evidence"
        for hypothesis in rendered["hypotheses"]
    )
    assert rendered["scientific_confirmation"] is False
    assert rendered["circuit_found"] is False


def test_campaign_rejects_non_discovery_or_opened_held_out_input() -> None:
    wrong_scope = _search_plan()
    wrong_scope["interpretation_scope"] = "validation"
    with pytest.raises(ValueError, match="discovery-only"):
        build_associative_recall_campaign(wrong_scope)

    opened_test = _search_plan()
    opened_test["held_out_test_opened"] = True
    with pytest.raises(ValueError, match="held-out"):
        build_associative_recall_campaign(opened_test)


def test_generate_mechanism_plan_writes_provenance_and_information_gain_choice(
    tmp_path: Path,
) -> None:
    source = tmp_path / "adaptive_search_plan.json"
    output = tmp_path / "mechanism_plan.json"
    _write(source, _search_plan())

    plan = generate_mechanism_plan(source, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted == plan
    assert plan["source_artifact"]["sha256"]
    assert plan["next_experiment"]["experiment_id"] == "interchange_query_state_at_mlp"
    assert plan["next_experiment"]["expected_information_gain_bits"] == pytest.approx(
        1.584962500721156
    )
    assert plan["state"]["surviving_hypotheses"] == [
        "heads_match_mlp_value",
        "mlp_match_heads_route",
        "split_head_match_route",
    ]
    assert plan["held_out_validation_reused"] is False
    assert plan["held_out_test_opened"] is False


def test_observation_falsifies_mechanisms_and_replans(tmp_path: Path) -> None:
    source = tmp_path / "adaptive_search_plan.json"
    observations = tmp_path / "observations.json"
    output = tmp_path / "mechanism_plan.json"
    _write(source, _search_plan())
    _write(observations, {"interchange_query_state_at_mlp": "no_slot_switch"})

    plan = generate_mechanism_plan(source, output, observations_path=observations)

    assert plan["state"]["surviving_hypotheses"] == ["heads_match_mlp_value"]
    assert plan["state"]["rejected_hypotheses"] == [
        "mlp_match_heads_route",
        "split_head_match_route",
    ]
    assert plan["next_experiment"] is None
    assert plan["observations"] == {"interchange_query_state_at_mlp": "no_slot_switch"}


def test_generate_mechanism_plan_refuses_overwrite_without_force(tmp_path: Path) -> None:
    source = tmp_path / "adaptive_search_plan.json"
    output = tmp_path / "mechanism_plan.json"
    _write(source, _search_plan())
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output exists"):
        generate_mechanism_plan(source, output)

    assert output.read_text(encoding="utf-8") == "existing\n"
