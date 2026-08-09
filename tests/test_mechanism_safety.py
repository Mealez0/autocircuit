from __future__ import annotations

from pathlib import Path

import pytest

from autocircuit.mechanism_plan import generate_mechanism_plan
from autocircuit.mechanism_synthesis import (
    build_associative_recall_campaign,
    evaluate_campaign,
    select_next_experiment,
)
from autocircuit.mechanisms import (
    MechanismCampaign,
    MechanismExperiment,
    MechanismHypothesis,
    MechanismPrediction,
    MechanismStep,
    MechanismVariable,
)


def _simple_hypothesis(hypothesis_id: str, outcomes: tuple[str, str]) -> MechanismHypothesis:
    variables = (
        MechanismVariable("input_state", "input"),
        MechanismVariable("answer", "output"),
    )
    return MechanismHypothesis(
        hypothesis_id=hypothesis_id,
        description=hypothesis_id,
        variables=variables,
        steps=(MechanismStep("decode", "decode", ("input_state",), "answer"),),
        predictions=(
            MechanismPrediction("cheap", outcomes[0], "cheap discriminator"),
            MechanismPrediction("expensive", outcomes[1], "expensive discriminator"),
        ),
    )


def _cost_campaign() -> MechanismCampaign:
    return MechanismCampaign(
        campaign_id="cost-test",
        task="associative_recall",
        hypotheses=(
            _simple_hypothesis("h1", ("left", "left")),
            _simple_hypothesis("h2", ("right", "right")),
        ),
        experiments=(
            MechanismExperiment("cheap", "interchange", "input_state", "answer", 0.5),
            MechanismExperiment("expensive", "interchange", "input_state", "answer", 2.0),
        ),
    )


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


def test_unknown_observation_fails_closed() -> None:
    campaign = _cost_campaign()
    with pytest.raises(ValueError, match="unknown mechanism experiment"):
        evaluate_campaign(campaign, {"not_registered": "left"})


def test_all_hypotheses_rejected_stops_planner_without_fabricating_winner() -> None:
    campaign = _cost_campaign()
    observations = {"cheap": "unpredicted"}
    state = evaluate_campaign(campaign, observations)

    assert state["surviving_hypotheses"] == []
    assert state["all_hypotheses_rejected"] is True
    assert state["scientific_confirmation"] is False
    assert select_next_experiment(campaign, observations) is None


def test_equal_information_gain_prefers_lower_cost_experiment() -> None:
    proposal = select_next_experiment(_cost_campaign(), {})
    assert proposal is not None
    assert proposal["experiment_id"] == "cheap"
    assert proposal["expected_information_gain_bits"] == pytest.approx(1.0)
    assert proposal["information_gain_per_cost"] == pytest.approx(2.0)


def test_campaign_requires_two_stable_positive_heads() -> None:
    plan = _search_plan()
    rows = plan["ranked_head_evidence"]
    assert isinstance(rows, list)
    rows[1]["stable_positive"] = False

    with pytest.raises(ValueError, match="two stable-positive"):
        build_associative_recall_campaign(plan)


def test_campaign_rejects_nonpositive_component_evidence() -> None:
    plan = _search_plan()
    context = plan["component_context"]
    assert isinstance(context, dict)
    context["mlp_family_specific_advantage"] = 0.0

    with pytest.raises(ValueError, match="positive attention and MLP"):
        build_associative_recall_campaign(plan)


def test_force_cannot_overwrite_source_search_plan(tmp_path: Path) -> None:
    import json

    source = tmp_path / "adaptive_search_plan.json"
    source.write_text(json.dumps(_search_plan()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot replace its source"):
        generate_mechanism_plan(source, source, force=True)
