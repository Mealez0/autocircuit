from __future__ import annotations

import pytest
from autocircuit.mechanism_synthesis import evaluate_campaign, select_next_experiment
from autocircuit.mechanisms import (
    MechanismCampaign,
    MechanismExperiment,
    MechanismHypothesis,
    MechanismPrediction,
    MechanismStep,
    MechanismVariable,
)


def _variables() -> tuple[MechanismVariable, ...]:
    return (
        MechanismVariable("query_key", "input"),
        MechanismVariable("match_slot", "latent"),
        MechanismVariable("retrieved_value", "latent"),
        MechanismVariable("answer", "output"),
    )


def _steps() -> tuple[MechanismStep, ...]:
    return (
        MechanismStep("match", "match", ("query_key",), "match_slot", ("head_3",)),
        MechanismStep(
            "retrieve",
            "retrieve",
            ("match_slot",),
            "retrieved_value",
            ("head_6",),
        ),
        MechanismStep(
            "decode",
            "decode",
            ("retrieved_value",),
            "answer",
            ("mlp_5",),
        ),
    )


def _hypothesis(hypothesis_id: str, outcomes: tuple[str, str]) -> MechanismHypothesis:
    return MechanismHypothesis(
        hypothesis_id=hypothesis_id,
        description=f"candidate {hypothesis_id}",
        variables=_variables(),
        steps=_steps(),
        predictions=(
            MechanismPrediction("e1", outcomes[0], "distinguishing prediction"),
            MechanismPrediction("e2", outcomes[1], "distinguishing prediction"),
        ),
    )


def _campaign() -> MechanismCampaign:
    return MechanismCampaign(
        campaign_id="assoc-recall-demo",
        task="associative_recall",
        hypotheses=(
            _hypothesis("h1", ("left", "same")),
            _hypothesis("h2", ("right", "same")),
            _hypothesis("h3", ("center", "different")),
        ),
        experiments=(
            MechanismExperiment("e1", "interchange", "query_key", "selected_slot", 1.0),
            MechanismExperiment("e2", "suppress", "retrieved_value", "answer", 1.0),
        ),
    )


def test_mechanism_hypothesis_is_a_validated_dag_with_stable_fingerprint() -> None:
    hypothesis = _hypothesis("h1", ("left", "same"))
    assert hypothesis.topological_steps() == ("match", "retrieve", "decode")
    assert hypothesis.fingerprint() == hypothesis.fingerprint()
    assert len(hypothesis.fingerprint()) == 64


def test_mechanism_hypothesis_rejects_cycles_fail_closed() -> None:
    with pytest.raises(ValueError, match="cycle"):
        MechanismHypothesis(
            hypothesis_id="cyclic",
            description="invalid",
            variables=(
                MechanismVariable("a", "latent"),
                MechanismVariable("b", "latent"),
            ),
            steps=(
                MechanismStep("a_from_b", "copy", ("b",), "a"),
                MechanismStep("b_from_a", "copy", ("a",), "b"),
            ),
            predictions=(MechanismPrediction("e1", "x", "invalid"),),
        )


def test_campaign_requires_complete_prediction_matrix() -> None:
    incomplete = MechanismHypothesis(
        hypothesis_id="incomplete",
        description="missing experiment prediction",
        variables=_variables(),
        steps=_steps(),
        predictions=(MechanismPrediction("e1", "left", "only one"),),
    )
    with pytest.raises(ValueError, match="prediction matrix"):
        MechanismCampaign(
            campaign_id="broken",
            task="associative_recall",
            hypotheses=(incomplete,),
            experiments=(
                MechanismExperiment("e1", "interchange", "query_key", "slot", 1.0),
                MechanismExperiment("e2", "suppress", "answer", "answer", 1.0),
            ),
        )


def test_falsification_rejects_only_hypotheses_that_conflict_with_observation() -> None:
    state = evaluate_campaign(_campaign(), {"e1": "left"})
    assert state["surviving_hypotheses"] == ["h1"]
    assert state["rejected_hypotheses"] == ["h2", "h3"]
    assert state["scientific_confirmation"] is False
    assert state["circuit_found"] is False


def test_information_gain_planner_prefers_experiment_that_separates_all_hypotheses() -> None:
    proposal = select_next_experiment(_campaign(), observations={})
    assert proposal is not None
    assert proposal["experiment_id"] == "e1"
    assert proposal["expected_information_gain_bits"] == pytest.approx(1.584962500721156)
    assert proposal["evidence_status"] == "proposal_only_not_evidence"


def test_no_experiment_is_selected_once_only_one_hypothesis_survives() -> None:
    assert select_next_experiment(_campaign(), observations={"e1": "left"}) is None
