"""Discovery-only mechanism hypothesis synthesis and active falsification planning."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from autocircuit.mechanisms import (
    MechanismCampaign,
    MechanismExperiment,
    MechanismHypothesis,
    MechanismPrediction,
    MechanismStep,
    MechanismVariable,
)

SYNTHESIS_VERSION = "mechanism-synthesis-0.1.0"
INTERPRETATION_SCOPE = "exploratory_discovery_only"


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_discovery_plan(search_plan: Mapping[str, Any]) -> None:
    if search_plan.get("interpretation_scope") != INTERPRETATION_SCOPE:
        raise ValueError("mechanism synthesis requires discovery-only exploratory evidence")
    if search_plan.get("held_out_validation_reused") is not False:
        raise ValueError("held-out validation cannot be reused for mechanism synthesis")
    if search_plan.get("held_out_test_opened") is not False:
        raise ValueError("held-out test data cannot be opened for mechanism synthesis")
    if search_plan.get("scientific_confirmation") is not False:
        raise ValueError("exploratory search plan cannot claim scientific confirmation")
    if search_plan.get("circuit_found") is not False:
        raise ValueError("exploratory search plan cannot claim a discovered circuit")


def _candidate_heads(search_plan: Mapping[str, Any]) -> tuple[int, int]:
    rows = search_plan.get("ranked_head_evidence")
    if not isinstance(rows, list):
        raise ValueError("search plan has no ranked head evidence")
    stable: list[int] = []
    seen: set[int] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"ranked head evidence row {row_index} is malformed")
        head = _integer(row.get("head_index"), f"ranked head {row_index} index")
        if head in seen:
            raise ValueError("ranked head evidence contains duplicate head indexes")
        seen.add(head)
        _finite(row.get("family_specific_advantage"), f"head {head} advantage")
        interval = row.get("family_specific_advantage_ci_95")
        if not isinstance(interval, list | tuple) or len(interval) != 2:
            raise ValueError(f"head {head} confidence interval is malformed")
        low = _finite(interval[0], f"head {head} confidence interval lower bound")
        high = _finite(interval[1], f"head {head} confidence interval upper bound")
        if low > high:
            raise ValueError(f"head {head} confidence interval is reversed")
        if row.get("stable_positive") is True and low > 0.0:
            stable.append(head)
    if len(stable) < 2:
        raise ValueError("mechanism synthesis requires two stable-positive attention heads")
    return stable[0], stable[1]


def _component_context(search_plan: Mapping[str, Any]) -> tuple[float, float]:
    context = search_plan.get("component_context")
    if not isinstance(context, dict):
        raise ValueError("search plan component context is missing")
    attention = _finite(
        context.get("attention_family_specific_advantage"),
        "attention family-specific advantage",
    )
    mlp = _finite(
        context.get("mlp_family_specific_advantage"),
        "MLP family-specific advantage",
    )
    if attention <= 0.0 or mlp <= 0.0:
        raise ValueError("mechanism grammar requires positive attention and MLP discovery evidence")
    return attention, mlp


def _variables() -> tuple[MechanismVariable, ...]:
    return (
        MechanismVariable("query_key", "input"),
        MechanismVariable("match_slot", "latent"),
        MechanismVariable("retrieved_value", "latent"),
        MechanismVariable("routed_value", "latent"),
        MechanismVariable("answer", "output"),
    )


def _experiments() -> tuple[MechanismExperiment, ...]:
    return (
        MechanismExperiment(
            "interchange_query_state_at_heads",
            "interchange",
            "head_query_state",
            "selected_slot",
            1.0,
            "test whether attention carries the query-to-slot selection state",
        ),
        MechanismExperiment(
            "interchange_query_state_at_mlp",
            "interchange",
            "mlp_query_state",
            "selected_slot",
            1.0,
            "test whether the MLP carries selection state or only value transformation",
        ),
        MechanismExperiment(
            "suppress_secondary_head",
            "suppress",
            "secondary_head",
            "slot_and_answer",
            0.8,
            "separate joint matching from downstream value routing",
        ),
        MechanismExperiment(
            "interchange_value_state_at_mlp",
            "interchange",
            "mlp_value_state",
            "slot_and_answer",
            1.0,
            "test whether MLP state changes value identity while preserving selected slot",
        ),
    )


def _predictions(rows: tuple[tuple[str, str], ...]) -> tuple[MechanismPrediction, ...]:
    return tuple(
        MechanismPrediction(experiment_id, outcome, "predeclared mechanism discriminator")
        for experiment_id, outcome in rows
    )


def build_associative_recall_campaign(search_plan: Mapping[str, Any]) -> MechanismCampaign:
    """Synthesize explicit competing causal programs from discovery-only carriers.

    The discovered components are used only as carrier hints. Their computational
    roles remain hypotheses until counterfactual interventions discriminate them.
    """

    _validate_discovery_plan(search_plan)
    selected_layer = _integer(search_plan.get("selected_layer"), "selected layer")
    primary_head, secondary_head = _candidate_heads(search_plan)
    attention_advantage, mlp_advantage = _component_context(search_plan)
    primary = f"head_{primary_head}"
    secondary = f"head_{secondary_head}"
    mlp = f"mlp_{selected_layer}"
    variables = _variables()

    heads_match_mlp_value = MechanismHypothesis(
        hypothesis_id="heads_match_mlp_value",
        description=(
            "attention jointly selects the matching fact slot; the MLP forms the retrieved "
            "value state before answer decoding"
        ),
        variables=variables,
        steps=(
            MechanismStep("match", "match", ("query_key",), "match_slot", (primary, secondary)),
            MechanismStep("retrieve", "retrieve", ("match_slot",), "retrieved_value", (mlp,)),
            MechanismStep(
                "route", "route", ("retrieved_value",), "routed_value", (primary, secondary)
            ),
            MechanismStep("decode", "decode", ("routed_value",), "answer", (mlp,)),
        ),
        predictions=_predictions(
            (
                ("interchange_query_state_at_heads", "donor_slot_switch"),
                ("interchange_query_state_at_mlp", "no_slot_switch"),
                ("suppress_secondary_head", "slot_and_answer_degrade"),
                ("interchange_value_state_at_mlp", "donor_value_recipient_slot"),
            )
        ),
    )
    mlp_match_heads_route = MechanismHypothesis(
        hypothesis_id="mlp_match_heads_route",
        description=(
            "the MLP computes the query-to-fact selection and value state; attention heads "
            "primarily route that state toward the answer"
        ),
        variables=variables,
        steps=(
            MechanismStep("match", "match", ("query_key",), "match_slot", (mlp,)),
            MechanismStep("retrieve", "retrieve", ("match_slot",), "retrieved_value", (mlp,)),
            MechanismStep(
                "route", "route", ("retrieved_value",), "routed_value", (primary, secondary)
            ),
            MechanismStep("decode", "decode", ("routed_value",), "answer", (mlp,)),
        ),
        predictions=_predictions(
            (
                ("interchange_query_state_at_heads", "no_slot_switch"),
                ("interchange_query_state_at_mlp", "donor_slot_switch"),
                ("suppress_secondary_head", "slot_preserved_answer_degrades"),
                ("interchange_value_state_at_mlp", "donor_slot_and_value_switch"),
            )
        ),
    )
    split_head_match_route = MechanismHypothesis(
        hypothesis_id="split_head_match_route",
        description=(
            "the primary head carries matching state, the secondary head routes retrieved "
            "value information, and the MLP integrates the routed value"
        ),
        variables=variables,
        steps=(
            MechanismStep("match", "match", ("query_key",), "match_slot", (primary,)),
            MechanismStep("retrieve", "retrieve", ("match_slot",), "retrieved_value", (mlp,)),
            MechanismStep("route", "route", ("retrieved_value",), "routed_value", (secondary,)),
            MechanismStep("decode", "decode", ("routed_value",), "answer", (mlp,)),
        ),
        predictions=_predictions(
            (
                ("interchange_query_state_at_heads", "donor_slot_switch"),
                (
                    "interchange_query_state_at_mlp",
                    "value_only_change_without_slot_switch",
                ),
                ("suppress_secondary_head", "slot_preserved_answer_degrades"),
                ("interchange_value_state_at_mlp", "donor_value_recipient_slot"),
            )
        ),
    )

    return MechanismCampaign(
        campaign_id="associative_recall_mechanism_v1",
        task="associative_recall",
        hypotheses=(heads_match_mlp_value, mlp_match_heads_route, split_head_match_route),
        experiments=_experiments(),
        selected_layer=selected_layer,
        candidate_carriers={
            "attention_heads": [primary_head, secondary_head],
            "mlp_layer": selected_layer,
            "attention_family_specific_advantage": attention_advantage,
            "mlp_family_specific_advantage": mlp_advantage,
            "role_assignment_status": "hypothesis_only_not_evidence",
        },
    )


def _validate_observations(
    campaign: MechanismCampaign, observations: Mapping[str, str]
) -> dict[str, str]:
    experiment_ids = {experiment.experiment_id for experiment in campaign.experiments}
    output: dict[str, str] = {}
    for experiment_id, outcome in observations.items():
        if experiment_id not in experiment_ids:
            raise ValueError(f"unknown mechanism experiment observation: {experiment_id}")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError("mechanism observation outcome must be a non-empty string")
        output[experiment_id] = outcome
    return output


def evaluate_campaign(
    campaign: MechanismCampaign, observations: Mapping[str, str]
) -> dict[str, Any]:
    """Falsify hypotheses whose preregistered predictions contradict observations."""

    checked = _validate_observations(campaign, observations)
    surviving: list[str] = []
    rejected: list[str] = []
    rejection_reasons: dict[str, list[dict[str, str]]] = {}
    for hypothesis in campaign.hypotheses:
        predictions = hypothesis.prediction_map()
        conflicts = [
            {
                "experiment_id": experiment_id,
                "predicted": predictions[experiment_id],
                "observed": outcome,
            }
            for experiment_id, outcome in checked.items()
            if predictions[experiment_id] != outcome
        ]
        if conflicts:
            rejected.append(hypothesis.hypothesis_id)
            rejection_reasons[hypothesis.hypothesis_id] = conflicts
        else:
            surviving.append(hypothesis.hypothesis_id)
    return {
        "synthesis_version": SYNTHESIS_VERSION,
        "observations_evaluated": len(checked),
        "surviving_hypotheses": surviving,
        "rejected_hypotheses": rejected,
        "rejection_reasons": rejection_reasons,
        "all_hypotheses_rejected": not surviving,
        "evidence_scope": INTERPRETATION_SCOPE,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def _entropy(probabilities: list[float]) -> float:
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability
    )


def select_next_experiment(
    campaign: MechanismCampaign, observations: Mapping[str, str]
) -> dict[str, Any] | None:
    """Choose the unobserved experiment with maximum information gain per unit cost."""

    checked = _validate_observations(campaign, observations)
    state = evaluate_campaign(campaign, checked)
    survivor_ids = set(state["surviving_hypotheses"])
    survivors = [
        hypothesis for hypothesis in campaign.hypotheses if hypothesis.hypothesis_id in survivor_ids
    ]
    if len(survivors) <= 1:
        return None

    candidate_scores: list[
        tuple[float, float, float, str, MechanismExperiment, dict[str, list[str]]]
    ] = []
    for experiment in campaign.experiments:
        if experiment.experiment_id in checked:
            continue
        partitions: dict[str, list[str]] = defaultdict(list)
        for hypothesis in survivors:
            outcome = hypothesis.prediction_map()[experiment.experiment_id]
            partitions[outcome].append(hypothesis.hypothesis_id)
        total = float(len(survivors))
        probabilities = [len(group) / total for group in partitions.values()]
        information_gain = _entropy(probabilities)
        utility = information_gain / float(experiment.cost)
        candidate_scores.append(
            (
                utility,
                information_gain,
                float(experiment.cost),
                experiment.experiment_id,
                experiment,
                dict(sorted(partitions.items())),
            )
        )
    if not candidate_scores:
        return None
    candidate_scores.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    utility, information_gain, _, _, experiment, partitions = candidate_scores[0]
    return {
        **experiment.to_dict(),
        "expected_information_gain_bits": information_gain,
        "information_gain_per_cost": utility,
        "surviving_hypothesis_count": len(survivors),
        "predicted_outcome_partition": partitions,
        "selection_rule": "max_expected_information_gain_per_cost",
        "evidence_status": "proposal_only_not_evidence",
    }
