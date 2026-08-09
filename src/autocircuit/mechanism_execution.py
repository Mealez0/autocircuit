"""Bind a selected mechanism falsifier to exact alignments and discovery inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from autocircuit.mechanism_artifacts import load_alignment_manifest
from autocircuit.mechanism_compile import InterventionRecipe, compile_experiment
from autocircuit.mechanism_counterfactuals import (
    CounterfactualPair,
    counterfactual_kind_for_experiment,
)
from autocircuit.mechanisms import (
    MechanismCampaign,
    MechanismExperiment,
    MechanismHypothesis,
    MechanismPrediction,
    MechanismStep,
    MechanismVariable,
)

INTERPRETATION_SCOPE = "exploratory_discovery_only"


def _guard(value: Mapping[str, Any], label: str) -> None:
    if value.get("interpretation_scope") != INTERPRETATION_SCOPE:
        raise ValueError(f"{label} is not discovery-only exploratory data")
    if value.get("held_out_validation_reused") is not False:
        raise ValueError(f"{label} indicates held-out validation reuse")
    if value.get("held_out_test_opened") is not False:
        raise ValueError(f"{label} indicates held-out test access")
    if value.get("scientific_confirmation") is not False:
        raise ValueError(f"{label} cannot claim scientific confirmation")
    if value.get("circuit_found") is not False:
        raise ValueError(f"{label} cannot claim a discovered circuit")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _optional_nonnegative_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _variables(raw: Any) -> tuple[MechanismVariable, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("serialized mechanism variables are malformed")
    values: list[MechanismVariable] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("serialized mechanism variable is malformed")
        values.append(
            MechanismVariable(
                _string(item.get("name"), "mechanism variable name"),
                _string(item.get("role"), "mechanism variable role"),
            )
        )
    return tuple(values)


def _steps(raw: Any) -> tuple[MechanismStep, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("serialized mechanism steps are malformed")
    values: list[MechanismStep] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("serialized mechanism step is malformed")
        inputs = item.get("inputs")
        carriers = item.get("carrier_hints")
        if not isinstance(inputs, list) or not all(isinstance(value, str) for value in inputs):
            raise ValueError("serialized mechanism step inputs are malformed")
        if not isinstance(carriers, list) or not all(
            isinstance(value, str) for value in carriers
        ):
            raise ValueError("serialized mechanism carrier hints are malformed")
        values.append(
            MechanismStep(
                _string(item.get("step_id"), "mechanism step id"),
                _string(item.get("operation"), "mechanism step operation"),
                tuple(inputs),
                _string(item.get("output"), "mechanism step output"),
                tuple(carriers),
            )
        )
    return tuple(values)


def _predictions(raw: Any) -> tuple[MechanismPrediction, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("serialized mechanism predictions are malformed")
    values: list[MechanismPrediction] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("serialized mechanism prediction is malformed")
        values.append(
            MechanismPrediction(
                _string(item.get("experiment_id"), "prediction experiment id"),
                _string(item.get("outcome"), "prediction outcome"),
                _string(item.get("rationale"), "prediction rationale"),
            )
        )
    return tuple(values)


def _hypotheses(raw: Any) -> tuple[MechanismHypothesis, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("serialized mechanism hypotheses are malformed")
    values: list[MechanismHypothesis] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("serialized mechanism hypothesis is malformed")
        if item.get("evidence_status") != "proposal_only_not_evidence":
            raise ValueError("serialized mechanism hypothesis upgraded its evidence status")
        hypothesis = MechanismHypothesis(
            hypothesis_id=_string(item.get("hypothesis_id"), "hypothesis id"),
            description=_string(item.get("description"), "hypothesis description"),
            variables=_variables(item.get("variables")),
            steps=_steps(item.get("steps")),
            predictions=_predictions(item.get("predictions")),
        )
        topological = item.get("topological_steps")
        if (
            not isinstance(topological, list)
            or tuple(topological) != hypothesis.topological_steps()
        ):
            raise ValueError("serialized mechanism topological order is inconsistent")
        values.append(hypothesis)
    return tuple(values)


def _experiments(raw: Any) -> tuple[MechanismExperiment, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("serialized mechanism experiments are malformed")
    values: list[MechanismExperiment] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("serialized mechanism experiment is malformed")
        if item.get("evidence_status") != "proposal_only_not_evidence":
            raise ValueError("serialized mechanism experiment upgraded its evidence status")
        cost = item.get("cost")
        if not isinstance(cost, int | float) or isinstance(cost, bool):
            raise ValueError("serialized mechanism experiment cost is malformed")
        values.append(
            MechanismExperiment(
                experiment_id=_string(item.get("experiment_id"), "experiment id"),
                intervention_kind=_string(
                    item.get("intervention_kind"), "experiment intervention kind"
                ),
                target=_string(item.get("target"), "experiment target"),
                readout=_string(item.get("readout"), "experiment readout"),
                cost=float(cost),
                rationale=_string(item.get("rationale"), "experiment rationale"),
            )
        )
    return tuple(values)


def _campaign(raw: Any) -> MechanismCampaign:
    if not isinstance(raw, dict):
        raise ValueError("mechanism plan campaign is malformed")
    _guard(raw, "mechanism campaign")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported serialized mechanism campaign schema")
    selected_layer = raw.get("selected_layer")
    if selected_layer is not None and (
        not isinstance(selected_layer, int) or isinstance(selected_layer, bool)
    ):
        raise ValueError("serialized mechanism selected layer is malformed")
    carriers = raw.get("candidate_carriers")
    if not isinstance(carriers, dict):
        raise ValueError("serialized mechanism candidate carriers are malformed")
    campaign = MechanismCampaign(
        campaign_id=_string(raw.get("campaign_id"), "campaign id"),
        task=_string(raw.get("task"), "campaign task"),
        hypotheses=_hypotheses(raw.get("hypotheses")),
        experiments=_experiments(raw.get("experiments")),
        selected_layer=selected_layer,
        candidate_carriers=dict(carriers),
    )
    if campaign.to_dict() != dict(raw):
        raise ValueError("serialized mechanism campaign does not round-trip exactly")
    return campaign


def _counterfactuals(raw_manifest: Mapping[str, Any]) -> tuple[CounterfactualPair, ...]:
    _guard(raw_manifest, "counterfactual manifest")
    raw_pairs = raw_manifest.get("counterfactuals")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("counterfactual manifest contains no pairs")
    pairs: list[CounterfactualPair] = []
    for raw in raw_pairs:
        if not isinstance(raw, dict):
            raise ValueError("serialized counterfactual pair is malformed")
        changed = raw.get("changed_variables")
        if not isinstance(changed, list | tuple) or not all(
            isinstance(value, str) for value in changed
        ):
            raise ValueError("serialized counterfactual changed variables are malformed")
        pair = CounterfactualPair(
            counterfactual_id=_string(raw.get("counterfactual_id"), "counterfactual id"),
            source_example_id=_string(raw.get("source_example_id"), "source example id"),
            source_family_id=_string(raw.get("source_family_id"), "source family id"),
            split=_string(raw.get("split"), "counterfactual split"),
            kind=_string(raw.get("kind"), "counterfactual kind"),
            base_prompt=_string(raw.get("base_prompt"), "counterfactual base prompt"),
            donor_prompt=_string(raw.get("donor_prompt"), "counterfactual donor prompt"),
            base_query_entity=_string(raw.get("base_query_entity"), "base query entity"),
            donor_query_entity=_string(raw.get("donor_query_entity"), "donor query entity"),
            base_target_text=_string(raw.get("base_target_text"), "base target text"),
            donor_target_text=_string(raw.get("donor_target_text"), "donor target text"),
            base_target_token_id=_integer(
                raw.get("base_target_token_id"), "base target token id"
            ),
            donor_target_token_id=_integer(
                raw.get("donor_target_token_id"), "donor target token id"
            ),
            changed_variables=tuple(changed),
            prompt_token_length=_integer(
                raw.get("prompt_token_length"), "counterfactual prompt token length"
            ),
            base_query_fact_index=_optional_nonnegative_integer(
                raw.get("base_query_fact_index"), "base query fact index"
            ),
            donor_query_fact_index=_optional_nonnegative_integer(
                raw.get("donor_query_fact_index"), "donor query fact index"
            ),
            evidence_status=_string(raw.get("evidence_status"), "counterfactual evidence status"),
        )
        pairs.append(pair)
    ids = [pair.counterfactual_id for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("counterfactual manifest contains duplicate ids")
    return tuple(sorted(pairs, key=lambda pair: pair.counterfactual_id))


@dataclass(frozen=True)
class PreparedMechanismExperiment:
    """Runtime-ready experiment selection before any model or CUDA execution."""

    experiment_id: str
    recipe: InterventionRecipe
    input_mode: str
    counterfactual_kind: str | None
    counterfactuals: tuple[CounterfactualPair, ...]

    def __post_init__(self) -> None:
        if self.input_mode not in {"base_donor", "base_only"}:
            raise ValueError("prepared mechanism input mode is invalid")
        if not self.counterfactuals:
            raise ValueError("prepared mechanism experiment has no discovery inputs")
        if self.input_mode == "base_donor" and self.counterfactual_kind is None:
            raise ValueError("base/donor execution requires a counterfactual kind")
        if self.input_mode == "base_only" and self.counterfactual_kind is not None:
            raise ValueError("base-only execution cannot require a donor kind")

    def to_manifest(self) -> dict[str, Any]:
        alignment_fingerprint = (
            self.recipe.alignment.fingerprint() if self.recipe.alignment is not None else None
        )
        return {
            "schema_version": 1,
            "interpretation_scope": INTERPRETATION_SCOPE,
            "experiment_id": self.experiment_id,
            "input_mode": self.input_mode,
            "counterfactual_kind": self.counterfactual_kind,
            "counterfactual_ids": [pair.counterfactual_id for pair in self.counterfactuals],
            "hook_site": self.recipe.hook_site,
            "position_index": self.recipe.position_index,
            "selected_heads": list(self.recipe.selected_heads),
            "operation": self.recipe.operation,
            "readout": self.recipe.readout,
            "alignment_fingerprint": alignment_fingerprint,
            "evidence_status": "proposal_only_not_evidence",
            "held_out_validation_reused": False,
            "held_out_test_opened": False,
            "scientific_confirmation": False,
            "circuit_found": False,
        }


def _base_only_inputs(pairs: Sequence[CounterfactualPair]) -> tuple[CounterfactualPair, ...]:
    chosen: dict[str, CounterfactualPair] = {}
    for pair in sorted(pairs, key=lambda item: (item.source_example_id, item.counterfactual_id)):
        chosen.setdefault(pair.source_example_id, pair)
    return tuple(chosen[key] for key in sorted(chosen))


def prepare_execution_bundle(
    mechanism_plan: Mapping[str, Any],
    counterfactual_manifest: Mapping[str, Any],
    alignment_manifest: Mapping[str, Any],
) -> PreparedMechanismExperiment:
    """Validate all upstream artifacts and bind the selected falsifier to exact inputs."""

    _guard(mechanism_plan, "mechanism plan")
    campaign = _campaign(mechanism_plan.get("campaign"))
    state = mechanism_plan.get("state")
    proposal = mechanism_plan.get("next_experiment")
    if not isinstance(state, dict):
        raise ValueError("mechanism plan state is malformed")
    survivors = state.get("surviving_hypotheses")
    if not isinstance(survivors, list) or len(survivors) < 2:
        raise ValueError("mechanism execution requires at least two surviving hypotheses")
    if not isinstance(proposal, dict):
        raise ValueError("mechanism plan has no executable next experiment")
    experiment_id = _string(proposal.get("experiment_id"), "next experiment id")

    alignments = load_alignment_manifest(alignment_manifest)
    recipe = compile_experiment(campaign, experiment_id, alignments)
    pairs = _counterfactuals(counterfactual_manifest)
    kind = counterfactual_kind_for_experiment(experiment_id)
    if kind is None:
        selected = _base_only_inputs(pairs)
        input_mode = "base_only"
    else:
        selected = tuple(pair for pair in pairs if pair.kind == kind)
        if not selected:
            raise ValueError(f"counterfactual manifest contains no required {kind} pairs")
        input_mode = "base_donor"
    return PreparedMechanismExperiment(
        experiment_id=experiment_id,
        recipe=recipe,
        input_mode=input_mode,
        counterfactual_kind=kind,
        counterfactuals=selected,
    )
