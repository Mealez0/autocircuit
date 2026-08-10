"""Operationalize mechanism predictions with discovery-only measurable outcomes."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

INCONCLUSIVE = "inconclusive"
_QUERY_EXPERIMENTS = {
    "interchange_query_state_at_heads",
    "interchange_query_state_at_mlp",
}


@dataclass(frozen=True)
class ObservationPolicy:
    """Frozen thresholds for turning per-pair measurements into one observation."""

    min_evaluable_pairs: int = 8
    min_consensus_fraction: float = 0.75
    logit_tie_tolerance: float = 1e-6
    slot_distance_tolerance: float = 1e-9
    answer_degradation_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_evaluable_pairs, int)
            or isinstance(self.min_evaluable_pairs, bool)
            or self.min_evaluable_pairs <= 0
        ):
            raise ValueError("min_evaluable_pairs must be a positive integer")
        if (
            isinstance(self.min_consensus_fraction, bool)
            or not isinstance(self.min_consensus_fraction, (int, float))
            or not math.isfinite(float(self.min_consensus_fraction))
            or not 0.5 < float(self.min_consensus_fraction) <= 1.0
        ):
            raise ValueError("min_consensus_fraction must lie in (0.5, 1]")
        for label, value in (
            ("logit_tie_tolerance", self.logit_tie_tolerance),
            ("slot_distance_tolerance", self.slot_distance_tolerance),
            ("answer_degradation_tolerance", self.answer_degradation_tolerance),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{label} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_evaluable_pairs": self.min_evaluable_pairs,
            "min_consensus_fraction": float(self.min_consensus_fraction),
            "logit_tie_tolerance": float(self.logit_tie_tolerance),
            "slot_distance_tolerance": float(self.slot_distance_tolerance),
            "answer_degradation_tolerance": float(self.answer_degradation_tolerance),
        }


@dataclass(frozen=True)
class PairMechanismMeasurement:
    """The two independent readouts used to classify one counterfactual pair."""

    counterfactual_id: str
    experiment_id: str
    base_margin: float
    patched_base_target_logit: float
    patched_donor_target_logit: float
    slot_distance_to_base: float
    slot_distance_to_donor: float

    def __post_init__(self) -> None:
        if not isinstance(self.counterfactual_id, str) or not self.counterfactual_id:
            raise ValueError("counterfactual id must be non-empty")
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("experiment id must be non-empty")
        for label, value in (
            ("base margin", self.base_margin),
            ("patched base-target logit", self.patched_base_target_logit),
            ("patched donor-target logit", self.patched_donor_target_logit),
            ("slot distance to base", self.slot_distance_to_base),
            ("slot distance to donor", self.slot_distance_to_donor),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{label} must be finite")
        if self.slot_distance_to_base < 0.0 or self.slot_distance_to_donor < 0.0:
            raise ValueError("slot distances must be non-negative")

    @property
    def patched_margin(self) -> float:
        return float(self.patched_base_target_logit) - float(self.patched_donor_target_logit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counterfactual_id": self.counterfactual_id,
            "experiment_id": self.experiment_id,
            "base_margin": float(self.base_margin),
            "patched_base_target_logit": float(self.patched_base_target_logit),
            "patched_donor_target_logit": float(self.patched_donor_target_logit),
            "patched_margin": self.patched_margin,
            "slot_distance_to_base": float(self.slot_distance_to_base),
            "slot_distance_to_donor": float(self.slot_distance_to_donor),
        }


def _answer_preference(
    measurement: PairMechanismMeasurement, policy: ObservationPolicy
) -> str:
    delta = (
        float(measurement.patched_donor_target_logit)
        - float(measurement.patched_base_target_logit)
    )
    tolerance = float(policy.logit_tie_tolerance)
    if delta > tolerance:
        return "donor"
    if delta < -tolerance:
        return "base"
    return "tie"


def _slot_preference(
    measurement: PairMechanismMeasurement, policy: ObservationPolicy
) -> str:
    base = float(measurement.slot_distance_to_base)
    donor = float(measurement.slot_distance_to_donor)
    tolerance = float(policy.slot_distance_tolerance)
    if base + tolerance < donor:
        return "base"
    if donor + tolerance < base:
        return "donor"
    return "tie"


def classify_pair_measurement(
    measurement: PairMechanismMeasurement,
    *,
    policy: ObservationPolicy | None = None,
) -> str:
    """Map one preregistered pair measurement to an operational mechanism outcome."""

    selected = policy if policy is not None else ObservationPolicy()
    answer = _answer_preference(measurement, selected)
    slot = _slot_preference(measurement, selected)
    if answer == "tie" or slot == "tie":
        return INCONCLUSIVE

    experiment_id = measurement.experiment_id
    if experiment_id in _QUERY_EXPERIMENTS:
        if slot == "donor" and answer == "donor":
            return "donor_slot_switch"
        if slot == "base" and answer == "base":
            return "no_slot_switch"
        if slot == "base" and answer == "donor":
            return "value_only_change_without_slot_switch"
        return INCONCLUSIVE

    if experiment_id == "interchange_value_state_at_mlp":
        if answer != "donor":
            return INCONCLUSIVE
        if slot == "base":
            return "donor_value_recipient_slot"
        if slot == "donor":
            return "donor_slot_and_value_switch"
        return INCONCLUSIVE

    if experiment_id == "suppress_secondary_head":
        degraded = measurement.patched_margin < (
            float(measurement.base_margin) - float(selected.answer_degradation_tolerance)
        )
        if not degraded:
            return INCONCLUSIVE
        if slot == "base":
            return "slot_preserved_answer_degrades"
        if slot == "donor":
            return "slot_and_answer_degrade"
        return INCONCLUSIVE

    raise ValueError(f"unknown mechanism experiment: {experiment_id}")


def aggregate_observation(
    outcomes: list[str] | tuple[str, ...],
    *,
    policy: ObservationPolicy | None = None,
) -> dict[str, Any]:
    """Require enough evaluable pairs and a deterministic consensus before replanning."""

    selected = policy if policy is not None else ObservationPolicy()
    if any(not isinstance(outcome, str) or not outcome for outcome in outcomes):
        raise ValueError("mechanism outcomes must be non-empty strings")
    evaluable = [outcome for outcome in outcomes if outcome != INCONCLUSIVE]
    inconclusive_count = len(outcomes) - len(evaluable)
    counts = Counter(evaluable)
    base = {
        "policy": selected.to_dict(),
        "total_pair_count": len(outcomes),
        "evaluable_pair_count": len(evaluable),
        "inconclusive_pair_count": inconclusive_count,
        "outcome_counts": dict(sorted(counts.items())),
        "scientific_confirmation": False,
        "circuit_found": False,
    }
    if len(evaluable) < selected.min_evaluable_pairs:
        return {
            **base,
            "status": "insufficient_evaluable_pairs",
            "observation": None,
            "consensus_fraction": None,
        }
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    winner, winner_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == winner_count:
        return {
            **base,
            "status": "inconclusive",
            "observation": None,
            "consensus_fraction": winner_count / len(evaluable),
        }
    consensus = winner_count / len(evaluable)
    if consensus < selected.min_consensus_fraction:
        return {
            **base,
            "status": "inconclusive",
            "observation": None,
            "consensus_fraction": consensus,
        }
    return {
        **base,
        "status": "observation_ready",
        "observation": winner,
        "consensus_fraction": consensus,
    }
