from __future__ import annotations

import pytest

from autocircuit.mechanism_outcomes import (
    ObservationPolicy,
    PairMechanismMeasurement,
    aggregate_observation,
    classify_pair_measurement,
)


def _measurement(
    experiment_id: str,
    *,
    base_margin: float = 2.0,
    patched_base_logit: float = 1.0,
    patched_donor_logit: float = 3.0,
    slot_distance_to_base: float = 2.0,
    slot_distance_to_donor: float = 0.1,
) -> PairMechanismMeasurement:
    return PairMechanismMeasurement(
        counterfactual_id="mcf-demo",
        experiment_id=experiment_id,
        base_margin=base_margin,
        patched_base_target_logit=patched_base_logit,
        patched_donor_target_logit=patched_donor_logit,
        slot_distance_to_base=slot_distance_to_base,
        slot_distance_to_donor=slot_distance_to_donor,
    )


def test_query_interchange_outcomes_are_jointly_defined_by_slot_and_answer() -> None:
    donor_switch = classify_pair_measurement(
        _measurement("interchange_query_state_at_mlp")
    )
    no_switch = classify_pair_measurement(
        _measurement(
            "interchange_query_state_at_mlp",
            patched_base_logit=4.0,
            patched_donor_logit=1.0,
            slot_distance_to_base=0.1,
            slot_distance_to_donor=2.0,
        )
    )
    value_only = classify_pair_measurement(
        _measurement(
            "interchange_query_state_at_mlp",
            patched_base_logit=1.0,
            patched_donor_logit=4.0,
            slot_distance_to_base=0.1,
            slot_distance_to_donor=2.0,
        )
    )

    assert donor_switch == "donor_slot_switch"
    assert no_switch == "no_slot_switch"
    assert value_only == "value_only_change_without_slot_switch"


def test_value_interchange_distinguishes_recipient_and_donor_slot() -> None:
    recipient = classify_pair_measurement(
        _measurement(
            "interchange_value_state_at_mlp",
            slot_distance_to_base=0.1,
            slot_distance_to_donor=2.0,
        )
    )
    donor = classify_pair_measurement(
        _measurement("interchange_value_state_at_mlp")
    )

    assert recipient == "donor_value_recipient_slot"
    assert donor == "donor_slot_and_value_switch"


def test_suppression_requires_answer_degradation_and_slot_readout() -> None:
    preserved = classify_pair_measurement(
        _measurement(
            "suppress_secondary_head",
            base_margin=3.0,
            patched_base_logit=1.2,
            patched_donor_logit=0.8,
            slot_distance_to_base=0.1,
            slot_distance_to_donor=2.0,
        )
    )
    degraded_slot = classify_pair_measurement(
        _measurement(
            "suppress_secondary_head",
            base_margin=3.0,
            patched_base_logit=1.2,
            patched_donor_logit=0.8,
            slot_distance_to_base=2.0,
            slot_distance_to_donor=0.1,
        )
    )
    no_answer_degradation = classify_pair_measurement(
        _measurement(
            "suppress_secondary_head",
            base_margin=0.2,
            patched_base_logit=2.0,
            patched_donor_logit=1.0,
            slot_distance_to_base=0.1,
            slot_distance_to_donor=2.0,
        )
    )

    assert preserved == "slot_preserved_answer_degrades"
    assert degraded_slot == "slot_and_answer_degrade"
    assert no_answer_degradation == "inconclusive"


def test_ties_and_nonfinite_measurements_fail_or_become_inconclusive() -> None:
    tied_answer = classify_pair_measurement(
        _measurement(
            "interchange_query_state_at_heads",
            patched_base_logit=1.0,
            patched_donor_logit=1.0,
        )
    )
    tied_slot = classify_pair_measurement(
        _measurement(
            "interchange_query_state_at_heads",
            slot_distance_to_base=1.0,
            slot_distance_to_donor=1.0,
        )
    )
    assert tied_answer == "inconclusive"
    assert tied_slot == "inconclusive"

    with pytest.raises(ValueError, match="finite"):
        _measurement("interchange_query_state_at_heads", base_margin=float("nan"))


def test_observation_aggregation_requires_preregistered_consensus() -> None:
    policy = ObservationPolicy(min_evaluable_pairs=4, min_consensus_fraction=0.75)
    result = aggregate_observation(
        [
            "donor_slot_switch",
            "donor_slot_switch",
            "donor_slot_switch",
            "no_slot_switch",
            "inconclusive",
        ],
        policy=policy,
    )

    assert result["status"] == "observation_ready"
    assert result["observation"] == "donor_slot_switch"
    assert result["evaluable_pair_count"] == 4
    assert result["inconclusive_pair_count"] == 1
    assert result["consensus_fraction"] == pytest.approx(0.75)


def test_observation_aggregation_never_forces_low_consensus_or_ties() -> None:
    policy = ObservationPolicy(min_evaluable_pairs=4, min_consensus_fraction=0.75)
    low_consensus = aggregate_observation(
        ["donor_slot_switch", "donor_slot_switch", "no_slot_switch", "no_slot_switch"],
        policy=policy,
    )
    insufficient = aggregate_observation(
        ["donor_slot_switch", "donor_slot_switch", "inconclusive"],
        policy=policy,
    )

    assert low_consensus["status"] == "inconclusive"
    assert low_consensus["observation"] is None
    assert insufficient["status"] == "insufficient_evaluable_pairs"
    assert insufficient["observation"] is None
