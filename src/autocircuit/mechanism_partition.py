"""Deterministic discovery-only partitioning for mechanism alignment and evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from autocircuit.mechanism_counterfactuals import CounterfactualPair

PARTITION_VERSION = "mechanism-discovery-partition-0.1.0"
DEFAULT_NAMESPACE = "autocircuit:mechanism-discovery-partition:v1"
PartitionRole = Literal["alignment_fit", "mechanism_eval"]


@dataclass(frozen=True)
class MechanismPartitionPolicy:
    """Freeze the internal discovery split used to avoid alignment/evaluation reuse."""

    fit_fraction: float = 2.0 / 3.0
    namespace: str = DEFAULT_NAMESPACE

    def __post_init__(self) -> None:
        if (
            isinstance(self.fit_fraction, bool)
            or not isinstance(self.fit_fraction, (int, float))
            or not math.isfinite(float(self.fit_fraction))
            or not 0.0 < float(self.fit_fraction) < 1.0
        ):
            raise ValueError("fit_fraction must lie strictly between zero and one")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("partition namespace must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_fraction": float(self.fit_fraction),
            "namespace": self.namespace,
        }


def _family_order(family_id: str, namespace: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{namespace}:{family_id}".encode()).hexdigest()
    return digest, family_id


def build_mechanism_partition(
    pairs: Sequence[CounterfactualPair],
    *,
    policy: MechanismPartitionPolicy | None = None,
) -> dict[str, Any]:
    """Assign whole discovery families to alignment-fit or mechanism-eval roles."""

    selected = policy if policy is not None else MechanismPartitionPolicy()
    if not pairs or any(pair.split != "discovery" for pair in pairs):
        raise ValueError("mechanism partition accepts discovery counterfactuals only")
    ids = [pair.counterfactual_id for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("mechanism partition received duplicate counterfactual ids")
    families = sorted(
        {pair.source_family_id for pair in pairs},
        key=lambda family_id: _family_order(family_id, selected.namespace),
    )
    if len(families) < 2:
        raise ValueError("mechanism partition requires at least two discovery families")

    raw_fit_count = int(round(len(families) * float(selected.fit_fraction)))
    fit_count = min(max(raw_fit_count, 1), len(families) - 1)
    fit_families = set(families[:fit_count])
    role_by_family: dict[str, PartitionRole] = {
        family_id: (
            "alignment_fit" if family_id in fit_families else "mechanism_eval"
        )
        for family_id in families
    }
    role_by_counterfactual_id = {
        pair.counterfactual_id: role_by_family[pair.source_family_id]
        for pair in sorted(pairs, key=lambda item: item.counterfactual_id)
    }
    fit_pair_count = sum(role == "alignment_fit" for role in role_by_counterfactual_id.values())
    eval_pair_count = len(role_by_counterfactual_id) - fit_pair_count
    return {
        "schema_version": 1,
        "partition_version": PARTITION_VERSION,
        "interpretation_scope": "exploratory_discovery_only",
        "policy": selected.to_dict(),
        "family_count": len(families),
        "fit_family_count": fit_count,
        "eval_family_count": len(families) - fit_count,
        "fit_counterfactual_count": fit_pair_count,
        "eval_counterfactual_count": eval_pair_count,
        "role_by_family": dict(sorted(role_by_family.items())),
        "role_by_counterfactual_id": dict(sorted(role_by_counterfactual_id.items())),
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def select_partition_role(
    pairs: Sequence[CounterfactualPair],
    partition: Mapping[str, Any],
    role: PartitionRole,
) -> list[CounterfactualPair]:
    """Select one recorded role and verify the partition covers inputs exactly."""

    if role not in {"alignment_fit", "mechanism_eval"}:
        raise ValueError(f"unknown mechanism partition role: {role}")
    if partition.get("interpretation_scope") != "exploratory_discovery_only":
        raise ValueError("mechanism partition is not discovery-only")
    mapping = partition.get("role_by_counterfactual_id")
    if not isinstance(mapping, dict):
        raise ValueError("mechanism partition role mapping is malformed")
    ids = {pair.counterfactual_id for pair in pairs}
    if set(mapping) != ids:
        raise ValueError("mechanism partition does not exactly cover counterfactual inputs")
    selected: list[CounterfactualPair] = []
    for pair in sorted(pairs, key=lambda item: item.counterfactual_id):
        recorded = mapping.get(pair.counterfactual_id)
        if recorded not in {"alignment_fit", "mechanism_eval"}:
            raise ValueError("mechanism partition contains an invalid role")
        if recorded == role:
            selected.append(pair)
    if not selected:
        raise ValueError(f"mechanism partition role is empty: {role}")
    return selected
