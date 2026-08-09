"""Adaptive discovery-only experiment planning from causal evidence.

The controller is deliberately separated from confirmatory validation. It consumes
already-produced discovery summaries, proposes a smaller causal experiment set,
and can promote stronger follow-up tests after observing discovery-only results.
A proposal is never treated as scientific evidence or a circuit claim.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

PLANNER_VERSION = "adaptive-search-controller-0.1.0"
INTERPRETATION_SCOPE = "exploratory_discovery_only"


@dataclass(frozen=True)
class SearchPolicy:
    """Compute budget and exploration policy for adaptive head search."""

    max_pair_proposals: int = 8
    exploration_pair_proposals: int = 2
    max_path_proposals: int = 2
    uncertainty_weight: float = 0.5

    def __post_init__(self) -> None:
        integer_fields = {
            "max_pair_proposals": self.max_pair_proposals,
            "exploration_pair_proposals": self.exploration_pair_proposals,
            "max_path_proposals": self.max_path_proposals,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.exploration_pair_proposals > self.max_pair_proposals:
            raise ValueError(
                "exploration_pair_proposals cannot exceed max_pair_proposals"
            )
        if not math.isfinite(self.uncertainty_weight) or self.uncertainty_weight < 0:
            raise ValueError("uncertainty_weight must be finite and non-negative")


@dataclass(frozen=True)
class HeadEvidence:
    head_index: int
    advantage: float
    ci_low: float
    ci_high: float
    matched_transfer: float
    positive_transfer_fraction: float
    uncertainty_weight: float

    @property
    def uncertainty_radius(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0

    @property
    def upper_confidence_score(self) -> float:
        return self.advantage + self.uncertainty_weight * self.uncertainty_radius

    @property
    def stable_positive(self) -> bool:
        return self.ci_low > 0.0


@dataclass(frozen=True)
class PairObservation:
    pair: tuple[int, int]
    advantage: float
    ci_low: float
    ci_high: float

    @property
    def stable_positive(self) -> bool:
        return self.ci_low > 0.0


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _interval(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two bounds")
    low = _finite_number(value[0], f"{label} lower bound")
    high = _finite_number(value[1], f"{label} upper bound")
    if low > high:
        raise ValueError(f"{label} lower bound exceeds upper bound")
    return low, high


def _head_evidence(
    summary: Mapping[str, Any], policy: SearchPolicy
) -> tuple[int, list[HeadEvidence]]:
    if summary.get("interpretation_scope") != INTERPRETATION_SCOPE:
        raise ValueError("head summary is not discovery-only exploratory evidence")
    n_heads = _integer(summary.get("n_heads"), "head count")
    if n_heads < 2:
        raise ValueError("adaptive pair search requires at least two heads")
    rows = summary.get("individual_head_ranking")
    if not isinstance(rows, list) or len(rows) != n_heads:
        raise ValueError("individual head ranking is incomplete")

    evidence: list[HeadEvidence] = []
    seen: set[int] = set()
    for row_index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise ValueError(f"head ranking row {row_index} is malformed")
        head = _integer(raw_row.get("head_index"), f"head ranking row {row_index} index")
        if head < 0 or head >= n_heads or head in seen:
            raise ValueError("head ranking contains duplicate or out-of-range indexes")
        seen.add(head)
        low, high = _interval(
            raw_row.get("family_specific_advantage_ci_95"),
            f"head {head} family-specific interval",
        )
        positive_fraction = _finite_number(
            raw_row.get("positive_transfer_fraction"),
            f"head {head} positive transfer fraction",
        )
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError("positive transfer fraction must lie in [0, 1]")
        evidence.append(
            HeadEvidence(
                head_index=head,
                advantage=_finite_number(
                    raw_row.get("family_specific_advantage"),
                    f"head {head} family-specific advantage",
                ),
                ci_low=low,
                ci_high=high,
                matched_transfer=_finite_number(
                    raw_row.get("matched_mean_transfer"),
                    f"head {head} matched transfer",
                ),
                positive_transfer_fraction=positive_fraction,
                uncertainty_weight=policy.uncertainty_weight,
            )
        )
    if seen != set(range(n_heads)):
        raise ValueError("head ranking does not cover every model head exactly once")
    evidence.sort(
        key=lambda item: (
            -int(item.stable_positive),
            -item.upper_confidence_score,
            -item.advantage,
            -item.matched_transfer,
            item.head_index,
        )
    )
    return n_heads, evidence


def _component_advantage(summary: Mapping[str, Any], intervention: str) -> float:
    ranking = summary.get("intervention_ranking")
    if not isinstance(ranking, list):
        raise ValueError("component intervention ranking is missing")
    matching = [
        row
        for row in ranking
        if isinstance(row, dict) and row.get("intervention") == intervention
    ]
    if len(matching) != 1:
        raise ValueError(f"component summary must contain one {intervention} row")
    return _finite_number(
        matching[0].get("family_specific_advantage"),
        f"{intervention} family-specific advantage",
    )


def _pair_observations(
    rows: Sequence[Mapping[str, Any]] | None, n_heads: int
) -> list[PairObservation]:
    if rows is None:
        return []
    observations: list[PairObservation] = []
    seen: set[tuple[int, int]] = set()
    for row_index, row in enumerate(rows):
        pair_value = row.get("pair")
        if not isinstance(pair_value, (list, tuple)) or len(pair_value) != 2:
            raise ValueError(f"pair result {row_index} has a malformed pair")
        left = _integer(pair_value[0], f"pair result {row_index} left head")
        right = _integer(pair_value[1], f"pair result {row_index} right head")
        pair = (min(left, right), max(left, right))
        if (
            left == right
            or pair[0] < 0
            or pair[1] >= n_heads
            or pair in seen
        ):
            raise ValueError("pair results contain duplicate or out-of-range pairs")
        seen.add(pair)
        low, high = _interval(
            row.get("family_specific_advantage_ci_95"),
            f"pair {pair} family-specific interval",
        )
        observations.append(
            PairObservation(
                pair=pair,
                advantage=_finite_number(
                    row.get("family_specific_advantage"),
                    f"pair {pair} family-specific advantage",
                ),
                ci_low=low,
                ci_high=high,
            )
        )
    observations.sort(key=lambda item: (-item.ci_low, -item.advantage, item.pair))
    return observations


def _pair_metrics(
    pair: tuple[int, int], evidence_by_head: Mapping[int, HeadEvidence]
) -> dict[str, float | int]:
    left = evidence_by_head[pair[0]]
    right = evidence_by_head[pair[1]]
    return {
        "stable_positive_heads": int(left.stable_positive) + int(right.stable_positive),
        "upper_confidence_score_sum": (
            left.upper_confidence_score + right.upper_confidence_score
        ),
        "family_specific_advantage_sum": left.advantage + right.advantage,
        "uncertainty_radius_sum": left.uncertainty_radius + right.uncertainty_radius,
    }


def _rank_exploitation_pairs(
    pairs: Sequence[tuple[int, int]], evidence_by_head: Mapping[int, HeadEvidence]
) -> list[tuple[int, int]]:
    return sorted(
        pairs,
        key=lambda pair: (
            -int(_pair_metrics(pair, evidence_by_head)["stable_positive_heads"]),
            -float(_pair_metrics(pair, evidence_by_head)["upper_confidence_score_sum"]),
            -float(_pair_metrics(pair, evidence_by_head)["family_specific_advantage_sum"]),
            pair,
        ),
    )


def _select_exploration_pairs(
    candidates: Sequence[tuple[int, int]],
    selected: Sequence[tuple[int, int]],
    count: int,
    evidence_by_head: Mapping[int, HeadEvidence],
) -> list[tuple[int, int]]:
    remaining = list(candidates)
    covered = {head for pair in selected for head in pair}
    chosen: list[tuple[int, int]] = []
    for _ in range(count):
        if not remaining:
            break
        remaining.sort(
            key=lambda pair: (
                -sum(head not in covered for head in pair),
                -float(_pair_metrics(pair, evidence_by_head)["uncertainty_radius_sum"]),
                -float(_pair_metrics(pair, evidence_by_head)["upper_confidence_score_sum"]),
                pair,
            )
        )
        pair = remaining.pop(0)
        chosen.append(pair)
        covered.update(pair)
    return chosen


def _pair_proposal(
    pair: tuple[int, int], mode: str, evidence_by_head: Mapping[int, HeadEvidence]
) -> dict[str, Any]:
    metrics = _pair_metrics(pair, evidence_by_head)
    return {
        "proposal_id": f"pair_patch_{pair[0]}_{pair[1]}",
        "kind": "pair_patch",
        "mode": mode,
        "heads": list(pair),
        "priority_basis": (
            "stable-positive individual causal evidence, uncertainty-aware upper confidence "
            "ranking, and deterministic coverage exploration"
        ),
        "evidence_status": "proposal_only_not_evidence",
        "source_evidence": metrics,
    }


def _necessity_followups(observations: Sequence[PairObservation]) -> list[dict[str, Any]]:
    return [
        {
            "proposal_id": f"leave_pair_out_{item.pair[0]}_{item.pair[1]}",
            "kind": "pair_necessity",
            "mode": "adaptive_follow_up",
            "heads": list(item.pair),
            "trigger": "observed_pair_ci_lower_bound_positive",
            "evidence_status": "proposal_only_not_evidence",
            "observed_pair_evidence": {
                "family_specific_advantage": item.advantage,
                "family_specific_advantage_ci_95": [item.ci_low, item.ci_high],
            },
        }
        for item in observations
        if item.stable_positive
    ]


def _path_proposals(
    ranked_heads: Sequence[HeadEvidence],
    mlp_advantage: float,
    attention_advantage: float,
    count: int,
) -> list[dict[str, Any]]:
    if mlp_advantage <= 0.0 or count == 0:
        return []
    candidates = [item for item in ranked_heads if item.upper_confidence_score > 0.0]
    return [
        {
            "proposal_id": f"head_{item.head_index}_to_mlp_path",
            "kind": "head_to_mlp_path",
            "mode": "mechanism_follow_up",
            "heads": [item.head_index],
            "target_component": "mlp_output",
            "priority_basis": (
                "positive MLP family-specific causal contribution plus high-priority head "
                "evidence"
            ),
            "evidence_status": "proposal_only_not_evidence",
            "source_evidence": {
                "head_family_specific_advantage": item.advantage,
                "head_family_specific_advantage_ci_95": [item.ci_low, item.ci_high],
                "mlp_family_specific_advantage": mlp_advantage,
                "attention_family_specific_advantage": attention_advantage,
            },
        }
        for item in candidates[:count]
    ]


def build_search_plan(
    head_summary: Mapping[str, Any],
    component_summary: Mapping[str, Any],
    *,
    pair_results: Sequence[Mapping[str, Any]] | None = None,
    policy: SearchPolicy | None = None,
) -> dict[str, Any]:
    """Build a deterministic adaptive plan using discovery-only causal summaries.

    Pair patches are chosen with an exploitation/exploration split. Exploitation
    favors heads whose matched-vs-permuted causal advantage is strong and whose
    confidence interval is stable; exploration spends a bounded budget covering
    heads that exploitation has not touched. Stable observed pair results promote
    a leave-pair-out necessity follow-up instead of blindly scanning every pair.
    """

    selected_policy = policy if policy is not None else SearchPolicy()
    head_layer = _integer(head_summary.get("selected_layer"), "head selected layer")
    component_layer = _integer(
        component_summary.get("selected_layer"), "component selected layer"
    )
    if head_layer != component_layer:
        raise ValueError("head and component summaries disagree on selected layer")

    n_heads, ranked_heads = _head_evidence(head_summary, selected_policy)
    evidence_by_head = {item.head_index: item for item in ranked_heads}
    observations = _pair_observations(pair_results, n_heads)
    observed_pairs = {item.pair for item in observations}

    all_pairs = list(itertools.combinations(range(n_heads), 2))
    unobserved_pairs = [pair for pair in all_pairs if pair not in observed_pairs]
    pair_budget = min(selected_policy.max_pair_proposals, len(unobserved_pairs))
    explore_budget = min(selected_policy.exploration_pair_proposals, pair_budget)
    exploit_budget = pair_budget - explore_budget

    exploitation_ranking = _rank_exploitation_pairs(unobserved_pairs, evidence_by_head)
    exploit_pairs = exploitation_ranking[:exploit_budget]
    exploit_set = set(exploit_pairs)
    remaining = [pair for pair in unobserved_pairs if pair not in exploit_set]
    explore_pairs = _select_exploration_pairs(
        remaining,
        exploit_pairs,
        explore_budget,
        evidence_by_head,
    )

    mlp_advantage = _component_advantage(component_summary, "mlp_output")
    attention_advantage = _component_advantage(component_summary, "attention_output")
    followups = _necessity_followups(observations)
    exploit_proposals = [
        _pair_proposal(pair, "exploit", evidence_by_head) for pair in exploit_pairs
    ]
    path_proposals = _path_proposals(
        ranked_heads,
        mlp_advantage,
        attention_advantage,
        selected_policy.max_path_proposals,
    )
    explore_proposals = [
        _pair_proposal(pair, "explore", evidence_by_head) for pair in explore_pairs
    ]
    proposals = followups + exploit_proposals + path_proposals + explore_proposals
    proposal_ids = [str(proposal["proposal_id"]) for proposal in proposals]
    if len(proposal_ids) != len(set(proposal_ids)):
        raise RuntimeError("adaptive search planner produced duplicate proposal identifiers")

    proposed_pair_count = len(exploit_pairs) + len(explore_pairs)
    pair_reduction_factor = (
        len(all_pairs) / proposed_pair_count if proposed_pair_count else None
    )
    return {
        "schema_version": 1,
        "planner_version": PLANNER_VERSION,
        "interpretation_scope": INTERPRETATION_SCOPE,
        "selected_layer": head_layer,
        "strategy": "causal_evidence_guided_ucb_with_coverage",
        "policy": {
            "max_pair_proposals": selected_policy.max_pair_proposals,
            "exploration_pair_proposals": selected_policy.exploration_pair_proposals,
            "max_path_proposals": selected_policy.max_path_proposals,
            "uncertainty_weight": selected_policy.uncertainty_weight,
        },
        "search_space": {
            "all_possible_head_pairs": len(all_pairs),
            "observed_head_pairs": len(observed_pairs),
            "proposed_head_pairs": proposed_pair_count,
            "pair_interventions_avoided_this_plan": len(all_pairs)
            - len(observed_pairs)
            - proposed_pair_count,
            "pair_reduction_factor": pair_reduction_factor,
        },
        "component_context": {
            "attention_family_specific_advantage": attention_advantage,
            "mlp_family_specific_advantage": mlp_advantage,
        },
        "ranked_head_evidence": [
            {
                "head_index": item.head_index,
                "family_specific_advantage": item.advantage,
                "family_specific_advantage_ci_95": [item.ci_low, item.ci_high],
                "stable_positive": item.stable_positive,
                "uncertainty_radius": item.uncertainty_radius,
                "upper_confidence_score": item.upper_confidence_score,
            }
            for item in ranked_heads
        ],
        "proposals": proposals,
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def next_proposal(
    plan: Mapping[str, Any], *, completed_proposal_ids: Set[str]
) -> dict[str, Any] | None:
    """Return the highest-priority unfinished proposal without mutating the plan."""

    proposals = plan.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError("search plan proposals are malformed")
    for row_index, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            raise ValueError(f"search proposal {row_index} is malformed")
        proposal_id = proposal.get("proposal_id")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError(f"search proposal {row_index} has no valid identifier")
        if proposal_id not in completed_proposal_ids:
            return dict(proposal)
    return None
