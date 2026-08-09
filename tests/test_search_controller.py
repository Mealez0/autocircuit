from __future__ import annotations

import math

import pytest

from autocircuit.search_controller import SearchPolicy, build_search_plan, next_proposal


def _head_row(
    head: int,
    advantage: float,
    low: float,
    high: float,
    *,
    matched: float | None = None,
    positive_fraction: float = 0.6,
) -> dict[str, object]:
    return {
        "intervention": f"head_{head}",
        "head_index": head,
        "family_count": 120,
        "matched_mean_transfer": advantage if matched is None else matched,
        "permuted_mean_transfer": 0.0,
        "family_specific_advantage": advantage,
        "family_specific_advantage_ci_95": [low, high],
        "positive_transfer_fraction": positive_fraction,
        "rank": head + 1,
    }


def _head_summary() -> dict[str, object]:
    rows = [
        _head_row(0, 0.010, -0.020, 0.040),
        _head_row(1, 0.028, -0.004, 0.060),
        _head_row(2, 0.008, -0.025, 0.035),
        _head_row(3, 0.083, 0.050, 0.116, matched=0.101, positive_fraction=0.78),
        _head_row(4, 0.015, -0.010, 0.042),
        _head_row(5, 0.005, -0.030, 0.040),
        _head_row(6, 0.076, 0.041, 0.109, matched=0.094, positive_fraction=0.74),
        _head_row(7, 0.018, -0.012, 0.048),
    ]
    rows.sort(key=lambda row: (-float(row["family_specific_advantage"]), int(row["head_index"])))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {
        "interpretation_scope": "exploratory_discovery_only",
        "protocol_version": "position-head-localization-0.1.0",
        "selected_layer": 5,
        "n_heads": 8,
        "family_count": 120,
        "individual_head_ranking": rows,
    }


def _component_summary(mlp_advantage: float = 0.171) -> dict[str, object]:
    return {
        "protocol_version": "position-component-localization-0.1.0",
        "selected_layer": 5,
        "intervention_ranking": [
            {
                "intervention": "mlp_output",
                "matched_mean_transfer": 0.19,
                "permuted_mean_transfer": 0.019,
                "family_specific_advantage": mlp_advantage,
            },
            {
                "intervention": "attention_output",
                "matched_mean_transfer": 0.18,
                "permuted_mean_transfer": 0.027,
                "family_specific_advantage": 0.153,
            },
        ],
    }


def test_plan_prioritizes_strong_stable_pair_and_reduces_pair_search() -> None:
    policy = SearchPolicy(max_pair_proposals=6, exploration_pair_proposals=2)
    plan = build_search_plan(_head_summary(), _component_summary(), policy=policy)

    pair_proposals = [
        proposal for proposal in plan["proposals"] if proposal["kind"] == "pair_patch"
    ]
    assert pair_proposals[0]["heads"] == [3, 6]
    assert pair_proposals[0]["mode"] == "exploit"
    assert len(pair_proposals) == 6
    assert plan["search_space"]["all_possible_head_pairs"] == 28
    assert plan["search_space"]["proposed_head_pairs"] == 6
    assert plan["search_space"]["pair_reduction_factor"] == pytest.approx(28 / 6)


def test_exploration_budget_covers_heads_not_used_by_exploitation() -> None:
    policy = SearchPolicy(max_pair_proposals=5, exploration_pair_proposals=2)
    plan = build_search_plan(_head_summary(), _component_summary(), policy=policy)

    exploit = [
        proposal
        for proposal in plan["proposals"]
        if proposal["kind"] == "pair_patch" and proposal["mode"] == "exploit"
    ]
    explore = [
        proposal
        for proposal in plan["proposals"]
        if proposal["kind"] == "pair_patch" and proposal["mode"] == "explore"
    ]
    exploit_heads = {head for proposal in exploit for head in proposal["heads"]}
    assert len(explore) == 2
    assert any(any(head not in exploit_heads for head in proposal["heads"]) for proposal in explore)


def test_positive_mlp_branch_emits_head_to_mlp_path_proposals() -> None:
    plan = build_search_plan(
        _head_summary(),
        _component_summary(),
        policy=SearchPolicy(max_pair_proposals=4, max_path_proposals=2),
    )

    paths = [proposal for proposal in plan["proposals"] if proposal["kind"] == "head_to_mlp_path"]
    assert [proposal["heads"] for proposal in paths] == [[3], [6]]
    assert all(proposal["evidence_status"] == "proposal_only_not_evidence" for proposal in paths)
    assert plan["scientific_confirmation"] is False
    assert plan["circuit_found"] is False
    assert plan["held_out_validation_reused"] is False
    assert plan["held_out_test_opened"] is False


def test_nonpositive_mlp_branch_does_not_spend_path_budget() -> None:
    plan = build_search_plan(_head_summary(), _component_summary(-0.01))
    assert not any(proposal["kind"] == "head_to_mlp_path" for proposal in plan["proposals"])


def test_stable_observed_pair_promotes_necessity_followup() -> None:
    pair_results = [
        {
            "pair": [3, 6],
            "family_specific_advantage": 0.159,
            "family_specific_advantage_ci_95": [0.130, 0.189],
        }
    ]
    plan = build_search_plan(
        _head_summary(),
        _component_summary(),
        pair_results=pair_results,
        policy=SearchPolicy(max_pair_proposals=4),
    )

    assert plan["proposals"][0]["proposal_id"] == "leave_pair_out_3_6"
    assert plan["proposals"][0]["kind"] == "pair_necessity"
    assert plan["proposals"][0]["trigger"] == "observed_pair_ci_lower_bound_positive"


def test_necessity_followups_are_bounded_and_ranked_by_stability() -> None:
    pair_results = [
        {
            "pair": [3, 6],
            "family_specific_advantage": 0.159,
            "family_specific_advantage_ci_95": [0.130, 0.189],
        },
        {
            "pair": [1, 3],
            "family_specific_advantage": 0.120,
            "family_specific_advantage_ci_95": [0.090, 0.150],
        },
        {
            "pair": [0, 6],
            "family_specific_advantage": 0.110,
            "family_specific_advantage_ci_95": [0.080, 0.145],
        },
    ]
    plan = build_search_plan(
        _head_summary(),
        _component_summary(),
        pair_results=pair_results,
        policy=SearchPolicy(max_pair_proposals=4, max_necessity_proposals=2),
    )

    necessity = [
        proposal for proposal in plan["proposals"] if proposal["kind"] == "pair_necessity"
    ]
    assert [proposal["heads"] for proposal in necessity] == [[3, 6], [1, 3]]
    assert plan["search_space"]["stable_observed_pairs"] == 3
    assert plan["search_space"]["proposed_necessity_followups"] == 2
    assert plan["search_space"]["necessity_followups_deferred"] == 1


def test_zero_necessity_budget_defers_stable_pair_followups() -> None:
    pair_results = [
        {
            "pair": [3, 6],
            "family_specific_advantage": 0.159,
            "family_specific_advantage_ci_95": [0.130, 0.189],
        }
    ]
    plan = build_search_plan(
        _head_summary(),
        _component_summary(),
        pair_results=pair_results,
        policy=SearchPolicy(max_necessity_proposals=0),
    )

    assert not any(proposal["kind"] == "pair_necessity" for proposal in plan["proposals"])
    assert plan["search_space"]["necessity_followups_deferred"] == 1


def test_uncertain_observed_pair_does_not_trigger_necessity_followup() -> None:
    pair_results = [
        {
            "pair": [3, 6],
            "family_specific_advantage": 0.04,
            "family_specific_advantage_ci_95": [-0.02, 0.09],
        }
    ]
    plan = build_search_plan(_head_summary(), _component_summary(), pair_results=pair_results)
    assert not any(proposal["kind"] == "pair_necessity" for proposal in plan["proposals"])


def test_next_proposal_skips_completed_ids_without_reordering_plan() -> None:
    plan = build_search_plan(
        _head_summary(), _component_summary(), policy=SearchPolicy(max_pair_proposals=3)
    )
    first = plan["proposals"][0]
    second = plan["proposals"][1]

    assert next_proposal(plan, completed_proposal_ids=set()) == first
    assert next_proposal(plan, completed_proposal_ids={first["proposal_id"]}) == second
    assert next_proposal(
        plan,
        completed_proposal_ids={proposal["proposal_id"] for proposal in plan["proposals"]},
    ) is None


def test_nonfinite_head_evidence_is_rejected_fail_closed() -> None:
    summary = _head_summary()
    rows = summary["individual_head_ranking"]
    assert isinstance(rows, list)
    rows[0]["family_specific_advantage"] = math.nan

    with pytest.raises(ValueError, match="finite"):
        build_search_plan(summary, _component_summary())


def test_policy_rejects_impossible_exploration_budget() -> None:
    with pytest.raises(ValueError, match="exploration_pair_proposals"):
        SearchPolicy(max_pair_proposals=2, exploration_pair_proposals=3)
