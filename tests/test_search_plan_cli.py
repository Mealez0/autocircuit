from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocircuit.search_controller import SearchPolicy
from autocircuit.search_plan import generate_search_plan


def _head_summary() -> dict[str, object]:
    rows = []
    advantages = [0.010, 0.028, 0.008, 0.083, 0.015, 0.005, 0.076, 0.018]
    intervals = [
        (-0.020, 0.040),
        (-0.004, 0.060),
        (-0.025, 0.035),
        (0.050, 0.116),
        (-0.010, 0.042),
        (-0.030, 0.040),
        (0.041, 0.109),
        (-0.012, 0.048),
    ]
    for head, advantage in enumerate(advantages):
        low, high = intervals[head]
        rows.append(
            {
                "intervention": f"head_{head}",
                "head_index": head,
                "family_count": 120,
                "matched_mean_transfer": advantage + 0.02,
                "permuted_mean_transfer": 0.02,
                "family_specific_advantage": advantage,
                "family_specific_advantage_ci_95": [low, high],
                "positive_transfer_fraction": 0.7 if head in {3, 6} else 0.55,
                "rank": head + 1,
            }
        )
    rows.sort(key=lambda row: -float(row["family_specific_advantage"]))
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


def _component_summary() -> dict[str, object]:
    return {
        "interpretation_scope": "exploratory_discovery_only",
        "protocol_version": "position-component-localization-0.1.0",
        "selected_layer": 5,
        "intervention_ranking": [
            {
                "intervention": "mlp_output",
                "family_specific_advantage": 0.171,
            },
            {
                "intervention": "attention_output",
                "family_specific_advantage": 0.153,
            },
        ],
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_generate_search_plan_writes_provenance_and_next_proposal(tmp_path: Path) -> None:
    head = tmp_path / "head_summary.json"
    component = tmp_path / "component_summary.json"
    output = tmp_path / "adaptive_search_plan.json"
    _write(head, _head_summary())
    _write(component, _component_summary())

    plan = generate_search_plan(
        head,
        component,
        output,
        policy=SearchPolicy(max_pair_proposals=6, exploration_pair_proposals=2),
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == plan
    assert plan["source_artifacts"]["head_summary"]["sha256"]
    assert plan["source_artifacts"]["component_summary"]["sha256"]
    assert plan["next_proposal"]["heads"] == [3, 6]
    assert plan["search_space"]["proposed_head_pairs"] == 6
    assert plan["held_out_validation_reused"] is False
    assert plan["held_out_test_opened"] is False


def test_generate_search_plan_refuses_overwrite_without_force(tmp_path: Path) -> None:
    head = tmp_path / "head_summary.json"
    component = tmp_path / "component_summary.json"
    output = tmp_path / "adaptive_search_plan.json"
    _write(head, _head_summary())
    _write(component, _component_summary())
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output exists"):
        generate_search_plan(head, component, output)

    assert output.read_text(encoding="utf-8") == "existing\n"


def test_head_set_feedback_promotes_stable_pair_necessity(tmp_path: Path) -> None:
    head = tmp_path / "head_summary.json"
    component = tmp_path / "component_summary.json"
    head_set = tmp_path / "head_set_summary.json"
    output = tmp_path / "adaptive_search_plan.json"
    _write(head, _head_summary())
    _write(component, _component_summary())
    _write(
        head_set,
        {
            "interpretation_scope": "exploratory_discovery_only",
            "selected_layer": 5,
            "n_heads": 8,
            "pair_patch_ranking": [
                {
                    "pair": [3, 6],
                    "family_specific_advantage": 0.159,
                    "family_specific_advantage_ci_95": [0.130, 0.189],
                }
            ],
        },
    )

    plan = generate_search_plan(head, component, output, head_set_summary_path=head_set)

    assert plan["next_proposal"]["proposal_id"] == "leave_pair_out_3_6"
    assert plan["source_artifacts"]["head_set_summary"]["sha256"]
    assert plan["search_space"]["observed_head_pairs"] == 1


def test_head_set_feedback_rejects_mismatched_layer(tmp_path: Path) -> None:
    head = tmp_path / "head_summary.json"
    component = tmp_path / "component_summary.json"
    head_set = tmp_path / "head_set_summary.json"
    output = tmp_path / "adaptive_search_plan.json"
    _write(head, _head_summary())
    _write(component, _component_summary())
    _write(
        head_set,
        {
            "interpretation_scope": "exploratory_discovery_only",
            "selected_layer": 4,
            "n_heads": 8,
            "pair_patch_ranking": [],
        },
    )

    with pytest.raises(ValueError, match="selected layer"):
        generate_search_plan(head, component, output, head_set_summary_path=head_set)
