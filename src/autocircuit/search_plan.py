"""Build a reproducible adaptive causal-search plan from discovery-only artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autocircuit.artifacts import sha256, write_json_durable
from autocircuit.search_controller import SearchPolicy, build_search_plan, next_proposal

INTERPRETATION_SCOPE = "exploratory_discovery_only"


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _discovery_only(summary: Mapping[str, Any], label: str) -> None:
    if summary.get("interpretation_scope") != INTERPRETATION_SCOPE:
        raise ValueError(f"{label} is not discovery-only exploratory evidence")


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _pair_feedback(
    head_set_summary: Mapping[str, Any],
    *,
    selected_layer: int,
    n_heads: int,
) -> list[Mapping[str, Any]]:
    _discovery_only(head_set_summary, "head-set summary")
    head_set_layer = _integer(
        head_set_summary.get("selected_layer"), "head-set selected layer"
    )
    if head_set_layer != selected_layer:
        raise ValueError("head-set selected layer does not match head localization")
    if _integer(head_set_summary.get("n_heads"), "head-set head count") != n_heads:
        raise ValueError("head-set head count does not match head localization")
    ranking = head_set_summary.get("pair_patch_ranking")
    if not isinstance(ranking, list):
        raise ValueError("head-set pair patch ranking is missing")
    if not all(isinstance(row, dict) for row in ranking):
        raise ValueError("head-set pair patch ranking contains malformed rows")
    return ranking


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def generate_search_plan(
    head_summary_path: Path,
    component_summary_path: Path,
    output_path: Path,
    *,
    head_set_summary_path: Path | None = None,
    policy: SearchPolicy | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create one deterministic plan without opening validation or test artifacts."""

    if output_path.exists() and not force:
        raise RuntimeError(f"output exists: {output_path}; use --force")
    head_summary = _read_json_object(head_summary_path, "head summary")
    component_summary = _read_json_object(component_summary_path, "component summary")
    _discovery_only(head_summary, "head summary")
    _discovery_only(component_summary, "component summary")

    selected_layer = _integer(head_summary.get("selected_layer"), "head selected layer")
    n_heads = _integer(head_summary.get("n_heads"), "head count")
    pair_results: list[Mapping[str, Any]] | None = None
    head_set_summary: dict[str, Any] | None = None
    if head_set_summary_path is not None:
        head_set_summary = _read_json_object(head_set_summary_path, "head-set summary")
        pair_results = _pair_feedback(
            head_set_summary,
            selected_layer=selected_layer,
            n_heads=n_heads,
        )

    plan = build_search_plan(
        head_summary,
        component_summary,
        pair_results=pair_results,
        policy=policy,
    )
    source_artifacts = {
        "head_summary": _source_record(head_summary_path),
        "component_summary": _source_record(component_summary_path),
    }
    if head_set_summary_path is not None:
        source_artifacts["head_set_summary"] = _source_record(head_set_summary_path)
    plan["source_artifacts"] = source_artifacts
    plan["next_proposal"] = next_proposal(plan, completed_proposal_ids=set())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_durable(output_path, plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-summary", type=Path, required=True)
    parser.add_argument("--component-summary", type=Path, required=True)
    parser.add_argument("--head-set-summary", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/adaptive_search_plan.json"),
    )
    parser.add_argument("--max-pair-proposals", type=int, default=8)
    parser.add_argument("--exploration-pair-proposals", type=int, default=2)
    parser.add_argument("--max-path-proposals", type=int, default=2)
    parser.add_argument("--uncertainty-weight", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        policy = SearchPolicy(
            max_pair_proposals=args.max_pair_proposals,
            exploration_pair_proposals=args.exploration_pair_proposals,
            max_path_proposals=args.max_path_proposals,
            uncertainty_weight=args.uncertainty_weight,
        )
        plan = generate_search_plan(
            args.head_summary,
            args.component_summary,
            args.output,
            head_set_summary_path=args.head_set_summary,
            policy=policy,
            force=args.force,
        )
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1

    print(f"Adaptive search plan: {args.output}")
    print(
        "Pair search: "
        f"{plan['search_space']['proposed_head_pairs']} proposed / "
        f"{plan['search_space']['all_possible_head_pairs']} possible"
    )
    proposal = plan["next_proposal"]
    if proposal is None:
        print("Next proposal: none")
    else:
        print(f"Next proposal: {proposal['proposal_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
