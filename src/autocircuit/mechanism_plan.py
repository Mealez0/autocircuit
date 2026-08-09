"""Build a discovery-only causal mechanism campaign and choose the next falsifier."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autocircuit.artifacts import sha256, write_json_durable
from autocircuit.mechanism_synthesis import (
    SYNTHESIS_VERSION,
    build_associative_recall_campaign,
    evaluate_campaign,
    select_next_experiment,
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_observations(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = _read_object(path, "mechanism observations")
    output: dict[str, str] = {}
    for experiment_id, outcome in value.items():
        if not isinstance(experiment_id, str) or not isinstance(outcome, str):
            raise ValueError("mechanism observations must map experiment ids to string outcomes")
        output[experiment_id] = outcome
    return output


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def generate_mechanism_plan(
    search_plan_path: Path,
    output_path: Path,
    *,
    observations_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create a deterministic mechanism plan without accessing held-out data."""

    if output_path.exists() and not force:
        raise RuntimeError(f"output exists: {output_path}; use --force")
    if output_path.resolve() == search_plan_path.resolve():
        raise ValueError("mechanism plan output cannot replace its source search plan")
    if observations_path is not None and output_path.resolve() == observations_path.resolve():
        raise ValueError("mechanism plan output cannot replace its observations")

    search_plan = _read_object(search_plan_path, "adaptive search plan")
    observations = _read_observations(observations_path)
    campaign = build_associative_recall_campaign(search_plan)
    state = evaluate_campaign(campaign, observations)
    next_experiment = select_next_experiment(campaign, observations)

    plan: dict[str, Any] = {
        "schema_version": 1,
        "synthesis_version": SYNTHESIS_VERSION,
        "interpretation_scope": "exploratory_discovery_only",
        "source_artifact": _source(search_plan_path),
        "observations": dict(sorted(observations.items())),
        "campaign": campaign.to_dict(),
        "state": state,
        "next_experiment": next_experiment,
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
    if observations_path is not None:
        plan["observations_artifact"] = _source(observations_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_durable(output_path, plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-plan", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mechanism_plan.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = generate_mechanism_plan(
            args.search_plan,
            args.output,
            observations_path=args.observations,
            force=args.force,
        )
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1

    state = plan["state"]
    if not isinstance(state, Mapping):
        raise RuntimeError("mechanism plan state is malformed")
    print(f"Mechanism plan: {args.output}")
    print(f"Surviving hypotheses: {len(state['surviving_hypotheses'])}")
    proposal = plan["next_experiment"]
    if proposal is None:
        print("Next falsification experiment: none")
    elif isinstance(proposal, Mapping):
        print(f"Next falsification experiment: {proposal['experiment_id']}")
    else:
        raise RuntimeError("mechanism plan proposal is malformed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
