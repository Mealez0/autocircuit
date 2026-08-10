"""Execute one discovery-only mechanism falsifier on the mechanism-eval population."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autocircuit.artifacts import sha256, write_json_durable
from autocircuit.mechanism_artifacts import load_alignment_manifest
from autocircuit.mechanism_execution import prepare_execution_bundle
from autocircuit.mechanism_outcomes import ObservationPolicy
from autocircuit.mechanism_runtime import MechanismRuntime, run_prepared_experiment
from autocircuit.mechanism_tlens_runtime import TransformerLensMechanismRuntime
from autocircuit.pipeline import PythiaAdapter

RUN_WORKFLOW_VERSION = "mechanism-run-workflow-0.1.0"
_OUTPUT_NAMES = ("experiment_report.json", "accepted_observation.json", "run_manifest.json")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def _runtime_identity(runtime: MechanismRuntime) -> dict[str, Any]:
    return {
        "model_id": runtime.model_id,
        "revision": runtime.revision,
        "resolved_revision": runtime.resolved_revision,
        "tokenizer_id": runtime.tokenizer_id,
        "dtype": runtime.dtype,
        "device": runtime.device,
    }


def _verify_runtime_identity(
    runtime: MechanismRuntime, alignment_manifest: Mapping[str, Any]
) -> None:
    identity = alignment_manifest.get("model_identity")
    if not isinstance(identity, dict):
        raise ValueError("alignment manifest model identity is missing")
    for key, actual in (
        ("model_id", runtime.model_id),
        ("tokenizer_id", runtime.tokenizer_id),
        ("dtype", runtime.dtype),
    ):
        expected = identity.get(key)
        if not isinstance(expected, str) or expected != actual:
            raise RuntimeError(f"runtime {key} mismatch: {actual!r} != {expected!r}")
    expected_resolved = identity.get("resolved_revision")
    if expected_resolved is not None:
        if not isinstance(expected_resolved, str) or runtime.resolved_revision != expected_resolved:
            raise RuntimeError(
                "runtime resolved revision mismatch: "
                f"{runtime.resolved_revision!r} != {expected_resolved!r}"
            )


def execute_mechanism_artifacts(
    mechanism_plan: Mapping[str, Any],
    counterfactual_manifest: Mapping[str, Any],
    alignment_manifest: Mapping[str, Any],
    runtime: MechanismRuntime,
    output_root: Path,
    *,
    source_artifacts: Mapping[str, Mapping[str, str]],
    policy: ObservationPolicy | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute a prepared mechanism experiment and publish a replan-ready observation."""

    existing = [output_root / name for name in _OUTPUT_NAMES if (output_root / name).exists()]
    if existing and not force:
        raise RuntimeError(
            "mechanism run outputs already exist; use --force: "
            + ", ".join(str(path) for path in existing)
        )
    if counterfactual_manifest.get("analysis_role") != "mechanism_eval":
        raise ValueError("mechanism execution requires the mechanism_eval counterfactual manifest")
    alignment_sources = alignment_manifest.get("source_artifacts")
    if (
        not isinstance(alignment_sources, dict)
        or "alignment_fit_counterfactuals" not in alignment_sources
    ):
        raise ValueError(
            "alignment manifest is not grounded in alignment_fit counterfactuals"
        )
    _verify_runtime_identity(runtime, alignment_manifest)
    alignments = load_alignment_manifest(alignment_manifest)
    prepared = prepare_execution_bundle(
        mechanism_plan,
        counterfactual_manifest,
        alignment_manifest,
    )
    result = run_prepared_experiment(
        runtime,
        prepared,
        alignments,
        policy=policy,
    )
    aggregation = result.get("aggregation")
    if not isinstance(aggregation, dict):
        raise RuntimeError("mechanism runtime returned malformed aggregation")
    status = aggregation.get("status")
    observation = aggregation.get("observation")
    if status == "observation_ready":
        if not isinstance(observation, str) or not observation:
            raise RuntimeError("ready mechanism observation has no categorical outcome")
        accepted = {prepared.experiment_id: observation}
    else:
        accepted = {}

    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "experiment_report.json"
    accepted_path = output_root / "accepted_observation.json"
    run_manifest_path = output_root / "run_manifest.json"
    report = {
        **result,
        "workflow_version": RUN_WORKFLOW_VERSION,
        "source_artifacts": {
            name: dict(record) for name, record in sorted(source_artifacts.items())
        },
        "runtime_identity": _runtime_identity(runtime),
        "accepted_observation": accepted or None,
    }
    write_json_durable(report_path, report)
    write_json_durable(accepted_path, accepted)
    artifact_hashes = {
        report_path.name: sha256(report_path),
        accepted_path.name: sha256(accepted_path),
    }
    run_manifest = {
        "schema_version": 1,
        "workflow_version": RUN_WORKFLOW_VERSION,
        "status": "complete",
        "observation_status": status,
        "interpretation_scope": "exploratory_discovery_only",
        "source_artifacts": {
            name: dict(record) for name, record in sorted(source_artifacts.items())
        },
        "artifact_hashes": artifact_hashes,
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
    write_json_durable(run_manifest_path, run_manifest)
    return run_manifest


def run_real_cli(
    mechanism_plan_path: Path,
    counterfactual_path: Path,
    alignment_path: Path,
    output_root: Path,
    *,
    device: str,
    policy: ObservationPolicy,
    force: bool,
) -> dict[str, Any]:
    """Load the exact fitted model identity and execute the selected falsifier."""

    mechanism_plan = _read_object(mechanism_plan_path, "mechanism plan")
    counterfactual_manifest = _read_object(counterfactual_path, "counterfactual manifest")
    alignment_manifest = _read_object(alignment_path, "alignment manifest")
    load_alignment_manifest(alignment_manifest)
    identity = alignment_manifest.get("model_identity")
    if not isinstance(identity, dict):
        raise ValueError("alignment manifest model identity is missing")
    model_id = identity.get("model_id")
    requested_revision = identity.get("requested_revision")
    resolved_revision = identity.get("resolved_revision")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("alignment manifest model id is malformed")
    if not isinstance(requested_revision, str) or not requested_revision:
        raise ValueError("alignment manifest requested revision is malformed")
    if resolved_revision is not None and (
        not isinstance(resolved_revision, str) or not resolved_revision
    ):
        raise ValueError("alignment manifest resolved revision is malformed")
    model = model_id.split("/")[-1]
    load_revision = resolved_revision or requested_revision
    adapter = PythiaAdapter(model, load_revision, device)
    runtime = TransformerLensMechanismRuntime(adapter)
    sources = {
        "mechanism_plan": _source_record(mechanism_plan_path),
        "mechanism_eval_counterfactuals": _source_record(counterfactual_path),
        "alignments": _source_record(alignment_path),
    }
    return execute_mechanism_artifacts(
        mechanism_plan,
        counterfactual_manifest,
        alignment_manifest,
        runtime,
        output_root,
        source_artifacts=sources,
        policy=policy,
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-plan", type=Path, required=True)
    parser.add_argument("--counterfactuals", type=Path, required=True)
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-evaluable-pairs", type=int, default=8)
    parser.add_argument("--min-consensus-fraction", type=float, default=0.75)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = ObservationPolicy(
            min_evaluable_pairs=args.min_evaluable_pairs,
            min_consensus_fraction=args.min_consensus_fraction,
        )
        manifest = run_real_cli(
            args.mechanism_plan,
            args.counterfactuals,
            args.alignments,
            args.output_root,
            device=args.device,
            policy=policy,
            force=args.force,
        )
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1
    print(f"Mechanism experiment complete: {args.output_root}")
    print(f"Observation status: {manifest['observation_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
