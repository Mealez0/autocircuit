"""Fit causal-variable alignments on a disjoint discovery-only population slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autocircuit.artifacts import sha256, write_json_durable
from autocircuit.datasets.associative_recall import ExamplePair
from autocircuit.datasets.validation import Tokenizer
from autocircuit.mechanism_artifacts import build_alignment_manifest
from autocircuit.mechanism_counterfactuals import (
    CounterfactualPair,
    build_counterfactual_manifest,
    build_discovery_counterfactuals,
)
from autocircuit.mechanism_execution import _campaign
from autocircuit.mechanism_partition import (
    MechanismPartitionPolicy,
    build_mechanism_partition,
    select_partition_role,
)
from autocircuit.mechanism_runtime import MechanismRuntime, fit_runtime_alignments
from autocircuit.mechanism_tlens_runtime import TransformerLensMechanismRuntime
from autocircuit.pipeline import (
    PythiaAdapter,
    _verify_frozen_discovery,
    _verify_validation_adapter,
)

FIT_WORKFLOW_VERSION = "mechanism-fit-workflow-0.1.0"
_OUTPUT_NAMES = (
    "counterfactuals.json",
    "alignment_fit_counterfactuals.json",
    "mechanism_eval_counterfactuals.json",
    "alignments.json",
    "fit_report.json",
    "run_manifest.json",
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_discovery_examples(path: Path) -> list[ExamplePair]:
    if not path.is_file():
        raise RuntimeError(f"missing matched discovery dataset: {path}")
    examples: list[ExamplePair] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed matched discovery JSONL at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"matched discovery JSONL line {line_number} is not an object"
            )
        try:
            example = ExamplePair(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid matched discovery example at line {line_number}"
            ) from exc
        if example.split != "discovery":
            raise ValueError(
                f"non-discovery example in matched dataset at line {line_number}"
            )
        examples.append(example)
    if not examples:
        raise ValueError("matched discovery dataset is empty")
    return examples


def _guard_mechanism_plan(value: Mapping[str, Any]) -> None:
    if value.get("interpretation_scope") != "exploratory_discovery_only":
        raise ValueError("mechanism fitting requires a discovery-only mechanism plan")
    if value.get("held_out_validation_reused") is not False:
        raise ValueError("mechanism plan indicates held-out validation reuse")
    if value.get("held_out_test_opened") is not False:
        raise ValueError("mechanism plan indicates held-out test access")
    if value.get("scientific_confirmation") is not False or value.get("circuit_found") is not False:
        raise ValueError("mechanism plan contains an invalid scientific claim")


def _json_digest(value: Any) -> str:
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def _complete_sources(
    pairs: Sequence[CounterfactualPair],
) -> tuple[list[CounterfactualPair], list[str]]:
    kinds_by_source: dict[str, set[str]] = {}
    for pair in pairs:
        kinds_by_source.setdefault(pair.source_example_id, set()).add(pair.kind)
    required = {"query_swap", "value_binding_swap"}
    complete_sources = {
        source for source, kinds in kinds_by_source.items() if required.issubset(kinds)
    }
    incomplete_sources = sorted(set(kinds_by_source) - complete_sources)
    selected = [
        pair
        for pair in sorted(pairs, key=lambda item: item.counterfactual_id)
        if pair.source_example_id in complete_sources
    ]
    if not selected:
        raise ValueError("no discovery examples support the complete mechanism experiment matrix")
    return selected, incomplete_sources


def _role_manifest(
    pairs: Sequence[CounterfactualPair],
    *,
    role: str,
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    source_count = len({pair.source_example_id for pair in pairs})
    manifest = build_counterfactual_manifest(pairs, {}, source_example_count=source_count)
    manifest["analysis_role"] = role
    manifest["partition_version"] = partition["partition_version"]
    return manifest


def _runtime_identity(
    runtime: MechanismRuntime, *, requested_revision: str
) -> dict[str, Any]:
    return {
        "model_id": runtime.model_id,
        "requested_revision": requested_revision,
        "resolved_revision": runtime.resolved_revision,
        "tokenizer_id": runtime.tokenizer_id,
        "dtype": runtime.dtype,
    }


def fit_discovery_artifacts(
    mechanism_plan: Mapping[str, Any],
    examples: Sequence[ExamplePair],
    tokenizer: Tokenizer,
    runtime: MechanismRuntime,
    output_root: Path,
    *,
    source_artifacts: Mapping[str, Mapping[str, str]],
    requested_revision: str,
    max_rank: int = 8,
    partition_policy: MechanismPartitionPolicy | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build, partition, fit, and publish all pre-execution mechanism artifacts."""

    _guard_mechanism_plan(mechanism_plan)
    existing = [output_root / name for name in _OUTPUT_NAMES if (output_root / name).exists()]
    if existing and not force:
        raise RuntimeError(
            "mechanism fit outputs already exist; use --force: "
            + ", ".join(str(path) for path in existing)
        )
    campaign = _campaign(mechanism_plan.get("campaign"))
    if not examples or any(example.split != "discovery" for example in examples):
        raise ValueError("mechanism fitting accepts discovery examples only")

    all_pairs, rejected = build_discovery_counterfactuals(examples, tokenizer)
    complete_pairs, incomplete_sources = _complete_sources(all_pairs)
    partition = build_mechanism_partition(complete_pairs, policy=partition_policy)
    fit_pairs = select_partition_role(complete_pairs, partition, "alignment_fit")
    eval_pairs = select_partition_role(complete_pairs, partition, "mechanism_eval")

    alignments, alignment_report = fit_runtime_alignments(
        runtime,
        campaign,
        fit_pairs,
        max_rank=max_rank,
    )
    full_manifest = build_counterfactual_manifest(
        all_pairs,
        rejected,
        source_example_count=len(examples),
    )
    full_manifest["analysis_partition"] = partition
    fit_manifest = _role_manifest(
        fit_pairs,
        role="alignment_fit",
        partition=partition,
    )
    eval_manifest = _role_manifest(
        eval_pairs,
        role="mechanism_eval",
        partition=partition,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    fit_counterfactual_path = output_root / "alignment_fit_counterfactuals.json"
    eval_counterfactual_path = output_root / "mechanism_eval_counterfactuals.json"
    full_counterfactual_path = output_root / "counterfactuals.json"
    alignment_path = output_root / "alignments.json"
    fit_report_path = output_root / "fit_report.json"
    run_manifest_path = output_root / "run_manifest.json"

    alignment_sources = {
        name: dict(record) for name, record in sorted(source_artifacts.items())
    }
    alignment_sources["alignment_fit_counterfactuals"] = {
        "path": str(fit_counterfactual_path),
        "sha256": _json_digest(fit_manifest),
    }
    alignment_manifest = build_alignment_manifest(
        alignments,
        source_artifacts=alignment_sources,
        model_identity=_runtime_identity(runtime, requested_revision=requested_revision),
    )
    fit_report = {
        "schema_version": 1,
        "workflow_version": FIT_WORKFLOW_VERSION,
        "interpretation_scope": "exploratory_discovery_only",
        "source_example_count": len(examples),
        "generated_counterfactual_count": len(all_pairs),
        "complete_counterfactual_count": len(complete_pairs),
        "incomplete_source_example_count": len(incomplete_sources),
        "incomplete_source_example_ids": incomplete_sources,
        "counterfactual_rejections": rejected,
        "partition": partition,
        "alignment_report": alignment_report,
        "model_identity": _runtime_identity(runtime, requested_revision=requested_revision),
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }

    values = {
        full_counterfactual_path: full_manifest,
        fit_counterfactual_path: fit_manifest,
        eval_counterfactual_path: eval_manifest,
        alignment_path: alignment_manifest,
        fit_report_path: fit_report,
    }
    for path, value in values.items():
        write_json_durable(path, value)
    expected_fit_hash = alignment_sources["alignment_fit_counterfactuals"]["sha256"]
    if sha256(fit_counterfactual_path) != expected_fit_hash:
        raise RuntimeError("published alignment-fit counterfactual hash drifted")
    artifact_hashes = {
        path.name: sha256(path) for path in sorted(values, key=lambda item: item.name)
    }
    run_manifest = {
        "schema_version": 1,
        "workflow_version": FIT_WORKFLOW_VERSION,
        "status": "complete",
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


def run_fit_cli(
    discovery_root: Path,
    mechanism_plan_path: Path,
    output_root: Path,
    *,
    device: str,
    max_rank: int,
    fit_fraction: float,
    force: bool,
) -> dict[str, Any]:
    """Verify frozen discovery provenance, load the real model, and fit alignments."""

    contract = _verify_frozen_discovery(discovery_root)
    matched_path = discovery_root / "matched_dataset.jsonl"
    examples = _read_discovery_examples(matched_path)
    mechanism_plan = _read_object(mechanism_plan_path, "mechanism plan")
    model = contract.get("model")
    load_revision = contract.get("load_revision")
    requested_revision = contract.get("requested_revision")
    if not all(
        isinstance(value, str) and value
        for value in (model, load_revision, requested_revision)
    ):
        raise RuntimeError("frozen discovery model identity is incomplete")
    adapter = PythiaAdapter(model, load_revision, device)
    _verify_validation_adapter(adapter, contract)
    runtime = TransformerLensMechanismRuntime(adapter)
    sources = {
        "discovery_run_manifest": _source_record(discovery_root / "run_manifest.json"),
        "matched_discovery_dataset": _source_record(matched_path),
        "mechanism_plan": _source_record(mechanism_plan_path),
    }
    return fit_discovery_artifacts(
        mechanism_plan,
        examples,
        adapter.tokenizer,
        runtime,
        output_root,
        source_artifacts=sources,
        requested_revision=requested_revision,
        max_rank=max_rank,
        partition_policy=MechanismPartitionPolicy(fit_fraction=fit_fraction),
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--mechanism-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-rank", type=int, default=8)
    parser.add_argument("--fit-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_fit_cli(
            args.discovery_root,
            args.mechanism_plan,
            args.output_root,
            device=args.device,
            max_rank=args.max_rank,
            fit_fraction=args.fit_fraction,
            force=args.force,
        )
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1
    print(f"Mechanism fit complete: {args.output_root}")
    print(f"Artifacts: {len(manifest['artifact_hashes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
