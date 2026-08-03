"""One-command discovery diagnosis and preregistered dataset-v2 search."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from autocircuit import __version__
from autocircuit.baseline import (
    BaselineMetrics,
    ExampleResult,
    make_example_result,
    make_failed_result,
    metrics_from_differences,
    successful_difference,
    write_example_results,
)
from autocircuit.candidates import (
    SEED_NAMESPACE,
    SELECTION_RULE,
    candidate_is_eligible,
    candidate_registry,
    frozen_config,
    frozen_toml,
    select_candidate,
    strongest_template,
)
from autocircuit.config import load_config
from autocircuit.datasets.associative_recall import (
    ExamplePair,
    generate_split,
    generation_seed_material,
    read_jsonl,
    write_jsonl,
)
from autocircuit.diagnostics import build_diagnostics, read_results, write_reports
from autocircuit.position_study import (
    decision_status,
    generate_matched_position_dataset,
    json_has_only_finite_numbers,
    paired_position_effects,
    position_metrics,
    secondary_summaries,
    validate_matched_dataset,
    validate_scoring_identity,
    write_dataset,
)
from autocircuit.runtime import select_device


class Adapter(Protocol):
    tokenizer: Any
    model_id: str
    revision: str
    device: str
    dtype: str
    resolved_revision: str | None
    revision_resolution_error: str | None
    tokenizer_id: str

    def score(self, examples: list[ExamplePair], batch_size: int) -> list[ExampleResult]: ...


class PythiaAdapter:
    def __init__(self, model: str, revision: str, requested_device: str) -> None:
        import torch
        from transformer_lens import HookedTransformer

        self.device = select_device(requested_device, torch.cuda)
        self.model_id = f"EleutherAI/{model}"
        self.revision = revision
        self.model = HookedTransformer.from_pretrained(model, device=self.device, revision=revision)
        self.tokenizer = self.model.tokenizer
        self.tokenizer_id = str(getattr(self.tokenizer, "name_or_path", self.model_id))
        self.dtype = str(self.model.cfg.dtype)
        resolved = getattr(getattr(self.model, "cfg", None), "_commit_hash", None)
        resolved = resolved or getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        self.resolved_revision = str(resolved) if resolved else None
        self.revision_resolution_error = (
            None if resolved else "commit SHA unavailable from loaded model"
        )

    def score(self, examples: list[ExamplePair], batch_size: int) -> list[ExampleResult]:
        import torch

        indexed = list(enumerate(examples))
        buckets: dict[int, list[tuple[int, ExamplePair]]] = {}
        for entry in indexed:
            buckets.setdefault(int(entry[1].metadata["prompt_token_length"]), []).append(entry)
        records: list[ExampleResult | None] = [None] * len(examples)
        with torch.inference_mode():
            for length in sorted(buckets):
                group = buckets[length]
                for start in range(0, len(group), batch_size):
                    chunk = group[start : start + batch_size]
                    items = [item for _, item in chunk]
                    try:
                        clean = self.model(
                            [item.clean_prompt for item in items], return_type="logits"
                        )[:, -1]
                        corrupt = self.model(
                            [item.corrupt_prompt for item in items], return_type="logits"
                        )[:, -1]
                        for row, (position, item) in enumerate(chunk):
                            records[position] = make_example_result(
                                item,
                                float(clean[row, item.target_token_id]),
                                float(clean[row, item.distractor_token_id]),
                                float(corrupt[row, item.target_token_id]),
                                float(corrupt[row, item.distractor_token_id]),
                            )
                    except (RuntimeError, ValueError) as exc:
                        for position, item in chunk:
                            records[position] = make_failed_result(item, exc)
        if any(record is None for record in records):
            raise RuntimeError("internal scoring error: missing result record")
        return [record for record in records if record is not None]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_resume(path: Path, expected_hash: str) -> None:
    if not path.is_file() or sha256(path) != expected_hash:
        raise RuntimeError(f"resume hash verification failed: {path}")


def _json(path: Path, value: Any) -> None:
    path.write_bytes(
        (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _position_fingerprint(args: argparse.Namespace) -> str:
    value = {
        "study": "matched-position-v1",
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_position_study(
    args: argparse.Namespace, adapter_factory: Callable[[str, str, str], Adapter] = PythiaAdapter
) -> dict[str, Any]:
    """Run the preregistered discovery-only matched-position experiment."""
    if args.model != "pythia-70m":
        raise ValueError("Pythia-70M is the only supported model")
    root = args.output / f"seed-{args.seed}-{args.revision.replace('/', '_')}"
    fingerprint = _position_fingerprint(args)
    if root.exists() and not (args.resume or args.force):
        raise RuntimeError(f"output exists: {root}; use --resume or --force")
    if root.exists() and args.resume:
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("resume requested but run manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_fingerprint") != fingerprint:
            raise RuntimeError("resume inputs do not match the recorded command/model")
        _validate_checkpoints(root, manifest)
        if manifest.get("complete") is True:
            final: dict[str, Any] = json.loads(
                (root / "final_status.json").read_text(encoding="utf-8")
            )
            print(f"Position study resumed: {final['status']}")
            print(f"Artifacts: {root}")
            return final
        state = manifest
    else:
        state = {
            "schema_version": 1,
            "study": "matched-position-v1",
            "input_fingerprint": fingerprint,
            "artifact_hashes": {},
            "complete": False,
        }
    if args.force and root.exists():
        import shutil

        shutil.rmtree(root)
        state = {
            "schema_version": 1,
            "study": "matched-position-v1",
            "input_fingerprint": fingerprint,
            "artifact_hashes": {},
            "complete": False,
        }
    root.mkdir(parents=True, exist_ok=True)
    _json(root / "run_manifest.json", state)
    adapter = adapter_factory(args.model, args.revision, args.device)

    dataset_path = root / "matched_dataset.jsonl"
    if "matched_dataset.jsonl" in state["artifact_hashes"]:
        examples = read_jsonl(dataset_path)
    else:
        examples = generate_matched_position_dataset(adapter.tokenizer, args.seed)
        write_dataset(dataset_path, examples)
        _checkpoint(root, state, dataset_path)
    # This is intentionally immediately before scoring, and raises on any mismatch.
    balance = validate_matched_dataset(examples, adapter.tokenizer)
    balance_path = root / "balance_report.json"
    _json(balance_path, balance)
    _checkpoint(root, state, balance_path)

    records_path = root / "examples.jsonl"
    if "examples.jsonl" in state["artifact_hashes"]:
        records = read_results(records_path)
    else:
        records = adapter.score(examples, args.batch_size)
        write_example_results(records_path, records)
        _checkpoint(root, state, records_path)
    validate_scoring_identity(examples, records)
    metrics = position_metrics(records)
    effects = paired_position_effects(records, seed=args.seed)
    secondary = secondary_summaries(records)
    metrics_path = root / "position_metrics.json"
    effects_path = root / "paired_position_effects.json"
    _json(metrics_path, metrics)
    _checkpoint(root, state, metrics_path)
    _json(effects_path, effects)
    _checkpoint(root, state, effects_path)
    secondary_path = root / "secondary_summaries.json"
    _json(secondary_path, secondary)
    _checkpoint(root, state, secondary_path)
    status = decision_status(metrics, effects, balance)
    final = {
        "status": status,
        "software_success": True,
        "scientific_eligibility": status.startswith("QUERY_FIRST"),
        "held_out_splits_opened": False,
        "activation_patching_performed": False,
    }
    report_lines = [
        "# Matched position discovery study",
        "",
        "## Preregistered primary results",
        "",
        f"**Decision status:** `{status}`",
        "",
        "| Position | Clean acc. | Corrupt acc. | Clean LD | Corrupt LD | Contrast | "
        "Joint | Examples | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for position in ("first", "interior", "last"):
        row = metrics[position]
        if row["example_count"]:
            report_lines.append(
                f"| {position} | {row['clean_pairwise_accuracy']:.6f} | "
                f"{row['corrupt_pairwise_accuracy']:.6f} | "
                f"{row['mean_clean_logit_difference']:.6f} | "
                f"{row['mean_corrupt_logit_difference']:.6f} | "
                f"{row['clean_corrupt_contrast']:.6f} | {row['joint_success_rate']:.6f} | "
                f"{row['example_count']} | {row['failed_count']} |"
            )
        else:
            report_lines.append(
                f"| {position} | n/a | n/a | n/a | n/a | n/a | n/a | 0 | "
                f"{row['failed_count']} |"
            )
    report_lines += [
        "",
        "The paired estimates and deterministic 95% family-bootstrap intervals are in "
        "`paired_position_effects.json`; the resampling unit is `matched_family_id`.",
        "",
        "## Secondary lexical/entity summaries",
        "",
        "Target, distractor, ordered-pair, and query-entity population summaries are reported "
        "in `balance_report.json`. These summaries are secondary and do not alter the "
        "decision rule.",
        "",
        "Descriptive performance by target token, distractor token, and query entity is in "
        "`secondary_summaries.json`, including counts, accuracies, logit differences, and "
        "contrasts. These results are secondary and never enter eligibility.",
        "",
        "## Software and scientific status",
        "",
        "- Software success: **yes** (scientific failure is a successful software outcome).",
        f"- Scientific eligibility: **{'yes' if final['scientific_eligibility'] else 'no'}**.",
        "- Held-out validation and test splits were **not opened**.",
        "- Activation patching was not performed.",
    ]
    report_path = root / "position_study.md"
    report_path.write_bytes(("\n".join(report_lines) + "\n").encode())
    _checkpoint(root, state, report_path)
    final_path = root / "final_status.json"
    _json(final_path, final)
    _checkpoint(root, state, final_path)
    if not json_has_only_finite_numbers(
        {"metrics": metrics, "effects": effects, "secondary": secondary}
    ):
        raise RuntimeError("non-finite value in JSON output")
    state |= {
        "complete": True,
        "git_commit": _git_commit(),
        "model_id": adapter.model_id,
        "tokenizer_id": adapter.tokenizer_id,
        "requested_revision": args.revision,
        "resolved_revision": adapter.resolved_revision,
        "device": adapter.device,
        "dtype": adapter.dtype,
        "expected_example_count": 360,
        "processed_example_count": len(records),
        "command_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    _json(root / "run_manifest.json", state)
    print(f"Position study: {status}")
    print(f"Artifacts: {root}")
    return final


def _fingerprint(args: argparse.Namespace, config_hash: str) -> str:
    value = {
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "config_hash": config_hash,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _checkpoint(root: Path, state: dict[str, Any], path: Path) -> None:
    state["artifact_hashes"][str(path.relative_to(root)).replace("\\", "/")] = sha256(path)
    _json(root / "run_manifest.json", state)


def _validate_checkpoints(root: Path, state: dict[str, Any]) -> None:
    for relative, digest in state.get("artifact_hashes", {}).items():
        verify_resume(root / relative, str(digest))


def _validate_external_artifacts(state: dict[str, Any]) -> None:
    frozen = state.get("external_artifacts", {}).get("frozen_v2_config")
    if frozen is not None:
        path = Path("configs/mvp_v2.toml")
        if frozen.get("logical_identity") != "configs/mvp_v2.toml":
            raise RuntimeError("invalid frozen v2 logical identity in run manifest")
        verify_resume(path, str(frozen["sha256"]))


def _metrics(records: list[ExampleResult]) -> BaselineMetrics:
    valid = [r for r in records if r.processing_status == "ok"]
    return metrics_from_differences(
        [successful_difference(r.clean_logit_difference) for r in valid],
        [successful_difference(r.corrupt_logit_difference) for r in valid],
        failed_count=len(records) - len(valid),
    )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_discovery(
    args: argparse.Namespace, adapter_factory: Callable[[str, str, str], Adapter] = PythiaAdapter
) -> dict[str, Any]:
    if args.model != "pythia-70m":
        raise ValueError("Pythia-70M is the only supported MVP model")
    config = load_config(Path("configs/mvp.toml"))
    config_hash = sha256(Path("configs/mvp.toml"))
    fingerprint = _fingerprint(args, config_hash)
    run_id = f"seed-{args.seed}-{args.revision.replace('/', '_')}"
    root = args.output / run_id
    if root.exists() and not (args.resume or args.force):
        raise RuntimeError(f"output exists: {root}; use --resume or --force")
    if root.exists() and args.resume:
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("resume requested but run manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_fingerprint") != fingerprint:
            raise RuntimeError("resume inputs do not match the recorded command/config/model")
        _validate_checkpoints(root, manifest)
        _validate_external_artifacts(manifest)
        if manifest.get("complete") is True:
            final_path = root / "final_status.json"
            if "final_status.json" not in manifest.get("artifact_hashes", {}):
                raise RuntimeError("completed resume has no hashed final status")
            final: dict[str, Any] = json.loads(final_path.read_text(encoding="utf-8"))
            print(f"Discovery pipeline resumed: {final['status']}")
            print(f"Artifacts: {root}")
            return final
        state = manifest
    else:
        state = {
            "schema_version": 2,
            "input_fingerprint": fingerprint,
            "artifact_hashes": {},
            "complete": False,
        }
    adapter = adapter_factory(args.model, args.revision, args.device)
    if args.force and root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    _json(root / "run_manifest.json", state)

    v1_dir = root / "v1"
    v1_dataset = v1_dir / "dataset"
    v1_dataset.mkdir(parents=True, exist_ok=True)
    v1_dataset_path = v1_dataset / "discovery.jsonl"
    if "v1/dataset/discovery.jsonl" in state["artifact_hashes"]:
        v1_examples = read_jsonl(v1_dataset_path)
        v1_rejections: dict[str, int] = state.get("v1_rejections", {})
    else:
        v1_examples, v1_rejections = generate_split(
            "discovery", config.discovery_examples, args.seed, adapter.tokenizer
        )
        write_jsonl(v1_dataset_path, v1_examples)
        state["v1_rejections"] = v1_rejections
        _checkpoint(root, state, v1_dataset_path)
    dataset_hash = sha256(v1_dataset_path)
    v1_records_path = v1_dir / "examples.jsonl"
    if "v1/examples.jsonl" in state["artifact_hashes"]:
        v1_records = read_results(v1_records_path)
    else:
        v1_records = adapter.score(v1_examples, args.batch_size)
        write_example_results(v1_records_path, v1_records)
        _checkpoint(root, state, v1_records_path)
    records_hash = sha256(v1_records_path)
    v1_metrics = _metrics(v1_records)
    v1_baseline_path = v1_dir / "baseline_results.json"
    _json(
        v1_baseline_path,
        {"metrics": asdict(v1_metrics), "rejections": v1_rejections},
    )
    _checkpoint(root, state, v1_baseline_path)
    diagnostics = build_diagnostics(v1_records)
    write_reports(diagnostics, v1_dir / "diagnostics.json", v1_dir / "diagnostics.md")
    _checkpoint(root, state, v1_dir / "diagnostics.json")
    _checkpoint(root, state, v1_dir / "diagnostics.md")

    registry = candidate_registry(strongest_template(diagnostics))
    registry_payload = json.dumps([asdict(item) for item in registry], sort_keys=True)
    registry_hash = hashlib.sha256(registry_payload.encode()).hexdigest()
    if state.get("candidate_registry_hash", registry_hash) != registry_hash:
        raise RuntimeError("candidate registry changed since the interrupted run")
    state["candidate_registry_hash"] = registry_hash
    state["candidate_seed_material"] = {
        item.candidate_id: generation_seed_material(
            args.seed, "discovery", SEED_NAMESPACE + ":" + item.candidate_id
        )
        for item in registry
    }
    candidate_metrics: dict[str, BaselineMetrics] = {}
    candidate_processed_counts: dict[str, int] = {}
    comparison: list[dict[str, Any]] = []
    dataset_hashes = {"v1": dataset_hash}
    for candidate in registry:
        candidate_dir = root / "candidates" / candidate.candidate_id
        dataset_dir = candidate_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / "discovery.jsonl"
        dataset_key = f"candidates/{candidate.candidate_id}/dataset/discovery.jsonl"
        if dataset_key in state["artifact_hashes"]:
            examples = read_jsonl(dataset_path)
            rejections = state.get("candidate_rejections", {}).get(candidate.candidate_id, {})
        else:
            examples, rejections = generate_split(
                "discovery",
                config.discovery_examples,
                args.seed,
                adapter.tokenizer,
                parameters=candidate.parameters,
                seed_namespace=SEED_NAMESPACE + ":" + candidate.candidate_id,
            )
            write_jsonl(dataset_path, examples)
            state.setdefault("candidate_rejections", {})[candidate.candidate_id] = rejections
            _checkpoint(root, state, dataset_path)
        digest = sha256(dataset_path)
        dataset_hashes[candidate.candidate_id] = digest
        candidate_records_path = candidate_dir / "examples.jsonl"
        records_key = f"candidates/{candidate.candidate_id}/examples.jsonl"
        if records_key in state["artifact_hashes"]:
            records = read_results(candidate_records_path)
        else:
            records = adapter.score(examples, args.batch_size)
            write_example_results(candidate_records_path, records)
            _checkpoint(root, state, candidate_records_path)
        valid_records = [record for record in records if record.processing_status == "ok"]
        candidate_processed_counts[candidate.candidate_id] = len(records)
        metrics = _metrics(records) if valid_records else None
        if metrics is not None:
            candidate_metrics[candidate.candidate_id] = metrics
        evaluation_status = (
            "NO_VALID_EXAMPLES"
            if metrics is None
            else "PROCESSING_FAILED"
            if metrics.failed_count
            else "COMPLETE"
        )
        summary = {
            "candidate": asdict(candidate),
            "evaluation_status": evaluation_status,
            "metrics": asdict(metrics) if metrics is not None else None,
            "eligible": metrics is not None and candidate_is_eligible(metrics, len(examples)),
            "expected_example_count": len(examples),
            "processed_example_count": len(records),
            "pre_model_rejections": rejections,
        }
        comparison.append(summary)
        candidate_baseline_path = candidate_dir / "baseline_results.json"
        _json(candidate_baseline_path, summary)
        _checkpoint(root, state, candidate_baseline_path)
    selected_id = select_candidate(
        candidate_metrics, {item.candidate_id: config.discovery_examples for item in registry}
    )
    status = "SELECTED_REQUIRES_HELD_OUT_VALIDATION" if selected_id else "NO_ELIGIBLE_CONFIGURATION"
    comparison_document = {
        "frozen_thresholds": {"clean_accuracy": 0.80, "clean_mean_logit_difference": 1.0},
        "selection_rule": SELECTION_RULE,
        "attempts": comparison,
        "selected_candidate_id": selected_id,
    }
    _json(root / "candidate_comparison.json", comparison_document)
    _checkpoint(root, state, root / "candidate_comparison.json")
    lines = [
        "# Candidate comparison",
        "",
        f"**Result:** {status}",
        "",
        "| Candidate | Eligible | Clean accuracy | Mean clean LD | Contrast |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison:
        metric = row["metrics"]
        if metric is None:
            lines.append(
                f"| {row['candidate']['candidate_id']} | False ({row['evaluation_status']}) | "
                "n/a | n/a | n/a |"
            )
        else:
            lines.append(
                "| {candidate} | {eligible} ({status}) | {accuracy:.4f} | {ld:.4f} | "
                "{contrast:.4f} |".format(
                    candidate=row["candidate"]["candidate_id"],
                    eligible=row["eligible"],
                    status=row["evaluation_status"],
                    accuracy=metric["clean_accuracy"],
                    ld=metric["clean_mean_logit_difference"],
                    contrast=metric["clean_corrupt_contrast"],
                )
            )
    (root / "candidate_comparison.md").write_bytes(("\n".join(lines) + "\n").encode())
    _checkpoint(root, state, root / "candidate_comparison.md")
    if selected_id:
        candidate = next(item for item in registry if item.candidate_id == selected_id)
        selected = frozen_config(
            candidate,
            candidate_metrics[selected_id],
            args.model,
            adapter.tokenizer_id,
            args.revision,
            adapter.resolved_revision,
            args.seed,
        )
        _json(
            root / "selected_v2.json",
            selected,
        )
        _checkpoint(root, state, root / "selected_v2.json")
        frozen_path = Path("configs/mvp_v2.toml")
        payload = frozen_toml(selected)
        expected_hash = hashlib.sha256(payload).hexdigest()
        if frozen_path.exists():
            if args.force:
                frozen_path.write_bytes(payload)
            elif args.resume and frozen_path.read_bytes() == payload:
                pass
            elif frozen_path.read_bytes() != payload:
                raise RuntimeError("configs/mvp_v2.toml differs from the selected frozen contract")
            else:
                raise RuntimeError("configs/mvp_v2.toml exists; use --resume to reuse or --force")
        else:
            frozen_path.write_bytes(payload)
        state.setdefault("external_artifacts", {})["frozen_v2_config"] = {
            "logical_identity": "configs/mvp_v2.toml",
            "sha256": expected_hash,
        }
        state["frozen_v2_config_hash"] = expected_hash
        _json(root / "run_manifest.json", state)
    final = {
        "status": status,
        "software_success": True,
        "scientific_eligibility": selected_id is not None,
        "causal_scanning_performed": False,
        "held_out_splits_opened": False,
    }
    _json(root / "final_status.json", final)
    _checkpoint(root, state, root / "final_status.json")
    import torch

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = torch.cuda.get_device_name() if cuda_available else None
    manifest = state | {
        "git_commit": _git_commit(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": _version("torch"),
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "transformer_lens_version": _version("transformer-lens"),
        "transformers_version": _version("transformers"),
        "model_id": adapter.model_id,
        "tokenizer_id": adapter.tokenizer_id,
        "requested_revision": args.revision,
        "resolved_revision": adapter.resolved_revision,
        "revision_resolution_error": adapter.revision_resolution_error,
        "device": adapter.device,
        "dtype": adapter.dtype,
        "seeds": {
            "base": args.seed,
            "candidate_namespace": SEED_NAMESPACE,
            "candidate_material": state["candidate_seed_material"],
        },
        "config_hashes": {"mvp": config_hash, "candidate_registry": registry_hash},
        "dataset_hashes": dataset_hashes,
        "record_hashes": {"v1": records_hash},
        "expected_example_counts": {"v1": config.discovery_examples}
        | {item.candidate_id: config.discovery_examples for item in registry},
        "processed_example_counts": {"v1": v1_metrics.example_count + v1_metrics.failed_count}
        | candidate_processed_counts,
        "command_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "created_at": datetime.now(UTC).isoformat(),
        "complete": True,
    }
    _json(root / "run_manifest.json", manifest)
    print(f"Discovery pipeline: {status}")
    print(f"Artifacts: {root}")
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser(
        "discovery", help="run discovery-only diagnosis and candidate search"
    )
    discovery.add_argument("--model", default="pythia-70m", choices=["pythia-70m"])
    discovery.add_argument("--revision", default="main")
    discovery.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    discovery.add_argument("--batch-size", type=int, default=8)
    discovery.add_argument("--seed", type=int, default=42)
    discovery.add_argument("--output", type=Path, default=Path("artifacts/discovery_pipeline"))
    reuse = discovery.add_mutually_exclusive_group()
    reuse.add_argument("--resume", action="store_true")
    reuse.add_argument("--force", action="store_true")
    position = commands.add_parser(
        "position-study", help="run the discovery-only matched query-position study"
    )
    position.add_argument("--model", default="pythia-70m", choices=["pythia-70m"])
    position.add_argument("--revision", default="main")
    position.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    position.add_argument("--batch-size", type=int, default=8)
    position.add_argument("--seed", type=int, default=42)
    position.add_argument("--output", type=Path, default=Path("artifacts/position_study"))
    position_reuse = position.add_mutually_exclusive_group()
    position_reuse.add_argument("--resume", action="store_true")
    position_reuse.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.seed < 0:
        parser.error("batch size must be positive and seed non-negative")
    try:
        if args.command == "position-study":
            run_position_study(args)
        else:
            run_discovery(args)
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
