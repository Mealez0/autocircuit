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
    metrics_from_differences,
    successful_difference,
    write_example_results,
)
from autocircuit.candidates import (
    SEED_NAMESPACE,
    SELECTION_RULE,
    candidate_registry,
    frozen_config,
    select_candidate,
    strongest_template,
)
from autocircuit.config import load_config
from autocircuit.datasets.associative_recall import ExamplePair, generate_split, write_jsonl
from autocircuit.diagnostics import build_diagnostics, write_reports
from autocircuit.runtime import select_device


class Adapter(Protocol):
    tokenizer: Any
    model_id: str
    revision: str
    device: str
    dtype: str

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
        self.dtype = str(self.model.cfg.dtype)

    def score(self, examples: list[ExamplePair], batch_size: int) -> list[ExampleResult]:
        records: list[ExampleResult] = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            for item in batch:
                clean = self.model(item.clean_prompt, return_type="logits")[0, -1]
                corrupt = self.model(item.corrupt_prompt, return_type="logits")[0, -1]
                records.append(
                    make_example_result(
                        item,
                        float(clean[item.target_token_id]),
                        float(clean[item.distractor_token_id]),
                        float(corrupt[item.target_token_id]),
                        float(corrupt[item.distractor_token_id]),
                    )
                )
        return records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_resume(path: Path, expected_hash: str) -> None:
    if not path.is_file() or sha256(path) != expected_hash:
        raise RuntimeError(f"resume hash verification failed: {path}")


def _json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


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
    run_id = f"seed-{args.seed}-{args.revision.replace('/', '_')}"
    root = args.output / run_id
    if root.exists() and not (args.resume or args.force):
        raise RuntimeError(f"output exists: {root}; use --resume or --force")
    if root.exists() and args.resume:
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("resume requested but run manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for candidate_id, digest in manifest["dataset_hashes"].items():
            dataset_path = (
                root / "v1" / "dataset" / "discovery.jsonl"
                if candidate_id == "v1"
                else root / "candidates" / candidate_id / "dataset" / "discovery.jsonl"
            )
            verify_resume(dataset_path, str(digest))
        verify_resume(root / "v1" / "examples.jsonl", manifest["record_hashes"]["v1"])
        final: dict[str, Any] = json.loads((root / "final_status.json").read_text(encoding="utf-8"))
        print(f"Discovery pipeline resumed: {final['status']}")
        print(f"Artifacts: {root}")
        return final
    adapter = adapter_factory(args.model, args.revision, args.device)
    if args.force and root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    v1_dir = root / "v1"
    v1_dataset = v1_dir / "dataset"
    v1_dataset.mkdir(parents=True, exist_ok=True)
    v1_examples, v1_rejections = generate_split(
        "discovery", config.discovery_examples, args.seed, adapter.tokenizer
    )
    dataset_hash = write_jsonl(v1_dataset / "discovery.jsonl", v1_examples)
    v1_records = adapter.score(v1_examples, args.batch_size)
    records_hash = write_example_results(v1_dir / "examples.jsonl", v1_records)
    v1_metrics = _metrics(v1_records)
    _json(
        v1_dir / "baseline_results.json",
        {"metrics": asdict(v1_metrics), "rejections": v1_rejections},
    )
    diagnostics = build_diagnostics(v1_records)
    write_reports(diagnostics, v1_dir / "diagnostics.json", v1_dir / "diagnostics.md")

    registry = candidate_registry(strongest_template(diagnostics))
    candidate_metrics: dict[str, BaselineMetrics] = {}
    comparison: list[dict[str, Any]] = []
    dataset_hashes = {"v1": dataset_hash}
    for candidate in registry:
        candidate_dir = root / "candidates" / candidate.candidate_id
        dataset_dir = candidate_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        examples, rejections = generate_split(
            "discovery",
            config.discovery_examples,
            args.seed,
            adapter.tokenizer,
            parameters=candidate.parameters,
            seed_namespace=SEED_NAMESPACE + ":" + candidate.candidate_id,
        )
        digest = write_jsonl(dataset_dir / "discovery.jsonl", examples)
        dataset_hashes[candidate.candidate_id] = digest
        records = adapter.score(examples, args.batch_size)
        write_example_results(candidate_dir / "examples.jsonl", records)
        metrics = _metrics(records)
        candidate_metrics[candidate.candidate_id] = metrics
        summary = {
            "candidate": asdict(candidate),
            "metrics": asdict(metrics),
            "eligible": metrics.clean_accuracy >= 0.80
            and metrics.clean_mean_logit_difference >= 1.0,
            "pre_model_rejections": rejections,
        }
        comparison.append(summary)
        _json(candidate_dir / "baseline_results.json", summary)
    selected_id = select_candidate(candidate_metrics)
    status = "SELECTED_REQUIRES_HELD_OUT_VALIDATION" if selected_id else "NO_ELIGIBLE_CONFIGURATION"
    comparison_document = {
        "frozen_thresholds": {"clean_accuracy": 0.80, "clean_mean_logit_difference": 1.0},
        "selection_rule": SELECTION_RULE,
        "attempts": comparison,
        "selected_candidate_id": selected_id,
    }
    _json(root / "candidate_comparison.json", comparison_document)
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
        lines.append(
            "| {candidate} | {eligible} | {accuracy:.4f} | {ld:.4f} | {contrast:.4f} |".format(
                candidate=row["candidate"]["candidate_id"],
                eligible=row["eligible"],
                accuracy=metric["clean_accuracy"],
                ld=metric["clean_mean_logit_difference"],
                contrast=metric["clean_corrupt_contrast"],
            )
        )
    (root / "candidate_comparison.md").write_bytes(("\n".join(lines) + "\n").encode())
    if selected_id:
        candidate = next(item for item in registry if item.candidate_id == selected_id)
        _json(
            root / "selected_v2.json",
            frozen_config(candidate, candidate_metrics[selected_id], args.model, args.revision),
        )
    final = {
        "status": status,
        "software_success": True,
        "scientific_eligibility": selected_id is not None,
        "causal_scanning_performed": False,
        "held_out_splits_opened": False,
    }
    _json(root / "final_status.json", final)
    manifest = {
        "git_commit": _git_commit(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": _version("torch"),
        "cuda_available": adapter.device == "cuda",
        "cuda_device_name": None,
        "transformer_lens_version": _version("transformer-lens"),
        "transformers_version": _version("transformers"),
        "model_id": adapter.model_id,
        "requested_revision": args.revision,
        "resolved_revision": adapter.revision,
        "device": adapter.device,
        "dtype": adapter.dtype,
        "seeds": {"base": args.seed, "candidate_namespace": SEED_NAMESPACE},
        "config_hashes": {"mvp": sha256(Path("configs/mvp.toml"))},
        "dataset_hashes": dataset_hashes,
        "record_hashes": {"v1": records_hash},
        "command_arguments": vars(args),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest["command_arguments"]["output"] = str(args.output)
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
    discovery.add_argument("--resume", action="store_true")
    discovery.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.seed < 0:
        parser.error("batch size must be positive and seed non-negative")
    try:
        run_discovery(args)
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
