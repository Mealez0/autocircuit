"""One-command discovery diagnosis and preregistered dataset-v2 search."""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from autocircuit import __version__
from autocircuit.artifacts import checkpoint as _checkpoint
from autocircuit.artifacts import sha256, verify_resume
from autocircuit.artifacts import validate_checkpoints as _validate_checkpoints
from autocircuit.artifacts import write_json as _json
from autocircuit.artifacts import write_json_durable as _validation_json
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
    STUDY_VERSION,
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
from autocircuit.position_validation import (
    DISCOVERY_ELIGIBLE_STATUS,
    FROZEN_PRIMARY_RULE,
    VALIDATED_STATUS,
    VALIDATION_MANIFEST_SCHEMA_VERSION,
    VALIDATION_POOL_ID,
    VALIDATION_PROTOCOL_VERSION,
    VALIDATION_SEED_NAMESPACE,
    generate_validation_dataset,
    validate_validation_dataset,
    validation_decision,
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


def _position_fingerprint_components(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "study_version": STUDY_VERSION,
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }


def _position_fingerprint(args: argparse.Namespace) -> str:
    return hashlib.sha256(
        json.dumps(_position_fingerprint_components(args), sort_keys=True).encode()
    ).hexdigest()


def _verify_frozen_discovery(root: Path) -> dict[str, Any]:
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("frozen discovery manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("study_version") != STUDY_VERSION:
        raise RuntimeError("frozen discovery run is incomplete or has the wrong study version")
    components = manifest.get("input_fingerprint_components")
    fingerprint = manifest.get("input_fingerprint")
    if not isinstance(components, dict) or not isinstance(fingerprint, str):
        raise RuntimeError("frozen discovery fingerprint fields are missing or malformed")
    if components.get("study_version") != STUDY_VERSION:
        raise RuntimeError("frozen discovery fingerprint has the wrong study version")
    model = components.get("model")
    revision = components.get("revision")
    seed = components.get("seed")
    if (
        model != "pythia-70m"
        or not isinstance(revision, str)
        or not revision
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise RuntimeError("frozen discovery model, revision, or seed is malformed")
    recomputed = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()
    if fingerprint != recomputed:
        raise RuntimeError("frozen discovery input fingerprint is inconsistent")
    arguments = manifest.get("command_arguments")
    expected_arguments = {
        "model": model,
        "revision": revision,
        "seed": seed,
        "device": components.get("device"),
        "batch_size": components.get("batch_size"),
    }
    if not isinstance(arguments, dict) or any(
        arguments.get(key) != value for key, value in expected_arguments.items()
    ):
        raise RuntimeError("frozen discovery command arguments are inconsistent")
    if manifest.get("requested_revision") != revision:
        raise RuntimeError("frozen discovery requested revision is inconsistent")
    if manifest.get("model_id") != f"EleutherAI/{model}":
        raise RuntimeError("frozen discovery model identity is inconsistent")
    resolved_revision = manifest.get("resolved_revision")
    resolution_error = manifest.get("revision_resolution_error")
    tokenizer_id = manifest.get("tokenizer_id")
    dtype = manifest.get("dtype")
    discovery_device = manifest.get("device")
    if resolved_revision is not None and (
        not isinstance(resolved_revision, str) or not resolved_revision
    ):
        raise RuntimeError("frozen discovery resolved revision is malformed")
    if resolved_revision is None and "revision_resolution_error" not in manifest:
        # position-study-2.0.0 was released before this diagnostic was written.
        # Preserve compatibility with that exact frozen contract while making
        # the absence explicit in the derived validation contract.
        resolution_error = "not recorded by position-study-2.0.0 manifest"
    if resolved_revision is None and (
        not isinstance(resolution_error, str) or not resolution_error
    ):
        raise RuntimeError("frozen discovery revision resolution metadata is missing")
    if resolved_revision is not None and resolution_error is not None:
        raise RuntimeError("frozen discovery revision resolution metadata is inconsistent")
    if not all(
        isinstance(value, str) and value for value in (tokenizer_id, dtype, discovery_device)
    ):
        raise RuntimeError("frozen discovery tokenizer, dtype, or device is malformed")
    for relative, digest in manifest.get("artifact_hashes", {}).items():
        verify_resume(root / relative, str(digest))
    required = {
        "matched_dataset.jsonl",
        "examples.jsonl",
        "balance_report.json",
        "position_metrics.json",
        "paired_position_effects.json",
        "secondary_summaries.json",
        "position_study.md",
        "final_status.json",
    }
    if not required <= set(manifest.get("artifact_hashes", {})):
        raise RuntimeError("frozen discovery manifest lacks required hashed artifacts")
    final = json.loads((root / "final_status.json").read_text(encoding="utf-8"))
    if not (
        final.get("software_success") is True
        and final.get("status") == DISCOVERY_ELIGIBLE_STATUS
        and final.get("scientific_eligibility") is True
        and final.get("held_out_splits_opened") is False
        and final.get("activation_patching_performed") is False
    ):
        raise RuntimeError("frozen discovery run is not eligible for held-out validation")
    if (
        manifest.get("expected_example_count") != 360
        or manifest.get("processed_example_count") != 360
    ):
        raise RuntimeError("frozen discovery example counts are incomplete")
    dataset = read_jsonl(root / "matched_dataset.jsonl")
    records = read_results(root / "examples.jsonl")
    if len(dataset) != 360 or len(records) != 360:
        raise RuntimeError("frozen discovery raw record counts are incomplete")
    validate_scoring_identity(dataset, records)
    _validate_result_semantics(records)
    if {item.metadata.get("generator_version") for item in dataset} != {STUDY_VERSION}:
        raise RuntimeError("frozen discovery dataset has the wrong generator version")
    balance = json.loads((root / "balance_report.json").read_text(encoding="utf-8"))
    if balance.get("all_invariants_passed") is not True:
        raise RuntimeError("frozen discovery balance invariants did not pass")
    stored_metrics = json.loads((root / "position_metrics.json").read_text(encoding="utf-8"))
    stored_effects = json.loads((root / "paired_position_effects.json").read_text(encoding="utf-8"))
    stored_secondary = json.loads((root / "secondary_summaries.json").read_text(encoding="utf-8"))
    metrics = position_metrics(records)
    effects = paired_position_effects(records, seed=seed)
    secondary = secondary_summaries(records)
    if (metrics, effects, secondary) != (stored_metrics, stored_effects, stored_secondary):
        raise RuntimeError("frozen discovery summaries do not match raw scoring records")
    if not json_has_only_finite_numbers(
        {"metrics": metrics, "effects": effects, "secondary": secondary}
    ):
        raise RuntimeError("frozen discovery raw recomputation contains non-finite values")
    if decision_status(metrics, effects, balance) != DISCOVERY_ELIGIBLE_STATUS:
        raise RuntimeError("frozen discovery artifacts do not recompute as eligible")
    discovery_commit = manifest.get("git_commit")
    if not isinstance(discovery_commit, str) or not discovery_commit:
        raise RuntimeError("frozen discovery commit is missing or malformed")
    return {
        "discovery_commit": discovery_commit,
        "discovery_study_version": STUDY_VERSION,
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "seed": seed,
        "model": model,
        "model_id": manifest.get("model_id"),
        "revision": revision,
        "requested_revision": revision,
        "load_revision": resolved_revision or revision,
        "resolved_revision": resolved_revision,
        "revision_resolution_error": resolution_error,
        "tokenizer_id": tokenizer_id,
        "dtype": dtype,
        "discovery_device": discovery_device,
        "exact_revision_available": resolved_revision is not None,
        "artifact_hashes": {
            "manifest": sha256(manifest_path),
            "dataset": sha256(root / "matched_dataset.jsonl"),
            "records": sha256(root / "examples.jsonl"),
            "metrics": sha256(root / "position_metrics.json"),
            "effects": sha256(root / "paired_position_effects.json"),
            "secondary": sha256(root / "secondary_summaries.json"),
            "final_status": sha256(root / "final_status.json"),
        },
        "frozen_primary_eligibility_rule": FROZEN_PRIMARY_RULE,
        "observed_discovery_metrics": metrics,
        "observed_discovery_effects": effects,
    }


def _validation_fingerprint_components(
    args: argparse.Namespace, contract: dict[str, Any]
) -> dict[str, Any]:
    return {
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "source_discovery_study_version": STUDY_VERSION,
        "discovery_artifact_hashes": contract["artifact_hashes"],
        "device": args.device,
        "batch_size": args.batch_size,
    }


def _validate_result_semantics(records: list[ExampleResult]) -> None:
    numeric_fields = (
        "clean_target_logit",
        "clean_distractor_logit",
        "clean_logit_difference",
        "corrupt_target_logit",
        "corrupt_distractor_logit",
        "corrupt_logit_difference",
        "clean_corrupt_recovery_span",
    )
    for record in records:
        values = [getattr(record, field) for field in numeric_fields]
        if record.processing_status == "ok":
            if record.error is not None or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in values
            ):
                raise RuntimeError("successful scoring record is internally inconsistent")
            assert isinstance(record.clean_target_logit, (int, float))
            assert isinstance(record.clean_distractor_logit, (int, float))
            assert isinstance(record.corrupt_target_logit, (int, float))
            assert isinstance(record.corrupt_distractor_logit, (int, float))
            clean = float(record.clean_target_logit) - float(record.clean_distractor_logit)
            corrupt = float(record.corrupt_target_logit) - float(record.corrupt_distractor_logit)
            if (
                record.clean_logit_difference != clean
                or record.corrupt_logit_difference != corrupt
                or record.clean_correct is not (clean > 0)
                or record.corrupt_correct is not (corrupt < 0)
                or record.clean_corrupt_recovery_span != clean - corrupt
            ):
                raise RuntimeError("successful scoring record derived fields are inconsistent")
        elif record.processing_status == "error":
            if any(value is not None for value in values) or not record.error:
                raise RuntimeError("failed scoring record is internally inconsistent")
            if record.clean_correct is not None or record.corrupt_correct is not None:
                raise RuntimeError("failed scoring record correctness is inconsistent")
        else:
            raise RuntimeError("scoring record has an unknown processing status")


def _verify_validation_adapter(adapter: Adapter, contract: dict[str, Any]) -> dict[str, Any]:
    load_revision = str(contract["load_revision"])
    frozen_resolved = contract["resolved_revision"]
    if adapter.model_id != contract["model_id"] or adapter.model_id != "EleutherAI/pythia-70m":
        raise RuntimeError("loaded validation model ID does not match frozen discovery")
    if adapter.revision != load_revision:
        raise RuntimeError("loaded validation revision does not match intended frozen revision")
    if frozen_resolved is not None and adapter.resolved_revision != frozen_resolved:
        raise RuntimeError("loaded validation resolved revision does not match frozen discovery")
    if adapter.tokenizer_id != contract["tokenizer_id"]:
        raise RuntimeError("loaded validation tokenizer does not match frozen discovery")
    if adapter.dtype != contract["dtype"]:
        raise RuntimeError("loaded validation dtype does not match frozen discovery")
    return {
        "model": contract["model"],
        "model_id": adapter.model_id,
        "requested_revision": contract["requested_revision"],
        "load_revision": load_revision,
        "frozen_resolved_revision": frozen_resolved,
        "validation_resolved_revision": adapter.resolved_revision,
        "exact_revision_matching_succeeded": (
            frozen_resolved is not None and adapter.resolved_revision == frozen_resolved
        ),
        "tokenizer_id": adapter.tokenizer_id,
        "dtype": adapter.dtype,
        "resolved_device": adapter.device,
    }


def _validation_checkpoint(root: Path, state: dict[str, Any], path: Path) -> None:
    state["artifact_hashes"][str(path.relative_to(root)).replace("\\", "/")] = sha256(path)
    _validation_json(root / "validation_manifest.json", state)


def _validate_validation_hash_paths(root: Path, state: dict[str, Any]) -> None:
    hashes = state.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise RuntimeError("validation artifact hashes are missing or malformed")
    normalized: set[str] = set()
    for raw in hashes:
        if not isinstance(raw, str) or "\\" in raw:
            raise RuntimeError("validation artifact hash path is unsafe")
        path = PurePosixPath(raw)
        canonical = path.as_posix()
        if path.is_absolute() or ".." in path.parts or canonical in normalized or canonical != raw:
            raise RuntimeError("validation artifact hash path is unsafe")
        normalized.add(canonical)
        candidate = (root / Path(*path.parts)).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise RuntimeError("validation artifact hash path escapes the output root")


def _validation_receipt_payload(
    contract: dict[str, Any], fingerprint: str, output_root: Path
) -> dict[str, Any]:
    output = str(output_root.resolve())
    claim_id = hashlib.sha256(f"{fingerprint}:{output}".encode()).hexdigest()
    return {
        "manifest_schema_version": VALIDATION_MANIFEST_SCHEMA_VERSION,
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "source_discovery_study_version": STUDY_VERSION,
        "frozen_discovery_artifact_hashes": contract["artifact_hashes"],
        "validation_fingerprint": fingerprint,
        "selected_output_root": output,
        "claim_id": claim_id,
        "resume_state": "claimed_one_shot_validation",
    }


def _untouched_initialized_validation(
    state: dict[str, Any],
    payload: dict[str, Any],
    components: dict[str, Any],
    contract: dict[str, Any],
) -> bool:
    """Return whether a manifest is the exact pre-model recovery checkpoint."""
    forbidden_model_state = {
        "model",
        "model_id",
        "load_revision",
        "frozen_resolved_revision",
        "validation_resolved_revision",
        "tokenizer_id",
        "dtype",
        "resolved_device",
        "processed_example_count",
        "complete_matched_family_count",
    }
    return (
        state.get("schema_version") == VALIDATION_MANIFEST_SCHEMA_VERSION
        and state.get("validation_protocol_version") == VALIDATION_PROTOCOL_VERSION
        and state.get("source_discovery_study_version") == STUDY_VERSION
        and state.get("lifecycle_state") == "initialized"
        and state.get("complete") is False
        and state.get("artifact_hashes") == {}
        and state.get("claim_id") == payload["claim_id"]
        and state.get("output_root") == payload["selected_output_root"]
        and state.get("input_fingerprint") == payload["validation_fingerprint"]
        and state.get("fingerprint_components") == components
        and state.get("frozen_discovery_artifact_hashes") == contract["artifact_hashes"]
        and not forbidden_model_state.intersection(state)
    )


def _claim_or_verify_validation_receipt(
    discovery_root: Path,
    payload: dict[str, Any],
    *,
    resume: bool,
) -> None:
    receipt = discovery_root / ".position_validation_receipt.json"
    if resume:
        if not receipt.is_file():
            raise RuntimeError("validation resume receipt is missing")
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        if recorded != payload:
            raise RuntimeError("validation receipt mismatch or tampering detected")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=discovery_root, prefix=".position_validation_receipt.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, receipt)
        except FileExistsError as exc:
            raise RuntimeError("this discovery contract already has a validation receipt") from exc
        except OSError as exc:
            if exc.errno not in {
                errno.EPERM,
                errno.EACCES,
                errno.EINVAL,
                errno.ENOSYS,
                errno.EOPNOTSUPP,
            }:
                raise
            try:
                receipt_descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as exists:
                raise RuntimeError(
                    "this discovery contract already has a validation receipt"
                ) from exists
            with os.fdopen(receipt_descriptor, "wb") as target:
                target.write(temporary.read_bytes())
                target.flush()
                os.fsync(target.fileno())
    finally:
        temporary.unlink(missing_ok=True)


def _validate_validation_contract(
    root: Path, state: dict[str, Any], discovery_contract: dict[str, Any]
) -> None:
    """Check checkpoint provenance without loading a model or tokenizer."""
    if "frozen_discovery_contract.json" in state.get("artifact_hashes", {}):
        recorded = json.loads((root / "frozen_discovery_contract.json").read_text(encoding="utf-8"))
        if recorded != discovery_contract:
            raise RuntimeError("frozen discovery contract changed during validation")
    if "validation_dataset.jsonl" in state.get("artifact_hashes", {}):
        examples = read_jsonl(root / "validation_dataset.jsonl")
        protocols = {item.metadata.get("validation_protocol_version") for item in examples}
        sources = {item.metadata.get("source_discovery_study_version") for item in examples}
        if protocols != {VALIDATION_PROTOCOL_VERSION} or sources != {STUDY_VERSION}:
            raise RuntimeError("validation dataset belongs to another protocol contract")
        if len(examples) != 360 or any(
            item.split != "validation"
            or item.seed != discovery_contract["seed"]
            or item.metadata.get("pool_identity") != VALIDATION_POOL_ID
            or item.metadata.get("seed_namespace") != VALIDATION_SEED_NAMESPACE
            or item.metadata.get("matched_family_id") != item.family_id
            for item in examples
        ):
            raise RuntimeError("validation dataset checkpoint provenance is inconsistent")
    if "validation_balance_report.json" in state.get("artifact_hashes", {}):
        balance = json.loads((root / "validation_balance_report.json").read_text(encoding="utf-8"))
        if (
            balance.get("validation_protocol_version") != VALIDATION_PROTOCOL_VERSION
            or balance.get("source_discovery_study_version") != STUDY_VERSION
        ):
            raise RuntimeError("validation balance report belongs to another protocol contract")


def _validate_completed_validation(
    root: Path,
    state: dict[str, Any],
    frozen_seed: int,
    contract: dict[str, Any],
    receipt_payload: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "frozen_discovery_contract.json",
        "validation_dataset.jsonl",
        "validation_examples.jsonl",
        "validation_balance_report.json",
        "validation_position_metrics.json",
        "validation_paired_effects.json",
        "validation_secondary_summaries.json",
        "validation_report.md",
        "validation_final_status.json",
    }
    if not required <= set(state.get("artifact_hashes", {})):
        raise RuntimeError("completed validation lacks required hashed artifacts")
    final: dict[str, Any] = json.loads(
        (root / "validation_final_status.json").read_text(encoding="utf-8")
    )
    dataset = read_jsonl(root / "validation_dataset.jsonl")
    records = read_results(root / "validation_examples.jsonl")
    if len(dataset) != 360 or len(records) != 360:
        raise RuntimeError("completed validation raw record counts are incomplete")
    validate_scoring_identity(dataset, records)
    _validate_result_semantics(records)
    balance = json.loads((root / "validation_balance_report.json").read_text(encoding="utf-8"))
    stored_metrics = json.loads(
        (root / "validation_position_metrics.json").read_text(encoding="utf-8")
    )
    stored_effects = json.loads(
        (root / "validation_paired_effects.json").read_text(encoding="utf-8")
    )
    stored_secondary = json.loads(
        (root / "validation_secondary_summaries.json").read_text(encoding="utf-8")
    )
    metrics = position_metrics(records)
    effects = paired_position_effects(records, seed=frozen_seed)
    secondary = secondary_summaries(records)
    if (metrics, effects, secondary) != (stored_metrics, stored_effects, stored_secondary):
        raise RuntimeError("completed validation summaries do not match raw scoring records")
    if not json_has_only_finite_numbers(
        {"metrics": metrics, "effects": effects, "secondary": secondary}
    ):
        raise RuntimeError("completed validation recomputation contains non-finite values")
    recomputed_status = validation_decision(metrics, effects, balance)
    status = final.get("status")
    confirmed = recomputed_status == VALIDATED_STATUS
    complete_families = final.get("complete_matched_family_count")
    receipt_digest = hashlib.sha256(
        json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        status != recomputed_status
        or state.get("complete") is not True
        or state.get("lifecycle_state") != "complete"
        or final.get("software_success") is not True
        or final.get("validation_protocol_version") != VALIDATION_PROTOCOL_VERSION
        or final.get("source_discovery_study_version") != STUDY_VERSION
        or final.get("expected_example_count") != 360
        or final.get("processed_example_count") != 360
        or state.get("expected_example_count") != 360
        or state.get("processed_example_count") != 360
        or state.get("complete_matched_family_count") != effects.get("complete_family_count")
        or state.get("held_out_validation_opened") is not True
        or state.get("held_out_test_opened") is not False
        or state.get("activation_patching_performed") is not False
        or final.get("expected_example_count") != state.get("expected_example_count")
        or final.get("processed_example_count") != state.get("processed_example_count")
        or not isinstance(complete_families, int)
        or not 0 <= complete_families <= 120
        or complete_families != effects.get("complete_family_count")
        or (confirmed and complete_families != 120)
        or final.get("held_out_validation_opened") is not True
        or final.get("held_out_test_opened") is not False
        or final.get("activation_patching_performed") is not False
        or final.get("scientific_validation_confirmed") is not confirmed
        or state.get("claim_id") != receipt_payload["claim_id"]
        or state.get("receipt_digest") != receipt_digest
        or state.get("model") != contract["model"]
        or state.get("model_id") != contract["model_id"]
        or state.get("requested_revision") != contract["requested_revision"]
        or state.get("load_revision") != contract["load_revision"]
        or state.get("frozen_resolved_revision") != contract["resolved_revision"]
        or state.get("tokenizer_id") != contract["tokenizer_id"]
        or state.get("dtype") != contract["dtype"]
        or (
            contract["resolved_revision"] is not None
            and state.get("validation_resolved_revision") != contract["resolved_revision"]
        )
    ):
        raise RuntimeError("completed validation final status is semantically inconsistent")
    return final


def run_position_validation(
    args: argparse.Namespace, adapter_factory: Callable[[str, str, str], Adapter] = PythiaAdapter
) -> dict[str, Any]:
    """Run the explicitly confirmed, one-shot held-out validation protocol."""
    if not args.resume and not args.confirm_open_validation:
        raise RuntimeError("fresh validation requires --confirm-open-validation")
    contract = _verify_frozen_discovery(args.discovery_root)
    seed = int(contract["seed"])
    revision = str(contract["requested_revision"])
    load_revision = str(contract["load_revision"])
    root = args.output / f"seed-{seed}-{revision.replace('/', '_')}"
    components = _validation_fingerprint_components(args, contract)
    fingerprint = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()
    receipt_payload = _validation_receipt_payload(contract, fingerprint, root)

    def initialized_state() -> dict[str, Any]:
        return {
            "schema_version": VALIDATION_MANIFEST_SCHEMA_VERSION,
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "source_discovery_study_version": STUDY_VERSION,
            "claim_id": receipt_payload["claim_id"],
            "lifecycle_state": "initialized",
            "input_fingerprint": fingerprint,
            "fingerprint_components": components,
            "frozen_discovery_artifact_hashes": contract["artifact_hashes"],
            "artifact_hashes": {},
            "complete": False,
            "discovery_commit": contract["discovery_commit"],
            "requested_device": args.device,
            "batch_size": args.batch_size,
            "frozen_seed": seed,
            "output_root": str(root.resolve()),
            "created_at": datetime.now(UTC).isoformat(),
            "held_out_test_opened": False,
            "activation_patching_performed": False,
        }

    receipt_path = args.discovery_root / ".position_validation_receipt.json"
    if args.resume and not root.exists():
        if not receipt_path.is_file():
            raise RuntimeError("--resume requires an existing validation output")
        try:
            _claim_or_verify_validation_receipt(
                args.discovery_root, receipt_payload, resume=True
            )
        except (json.JSONDecodeError, RuntimeError) as exc:
            raise RuntimeError(
                "validation output is absent and its receipt is not the exact selected claim"
            ) from exc
        root.mkdir(parents=True)
        _validation_json(root / "validation_manifest.json", initialized_state())
    if root.exists() and not args.resume:
        # A crash between mkdir and initial-manifest publication leaves no
        # scientific or model state. Only that exactly empty directory is a
        # safe fresh-run recovery boundary.
        if any(root.iterdir()):
            raise RuntimeError(
                "validation output already exists; one-shot runs cannot be recomputed"
            )
        root.rmdir()
    if root.exists():
        manifest_path = root / "validation_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("validation resume manifest is missing")
        state = json.loads(manifest_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != VALIDATION_MANIFEST_SCHEMA_VERSION:
            raise RuntimeError("validation manifest schema version mismatch")
        if state.get("validation_protocol_version") != VALIDATION_PROTOCOL_VERSION:
            raise RuntimeError("validation protocol version mismatch")
        if (
            state.get("source_discovery_study_version") != STUDY_VERSION
            or state.get("claim_id") != receipt_payload["claim_id"]
            or state.get("frozen_discovery_artifact_hashes") != contract["artifact_hashes"]
            or state.get("output_root") != str(root.resolve())
            or state.get("requested_device") != args.device
            or state.get("batch_size") != args.batch_size
            or state.get("frozen_seed") != seed
            or state.get("lifecycle_state")
            not in {
                "initialized",
                "receipt_claimed",
                "model_verified",
                "dataset_written",
                "scoring_written",
                "reports_written",
                "complete",
            }
        ):
            raise RuntimeError("validation manifest provenance is inconsistent")
        if (
            state.get("fingerprint_components") != components
            or state.get("input_fingerprint") != fingerprint
        ):
            raise RuntimeError("validation resume fingerprint mismatch")
        receipt_path = args.discovery_root / ".position_validation_receipt.json"
        if receipt_path.is_file():
            try:
                _claim_or_verify_validation_receipt(
                    args.discovery_root, receipt_payload, resume=True
                )
                if _untouched_initialized_validation(
                    state, receipt_payload, components, contract
                ):
                    state["lifecycle_state"] = "receipt_claimed"
                    state["receipt_digest"] = hashlib.sha256(
                        json.dumps(
                            receipt_payload, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                    _validation_json(manifest_path, state)
            except (json.JSONDecodeError, RuntimeError) as exc:
                # The Windows O_EXCL fallback can be interrupted after the
                # receipt name is reserved but before its payload is durable.
                # Recover only from the untouched initialized lifecycle and a
                # partial payload that still binds this exact claim.
                try:
                    partial = json.loads(receipt_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    partial = None
                recoverable = (
                    _untouched_initialized_validation(
                        state, receipt_payload, components, contract
                    )
                    and (
                        partial is None
                        or (
                            isinstance(partial, dict)
                            and partial.get("claim_id") == receipt_payload["claim_id"]
                            and partial.get("selected_output_root")
                            == receipt_payload["selected_output_root"]
                            and partial.get("validation_fingerprint")
                            == receipt_payload["validation_fingerprint"]
                        )
                    )
                )
                if not recoverable:
                    raise RuntimeError(
                        "validation receipt mismatch or tampering detected"
                    ) from exc
                _validation_json(receipt_path, receipt_payload)
                _claim_or_verify_validation_receipt(
                    args.discovery_root, receipt_payload, resume=True
                )
                state["lifecycle_state"] = "receipt_claimed"
                state["receipt_digest"] = hashlib.sha256(
                    json.dumps(
                        receipt_payload, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                _validation_json(manifest_path, state)
        elif (
            args.resume
            and _untouched_initialized_validation(
                state, receipt_payload, components, contract
            )
        ):
            _claim_or_verify_validation_receipt(args.discovery_root, receipt_payload, resume=False)
            state["lifecycle_state"] = "receipt_claimed"
            state["receipt_digest"] = hashlib.sha256(
                json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            _validation_json(manifest_path, state)
        else:
            raise RuntimeError("validation receipt is missing and manifest is not recoverable")
        _validate_validation_hash_paths(root, state)
        _validate_checkpoints(root, state)
        _validate_validation_contract(root, state, contract)
        if state.get("complete") is True:
            return _validate_completed_validation(
                root, state, seed, contract, receipt_payload
            )
    else:
        state = initialized_state()
        root.mkdir(parents=True)
        initial_manifest = root / "validation_manifest.json"
        _validation_json(initial_manifest, state)
        receipt_path = args.discovery_root / ".position_validation_receipt.json"
        receipt_preexisted = receipt_path.exists()
        try:
            _claim_or_verify_validation_receipt(args.discovery_root, receipt_payload, resume=False)
        except Exception:
            # If publication reserved/wrote a receipt before interruption,
            # retain the exact initialized state so the Windows fallback can
            # be verified and recovered by --resume. With no receipt, this is
            # an untouched failed claim and can be removed safely.
            if receipt_preexisted or not receipt_path.exists():
                initial_manifest.unlink()
                root.rmdir()
            raise
        state["lifecycle_state"] = "receipt_claimed"
        state["receipt_digest"] = hashlib.sha256(
            json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _validation_json(initial_manifest, state)
    contract_path = root / "frozen_discovery_contract.json"
    if "frozen_discovery_contract.json" not in state["artifact_hashes"]:
        _validation_json(contract_path, contract)
        _validation_checkpoint(root, state, contract_path)
    adapter = adapter_factory(str(contract["model"]), load_revision, args.device)
    model_identity = _verify_validation_adapter(adapter, contract)
    state |= model_identity | {"lifecycle_state": "model_verified"}
    _validation_json(root / "validation_manifest.json", state)
    dataset_path = root / "validation_dataset.jsonl"
    if "validation_dataset.jsonl" in state["artifact_hashes"]:
        examples = read_jsonl(dataset_path)
    else:
        examples = generate_validation_dataset(adapter.tokenizer, seed)
        write_dataset(dataset_path, examples)
        _validation_checkpoint(root, state, dataset_path)
        state["lifecycle_state"] = "dataset_written"
        _validation_json(root / "validation_manifest.json", state)
    balance = validate_validation_dataset(examples, adapter.tokenizer, seed)
    balance_path = root / "validation_balance_report.json"
    _validation_json(balance_path, balance)
    _validation_checkpoint(root, state, balance_path)
    records_path = root / "validation_examples.jsonl"
    if "validation_examples.jsonl" in state["artifact_hashes"]:
        records = read_results(records_path)
    else:
        records = adapter.score(examples, args.batch_size)
        validate_scoring_identity(examples, records)
        write_example_results(records_path, records)
        _validation_checkpoint(root, state, records_path)
        state["lifecycle_state"] = "scoring_written"
        _validation_json(root / "validation_manifest.json", state)
    validate_scoring_identity(examples, records)
    _validate_result_semantics(records)
    metrics = position_metrics(records)
    effects = paired_position_effects(records, seed=seed)
    secondary = secondary_summaries(records)
    outputs = {
        "validation_position_metrics.json": metrics,
        "validation_paired_effects.json": effects,
        "validation_secondary_summaries.json": secondary,
    }
    for name, value in outputs.items():
        path = root / name
        _validation_json(path, value)
        _validation_checkpoint(root, state, path)
    status = validation_decision(metrics, effects, balance)
    final = {
        "status": status,
        "software_success": True,
        "held_out_validation_opened": True,
        "held_out_test_opened": False,
        "activation_patching_performed": False,
        "scientific_validation_confirmed": status == VALIDATED_STATUS,
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "source_discovery_study_version": STUDY_VERSION,
        "expected_example_count": 360,
        "processed_example_count": len(records),
        "complete_matched_family_count": effects["complete_family_count"],
    }
    report = [
        "# One-shot matched-position held-out validation",
        "",
        "## Frozen discovery evidence",
        f"- Discovery version: `{STUDY_VERSION}`",
        f"- Discovery commit: `{contract['discovery_commit']}`",
        f"- Model ID: `{contract['model_id']}`",
        f"- Requested revision: `{contract['requested_revision']}`",
        f"- Frozen resolved revision: `{contract['resolved_revision'] or 'unavailable'}`",
        f"- Seed: `{seed}`",
        "- Artifact provenance: all hashes in `frozen_discovery_contract.json` verified.",
        "",
        "## Primary held-out validation result",
        f"**Status:** `{status}`",
        "The primary confirmatory claim concerns the frozen first-versus-last candidate.",
        "",
        "| Position | Clean acc. | Corrupt acc. | Clean LD | Corrupt LD | Contrast | "
        "Joint | N | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for position in ("first", "interior", "last"):
        row = metrics[position]
        if row["example_count"]:
            report.append(
                f"| {position} | {row['clean_pairwise_accuracy']:.6f} | "
                f"{row['corrupt_pairwise_accuracy']:.6f} | "
                f"{row['mean_clean_logit_difference']:.6f} | "
                f"{row['mean_corrupt_logit_difference']:.6f} | "
                f"{row['clean_corrupt_contrast']:.6f} | "
                f"{row['joint_success_rate']:.6f} | {row['example_count']} | "
                f"{row['failed_count']} |"
            )
        else:
            report.append(
                f"| {position} | n/a | n/a | n/a | n/a | n/a | n/a | "
                f"{row['example_count']} | {row['failed_count']} |"
            )
    comparisons = effects["comparisons"]

    def effect_line(key: str, label: str) -> str:
        effect = comparisons[key]
        interval = effect["ci_95"]
        if effect["estimate"] is None or interval is None:
            return f"- {label}: unavailable"
        return (
            f"- {label}: estimate {effect['estimate']:.6f}; "
            f"95% matched-family bootstrap CI [{interval[0]:.6f}, {interval[1]:.6f}]"
        )

    report += [
        "",
        "### Primary first-versus-last effect",
        effect_line(
            "first_minus_last_clean_logit_difference", "First minus last clean logit difference"
        ),
        effect_line("first_minus_last_contrast", "First minus last contrast (supporting)"),
        "",
        "## Supporting validation results",
        "Interior-versus-last is a supporting replication.",
        effect_line(
            "interior_minus_last_clean_logit_difference",
            "Interior minus last clean logit difference",
        ),
        effect_line("interior_minus_last_contrast", "Interior minus last contrast"),
        "",
        "### Descriptive first-versus-interior result",
        "No first-specific superiority is assumed.",
        effect_line(
            "first_minus_interior_clean_logit_difference",
            "First minus interior clean logit difference",
        ),
        effect_line("first_minus_interior_contrast", "First minus interior contrast"),
        "Correctness transitions and lexical/entity summaries are supporting only and are "
        "recorded in the paired-effects and secondary-summary JSON artifacts.",
        "",
        "## Software and scientific status",
        "- Software success: **yes**.",
        f"- Scientific validation confirmed: **{'yes' if status == VALIDATED_STATUS else 'no'}**.",
        "- Test split opened: **no**.",
        "- Activation patching performed: **no**.",
        "- Exact model revision match: **{}**.".format(
            "yes" if model_identity["exact_revision_matching_succeeded"] else "unavailable"
        ),
    ]
    report_path = root / "validation_report.md"
    report_path.write_bytes(("\n".join(report) + "\n").encode())
    _validation_checkpoint(root, state, report_path)
    final_path = root / "validation_final_status.json"
    _validation_json(final_path, final)
    _validation_checkpoint(root, state, final_path)
    state["lifecycle_state"] = "reports_written"
    _validation_json(root / "validation_manifest.json", state)
    if not json_has_only_finite_numbers(
        {"metrics": metrics, "effects": effects, "secondary": secondary}
    ):
        raise RuntimeError("non-finite validation result")
    state |= {
        "complete": True,
        "lifecycle_state": "complete",
        "expected_example_count": 360,
        "processed_example_count": len(records),
        "model_id": adapter.model_id,
        "revision": revision,
        "complete_matched_family_count": effects["complete_family_count"],
        "completed_at": datetime.now(UTC).isoformat(),
        "held_out_validation_opened": True,
        "held_out_test_opened": False,
        "activation_patching_performed": False,
    }
    _validation_json(root / "validation_manifest.json", state)
    return final


def _validate_position_contract(root: Path, state: dict[str, Any]) -> None:
    """Reject resumable artifacts that belong to another study contract."""
    if state.get("study_version") != STUDY_VERSION:
        raise RuntimeError("resume study version does not match the current study contract")
    if "matched_dataset.jsonl" in state.get("artifact_hashes", {}):
        dataset = read_jsonl(root / "matched_dataset.jsonl")
        versions = {item.metadata.get("generator_version") for item in dataset}
        if versions != {STUDY_VERSION}:
            raise RuntimeError("resumed dataset belongs to another generator version")
    if "balance_report.json" in state.get("artifact_hashes", {}):
        balance = json.loads((root / "balance_report.json").read_text(encoding="utf-8"))
        if balance.get("generator_version") != STUDY_VERSION:
            raise RuntimeError("resumed balance report belongs to another generator version")


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
        if manifest.get("study_version") != STUDY_VERSION:
            raise RuntimeError("resume study version does not match the current study contract")
        if manifest.get("input_fingerprint_components") != _position_fingerprint_components(args):
            raise RuntimeError("resume fingerprint components do not match the study contract")
        if manifest.get("input_fingerprint") != fingerprint:
            raise RuntimeError("resume inputs do not match the recorded command/model")
        _validate_checkpoints(root, manifest)
        _validate_position_contract(root, manifest)
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
            "study_version": STUDY_VERSION,
            "input_fingerprint": fingerprint,
            "input_fingerprint_components": _position_fingerprint_components(args),
            "artifact_hashes": {},
            "complete": False,
        }
    if args.force and root.exists():
        import shutil

        shutil.rmtree(root)
        state = {
            "schema_version": 1,
            "study_version": STUDY_VERSION,
            "input_fingerprint": fingerprint,
            "input_fingerprint_components": _position_fingerprint_components(args),
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
        validate_scoring_identity(examples, records)
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
                f"| {position} | n/a | n/a | n/a | n/a | n/a | n/a | 0 | {row['failed_count']} |"
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
    validation = commands.add_parser(
        "position-validation", help="run one-shot held-out matched-position validation"
    )
    validation.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    validation.add_argument("--batch-size", type=int, default=8)
    validation.add_argument(
        "--discovery-root", type=Path, default=Path("artifacts/position_study/seed-42-main")
    )
    validation.add_argument("--output", type=Path, default=Path("artifacts/position_validation"))
    validation.add_argument("--confirm-open-validation", action="store_true")
    validation.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or getattr(args, "seed", 0) < 0:
        parser.error("batch size must be positive and seed non-negative")
    try:
        if args.command == "position-validation":
            run_position_validation(args)
        elif args.command == "position-study":
            run_position_study(args)
        else:
            run_discovery(args)
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
