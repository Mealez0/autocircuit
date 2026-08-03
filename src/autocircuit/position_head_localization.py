"""Discovery-only attention-head localization in a provenance-selected block."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autocircuit.datasets.associative_recall import ExamplePair, read_jsonl
from autocircuit.pipeline import (
    PythiaAdapter,
    _git_commit,
    _verify_frozen_discovery,
    _verify_validation_adapter,
    sha256,
    verify_resume,
)
from autocircuit.position_component_localization import (
    INTERVENTIONS as COMPONENT_INTERVENTIONS,
)
from autocircuit.position_component_localization import (
    PROTOCOL_VERSION as COMPONENT_PROTOCOL_VERSION,
)
from autocircuit.position_component_localization import STATUS as COMPONENT_STATUS
from autocircuit.position_component_localization import ComponentRecord
from autocircuit.position_component_localization import summarize as summarize_components
from autocircuit.position_localization import (
    DIRECTIONS,
    LOCALIZATION_PROTOCOL_VERSION,
    LOCALIZATION_STATUS,
    PROMPT_VARIANTS,
    SOURCE_MODES,
    FirstLastPair,
    _prompt,
    _task_scores,
    build_first_last_pairs,
)

PROTOCOL_VERSION = "position-head-localization-0.1.0"
STATUS = "EXPLORATORY_ATTENTION_HEAD_LOCALIZATION_COMPLETE"
REFERENCE_INTERVENTIONS = ("aggregate_attention_output", "identity_noop")
BOOTSTRAP_SAMPLES = 10_000
MODEL_IDENTITY_FIELDS = (
    "model",
    "model_id",
    "tokenizer_id",
    "requested_revision",
    "load_revision",
    "frozen_resolved_revision",
    "dtype",
    "exact_revision_matching_succeeded",
    "validation_resolved_revision",
)
FACTORIAL_CELLS = {(variant, direction) for variant in PROMPT_VARIANTS for direction in DIRECTIONS}
REQUIRED_LAYER_ARTIFACTS = {
    "layer_scan_records.jsonl",
    "layer_scan_summary.json",
    "localization_final_status.json",
}
REQUIRED_COMPONENT_ARTIFACTS = {
    "component_records.jsonl",
    "component_summary.json",
    "component_final_status.json",
}
STRICT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class HeadRecord:
    family_id: str
    prompt_variant: str
    direction: str
    source_mode: str
    selected_layer: int
    intervention: str
    head_index: int | None
    hook_site: str
    source_family_id: str
    destination_family_id: str
    source_score: float
    destination_baseline: float
    patched_score: float
    oriented_causal_transfer: float
    layer_run_manifest_hash: str
    component_run_manifest_hash: str


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"malformed {label}")
    return value


def _verify_hash_entries(root: Path, entries: Any, label: str) -> None:
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError(f"{label} has no artifact hashes")
    resolved_root = root.resolve()
    for relative, digest in entries.items():
        path = root / str(relative)
        if path.resolve().parent != resolved_root or not isinstance(digest, str):
            raise RuntimeError(f"unsafe {label} artifact hash entry")
        verify_resume(path, digest)


def _require_hashed_artifacts(
    root: Path, entries: Any, required: set[str], label: str
) -> dict[str, str]:
    if not isinstance(entries, dict) or not required <= set(entries):
        missing = sorted(required - set(entries if isinstance(entries, dict) else {}))
        raise RuntimeError(f"{label} lacks required hashed artifacts: {missing}")
    _verify_hash_entries(root, entries, label)
    return {name: str(entries[name]) for name in sorted(required)}


def _read_jsonl_dicts(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"malformed {label} row {line_number}")
        rows.append(value)
    if not rows:
        raise RuntimeError(f"{label} is empty")
    return rows


def _expected_identity(discovery: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "model": discovery.get("model"),
        "model_id": discovery.get("model_id"),
        "tokenizer_id": discovery.get("tokenizer_id"),
        "requested_revision": discovery.get("requested_revision"),
        "load_revision": discovery.get("load_revision"),
        "frozen_resolved_revision": discovery.get("resolved_revision"),
        "dtype": discovery.get("dtype"),
        "exact_revision_matching_succeeded": discovery.get("exact_revision_available"),
        "validation_resolved_revision": discovery.get("resolved_revision"),
    }
    nullable = {"frozen_resolved_revision", "validation_resolved_revision"}
    if any(value is None for key, value in mapping.items() if key not in nullable):
        raise RuntimeError("frozen discovery model identity is incomplete")
    return mapping


def _verify_manifest_identity(
    manifest: dict[str, Any], discovery: dict[str, Any], label: str
) -> dict[str, Any]:
    expected = _expected_identity(discovery)
    missing = [key for key in MODEL_IDENTITY_FIELDS if key not in manifest]
    if missing:
        raise RuntimeError(f"{label} model identity fields are missing: {missing}")
    if any(manifest[key] != expected[key] for key in MODEL_IDENTITY_FIELDS):
        raise RuntimeError(f"{label} and discovery model identity disagree")
    return {key: manifest[key] for key in MODEL_IDENTITY_FIELDS}


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"{label} is not a finite number")
    return float(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=STRICT_TOLERANCE, abs_tol=STRICT_TOLERANCE)


def verify_layer_run(root: Path, discovery: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / "localization_manifest.json"
    summary_path = root / "layer_scan_summary.json"
    final_path = root / "localization_final_status.json"
    records_path = root / "layer_scan_records.jsonl"
    manifest = _load_json(manifest_path, "layer manifest")
    summary = _load_json(summary_path, "layer summary")
    final = _load_json(final_path, "layer final status")
    _require_hashed_artifacts(
        root, manifest.get("artifact_hashes"), REQUIRED_LAYER_ARTIFACTS, "layer run"
    )
    identity = _verify_manifest_identity(manifest, discovery, "layer run")
    if not (
        manifest.get("complete") is True
        and manifest.get("protocol_version") == LOCALIZATION_PROTOCOL_VERSION
        and manifest.get("discovery_artifact_hashes") == discovery["artifact_hashes"]
        and manifest.get("held_out_validation_reused") is False
        and manifest.get("held_out_test_opened") is False
        and final.get("status") == LOCALIZATION_STATUS
        and final.get("software_success") is True
        and final.get("activation_patching_performed") is True
        and final.get("held_out_validation_reused") is False
        and final.get("held_out_test_opened") is False
        and final.get("circuit_found") is False
        and summary.get("protocol_version") == LOCALIZATION_PROTOCOL_VERSION
    ):
        raise RuntimeError("layer localization run is not eligible")
    rows = _read_jsonl_dicts(records_path, "layer records")
    families = {str(row.get("family_id")) for row in rows}
    sites = {str(row.get("site")) for row in rows}
    if not families or "None" in families or not sites or "None" in sites:
        raise RuntimeError("layer records have malformed families or sites")
    cells: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        variant, direction, mode = (
            row.get("prompt_variant"),
            row.get("direction"),
            row.get("source_mode"),
        )
        site, family = str(row.get("site")), str(row.get("family_id"))
        if (
            variant not in PROMPT_VARIANTS
            or direction not in DIRECTIONS
            or mode not in SOURCE_MODES
        ):
            raise RuntimeError("layer records contain an invalid factorial label")
        key = (site, family, str(mode))
        cell = (str(variant), str(direction))
        if cell in cells[key]:
            raise RuntimeError("layer records contain duplicate factorial cells")
        cells[key].add(cell)
        values[(site, str(mode))].append(
            _finite_number(row.get("causal_transfer"), "layer causal transfer")
        )
    expected_keys = {
        (site, family, mode) for site in sites for family in families for mode in SOURCE_MODES
    }
    if set(cells) != expected_keys or any(
        cell_set != FACTORIAL_CELLS for cell_set in cells.values()
    ):
        raise RuntimeError("layer records are factorially incomplete")
    expected_count = len(sites) * len(families) * len(SOURCE_MODES) * len(FACTORIAL_CELLS)
    if len(rows) != expected_count:
        raise RuntimeError("layer record count is inconsistent")
    if not (
        manifest.get("family_count") == len(families)
        and final.get("matched_family_count") == len(families)
        and manifest.get("record_count") == len(rows)
        and final.get("record_count") == len(rows)
        and summary.get("record_count") == len(rows)
    ):
        raise RuntimeError("layer family or record counts are inconsistent")
    ranking = summary.get("site_ranking")
    if not isinstance(ranking, list) or len(ranking) != len(sites):
        raise RuntimeError("layer ranking is malformed")
    recomputed: list[dict[str, Any]] = []
    for site in sites:
        matched = _mean(values[(site, "matched")])
        permuted = _mean(values[(site, "permuted")])
        recomputed.append(
            {
                "site": site,
                "matched": matched,
                "permuted": permuted,
                "advantage": matched - permuted,
            }
        )
    recomputed.sort(key=lambda row: (-row["advantage"], -row["matched"], row["site"]))
    for index, (stored, computed) in enumerate(zip(ranking, recomputed, strict=True), 1):
        if not isinstance(stored, dict) or stored.get("site") != computed["site"]:
            raise RuntimeError("layer ranking order is inconsistent with raw records")
        if stored.get("rank") != index:
            raise RuntimeError("layer ranking ranks are inconsistent")
        for stored_key, computed_key in (
            ("matched_mean_causal_transfer", "matched"),
            ("permuted_mean_causal_transfer", "permuted"),
            ("family_specific_transfer_advantage", "advantage"),
        ):
            if not _close(
                _finite_number(stored.get(stored_key), f"layer ranking {stored_key}"),
                float(computed[computed_key]),
            ):
                raise RuntimeError("layer ranking statistics disagree with raw records")
    top = ranking[0]
    site = top.get("site")
    if not isinstance(site, str):
        raise RuntimeError("top layer site is malformed")
    import re

    match = re.fullmatch(r"blocks\.(\d+)\.hook_resid_(?:pre|post)", site)
    if match is None or float(top["family_specific_transfer_advantage"]) <= 0:
        raise RuntimeError("top layer boundary is ineligible")
    return {
        "layer": int(match.group(1)),
        "top_boundary": top,
        "identity": identity,
        "hashes": {
            "manifest": sha256(manifest_path),
            "summary": sha256(summary_path),
            "final": sha256(final_path),
            "records": sha256(records_path),
        },
    }


def _component_attention_row(summary: dict[str, Any]) -> dict[str, float]:
    ranking = summary.get("intervention_ranking")
    if not isinstance(ranking, list) or not ranking:
        raise RuntimeError("component ranking is missing")
    matches = [
        row
        for row in ranking
        if isinstance(row, dict) and row.get("intervention") == "attention_output"
    ]
    if len(matches) != 1:
        raise RuntimeError("component attention ranking is malformed")
    row = matches[0]
    result: dict[str, float] = {}
    for key in (
        "matched_mean_transfer",
        "permuted_mean_transfer",
        "family_specific_advantage",
    ):
        value = row.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"component attention {key} is malformed")
        result[key] = float(value)
    return result


def verify_component_run(
    root: Path,
    discovery: dict[str, Any],
    layer_run: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = root / "component_manifest.json"
    summary_path = root / "component_summary.json"
    final_path = root / "component_final_status.json"
    manifest = _load_json(manifest_path, "component manifest")
    summary = _load_json(summary_path, "component summary")
    final = _load_json(final_path, "component final status")
    records_path = root / "component_records.jsonl"
    _require_hashed_artifacts(
        root,
        manifest.get("artifact_hashes"),
        REQUIRED_COMPONENT_ARTIFACTS,
        "component run",
    )
    identity = _verify_manifest_identity(manifest, discovery, "component run")
    recorded_layer_hashes = {
        key: layer_run["hashes"][key] for key in ("manifest", "summary", "final")
    }
    if not (
        manifest.get("complete") is True
        and manifest.get("protocol_version") == COMPONENT_PROTOCOL_VERSION
        and manifest.get("discovery_hashes") == discovery["artifact_hashes"]
        and manifest.get("layer_run_hashes") == recorded_layer_hashes
        and manifest.get("selected_layer") == layer_run["layer"]
        and manifest.get("held_out_validation_reused") is False
        and manifest.get("held_out_test_opened") is False
    ):
        raise RuntimeError("component localization manifest is not eligible")
    if identity != layer_run.get("identity"):
        raise RuntimeError("layer and component model identity disagree")
    if not (
        final.get("status") == COMPONENT_STATUS
        and final.get("software_success") is True
        and final.get("activation_patching_performed") is True
        and final.get("circuit_found") is False
        and final.get("held_out_validation_reused") is False
        and final.get("held_out_test_opened") is False
        and final.get("selected_layer") == layer_run["layer"]
        and summary.get("protocol_version") == COMPONENT_PROTOCOL_VERSION
        and summary.get("selected_layer") == layer_run["layer"]
    ):
        raise RuntimeError("component localization status is inconsistent")
    raw_rows = _read_jsonl_dicts(records_path, "component records")
    records: list[ComponentRecord] = []
    cells: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    families: set[str] = set()
    for row in raw_rows:
        intervention = row.get("intervention")
        family = row.get("family_id")
        variant = row.get("prompt_variant")
        direction = row.get("direction")
        mode = row.get("source_mode")
        if intervention not in COMPONENT_INTERVENTIONS or not isinstance(family, str):
            raise RuntimeError("component records contain an invalid intervention or family")
        if (
            variant not in PROMPT_VARIANTS
            or direction not in DIRECTIONS
            or mode not in SOURCE_MODES
        ):
            raise RuntimeError("component records contain an invalid factorial label")
        if row.get("layer") != layer_run["layer"]:
            raise RuntimeError("component records disagree on selected layer")
        key = (str(intervention), family, str(mode))
        cell = (str(variant), str(direction))
        if cell in cells[key]:
            raise RuntimeError("component records contain duplicate factorial cells")
        cells[key].add(cell)
        families.add(family)
        try:
            records.append(ComponentRecord(**row))
        except TypeError as exc:
            raise RuntimeError("component record schema is malformed") from exc
    expected_keys = {
        (intervention, family, mode)
        for intervention in COMPONENT_INTERVENTIONS
        for family in families
        for mode in SOURCE_MODES
    }
    if set(cells) != expected_keys or any(
        cell_set != FACTORIAL_CELLS for cell_set in cells.values()
    ):
        raise RuntimeError("component records are factorially incomplete")
    expected_count = (
        len(families) * len(COMPONENT_INTERVENTIONS) * len(SOURCE_MODES) * len(FACTORIAL_CELLS)
    )
    if len(records) != expected_count or len(families) != 120:
        raise RuntimeError("component family or record count is incomplete")
    if not (
        summary.get("record_count") == len(records)
        and final.get("record_count") == len(records)
        and final.get("matched_family_count") == len(families)
    ):
        raise RuntimeError("component recorded counts are inconsistent")
    recomputed = summarize_components(records, seed=int(discovery["seed"]))
    stored_ranking = summary.get("intervention_ranking")
    recomputed_ranking = recomputed["intervention_ranking"]
    if not isinstance(stored_ranking, list) or len(stored_ranking) != len(COMPONENT_INTERVENTIONS):
        raise RuntimeError("component ranking is malformed")
    if {row.get("intervention") for row in stored_ranking if isinstance(row, dict)} != set(
        COMPONENT_INTERVENTIONS
    ):
        raise RuntimeError("component ranking interventions are incomplete or duplicated")
    for stored, computed in zip(stored_ranking, recomputed_ranking, strict=True):
        if not isinstance(stored, dict) or stored.get("intervention") != computed["intervention"]:
            raise RuntimeError("component ranking order disagrees with raw records")
        if stored.get("family_count") != len(families):
            raise RuntimeError("component ranking family count is inconsistent")
        matched = _finite_number(stored.get("matched_mean_transfer"), "component matched mean")
        permuted = _finite_number(stored.get("permuted_mean_transfer"), "component permuted mean")
        advantage = _finite_number(
            stored.get("family_specific_advantage"), "component family advantage"
        )
        if not _close(advantage, matched - permuted):
            raise RuntimeError("component advantage is not matched minus permuted")
        for metric_key in (
            "matched_mean_transfer",
            "permuted_mean_transfer",
            "family_specific_advantage",
        ):
            if not _close(float(stored[metric_key]), float(computed[metric_key])):
                raise RuntimeError("component summary disagrees with raw records")
        stored_ci = stored.get("family_specific_advantage_ci_95")
        computed_ci = computed.get("family_specific_advantage_ci_95")
        if (
            not isinstance(stored_ci, list)
            or not isinstance(computed_ci, list)
            or len(stored_ci) != 2
            or any(
                not _close(_finite_number(left, "component bootstrap interval"), float(right))
                for left, right in zip(stored_ci, computed_ci, strict=True)
            )
        ):
            raise RuntimeError("component bootstrap interval disagrees with raw records")
    if summary.get("leading_atomic_component") != recomputed.get("leading_atomic_component"):
        raise RuntimeError("component leading atomic component is inconsistent")
    if not isinstance(manifest.get("parallel_attn_mlp"), bool):
        raise RuntimeError("component architecture identity is missing")
    return {
        "layer": int(layer_run["layer"]),
        "attention": _component_attention_row(recomputed),
        "hashes": {
            "manifest": sha256(manifest_path),
            "summary": sha256(summary_path),
            "final": sha256(final_path),
            "records": sha256(records_path),
        },
        "identity": identity,
        "parallel_attn_mlp": manifest.get("parallel_attn_mlp"),
    }


def _patch_head_query(source: Any, head_index: int) -> Callable[..., Any]:
    if head_index < 0:
        raise ValueError("head index must be non-negative")

    def callback(value: Any, hook: Any = None) -> Any:
        del hook
        if getattr(value, "ndim", None) != 4 or getattr(source, "ndim", None) != 4:
            raise RuntimeError("hook_z activation must have shape [batch, position, head, d_head]")
        if value.shape != source.shape:
            raise RuntimeError("source and destination hook_z shapes differ")
        if head_index >= value.shape[2]:
            raise ValueError("head index is outside hook_z head dimension")
        patched = value.clone()
        patched[:, -1, head_index, :] = source[:, -1, head_index, :]
        return patched

    return callback


def _patch_final_position(source: Any) -> Callable[..., Any]:
    def callback(value: Any, hook: Any = None) -> Any:
        del hook
        if getattr(value, "ndim", None) != 3 or getattr(source, "ndim", None) != 3:
            raise RuntimeError("attention output must have shape [batch, position, d_model]")
        if value.shape != source.shape:
            raise RuntimeError("source and destination attention output shapes differ")
        patched = value.clone()
        patched[:, -1, :] = source[:, -1, :]
        return patched

    return callback


def _identity_final_position() -> Callable[..., Any]:
    def callback(value: Any, hook: Any = None) -> Any:
        del hook
        patched = value.clone()
        patched[:, -1, :] = value[:, -1, :]
        if not patched.equal(value):
            raise RuntimeError("identity no-op changed the activation")
        return patched

    return callback


def _mode_value(value: Any, mode: str) -> Any:
    import torch

    if mode == "matched":
        return value
    if mode == "permuted":
        return torch.roll(value, shifts=1, dims=0)
    raise ValueError(f"unknown source mode: {mode}")


def _mode_families(items: list[ExamplePair], mode: str) -> list[str]:
    ids = [item.family_id for item in items]
    return ids if mode == "matched" else [ids[-1], *ids[:-1]]


def deterministic_batches(pairs: list[FirstLastPair], batch_size: int) -> list[list[FirstLastPair]]:
    """Group compatible families without ever creating a singleton batch."""
    if batch_size < 2:
        raise ValueError("batch size must be at least 2 for family permutation")
    buckets: dict[int, list[FirstLastPair]] = defaultdict(list)
    for pair in pairs:
        length = pair.first.metadata.get("prompt_token_length")
        if not isinstance(length, int) or isinstance(length, bool):
            raise ValueError("pair has no valid token length")
        buckets[length].append(pair)
    batches: list[list[FirstLastPair]] = []
    for length in sorted(buckets):
        bucket = buckets[length]
        if len(bucket) == 1:
            raise ValueError(
                f"token-length bucket {length} has one family; "
                "within-bucket permutation is impossible"
            )
        start = 0
        while start < len(bucket):
            remaining = len(bucket) - start
            take = min(batch_size, remaining)
            if remaining > batch_size and remaining - batch_size == 1:
                take = batch_size - 1
            batch = bucket[start : start + take]
            if len(batch) < 2:
                raise RuntimeError("deterministic batching produced a singleton")
            batches.append(batch)
            start += take
    return batches


def scan_heads(
    model: Any,
    pairs: list[FirstLastPair],
    batch_size: int,
    layer: int,
    layer_manifest_hash: str,
    component_manifest_hash: str,
) -> list[HeadRecord]:
    import torch

    if not layer_manifest_hash or not component_manifest_hash:
        raise ValueError("upstream manifest hashes are required")
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    if layer < 0 or layer >= n_layers:
        raise ValueError("selected layer is outside the model")
    if n_heads <= 0:
        raise ValueError("model must expose a positive head count")
    z_site = f"blocks.{layer}.attn.hook_z"
    attn_site = f"blocks.{layer}.hook_attn_out"
    batches = deterministic_batches(pairs, batch_size)
    records: list[HeadRecord] = []
    with torch.inference_mode():
        for chunk in batches:
            for variant in PROMPT_VARIANTS:
                for direction in DIRECTIONS:
                    source_items = [
                        pair.first if direction == "first_to_last" else pair.last for pair in chunk
                    ]
                    destination_items = [
                        pair.last if direction == "first_to_last" else pair.first for pair in chunk
                    ]
                    orientation = 1.0 if direction == "first_to_last" else -1.0
                    source_prompts = [_prompt(item, variant) for item in source_items]
                    destination_prompts = [_prompt(item, variant) for item in destination_items]
                    source_logits, source_cache = model.run_with_cache(
                        source_prompts,
                        return_type="logits",
                        names_filter=[z_site, attn_site],
                    )
                    destination_logits, _ = model.run_with_cache(
                        destination_prompts,
                        return_type="logits",
                        names_filter=[attn_site],
                    )
                    source_scores = _task_scores(source_logits, source_items, variant)
                    destination_scores = _task_scores(
                        destination_logits, destination_items, variant
                    )
                    for mode in SOURCE_MODES:
                        mode_scores = _mode_value(source_scores, mode)
                        family_ids = _mode_families(source_items, mode)
                        interventions: list[tuple[str, int | None, str, Callable[..., Any]]] = [
                            (
                                f"head_{head}",
                                head,
                                z_site,
                                _patch_head_query(_mode_value(source_cache[z_site], mode), head),
                            )
                            for head in range(n_heads)
                        ]
                        interventions.extend(
                            [
                                (
                                    "aggregate_attention_output",
                                    None,
                                    attn_site,
                                    _patch_final_position(
                                        _mode_value(source_cache[attn_site], mode)
                                    ),
                                ),
                                ("identity_noop", None, attn_site, _identity_final_position()),
                            ]
                        )
                        for name, head, site, callback in interventions:
                            patched_logits = model.run_with_hooks(
                                destination_prompts,
                                return_type="logits",
                                fwd_hooks=[(site, callback)],
                            )
                            patched_scores = _task_scores(
                                patched_logits, destination_items, variant
                            )
                            if name == "identity_noop" and not torch.equal(
                                patched_scores, destination_scores
                            ):
                                raise RuntimeError("identity no-op score was not exactly unchanged")
                            for row, destination in enumerate(destination_items):
                                source_score = float(mode_scores[row])
                                baseline = float(destination_scores[row])
                                patched_score = float(patched_scores[row])
                                transfer = (
                                    0.0
                                    if name == "identity_noop"
                                    else orientation * (patched_score - baseline)
                                )
                                if not all(
                                    math.isfinite(value)
                                    for value in (
                                        source_score,
                                        baseline,
                                        patched_score,
                                        transfer,
                                    )
                                ):
                                    raise RuntimeError("non-finite head-localization record")
                                records.append(
                                    HeadRecord(
                                        family_id=destination.family_id,
                                        prompt_variant=variant,
                                        direction=direction,
                                        source_mode=mode,
                                        selected_layer=layer,
                                        intervention=name,
                                        head_index=head,
                                        hook_site=site,
                                        source_family_id=family_ids[row],
                                        destination_family_id=destination.family_id,
                                        source_score=source_score,
                                        destination_baseline=baseline,
                                        patched_score=patched_score,
                                        oriented_causal_transfer=transfer,
                                        layer_run_manifest_hash=layer_manifest_hash,
                                        component_run_manifest_hash=component_manifest_hash,
                                    )
                                )
    return records


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _bootstrap(values: list[float], seed: int, samples: int) -> list[float]:
    import torch

    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indexes = torch.randint(len(values), (samples, len(values)), generator=generator)
    means = tensor[indexes].mean(dim=1)
    bounds = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=tensor.dtype))
    return [float(bounds[0]), float(bounds[1])]


def validate_head_records(
    records: list[HeadRecord], n_heads: int, expected_families: set[str] | None = None
) -> list[str]:
    if not records:
        raise ValueError("head records are empty")
    expected_interventions = {f"head_{head}" for head in range(n_heads)} | set(
        REFERENCE_INTERVENTIONS
    )
    layers = {record.selected_layer for record in records}
    if len(layers) != 1:
        raise RuntimeError("head records mix selected layers")
    families = {record.family_id for record in records}
    if expected_families is not None and families != expected_families:
        raise RuntimeError("head records contain missing or unexpected families")
    cells: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        if record.intervention not in expected_interventions:
            raise RuntimeError("head records contain an unexpected intervention")
        if record.prompt_variant not in PROMPT_VARIANTS:
            raise RuntimeError("head records contain an invalid prompt variant")
        if record.direction not in DIRECTIONS:
            raise RuntimeError("head records contain an invalid direction")
        if record.source_mode not in SOURCE_MODES:
            raise RuntimeError("head records contain an invalid source mode")
        expected_head = (
            int(record.intervention.removeprefix("head_"))
            if record.intervention.startswith("head_")
            else None
        )
        if record.head_index != expected_head:
            raise RuntimeError("head record index is inconsistent with its intervention")
        if record.destination_family_id != record.family_id:
            raise RuntimeError("head record destination family is inconsistent")
        key = (record.intervention, record.family_id, record.source_mode)
        cell = (record.prompt_variant, record.direction)
        if cell in cells[key]:
            raise RuntimeError("head records contain duplicate factorial cells")
        cells[key].add(cell)
    expected_keys = {
        (intervention, family, mode)
        for intervention in expected_interventions
        for family in families
        for mode in SOURCE_MODES
    }
    if set(cells) != expected_keys or any(
        cell_set != FACTORIAL_CELLS for cell_set in cells.values()
    ):
        raise RuntimeError("head records are factorially incomplete")
    expected_count = (
        len(families)
        * len(PROMPT_VARIANTS)
        * len(DIRECTIONS)
        * len(SOURCE_MODES)
        * (n_heads + len(REFERENCE_INTERVENTIONS))
    )
    if len(records) != expected_count:
        raise RuntimeError("head record count is inconsistent with the factorial design")
    return sorted(families)


def summarize(
    records: list[HeadRecord],
    n_heads: int,
    previous_attention: dict[str, float],
    *,
    seed: int = 42,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    families = validate_head_records(records, n_heads)
    layers = {record.selected_layer for record in records}
    expected = {f"head_{head}" for head in range(n_heads)} | set(REFERENCE_INTERVENTIONS)
    if {record.intervention for record in records} != expected:
        raise RuntimeError("head interventions are incomplete")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.intervention, record.family_id, record.source_mode)].append(
            record.oriented_causal_transfer
        )
    rows: dict[str, dict[str, Any]] = {}
    for intervention in sorted(expected):
        matched: list[float] = []
        permuted: list[float] = []
        advantages: list[float] = []
        for family in families:
            matched_values = grouped.get((intervention, family, "matched"), [])
            permuted_values = grouped.get((intervention, family, "permuted"), [])
            if not matched_values or not permuted_values:
                raise RuntimeError("matched-family head records are incomplete")
            matched_mean = _mean(matched_values)
            permuted_mean = _mean(permuted_values)
            matched.append(matched_mean)
            permuted.append(permuted_mean)
            advantages.append(matched_mean - permuted_mean)
        digest = hashlib.sha256(f"{seed}:{intervention}".encode()).hexdigest()
        row_seed = int(digest[:16], 16) % (2**63 - 1)
        rows[intervention] = {
            "intervention": intervention,
            "head_index": (
                int(intervention.removeprefix("head_"))
                if intervention.startswith("head_")
                else None
            ),
            "family_count": len(families),
            "matched_mean_transfer": _mean(matched),
            "permuted_mean_transfer": _mean(permuted),
            "family_specific_advantage": _mean(advantages),
            "family_specific_advantage_ci_95": _bootstrap(advantages, row_seed, bootstrap_samples),
            "positive_transfer_fraction": sum(value > 0 for value in matched) / len(matched),
        }
    ranking = [rows[f"head_{head}"] for head in range(n_heads)]
    ranking.sort(
        key=lambda row: (
            -float(row["family_specific_advantage"]),
            -float(row["matched_mean_transfer"]),
            int(row["head_index"]),
        )
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    aggregate = rows["aggregate_attention_output"]
    noop = rows["identity_noop"]
    if any(
        float(noop[key]) != 0.0
        for key in (
            "matched_mean_transfer",
            "permuted_mean_transfer",
            "family_specific_advantage",
        )
    ):
        raise RuntimeError("identity no-op summary is not exactly zero")
    sum_matched = sum(float(row["matched_mean_transfer"]) for row in ranking)
    sum_advantage = sum(float(row["family_specific_advantage"]) for row in ranking)
    composition = {
        "interpretation": "composition_and_reproducibility_diagnostic_not_additivity",
        "sum_individual_head_matched_mean_transfer": sum_matched,
        "aggregate_attention_matched_mean_transfer": aggregate["matched_mean_transfer"],
        "matched_transfer_difference": sum_matched - float(aggregate["matched_mean_transfer"]),
        "sum_individual_head_family_specific_advantage": sum_advantage,
        "aggregate_attention_family_specific_advantage": aggregate["family_specific_advantage"],
        "family_specific_advantage_difference": sum_advantage
        - float(aggregate["family_specific_advantage"]),
        "previous_component_attention_matched_mean_transfer": previous_attention[
            "matched_mean_transfer"
        ],
        "current_aggregate_attention_matched_mean_transfer": aggregate["matched_mean_transfer"],
        "component_attention_matched_difference": float(aggregate["matched_mean_transfer"])
        - previous_attention["matched_mean_transfer"],
        "previous_component_attention_permuted_mean_transfer": previous_attention[
            "permuted_mean_transfer"
        ],
        "current_aggregate_attention_permuted_mean_transfer": aggregate["permuted_mean_transfer"],
        "component_attention_permuted_difference": float(aggregate["permuted_mean_transfer"])
        - previous_attention["permuted_mean_transfer"],
        "previous_component_attention_family_specific_advantage": previous_attention[
            "family_specific_advantage"
        ],
        "current_aggregate_attention_family_specific_advantage": aggregate[
            "family_specific_advantage"
        ],
        "component_attention_family_specific_advantage_difference": float(
            aggregate["family_specific_advantage"]
        )
        - previous_attention["family_specific_advantage"],
    }
    return {
        "interpretation_scope": "exploratory_discovery_only",
        "protocol_version": PROTOCOL_VERSION,
        "selected_layer": next(iter(layers)),
        "n_heads": n_heads,
        "family_count": len(families),
        "record_count": len(records),
        "individual_head_ranking": ranking,
        "leading_head": ranking[0]["intervention"],
        "leading_head_interpretation": "exploratory localization candidate, not a circuit",
        "aggregate_attention_reference": aggregate,
        "identity_noop_reference": noop,
        "composition_audit": composition,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_records(path: Path, records: list[HeadRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _write_incomplete_manifest(
    root: Path,
    stage: str,
    upstream: dict[str, Any],
    exc: BaseException | None = None,
) -> None:
    artifact_names = (
        "head_records.jsonl",
        "head_summary.json",
        "head_localization.md",
        "head_final_status.json",
    )
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "complete": False,
        "software_success": False,
        "failure_stage": stage,
        "upstream_provenance": upstream,
        "artifact_hashes": {
            name: sha256(root / name) for name in artifact_names if (root / name).is_file()
        },
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
    if exc is not None:
        manifest["exception_type"] = type(exc).__name__
        manifest["exception_message"] = str(exc)
    _write_json(root / "head_manifest.json", manifest)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    audit = summary["composition_audit"]
    lines = [
        "# Exploratory discovery-only attention-head localization",
        "",
        f"- Selected layer: `{summary['selected_layer']}`",
        f"- Dynamically detected heads: `{summary['n_heads']}`",
        f"- Matched families: `{summary['family_count']}`",
        f"- Leading exploratory head: `{summary['leading_head']}`",
        "- Interpretation: localization candidate only; this is not a circuit claim.",
        "",
        "## Composition and reproducibility diagnostic",
        "",
        "These comparisons do not assume or establish additive head composition.",
        "- Sum of individual-head matched means: "
        f"`{audit['sum_individual_head_matched_mean_transfer']}`",
        "- Aggregate-attention matched mean: "
        f"`{audit['aggregate_attention_matched_mean_transfer']}`",
        "- Current-minus-previous aggregate matched difference: "
        f"`{audit['component_attention_matched_difference']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _target_root(args: argparse.Namespace, discovery: dict[str, Any]) -> Path:
    seed = int(discovery["seed"])
    revision = str(discovery["requested_revision"]).replace("/", "_")
    output: Path = args.output
    return output / f"seed-{seed}-{revision}"


def run(
    args: argparse.Namespace,
    adapter_factory: Callable[[str, str, str], Any] = PythiaAdapter,
) -> dict[str, Any]:
    discovery = _verify_frozen_discovery(args.discovery_root)
    layer_run = verify_layer_run(args.layer_root, discovery)
    component_run = verify_component_run(args.component_root, discovery, layer_run)
    if component_run["layer"] != layer_run["layer"]:
        raise RuntimeError("residual and component selected layers disagree")
    root = _target_root(args, discovery)
    if root.exists():
        if not args.force:
            raise RuntimeError("head output exists; use --force")
        if root.resolve().parent != args.output.resolve():
            raise RuntimeError("refusing unsafe force target")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    upstream = {
        "discovery_artifact_hashes": discovery["artifact_hashes"],
        "layer_run_hashes": layer_run["hashes"],
        "component_run_hashes": component_run["hashes"],
        "model_identity": layer_run["identity"],
        "selected_layer": layer_run["layer"],
    }
    incomplete = {
        "status": "EXPLORATORY_ATTENTION_HEAD_LOCALIZATION_INCOMPLETE",
        "software_success": False,
        "interpretation_scope": "exploratory_discovery_only",
        "selected_layer": layer_run["layer"],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
    _write_json(root / "head_final_status.json", incomplete)
    _write_incomplete_manifest(root, "preflight", upstream)
    stage = "preflight"
    try:
        examples = read_jsonl(args.discovery_root / "matched_dataset.jsonl")
        pairs = build_first_last_pairs(examples)
        deterministic_batches(pairs, args.batch_size)
        stage = "model_loading"
        adapter = adapter_factory(
            str(discovery["model"]), str(discovery["load_revision"]), args.device
        )
        identity = _verify_validation_adapter(adapter, discovery)
        n_heads = int(adapter.model.cfg.n_heads)
        parallel_attn_mlp = component_run.get("parallel_attn_mlp")
        if not isinstance(parallel_attn_mlp, bool) or parallel_attn_mlp != bool(
            getattr(adapter.model.cfg, "parallel_attn_mlp", False)
        ):
            raise RuntimeError("component architecture identity disagrees with loaded model")
        stage = "scanning"
        records = scan_heads(
            adapter.model,
            pairs,
            args.batch_size,
            int(layer_run["layer"]),
            str(layer_run["hashes"]["manifest"]),
            str(component_run["hashes"]["manifest"]),
        )
        records_path = root / "head_records.jsonl"
        _write_records(records_path, records)
        stage = "record_validation"
        validate_head_records(records, n_heads, {pair.family_id for pair in pairs})
        if len(pairs) != 120:
            raise RuntimeError("head localization requires exactly 120 matched families")
        stage = "summarization"
        summary = summarize(
            records,
            n_heads,
            component_run["attention"],
            seed=int(discovery["seed"]),
        )
        summary["upstream_provenance"] = {
            "discovery_artifact_hashes": discovery["artifact_hashes"],
            "layer_run_hashes": layer_run["hashes"],
            "component_run_hashes": component_run["hashes"],
            "model_identity": identity,
        }
        summary_path = root / "head_summary.json"
        report_path = root / "head_localization.md"
        final_path = root / "head_final_status.json"
        _write_json(summary_path, summary)
        _write_report(report_path, summary)
        stage = "finalization"
        final = {
            "status": STATUS,
            "software_success": True,
            "interpretation_scope": "exploratory_discovery_only",
            "selected_layer": layer_run["layer"],
            "dynamically_detected_head_count": n_heads,
            "matched_family_count": len(pairs),
            "record_count": len(records),
            "leading_head": summary["leading_head"],
            "held_out_validation_reused": False,
            "held_out_test_opened": False,
            "activation_patching_performed": True,
            "scientific_confirmation": False,
            "circuit_found": False,
        }
        _write_json(final_path, final)
        artifact_paths = (records_path, summary_path, report_path, final_path)
        manifest = identity | {
            "protocol_version": PROTOCOL_VERSION,
            "complete": True,
            "git_commit": _git_commit(),
            "created_at": datetime.now(UTC).isoformat(),
            "selected_layer": layer_run["layer"],
            "n_heads": n_heads,
            "discovery_artifact_hashes": discovery["artifact_hashes"],
            "layer_run_hashes": layer_run["hashes"],
            "component_run_hashes": component_run["hashes"],
            "held_out_validation_reused": False,
            "held_out_test_opened": False,
            "artifact_hashes": {path.name: sha256(path) for path in artifact_paths},
        }
        _write_json(root / "head_manifest.json", manifest)
        print(f"Head localization: {STATUS}")
        print(f"Artifacts: {root}")
        return final
    except Exception as exc:
        incomplete["error_type"] = type(exc).__name__
        incomplete["error"] = str(exc)
        incomplete["traceback"] = traceback.format_exc()
        incomplete["failure_stage"] = stage
        _write_json(root / "head_final_status.json", incomplete)
        _write_incomplete_manifest(root, stage, upstream, exc)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--discovery-root", type=Path, default=Path("artifacts/position_study/seed-42-main")
    )
    parser.add_argument(
        "--layer-root",
        type=Path,
        default=Path("artifacts/position_localization/seed-42-main"),
    )
    parser.add_argument(
        "--component-root",
        type=Path,
        default=Path("artifacts/position_component_localization/seed-42-main"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/position_head_localization"))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 2:
        parser.error("batch size must be at least 2 for family permutation")
    try:
        run(args)
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
