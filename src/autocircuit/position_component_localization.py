"""Exploratory attention-vs-MLP localization inside the selected position-effect block."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autocircuit.artifacts import sha256, verify_resume
from autocircuit.datasets.associative_recall import ExamplePair, read_jsonl
from autocircuit.pipeline import (
    PythiaAdapter,
    _git_commit,
    _verify_frozen_discovery,
    _verify_validation_adapter,
)
from autocircuit.position_localization import (
    DIRECTIONS,
    LOCALIZATION_PROTOCOL_VERSION,
    LOCALIZATION_STATUS,
    PROMPT_VARIANTS,
    SOURCE_MODES,
    FirstLastPair,
    _patch_query_position,
    _prompt,
    _task_scores,
    build_first_last_pairs,
)

PROTOCOL_VERSION = "position-component-localization-0.1.0"
STATUS = "EXPLORATORY_BLOCK_COMPONENT_LOCALIZATION_COMPLETE"
INTERVENTIONS = (
    "residual_pre",
    "attention_output",
    "mlp_output",
    "attention_plus_mlp",
    "residual_post",
)


@dataclass(frozen=True)
class ComponentRecord:
    family_id: str
    prompt_variant: str
    direction: str
    source_mode: str
    layer: int
    intervention: str
    source_family_id: str
    source_score: float
    destination_score: float
    patched_score: float
    causal_transfer: float


def intervention_sites(layer: int) -> dict[str, tuple[str, ...]]:
    if layer < 0:
        raise ValueError("layer must be non-negative")
    prefix = f"blocks.{layer}"
    attn = f"{prefix}.hook_attn_out"
    mlp = f"{prefix}.hook_mlp_out"
    return {
        "residual_pre": (f"{prefix}.hook_resid_pre",),
        "attention_output": (attn,),
        "mlp_output": (mlp,),
        "attention_plus_mlp": (attn, mlp),
        "residual_post": (f"{prefix}.hook_resid_post",),
    }


def select_layer(summary: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    ranking = summary.get("site_ranking")
    if not isinstance(ranking, list) or not ranking or not isinstance(ranking[0], dict):
        raise RuntimeError("layer ranking is missing")
    top = dict(ranking[0])
    site = top.get("site")
    advantage = top.get("family_specific_transfer_advantage")
    if not isinstance(site, str):
        raise RuntimeError("top layer site is malformed")
    if not isinstance(advantage, (int, float)) or isinstance(advantage, bool):
        raise RuntimeError("top layer advantage is malformed")
    if not math.isfinite(float(advantage)) or float(advantage) <= 0:
        raise RuntimeError("top layer advantage must be positive")
    match = re.fullmatch(r"blocks\.(\d+)\.hook_resid_(?:pre|post)", site)
    if match is None:
        raise RuntimeError("top layer site is not a residual boundary")
    return int(match.group(1)), top


def verify_layer_run(root: Path, discovery: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / "localization_manifest.json"
    summary_path = root / "layer_scan_summary.json"
    final_path = root / "localization_final_status.json"
    if not all(path.is_file() for path in (manifest_path, summary_path, final_path)):
        raise RuntimeError("layer localization artifacts are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("complete") is True
        and manifest.get("protocol_version") == LOCALIZATION_PROTOCOL_VERSION
        and manifest.get("held_out_validation_reused") is False
        and manifest.get("held_out_test_opened") is False
        and manifest.get("discovery_artifact_hashes") == discovery["artifact_hashes"]
    ):
        raise RuntimeError("layer localization manifest is not eligible")
    for relative, digest in manifest.get("artifact_hashes", {}).items():
        path = root / str(relative)
        if path.resolve().parent != root.resolve() or not isinstance(digest, str):
            raise RuntimeError("unsafe layer artifact hash entry")
        verify_resume(path, digest)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if not (
        final.get("status") == LOCALIZATION_STATUS
        and final.get("software_success") is True
        and final.get("activation_patching_performed") is True
        and final.get("held_out_test_opened") is False
        and final.get("circuit_found") is False
    ):
        raise RuntimeError("layer localization status is inconsistent")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    layer, top = select_layer(summary)
    return {
        "layer": layer,
        "top_boundary": top,
        "hashes": {
            "manifest": sha256(manifest_path),
            "summary": sha256(summary_path),
            "final": sha256(final_path),
        },
    }


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


def scan_components(
    model: Any,
    pairs: list[FirstLastPair],
    batch_size: int,
    layer: int,
) -> list[ComponentRecord]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if layer < 0 or layer >= int(model.cfg.n_layers):
        raise ValueError("selected layer is outside the model")
    interventions = intervention_sites(layer)
    sites = sorted({site for group in interventions.values() for site in group})
    buckets: dict[int, list[FirstLastPair]] = defaultdict(list)
    for pair in pairs:
        length = pair.first.metadata.get("prompt_token_length")
        if not isinstance(length, int) or isinstance(length, bool):
            raise ValueError("pair has no valid token length")
        buckets[length].append(pair)
    records: list[ComponentRecord] = []
    with torch.inference_mode():
        for length in sorted(buckets):
            bucket = buckets[length]
            for start in range(0, len(bucket), batch_size):
                chunk = bucket[start : start + batch_size]
                for variant in PROMPT_VARIANTS:
                    for direction in DIRECTIONS:
                        source_items = [
                            pair.first if direction == "first_to_last" else pair.last
                            for pair in chunk
                        ]
                        destination_items = [
                            pair.last if direction == "first_to_last" else pair.first
                            for pair in chunk
                        ]
                        orientation = 1.0 if direction == "first_to_last" else -1.0
                        source_prompts = [_prompt(item, variant) for item in source_items]
                        destination_prompts = [
                            _prompt(item, variant) for item in destination_items
                        ]
                        source_logits, cache = model.run_with_cache(
                            source_prompts, return_type="logits", names_filter=sites
                        )
                        destination_logits = model(destination_prompts, return_type="logits")
                        source_scores = _task_scores(source_logits, source_items, variant)
                        destination_scores = _task_scores(
                            destination_logits, destination_items, variant
                        )
                        modes = SOURCE_MODES if len(chunk) > 1 else ("matched",)
                        for mode in modes:
                            mode_scores = _mode_value(source_scores, mode)
                            family_ids = _mode_families(source_items, mode)
                            for intervention in INTERVENTIONS:
                                hooks = [
                                    (
                                        site,
                                        _patch_query_position(_mode_value(cache[site], mode)),
                                    )
                                    for site in interventions[intervention]
                                ]
                                patched_logits = model.run_with_hooks(
                                    destination_prompts,
                                    return_type="logits",
                                    fwd_hooks=hooks,
                                )
                                patched_scores = _task_scores(
                                    patched_logits, destination_items, variant
                                )
                                for row, destination in enumerate(destination_items):
                                    source_score = float(mode_scores[row])
                                    destination_score = float(destination_scores[row])
                                    patched_score = float(patched_scores[row])
                                    transfer = orientation * (
                                        patched_score - destination_score
                                    )
                                    if not all(
                                        math.isfinite(value)
                                        for value in (
                                            source_score,
                                            destination_score,
                                            patched_score,
                                            transfer,
                                        )
                                    ):
                                        raise RuntimeError("non-finite component record")
                                    records.append(
                                        ComponentRecord(
                                            family_id=destination.family_id,
                                            prompt_variant=variant,
                                            direction=direction,
                                            source_mode=mode,
                                            layer=layer,
                                            intervention=intervention,
                                            source_family_id=family_ids[row],
                                            source_score=source_score,
                                            destination_score=destination_score,
                                            patched_score=patched_score,
                                            causal_transfer=transfer,
                                        )
                                    )
    return records


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _bootstrap(values: list[float], seed: int, samples: int = 10_000) -> list[float]:
    import torch

    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indexes = torch.randint(len(values), (samples, len(values)), generator=generator)
    means = tensor[indexes].mean(dim=1)
    levels = torch.tensor([0.025, 0.975], dtype=tensor.dtype)
    bounds = torch.quantile(means, levels)
    return [float(bounds[0]), float(bounds[1])]


def summarize(records: list[ComponentRecord], seed: int = 42) -> dict[str, Any]:
    if not records:
        raise ValueError("component scan is empty")
    layers = {record.layer for record in records}
    if len(layers) != 1:
        raise ValueError("component scan mixes layers")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.intervention, record.family_id, record.source_mode)].append(
            record.causal_transfer
        )
    families = sorted({record.family_id for record in records})
    ranking: list[dict[str, Any]] = []
    for intervention in INTERVENTIONS:
        matched: list[float] = []
        permuted: list[float] = []
        advantages: list[float] = []
        for family in families:
            m = grouped.get((intervention, family, "matched"), [])
            p = grouped.get((intervention, family, "permuted"), [])
            if not m:
                raise RuntimeError("matched family records are incomplete")
            m_mean = _mean(m)
            matched.append(m_mean)
            if p:
                p_mean = _mean(p)
                permuted.append(p_mean)
                advantages.append(m_mean - p_mean)
        digest = hashlib.sha256(f"{seed}:{intervention}".encode()).hexdigest()
        row_seed = int(digest[:16], 16) % (2**63 - 1)
        ranking.append(
            {
                "intervention": intervention,
                "family_count": len(matched),
                "matched_mean_transfer": _mean(matched),
                "permuted_mean_transfer": _mean(permuted) if permuted else 0.0,
                "family_specific_advantage": _mean(advantages) if advantages else 0.0,
                "family_specific_advantage_ci_95": (
                    _bootstrap(advantages, row_seed) if advantages else None
                ),
            }
        )
    ranking.sort(
        key=lambda row: (
            float(row["family_specific_advantage"]),
            float(row["matched_mean_transfer"]),
        ),
        reverse=True,
    )
    atomic = [
        row
        for row in ranking
        if row["intervention"] in {"attention_output", "mlp_output"}
    ]
    atomic.sort(key=lambda row: float(row["family_specific_advantage"]), reverse=True)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "selected_layer": next(iter(layers)),
        "record_count": len(records),
        "intervention_ranking": ranking,
        "leading_atomic_component": atomic[0]["intervention"],
        "interpretation_scope": "exploratory_discovery_only",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    args: argparse.Namespace,
    adapter_factory: Callable[[str, str, str], Any] = PythiaAdapter,
) -> dict[str, Any]:
    discovery = _verify_frozen_discovery(args.discovery_root)
    layer_run = verify_layer_run(args.layer_root, discovery)
    seed = int(discovery["seed"])
    revision = str(discovery["requested_revision"])
    root = args.output / f"seed-{seed}-{revision.replace('/', '_')}"
    if root.exists() and args.force:
        shutil.rmtree(root)
    if root.exists():
        raise RuntimeError("component output exists; use --force")
    root.mkdir(parents=True)
    adapter = adapter_factory(
        str(discovery["model"]), str(discovery["load_revision"]), args.device
    )
    identity = _verify_validation_adapter(adapter, discovery)
    examples = read_jsonl(args.discovery_root / "matched_dataset.jsonl")
    pairs = build_first_last_pairs(examples)
    records = scan_components(adapter.model, pairs, args.batch_size, int(layer_run["layer"]))
    summary = summarize(records, seed)
    records_path = root / "component_records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    summary_path = root / "component_summary.json"
    final_path = root / "component_final_status.json"
    _write_json(summary_path, summary)
    final = {
        "status": STATUS,
        "software_success": True,
        "scientific_confirmation": False,
        "selected_layer": layer_run["layer"],
        "matched_family_count": len(pairs),
        "record_count": len(records),
        "leading_atomic_component": summary["leading_atomic_component"],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "activation_patching_performed": True,
        "circuit_found": False,
    }
    _write_json(final_path, final)
    manifest = identity | {
        "protocol_version": PROTOCOL_VERSION,
        "complete": True,
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "discovery_hashes": discovery["artifact_hashes"],
        "layer_run_hashes": layer_run["hashes"],
        "selected_layer": layer_run["layer"],
        "parallel_attn_mlp": bool(getattr(adapter.model.cfg, "parallel_attn_mlp", False)),
        "artifact_hashes": {
            records_path.name: sha256(records_path),
            summary_path.name: sha256(summary_path),
            final_path.name: sha256(final_path),
        },
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
    }
    _write_json(root / "component_manifest.json", manifest)
    print(f"Component localization: {STATUS}")
    print(f"Artifacts: {root}")
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--discovery-root",
        type=Path,
        default=Path("artifacts/position_study/seed-42-main"),
    )
    parser.add_argument(
        "--layer-root",
        type=Path,
        default=Path("artifacts/position_localization/seed-42-main"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/position_component_localization"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    try:
        run(args)
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
