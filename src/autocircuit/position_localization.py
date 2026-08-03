"""Exploratory causal localization for the matched query-position effect.

Only the frozen discovery population is used. The untouched test split is never
resolved, generated, read, or scored by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
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
from autocircuit.position_study import POSITIONS, STUDY_VERSION

LOCALIZATION_PROTOCOL_VERSION = "position-localization-0.1.0"
LOCALIZATION_STATUS = "EXPLORATORY_LAYER_LOCALIZATION_COMPLETE"
PROMPT_VARIANTS = ("clean", "corrupt")
DIRECTIONS = ("first_to_last", "last_to_first")
SOURCE_MODES = ("matched", "permuted")
BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class FirstLastPair:
    family_id: str
    first: ExamplePair
    last: ExamplePair


@dataclass(frozen=True)
class LayerScanRecord:
    family_id: str
    prompt_variant: str
    direction: str
    source_mode: str
    site: str
    site_index: int
    source_family_id: str
    destination_family_id: str
    source_task_score: float
    destination_task_score: float
    patched_task_score: float
    available_gap: float
    causal_transfer: float
    normalized_transfer: float | None


def residual_stream_sites(n_layers: int) -> list[str]:
    """Return the n_layers + 1 residual-stream boundaries in execution order."""
    if n_layers <= 0:
        raise ValueError("model must contain at least one transformer layer")
    return [f"blocks.{layer}.hook_resid_pre" for layer in range(n_layers)] + [
        f"blocks.{n_layers - 1}.hook_resid_post"
    ]


def build_first_last_pairs(examples: list[ExamplePair]) -> list[FirstLastPair]:
    """Validate and pair the frozen discovery-only matched-position population."""
    if len(examples) != 360:
        raise ValueError("localization requires exactly 360 discovery examples")
    families: dict[str, dict[str, ExamplePair]] = defaultdict(dict)
    for item in examples:
        if item.split != "discovery":
            raise ValueError("localization accepts only the discovery split")
        if item.metadata.get("generator_version") != STUDY_VERSION:
            raise ValueError("localization dataset has the wrong discovery version")
        position = item.metadata.get("normalized_query_position")
        if position not in POSITIONS:
            raise ValueError("localization dataset has an invalid query position")
        if position in families[item.family_id]:
            raise ValueError("localization family contains a duplicate position")
        families[item.family_id][str(position)] = item
    if len(families) != 120:
        raise ValueError("localization requires exactly 120 matched families")

    pairs: list[FirstLastPair] = []
    invariant_fields = (
        "family_id",
        "split",
        "target_text",
        "distractor_text",
        "target_token_id",
        "distractor_token_id",
        "changed_factor",
        "seed",
        "template_id",
    )
    for family_id in sorted(families):
        variants = families[family_id]
        if set(variants) != set(POSITIONS):
            raise ValueError("localization family does not contain all three positions")
        first = variants["first"]
        last = variants["last"]
        if any(getattr(first, field) != getattr(last, field) for field in invariant_fields):
            raise ValueError("first and last variants do not share the same task identity")
        first_length = first.metadata.get("prompt_token_length")
        last_length = last.metadata.get("prompt_token_length")
        if (
            not isinstance(first_length, int)
            or isinstance(first_length, bool)
            or first_length <= 0
            or first_length != last_length
        ):
            raise ValueError("first and last variants do not share one token length")
        pairs.append(FirstLastPair(family_id, first, last))
    return pairs


def _prompt(item: ExamplePair, variant: str) -> str:
    if variant == "clean":
        return item.clean_prompt
    if variant == "corrupt":
        return item.corrupt_prompt
    raise ValueError(f"unknown prompt variant: {variant}")


def _task_scores(logits: Any, items: list[ExamplePair], variant: str) -> Any:
    """Return a task-aligned score: clean LD and negated corrupt LD."""
    import torch

    rows = torch.arange(len(items), device=logits.device)
    targets = torch.tensor(
        [item.target_token_id for item in items],
        device=logits.device,
        dtype=torch.long,
    )
    distractors = torch.tensor(
        [item.distractor_token_id for item in items],
        device=logits.device,
        dtype=torch.long,
    )
    raw = logits[rows, -1, targets] - logits[rows, -1, distractors]
    return raw if variant == "clean" else -raw


def _patch_query_position(source: Any) -> Callable[[Any, Any], Any]:
    """Build a TransformerLens hook that patches only the final query position."""

    def hook(value: Any, hook_point: Any) -> Any:
        del hook_point
        if value.shape != source.shape:
            raise RuntimeError("source and destination activation shapes differ")
        patched = value.clone()
        patched[:, -1, :] = source[:, -1, :]
        return patched

    return hook


def run_residual_stream_scan(
    model: Any,
    pairs: list[FirstLastPair],
    batch_size: int,
) -> list[LayerScanRecord]:
    """Run bidirectional query-position patching at every residual boundary."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    sites = residual_stream_sites(int(model.cfg.n_layers))
    buckets: dict[int, list[FirstLastPair]] = defaultdict(list)
    for pair in pairs:
        length = pair.first.metadata.get("prompt_token_length")
        if not isinstance(length, int) or isinstance(length, bool):
            raise ValueError("localization pair has no valid prompt token length")
        buckets[length].append(pair)

    records: list[LayerScanRecord] = []
    with torch.inference_mode():
        for length in sorted(buckets):
            bucket = buckets[length]
            for start in range(0, len(bucket), batch_size):
                chunk = bucket[start : start + batch_size]
                for variant in PROMPT_VARIANTS:
                    for direction in DIRECTIONS:
                        if direction == "first_to_last":
                            source_items = [pair.first for pair in chunk]
                            destination_items = [pair.last for pair in chunk]
                            orientation = 1.0
                        else:
                            source_items = [pair.last for pair in chunk]
                            destination_items = [pair.first for pair in chunk]
                            orientation = -1.0
                        source_prompts = [_prompt(item, variant) for item in source_items]
                        destination_prompts = [
                            _prompt(item, variant) for item in destination_items
                        ]
                        source_logits, source_cache = model.run_with_cache(
                            source_prompts,
                            return_type="logits",
                            names_filter=sites,
                        )
                        destination_logits = model(
                            destination_prompts,
                            return_type="logits",
                        )
                        source_scores = _task_scores(source_logits, source_items, variant)
                        destination_scores = _task_scores(
                            destination_logits,
                            destination_items,
                            variant,
                        )
                        for site_index, site in enumerate(sites):
                            source_activation = source_cache[site]
                            modes = SOURCE_MODES if len(chunk) > 1 else ("matched",)
                            for source_mode in modes:
                                if source_mode == "matched":
                                    patch_source = source_activation
                                    source_family_ids = [
                                        item.family_id for item in source_items
                                    ]
                                else:
                                    patch_source = torch.roll(
                                        source_activation,
                                        shifts=1,
                                        dims=0,
                                    )
                                    source_family_ids = [source_items[-1].family_id] + [
                                        item.family_id for item in source_items[:-1]
                                    ]
                                patched_logits = model.run_with_hooks(
                                    destination_prompts,
                                    return_type="logits",
                                    fwd_hooks=[
                                        (site, _patch_query_position(patch_source))
                                    ],
                                )
                                patched_scores = _task_scores(
                                    patched_logits,
                                    destination_items,
                                    variant,
                                )
                                for row, destination in enumerate(destination_items):
                                    source_score = float(source_scores[row])
                                    destination_score = float(destination_scores[row])
                                    patched_score = float(patched_scores[row])
                                    available_gap = orientation * (
                                        source_score - destination_score
                                    )
                                    causal_transfer = orientation * (
                                        patched_score - destination_score
                                    )
                                    normalized = (
                                        causal_transfer / available_gap
                                        if abs(available_gap) > 1e-12
                                        else None
                                    )
                                    finite_values = (
                                        source_score,
                                        destination_score,
                                        patched_score,
                                        available_gap,
                                        causal_transfer,
                                    )
                                    if not all(
                                        math.isfinite(value) for value in finite_values
                                    ):
                                        raise RuntimeError(
                                            "non-finite value in localization record"
                                        )
                                    if normalized is not None and not math.isfinite(
                                        normalized
                                    ):
                                        raise RuntimeError(
                                            "non-finite normalized localization record"
                                        )
                                    records.append(
                                        LayerScanRecord(
                                            family_id=destination.family_id,
                                            prompt_variant=variant,
                                            direction=direction,
                                            source_mode=source_mode,
                                            site=site,
                                            site_index=site_index,
                                            source_family_id=source_family_ids[row],
                                            destination_family_id=destination.family_id,
                                            source_task_score=source_score,
                                            destination_task_score=destination_score,
                                            patched_task_score=patched_score,
                                            available_gap=available_gap,
                                            causal_transfer=causal_transfer,
                                            normalized_transfer=normalized,
                                        )
                                    )
    return records


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate an empty mean")
    return sum(values) / len(values)


def _group_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _bootstrap_mean_interval(
    values: list[float],
    samples: int,
    seed: int,
) -> list[float]:
    import torch

    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indexes = torch.randint(
        len(values),
        (samples, len(values)),
        generator=generator,
        device="cpu",
    )
    means = tensor[indexes].mean(dim=1)
    levels = torch.tensor([0.025, 0.975], dtype=tensor.dtype)
    quantiles = torch.quantile(means, levels)
    return [float(quantiles[0]), float(quantiles[1])]


def _bootstrap_ratio_interval(
    numerators: list[float],
    denominators: list[float],
    samples: int,
    seed: int,
) -> list[float] | None:
    import torch

    numerator = torch.tensor(numerators, dtype=torch.float64)
    denominator = torch.tensor(denominators, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indexes = torch.randint(
        len(numerators),
        (samples, len(numerators)),
        generator=generator,
        device="cpu",
    )
    numerator_means = numerator[indexes].mean(dim=1)
    denominator_means = denominator[indexes].mean(dim=1)
    valid = denominator_means.abs() > 1e-12
    ratios = numerator_means[valid] / denominator_means[valid]
    if int(ratios.numel()) < max(100, samples // 2):
        return None
    levels = torch.tensor([0.025, 0.975], dtype=ratios.dtype)
    quantiles = torch.quantile(ratios, levels)
    return [float(quantiles[0]), float(quantiles[1])]


def summarize_layer_scan(
    records: list[LayerScanRecord],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate family-level causal transfer and rank candidate boundaries."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    groups: dict[tuple[str, str, str, str], list[LayerScanRecord]] = defaultdict(list)
    for record in records:
        groups[
            (
                record.prompt_variant,
                record.direction,
                record.source_mode,
                record.site,
            )
        ].append(record)

    rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        variant, direction, source_mode, site = key
        group = groups[key]
        transfers = [record.causal_transfer for record in group]
        gaps = [record.available_gap for record in group]
        mean_transfer = _mean(transfers)
        mean_gap = _mean(gaps)
        normalized = mean_transfer / mean_gap if abs(mean_gap) > 1e-12 else None
        key_text = "|".join(key)
        rows.append(
            {
                "prompt_variant": variant,
                "direction": direction,
                "source_mode": source_mode,
                "site": site,
                "site_index": group[0].site_index,
                "family_count": len(group),
                "mean_source_task_score": _mean(
                    [record.source_task_score for record in group]
                ),
                "mean_destination_task_score": _mean(
                    [record.destination_task_score for record in group]
                ),
                "mean_patched_task_score": _mean(
                    [record.patched_task_score for record in group]
                ),
                "mean_available_gap": mean_gap,
                "mean_causal_transfer": mean_transfer,
                "causal_transfer_ci_95": _bootstrap_mean_interval(
                    transfers,
                    bootstrap_samples,
                    _group_seed(seed, key_text + "|transfer"),
                ),
                "normalized_transfer": normalized,
                "normalized_transfer_ci_95": _bootstrap_ratio_interval(
                    transfers,
                    gaps,
                    bootstrap_samples,
                    _group_seed(seed, key_text + "|ratio"),
                ),
                "positive_transfer_fraction": sum(value > 0 for value in transfers)
                / len(transfers),
            }
        )

    site_indices: dict[str, int] = {}
    matched_by_site: dict[str, list[float]] = defaultdict(list)
    permuted_by_site: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        site = str(row["site"])
        site_indices[site] = int(row["site_index"])
        if row["source_mode"] == "matched":
            matched_by_site[site].append(float(row["mean_causal_transfer"]))
        else:
            permuted_by_site[site].append(float(row["mean_causal_transfer"]))

    ranking: list[dict[str, Any]] = []
    for site in sorted(site_indices, key=site_indices.get):
        matched_mean = _mean(matched_by_site[site])
        permuted_values = permuted_by_site.get(site, [])
        permuted_mean = _mean(permuted_values) if permuted_values else 0.0
        ranking.append(
            {
                "site": site,
                "site_index": site_indices[site],
                "matched_mean_causal_transfer": matched_mean,
                "permuted_mean_causal_transfer": permuted_mean,
                "family_specific_transfer_advantage": matched_mean - permuted_mean,
            }
        )
    ranking.sort(
        key=lambda row: (
            float(row["family_specific_transfer_advantage"]),
            float(row["matched_mean_causal_transfer"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    return {
        "protocol_version": LOCALIZATION_PROTOCOL_VERSION,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "record_count": len(records),
        "group_summaries": rows,
        "site_ranking": ranking,
        "interpretation_scope": "exploratory_discovery_only",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_records(path: Path, records: list[LayerScanRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(asdict(record), allow_nan=False, sort_keys=True) + "\n"
            )


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    ranking = summary["site_ranking"]
    lines = [
        "# Exploratory residual-stream localization",
        "",
        f"**Status:** `{LOCALIZATION_STATUS}`",
        "",
        "This scan patches the final query-token residual stream between matched first and",
        "last query-position prompts. It is exploratory and uses only the frozen discovery",
        "population. It does not constitute circuit confirmation.",
        "",
        "## Candidate residual boundaries",
        "",
        "| Rank | Site | Matched transfer | Permuted transfer | Family-specific advantage |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            "| {rank} | `{site}` | {matched:.6f} | {permuted:.6f} | {advantage:.6f} |".format(
                rank=row["rank"],
                site=row["site"],
                matched=row["matched_mean_causal_transfer"],
                permuted=row["permuted_mean_causal_transfer"],
                advantage=row["family_specific_transfer_advantage"],
            )
        )
    lines += [
        "",
        "## Safety and interpretation",
        "",
        "- Discovery split only: **yes**.",
        "- Held-out validation reused: **no**.",
        "- Test split opened: **no**.",
        "- Activation patching performed: **yes**.",
        "- Circuit found or confirmed: **no**.",
        "",
        "The ranking is a candidate-localization output. Head, MLP, edge, ablation, null,",
        "and final test controls are still required before making a circuit claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_localization(
    args: argparse.Namespace,
    adapter_factory: Callable[[str, str, str], Any] = PythiaAdapter,
) -> dict[str, Any]:
    contract = _verify_frozen_discovery(args.discovery_root)
    seed = int(contract["seed"])
    revision = str(contract["requested_revision"])
    root = args.output / f"seed-{seed}-{revision.replace('/', '_')}"
    manifest_path = root / "localization_manifest.json"

    if root.exists() and args.force:
        shutil.rmtree(root)
    if root.exists() and args.resume:
        if not manifest_path.is_file():
            raise RuntimeError("localization resume manifest is missing")
        manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.get("complete") is not True:
            raise RuntimeError("incomplete localization requires --force to recompute")
        for relative, digest in manifest.get("artifact_hashes", {}).items():
            verify_resume(root / str(relative), str(digest))
        final: dict[str, Any] = json.loads(
            (root / "localization_final_status.json").read_text(encoding="utf-8")
        )
        print(f"Position localization resumed: {final['status']}")
        print(f"Artifacts: {root}")
        return final
    if root.exists():
        raise RuntimeError("localization output exists; use --resume or --force")

    root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "protocol_version": LOCALIZATION_PROTOCOL_VERSION,
        "complete": False,
        "artifact_hashes": {},
        "discovery_artifact_hashes": contract["artifact_hashes"],
        "discovery_commit": contract["discovery_commit"],
        "seed": seed,
        "requested_revision": revision,
        "requested_device": args.device,
        "batch_size": args.batch_size,
        "created_at": datetime.now(UTC).isoformat(),
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
    }
    _write_json(manifest_path, manifest)

    adapter = adapter_factory(
        str(contract["model"]),
        str(contract["load_revision"]),
        args.device,
    )
    model_identity = _verify_validation_adapter(adapter, contract)
    examples = read_jsonl(args.discovery_root / "matched_dataset.jsonl")
    pairs = build_first_last_pairs(examples)
    records = run_residual_stream_scan(adapter.model, pairs, args.batch_size)
    summary = summarize_layer_scan(records, seed=seed)

    records_path = root / "layer_scan_records.jsonl"
    summary_path = root / "layer_scan_summary.json"
    report_path = root / "layer_scan.md"
    final_path = root / "localization_final_status.json"
    _write_records(records_path, records)
    _write_json(summary_path, summary)
    _write_report(report_path, summary)
    final = {
        "status": LOCALIZATION_STATUS,
        "software_success": True,
        "scientific_confirmation": False,
        "interpretation_scope": "exploratory_discovery_only",
        "matched_family_count": len(pairs),
        "residual_boundary_count": len(
            residual_stream_sites(int(adapter.model.cfg.n_layers))
        ),
        "record_count": len(records),
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "activation_patching_performed": True,
        "circuit_found": False,
    }
    _write_json(final_path, final)

    artifact_paths = (records_path, summary_path, report_path, final_path)
    manifest |= model_identity | {
        "complete": True,
        "git_commit": _git_commit(),
        "completed_at": datetime.now(UTC).isoformat(),
        "family_count": len(pairs),
        "record_count": len(records),
        "artifact_hashes": {
            str(path.relative_to(root)).replace("\\", "/"): sha256(path)
            for path in artifact_paths
        },
        "command_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_json(manifest_path, manifest)
    print(f"Position localization: {LOCALIZATION_STATUS}")
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
        "--output",
        type=Path,
        default=Path("artifacts/position_localization"),
    )
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--resume", action="store_true")
    reuse.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    try:
        run_localization(args)
    except Exception as exc:
        print(f"SOFTWARE FAILURE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
