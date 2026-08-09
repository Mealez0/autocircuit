"""Exploratory discovery-only causal analysis of attention-head sets."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

from autocircuit.artifacts import sha256
from autocircuit.datasets.associative_recall import read_jsonl
from autocircuit.pipeline import (
    PythiaAdapter,
    _git_commit,
    _verify_frozen_discovery,
    _verify_validation_adapter,
)
from autocircuit.position_component_localization import (
    PROTOCOL_VERSION as COMPONENT_PROTOCOL_VERSION,
)
from autocircuit.position_head_localization import (
    PROTOCOL_VERSION as HEAD_PROTOCOL_VERSION,
)
from autocircuit.position_head_localization import (
    REFERENCE_INTERVENTIONS,
    HeadRecord,
    _bootstrap,
    _close,
    _finite_number,
    _identity_final_position,
    _mean,
    _mode_families,
    _mode_value,
    _patch_final_position,
    _read_jsonl_dicts,
    _require_hashed_artifacts,
    _verify_manifest_identity,
    deterministic_batches,
    validate_head_records,
    verify_component_run,
    verify_layer_run,
)
from autocircuit.position_head_localization import (
    STATUS as HEAD_STATUS,
)
from autocircuit.position_head_localization import (
    summarize as summarize_heads,
)
from autocircuit.position_localization import (
    DIRECTIONS,
    PROMPT_VARIANTS,
    SOURCE_MODES,
    FirstLastPair,
    _prompt,
    _task_scores,
    build_first_last_pairs,
)

PROTOCOL_VERSION = "position-head-set-analysis-0.1.0"
STATUS = "EXPLORATORY_ATTENTION_HEAD_SET_ANALYSIS_COMPLETE"
BOOTSTRAP_SAMPLES = 10_000
REPRODUCIBILITY_ABS_TOLERANCE = 1e-9
HEAD_REQUIRED_ARTIFACTS = {
    "head_records.jsonl",
    "head_summary.json",
    "head_final_status.json",
}


@dataclass(frozen=True)
class HeadSetRecord:
    family_id: str
    prompt_variant: str
    direction: str
    source_mode: str
    selected_layer: int
    intervention: str
    included_heads: tuple[int, ...]
    hook_site: str
    source_family_id: str
    destination_family_id: str
    source_score: float
    destination_baseline: float
    patched_score: float
    oriented_causal_transfer: float
    layer_run_manifest_hash: str
    component_run_manifest_hash: str
    head_run_manifest_hash: str


class ReproducibilityError(RuntimeError):
    """Fail-closed error carrying machine-readable reproduction diagnostics."""

    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(
            "reproducibility control failed: "
            + json.dumps(details, sort_keys=True, default=str)
        )


def unordered_head_pairs(n_heads: int) -> list[tuple[int, int]]:
    if n_heads < 2:
        raise ValueError("head-set analysis requires at least two heads")
    return list(itertools.combinations(range(n_heads), 2))


def intervention_heads(n_heads: int) -> dict[str, tuple[int, ...]]:
    heads = tuple(range(n_heads))
    pairs = unordered_head_pairs(n_heads)
    result: dict[str, tuple[int, ...]] = {
        f"pair_patch_{left}_{right}": (left, right) for left, right in pairs
    }
    result["all_heads_z_patch"] = heads
    result |= {
        f"leave_one_out_{head}": tuple(value for value in heads if value != head) for head in heads
    }
    result |= {
        f"leave_pair_out_{left}_{right}": tuple(
            value for value in heads if value not in (left, right)
        )
        for left, right in pairs
    }
    return result


def _patch_head_set(source: Any, heads: tuple[int, ...]) -> Callable[..., Any]:
    if len(set(heads)) != len(heads) or any(head < 0 for head in heads):
        raise ValueError("patched head indexes must be unique and non-negative")

    def callback(value: Any, hook: Any = None) -> Any:
        del hook
        if getattr(value, "ndim", None) != 4 or getattr(source, "ndim", None) != 4:
            raise RuntimeError("hook_z activation must have shape [batch, position, head, d_head]")
        if value.shape != source.shape:
            raise RuntimeError("source and destination hook_z shapes differ")
        if any(head >= value.shape[2] for head in heads):
            raise ValueError("head index is outside hook_z head dimension")
        patched = value.clone()
        if heads:
            patched[:, -1, list(heads), :] = source[:, -1, list(heads), :]
        elif not patched.equal(value):
            raise RuntimeError("empty head-set intervention changed the activation")
        return patched

    return callback


def _load_head_records(path: Path) -> list[HeadRecord]:
    records: list[HeadRecord] = []
    for row in _read_jsonl_dicts(path, "head records"):
        try:
            records.append(HeadRecord(**row))
        except TypeError as exc:
            raise RuntimeError("head record schema is malformed") from exc
    return records


def _compare_stat_rows(stored: dict[str, Any], computed: dict[str, Any], label: str) -> None:
    if stored.get("intervention") != computed.get("intervention"):
        raise RuntimeError(f"{label} ordering is inconsistent")
    for key in (
        "family_count",
        "head_index",
    ):
        if stored.get(key) != computed.get(key):
            raise RuntimeError(f"{label} {key} is inconsistent")
    for key in (
        "matched_mean_transfer",
        "permuted_mean_transfer",
        "family_specific_advantage",
        "positive_transfer_fraction",
    ):
        if not _close(_finite_number(stored.get(key), f"{label} {key}"), float(computed[key])):
            raise RuntimeError(f"{label} {key} disagrees with raw records")
    stored_ci = stored.get("family_specific_advantage_ci_95")
    computed_ci = computed.get("family_specific_advantage_ci_95")
    if (
        not isinstance(stored_ci, list)
        or not isinstance(computed_ci, list)
        or len(stored_ci) != 2
        or any(
            not _close(_finite_number(left, f"{label} interval"), float(right))
            for left, right in zip(stored_ci, computed_ci, strict=True)
        )
    ):
        raise RuntimeError(f"{label} bootstrap interval disagrees with raw records")


def verify_head_run(
    root: Path,
    discovery: dict[str, Any],
    layer_run: dict[str, Any],
    component_run: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = root / "head_manifest.json"
    summary_path = root / "head_summary.json"
    final_path = root / "head_final_status.json"
    records_path = root / "head_records.jsonl"
    for path in (manifest_path, summary_path, final_path):
        if not path.is_file():
            raise RuntimeError(f"missing head localization artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (manifest, summary, final)):
        raise RuntimeError("head localization JSON artifacts are malformed")
    _require_hashed_artifacts(
        root, manifest.get("artifact_hashes"), HEAD_REQUIRED_ARTIFACTS, "head run"
    )
    identity = _verify_manifest_identity(manifest, discovery, "head run")
    if identity != layer_run["identity"] or identity != component_run["identity"]:
        raise RuntimeError("head localization model identity disagrees with upstream")
    if not (
        manifest.get("complete") is True
        and manifest.get("protocol_version") == HEAD_PROTOCOL_VERSION
        and manifest.get("selected_layer") == layer_run["layer"] == component_run["layer"]
        and manifest.get("discovery_artifact_hashes") == discovery["artifact_hashes"]
        and manifest.get("layer_run_hashes") == layer_run["hashes"]
        and manifest.get("component_run_hashes") == component_run["hashes"]
        and manifest.get("held_out_validation_reused") is False
        and manifest.get("held_out_test_opened") is False
        and final.get("status") == HEAD_STATUS
        and final.get("software_success") is True
        and final.get("interpretation_scope") == "exploratory_discovery_only"
        and final.get("held_out_validation_reused") is False
        and final.get("held_out_test_opened") is False
        and final.get("activation_patching_performed") is True
        and final.get("scientific_confirmation") is False
        and final.get("circuit_found") is False
        and summary.get("protocol_version") == HEAD_PROTOCOL_VERSION
    ):
        raise RuntimeError("head localization run is not eligible")
    n_heads = manifest.get("n_heads")
    if not isinstance(n_heads, int) or isinstance(n_heads, bool) or n_heads < 2:
        raise RuntimeError("head localization head count is malformed")
    records = _load_head_records(records_path)
    families = validate_head_records(records, n_heads)
    if len(families) != 120:
        raise RuntimeError("head localization family count is incomplete")
    recomputed = summarize_heads(
        records, n_heads, component_run["attention"], seed=int(discovery["seed"])
    )
    stored_ranking = summary.get("individual_head_ranking")
    if not isinstance(stored_ranking, list) or len(stored_ranking) != n_heads:
        raise RuntimeError("head localization ranking is malformed")
    for stored, computed in zip(stored_ranking, recomputed["individual_head_ranking"], strict=True):
        if not isinstance(stored, dict):
            raise RuntimeError("head localization ranking row is malformed")
        _compare_stat_rows(stored, computed, "head ranking")
        if stored.get("rank") != computed.get("rank"):
            raise RuntimeError("head localization rank is inconsistent")
    for key in ("aggregate_attention_reference", "identity_noop_reference"):
        stored = summary.get(key)
        computed = recomputed[key]
        if not isinstance(stored, dict):
            raise RuntimeError(f"head localization {key} is malformed")
        _compare_stat_rows(stored, computed, key)
    expected_count = (
        120
        * len(PROMPT_VARIANTS)
        * len(DIRECTIONS)
        * len(SOURCE_MODES)
        * (n_heads + len(REFERENCE_INTERVENTIONS))
    )
    if not (
        len(records) == expected_count
        and summary.get("record_count") == expected_count
        and final.get("record_count") == expected_count
        and summary.get("family_count") == final.get("matched_family_count") == 120
        and summary.get("selected_layer") == final.get("selected_layer") == layer_run["layer"]
        and summary.get("n_heads") == final.get("dynamically_detected_head_count") == n_heads
        and summary.get("leading_head") == final.get("leading_head") == recomputed["leading_head"]
    ):
        raise RuntimeError("head localization counts or provenance are inconsistent")
    ranked_heads = [int(row["head_index"]) for row in recomputed["individual_head_ranking"]]
    return {
        "layer": int(layer_run["layer"]),
        "n_heads": n_heads,
        "records": records,
        "summary": recomputed,
        "candidate_pair": tuple(sorted(ranked_heads[:2])),
        "negative_control_pair": tuple(sorted(ranked_heads[-2:])),
        "selection_rule": (
            "two highest and two lowest individual heads by family-specific advantage"
        ),
        "identity": identity,
        "hashes": {
            "manifest": sha256(manifest_path),
            "summary": sha256(summary_path),
            "final": sha256(final_path),
            "records": sha256(records_path),
        },
    }


def scan_head_sets(
    model: Any,
    pairs: list[FirstLastPair],
    batch_size: int,
    layer: int,
    layer_manifest_hash: str,
    component_manifest_hash: str,
    head_manifest_hash: str,
) -> list[HeadSetRecord]:
    import torch

    if not all((layer_manifest_hash, component_manifest_hash, head_manifest_hash)):
        raise ValueError("all upstream manifest hashes are required")
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    if layer < 0 or layer >= n_layers:
        raise ValueError("selected layer is outside the model")
    head_interventions = intervention_heads(n_heads)
    z_site = f"blocks.{layer}.attn.hook_z"
    attn_site = f"blocks.{layer}.hook_attn_out"
    records: list[HeadSetRecord] = []
    with torch.inference_mode():
        for chunk in deterministic_batches(pairs, batch_size):
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
                        callbacks: list[tuple[str, tuple[int, ...], str, Callable[..., Any]]] = [
                            (
                                name,
                                included,
                                z_site,
                                _patch_head_set(_mode_value(source_cache[z_site], mode), included),
                            )
                            for name, included in head_interventions.items()
                        ]
                        callbacks.extend(
                            [
                                (
                                    "aggregate_attention_output",
                                    tuple(range(n_heads)),
                                    attn_site,
                                    _patch_final_position(
                                        _mode_value(source_cache[attn_site], mode)
                                    ),
                                ),
                                ("identity_noop", (), attn_site, _identity_final_position()),
                            ]
                        )
                        for name, included, site, callback in callbacks:
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
                                raise RuntimeError(
                                    "identity no-op score was not exactly unchanged: "
                                    + json.dumps(
                                        {
                                            "exact_equality": False,
                                            "device": str(patched_scores.device),
                                            "dtype": str(patched_scores.dtype),
                                            "hook_site": attn_site,
                                        },
                                        sort_keys=True,
                                    )
                                )
                            for row, destination in enumerate(destination_items):
                                baseline, patched = (
                                    float(destination_scores[row]),
                                    float(patched_scores[row]),
                                )
                                transfer = (
                                    0.0
                                    if name == "identity_noop"
                                    else orientation * (patched - baseline)
                                )
                                if not all(
                                    math.isfinite(value)
                                    for value in (
                                        float(mode_scores[row]),
                                        baseline,
                                        patched,
                                        transfer,
                                    )
                                ):
                                    raise RuntimeError("non-finite head-set record")
                                records.append(
                                    HeadSetRecord(
                                        family_id=destination.family_id,
                                        prompt_variant=variant,
                                        direction=direction,
                                        source_mode=mode,
                                        selected_layer=layer,
                                        intervention=name,
                                        included_heads=included,
                                        hook_site=site,
                                        source_family_id=family_ids[row],
                                        destination_family_id=destination.family_id,
                                        source_score=float(mode_scores[row]),
                                        destination_baseline=baseline,
                                        patched_score=patched,
                                        oriented_causal_transfer=transfer,
                                        layer_run_manifest_hash=layer_manifest_hash,
                                        component_run_manifest_hash=component_manifest_hash,
                                        head_run_manifest_hash=head_manifest_hash,
                                    )
                                )
    return records


def _expected_interventions(n_heads: int) -> dict[str, tuple[int, ...]]:
    return intervention_heads(n_heads) | {
        "aggregate_attention_output": tuple(range(n_heads)),
        "identity_noop": (),
    }


def validate_head_set_records(
    records: list[HeadSetRecord], n_heads: int, expected_families: set[str] | None = None
) -> list[str]:
    if not records:
        raise ValueError("head-set records are empty")
    expected = _expected_interventions(n_heads)
    families = {record.family_id for record in records}
    layers = {record.selected_layer for record in records}
    if len(layers) != 1:
        raise RuntimeError("head-set records mix selected layers")
    if expected_families is not None and families != expected_families:
        raise RuntimeError("head-set records contain missing or unexpected families")
    cells: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        if record.intervention not in expected:
            raise RuntimeError("head-set record has an unexpected intervention")
        if tuple(record.included_heads) != expected[record.intervention]:
            raise RuntimeError("head-set record has inconsistent included heads")
        if (
            record.prompt_variant not in PROMPT_VARIANTS
            or record.direction not in DIRECTIONS
            or record.source_mode not in SOURCE_MODES
        ):
            raise RuntimeError("head-set record has an invalid factorial label")
        if record.destination_family_id != record.family_id:
            raise RuntimeError("head-set destination family is inconsistent")
        key = (record.intervention, record.family_id, record.source_mode)
        cell = (record.prompt_variant, record.direction)
        if cell in cells[key]:
            raise RuntimeError("head-set records contain duplicate factorial cells")
        cells[key].add(cell)
    factorial = {(variant, direction) for variant in PROMPT_VARIANTS for direction in DIRECTIONS}
    expected_keys = {
        (intervention, family, mode)
        for intervention in expected
        for family in families
        for mode in SOURCE_MODES
    }
    if set(cells) != expected_keys or any(value != factorial for value in cells.values()):
        raise RuntimeError("head-set records are factorially incomplete")
    expected_count = len(families) * 2 * 2 * 2 * len(expected)
    if len(records) != expected_count:
        raise RuntimeError("head-set record count is inconsistent")
    return sorted(families)


def _seed(seed: int, label: str) -> int:
    return int(hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:16], 16) % (2**63 - 1)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if abs(denominator) > 1e-12 else None


def _family_profiles(
    records: list[HeadSetRecord], n_heads: int, seed: int
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], float], list[str]]:
    families = validate_head_set_records(records, n_heads)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.intervention, record.family_id, record.source_mode)].append(
            record.oriented_causal_transfer
        )
    rows: dict[str, dict[str, Any]] = {}
    profiles: dict[tuple[str, str, str], float] = {}
    for intervention in _expected_interventions(n_heads):
        matched, permuted, advantages = [], [], []
        for family in families:
            matched_mean = _mean(grouped[(intervention, family, "matched")])
            permuted_mean = _mean(grouped[(intervention, family, "permuted")])
            profiles[(intervention, family, "matched")] = matched_mean
            profiles[(intervention, family, "permuted")] = permuted_mean
            matched.append(matched_mean)
            permuted.append(permuted_mean)
            advantages.append(matched_mean - permuted_mean)
        rows[intervention] = {
            "intervention": intervention,
            "family_count": len(families),
            "matched_mean_transfer": _mean(matched),
            "permuted_mean_transfer": _mean(permuted),
            "family_specific_advantage": _mean(advantages),
            "family_specific_advantage_ci_95": _bootstrap(
                advantages, _seed(seed, intervention), BOOTSTRAP_SAMPLES
            ),
            "positive_transfer_fraction": sum(value > 0 for value in matched) / len(matched),
        }
    return rows, profiles, families


def _loss_row(
    reference: str,
    intervention: str,
    profiles: dict[tuple[str, str, str], float],
    families: list[str],
    seed: int,
) -> dict[str, Any]:
    matched_losses, advantage_losses = [], []
    for family in families:
        ref_m = profiles[(reference, family, "matched")]
        ref_p = profiles[(reference, family, "permuted")]
        value_m = profiles[(intervention, family, "matched")]
        value_p = profiles[(intervention, family, "permuted")]
        matched_losses.append(ref_m - value_m)
        advantage_losses.append((ref_m - ref_p) - (value_m - value_p))
    return {
        "intervention": intervention,
        "matched_transfer_loss": _mean(matched_losses),
        "matched_transfer_loss_ci_95": _bootstrap(
            matched_losses, _seed(seed, intervention + "|matched-loss"), BOOTSTRAP_SAMPLES
        ),
        "family_specific_advantage_loss": _mean(advantage_losses),
        "family_specific_advantage_loss_ci_95": _bootstrap(
            advantage_losses, _seed(seed, intervention + "|advantage-loss"), BOOTSTRAP_SAMPLES
        ),
        "positive_matched_transfer_loss_fraction": sum(
            value > 0 for value in matched_losses
        )
        / len(matched_losses),
        "positive_family_specific_advantage_loss_fraction": sum(
            value > 0 for value in advantage_losses
        )
        / len(advantage_losses),
    }


def _upstream_profiles(records: list[HeadRecord]) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.intervention, record.family_id, record.source_mode)].append(
            record.oriented_causal_transfer
        )
    return {key: _mean(values) for key, values in grouped.items()}


def _interaction_row(
    pair: tuple[int, int],
    profiles: dict[tuple[str, str, str], float],
    upstream: dict[tuple[str, str, str], float],
    families: list[str],
    seed: int,
) -> dict[str, Any]:
    left, right = pair
    intervention = f"pair_patch_{left}_{right}"
    matched_values, advantage_values = [], []
    for family in families:
        pair_m = profiles[(intervention, family, "matched")]
        pair_p = profiles[(intervention, family, "permuted")]
        left_m, left_p = (
            upstream[(f"head_{left}", family, "matched")],
            upstream[(f"head_{left}", family, "permuted")],
        )
        right_m, right_p = (
            upstream[(f"head_{right}", family, "matched")],
            upstream[(f"head_{right}", family, "permuted")],
        )
        matched_values.append(pair_m - left_m - right_m)
        advantage_values.append((pair_m - pair_p) - (left_m - left_p) - (right_m - right_p))
    mean_matched = _mean(matched_values)
    mean_advantage = _mean(advantage_values)
    matched_interval = _bootstrap(
        matched_values, _seed(seed, intervention + "|matched-interaction"), BOOTSTRAP_SAMPLES
    )
    advantage_interval = _bootstrap(
        advantage_values,
        _seed(seed, intervention + "|advantage-interaction"),
        BOOTSTRAP_SAMPLES,
    )

    def interval_label(interval: list[float]) -> str:
        if interval[0] > 0:
            return "super_additive_diagnostic"
        if interval[1] < 0:
            return "sub_additive_diagnostic"
        return "no_clear_interaction"

    return {
        "pair": [left, right],
        "intervention": intervention,
        "matched_mean_interaction": mean_matched,
        "matched_interaction_ci_95": matched_interval,
        "matched_positive_interaction_fraction": sum(value > 0 for value in matched_values)
        / len(matched_values),
        "family_specific_advantage_mean_interaction": mean_advantage,
        "family_specific_advantage_interaction_ci_95": advantage_interval,
        "family_specific_advantage_positive_interaction_fraction": sum(
            value > 0 for value in advantage_values
        )
        / len(advantage_values),
        "matched_interaction_interpretation": interval_label(matched_interval),
        "family_specific_advantage_interaction_interpretation": interval_label(
            advantage_interval
        ),
    }


def _rank(rows: list[dict[str, Any]], key: str, pair_key: str = "pair") -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (-float(row[key]), tuple(int(value) for value in row[pair_key])),
    )
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    return ranked


def _specificity(
    ranking: list[dict[str, Any]],
    key: str,
    candidate: tuple[int, int],
    negative: tuple[int, int],
) -> dict[str, Any]:
    candidate_row = next(row for row in ranking if tuple(row["pair"]) == candidate)
    negative_row = next(row for row in ranking if tuple(row["pair"]) == negative)
    noncandidate = [float(row[key]) for row in ranking if tuple(row["pair"]) != candidate]
    value = float(candidate_row[key])
    if not noncandidate:
        return {
            "metric": key,
            "candidate_rank": candidate_row["rank"],
            "candidate_empirical_percentile": None,
            "percentile_convention": (
                "fraction of noncandidate pairs with metric less than or equal to candidate"
            ),
            "candidate_difference_from_median_noncandidate": None,
            "candidate_difference_from_negative_control": value
            - float(negative_row[key]),
            "pairs_exceeding_candidate": 0,
            "comparison_status": "undefined_no_noncandidate_pairs",
            "comparison_scope": "empirical_discovery_population_not_held_out_null",
        }
    ordered = sorted(noncandidate)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "metric": key,
        "candidate_rank": candidate_row["rank"],
        "candidate_empirical_percentile": 100
        * sum(noncandidate_value <= value for noncandidate_value in noncandidate)
        / len(noncandidate),
        "percentile_convention": (
            "fraction of noncandidate pairs with metric less than or equal to candidate"
        ),
        "candidate_difference_from_median_noncandidate": value - median,
        "candidate_difference_from_negative_control": value - float(negative_row[key]),
        "pairs_exceeding_candidate": sum(float(row[key]) > value for row in ranking),
        "comparison_scope": "empirical_discovery_population_not_held_out_null",
    }


def summarize_head_sets(
    records: list[HeadSetRecord],
    n_heads: int,
    head_run: dict[str, Any],
    component_run: dict[str, Any],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    rows, profiles, families = _family_profiles(records, n_heads, seed)
    pairs = unordered_head_pairs(n_heads)
    all_heads = rows["all_heads_z_patch"]
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
    pair_rows: list[dict[str, Any]] = []
    for left, right in pairs:
        row = rows[f"pair_patch_{left}_{right}"] | {"pair": [left, right]}
        row["matched_transfer_recovery_fraction"] = _ratio(
            float(row["matched_mean_transfer"]), float(all_heads["matched_mean_transfer"])
        )
        row["family_specific_advantage_recovery_fraction"] = _ratio(
            float(row["family_specific_advantage"]),
            float(all_heads["family_specific_advantage"]),
        )
        pair_rows.append(row)
    leave_one = [
        _loss_row("all_heads_z_patch", f"leave_one_out_{head}", profiles, families, seed)
        | {"head_index": head}
        for head in range(n_heads)
    ]
    leave_pair = [
        _loss_row(
            "all_heads_z_patch",
            f"leave_pair_out_{left}_{right}",
            profiles,
            families,
            seed,
        )
        | {"pair": [left, right]}
        for left, right in pairs
    ]
    upstream = _upstream_profiles(head_run["records"])
    interactions = [_interaction_row(pair, profiles, upstream, families, seed) for pair in pairs]
    pair_ranking = _rank(pair_rows, "family_specific_advantage")
    leave_pair_ranking = _rank(leave_pair, "family_specific_advantage_loss")
    recovery_rankable = [
        row for row in pair_rows if row["family_specific_advantage_recovery_fraction"] is not None
    ]
    recovery_ranking = _rank(recovery_rankable, "family_specific_advantage_recovery_fraction")
    leave_one.sort(
        key=lambda row: (-float(row["family_specific_advantage_loss"]), int(row["head_index"]))
    )
    for index, row in enumerate(leave_one, 1):
        row["rank"] = index
    interactions = _rank(interactions, "family_specific_advantage_mean_interaction")
    candidate = tuple(head_run["candidate_pair"])
    negative = tuple(head_run["negative_control_pair"])
    candidate_row = next(row for row in pair_ranking if tuple(row["pair"]) == candidate)
    candidate_leave = next(row for row in leave_pair_ranking if tuple(row["pair"]) == candidate)
    candidate_interaction = next(row for row in interactions if tuple(row["pair"]) == candidate)
    reproducibility = {
        "all_heads_z_minus_aggregate_matched": float(all_heads["matched_mean_transfer"])
        - float(aggregate["matched_mean_transfer"]),
        "all_heads_z_minus_aggregate_permuted": float(all_heads["permuted_mean_transfer"])
        - float(aggregate["permuted_mean_transfer"]),
        "all_heads_z_minus_aggregate_family_specific_advantage": float(
            all_heads["family_specific_advantage"]
        )
        - float(aggregate["family_specific_advantage"]),
        "aggregate_minus_prior_component_matched": float(aggregate["matched_mean_transfer"])
        - float(component_run["attention"]["matched_mean_transfer"]),
        "aggregate_minus_prior_component_permuted": float(aggregate["permuted_mean_transfer"])
        - float(component_run["attention"]["permuted_mean_transfer"]),
        "aggregate_minus_prior_component_family_specific_advantage": float(
            aggregate["family_specific_advantage"]
        )
        - float(component_run["attention"]["family_specific_advantage"]),
        "upstream_individual_heads_recomputed_exactly": True,
        "identity_noop_exact_zero": True,
    }
    return {
        "interpretation_scope": "exploratory_discovery_only",
        "protocol_version": PROTOCOL_VERSION,
        "selected_layer": records[0].selected_layer,
        "n_heads": n_heads,
        "family_count": len(families),
        "record_count": len(records),
        "selection_rule": head_run["selection_rule"],
        "candidate_pair": list(candidate),
        "negative_control_pair": list(negative),
        "all_heads_z_reference": all_heads,
        "aggregate_attention_reference": aggregate,
        "identity_noop_reference": noop,
        "pair_patch_ranking": pair_ranking,
        "leave_one_out_ranking": leave_one,
        "leave_pair_out_ranking": leave_pair_ranking,
        "transfer_recovery_ranking": recovery_ranking,
        "pair_interaction_ranking": interactions,
        "candidate_pair_diagnostics": {
            "pair_patch": candidate_row,
            "leave_pair_out": candidate_leave,
            "interaction": candidate_interaction,
            "interpretation": (
                "transfer recovery and leave-out diagnostics, not full behavioral "
                "sufficiency or necessity"
            ),
        },
        "specificity_comparison": {
            "pair_family_specific_advantage": _specificity(
                pair_ranking, "family_specific_advantage", candidate, negative
            ),
            "pair_leave_out_family_specific_loss": _specificity(
                leave_pair_ranking,
                "family_specific_advantage_loss",
                candidate,
                negative,
            ),
            "transfer_recovery_fraction": (
                _specificity(
                    recovery_ranking,
                    "family_specific_advantage_recovery_fraction",
                    candidate,
                    negative,
                )
                if len(recovery_ranking) == len(pairs)
                else {"status": "undefined_due_to_unstable_denominator"}
            ),
        },
        "reproducibility_diagnostics": reproducibility,
    }


def enforce_reproducibility(
    summary: dict[str, Any],
    component_run: dict[str, Any],
    *,
    device: str,
    dtype: str,
    selected_layer: int,
    tolerance: float = REPRODUCIBILITY_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Require deterministic reference interventions to reproduce upstream values."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("reproducibility tolerance must be finite and non-negative")
    all_heads = summary["all_heads_z_reference"]
    aggregate = summary["aggregate_attention_reference"]
    prior = component_run["attention"]
    metrics = (
        "matched_mean_transfer",
        "permuted_mean_transfer",
        "family_specific_advantage",
    )
    comparisons: list[dict[str, Any]] = []
    for metric in metrics:
        comparisons.append(
            {
                "comparison": "all_heads_z_patch_vs_aggregate_attention_output",
                "metric": metric,
                "expected": float(aggregate[metric]),
                "observed": float(all_heads[metric]),
            }
        )
        comparisons.append(
            {
                "comparison": "aggregate_attention_output_vs_prior_component_attention",
                "metric": metric,
                "expected": float(prior[metric]),
                "observed": float(aggregate[metric]),
            }
        )
    failures: list[dict[str, Any]] = []
    for comparison in comparisons:
        comparison["difference"] = comparison["observed"] - comparison["expected"]
        comparison["absolute_difference"] = abs(float(comparison["difference"]))
        comparison["tolerance"] = tolerance
        comparison["passed"] = comparison["absolute_difference"] <= tolerance
        if not comparison["passed"]:
            failures.append(dict(comparison))
    result = {
        "status": "passed" if not failures else "failed",
        "absolute_tolerance": tolerance,
        "device": device,
        "dtype": dtype,
        "selected_layer": selected_layer,
        "protocol_versions": {
            "head_set": PROTOCOL_VERSION,
            "head_localization": HEAD_PROTOCOL_VERSION,
            "component_localization": COMPONENT_PROTOCOL_VERSION,
        },
        "comparisons": comparisons,
        "failures": failures,
    }
    if failures:
        raise ReproducibilityError(result)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_records(path: Path, records: list[HeadSetRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Exploratory discovery-only attention-head-set analysis",
                "",
                f"- Selected layer: `{summary['selected_layer']}`",
                f"- Candidate pair: `{summary['candidate_pair']}`",
                f"- Negative-control pair: `{summary['negative_control_pair']}`",
                "- Scope: transfer recovery, leave-out, interaction, and empirical "
                "specificity diagnostics.",
                "- This is not confirmatory validation and does not establish a circuit.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_incomplete(
    root: Path, stage: str, upstream: dict[str, Any], exc: BaseException | None = None
) -> None:
    names = (
        "head_set_records.jsonl",
        "head_set_summary.json",
        "head_set_analysis.md",
        "head_set_final_status.json",
    )
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "complete": False,
        "software_success": False,
        "failure_stage": stage,
        "upstream_provenance": upstream,
        "artifact_hashes": {name: sha256(root / name) for name in names if (root / name).is_file()},
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
    if exc is not None:
        manifest |= {"exception_type": type(exc).__name__, "exception_message": str(exc)}
        if isinstance(exc, ReproducibilityError):
            manifest["reproducibility_failure"] = exc.details
    _write_json(root / "head_set_manifest.json", manifest)


def run(
    args: argparse.Namespace,
    adapter_factory: Callable[[str, str, str], Any] = PythiaAdapter,
) -> dict[str, Any]:
    # Before trusted discovery provenance resolves seed/revision, a normal
    # run-scoped incomplete manifest cannot be placed safely or authoritatively.
    if args.batch_size < 2:
        raise ValueError("batch size must be at least 2 for family permutation")
    discovery = _verify_frozen_discovery(args.discovery_root)
    layer_run = verify_layer_run(args.layer_root, discovery)
    component_run = verify_component_run(args.component_root, discovery, layer_run)
    head_run = verify_head_run(args.head_root, discovery, layer_run, component_run)
    seed, revision = int(discovery["seed"]), str(discovery["requested_revision"])
    root = args.output / f"seed-{seed}-{revision.replace('/', '_')}"
    if root.exists():
        if not args.force:
            raise RuntimeError("head-set output exists; use --force")
        if root.resolve().parent != args.output.resolve():
            raise RuntimeError("refusing unsafe force target")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    upstream = {
        "discovery_artifact_hashes": discovery["artifact_hashes"],
        "layer_run_hashes": layer_run["hashes"],
        "component_run_hashes": component_run["hashes"],
        "head_run_hashes": head_run["hashes"],
        "model_identity": head_run["identity"],
        "selected_layer": head_run["layer"],
        "n_heads": head_run["n_heads"],
        "selection_rule": head_run["selection_rule"],
        "candidate_pair": list(head_run["candidate_pair"]),
        "negative_control_pair": list(head_run["negative_control_pair"]),
    }
    incomplete = {
        "status": "EXPLORATORY_ATTENTION_HEAD_SET_ANALYSIS_INCOMPLETE",
        "software_success": False,
        "interpretation_scope": "exploratory_discovery_only",
        "selected_layer": head_run["layer"],
        "candidate_pair": list(head_run["candidate_pair"]),
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
    final_path = root / "head_set_final_status.json"
    _write_json(final_path, incomplete)
    _write_incomplete(root, "preflight", upstream)
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
        if int(adapter.model.cfg.n_heads) != head_run["n_heads"]:
            raise RuntimeError("loaded model head count disagrees with provenance")
        if (
            bool(getattr(adapter.model.cfg, "parallel_attn_mlp", False))
            != component_run["parallel_attn_mlp"]
        ):
            raise RuntimeError("loaded model architecture disagrees with provenance")
        stage = "scanning"
        records = scan_head_sets(
            adapter.model,
            pairs,
            args.batch_size,
            head_run["layer"],
            layer_run["hashes"]["manifest"],
            component_run["hashes"]["manifest"],
            head_run["hashes"]["manifest"],
        )
        records_path = root / "head_set_records.jsonl"
        _write_records(records_path, records)
        stage = "record_validation"
        validate_head_set_records(records, head_run["n_heads"], {pair.family_id for pair in pairs})
        if len(pairs) != 120:
            raise RuntimeError("head-set analysis requires exactly 120 matched families")
        stage = "summarization"
        summary = summarize_head_sets(
            records, head_run["n_heads"], head_run, component_run, seed=seed
        )
        stage = "reproducibility_enforcement"
        summary["reproducibility_enforcement"] = enforce_reproducibility(
            summary,
            component_run,
            device=str(identity.get("resolved_device", args.device)),
            dtype=str(identity.get("dtype", "unknown")),
            selected_layer=head_run["layer"],
        )
        summary["upstream_provenance"] = upstream
        summary_path = root / "head_set_summary.json"
        report_path = root / "head_set_analysis.md"
        _write_json(summary_path, summary)
        _write_report(report_path, summary)
        stage = "finalization"
        final = {
            "status": STATUS,
            "software_success": True,
            "interpretation_scope": "exploratory_discovery_only",
            "selected_layer": head_run["layer"],
            "dynamically_detected_head_count": head_run["n_heads"],
            "matched_family_count": len(pairs),
            "record_count": len(records),
            "candidate_pair": list(head_run["candidate_pair"]),
            "held_out_validation_reused": False,
            "held_out_test_opened": False,
            "activation_patching_performed": True,
            "scientific_confirmation": False,
            "circuit_found": False,
        }
        _write_json(final_path, final)
        artifacts = (records_path, summary_path, report_path, final_path)
        manifest = identity | {
            "protocol_version": PROTOCOL_VERSION,
            "complete": True,
            "git_commit": _git_commit(),
            "created_at": datetime.now(UTC).isoformat(),
            "selected_layer": head_run["layer"],
            "n_heads": head_run["n_heads"],
            "candidate_pair": list(head_run["candidate_pair"]),
            "negative_control_pair": list(head_run["negative_control_pair"]),
            "selection_rule": head_run["selection_rule"],
            "discovery_artifact_hashes": discovery["artifact_hashes"],
            "layer_run_hashes": layer_run["hashes"],
            "component_run_hashes": component_run["hashes"],
            "head_run_hashes": head_run["hashes"],
            "held_out_validation_reused": False,
            "held_out_test_opened": False,
            "artifact_hashes": {path.name: sha256(path) for path in artifacts},
        }
        _write_json(root / "head_set_manifest.json", manifest)
        print(f"Head-set analysis: {STATUS}")
        print(f"Artifacts: {root}")
        return final
    except Exception as exc:
        incomplete |= {
            "failure_stage": stage,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, ReproducibilityError):
            incomplete["reproducibility_failure"] = exc.details
        _write_json(final_path, incomplete)
        _write_incomplete(root, stage, upstream, exc)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--discovery-root", type=Path, default=Path("artifacts/position_study/seed-42-main")
    )
    parser.add_argument(
        "--layer-root", type=Path, default=Path("artifacts/position_localization/seed-42-main")
    )
    parser.add_argument(
        "--component-root",
        type=Path,
        default=Path("artifacts/position_component_localization/seed-42-main"),
    )
    parser.add_argument(
        "--head-root",
        type=Path,
        default=Path("artifacts/position_head_localization/seed-42-main"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/position_head_set_analysis"))
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
