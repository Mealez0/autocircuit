"""Preregistered, compact dataset-v2 candidate registry and selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, cast

from autocircuit.baseline import BaselineMetrics, baseline_passes
from autocircuit.datasets.associative_recall import V2_GENERATOR_VERSION, GenerationParameters

SELECTION_RULE = (
    "eligible (accuracy>=0.80 and mean clean LD>=1.0), then descending accuracy, "
    "mean clean LD, contrast, then lexical candidate_id"
)
SEED_NAMESPACE = "autocircuit:dataset-v2:candidate-evaluation:v1"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    rationale: str
    parameters: GenerationParameters
    generator_version: str = V2_GENERATOR_VERSION


def candidate_registry(strongest_template: str) -> tuple[Candidate, ...]:
    fixed = GenerationParameters(allowed_templates=(strongest_template,), relation_mode="fixed")
    candidates = (
        Candidate("v1-control", "Existing v1 population control.", GenerationParameters()),
        Candidate(
            "discovery-best-template",
            "Discovery-selected template; ties resolved lexically.",
            fixed,
        ),
        Candidate(
            "short-all",
            "Reduce interference with 3–4 facts.",
            GenerationParameters(maximum_fact_count=4),
        ),
        Candidate(
            "short-best-template",
            "3–4 facts and discovery-selected template.",
            GenerationParameters(
                allowed_templates=(strongest_template,),
                maximum_fact_count=4,
                relation_mode="fixed",
            ),
        ),
        Candidate(
            "query-last",
            "Query only the last presented fact.",
            GenerationParameters(allowed_query_positions=("last",)),
        ),
        Candidate(
            "short-query-last",
            "3–4 facts and last-position query.",
            GenerationParameters(maximum_fact_count=4, allowed_query_positions=("last",)),
        ),
        Candidate(
            "blank-line-separator",
            "Explicit stable blank-line separator.",
            GenerationParameters(separator="blank_line"),
        ),
    )
    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))


def strongest_template(diagnostics: dict[str, Any]) -> str:
    groups = diagnostics["groups"]["template_id"]
    winner = sorted(groups, key=lambda g: (-g["clean_accuracy"], -g["mean_clean_ld"], g["value"]))[
        0
    ]
    return str(winner["value"]).removesuffix("-v1")


def candidate_is_eligible(metrics: BaselineMetrics, generated_count: int) -> bool:
    return (
        baseline_passes(metrics)
        and metrics.failed_count == 0
        and metrics.example_count + metrics.failed_count == generated_count
    )


def select_candidate(
    results: dict[str, BaselineMetrics], generated_counts: dict[str, int] | None = None
) -> str | None:
    eligible = [
        (candidate_id, metrics)
        for candidate_id, metrics in results.items()
        if candidate_is_eligible(
            metrics, generated_counts[candidate_id] if generated_counts else metrics.example_count
        )
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            -item[1].clean_accuracy,
            -item[1].clean_mean_logit_difference,
            -item[1].clean_corrupt_contrast,
            item[0],
        ),
    )[0]


def frozen_config(
    candidate: Candidate,
    metrics: BaselineMetrics,
    model: str,
    tokenizer: str,
    requested_revision: str,
    resolved_revision: str | None,
    base_seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "discovery_selected_requires_held_out_validation",
        "selected_candidate_id": candidate.candidate_id,
        "parameters": asdict(candidate.parameters),
        "generator_version": candidate.generator_version,
        "discovery_selection_metrics": asdict(metrics),
        "seed_namespace": SEED_NAMESPACE,
        "base_seed": base_seed,
        "model": model,
        "tokenizer": tokenizer,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "frozen_thresholds": {"clean_accuracy": 0.80, "clean_mean_logit_difference": 1.0},
        "selection_rule": SELECTION_RULE,
    }


def frozen_toml(value: dict[str, Any]) -> bytes:
    """Serialize the declarative v2 contract without timestamps or machine paths."""

    def quote(item: object) -> str:
        return json.dumps(item, ensure_ascii=False)

    lines = [
        f"schema_version = {value['schema_version']}",
        f"status = {quote(value['status'])}",
        f"selected_candidate_id = {quote(value['selected_candidate_id'])}",
        f"generator_version = {quote(value['generator_version'])}",
        f"base_seed = {value['base_seed']}",
        f"seed_namespace = {quote(value['seed_namespace'])}",
        f"model = {quote(value['model'])}",
        f"tokenizer = {quote(value['tokenizer'])}",
        f"requested_revision = {quote(value['requested_revision'])}",
        f"resolved_revision = {quote(value['resolved_revision'] or '')}",
        f"selection_rule = {quote(value['selection_rule'])}",
        "",
        "[generation]",
    ]
    parameters = cast(dict[str, object], value["parameters"])
    for key in sorted(parameters):
        item = parameters[key]
        rendered = (
            "[" + ", ".join(quote(part) for part in item) + "]"
            if isinstance(item, list | tuple)
            else quote(item)
            if isinstance(item, str)
            else str(item)
        )
        lines.append(f"{key} = {rendered}")
    for section in ("frozen_thresholds", "discovery_selection_metrics"):
        lines += ["", f"[{section}]"]
        section_values = cast(dict[str, object], value[section])
        for key, item in sorted(section_values.items()):
            lines.append(f"{key} = {str(item).lower() if isinstance(item, bool) else item}")
    return ("\n".join(lines) + "\n").encode("utf-8")
