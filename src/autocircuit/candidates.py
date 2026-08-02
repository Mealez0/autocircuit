"""Preregistered, compact dataset-v2 candidate registry and selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from autocircuit.baseline import BaselineMetrics, baseline_passes
from autocircuit.datasets.associative_recall import GENERATOR_VERSION, GenerationParameters

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
    generator_version: str = GENERATOR_VERSION


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


def select_candidate(results: dict[str, BaselineMetrics]) -> str | None:
    eligible = [
        (candidate_id, metrics)
        for candidate_id, metrics in results.items()
        if baseline_passes(metrics)
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
    candidate: Candidate, metrics: BaselineMetrics, model: str, revision: str
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "discovery_selected_requires_held_out_validation",
        "selected_candidate_id": candidate.candidate_id,
        "parameters": asdict(candidate.parameters),
        "generator_version": candidate.generator_version,
        "discovery_selection_metrics": asdict(metrics),
        "seed_namespace": SEED_NAMESPACE,
        "model": model,
        "revision": revision,
        "selection_rule": SELECTION_RULE,
    }
