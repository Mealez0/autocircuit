"""Deterministic discovery-only behavioral diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from autocircuit.baseline import ExampleResult, successful_difference

GROUPS: dict[str, Callable[[ExampleResult], str]] = {
    "template_id": lambda r: r.template_id,
    "fact_count": lambda r: str(r.fact_count),
    "normalized_query_position": lambda r: r.normalized_query_position,
    "query_fact_index": lambda r: str(r.query_fact_index),
    "prompt_token_length": lambda r: str(r.prompt_token_length),
    "target_text": lambda r: r.target_text,
    "distractor_text": lambda r: r.distractor_text,
    "target_distractor_pair": lambda r: f"{r.target_text} | {r.distractor_text}",
    "entity": lambda r: r.query_entity,
    "correctness_quadrant": lambda r: correctness_quadrant(r.clean_correct, r.corrupt_correct),
}


def correctness_quadrant(clean: bool | None, corrupt: bool | None) -> str:
    if clean is None or corrupt is None:
        return "processing failure"
    clean_label = "correct" if clean else "incorrect"
    corrupt_label = "correct" if corrupt else "incorrect"
    return f"clean {clean_label}, corrupt {corrupt_label}"


def _bootstrap(values: list[float], seed: int, samples: int) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    return sorted(sum(rng.choice(values) for _ in range(size)) / size for _ in range(samples))


def _ci(values: list[float], seed: int, samples: int) -> list[float]:
    estimates = _bootstrap(values, seed, samples)
    return [estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]]


def group_metrics(
    records: Sequence[ExampleResult],
    *,
    bootstrap_seed: int = 1729,
    bootstrap_samples: int = 1000,
    minimum_count: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    valid = [r for r in records if r.processing_status == "ok"]
    output: dict[str, list[dict[str, Any]]] = {}
    for dimension, key_function in GROUPS.items():
        buckets: dict[str, list[ExampleResult]] = {}
        for record in valid:
            buckets.setdefault(key_function(record), []).append(record)
        output[dimension] = []
        for key in sorted(buckets):
            bucket = buckets[key]
            clean = [successful_difference(r.clean_logit_difference) for r in bucket]
            corrupt = [successful_difference(r.corrupt_logit_difference) for r in bucket]
            accuracy = [float(bool(r.clean_correct)) for r in bucket]
            seed = bootstrap_seed + int(
                hashlib.sha256(f"{dimension}:{key}".encode()).hexdigest()[:8], 16
            )
            output[dimension].append(
                {
                    "value": key,
                    "count": len(bucket),
                    "clean_accuracy": statistics.fmean(accuracy),
                    "corrupt_accuracy": statistics.fmean(
                        float(bool(r.corrupt_correct)) for r in bucket
                    ),
                    "mean_clean_ld": statistics.fmean(clean),
                    "median_clean_ld": statistics.median(clean),
                    "mean_corrupt_ld": statistics.fmean(corrupt),
                    "median_corrupt_ld": statistics.median(corrupt),
                    "mean_clean_corrupt_contrast": statistics.fmean(
                        a - b for a, b in zip(clean, corrupt, strict=True)
                    ),
                    "clean_ld_standard_deviation": statistics.pstdev(clean),
                    "clean_ld_mean_bootstrap_95_ci": _ci(clean, seed, bootstrap_samples),
                    "clean_accuracy_bootstrap_95_ci": _ci(accuracy, seed + 1, bootstrap_samples),
                    "bootstrap_seed": seed,
                    "underpowered": len(bucket) < minimum_count,
                }
            )
    return output


def build_diagnostics(records: Sequence[ExampleResult], **kwargs: Any) -> dict[str, Any]:
    valid = [r for r in records if r.processing_status == "ok"]

    def serialized(rows: Sequence[ExampleResult]) -> list[dict[str, Any]]:
        return [asdict(r) for r in rows[:25]]

    by_clean = sorted(
        valid, key=lambda r: (successful_difference(r.clean_logit_difference), r.example_id)
    )
    by_span = sorted(
        valid,
        key=lambda r: (successful_difference(r.clean_corrupt_recovery_span), r.example_id),
    )
    return {
        "split": "discovery",
        "groups": group_metrics(records, **kwargs),
        "examples": {
            "lowest_clean_ld": serialized(by_clean),
            "highest_clean_ld": serialized(list(reversed(by_clean))),
            "smallest_contrast": serialized(by_span),
            "largest_contrast": serialized(list(reversed(by_span))),
        },
    }


def read_results(path: Path) -> list[ExampleResult]:
    return [
        ExampleResult(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
    ]


def write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    lines = [
        "# Discovery diagnostics",
        "",
        "> Discovery data only; held-out splits were not opened.",
        "",
    ]
    for dimension, groups in report["groups"].items():
        lines += [
            f"## {dimension}",
            "",
            "| Value | n | Clean accuracy | Mean clean LD | Power |",
            "|---|---:|---:|---:|---|",
        ]
        lines += [
            "| {value} | {count} | {accuracy:.4f} | {ld:.4f} | {power} |".format(
                value=g["value"],
                count=g["count"],
                accuracy=g["clean_accuracy"],
                ld=g["mean_clean_ld"],
                power="underpowered" if g["underpowered"] else "adequate",
            )
            for g in groups
        ]
        lines.append("")
    markdown_path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-count", type=int, default=8)
    args = parser.parse_args(argv)
    records = read_results(args.examples)
    if any(record.split != "discovery" for record in records):
        parser.error("diagnostics accepts discovery records only")
    args.output.mkdir(parents=True, exist_ok=True)
    write_reports(
        build_diagnostics(records, minimum_count=args.minimum_count),
        args.output / "diagnostics.json",
        args.output / "diagnostics.md",
    )


if __name__ == "__main__":
    main()
