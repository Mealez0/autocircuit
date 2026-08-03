from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from autocircuit.datasets.associative_recall import ExamplePair
from autocircuit.position_localization import (
    FirstLastPair,
    LayerScanRecord,
    build_first_last_pairs,
    build_parser,
    residual_stream_sites,
    run_residual_stream_scan,
    summarize_layer_scan,
)
from autocircuit.position_study import STUDY_VERSION


def make_example(family: int, position: str) -> ExamplePair:
    family_id = f"family-{family:03d}"
    return ExamplePair(
        example_id=f"{family_id}-{position}",
        family_id=family_id,
        split="discovery",
        clean_prompt=f"{family_id}|{position}|clean",
        corrupt_prompt=f"{family_id}|{position}|corrupt",
        target_text=" target",
        distractor_text=" distractor",
        target_token_id=1,
        distractor_token_id=2,
        changed_factor="queried_value_swap",
        seed=42,
        template_id="chooses-position-study-v2",
        metadata={
            "generator_version": STUDY_VERSION,
            "normalized_query_position": position,
            "prompt_token_length": 4,
            "matched_family_id": family_id,
        },
    )


def make_population() -> list[ExamplePair]:
    return [
        make_example(family, position)
        for family in range(120)
        for position in ("first", "interior", "last")
    ]


def test_residual_stream_sites_are_ordered_layer_boundaries() -> None:
    assert residual_stream_sites(2) == [
        "blocks.0.hook_resid_pre",
        "blocks.1.hook_resid_pre",
        "blocks.1.hook_resid_post",
    ]
    with pytest.raises(ValueError, match="at least one"):
        residual_stream_sites(0)


def test_discovery_population_is_paired_and_non_discovery_is_rejected() -> None:
    population = make_population()
    pairs = build_first_last_pairs(population)
    assert len(pairs) == 120
    assert pairs[0].first.metadata["normalized_query_position"] == "first"
    assert pairs[0].last.metadata["normalized_query_position"] == "last"

    changed = population.copy()
    changed[0] = replace(changed[0], split="validation")
    with pytest.raises(ValueError, match="only the discovery"):
        build_first_last_pairs(changed)


class FakeHookModel:
    cfg = SimpleNamespace(n_layers=2)

    def _parts(self, prompt: str) -> tuple[str, str, str]:
        family, position, variant = prompt.split("|")
        return family, position, variant

    def _baseline_score(self, prompt: str) -> float:
        _, position, _ = self._parts(prompt)
        return 4.0 if position == "first" else 1.0

    def _site_score(self, prompt: str, site: str) -> float:
        _, position, _ = self._parts(prompt)
        values = {
            "blocks.0.hook_resid_pre": {"first": 1.0, "last": 1.0},
            "blocks.1.hook_resid_pre": {"first": 3.0, "last": 1.0},
            "blocks.1.hook_resid_post": {"first": 4.0, "last": 1.0},
        }
        return values[site][position]

    def _logits(self, prompts: list[str], scores: list[float]) -> torch.Tensor:
        logits = torch.zeros((len(prompts), 4, 3), dtype=torch.float32)
        for row, (prompt, score) in enumerate(zip(prompts, scores, strict=True)):
            _, _, variant = self._parts(prompt)
            raw = score if variant == "clean" else -score
            logits[row, -1, 1] = raw / 2
            logits[row, -1, 2] = -raw / 2
        return logits

    def __call__(self, prompts: list[str], return_type: str) -> torch.Tensor:
        assert return_type == "logits"
        return self._logits(prompts, [self._baseline_score(prompt) for prompt in prompts])

    def run_with_cache(
        self,
        prompts: list[str],
        *,
        return_type: str,
        names_filter: list[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self(prompts, return_type)
        cache: dict[str, torch.Tensor] = {}
        for site in names_filter:
            activation = torch.zeros((len(prompts), 4, 1), dtype=torch.float32)
            activation[:, -1, 0] = torch.tensor(
                [self._site_score(prompt, site) for prompt in prompts]
            )
            cache[site] = activation
        return logits, cache

    def run_with_hooks(
        self,
        prompts: list[str],
        *,
        return_type: str,
        fwd_hooks: list[tuple[str, Any]],
    ) -> torch.Tensor:
        assert return_type == "logits"
        site, hook = fwd_hooks[0]
        destination = torch.zeros((len(prompts), 4, 1), dtype=torch.float32)
        destination[:, -1, 0] = torch.tensor(
            [self._site_score(prompt, site) for prompt in prompts]
        )
        patched = hook(destination, None)
        scores = [float(value) for value in patched[:, -1, 0]]
        return self._logits(prompts, scores)


def tiny_pairs() -> list[FirstLastPair]:
    return [
        FirstLastPair(
            family_id=f"family-{family}",
            first=make_example(family, "first"),
            last=make_example(family, "last"),
        )
        for family in range(2)
    ]


def test_bidirectional_scan_localizes_the_late_residual_boundary() -> None:
    records = run_residual_stream_scan(FakeHookModel(), tiny_pairs(), batch_size=2)
    assert len(records) == 48

    matched_clean_forward = [
        record
        for record in records
        if record.prompt_variant == "clean"
        and record.direction == "first_to_last"
        and record.source_mode == "matched"
    ]
    by_site = {record.site: record for record in matched_clean_forward if record.family_id == "family-0"}
    assert by_site["blocks.0.hook_resid_pre"].causal_transfer == pytest.approx(0.0)
    assert by_site["blocks.1.hook_resid_pre"].causal_transfer == pytest.approx(2.0)
    assert by_site["blocks.1.hook_resid_post"].causal_transfer == pytest.approx(3.0)
    assert by_site["blocks.1.hook_resid_post"].normalized_transfer == pytest.approx(1.0)

    summary = summarize_layer_scan(records, bootstrap_samples=100, seed=42)
    assert summary["site_ranking"][0]["site"] == "blocks.1.hook_resid_post"
    assert summary["record_count"] == 48


def test_summary_uses_family_level_transfer_and_cli_has_no_held_out_controls() -> None:
    records = [
        LayerScanRecord(
            family_id=f"family-{index}",
            prompt_variant="clean",
            direction="first_to_last",
            source_mode="matched",
            site="blocks.0.hook_resid_pre",
            site_index=0,
            source_family_id=f"family-{index}",
            destination_family_id=f"family-{index}",
            source_task_score=3.0,
            destination_task_score=1.0,
            patched_task_score=2.0,
            available_gap=2.0,
            causal_transfer=1.0,
            normalized_transfer=0.5,
        )
        for index in range(4)
    ]
    summary = summarize_layer_scan(records, bootstrap_samples=100, seed=42)
    row = summary["group_summaries"][0]
    assert row["family_count"] == 4
    assert row["mean_causal_transfer"] == pytest.approx(1.0)
    assert row["normalized_transfer"] == pytest.approx(0.5)

    parser = build_parser()
    parsed = parser.parse_args(["--device", "cpu", "--batch-size", "2"])
    assert parsed.device == "cpu"
    assert not hasattr(parsed, "validation_root")
    assert not hasattr(parsed, "test_root")
