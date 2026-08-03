from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from autocircuit.datasets.associative_recall import ExamplePair
from autocircuit.position_component_localization import (
    ComponentRecord,
    build_parser,
    intervention_sites,
    scan_components,
    select_layer,
    summarize,
)
from autocircuit.position_localization import FirstLastPair
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


def tiny_pairs() -> list[FirstLastPair]:
    return [
        FirstLastPair(
            family_id=f"family-{family:03d}",
            first=make_example(family, "first"),
            last=make_example(family, "last"),
        )
        for family in range(2)
    ]


def test_intervention_sites_and_layer_selection() -> None:
    assert intervention_sites(5)["attention_plus_mlp"] == (
        "blocks.5.hook_attn_out",
        "blocks.5.hook_mlp_out",
    )
    with pytest.raises(ValueError, match="non-negative"):
        intervention_sites(-1)
    layer, top = select_layer(
        {
            "site_ranking": [
                {
                    "site": "blocks.5.hook_resid_post",
                    "family_specific_transfer_advantage": 0.6,
                }
            ]
        }
    )
    assert layer == 5
    assert top["site"] == "blocks.5.hook_resid_post"


class FakeModel:
    cfg = SimpleNamespace(n_layers=6, parallel_attn_mlp=True)

    def _parts(self, prompt: str) -> tuple[str, str, str]:
        return tuple(prompt.split("|"))  # type: ignore[return-value]

    def _state(self, prompt: str) -> dict[str, float]:
        _, position, _ = self._parts(prompt)
        if position == "first":
            return {"pre": 3.0, "attn": 1.5, "mlp": 0.5, "post": 5.0}
        return {"pre": 1.0, "attn": 0.5, "mlp": 0.2, "post": 1.0}

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
        return self._logits(prompts, [self._state(prompt)["post"] for prompt in prompts])

    def run_with_cache(
        self,
        prompts: list[str],
        *,
        return_type: str,
        names_filter: list[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cache: dict[str, torch.Tensor] = {}
        for site in names_filter:
            key = (
                "pre"
                if site.endswith("hook_resid_pre")
                else "attn"
                if site.endswith("hook_attn_out")
                else "mlp"
                if site.endswith("hook_mlp_out")
                else "post"
            )
            value = torch.zeros((len(prompts), 4, 1), dtype=torch.float32)
            value[:, -1, 0] = torch.tensor([self._state(prompt)[key] for prompt in prompts])
            cache[site] = value
        return self(prompts, return_type), cache

    def run_with_hooks(
        self,
        prompts: list[str],
        *,
        return_type: str,
        fwd_hooks: list[tuple[str, Any]],
    ) -> torch.Tensor:
        assert return_type == "logits"
        states = [self._state(prompt) for prompt in prompts]
        names = {site for site, _ in fwd_hooks}
        for site, hook in fwd_hooks:
            key = (
                "pre"
                if site.endswith("hook_resid_pre")
                else "attn"
                if site.endswith("hook_attn_out")
                else "mlp"
                if site.endswith("hook_mlp_out")
                else "post"
            )
            current = torch.zeros((len(prompts), 4, 1), dtype=torch.float32)
            current[:, -1, 0] = torch.tensor([state[key] for state in states])
            patched = hook(current, hook=None)
            for row, state in enumerate(states):
                state[key] = float(patched[row, -1, 0])
        scores: list[float] = []
        for state in states:
            if any(name.endswith("hook_resid_post") for name in names):
                score = state["post"]
            elif any(name.endswith("hook_resid_pre") for name in names):
                score = state["pre"]
            else:
                score = state["post"] + state["attn"] - 0.5 + state["mlp"] - 0.2
            scores.append(score)
        return self._logits(prompts, scores)


def test_component_scan_separates_attention_and_mlp() -> None:
    records = scan_components(FakeModel(), tiny_pairs(), batch_size=2, layer=5)
    assert len(records) == 80
    selected = {
        record.intervention: record
        for record in records
        if record.prompt_variant == "clean"
        and record.direction == "first_to_last"
        and record.source_mode == "matched"
        and record.family_id == "family-000"
    }
    assert selected["attention_output"].causal_transfer == pytest.approx(1.0)
    assert selected["mlp_output"].causal_transfer == pytest.approx(0.3)
    assert selected["attention_plus_mlp"].causal_transfer == pytest.approx(1.3)
    assert selected["residual_post"].causal_transfer == pytest.approx(4.0)
    summary = summarize(records, seed=42)
    assert summary["selected_layer"] == 5
    assert summary["leading_atomic_component"] == "attention_output"


def test_permuted_source_score_and_cli_are_correct() -> None:
    records = scan_components(FakeModel(), tiny_pairs(), batch_size=2, layer=5)
    record = next(
        item
        for item in records
        if item.prompt_variant == "clean"
        and item.direction == "first_to_last"
        and item.source_mode == "permuted"
        and item.intervention == "residual_post"
        and item.family_id == "family-000"
    )
    assert record.source_family_id == "family-001"
    assert record.source_score == pytest.approx(5.0)
    parsed = build_parser().parse_args(["--device", "cpu", "--batch-size", "2"])
    assert parsed.device == "cpu"
    assert not hasattr(parsed, "validation_root")
    assert not hasattr(parsed, "test_root")


def test_summary_computes_family_specific_advantage() -> None:
    records: list[ComponentRecord] = []
    for intervention in (
        "residual_pre",
        "attention_output",
        "mlp_output",
        "attention_plus_mlp",
        "residual_post",
    ):
        for index in range(4):
            for mode in ("matched", "permuted"):
                transfer = 1.0 if intervention == "attention_output" and mode == "matched" else 0.2
                records.append(
                    ComponentRecord(
                        family_id=f"family-{index}",
                        prompt_variant="clean",
                        direction="first_to_last",
                        source_mode=mode,
                        layer=5,
                        intervention=intervention,
                        source_family_id=f"source-{index}-{mode}",
                        source_score=3.0,
                        destination_score=1.0,
                        patched_score=1.0 + transfer,
                        causal_transfer=transfer,
                    )
                )
    summary = summarize(records, seed=42)
    attention = next(
        row
        for row in summary["intervention_ranking"]
        if row["intervention"] == "attention_output"
    )
    assert attention["family_specific_advantage"] == pytest.approx(0.8)
