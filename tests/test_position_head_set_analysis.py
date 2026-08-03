from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from autocircuit.datasets.associative_recall import ExamplePair
from autocircuit.pipeline import sha256
from autocircuit.position_head_localization import (
    HeadRecord,
    scan_heads,
)
from autocircuit.position_head_localization import (
    summarize as summarize_heads,
)
from autocircuit.position_head_set_analysis import (
    REPRODUCIBILITY_ABS_TOLERANCE,
    HeadSetRecord,
    ReproducibilityError,
    _bootstrap,
    _interaction_row,
    _loss_row,
    _patch_head_set,
    _ratio,
    _seed,
    _specificity,
    _write_incomplete,
    build_parser,
    enforce_reproducibility,
    intervention_heads,
    run,
    scan_head_sets,
    summarize_head_sets,
    unordered_head_pairs,
    validate_head_set_records,
    verify_head_run,
)
from autocircuit.position_localization import FirstLastPair
from autocircuit.position_study import STUDY_VERSION


def make_example(family: int, position: str, length: int = 4) -> ExamplePair:
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
            "prompt_token_length": length,
        },
    )


def tiny_pairs(count: int = 2) -> list[FirstLastPair]:
    return [
        FirstLastPair(
            f"family-{family:03d}",
            make_example(family, "first"),
            make_example(family, "last"),
        )
        for family in range(count)
    ]


class FakeModel:
    cfg = SimpleNamespace(n_layers=6, n_heads=4, parallel_attn_mlp=True)

    @staticmethod
    def _parts(prompt: str) -> tuple[str, str, str]:
        family, position, variant = prompt.split("|")
        return family, position, variant

    def _z(self, prompts: list[str]) -> torch.Tensor:
        value = torch.zeros((len(prompts), 4, self.cfg.n_heads, 1))
        for row, prompt in enumerate(prompts):
            family, position, _ = self._parts(prompt)
            offset = int(family[-3:]) * 0.1
            base = 2.0 if position == "first" else 0.5
            value[row, -1, :, 0] = torch.tensor(
                [base + offset, 2 * base + offset, 3 * base + offset, 4 * base + offset]
            )
        return value

    @staticmethod
    def _attn(z: torch.Tensor) -> torch.Tensor:
        return z.sum(dim=2)

    def _logits(self, prompts: list[str], attn: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((len(prompts), 4, 3))
        for row, prompt in enumerate(prompts):
            raw = attn[row, -1, 0]
            if self._parts(prompt)[2] == "corrupt":
                raw = -raw
            logits[row, -1, 1], logits[row, -1, 2] = raw / 2, -raw / 2
        return logits

    def run_with_cache(
        self, prompts: list[str], *, return_type: str, names_filter: list[str]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert return_type == "logits"
        z = self._z(prompts)
        attn = self._attn(z)
        cache = {
            site: z.clone() if site.endswith("attn.hook_z") else attn.clone()
            for site in names_filter
        }
        return self._logits(prompts, attn), cache

    def run_with_hooks(
        self,
        prompts: list[str],
        *,
        return_type: str,
        fwd_hooks: list[tuple[str, Any]],
    ) -> torch.Tensor:
        assert return_type == "logits"
        z = self._z(prompts)
        attn = self._attn(z)
        for site, callback in fwd_hooks:
            if site.endswith("attn.hook_z"):
                z = callback(z, hook=None)
                attn = self._attn(z)
            else:
                attn = callback(attn, hook=None)
        return self._logits(prompts, attn)


class InstrumentedTwoHeadModel(FakeModel):
    cfg = SimpleNamespace(n_layers=6, n_heads=2, parallel_attn_mlp=True)

    def __init__(self) -> None:
        self.hook_calls: list[tuple[str, bool]] = []

    def _z(self, prompts: list[str]) -> torch.Tensor:
        value = torch.zeros((len(prompts), 4, self.cfg.n_heads, 1))
        for row, prompt in enumerate(prompts):
            family, position, _ = self._parts(prompt)
            offset = int(family[-3:]) * 0.1
            base = 2.0 if position == "first" else 0.5
            value[row, -1, :, 0] = torch.tensor([base + offset, 2 * base + offset])
        return value

    def run_with_hooks(
        self,
        prompts: list[str],
        *,
        return_type: str,
        fwd_hooks: list[tuple[str, Any]],
    ) -> torch.Tensor:
        assert return_type == "logits"
        z = self._z(prompts)
        attn = self._attn(z)
        for site, callback in fwd_hooks:
            if site.endswith("attn.hook_z"):
                before = z.clone()
                z = callback(z, hook=None)
                self.hook_calls.append((site, torch.equal(z, before)))
                attn = self._attn(z)
            else:
                before = attn.clone()
                attn = callback(attn, hook=None)
                self.hook_calls.append((site, torch.equal(attn, before)))
        return self._logits(prompts, attn)


def test_dynamic_pair_enumeration_and_intervention_count() -> None:
    assert unordered_head_pairs(4) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    interventions = intervention_heads(4)
    assert len(interventions) == 6 + 1 + 4 + 6
    assert interventions["all_heads_z_patch"] == (0, 1, 2, 3)
    assert interventions["leave_one_out_2"] == (0, 1, 3)
    assert interventions["leave_pair_out_1_3"] == (0, 2)
    assert len(intervention_heads(8)) + 2 == 67
    two_head_interventions = intervention_heads(2)
    assert two_head_interventions["leave_pair_out_0_1"] == ()
    with pytest.raises(ValueError, match="at least two"):
        unordered_head_pairs(1)


@pytest.mark.parametrize("heads", [(), (1, 3), (0, 1, 2, 3), (0, 2, 3), (0, 2)])
def test_head_set_hook_exactly_patches_selected_final_slices(heads: tuple[int, ...]) -> None:
    destination = torch.arange(64.0).reshape(2, 4, 4, 2)
    source = destination + 100
    patched = _patch_head_set(source, heads)(destination, hook=None)
    assert torch.equal(patched[:, :-1], destination[:, :-1])
    for head in range(4):
        expected = source if head in heads else destination
        assert torch.equal(patched[:, -1, head], expected[:, -1, head])
    assert patched.dtype == destination.dtype and patched.device == destination.device


def test_head_set_hook_rejects_malformed_shapes_and_indexes() -> None:
    good = torch.zeros((2, 4, 4, 2))
    with pytest.raises(RuntimeError, match="must have shape"):
        _patch_head_set(torch.zeros((2, 4, 8)), (0,))(good, hook=None)
    with pytest.raises(RuntimeError, match="shapes differ"):
        _patch_head_set(torch.zeros((1, 4, 4, 2)), (0,))(good, hook=None)
    with pytest.raises(ValueError, match="outside"):
        _patch_head_set(good, (4,))(good, hook=None)
    with pytest.raises(ValueError, match="unique"):
        _patch_head_set(good, (1, 1))


def _scan() -> list[HeadSetRecord]:
    return scan_head_sets(FakeModel(), tiny_pairs(), 2, 5, "layer", "component", "head")


def test_scan_has_complete_factorial_count_provenance_and_exact_noop() -> None:
    records = _scan()
    assert len(records) == 2 * 2 * 2 * 2 * (6 + 1 + 4 + 6 + 1 + 1)
    validate_head_set_records(records, 4, {"family-000", "family-001"})
    assert {record.source_mode for record in records} == {"matched", "permuted"}
    assert all(
        record.oriented_causal_transfer == 0.0
        for record in records
        if record.intervention == "identity_noop"
    )
    permuted = next(
        record
        for record in records
        if record.intervention == "pair_patch_2_3"
        and record.family_id == "family-000"
        and record.prompt_variant == "clean"
        and record.direction == "first_to_last"
        and record.source_mode == "permuted"
    )
    assert permuted.source_family_id == "family-001"
    assert permuted.source_score == pytest.approx(20.4)
    assert permuted.oriented_causal_transfer == pytest.approx(10.7)


def test_two_head_leave_pair_out_executes_empty_z_hook_and_is_distinct_from_noop() -> None:
    model = InstrumentedTwoHeadModel()
    records = scan_head_sets(model, tiny_pairs(), 2, 5, "layer", "component", "head")
    leave_pair = [record for record in records if record.intervention == "leave_pair_out_0_1"]
    identity = [record for record in records if record.intervention == "identity_noop"]
    z_site = "blocks.5.attn.hook_z"
    attn_site = "blocks.5.hook_attn_out"

    assert len(leave_pair) == len(identity) == 2 * 2 * 2 * 2
    assert {record.source_mode for record in leave_pair} == {"matched", "permuted"}
    assert all(record.included_heads == () for record in leave_pair)
    assert all(record.hook_site == z_site for record in leave_pair)
    assert all(record.oriented_causal_transfer == 0.0 for record in leave_pair)
    assert all(record.intervention == "identity_noop" for record in identity)
    assert all(record.included_heads == () for record in identity)
    assert all(record.hook_site == attn_site for record in identity)
    assert all(record.oriented_causal_transfer == 0.0 for record in identity)

    z_calls = [unchanged for site, unchanged in model.hook_calls if site == z_site]
    attn_calls = [unchanged for site, unchanged in model.hook_calls if site == attn_site]
    assert len(z_calls) == 5 * 2 * 2 * 2
    assert sum(z_calls) == 2 * 2 * 2
    assert len(attn_calls) == 2 * 2 * 2 * 2
    assert sum(attn_calls) == 2 * 2 * 2


def test_pair_and_all_head_transfers_have_exact_orientation_in_all_four_cells() -> None:
    records = _scan()
    for intervention, expected in (("pair_patch_2_3", 10.5), ("all_heads_z_patch", 15.0)):
        observed = {
            (record.prompt_variant, record.direction): record.oriented_causal_transfer
            for record in records
            if record.intervention == intervention
            and record.family_id == "family-000"
            and record.source_mode == "matched"
        }
        assert observed == {
            ("clean", "first_to_last"): pytest.approx(expected),
            ("clean", "last_to_first"): pytest.approx(expected),
            ("corrupt", "first_to_last"): pytest.approx(expected),
            ("corrupt", "last_to_first"): pytest.approx(expected),
        }


def test_record_validation_rejects_missing_duplicate_and_wrong_heads() -> None:
    records = _scan()
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_head_set_records(records[:-1], 4)
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_head_set_records([*records, records[0]], 4)
    with pytest.raises(RuntimeError, match="included heads"):
        validate_head_set_records([replace(records[0], included_heads=(3,)), *records[1:]], 4)


def _head_run(pairs: list[FirstLastPair]) -> dict[str, Any]:
    records = scan_heads(FakeModel(), pairs, 2, 5, "layer", "component")
    previous = {
        "matched_mean_transfer": 15.0,
        "permuted_mean_transfer": 15.0,
        "family_specific_advantage": 0.0,
    }
    summary = summarize_heads(records, 4, previous, bootstrap_samples=100)
    ranking = summary["individual_head_ranking"]
    ranked = [int(row["head_index"]) for row in ranking]
    return {
        "records": records,
        "summary": summary,
        "candidate_pair": tuple(sorted(ranked[:2])),
        "negative_control_pair": tuple(sorted(ranked[-2:])),
        "selection_rule": (
            "two highest and two lowest individual heads by family-specific advantage"
        ),
    }


def test_candidate_and_negative_pairs_are_automatic() -> None:
    head_run = _head_run(tiny_pairs())
    ranking = [int(row["head_index"]) for row in head_run["summary"]["individual_head_ranking"]]
    assert head_run["candidate_pair"] == tuple(sorted(ranking[:2]))
    assert head_run["negative_control_pair"] == tuple(sorted(ranking[-2:]))


def test_head_upstream_hash_provenance_and_raw_reconstruction(tmp_path: Path) -> None:
    root = tmp_path / "head"
    root.mkdir()
    identity = {
        "model": "pythia-70m",
        "model_id": "EleutherAI/pythia-70m",
        "tokenizer_id": "EleutherAI/pythia-70m",
        "requested_revision": "main",
        "load_revision": "main",
        "frozen_resolved_revision": None,
        "dtype": "torch.float32",
        "exact_revision_matching_succeeded": False,
        "validation_resolved_revision": None,
    }
    discovery = {
        "seed": 42,
        "model": "pythia-70m",
        "model_id": "EleutherAI/pythia-70m",
        "tokenizer_id": "EleutherAI/pythia-70m",
        "requested_revision": "main",
        "load_revision": "main",
        "resolved_revision": None,
        "dtype": "torch.float32",
        "exact_revision_available": False,
        "artifact_hashes": {"dataset": "discovery"},
    }
    layer = {
        "layer": 5,
        "identity": identity,
        "hashes": {"manifest": "lm", "summary": "ls", "final": "lf", "records": "lr"},
    }
    component = {
        "layer": 5,
        "identity": identity,
        "hashes": {"manifest": "cm", "summary": "cs", "final": "cf", "records": "cr"},
        "attention": {
            "matched_mean_transfer": 10.0,
            "permuted_mean_transfer": 5.0,
            "family_specific_advantage": 5.0,
        },
    }
    records: list[HeadRecord] = []
    for family in range(120):
        for intervention in [
            *(f"head_{head}" for head in range(8)),
            "aggregate_attention_output",
            "identity_noop",
        ]:
            head_index = (
                int(intervention.removeprefix("head_"))
                if intervention.startswith("head_")
                else None
            )
            for variant in ("clean", "corrupt"):
                for direction in ("first_to_last", "last_to_first"):
                    for mode in ("matched", "permuted"):
                        if intervention == "identity_noop":
                            transfer = 0.0
                        elif intervention == "aggregate_attention_output":
                            transfer = 10.0 if mode == "matched" else 5.0
                        else:
                            base = float(head_index or 0)
                            transfer = base + (1.0 if mode == "matched" else 0.0)
                        records.append(
                            HeadRecord(
                                family_id=f"family-{family:03d}",
                                prompt_variant=variant,
                                direction=direction,
                                source_mode=mode,
                                selected_layer=5,
                                intervention=intervention,
                                head_index=head_index,
                                hook_site="site",
                                source_family_id=f"source-{family:03d}-{mode}",
                                destination_family_id=f"family-{family:03d}",
                                source_score=2.0,
                                destination_baseline=1.0,
                                patched_score=1.0 + transfer,
                                oriented_causal_transfer=transfer,
                                layer_run_manifest_hash="lm",
                                component_run_manifest_hash="cm",
                            )
                        )
    summary = summarize_heads(records, 8, component["attention"], seed=42)
    records_path = root / "head_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record.__dict__, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "head_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    final = {
        "status": "EXPLORATORY_ATTENTION_HEAD_LOCALIZATION_COMPLETE",
        "software_success": True,
        "interpretation_scope": "exploratory_discovery_only",
        "selected_layer": 5,
        "dynamically_detected_head_count": 8,
        "matched_family_count": 120,
        "record_count": len(records),
        "leading_head": summary["leading_head"],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "activation_patching_performed": True,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
    (root / "head_final_status.json").write_text(json.dumps(final), encoding="utf-8")
    manifest = identity | {
        "complete": True,
        "protocol_version": "position-head-localization-0.1.0",
        "selected_layer": 5,
        "n_heads": 8,
        "discovery_artifact_hashes": discovery["artifact_hashes"],
        "layer_run_hashes": layer["hashes"],
        "component_run_hashes": component["hashes"],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "artifact_hashes": {
            name: sha256(root / name)
            for name in ("head_records.jsonl", "head_summary.json", "head_final_status.json")
        },
    }
    (root / "head_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    verified = verify_head_run(root, discovery, layer, component)
    assert verified["candidate_pair"] == (6, 7)
    assert verified["negative_control_pair"] == (0, 1)
    assert len(verified["records"]) == 9600
    records_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash"):
        verify_head_run(root, discovery, layer, component)


def test_summary_recovery_losses_interactions_specificity_and_bootstrap() -> None:
    pairs = tiny_pairs()
    records = scan_head_sets(FakeModel(), pairs, 2, 5, "layer", "component", "head")
    head_run = _head_run(pairs)
    component = {
        "attention": {
            "matched_mean_transfer": 15.0,
            "permuted_mean_transfer": 15.0,
            "family_specific_advantage": 0.0,
        }
    }
    provisional = summarize_head_sets(records, 4, head_run, component)
    component["attention"] = {
        key: provisional["aggregate_attention_reference"][key]
        for key in (
            "matched_mean_transfer",
            "permuted_mean_transfer",
            "family_specific_advantage",
        )
    }
    first = summarize_head_sets(records, 4, head_run, component)
    second = summarize_head_sets(records, 4, head_run, component)
    assert first == second
    assert len(first["pair_patch_ranking"]) == 6
    assert len(first["leave_one_out_ranking"]) == 4
    assert len(first["leave_pair_out_ranking"]) == 6
    assert len(first["pair_interaction_ranking"]) == 6
    assert first["candidate_pair_diagnostics"]["pair_patch"]["pair"] == list(
        head_run["candidate_pair"]
    )
    assert (
        "candidate_difference_from_negative_control"
        in first["specificity_comparison"]["pair_family_specific_advantage"]
    )
    assert first["identity_noop_reference"]["matched_mean_transfer"] == 0.0
    assert first["reproducibility_diagnostics"]["upstream_individual_heads_recomputed_exactly"]
    interaction = first["pair_interaction_ranking"][0]
    assert len(interaction["matched_interaction_ci_95"]) == 2
    assert interaction["matched_interaction_interpretation"] in {
        "super_additive_diagnostic",
        "sub_additive_diagnostic",
        "no_clear_interaction",
    }
    diagnostics = first["reproducibility_diagnostics"]
    assert diagnostics["all_heads_z_minus_aggregate_matched"] == 0.0
    assert diagnostics["all_heads_z_minus_aggregate_permuted"] == 0.0
    assert diagnostics["all_heads_z_minus_aggregate_family_specific_advantage"] == 0.0
    assert diagnostics["aggregate_minus_prior_component_matched"] == 0.0
    assert diagnostics["aggregate_minus_prior_component_permuted"] == 0.0
    assert diagnostics["aggregate_minus_prior_component_family_specific_advantage"] == 0.0


def test_zero_denominator_is_explicit() -> None:
    assert _ratio(1.0, 0.0) is None
    assert _ratio(1.0, 1e-13) is None
    assert _ratio(1.0, 2.0) == 0.5


def test_hand_calculated_recovery_and_leave_out_losses() -> None:
    assert _ratio(4.0, 8.0) == 0.5
    assert _ratio(3.0, 12.0) == 0.25
    families = ["a", "b"]
    profiles = {
        ("all", "a", "matched"): 10.0,
        ("all", "a", "permuted"): 4.0,
        ("all", "b", "matched"): 20.0,
        ("all", "b", "permuted"): 8.0,
        ("leave_one", "a", "matched"): 7.0,
        ("leave_one", "a", "permuted"): 3.0,
        ("leave_one", "b", "matched"): 18.0,
        ("leave_one", "b", "permuted"): 9.0,
        ("leave_pair", "a", "matched"): 6.0,
        ("leave_pair", "a", "permuted"): 2.0,
        ("leave_pair", "b", "matched"): 15.0,
        ("leave_pair", "b", "permuted"): 7.0,
    }
    single = _loss_row("all", "leave_one", profiles, families, 42)
    assert single["matched_transfer_loss"] == pytest.approx(2.5)
    assert single["family_specific_advantage_loss"] == pytest.approx(2.5)
    assert single["positive_matched_transfer_loss_fraction"] == 1.0
    assert single["positive_family_specific_advantage_loss_fraction"] == 1.0
    pair = _loss_row("all", "leave_pair", profiles, families, 42)
    assert pair["matched_transfer_loss"] == pytest.approx(4.5)
    assert pair["family_specific_advantage_loss"] == pytest.approx(3.0)
    assert pair["positive_matched_transfer_loss_fraction"] == 1.0
    assert pair["positive_family_specific_advantage_loss_fraction"] == 1.0


def test_hand_calculated_family_paired_interaction() -> None:
    families = ["a", "b", "c"]
    profiles = {
        ("pair_patch_0_1", "a", "matched"): 10.0,
        ("pair_patch_0_1", "a", "permuted"): 4.0,
        ("pair_patch_0_1", "b", "matched"): 20.0,
        ("pair_patch_0_1", "b", "permuted"): 6.0,
        ("pair_patch_0_1", "c", "matched"): 5.0,
        ("pair_patch_0_1", "c", "permuted"): 1.0,
    }
    upstream = {
        ("head_0", "a", "matched"): 1.0,
        ("head_0", "a", "permuted"): 0.0,
        ("head_1", "a", "matched"): 2.0,
        ("head_1", "a", "permuted"): 1.0,
        ("head_0", "b", "matched"): 8.0,
        ("head_0", "b", "permuted"): 2.0,
        ("head_1", "b", "matched"): 15.0,
        ("head_1", "b", "permuted"): 5.0,
        ("head_0", "c", "matched"): 2.0,
        ("head_0", "c", "permuted"): 1.0,
        ("head_1", "c", "matched"): 5.0,
        ("head_1", "c", "permuted"): 1.0,
    }
    matched_interactions = [7.0, -3.0, -2.0]
    advantage_interactions = [4.0, -2.0, -1.0]
    row = _interaction_row((0, 1), profiles, upstream, families, 42)
    assert row["matched_mean_interaction"] == pytest.approx(sum(matched_interactions) / 3)
    assert row["family_specific_advantage_mean_interaction"] == pytest.approx(
        sum(advantage_interactions) / 3
    )
    assert row["matched_positive_interaction_fraction"] == pytest.approx(1 / 3)
    assert row["family_specific_advantage_positive_interaction_fraction"] == pytest.approx(1 / 3)
    assert row["matched_interaction_ci_95"] == pytest.approx([-3.0, 7.0])
    assert row["family_specific_advantage_interaction_ci_95"] == pytest.approx([-2.0, 4.0])
    assert row["matched_interaction_interpretation"] == "no_clear_interaction"
    assert row["family_specific_advantage_interaction_interpretation"] == (
        "no_clear_interaction"
    )

    # A complete permutation preserves the arithmetic mean, but destroys the
    # matched-family profile and therefore changes its fraction and interval.
    mispaired_matched = [-6.0, 7.0, 1.0]
    mispaired_advantage = [-5.0, 4.0, 2.0]
    assert mispaired_matched != matched_interactions
    assert mispaired_advantage != advantage_interactions
    assert sum(value > 0 for value in mispaired_matched) / 3 == pytest.approx(2 / 3)
    assert sum(value > 0 for value in mispaired_advantage) / 3 == pytest.approx(2 / 3)
    mispaired_matched_interval = _bootstrap(
        mispaired_matched, _seed(42, "pair_patch_0_1|matched-interaction"), 10_000
    )
    mispaired_advantage_interval = _bootstrap(
        mispaired_advantage,
        _seed(42, "pair_patch_0_1|advantage-interaction"),
        10_000,
    )
    assert mispaired_matched_interval == pytest.approx([-6.0, 7.0])
    assert mispaired_advantage_interval == pytest.approx([-5.0, 4.0])
    assert mispaired_matched_interval != row["matched_interaction_ci_95"]
    assert mispaired_advantage_interval != row[
        "family_specific_advantage_interaction_ci_95"
    ]

    # A wrong-family index that repeats b and omits other head-1 families also
    # changes the mean, catching non-bijective indexing defects.
    repeated_wrong_family = [-6.0, -3.0, -12.0]
    assert sum(repeated_wrong_family) / 3 != pytest.approx(
        row["matched_mean_interaction"]
    )


@pytest.mark.parametrize(
    ("candidate_value", "other_values", "expected_percentile"),
    [
        (10.0, [1.0, 2.0, 3.0], 100.0),
        (0.0, [1.0, 2.0, 3.0], 0.0),
        (3.0, [1.0, 4.0, 5.0], 100 / 3),
        (3.0, [3.0, 3.0, 4.0], 200 / 3),
    ],
)
def test_specificity_percentile_uses_only_noncandidates(
    candidate_value: float, other_values: list[float], expected_percentile: float
) -> None:
    pairs = ([1, 2], [0, 1], [0, 2], [0, 3])
    values = [candidate_value, *other_values]
    rows = [
        {"pair": pair, "metric": value, "rank": index}
        for index, (pair, value) in enumerate(zip(pairs, values, strict=True), 1)
    ]
    result = _specificity(rows, "metric", (1, 2), (0, 1))
    ordered = sorted(other_values)
    median = ordered[1]
    assert result["candidate_empirical_percentile"] == pytest.approx(expected_percentile)
    assert result["candidate_difference_from_median_noncandidate"] == pytest.approx(
        candidate_value - median
    )
    assert result["candidate_difference_from_negative_control"] == pytest.approx(
        candidate_value - other_values[0]
    )
    assert result["pairs_exceeding_candidate"] == sum(
        value > candidate_value for value in other_values
    )


def test_two_head_specificity_is_explicitly_undefined_without_noncandidates() -> None:
    result = _specificity(
        [{"pair": [0, 1], "metric": 2.0, "rank": 1}],
        "metric",
        (0, 1),
        (0, 1),
    )
    assert result["candidate_empirical_percentile"] is None
    assert result["candidate_difference_from_median_noncandidate"] is None
    assert result["candidate_difference_from_negative_control"] == 0.0
    assert result["comparison_status"] == "undefined_no_noncandidate_pairs"


def _interaction_with_values(values: list[float]) -> dict[str, Any]:
    families = [f"f{index}" for index in range(len(values))]
    profiles: dict[tuple[str, str, str], float] = {}
    upstream: dict[tuple[str, str, str], float] = {}
    for family, value in zip(families, values, strict=True):
        profiles[("pair_patch_0_1", family, "matched")] = value
        profiles[("pair_patch_0_1", family, "permuted")] = 0.0
        for head in (0, 1):
            upstream[(f"head_{head}", family, "matched")] = 0.0
            upstream[(f"head_{head}", family, "permuted")] = 0.0
    return _interaction_row((0, 1), profiles, upstream, families, 42)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 1.0, 2.0], "super_additive_diagnostic"),
        ([-1.0, -2.0, -1.0, -2.0], "sub_additive_diagnostic"),
        ([10.0, -1.0, -1.0, -1.0], "no_clear_interaction"),
        ([-10.0, 1.0, 1.0, 1.0], "no_clear_interaction"),
        ([0.0, 0.0, 0.0, 0.0], "no_clear_interaction"),
    ],
)
def test_interaction_labels_are_interval_aware(values: list[float], expected: str) -> None:
    row = _interaction_with_values(values)
    assert row["matched_interaction_interpretation"] == expected
    assert row["family_specific_advantage_interaction_interpretation"] == expected


def _repro_summary(offset: float = 0.0) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate = {
        "matched_mean_transfer": 3.0,
        "permuted_mean_transfer": 1.0,
        "family_specific_advantage": 2.0,
    }
    summary = {
        "all_heads_z_reference": {
            key: value + offset for key, value in aggregate.items()
        },
        "aggregate_attention_reference": dict(aggregate),
    }
    return summary, {"attention": dict(aggregate)}


def test_reproducibility_exact_and_within_tolerance_pass() -> None:
    summary, component = _repro_summary()
    exact = enforce_reproducibility(
        summary, component, device="cpu", dtype="torch.float32", selected_layer=5
    )
    assert exact["status"] == "passed"
    summary, component = _repro_summary(REPRODUCIBILITY_ABS_TOLERANCE / 2)
    within = enforce_reproducibility(
        summary, component, device="cpu", dtype="torch.float32", selected_layer=5
    )
    assert within["status"] == "passed"


def test_reproducibility_beyond_tolerance_fails_with_structured_details() -> None:
    summary, component = _repro_summary(REPRODUCIBILITY_ABS_TOLERANCE * 2)
    with pytest.raises(ReproducibilityError) as captured:
        enforce_reproducibility(
            summary, component, device="cuda", dtype="torch.float32", selected_layer=5
        )
    assert captured.value.details["status"] == "failed"
    assert captured.value.details["failures"]
    assert captured.value.details["device"] == "cuda"


def test_empirical_pair_ranking_tie_breaks_by_pair() -> None:
    pairs = tiny_pairs()
    records = scan_head_sets(FakeModel(), pairs, 2, 5, "layer", "component", "head")
    # Remove family offsets so all family-specific advantages tie at zero.
    records = [replace(record, oriented_causal_transfer=0.0) for record in records]
    head_run = _head_run(pairs)
    component = {
        "attention": {
            "matched_mean_transfer": 0.0,
            "permuted_mean_transfer": 0.0,
            "family_specific_advantage": 0.0,
        }
    }
    summary = summarize_head_sets(records, 4, head_run, component)
    assert [row["pair"] for row in summary["pair_patch_ranking"]] == [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ]
    assert summary["specificity_comparison"]["transfer_recovery_fraction"]["status"] == (
        "undefined_due_to_unstable_denominator"
    )


def test_cli_has_no_validation_or_test_arguments() -> None:
    parsed = build_parser().parse_args([])
    assert hasattr(parsed, "head_root") and hasattr(parsed, "component_root")
    assert not hasattr(parsed, "validation_root")
    assert not hasattr(parsed, "test_root")


def test_incomplete_manifest_hashes_diagnostics_without_self_hash(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "head_set_final_status.json").write_text("{}", encoding="utf-8")
    _write_incomplete(root, "scanning", {"candidate_pair": [1, 2]}, RuntimeError("boom"))
    manifest = json.loads((root / "head_set_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["software_success"] is False
    assert manifest["failure_stage"] == "scanning"
    assert "head_set_final_status.json" in manifest["artifact_hashes"]
    assert "head_set_manifest.json" not in manifest["artifact_hashes"]


def test_no_model_loading_when_provenance_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = False

    def fail(_: Path) -> dict[str, Any]:
        raise RuntimeError("provenance failed")

    def factory(*_: str) -> Any:
        nonlocal loaded
        loaded = True
        raise AssertionError

    monkeypatch.setattr("autocircuit.position_head_set_analysis._verify_frozen_discovery", fail)
    args = argparse.Namespace(
        discovery_root=tmp_path,
        layer_root=tmp_path,
        component_root=tmp_path,
        head_root=tmp_path,
        output=tmp_path / "output",
        device="cpu",
        batch_size=2,
        force=False,
    )
    with pytest.raises(RuntimeError, match="provenance failed"):
        run(args, factory)
    assert loaded is False


def test_force_scope_and_existing_output_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    discovery = {
        "seed": 42,
        "requested_revision": "main",
        "model": "pythia-70m",
        "load_revision": "main",
        "artifact_hashes": {},
    }
    identity = {key: "value" for key in ("model",)}
    layer = {"layer": 5, "hashes": {}, "identity": identity}
    component = {"layer": 5, "hashes": {}, "identity": identity, "attention": {}}
    head = {
        "layer": 5,
        "n_heads": 4,
        "hashes": {},
        "identity": identity,
        "candidate_pair": (2, 3),
        "negative_control_pair": (0, 1),
        "selection_rule": "rule",
    }
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis._verify_frozen_discovery", lambda _: discovery
    )
    monkeypatch.setattr("autocircuit.position_head_set_analysis.verify_layer_run", lambda *_: layer)
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.verify_component_run", lambda *_: component
    )
    monkeypatch.setattr("autocircuit.position_head_set_analysis.verify_head_run", lambda *_: head)
    output = tmp_path / "output"
    target = output / "seed-42-main"
    target.mkdir(parents=True)
    sibling = output / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")
    args = argparse.Namespace(
        discovery_root=tmp_path,
        layer_root=tmp_path,
        component_root=tmp_path,
        head_root=tmp_path,
        output=output,
        device="cpu",
        batch_size=2,
        force=False,
    )
    with pytest.raises(RuntimeError, match="use --force"):
        run(args)
    args.force = True
    monkeypatch.setattr("autocircuit.position_head_set_analysis.read_jsonl", lambda _: [])
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.build_first_last_pairs", lambda _: tiny_pairs(1)
    )
    with pytest.raises(ValueError, match="one family"):
        run(args)
    assert sibling.read_text(encoding="utf-8") == "keep"


def _mock_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, component_offset: float = 0.0
) -> tuple[argparse.Namespace, SimpleNamespace, dict[str, Any]]:
    pairs = tiny_pairs(120)
    identity = {
        "model": "pythia-70m",
        "model_id": "EleutherAI/pythia-70m",
        "tokenizer_id": "EleutherAI/pythia-70m",
        "requested_revision": "main",
        "load_revision": "main",
        "frozen_resolved_revision": None,
        "dtype": "torch.float32",
        "exact_revision_matching_succeeded": False,
        "validation_resolved_revision": None,
    }
    discovery = {
        "seed": 42,
        "requested_revision": "main",
        "load_revision": "main",
        "model": "pythia-70m",
        "artifact_hashes": {"dataset": "frozen"},
    }
    layer = {"layer": 5, "hashes": {"manifest": "layer"}, "identity": identity}
    upstream_records = scan_heads(FakeModel(), pairs, 120, 5, "layer", "component")
    upstream_summary = summarize_heads(
        upstream_records,
        4,
        {
            "matched_mean_transfer": 15.0,
            "permuted_mean_transfer": 15.0,
            "family_specific_advantage": 0.0,
        },
        bootstrap_samples=100,
    )
    aggregate = upstream_summary["aggregate_attention_reference"]
    component = {
        "layer": 5,
        "hashes": {"manifest": "component"},
        "identity": identity,
        "parallel_attn_mlp": True,
        "attention": {
            "matched_mean_transfer": float(aggregate["matched_mean_transfer"])
            + component_offset,
            "permuted_mean_transfer": float(aggregate["permuted_mean_transfer"]),
            "family_specific_advantage": float(aggregate["family_specific_advantage"]),
        },
    }
    ranked = [int(row["head_index"]) for row in upstream_summary["individual_head_ranking"]]
    head = {
        "layer": 5,
        "n_heads": 4,
        "hashes": {"manifest": "head"},
        "identity": identity,
        "records": upstream_records,
        "summary": upstream_summary,
        "candidate_pair": tuple(sorted(ranked[:2])),
        "negative_control_pair": tuple(sorted(ranked[-2:])),
        "selection_rule": "upstream ranking rule",
    }
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis._verify_frozen_discovery", lambda _: discovery
    )
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.verify_layer_run", lambda *_: layer
    )
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.verify_component_run", lambda *_: component
    )
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.verify_head_run", lambda *_: head
    )
    monkeypatch.setattr("autocircuit.position_head_set_analysis.read_jsonl", lambda _: [])
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.build_first_last_pairs", lambda _: pairs
    )
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis._verify_validation_adapter",
        lambda *_: {
            "model_id": "EleutherAI/pythia-70m",
            "resolved_device": "cpu",
            "dtype": "torch.float32",
        },
    )
    args = argparse.Namespace(
        discovery_root=tmp_path,
        layer_root=tmp_path,
        component_root=tmp_path,
        head_root=tmp_path,
        output=tmp_path / "output",
        device="cpu",
        batch_size=120,
        force=False,
    )
    return args, SimpleNamespace(model=FakeModel()), head


def test_real_success_lifecycle_writes_complete_verified_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, adapter, head = _mock_lifecycle(monkeypatch, tmp_path)
    final = run(args, lambda *_: adapter)
    root = args.output / "seed-42-main"
    expected_names = {
        "head_set_records.jsonl",
        "head_set_summary.json",
        "head_set_analysis.md",
        "head_set_final_status.json",
        "head_set_manifest.json",
    }
    assert {path.name for path in root.iterdir()} == expected_names
    summary = json.loads((root / "head_set_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "head_set_manifest.json").read_text(encoding="utf-8"))
    assert final["software_success"] is True
    assert final["matched_family_count"] == 120
    assert final["record_count"] == 120 * 2 * 2 * 2 * 19
    assert final["candidate_pair"] == list(head["candidate_pair"])
    assert summary["negative_control_pair"] == list(head["negative_control_pair"])
    assert len(summary["pair_patch_ranking"]) == 6
    assert len(summary["leave_one_out_ranking"]) == 4
    assert summary["reproducibility_enforcement"]["status"] == "passed"
    assert summary["identity_noop_reference"]["matched_mean_transfer"] == 0.0
    assert final["held_out_validation_reused"] is False
    assert final["held_out_test_opened"] is False
    assert final["scientific_confirmation"] is False
    assert final["circuit_found"] is False
    assert manifest["complete"] is True
    assert "head_set_manifest.json" not in manifest["artifact_hashes"]
    for name, digest in manifest["artifact_hashes"].items():
        assert sha256(root / name) == digest


def _assert_failed_run(root: Path, stage: str, message: str) -> dict[str, Any]:
    final = json.loads((root / "head_set_final_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "head_set_manifest.json").read_text(encoding="utf-8"))
    assert final["software_success"] is False
    assert final["failure_stage"] == stage
    assert message in final["exception_message"]
    assert manifest["complete"] is False
    assert manifest["software_success"] is False
    assert manifest["failure_stage"] == stage
    assert message in manifest["exception_message"]
    return manifest


def test_run_scanning_failure_is_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, adapter, _ = _mock_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.scan_head_sets",
        lambda *_: (_ for _ in ()).throw(RuntimeError("scan boom")),
    )
    with pytest.raises(RuntimeError, match="scan boom"):
        run(args, lambda *_: adapter)
    manifest = _assert_failed_run(args.output / "seed-42-main", "scanning", "scan boom")
    assert "head_set_records.jsonl" not in manifest["artifact_hashes"]


def test_run_post_record_validation_failure_hashes_partial_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, adapter, _ = _mock_lifecycle(monkeypatch, tmp_path)
    partial = _scan()[:1]
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.scan_head_sets", lambda *_: partial
    )
    with pytest.raises(RuntimeError, match="families"):
        run(args, lambda *_: adapter)
    root = args.output / "seed-42-main"
    manifest = _assert_failed_run(root, "record_validation", "families")
    assert sha256(root / "head_set_records.jsonl") == manifest["artifact_hashes"][
        "head_set_records.jsonl"
    ]


def test_run_summarization_failure_is_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, adapter, _ = _mock_lifecycle(monkeypatch, tmp_path)
    complete = scan_head_sets(FakeModel(), tiny_pairs(120), 120, 5, "l", "c", "h")
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.scan_head_sets", lambda *_: complete
    )
    monkeypatch.setattr(
        "autocircuit.position_head_set_analysis.summarize_head_sets",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("summary boom")),
    )
    with pytest.raises(RuntimeError, match="summary boom"):
        run(args, lambda *_: adapter)
    manifest = _assert_failed_run(
        args.output / "seed-42-main", "summarization", "summary boom"
    )
    assert "head_set_records.jsonl" in manifest["artifact_hashes"]


def test_run_reproducibility_failure_cannot_report_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, adapter, _ = _mock_lifecycle(
        monkeypatch, tmp_path, component_offset=REPRODUCIBILITY_ABS_TOLERANCE * 10
    )
    with pytest.raises(ReproducibilityError):
        run(args, lambda *_: adapter)
    root = args.output / "seed-42-main"
    manifest = _assert_failed_run(
        root, "reproducibility_enforcement", "reproducibility control failed"
    )
    assert manifest["reproducibility_failure"]["status"] == "failed"
    assert manifest["reproducibility_failure"]["failures"]
    assert "head_set_records.jsonl" in manifest["artifact_hashes"]
