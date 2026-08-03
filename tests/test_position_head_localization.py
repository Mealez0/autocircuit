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
from autocircuit.position_component_localization import PROTOCOL_VERSION as COMPONENT_VERSION
from autocircuit.position_component_localization import STATUS as COMPONENT_STATUS
from autocircuit.position_component_localization import ComponentRecord
from autocircuit.position_component_localization import summarize as component_summarize
from autocircuit.position_head_localization import (
    HeadRecord,
    _identity_final_position,
    _patch_final_position,
    _patch_head_query,
    build_parser,
    deterministic_batches,
    main,
    run,
    scan_heads,
    summarize,
    validate_head_records,
    verify_component_run,
    verify_layer_run,
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
    cfg = SimpleNamespace(n_layers=6, n_heads=3, parallel_attn_mlp=True)

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
                [base + offset, 2 * base + offset, 3 * base + offset]
            )
        return value

    @staticmethod
    def _attn(z: torch.Tensor) -> torch.Tensor:
        return z.sum(dim=2)

    def _logits(self, prompts: list[str], attn: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((len(prompts), 4, 3))
        scores = attn[:, -1, 0]
        for row, prompt in enumerate(prompts):
            variant = self._parts(prompt)[2]
            raw = scores[row] if variant == "clean" else -scores[row]
            logits[row, -1, 1] = raw / 2
            logits[row, -1, 2] = -raw / 2
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


def test_head_hook_exact_slice_and_keyword_compatibility() -> None:
    destination = torch.arange(48.0).reshape(2, 3, 4, 2)
    source = destination + 100
    result = _patch_head_query(source, 2)(destination, hook=None)
    assert torch.equal(result[:, -1, 2, :], source[:, -1, 2, :])
    assert torch.equal(result[:, :-1, :, :], destination[:, :-1, :, :])
    assert torch.equal(result[:, -1, :2, :], destination[:, -1, :2, :])
    assert torch.equal(result[:, -1, 3:, :], destination[:, -1, 3:, :])
    assert result.dtype == destination.dtype and result.device == destination.device


def test_head_hook_rejects_shape_and_index_errors() -> None:
    good = torch.zeros((2, 3, 4, 2))
    with pytest.raises(RuntimeError, match="must have shape"):
        _patch_head_query(torch.zeros((2, 3, 8)), 0)(good, hook=None)
    with pytest.raises(RuntimeError, match="shapes differ"):
        _patch_head_query(torch.zeros((1, 3, 4, 2)), 0)(good, hook=None)
    with pytest.raises(ValueError, match="non-negative"):
        _patch_head_query(good, -1)
    with pytest.raises(ValueError, match="outside"):
        _patch_head_query(good, 4)(good, hook=None)


def test_reference_hooks_patch_final_position_and_identity_exactly() -> None:
    destination = torch.arange(24.0).reshape(2, 3, 4)
    source = destination + 50
    patched = _patch_final_position(source)(destination, hook=None)
    assert torch.equal(patched[:, :-1], destination[:, :-1])
    assert torch.equal(patched[:, -1], source[:, -1])
    assert torch.equal(_identity_final_position()(destination, hook=None), destination)


def test_scan_dynamic_heads_full_factorial_and_references() -> None:
    records = scan_heads(FakeModel(), tiny_pairs(), 2, 5, "layer", "component")
    assert len(records) == 2 * 2 * 2 * 2 * (3 + 2)
    assert {record.intervention for record in records} == {
        "head_0",
        "head_1",
        "head_2",
        "aggregate_attention_output",
        "identity_noop",
    }
    assert {record.prompt_variant for record in records} == {"clean", "corrupt"}
    assert {record.direction for record in records} == {"first_to_last", "last_to_first"}
    assert {record.source_mode for record in records} == {"matched", "permuted"}
    assert all(
        record.oriented_causal_transfer == 0.0
        for record in records
        if record.intervention == "identity_noop"
    )
    aggregate = next(
        record
        for record in records
        if record.intervention == "aggregate_attention_output"
        and record.prompt_variant == "clean"
        and record.direction == "first_to_last"
        and record.source_mode == "matched"
    )
    assert aggregate.oriented_causal_transfer == pytest.approx(9.0)
    oriented = {
        (record.prompt_variant, record.direction): record.oriented_causal_transfer
        for record in records
        if record.intervention == "aggregate_attention_output"
        and record.source_mode == "matched"
        and record.family_id == "family-000"
    }
    assert oriented == {
        ("clean", "first_to_last"): pytest.approx(9.0),
        ("clean", "last_to_first"): pytest.approx(9.0),
        ("corrupt", "first_to_last"): pytest.approx(9.0),
        ("corrupt", "last_to_first"): pytest.approx(9.0),
    }
    permuted = next(
        record
        for record in records
        if record.intervention == "aggregate_attention_output"
        and record.prompt_variant == "clean"
        and record.direction == "first_to_last"
        and record.source_mode == "permuted"
        and record.family_id == "family-000"
    )
    assert permuted.source_family_id == "family-001"
    assert permuted.source_score == pytest.approx(12.3)
    assert permuted.oriented_causal_transfer == pytest.approx(9.3)


def test_deterministic_batch_rebalancing_and_impossible_inputs() -> None:
    pairs = tiny_pairs(9)
    first = deterministic_batches(pairs, 8)
    second = deterministic_batches(pairs, 8)
    assert [[pair.family_id for pair in batch] for batch in first] == [
        [f"family-{index:03d}" for index in range(7)],
        ["family-007", "family-008"],
    ]
    assert [[pair.family_id for pair in batch] for batch in first] == [
        [pair.family_id for pair in batch] for batch in second
    ]
    with pytest.raises(ValueError, match="at least 2"):
        deterministic_batches(pairs, 1)
    with pytest.raises(ValueError, match="one family"):
        deterministic_batches(tiny_pairs(1), 8)


def test_cli_rejects_batch_size_one_before_run(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_run(_: argparse.Namespace) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr("autocircuit.position_head_localization.run", fake_run)
    with pytest.raises(SystemExit, match="2"):
        main(["--batch-size", "1"])
    assert called is False


def test_rebalanced_scan_has_complete_source_mode_coverage() -> None:
    records = scan_heads(FakeModel(), tiny_pairs(9), 8, 5, "layer", "component")
    coverage = {
        (record.family_id, record.source_mode)
        for record in records
        if record.intervention == "head_0"
    }
    assert coverage == {
        (f"family-{family:03d}", mode) for family in range(9) for mode in ("matched", "permuted")
    }


def _record(family: int, intervention: str, mode: str, transfer: float) -> HeadRecord:
    head = int(intervention.removeprefix("head_")) if intervention.startswith("head_") else None
    return HeadRecord(
        family_id=f"family-{family}",
        prompt_variant="clean",
        direction="first_to_last",
        source_mode=mode,
        selected_layer=5,
        intervention=intervention,
        head_index=head,
        hook_site="site",
        source_family_id=f"source-{family}",
        destination_family_id=f"family-{family}",
        source_score=2.0,
        destination_baseline=1.0,
        patched_score=1.0 + transfer,
        oriented_causal_transfer=transfer,
        layer_run_manifest_hash="layer",
        component_run_manifest_hash="component",
    )


def test_summary_family_aggregation_ranking_bootstrap_and_audit() -> None:
    records: list[HeadRecord] = []
    for family in range(4):
        for intervention in (
            "head_0",
            "head_1",
            "head_2",
            *(
                "aggregate_attention_output",
                "identity_noop",
            ),
        ):
            for mode in ("matched", "permuted"):
                if intervention == "identity_noop":
                    transfer = 0.0
                elif intervention == "head_0":
                    transfer = 2.0 if mode == "matched" else 1.0
                elif intervention == "head_1":
                    transfer = 2.0 if mode == "matched" else 1.0
                elif intervention == "head_2":
                    transfer = 1.0 if mode == "matched" else 0.0
                else:
                    transfer = 4.0 if mode == "matched" else 1.0
                base = _record(family, intervention, mode, transfer)
                for variant in ("clean", "corrupt"):
                    for direction in ("first_to_last", "last_to_first"):
                        records.append(replace(base, prompt_variant=variant, direction=direction))
    previous = {
        "matched_mean_transfer": 3.5,
        "permuted_mean_transfer": 0.5,
        "family_specific_advantage": 3.0,
    }
    first = summarize(records, 3, previous, bootstrap_samples=100)
    second = summarize(records, 3, previous, bootstrap_samples=100)
    assert first == second
    assert [row["head_index"] for row in first["individual_head_ranking"]] == [0, 1, 2]
    assert first["individual_head_ranking"][0]["family_count"] == 4
    assert first["identity_noop_reference"]["matched_mean_transfer"] == 0.0
    audit = first["composition_audit"]
    assert audit["sum_individual_head_matched_mean_transfer"] == 5.0
    assert audit["matched_transfer_difference"] == 1.0
    assert audit["component_attention_matched_difference"] == 0.5
    assert audit["component_attention_permuted_difference"] == 0.5


def test_exact_factorial_validation_rejects_missing_duplicate_and_invalid_cells() -> None:
    complete = scan_heads(FakeModel(), tiny_pairs(), 2, 5, "layer", "component")
    validate_head_records(complete, 3, {pair.family_id for pair in tiny_pairs()})
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_head_records(complete[:-1], 3)
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_head_records([*complete, complete[0]], 3)
    with pytest.raises(RuntimeError, match="invalid prompt"):
        validate_head_records([replace(complete[0], prompt_variant="invalid"), *complete[1:]], 3)
    with pytest.raises(RuntimeError, match="invalid direction"):
        validate_head_records([replace(complete[0], direction="invalid"), *complete[1:]], 3)
    with pytest.raises(RuntimeError, match="invalid source"):
        validate_head_records([replace(complete[0], source_mode="invalid"), *complete[1:]], 3)
    with pytest.raises(RuntimeError, match="index"):
        validate_head_records([replace(complete[0], head_index=2), *complete[1:]], 3)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def component_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root.mkdir()
    discovery = {
        "seed": 42,
        "artifact_hashes": {"dataset": "abc"},
        "model": "pythia-70m",
        "model_id": "EleutherAI/pythia-70m",
        "tokenizer_id": "EleutherAI/pythia-70m",
        "requested_revision": "main",
        "load_revision": "main",
        "resolved_revision": None,
        "dtype": "torch.float32",
        "exact_revision_available": False,
    }
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
    layer = {
        "layer": 5,
        "hashes": {"manifest": "m", "summary": "s", "final": "f", "records": "r"},
        "identity": identity,
    }
    records: list[ComponentRecord] = []
    for family in range(120):
        for intervention_index, intervention in enumerate(
            (
                "residual_pre",
                "attention_output",
                "mlp_output",
                "attention_plus_mlp",
                "residual_post",
            )
        ):
            for variant in ("clean", "corrupt"):
                for direction in ("first_to_last", "last_to_first"):
                    for mode in ("matched", "permuted"):
                        transfer = float(intervention_index + 1) + (
                            0.5 if mode == "matched" else 0.0
                        )
                        records.append(
                            ComponentRecord(
                                family_id=f"family-{family:03d}",
                                prompt_variant=variant,
                                direction=direction,
                                source_mode=mode,
                                layer=5,
                                intervention=intervention,
                                source_family_id=f"source-{family:03d}-{mode}",
                                source_score=3.0,
                                destination_score=1.0,
                                patched_score=1.0 + transfer,
                                causal_transfer=transfer,
                            )
                        )
    records_path = root / "component_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record.__dict__, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = component_summarize(records, seed=42)
    final = {
        "status": COMPONENT_STATUS,
        "software_success": True,
        "activation_patching_performed": True,
        "circuit_found": False,
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "selected_layer": 5,
        "matched_family_count": 120,
        "record_count": len(records),
    }
    _write(root / "component_summary.json", summary)
    _write(root / "component_final_status.json", final)
    manifest = identity | {
        "complete": True,
        "protocol_version": COMPONENT_VERSION,
        "discovery_hashes": discovery["artifact_hashes"],
        "layer_run_hashes": {key: layer["hashes"][key] for key in ("manifest", "summary", "final")},
        "selected_layer": 5,
        "parallel_attn_mlp": True,
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "artifact_hashes": {
            "component_records.jsonl": sha256(records_path),
            "component_summary.json": sha256(root / "component_summary.json"),
            "component_final_status.json": sha256(root / "component_final_status.json"),
        },
    }
    _write(root / "component_manifest.json", manifest)
    return discovery, layer


def layer_fixture(root: Path) -> dict[str, Any]:
    root.mkdir()
    discovery = {
        "seed": 42,
        "artifact_hashes": {"dataset": "abc"},
        "model": "pythia-70m",
        "model_id": "EleutherAI/pythia-70m",
        "tokenizer_id": "EleutherAI/pythia-70m",
        "requested_revision": "main",
        "load_revision": "main",
        "resolved_revision": None,
        "dtype": "torch.float32",
        "exact_revision_available": False,
    }
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
    rows: list[dict[str, Any]] = []
    sites = ("blocks.4.hook_resid_post", "blocks.5.hook_resid_post")
    for family in range(2):
        for site_index, site in enumerate(sites):
            for variant in ("clean", "corrupt"):
                for direction in ("first_to_last", "last_to_first"):
                    for mode in ("matched", "permuted"):
                        transfer = (
                            (2.0 if site_index else 1.0)
                            if mode == "matched"
                            else (0.5 if site_index else 0.2)
                        )
                        rows.append(
                            {
                                "family_id": f"family-{family}",
                                "prompt_variant": variant,
                                "direction": direction,
                                "source_mode": mode,
                                "site": site,
                                "site_index": site_index,
                                "source_family_id": f"source-{family}-{mode}",
                                "destination_family_id": f"family-{family}",
                                "source_task_score": 3.0,
                                "destination_task_score": 1.0,
                                "patched_task_score": 1.0 + transfer,
                                "available_gap": 2.0,
                                "causal_transfer": transfer,
                                "normalized_transfer": transfer / 2,
                            }
                        )
    records_path = root / "layer_scan_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    ranking = [
        {
            "site": "blocks.5.hook_resid_post",
            "site_index": 1,
            "matched_mean_causal_transfer": 2.0,
            "permuted_mean_causal_transfer": 0.5,
            "family_specific_transfer_advantage": 1.5,
            "rank": 1,
        },
        {
            "site": "blocks.4.hook_resid_post",
            "site_index": 0,
            "matched_mean_causal_transfer": 1.0,
            "permuted_mean_causal_transfer": 0.2,
            "family_specific_transfer_advantage": 0.8,
            "rank": 2,
        },
    ]
    _write(
        root / "layer_scan_summary.json",
        {
            "protocol_version": "position-localization-0.1.0",
            "record_count": len(rows),
            "site_ranking": ranking,
        },
    )
    _write(
        root / "localization_final_status.json",
        {
            "status": "EXPLORATORY_LAYER_LOCALIZATION_COMPLETE",
            "software_success": True,
            "activation_patching_performed": True,
            "held_out_validation_reused": False,
            "held_out_test_opened": False,
            "circuit_found": False,
            "matched_family_count": 2,
            "record_count": len(rows),
        },
    )
    manifest = identity | {
        "complete": True,
        "protocol_version": "position-localization-0.1.0",
        "discovery_artifact_hashes": discovery["artifact_hashes"],
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "family_count": 2,
        "record_count": len(rows),
        "artifact_hashes": {
            name: sha256(root / name)
            for name in (
                "layer_scan_records.jsonl",
                "layer_scan_summary.json",
                "localization_final_status.json",
            )
        },
    }
    _write(root / "localization_manifest.json", manifest)
    return discovery


def test_layer_upstream_recomputes_ranking_and_requires_identity_and_hashes(
    tmp_path: Path,
) -> None:
    discovery = layer_fixture(tmp_path / "layer")
    root = tmp_path / "layer"
    assert verify_layer_run(root, discovery)["layer"] == 5
    manifest_path = root / "localization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifact_hashes"]["layer_scan_records.jsonl"]
    _write(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="required hashed artifacts"):
        verify_layer_run(root, discovery)


def test_layer_upstream_rejects_identity_and_raw_ranking_disagreement(tmp_path: Path) -> None:
    discovery = layer_fixture(tmp_path / "layer")
    root = tmp_path / "layer"
    manifest_path = root / "localization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_id"] = "wrong"
    _write(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="identity disagree"):
        verify_layer_run(root, discovery)


def test_component_upstream_verification_hash_layer_identity_and_ranking(tmp_path: Path) -> None:
    discovery, layer = component_fixture(tmp_path / "component")
    verified = verify_component_run(tmp_path / "component", discovery, layer)
    assert verified["layer"] == 5
    assert verified["attention"]["family_specific_advantage"] == 0.5

    summary = tmp_path / "component" / "component_summary.json"
    summary.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash"):
        verify_component_run(tmp_path / "component", discovery, layer)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("incomplete", "not eligible"),
        ("layer", "not eligible"),
        ("identity", "identity"),
        ("ranking", "ranking"),
    ],
)
def test_component_upstream_rejections(tmp_path: Path, change: str, message: str) -> None:
    discovery, layer = component_fixture(tmp_path / "component")
    root = tmp_path / "component"
    manifest_path = root / "component_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if change == "incomplete":
        manifest["complete"] = False
    elif change == "layer":
        manifest["selected_layer"] = 4
    elif change == "identity":
        manifest["model_id"] = "wrong"
    else:
        summary_path = root / "component_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["intervention_ranking"] = []
        _write(summary_path, summary)
        manifest["artifact_hashes"]["component_summary.json"] = sha256(summary_path)
    _write(manifest_path, manifest)
    with pytest.raises(RuntimeError, match=message):
        verify_component_run(root, discovery, layer)


def test_cli_has_only_discovery_inputs() -> None:
    parsed = build_parser().parse_args([])
    assert hasattr(parsed, "discovery_root")
    assert hasattr(parsed, "layer_root")
    assert hasattr(parsed, "component_root")
    assert not hasattr(parsed, "validation_root")
    assert not hasattr(parsed, "test_root")


def test_no_model_loading_when_upstream_verification_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = False

    def fail(_: Path) -> dict[str, Any]:
        raise RuntimeError("upstream failed")

    def factory(*_: str) -> Any:
        nonlocal loaded
        loaded = True
        raise AssertionError

    monkeypatch.setattr("autocircuit.position_head_localization._verify_frozen_discovery", fail)
    args = argparse.Namespace(
        discovery_root=tmp_path / "discovery",
        layer_root=tmp_path / "layer",
        component_root=tmp_path / "component",
        output=tmp_path / "output",
        device="cpu",
        batch_size=2,
        force=False,
    )
    with pytest.raises(RuntimeError, match="upstream failed"):
        run(args, factory)
    assert loaded is False


def _mock_verified_upstreams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> argparse.Namespace:
    discovery = {
        "seed": 42,
        "requested_revision": "main",
        "load_revision": "main",
        "model": "pythia-70m",
        "artifact_hashes": {"dataset": "frozen"},
    }
    identity = {"model": "pythia-70m"}
    layer = {"layer": 5, "hashes": {"manifest": "layer"}, "identity": identity}
    component = {
        "layer": 5,
        "hashes": {"manifest": "component"},
        "attention": {
            "matched_mean_transfer": 1.0,
            "permuted_mean_transfer": 0.0,
            "family_specific_advantage": 1.0,
        },
        "parallel_attn_mlp": True,
    }
    monkeypatch.setattr(
        "autocircuit.position_head_localization._verify_frozen_discovery", lambda _: discovery
    )
    monkeypatch.setattr("autocircuit.position_head_localization.verify_layer_run", lambda *_: layer)
    monkeypatch.setattr(
        "autocircuit.position_head_localization.verify_component_run", lambda *_: component
    )
    monkeypatch.setattr("autocircuit.position_head_localization.read_jsonl", lambda _: [])
    monkeypatch.setattr(
        "autocircuit.position_head_localization.build_first_last_pairs", lambda _: tiny_pairs()
    )
    monkeypatch.setattr(
        "autocircuit.position_head_localization._verify_validation_adapter",
        lambda *_: {"model_id": "EleutherAI/pythia-70m"},
    )
    monkeypatch.setattr("autocircuit.position_head_localization.read_jsonl", lambda _: [])
    return argparse.Namespace(
        discovery_root=tmp_path,
        layer_root=tmp_path,
        component_root=tmp_path,
        output=tmp_path / "output",
        device="cpu",
        batch_size=8,
        force=False,
    )


def test_preflight_failure_writes_incomplete_manifest_before_model_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _mock_verified_upstreams(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "autocircuit.position_head_localization.build_first_last_pairs", lambda _: tiny_pairs(1)
    )
    loaded = False

    def factory(*_: str) -> Any:
        nonlocal loaded
        loaded = True
        return SimpleNamespace(model=FakeModel())

    with pytest.raises(ValueError, match="one family"):
        run(args, factory)
    assert loaded is False
    root = args.output / "seed-42-main"
    manifest = json.loads((root / "head_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["software_success"] is False
    assert manifest["failure_stage"] == "preflight"
    assert manifest["exception_type"] == "ValueError"


def test_scanning_failure_writes_authoritative_incomplete_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _mock_verified_upstreams(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "autocircuit.position_head_localization.build_first_last_pairs", lambda _: tiny_pairs(120)
    )
    monkeypatch.setattr(
        "autocircuit.position_head_localization.scan_heads",
        lambda *_: (_ for _ in ()).throw(RuntimeError("scan failed")),
    )
    with pytest.raises(RuntimeError, match="scan failed"):
        run(args, lambda *_: SimpleNamespace(model=FakeModel()))
    root = args.output / "seed-42-main"
    manifest = json.loads((root / "head_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["failure_stage"] == "scanning"
    assert "head_final_status.json" in manifest["artifact_hashes"]
    assert "head_records.jsonl" not in manifest["artifact_hashes"]


def test_failure_after_partial_records_hashes_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _mock_verified_upstreams(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "autocircuit.position_head_localization.build_first_last_pairs", lambda _: tiny_pairs(120)
    )
    monkeypatch.setattr(
        "autocircuit.position_head_localization.scan_heads",
        lambda *_: [_record(0, "head_0", "matched", 1.0)],
    )
    with pytest.raises(RuntimeError, match="families"):
        run(args, lambda *_: SimpleNamespace(model=FakeModel()))
    root = args.output / "seed-42-main"
    manifest = json.loads((root / "head_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["failure_stage"] == "record_validation"
    assert "head_records.jsonl" in manifest["artifact_hashes"]
    assert sha256(root / "head_records.jsonl") == manifest["artifact_hashes"]["head_records.jsonl"]


def test_existing_output_refusal_and_narrow_force_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    discovery = {
        "seed": 42,
        "requested_revision": "main",
        "model": "pythia-70m",
        "load_revision": "main",
        "artifact_hashes": {},
    }
    identity = {"model": "pythia-70m"}
    layer = {"layer": 5, "hashes": {}, "identity": identity}
    component = {
        "layer": 5,
        "hashes": {},
        "attention": {},
        "parallel_attn_mlp": True,
    }
    monkeypatch.setattr(
        "autocircuit.position_head_localization._verify_frozen_discovery",
        lambda _: discovery,
    )
    monkeypatch.setattr("autocircuit.position_head_localization.verify_layer_run", lambda *_: layer)
    monkeypatch.setattr(
        "autocircuit.position_head_localization.verify_component_run", lambda *_: component
    )
    monkeypatch.setattr("autocircuit.position_head_localization.read_jsonl", lambda _: [])
    monkeypatch.setattr(
        "autocircuit.position_head_localization.build_first_last_pairs", lambda _: tiny_pairs()
    )
    output = tmp_path / "output"
    target = output / "seed-42-main"
    target.mkdir(parents=True)
    sibling = output / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")
    args = argparse.Namespace(
        discovery_root=tmp_path,
        layer_root=tmp_path,
        component_root=tmp_path,
        output=output,
        device="cpu",
        batch_size=2,
        force=False,
    )
    with pytest.raises(RuntimeError, match="use --force"):
        run(args, lambda *_: None)
    args.force = True
    with pytest.raises(RuntimeError, match="model failure"):
        run(args, lambda *_: (_ for _ in ()).throw(RuntimeError("model failure")))
    assert sibling.read_text(encoding="utf-8") == "keep"
    failure = json.loads((target / "head_final_status.json").read_text(encoding="utf-8"))
    assert failure["software_success"] is False
    assert failure["held_out_test_opened"] is False


def test_record_dataclass_preserves_upstream_provenance() -> None:
    record = _record(0, "head_0", "matched", 1.0)
    changed = replace(record, layer_run_manifest_hash="abc", component_run_manifest_hash="def")
    assert changed.layer_run_manifest_hash == "abc"
    assert changed.component_run_manifest_hash == "def"


def test_success_status_and_manifest_hash_every_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    discovery = {
        "seed": 42,
        "requested_revision": "main",
        "load_revision": "main",
        "model": "pythia-70m",
        "artifact_hashes": {"dataset": "frozen"},
    }
    identity = {"model": "pythia-70m"}
    layer = {"layer": 5, "hashes": {"manifest": "layer"}, "identity": identity}
    component = {
        "layer": 5,
        "hashes": {"manifest": "component"},
        "attention": {
            "matched_mean_transfer": 1.0,
            "permuted_mean_transfer": 0.0,
            "family_specific_advantage": 1.0,
        },
        "parallel_attn_mlp": True,
    }
    adapter = SimpleNamespace(model=FakeModel())
    monkeypatch.setattr(
        "autocircuit.position_head_localization._verify_frozen_discovery", lambda _: discovery
    )
    monkeypatch.setattr("autocircuit.position_head_localization.verify_layer_run", lambda *_: layer)
    monkeypatch.setattr(
        "autocircuit.position_head_localization.verify_component_run", lambda *_: component
    )
    monkeypatch.setattr(
        "autocircuit.position_head_localization._verify_validation_adapter",
        lambda *_: {"model_id": "EleutherAI/pythia-70m"},
    )
    monkeypatch.setattr("autocircuit.position_head_localization.read_jsonl", lambda _: [])
    monkeypatch.setattr(
        "autocircuit.position_head_localization.build_first_last_pairs", lambda _: tiny_pairs(120)
    )
    args = argparse.Namespace(
        discovery_root=tmp_path,
        layer_root=tmp_path,
        component_root=tmp_path,
        output=tmp_path / "output",
        device="cpu",
        batch_size=120,
        force=False,
    )
    final = run(args, lambda *_: adapter)
    root = args.output / "seed-42-main"
    assert final["software_success"] is True
    assert final["held_out_validation_reused"] is False
    assert final["held_out_test_opened"] is False
    assert final["activation_patching_performed"] is True
    assert final["scientific_confirmation"] is False
    assert final["circuit_found"] is False
    assert final["matched_family_count"] == 120
    assert final["record_count"] == 120 * 2 * 2 * 2 * (3 + 2)
    summary = json.loads((root / "head_summary.json").read_text(encoding="utf-8"))
    assert len(summary["individual_head_ranking"]) == 3
    assert summary["leading_head"] in {"head_0", "head_1", "head_2"}
    assert summary["aggregate_attention_reference"]["family_count"] == 120
    assert summary["identity_noop_reference"]["matched_mean_transfer"] == 0.0
    manifest = json.loads((root / "head_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["artifact_hashes"]) == {
        "head_records.jsonl",
        "head_summary.json",
        "head_localization.md",
        "head_final_status.json",
    }
    for relative, digest in manifest["artifact_hashes"].items():
        assert sha256(root / relative) == digest
