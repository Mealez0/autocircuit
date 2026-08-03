from __future__ import annotations

import argparse
import builtins
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import autocircuit.position_validation as position_validation
from autocircuit import pipeline
from autocircuit.baseline import make_example_result, make_failed_result, write_example_results
from autocircuit.datasets.associative_recall import ENTITIES, VALUES, ExamplePair
from autocircuit.position_study import (
    STUDY_VERSION,
    generate_matched_position_dataset,
    paired_position_effects,
    position_metrics,
    secondary_summaries,
    validate_matched_dataset,
    write_dataset,
)
from autocircuit.position_validation import (
    NOT_CONFIRMED_STATUS,
    VALIDATED_STATUS,
    VALIDATION_PROTOCOL_VERSION,
    generate_validation_dataset,
    validate_validation_dataset,
    validation_decision,
    validation_lexical_pools,
)


class FakeTokenizer:
    name_or_path = "offline"

    def __init__(self) -> None:
        self.mapping = {
            value: index
            for index, value in enumerate(sum((list(group) for group in VALUES), []), 10)
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.mapping[text]] if text in self.mapping else list(range(len(text.split())))


class FakeAdapter:
    tokenizer = FakeTokenizer()
    model_id = "EleutherAI/pythia-70m"
    revision = "main"
    device = "cpu"
    dtype = "float32"
    tokenizer_id = "offline"
    resolved_revision = None
    revision_resolution_error = "offline"

    def score(self, examples: list[ExamplePair], batch_size: int):
        del batch_size
        clean = {"first": 2.0, "interior": 2.0, "last": 0.0}
        return [
            make_example_result(
                item,
                clean[str(item.metadata["normalized_query_position"])],
                0.0,
                -1.0,
                0.0,
            )
            for item in examples
        ]


@pytest.fixture(autouse=True)
def reduced_bootstrap_for_offline_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production remains at 10,000; offline protocol tests inject 100 samples."""
    monkeypatch.setattr(
        pipeline,
        "paired_position_effects",
        lambda records, seed: paired_position_effects(records, seed=seed, samples=100),
    )


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def make_discovery(
    root: Path,
    *,
    eligible: bool = True,
    complete: bool = True,
    resolved_revision: str | None = None,
) -> None:
    root.mkdir(parents=True)
    tokenizer = FakeTokenizer()
    examples = generate_matched_position_dataset(tokenizer, 42)
    records = FakeAdapter().score(examples, 8)
    balance = validate_matched_dataset(examples, tokenizer)
    metrics = position_metrics(records)
    effects = paired_position_effects(records, seed=42, samples=100)
    secondary = secondary_summaries(records)
    write_dataset(root / "matched_dataset.jsonl", examples)
    write_example_results(root / "examples.jsonl", records)
    _json(root / "balance_report.json", balance)
    _json(root / "position_metrics.json", metrics)
    _json(root / "paired_position_effects.json", effects)
    _json(root / "secondary_summaries.json", secondary)
    (root / "position_study.md").write_text("# Frozen discovery study\n")
    final = {
        "status": (
            "QUERY_FIRST_DISCOVERY_ELIGIBLE_REQUIRES_HELD_OUT_VALIDATION"
            if eligible
            else "POSITION_EFFECT_UNCONFIRMED"
        ),
        "scientific_eligibility": eligible,
        "software_success": True,
        "held_out_splits_opened": False,
        "activation_patching_performed": False,
    }
    _json(root / "final_status.json", final)
    artifact_names = [
        "matched_dataset.jsonl",
        "examples.jsonl",
        "balance_report.json",
        "position_metrics.json",
        "paired_position_effects.json",
        "secondary_summaries.json",
        "position_study.md",
        "final_status.json",
    ]
    components = {
        "study_version": STUDY_VERSION,
        "model": "pythia-70m",
        "revision": "main",
        "device": "cpu",
        "batch_size": 8,
        "seed": 42,
    }
    manifest = {
        "complete": complete,
        "study_version": STUDY_VERSION,
        "input_fingerprint_components": components,
        "input_fingerprint": hashlib.sha256(
            json.dumps(components, sort_keys=True).encode()
        ).hexdigest(),
        "expected_example_count": 360,
        "processed_example_count": 360,
        "git_commit": "discovery-commit",
        "model_id": "EleutherAI/pythia-70m",
        "requested_revision": "main",
        "resolved_revision": resolved_revision,
        "revision_resolution_error": (
            None if resolved_revision else "offline fake has no resolved revision"
        ),
        "tokenizer_id": "offline",
        "dtype": "float32",
        "device": "cpu",
        "command_arguments": {
            "model": "pythia-70m",
            "revision": "main",
            "seed": 42,
            "device": "cpu",
            "batch_size": 8,
        },
        "artifact_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in artifact_names
        },
    }
    _json(root / "run_manifest.json", manifest)


def validation_args(tmp_path: Path, **changes: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "command": "position-validation",
        "device": "cpu",
        "batch_size": 8,
        "discovery_root": tmp_path / "discovery",
        "output": tmp_path / "validation",
        "confirm_open_validation": True,
        "resume": False,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_validation_cli_exposes_only_frozen_protocol_controls() -> None:
    parser = pipeline.build_parser()
    for forbidden in ("--force", "--seed", "--threshold", "--model", "--revision"):
        with pytest.raises(SystemExit):
            parser.parse_args(["position-validation", forbidden, "changed"])
    with pytest.raises(ValueError, match="only resolve validation"):
        validation_lexical_pools("test")


def test_validation_generation_is_stable_balanced_and_validation_only(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    first = generate_validation_dataset(tokenizer, 42)
    second = generate_validation_dataset(tokenizer, 42)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_dataset(a, first)
    write_dataset(b, second)
    assert a.read_bytes() == b.read_bytes()
    assert len(first) == 360
    assert {item.split for item in first} == {"validation"}
    text = "".join(item.clean_prompt + item.corrupt_prompt for item in first)
    assert all(entity in text for entity in ENTITIES[1])
    assert not any(entity in text for entity in ENTITIES[0] + ENTITIES[2])
    assert all(value.strip() in text for value in VALUES[1])
    assert not any(value.strip() in text for value in VALUES[0] + VALUES[2])
    families: dict[str, list[ExamplePair]] = defaultdict(list)
    for item in first:
        families[item.family_id].append(item)
    assert len(families) == 120
    assert all(
        {item.metadata["normalized_query_position"] for item in family}
        == set(("first", "interior", "last"))
        for family in families.values()
    )
    report = validate_validation_dataset(first, tokenizer, 42)
    assert report["ordered_pair_observed_count"] == 120
    assert report["ordered_pair_missing_count"] == 12
    assert set(report["target_counts"].values()) == {10}
    assert set(report["distractor_counts"].values()) == {10}
    assert set(report["query_entity_counts"].values()) == {12}
    # Regression guard: validation additions do not change discovery generation.
    discovery_a = tmp_path / "discovery-a.jsonl"
    discovery_b = tmp_path / "discovery-b.jsonl"
    write_dataset(discovery_a, generate_matched_position_dataset(tokenizer, 42))
    write_dataset(discovery_b, generate_matched_position_dataset(tokenizer, 42))
    assert discovery_a.read_bytes() == discovery_b.read_bytes()
    assert hashlib.sha256(discovery_a.read_bytes()).hexdigest() == (
        "68cb3ff72c0084f130c08784b4947aa31c305104f4e3930726c076a3ea48ddec"
    )


def test_validation_semantic_prompt_tamper_is_rejected() -> None:
    examples = generate_validation_dataset(FakeTokenizer(), 42)
    changed = examples.copy()
    changed[0] = replace(changed[0], corrupt_prompt=changed[0].corrupt_prompt + " ")
    with pytest.raises(ValueError, match="reconstruction"):
        validate_validation_dataset(changed, FakeTokenizer(), 42)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("entity", ENTITIES[0][0]),
        ("entity", ENTITIES[2][0]),
        ("value", VALUES[0][0]),
        ("value", VALUES[2][0]),
    ],
)
def test_validation_rejects_third_fact_from_nonvalidation_pool(
    field: str, replacement: str
) -> None:
    examples = generate_validation_dataset(FakeTokenizer(), 42)
    family_id = examples[0].family_id
    changed = examples.copy()
    for index, item in enumerate(changed):
        if item.family_id != family_id:
            continue
        assignments = [list(pair) for pair in item.metadata["assignments"]]
        assignments[2][0 if field == "entity" else 1] = replacement
        changed[index] = replace(item, metadata=item.metadata | {"assignments": assignments})
    with pytest.raises(ValueError, match="out-of-pool"):
        validate_validation_dataset(changed, FakeTokenizer(), 42)


def test_validation_rejects_identity_seed_and_namespace_tampering() -> None:
    examples = generate_validation_dataset(FakeTokenizer(), 42)

    def rejected(item: ExamplePair) -> None:
        changed = examples.copy()
        changed[0] = item
        with pytest.raises(ValueError):
            validate_validation_dataset(changed, FakeTokenizer(), 42)

    first = examples[0]
    rejected(first.__class__(**(asdict(first) | {"seed": 43})))
    rejected(replace(first, metadata=first.metadata | {"matched_family_id": "forged"}))
    rejected(replace(first, metadata=first.metadata | {"seed_namespace": "test"}))
    rejected(replace(first, example_id="forged-example"))

    family_id = first.family_id
    forged_family = examples.copy()
    for index, item in enumerate(forged_family):
        if item.family_id == family_id:
            forged_family[index] = replace(
                item,
                family_id="validation-family-forged",
                metadata=item.metadata | {"matched_family_id": "validation-family-forged"},
            )
    with pytest.raises(ValueError, match="family ID"):
        validate_validation_dataset(forged_family, FakeTokenizer(), 42)


def test_frozen_validation_decision_boundaries_and_supporting_independence() -> None:
    metrics = {
        position: {
            "clean_pairwise_accuracy": 0.8,
            "mean_clean_logit_difference": 1.0,
            "example_count": 120,
            "failed_count": 0,
        }
        for position in ("first", "interior", "last")
    }
    effects = {
        "complete_family_count": 120,
        "comparisons": {
            "first_minus_last_clean_logit_difference": {
                "estimate": 0.1,
                "ci_95": [0.001, 1.0],
            },
            "first_minus_interior_clean_logit_difference": {
                "estimate": -100.0,
                "ci_95": [-200.0, -50.0],
            },
        },
    }
    balance = {"all_invariants_passed": True}
    assert validation_decision(metrics, effects, balance) == VALIDATED_STATUS
    for position in ("first", "interior", "last"):
        failed = json.loads(json.dumps(metrics))
        failed[position].update({"example_count": 119, "failed_count": 1})
        assert validation_decision(failed, effects, balance) == NOT_CONFIRMED_STATUS
    incomplete = json.loads(json.dumps(effects))
    incomplete["complete_family_count"] = 119
    assert validation_decision(metrics, incomplete, balance) == NOT_CONFIRMED_STATUS
    for key, value in (("clean_pairwise_accuracy", 0.799), ("mean_clean_logit_difference", 0.999)):
        failed = json.loads(json.dumps(metrics))
        failed["first"][key] = value
        assert validation_decision(failed, effects, balance) == NOT_CONFIRMED_STATUS


def test_confirmation_and_discovery_contract_fail_before_model_loading(tmp_path: Path) -> None:
    make_discovery(tmp_path / "discovery")
    with pytest.raises(RuntimeError, match="confirm-open-validation"):
        pipeline.run_position_validation(
            validation_args(tmp_path, confirm_open_validation=False),
            lambda *unused: (_ for _ in ()).throw(AssertionError("model loaded")),
        )
    for mutation, message in (
        ("eligible", "eligible"),
        ("complete", "incomplete"),
        ("version", "version"),
    ):
        other = tmp_path / mutation
        make_discovery(other, eligible=mutation != "eligible", complete=mutation != "complete")
        if mutation == "version":
            manifest = json.loads((other / "run_manifest.json").read_text())
            manifest["study_version"] = "position-study-1.0.0"
            _json(other / "run_manifest.json", manifest)
        with pytest.raises(RuntimeError, match=message):
            pipeline.run_position_validation(
                validation_args(tmp_path, discovery_root=other),
                lambda *unused: (_ for _ in ()).throw(AssertionError("model loaded")),
            )
    artifact = tmp_path / "discovery" / "position_metrics.json"
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="hash verification"):
        pipeline.run_position_validation(
            validation_args(tmp_path),
            lambda *unused: (_ for _ in ()).throw(AssertionError("model loaded")),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_seed", "seed"),
        ("missing_revision", "revision"),
        ("altered_model", "model"),
        ("altered_components", "fingerprint"),
        ("recomputed_inconsistent", "command arguments"),
    ],
)
def test_discovery_manifest_identity_fields_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = tmp_path / mutation
    make_discovery(root)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    components = manifest["input_fingerprint_components"]
    if mutation == "missing_seed":
        components.pop("seed")
    elif mutation == "missing_revision":
        components.pop("revision")
    elif mutation == "altered_model":
        components["model"] = "other-model"
    elif mutation == "altered_components":
        components["seed"] = 43
    else:
        components["seed"] = 43
        manifest["input_fingerprint"] = hashlib.sha256(
            json.dumps(components, sort_keys=True).encode()
        ).hexdigest()
    _json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match=message):
        pipeline.run_position_validation(
            validation_args(tmp_path, discovery_root=root),
            lambda *unused: (_ for _ in ()).throw(AssertionError("model loaded")),
        )


def test_discovery_eligibility_is_recomputed_from_hash_consistent_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "discovery"
    make_discovery(root)
    metrics_path = root / "position_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["first"]["clean_pairwise_accuracy"] = 0.0
    _json(metrics_path, metrics)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_hashes"]["position_metrics.json"] = hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    _json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="summaries do not match raw scoring records"):
        pipeline.run_position_validation(
            validation_args(tmp_path),
            lambda *unused: (_ for _ in ()).throw(AssertionError("model loaded")),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clean_logit_difference", 999.0, "derived fields"),
        ("corrupt_logit_difference", -999.0, "derived fields"),
        ("family_id", "substituted-family", "family ID"),
        ("normalized_query_position", "last", "position"),
    ],
)
def test_discovery_raw_record_tampering_fails_before_model_loading(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    root = tmp_path / "discovery"
    make_discovery(root)
    records_path = root / "examples.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[0][field] = value
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_hashes"]["examples.jsonl"] = hashlib.sha256(
        records_path.read_bytes()
    ).hexdigest()
    _json(manifest_path, manifest)
    with pytest.raises((RuntimeError, ValueError), match=message):
        pipeline.run_position_validation(
            validation_args(tmp_path),
            lambda *unused: (_ for _ in ()).throw(AssertionError("model loaded")),
        )


@pytest.mark.parametrize(
    "name",
    ["position_metrics.json", "paired_position_effects.json", "secondary_summaries.json"],
)
def test_discovery_stored_summary_tampering_fails_before_model_loading(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / "discovery"
    make_discovery(root)
    path = root / name
    value = json.loads(path.read_text())
    value["tampered"] = True
    _json(path, value)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_hashes"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="summaries do not match"):
        pipeline.run_position_validation(
            validation_args(tmp_path),
            lambda *unused: (_ for _ in ()).throw(AssertionError("model loaded")),
        )


def test_resume_without_existing_output_fails_before_opening_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_discovery(tmp_path / "discovery")
    monkeypatch.setattr(
        pipeline,
        "generate_validation_dataset",
        lambda *unused: (_ for _ in ()).throw(AssertionError("validation generated")),
    )
    with pytest.raises(RuntimeError, match="(requires an existing|exact selected claim)"):
        pipeline.run_position_validation(
            validation_args(tmp_path, resume=True, confirm_open_validation=False),
            lambda *unused: (_ for _ in ()).throw(AssertionError("model loaded")),
        )


@pytest.mark.parametrize(("field", "message"), [("model", "model ID"), ("tokenizer", "tokenizer")])
def test_loaded_model_identity_mismatch_fails_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, message: str
) -> None:
    make_discovery(tmp_path / "discovery")
    monkeypatch.setattr(
        pipeline,
        "generate_validation_dataset",
        lambda *unused: (_ for _ in ()).throw(AssertionError("validation generated")),
    )

    class WrongAdapter(FakeAdapter):
        model_id = "EleutherAI/wrong" if field == "model" else FakeAdapter.model_id
        tokenizer_id = "wrong-tokenizer" if field == "tokenizer" else FakeAdapter.tokenizer_id

    with pytest.raises(RuntimeError, match=message):
        pipeline.run_position_validation(validation_args(tmp_path), lambda *unused: WrongAdapter())
    assert (tmp_path / "validation" / "seed-42-main" / "validation_manifest.json").is_file()


def test_frozen_resolved_revision_is_loaded_and_verified(tmp_path: Path) -> None:
    make_discovery(tmp_path / "discovery", resolved_revision="frozen-sha")

    class ExactAdapter(FakeAdapter):
        revision = "frozen-sha"
        resolved_revision = "frozen-sha"

    final = pipeline.run_position_validation(
        validation_args(tmp_path), lambda *unused: ExactAdapter()
    )
    assert final["software_success"] is True
    manifest = json.loads(
        (tmp_path / "validation" / "seed-42-main" / "validation_manifest.json").read_text()
    )
    assert manifest["load_revision"] == "frozen-sha"
    assert manifest["validation_resolved_revision"] == "frozen-sha"
    assert manifest["exact_revision_matching_succeeded"] is True


def test_resolved_revision_mismatch_fails_and_requested_revision_fallback_is_explicit(
    tmp_path: Path,
) -> None:
    mismatch = tmp_path / "mismatch"
    make_discovery(mismatch / "discovery", resolved_revision="frozen-sha")

    class WrongRevisionAdapter(FakeAdapter):
        revision = "frozen-sha"
        resolved_revision = "other-sha"

    with pytest.raises(RuntimeError, match="resolved revision"):
        pipeline.run_position_validation(
            validation_args(mismatch), lambda *unused: WrongRevisionAdapter()
        )
    fallback = tmp_path / "fallback"
    make_discovery(fallback / "discovery")
    contract = pipeline._verify_frozen_discovery(fallback / "discovery")
    assert contract["load_revision"] == "main"
    assert contract["exact_revision_available"] is False


def test_merged_position_study_manifest_without_resolution_error_is_supported(
    tmp_path: Path,
) -> None:
    root = tmp_path / "discovery"
    make_discovery(root)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["resolved_revision"] is None
    del manifest["revision_resolution_error"]
    _json(manifest_path, manifest)

    contract = pipeline._verify_frozen_discovery(root)
    assert contract["load_revision"] == "main"
    assert contract["revision_resolution_error"] == (
        "not recorded by position-study-2.0.0 manifest"
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_loaded_dtype_must_match_frozen_discovery(tmp_path: Path, dtype: str) -> None:
    make_discovery(tmp_path / "discovery")

    class WrongDtypeAdapter(FakeAdapter):
        pass

    WrongDtypeAdapter.dtype = dtype
    with pytest.raises(RuntimeError, match="dtype"):
        pipeline.run_position_validation(
            validation_args(tmp_path), lambda *unused: WrongDtypeAdapter()
        )
    manifest = json.loads(
        (tmp_path / "validation" / "seed-42-main" / "validation_manifest.json").read_text()
    )
    assert manifest["lifecycle_state"] == "receipt_claimed"


def test_empty_pre_manifest_output_is_recoverable_but_ambiguous_output_is_not(
    tmp_path: Path,
) -> None:
    make_discovery(tmp_path / "discovery")
    root = tmp_path / "validation" / "seed-42-main"
    root.mkdir(parents=True)
    final = pipeline.run_position_validation(
        validation_args(tmp_path), lambda *unused: FakeAdapter()
    )
    assert final["software_success"] is True

    other = tmp_path / "ambiguous"
    make_discovery(other / "discovery")
    ambiguous = other / "validation" / "seed-42-main"
    ambiguous.mkdir(parents=True)
    (ambiguous / "orphan.tmp").write_text("partial")
    with pytest.raises(RuntimeError, match="already exists"):
        pipeline.run_position_validation(
            validation_args(other), lambda *unused: FakeAdapter()
        )


def test_receipt_claim_interruption_is_resumable_only_at_selected_root(tmp_path: Path) -> None:
    make_discovery(tmp_path / "discovery")
    with pytest.raises(AssertionError, match="interrupted after receipt"):
        pipeline.run_position_validation(
            validation_args(tmp_path),
            lambda *unused: (_ for _ in ()).throw(AssertionError("interrupted after receipt")),
        )
    root = tmp_path / "validation" / "seed-42-main"
    assert root.is_dir()
    assert (root / "validation_manifest.json").is_file()
    assert (tmp_path / "discovery" / ".position_validation_receipt.json").is_file()
    with pytest.raises(RuntimeError, match="already has a validation receipt"):
        pipeline.run_position_validation(
            validation_args(tmp_path, output=tmp_path / "alternate"),
            lambda *unused: FakeAdapter(),
        )
    final = pipeline.run_position_validation(
        validation_args(tmp_path, resume=True, confirm_open_validation=False),
        lambda *unused: FakeAdapter(),
    )
    assert final["software_success"] is True


def test_matching_partial_fallback_receipt_recovers_only_untouched_initial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_discovery(tmp_path / "discovery")
    original = pipeline._claim_or_verify_validation_receipt

    def interrupt_with_partial(
        discovery_root: Path, payload: dict[str, object], *, resume: bool
    ) -> None:
        assert not resume
        partial = {
            "claim_id": payload["claim_id"],
            "selected_output_root": payload["selected_output_root"],
            "validation_fingerprint": payload["validation_fingerprint"],
        }
        _json(discovery_root / ".position_validation_receipt.json", partial)
        raise RuntimeError("fallback publication interrupted")

    monkeypatch.setattr(pipeline, "_claim_or_verify_validation_receipt", interrupt_with_partial)
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.run_position_validation(
            validation_args(tmp_path), lambda *unused: FakeAdapter()
        )
    root = tmp_path / "validation" / "seed-42-main"
    manifest = json.loads((root / "validation_manifest.json").read_text())
    assert manifest["lifecycle_state"] == "initialized"
    assert manifest["artifact_hashes"] == {}

    monkeypatch.setattr(pipeline, "_claim_or_verify_validation_receipt", original)
    final = pipeline.run_position_validation(
        validation_args(tmp_path, resume=True, confirm_open_validation=False),
        lambda *unused: FakeAdapter(),
    )
    assert final["software_success"] is True


def test_truncated_receipt_recovers_only_exact_untouched_initialized_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_discovery(tmp_path / "discovery")
    original = pipeline._claim_or_verify_validation_receipt

    def interrupt_with_truncated_receipt(
        discovery_root: Path, payload: dict[str, object], *, resume: bool
    ) -> None:
        del payload
        assert not resume
        (discovery_root / ".position_validation_receipt.json").write_text('{"claim_id":')
        raise RuntimeError("truncated receipt")

    monkeypatch.setattr(
        pipeline, "_claim_or_verify_validation_receipt", interrupt_with_truncated_receipt
    )
    with pytest.raises(RuntimeError, match="truncated"):
        pipeline.run_position_validation(
            validation_args(tmp_path), lambda *unused: FakeAdapter()
        )
    monkeypatch.setattr(pipeline, "_claim_or_verify_validation_receipt", original)
    root = tmp_path / "validation" / "seed-42-main"
    final = pipeline.run_position_validation(
        validation_args(tmp_path, resume=True, confirm_open_validation=False),
        lambda *unused: FakeAdapter(),
    )
    assert final["software_success"] is True

    manifest_path = root / "validation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["complete"] = False
    manifest["lifecycle_state"] = "initialized"
    manifest["artifact_hashes"] = {}
    manifest["model_id"] = "model-state-was-reached"
    _json(manifest_path, manifest)
    (tmp_path / "discovery" / ".position_validation_receipt.json").write_text("{")
    with pytest.raises(RuntimeError, match="receipt mismatch"):
        pipeline.run_position_validation(
            validation_args(tmp_path, resume=True, confirm_open_validation=False),
            lambda *unused: FakeAdapter(),
        )


def test_valid_receipt_reconstructs_only_its_missing_selected_output(tmp_path: Path) -> None:
    make_discovery(tmp_path / "discovery")
    with pytest.raises(AssertionError, match="claimed"):
        pipeline.run_position_validation(
            validation_args(tmp_path),
            lambda *unused: (_ for _ in ()).throw(AssertionError("claimed")),
        )
    root = tmp_path / "validation" / "seed-42-main"
    for path in root.iterdir():
        path.unlink()
    root.rmdir()
    final = pipeline.run_position_validation(
        validation_args(tmp_path, resume=True, confirm_open_validation=False),
        lambda *unused: FakeAdapter(),
    )
    assert final["software_success"] is True

    alternate = validation_args(
        tmp_path,
        output=tmp_path / "alternate",
        resume=True,
        confirm_open_validation=False,
    )
    with pytest.raises(RuntimeError, match="exact selected claim"):
        pipeline.run_position_validation(alternate, lambda *unused: FakeAdapter())


def test_receipt_windows_fallback_is_exclusive_and_leaves_no_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery = tmp_path / "discovery"
    discovery.mkdir()
    payload = {
        "manifest_schema_version": pipeline.VALIDATION_MANIFEST_SCHEMA_VERSION,
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "source_discovery_study_version": STUDY_VERSION,
        "frozen_discovery_artifact_hashes": {"manifest": "hash"},
        "validation_fingerprint": "fingerprint",
        "selected_output_root": str((tmp_path / "output").resolve()),
        "claim_id": "claim",
        "resume_state": "claimed_one_shot_validation",
    }
    monkeypatch.setattr(
        pipeline.os,
        "link",
        lambda *unused: (_ for _ in ()).throw(OSError(pipeline.errno.EPERM, "unsupported")),
    )
    pipeline._claim_or_verify_validation_receipt(discovery, payload, resume=False)
    pipeline._claim_or_verify_validation_receipt(discovery, payload, resume=True)
    with pytest.raises(RuntimeError, match="already has"):
        pipeline._claim_or_verify_validation_receipt(discovery, payload, resume=False)
    assert not list(discovery.glob("*.tmp"))
    receipt = discovery / ".position_validation_receipt.json"
    receipt.write_text("partial")
    with pytest.raises(json.JSONDecodeError):
        pipeline._claim_or_verify_validation_receipt(discovery, payload, resume=True)


def test_atomic_validation_json_preserves_previous_file_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "validation_manifest.json"
    path.write_text('{"old":true}\n')
    monkeypatch.setattr(
        pipeline.os,
        "replace",
        lambda *unused: (_ for _ in ()).throw(OSError("interrupted before publish")),
    )
    with pytest.raises(OSError, match="interrupted"):
        pipeline._validation_json(path, {"new": True})
    assert path.read_text() == '{"old":true}\n'
    assert not list(tmp_path.glob(".validation_manifest.json.*"))


def test_offline_one_shot_resume_firewall_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_discovery(tmp_path / "discovery")
    originals = {
        "builtin": builtins.open,
        "open": Path.open,
        "read_text": Path.read_text,
        "read_bytes": Path.read_bytes,
        "stat": Path.stat,
    }

    def reject(value: object) -> None:
        path = Path(value)
        if path.name == "test.jsonl":
            raise AssertionError("test split file access")

    def builtin_open(file: object, *args: object, **kwargs: object):
        reject(file)
        return originals["builtin"](file, *args, **kwargs)

    def path_call(name: str):
        def guarded(path: Path, *args: object, **kwargs: object):
            reject(path)
            return originals[name](path, *args, **kwargs)

        return guarded

    monkeypatch.setattr(builtins, "open", builtin_open)
    for name in ("open", "read_text", "read_bytes", "stat"):
        monkeypatch.setattr(Path, name, path_call(name))
    original_generator = pipeline.generate_split

    def guarded_generator(split: str, *args: object, **kwargs: object):
        if split == "test":
            raise AssertionError("test population generated")
        return original_generator(split, *args, **kwargs)

    monkeypatch.setattr(pipeline, "generate_split", guarded_generator)
    original_pool_resolver = position_validation.validation_lexical_pools

    def guarded_pool_resolver(split: str = "validation"):
        if split != "validation":
            raise AssertionError("non-validation lexical pool selected")
        pools = original_pool_resolver(split)
        if pools == (ENTITIES[2], VALUES[2]):
            raise AssertionError("test lexical pool selected")
        return pools

    monkeypatch.setattr(position_validation, "validation_lexical_pools", guarded_pool_resolver)
    final = pipeline.run_position_validation(
        validation_args(tmp_path), lambda *unused: FakeAdapter()
    )
    assert final["held_out_validation_opened"] and not final["held_out_test_opened"]
    root = tmp_path / "validation" / "seed-42-main"
    required = {
        "validation_manifest.json",
        "frozen_discovery_contract.json",
        "validation_dataset.jsonl",
        "validation_examples.jsonl",
        "validation_balance_report.json",
        "validation_position_metrics.json",
        "validation_paired_effects.json",
        "validation_secondary_summaries.json",
        "validation_report.md",
        "validation_final_status.json",
    }
    assert required == {path.name for path in root.iterdir()}
    assert all(
        "NaN" not in path.read_text() and "Infinity" not in path.read_text()
        for path in root.glob("*.json")
    )
    resumed = validation_args(tmp_path, resume=True, confirm_open_validation=False)
    manifest_path = root / "validation_manifest.json"
    interrupted = json.loads(manifest_path.read_text())
    interrupted["complete"] = False
    _json(manifest_path, interrupted)
    alternate = validation_args(tmp_path, output=tmp_path / "alternate")
    with pytest.raises(RuntimeError, match="already has a validation receipt"):
        pipeline.run_position_validation(alternate, lambda *unused: FakeAdapter())
    with pytest.raises(RuntimeError, match="(requires an existing|exact selected claim)"):
        pipeline.run_position_validation(
            validation_args(
                tmp_path,
                output=tmp_path / "alternate",
                resume=True,
                confirm_open_validation=False,
            ),
            lambda *unused: FakeAdapter(),
        )
    assert pipeline.run_position_validation(resumed, lambda *unused: FakeAdapter()) == final
    assert (
        pipeline.run_position_validation(
            resumed, lambda *unused: (_ for _ in ()).throw(AssertionError("model reloaded"))
        )
        == final
    )
    with pytest.raises(RuntimeError, match="cannot be recomputed"):
        pipeline.run_position_validation(validation_args(tmp_path), lambda *unused: FakeAdapter())
    with pytest.raises(RuntimeError, match="already has a validation receipt"):
        pipeline.run_position_validation(
            validation_args(tmp_path, output=tmp_path / "completed-alternate"),
            lambda *unused: FakeAdapter(),
        )
    manifest = json.loads(manifest_path.read_text())
    manifest["validation_protocol_version"] = "old"
    _json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="protocol version"):
        pipeline.run_position_validation(resumed, lambda *unused: FakeAdapter())
    manifest["validation_protocol_version"] = VALIDATION_PROTOCOL_VERSION
    _json(manifest_path, manifest)
    consistent_manifest = manifest_path.read_bytes()

    def inconsistent_artifact(name: str, mutate: object) -> None:
        path = root / name
        original = path.read_bytes()
        value = json.loads(original)
        assert callable(mutate)
        mutate(value)
        _json(path, value)
        current_manifest = json.loads(consistent_manifest)
        current_manifest["artifact_hashes"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        _json(manifest_path, current_manifest)
        with pytest.raises(
            RuntimeError, match="(semantically inconsistent|summaries do not match)"
        ):
            pipeline.run_position_validation(resumed, lambda *unused: FakeAdapter())
        path.write_bytes(original)
        manifest_path.write_bytes(consistent_manifest)

    inconsistent_artifact(
        "validation_position_metrics.json",
        lambda value: value["first"].update({"clean_pairwise_accuracy": 0.79}),
    )
    inconsistent_artifact(
        "validation_paired_effects.json",
        lambda value: value["comparisons"]["first_minus_last_clean_logit_difference"].update(
            {"ci_95": [0.0, 1.0]}
        ),
    )
    inconsistent_artifact(
        "validation_final_status.json",
        lambda value: value.update(
            {"status": NOT_CONFIRMED_STATUS, "scientific_validation_confirmed": False}
        ),
    )
    inconsistent_artifact(
        "validation_final_status.json",
        lambda value: value.update({"software_success": False}),
    )
    count_manifest = json.loads(consistent_manifest)
    count_manifest["processed_example_count"] = 359
    _json(manifest_path, count_manifest)
    with pytest.raises(RuntimeError, match="semantically inconsistent"):
        pipeline.run_position_validation(resumed, lambda *unused: FakeAdapter())
    manifest_path.write_bytes(consistent_manifest)

    receipt = tmp_path / "discovery" / ".position_validation_receipt.json"
    recorded_receipt = json.loads(receipt.read_text())
    recorded_receipt["selected_output_root"] = "tampered"
    _json(receipt, recorded_receipt)
    with pytest.raises(RuntimeError, match="receipt mismatch"):
        pipeline.run_position_validation(resumed, lambda *unused: FakeAdapter())


def test_bad_scoring_identity_not_checkpointed_and_scientific_failure_is_success(
    tmp_path: Path,
) -> None:
    make_discovery(tmp_path / "discovery")

    class DuplicateAdapter(FakeAdapter):
        def score(self, examples: list[ExamplePair], batch_size: int):
            records = super().score(examples, batch_size)
            return records[:-1] + [records[0]]

    with pytest.raises(ValueError, match="not unique"):
        pipeline.run_position_validation(
            validation_args(tmp_path), lambda *unused: DuplicateAdapter()
        )
    root = tmp_path / "validation" / "seed-42-main"
    assert not (root / "validation_examples.jsonl").exists()

    other = tmp_path / "failure"
    make_discovery(other / "discovery")

    class FailingAdapter(FakeAdapter):
        def score(self, examples: list[ExamplePair], batch_size: int):
            records = super().score(examples, batch_size)
            for position in ("first", "interior", "last"):
                index = next(
                    i
                    for i, item in enumerate(examples)
                    if item.metadata["normalized_query_position"] == position
                )
                records[index] = make_failed_result(examples[index], "offline failure")
            return records

    final = pipeline.run_position_validation(
        validation_args(other), lambda *unused: FailingAdapter()
    )
    assert final["software_success"] and final["status"] == NOT_CONFIRMED_STATUS


def test_entirely_failed_position_is_reported_as_scientific_failure(tmp_path: Path) -> None:
    make_discovery(tmp_path / "discovery")

    class FailedLastAdapter(FakeAdapter):
        def score(self, examples: list[ExamplePair], batch_size: int):
            records = super().score(examples, batch_size)
            return [
                make_failed_result(item, "complete last-position failure")
                if item.metadata["normalized_query_position"] == "last"
                else record
                for item, record in zip(examples, records, strict=True)
            ]

    final = pipeline.run_position_validation(
        validation_args(tmp_path), lambda *unused: FailedLastAdapter()
    )
    root = tmp_path / "validation" / "seed-42-main"
    assert final["software_success"] is True
    assert final["status"] == NOT_CONFIRMED_STATUS
    assert final["scientific_validation_confirmed"] is False
    assert final["held_out_test_opened"] is False
    assert {
        "validation_manifest.json",
        "frozen_discovery_contract.json",
        "validation_dataset.jsonl",
        "validation_examples.jsonl",
        "validation_balance_report.json",
        "validation_position_metrics.json",
        "validation_paired_effects.json",
        "validation_secondary_summaries.json",
        "validation_report.md",
        "validation_final_status.json",
    } == {path.name for path in root.iterdir()}
    report = (root / "validation_report.md").read_text()
    assert "| last | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 120 |" in report


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clean_logit_difference", 999.0),
        ("corrupt_logit_difference", -999.0),
        ("processing_status", "error"),
        ("error", "substituted failure"),
    ],
)
def test_completed_resume_rejects_hash_consistent_raw_record_tampering(
    tmp_path: Path, field: str, value: object
) -> None:
    make_discovery(tmp_path / "discovery")
    pipeline.run_position_validation(validation_args(tmp_path), lambda *unused: FakeAdapter())
    root = tmp_path / "validation" / "seed-42-main"
    records_path = root / "validation_examples.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[0][field] = value
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    manifest_path = root / "validation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_hashes"]["validation_examples.jsonl"] = hashlib.sha256(
        records_path.read_bytes()
    ).hexdigest()
    _json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="inconsistent"):
        pipeline.run_position_validation(
            validation_args(tmp_path, resume=True, confirm_open_validation=False),
            lambda *unused: (_ for _ in ()).throw(AssertionError("model loaded")),
        )
