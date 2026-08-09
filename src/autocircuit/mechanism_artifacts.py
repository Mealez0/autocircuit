"""Serializable, hash-checked artifacts for discovery-only causal alignments."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import torch

from autocircuit.causal_subspace import ActivationAlignment, CausalSubspace

ALIGNMENT_ARTIFACT_VERSION = "mechanism-alignments-0.1.0"
INTERPRETATION_SCOPE = "exploratory_discovery_only"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _guard_discovery_artifact(value: Mapping[str, Any], label: str) -> None:
    if value.get("interpretation_scope") != INTERPRETATION_SCOPE:
        raise ValueError(f"{label} is not discovery-only exploratory data")
    if value.get("held_out_validation_reused") is not False:
        raise ValueError(f"{label} indicates held-out validation reuse")
    if value.get("held_out_test_opened") is not False:
        raise ValueError(f"{label} indicates held-out test access")
    if value.get("scientific_confirmation") is not False:
        raise ValueError(f"{label} cannot claim scientific confirmation")
    if value.get("circuit_found") is not False:
        raise ValueError(f"{label} cannot claim a discovered circuit")


def _subspace_to_dict(subspace: CausalSubspace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "variable_name": subspace.variable_name,
        "basis": subspace.basis.tolist(),
        "class_labels": list(subspace.class_labels),
        "fit_sample_count": subspace.fit_sample_count,
        "singular_values": list(subspace.singular_values),
        "between_class_energy_fraction": subspace.between_class_energy_fraction,
        "evidence_status": subspace.evidence_status,
        "fingerprint": subspace.fingerprint(),
    }


def _subspace_from_dict(value: Mapping[str, Any]) -> CausalSubspace:
    if value.get("schema_version") != 1:
        raise ValueError("unsupported causal subspace artifact schema")
    basis = value.get("basis")
    class_labels = value.get("class_labels")
    singular_values = value.get("singular_values")
    if not isinstance(basis, list) or not basis or not all(isinstance(row, list) for row in basis):
        raise ValueError("causal subspace artifact basis is malformed")
    if not isinstance(class_labels, list) or not all(
        isinstance(label, str) for label in class_labels
    ):
        raise ValueError("causal subspace artifact class labels are malformed")
    if not isinstance(singular_values, list) or not all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in singular_values
    ):
        raise ValueError("causal subspace artifact singular values are malformed")
    variable_name = value.get("variable_name")
    fit_sample_count = value.get("fit_sample_count")
    energy = value.get("between_class_energy_fraction")
    evidence_status = value.get("evidence_status")
    if not isinstance(variable_name, str):
        raise ValueError("causal subspace artifact variable is malformed")
    if not isinstance(fit_sample_count, int) or isinstance(fit_sample_count, bool):
        raise ValueError("causal subspace artifact sample count is malformed")
    if not isinstance(energy, int | float) or isinstance(energy, bool):
        raise ValueError("causal subspace artifact energy is malformed")
    if not isinstance(evidence_status, str):
        raise ValueError("causal subspace artifact evidence status is malformed")
    try:
        basis_tensor = torch.tensor(basis, dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("causal subspace artifact basis is not numeric") from exc
    subspace = CausalSubspace(
        variable_name=variable_name,
        basis=basis_tensor,
        class_labels=tuple(class_labels),
        fit_sample_count=fit_sample_count,
        singular_values=tuple(float(item) for item in singular_values),
        between_class_energy_fraction=float(energy),
        evidence_status=evidence_status,
    )
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != subspace.fingerprint():
        raise ValueError("causal subspace artifact fingerprint verification failed")
    return subspace


def _alignment_to_dict(alignment: ActivationAlignment) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "alignment_id": alignment.alignment_id,
        "variable_name": alignment.variable_name,
        "hook_site": alignment.hook_site,
        "position_index": alignment.position_index,
        "selected_heads": list(alignment.selected_heads),
        "subspace": _subspace_to_dict(alignment.subspace),
        "evidence_status": alignment.evidence_status,
        "fingerprint": alignment.fingerprint(),
    }


def _alignment_from_dict(value: Mapping[str, Any]) -> ActivationAlignment:
    if value.get("schema_version") != 1:
        raise ValueError("unsupported activation alignment artifact schema")
    alignment_id = value.get("alignment_id")
    variable_name = value.get("variable_name")
    hook_site = value.get("hook_site")
    position_index = value.get("position_index")
    selected_heads = value.get("selected_heads")
    subspace_value = value.get("subspace")
    evidence_status = value.get("evidence_status")
    if not all(isinstance(item, str) for item in (alignment_id, variable_name, hook_site)):
        raise ValueError("activation alignment identity fields are malformed")
    if not isinstance(position_index, int) or isinstance(position_index, bool):
        raise ValueError("activation alignment position is malformed")
    if not isinstance(selected_heads, list) or not all(
        isinstance(head, int) and not isinstance(head, bool) for head in selected_heads
    ):
        raise ValueError("activation alignment head indexes are malformed")
    if not isinstance(subspace_value, dict):
        raise ValueError("activation alignment subspace is malformed")
    if not isinstance(evidence_status, str):
        raise ValueError("activation alignment evidence status is malformed")
    alignment = ActivationAlignment(
        alignment_id=alignment_id,
        variable_name=variable_name,
        hook_site=hook_site,
        position_index=position_index,
        selected_heads=tuple(selected_heads),
        subspace=_subspace_from_dict(subspace_value),
        evidence_status=evidence_status,
    )
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != alignment.fingerprint():
        raise ValueError("activation alignment artifact fingerprint verification failed")
    return alignment


def _source_artifacts(value: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    if not value:
        raise ValueError("alignment artifact requires source artifact provenance")
    output: dict[str, dict[str, str]] = {}
    for name, record in sorted(value.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("alignment source artifact name is malformed")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError("alignment source artifact path is malformed")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("alignment source artifact SHA-256 is malformed")
        output[name] = {"path": path, "sha256": digest}
    return output


def _model_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    required = ("model_id", "requested_revision", "tokenizer_id", "dtype")
    if any(not isinstance(value.get(key), str) or not value.get(key) for key in required):
        raise ValueError("alignment model identity is incomplete")
    resolved = value.get("resolved_revision")
    if resolved is not None and (not isinstance(resolved, str) or not resolved):
        raise ValueError("alignment resolved revision is malformed")
    return {
        "model_id": value["model_id"],
        "requested_revision": value["requested_revision"],
        "resolved_revision": resolved,
        "tokenizer_id": value["tokenizer_id"],
        "dtype": value["dtype"],
    }


def build_alignment_manifest(
    alignments: Mapping[str, ActivationAlignment],
    *,
    source_artifacts: Mapping[str, Mapping[str, str]],
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize fitted discovery alignments with hashes and model provenance."""

    if not alignments:
        raise ValueError("alignment manifest requires at least one activation alignment")
    serialized: dict[str, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    for key, alignment in sorted(alignments.items()):
        if not isinstance(key, str) or not key:
            raise ValueError("alignment manifest key is malformed")
        if not isinstance(alignment, ActivationAlignment):
            raise ValueError("alignment manifest values must be ActivationAlignment objects")
        fingerprint = alignment.fingerprint()
        if fingerprint in fingerprints:
            raise ValueError("alignment manifest contains duplicate alignment fingerprints")
        fingerprints.add(fingerprint)
        serialized[key] = _alignment_to_dict(alignment)
    return {
        "schema_version": 1,
        "artifact_version": ALIGNMENT_ARTIFACT_VERSION,
        "interpretation_scope": INTERPRETATION_SCOPE,
        "source_artifacts": _source_artifacts(source_artifacts),
        "model_identity": _model_identity(model_identity),
        "alignments": serialized,
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }


def load_alignment_manifest(value: Mapping[str, Any]) -> dict[str, ActivationAlignment]:
    """Validate and reconstruct an alignment manifest, rejecting tampering fail-closed."""

    _guard_discovery_artifact(value, "alignment manifest")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_version") != ALIGNMENT_ARTIFACT_VERSION
    ):
        raise ValueError("unsupported alignment manifest schema or version")
    sources = value.get("source_artifacts")
    identity = value.get("model_identity")
    raw_alignments = value.get("alignments")
    if not isinstance(sources, dict):
        raise ValueError("alignment manifest source provenance is malformed")
    if not isinstance(identity, dict):
        raise ValueError("alignment manifest model identity is malformed")
    if not isinstance(raw_alignments, dict) or not raw_alignments:
        raise ValueError("alignment manifest contains no alignments")
    _source_artifacts(sources)
    _model_identity(identity)
    output: dict[str, ActivationAlignment] = {}
    fingerprints: set[str] = set()
    for key, raw in sorted(raw_alignments.items()):
        if not isinstance(key, str) or not key or not isinstance(raw, dict):
            raise ValueError("alignment manifest entry is malformed")
        alignment = _alignment_from_dict(raw)
        fingerprint = alignment.fingerprint()
        if fingerprint in fingerprints:
            raise ValueError("alignment manifest contains duplicate alignment fingerprints")
        fingerprints.add(fingerprint)
        output[key] = alignment
    return output
