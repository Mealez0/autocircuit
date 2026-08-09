"""Causal-variable subspaces for discovery-only interchange interventions.

A subspace is an alignment hypothesis, not evidence that a neural component has
an interpreted role. The primitives here deliberately operate on activations
without accessing datasets, models, validation data, or the untouched test split.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass

import torch

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_SLUG = re.compile(r"^[a-z][a-z0-9_-]*$")
_ORTHONORMAL_ATOL = 1e-10


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _slug(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase slug")
    return value


def _canonical_basis(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("subspace basis must be a rank-2 tensor")
    basis = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if basis.shape[0] <= 0 or basis.shape[1] <= 0:
        raise ValueError("subspace basis dimensions must be positive")
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("subspace basis must be finite")
    gram = basis.T @ basis
    identity = torch.eye(basis.shape[1], dtype=torch.float64)
    if not torch.allclose(gram, identity, atol=_ORTHONORMAL_ATOL, rtol=0.0):
        raise ValueError("subspace basis must be orthonormal")
    return basis


def _canonicalize_column_signs(basis: torch.Tensor) -> torch.Tensor:
    canonical = basis.clone()
    for column in range(canonical.shape[1]):
        vector = canonical[:, column]
        pivot = int(torch.argmax(torch.abs(vector)).item())
        if float(vector[pivot]) < 0.0:
            canonical[:, column] *= -1.0
    return canonical


@dataclass(frozen=True)
class CausalSubspace:
    """An orthonormal activation subspace aligned to one proposed causal variable."""

    variable_name: str
    basis: torch.Tensor
    class_labels: tuple[str, ...]
    fit_sample_count: int
    singular_values: tuple[float, ...]
    between_class_energy_fraction: float
    evidence_status: str = "proposal_only_not_evidence"

    def __post_init__(self) -> None:
        _identifier(self.variable_name, "variable name")
        canonical = _canonical_basis(self.basis)
        object.__setattr__(self, "basis", canonical)
        if len(self.class_labels) < 2 or len(set(self.class_labels)) != len(self.class_labels):
            raise ValueError("subspace class labels must contain at least two unique labels")
        if any(not isinstance(label, str) or not label for label in self.class_labels):
            raise ValueError("subspace class labels must be non-empty strings")
        if (
            not isinstance(self.fit_sample_count, int)
            or isinstance(self.fit_sample_count, bool)
            or self.fit_sample_count < len(self.class_labels)
        ):
            raise ValueError("fit sample count must cover every class")
        if len(self.singular_values) < self.rank:
            raise ValueError("singular values must cover the fitted subspace rank")
        if any(value < 0.0 or not math.isfinite(value) for value in self.singular_values):
            raise ValueError("singular values must be finite and non-negative")
        if (
            isinstance(self.between_class_energy_fraction, bool)
            or not isinstance(self.between_class_energy_fraction, (int, float))
            or not math.isfinite(float(self.between_class_energy_fraction))
            or not 0.0 <= float(self.between_class_energy_fraction) <= 1.0
        ):
            raise ValueError("between-class energy fraction must lie in [0, 1]")
        if self.evidence_status != "proposal_only_not_evidence":
            raise ValueError("causal subspaces are proposal-only until intervened on")

    @property
    def feature_dim(self) -> int:
        return int(self.basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def _compatible_basis(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape[-1] != self.feature_dim:
            raise ValueError("activation feature dimension does not match subspace")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("activation values must be finite")
        return self.basis.to(device=value.device, dtype=value.dtype)

    def project(self, value: torch.Tensor) -> torch.Tensor:
        """Project the final tensor dimension onto this aligned subspace."""

        basis = self._compatible_basis(value)
        return (value @ basis) @ basis.T

    def orthogonal_component(self, value: torch.Tensor) -> torch.Tensor:
        """Return the component deliberately preserved by an interchange."""

        return value - self.project(value)

    def interchange(self, base: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
        """Replace only the aligned component of ``base`` with ``donor``'s component."""

        if not isinstance(base, torch.Tensor) or not isinstance(donor, torch.Tensor):
            raise ValueError("base and donor activations must be tensors")
        if base.shape != donor.shape:
            raise RuntimeError("base and donor activation shapes differ")
        self._compatible_basis(base)
        if donor.device != base.device or donor.dtype != base.dtype:
            donor = donor.to(device=base.device, dtype=base.dtype)
        self._compatible_basis(donor)
        return base + self.project(donor - base)

    def fingerprint(self) -> str:
        """Return a byte-stable fingerprint independent of tensor storage details."""

        digest = hashlib.sha256()
        digest.update(b"autocircuit-causal-subspace-v1\0")
        for value in (
            self.variable_name,
            *self.class_labels,
            str(self.fit_sample_count),
            str(self.feature_dim),
            str(self.rank),
            self.evidence_status,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for singular_value in self.singular_values:
            digest.update(struct.pack("<d", float(singular_value)))
        digest.update(struct.pack("<d", float(self.between_class_energy_fraction)))
        for coordinate in self.basis.reshape(-1).tolist():
            digest.update(struct.pack("<d", float(coordinate)))
        return digest.hexdigest()


@dataclass(frozen=True)
class ActivationAlignment:
    """Map a proposed causal variable subspace to an exact TransformerLens hook slice."""

    alignment_id: str
    variable_name: str
    hook_site: str
    position_index: int
    selected_heads: tuple[int, ...]
    subspace: CausalSubspace
    evidence_status: str = "proposal_only_not_evidence"

    def __post_init__(self) -> None:
        _slug(self.alignment_id, "alignment id")
        _identifier(self.variable_name, "alignment variable name")
        if self.variable_name != self.subspace.variable_name:
            raise ValueError("alignment variable and subspace variable disagree")
        if not isinstance(self.hook_site, str) or not self.hook_site:
            raise ValueError("alignment hook site must be non-empty")
        if not isinstance(self.position_index, int) or isinstance(self.position_index, bool):
            raise ValueError("alignment position index must be an integer")
        if len(self.selected_heads) != len(set(self.selected_heads)) or any(
            not isinstance(head, int) or isinstance(head, bool) or head < 0
            for head in self.selected_heads
        ):
            raise ValueError("alignment head indexes must be unique non-negative integers")
        if self.evidence_status != "proposal_only_not_evidence":
            raise ValueError("activation alignments are proposal-only until causally tested")

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"autocircuit-activation-alignment-v1\0")
        for value in (
            self.alignment_id,
            self.variable_name,
            self.hook_site,
            str(self.position_index),
            *(str(head) for head in self.selected_heads),
            self.subspace.fingerprint(),
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


def fit_causal_subspace(
    variable_name: str,
    activations: torch.Tensor,
    labels: list[str] | tuple[str, ...],
    *,
    rank: int,
) -> CausalSubspace:
    """Fit a deterministic supervised between-class subspace on discovery activations.

    The estimator uses weighted class-centroid contrasts and SVD. It intentionally
    does not claim that linear separability establishes causality; the returned
    object remains proposal-only until interchange interventions test it.
    """

    _identifier(variable_name, "variable name")
    if not isinstance(activations, torch.Tensor) or activations.ndim != 2:
        raise ValueError("activations must have shape [sample, feature]")
    if activations.shape[0] != len(labels):
        raise ValueError("activation sample count and label count disagree")
    if activations.shape[0] < 2 or activations.shape[1] < 1:
        raise ValueError("causal subspace fitting requires samples and features")
    values = activations.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("activations must be finite")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("labels must be non-empty strings")
    classes = tuple(sorted(set(labels)))
    if len(classes) < 2:
        raise ValueError("causal subspace fitting requires at least two classes")
    max_rank = min(int(values.shape[1]), len(classes) - 1)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0 or rank > max_rank:
        raise ValueError(f"rank must lie in [1, {max_rank}]")

    global_mean = values.mean(dim=0)
    centroid_rows: list[torch.Tensor] = []
    for label in classes:
        indexes = [index for index, item in enumerate(labels) if item == label]
        class_values = values[indexes]
        centroid_rows.append(math.sqrt(len(indexes)) * (class_values.mean(dim=0) - global_mean))
    contrasts = torch.stack(centroid_rows)
    _, singular_values, vh = torch.linalg.svd(contrasts, full_matrices=False)
    energy = singular_values.square()
    total_energy = float(energy.sum())
    if not math.isfinite(total_energy) or total_energy <= 1e-18:
        raise ValueError("labels have no finite between-class activation signal")
    basis = _canonicalize_column_signs(vh[:rank].T.contiguous())
    selected_energy = float(energy[:rank].sum())
    return CausalSubspace(
        variable_name=variable_name,
        basis=basis,
        class_labels=classes,
        fit_sample_count=int(values.shape[0]),
        singular_values=tuple(float(value) for value in singular_values.tolist()),
        between_class_energy_fraction=selected_energy / total_energy,
    )
