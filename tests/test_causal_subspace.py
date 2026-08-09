from __future__ import annotations

import pytest
import torch

from autocircuit.causal_subspace import CausalSubspace, fit_causal_subspace


def _training_data() -> tuple[torch.Tensor, list[str]]:
    activations = torch.tensor(
        [
            [-2.0, -1.0],
            [-2.0, 1.0],
            [2.0, -1.0],
            [2.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return activations, ["left", "left", "right", "right"]


def test_fit_causal_subspace_is_deterministic_orthonormal_and_discriminative() -> None:
    activations, labels = _training_data()
    first = fit_causal_subspace("query_slot", activations, labels, rank=1)
    second = fit_causal_subspace("query_slot", activations, labels, rank=1)

    assert first.fingerprint() == second.fingerprint()
    assert first.rank == 1
    assert first.feature_dim == 2
    assert first.between_class_energy_fraction == pytest.approx(1.0)
    gram = first.basis.T @ first.basis
    assert torch.allclose(gram, torch.eye(1, dtype=torch.float64), atol=1e-12, rtol=0.0)
    assert first.evidence_status == "proposal_only_not_evidence"


def test_subspace_interchange_replaces_only_aligned_direction() -> None:
    activations, labels = _training_data()
    subspace = fit_causal_subspace("query_slot", activations, labels, rank=1)
    base = torch.tensor([[1.0, 10.0]], dtype=torch.float64)
    donor = torch.tensor([[3.0, 99.0]], dtype=torch.float64)

    patched = subspace.interchange(base, donor)

    assert patched[0, 0].item() == pytest.approx(3.0)
    assert patched[0, 1].item() == pytest.approx(10.0)
    assert torch.allclose(
        subspace.orthogonal_component(patched),
        subspace.orthogonal_component(base),
        atol=1e-12,
        rtol=0.0,
    )


def test_subspace_fingerprint_is_invariant_to_training_row_order() -> None:
    activations, labels = _training_data()
    first = fit_causal_subspace("query_slot", activations, labels, rank=1)
    order = torch.tensor([3, 1, 2, 0])
    second = fit_causal_subspace(
        "query_slot",
        activations[order],
        [labels[index] for index in order.tolist()],
        rank=1,
    )

    assert first.fingerprint() == second.fingerprint()


def test_fit_causal_subspace_rejects_invalid_or_nonfinite_inputs() -> None:
    activations, labels = _training_data()
    with pytest.raises(ValueError, match="rank"):
        fit_causal_subspace("query_slot", activations, labels, rank=2)
    with pytest.raises(ValueError, match="label count"):
        fit_causal_subspace("query_slot", activations, labels[:-1], rank=1)

    invalid = activations.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_causal_subspace("query_slot", invalid, labels, rank=1)


def test_direct_subspace_rejects_non_orthonormal_basis() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        CausalSubspace(
            variable_name="query_slot",
            basis=torch.tensor([[1.0], [1.0]], dtype=torch.float64),
            class_labels=("left", "right"),
            fit_sample_count=4,
            singular_values=(1.0,),
            between_class_energy_fraction=1.0,
        )
