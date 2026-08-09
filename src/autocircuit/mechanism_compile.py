"""Compile abstract discovery-only mechanism experiments to hook-level recipes.

Compilation makes an experiment executable without upgrading its evidentiary
status. Subspace alignments remain hypotheses until the resulting interventions
are actually run and evaluated under the declared discovery protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from autocircuit.causal_subspace import ActivationAlignment
from autocircuit.mechanisms import MechanismCampaign, MechanismExperiment


@dataclass(frozen=True)
class InterventionRecipe:
    """An exact TransformerLens hook slice and operation for one experiment."""

    experiment_id: str
    hook_site: str
    position_index: int
    selected_heads: tuple[int, ...]
    operation: str
    intervention_kind: str
    readout: str
    alignment: ActivationAlignment | None = None
    evidence_status: str = "proposal_only_not_evidence"

    def __post_init__(self) -> None:
        if not isinstance(self.hook_site, str) or not self.hook_site:
            raise ValueError("intervention recipe hook site must be non-empty")
        if not isinstance(self.position_index, int) or isinstance(self.position_index, bool):
            raise ValueError("intervention recipe position must be an integer")
        if len(self.selected_heads) != len(set(self.selected_heads)) or any(
            not isinstance(head, int) or isinstance(head, bool) or head < 0
            for head in self.selected_heads
        ):
            raise ValueError("intervention recipe head indexes are invalid")
        if self.operation not in {"interchange_subspace", "zero_selected_heads"}:
            raise ValueError("unknown intervention recipe operation")
        if self.operation == "interchange_subspace" and self.alignment is None:
            raise ValueError("subspace interchange requires an activation alignment")
        if self.operation == "zero_selected_heads" and self.alignment is not None:
            raise ValueError("head suppression cannot carry a subspace alignment")
        if self.evidence_status != "proposal_only_not_evidence":
            raise ValueError("compiled interventions remain proposal-only")


def _experiment(campaign: MechanismCampaign, experiment_id: str) -> MechanismExperiment:
    matches = [
        experiment
        for experiment in campaign.experiments
        if experiment.experiment_id == experiment_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate mechanism experiment: {experiment_id}")
    return matches[0]


def _carriers(campaign: MechanismCampaign) -> tuple[int, tuple[int, int]]:
    layer = campaign.selected_layer
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
        raise ValueError("mechanism campaign has no valid selected layer")
    carriers = campaign.candidate_carriers
    if not isinstance(carriers, dict):
        raise ValueError("mechanism campaign has no candidate carriers")
    heads = carriers.get("attention_heads")
    if (
        not isinstance(heads, list | tuple)
        or len(heads) != 2
        or any(not isinstance(head, int) or isinstance(head, bool) or head < 0 for head in heads)
        or heads[0] == heads[1]
    ):
        raise ValueError("mechanism campaign must expose exactly two candidate attention heads")
    mlp_layer = carriers.get("mlp_layer")
    if mlp_layer != layer:
        raise ValueError("mechanism campaign MLP carrier disagrees with selected layer")
    if carriers.get("role_assignment_status") != "hypothesis_only_not_evidence":
        raise ValueError("mechanism carrier roles must remain hypothesis-only")
    return layer, (int(heads[0]), int(heads[1]))


def _require_alignment(
    alignments: Mapping[str, ActivationAlignment],
    key: str,
    *,
    hook_site: str,
    selected_heads: tuple[int, ...],
) -> ActivationAlignment:
    alignment = alignments.get(key)
    if not isinstance(alignment, ActivationAlignment):
        raise ValueError(f"missing activation alignment: {key}")
    if alignment.hook_site != hook_site:
        raise ValueError(f"activation alignment hook site mismatch for {key}")
    if alignment.position_index != -1:
        raise ValueError(f"activation alignment position mismatch for {key}")
    if alignment.selected_heads != selected_heads:
        raise ValueError(f"activation alignment selected heads mismatch for {key}")
    return alignment


def compile_experiment(
    campaign: MechanismCampaign,
    experiment_id: str,
    alignments: Mapping[str, ActivationAlignment],
) -> InterventionRecipe:
    """Compile one abstract mechanism experiment to an exact hook-level recipe."""

    experiment = _experiment(campaign, experiment_id)
    layer, heads = _carriers(campaign)
    z_site = f"blocks.{layer}.attn.hook_z"
    mlp_site = f"blocks.{layer}.hook_mlp_out"

    if experiment_id == "interchange_query_state_at_heads":
        alignment = _require_alignment(
            alignments,
            "head_query_state",
            hook_site=z_site,
            selected_heads=heads,
        )
        return InterventionRecipe(
            experiment_id=experiment_id,
            hook_site=z_site,
            position_index=-1,
            selected_heads=heads,
            operation="interchange_subspace",
            intervention_kind=experiment.intervention_kind,
            readout=experiment.readout,
            alignment=alignment,
        )
    if experiment_id == "interchange_query_state_at_mlp":
        alignment = _require_alignment(
            alignments,
            "mlp_query_state",
            hook_site=mlp_site,
            selected_heads=(),
        )
        return InterventionRecipe(
            experiment_id=experiment_id,
            hook_site=mlp_site,
            position_index=-1,
            selected_heads=(),
            operation="interchange_subspace",
            intervention_kind=experiment.intervention_kind,
            readout=experiment.readout,
            alignment=alignment,
        )
    if experiment_id == "interchange_value_state_at_mlp":
        alignment = _require_alignment(
            alignments,
            "mlp_value_state",
            hook_site=mlp_site,
            selected_heads=(),
        )
        return InterventionRecipe(
            experiment_id=experiment_id,
            hook_site=mlp_site,
            position_index=-1,
            selected_heads=(),
            operation="interchange_subspace",
            intervention_kind=experiment.intervention_kind,
            readout=experiment.readout,
            alignment=alignment,
        )
    if experiment_id == "suppress_secondary_head":
        return InterventionRecipe(
            experiment_id=experiment_id,
            hook_site=z_site,
            position_index=-1,
            selected_heads=(heads[1],),
            operation="zero_selected_heads",
            intervention_kind=experiment.intervention_kind,
            readout=experiment.readout,
            alignment=None,
        )
    raise ValueError(f"mechanism experiment has no compiler rule: {experiment_id}")


def _resolved_position(value: torch.Tensor, position_index: int) -> int:
    if value.ndim < 2:
        raise RuntimeError("hook activation has no position dimension")
    position = position_index if position_index >= 0 else int(value.shape[1]) + position_index
    if position < 0 or position >= int(value.shape[1]):
        raise ValueError("intervention position is outside activation sequence")
    return position


def _interchange_callback(
    recipe: InterventionRecipe, donor_activation: torch.Tensor
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    alignment = recipe.alignment
    if alignment is None:
        raise RuntimeError("compiled interchange recipe lost its activation alignment")

    def callback(value: torch.Tensor, hook: Any = None) -> torch.Tensor:
        del hook
        if not isinstance(value, torch.Tensor) or not isinstance(donor_activation, torch.Tensor):
            raise RuntimeError("hook and donor activations must be tensors")
        if value.shape != donor_activation.shape:
            raise RuntimeError("donor and destination activation shapes differ")
        if donor_activation.device != value.device or donor_activation.dtype != value.dtype:
            donor = donor_activation.to(device=value.device, dtype=value.dtype)
        else:
            donor = donor_activation
        position = _resolved_position(value, recipe.position_index)
        patched = value.clone()
        if recipe.selected_heads:
            if value.ndim != 4:
                raise RuntimeError(
                    "selected-head subspace intervention requires hook_z shape "
                    "[batch, position, head, d_head]"
                )
            if any(head >= int(value.shape[2]) for head in recipe.selected_heads):
                raise ValueError("selected head is outside hook_z head dimension")
            head_indexes = list(recipe.selected_heads)
            base_slice = value[:, position, head_indexes, :]
            donor_slice = donor[:, position, head_indexes, :]
            flat_base = base_slice.reshape(base_slice.shape[0], -1)
            flat_donor = donor_slice.reshape(donor_slice.shape[0], -1)
            if alignment.subspace.feature_dim != int(flat_base.shape[-1]):
                raise RuntimeError("alignment feature dimension does not match selected heads")
            interchanged = alignment.subspace.interchange(flat_base, flat_donor)
            patched[:, position, head_indexes, :] = interchanged.reshape_as(base_slice)
        else:
            if value.ndim != 3:
                raise RuntimeError(
                    "component subspace intervention requires shape [batch, position, feature]"
                )
            base_slice = value[:, position, :]
            donor_slice = donor[:, position, :]
            if alignment.subspace.feature_dim != int(base_slice.shape[-1]):
                raise RuntimeError("alignment feature dimension does not match hook activation")
            patched[:, position, :] = alignment.subspace.interchange(base_slice, donor_slice)
        return patched

    return callback


def _suppression_callback(
    recipe: InterventionRecipe,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    def callback(value: torch.Tensor, hook: Any = None) -> torch.Tensor:
        del hook
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            raise RuntimeError(
                "head suppression requires hook_z shape [batch, position, head, d_head]"
            )
        position = _resolved_position(value, recipe.position_index)
        if any(head >= int(value.shape[2]) for head in recipe.selected_heads):
            raise ValueError("selected head is outside hook_z head dimension")
        patched = value.clone()
        if recipe.selected_heads:
            patched[:, position, list(recipe.selected_heads), :] = 0
        return patched

    return callback


def build_hook_callback(
    recipe: InterventionRecipe,
    donor_activation: torch.Tensor | None = None,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """Build a TransformerLens-compatible callback without mutating source tensors."""

    if recipe.operation == "interchange_subspace":
        if donor_activation is None:
            raise ValueError("subspace interchange requires a donor activation")
        return _interchange_callback(recipe, donor_activation)
    if recipe.operation == "zero_selected_heads":
        if donor_activation is not None:
            raise ValueError("head suppression does not accept a donor activation")
        return _suppression_callback(recipe)
    raise ValueError(f"unknown compiled intervention operation: {recipe.operation}")
