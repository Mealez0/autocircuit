"""Runtime core for fitting and executing discovery-only mechanism falsifiers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from autocircuit.causal_subspace import (
    ActivationAlignment,
    fit_causal_subspace,
)
from autocircuit.mechanism_compile import build_hook_callback
from autocircuit.mechanism_counterfactuals import CounterfactualPair
from autocircuit.mechanism_execution import PreparedMechanismExperiment
from autocircuit.mechanism_outcomes import (
    ObservationPolicy,
    PairMechanismMeasurement,
    aggregate_observation,
    classify_pair_measurement,
)
from autocircuit.mechanisms import MechanismCampaign

RUNTIME_VERSION = "mechanism-runtime-0.1.0"
INTERPRETATION_SCOPE = "exploratory_discovery_only"
HookCallback = Callable[[torch.Tensor, Any], torch.Tensor]
Hook = tuple[str, HookCallback]


@dataclass(frozen=True)
class RuntimeForward:
    """One model forward reduced to final logits and explicitly requested hook caches."""

    final_logits: torch.Tensor
    cache: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.final_logits, torch.Tensor) or self.final_logits.ndim != 1:
            raise ValueError("runtime final logits must be a rank-1 tensor")
        if not bool(torch.isfinite(self.final_logits).all()):
            raise ValueError("runtime final logits must be finite")
        if not isinstance(self.cache, dict):
            raise ValueError("runtime cache must be a dictionary")
        for name, value in self.cache.items():
            if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
                raise ValueError("runtime cache entries are malformed")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"runtime cache contains non-finite values: {name}")


class MechanismRuntime(Protocol):
    """Minimal model interface shared by fake tests and TransformerLens execution."""

    model_id: str
    revision: str
    resolved_revision: str | None
    tokenizer_id: str
    dtype: str
    device: str

    def forward(
        self,
        prompt: str,
        *,
        cache_sites: tuple[str, ...],
        hooks: tuple[Hook, ...] = (),
    ) -> RuntimeForward: ...


def _campaign_carriers(campaign: MechanismCampaign) -> tuple[int, tuple[int, int]]:
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
        raise ValueError("mechanism campaign must expose exactly two attention heads")
    return layer, (int(heads[0]), int(heads[1]))


def _final_position(value: torch.Tensor, *, rank: int) -> torch.Tensor:
    if value.ndim != rank:
        raise RuntimeError(f"runtime hook tensor must have rank {rank}")
    if value.shape[0] != 1 or value.shape[1] <= 0:
        raise RuntimeError("runtime fitting currently requires one non-empty prompt at a time")
    return value[:, -1]


def _fit_rank(labels: Sequence[str], feature_dim: int, max_rank: int) -> int:
    if not isinstance(max_rank, int) or isinstance(max_rank, bool) or max_rank <= 0:
        raise ValueError("max_rank must be a positive integer")
    class_count = len(set(labels))
    if class_count < 2:
        raise ValueError("causal-variable fitting requires at least two observed classes")
    return min(max_rank, class_count - 1, feature_dim)


def _fit_alignment(
    *,
    key: str,
    variable_name: str,
    hook_site: str,
    position_index: int,
    selected_heads: tuple[int, ...],
    samples: list[torch.Tensor],
    labels: list[str],
    max_rank: int,
) -> ActivationAlignment:
    if not samples or len(samples) != len(labels):
        raise ValueError(f"alignment {key} has incomplete training samples")
    matrix = torch.cat(samples, dim=0).to(device="cpu", dtype=torch.float64)
    rank = _fit_rank(labels, int(matrix.shape[-1]), max_rank)
    subspace = fit_causal_subspace(variable_name, matrix, labels, rank=rank)
    return ActivationAlignment(
        alignment_id=key.replace("_", "-"),
        variable_name=variable_name,
        hook_site=hook_site,
        position_index=position_index,
        selected_heads=selected_heads,
        subspace=subspace,
    )


def fit_runtime_alignments(
    runtime: MechanismRuntime,
    campaign: MechanismCampaign,
    counterfactuals: Sequence[CounterfactualPair],
    *,
    max_rank: int = 8,
) -> tuple[dict[str, ActivationAlignment], dict[str, Any]]:
    """Fit four proposal-only causal-variable alignments on discovery prompts."""

    if not counterfactuals or any(pair.split != "discovery" for pair in counterfactuals):
        raise ValueError("runtime alignment fitting accepts discovery counterfactuals only")
    layer, heads = _campaign_carriers(campaign)
    z_site = f"blocks.{layer}.attn.hook_z"
    mlp_site = f"blocks.{layer}.hook_mlp_out"
    downstream_site = f"blocks.{layer}.hook_resid_post"

    head_query_samples: list[torch.Tensor] = []
    mlp_query_samples: list[torch.Tensor] = []
    downstream_samples: list[torch.Tensor] = []
    query_labels: list[str] = []
    mlp_value_samples: list[torch.Tensor] = []
    value_labels: list[str] = []

    for pair in sorted(counterfactuals, key=lambda item: item.counterfactual_id):
        if pair.kind == "query_swap":
            if pair.base_query_fact_index is None or pair.donor_query_fact_index is None:
                raise ValueError("query counterfactual is missing fact-slot grounding")
            for prompt, slot in (
                (pair.base_prompt, pair.base_query_fact_index),
                (pair.donor_prompt, pair.donor_query_fact_index),
            ):
                forward = runtime.forward(
                    prompt,
                    cache_sites=(z_site, mlp_site, downstream_site),
                )
                try:
                    z = _final_position(forward.cache[z_site], rank=4)
                    mlp = _final_position(forward.cache[mlp_site], rank=3)
                    downstream = _final_position(forward.cache[downstream_site], rank=3)
                except KeyError as exc:
                    raise RuntimeError(f"runtime omitted required fitting hook: {exc.args[0]}") from exc
                if any(head >= int(z.shape[1]) for head in heads):
                    raise ValueError("candidate attention head is outside runtime hook_z shape")
                selected = z[:, list(heads), :].reshape(1, -1)
                head_query_samples.append(selected)
                mlp_query_samples.append(mlp)
                downstream_samples.append(downstream)
                query_labels.append(f"slot_{slot}")
        elif pair.kind == "value_binding_swap":
            for prompt, token_id in (
                (pair.base_prompt, pair.base_target_token_id),
                (pair.donor_prompt, pair.donor_target_token_id),
            ):
                forward = runtime.forward(prompt, cache_sites=(mlp_site,))
                try:
                    mlp = _final_position(forward.cache[mlp_site], rank=3)
                except KeyError as exc:
                    raise RuntimeError(f"runtime omitted required fitting hook: {mlp_site}") from exc
                mlp_value_samples.append(mlp)
                value_labels.append(f"token_{token_id}")
        else:
            raise ValueError(f"unknown mechanism counterfactual kind: {pair.kind}")

    alignments = {
        "head_query_state": _fit_alignment(
            key="head_query_state",
            variable_name="query_slot",
            hook_site=z_site,
            position_index=-1,
            selected_heads=heads,
            samples=head_query_samples,
            labels=query_labels,
            max_rank=max_rank,
        ),
        "mlp_query_state": _fit_alignment(
            key="mlp_query_state",
            variable_name="query_slot",
            hook_site=mlp_site,
            position_index=-1,
            selected_heads=(),
            samples=mlp_query_samples,
            labels=query_labels,
            max_rank=max_rank,
        ),
        "downstream_slot_state": _fit_alignment(
            key="downstream_slot_state",
            variable_name="query_slot",
            hook_site=downstream_site,
            position_index=-1,
            selected_heads=(),
            samples=downstream_samples,
            labels=query_labels,
            max_rank=max_rank,
        ),
        "mlp_value_state": _fit_alignment(
            key="mlp_value_state",
            variable_name="retrieved_value",
            hook_site=mlp_site,
            position_index=-1,
            selected_heads=(),
            samples=mlp_value_samples,
            labels=value_labels,
            max_rank=max_rank,
        ),
    }
    report = {
        "runtime_version": RUNTIME_VERSION,
        "fit_scope": INTERPRETATION_SCOPE,
        "selected_layer": layer,
        "attention_heads": list(heads),
        "max_rank": max_rank,
        "query_training_sample_count": len(query_labels),
        "query_class_count": len(set(query_labels)),
        "value_training_sample_count": len(value_labels),
        "value_class_count": len(set(value_labels)),
        "alignment_fingerprints": {
            key: alignment.fingerprint() for key, alignment in sorted(alignments.items())
        },
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
    return alignments, report


def _readout_alignment(
    alignments: Mapping[str, ActivationAlignment], recipe_site: str
) -> ActivationAlignment:
    downstream = alignments.get("downstream_slot_state")
    if not isinstance(downstream, ActivationAlignment):
        raise ValueError("missing downstream slot alignment")
    prefix = recipe_site.split(".", 2)[:2]
    expected_prefix = ".".join(prefix)
    if downstream.hook_site != f"{expected_prefix}.hook_resid_post":
        raise ValueError("downstream slot alignment is not from the intervention layer")
    if downstream.position_index != -1 or downstream.selected_heads:
        raise ValueError("downstream slot alignment must read the full final-position residual")
    if downstream.variable_name != "query_slot":
        raise ValueError("downstream slot alignment must represent query_slot")
    return downstream


def _projected_final(
    alignment: ActivationAlignment, activation: torch.Tensor
) -> torch.Tensor:
    value = _final_position(activation, rank=3)
    if alignment.subspace.feature_dim != int(value.shape[-1]):
        raise RuntimeError("downstream slot subspace dimension does not match runtime activation")
    return alignment.subspace.project(value.to(device="cpu", dtype=torch.float64))


def _distance(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise RuntimeError("slot-readout reference shapes differ")
    value = float(torch.linalg.vector_norm(left - right).item())
    if not math.isfinite(value):
        raise RuntimeError("slot-readout distance is non-finite")
    return value


def _logit(logits: torch.Tensor, token_id: int) -> float:
    if token_id < 0 or token_id >= int(logits.shape[0]):
        raise ValueError("counterfactual target token id is outside runtime vocabulary")
    value = float(logits[token_id].item())
    if not math.isfinite(value):
        raise RuntimeError("runtime target logit is non-finite")
    return value


def run_prepared_experiment(
    runtime: MechanismRuntime,
    prepared: PreparedMechanismExperiment,
    alignments: Mapping[str, ActivationAlignment],
    *,
    policy: ObservationPolicy | None = None,
) -> dict[str, Any]:
    """Execute one selected falsifier and emit a consensus-gated observation."""

    selected_policy = policy if policy is not None else ObservationPolicy()
    downstream = _readout_alignment(alignments, prepared.recipe.hook_site)
    measurements: list[dict[str, Any]] = []
    outcomes: list[str] = []

    for pair in prepared.counterfactuals:
        if pair.split != "discovery":
            raise ValueError("runtime mechanism execution accepts discovery inputs only")
        base = runtime.forward(pair.base_prompt, cache_sites=(downstream.hook_site,))
        donor_sites = tuple(
            sorted({downstream.hook_site, prepared.recipe.hook_site})
        )
        donor = runtime.forward(pair.donor_prompt, cache_sites=donor_sites)
        if prepared.recipe.operation == "interchange_subspace":
            try:
                donor_activation = donor.cache[prepared.recipe.hook_site]
            except KeyError as exc:
                raise RuntimeError("runtime omitted donor intervention hook") from exc
            callback = build_hook_callback(prepared.recipe, donor_activation)
        else:
            callback = build_hook_callback(prepared.recipe)
        patched = runtime.forward(
            pair.base_prompt,
            cache_sites=(downstream.hook_site,),
            hooks=((prepared.recipe.hook_site, callback),),
        )
        try:
            base_slot = _projected_final(downstream, base.cache[downstream.hook_site])
            donor_slot = _projected_final(downstream, donor.cache[downstream.hook_site])
            patched_slot = _projected_final(downstream, patched.cache[downstream.hook_site])
        except KeyError as exc:
            raise RuntimeError("runtime omitted downstream slot-readout hook") from exc

        base_target_logit = _logit(base.final_logits, pair.base_target_token_id)
        base_donor_logit = _logit(base.final_logits, pair.donor_target_token_id)
        measurement = PairMechanismMeasurement(
            counterfactual_id=pair.counterfactual_id,
            experiment_id=prepared.experiment_id,
            base_margin=base_target_logit - base_donor_logit,
            patched_base_target_logit=_logit(
                patched.final_logits, pair.base_target_token_id
            ),
            patched_donor_target_logit=_logit(
                patched.final_logits, pair.donor_target_token_id
            ),
            slot_distance_to_base=_distance(patched_slot, base_slot),
            slot_distance_to_donor=_distance(patched_slot, donor_slot),
        )
        outcome = classify_pair_measurement(measurement, policy=selected_policy)
        measurements.append({**measurement.to_dict(), "outcome": outcome})
        outcomes.append(outcome)

    aggregation = aggregate_observation(outcomes, policy=selected_policy)
    return {
        "schema_version": 1,
        "runtime_version": RUNTIME_VERSION,
        "interpretation_scope": INTERPRETATION_SCOPE,
        "experiment_id": prepared.experiment_id,
        "recipe": prepared.to_manifest(),
        "observation_policy": selected_policy.to_dict(),
        "measurements": measurements,
        "aggregation": aggregation,
        "held_out_validation_reused": False,
        "held_out_test_opened": False,
        "scientific_confirmation": False,
        "circuit_found": False,
    }
