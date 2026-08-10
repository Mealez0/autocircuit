from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from autocircuit.mechanism_tlens_runtime import TransformerLensMechanismRuntime


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run_with_hooks(
        self,
        prompt: str,
        *,
        fwd_hooks: list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]],
        return_type: str,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "prompt": prompt,
                "hook_names": [name for name, _ in fwd_hooks],
                "return_type": return_type,
            }
        )
        activations = {
            "blocks.5.hook_mlp_out": torch.tensor(
                [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32
            ),
            "blocks.5.hook_resid_post": torch.tensor(
                [[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32
            ),
        }
        for name, callback in fwd_hooks:
            value = activations.get(name)
            if value is None:
                value = torch.zeros((1, 2, 2), dtype=torch.float32)
            activations[name] = callback(value, object())
        logits = torch.zeros((1, 2, 32), dtype=torch.float32)
        logits[0, -1, 10] = 2.0
        logits[0, -1, 11] = 1.0
        return logits


class FakeAdapter:
    model_id = "EleutherAI/pythia-70m"
    revision = "step143000"
    resolved_revision = "deadbeef"
    tokenizer_id = "EleutherAI/pythia-70m"
    dtype = "torch.float32"
    device = "cpu"

    def __init__(self) -> None:
        self.model = FakeModel()


def test_tlens_runtime_caches_post_intervention_activation_and_final_logits() -> None:
    adapter = FakeAdapter()
    runtime = TransformerLensMechanismRuntime(adapter)

    def add_ten(value: torch.Tensor, hook: Any) -> torch.Tensor:
        del hook
        return value + 10.0

    result = runtime.forward(
        "prompt",
        cache_sites=("blocks.5.hook_mlp_out", "blocks.5.hook_resid_post"),
        hooks=(("blocks.5.hook_mlp_out", add_ten),),
    )

    assert result.final_logits.shape == (32,)
    assert result.final_logits[10].item() == pytest.approx(2.0)
    assert result.cache["blocks.5.hook_mlp_out"][0, -1].tolist() == [13.0, 14.0]
    assert result.cache["blocks.5.hook_resid_post"][0, -1].tolist() == [7.0, 8.0]
    assert adapter.model.calls == [
        {
            "prompt": "prompt",
            "hook_names": ["blocks.5.hook_mlp_out", "blocks.5.hook_resid_post"],
            "return_type": "logits",
        }
    ]


def test_tlens_runtime_applies_multiple_user_hooks_in_declared_order() -> None:
    runtime = TransformerLensMechanismRuntime(FakeAdapter())

    def times_two(value: torch.Tensor, hook: Any) -> torch.Tensor:
        del hook
        return value * 2.0

    def plus_one(value: torch.Tensor, hook: Any) -> torch.Tensor:
        del hook
        return value + 1.0

    result = runtime.forward(
        "prompt",
        cache_sites=("blocks.5.hook_mlp_out",),
        hooks=(
            ("blocks.5.hook_mlp_out", times_two),
            ("blocks.5.hook_mlp_out", plus_one),
        ),
    )

    assert result.cache["blocks.5.hook_mlp_out"][0, -1].tolist() == [7.0, 9.0]


def test_tlens_runtime_rejects_duplicate_cache_sites_or_invalid_model_output() -> None:
    runtime = TransformerLensMechanismRuntime(FakeAdapter())
    with pytest.raises(ValueError, match="cache sites"):
        runtime.forward(
            "prompt",
            cache_sites=("blocks.5.hook_mlp_out", "blocks.5.hook_mlp_out"),
        )

    class BadModel(FakeModel):
        def run_with_hooks(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            del args, kwargs
            return torch.zeros((2, 32), dtype=torch.float32)

    bad = FakeAdapter()
    bad.model = BadModel()
    with pytest.raises(RuntimeError, match="logits shape"):
        TransformerLensMechanismRuntime(bad).forward(
            "prompt",
            cache_sites=("blocks.5.hook_mlp_out",),
        )


def test_tlens_runtime_fails_if_requested_hook_never_executes() -> None:
    class MissingHookModel(FakeModel):
        def run_with_hooks(
            self,
            prompt: str,
            *,
            fwd_hooks: list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]],
            return_type: str,
        ) -> torch.Tensor:
            del prompt, fwd_hooks, return_type
            return torch.zeros((1, 2, 32), dtype=torch.float32)

    adapter = FakeAdapter()
    adapter.model = MissingHookModel()
    runtime = TransformerLensMechanismRuntime(adapter)
    with pytest.raises(RuntimeError, match="requested cache hooks did not execute"):
        runtime.forward("prompt", cache_sites=("blocks.5.hook_mlp_out",))
