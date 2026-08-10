"""TransformerLens-backed runtime adapter for causal mechanism experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol

import torch

from autocircuit.mechanism_runtime import Hook, RuntimeForward


class _TransformerLensModel(Protocol):
    def run_with_hooks(
        self,
        prompt: str,
        *,
        fwd_hooks: list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]],
        return_type: str,
    ) -> torch.Tensor: ...


class _Adapter(Protocol):
    model: _TransformerLensModel
    model_id: str
    revision: str
    resolved_revision: str | None
    tokenizer_id: str
    dtype: str
    device: str


class TransformerLensMechanismRuntime:
    """Expose an existing PythiaAdapter model through the mechanism runtime protocol.

    User interventions and requested cache captures are composed into one callback
    per hook site. Cache capture therefore observes the post-intervention activation,
    while ``run_with_hooks`` owns temporary-hook cleanup.
    """

    def __init__(self, adapter: _Adapter) -> None:
        self._adapter = adapter
        self.model = adapter.model
        self.model_id = adapter.model_id
        self.revision = adapter.revision
        self.resolved_revision = adapter.resolved_revision
        self.tokenizer_id = adapter.tokenizer_id
        self.dtype = adapter.dtype
        self.device = adapter.device

    @staticmethod
    def _validate_sites(cache_sites: tuple[str, ...], hooks: tuple[Hook, ...]) -> None:
        if not cache_sites:
            raise ValueError("mechanism runtime requires at least one cache site")
        if len(cache_sites) != len(set(cache_sites)) or any(
            not isinstance(site, str) or not site for site in cache_sites
        ):
            raise ValueError("mechanism runtime cache sites must be unique non-empty strings")
        if any(not isinstance(site, str) or not site for site, _ in hooks):
            raise ValueError("mechanism runtime hook sites must be non-empty strings")

    @torch.inference_mode()
    def forward(
        self,
        prompt: str,
        *,
        cache_sites: tuple[str, ...],
        hooks: tuple[Hook, ...] = (),
    ) -> RuntimeForward:
        """Run one prompt with temporary interventions and explicit post-hook caches."""

        if not isinstance(prompt, str) or not prompt:
            raise ValueError("mechanism runtime prompt must be non-empty")
        self._validate_sites(cache_sites, hooks)
        requested = set(cache_sites)
        user_hooks: dict[str, list[Callable[[torch.Tensor, Any], torch.Tensor]]] = defaultdict(list)
        for site, callback in hooks:
            user_hooks[site].append(callback)

        cache: dict[str, torch.Tensor] = {}
        combined_hooks: list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = []
        for site in sorted(requested | set(user_hooks)):
            callbacks = tuple(user_hooks.get(site, ()))
            should_cache = site in requested

            def composed(
                value: torch.Tensor,
                hook: Any,
                *,
                _site: str = site,
                _callbacks: tuple[Callable[[torch.Tensor, Any], torch.Tensor], ...] = callbacks,
                _should_cache: bool = should_cache,
            ) -> torch.Tensor:
                current = value
                for callback in _callbacks:
                    updated = callback(current, hook)
                    if not isinstance(updated, torch.Tensor):
                        raise RuntimeError(
                            f"mechanism intervention at {_site} did not return a tensor"
                        )
                    current = updated
                if _should_cache:
                    cache[_site] = current.detach().to(device="cpu").clone()
                return current

            combined_hooks.append((site, composed))

        logits = self.model.run_with_hooks(
            prompt,
            fwd_hooks=combined_hooks,
            return_type="logits",
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError(
                "TransformerLens mechanism runtime expected logits shape [batch, position, vocab]"
            )
        if int(logits.shape[0]) != 1 or int(logits.shape[1]) <= 0:
            raise RuntimeError(
                "TransformerLens mechanism runtime requires one non-empty prompt per forward"
            )
        missing = sorted(requested - set(cache))
        if missing:
            raise RuntimeError(
                "requested cache hooks did not execute: " + ", ".join(missing)
            )
        final_logits = logits[0, -1].detach().to(device="cpu").clone()
        return RuntimeForward(final_logits=final_logits, cache=cache)
