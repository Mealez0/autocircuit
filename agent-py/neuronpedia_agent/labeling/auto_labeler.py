"""LLM-based labels for graph supernodes."""

from __future__ import annotations

from typing import Any, Mapping

import anthropic

from ..analysis.grouping_engine import Supernode

LABELING_PROMPT = """
You are analyzing a group of features in a language model's computational pathway. Your task is to create a concise label (2-5 words) that describes what this group of features collectively does.

Context:
- Input prompt: "{prompt}"
- Target output: "{target_logit}"
- Functional role: {functional_role} (input_detector, relational_processor, or output_promoter)
- Layer range: {layer_min} to {layer_max}

Features in this group:
{feature_details}

Guidelines for labeling:
1. For input detectors: Describe what tokens/patterns they detect (e.g., "state names", "capital-of preposition")
2. For relational processors: Describe the relationship they encode (e.g., "capital-state mapping", "location relations")
3. For output promoters: Describe what they predict (e.g., "say [city name]", "promote Texas cities")
4. Be specific but concise (2-5 words)
5. Avoid technical jargon like "features" or "nodes"

Generate a label:
"""


class AutoLabeler:
    """Generate human-readable labels for supernodes using an LLM.

    API failures are intentionally not converted into plausible-looking labels.
    Callers receive the exception and can decide whether to stop, retry, or skip
    labeling.  This keeps agent execution failures observable.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_label(
        self,
        supernode: Supernode,
        node_data: Mapping[str, Mapping[str, Any]],
        prompt: str = "",
        target_logit: str = "",
    ) -> str:
        labeling_prompt = self._create_labeling_prompt(
            supernode,
            node_data,
            prompt,
            target_logit,
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=50,
            temperature=0.7,
            messages=[{"role": "user", "content": labeling_prompt}],
        )

        text_parts = [
            str(getattr(block, "text", "")).strip()
            for block in message.content
            if getattr(block, "text", None)
        ]
        label = " ".join(part for part in text_parts if part).strip()
        if not label:
            raise RuntimeError("Labeling API returned no text content")

        words = label.split()
        return " ".join(words[:5])

    def _create_labeling_prompt(
        self,
        supernode: Supernode,
        node_data: Mapping[str, Mapping[str, Any]],
        prompt: str,
        target_logit: str,
    ) -> str:
        feature_details: list[str] = []
        for node_id in supernode.node_ids:
            node = node_data.get(str(node_id), {})
            explanation = node.get("explanation", "No explanation")
            layer = node.get("layer", "?")
            feature_idx = node.get("feature_index", "?")
            detail = f"- Layer {layer}, Feature {feature_idx}: {explanation}"

            top_logits = node.get("top_logits", [])
            if isinstance(top_logits, list) and top_logits:
                parts: list[str] = []
                for item in top_logits[:3]:
                    if not isinstance(item, Mapping):
                        continue
                    token = item.get("token", "?")
                    value = item.get("value")
                    if isinstance(value, (int, float)):
                        parts.append(f"{token} ({value:.2f})")
                    else:
                        parts.append(str(token))
                if parts:
                    detail += f"\n  Top logits: {', '.join(parts)}"

            feature_details.append(detail)

        return LABELING_PROMPT.format(
            prompt=prompt,
            target_logit=target_logit,
            functional_role=supernode.functional_role,
            layer_min=supernode.layer_range[0],
            layer_max=supernode.layer_range[1],
            feature_details="\n".join(feature_details),
        )
