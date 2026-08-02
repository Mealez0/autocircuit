"""Explicit, network-capable TransformerLens smoke test."""

from __future__ import annotations

import argparse
import sys

MODEL_ALIASES = {"pythia-70m": "EleutherAI/pythia-70m"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="pythia-70m", choices=sorted(MODEL_ALIASES))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformer_lens import HookedTransformer

        from autocircuit.runtime import select_device

        device = select_device(args.device, torch.cuda)
        model_name = MODEL_ALIASES[args.model]
        print(f"Loading {model_name} on {device} ...")
        model = HookedTransformer.from_pretrained(model_name, device=device)
        tokens = model.to_tokens("John likes apples. Mary likes books. John likes")
        logits, cache = model.run_with_cache(tokens)
        expected = (tokens.shape[0], tokens.shape[1], model.cfg.d_vocab)
        if tuple(logits.shape) != expected:
            raise AssertionError(f"logits shape {tuple(logits.shape)} != expected {expected}")
        print(f"Logits shape: {tuple(logits.shape)}")
        print(f"Cached hooks: {len(cache)}")
        if not cache:
            raise AssertionError("run_with_cache returned no hook activations")
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print("PASS: model forward pass and activation cache succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
