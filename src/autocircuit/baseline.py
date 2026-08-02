"""Pythia-70M next-token baseline evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autocircuit.datasets.associative_recall import read_jsonl
from autocircuit.runtime import select_device


@dataclass(frozen=True)
class BaselineMetrics:
    clean_accuracy: float
    corrupt_accuracy: float
    clean_mean_logit_difference: float
    corrupt_mean_logit_difference: float
    clean_corrupt_contrast: float
    example_count: int
    failed_count: int


def metrics_from_differences(
    clean: Sequence[float], corrupt: Sequence[float], *, failed_count: int = 0
) -> BaselineMetrics:
    if len(clean) != len(corrupt) or not clean:
        raise ValueError("clean and corrupt metrics must have equal non-zero lengths")
    clean_mean = sum(clean) / len(clean)
    corrupt_mean = sum(corrupt) / len(corrupt)
    return BaselineMetrics(
        clean_accuracy=sum(value > 0 for value in clean) / len(clean),
        # The corrupt prompt should prefer the distractor, hence negative clean-target LD.
        corrupt_accuracy=sum(value < 0 for value in corrupt) / len(corrupt),
        clean_mean_logit_difference=clean_mean,
        corrupt_mean_logit_difference=corrupt_mean,
        clean_corrupt_contrast=clean_mean - corrupt_mean,
        example_count=len(clean),
        failed_count=failed_count,
    )


def baseline_passes(metrics: BaselineMetrics) -> bool:
    return metrics.clean_accuracy >= 0.80 and metrics.clean_mean_logit_difference >= 1.0


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    dataset: Path,
    model_name: str,
    requested_device: str,
    split: str,
    batch_size: int,
    output_root: Path,
    seed: int,
    revision: str,
) -> Path:
    import torch
    from transformer_lens import HookedTransformer

    random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(requested_device, torch.cuda)
    model_id = f"EleutherAI/{model_name}"
    model = HookedTransformer.from_pretrained(model_name, device=device, revision=revision)
    examples = read_jsonl(dataset / f"{split}.jsonl")
    clean_differences: list[float] = []
    corrupt_differences: list[float] = []
    failed = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        buckets: dict[int, list[Any]] = {}
        for item in batch:
            buckets.setdefault(int(item.metadata["prompt_token_length"]), []).append(item)
        for group in buckets.values():
            try:
                clean_logits = model([item.clean_prompt for item in group], return_type="logits")[
                    :, -1
                ]
                corrupt_logits = model(
                    [item.corrupt_prompt for item in group], return_type="logits"
                )[:, -1]
                for row, item in enumerate(group):
                    target, distractor = item.target_token_id, item.distractor_token_id
                    clean_differences.append(
                        float(clean_logits[row, target] - clean_logits[row, distractor])
                    )
                    corrupt_differences.append(
                        float(corrupt_logits[row, target] - corrupt_logits[row, distractor])
                    )
            except (RuntimeError, ValueError):
                failed += len(group)
    metrics = metrics_from_differences(clean_differences, corrupt_differences, failed_count=failed)
    manifest_path = dataset / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    passed = baseline_passes(metrics)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{split}-{seed}"
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "baseline": "PASS" if passed else "FAIL",
        "split": split,
        "metrics": asdict(metrics),
        "acceptance": {"minimum_clean_accuracy": 0.8, "minimum_clean_mean_ld": 1.0},
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer": str(getattr(model.tokenizer, "name_or_path", model_id)),
        "device": device,
        "dtype": str(model.cfg.dtype),
        "seed": seed,
        "batch_size": batch_size,
        "config_hash": manifest["config_hash"],
        "dataset_hash": _hash(dataset / f"{split}.jsonl"),
        "dataset_manifest_hash": _hash(manifest_path),
        "created_at": datetime.now(UTC).isoformat(),
    }
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"BASELINE: {'PASS' if passed else 'FAIL'}")
    print(f"Results: {result_path}")
    return result_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="pythia-70m", choices=["pythia-70m"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="discovery", choices=["discovery", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    evaluate(
        args.dataset,
        args.model,
        args.device,
        args.split,
        args.batch_size,
        args.output,
        args.seed,
        args.revision,
    )


if __name__ == "__main__":
    main()
