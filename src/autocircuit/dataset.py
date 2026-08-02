"""Command-line entry point for associative-recall dataset generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from autocircuit.config import load_config
from autocircuit.datasets.associative_recall import (
    GENERATOR_VERSION,
    SPLITS,
    generate_dataset,
    write_jsonl,
)
from autocircuit.datasets.validation import validate_split_leakage


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(config_path: Path, output: Path) -> None:
    from transformers import AutoTokenizer

    config = load_config(config_path)
    model_id = f"EleutherAI/{config.model}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    examples, rejected = generate_dataset(config, tokenizer)
    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for split in SPLITS:
        selected = [item for item in examples if item.split == split]
        filename = f"{split}.jsonl"
        hashes[filename] = write_jsonl(output / filename, selected)
        sizes[split] = len(selected)
    leakage = validate_split_leakage(examples)
    # A seed-derived timestamp preserves the byte-reproducibility contract.
    generated_at = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(seconds=config.seed)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": config.seed,
        "split_sizes": sizes,
        "config_hash": _sha256(config_path),
        "file_hashes": hashes,
        "created_at": generated_at.isoformat(),
        "model_id": model_id,
        "tokenizer_id": str(getattr(tokenizer, "name_or_path", model_id)),
        "rejected_examples": {"total": sum(rejected.values()), "reasons": rejected},
        "leakage_validation": {"passed": leakage.passed, "reasons": leakage.reasons},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(" ".join(f"{name}={size}" for name, size in sizes.items()))
    print(f"VALIDATION: {'PASS' if leakage.passed else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("generate")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    generate(args.config, args.output)


if __name__ == "__main__":
    main()
