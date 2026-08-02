"""Typed loading for the frozen MVP experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class MVPConfig:
    model: str
    seed: int
    discovery_examples: int
    validation_examples: int
    test_examples: int


def load_config(path: str | Path) -> MVPConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = raw["experiment"]
    splits = raw["splits"]
    config = MVPConfig(
        model=str(experiment["model"]),
        seed=int(experiment["seed"]),
        discovery_examples=int(splits["discovery"]),
        validation_examples=int(splits["validation"]),
        test_examples=int(splits["test"]),
    )
    if min(config.discovery_examples, config.validation_examples, config.test_examples) <= 0:
        raise ValueError("all split sizes must be positive")
    return config
