"""Runtime device selection and dependency compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Protocol

from packaging.version import InvalidVersion, Version

SUPPORTED_TRANSFORMER_LENS = "2.15.4"
SUPPORTED_TRANSFORMERS = "4.51.3"


class TorchCuda(Protocol):
    def is_available(self) -> bool: ...


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    message: str


def select_device(requested: str, cuda: TorchCuda) -> str:
    """Resolve auto/cpu/cuda without initializing or downloading a model."""
    normalized = requested.lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if normalized == "auto":
        return "cuda" if cuda.is_available() else "cpu"
    if normalized == "cuda" and not cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return normalized


def check_compatibility(transformer_lens: str, transformers: str) -> CompatibilityResult:
    """Validate the reproducible dependency pair used by this MVP."""
    try:
        lens_version = Version(transformer_lens)
        hf_version = Version(transformers)
    except InvalidVersion as exc:
        return CompatibilityResult(False, f"Could not parse dependency version: {exc}")

    expected_lens = Version(SUPPORTED_TRANSFORMER_LENS)
    expected_hf = Version(SUPPORTED_TRANSFORMERS)
    if lens_version != expected_lens or hf_version != expected_hf:
        return CompatibilityResult(
            False,
            "Unsupported TransformerLens/Transformers combination: "
            f"got {lens_version}/{hf_version}; expected "
            f"{expected_lens}/{expected_hf}. Reinstall with `pip install -e .`.",
        )
    return CompatibilityResult(True, f"compatible ({lens_version} / {hf_version})")


def installed_version(distribution: str) -> str | None:
    """Read package metadata without importing heavyweight libraries."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None
