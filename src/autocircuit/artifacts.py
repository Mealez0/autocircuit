"""Deterministic and durable artifact I/O primitives.

These helpers intentionally contain no experiment, model, or scientific-decision
logic. Keeping byte serialization and integrity checks here lets orchestration
code share one reproducible artifact contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_resume(path: Path, expected_hash: str) -> None:
    """Fail closed when a resumable artifact is absent or has changed."""
    if not path.is_file() or sha256(path) != expected_hash:
        raise RuntimeError(f"resume hash verification failed: {path}")


def write_json(path: Path, value: Any) -> None:
    """Write deterministic UTF-8 JSON using the historical AutoCircuit format."""
    path.write_bytes(
        (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def write_json_durable(path: Path, value: Any) -> None:
    """Atomically and durably replace JSON without truncating prior state."""
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # Directory handles/fsync are unavailable on some Windows filesystems.
            pass
    finally:
        temporary.unlink(missing_ok=True)
