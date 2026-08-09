from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autocircuit import pipeline


def test_sha256_and_resume_verification_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"autocircuit-artifact\n")
    expected = hashlib.sha256(b"autocircuit-artifact\n").hexdigest()

    assert pipeline.sha256(artifact) == expected
    pipeline.verify_resume(artifact, expected)

    artifact.write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match=r"resume hash verification failed: .*artifact\.bin"):
        pipeline.verify_resume(artifact, expected)


def test_json_writer_is_deterministic_utf8_and_rejects_nan(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {"z": "ğ", "a": [3, 2, 1], "nested": {"b": True, "a": None}}

    pipeline._json(first, payload)
    pipeline._json(second, payload)

    expected = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    assert first.read_bytes() == expected
    assert second.read_bytes() == expected
    assert b"\r\n" not in expected

    with pytest.raises(ValueError):
        pipeline._json(tmp_path / "nan.json", {"value": float("nan")})


def test_validation_json_atomically_replaces_and_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    pipeline._validation_json(path, {"generation": 1, "complete": False})
    first = path.read_bytes()

    pipeline._validation_json(path, {"generation": 2, "complete": True})
    second = path.read_bytes()

    assert first != second
    assert json.loads(second) == {"complete": True, "generation": 2}
    assert second.endswith(b"\n")
    assert not list(tmp_path.glob(".manifest.json.*"))
