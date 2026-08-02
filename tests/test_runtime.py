from __future__ import annotations

import pytest

from autocircuit.runtime import check_compatibility, select_device


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


def test_auto_selects_available_cuda() -> None:
    assert select_device("auto", FakeCuda(True)) == "cuda"
    assert select_device("auto", FakeCuda(False)) == "cpu"


def test_explicit_unavailable_cuda_fails() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device("cuda", FakeCuda(False))


def test_supported_dependency_pair() -> None:
    assert check_compatibility("2.15.4", "4.51.3").compatible


def test_transformers_5_is_rejected_with_actionable_message() -> None:
    result = check_compatibility("2.15.4", "5.0.0")
    assert not result.compatible
    assert "pip install -e ." in result.message
