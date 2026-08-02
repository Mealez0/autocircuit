import pytest

from autocircuit.baseline import baseline_passes, metrics_from_differences


def test_baseline_metrics_from_synthetic_differences() -> None:
    metrics = metrics_from_differences(
        [2.0, 1.0, -1.0, 2.0], [-2.0, -1.0, 1.0, -2.0], failed_count=1
    )
    assert metrics.clean_accuracy == pytest.approx(0.75)
    assert metrics.corrupt_accuracy == pytest.approx(0.75)
    assert metrics.clean_mean_logit_difference == pytest.approx(1.0)
    assert metrics.corrupt_mean_logit_difference == pytest.approx(-1.0)
    assert metrics.clean_corrupt_contrast == pytest.approx(2.0)
    assert metrics.failed_count == 1
    assert not baseline_passes(metrics)


def test_baseline_frozen_acceptance_thresholds() -> None:
    passing = metrics_from_differences([1.5] * 8 + [-1.0] * 2, [-1.0] * 10)
    assert passing.clean_accuracy == 0.8
    assert baseline_passes(passing)
    with pytest.raises(ValueError, match="non-zero"):
        metrics_from_differences([], [])
