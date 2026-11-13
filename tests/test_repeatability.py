import numpy as np
import pytest

from surge_repeatability import evaluate_metrics, measure_repeats


def metrics_fixture() -> dict[str, np.ndarray]:
    return {
        "pairwise_stft_power_magnitude_distance": np.zeros((3, 9, 4)),
        "pairwise_stft_short_power_magnitude_distance": np.zeros((3, 9, 4)),
        "pairwise_log_mel_rms_db": np.zeros((3, 9, 4)),
        "active_mel_bins": np.ones((9, 4), dtype=int),
    }


def test_tolerance_accepts_nonzero_error_and_inclusive_boundaries():
    metrics = metrics_fixture()
    metrics["pairwise_stft_power_magnitude_distance"][:] = 0.01
    metrics["pairwise_stft_short_power_magnitude_distance"][:] = 0.009
    metrics["pairwise_log_mel_rms_db"][:] = 0.1
    assert evaluate_metrics(metrics)["accepted"]


@pytest.mark.parametrize("key, value", [
    ("pairwise_stft_power_magnitude_distance", 0.0101),
    ("pairwise_stft_short_power_magnitude_distance", 0.018),
    ("pairwise_log_mel_rms_db", 0.1001),
])
def test_one_failing_pair_or_grid_point_rejects_patch_even_if_average_is_small(key, value):
    metrics = metrics_fixture()
    metrics[key][2, 8, 3] = value
    result = evaluate_metrics(metrics)
    assert not result["accepted"]
    assert len(result["failed_comparisons"]) == 1
    failure = result["failed_comparisons"][0]
    assert (failure["repeat_pair_index"], failure["pitch_index"], failure["velocity_index"]) == (2, 8, 3)


def test_inactive_cells_cannot_pass_as_zero_variation():
    metrics = metrics_fixture()
    metrics["active_mel_bins"][0, 0] = 0
    result = evaluate_metrics(metrics)
    assert not result["accepted"]
    assert result["inactive_grid_cells"] == [[0, 0]]


def test_invalid_tolerances_and_nonfinite_evidence_are_rejected():
    with pytest.raises(ValueError, match="positive finite"):
        evaluate_metrics(metrics_fixture(), magnitude_limit=float("inf"))
    metrics = metrics_fixture()
    metrics["pairwise_log_mel_rms_db"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluate_metrics(metrics)


def test_per_resolution_measurement_does_not_hide_unstable_short_window():
    base = {
        "stft_power": np.ones((1, 1, 2, 4), dtype=np.float32),
        "stft_short_power": np.ones((1, 1, 2, 4), dtype=np.float32),
        "log_mel_db": np.zeros((1, 1, 2, 4), dtype=np.float32),
        "rms": np.ones((1, 1, 4), dtype=np.float32),
    }
    changed = {key: value.copy() for key, value in base.items()}
    changed["stft_short_power"] *= 1.018 ** 2
    measured = measure_repeats([base, changed])
    assert measured["pairwise_multires_magnitude_distance"][0, 0, 0] < 0.01
    assert measured["pairwise_stft_short_power_magnitude_distance"][0, 0, 0] > 0.01
    assert not evaluate_metrics(measured)["accepted"]
