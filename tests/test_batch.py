import numpy as np
import pytest

from surge_batch import repeat_distances, repeat_statistics, save_immutable
from surge_dataset import extract_features, validate_audio


def small_features(amplitude: float) -> dict[str, np.ndarray]:
    power = np.full((1, 1, 2, 3), amplitude ** 2, dtype=np.float32)
    return {
        "stft_power": power,
        "stft_short_power": power.copy(),
        "log_mel_db": np.full_like(power, 20 * np.log10(amplitude)),
        "rms": np.full((1, 1, 3), amplitude, dtype=np.float32),
    }


def test_repeat_statistics_are_sample_variance_not_waveform_averaging():
    values = np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.float32)
    mean, variance = repeat_statistics(values)
    np.testing.assert_array_equal(mean, [[3, 4]])
    np.testing.assert_array_equal(variance, [[4, 4]])
    with pytest.raises(ValueError, match="two repeats"):
        repeat_statistics(values[:1])


def test_repeat_distance_is_zero_for_identical_and_detects_amplitude_changes():
    quiet = small_features(0.1)
    loud = small_features(0.2)
    same = repeat_distances([quiet, quiet, quiet])
    assert same["pairwise_log_mel_rms_db"].shape == (3, 1, 1)
    assert not same["pairwise_log_mel_rms_db"].any()
    assert not same["pairwise_multires_magnitude_distance"].any()
    changed = repeat_distances([quiet, loud])
    np.testing.assert_allclose(changed["pairwise_log_mel_rms_db"], 20 * np.log10(2), atol=1e-5)
    np.testing.assert_allclose(changed["pairwise_multires_magnitude_distance"], 2 / 3, atol=1e-6)
    with pytest.raises(ValueError, match="two repeats"):
        repeat_distances([quiet])


def test_silent_grid_cells_are_explicit_and_never_look_like_active_matches():
    silence = np.zeros((2, 96000), dtype=np.float32)
    with pytest.raises(ValueError, match="silent"):
        validate_audio(silence)
    features = extract_features(silence, require_signal=False)
    repeats = [{key: value[None, None] for key, value in features.items()}] * 3
    for repeated in repeats:
        repeated["stft_short_power"] = repeated["stft_power"]
    result = repeat_distances(repeats)
    assert not result["active_mel_bins"].any()
    assert not result["pairwise_multires_magnitude_distance"].any()
    assert np.isfinite(features["mfcc"]).all()


def test_resume_never_overwrites_or_partially_rewrites_patch_provenance(tmp_path):
    save_immutable(tmp_path, {"original.component": b"old state"})
    with pytest.raises(ValueError, match="refusing to overwrite"):
        save_immutable(tmp_path, {"sanitizer.json": b"new report", "original.component": b"different state"})
    assert (tmp_path / "original.component").read_bytes() == b"old state"
    assert not (tmp_path / "sanitizer.json").exists()
    save_immutable(tmp_path, {"original.component": b"old state", "sanitizer.json": b"valid report"})
    assert (tmp_path / "sanitizer.json").read_bytes() == b"valid report"
