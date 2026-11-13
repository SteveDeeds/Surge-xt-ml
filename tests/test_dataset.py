import numpy as np
import pytest
from unittest.mock import MagicMock

from surge_dataset import (
    HOP_LENGTH, N_FFT, N_MELS, N_MFCC, PITCHES, SAMPLE_RATE, VELOCITIES,
    extract_features, note_events, render_note, validate_audio,
)


def sine_audio(amplitude: float = 0.1) -> np.ndarray:
    time = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    channel = (amplitude * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    return np.stack([channel, -channel])


def test_grid_and_note_off_timing():
    assert len(PITCHES) * len(VELOCITIES) == 36
    assert PITCHES == tuple(12 + 12 * octave for octave in range(9))
    for note in PITCHES:
        for velocity in VELOCITIES:
            assert note_events(note, velocity) == [
                (bytes([0x90, note, velocity]), 0.0),
                (bytes([0x80, note, 0]), 1.0),
            ]


def test_preset_load_is_drained_before_first_note():
    plugin = MagicMock()
    expected = sine_audio()
    plugin.process.side_effect = [np.zeros((2, 4800), dtype=np.float32), expected]
    actual = render_note(plugin, b"saved patch", 60, 96)
    assert plugin.preset_data == b"saved patch"
    pre_roll, recording = plugin.process.call_args_list
    assert pre_roll.args == ([],)
    assert pre_roll.kwargs["duration"] == 0.1
    assert pre_roll.kwargs["reset"] is True
    assert recording.args == (note_events(60, 96),)
    assert recording.kwargs["duration"] == 2.0
    assert recording.kwargs["reset"] is False
    assert actual is expected


def test_features_preserve_energy_and_grid_shape():
    features = extract_features(sine_audio())
    frames = 1 + 2 * SAMPLE_RATE // HOP_LENGTH
    assert features["stft_power"].shape == (N_FFT // 2 + 1, frames)
    assert features["log_mel_db"].shape == (N_MELS, frames)
    assert features["mfcc"].shape == (N_MFCC, frames)
    assert features["rms"].shape == (frames,)
    for value in features.values():
        assert np.isfinite(value).all()
    peak_bin = np.argmax(features["stft_power"][:, frames // 2])
    assert abs(peak_bin * SAMPLE_RATE / N_FFT - 440) <= SAMPLE_RATE / N_FFT
    assert features["rms"][frames // 2] == pytest.approx(0.1 / np.sqrt(2), rel=0.01)


def test_log_mel_uses_shared_reference_not_per_recording_normalization():
    quiet = extract_features(sine_audio(0.05))
    loud = extract_features(sine_audio(0.1))
    active = quiet["log_mel_db"] > -60
    np.testing.assert_allclose(
        (loud["log_mel_db"] - quiet["log_mel_db"])[active],
        20 * np.log10(2), atol=1e-4,
    )


@pytest.mark.parametrize("kind", ["short", "silent", "nonfinite"])
def test_invalid_render_is_rejected(kind):
    audio = sine_audio()
    if kind == "short":
        audio = audio[:, :-1]
    elif kind == "silent":
        audio[:] = 0
    else:
        audio[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_audio(audio)
