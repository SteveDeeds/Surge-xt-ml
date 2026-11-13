"""Verify the complete saved initial-patch dataset without running the synth."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "complete", "Dataset is incomplete")
    require(len(manifest["samples"]) == 36, "Expected 36 recordings")
    expected = {(note, velocity) for note in range(12, 109, 12) for velocity in (32, 64, 96, 127)}
    actual = {(sample["midi_note"], sample["velocity"]) for sample in manifest["samples"]}
    require(actual == expected, "Pitch/velocity coverage differs from requested grid")
    settings = manifest["render"]
    require(settings["hold_seconds"] == settings["tail_seconds"] == 1, "Incorrect note/tail duration")
    require(settings["sample_rate"] == 48000, "Incorrect sample rate")
    require(settings["preset_restored_per_recording"], "Missing per-recording state restoration")
    require(
        hashlib.sha256((directory / "patch.vstpreset").read_bytes()).hexdigest()
        == manifest["patch"]["preset_sha256"], "Patch state checksum mismatch",
    )
    max_pitch_error = 0.0
    for sample in manifest["samples"]:
        path = directory / sample["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == sample["sha256"],
                f"Checksum mismatch: {path.name}")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        info = sf.info(path)
        require(rate == 48000 and audio.shape == (96000, 2), f"Invalid audio format: {path.name}")
        require(info.subtype == "FLOAT", f"Not a float WAV: {path.name}")
        require(bool(np.isfinite(audio).all()), f"Nonfinite audio: {path.name}")
        require(float(np.max(np.abs(audio))) > 1e-8, f"Silent audio: {path.name}")
        # This pitch check applies to the simple initial patch, not arbitrary future patches.
        held = audio[9600:43200]
        spectrum = np.fft.rfft(held * np.hanning(len(held))[:, None], n=262144, axis=0)
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
        peak_bin = int(np.argmax(power[1:]) + 1)
        measured_hz = peak_bin * rate / 262144
        error = abs(12 * np.log2(measured_hz / sample["frequency_hz"]))
        max_pitch_error = max(max_pitch_error, float(error))
        require(error < 0.5, f"Initial-patch pitch mismatch: {path.name}, {error:.3f} semitones")
    shapes = {
        "stft_power": (9, 4, 4097, 188),
        "log_mel_db": (9, 4, 128, 188),
        "mfcc": (9, 4, 20, 188),
        "rms": (9, 4, 188),
    }
    with np.load(directory / "features.npz", allow_pickle=False) as features:
        for key, expected_shape in shapes.items():
            values = features[key]
            require(values.shape == expected_shape, f"Incorrect {key} shape: {values.shape}")
            require(values.dtype == np.float32, f"Incorrect {key} dtype")
            require(bool(np.isfinite(values).all()), f"Nonfinite {key}")
        np.testing.assert_array_equal(features["pitches_midi"], np.arange(12, 109, 12))
        np.testing.assert_array_equal(features["velocities"], [32, 64, 96, 127])
        np.testing.assert_allclose(features["time_seconds"], np.arange(188) * 512 / 48000)
        require(features["stft_frequency_hz"].shape == (4097,), "Missing STFT coordinates")
        require(features["mel_center_hz"].shape == (128,), "Missing mel coordinates")
    require((directory / "preview.png").is_file(), "Missing preview")
    return {
        "verified_recordings": 36,
        "total_audio_seconds": 72,
        "max_initial_patch_pitch_error_semitones": max_pitch_error,
        "max_last_50ms_rms": max(sample["last_50ms_rms"] for sample in manifest["samples"]),
        "feature_shapes": shapes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    print(json.dumps(result, indent=2))
