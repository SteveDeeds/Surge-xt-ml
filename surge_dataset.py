"""Render one Surge XT patch over a fixed pitch/velocity grid, then extract features."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import warnings
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import numpy.typing as npt
import soundfile as sf
from pedalboard import VST3Plugin

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEFAULT_PLUGIN = (
    ROOT / ".tools" / "SurgeXT-1.3.4" / "Surge Synth Team" / "Surge XT.vst3"
    / "Contents" / "x86_64-win" / "Surge XT.vst3"
)
SAMPLE_RATE = 48_000
HOLD_SECONDS = 1.0
TAIL_SECONDS = 1.0
PITCHES = tuple(range(12, 109, 12))
VELOCITIES = (32, 64, 96, 127)
N_FFT = 8192
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20
PRE_ROLL_SECONDS = 0.1
FloatArray = npt.NDArray[np.float32]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def note_events(note: int, velocity: int) -> list[tuple[bytes, float]]:
    return [
        (bytes([0x90, note, velocity]), 0.0),
        (bytes([0x80, note, 0]), HOLD_SECONDS),
    ]


def validate_audio(audio: FloatArray, *, require_signal: bool = True) -> None:
    expected = (2, int(SAMPLE_RATE * (HOLD_SECONDS + TAIL_SECONDS)))
    if audio.shape != expected:
        raise ValueError(f"Expected stereo audio {expected}, got {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError("Synth returned nonfinite audio")
    if require_signal and np.max(np.abs(audio)) < 1e-8:
        raise ValueError("Synth returned a silent recording")


def render_note(
    plugin: VST3Plugin, state: bytes, note: int, velocity: int, *,
    require_signal: bool = True,
) -> FloatArray:
    plugin.preset_data = state
    # Surge applies queued patch loads on the audio thread, before it can keep MIDI notes.
    plugin.process(
        [], duration=PRE_ROLL_SECONDS, sample_rate=SAMPLE_RATE,
        num_channels=2, buffer_size=512, reset=True,
    )
    audio = plugin.process(
        note_events(note, velocity), duration=HOLD_SECONDS + TAIL_SECONDS,
        sample_rate=SAMPLE_RATE, num_channels=2, buffer_size=512, reset=False,
    )
    validate_audio(audio, require_signal=require_signal)
    return audio


@cache
def mel_filterbank() -> FloatArray:
    return librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0,
        fmax=SAMPLE_RATE / 2, htk=False, norm="slaney",
    )


def extract_features(
    audio: FloatArray, *, require_signal: bool = True,
) -> dict[str, FloatArray]:
    validate_audio(audio, require_signal=require_signal)
    spectrum = librosa.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True, pad_mode="constant"
    )
    # Average channel powers rather than waveforms to avoid stereo phase cancellation.
    power = np.mean(np.abs(spectrum) ** 2, axis=0).astype(np.float32)
    mel = mel_filterbank() @ power
    log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC, dct_type=2, norm="ortho")
    channel_rms = librosa.feature.rms(
        y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH,
        center=True, pad_mode="constant",
    )
    rms = np.sqrt(np.mean(channel_rms[:, 0, :] ** 2, axis=0))
    return {
        "stft_power": power,
        "log_mel_db": log_mel.astype(np.float32),
        "mfcc": mfcc.astype(np.float32),
        "rms": rms.astype(np.float32),
    }


def save_preview(
    path: Path, audio: FloatArray, features: dict[str, FloatArray]
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), constrained_layout=True)
    axes[0].plot(np.arange(audio.shape[1]) / SAMPLE_RATE, audio.T, linewidth=0.4)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("C4, velocity 96: 1 s held + 1 s release")
    time_step = HOP_LENGTH / SAMPLE_RATE
    end = (features["mfcc"].shape[-1] - 0.5) * time_step
    for ax, key, label in zip(
        axes[1:], ("stft_power", "log_mel_db", "mfcc"),
        ("STFT frequency (Hz)", "Mel band", "MFCC coefficient"), strict=True,
    ):
        matrix = features[key]
        if key == "stft_power":
            matrix = librosa.power_to_db(matrix, ref=1.0, amin=1e-10, top_db=None)
        upper = SAMPLE_RATE / 2 if key == "stft_power" else matrix.shape[0]
        image = ax.imshow(
            matrix, origin="lower", aspect="auto",
            extent=(-time_step / 2, end, 0, upper), cmap="magma",
        )
        ax.set_ylabel(label)
        fig.colorbar(image, ax=ax, label="dB (ref=1)" if key != "mfcc" else "Value")
    for ax in axes:
        ax.axvline(HOLD_SECONDS, color="cyan", linestyle="--", linewidth=1)
        ax.set_xlim(0, HOLD_SECONDS + TAIL_SECONDS)
        ax.set_xlabel("Time (s); dashed line = note-off")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_dataset(plugin_path: Path, output: Path, preset: Path | None) -> None:
    if not plugin_path.exists():
        raise FileNotFoundError(f"Surge XT VST3 not found: {plugin_path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dataset: {output}")
    plugin = VST3Plugin(str(plugin_path))
    if not plugin.is_instrument:
        raise ValueError(f"Not an instrument: {plugin_path}")
    if preset is not None:
        plugin.load_preset(str(preset.resolve(strict=True)))
    plugin.process(
        [], duration=PRE_ROLL_SECONDS, sample_rate=SAMPLE_RATE,
        num_channels=2, buffer_size=512, reset=True,
    )
    state = plugin.preset_data
    raw_state = plugin.raw_state
    parameters = {
        name: {"raw_value": float(parameter.raw_value), "display": str(getattr(plugin, name))}
        for name, parameter in plugin.parameters.items()
    }
    output.mkdir(parents=True)
    wav_dir = output / "audio"
    wav_dir.mkdir()
    (output / "patch.vstpreset").write_bytes(state)
    (output / "patch.raw").write_bytes(raw_state)
    write_json(output / "parameters.json", parameters)
    manifest: dict = {
        "status": "in_progress",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch": {
            "selection": "Surge XT initial patch" if preset is None else preset.name,
            "source_preset": None if preset is None else str(preset.resolve()),
            "preset_sha256": sha256(state),
            "raw_state_sha256": sha256(raw_state),
            "snapshot": "patch.vstpreset",
        },
        "plugin_path": str(plugin_path.resolve()),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("pedalboard", "numpy", "librosa", "soundfile", "matplotlib")
        },
        "render": {
            "sample_rate": SAMPLE_RATE, "channels": 2, "wav_subtype": "FLOAT",
            "hold_seconds": HOLD_SECONDS, "tail_seconds": TAIL_SECONDS,
            "buffer_size": 512, "pitch_convention": "scientific; C0=MIDI12",
            "pitches_midi": PITCHES, "velocities": VELOCITIES,
            "fresh_plugin_per_recording": False,
            "preset_restored_per_recording": True,
            "discarded_silent_pre_roll_seconds": PRE_ROLL_SECONDS,
            "normalization": "none",
            "randomness": "Plugin oscillator/noise RNG is not externally seeded.",
        },
        "analysis": {
            "n_fft": N_FFT, "hop_length": HOP_LENGTH,
            "window": "hann", "center": True, "pad_mode": "constant",
            "n_mels": N_MELS, "mel_scale": "slaney", "mel_norm": "slaney",
            "fmin": 0.0, "fmax": SAMPLE_RATE / 2, "n_mfcc": N_MFCC,
            "mfcc_dct_type": 2, "mfcc_norm": "ortho",
            "db_reference": 1.0, "db_amin": 1e-10, "db_top_db": None,
            "stereo_reduction": "mean of channel powers, not waveforms",
            "tensor_axes": ["pitch", "velocity", "feature", "time"],
            "rms_axes": ["pitch", "velocity", "time"],
            "time_reference": "center of each zero-padded analysis frame",
        },
        "samples": [],
    }
    install = ROOT / ".tools" / "SurgeXT-1.3.4" / "install.json"
    if plugin_path.resolve() == DEFAULT_PLUGIN.resolve() and install.exists():
        manifest["installation"] = json.loads(install.read_text(encoding="utf-8-sig"))
    binaries = sorted(plugin_path.rglob("*.vst3")) if plugin_path.is_dir() else [plugin_path]
    if not binaries:
        raise FileNotFoundError(f"No VST3 binary inside {plugin_path}")
    manifest["plugin_binaries"] = {
        str(binary.relative_to(plugin_path) if plugin_path.is_dir() else binary.name):
        sha256(binary.read_bytes()) for binary in binaries
    }
    write_json(output / "manifest.json", manifest)
    tensors: dict[str, list[FloatArray]] = {}
    preview_audio = None
    preview_features = None
    for octave, note in enumerate(PITCHES):
        for velocity in VELOCITIES:
            audio = render_note(plugin, state, note, velocity)
            features = extract_features(audio)
            for key, value in features.items():
                tensors.setdefault(key, []).append(value)
            name = f"C{octave}_midi{note:03d}_v{velocity:03d}.wav"
            wav_path = wav_dir / name
            sf.write(wav_path, audio.T, SAMPLE_RATE, subtype="FLOAT")
            peak = float(np.max(np.abs(audio)))
            if peak >= 1.0:
                warnings.warn(f"{name}: peak {peak:.3f} exceeds full scale; float WAV retained.")
            split = int(SAMPLE_RATE * HOLD_SECONDS)
            manifest["samples"].append({
                "note": f"C{octave}", "midi_note": note, "velocity": velocity,
                "frequency_hz": float(librosa.midi_to_hz(note)),
                "path": f"audio/{name}", "sha256": sha256(wav_path.read_bytes()),
                "frames": audio.shape[1], "peak": peak,
                "held_rms": float(np.sqrt(np.mean(audio[:, :split] ** 2))),
                "tail_rms": float(np.sqrt(np.mean(audio[:, split:] ** 2))),
                "last_50ms_rms": float(np.sqrt(np.mean(audio[:, -2400:] ** 2))),
            })
            if note == 60 and velocity == 96:
                preview_audio, preview_features = audio, features
            write_json(output / "manifest.json", manifest)
    arrays = {
        key: np.stack(values).reshape(len(PITCHES), len(VELOCITIES), *values[0].shape)
        for key, values in tensors.items()
    }
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {key} features")
    frame_times = np.arange(arrays["log_mel_db"].shape[-1]) * HOP_LENGTH / SAMPLE_RATE
    np.savez_compressed(
        output / "features.npz", **arrays, pitches_midi=np.array(PITCHES),
        velocities=np.array(VELOCITIES), time_seconds=frame_times,
        stft_frequency_hz=librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT),
        mel_center_hz=librosa.mel_frequencies(n_mels=N_MELS + 2, fmax=SAMPLE_RATE / 2)[1:-1],
    )
    if preview_audio is None or preview_features is None:
        raise RuntimeError("C4/velocity96 preview was not generated")
    save_preview(output / "preview.png", preview_audio, preview_features)
    manifest["feature_shapes"] = {key: list(value.shape) for key, value in arrays.items()}
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(output.resolve()), "recordings": len(manifest["samples"]),
        "feature_shapes": manifest["feature_shapes"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "initial_patch")
    parser.add_argument("--preset", type=Path, help="Optional native .vstpreset (not .fxp)")
    args = parser.parse_args()
    build_dataset(args.plugin, args.output, args.preset)


if __name__ == "__main__":
    main()
