"""Paired original/sanitized factory grids with repeat-level data and uncertainty."""

import argparse
import gc
import importlib.metadata
import itertools
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pedalboard import VST3Plugin

from surge_dataset import (
    DEFAULT_PLUGIN, HOLD_SECONDS, HOP_LENGTH, N_FFT, N_MELS, PITCHES,
    PRE_ROLL_SECONDS, ROOT, SAMPLE_RATE, TAIL_SECONDS, VELOCITIES,
    extract_features, mel_filterbank, render_note, save_preview, sha256, write_json,
)
from surge_patches import (
    audit_patch, component_from_host, component_to_host, parse_patch,
    read_fxp, sanitize_patch,
)

FACTORY = ROOT / ".tools" / "SurgeXT-1.3.4" / "Surge Synth Team" / "SurgeXTData" / "patches_factory"
PATCHES = (
    ("bass_fm", "Basses", "FM Bass 1.fxp"),
    ("keys_artificial", "Keys", "Artificial 1.fxp"),
    ("lead_acidofil", "Leads", "Acidofil.fxp"),
    ("pad_drift", "Pads", "Alias Pornography.fxp"),
    ("brass_brassy", "Brass", "Brassy.fxp"),
    ("wind_clarinet", "Winds", "Clarinet.fxp"),
    ("pluck_gliss", "Plucks", "80s Gliss.fxp"),
    ("percussion_drum", "Percussion", "Drum One.fxp"),
    ("poly_1804", "Polysynths", "1804.fxp"),
    ("fx_aggero", "FX", "Aggero.fxp"),
)
REPEATS = 3
SHORT_FFT = 2048
ISOLATION_PEAK_LIMIT = 1e-7


def prime(plugin: VST3Plugin) -> None:
    plugin.process(
        [], duration=PRE_ROLL_SECONDS, sample_rate=SAMPLE_RATE,
        num_channels=2, buffer_size=512, reset=True,
    )


def set_component(plugin: VST3Plugin, component: bytes) -> bytes:
    plugin.raw_state = component_to_host(component)
    prime(plugin)
    actual = component_from_host(plugin.raw_state)
    expected_meta = parse_patch(component).find("meta")
    actual_meta = parse_patch(actual).find("meta")
    if expected_meta is None or actual_meta is None or expected_meta.get("name") != actual_meta.get("name"):
        raise ValueError("Factory patch identity did not survive native state restoration")
    return actual


def check_isolation(plugin: VST3Plugin, state: bytes) -> dict:
    plugin.preset_data = state
    prime(plugin)
    output = plugin.process(
        [], duration=2.0, sample_rate=SAMPLE_RATE,
        num_channels=2, buffer_size=512, reset=False,
    )
    if output.shape != (2, 96000) or not np.isfinite(output).all():
        raise ValueError("Invalid isolation probe output")
    peak = float(np.max(np.abs(output)))
    if peak >= ISOLATION_PEAK_LIMIT:
        raise ValueError(f"Non-silent output after patch restoration: peak={peak}")
    return {"no_midi_seconds": 2, "peak": peak, "required_peak_below": ISOLATION_PEAK_LIMIT}


def batch_features(audio: np.ndarray) -> dict[str, np.ndarray]:
    features = extract_features(audio, require_signal=False)
    short = librosa.stft(
        audio, n_fft=SHORT_FFT, hop_length=HOP_LENGTH, center=True, pad_mode="constant",
    )
    features["stft_short_power"] = np.mean(np.abs(short) ** 2, axis=0).astype(np.float32)
    return features


def repeat_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] < 2:
        raise ValueError("At least two repeats are required for sample variance")
    return (
        np.mean(values, axis=0, dtype=np.float64).astype(np.float32),
        np.var(values, axis=0, ddof=1, dtype=np.float64).astype(np.float32),
    )


def repeat_distances(repeats: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if len(repeats) < 2:
        raise ValueError("At least two repeats are required")
    mean_mel = np.mean([r["log_mel_db"] for r in repeats], axis=0)
    active = mean_mel > np.max(mean_mel, axis=(-2, -1), keepdims=True) - 60
    active &= np.mean([r["rms"] for r in repeats], axis=0)[..., None, :] > 1e-5
    mel_distances, mr_distances = [], []
    for left, right in itertools.combinations(repeats, 2):
        diff = left["log_mel_db"] - right["log_mel_db"]
        counts = np.sum(active, axis=(-2, -1))
        mel_distances.append(np.sqrt(
            np.sum(np.where(active, diff ** 2, 0), axis=(-2, -1))
            / np.maximum(counts, 1)
        ))
        resolutions = []
        for key in ("stft_power", "stft_short_power"):
            a, b = np.sqrt(left[key]), np.sqrt(right[key])
            numerator = np.sqrt(np.sum((a - b) ** 2, axis=(-2, -1), dtype=np.float64))
            denominator = (
                np.sqrt(np.sum(a ** 2, axis=(-2, -1), dtype=np.float64))
                + np.sqrt(np.sum(b ** 2, axis=(-2, -1), dtype=np.float64))
            ) / 2
            resolutions.append(numerator / np.maximum(denominator, 1e-12))
        mr_distances.append(np.mean(resolutions, axis=0))
    return {
        "pairwise_log_mel_rms_db": np.asarray(mel_distances, dtype=np.float32),
        "pairwise_multires_magnitude_distance": np.asarray(mr_distances, dtype=np.float32),
        "active_mel_bins": np.sum(active, axis=(-2, -1)),
    }


def coordinates() -> dict[str, np.ndarray]:
    return {
        "pitches_midi": np.array(PITCHES), "velocities": np.array(VELOCITIES),
        "time_seconds": np.arange(188) * HOP_LENGTH / SAMPLE_RATE,
        "stft_frequency_hz": librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT),
        "stft_short_frequency_hz": librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=SHORT_FFT),
        "mel_center_hz": librosa.mel_frequencies(n_mels=N_MELS + 2, fmax=SAMPLE_RATE / 2)[1:-1],
    }


def save_immutable(directory: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        path = directory / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Existing provenance differs; refusing to overwrite: {path}")
    for name, data in files.items():
        path = directory / name
        if not path.exists():
            path.write_bytes(data)


def render_variant(plugin: VST3Plugin, output: Path, state: bytes) -> dict:
    output.mkdir()
    (output / "audio").mkdir()
    (output / "patch.vstpreset").write_bytes(state)
    manifest = {
        "status": "in_progress", "state_sha256": sha256(state),
        "samples": [], "repeats": REPEATS,
        "isolation_before": check_isolation(plugin, state),
    }
    write_json(output / "manifest.json", manifest)
    paths = []
    for repeat in range(REPEATS):
        tensors: dict[str, list[np.ndarray]] = {}
        for pitch_index, note in enumerate(PITCHES):
            for velocity in VELOCITIES:
                audio = render_note(plugin, state, note, velocity, require_signal=False)
                features = batch_features(audio)
                for key, value in features.items():
                    if not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite {key} features")
                    tensors.setdefault(key, []).append(value)
                name = f"r{repeat}_C{pitch_index}_midi{note:03d}_v{velocity:03d}.wav"
                path = output / "audio" / name
                sf.write(path, audio.T, SAMPLE_RATE, subtype="FLOAT")
                peak = float(np.max(np.abs(audio)))
                manifest["samples"].append({
                    "repeat": repeat, "midi_note": note, "velocity": velocity,
                    "path": f"audio/{name}", "sha256": sha256(path.read_bytes()),
                    "waveform_sha256": sha256(audio.tobytes()), "frames": audio.shape[1],
                    "peak": peak, "silent": peak < 1e-8, "over_full_scale": peak >= 1,
                    "held_rms": float(np.sqrt(np.mean(audio[:, :48000] ** 2))),
                    "tail_rms": float(np.sqrt(np.mean(audio[:, 48000:] ** 2))),
                    "last_50ms_rms": float(np.sqrt(np.mean(audio[:, -2400:] ** 2))),
                })
                if repeat == 0 and note == 60 and velocity == 96:
                    save_preview(output / "preview.png", audio, features)
                write_json(output / "manifest.json", manifest)
        arrays = {
            key: np.stack(values).reshape(9, 4, *values[0].shape)
            for key, values in tensors.items()
        }
        feature_path = output / f"features_r{repeat}.npz"
        np.savez_compressed(feature_path, **arrays, **coordinates())
        paths.append(feature_path)
        del tensors, arrays, features, audio
        gc.collect()
        print(f"{output.parent.name}/{output.name}: repeat {repeat + 1}/{REPEATS}", flush=True)
    if all(sample["silent"] for sample in manifest["samples"]):
        raise ValueError(f"Patch produced no signal anywhere on the requested grid: {output}")
    manifest["isolation_after"] = check_isolation(plugin, state)
    means, variances = {}, {}
    with np.load(paths[0], allow_pickle=False) as first:
        keys = [key for key in first.files if key not in coordinates()]
    # Read one representation at a time to bound memory while retaining all repeats.
    for key in keys:
        values = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                values.append(data[key])
        means[key], variances[key] = repeat_statistics(np.stack(values))
        del values
    mean_mel_power = mel_filterbank() @ means["stft_power"]
    mean_power_log_mel = librosa.power_to_db(mean_mel_power, ref=1.0, amin=1e-10, top_db=None)
    np.savez_compressed(
        output / "feature_mean.npz", **means,
        log_mel_of_mean_power=mean_power_log_mel.astype(np.float32), **coordinates(),
    )
    np.savez_compressed(output / "feature_variance.npz", **variances, **coordinates())
    del means, variances
    repeated = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            repeated.append({
                key: data[key] for key in ("stft_power", "stft_short_power", "log_mel_db", "rms")
            })
    distances = repeat_distances(repeated)
    np.savez_compressed(
        output / "repeatability.npz", **distances,
        pairs=np.array(list(itertools.combinations(range(REPEATS), 2))),
        pitches_midi=np.array(PITCHES), velocities=np.array(VELOCITIES),
    )
    waveform_identical = []
    for note in PITCHES:
        for velocity in VELOCITIES:
            hashes = {
                sample["waveform_sha256"] for sample in manifest["samples"]
                if sample["midi_note"] == note and sample["velocity"] == velocity
            }
            waveform_identical.append(len(hashes) == 1)
    manifest["summary"] = {
        "recordings": len(manifest["samples"]),
        "silent_recordings": sum(s["silent"] for s in manifest["samples"]),
        "over_full_scale_recordings": sum(s["over_full_scale"] for s in manifest["samples"]),
        "waveform_identical_grid_points": sum(waveform_identical),
        "grid_points": 36,
        "multires_magnitude_distance_mean": float(np.mean(distances["pairwise_multires_magnitude_distance"])),
        "multires_magnitude_distance_max": float(np.max(distances["pairwise_multires_magnitude_distance"])),
        "active_log_mel_rms_db_mean": float(np.mean(distances["pairwise_log_mel_rms_db"])),
        "active_log_mel_rms_db_max": float(np.max(distances["pairwise_log_mel_rms_db"])),
        "no_active_signal_grid_points": int(np.sum(distances["active_mel_bins"] == 0)),
        "residual_variation_detected": not all(waveform_identical),
        "max_last_50ms_rms": max(s["last_50ms_rms"] for s in manifest["samples"]),
        "deterministic_claim": False,
    }
    manifest["files_sha256"] = {
        p.name: sha256(p.read_bytes()) for p in sorted(output.glob("*.npz"))
    }
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    return manifest["summary"]


def run_batch(output: Path, *, resume: bool = False) -> None:
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {output}; use --resume for completed variants")
    output.mkdir(parents=True, exist_ok=True)
    plan = [{"id": slug, "factory_path": str(Path(category) / name)}
            for slug, category, name in PATCHES]
    source_hashes = {p["id"]: sha256(read_fxp(FACTORY / p["factory_path"])) for p in plan}
    settings = {
        "patches": plan, "component_sha256": source_hashes,
        "variants": ["original", "sanitized"], "repeats": REPEATS,
        "sample_rate": SAMPLE_RATE, "channels": 2, "wav_subtype": "FLOAT",
        "hold_seconds": HOLD_SECONDS, "tail_seconds": TAIL_SECONDS,
        "pitches_midi": PITCHES, "velocities": VELOCITIES,
        "n_fft": N_FFT, "short_n_fft": SHORT_FFT, "hop_length": HOP_LENGTH,
        "window": "hann", "center": True, "pad_mode": "constant",
        "n_mels": N_MELS, "mel_scale": "slaney", "mel_norm": "slaney",
        "db_reference": 1.0, "db_amin": 1e-10, "db_top_db": None,
        "n_mfcc": 20, "mfcc_dct_type": 2, "mfcc_norm": "ortho",
        "stereo_reduction": "mean channel powers",
        "normalization": "none", "preset_restored_per_recording": True,
        "pre_roll_seconds": PRE_ROLL_SECONDS, "buffer_size": 512,
        "seed": None, "variance_ddof": 1,
        "pitch_convention": "scientific; C0=MIDI12, C8=MIDI108",
        "repeat_tensor_axes": ["pitch", "velocity", "feature", "time"],
        "repeat_distance_axes": ["repeat_pair", "pitch", "velocity"],
        "log_mel_distance_mask": "mean log-mel within 60 dB of cell maximum AND mean RMS>1e-5",
        "multires_distance": "mean across FFT sizes of magnitude L2 difference divided by mean L2 norm",
        "sanitizer": "retrigger_on_drift_zero_keytrigger_remove_random_alternate",
        "noise_lfo_policy": "retain noise/S&H/formula textures; measure three repeats",
        "transport_policy": "same instance, full state restore and fixed DSP pre-roll for every recording",
    }
    manifest_path = output / "batch.json"
    if manifest_path.exists():
        batch = json.loads(manifest_path.read_text(encoding="utf-8"))
        if batch["settings"] != json.loads(json.dumps(settings)):
            raise ValueError("Resume settings/source patches differ from saved batch")
        if batch["pipeline_sha256"] != pipeline_hashes():
            raise ValueError("Resume pipeline code differs from saved batch")
    else:
        batch = {
            "status": "in_progress", "created_utc": datetime.now(timezone.utc).isoformat(),
            "settings": settings, "pipeline_sha256": pipeline_hashes(),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("pedalboard", "numpy", "librosa", "soundfile", "matplotlib")},
            "plugin_sha256": sha256(DEFAULT_PLUGIN.read_bytes()),
            "thread_environment": {
                name: os.environ.get(name) for name in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
            },
            "patches": [],
        }
        write_json(manifest_path, batch)
    if batch["plugin_sha256"] != sha256(DEFAULT_PLUGIN.read_bytes()):
        raise ValueError("Resume plugin binary differs")
    current_threads = {
        name: os.environ.get(name) for name in
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
    }
    if batch["thread_environment"] != current_threads:
        raise ValueError("Resume numerical thread environment differs")
    snapshot = output / "pipeline"
    snapshot.mkdir(exist_ok=True)
    save_immutable(snapshot, {name: (ROOT / name).read_bytes() for name in pipeline_hashes()})
    started = time.perf_counter()
    plugin = VST3Plugin(str(DEFAULT_PLUGIN))
    for slug, category, name in PATCHES:
        directory = output / slug
        directory.mkdir(exist_ok=True)
        fxp_path = FACTORY / category / name
        factory = read_fxp(fxp_path)
        canonical = set_component(plugin, factory)
        original_state = plugin.preset_data
        sanitized, report = sanitize_patch(canonical)
        actual = set_component(plugin, sanitized)
        # Verify the engine retained the requested controls and removed routes.
        _, verification = sanitize_patch(actual)
        if verification["edits"] or verification["removed_routes"]:
            raise ValueError(f"Sanitizer controls were not retained by the engine: {slug}")
        sanitized_state = plugin.preset_data
        report["engine_readback_verified"] = True
        save_immutable(directory, {
            "sanitizer.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"),
            "original.component": canonical,
            "sanitized.component": actual,
            "source.fxp": fxp_path.read_bytes(),
        })
        entry = {
            "id": slug, "factory_path": str(Path(category) / name),
            "fxp_sha256": sha256(fxp_path.read_bytes()),
            "original_component_sha256": sha256(canonical),
            "sanitized_component_sha256": sha256(actual),
            "original_audit": audit_patch(canonical), "variants": {},
        }
        for variant, state in (("original", original_state), ("sanitized", sanitized_state)):
            target = directory / variant
            if target.exists():
                saved = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
                if saved["status"] != "complete":
                    raise ValueError(f"Incomplete variant needs inspection before resuming: {target}")
                if saved["state_sha256"] != sha256(state):
                    raise ValueError(f"Saved state differs while resuming: {target}")
                entry["variants"][variant] = saved["summary"]
            else:
                entry["variants"][variant] = render_variant(plugin, target, state)
            batch["patches"] = [p for p in batch["patches"] if p["id"] != slug] + [entry]
            write_json(manifest_path, batch)
        del canonical, sanitized, actual, original_state, sanitized_state
        gc.collect()
    batch["status"] = "complete"
    batch["elapsed_seconds_this_run"] = time.perf_counter() - started
    batch["recordings"] = sum(
        v["recordings"] for p in batch["patches"] for v in p["variants"].values()
    )
    write_json(manifest_path, batch)
    print(json.dumps({"output": str(output), "recordings": batch["recordings"]}), flush=True)


def pipeline_hashes() -> dict[str, str]:
    return {name: sha256((ROOT / name).read_bytes())
            for name in ("surge_batch.py", "surge_patches.py", "surge_dataset.py")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "factory_ten")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_batch(args.output, resume=args.resume)
