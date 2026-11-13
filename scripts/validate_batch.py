"""Verify paired factory data and repeat statistics. Run with python -m scripts.validate_batch."""

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import soundfile as sf

from surge_batch import (
    FACTORY, PATCHES, REPEATS, coordinates,
    repeat_distances, repeat_statistics,
)
from surge_dataset import DEFAULT_PLUGIN, PITCHES, VELOCITIES
from surge_patches import parse_patch, sanitize_patch

SHAPES = {
    "stft_power": (9, 4, 4097, 188),
    "stft_short_power": (9, 4, 1025, 188),
    "log_mel_db": (9, 4, 128, 188),
    "mfcc": (9, 4, 20, 188),
    "rms": (9, 4, 188),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preset_component(path: Path) -> bytes:
    data = path.read_bytes()
    require(data[:4] == b"VST3" and len(data) >= 48, f"Invalid VST3 preset: {path}")
    offset = struct.unpack_from("<Q", data, 40)[0]
    require(data[offset:offset + 4] == b"List", f"Missing VST3 chunk list: {path}")
    count = struct.unpack_from("<I", data, offset + 4)[0]
    require(offset + 8 + count * 20 <= len(data), f"Truncated VST3 chunk list: {path}")
    for index in range(count):
        position = offset + 8 + index * 20
        if data[position:position + 4] == b"Comp":
            start, size = struct.unpack_from("<QQ", data, position + 4)
            require(start + size <= len(data), f"Truncated component: {path}")
            return data[start:start + size]
    raise ValueError(f"No component in preset: {path}")


def validate_variant(directory: Path, expected_component: bytes) -> dict:
    manifest = read_json(directory / "manifest.json")
    require(manifest["status"] == "complete", f"Incomplete variant: {directory}")
    require(manifest["repeats"] == REPEATS, "Incorrect repeat count")
    require(digest(directory / "patch.vstpreset") == manifest["state_sha256"], "State checksum mismatch")
    require(preset_component(directory / "patch.vstpreset") == expected_component,
            f"VST3 preset and audited component differ: {directory}")
    require(len(manifest["samples"]) == 108, f"Expected 108 recordings: {directory}")
    expected_cells = set(
        (repeat, note, velocity)
        for repeat in range(REPEATS) for note in PITCHES for velocity in VELOCITIES
    )
    cells = {(s["repeat"], s["midi_note"], s["velocity"]) for s in manifest["samples"]}
    require(cells == expected_cells, f"Missing pitch/velocity/repeat cells: {directory}")
    require(len(list((directory / "audio").glob("*.wav"))) == 108, "Unexpected WAV count")
    for position in ("before", "after"):
        isolation = manifest[f"isolation_{position}"]
        require(isolation["peak"] < 1e-7 and isolation["no_midi_seconds"] == 2,
                f"Isolation check failed: {directory}")
    for sample in manifest["samples"]:
        path = directory / sample["path"]
        require(digest(path) == sample["sha256"], f"WAV checksum mismatch: {path}")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 48000 and audio.shape == (96000, 2), f"Invalid WAV: {path}")
        require(sf.info(path).subtype == "FLOAT", f"Not float WAV: {path}")
        require(bool(np.isfinite(audio).all()), f"Nonfinite WAV: {path}")
        require(hashlib.sha256(audio.T.copy().tobytes()).hexdigest() == sample["waveform_sha256"],
                f"Waveform checksum mismatch: {path}")
        peak = float(np.max(np.abs(audio)))
        require((peak < 1e-8) == sample["silent"], f"Incorrect silence flag: {path}")
        require((peak >= 1) == sample["over_full_scale"], f"Incorrect full-scale flag: {path}")
    for name, checksum in manifest["files_sha256"].items():
        require(digest(directory / name) == checksum, f"Feature checksum mismatch: {directory / name}")
    repeat_paths = [directory / f"features_r{repeat}.npz" for repeat in range(REPEATS)]
    for path in repeat_paths:
        with np.load(path, allow_pickle=False) as data:
            for name, expected in coordinates().items():
                np.testing.assert_allclose(data[name], expected)
    with (
        np.load(directory / "feature_mean.npz", allow_pickle=False) as saved_mean,
        np.load(directory / "feature_variance.npz", allow_pickle=False) as saved_var,
    ):
        for key, shape in SHAPES.items():
            values = []
            for path in repeat_paths:
                with np.load(path, allow_pickle=False) as data:
                    value = data[key]
                require(value.shape == shape and value.dtype == np.float32,
                        f"Incorrect tensor shape/dtype: {directory}, {key}")
                require(bool(np.isfinite(value).all()), f"Nonfinite features: {directory}, {key}")
                if key.endswith("_power") or key == "rms":
                    require(bool((value >= 0).all()), f"Negative power/RMS: {directory}, {key}")
                values.append(value)
            mean, variance = repeat_statistics(np.stack(values))
            np.testing.assert_allclose(saved_mean[key], mean, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(saved_var[key], variance, rtol=1e-6, atol=1e-8)
            require(bool((saved_var[key] >= 0).all()), f"Negative sample variance: {directory}, {key}")
            del values, mean, variance, value
    repeated = []
    for path in repeat_paths:
        with np.load(path, allow_pickle=False) as data:
            repeated.append({
                key: data[key] for key in ("stft_power", "stft_short_power", "log_mel_db", "rms")
            })
    computed = repeat_distances(repeated)
    with np.load(directory / "repeatability.npz", allow_pickle=False) as saved:
        for key, value in computed.items():
            np.testing.assert_allclose(saved[key], value, rtol=1e-6, atol=1e-7)
    # State files are independently checked against the engine-readback components.
    parse_patch(expected_component)
    require((directory / "preview.png").is_file(), f"Missing preview: {directory}")
    return manifest["summary"]


def validate_batch(directory: Path) -> dict:
    batch = read_json(directory / "batch.json")
    require(batch["status"] == "complete", "Batch is not complete")
    require(batch["recordings"] == 2160, "Incorrect total recording count")
    snapshot = directory / "pipeline"
    for name, checksum in batch["pipeline_sha256"].items():
        require(digest(snapshot / name) == checksum, "Frozen rendering pipeline checksum mismatch")
    require(batch["plugin_sha256"] == digest(DEFAULT_PLUGIN), "Plugin checksum mismatch")
    expected_ids = {slug for slug, _, _ in PATCHES}
    require({p["id"] for p in batch["patches"]} == expected_ids and len(batch["patches"]) == 10,
            "Expected ten unique patches")
    rows = []
    for entry in batch["patches"]:
        patch_dir = directory / entry["id"]
        require(digest(patch_dir / "source.fxp") == entry["fxp_sha256"], "Factory source checksum mismatch")
        require(digest(FACTORY / entry["factory_path"]) == entry["fxp_sha256"], "Installed factory source changed")
        require(digest(patch_dir / "original.component") == entry["original_component_sha256"],
                "Original component checksum mismatch")
        require(digest(patch_dir / "sanitized.component") == entry["sanitized_component_sha256"],
                "Sanitized component checksum mismatch")
        sanitized = (patch_dir / "sanitized.component").read_bytes()
        _, verification = sanitize_patch(sanitized)
        require(not verification["edits"] and not verification["removed_routes"],
                "Sanitized controls/routes are not as requested")
        report = read_json(patch_dir / "sanitizer.json")
        require(report["engine_readback_verified"], "Sanitizer lacks engine readback verification")
        for variant in ("original", "sanitized"):
            result = validate_variant(
                patch_dir / variant, (patch_dir / f"{variant}.component").read_bytes(),
            )
            require(result == entry["variants"][variant], "Batch and variant summaries differ")
            rows.append({"patch": entry["id"], "variant": variant, **result})
        print(f"Verified {entry['id']}: 216 recordings and both variant statistics", flush=True)
    report = {
        "status": "verified", "patches": 10, "variants": 20,
        "recordings": 2160, "audio_seconds": 4320, "rows": rows,
        "limitation": "Three repeats estimate within-patch variation; they do not prove determinism.",
    }
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate_batch(args.directory)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
