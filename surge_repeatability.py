"""Admit only sanitized patches repeatable within numerical spectral tolerances."""

import argparse
import gc
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from pedalboard import VST3Plugin

from surge_batch import (
    REPEATS, batch_features, check_isolation, coordinates, repeat_distances,
)
from surge_dataset import (
    DEFAULT_PLUGIN, PITCHES, ROOT, VELOCITIES, render_note, sha256, write_json,
)

MAGNITUDE_LIMIT = 0.01
LOG_MEL_DB_LIMIT = 0.1
FEATURE_KEYS = ("stft_power", "stft_short_power", "log_mel_db", "rms")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_features(path: Path, checksum: str | None = None) -> dict[str, np.ndarray]:
    if checksum is not None and sha256(path.read_bytes()) != checksum:
        raise ValueError(f"Feature checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in ("pitches_midi", "velocities", "time_seconds"):
            np.testing.assert_array_equal(data[key], coordinates()[key])
        return {key: data[key] for key in FEATURE_KEYS}


def measure_repeats(repeats: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    for repeated in repeats:
        if set(repeated) != set(FEATURE_KEYS):
            raise ValueError("Missing or unexpected repeatability features")
        for key, value in repeated.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {key}")
            if key != "log_mel_db" and np.any(value < 0):
                raise ValueError(f"Negative power/RMS in {key}")
    metrics = repeat_distances(repeats)
    for key in ("stft_power", "stft_short_power"):
        distances = []
        for left, right in itertools.combinations(repeats, 2):
            if left[key].shape != right[key].shape:
                raise ValueError(f"Repeat shape mismatch: {key}")
            a, b = np.sqrt(left[key]), np.sqrt(right[key])
            difference = np.sqrt(np.sum((a - b) ** 2, axis=(-2, -1), dtype=np.float64))
            scale = (
                np.sqrt(np.sum(a ** 2, axis=(-2, -1), dtype=np.float64))
                + np.sqrt(np.sum(b ** 2, axis=(-2, -1), dtype=np.float64))
            ) / 2
            distances.append(difference / np.maximum(scale, 1e-12))
        metrics[f"pairwise_{key}_magnitude_distance"] = np.asarray(distances)
    return metrics


def evaluate_metrics(
    metrics: dict[str, np.ndarray], *,
    magnitude_limit: float = MAGNITUDE_LIMIT, log_mel_db_limit: float = LOG_MEL_DB_LIMIT,
) -> dict:
    if (not np.isfinite([magnitude_limit, log_mel_db_limit]).all()
            or magnitude_limit <= 0 or log_mel_db_limit <= 0):
        raise ValueError("Repeatability limits must be positive finite values")
    keys = (
        "pairwise_stft_power_magnitude_distance",
        "pairwise_stft_short_power_magnitude_distance",
        "pairwise_log_mel_rms_db",
    )
    for key in (*keys, "active_mel_bins"):
        if not np.isfinite(metrics[key]).all():
            raise ValueError(f"Nonfinite repeatability metric: {key}")
    failures = []
    for key, limit in zip(keys, (magnitude_limit, magnitude_limit, log_mel_db_limit), strict=True):
        for pair, pitch, velocity in np.argwhere(metrics[key] > limit):
            failures.append({
                "metric": key, "repeat_pair_index": int(pair),
                "pitch_index": int(pitch), "velocity_index": int(velocity),
                "value": float(metrics[key][pair, pitch, velocity]), "limit": limit,
            })
    inactive = np.argwhere(metrics["active_mel_bins"] <= 0)
    return {
        "accepted": not failures and len(inactive) == 0,
        "limits": {"per_resolution_magnitude": magnitude_limit, "active_log_mel_rms_db": log_mel_db_limit},
        "maxima": {key: float(np.max(metrics[key])) for key in keys},
        "failed_comparisons": failures,
        "inactive_grid_cells": inactive.tolist(),
        "rule": "All repeat pairs and every pitch/velocity cell must pass both resolutions and log-mel.",
        "inactive_policy": "No active evidence is not a pass; exclude the patch from this full-grid dataset.",
    }


def worker(plan_path: Path, run: int) -> None:
    plan = read_json(plan_path)
    output = plan_path.parent / f"fresh_process_{run}"
    output.mkdir()
    plugin = VST3Plugin(str(DEFAULT_PLUGIN))
    result = {"status": "in_progress", "pid": os.getpid(), "patches": []}
    for candidate in plan["candidates"]:
        directory = Path(candidate["directory"])
        state = (directory / "patch.vstpreset").read_bytes()
        if sha256(state) != candidate["state_sha256"]:
            raise ValueError(f"State changed since screening: {directory}")
        isolation_before = check_isolation(plugin, state)
        tensors: dict[str, list[np.ndarray]] = {key: [] for key in FEATURE_KEYS}
        for note in PITCHES:
            for velocity in VELOCITIES:
                audio = render_note(plugin, state, note, velocity, require_signal=False)
                features = batch_features(audio)
                for key in FEATURE_KEYS:
                    tensors[key].append(features[key])
        arrays = {
            key: np.stack(values).reshape(9, 4, *values[0].shape)
            for key, values in tensors.items()
        }
        isolation_after = check_isolation(plugin, state)
        feature_path = output / f"{candidate['id']}.npz"
        # These are verification artifacts, not another training-tensor dimension.
        np.savez(feature_path, **arrays, **coordinates())
        result["patches"].append({
            "id": candidate["id"], "path": str(feature_path.resolve()),
            "sha256": sha256(feature_path.read_bytes()),
            "isolation_before": isolation_before, "isolation_after": isolation_after,
        })
        write_json(output / "run.json", result)
        del tensors, arrays, features, audio
        gc.collect()
        print(f"Fresh process {run}: {candidate['id']}", flush=True)
    result["status"] = "complete"
    write_json(output / "run.json", result)


def select_dataset(batch_directory: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite repeatability selection: {output}")
    batch = read_json(batch_directory / "batch.json")
    validation = read_json(batch_directory / "validation.json")
    if batch["status"] != "complete" or validation["status"] != "verified":
        raise ValueError("A complete, verified source batch is required")
    if len(batch["patches"]) != 10 or validation["recordings"] != 2160:
        raise ValueError("Unexpected source batch scope")
    if batch["plugin_sha256"] != sha256(DEFAULT_PLUGIN.read_bytes()):
        raise ValueError("Plugin differs from the verified source batch")
    for patch in batch["patches"]:
        for variant in ("original", "sanitized"):
            matches = [
                row for row in validation["rows"]
                if row["patch"] == patch["id"] and row["variant"] == variant
            ]
            if (len(matches) != 1
                    or {k: v for k, v in matches[0].items() if k not in ("patch", "variant")}
                    != patch["variants"][variant]):
                raise ValueError("Source validation does not match the batch summaries")
    output.mkdir(parents=True)
    snapshot = output / "pipeline"
    snapshot.mkdir()
    gate_sources = {}
    for name in ("surge_repeatability.py", "surge_batch.py", "surge_dataset.py", "surge_patches.py"):
        source = (ROOT / name).read_bytes()
        (snapshot / name).write_bytes(source)
        gate_sources[name] = sha256(source)
    report = {
        "status": "in_progress", "source_batch": str(batch_directory.resolve()),
        "source_batch_sha256": sha256((batch_directory / "batch.json").read_bytes()),
        "source_validation_sha256": sha256((batch_directory / "validation.json").read_bytes()),
        "gate_code_sha256": sha256(Path(__file__).read_bytes()),
        "gate_pipeline_sha256": gate_sources,
        "plugin_sha256": batch["plugin_sha256"],
        "packages": batch["packages"],
        "limits": {"per_resolution_magnitude": MAGNITUDE_LIMIT, "active_log_mel_rms_db": LOG_MEL_DB_LIMIT},
        "policy": "Sanitized only; screen all existing repeats, then two fresh processes against canonical repeat 0.",
        "screen": [], "fresh_process_confirmation": [], "accepted": [], "excluded": [],
        "scope": "Empirical repeatability for this engine, settings and grid; not universal determinism.",
    }
    candidates = []
    for patch in batch["patches"]:
        directory = batch_directory / patch["id"] / "sanitized"
        manifest = read_json(directory / "manifest.json")
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete source variant: {directory}")
        repeated = [
            load_features(directory / f"features_r{repeat}.npz",
                          manifest["files_sha256"][f"features_r{repeat}.npz"])
            for repeat in range(REPEATS)
        ]
        metrics = measure_repeats(repeated)
        decision = evaluate_metrics(metrics)
        report["screen"].append({"id": patch["id"], **decision})
        if decision["accepted"]:
            candidates.append({
                "id": patch["id"], "directory": str(directory.resolve()),
                "state_sha256": manifest["state_sha256"],
                "reference_features_sha256": manifest["files_sha256"]["features_r0.npz"],
            })
        else:
            report["excluded"].append({"id": patch["id"], "stage": "same_process_screen"})
        write_json(output / "selection.json", report)
        del repeated, metrics
        gc.collect()
        print(f"Screen {patch['id']}: {'pass' if decision['accepted'] else 'exclude'}", flush=True)
    if not candidates:
        raise ValueError(f"No sanitized patches passed the screening limits; see {output / 'selection.json'}")
    plan_path = output / "confirmation_plan.json"
    write_json(plan_path, {"candidates": candidates})
    env = dict(os.environ)
    env.update({name: "1" for name in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
    runs = []
    for run in range(2):
        log_path = output / f"fresh_process_{run}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                 str(plan_path.resolve()), "--run", str(run)],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Fresh-process verification failed; see {log_path}")
        result = read_json(output / f"fresh_process_{run}" / "run.json")
        if result["status"] != "complete":
            raise ValueError(f"Incomplete fresh-process run {run}")
        runs.append({patch["id"]: patch for patch in result["patches"]})
    for candidate in candidates:
        reference = Path(candidate["directory"]) / "features_r0.npz"
        repeated = [load_features(reference, candidate["reference_features_sha256"])]
        for run in runs:
            record = run[candidate["id"]]
            repeated.append(load_features(Path(record["path"]), record["sha256"]))
        metrics = measure_repeats(repeated)
        decision = evaluate_metrics(metrics)
        np.savez_compressed(output / f"{candidate['id']}_confirmation_metrics.npz", **metrics)
        report["fresh_process_confirmation"].append({"id": candidate["id"], **decision})
        if decision["accepted"]:
            manifest = read_json(Path(candidate["directory"]) / "manifest.json")
            recordings = [s for s in manifest["samples"] if s["repeat"] == 0]
            if len(recordings) != 36:
                raise ValueError("Canonical render does not cover the full grid")
            report["accepted"].append({
                "id": candidate["id"], "variant": "sanitized", "canonical_repeat": 0,
                "features": str(reference.resolve()),
                "features_sha256": candidate["reference_features_sha256"],
                "state": str((Path(candidate["directory"]) / "patch.vstpreset").resolve()),
                "state_sha256": candidate["state_sha256"],
                "audio_directory": str((Path(candidate["directory"]) / "audio").resolve()),
                "variant_directory": str(Path(candidate["directory"]).resolve()),
                "recordings": [{"midi_note": s["midi_note"], "velocity": s["velocity"],
                                "path": s["path"], "sha256": s["sha256"]} for s in recordings],
            })
        else:
            report["excluded"].append({"id": candidate["id"], "stage": "fresh_process_confirmation"})
        write_json(output / "selection.json", report)
        del repeated, metrics
        gc.collect()
    if not report["accepted"]:
        raise ValueError(f"No patches passed fresh-process confirmation; see {output / 'selection.json'}")
    report["training_axes"] = ["patch", "pitch", "velocity", "feature", "time"]
    report["diagnostic_repeats_are_not_training_examples"] = True
    report["status"] = "complete"
    write_json(output / "selection.json", report)
    print(json.dumps({
        "accepted": [p["id"] for p in report["accepted"]],
        "excluded": report["excluded"], "selection": str(output / "selection.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=ROOT / "data" / "factory_ten")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "repeatable_ten")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        if args.run is None:
            parser.error("--worker requires --run")
        worker(args.worker, args.run)
    else:
        select_dataset(args.batch, args.output)
