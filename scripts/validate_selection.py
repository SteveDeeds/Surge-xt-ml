"""Verify the canonical repeatability-only index; run as python -m scripts.validate_selection."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.validate_batch import SHAPES, digest, read_json, require
from surge_batch import coordinates
from surge_dataset import PITCHES, VELOCITIES
from surge_repeatability import LOG_MEL_DB_LIMIT, MAGNITUDE_LIMIT, evaluate_metrics
from surge_dataset import write_json


def validate_selection(directory: Path) -> dict:
    selection = read_json(directory / "selection.json")
    require(selection["status"] == "complete", "Selection is incomplete")
    require(selection["limits"] == {
        "per_resolution_magnitude": MAGNITUDE_LIMIT, "active_log_mel_rms_db": LOG_MEL_DB_LIMIT,
    }, "Eligibility limits differ from the approved limits")
    require(bool(selection["accepted"]), "No eligible patches")
    require(selection["training_axes"] == ["patch", "pitch", "velocity", "feature", "time"],
            "Training index contains an unexpected axis")
    source = Path(selection["source_batch"])
    require(digest(source / "batch.json") == selection["source_batch_sha256"], "Source batch changed")
    require(digest(source / "validation.json") == selection["source_validation_sha256"],
            "Source validation changed")
    for name, checksum in selection["gate_pipeline_sha256"].items():
        require(digest(directory / "pipeline" / name) == checksum, "Gate pipeline snapshot changed")
    accepted_ids = {patch["id"] for patch in selection["accepted"]}
    excluded_ids = {patch["id"] for patch in selection["excluded"]}
    require(len(accepted_ids) == len(selection["accepted"]), "Duplicate accepted patches")
    require(not accepted_ids.intersection(excluded_ids), "Accepted and excluded sets overlap")
    require(len(accepted_ids | excluded_ids) == 10, "Selection does not account for all ten patches")
    screened = {record["id"]: record for record in selection["screen"]}
    confirmed = {record["id"]: record for record in selection["fresh_process_confirmation"]}
    runs = [read_json(directory / f"fresh_process_{run}" / "run.json") for run in (0, 1)]
    for run in runs:
        require(run["status"] == "complete", "Incomplete fresh-process verification")
    worst_magnitude, worst_log_mel = 0.0, 0.0
    for patch in selection["accepted"]:
        require(screened[patch["id"]]["accepted"] and confirmed[patch["id"]]["accepted"],
                "Accepted patch did not pass both stages")
        for stage in (screened[patch["id"]], confirmed[patch["id"]]):
            require(not stage["failed_comparisons"] and not stage["inactive_grid_cells"],
                    "Accepted patch has a failed comparison or inactive cell")
        with np.load(directory / f"{patch['id']}_confirmation_metrics.npz", allow_pickle=False) as data:
            metrics = {key: data[key] for key in data.files}
        require(evaluate_metrics(metrics)["accepted"], "Fresh-process metrics fail current approved limits")
        for key in ("pairwise_stft_power_magnitude_distance",
                    "pairwise_stft_short_power_magnitude_distance"):
            require(metrics[key].shape == (3, 9, 4), "Incomplete pairwise confirmation grid")
            worst_magnitude = max(worst_magnitude, float(np.max(metrics[key])))
        worst_log_mel = max(worst_log_mel, float(np.max(metrics["pairwise_log_mel_rms_db"])))
        for run in runs:
            records = [r for r in run["patches"] if r["id"] == patch["id"]]
            require(len(records) == 1, "Missing fresh-process evidence")
            record = records[0]
            require(digest(Path(record["path"])) == record["sha256"], "Fresh-process data changed")
            for position in ("before", "after"):
                require(record[f"isolation_{position}"]["peak"] < 1e-7, "Fresh-process tail isolation failed")
        require(patch["canonical_repeat"] == 0 and patch["variant"] == "sanitized",
                "Unexpected canonical variant or repeat")
        require(digest(Path(patch["features"])) == patch["features_sha256"], "Canonical features changed")
        require(digest(Path(patch["state"])) == patch["state_sha256"], "Canonical state changed")
        with np.load(patch["features"], allow_pickle=False) as data:
            for key, shape in SHAPES.items():
                require(data[key].shape == shape, f"Unexpected canonical shape for {key}")
                require(bool(np.isfinite(data[key]).all()), f"Nonfinite canonical {key}")
            for key, value in coordinates().items():
                np.testing.assert_array_equal(data[key], value)
        cells = {(sample["midi_note"], sample["velocity"]) for sample in patch["recordings"]}
        require(len(patch["recordings"]) == 36 and cells == {
            (note, velocity) for note in PITCHES for velocity in VELOCITIES
        }, "Canonical grid coverage is incomplete")
        for sample in patch["recordings"]:
            path = Path(patch["variant_directory"]) / sample["path"]
            require(path.name.startswith("r0_"), "Noncanonical repeat in training index")
            require(digest(path) == sample["sha256"], "Canonical WAV checksum mismatch")
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            require(rate == 48000 and audio.shape == (96000, 2), "Canonical WAV format mismatch")
            require(bool(np.isfinite(audio).all()), "Nonfinite canonical WAV")
    report = {
        "status": "verified",
        "accepted": sorted(accepted_ids), "excluded": sorted(excluded_ids),
        "canonical_recordings": 36 * len(accepted_ids),
        "training_repeat_axis": False,
        "worst_fresh_process_per_resolution_magnitude_distance": worst_magnitude,
        "worst_fresh_process_active_log_mel_rms_db": worst_log_mel,
    }
    write_json(directory / "validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(validate_selection(args.directory))
