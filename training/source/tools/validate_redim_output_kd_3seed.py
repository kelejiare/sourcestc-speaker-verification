#!/usr/bin/env python3
"""Validate the fixed three-seed 3D-Speaker ReDim output adapter.

The candidate is fitted on 3D-Speaker train speakers only.  Selection settings
are frozen by the preceding pilot.  This script evaluates the three seeds and
their deterministic arithmetic-mean soup on the frozen coverage protocol, then
scores the untouched HI-MIA dev split and the separately frozen H1 protocol.
It creates evidence and an adapter artifact, never a submission package.
"""

import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_redim_output_kd_support import (
    DURATIONS,
    ENROLL_WEIGHTS,
    MODEL_DIRS,
    MODEL_NAMES,
    SHORT_WEIGHTS,
    concat,
    evaluate,
    l2,
    load_redim,
    load_trials,
    load_acoustic,
    train_adapter,
    transform,
)
from tools.score_lodo_coverage import compute_eer_percent, speaker_bootstrap


OUTPUT_DIR = ROOT / "records/2026-09-04/019_distillation_redim_output_kd_3seed"
OUTPUT = OUTPUT_DIR / "VALIDATION_RESULT.json"
ADAPTERS = OUTPUT_DIR / "distillation_redim_output_kd_soup.npz"
TRACE = OUTPUT_DIR / "EXTERNAL_SCORE_TRACE.jsonl"
SEEDS = (824301, 824302, 824303)
CONFIG = {
    "id": "r16_balanced_frozen_after_pilot",
    "rank": 16,
    "updates": 220,
    "lr": 0.003,
    "margin": 0.15,
    "scale": 24.0,
    "ce": 1.0,
    "proto": 1.0,
    "identity": 2.0,
}
BASE = ROOT / "records/2026-09-04/014_distillation_cn_redimnet2/artifacts"
EXTERNALS = {
    "himia_dev": {
        "cache": BASE / "himia_dual_calibration/cache",
        "trials": BASE / "himia_calibration_trial_freeze/TRIALS.jsonl",
        "split": "dev",
        "expected_cache": 576,
        "expected_trials": 144,
    },
    "h1_frozen": {
        "cache": BASE / "h1_one_shot/cache",
        "trials": BASE / "h1_trial_freeze_v2/H1_TRIALS.jsonl",
        "split": None,
        "expected_cache": 144,
        "expected_trials": 240,
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_external_cache(directory, expected_count):
    output = {}
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(str(path), allow_pickle=False) as source:
            utterance = str(source["utterance_stable_id"])
            if utterance in output:
                raise ValueError("duplicate external cache utterance")
            row = {
                "speaker": str(source["speaker_stable_id"]),
                "distance": str(source["distance_stable_id"]),
            }
            for prefix, width in (("acoustic", 448), ("lm", 192), ("ptn", 192)):
                for suffix, views in (("full", 1), ("0p5", 2), ("1p0", 2)):
                    value = np.asarray(source[prefix + "_" + suffix], dtype=np.float32)
                    expected = (width,) if views == 1 else (views, width)
                    if value.shape != expected or not np.isfinite(value).all():
                        raise ValueError("invalid external cache tensor")
                    row[prefix + "_" + suffix] = l2(value)
            output[utterance] = row
    if len(output) != expected_count:
        raise ValueError("external cache count changed: {} != {}".format(
            len(output), expected_count
        ))
    return output


def load_external_trials(path, split, expected_count):
    with Path(path).open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if split is not None:
        rows = [row for row in rows if row.get("calibration_split") == split]
    if len(rows) != expected_count:
        raise ValueError("external trial count changed: {} != {}".format(
            len(rows), expected_count
        ))
    return rows


def external_trace(dataset, cache, trials, adapters, adapt_mix):
    trace = []
    for raw in trials:
        enrollment_id = str(raw["enrollment"]["utterance_stable_id"])
        test_id = str(raw["test"]["utterance_stable_id"])
        enrollment_row = cache[enrollment_id]
        test_row = cache[test_id]
        enrollment_acoustic = enrollment_row["acoustic_full"]
        enrollment = concat([
            enrollment_acoustic, enrollment_row["lm_full"], enrollment_row["ptn_full"]
        ], ENROLL_WEIGHTS)
        for duration in ("full", "0p5", "1p0"):
            acoustic_values = test_row["acoustic_" + duration]
            if duration == "full":
                acoustic_values = acoustic_values[None, :]
                candidate_values = [concat([
                    acoustic_values[0], test_row["lm_full"], test_row["ptn_full"]
                ], ENROLL_WEIGHTS)]
            else:
                key = "short05" if duration == "0p5" else "short10"
                candidate_values = []
                for view in range(2):
                    lm = test_row["lm_" + duration][view]
                    ptn = test_row["ptn_" + duration][view]
                    if adapt_mix > 0.0:
                        lm = transform(lm, adapters["lm"][key], adapt_mix)
                        ptn = transform(ptn, adapters["ptn"][key], adapt_mix)
                    candidate_values.append(concat([
                        acoustic_values[view], lm, ptn
                    ], SHORT_WEIGHTS))
            baseline_scores = [float(np.dot(enrollment_acoustic, item)) for item in acoustic_values]
            candidate_scores = [float(np.dot(enrollment, item)) for item in candidate_values]
            trace.append({
                "fold": dataset,
                "base_scenario": "distance_" + test_row["distance"][:12],
                "duration": duration,
                "trial_id": str(raw["trial_id"]),
                "label": int(raw["label"]),
                "enrollment_speaker_id": str(raw["enrollment_speaker_id"]),
                "test_speaker_id": str(raw["test_speaker_id"]),
                "baseline_score": float(np.mean(baseline_scores)),
                "candidate_score": float(np.mean(candidate_scores)),
            })
    return trace


def summarize_external(trace):
    grouped = defaultdict(list)
    for row in trace:
        grouped[(row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, rows in sorted(grouped.items()):
        labels = [row["label"] for row in rows]
        baseline = compute_eer_percent(labels, [row["baseline_score"] for row in rows])
        candidate = compute_eer_percent(labels, [row["candidate_score"] for row in rows])
        cells.append({
            "base_scenario": key[0], "duration": key[1], "trial_count": len(rows),
            "baseline_eer_percent": baseline, "candidate_eer_percent": candidate,
            "improvement_points": baseline - candidate,
        })
    short = [cell for cell in cells if cell["duration"] != "full"]
    return {
        "macro_improvement_points": float(np.mean([
            cell["improvement_points"] for cell in cells
        ])),
        "short_macro_improvement_points": float(np.mean([
            cell["improvement_points"] for cell in short
        ])),
        "worst_cell_improvement_points": float(min(
            cell["improvement_points"] for cell in cells
        )),
        "cells": cells,
    }


def main():
    torch.set_num_threads(4)
    acoustic, acoustic_index, metadata = load_acoustic()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in MODEL_NAMES}
    trials = load_trials()
    fit_audits = []
    seed_adapters = {}
    seed_metrics = []
    for seed in SEEDS:
        print("FIT SEED {}".format(seed), flush=True)
        adapters = {name: {} for name in MODEL_NAMES}
        for model_offset, name in enumerate(MODEL_NAMES):
            for duration_offset, duration in enumerate(DURATIONS):
                delta, audit = train_adapter(
                    redims[name], metadata, duration, CONFIG,
                    seed + model_offset * 10 + duration_offset,
                )
                adapters[name][duration] = delta
                audit.update({"seed": seed, "model": name})
                fit_audits.append(audit)
        metrics = evaluate(acoustic, acoustic_index, redims, trials, adapters, 1.0)
        seed_adapters[seed] = adapters
        seed_metrics.append({"seed": seed, "metrics": metrics})
        print(json.dumps({
            "seed": seed, "macro": metrics["macro"],
            "short": metrics["short_macro"], "worst": metrics["worst_cell"],
            "folds": metrics["folds"],
        }, sort_keys=True), flush=True)

    soup = {name: {} for name in MODEL_NAMES}
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "training_scope": np.asarray("3dspeaker_train_only"),
        "seeds": np.asarray(SEEDS, dtype=np.int64),
    }
    for name in MODEL_NAMES:
        for duration in DURATIONS:
            value = np.mean(np.stack([
                seed_adapters[seed][name][duration] for seed in SEEDS
            ]).astype(np.float64), axis=0).astype(np.float32)
            if value.shape != (192, 192) or not np.isfinite(value).all():
                raise ValueError("invalid soup adapter")
            soup[name][duration] = value
            arrays[name + "__" + duration] = value
    atomic_npz(ADAPTERS, arrays)
    coverage = evaluate(acoustic, acoustic_index, redims, trials, soup, 1.0)

    external_results = {}
    all_trace = []
    for offset, (dataset, spec) in enumerate(EXTERNALS.items()):
        print("SCORE {}".format(dataset), flush=True)
        cache = load_external_cache(spec["cache"], spec["expected_cache"])
        external_trials = load_external_trials(
            spec["trials"], spec["split"], spec["expected_trials"]
        )
        required = {
            str(row[key]["utterance_stable_id"])
            for row in external_trials for key in ("enrollment", "test")
        }
        if not required.issubset(cache):
            raise ValueError("external trial endpoint missing from cache")
        unadapted_trace = external_trace(dataset, cache, external_trials, soup, 0.0)
        adapted_trace = external_trace(dataset, cache, external_trials, soup, 1.0)
        metrics = summarize_external(adapted_trace)
        unadapted = summarize_external(unadapted_trace)
        bootstrap = speaker_bootstrap(
            adapted_trace, fold=dataset, repeats=2000,
            seed=824401 + offset, short_only=True,
        )
        external_results[dataset] = {
            "cache_count": len(cache), "trial_count": len(external_trials),
            "cache_path": str(spec["cache"]), "trial_path": str(spec["trials"]),
            "trial_sha256": sha256_file(spec["trials"]),
            "unadapted_dual_vs_acoustic": unadapted,
            "adapted_candidate_vs_acoustic": metrics,
            "short_speaker_bootstrap_vs_acoustic": bootstrap,
        }
        all_trace.extend(adapted_trace)
        print(json.dumps({
            "dataset": dataset,
            "macro": metrics["macro_improvement_points"],
            "short": metrics["short_macro_improvement_points"],
            "worst": metrics["worst_cell_improvement_points"],
            "ci_upper": bootstrap["delta_ci95_upper"],
        }, sort_keys=True), flush=True)
    atomic_jsonl(TRACE, all_trace)

    gate_checks = {
        "coverage_macro_at_least_2p6": coverage["macro"] >= 2.6,
        "coverage_short_at_least_3": coverage["short_macro"] >= 3.0,
        "coverage_worst_at_least_minus_0p2": coverage["worst_cell"] >= -0.2,
        "coverage_all_folds_positive": all(
            value > 0.0 for value in coverage["folds"].values()
        ),
        "each_external_macro_nonnegative": all(
            value["adapted_candidate_vs_acoustic"]["macro_improvement_points"] >= 0.0
            for value in external_results.values()
        ),
        "each_external_short_positive": all(
            value["adapted_candidate_vs_acoustic"]["short_macro_improvement_points"] > 0.0
            for value in external_results.values()
        ),
        "each_external_worst_at_least_minus_0p2": all(
            value["adapted_candidate_vs_acoustic"]["worst_cell_improvement_points"] >= -0.2
            for value in external_results.values()
        ),
        "each_external_bootstrap_upper_below_zero": all(
            value["short_speaker_bootstrap_vs_acoustic"]["delta_ci95_upper"] < 0.0
            for value in external_results.values()
        ),
    }
    status = (
        "THREE_SEED_EXTERNAL_GATE_PASS_BUILD_ALLOWED"
        if all(gate_checks.values())
        else "THREE_SEED_EXTERNAL_GATE_FAILED_DO_NOT_PACKAGE"
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "distillation_redim_output_kd_three_seed_external_validation",
        "status": status, "package_created": False, "submission_created": False,
        "training_scope": "3dspeaker_train_only",
        "config": CONFIG, "seeds": list(SEEDS), "fit_audits": fit_audits,
        "seed_coverage_metrics": seed_metrics,
        "soup_coverage_metrics": coverage,
        "external_results": external_results,
        "gate_checks": gate_checks,
        "adapter_artifact": {
            "path": str(ADAPTERS), "bytes": ADAPTERS.stat().st_size,
            "sha256": sha256_file(ADAPTERS),
        },
        "external_trace": {
            "path": str(TRACE), "rows": len(all_trace), "sha256": sha256_file(TRACE)
        },
        "limitations": [
            "This is proxy evaluation and cannot guarantee a hidden score above 91.",
            "H1 has been inspected by prior model-selection work and is confirmatory only.",
            "The separately executed all-source LODO gate remains a distinct required audit.",
        ],
    }
    atomic_json(OUTPUT, report)
    print(json.dumps({
        "status": status, "gate_checks": gate_checks,
        "coverage": {"macro": coverage["macro"], "short": coverage["short_macro"],
                     "worst": coverage["worst_cell"], "folds": coverage["folds"]},
        "external": external_results,
        "adapter": report["adapter_artifact"], "report": str(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
