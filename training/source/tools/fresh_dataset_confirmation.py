#!/usr/bin/env python3
"""One-shot confirmation of the frozen distillation fusion on three unused corpora.

The fusion point is read from the already-written frontier report.  Trials are
deterministically frozen before scoring.  SeniorTalk, Common Voice Beijing and
MagicData were not used to fit the selected multidomain adapters or select the
frontier point.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_redim_output_kd_support import MODEL_DIRS, MODEL_NAMES, l2, load_redim
from tools.search_multidomain_fusion_frontier import (
    adapted,
    branch_weights,
    candidate_score,
    load_adapters,
)
from tools.score_lodo_coverage import compute_eer_percent, speaker_bootstrap
from tools.validate_redim_output_kd_3seed import atomic_json, atomic_jsonl, sha256_file


OUTPUT_DIR = ROOT / "records/2026-09-04/022_fresh_dataset_confirmation"
FRONTIER = ROOT / (
    "records/2026-09-04/021_distillation_multidomain_fusion_frontier/FRONTIER_RESULT.json"
)
ACOUSTIC_CACHE = ROOT / (
    "records/2026-09-02/013_distillation_domain_kd/artifacts/"
    "combined_cache_post_cv/cache_v2.npz"
)
TRIALS_PATH = OUTPUT_DIR / "FROZEN_TRIALS.jsonl"
TRACE_PATH = OUTPUT_DIR / "SCORE_TRACE.jsonl"
RECEIPT = OUTPUT_DIR / "TRIAL_FREEZE_RECEIPT.json"
OUTPUT = OUTPUT_DIR / "CONFIRMATION_RESULT.json"
DATASETS = ("seniortalk", "commonvoice_beijing", "magicdata_slr68")
EXPECTED_CONFIG = {
    "adapter_mix": 1.0,
    "aggregation": "mean_delta",
    "enroll_acoustic": 0.86,
    "ptn_ratio": 0.05,
    "short_acoustic": 0.2,
}


def stable_hash(*parts):
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def load_acoustic_post():
    with np.load(str(ACOUSTIC_CACHE), allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    index = defaultdict(list)
    metadata = {}
    for position, utterance in enumerate(arrays["utterance_id"]):
        utterance = str(utterance)
        index[utterance].append(position)
        row = {
            "dataset": str(arrays["dataset"][position]),
            "speaker": str(arrays["speaker_id"][position]),
            "split": str(arrays["split"][position]),
        }
        if utterance in metadata and metadata[utterance] != row:
            raise ValueError("inconsistent acoustic metadata")
        metadata[utterance] = row
    for utterance, positions in index.items():
        positions.sort(key=lambda item: str(arrays["embedding_id"][item]))
        if len(positions) != 2:
            raise ValueError("fresh confirmation expects two fixed views")
    return arrays, dict(index), metadata


def freeze_trials(metadata, available):
    trials = []
    audits = {}
    for dataset in DATASETS:
        speakers = defaultdict(list)
        for utterance, row in metadata.items():
            if row["dataset"] == dataset and utterance in available:
                speakers[row["speaker"]].append(utterance)
        speakers = {
            speaker: sorted(utterances, key=lambda item: stable_hash(dataset, item))
            for speaker, utterances in speakers.items() if len(utterances) >= 2
        }
        ordered = sorted(speakers, key=lambda item: stable_hash(dataset, item))
        if len(ordered) < 2:
            raise ValueError("fresh dataset needs at least two eligible speakers")
        positive_count = 0
        negative_count = 0
        for speaker_index, speaker in enumerate(ordered):
            enrollment = speakers[speaker][0]
            positives = speakers[speaker][1:6]
            for positive_index, test in enumerate(positives):
                raw = {
                    "dataset": dataset, "base_scenario": "fresh_" + dataset,
                    "enrollment_utterance_id": enrollment,
                    "test_utterance_id": test,
                    "enrollment_speaker_id": speaker,
                    "test_speaker_id": speaker, "label": 1,
                }
                raw["trial_id"] = stable_hash(
                    dataset, enrollment, test, "1", str(positive_index)
                )
                trials.append(raw)
                positive_count += 1
                negative_speaker = ordered[
                    (speaker_index + 1 + positive_index) % len(ordered)
                ]
                if negative_speaker == speaker:
                    negative_speaker = ordered[(speaker_index + 1) % len(ordered)]
                negative_utterances = speakers[negative_speaker]
                negative_test = negative_utterances[
                    positive_index % len(negative_utterances)
                ]
                raw = {
                    "dataset": dataset, "base_scenario": "fresh_" + dataset,
                    "enrollment_utterance_id": enrollment,
                    "test_utterance_id": negative_test,
                    "enrollment_speaker_id": speaker,
                    "test_speaker_id": negative_speaker, "label": 0,
                }
                raw["trial_id"] = stable_hash(
                    dataset, enrollment, negative_test, "0", str(positive_index)
                )
                trials.append(raw)
                negative_count += 1
        audits[dataset] = {
            "eligible_speaker_count": len(ordered),
            "utterance_count": sum(len(item) for item in speakers.values()),
            "positive_trials": positive_count, "negative_trials": negative_count,
        }
    if len({row["trial_id"] for row in trials}) != len(trials):
        raise ValueError("duplicate frozen trial id")
    return sorted(trials, key=lambda row: row["trial_id"]), audits


def score(acoustic, acoustic_index, redims, adapters, config, trials):
    enroll_weights = branch_weights(config["enroll_acoustic"], config["ptn_ratio"])
    short_weights = branch_weights(config["short_acoustic"], config["ptn_ratio"])
    trace = []
    for raw in trials:
        enrollment_id = raw["enrollment_utterance_id"]
        test_id = raw["test_utterance_id"]
        enrollment_index = acoustic_index[enrollment_id][0]
        enrollment = [
            l2(acoustic["full"][enrollment_index]),
            redims["lm"][enrollment_id]["full"],
            redims["ptn"][enrollment_id]["full"],
        ]
        for duration, acoustic_key, redim_key in (
            ("full", "full", "full"),
            ("0p5", "short05", "short05"),
            ("1p0", "short10", "short10"),
        ):
            indices = [acoustic_index[test_id][0]] if duration == "full" else acoustic_index[test_id]
            baseline_scores = []
            candidate_scores = []
            for view, position in enumerate(indices):
                test_acoustic = l2(acoustic[acoustic_key][position])
                baseline_scores.append(float(np.dot(enrollment[0], test_acoustic)))
                if duration == "full":
                    test = [
                        test_acoustic, redims["lm"][test_id]["full"],
                        redims["ptn"][test_id]["full"],
                    ]
                    weights = enroll_weights
                else:
                    key = "short05" if duration == "0p5" else "short10"
                    test = [
                        test_acoustic,
                        adapted(redims["lm"][test_id][redim_key][view], adapters,
                                config["aggregation"], "lm", key,
                                config["adapter_mix"]),
                        adapted(redims["ptn"][test_id][redim_key][view], adapters,
                                config["aggregation"], "ptn", key,
                                config["adapter_mix"]),
                    ]
                    weights = short_weights
                candidate_scores.append(candidate_score(
                    enrollment, test, enroll_weights, weights
                ))
            trace.append({
                "fold": raw["dataset"], "base_scenario": raw["base_scenario"],
                "duration": duration, "trial_id": raw["trial_id"],
                "label": raw["label"],
                "enrollment_speaker_id": raw["enrollment_speaker_id"],
                "test_speaker_id": raw["test_speaker_id"],
                "baseline_score": float(np.mean(baseline_scores)),
                "candidate_score": float(np.mean(candidate_scores)),
            })
    return trace


def summarize(trace):
    grouped = defaultdict(list)
    for row in trace:
        grouped[(row["fold"], row["duration"])].append(row)
    cells = []
    for key, rows in sorted(grouped.items()):
        labels = [row["label"] for row in rows]
        baseline = compute_eer_percent(labels, [row["baseline_score"] for row in rows])
        candidate = compute_eer_percent(labels, [row["candidate_score"] for row in rows])
        cells.append({
            "dataset": key[0], "duration": key[1], "trial_count": len(rows),
            "baseline_eer_percent": baseline, "candidate_eer_percent": candidate,
            "improvement_points": baseline - candidate,
        })
    folds = {}
    for dataset in DATASETS:
        selected = [row for row in cells if row["dataset"] == dataset]
        short = [row for row in selected if row["duration"] != "full"]
        folds[dataset] = {
            "macro_improvement_points": float(np.mean([
                row["improvement_points"] for row in selected
            ])),
            "short_macro_improvement_points": float(np.mean([
                row["improvement_points"] for row in short
            ])),
            "worst_cell_improvement_points": float(min(
                row["improvement_points"] for row in selected
            )),
        }
    return {
        "macro_improvement_points": float(np.mean([
            row["improvement_points"] for row in cells
        ])),
        "short_macro_improvement_points": float(np.mean([
            row["improvement_points"] for row in cells if row["duration"] != "full"
        ])),
        "worst_cell_improvement_points": float(min(
            row["improvement_points"] for row in cells
        )),
        "folds": folds, "cells": cells,
    }


def main():
    with FRONTIER.open("r", encoding="utf-8") as stream:
        frontier = json.load(stream)
    config = frontier["best"]["config"]
    if config != EXPECTED_CONFIG or not frontier["best"]["point_gate_pass"]:
        raise ValueError("frontier candidate changed after freeze")
    acoustic, acoustic_index, metadata = load_acoustic_post()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in MODEL_NAMES}
    adapters = load_adapters()
    available = set(redims["lm"]) & set(redims["ptn"])
    trials, trial_audit = freeze_trials(metadata, available)
    required = {
        row[key] for row in trials
        for key in ("enrollment_utterance_id", "test_utterance_id")
    }
    if not required.issubset(redims["lm"]) or not required.issubset(redims["ptn"]):
        raise ValueError("fresh trial endpoint missing ReDim cache")
    atomic_jsonl(TRIALS_PATH, trials)
    atomic_json(RECEIPT, {
        "schema_version": 1, "artifact_kind": "distillation_fresh_trial_freeze",
        "candidate_config": config, "frontier_sha256": sha256_file(FRONTIER),
        "acoustic_cache_sha256": sha256_file(ACOUSTIC_CACHE), "trial_audit": trial_audit,
        "trials": {"path": str(TRIALS_PATH), "rows": len(trials),
                   "sha256": sha256_file(TRIALS_PATH)},
        "frozen_before_scoring": True,
    })
    trace = score(acoustic, acoustic_index, redims, adapters, config, trials)
    atomic_jsonl(TRACE_PATH, trace)
    metrics = summarize(trace)
    bootstrap = {
        dataset: speaker_bootstrap(
            trace, fold=dataset, repeats=2000,
            seed=824701 + DATASETS.index(dataset), short_only=True,
        )
        for dataset in DATASETS
    }
    gates = {
        "combined_macro_positive": metrics["macro_improvement_points"] > 0.0,
        "combined_short_positive": metrics["short_macro_improvement_points"] > 0.0,
        "worst_cell_at_least_minus_0p2": metrics["worst_cell_improvement_points"] >= -0.2,
        "each_dataset_macro_positive": all(
            row["macro_improvement_points"] > 0.0 for row in metrics["folds"].values()
        ),
        "each_dataset_short_positive": all(
            row["short_macro_improvement_points"] > 0.0
            for row in metrics["folds"].values()
        ),
        "each_dataset_bootstrap_upper_below_zero": all(
            row["delta_ci95_upper"] < 0.0 for row in bootstrap.values()
        ),
    }
    status = (
        "FRESH_CONFIRMATION_ALL_GATES_PASS_BUILD_ALLOWED"
        if all(gates.values())
        else "FRESH_CONFIRMATION_FAILED_DO_NOT_PACKAGE"
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "distillation_fresh_three_dataset_confirmation",
        "status": status, "package_created": False, "submission_created": False,
        "candidate_config": config, "trial_audit": trial_audit,
        "metrics": metrics, "speaker_bootstrap": bootstrap, "gate_checks": gates,
        "trials": {"path": str(TRIALS_PATH), "rows": len(trials),
                   "sha256": sha256_file(TRIALS_PATH)},
        "trace": {"path": str(TRACE_PATH), "rows": len(trace),
                  "sha256": sha256_file(TRACE_PATH)},
        "limitations": [
            "These are public proxy corpora and cannot guarantee a hidden score above 91.",
            "The corpora are dataset-disjoint from adapter fitting and frontier selection.",
        ],
    }
    atomic_json(OUTPUT, report)
    print(json.dumps({
        "status": status, "metrics": metrics, "bootstrap": bootstrap,
        "gates": gates, "report": str(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
