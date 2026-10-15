#!/usr/bin/env python3
"""Cross-domain reweighting study around the released distillation embedding blocks.

No acoustic model or adapter is changed.  Candidate selection uses coverage
LODO plus AliMeeting Eval/ChildMandarin Dev; AliMeeting Test is opened only
after the configuration has been frozen.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_redim_output_kd_support import MODEL_DIRS, l2, load_redim, load_trials, load_acoustic
from tools.score_lodo_coverage import compute_eer_percent, speaker_bootstrap
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.nap_lodo import adapt, atomic_json, atomic_jsonl, load_distillation_adapters, sha256_file


OUTPUT_DIR = ROOT / "records/2026-09-05/039_distillation_reweight_validation"
RESULT_PATH = OUTPUT_DIR / "VALIDATION_RESULT.json"
DEVELOPMENT_TRACE = OUTPUT_DIR / "DEVELOPMENT_TRACE.jsonl"
FINAL_TRACE = OUTPUT_DIR / "FINAL_CONFIRMATION_TRACE.jsonl"
CURRENT_ENROLL = np.asarray([0.86, 0.133, 0.007], dtype=np.float64)
CURRENT_SHORT = np.asarray([0.20, 0.76, 0.04], dtype=np.float64)


def weight_vector(acoustic_weight, ptn_ratio):
    acoustic_weight = float(acoustic_weight)
    ptn_ratio = float(ptn_ratio)
    rest = 1.0 - acoustic_weight
    value = np.asarray([
        acoustic_weight, rest * (1.0 - ptn_ratio), rest * ptn_ratio
    ], dtype=np.float64)
    if np.any(value < 0.0) or abs(float(value.sum()) - 1.0) > 1.0e-12:
        raise ValueError("invalid weight vector")
    return value


def score_dots(dots, enroll, test):
    coefficients = np.sqrt(np.asarray(enroll) * np.asarray(test))
    return float(np.dot(coefficients, np.asarray(dots, dtype=np.float64)))


def coverage_components(adapters):
    acoustic, acoustic_index, _ = load_acoustic()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in ("lm", "ptn")}
    rows = []
    trials = load_trials()
    for index, raw in enumerate(trials):
        if index % 1000 == 0:
            print("COVERAGE component {}/{}".format(index, len(trials)), flush=True)
        enrollment_id = str(raw["enrollment_utterance_id"])
        test_id = str(raw["test_utterance_id"])
        enrollment_index = acoustic_index[enrollment_id][0]
        enrollment = [
            l2(acoustic["full"][enrollment_index]), redims["lm"][enrollment_id]["full"],
            redims["ptn"][enrollment_id]["full"],
        ]
        for duration, acoustic_key, redim_key, adapter_key in (
            ("full", "full", "full", None),
            ("0p5", "short05", "short05", "short05"),
            ("1p0", "short10", "short10", "short10"),
        ):
            indices = [acoustic_index[test_id][0]] if duration == "full" else acoustic_index[test_id]
            view_dots = []
            for view, cache_index in enumerate(indices):
                if duration == "full":
                    test = [l2(acoustic[acoustic_key][cache_index]), redims["lm"][test_id]["full"],
                            redims["ptn"][test_id]["full"]]
                else:
                    test = [
                        l2(acoustic[acoustic_key][cache_index]),
                        adapt(redims["lm"][test_id][redim_key][view], adapters["lm"][adapter_key]),
                        adapt(redims["ptn"][test_id][redim_key][view], adapters["ptn"][adapter_key]),
                    ]
                view_dots.append([float(np.dot(a, b)) for a, b in zip(enrollment, test)])
            dots = np.mean(np.asarray(view_dots, dtype=np.float64), axis=0)
            current_test = CURRENT_ENROLL if duration == "full" else CURRENT_SHORT
            rows.append({
                "suite": "coverage_lodo", "fold": str(raw["dataset"]),
                "dataset": str(raw["dataset"]), "base_scenario": str(raw["base_scenario"]),
                "duration": duration, "trial_id": str(raw["trial_id"]),
                "label": int(raw["label"]),
                "enrollment_speaker_id": str(raw["enrollment_speaker_id"]),
                "test_speaker_id": str(raw["test_speaker_id"]),
                "dots": [float(item) for item in dots],
                "distillation_score": score_dots(dots, CURRENT_ENROLL, current_test),
            })
    return rows


def external_components(suite, sources, trials, cache, adapters):
    del sources
    rows = []
    for index, raw in enumerate(trials):
        if index % 1000 == 0:
            print("{} component {}/{}".format(suite["name"], index, len(trials)), flush=True)
        enrollment_cache = cache[raw["enrollment_utterance_id"]]
        test_cache = cache[raw["test_utterance_id"]]
        enrollment = [l2(enrollment_cache[name]["full"]) for name in ("acoustic", "lm", "ptn")]
        for duration, cache_key, adapter_key in (
            ("full", "full", None), ("0p5", "0p5", "short05"),
            ("1p0", "1p0", "short10"),
        ):
            view_dots = []
            for view in range(1 if duration == "full" else 2):
                if duration == "full":
                    test = [l2(test_cache[name]["full"]) for name in ("acoustic", "lm", "ptn")]
                else:
                    test = [
                        l2(test_cache["acoustic"][cache_key][view]),
                        adapt(test_cache["lm"][cache_key][view], adapters["lm"][adapter_key]),
                        adapt(test_cache["ptn"][cache_key][view], adapters["ptn"][adapter_key]),
                    ]
                view_dots.append([float(np.dot(a, b)) for a, b in zip(enrollment, test)])
            dots = np.mean(np.asarray(view_dots, dtype=np.float64), axis=0)
            current_test = CURRENT_ENROLL if duration == "full" else CURRENT_SHORT
            rows.append({
                "suite": suite["name"], "fold": str(raw["dataset"]),
                "dataset": str(raw["dataset"]), "base_scenario": str(raw["base_scenario"]),
                "duration": duration, "trial_id": str(raw["trial_id"]),
                "label": int(raw["label"]),
                "enrollment_speaker_id": str(raw["enrollment_speaker_id"]),
                "test_speaker_id": str(raw["test_speaker_id"]),
                "dots": [float(item) for item in dots],
                "distillation_score": score_dots(dots, CURRENT_ENROLL, current_test),
            })
    return rows


def score_rows(rows, config):
    enroll = weight_vector(config["enroll_acoustic"], config["enroll_ptn_ratio"])
    short05 = weight_vector(config["short05_acoustic"], config["short05_ptn_ratio"])
    short10 = weight_vector(config["short10_acoustic"], config["short10_ptn_ratio"])
    scored = []
    for row in rows:
        test = (enroll if row["duration"] == "full" else
                short05 if row["duration"] == "0p5" else short10)
        scored.append({
            "suite": row["suite"], "fold": row["fold"], "dataset": row["dataset"],
            "base_scenario": row["base_scenario"], "duration": row["duration"],
            "trial_id": row["trial_id"], "label": row["label"],
            "enrollment_speaker_id": row["enrollment_speaker_id"],
            "test_speaker_id": row["test_speaker_id"],
            "baseline_score": row["distillation_score"],
            "candidate_score": score_dots(row["dots"], enroll, test),
        })
    return scored


def summarize(scored, bootstrap=0):
    groups = defaultdict(list)
    for row in scored:
        groups[(row["dataset"], row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, rows in sorted(groups.items()):
        labels = [row["label"] for row in rows]
        baseline = compute_eer_percent(labels, [row["baseline_score"] for row in rows])
        candidate = compute_eer_percent(labels, [row["candidate_score"] for row in rows])
        cells.append({
            "dataset": key[0], "base_scenario": key[1], "duration": key[2],
            "trial_count": len(rows), "distillation_eer_percent": baseline,
            "candidate_eer_percent": candidate,
            "improvement_points_vs_distillation": baseline - candidate,
        })
    datasets = {
        dataset: float(np.mean([cell["improvement_points_vs_distillation"] for cell in cells
                                if cell["dataset"] == dataset]))
        for dataset in sorted({cell["dataset"] for cell in cells})
    }
    macro = float(np.mean([cell["improvement_points_vs_distillation"] for cell in cells]))
    short = float(np.mean([cell["improvement_points_vs_distillation"] for cell in cells
                           if cell["duration"] != "full"]))
    worst = float(min(cell["improvement_points_vs_distillation"] for cell in cells))
    bootstraps = {}
    if bootstrap:
        for offset, dataset in enumerate(sorted(datasets)):
            bootstraps[dataset] = speaker_bootstrap(
                scored, dataset, bootstrap, 20260908 + offset, True
            )
    return {
        "cells": cells, "datasets": datasets, "speaker_bootstrap": bootstraps,
        "macro_improvement_points_vs_distillation": macro,
        "short_macro_improvement_points_vs_distillation": short,
        "worst_cell_improvement_points_vs_distillation": worst,
    }


def config_grid():
    coarse = []
    for enroll_acoustic in (0.76, 0.80, 0.83, 0.86, 0.89, 0.92):
        for ptn_ratio in (0.0, 0.025, 0.05, 0.10, 0.15):
            for short_acoustic in (0.08, 0.14, 0.20, 0.26, 0.32, 0.38):
                coarse.append({
                    "enroll_acoustic": enroll_acoustic, "enroll_ptn_ratio": ptn_ratio,
                    "short05_acoustic": short_acoustic, "short05_ptn_ratio": ptn_ratio,
                    "short10_acoustic": 0.20, "short10_ptn_ratio": 0.05,
                })
    # Explicit nearby asymmetric ratios test whether PTN helps full and short
    # conditions differently.  Bounds remain deliberately conservative.
    asymmetric = []
    for enroll_acoustic in (0.80, 0.83, 0.86, 0.89):
        for short_acoustic in (0.14, 0.20, 0.26, 0.32):
            for enroll_ratio in (0.0, 0.025, 0.05, 0.10):
                for short_ratio in (0.0, 0.025, 0.05, 0.10, 0.15):
                    asymmetric.append({
                        "enroll_acoustic": enroll_acoustic, "enroll_ptn_ratio": enroll_ratio,
                        "short05_acoustic": short_acoustic, "short05_ptn_ratio": short_ratio,
                        "short10_acoustic": 0.20, "short10_ptn_ratio": 0.05,
                    })
    unique = {}
    for config in coarse + asymmetric:
        unique[json.dumps(config, sort_keys=True)] = config
    return list(unique.values())


def main():
    adapters = load_distillation_adapters()
    coverage = coverage_components(adapters)
    dev_suite = SUITES[0]
    sources, trials, cache = load_suite(dev_suite)
    external_dev = external_components(dev_suite, sources, trials, cache, adapters)
    grid = []
    for config in config_grid():
        coverage_metrics = summarize(score_rows(coverage, config))
        external_metrics = summarize(score_rows(external_dev, config))
        checks = {
            "coverage_macro_ge_0p10": coverage_metrics["macro_improvement_points_vs_distillation"] >= 0.10,
            "coverage_short_nonnegative": coverage_metrics["short_macro_improvement_points_vs_distillation"] >= 0.0,
            "coverage_worst_ge_minus_0p2": coverage_metrics["worst_cell_improvement_points_vs_distillation"] >= -0.20,
            "coverage_each_dataset_nonnegative": all(value >= 0.0 for value in coverage_metrics["datasets"].values()),
            "external_macro_positive": external_metrics["macro_improvement_points_vs_distillation"] > 0.0,
            "external_short_positive": external_metrics["short_macro_improvement_points_vs_distillation"] > 0.0,
            "external_worst_ge_minus_0p2": external_metrics["worst_cell_improvement_points_vs_distillation"] >= -0.20,
            "external_each_dataset_nonnegative": all(value >= 0.0 for value in external_metrics["datasets"].values()),
        }
        grid.append({"config": config, "coverage": coverage_metrics,
                     "external_development": external_metrics, "checks": checks,
                     "eligible": all(checks.values())})
    eligible = [row for row in grid if row["eligible"]]
    eligible.sort(key=lambda row: (
        min(row["coverage"]["macro_improvement_points_vs_distillation"],
            row["external_development"]["macro_improvement_points_vs_distillation"]),
        row["external_development"]["short_macro_improvement_points_vs_distillation"],
        row["coverage"]["short_macro_improvement_points_vs_distillation"],
        row["coverage"]["worst_cell_improvement_points_vs_distillation"],
    ), reverse=True)
    selected = eligible[0] if eligible else None
    if selected is None:
        ranked = sorted(grid, key=lambda row: (
            sum(row["checks"].values()),
            min(row["coverage"]["macro_improvement_points_vs_distillation"],
                row["external_development"]["macro_improvement_points_vs_distillation"]),
        ), reverse=True)
        result = {
            "schema_version": 1, "status": "REWEIGHT_GATE_FAILED_DO_NOT_PACKAGE",
            "grid_points": len(grid), "eligible_count": 0, "top_near_misses": ranked[:20],
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        atomic_json(RESULT_PATH, result)
        print(json.dumps({"status": result["status"], "grid_points": len(grid),
                          "best": ranked[0]}, ensure_ascii=False, indent=2), flush=True)
        return
    selected = json.loads(json.dumps(selected, sort_keys=True))
    development_scored = score_rows(coverage + external_dev, selected["config"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(DEVELOPMENT_TRACE, development_scored)

    final_suite = SUITES[1]
    final_sources, final_trials, final_cache = load_suite(final_suite)
    final_components = external_components(
        final_suite, final_sources, final_trials, final_cache, adapters
    )
    final_scored = score_rows(final_components, selected["config"])
    final_metrics = summarize(final_scored, bootstrap=2000)
    atomic_jsonl(FINAL_TRACE, final_scored)
    final_checks = {
        "macro_positive": final_metrics["macro_improvement_points_vs_distillation"] > 0.0,
        "short_macro_positive": final_metrics["short_macro_improvement_points_vs_distillation"] > 0.0,
        "worst_ge_minus_0p2": final_metrics["worst_cell_improvement_points_vs_distillation"] >= -0.20,
    }
    status = "FINAL_CONFIRMATION_PASS" if all(final_checks.values()) else "FINAL_CONFIRMATION_FAIL_DO_NOT_PACKAGE"
    result = {
        "schema_version": 1, "artifact_kind": "joint_distillation_block_reweight_validation",
        "status": status, "distillation_online_anchor": 89.76798, "target_online_score": 91.0,
        "grid_points": len(grid), "eligible_count": len(eligible),
        "frozen_selection_before_final_confirmation": selected,
        "final_confirmation": final_metrics, "final_checks": final_checks,
        "development_trace": {"path": str(DEVELOPMENT_TRACE.relative_to(ROOT)),
                              "sha256": sha256_file(DEVELOPMENT_TRACE)},
        "final_trace": {"path": str(FINAL_TRACE.relative_to(ROOT)),
                        "sha256": sha256_file(FINAL_TRACE)},
        "limitations": [
            "Public cross-domain validation cannot guarantee a hidden leaderboard score of 91.",
            "The acoustic models and KD adapters are byte-identical to released distillation.",
        ],
    }
    atomic_json(RESULT_PATH, result)
    print(json.dumps({
        "status": status, "grid_points": len(grid), "eligible_count": len(eligible),
        "selected": selected, "final_confirmation": final_metrics,
        "result": str(RESULT_PATH), "sha256": sha256_file(RESULT_PATH),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
