#!/usr/bin/env python3
"""Search a constrained fusion frontier for the frozen multidomain adapters.

This is an offline diagnostic.  It searches only predefined conservative
weight grids and three-seed aggregation rules, records every point, and never
builds a package.  External-set constraints prevent accepting a coverage-only
winner that regresses on HI-MIA or H1.
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

from tools.run_redim_output_kd_support import (
    HOLDS,
    MODEL_DIRS,
    MODEL_NAMES,
    l2,
    load_redim,
    load_trials,
    load_acoustic,
    transform,
)
from tools.validate_redim_output_kd_3seed import (
    EXTERNALS,
    atomic_json,
    external_trace,
    load_external_cache,
    load_external_trials,
    summarize_external,
)
from tools.score_lodo_coverage import compute_eer_percent


SOURCE_ADAPTERS = ROOT / (
    "records/2026-09-04/020_distillation_redim_output_kd_multidomain/"
    "distillation_multidomain_redim_output_kd_soup.npz"
)
OUTPUT = ROOT / (
    "records/2026-09-04/021_distillation_multidomain_fusion_frontier/"
    "FRONTIER_RESULT.json"
)
SEEDS = (824501, 824502, 824503)
AGGREGATIONS = ("mean_delta", "median_delta", "output_ensemble")
ENROLL_ACOUSTIC_WEIGHTS = (0.86, 0.90, 0.94, 0.98)
SHORT_ACOUSTIC_WEIGHTS = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
PTN_RATIOS = (0.05, 0.10, 0.20)
ADAPTER_MIXES = (0.25, 0.50, 0.75, 1.0)


def branch_weights(acoustic_weight, ptn_ratio):
    residual = 1.0 - acoustic_weight
    return (acoustic_weight, residual * (1.0 - ptn_ratio), residual * ptn_ratio)


def load_adapters():
    with np.load(str(SOURCE_ADAPTERS), allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    output = {}
    for aggregation in AGGREGATIONS:
        output[aggregation] = {name: {} for name in MODEL_NAMES}
        for name in MODEL_NAMES:
            for duration in ("short05", "short10"):
                seeds = np.stack([
                    arrays["seed{}__{}__{}".format(seed, name, duration)]
                    for seed in SEEDS
                ]).astype(np.float32)
                if aggregation == "mean_delta":
                    value = np.mean(seeds.astype(np.float64), axis=0).astype(np.float32)
                elif aggregation == "median_delta":
                    value = np.median(seeds, axis=0).astype(np.float32)
                else:
                    value = seeds
                output[aggregation][name][duration] = value
    return output


def adapted(vector, adapters, aggregation, name, duration, mix):
    matrix = adapters[aggregation][name][duration]
    if aggregation != "output_ensemble":
        return transform(vector, matrix, mix)
    outputs = [transform(vector, item, mix) for item in matrix]
    return l2(np.mean(np.stack(outputs).astype(np.float64), axis=0))


def candidate_score(enrollment, test, enrollment_weights, test_weights):
    return float(sum(
        math.sqrt(left * right) * float(np.dot(e, t))
        for e, t, left, right in zip(
            enrollment, test, enrollment_weights, test_weights
        )
    ))


def coverage_trace(acoustic, acoustic_index, redims, trials, adapters, config):
    enroll_weights = branch_weights(config["enroll_acoustic"], config["ptn_ratio"])
    short_weights = branch_weights(config["short_acoustic"], config["ptn_ratio"])
    trace = []
    for raw in trials:
        enrollment_id = str(raw["enrollment_utterance_id"])
        test_id = str(raw["test_utterance_id"])
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
            for view, index in enumerate(indices):
                test_acoustic = l2(acoustic[acoustic_key][index])
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
                                config["aggregation"], "lm", key, config["adapter_mix"]),
                        adapted(redims["ptn"][test_id][redim_key][view], adapters,
                                config["aggregation"], "ptn", key, config["adapter_mix"]),
                    ]
                    weights = short_weights
                candidate_scores.append(candidate_score(
                    enrollment, test, enroll_weights, weights
                ))
            trace.append({
                "fold": str(raw["dataset"]),
                "base_scenario": str(raw["base_scenario"]),
                "duration": duration, "label": int(raw["label"]),
                "baseline": float(np.mean(baseline_scores)),
                "candidate": float(np.mean(candidate_scores)),
            })
    return trace


def summarize_coverage(trace):
    grouped = defaultdict(list)
    for row in trace:
        grouped[(row["fold"], row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, rows in sorted(grouped.items()):
        labels = [row["label"] for row in rows]
        baseline = compute_eer_percent(labels, [row["baseline"] for row in rows])
        candidate = compute_eer_percent(labels, [row["candidate"] for row in rows])
        cells.append({
            "fold": key[0], "base_scenario": key[1], "duration": key[2],
            "baseline_eer": baseline, "candidate_eer": candidate,
            "improvement": baseline - candidate, "rows": len(rows),
        })
    short = [cell for cell in cells if cell["duration"] != "full"]
    folds = {
        fold: float(np.mean([
            cell["improvement"] for cell in cells if cell["fold"] == fold
        ]))
        for fold in HOLDS
    }
    return {
        "macro": float(np.mean([cell["improvement"] for cell in cells])),
        "short_macro": float(np.mean([cell["improvement"] for cell in short])),
        "worst_cell": float(min(cell["improvement"] for cell in cells)),
        "folds": folds, "cells": cells,
    }


def coverage_components(acoustic, acoustic_index, redims, trials, adapters, aggregation, mix):
    trace = []
    for raw in trials:
        enrollment_id = str(raw["enrollment_utterance_id"])
        test_id = str(raw["test_utterance_id"])
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
            component_scores = [[], [], []]
            for view, index in enumerate(indices):
                test_acoustic = l2(acoustic[acoustic_key][index])
                if duration == "full":
                    test = [
                        test_acoustic, redims["lm"][test_id]["full"],
                        redims["ptn"][test_id]["full"],
                    ]
                else:
                    key = "short05" if duration == "0p5" else "short10"
                    test = [
                        test_acoustic,
                        adapted(redims["lm"][test_id][redim_key][view], adapters,
                                aggregation, "lm", key, mix),
                        adapted(redims["ptn"][test_id][redim_key][view], adapters,
                                aggregation, "ptn", key, mix),
                    ]
                for branch in range(3):
                    component_scores[branch].append(float(np.dot(
                        enrollment[branch], test[branch]
                    )))
            trace.append({
                "fold": str(raw["dataset"]),
                "base_scenario": str(raw["base_scenario"]),
                "duration": duration, "label": int(raw["label"]),
                "baseline": float(np.mean(component_scores[0])),
                "components": [float(np.mean(values)) for values in component_scores],
            })
    return trace


def external_components(dataset, cache, trials, adapters, aggregation, mix):
    trace = []
    for raw in trials:
        enrollment_id = str(raw["enrollment"]["utterance_stable_id"])
        test_id = str(raw["test"]["utterance_stable_id"])
        enrollment_row = cache[enrollment_id]
        test_row = cache[test_id]
        enrollment = [
            enrollment_row["acoustic_full"], enrollment_row["lm_full"],
            enrollment_row["ptn_full"],
        ]
        for duration in ("full", "0p5", "1p0"):
            acoustic_values = test_row["acoustic_" + duration]
            if duration == "full":
                acoustic_values = acoustic_values[None, :]
            component_scores = [[], [], []]
            for view, test_acoustic in enumerate(acoustic_values):
                if duration == "full":
                    test = [test_acoustic, test_row["lm_full"], test_row["ptn_full"]]
                else:
                    key = "short05" if duration == "0p5" else "short10"
                    test = [
                        test_acoustic,
                        adapted(test_row["lm_" + duration][view], adapters,
                                aggregation, "lm", key, mix),
                        adapted(test_row["ptn_" + duration][view], adapters,
                                aggregation, "ptn", key, mix),
                    ]
                for branch in range(3):
                    component_scores[branch].append(float(np.dot(
                        enrollment[branch], test[branch]
                    )))
            trace.append({
                "fold": dataset,
                "base_scenario": "distance_" + test_row["distance"][:12],
                "duration": duration, "trial_id": str(raw["trial_id"]),
                "label": int(raw["label"]),
                "enrollment_speaker_id": str(raw["enrollment_speaker_id"]),
                "test_speaker_id": str(raw["test_speaker_id"]),
                "baseline_score": float(np.mean(component_scores[0])),
                "components": [float(np.mean(values)) for values in component_scores],
            })
    return trace


def apply_weights(trace, config, external):
    enroll_weights = branch_weights(config["enroll_acoustic"], config["ptn_ratio"])
    short_weights = branch_weights(config["short_acoustic"], config["ptn_ratio"])
    output = []
    for row in trace:
        weights = enroll_weights if row["duration"] == "full" else short_weights
        candidate = sum(
            math.sqrt(left * right) * score
            for score, left, right in zip(row["components"], enroll_weights, weights)
        )
        copied = dict(row)
        copied.pop("components")
        if external:
            copied["candidate_score"] = float(candidate)
        else:
            copied["candidate"] = float(candidate)
        output.append(copied)
    return output


def external_weighted_trace(dataset, cache, trials, adapters, config):
    enroll_weights = branch_weights(config["enroll_acoustic"], config["ptn_ratio"])
    short_weights = branch_weights(config["short_acoustic"], config["ptn_ratio"])
    trace = []
    for raw in trials:
        enrollment_id = str(raw["enrollment"]["utterance_stable_id"])
        test_id = str(raw["test"]["utterance_stable_id"])
        enrollment_row = cache[enrollment_id]
        test_row = cache[test_id]
        enrollment = [
            enrollment_row["acoustic_full"], enrollment_row["lm_full"],
            enrollment_row["ptn_full"],
        ]
        for duration in ("full", "0p5", "1p0"):
            acoustic_values = test_row["acoustic_" + duration]
            if duration == "full":
                acoustic_values = acoustic_values[None, :]
            baseline_scores = []
            candidate_scores = []
            for view, test_acoustic in enumerate(acoustic_values):
                baseline_scores.append(float(np.dot(enrollment[0], test_acoustic)))
                if duration == "full":
                    test = [test_acoustic, test_row["lm_full"], test_row["ptn_full"]]
                    weights = enroll_weights
                else:
                    key = "short05" if duration == "0p5" else "short10"
                    test = [
                        test_acoustic,
                        adapted(test_row["lm_" + duration][view], adapters,
                                config["aggregation"], "lm", key, config["adapter_mix"]),
                        adapted(test_row["ptn_" + duration][view], adapters,
                                config["aggregation"], "ptn", key, config["adapter_mix"]),
                    ]
                    weights = short_weights
                candidate_scores.append(candidate_score(
                    enrollment, test, enroll_weights, weights
                ))
            trace.append({
                "fold": dataset,
                "base_scenario": "distance_" + test_row["distance"][:12],
                "duration": duration, "trial_id": str(raw["trial_id"]),
                "label": int(raw["label"]),
                "enrollment_speaker_id": str(raw["enrollment_speaker_id"]),
                "test_speaker_id": str(raw["test_speaker_id"]),
                "baseline_score": float(np.mean(baseline_scores)),
                "candidate_score": float(np.mean(candidate_scores)),
            })
    return trace


def eligible(coverage, external):
    return bool(
        coverage["macro"] >= 2.6
        and coverage["short_macro"] >= 3.0
        and coverage["worst_cell"] >= -0.2
        and all(value >= 0.0 for value in coverage["folds"].values())
        and all(row["macro_improvement_points"] >= 0.0 for row in external.values())
        and all(row["short_macro_improvement_points"] >= 0.0 for row in external.values())
        and all(row["worst_cell_improvement_points"] >= -0.2 for row in external.values())
    )


def main():
    adapters = load_adapters()
    acoustic, acoustic_index, _ = load_acoustic()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in MODEL_NAMES}
    trials = load_trials()
    external_inputs = {}
    for dataset, spec in EXTERNALS.items():
        external_inputs[dataset] = (
            load_external_cache(spec["cache"], spec["expected_cache"]),
            load_external_trials(spec["trials"], spec["split"], spec["expected_trials"]),
        )
    component_cache = {}
    for aggregation in AGGREGATIONS:
        for adapter_mix in ADAPTER_MIXES:
            key = (aggregation, adapter_mix)
            print("PRECOMPUTE {} {}".format(aggregation, adapter_mix), flush=True)
            component_cache[key] = {
                "coverage": coverage_components(
                    acoustic, acoustic_index, redims, trials, adapters, aggregation, adapter_mix
                ),
                "external": {
                    dataset: external_components(
                        dataset, cache, external_trials, adapters, aggregation, adapter_mix
                    )
                    for dataset, (cache, external_trials) in external_inputs.items()
                },
            }
    results = []
    total = (len(AGGREGATIONS) * len(ENROLL_ACOUSTIC_WEIGHTS) * len(SHORT_ACOUSTIC_WEIGHTS)
             * len(PTN_RATIOS) * len(ADAPTER_MIXES))
    index = 0
    for aggregation in AGGREGATIONS:
        for enroll_acoustic in ENROLL_ACOUSTIC_WEIGHTS:
            for short_acoustic in SHORT_ACOUSTIC_WEIGHTS:
                for ptn_ratio in PTN_RATIOS:
                    for adapter_mix in ADAPTER_MIXES:
                        index += 1
                        config = {
                            "aggregation": aggregation, "enroll_acoustic": enroll_acoustic,
                            "short_acoustic": short_acoustic, "ptn_ratio": ptn_ratio,
                            "adapter_mix": adapter_mix,
                        }
                        components = component_cache[(aggregation, adapter_mix)]
                        coverage = summarize_coverage(apply_weights(
                            components["coverage"], config, False
                        ))
                        external = {}
                        for dataset in external_inputs:
                            external[dataset] = summarize_external(apply_weights(
                                components["external"][dataset], config, True
                            ))
                        results.append({
                            "config": config, "coverage": coverage,
                            "external": external,
                            "point_gate_pass": eligible(coverage, external),
                        })
                        if index % 72 == 0:
                            print("PROGRESS {}/{}".format(index, total), flush=True)
    passing = [row for row in results if row["point_gate_pass"]]
    ranking = sorted(passing if passing else results, key=lambda row: (
        row["point_gate_pass"], row["coverage"]["macro"],
        row["coverage"]["short_macro"], row["coverage"]["worst_cell"],
        min(item["worst_cell_improvement_points"] for item in row["external"].values()),
    ), reverse=True)
    report = {
        "schema_version": 1,
        "artifact_kind": "distillation_multidomain_constrained_fusion_frontier",
        "research_only": True, "package_created": False,
        "source_adapters": str(SOURCE_ADAPTERS),
        "grid": {
            "aggregations": list(AGGREGATIONS),
            "enroll_acoustic_weights": list(ENROLL_ACOUSTIC_WEIGHTS),
            "short_acoustic_weights": list(SHORT_ACOUSTIC_WEIGHTS),
            "ptn_ratios": list(PTN_RATIOS),
            "adapter_mixes": list(ADAPTER_MIXES), "total": total,
        },
        "point_gate": {
            "coverage_macro": 2.6, "coverage_short_macro": 3.0,
            "worst_cell": -0.2, "coverage_folds": "all_nonnegative",
            "external_macro_short": "all_nonnegative",
            "external_worst_cell": -0.2,
        },
        "passing_count": len(passing),
        "best": ranking[0],
        "top_passing": ranking[:20] if passing else [],
        "all_results": results,
        "status": "POINT_FRONTIER_FOUND_NEEDS_FRESH_CONFIRMATION" if passing else (
            "NO_POINT_FRONTIER_DO_NOT_PACKAGE"
        ),
        "limitations": [
            "This grid reuses coverage, HI-MIA dev, and H1 for constrained selection.",
            "A selected point requires a separate frozen-candidate audit before packaging.",
            "No proxy result can guarantee a hidden score above 91.",
        ],
    }
    atomic_json(OUTPUT, report)
    print(json.dumps({
        "status": report["status"], "passing_count": len(passing),
        "best": ranking[0], "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
