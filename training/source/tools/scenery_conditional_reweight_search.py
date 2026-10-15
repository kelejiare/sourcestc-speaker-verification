#!/usr/bin/env python3
"""Search source-domain weights using the official two-scenery path contract.

The starter kit exposes ``data_scenery1`` and ``data_scenery2``.  The official
task page states that the 15 EERs are derived from five base scenarios across
two source datasets, each with full/0.5/1.0 test duration.  This experiment
keeps the joint models and adapters fixed and searches separate, cosine-exact
block weights for the two observable source domains.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.score_lodo_coverage import compute_eer_percent
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.distillation_reweight_validation import (
    coverage_components,
    external_components,
    score_dots,
    weight_vector,
)
from tools.nap_lodo import atomic_json, sha256_file
from joint_kd_inference import (
    ENROLL_WEIGHTS,
    SHORT05_WEIGHTS,
    SHORT10_WEIGHTS,
    _load_joint_adapter,
)


OUTPUT_DIR = ROOT / "records/2026-09-06/059_mobile_scenery_conditional_reweight"
RESULT_PATH = OUTPUT_DIR / "SEARCH_RESULT.json"
ADAPTER_PATH = ROOT / "model/joint_kd_adapter.npz"
SCENERY1_DATASETS = {
    "3dspeaker", "childmandarin", "childmandarin_dev_holdout",
}
SCENERY2_DATASETS = {
    "aishell4", "alimeeting_eval_holdout", "hi_mia_internal_h1",
}


def load_adapters():
    matrices, receipt = _load_joint_adapter(ADAPTER_PATH)
    return {
        "lm": {"short05": matrices["lm_short05"], "short10": matrices["lm_short10"]},
        "ptn": {"short05": matrices["ptn_short05"], "short10": matrices["ptn_short10"]},
    }, receipt


def vectors(kind):
    if kind == "enroll":
        acoustic_values = (0.50, 0.56, 0.62, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92)
        ratios = (0.0, 0.05, 0.10, 0.20, 0.30)
        anchors = (ENROLL_WEIGHTS,)
    else:
        acoustic_values = (0.05, 0.10, 0.15, 0.20, 0.26, 0.32, 0.38, 0.44, 0.50)
        ratios = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
        anchors = (SHORT05_WEIGHTS, SHORT10_WEIGHTS)
    unique = {}
    for acoustic_value in acoustic_values:
        for ratio in ratios:
            value = weight_vector(acoustic_value, ratio)
            unique[tuple(round(float(item), 12) for item in value)] = value
    for value in anchors:
        unique[tuple(round(float(item), 12) for item in value)] = np.asarray(value)
    return [unique[key] for key in sorted(unique)]


def baseline_score(row):
    right = (
        ENROLL_WEIGHTS if row["duration"] == "full" else
        SHORT05_WEIGHTS if row["duration"] == "0p5" else SHORT10_WEIGHTS
    )
    return score_dots(row["dots"], ENROLL_WEIGHTS, right)


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["dataset"], row["base_scenario"], row["duration"])].append(row)
    return dict(grouped)


def cell_table(grouped, enroll, test, duration):
    output = []
    for key, rows in sorted(grouped.items()):
        if key[3] != duration:
            continue
        labels = [row["label"] for row in rows]
        baseline = compute_eer_percent(labels, [baseline_score(row) for row in rows])
        candidate = compute_eer_percent(labels, [
            score_dots(row["dots"], enroll, test) for row in rows
        ])
        output.append({
            "suite": key[0], "dataset": key[1], "base_scenario": key[2],
            "duration": key[3], "trial_count": len(rows),
            "joint_eer_percent": baseline, "candidate_eer_percent": candidate,
            "improvement_points_vs_joint": baseline - candidate,
        })
    return output


def metrics(cells):
    datasets = {
        dataset: float(np.mean([
            cell["improvement_points_vs_joint"] for cell in cells
            if cell["dataset"] == dataset
        ]))
        for dataset in sorted({cell["dataset"] for cell in cells})
    }
    suites = {
        suite: float(np.mean([
            cell["improvement_points_vs_joint"] for cell in cells
            if cell["suite"] == suite
        ]))
        for suite in sorted({cell["suite"] for cell in cells})
    }
    return {
        "macro": float(np.mean([cell["improvement_points_vs_joint"] for cell in cells])),
        "short_macro": float(np.mean([
            cell["improvement_points_vs_joint"] for cell in cells
            if cell["duration"] != "full"
        ])),
        "worst": float(min(cell["improvement_points_vs_joint"] for cell in cells)),
        "datasets": datasets, "suites": suites, "cells": cells,
    }


def eligible(summary):
    checks = {
        "macro_positive": summary["macro"] > 0.0,
        "short_nonnegative": summary["short_macro"] >= 0.0,
        "worst_ge_minus_0p2": summary["worst"] >= -0.2,
        "every_dataset_nonnegative": all(value >= 0.0 for value in summary["datasets"].values()),
        "every_suite_nonnegative": all(value >= 0.0 for value in summary["suites"].values()),
    }
    return checks, all(checks.values())


def search_domain(rows, name):
    grouped = group_rows(rows)
    enroll_grid = vectors("enroll")
    short_grid = vectors("short")
    results = []
    for enroll in enroll_grid:
        full_cells = cell_table(grouped, enroll, enroll, "full")
        short_options = {}
        for duration in ("0p5", "1p0"):
            values = []
            for short in short_grid:
                cells = cell_table(grouped, enroll, short, duration)
                summary = metrics(cells)
                values.append((short, summary))
            values.sort(key=lambda item: (
                min(item[1]["datasets"].values()), item[1]["macro"], item[1]["worst"]
            ), reverse=True)
            short_options[duration] = values[:12]
        for short05, cells05 in short_options["0p5"]:
            for short10, cells10 in short_options["1p0"]:
                summary = metrics(full_cells + cells05["cells"] + cells10["cells"])
                checks, passes = eligible(summary)
                results.append({
                    "domain": name,
                    "weights": {
                        "enroll": [float(item) for item in enroll],
                        "short05": [float(item) for item in short05],
                        "short10": [float(item) for item in short10],
                    },
                    "metrics": summary, "checks": checks, "eligible": passes,
                })
    results.sort(key=lambda row: (
        row["eligible"], min(row["metrics"]["suites"].values()),
        min(row["metrics"]["datasets"].values()), row["metrics"]["macro"],
        row["metrics"]["short_macro"], row["metrics"]["worst"],
    ), reverse=True)
    return results


def main():
    adapters, adapter_receipt = load_adapters()
    rows = coverage_components(adapters)
    for suite in (SUITES[0], SUITES[2]):
        sources, trials, cache = load_suite(suite)
        rows.extend(external_components(suite, sources, trials, cache, adapters))
    unknown = sorted({row["dataset"] for row in rows} - SCENERY1_DATASETS - SCENERY2_DATASETS)
    if unknown:
        raise ValueError("unmapped proxy datasets: {}".format(unknown))
    domain_rows = {
        "data_scenery1": [row for row in rows if row["dataset"] in SCENERY1_DATASETS],
        "data_scenery2": [row for row in rows if row["dataset"] in SCENERY2_DATASETS],
    }
    searches = {}
    for name, selected_rows in domain_rows.items():
        print("SEARCH {} rows={}".format(name, len(selected_rows)), flush=True)
        searches[name] = search_domain(selected_rows, name)
        best = searches[name][0]
        print("  best macro={:.6f} short={:.6f} worst={:.6f} eligible={}".format(
            best["metrics"]["macro"], best["metrics"]["short_macro"],
            best["metrics"]["worst"], best["eligible"],
        ), flush=True)
    selected = {name: values[0] for name, values in searches.items()}
    aggregate_cells = []
    for row in selected.values():
        aggregate_cells.extend(row["metrics"]["cells"])
    aggregate = metrics(aggregate_cells)
    release_checks = {
        "both_domains_eligible": all(row["eligible"] for row in selected.values()),
        "aggregate_macro_ge_0p5": aggregate["macro"] >= 0.5,
        "aggregate_short_positive": aggregate["short_macro"] > 0.0,
        "aggregate_worst_ge_minus_0p2": aggregate["worst"] >= -0.2,
    }
    status = (
        "DEVELOPMENT_GATE_PASS_REQUIRES_ALIMEETING_TEST_AUDIT"
        if all(release_checks.values()) else "DEVELOPMENT_GATE_FAILED_DO_NOT_PACKAGE"
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "mobile_official_two_scenery_conditional_reweight_search",
        "status": status,
        "official_contract_evidence": {
            "starter_kit_paths": ["data_scenery1", "data_scenery2"],
            "source_dataset_count": 2,
            "base_scenario_count": 5,
            "duration_variants": ["full", "0p5", "1p0"],
            "final_scenario_count": 15,
        },
        "adapter_receipt": adapter_receipt,
        "row_counts": {name: len(value) for name, value in domain_rows.items()},
        "grid": {
            "enroll_vectors": len(vectors("enroll")),
            "short_vectors": len(vectors("short")),
            "retained_short_options_per_enroll_duration": 12,
        },
        "selected": selected,
        "aggregate": aggregate,
        "release_checks": release_checks,
        "top_candidates": {name: values[:10] for name, values in searches.items()},
        "limitations": [
            "The starter kit identifies two source directories but does not explicitly publish which source is numbered 1 versus 2.",
            "The mapping used here follows the starter-kit order and must be protected by a joint fallback for unknown paths.",
            "Public proxy improvement cannot guarantee the hidden leaderboard score.",
        ],
    }
    atomic_json(RESULT_PATH, payload)
    print(json.dumps({
        "status": status,
        "selected": {name: {"weights": row["weights"], "metrics": {
            key: row["metrics"][key] for key in ("macro", "short_macro", "worst", "datasets", "suites")
        }} for name, row in selected.items()},
        "aggregate": {key: aggregate[key] for key in ("macro", "short_macro", "worst", "datasets", "suites")},
        "release_checks": release_checks,
        "result": str(RESULT_PATH.relative_to(ROOT)), "sha256": sha256_file(RESULT_PATH),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
