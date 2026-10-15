#!/usr/bin/env python3
"""Train separate three-seed ReDim KD adapters for the two official sources.

The acoustic checkpoints and feature frontends stay byte-identical to distillation/joint.
Only four 192x192 output transforms per observable source directory are
changed.  Train/evaluation speakers remain disjoint through the existing
frozen manifests and cache splits.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_redim_output_kd_support import (
    MODEL_DIRS, MODEL_NAMES, load_redim, load_acoustic,
)
from tools.train_allsource_all_himia import all_himia_ids
from tools.validate_redim_output_kd_3seed import EXTERNALS, load_external_cache
from tools.score_lodo_coverage import compute_eer_percent
from tools.adapter_prior_search import fit_candidate
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.distillation_reweight_validation import (
    coverage_components, external_components, score_dots,
)
from tools.nap_lodo import atomic_json, sha256_file
from joint_kd_inference import (
    ENROLL_WEIGHTS, SHORT05_WEIGHTS, SHORT10_WEIGHTS, _load_joint_adapter,
)


OUTPUT_DIR = ROOT / "records/2026-09-06/060_mobile_scenery_conditional_adapters"
RESULT_PATH = OUTPUT_DIR / "SEARCH_RESULT.json"
ADAPTER_PATH = ROOT / "model/joint_kd_adapter.npz"
WEIGHT_RESULT = ROOT / "records/2026-09-06/059_mobile_scenery_conditional_reweight/SEARCH_RESULT.json"
SCENERY1_DATASETS = {"3dspeaker", "childmandarin", "childmandarin_dev_holdout"}
SCENERY2_DATASETS = {"aishell4", "alimeeting_eval_holdout", "hi_mia_internal_h1"}


CONFIGS = {
    "data_scenery1": (
        {"id": "mobile_d40_c35", "priors": {
            "3dspeaker": .40, "childmandarin": .35, "commonvoice_beijing": .10,
            "magicdata_slr68": .05, "seniortalk": .05, "aishell4": .025,
            "himia_train": .025}},
        {"id": "mobile_d50_c30", "priors": {
            "3dspeaker": .50, "childmandarin": .30, "commonvoice_beijing": .08,
            "magicdata_slr68": .04, "seniortalk": .04, "aishell4": .02,
            "himia_train": .02}},
        {"id": "mobile_d30_c50", "priors": {
            "3dspeaker": .30, "childmandarin": .50, "commonvoice_beijing": .08,
            "magicdata_slr68": .04, "seniortalk": .04, "aishell4": .02,
            "himia_train": .02}},
        {"id": "mobile_d42_c42", "priors": {
            "3dspeaker": .42, "childmandarin": .42, "commonvoice_beijing": .06,
            "magicdata_slr68": .03, "seniortalk": .03, "aishell4": .02,
            "himia_train": .02}},
    ),
    "data_scenery2": (
        {"id": "meeting_h50_a25", "priors": {
            "himia_train": .50, "aishell4": .25, "3dspeaker": .10,
            "childmandarin": .05, "seniortalk": .05,
            "commonvoice_beijing": .03, "magicdata_slr68": .02}},
        {"id": "meeting_h60_a20", "priors": {
            "himia_train": .60, "aishell4": .20, "3dspeaker": .08,
            "childmandarin": .04, "seniortalk": .04,
            "commonvoice_beijing": .02, "magicdata_slr68": .02}},
        {"id": "meeting_h40_a35", "priors": {
            "himia_train": .40, "aishell4": .35, "3dspeaker": .10,
            "childmandarin": .05, "seniortalk": .04,
            "commonvoice_beijing": .03, "magicdata_slr68": .03}},
        {"id": "meeting_h45_a20_d20", "priors": {
            "himia_train": .45, "aishell4": .20, "3dspeaker": .20,
            "childmandarin": .05, "seniortalk": .04,
            "commonvoice_beijing": .03, "magicdata_slr68": .03}},
    ),
}


def current_adapters():
    matrices, receipt = _load_joint_adapter(ADAPTER_PATH)
    return {
        "lm": {"short05": matrices["lm_short05"], "short10": matrices["lm_short10"]},
        "ptn": {"short05": matrices["ptn_short05"], "short10": matrices["ptn_short10"]},
    }, receipt


def read_weights():
    with WEIGHT_RESULT.open("r", encoding="utf-8") as stream:
        selected = json.load(stream)["selected"]
    return {
        domain: {
            key: np.asarray(row["weights"][key], dtype=np.float64)
            for key in ("enroll", "short05", "short10")
        }
        for domain, row in selected.items()
    }


def row_key(row):
    return (row["suite"], row["dataset"], row["base_scenario"],
            row["duration"], row["trial_id"])


def baseline_lookup(current):
    rows = coverage_components(current)
    for suite in (SUITES[0], SUITES[2]):
        sources, trials, cache = load_suite(suite)
        rows.extend(external_components(suite, sources, trials, cache, current))
    output = {}
    for row in rows:
        right = (ENROLL_WEIGHTS if row["duration"] == "full" else
                 SHORT05_WEIGHTS if row["duration"] == "0p5" else SHORT10_WEIGHTS)
        output[row_key(row)] = score_dots(row["dots"], ENROLL_WEIGHTS, right)
    return output


def candidate_rows(candidate):
    rows = coverage_components(candidate)
    for suite in (SUITES[0], SUITES[2]):
        sources, trials, cache = load_suite(suite)
        rows.extend(external_components(suite, sources, trials, cache, candidate))
    return rows


def summarize(rows, baseline, weights, datasets):
    grouped = defaultdict(list)
    for row in rows:
        if row["dataset"] in datasets:
            grouped[(row["suite"], row["dataset"], row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, selected in sorted(grouped.items()):
        duration = key[3]
        right = weights["enroll"] if duration == "full" else weights["short05"] if duration == "0p5" else weights["short10"]
        labels = [row["label"] for row in selected]
        base = compute_eer_percent(labels, [baseline[row_key(row)] for row in selected])
        proposed = compute_eer_percent(labels, [
            score_dots(row["dots"], weights["enroll"], right) for row in selected
        ])
        cells.append({
            "suite": key[0], "dataset": key[1], "base_scenario": key[2],
            "duration": duration, "trial_count": len(selected),
            "joint_eer_percent": base, "candidate_eer_percent": proposed,
            "improvement_points_vs_joint": base - proposed,
        })
    by_dataset = {
        dataset: float(np.mean([cell["improvement_points_vs_joint"] for cell in cells if cell["dataset"] == dataset]))
        for dataset in sorted({cell["dataset"] for cell in cells})
    }
    by_suite = {
        suite: float(np.mean([cell["improvement_points_vs_joint"] for cell in cells if cell["suite"] == suite]))
        for suite in sorted({cell["suite"] for cell in cells})
    }
    return {
        "macro": float(np.mean([cell["improvement_points_vs_joint"] for cell in cells])),
        "short_macro": float(np.mean([cell["improvement_points_vs_joint"] for cell in cells if cell["duration"] != "full"])),
        "worst": float(min(cell["improvement_points_vs_joint"] for cell in cells)),
        "datasets": by_dataset, "suites": by_suite, "cells": cells,
    }


def checks(metrics):
    values = {
        "macro_positive": metrics["macro"] > 0.0,
        "short_positive": metrics["short_macro"] > 0.0,
        "worst_ge_minus_0p2": metrics["worst"] >= -0.2,
        "every_dataset_nonnegative": all(value >= 0.0 for value in metrics["datasets"].values()),
        "every_suite_nonnegative": all(value >= 0.0 for value in metrics["suites"].values()),
    }
    return values, all(values.values())


def main():
    torch.set_num_threads(4)
    for configs in CONFIGS.values():
        for config in configs:
            if abs(sum(config["priors"].values()) - 1.0) > 1.0e-12:
                raise ValueError("priors do not sum to one: {}".format(config["id"]))
    current, adapter_receipt = current_adapters()
    baseline = baseline_lookup(current)
    weights = read_weights()
    _, _, metadata = load_acoustic()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in MODEL_NAMES}
    himia_spec = EXTERNALS["himia_dev"]
    himia = load_external_cache(himia_spec["cache"], himia_spec["expected_cache"])
    himia_ids, _ = all_himia_ids(himia_spec["trials"])
    results = {}
    adapters = {}
    for domain, configs in CONFIGS.items():
        datasets = SCENERY1_DATASETS if domain == "data_scenery1" else SCENERY2_DATASETS
        values = []
        for index, config in enumerate(configs, 1):
            print("FIT {} {}/{} {}".format(domain, index, len(configs), config["id"]), flush=True)
            candidate, audits = fit_candidate(redims, metadata, himia, himia_ids, config)
            adapters[config["id"]] = candidate
            metric = summarize(candidate_rows(candidate), baseline, weights[domain], datasets)
            gate, passes = checks(metric)
            values.append({
                "config": config, "metrics": metric, "checks": gate,
                "eligible": passes,
                "fit_elapsed_seconds": float(sum(item["elapsed_seconds"] for item in audits)),
                "adapter_frobenius": {
                    model + "_" + duration: float(np.linalg.norm(candidate[model][duration]))
                    for model in ("lm", "ptn") for duration in ("short05", "short10")
                },
            })
            print("  macro={:.6f} short={:.6f} worst={:.6f} pass={}".format(
                metric["macro"], metric["short_macro"], metric["worst"], passes), flush=True)
        values.sort(key=lambda row: (
            row["eligible"], min(row["metrics"]["suites"].values()),
            min(row["metrics"]["datasets"].values()), row["metrics"]["macro"],
            row["metrics"]["short_macro"], row["metrics"]["worst"],
        ), reverse=True)
        results[domain] = values
    selected = {domain: values[0] for domain, values in results.items()}
    all_cells = []
    for row in selected.values():
        all_cells.extend(row["metrics"]["cells"])
    aggregate = {
        "macro": float(np.mean([cell["improvement_points_vs_joint"] for cell in all_cells])),
        "short_macro": float(np.mean([cell["improvement_points_vs_joint"] for cell in all_cells if cell["duration"] != "full"])),
        "worst": float(min(cell["improvement_points_vs_joint"] for cell in all_cells)),
    }
    release_checks = {
        "both_domains_eligible": all(row["eligible"] for row in selected.values()),
        "macro_ge_0p8": aggregate["macro"] >= 0.8,
        "short_macro_ge_0p8": aggregate["short_macro"] >= 0.8,
        "worst_ge_minus_0p2": aggregate["worst"] >= -0.2,
    }
    status = "DEVELOPMENT_GATE_PASS_REQUIRES_WEIGHT_REFINEMENT" if all(release_checks.values()) else "DEVELOPMENT_GATE_FAILED_DO_NOT_PACKAGE"
    payload = {
        "schema_version": 1,
        "artifact_kind": "mobile_two_scenery_three_seed_output_kd_search",
        "status": status,
        "baseline": "joint_joint_kd_reweight",
        "adapter_receipt": adapter_receipt,
        "fixed_domain_weights": {domain: {key: value.tolist() for key, value in row.items()} for domain, row in weights.items()},
        "three_seed_ids": [825001, 825002, 825003],
        "selected": selected, "aggregate": aggregate,
        "release_checks": release_checks, "all_results": results,
        "limitations": [
            "All public corpora are proxy evidence and cannot guarantee the hidden score.",
            "This stage selects priors under fixed domain weights; a passing candidate still requires a frozen weight refinement and runtime gate.",
        ],
    }
    atomic_json(RESULT_PATH, payload)
    print(json.dumps({
        "status": status,
        "selected": {domain: {"id": row["config"]["id"], "metrics": {key: row["metrics"][key] for key in ("macro", "short_macro", "worst", "datasets", "suites")}} for domain, row in selected.items()},
        "aggregate": aggregate, "release_checks": release_checks,
        "result": str(RESULT_PATH.relative_to(ROOT)), "sha256": sha256_file(RESULT_PATH),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
