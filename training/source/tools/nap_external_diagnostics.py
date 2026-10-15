#!/usr/bin/env python3
"""Read-only external diagnostics for the frozen joint duration-NAP candidate.

The joint hyperparameters are read from the completed LODO search and are never
changed here.  Nuisance bases are fitted on the six development corpora, then
the fixed candidate is compared directly with the released distillation embedding
contract on already-consumed HI-MIA, ChildMandarin and AliMeeting caches.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.score_lodo_coverage import compute_eer_percent, speaker_bootstrap
from tools.nap_lodo import (
    ADAPTER_PATH,
    MODEL_DIRS,
    DISTILLATION_ENROLL,
    DISTILLATION_SHORT,
    adapt,
    atomic_json,
    atomic_jsonl,
    candidate_score,
    fit_duration_nuisance,
    l2,
    load_redim,
    load_distillation_adapters,
    project,
    sha256_file,
)


SEARCH_RESULT = ROOT / "records/2026-09-05/033_joint_duration_nap_lodo/LODO_RESULT.json"
OUTPUT_DIR = ROOT / "records/2026-09-05/033_joint_duration_nap_lodo/external_diagnostics"
TRACE_PATH = OUTPUT_DIR / "SCORE_TRACE.jsonl"
RESULT_PATH = OUTPUT_DIR / "EXTERNAL_DIAGNOSTICS.json"

SUITES = (
    {
        "name": "new_holdouts",
        "sources": ROOT / "records/2026-09-04/027_distillation_new_holdout_cache/FROZEN_SOURCES.jsonl",
        "trials": ROOT / "records/2026-09-04/028_distillation_new_holdout_score/FROZEN_TRIALS.jsonl",
        "cache": ROOT / "records/2026-09-04/027_distillation_new_holdout_cache/cache",
        "nested": False,
    },
    {
        "name": "alimeeting_test",
        "sources": ROOT / "records/2026-09-04/030_distillation_alimeeting_test_cache/FROZEN_SOURCES.jsonl",
        "trials": ROOT / "records/2026-09-04/031_distillation_alimeeting_test_final_score/FROZEN_TRIALS.jsonl",
        "cache": ROOT / "records/2026-09-04/030_distillation_alimeeting_test_cache/cache",
        "nested": False,
    },
    {
        "name": "himia_h1",
        "sources": ROOT / "records/2026-09-04/014_distillation_cn_redimnet2/artifacts/h1_trial_freeze_v2/H1_SELECTED_SOURCES.jsonl",
        "trials": ROOT / "records/2026-09-04/014_distillation_cn_redimnet2/artifacts/h1_trial_freeze_v2/H1_TRIALS.jsonl",
        "cache": ROOT / "records/2026-09-04/014_distillation_cn_redimnet2/artifacts/h1_one_shot/cache",
        "nested": True,
    },
)

CACHE_KEYS = {
    "schema_version", "utterance_stable_id", "speaker_stable_id",
    "distance_stable_id", "source_sha256", "source_samples",
    "acoustic_full", "acoustic_0p5", "acoustic_1p0", "lm_full", "lm_0p5", "lm_1p0",
    "ptn_full", "ptn_0p5", "ptn_1p0",
}


def cache_path(cache_dir, utterance):
    name = hashlib.sha256(utterance.encode("utf-8")).hexdigest() + ".npz"
    return Path(cache_dir) / name


def scalar(payload, key):
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError("{} is not scalar".format(key))
    return value.item()


def validate_cache(path, record):
    with np.load(str(path), allow_pickle=False) as payload:
        if set(payload.files) != CACHE_KEYS or int(scalar(payload, "schema_version")) != 1:
            raise ValueError("cache schema mismatch: {}".format(path))
        for key in ("utterance_stable_id", "speaker_stable_id", "distance_stable_id", "source_sha256"):
            if str(scalar(payload, key)) != str(record[key]):
                raise ValueError("cache binding mismatch: {} {}".format(path, key))
        output = {}
        for model, dimension in (("acoustic", 448), ("lm", 192), ("ptn", 192)):
            output[model] = {}
            for duration, shape in (("full", (dimension,)), ("0p5", (2, dimension)), ("1p0", (2, dimension))):
                value = np.asarray(payload[model + "_" + duration], dtype=np.float32).copy()
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("cache embedding mismatch: {}".format(path))
                norms = np.linalg.norm(value.reshape(-1, dimension), axis=1)
                if float(np.max(np.abs(norms - 1.0))) > 2.0e-4:
                    raise ValueError("cache norm mismatch: {}".format(path))
                output[model][duration] = value
        output["source_samples"] = int(scalar(payload, "source_samples"))
    return output


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("empty JSONL: {}".format(path))
    return rows


def distillation_score(enrollment, test, duration):
    test_weights = DISTILLATION_ENROLL if duration == "full" else DISTILLATION_SHORT
    value = sum(
        math.sqrt(float(left * right)) * float(np.dot(a, b))
        for a, b, left, right in zip(enrollment, test, DISTILLATION_ENROLL, test_weights)
    )
    if not math.isfinite(value):
        raise ValueError("non-finite distillation score")
    return float(value)


def normalize_trial(raw, nested):
    if nested:
        enrollment_id = str(raw["enrollment"]["utterance_stable_id"])
        test_id = str(raw["test"]["utterance_stable_id"])
        dataset = "hi_mia_internal_h1"
        base_scenario = "himia_h1"
    else:
        enrollment_id = str(raw["enrollment_utterance_id"])
        test_id = str(raw["test_utterance_id"])
        dataset = str(raw["dataset"])
        base_scenario = str(raw["base_scenario"])
    return {
        "trial_id": str(raw["trial_id"]),
        "dataset": dataset,
        "base_scenario": base_scenario,
        "enrollment_utterance_id": enrollment_id,
        "test_utterance_id": test_id,
        "enrollment_speaker_id": str(raw["enrollment_speaker_id"]),
        "test_speaker_id": str(raw["test_speaker_id"]),
        "label": int(raw["label"]),
    }


def load_suite(suite):
    sources = read_jsonl(suite["sources"])
    records = {str(row["utterance_stable_id"]): row for row in sources}
    if len(records) != len(sources):
        raise ValueError("duplicate source id in {}".format(suite["name"]))
    cache = {}
    for identifier, record in records.items():
        cache[identifier] = validate_cache(cache_path(suite["cache"], identifier), record)
    trials = [normalize_trial(row, suite["nested"]) for row in read_jsonl(suite["trials"])]
    for row in trials:
        if row["enrollment_utterance_id"] not in cache or row["test_utterance_id"] not in cache:
            raise ValueError("trial/cache binding failure in {}".format(suite["name"]))
    return sources, trials, cache


def score_suite(suite, trials, cache, adapters, bases, config):
    rank = int(config["rank"])
    strength = float(config["strength"])
    enroll_extra = float(config["enroll_extra_total"])
    rows = []
    for raw in trials:
        enrollment_cache = cache[raw["enrollment_utterance_id"]]
        test_cache = cache[raw["test_utterance_id"]]
        enrollment = [l2(enrollment_cache[name]["full"]) for name in ("acoustic", "lm", "ptn")]
        enrollment_nap = {
            duration: project(enrollment[1], bases[duration], rank, strength)
            for duration in ("short05", "short10")
        }
        for duration, cache_key, adapter_key in (
            ("full", "full", None),
            ("0p5", "0p5", "short05"),
            ("1p0", "1p0", "short10"),
        ):
            base_scores = []
            nap05_scores = []
            nap10_scores = []
            views = 1 if duration == "full" else 2
            for view in range(views):
                if duration == "full":
                    test = [l2(test_cache[name]["full"]) for name in ("acoustic", "lm", "ptn")]
                    lm05 = project(test[1], bases["short05"], rank, strength)
                    lm10 = project(test[1], bases["short10"], rank, strength)
                else:
                    test = [
                        l2(test_cache["acoustic"][cache_key][view]),
                        adapt(test_cache["lm"][cache_key][view], adapters["lm"][adapter_key]),
                        adapt(test_cache["ptn"][cache_key][view], adapters["ptn"][adapter_key]),
                    ]
                    lm05 = project(test[1], bases["short05"], rank, strength)
                    lm10 = project(test[1], bases["short10"], rank, strength)
                base_scores.append(distillation_score(enrollment, test, duration))
                nap05_scores.append(float(np.dot(enrollment_nap["short05"], lm05)))
                nap10_scores.append(float(np.dot(enrollment_nap["short10"], lm10)))
            base = float(np.mean(base_scores))
            nap05 = float(np.mean(nap05_scores))
            nap10 = float(np.mean(nap10_scores))
            short_extra = (
                float(config["short05_extra"]) if duration == "0p5"
                else float(config["short10_extra"])
            )
            proposed = candidate_score(
                base, nap05, nap10, duration, enroll_extra, short_extra
            )
            rows.append({
                "suite": suite["name"],
                "fold": raw["dataset"],
                "dataset": raw["dataset"],
                "base_scenario": raw["base_scenario"],
                "duration": duration,
                "scenario": raw["base_scenario"] + "_" + duration,
                "trial_id": raw["trial_id"],
                "label": raw["label"],
                "enrollment_speaker_id": raw["enrollment_speaker_id"],
                "test_speaker_id": raw["test_speaker_id"],
                "enrollment_utterance_id": raw["enrollment_utterance_id"],
                "test_utterance_id": raw["test_utterance_id"],
                "baseline_score": base,
                "candidate_score": proposed,
                "nap05_score": nap05,
                "nap10_score": nap10,
            })
    return rows


def summarize(rows, bootstrap_repeats):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["dataset"], row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, selected in sorted(grouped.items()):
        labels = [row["label"] for row in selected]
        baseline = compute_eer_percent(labels, [row["baseline_score"] for row in selected])
        candidate = compute_eer_percent(labels, [row["candidate_score"] for row in selected])
        cells.append({
            "suite": key[0], "dataset": key[1], "base_scenario": key[2],
            "duration": key[3], "trial_count": len(selected),
            "distillation_eer_percent": baseline, "candidate_eer_percent": candidate,
            "improvement_points_vs_distillation": baseline - candidate,
        })
    dataset_metrics = {}
    bootstraps = {}
    for index, dataset in enumerate(sorted({row["dataset"] for row in rows})):
        selected = [cell for cell in cells if cell["dataset"] == dataset]
        short = [cell for cell in selected if cell["duration"] != "full"]
        dataset_metrics[dataset] = {
            "cell_count": len(selected),
            "macro_improvement_points_vs_distillation": float(np.mean([
                cell["improvement_points_vs_distillation"] for cell in selected
            ])),
            "short_macro_improvement_points_vs_distillation": float(np.mean([
                cell["improvement_points_vs_distillation"] for cell in short
            ])),
            "worst_cell_improvement_points_vs_distillation": float(min(
                cell["improvement_points_vs_distillation"] for cell in selected
            )),
        }
        bootstraps[dataset] = speaker_bootstrap(
            rows, dataset, bootstrap_repeats, 20260905 + index, True
        )
    summary = {
        "cells": cells,
        "datasets": dataset_metrics,
        "macro_improvement_points_vs_distillation": float(np.mean([
            cell["improvement_points_vs_distillation"] for cell in cells
        ])),
        "short_macro_improvement_points_vs_distillation": float(np.mean([
            cell["improvement_points_vs_distillation"] for cell in cells
            if cell["duration"] != "full"
        ])),
        "worst_cell_improvement_points_vs_distillation": float(min(
            cell["improvement_points_vs_distillation"] for cell in cells
        )),
        "speaker_bootstrap": bootstraps,
    }
    checks = {
        "external_macro_positive": summary["macro_improvement_points_vs_distillation"] > 0.0,
        "external_short_macro_positive": summary["short_macro_improvement_points_vs_distillation"] > 0.0,
        "no_cell_regresses_over_0p2": summary["worst_cell_improvement_points_vs_distillation"] >= -0.2,
        "each_dataset_macro_nonnegative": all(
            value["macro_improvement_points_vs_distillation"] >= 0.0
            for value in dataset_metrics.values()
        ),
    }
    summary["gate_checks"] = checks
    summary["external_diagnostic_pass"] = bool(all(checks.values()))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_repeats < 1000:
        raise ValueError("at least 1000 bootstrap repeats are required")

    search = json.loads(SEARCH_RESULT.read_text(encoding="utf-8"))
    if search.get("status") != "LODO_GATE_PASS" or not search.get("selected"):
        raise ValueError("joint LODO result is not a selected gate-pass candidate")
    config = dict(search["selected"]["config"])
    adapters = load_distillation_adapters()
    redim = load_redim(MODEL_DIRS["lm"])
    bases = {}
    basis_audits = {}
    for duration in ("short05", "short10"):
        basis, audit = fit_duration_nuisance(
            redim, adapters["lm"][duration], duration, held_out=None
        )
        bases[duration] = basis
        basis_audits[duration] = audit

    trace = []
    input_receipts = []
    for suite in SUITES:
        print("VALIDATE {}".format(suite["name"]), flush=True)
        sources, trials, cache = load_suite(suite)
        print("SCORE {} trials={}".format(suite["name"], len(trials)), flush=True)
        trace.extend(score_suite(suite, trials, cache, adapters, bases, config))
        input_receipts.append({
            "suite": suite["name"], "source_count": len(sources),
            "trial_count": len(trials), "cache_count": len(cache),
            "sources_sha256": sha256_file(suite["sources"]),
            "trials_sha256": sha256_file(suite["trials"]),
        })
    atomic_jsonl(TRACE_PATH, trace)
    summary = summarize(trace, args.bootstrap_repeats)
    result = {
        "schema_version": 1,
        "artifact_kind": "joint_duration_nap_fixed_candidate_external_diagnostics",
        "status": "EXTERNAL_DIAGNOSTIC_PASS" if summary["external_diagnostic_pass"]
                  else "EXTERNAL_DIAGNOSTIC_FAIL_DO_NOT_PACKAGE",
        "selection_result": {
            "path": str(SEARCH_RESULT.relative_to(ROOT)),
            "sha256": sha256_file(SEARCH_RESULT),
        },
        "distillation_adapter": {
            "path": str(ADAPTER_PATH.relative_to(ROOT)),
            "sha256": sha256_file(ADAPTER_PATH),
        },
        "config_frozen_before_external_diagnostics": config,
        "basis_fit": basis_audits,
        "inputs": input_receipts,
        "score_trace": {
            "path": str(TRACE_PATH.relative_to(ROOT)),
            "rows": len(trace), "sha256": sha256_file(TRACE_PATH),
        },
        "summary": summary,
        "limitations": [
            "These are previously consumed public development caches, not fresh holdouts.",
            "They are used only as a rejection diagnostic after LODO selection.",
            "A pass cannot guarantee a hidden leaderboard score of 91.",
        ],
    }
    atomic_json(RESULT_PATH, result)
    print(json.dumps({
        "status": result["status"], "summary": summary,
        "result": str(RESULT_PATH), "sha256": sha256_file(RESULT_PATH),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
