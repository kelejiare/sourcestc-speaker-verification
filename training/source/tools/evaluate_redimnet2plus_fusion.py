#!/usr/bin/env python3
"""Evaluate a deployable bilateral ReDimNet2+ fusion over the exact duration_metric graph.

The candidate embedding is a concatenation of two independently unit-normalized
blocks.  Consequently, the score computed here is exactly the cosine score that
the submitted embedding will produce; there is no one-sided or cell-wise oracle
shortcut in the release decision.
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

from tools.score_lodo_coverage import compute_eer_percent
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.nap_lodo import adapt, atomic_json, l2
from joint_kd_inference import (
    ENROLL_WEIGHTS,
    SHORT05_WEIGHTS,
    SHORT10_WEIGHTS,
    _load_joint_adapter,
)
from short_duration_metric import load_short_metric_adapter
from short_pt_inference import load_short_pt_adapter


JOINT_ADAPTER = ROOT / "model/joint_kd_adapter.npz"
SHORT_METRIC_ADAPTER = ROOT / "model/short_duration_metric_adapter.npz"
SHORT_PT_ADAPTER = ROOT / "model/short_duration_pt_adapter.npz"
BRANCH_ROOT = (
    ROOT
    / "records/2026-09-11/086_mobile_expert_redimnet2_mobile_acoustic_expert"
    / "plus_redimnet2plus_lmft"
)
OUTPUT_ROOT = ROOT / "records/2026-09-12/118_plus_redimnet2plus_fusion"
JOINT_DIM = 832
BRANCH_DIM = 192
DURATION_METRIC_ALPHA = 0.16


def shard_path(cache_dir, identifier):
    name = hashlib.sha256(identifier.encode("utf-8")).hexdigest() + ".npz"
    return Path(cache_dir) / name


def fuse(blocks, weights):
    output = np.concatenate(
        [math.sqrt(float(weight)) * l2(block) for block, weight in zip(blocks, weights)]
    ).astype(np.float32)
    return l2(output)


def joint_vector(record, duration, view, matrices):
    if duration == "full":
        blocks = [l2(record[name]["full"]) for name in ("acoustic", "lm", "ptn")]
        return fuse(blocks, ENROLL_WEIGHTS)
    cache_key = "0p5" if duration == "0p5" else "1p0"
    adapter_key = "short05" if duration == "0p5" else "short10"
    blocks = [
        l2(record["acoustic"][cache_key][view]),
        adapt(record["lm"][cache_key][view], matrices["lm_" + adapter_key]),
        adapt(record["ptn"][cache_key][view], matrices["ptn_" + adapter_key]),
    ]
    weights = SHORT05_WEIGHTS if duration == "0p5" else SHORT10_WEIGHTS
    return fuse(blocks, weights)


def adapt_pt(vector, weight, bias):
    output = np.array(l2(vector), dtype=np.float32, copy=True)
    scale = float(np.linalg.norm(output[:256]))
    if not math.isfinite(scale) or scale <= 1.0e-12:
        raise RuntimeError("invalid PT sub-block")
    unit = output[:256] / scale
    proposed = np.einsum("j,ij->i", unit, weight, optimize=False) + bias
    output[:256] = np.float32(scale) * l2(proposed)
    return l2(output)


def cml(vector, factors):
    base = l2(vector)
    experts = [l2(base + np.matmul(b, np.matmul(a, base))) for a, b in factors]
    return fuse([base] + experts, [1.0 - DURATION_METRIC_ALPHA] + [DURATION_METRIC_ALPHA / 3.0] * 3)


def load_branch_cache(suite_name, sources):
    cache_dir = BRANCH_ROOT / (suite_name + "_cache")
    output = {}
    for source in sources:
        identifier = str(source["utterance_stable_id"])
        path = shard_path(cache_dir, identifier)
        with np.load(str(path), allow_pickle=False) as payload:
            expected = {"schema_version", "utterance_id", "source_sha256", "full", "0p5", "1p0"}
            if set(payload.files) != expected or int(payload["schema_version"]) != 1:
                raise RuntimeError("branch cache schema mismatch: {}".format(path))
            if str(payload["utterance_id"]) != identifier:
                raise RuntimeError("branch cache identity mismatch: {}".format(path))
            if str(payload["source_sha256"]) != str(source["source_sha256"]):
                raise RuntimeError("branch cache source mismatch: {}".format(path))
            record = {
                key: np.asarray(payload[key], dtype=np.float32).copy()
                for key in ("full", "0p5", "1p0")
            }
        for key, shape in (("full", (BRANCH_DIM,)), ("0p5", (2, BRANCH_DIM)), ("1p0", (2, BRANCH_DIM))):
            if record[key].shape != shape or not np.isfinite(record[key]).all():
                raise RuntimeError("branch cache tensor mismatch: {} {}".format(path, key))
            norms = np.linalg.norm(record[key].reshape(-1, BRANCH_DIM), axis=1)
            if float(np.max(np.abs(norms - 1.0))) > 2.0e-4:
                raise RuntimeError("branch cache norm mismatch: {} {}".format(path, key))
        output[identifier] = record
    return output


def is_mobile_dataset(dataset):
    return str(dataset).startswith("childmandarin")


def build_rows(suite, sources, trials, joint_cache, branch_cache):
    matrices, _ = _load_joint_adapter(JOINT_ADAPTER)
    factors, _ = load_short_metric_adapter(SHORT_METRIC_ADAPTER)
    pt_weight, pt_bias, _ = load_short_pt_adapter(SHORT_PT_ADAPTER)
    joint_vectors = {}
    for identifier, record in joint_cache.items():
        joint_vectors[identifier] = {
            duration: [
                joint_vector(record, duration, view, matrices)
                for view in range(1 if duration == "full" else 2)
            ]
            for duration in ("full", "0p5", "1p0")
        }
    mobile_ids = set()
    for trial in trials:
        if is_mobile_dataset(trial["dataset"]):
            mobile_ids.add(trial["enrollment_utterance_id"])
            mobile_ids.add(trial["test_utterance_id"])
    duration_metric_vectors = {}
    for identifier, durations in joint_vectors.items():
        materialized = {}
        for duration, vectors in durations.items():
            materialized[duration] = []
            for vector in vectors:
                candidate = vector
                if duration == "0p5" and identifier in mobile_ids:
                    candidate = adapt_pt(candidate, pt_weight, pt_bias)
                materialized[duration].append(cml(candidate, factors))
        duration_metric_vectors[identifier] = materialized
    rows = []
    for trial in trials:
        enrollment_id = trial["enrollment_utterance_id"]
        test_id = trial["test_utterance_id"]
        enrollment_duration_metric = duration_metric_vectors[enrollment_id]["full"][0]
        enrollment_plus = l2(branch_cache[enrollment_id]["full"])
        for duration in ("full", "0p5", "1p0"):
            baseline_scores = []
            branch_scores = []
            for view, candidate_duration_metric in enumerate(duration_metric_vectors[test_id][duration]):
                candidate_plus = l2(
                    branch_cache[test_id][duration]
                    if duration == "full"
                    else branch_cache[test_id][duration][view]
                )
                baseline_scores.append(float(np.dot(enrollment_duration_metric, candidate_duration_metric)))
                branch_scores.append(float(np.dot(enrollment_plus, candidate_plus)))
            rows.append({
                "suite": suite["name"],
                "dataset": str(trial["dataset"]),
                "base_scenario": str(trial["base_scenario"]),
                "duration": duration,
                "label": int(trial["label"]),
                "duration_metric": float(np.mean(baseline_scores)),
                "plus": float(np.mean(branch_scores)),
            })
    return rows


def score_coefficients(alpha_full, alpha_query):
    return (
        math.sqrt((1.0 - alpha_full) * (1.0 - alpha_query)),
        math.sqrt(alpha_full * alpha_query),
    )


def summarize(rows, weights):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, selected in sorted(grouped.items()):
        duration = key[2]
        # Runtime enables the Plus block only for data_scenery2 (meeting).
        # Source-one/mobile embeddings are exact duration_metric with a zero auxiliary
        # block, so their score coefficients must also remain (1, 0) here.
        if is_mobile_dataset(key[0]):
            left, right = 1.0, 0.0
        else:
            left, right = score_coefficients(weights["full"], weights[duration])
        labels = [row["label"] for row in selected]
        baseline_scores = [row["duration_metric"] for row in selected]
        candidate_scores = [left * row["duration_metric"] + right * row["plus"] for row in selected]
        baseline = compute_eer_percent(labels, baseline_scores)
        candidate = compute_eer_percent(labels, candidate_scores)
        cells.append({
            "dataset": key[0],
            "base_scenario": key[1],
            "duration": duration,
            "trial_count": len(selected),
            "duration_metric_eer_percent": baseline,
            "candidate_eer_percent": candidate,
            "improvement_points_vs_duration_metric": baseline - candidate,
        })
    datasets = {}
    for dataset in sorted({cell["dataset"] for cell in cells}):
        selected = [cell for cell in cells if cell["dataset"] == dataset]
        datasets[dataset] = float(np.mean([cell["improvement_points_vs_duration_metric"] for cell in selected]))
    values = [cell["improvement_points_vs_duration_metric"] for cell in cells]
    short = [cell["improvement_points_vs_duration_metric"] for cell in cells if cell["duration"] != "full"]
    return {
        "cells": cells,
        "datasets": datasets,
        "macro": float(np.mean(values)),
        "short_macro": float(np.mean(short)),
        "worst": float(min(values)),
    }


def search(rows):
    full_grid = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
    query_grid = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
    results = []
    for full in full_grid:
        for short05 in query_grid:
            for short10 in query_grid:
                weights = {"full": full, "0p5": short05, "1p0": short10}
                metrics = summarize(rows, weights)
                checks = {
                    "macro_positive": metrics["macro"] > 0.0,
                    "short_positive": metrics["short_macro"] > 0.0,
                    "worst_ge_minus_0p2": metrics["worst"] >= -0.2,
                    "each_dataset_nonnegative": all(value >= 0.0 for value in metrics["datasets"].values()),
                }
                results.append({
                    "weights": weights,
                    "metrics": metrics,
                    "checks": checks,
                    "eligible": all(checks.values()),
                })
    results.sort(
        key=lambda row: (
            row["eligible"],
            row["metrics"]["macro"],
            row["metrics"]["short_macro"],
            row["metrics"]["worst"],
            -sum(row["weights"].values()),
        ),
        reverse=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("new_holdouts", "alimeeting_test"), required=True)
    parser.add_argument("--mode", choices=("search", "fixed", "audit"), required=True)
    parser.add_argument("--alpha-full", type=float)
    parser.add_argument("--alpha-0p5", type=float)
    parser.add_argument("--alpha-1p0", type=float)
    args = parser.parse_args()
    suite = SUITES[0] if args.suite == "new_holdouts" else SUITES[1]
    sources, trials, joint_cache = load_suite(suite)
    branch_cache = load_branch_cache(suite["name"], sources)
    rows = build_rows(suite, sources, trials, joint_cache, branch_cache)
    if args.mode == "audit":
        profiles = {
            "frozen_development": {"full": 0.03, "0p5": 0.20, "1p0": 0.0},
            "small_independent": {"full": 0.03, "0p5": 0.01, "1p0": 0.05},
            "consensus_low": {"full": 0.03, "0p5": 0.05, "1p0": 0.01},
            "consensus_mid": {"full": 0.03, "0p5": 0.10, "1p0": 0.01},
        }
        payload = {
            "schema_version": 1,
            "artifact_kind": "plus_redimnet2plus_route_exact_weight_audit",
            "status": "DIAGNOSTIC_ONLY",
            "suite": suite["name"],
            "online_anchor": {"release": "duration_metric-short_pt-pt-then-short_metric-cml", "score": 90.9618},
            "hidden_score_guarantee": False,
            "profiles": {
                name: {"weights": weights, "metrics": summarize(rows, weights)}
                for name, weights in profiles.items()
            },
        }
    elif args.mode == "search":
        results = search(rows)
        payload = {
            "schema_version": 1,
            "artifact_kind": "plus_redimnet2plus_exact_bilateral_search",
            "status": "DEVELOPMENT_SEARCH_ONLY",
            "suite": suite["name"],
            "online_anchor": {"release": "duration_metric-short_pt-pt-then-short_metric-cml", "score": 90.9618},
            "hidden_score_guarantee": False,
            "grid_size": len(results),
            "eligible_count": sum(row["eligible"] for row in results),
            "top_candidates": results[:30],
        }
    else:
        values = (args.alpha_full, args.alpha_0p5, args.alpha_1p0)
        if any(value is None or value < 0.0 or value >= 1.0 for value in values):
            raise ValueError("fixed mode requires three alphas in [0,1)")
        weights = {"full": values[0], "0p5": values[1], "1p0": values[2]}
        metrics = summarize(rows, weights)
        payload = {
            "schema_version": 1,
            "artifact_kind": "plus_redimnet2plus_exact_bilateral_confirmation",
            "status": "INDEPENDENT_CONFIRMATION",
            "suite": suite["name"],
            "online_anchor": {"release": "duration_metric-short_pt-pt-then-short_metric-cml", "score": 90.9618},
            "hidden_score_guarantee": False,
            "weights": weights,
            "metrics": metrics,
            "release_checks": {
                "macro_positive": metrics["macro"] > 0.0,
                "short_positive": metrics["short_macro"] > 0.0,
                "worst_ge_minus_0p": metrics["worst"] >= -0.2,
            },
        }
    output = OUTPUT_ROOT / (args.suite + "_" + args.mode.upper() + ".json")
    atomic_json(output, payload)
    if args.mode == "search":
        summary = payload["top_candidates"][:5]
    elif args.mode == "audit":
        summary = payload["profiles"]
    else:
        summary = [payload["metrics"]]
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "summary": summary,
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
