#!/usr/bin/env python3
"""Formal frozen-protocol gate for batch_centering unlabeled teacher-block centering."""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_protocol import pairs
from tools.evaluate_distillation_gates import compute_eer_percent
from tools.evaluate import DURATIONS, data, pairs_for
from tools.evaluate_protocol_gate import bootstrap_correct, write_json


LORA = ROOT / "records/2026-09-13/126_lora_lora_duration_experts"
OUTPUT = ROOT / "records/2026-09-16/156_batch_centering_transductive_centering/evaluations/B7_GATE_REPORT.json"
POLICIES = {
    "source1": {"long3": 0.45, "short05": 0.32, "short10": 0.25},
    "source2": {"long3": 0.60, "short05": 0.1090909090909091, "short10": 0.70},
}
STRENGTHS = {"source1": 1.05, "source2": 1.00}


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-8)


def centered(values, strength):
    values = unit(values)
    return unit(values - np.float32(strength) * values.mean(axis=0, keepdims=True))


def scores(anchor, teacher, trial, duration, policy):
    left, right, labels = trial.T
    anchor_score = np.sum(
        anchor["long3"][left].astype(np.float64) * anchor[duration][right], axis=1
    )
    teacher_score = np.sum(
        teacher["long3"][left].astype(np.float64) * teacher[duration][right], axis=1
    )
    a = POLICIES[policy]
    value = (
        math.sqrt((1.0 - a["long3"]) * (1.0 - a[duration])) * anchor_score
        + math.sqrt(a["long3"] * a[duration]) * teacher_score
    )
    return labels, value


def protocol(name, rows, anchor, original, candidate, policy, trial, weights=None):
    if weights is None:
        weights = np.ones(len(trial), dtype=np.float64)
    cells, before, after = [], [], []
    for duration, _ in DURATIONS:
        labels, old_score = scores(anchor, original, trial, duration, policy)
        _, new_score = scores(anchor, candidate, trial, duration, policy)
        old_eer = compute_eer_percent(labels, old_score, weights)
        new_eer = compute_eer_percent(labels, new_score, weights)
        cells.append({
            "duration": duration,
            "calibrated_fusion_eer": old_eer,
            "candidate_eer": new_eer,
            "delta": new_eer - old_eer,
        })
        before.append(old_score)
        after.append(new_score)
    used = sorted(set(trial[:, 0].tolist() + trial[:, 1].tolist()))
    mapping = {old: new for new, old in enumerate(used)}
    local_trial = np.asarray([
        (mapping[int(left)], mapping[int(right)], int(label))
        for left, right, label in trial
    ], dtype=np.int64)
    boot = bootstrap_correct(
        [rows[index] for index in used], local_trial, before, after,
        np.asarray(weights, dtype=np.float64), repeats=800, seed=2026091671,
    )
    return {
        "name": name,
        "policy": policy,
        "files": len(rows),
        "trials": len(trial),
        "positive_trials": int(trial[:, 2].sum()),
        "cells": cells,
        "macro_delta": float(np.mean([cell["delta"] for cell in cells])),
        "speaker_cluster_bootstrap": boot,
    }


def load_vectors(dataset, policy):
    rows, _ = data(dataset)
    folder = LORA / "evaluations" / dataset
    anchor = dict(np.load(folder / "plus_online_anchor/embeddings.npz", allow_pickle=False))
    teacher = dict(np.load(folder / "lora_duration_s5201/embeddings.npz", allow_pickle=False))
    expected = [row["id"] for row in rows]
    if list(anchor["ids"]) != expected or list(teacher["ids"]) != expected:
        raise RuntimeError("cache order mismatch: " + dataset)
    original = {key: unit(teacher[key]) for key, _ in DURATIONS}
    candidate = {
        key: centered(original[key], STRENGTHS[policy]) for key, _ in DURATIONS
    }
    anchor = {key: unit(anchor[key]) for key, _ in DURATIONS}
    return rows, anchor, original, candidate


def subset(rows, arrays, indices):
    return [rows[index] for index in indices], {
        key: value[indices] for key, value in arrays.items()
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protocols = []

    rows, anchor, original, candidate = load_vectors("dev", "source1")
    for domain in ("child", "youth"):
        indices = np.asarray([i for i, row in enumerate(rows) if row["domain"] == domain])
        local_rows, local_anchor = subset(rows, anchor, indices)
        _, local_original = subset(rows, original, indices)
        _, local_candidate = subset(rows, candidate, indices)
        trial = pairs(local_rows, {domain}, {domain})
        protocols.append(protocol(
            "selection_{}_within".format(domain), local_rows, local_anchor,
            local_original, local_candidate, "source1", trial,
        ))
    trial = pairs(rows, {"child"}, {"child", "youth"})
    right_domain = np.asarray([rows[int(index)]["domain"] for index in trial[:, 1]])
    weights = np.ones(len(trial), dtype=np.float64)
    for domain, mass in (("child", 7.0), ("youth", 6.0)):
        mask = right_domain == domain
        weights[mask] = mass / max(1, int(mask.sum()))
    protocols.append(protocol(
        "selection_cross_age_7_6", rows, anchor, original, candidate,
        "source1", trial, weights,
    ))

    for dataset, policy, phase in (
        ("confirm_child", "source1", "confirmation"),
        ("nearfar_dev", "source2", "selection"),
        ("confirm_nearfar", "source2", "confirmation"),
    ):
        rows, anchor, original, candidate = load_vectors(dataset, policy)
        protocols.append(protocol(
            "{}_{}".format(phase, dataset), rows, anchor, original, candidate,
            policy, pairs_for(rows),
        ))

    cells = [
        {"protocol": item["name"], **cell}
        for item in protocols for cell in item["cells"]
    ]
    macro = float(np.mean([cell["delta"] for cell in cells]))
    worst = float(max(cell["delta"] for cell in cells))
    selection = float(np.mean([
        cell["delta"] for cell in cells if cell["protocol"].startswith("selection")
    ]))
    confirmation = float(np.mean([
        cell["delta"] for cell in cells if cell["protocol"].startswith("confirmation")
    ]))
    ci_passes = sum(
        item["speaker_cluster_bootstrap"]["ci95"][1] < 0.0 for item in protocols
    )
    gate = {
        "macro_delta_at_most_minus_0p75": macro <= -0.75,
        "worst_cell_regression_at_most_0p2": worst <= 0.2,
        "selection_macro_improves": selection < 0.0,
        "confirmation_macro_improves": confirmation < 0.0,
        "at_least_half_protocol_ci_upper_below_zero": ci_passes >= 3,
    }
    authorized = all(gate.values())
    result = {
        "schema": "batch_centering_unlabeled_group_teacher_centering_gate_v1",
        "status": "PASS_STRICT_GATE" if authorized else "STRICT_CONFIDENCE_GATE_FAIL",
        "online_anchor": {"release": "calibrated_fusion", "score": 91.71295},
        "configuration": {
            "grouping": "official_source_plus_existing_duration_policy",
            "minimum_runtime_group_size": 8,
            "source1_strength": STRENGTHS["source1"],
            "source2_strength": STRENGTHS["source2"],
            "teacher_block_only": True,
            "input_labels_or_trials_used": False,
        },
        "protocols": protocols,
        "cell_count": len(cells),
        "macro_delta": macro,
        "worst_cell_delta": worst,
        "selection_macro_delta": selection,
        "confirmation_macro_delta": confirmation,
        "protocol_ci_upper_below_zero_count": ci_passes,
        "gate": {**gate, "authorized_for_packaging": authorized},
        "mechanical_score_projection": 91.71295 - macro,
        "limitations": [
            "The score projection mechanically subtracts proxy macro EER delta and is not a calibrated hidden-score forecast.",
            "The hidden evaluation source composition and batch group sizes are unknown.",
            "The runtime transform is unlabeled and uses no trial or speaker metadata, but it is transductive batch adaptation and must be disclosed.",
            "No public proxy result can guarantee a hidden leaderboard score.",
        ],
        "hidden_score_guarantee": False,
    }
    write_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
