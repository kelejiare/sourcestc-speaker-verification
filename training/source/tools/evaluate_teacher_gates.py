#!/usr/bin/env python3
"""Evaluate a teacher_gate patch with the *actual* calibrated_fusion embedding geometry.

This is deliberately different from a teacher-only proxy: the score is
reconstructed from the frozen 3520-D plus anchor and the 256-D w2v block using
the exact source/duration coefficients that calibrated_fusion submits.  It is a public,
offline gate only and has no access to the competition test set.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_distillation_gates import compute_eer_percent
from tools.relation_distillation import paired_trials, sha256_file
from tools.evaluate import DURATIONS, data, pairs_for
from native_w2vbert2 import NativeFeatures, load_native_teacher
from lora_duration_model import apply_merged_patch
from source_duration_policy import SOURCE1_POLICY
from calibrated_teacher_fusion import SOURCE2_POLICY


RECORD = ROOT / "records/2026-09-14/148_teacher_gate_rawcrop_relation"
LORA_RECORD = ROOT / "records/2026-09-13/126_lora_lora_duration_experts"
BASE_PATCH = LORA_RECORD / "lora_duration_s5201/merged_patch.pt"
TEACHER = ROOT / "model/w2vbert2-multidomain-speaker-fp16.pt"
CONFIG = ROOT / "records/2026-09-12/120_teacher_w2vbert2_teacher_screen/base_config/config.json"
NAME_TO_DURATION = {"long3": "full3", "short05": "short05", "short10": "short10"}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract_patch(patch, waves, device):
    """Return long/0.5s/1s vectors using exactly the teacher_fusion centre crops."""
    patch = Path(patch)
    model = load_native_teacher(TEACHER, CONFIG)
    apply_merged_patch(model, patch, sha256_file(patch))
    model.to(device).eval()
    features = NativeFeatures()
    values = {}
    started = time.time()
    with torch.inference_mode():
        for key, samples in DURATIONS:
            start = (48000 - samples) // 2
            pieces = []
            for offset in range(0, len(waves), 4):
                batch = np.ascontiguousarray(waves[offset:offset + 4, start:start + samples], dtype=np.float32)
                pieces.append(model(features(batch).to(device)).cpu().numpy().astype(np.float32))
            value = np.concatenate(pieces, axis=0)
            value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1.0e-8)
            if value.shape != (len(waves), 256) or not np.isfinite(value).all():
                raise RuntimeError("teacher_gate candidate embedding contract failed for " + key)
            values[key] = value
            print(json.dumps({"phase": "teacher_gate_eval_extract", "duration": key,
                              "files": len(waves), "seconds": time.time() - started}), flush=True)
    return values


def route_policy(domain):
    # Public manifests label the real conference/near-far evaluation as
    # meeting.  All other held-out rows follow the source-one route.
    return SOURCE2_POLICY if str(domain) == "meeting" else SOURCE1_POLICY


def composed_scores(rows, anchor, student, pairs, duration):
    left, right, labels = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    test_key = NAME_TO_DURATION[duration]
    enrollment_alpha = np.asarray([float(route_policy(rows[i]["domain"])["full3"]) for i in left], dtype=np.float64)
    test_alpha = np.asarray([float(route_policy(rows[i]["domain"])[test_key]) for i in right], dtype=np.float64)
    anchor_score = np.sum(anchor["long3"][left] * anchor[duration][right], axis=1, dtype=np.float64)
    student_score = np.sum(student["long3"][left] * student[duration][right], axis=1, dtype=np.float64)
    score = (np.sqrt((1.0 - enrollment_alpha) * (1.0 - test_alpha)) * anchor_score +
             np.sqrt(enrollment_alpha * test_alpha) * student_score)
    return labels.astype(np.int64), score.astype(np.float64)


def bootstrap_delta(rows, pairs, anchor, old, new, repeats=300, seed=6404):
    """Cluster bootstrap over speakers, reporting candidate-minus-calibrated_fusion EER."""
    speakers = sorted({str(row["speaker"]) for row in rows})
    if len(speakers) < 10:
        return {"speakers": len(speakers), "ci95": None, "reason": "too_few_speakers"}
    sid = np.asarray([speakers.index(str(row["speaker"])) for row in rows], dtype=np.int64)
    old_parts, new_parts, label_parts = [], [], []
    for duration, _ in DURATIONS:
        labels, old_score = composed_scores(rows, anchor, old, pairs, duration)
        _, new_score = composed_scores(rows, anchor, new, pairs, duration)
        old_parts.append(old_score); new_parts.append(new_score); label_parts.append(labels)
    old_score = np.concatenate(old_parts); new_score = np.concatenate(new_parts); labels = np.concatenate(label_parts)
    pair_left = np.concatenate([pairs[:, 0] for _ in DURATIONS])
    pair_right = np.concatenate([pairs[:, 1] for _ in DURATIONS])
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(repeats):
        count = np.bincount(rng.randint(len(speakers), size=len(speakers)), minlength=len(speakers))
        weight = np.where(pair_left == pair_right, count[sid[pair_left]],
                          count[sid[pair_left]] * count[sid[pair_right]]).astype(np.float64)
        if weight[labels == 0].sum() and weight[labels == 1].sum():
            values.append(compute_eer_percent(labels, new_score, weight) -
                          compute_eer_percent(labels, old_score, weight))
    return {"speakers": len(speakers), "valid_replicates": len(values),
            "ci95": np.quantile(values, [.025, .975]).tolist()}


def evaluate(patch, datasets, device, tag, active_domains=None,
             active_duration_domains=None):
    patch = Path(patch)
    if not patch.is_file():
        raise FileNotFoundError(patch)
    old_sha = sha256_file(BASE_PATCH)
    candidate_sha = sha256_file(patch)
    active_domains = None if active_domains is None else tuple(sorted(set(active_domains)))
    if active_duration_domains is not None:
        active_duration_domains = {
            str(key): tuple(sorted(set(value)))
            for key, value in active_duration_domains.items()
        }
        if set(active_duration_domains) - {key for key, _ in DURATIONS}:
            raise ValueError("unknown duration in active_duration_domains")
    result = {"schema": "teacher_gate_exact_calibrated_fusion_geometry_gate_v1", "candidate_patch": str(patch),
              "candidate_sha256": candidate_sha, "base_patch": str(BASE_PATCH),
              "base_patch_sha256": old_sha, "candidate_active_domains": active_domains,
              "candidate_active_duration_domains": active_duration_domains,
              "datasets": {}, "hidden_score_guarantee": False}
    all_cells = []
    for dataset in datasets:
        rows, waves = data(dataset)
        anchor_file = LORA_RECORD / "evaluations" / dataset / "plus_online_anchor/embeddings.npz"
        if not anchor_file.is_file():
            raise FileNotFoundError("missing frozen plus anchor cache: " + str(anchor_file))
        anchor = np.load(anchor_file, allow_pickle=False)
        if list(anchor["ids"]) != [row["id"] for row in rows]:
            raise RuntimeError("frozen anchor cache row order changed: " + dataset)
        old = extract_patch(BASE_PATCH, waves, device)
        new = extract_patch(patch, waves, device)
        if active_domains is not None or active_duration_domains is not None:
            # Route-scoped candidates must be evaluated as they will deploy:
            # untouched domains retain the exact parent teacher embeddings.
            for key, _ in DURATIONS:
                active = (active_duration_domains.get(key, active_domains)
                          if active_duration_domains is not None else active_domains)
                if active is None:
                    continue
                disabled = np.asarray([str(row["domain"]) not in active for row in rows], dtype=bool)
                new[key][disabled] = old[key][disabled]
        cells = []
        for domain in sorted({str(row["domain"]) for row in rows}):
            indices = np.asarray([i for i, row in enumerate(rows) if str(row["domain"]) == domain], dtype=np.int64)
            subset = [rows[i] for i in indices]
            pairs = pairs_for(subset)
            local_anchor = {key: anchor[key][indices] for key, _ in DURATIONS}
            local_old = {key: old[key][indices] for key, _ in DURATIONS}
            local_new = {key: new[key][indices] for key, _ in DURATIONS}
            for duration, _ in DURATIONS:
                labels, old_scores = composed_scores(subset, local_anchor, local_old, pairs, duration)
                _, new_scores = composed_scores(subset, local_anchor, local_new, pairs, duration)
                cell = {"dataset": dataset, "domain": domain, "duration": duration,
                        "trials": int(labels.size), "calibrated_fusion_eer_percent": compute_eer_percent(labels, old_scores),
                        "candidate_eer_percent": compute_eer_percent(labels, new_scores)}
                cell["delta_eer_points"] = cell["candidate_eer_percent"] - cell["calibrated_fusion_eer_percent"]
                cells.append(cell); all_cells.append(cell)
            result["datasets"].setdefault(dataset, {})[domain] = {
                "cells": cells[-3:], "speaker_bootstrap": bootstrap_delta(subset, pairs, local_anchor, local_old, local_new)
            }
    result["macro_delta_eer_points"] = float(np.mean([cell["delta_eer_points"] for cell in all_cells]))
    short = [cell["delta_eer_points"] for cell in all_cells if cell["duration"] != "long3"]
    result["short_macro_delta_eer_points"] = float(np.mean(short)) if short else None
    long_regressions = [cell for cell in all_cells if cell["duration"] == "long3" and cell["delta_eer_points"] > .2]
    result["release_gate"] = {
        "candidate_beats_calibrated_fusion_macro": bool(result["macro_delta_eer_points"] < 0.0),
        "no_long_cell_regression_gt_0p2": not long_regressions,
        "long_regressions": long_regressions,
        "authorized_for_submission": False,
        "reason": "Offline public gates alone cannot authorize an online score claim."
    }
    output = RECORD / "evaluations" / tag / "GATE_REPORT.json"
    write_json(output, result)
    print(json.dumps({"gate_report": str(output), "macro_delta": result["macro_delta_eer_points"],
                      "short_macro_delta": result["short_macro_delta_eer_points"],
                      "no_long_regression": not long_regressions}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True)
    parser.add_argument("--datasets", nargs="+", choices=("dev", "nearfar_dev", "confirm_child", "confirm_nearfar"), default=("dev", "nearfar_dev"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.patch, args.datasets, args.device, args.tag)


if __name__ == "__main__":
    main()
