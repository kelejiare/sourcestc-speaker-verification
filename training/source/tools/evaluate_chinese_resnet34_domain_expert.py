#!/usr/bin/env python3
"""Evaluate a trained acoustic ResNet34 expert against the online joint anchor.

The model is selected only on speaker-disjoint ChildMandarin development and
AliMeeting evaluation trials.  The search is scenery- and duration-specific,
matching what the official runtime can determine from the input path.  This is
an offline risk gate; it does not claim a hidden leaderboard score.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import w24_external_support as cache_io
from tools import mobile_joint_joint_alimeeting_resnet34_search as joint_joint
from tools.score_lodo_coverage import compute_eer_percent
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.nap_lodo import atomic_json, sha256_file
from wespeaker_resnet34_model import ResNet34


OUTPUT_ROOT = ROOT / "records/2026-09-11/085_short_pt_chinese_acoustic_domain_expert"
GRID = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
        0.40, 0.50, 0.65, 0.80, 1.0)


def load_candidate(path):
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    model = ResNet34(feat_dim=80, embed_dim=256)
    incompatible = model.load_state_dict(payload, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("short_pt trained checkpoint strict-load mismatch")
    model.eval()
    return model, len(payload), sum(p.numel() for p in model.parameters())


def summarize(rows, dataset, duration, alpha):
    selected = [
        row for row in rows
        if row["dataset"] == dataset and row["duration"] == duration
    ]
    if not selected:
        raise RuntimeError("empty evaluation cell")
    labels = [row["label"] for row in selected]
    anchor_scores = [row["joint"] for row in selected]
    expert_scores = [row["auxiliary"] for row in selected]
    fused_scores = [
        (1.0 - alpha) * anchor + alpha * expert
        for anchor, expert in zip(anchor_scores, expert_scores)
    ]
    anchor = compute_eer_percent(labels, anchor_scores)
    expert = compute_eer_percent(labels, expert_scores)
    candidate = compute_eer_percent(labels, fused_scores)
    return {
        "dataset": dataset,
        "duration": duration,
        "alpha": float(alpha),
        "trial_count": len(selected),
        "joint_eer_percent": float(anchor),
        "expert_eer_percent": float(expert),
        "candidate_eer_percent": float(candidate),
        "improvement_points_vs_joint": float(anchor - candidate),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    output_dir = OUTPUT_ROOT / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_sha256 = sha256_file(checkpoint)

    cache_io.OUTPUT_DIR = output_dir
    cache_io.MODEL_PATH = checkpoint
    cache_io.MODEL_SHA256 = model_sha256
    cache_io.BRANCH_DIM = 256
    cache_io.BRANCH_CMN = True
    cache_io.load_model = lambda: load_candidate(checkpoint)

    suite = SUITES[0]
    sources, trials, joint_cache = load_suite(suite)
    _expert_cache, cache_receipt = cache_io.build_cache(
        suite, sources, args.threads
    )
    joint_joint.BRANCH_CACHE = output_dir / "new_holdouts_cache"
    joint_joint.branch_io.BRANCH_DIM = 256
    joint_joint.branch_io.BRANCH_CMN = True
    adapters, adapter_receipt = joint_joint.joint_adapters()
    rows = joint_joint.component_rows(sources, trials, joint_cache, adapters)

    cells = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for duration in ("full", "0p5", "1p0"):
            candidates = [summarize(rows, dataset, duration, alpha) for alpha in GRID]
            candidates.sort(key=lambda row: (
                row["improvement_points_vs_joint"], -row["alpha"]
            ), reverse=True)
            cells.append({
                "dataset": dataset,
                "duration": duration,
                "best": candidates[0],
                "grid": candidates,
            })

    best_rows = [cell["best"] for cell in cells]
    macro = float(np.mean([
        row["improvement_points_vs_joint"] for row in best_rows
    ]))
    conservative = [
        row for row in best_rows
        if row["alpha"] <= 0.20 and row["improvement_points_vs_joint"] >= 0.0
    ]
    payload = {
        "schema_version": 1,
        "artifact_kind": "short_pt_chinese_acoustic_domain_expert_external_gate",
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": model_sha256,
            "bytes": checkpoint.stat().st_size,
        },
        "suite": suite["name"],
        "cache_receipt": cache_receipt,
        "adapter_receipt": adapter_receipt,
        "cells": cells,
        "oracle_cellwise_macro_improvement_points": macro,
        "conservative_nonnegative_cell_count": len(conservative),
        "status": "EXPERIMENT_ONLY_NOT_RELEASED",
        "limitations": [
            "The per-cell oracle is diagnostic and is not a release policy.",
            "Public speaker-disjoint proxies cannot guarantee hidden score.",
            "A release needs a frozen scenery-level policy and further holdouts.",
        ],
    }
    output = output_dir / "EXTERNAL_GATE.json"
    atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "checkpoint_sha256": model_sha256,
        "oracle_cellwise_macro_improvement_points": macro,
        "best": best_rows,
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256_file(output),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
