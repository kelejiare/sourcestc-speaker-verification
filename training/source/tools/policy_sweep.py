#!/usr/bin/env python3
"""Audit the frozen source_policy source-aware policy on cached proxy embeddings."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_distillation_gates import compute_eer_percent
from tools.evaluate import data, pairs_for

EVAL_ROOT = ROOT / "records/2026-09-13/126_lora_lora_duration_experts/evaluations"
OUT = ROOT / "records/2026-09-13/136_source_policy_source_optimized/ALPHA_SWEEP.json"
SOURCE1_POLICY = {"full3": 0.45, "short05": 0.32, "short10": 0.25}
SOURCE2_POLICY = {"full3": 0.60, "short05": 0.30, "short10": 0.70}


def evaluate(dataset):
    rows, _ = data(dataset)
    folder = EVAL_ROOT / dataset
    anchor = np.load(folder / "plus_online_anchor/embeddings.npz", allow_pickle=False)
    candidate = np.load(folder / "lora_duration_s5201/embeddings.npz", allow_pickle=False)
    if not np.array_equal(anchor["ids"], candidate["ids"]):
        raise RuntimeError("embedding IDs differ for {}".format(dataset))
    cells = []
    for domain in sorted({row["domain"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["domain"] == domain]
        pairs = pairs_for([rows[i] for i in indices])
        policy = SOURCE2_POLICY if domain == "meeting" else SOURCE1_POLICY
        source = "data_scenery2_joint" if domain == "meeting" else "data_scenery1_mobile"
        for duration in ("long3", "short05", "short10"):
            old = np.sum(anchor["long3"][indices][pairs[:, 0]] * anchor[duration][indices][pairs[:, 1]], axis=1)
            new = np.sum(candidate["long3"][indices][pairs[:, 0]] * candidate[duration][indices][pairs[:, 1]], axis=1)
            alpha = policy["full3" if duration == "long3" else duration]
            old_eer = compute_eer_percent(pairs[:, 2], old)
            fused_eer = compute_eer_percent(pairs[:, 2], (1.0 - alpha) * old + alpha * new)
            cells.append({
                "domain": domain,
                "source": source,
                "duration": duration,
                "alpha": alpha,
                "anchor_eer": old_eer,
                "fused_eer": fused_eer,
                "delta_eer": fused_eer - old_eer,
                "speakers": len({rows[i]["speaker"] for i in indices}),
                "trials": int(len(pairs)),
            })
    return cells


def summary(dataset, cells):
    deltas = [cell["delta_eer"] for cell in cells]
    return {
        "dataset": dataset,
        "macro_delta_eer": float(np.mean(deltas)),
        "worst_cell_delta_eer": float(np.max(deltas)),
        "cells": cells,
    }


def main():
    datasets = ("dev", "nearfar_dev", "confirm_child", "confirm_nearfar")
    all_cells = {dataset: evaluate(dataset) for dataset in datasets}
    selection = [summary(dataset, all_cells[dataset]) for dataset in ("dev", "nearfar_dev")]
    confirmation = [summary(dataset, all_cells[dataset]) for dataset in ("confirm_child", "confirm_nearfar")]
    status = all(item["macro_delta_eer"] < -0.50 and item["worst_cell_delta_eer"] <= 0.0 for item in selection)
    result = {
        "schema_version": 1,
        "status": "PASS" if status else "FAIL",
        "selection_protocol": "source mapping and policy frozen for dev+nearfar gate; confirmation reported posthoc",
        "source1_policy": SOURCE1_POLICY,
        "source2_policy": SOURCE2_POLICY,
        "selection": selection,
        "confirmation_posthoc": confirmation,
        "all_known_worst_cell_delta_eer": float(max(cell["delta_eer"] for cells in all_cells.values() for cell in cells)),
        "anchor": "plus exact 3520-D online release, score 90.96288",
        "candidate": "lora duration LoRA merged patch, 256-D native output",
        "hidden_score_guarantee": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
