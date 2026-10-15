#!/usr/bin/env python3
"""Reproduce teacher_fusion weight selection without using the official test set.

The script only combines already materialized, independently generated
embeddings.  The fixed release policy is selected on ``dev`` and
``nearfar_dev``; confirmation sets are reported post-hoc and cannot change the
choice.  A negative EER delta means improvement over the exact plus anchor.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_distillation_gates import compute_eer_percent
from tools.evaluate import data, pairs_for


LORA_ROOT = ROOT / "records/2026-09-13/126_lora_lora_duration_experts/evaluations"
OUT = ROOT / "records/2026-09-13/127_teacher_fusion_lora_duration_release/ALPHA_SWEEP.json"
POLICY = {"full3": 0.15, "short05": 0.25, "short10": 0.20}


def one_dataset(dataset, alpha_map):
    rows, _ = data(dataset)
    folder = LORA_ROOT / dataset
    anchor = np.load(folder / "plus_online_anchor/embeddings.npz", allow_pickle=False)
    candidate = np.load(folder / "lora_duration_s5201/embeddings.npz", allow_pickle=False)
    if not np.array_equal(anchor["ids"], candidate["ids"]):
        raise RuntimeError("embedding IDs differ for {}".format(dataset))
    cells = []
    for domain in sorted({row["domain"] for row in rows}):
        ix = [i for i, row in enumerate(rows) if row["domain"] == domain]
        pairs = pairs_for([rows[i] for i in ix])
        for duration in ("long3", "short05", "short10"):
            old = np.sum(anchor["long3"][ix][pairs[:, 0]] * anchor[duration][ix][pairs[:, 1]], axis=1)
            new = np.sum(candidate["long3"][ix][pairs[:, 0]] * candidate[duration][ix][pairs[:, 1]], axis=1)
            alpha = float(alpha_map["full3" if duration == "long3" else duration])
            old_eer = compute_eer_percent(pairs[:, 2], old)
            fused_eer = compute_eer_percent(pairs[:, 2], (1.0 - alpha) * old + alpha * new)
            cells.append({
                "domain": domain, "duration": duration, "alpha": alpha,
                "anchor_eer": old_eer, "fused_eer": fused_eer,
                "delta_eer": fused_eer - old_eer,
                "speakers": len({row["speaker"] for row in [rows[i] for i in ix]}),
                "trials": int(len(pairs)),
            })
    return cells


def main():
    datasets = {}
    for dataset in ("dev", "nearfar_dev", "confirm_child", "confirm_nearfar"):
        datasets[dataset] = one_dataset(dataset, POLICY)
    selection = []
    for dataset in ("dev", "nearfar_dev"):
        cells = datasets[dataset]
        selection.append({
            "dataset": dataset,
            "macro_delta_eer": float(np.mean([cell["delta_eer"] for cell in cells])),
            "worst_cell_delta_eer": float(np.max([cell["delta_eer"] for cell in cells])),
            "cells": cells,
        })
    posthoc = []
    for dataset in ("confirm_child", "confirm_nearfar"):
        cells = datasets[dataset]
        posthoc.append({
            "dataset": dataset,
            "macro_delta_eer": float(np.mean([cell["delta_eer"] for cell in cells])),
            "worst_cell_delta_eer": float(np.max([cell["delta_eer"] for cell in cells])),
            "cells": cells,
        })
    # Gate is intentionally modest: both predeclared proxy groups must improve
    # on average.  It does not claim a hidden-score prediction or guarantee.
    status = all(item["macro_delta_eer"] < -0.25 for item in selection)
    result = {
        "schema_version": 1,
        "status": "PASS" if status else "FAIL",
        "selection_protocol": "fixed before confirmation inspection; no official test used",
        "policy": POLICY,
        "selection": selection,
        "confirmation_posthoc": posthoc,
        "anchor": "plus exact 3520-D online release, score 90.96288",
        "candidate": "lora duration LoRA merged patch, 256-D native output",
        "hidden_score_guarantee": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
