#!/usr/bin/env python3
"""Refine scenery1 weights after freezing the passing mobile adapter.

The search baseline is byte-identical joint.  Full, 0.5 s, and 1.0 s weights are
selected jointly under per-cell, per-dataset, and per-suite non-regression.
"""

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nap_external_diagnostics import SUITES, load_suite
from tools.distillation_reweight_validation import coverage_components, external_components
from tools.nap_lodo import atomic_json, sha256_file
from tools.scenery_conditional_adapter_search import (
    SCENERY1_DATASETS, baseline_lookup, current_adapters,
)
from tools.scenery_conditional_reweight_search import search_domain


OUTPUT_DIR = ROOT / "records/2026-09-06/062_mobile_weight_refinement"
RESULT_PATH = OUTPUT_DIR / "SEARCH_RESULT.json"
MOBILE_PATH = ROOT / "records/2026-09-06/061_mobile_mobile_adapter_mix/MOBILE_ADAPTER.npz"


def load_mobile():
    expected = {
        "schema_version", "config_id", "mix05", "mix10", "seed_ids",
        "lm_short05", "lm_short10", "ptn_short05", "ptn_short10",
    }
    with np.load(str(MOBILE_PATH), allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise RuntimeError("mobile adapter key schema mismatch")
        if int(np.asarray(payload["schema_version"])) != 1:
            raise RuntimeError("mobile adapter schema mismatch")
        if str(np.asarray(payload["config_id"]).item()) != "mobile_d42_c42":
            raise RuntimeError("mobile adapter config mismatch")
        if float(np.asarray(payload["mix05"])) != 1.0 or float(np.asarray(payload["mix10"])) != 0.0:
            raise RuntimeError("mobile adapter mix mismatch")
        output = {
            model: {
                duration: np.ascontiguousarray(payload[model + "_" + duration], dtype=np.float32)
                for duration in ("short05", "short10")
            }
            for model in ("lm", "ptn")
        }
    return output


def main():
    current, current_receipt = current_adapters()
    baseline = baseline_lookup(current)
    candidate = load_mobile()
    rows = coverage_components(candidate)
    sources, trials, cache = load_suite(SUITES[0])
    rows.extend(external_components(SUITES[0], sources, trials, cache, candidate))
    rows = [row for row in rows if row["dataset"] in SCENERY1_DATASETS]
    print("SEARCH mobile rows={}".format(len(rows)), flush=True)
    results = search_domain(rows, "data_scenery1")
    selected = results[0]
    release_checks = {
        "eligible": bool(selected["eligible"]),
        "macro_ge_0p8": selected["metrics"]["macro"] >= 0.8,
        "short_macro_ge_0p8": selected["metrics"]["short_macro"] >= 0.8,
        "worst_ge_minus_0p2": selected["metrics"]["worst"] >= -0.2,
    }
    status = (
        "MOBILE_WEIGHT_GATE_PASS_REQUIRES_BOOTSTRAP"
        if all(release_checks.values()) else
        "MOBILE_WEIGHT_GATE_FAILED_DO_NOT_PACKAGE"
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "mobile_mobile_adapter_joint_weight_refinement",
        "status": status,
        "baseline": "joint_joint_kd_reweight",
        "joint_adapter_receipt": current_receipt,
        "mobile_adapter": {
            "path": str(MOBILE_PATH.relative_to(ROOT)),
            "bytes": MOBILE_PATH.stat().st_size,
            "sha256": sha256_file(MOBILE_PATH),
        },
        "selected": selected,
        "release_checks": release_checks,
        "top_candidates": results[:20],
        "limitations": [
            "The source-directory mapping follows the official starter-kit order.",
            "Proxy improvement cannot guarantee the hidden leaderboard score.",
        ],
    }
    atomic_json(RESULT_PATH, payload)
    print(json.dumps({
        "status": status,
        "weights": selected["weights"],
        "metrics": {key: selected["metrics"][key] for key in (
            "macro", "short_macro", "worst", "datasets", "suites")},
        "checks": release_checks,
        "result": str(RESULT_PATH.relative_to(ROOT)),
        "sha256": sha256_file(RESULT_PATH),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
