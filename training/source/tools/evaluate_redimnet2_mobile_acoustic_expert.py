#!/usr/bin/env python3
"""External gate for a fine-tuned ReDimNet2 mobile acoustic expert."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redimnet2.redimnet2 import ReDimNet2Wrap
from speaker_inference import l2_normalize
from tools import w24_external_support as cache_io
from tools import mobile_joint_joint_alimeeting_resnet34_search as joint_joint
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.nap_lodo import atomic_json, sha256_file
from tools.evaluate_chinese_resnet34_domain_expert import summarize


OUTPUT_ROOT = ROOT / "records/2026-09-11/086_mobile_expert_redimnet2_mobile_acoustic_expert"
GRID = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
        0.40, 0.50, 0.65, 0.80, 1.0)


def load_candidate(path):
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"model_config", "state_dict"}:
        raise RuntimeError("mobile_expert checkpoint payload schema mismatch")
    if int(payload["model_config"].get("embed_dim", -1)) != 192:
        raise RuntimeError("mobile_expert checkpoint embedding dimension mismatch")
    model = ReDimNet2Wrap(**payload["model_config"])
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("mobile_expert checkpoint strict-load mismatch")
    model.eval()
    return model, len(payload["state_dict"]), sum(p.numel() for p in model.parameters())


def embed_waveforms(model, waveforms, _config):
    lengths = {int(np.asarray(value).size) for value in waveforms}
    if len(lengths) != 1:
        return np.stack([
            embed_waveforms(model, [value], _config)[0] for value in waveforms
        ])
    tensor = torch.from_numpy(np.stack([
        np.ascontiguousarray(value, dtype=np.float32) for value in waveforms
    ])).float()
    with torch.inference_mode():
        output = model(tensor).float().cpu().numpy()
    return np.stack([
        l2_normalize(value, "mobile_expert acoustic expert") for value in output
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--suite", choices=("new_holdouts", "alimeeting_test"), default="new_holdouts")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    output_dir = OUTPUT_ROOT / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_io.OUTPUT_DIR = output_dir
    cache_io.MODEL_PATH = checkpoint
    cache_io.MODEL_SHA256 = sha256_file(checkpoint)
    cache_io.BRANCH_DIM = 192
    cache_io.BRANCH_CMN = False
    cache_io.load_model = lambda: load_candidate(checkpoint)
    cache_io.embed = embed_waveforms

    suite = SUITES[0] if args.suite == "new_holdouts" else SUITES[1]
    sources, trials, joint_cache = load_suite(suite)
    _cache, cache_receipt = cache_io.build_cache(suite, sources, args.threads)
    joint_joint.BRANCH_CACHE = output_dir / (suite["name"] + "_cache")
    joint_joint.branch_io.BRANCH_DIM = 192
    adapters, adapter_receipt = joint_joint.joint_adapters()
    rows = joint_joint.component_rows(sources, trials, joint_cache, adapters)

    best = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for duration in ("full", "0p5", "1p0"):
            candidates = [summarize(rows, dataset, duration, alpha) for alpha in GRID]
            candidates.sort(key=lambda row: (
                row["improvement_points_vs_joint"], -row["alpha"]
            ), reverse=True)
            best.append(candidates[0])
    macro = float(np.mean([
        row["improvement_points_vs_joint"] for row in best
    ]))
    payload = {
        "schema_version": 1,
        "artifact_kind": "mobile_expert_redimnet2_mobile_acoustic_expert_external_gate",
        "status": "EXPERIMENT_ONLY_NOT_RELEASED",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "suite": suite["name"],
        "cache_receipt": cache_receipt,
        "adapter_receipt": adapter_receipt,
        "best": best,
        "oracle_cellwise_macro_improvement_points": macro,
        "limitations": [
            "Cellwise best weights are an oracle upper bound, not a release policy.",
            "Public proxy results cannot guarantee the hidden leaderboard score.",
        ],
    }
    result_path = output_dir / "EXTERNAL_GATE.json"
    atomic_json(result_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "best": best,
        "oracle_cellwise_macro_improvement_points": macro,
        "output": str(result_path.relative_to(ROOT)),
        "output_sha256": sha256_file(result_path),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
