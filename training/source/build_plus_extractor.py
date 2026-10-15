#!/usr/bin/env python3
"""Fail-closed competition entrypoint for the plus candidate."""

import json
import os
import sys
from pathlib import Path

import torch

from ensemble_inference import DurationAdaptiveEnsembleExtractor
from strict_submission_runtime import run_strict_inference
from runtime_io import assert_online_acoustic_contract, find_input_root, find_output_dir
from joint_kd_inference import JointKDReweightExtractor
from source_adapted_inference import SourceAdaptedExtractor
from duration_metric_inference import DurationMetricExtractor
from plus_feature_fusion import (
    PlusFeatureFusionExtractor,
)


PROJECT_DIR = Path(__file__).resolve().parent


def required(env_name, names):
    candidates = []
    configured = os.environ.get(env_name)
    if configured:
        candidates.append(Path(configured))
    for name in names:
        candidates.extend((PROJECT_DIR / name, PROJECT_DIR / "model" / name))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("{} not found; checked {}".format(env_name, candidates))


def write_report(report):
    configured = os.environ.get("GAME823_RUNTIME_REPORT")
    if not configured:
        return
    destination = Path(configured).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(destination))


def main():
    input_root = find_input_root()
    output_dir = find_output_dir()
    pt = required("GAME823_PT_MODEL_PATH", ("model_pt.pt",))
    camp = required("GAME823_CAMPPLUS_MODEL_PATH", ("campplus_cn_en_common.pt",))
    lm = required("GAME823_REDIM_LM_MODEL_PATH", ("redimnet2-b6-vb2-vox2-cnc2-lm.pt",))
    ptn = required("GAME823_REDIM_PTN_MODEL_PATH", ("redimnet2-b6-vb2-vox2-cnc2-ptn.pt",))
    adapter = required("GAME823_JOINT_ADAPTER_PATH", ("joint_kd_adapter.npz",))
    mobile = required("GAME823_MOBILE_ADAPTER_PATH", ("mobile_kd_adapter.npz",))
    short_metric = required("GAME823_SHORT_METRIC_ADAPTER_PATH", ("short_duration_metric_adapter.npz",))
    short_pt = required("GAME823_SHORT_PT_ADAPTER_PATH", ("short_duration_pt_adapter.npz",))
    plus = required("GAME823_REDIMNET2PLUS_MODEL_PATH", ("redimnet2plus-b6-lmft-robust.pt",))

    torch.manual_seed(0)
    torch.set_num_threads(max(1, int(os.environ.get("GAME823_TORCH_THREADS", "4"))))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device_name = os.environ.get("GAME823_DEVICE", "auto")
    maximum = float(os.environ.get("GAME823_MAX_CHUNK_SECONDS", "60.0"))
    hop = float(os.environ.get("GAME823_CHUNK_HOP_SECONDS", "30.0"))
    base = DurationAdaptiveEnsembleExtractor(
        pt, camp, device_name=device_name,
        max_chunk_seconds=maximum, chunk_hop_seconds=hop,
    )
    assert_online_acoustic_contract(base)
    joint = JointKDReweightExtractor(
        pt, camp, lm, ptn, adapter, device_name=device_name,
        max_chunk_seconds=maximum, chunk_hop_seconds=hop,
        base_extractor=base,
    )
    source_adapted = SourceAdaptedExtractor(joint, mobile)
    duration_metric = DurationMetricExtractor(source_adapted, short_metric, short_pt)
    extractor = PlusFeatureFusionExtractor(duration_metric, plus)
    print("release: plus-duration_metric-anchored-redimnet2plus")
    print("input root: {}".format(input_root))
    print("output directory: {}".format(output_dir))
    print("device: {}".format(joint.device))
    print("output dimension: {}".format(extractor.output_dim))
    report = run_strict_inference(
        extractor, input_root=input_root, output_dir=output_dir
    )
    report.update({
        "release": "plus-duration_metric-anchored-redimnet2plus",
        "online_anchor": "duration_metric-short_pt-pt-then-short_metric-cml",
        "online_anchor_score": 90.9618,
        "requested_target_score": 92.0,
        "hidden_score_guarantee": False,
    })
    write_report(report)
    print("result: {}".format(report["output_zip"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("plus failed: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        raise
