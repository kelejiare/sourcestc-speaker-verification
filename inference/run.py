#!/usr/bin/env python3
"""Fail-closed batch_centering entrypoint: unchanged calibrated_fusion plus unlabeled group centering."""

import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_DIR / "vendor"
if VENDOR_DIR.is_dir() and str(VENDOR_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(VENDOR_DIR))

import torch

from ensemble_inference import DurationAdaptiveEnsembleExtractor
from batch_submission_runtime import run_batch_centering_inference
from runtime_io import assert_online_acoustic_contract, find_input_root, find_output_dir
from joint_kd_inference import JointKDReweightExtractor
from source_adapted_inference import SourceAdaptedExtractor
from duration_metric_inference import DurationMetricExtractor
from plus_feature_fusion import PlusFeatureFusionExtractor
from calibrated_teacher_fusion import CalibratedTeacherFusionExtractor


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


def required_dir(env_name, names):
    candidates = []
    configured = os.environ.get(env_name)
    if configured:
        candidates.append(Path(configured))
    candidates.extend(PROJECT_DIR / name for name in names)
    for path in candidates:
        if path.is_dir() and (path / "config.json").is_file() and (path / "preprocessor_config.json").is_file():
            return path.resolve()
    raise FileNotFoundError("{} not found; checked {}".format(env_name, candidates))


def write_report(report):
    configured = os.environ.get("GAME823_RUNTIME_REPORT")
    if not configured:
        return
    destination = Path(configured).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    teacher = required("GAME823_W2VBERT2_MODEL_PATH", ("w2vbert2-multidomain-speaker-fp16.pt",))
    patch = required("GAME823_LORA_PATCH_PATH", ("lora_duration_patch.pt",))
    teacher_config = required_dir("GAME823_W2VBERT2_CONFIG_DIR", ("w2vbert2_config", "model/w2vbert2_config"))

    torch.manual_seed(0)
    torch.set_num_threads(max(1, int(os.environ.get("GAME823_TORCH_THREADS", "4"))))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device_name = os.environ.get("GAME823_DEVICE", "auto")
    maximum = float(os.environ.get("GAME823_MAX_CHUNK_SECONDS", "60.0"))
    hop = float(os.environ.get("GAME823_CHUNK_HOP_SECONDS", "30.0"))
    base = DurationAdaptiveEnsembleExtractor(pt, camp, device_name=device_name, max_chunk_seconds=maximum, chunk_hop_seconds=hop)
    assert_online_acoustic_contract(base)
    joint = JointKDReweightExtractor(pt, camp, lm, ptn, adapter, device_name=device_name, max_chunk_seconds=maximum, chunk_hop_seconds=hop, base_extractor=base)
    source_adapted = SourceAdaptedExtractor(joint, mobile)
    duration_metric = DurationMetricExtractor(source_adapted, short_metric, short_pt)
    plus_extractor = PlusFeatureFusionExtractor(duration_metric, plus)
    extractor = CalibratedTeacherFusionExtractor(plus_extractor, teacher, teacher_config, patch, device=str(joint.device))

    model_name = "SourceSTC speaker verification"
    print("model: " + model_name)
    print("input root: {}".format(input_root))
    print("output directory: {}".format(output_dir))
    print("device: {}".format(joint.device))
    print("output dimension: {}".format(extractor.output_dim))
    report = run_batch_centering_inference(extractor, input_root=input_root, output_dir=output_dir)
    report.update({
        "model": model_name,
        "hidden_score_guarantee": False,
        "teacher_audit": extractor.audit_summary(),
    })
    write_report(report)
    print("BATCH_CENTERING_GROUP_AUDIT " + json.dumps(report["groups"], sort_keys=True))
    print("result: {}".format(report["output_zip"]))


if __name__ == "__main__":
    main()
