#!/usr/bin/env python3
"""Evaluate the official Mandarin ERes2NetV2-w24 as a distillation auxiliary branch.

The cache is resumable and bound to the frozen external source manifests.
Development uses only AliMeeting Eval and ChildMandarin Dev.  AliMeeting Test
is not touched unless ``--suite alimeeting_test`` is explicitly requested
after a development candidate has been frozen.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_features import FbankConfig, extract_kaldi_fbank_torch, load_audio
from eres2netv2_model import ERes2NetV2
from tools.build_embedding_cache import deterministic_crop_start
from tools.score_lodo_coverage import compute_eer_percent
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.nap_lodo import adapt, atomic_json, load_distillation_adapters, sha256_file


OUTPUT_DIR = ROOT / "records/2026-09-05/049_w24_external_support"
MODEL_PATH = ROOT / "model/pretrained_eres2netv2w24s4ep4.ckpt"
MODEL_SHA256 = "4f5a0bf73c61eb41b174e1bb54e7ee3c83233892be8e0af1f187024e8e581a35"
BRANCH_DIM = 192
BRANCH_CMN = True
ENROLL_WEIGHTS = (0.86, 0.133, 0.007)
SHORT_WEIGHTS = (0.20, 0.76, 0.04)


def l2(value):
    value = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not np.isfinite(value).all() or norm <= 1.0e-12:
        raise ValueError("invalid embedding")
    return np.ascontiguousarray(value / norm, dtype=np.float32)


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def shard_path(cache_dir, identifier):
    return cache_dir / (hashlib.sha256(identifier.encode("utf-8")).hexdigest() + ".npz")


def load_model():
    if sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise ValueError("w24 checkpoint digest mismatch")
    payload = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
    model = ERes2NetV2(
        feat_dim=80, embedding_size=192, baseWidth=24, scale=4, expansion=4
    )
    incompatible = model.load_state_dict(payload, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("w24 strict load failed")
    model.eval()
    return model, len(payload), sum(parameter.numel() for parameter in model.parameters())


def embed(model, waveforms, config):
    features = [extract_kaldi_fbank_torch(waveform, config)[0] for waveform in waveforms]
    if len({feature.shape[0] for feature in features}) != 1:
        return np.stack([embed(model, [waveform], config)[0] for waveform in waveforms])
    with torch.inference_mode():
        values = model(torch.from_numpy(np.stack(features)).float()).float().cpu().numpy()
    return np.stack([l2(value) for value in values])


def validate_shard(path, source):
    with np.load(str(path), allow_pickle=False) as payload:
        expected = {"schema_version", "utterance_id", "source_sha256", "full", "0p5", "1p0"}
        if set(payload.files) != expected or int(payload["schema_version"]) != 1:
            raise ValueError("w24 shard schema mismatch")
        if str(payload["utterance_id"]) != str(source["utterance_stable_id"]):
            raise ValueError("w24 shard identity mismatch")
        if str(payload["source_sha256"]) != str(source["source_sha256"]):
            raise ValueError("w24 shard source mismatch")
        output = {}
        for key, shape in (("full", (BRANCH_DIM,)), ("0p5", (2, BRANCH_DIM)), ("1p0", (2, BRANCH_DIM))):
            value = np.asarray(payload[key], dtype=np.float32).copy()
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("w24 shard embedding mismatch")
            norms = np.linalg.norm(value.reshape(-1, BRANCH_DIM), axis=1)
            if float(np.max(np.abs(norms - 1.0))) > 2.0e-4:
                raise ValueError("w24 shard norm mismatch")
            output[key] = value
    return output


def build_cache(suite, sources, threads):
    cache_dir = OUTPUT_DIR / (suite["name"] + "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(threads)))
    model, tensors, parameters = load_model()
    config = FbankConfig(cmn=BRANCH_CMN, dither=0.0)
    pending = [source for source in sources if not shard_path(
        cache_dir, str(source["utterance_stable_id"])
    ).is_file()]
    started = time.perf_counter()
    for offset, source in enumerate(pending, 1):
        identifier = str(source["utterance_stable_id"])
        waveform = load_audio(ROOT / source["source_path"], 16000, remove_dc=False)
        crops = {}
        for key, samples in (("0p5", 8000), ("1p0", 16000)):
            if waveform.size < samples:
                raise ValueError("source shorter than official crop")
            values = []
            for view in (0, 1):
                start = deterministic_crop_start(waveform.size, samples, view, 2)
                values.append(np.ascontiguousarray(waveform[start:start + samples], dtype=np.float32))
            crops[key] = embed(model, values, config)
        arrays = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "utterance_id": np.asarray(identifier),
            "source_sha256": np.asarray(str(source["source_sha256"])),
            "full": embed(model, [waveform], config)[0],
            "0p5": crops["0p5"], "1p0": crops["1p0"],
        }
        path = shard_path(cache_dir, identifier)
        atomic_npz(path, arrays); validate_shard(path, source)
        if offset == 1 or offset % 25 == 0 or offset == len(pending):
            print("{} cache {}/{} new; reused={}; elapsed={:.1f}s".format(
                suite["name"], offset, len(pending), len(sources) - len(pending),
                time.perf_counter() - started,
            ), flush=True)
    cache = {
        str(source["utterance_stable_id"]): validate_shard(
            shard_path(cache_dir, str(source["utterance_stable_id"])), source
        ) for source in sources
    }
    return cache, {
        "cache_dir": str(cache_dir.relative_to(ROOT)), "source_count": len(cache),
        "new_count": len(pending), "tensor_count": tensors,
        "parameter_count": parameters, "model_sha256": MODEL_SHA256,
        "elapsed_seconds": time.perf_counter() - started,
    }


def distillation_score(enrollment, test, duration):
    right = ENROLL_WEIGHTS if duration == "full" else SHORT_WEIGHTS
    return float(sum(
        math.sqrt(float(a_weight * b_weight)) * float(np.dot(a, b))
        for a, b, a_weight, b_weight in zip(enrollment, test, ENROLL_WEIGHTS, right)
    ))


def component_rows(trials, distillation_cache, w24_cache, adapters):
    rows = []
    for trial in trials:
        enrollment_id = trial["enrollment_utterance_id"]
        test_id = trial["test_utterance_id"]
        enrollment = [l2(distillation_cache[enrollment_id][name]["full"]) for name in ("acoustic", "lm", "ptn")]
        enrollment_w24 = l2(w24_cache[enrollment_id]["full"])
        for duration, cache_key, adapter_key in (
            ("full", "full", None), ("0p5", "0p5", "short05"),
            ("1p0", "1p0", "short10"),
        ):
            distillation_scores = []; w24_scores = []
            for view in range(1 if duration == "full" else 2):
                if duration == "full":
                    test = [l2(distillation_cache[test_id][name]["full"]) for name in ("acoustic", "lm", "ptn")]
                    test_w24 = l2(w24_cache[test_id]["full"])
                else:
                    test = [
                        l2(distillation_cache[test_id]["acoustic"][cache_key][view]),
                        adapt(distillation_cache[test_id]["lm"][cache_key][view], adapters["lm"][adapter_key]),
                        adapt(distillation_cache[test_id]["ptn"][cache_key][view], adapters["ptn"][adapter_key]),
                    ]
                    test_w24 = l2(w24_cache[test_id][cache_key][view])
                distillation_scores.append(distillation_score(enrollment, test, duration))
                w24_scores.append(float(np.dot(enrollment_w24, test_w24)))
            rows.append({
                "dataset": trial["dataset"], "base_scenario": trial["base_scenario"],
                "duration": duration, "label": int(trial["label"]),
                "distillation": float(np.mean(distillation_scores)), "w24": float(np.mean(w24_scores)),
            })
    return rows


def summarize(rows, alpha05, alpha10, alpha_full):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, selected in sorted(grouped.items()):
        alpha = alpha_full if key[2] == "full" else alpha05 if key[2] == "0p5" else alpha10
        labels = [row["label"] for row in selected]
        baseline = compute_eer_percent(labels, [row["distillation"] for row in selected])
        candidate = compute_eer_percent(labels, [
            (1.0 - alpha) * row["distillation"] + alpha * row["w24"] for row in selected
        ])
        cells.append({
            "dataset": key[0], "base_scenario": key[1], "duration": key[2],
            "distillation_eer_percent": baseline, "candidate_eer_percent": candidate,
            "improvement_points_vs_distillation": baseline - candidate,
        })
    datasets = {}
    for dataset in sorted(set(cell["dataset"] for cell in cells)):
        selected = [cell for cell in cells if cell["dataset"] == dataset]
        datasets[dataset] = float(np.mean([cell["improvement_points_vs_distillation"] for cell in selected]))
    return {
        "cells": cells, "datasets": datasets,
        "macro": float(np.mean([cell["improvement_points_vs_distillation"] for cell in cells])),
        "short_macro": float(np.mean([cell["improvement_points_vs_distillation"] for cell in cells if cell["duration"] != "full"])),
        "worst": float(min(cell["improvement_points_vs_distillation"] for cell in cells)),
    }


def search(rows):
    results = []
    for full in (0.005, 0.01, 0.02, 0.04, 0.06, 0.10):
        for short05 in (0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            for short10 in (0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
                metrics = summarize(rows, short05, short10, full)
                checks = {
                    "macro_positive": metrics["macro"] > 0.0,
                    "short_positive": metrics["short_macro"] > 0.0,
                    "worst_ge_minus_0p2": metrics["worst"] >= -0.2,
                    "each_dataset_nonnegative": all(value >= 0.0 for value in metrics["datasets"].values()),
                }
                results.append({
                    "weights": {"full": full, "0p5": short05, "1p0": short10},
                    "metrics": metrics, "checks": checks, "eligible": all(checks.values()),
                })
    results.sort(key=lambda row: (
        row["eligible"], row["metrics"]["macro"], row["metrics"]["short_macro"],
        row["metrics"]["worst"],
    ), reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("new_holdouts", "alimeeting_test"), default="new_holdouts")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    suite = SUITES[0] if args.suite == "new_holdouts" else SUITES[1]
    sources, trials, distillation_cache = load_suite(suite)
    w24_cache, cache_receipt = build_cache(suite, sources, args.threads)
    rows = component_rows(trials, distillation_cache, w24_cache, load_distillation_adapters())
    results = search(rows)
    output = OUTPUT_DIR / (suite["name"] + "_RESULT.json")
    payload = {
        "schema_version": 1, "artifact_kind": "joint_w24_auxiliary_pilot",
        "suite": suite["name"],
        "status": "DEVELOPMENT_HAS_CANDIDATE" if results[0]["eligible"] else "GATE_FAILED_DO_NOT_PACKAGE",
        "cache_receipt": cache_receipt,
        "grid_size": len(results),
        "eligible_count": sum(row["eligible"] for row in results),
        "top_candidates": results[:20],
        "limitations": ["Public proxy evidence cannot guarantee the hidden score."],
    }
    atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "eligible_count": payload["eligible_count"],
        "best": results[0], "output": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
