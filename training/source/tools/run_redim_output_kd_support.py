#!/usr/bin/env python3
"""Pilot a short-to-full metric adapter on the frozen dual ReDimNet2 branch.

Only the already cached 3D-Speaker train speakers are used for fitting.  The
coverage test speakers are disjoint, while AISHELL-4 and ChildMandarin are
both speaker- and dataset-disjoint.  This is a cheap go/no-go pilot before
building any additional acoustic caches; it cannot package or submit.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.score_lodo_coverage import compute_eer_percent


BASE = ROOT / "records/2026-09-04/014_distillation_cn_redimnet2/artifacts"
MODEL_DIRS = {
    "lm": BASE / "stage1_b6_lm/redim_cache",
    "ptn": BASE / "stage1_b6_ptn/redim_cache",
}
ACOUSTIC_CACHE = ROOT / (
    "records/2026-09-02/013_distillation_domain_kd/artifacts/"
    "combined_cache_pre_cv/cache_v2.npz"
)
TRIALS = ROOT / (
    "records/2026-09-02/013_distillation_domain_kd/artifacts/lodo_sources_pre_cv/"
    "coverage_validation/secondary_coverage_trials.jsonl"
)
OUTPUT = ROOT / (
    "records/2026-09-04/017_distillation_redim_output_kd/PILOT_RESULT.json"
)
HOLDS = ("3dspeaker", "aishell4", "childmandarin")
MODEL_NAMES = ("lm", "ptn")
DURATIONS = ("short05", "short10")
ENROLL_WEIGHTS = (0.90, 0.09, 0.01)
SHORT_WEIGHTS = (0.20, 0.72, 0.08)

CONFIGS = (
    {"id": "r8_conservative", "rank": 8, "updates": 180, "lr": 0.003,
     "margin": 0.15, "scale": 24.0, "ce": 1.0, "proto": 2.0, "identity": 4.0},
    {"id": "r16_balanced", "rank": 16, "updates": 220, "lr": 0.003,
     "margin": 0.15, "scale": 24.0, "ce": 1.0, "proto": 1.0, "identity": 2.0},
    {"id": "r32_discriminative", "rank": 32, "updates": 260, "lr": 0.002,
     "margin": 0.20, "scale": 28.0, "ce": 1.5, "proto": 1.0, "identity": 1.0},
)
MIXES = (0.25, 0.50, 0.75, 1.0)


def l2(value):
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm <= 1.0e-12):
        raise ValueError("non-finite or zero-norm embedding")
    return value / norm


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_acoustic():
    with np.load(str(ACOUSTIC_CACHE), allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    by_utterance = defaultdict(list)
    metadata = {}
    for index, utterance in enumerate(arrays["utterance_id"]):
        utterance = str(utterance)
        by_utterance[utterance].append(index)
        metadata[utterance] = {
            "dataset": str(arrays["dataset"][index]),
            "split": str(arrays["split"][index]),
            "speaker": str(arrays["speaker_id"][index]),
        }
    for utterance, indices in by_utterance.items():
        indices.sort(
            key=lambda item: int(str(arrays["embedding_id"][item]).rsplit("view", 1)[-1])
        )
    return arrays, dict(by_utterance), metadata


def load_redim(directory):
    output = {}
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(str(path), allow_pickle=False) as source:
            utterance = str(source["utterance_id"])
            if utterance in output:
                raise ValueError("duplicate ReDim cache utterance")
            output[utterance] = {
                "speaker": str(source["speaker_id"]),
                "dataset": str(source["dataset"]),
                "full": l2(source["full"]),
                "short05": l2(source["short05"]),
                "short10": l2(source["short10"]),
            }
    if len(output) != 1544:
        raise ValueError("ReDim cache count changed")
    return output


def load_trials():
    with TRIALS.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 4691:
        raise ValueError("coverage trial count changed")
    return rows


def training_arrays(redim, metadata, duration):
    utterances = sorted(
        utterance for utterance, row in redim.items()
        if row["dataset"] == "3dspeaker"
        and metadata[utterance]["split"] == "train"
    )
    speakers = sorted(set(redim[utterance]["speaker"] for utterance in utterances))
    speaker_to_class = {speaker: index for index, speaker in enumerate(speakers)}
    prototype_rows = defaultdict(list)
    for utterance in utterances:
        prototype_rows[redim[utterance]["speaker"]].append(redim[utterance]["full"])
    prototypes = np.stack([
        l2(np.mean(np.stack(prototype_rows[speaker]), axis=0)) for speaker in speakers
    ]).astype(np.float32)
    features = []
    labels = []
    speaker_row_counts = defaultdict(int)
    for utterance in utterances:
        speaker = redim[utterance]["speaker"]
        for vector in redim[utterance][duration]:
            features.append(vector)
            labels.append(speaker_to_class[speaker])
            speaker_row_counts[speaker] += 1
    features = l2(np.stack(features)).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    row_speakers = [speakers[label] for label in labels]
    weights = np.asarray([
        1.0 / (len(speakers) * speaker_row_counts[speaker])
        for speaker in row_speakers
    ], dtype=np.float64)
    weights /= weights.sum()
    return features, labels, prototypes, weights


def train_adapter(redim, metadata, duration, config, seed):
    features, labels, prototypes, weights = training_arrays(
        redim, metadata, duration
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    torch.manual_seed(int(seed))
    x_all = torch.from_numpy(features)
    labels_all = torch.from_numpy(labels)
    protos = torch.from_numpy(prototypes)
    sampling = torch.from_numpy(weights)
    rank = int(config["rank"])
    A = torch.nn.Parameter(torch.randn((rank, 192), generator=generator) * 0.01)
    B = torch.nn.Parameter(torch.zeros((192, rank), dtype=torch.float32))
    optimizer = torch.optim.Adam([A, B], lr=float(config["lr"]))
    batch_size = min(256, len(features))
    trace = []
    started = time.perf_counter()
    for update in range(int(config["updates"])):
        batch = torch.multinomial(
            sampling, batch_size, replacement=True, generator=generator
        )
        x = x_all.index_select(0, batch)
        target = labels_all.index_select(0, batch)
        proposed = torch.nn.functional.normalize(
            x + torch.matmul(torch.matmul(x, A.t()), B.t()), dim=1
        )
        target_proto = protos.index_select(0, target)
        proto_loss = (1.0 - torch.sum(proposed * target_proto, dim=1)).mean()
        identity_loss = (1.0 - torch.sum(proposed * x, dim=1)).mean()
        cosine = torch.matmul(proposed, protos.t()).clamp(-1.0, 1.0)
        margin_logits = cosine.clone()
        margin_logits[torch.arange(len(batch)), target] -= float(config["margin"])
        ce_loss = torch.nn.functional.cross_entropy(
            float(config["scale"]) * margin_logits, target
        )
        loss = (
            float(config["ce"]) * ce_loss
            + float(config["proto"]) * proto_loss
            + float(config["identity"]) * identity_loss
            + 1.0e-4 * (torch.mean(A * A) + torch.mean(B * B))
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([A, B], 2.0)
        optimizer.step()
        if update in (0, 19, 79, 159, int(config["updates"]) - 1):
            trace.append({
                "update": update + 1, "loss": float(loss.detach()),
                "ce": float(ce_loss.detach()), "proto": float(proto_loss.detach()),
                "identity": float(identity_loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            })
    delta = torch.matmul(B, A).detach().cpu().numpy().astype(np.float32)
    return delta, {
        "duration": duration, "rows": len(features), "speakers": len(prototypes),
        "seconds": time.perf_counter() - started,
        "delta_frobenius": float(np.linalg.norm(delta)), "trace": trace,
    }


def transform(vector, delta, mix):
    source = l2(vector)
    proposed = l2(source + np.matmul(source, delta.T))
    return l2((1.0 - mix) * source + mix * proposed)


def concat(blocks, weights):
    return l2(np.concatenate([
        math.sqrt(weight) * l2(block) for block, weight in zip(blocks, weights)
    ]).astype(np.float32))


def evaluate(acoustic, acoustic_index, redims, trials, adapters, mix):
    rows = []
    for trial in trials:
        enrollment_id = str(trial["enrollment_utterance_id"])
        test_id = str(trial["test_utterance_id"])
        enrollment_index = acoustic_index[enrollment_id][0]
        enrollment = concat([
            acoustic["full"][enrollment_index], redims["lm"][enrollment_id]["full"],
            redims["ptn"][enrollment_id]["full"],
        ], ENROLL_WEIGHTS)
        baseline_enrollment = l2(acoustic["full"][enrollment_index])
        for duration, acoustic_key, redim_key in (
            ("full", "full", "full"),
            ("0p5", "short05", "short05"),
            ("1p0", "short10", "short10"),
        ):
            baseline_scores = []
            candidate_scores = []
            test_indices = (
                [acoustic_index[test_id][0]] if duration == "full" else acoustic_index[test_id]
            )
            for view, test_index in enumerate(test_indices):
                baseline_test = l2(acoustic[acoustic_key][test_index])
                if duration == "full":
                    candidate_test = concat([
                        baseline_test, redims["lm"][test_id]["full"],
                        redims["ptn"][test_id]["full"],
                    ], ENROLL_WEIGHTS)
                else:
                    duration_key = "short05" if duration == "0p5" else "short10"
                    candidate_test = concat([
                        baseline_test,
                        transform(redims["lm"][test_id][redim_key][view],
                                  adapters["lm"][duration_key], mix),
                        transform(redims["ptn"][test_id][redim_key][view],
                                  adapters["ptn"][duration_key], mix),
                    ], SHORT_WEIGHTS)
                baseline_scores.append(float(np.dot(baseline_enrollment, baseline_test)))
                candidate_scores.append(float(np.dot(enrollment, candidate_test)))
            rows.append({
                "dataset": str(trial["dataset"]),
                "base_scenario": str(trial["base_scenario"]),
                "duration": duration, "label": int(trial["label"]),
                "baseline": float(np.mean(baseline_scores)),
                "candidate": float(np.mean(candidate_scores)),
            })
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, selected in sorted(grouped.items()):
        labels = [row["label"] for row in selected]
        baseline = compute_eer_percent(labels, [row["baseline"] for row in selected])
        candidate = compute_eer_percent(labels, [row["candidate"] for row in selected])
        cells.append({
            "dataset": key[0], "base_scenario": key[1], "duration": key[2],
            "baseline_eer": baseline, "candidate_eer": candidate,
            "improvement": baseline - candidate, "rows": len(selected),
        })
    short = [cell for cell in cells if cell["duration"] != "full"]
    folds = {}
    for dataset in HOLDS:
        chosen = [cell for cell in cells if cell["dataset"] == dataset]
        folds[dataset] = float(np.mean([cell["improvement"] for cell in chosen]))
    return {
        "macro": float(np.mean([cell["improvement"] for cell in cells])),
        "short_macro": float(np.mean([cell["improvement"] for cell in short])),
        "worst_cell": float(min(cell["improvement"] for cell in cells)),
        "folds": folds, "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--configs", default=",".join(item["id"] for item in CONFIGS))
    parser.add_argument("--seed", type=int, default=824101)
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(args.threads)))
    selected = {item.strip() for item in args.configs.split(",") if item.strip()}
    configs = [item for item in CONFIGS if item["id"] in selected]
    if not configs or selected != {item["id"] for item in configs}:
        raise ValueError("invalid config selection")
    acoustic, acoustic_index, metadata = load_acoustic()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in MODEL_NAMES}
    trials = load_trials()
    required_ids = {
        str(row[key]) for row in trials
        for key in ("enrollment_utterance_id", "test_utterance_id")
    }
    for name in MODEL_NAMES:
        if not required_ids.issubset(redims[name]):
            raise ValueError("{} cache misses trial endpoints".format(name))
    results = []
    audits = []
    # Verify the unadapted dual fusion reproduces the archived point estimate.
    zero = np.zeros((192, 192), dtype=np.float32)
    base_metrics = evaluate(
        acoustic, acoustic_index, redims, trials,
        {name: {duration: zero for duration in DURATIONS} for name in MODEL_NAMES},
        0.0,
    )
    if abs(base_metrics["macro"] - 2.451356469508688) > 1.0e-9:
        raise ValueError("archived dual-fusion reproduction failed")
    for config in configs:
        print("CONFIG {}".format(config["id"]), flush=True)
        adapters = {name: {} for name in MODEL_NAMES}
        for model_offset, name in enumerate(MODEL_NAMES):
            for duration_offset, duration in enumerate(DURATIONS):
                adapter, audit = train_adapter(
                    redims[name], metadata, duration, config,
                    int(args.seed) + model_offset * 10 + duration_offset,
                )
                adapters[name][duration] = adapter
                audit.update({"config_id": config["id"], "model": name})
                audits.append(audit)
                print("FIT {} {} {} {:.2f}s".format(
                    config["id"], name, duration, audit["seconds"]
                ), flush=True)
        for mix in MIXES:
            metrics = evaluate(acoustic, acoustic_index, redims, trials, adapters, mix)
            results.append({"config_id": config["id"], "mix": mix, "metrics": metrics})
        best = max(
            [row for row in results if row["config_id"] == config["id"]],
            key=lambda row: (row["metrics"]["macro"], row["metrics"]["short_macro"],
                             row["metrics"]["worst_cell"]),
        )
        print(json.dumps({
            "config": config["id"], "mix": best["mix"],
            "macro": best["metrics"]["macro"],
            "short_macro": best["metrics"]["short_macro"],
            "worst_cell": best["metrics"]["worst_cell"],
            "folds": best["metrics"]["folds"],
        }, sort_keys=True), flush=True)
    ranked = sorted(results, key=lambda row: (
        row["metrics"]["macro"], row["metrics"]["short_macro"],
        row["metrics"]["worst_cell"],
    ), reverse=True)
    eligible = [row for row in ranked if (
        row["metrics"]["macro"] >= 2.0
        and row["metrics"]["short_macro"] >= 3.0
        and row["metrics"]["worst_cell"] >= -0.2
        and all(value >= 0.0 for value in row["metrics"]["folds"].values())
    )]
    report = {
        "schema_version": 1,
        "artifact_kind": "distillation_redim_output_prototype_aam_pilot",
        "research_only": True, "submission_created": False,
        "training_scope": "3dspeaker_train_only",
        "acoustic_cache": {"path": str(ACOUSTIC_CACHE), "sha256": sha256_file(ACOUSTIC_CACHE)},
        "trial_file": {"path": str(TRIALS), "sha256": sha256_file(TRIALS)},
        "redim_cache_counts": {name: len(redims[name]) for name in MODEL_NAMES},
        "unadapted_dual_reproduction": base_metrics,
        "configs": configs, "fit_audits": audits, "all_results": results,
        "eligible_count": len(eligible), "best": ranked[0],
        "eligible_best": eligible[0] if eligible else None,
        "status": "PILOT_POINT_GATE_PASS" if eligible else "PILOT_POINT_GATE_FAILED_DO_NOT_PACKAGE",
    }
    atomic_json(OUTPUT, report)
    print(json.dumps({
        "status": report["status"], "eligible_count": len(eligible),
        "best": ranked[0], "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
