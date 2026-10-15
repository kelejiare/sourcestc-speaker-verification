#!/usr/bin/env python3
"""Train three all-source ReDim short-to-full adapters and run diagnostics.

All included sources have already been consumed by prior analysis.  This stage
is deliberately marked development-only; it must be confirmed on a new child
development holdout before it can authorize any package.
"""

import json
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

from tools.fresh_dataset_confirmation import (
    FRONTIER,
    TRIALS_PATH as FRESH_TRIALS,
    load_acoustic_post,
    score as score_fresh,
    summarize as summarize_fresh,
)
from tools.run_redim_output_kd_support import (
    DURATIONS,
    MODEL_DIRS,
    MODEL_NAMES,
    l2,
    load_redim,
    load_trials,
    load_acoustic,
)
from tools.search_multidomain_fusion_frontier import (
    coverage_components,
    external_components,
    load_adapters as _unused_load_previous_adapters,
    apply_weights,
    summarize_coverage,
)
from tools.train_redim_output_kd_multidomain import himia_training_ids
from tools.validate_redim_output_kd_3seed import (
    EXTERNALS,
    atomic_json,
    atomic_npz,
    load_external_cache,
    load_external_trials,
    sha256_file,
    summarize_external,
)


OUTPUT_DIR = ROOT / "records/2026-09-04/023_distillation_allsource_output_kd"
OUTPUT = OUTPUT_DIR / "DEVELOPMENT_RESULT.json"
ADAPTERS = OUTPUT_DIR / "distillation_allsource_three_seed_adapters.npz"
SEEDS = (824801, 824802, 824803)
CONFIG = {
    "id": "rank24_allsource_balanced_fixed",
    "rank": 24, "updates": 400, "lr": 0.002,
    "margin": 0.15, "scale": 24.0,
    "ce": 1.0, "proto": 1.0, "identity": 3.0,
}
DATASET_PRIORS = {
    "3dspeaker": 0.28,
    "himia_train": 0.22,
    "childmandarin": 0.20,
    "aishell4": 0.10,
    "seniortalk": 0.08,
    "commonvoice_beijing": 0.07,
    "magicdata_slr68": 0.05,
}


def build_records(redim, metadata, himia, himia_ids, model, duration):
    records = []
    for utterance, row in redim.items():
        dataset = row["dataset"]
        if dataset == "3dspeaker" and metadata[utterance]["split"] != "train":
            continue
        if dataset not in DATASET_PRIORS or dataset == "himia_train":
            continue
        records.append({
            "dataset": dataset, "speaker": dataset + ":" + row["speaker"],
            "full": row["full"], "short": row[duration],
        })
    suffix = "0p5" if duration == "short05" else "1p0"
    for utterance in sorted(himia_ids):
        row = himia[utterance]
        records.append({
            "dataset": "himia_train", "speaker": "himia:" + row["speaker"],
            "full": row[model + "_full"], "short": row[model + "_" + suffix],
        })
    if set(row["dataset"] for row in records) != set(DATASET_PRIORS):
        raise ValueError("all-source training dataset set changed")
    return records


def training_arrays(records):
    prototype_rows = defaultdict(list)
    for row in records:
        prototype_rows[row["speaker"]].append(row["full"])
    speakers = sorted(prototype_rows)
    speaker_to_class = {speaker: index for index, speaker in enumerate(speakers)}
    prototypes = np.stack([
        l2(np.mean(np.stack(prototype_rows[speaker]), axis=0))
        for speaker in speakers
    ]).astype(np.float32)
    features = []
    labels = []
    datasets = []
    row_speakers = []
    for row in records:
        for vector in row["short"]:
            features.append(vector)
            labels.append(speaker_to_class[row["speaker"]])
            datasets.append(row["dataset"])
            row_speakers.append(row["speaker"])
    features = l2(np.stack(features)).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    datasets = np.asarray(datasets)
    row_speakers = np.asarray(row_speakers)
    weights = np.zeros(len(features), dtype=np.float64)
    for dataset, prior in DATASET_PRIORS.items():
        positions = np.flatnonzero(datasets == dataset)
        active_speakers = sorted(set(str(row_speakers[index]) for index in positions))
        for speaker in active_speakers:
            speaker_positions = positions[row_speakers[positions] == speaker]
            weights[speaker_positions] = prior / (
                len(active_speakers) * len(speaker_positions)
            )
    weights /= weights.sum()
    audit = {
        "feature_rows": len(features), "speaker_classes": len(speakers),
        "dataset_records": {
            dataset: sum(row["dataset"] == dataset for row in records)
            for dataset in DATASET_PRIORS
        },
        "dataset_speakers": {
            dataset: len(set(row["speaker"] for row in records if row["dataset"] == dataset))
            for dataset in DATASET_PRIORS
        },
    }
    return features, labels, prototypes, weights, audit


def train(redim, metadata, himia, himia_ids, model, duration, seed):
    records = build_records(redim, metadata, himia, himia_ids, model, duration)
    features, labels, prototypes, weights, audit = training_arrays(records)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    x_all = torch.from_numpy(features)
    labels_all = torch.from_numpy(labels)
    protos = torch.from_numpy(prototypes)
    sampling = torch.from_numpy(weights)
    A = torch.nn.Parameter(
        torch.randn((CONFIG["rank"], 192), generator=generator) * 0.01
    )
    B = torch.nn.Parameter(torch.zeros((192, CONFIG["rank"]), dtype=torch.float32))
    optimizer = torch.optim.Adam([A, B], lr=CONFIG["lr"])
    trace = []
    started = time.perf_counter()
    for update in range(CONFIG["updates"]):
        batch = torch.multinomial(
            sampling, min(384, len(features)), replacement=True, generator=generator
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
        logits = cosine.clone()
        logits[torch.arange(len(batch)), target] -= CONFIG["margin"]
        ce_loss = torch.nn.functional.cross_entropy(CONFIG["scale"] * logits, target)
        loss = (
            CONFIG["ce"] * ce_loss + CONFIG["proto"] * proto_loss
            + CONFIG["identity"] * identity_loss
            + 1.0e-4 * (torch.mean(A * A) + torch.mean(B * B))
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite all-source loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([A, B], 2.0)
        optimizer.step()
        if update in (0, 39, 119, 239, CONFIG["updates"] - 1):
            trace.append({
                "update": update + 1, "loss": float(loss.detach()),
                "ce": float(ce_loss.detach()), "proto": float(proto_loss.detach()),
                "identity": float(identity_loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            })
    delta = torch.matmul(B, A).detach().cpu().numpy().astype(np.float32)
    audit.update({
        "seed": seed, "model": model, "duration": duration,
        "elapsed_seconds": time.perf_counter() - started,
        "delta_frobenius": float(np.linalg.norm(delta)), "trace": trace,
    })
    return delta, audit


def main():
    torch.set_num_threads(4)
    with FRONTIER.open("r", encoding="utf-8") as stream:
        candidate = json.load(stream)["best"]["config"]
    if candidate != {
        "adapter_mix": 1.0, "aggregation": "mean_delta", "enroll_acoustic": 0.86,
        "ptn_ratio": 0.05, "short_acoustic": 0.2,
    }:
        raise ValueError("fixed candidate weights changed")
    acoustic, acoustic_index, _ = load_acoustic()
    _, _, metadata = load_acoustic_post()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in MODEL_NAMES}
    himia_spec = EXTERNALS["himia_dev"]
    himia = load_external_cache(himia_spec["cache"], himia_spec["expected_cache"])
    himia_ids, himia_speakers = himia_training_ids(himia_spec["trials"])
    fit_audits = []
    seed_adapters = {}
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "training_scope": np.asarray("seven_consumed_public_domains"),
        "seeds": np.asarray(SEEDS, dtype=np.int64),
    }
    for seed in SEEDS:
        print("FIT SEED {}".format(seed), flush=True)
        seed_adapters[seed] = {name: {} for name in MODEL_NAMES}
        for model_offset, name in enumerate(MODEL_NAMES):
            for duration_offset, duration in enumerate(DURATIONS):
                delta, audit = train(
                    redims[name], metadata, himia, himia_ids, name, duration,
                    seed + model_offset * 10 + duration_offset,
                )
                seed_adapters[seed][name][duration] = delta
                arrays["seed{}__{}__{}".format(seed, name, duration)] = delta
                fit_audits.append(audit)
    soup = {name: {} for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        for duration in DURATIONS:
            value = np.mean(np.stack([
                seed_adapters[seed][name][duration] for seed in SEEDS
            ]).astype(np.float64), axis=0).astype(np.float32)
            soup[name][duration] = value
            arrays[name + "__" + duration] = value
    atomic_npz(ADAPTERS, arrays)
    wrapped = {"mean_delta": soup}

    coverage_components_rows = coverage_components(
        acoustic, acoustic_index, redims, load_trials(), wrapped, "mean_delta", 1.0
    )
    coverage = summarize_coverage(apply_weights(
        coverage_components_rows, candidate, False
    ))
    external = {}
    for dataset, spec in EXTERNALS.items():
        cache = load_external_cache(spec["cache"], spec["expected_cache"])
        trials = load_external_trials(spec["trials"], spec["split"], spec["expected_trials"])
        rows = external_components(
            dataset, cache, trials, wrapped, "mean_delta", 1.0
        )
        external[dataset] = summarize_external(apply_weights(rows, candidate, True))
    acoustic_post, acoustic_post_index, _ = load_acoustic_post()
    with FRESH_TRIALS.open("r", encoding="utf-8") as stream:
        fresh_trials = [json.loads(line) for line in stream]
    fresh = summarize_fresh(score_fresh(
        acoustic_post, acoustic_post_index, redims, wrapped, candidate, fresh_trials
    ))
    diagnostics = {
        "coverage": coverage, "external": external, "consumed_fresh": fresh,
    }
    checks = {
        "coverage_macro_at_least_2p6": coverage["macro"] >= 2.6,
        "coverage_short_at_least_3": coverage["short_macro"] >= 3.0,
        "coverage_worst_at_least_minus_0p2": coverage["worst_cell"] >= -0.2,
        "external_worst_at_least_minus_0p2": all(
            row["worst_cell_improvement_points"] >= -0.2 for row in external.values()
        ),
        "fresh_macro_positive": fresh["macro_improvement_points"] > 0.0,
        "fresh_short_positive": fresh["short_macro_improvement_points"] > 0.0,
        "fresh_worst_at_least_minus_0p2": fresh["worst_cell_improvement_points"] >= -0.2,
    }
    status = (
        "DEVELOPMENT_DIAGNOSTIC_PASS_REQUIRES_NEW_CHILD_HOLDOUT"
        if all(checks.values())
        else "DEVELOPMENT_DIAGNOSTIC_FAILED_DO_NOT_BUILD_HOLDOUT"
    )
    report = {
        "schema_version": 1, "artifact_kind": "distillation_allsource_output_kd_development",
        "status": status, "research_only": True, "package_created": False,
        "candidate_weights": candidate, "adapter_config": CONFIG,
        "dataset_priors": DATASET_PRIORS, "seeds": list(SEEDS),
        "himia_train_speakers": len(himia_speakers),
        "fit_audits": fit_audits, "diagnostics": diagnostics, "gate_checks": checks,
        "adapter_artifact": {"path": str(ADAPTERS), "bytes": ADAPTERS.stat().st_size,
                             "sha256": sha256_file(ADAPTERS)},
        "limitations": [
            "All current diagnostic corpora are consumed development data.",
            "A completely new ChildMandarin dev subset is required before packaging.",
            "No public proxy can guarantee a hidden score above 91.",
        ],
    }
    atomic_json(OUTPUT, report)
    print(json.dumps({
        "status": status, "checks": checks,
        "coverage": {"macro": coverage["macro"], "short": coverage["short_macro"],
                     "worst": coverage["worst_cell"], "folds": coverage["folds"]},
        "external": external, "fresh": fresh,
        "adapter": report["adapter_artifact"], "report": str(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
