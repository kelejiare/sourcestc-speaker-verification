#!/usr/bin/env python3
"""Fit and validate a two-domain short-to-full ReDimNet2 output adapter.

Training data is restricted to 3D-Speaker train speakers and the frozen
HI-MIA calibration-train speakers.  HI-MIA dev speakers, the frozen coverage
speakers/corpora, and H1 are used only for evaluation.  The configuration is
fixed before these evaluations and no candidate package is created here.
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

from tools.run_redim_output_kd_support import (
    DURATIONS,
    MODEL_DIRS,
    MODEL_NAMES,
    evaluate,
    l2,
    load_redim,
    load_trials,
    load_acoustic,
)
from tools.validate_redim_output_kd_3seed import (
    BASE,
    EXTERNALS,
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    external_trace,
    load_external_cache,
    load_external_trials,
    sha256_file,
    summarize_external,
)
from tools.score_lodo_coverage import speaker_bootstrap


OUTPUT_DIR = ROOT / "records/2026-09-04/020_distillation_redim_output_kd_multidomain"
OUTPUT = OUTPUT_DIR / "VALIDATION_RESULT.json"
ADAPTERS = OUTPUT_DIR / "distillation_multidomain_redim_output_kd_soup.npz"
TRACE = OUTPUT_DIR / "EXTERNAL_SCORE_TRACE.jsonl"
SEEDS = (824501, 824502, 824503)
CONFIG = {
    "id": "r16_balanced_two_domain_fixed",
    "rank": 16,
    "updates": 260,
    "lr": 0.0025,
    "margin": 0.15,
    "scale": 24.0,
    "ce": 1.0,
    "proto": 1.0,
    "identity": 2.5,
    "dataset_sampling": "equal_then_speaker_equal_then_view_equal",
}


def himia_training_ids(trial_path):
    with Path(trial_path).open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    selected = [row for row in rows if row.get("calibration_split") == "train"]
    if len(selected) != 720:
        raise ValueError("HI-MIA train trial count changed")
    utterances = {
        str(row[key]["utterance_stable_id"])
        for row in selected for key in ("enrollment", "test")
    }
    speakers = {
        str(row[key]["speaker_stable_id"])
        for row in selected for key in ("enrollment", "test")
    }
    if len(speakers) != 30:
        raise ValueError("HI-MIA calibration-train speaker count changed")
    return utterances, speakers


def training_arrays(redim, acoustic_metadata, himia, himia_ids, model, duration):
    records = []
    for utterance, row in redim.items():
        if row["dataset"] != "3dspeaker" or acoustic_metadata[utterance]["split"] != "train":
            continue
        records.append({
            "dataset": "3dspeaker",
            "speaker": "3dspeaker:" + row["speaker"],
            "full": row["full"],
            "short": row[duration],
        })
    suffix = "0p5" if duration == "short05" else "1p0"
    for utterance in sorted(himia_ids):
        row = himia[utterance]
        records.append({
            "dataset": "himia_train",
            "speaker": "himia:" + row["speaker"],
            "full": row[model + "_full"],
            "short": row[model + "_" + suffix],
        })
    datasets = sorted(set(row["dataset"] for row in records))
    if datasets != ["3dspeaker", "himia_train"]:
        raise ValueError("two-domain training contract violated")
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
    row_datasets = []
    row_speakers = []
    for row in records:
        for vector in row["short"]:
            features.append(vector)
            labels.append(speaker_to_class[row["speaker"]])
            row_datasets.append(row["dataset"])
            row_speakers.append(row["speaker"])
    features = l2(np.stack(features)).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    row_datasets = np.asarray(row_datasets)
    row_speakers = np.asarray(row_speakers)
    weights = np.zeros(len(features), dtype=np.float64)
    for dataset in datasets:
        positions = np.flatnonzero(row_datasets == dataset)
        dataset_speakers = sorted(set(str(row_speakers[index]) for index in positions))
        for speaker in dataset_speakers:
            speaker_positions = positions[row_speakers[positions] == speaker]
            weights[speaker_positions] = 1.0 / (
                len(datasets) * len(dataset_speakers) * len(speaker_positions)
            )
    weights /= weights.sum()
    audit = {
        "dataset_record_counts": {
            dataset: sum(row["dataset"] == dataset for row in records)
            for dataset in datasets
        },
        "dataset_speaker_counts": {
            dataset: len(set(row["speaker"] for row in records if row["dataset"] == dataset))
            for dataset in datasets
        },
        "feature_rows": len(features), "speaker_classes": len(speakers),
    }
    return features, labels, prototypes, weights, audit


def train_adapter(redim, metadata, himia, himia_ids, model, duration, seed):
    features, labels, prototypes, weights, data_audit = training_arrays(
        redim, metadata, himia, himia_ids, model, duration
    )
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
    batch_size = min(320, len(features))
    trace = []
    started = time.perf_counter()
    for update in range(CONFIG["updates"]):
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
        logits = cosine.clone()
        logits[torch.arange(len(batch)), target] -= CONFIG["margin"]
        ce_loss = torch.nn.functional.cross_entropy(CONFIG["scale"] * logits, target)
        loss = (
            CONFIG["ce"] * ce_loss + CONFIG["proto"] * proto_loss
            + CONFIG["identity"] * identity_loss
            + 1.0e-4 * (torch.mean(A * A) + torch.mean(B * B))
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite multidomain training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([A, B], 2.0)
        optimizer.step()
        if update in (0, 19, 79, 159, CONFIG["updates"] - 1):
            trace.append({
                "update": update + 1, "loss": float(loss.detach()),
                "ce": float(ce_loss.detach()), "proto": float(proto_loss.detach()),
                "identity": float(identity_loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            })
    delta = torch.matmul(B, A).detach().cpu().numpy().astype(np.float32)
    audit = dict(data_audit)
    audit.update({
        "seed": seed, "model": model, "duration": duration,
        "elapsed_seconds": time.perf_counter() - started,
        "delta_frobenius": float(np.linalg.norm(delta)), "trace": trace,
    })
    return delta, audit


def score_external(dataset, spec, soup):
    cache = load_external_cache(spec["cache"], spec["expected_cache"])
    trials = load_external_trials(spec["trials"], spec["split"], spec["expected_trials"])
    required = {
        str(row[key]["utterance_stable_id"])
        for row in trials for key in ("enrollment", "test")
    }
    if not required.issubset(cache):
        raise ValueError("external trial endpoint missing from cache")
    unadapted_trace = external_trace(dataset, cache, trials, soup, 0.0)
    adapted_trace = external_trace(dataset, cache, trials, soup, 1.0)
    return {
        "cache": cache, "trials": trials,
        "unadapted_trace": unadapted_trace, "adapted_trace": adapted_trace,
        "unadapted": summarize_external(unadapted_trace),
        "adapted": summarize_external(adapted_trace),
    }


def main():
    torch.set_num_threads(4)
    acoustic, acoustic_index, metadata = load_acoustic()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in MODEL_NAMES}
    coverage_trials = load_trials()
    himia_spec = EXTERNALS["himia_dev"]
    himia_cache = load_external_cache(himia_spec["cache"], himia_spec["expected_cache"])
    himia_ids, himia_speakers = himia_training_ids(himia_spec["trials"])
    if not himia_ids.issubset(himia_cache):
        raise ValueError("HI-MIA training endpoint missing from cache")
    fit_audits = []
    seed_adapters = {}
    seed_metrics = []
    for seed in SEEDS:
        print("FIT SEED {}".format(seed), flush=True)
        adapters = {name: {} for name in MODEL_NAMES}
        for model_offset, name in enumerate(MODEL_NAMES):
            for duration_offset, duration in enumerate(DURATIONS):
                delta, audit = train_adapter(
                    redims[name], metadata, himia_cache, himia_ids,
                    name, duration, seed + model_offset * 10 + duration_offset,
                )
                adapters[name][duration] = delta
                fit_audits.append(audit)
        coverage = evaluate(acoustic, acoustic_index, redims, coverage_trials, adapters, 1.0)
        himia = score_external("himia_dev", himia_spec, adapters)
        seed_adapters[seed] = adapters
        seed_metrics.append({
            "seed": seed, "coverage": coverage, "himia_dev": himia["adapted"]
        })
        print(json.dumps({
            "seed": seed, "coverage_macro": coverage["macro"],
            "coverage_short": coverage["short_macro"],
            "himia_macro": himia["adapted"]["macro_improvement_points"],
            "himia_short": himia["adapted"]["short_macro_improvement_points"],
            "himia_worst": himia["adapted"]["worst_cell_improvement_points"],
        }, sort_keys=True), flush=True)

    soup = {name: {} for name in MODEL_NAMES}
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "training_scope": np.asarray("3dspeaker_train_plus_himia_calibration_train"),
        "seeds": np.asarray(SEEDS, dtype=np.int64),
    }
    for name in MODEL_NAMES:
        for duration in DURATIONS:
            for seed in SEEDS:
                arrays["seed{}__{}__{}".format(seed, name, duration)] = (
                    seed_adapters[seed][name][duration]
                )
            value = np.mean(np.stack([
                seed_adapters[seed][name][duration] for seed in SEEDS
            ]).astype(np.float64), axis=0).astype(np.float32)
            if value.shape != (192, 192) or not np.isfinite(value).all():
                raise ValueError("invalid deployment soup")
            soup[name][duration] = value
            arrays[name + "__" + duration] = value
    atomic_npz(ADAPTERS, arrays)

    coverage = evaluate(acoustic, acoustic_index, redims, coverage_trials, soup, 1.0)
    external_results = {}
    all_trace = []
    for offset, (dataset, spec) in enumerate(EXTERNALS.items()):
        print("SCORE {}".format(dataset), flush=True)
        scored = score_external(dataset, spec, soup)
        bootstrap = speaker_bootstrap(
            scored["adapted_trace"], fold=dataset, repeats=2000,
            seed=824601 + offset, short_only=True,
        )
        external_results[dataset] = {
            "cache_count": len(scored["cache"]), "trial_count": len(scored["trials"]),
            "unadapted_dual_vs_acoustic": scored["unadapted"],
            "adapted_candidate_vs_acoustic": scored["adapted"],
            "short_speaker_bootstrap_vs_acoustic": bootstrap,
            "trial_path": str(spec["trials"]),
            "trial_sha256": sha256_file(spec["trials"]),
        }
        all_trace.extend(scored["adapted_trace"])
        print(json.dumps({
            "dataset": dataset,
            "macro": scored["adapted"]["macro_improvement_points"],
            "short": scored["adapted"]["short_macro_improvement_points"],
            "worst": scored["adapted"]["worst_cell_improvement_points"],
            "ci_upper": bootstrap["delta_ci95_upper"],
        }, sort_keys=True), flush=True)
    atomic_jsonl(TRACE, all_trace)

    gates = {
        "coverage_macro_at_least_2p6": coverage["macro"] >= 2.6,
        "coverage_short_at_least_3": coverage["short_macro"] >= 3.0,
        "coverage_worst_at_least_minus_0p2": coverage["worst_cell"] >= -0.2,
        "coverage_all_folds_positive": all(value > 0.0 for value in coverage["folds"].values()),
        "each_external_macro_nonnegative": all(
            row["adapted_candidate_vs_acoustic"]["macro_improvement_points"] >= 0.0
            for row in external_results.values()
        ),
        "each_external_short_positive": all(
            row["adapted_candidate_vs_acoustic"]["short_macro_improvement_points"] > 0.0
            for row in external_results.values()
        ),
        "each_external_worst_at_least_minus_0p2": all(
            row["adapted_candidate_vs_acoustic"]["worst_cell_improvement_points"] >= -0.2
            for row in external_results.values()
        ),
        "each_external_bootstrap_upper_below_zero": all(
            row["short_speaker_bootstrap_vs_acoustic"]["delta_ci95_upper"] < 0.0
            for row in external_results.values()
        ),
    }
    status = (
        "MULTIDOMAIN_EXTERNAL_GATE_PASS_BUILD_ALLOWED"
        if all(gates.values())
        else "MULTIDOMAIN_EXTERNAL_GATE_FAILED_DO_NOT_PACKAGE"
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "distillation_redim_output_kd_multidomain_three_seed_validation",
        "status": status, "package_created": False, "submission_created": False,
        "config": CONFIG, "seeds": list(SEEDS),
        "training_scope": {
            "datasets": ["3dspeaker_train", "himia_calibration_train"],
            "himia_training_endpoint_count": len(himia_ids),
            "himia_training_speaker_count": len(himia_speakers),
            "speaker_disjoint_from_himia_dev": True,
        },
        "fit_audits": fit_audits, "seed_metrics": seed_metrics,
        "soup_coverage_metrics": coverage, "external_results": external_results,
        "gate_checks": gates,
        "adapter_artifact": {
            "path": str(ADAPTERS), "bytes": ADAPTERS.stat().st_size,
            "sha256": sha256_file(ADAPTERS),
        },
        "score_trace": {
            "path": str(TRACE), "rows": len(all_trace), "sha256": sha256_file(TRACE)
        },
        "limitations": [
            "Proxy results cannot guarantee a hidden competition score above 91.",
            "H1 was consumed by earlier model work and is confirmatory rather than untouched.",
            "The strict leave-one-dataset-out audit is a separate release requirement.",
        ],
    }
    atomic_json(OUTPUT, report)
    print(json.dumps({
        "status": status, "gates": gates,
        "coverage": {"macro": coverage["macro"], "short": coverage["short_macro"],
                     "worst": coverage["worst_cell"], "folds": coverage["folds"]},
        "external": external_results, "adapter": report["adapter_artifact"],
        "report": str(OUTPUT),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
