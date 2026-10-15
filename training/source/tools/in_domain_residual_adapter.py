#!/usr/bin/env python3
"""Speaker-disjoint AliMeeting/Child residual adaptation around released distillation.

Stage A selects a low-rank residual using deterministic held-out speakers from
AliMeeting Eval and ChildMandarin dev.  Stage B refits that frozen recipe on
all speakers in those two corpora, then evaluates without tuning on
AliMeeting Test, the original six-corpus coverage set, and HI-MIA H1.

The extra residual composes algebraically with the released distillation 192-D LM
adapter, so a successful result would not add a model or runtime dependency.
This script is research-only and never builds a submission package.
"""

import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_redim_output_kd_support import (
    MODEL_DIRS, l2, load_redim, load_trials, load_acoustic,
)
from tools.score_lodo_coverage import compute_eer_percent
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.nap_lodo import (
    adapt, atomic_json, atomic_jsonl, load_distillation_adapters, sha256_file,
    distillation_score,
)


OUTPUT_DIR = ROOT / "records/2026-09-05/045_in_domain_residual_adapter"
RESULT_PATH = OUTPUT_DIR / "VALIDATION_RESULT.json"
ADAPTER_PATH = OUTPUT_DIR / "CANDIDATE_ADAPTERS.npz"
TRACE_PATH = OUTPUT_DIR / "SCORE_TRACE.jsonl"
SEEDS = (826201, 826202, 826203)
CONFIGS = (
    {"id": "r4_id8", "rank": 4, "identity": 8.0},
    {"id": "r8_id4", "rank": 8, "identity": 4.0},
    {"id": "r16_id2", "rank": 16, "identity": 2.0},
)
MIXES = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0)
DURATIONS = ("short05", "short10")


def stable_order(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def split_speakers(sources):
    by_dataset = defaultdict(set)
    for row in sources:
        by_dataset[str(row["dataset"])].add(str(row["speaker_stable_id"]))
    train = set(); dev = set(); audit = {}
    for dataset, speakers in sorted(by_dataset.items()):
        ordered = sorted(speakers, key=stable_order)
        dev_count = max(2, int(round(0.20 * len(ordered))))
        dataset_dev = set(ordered[-dev_count:])
        dataset_train = set(ordered[:-dev_count])
        if len(dataset_train) < 10:
            raise ValueError("insufficient train speakers")
        train.update(dataset_train); dev.update(dataset_dev)
        audit[dataset] = {"train": len(dataset_train), "dev": len(dataset_dev)}
    if train & dev:
        raise ValueError("speaker split leakage")
    return train, dev, audit


def training_arrays(sources, cache, speakers, current, duration):
    cache_key = "0p5" if duration == "short05" else "1p0"
    prototype_rows = defaultdict(list)
    selected = [row for row in sources if str(row["speaker_stable_id"]) in speakers]
    for row in selected:
        speaker = str(row["speaker_stable_id"])
        prototype_rows[speaker].append(cache[row["utterance_stable_id"]]["lm"]["full"])
    classes = sorted(prototype_rows)
    class_index = {speaker: index for index, speaker in enumerate(classes)}
    prototypes = np.stack([
        l2(np.mean(np.stack(prototype_rows[speaker]), axis=0)) for speaker in classes
    ]).astype(np.float32)
    features = []; labels = []; datasets = []; row_speakers = []
    for row in selected:
        speaker = str(row["speaker_stable_id"])
        dataset = str(row["dataset"])
        for vector in cache[row["utterance_stable_id"]]["lm"][cache_key]:
            features.append(adapt(vector, current["lm"][duration]))
            labels.append(class_index[speaker]); datasets.append(dataset)
            row_speakers.append(speaker)
    features = np.stack(features).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    datasets = np.asarray(datasets); row_speakers = np.asarray(row_speakers)
    sampling = np.zeros(len(features), dtype=np.float64)
    active_datasets = sorted(set(str(item) for item in datasets))
    for dataset in active_datasets:
        positions = np.flatnonzero(datasets == dataset)
        active_speakers = sorted(set(str(row_speakers[index]) for index in positions))
        for speaker in active_speakers:
            speaker_positions = positions[row_speakers[positions] == speaker]
            sampling[speaker_positions] = 1.0 / (
                len(active_datasets) * len(active_speakers) * len(speaker_positions)
            )
    sampling /= sampling.sum()
    return {
        "features": features, "labels": labels, "prototypes": prototypes,
        "sampling": sampling,
        "audit": {"datasets": active_datasets, "speakers": len(classes),
                  "rows": len(features)},
    }


def train_one(data, config, seed):
    torch.manual_seed(int(seed))
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(seed))
    x_all = torch.from_numpy(data["features"])
    labels_all = torch.from_numpy(data["labels"])
    prototypes = torch.from_numpy(data["prototypes"])
    sampling = torch.from_numpy(data["sampling"])
    rank = int(config["rank"])
    A = torch.nn.Parameter(torch.randn((rank, 192), generator=generator) * 0.01)
    B = torch.nn.Parameter(torch.zeros((192, rank), dtype=torch.float32))
    optimizer = torch.optim.Adam([A, B], lr=0.002)
    trace = []
    started = time.perf_counter()
    for update in range(400):
        batch = torch.multinomial(sampling, min(256, len(x_all)), True,
                                  generator=generator)
        x = x_all.index_select(0, batch)
        target = labels_all.index_select(0, batch)
        proposed = torch.nn.functional.normalize(
            x + torch.matmul(torch.matmul(x, A.t()), B.t()), dim=1
        )
        target_proto = prototypes.index_select(0, target)
        proto_loss = (1.0 - torch.sum(proposed * target_proto, dim=1)).mean()
        identity_loss = (1.0 - torch.sum(proposed * x, dim=1)).mean()
        logits = torch.matmul(proposed, prototypes.t()).clamp(-1.0, 1.0)
        logits[torch.arange(len(batch)), target] -= 0.15
        ce_loss = torch.nn.functional.cross_entropy(24.0 * logits, target)
        loss = ce_loss + proto_loss + float(config["identity"]) * identity_loss
        loss += 1.0e-4 * (torch.mean(A * A) + torch.mean(B * B))
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite residual loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([A, B], 2.0); optimizer.step()
        if update in (0, 39, 119, 239, 399):
            trace.append({"update": update + 1, "loss": float(loss.detach()),
                          "identity": float(identity_loss.detach())})
    delta = torch.matmul(B, A).detach().numpy().astype(np.float32)
    return delta, {"seed": int(seed), "elapsed_seconds": time.perf_counter() - started,
                   "frobenius": float(np.linalg.norm(delta)), "trace": trace}


def fit(sources, cache, speakers, current, config):
    output = {}; audit = {}
    for offset, duration in enumerate(DURATIONS):
        data = training_arrays(sources, cache, speakers, current, duration)
        values = []; seed_audits = []
        for seed in SEEDS:
            delta, seed_audit = train_one(data, config, seed + offset * 10)
            values.append(delta); seed_audits.append(seed_audit)
        output[duration] = np.mean(np.stack(values).astype(np.float64), axis=0).astype(np.float32)
        audit[duration] = {"data": data["audit"], "seeds": seed_audits,
                           "soup_frobenius": float(np.linalg.norm(output[duration]))}
    return output, audit


def compose(current, extra, mix):
    # adapt(adapt(x, current), mix*extra) is directionally identical to
    # adapt(x, current + mix*extra + mix*extra@current).
    return (current + float(mix) * extra
            + float(mix) * np.matmul(extra, current)).astype(np.float32)


def dev_trials(sources, dev_speakers):
    by_dataset_speaker = defaultdict(lambda: defaultdict(list))
    for row in sources:
        speaker = str(row["speaker_stable_id"])
        if speaker in dev_speakers:
            by_dataset_speaker[str(row["dataset"])][speaker].append(row)
    trials = []
    for dataset, by_speaker in sorted(by_dataset_speaker.items()):
        speakers = sorted(by_speaker)
        for index, speaker in enumerate(speakers):
            records = sorted(by_speaker[speaker], key=lambda row: row["utterance_stable_id"])
            if dataset == "alimeeting_eval_holdout":
                enrollment = next(row for row in records if row["domain"].endswith("near"))
                positives = [row for row in records if row["domain"].endswith("far")][:2]
            else:
                enrollment = records[0]; positives = records[1:3]
            negative_speaker = speakers[(index + 1) % len(speakers)]
            negatives_all = sorted(by_speaker[negative_speaker],
                                   key=lambda row: row["utterance_stable_id"])
            if dataset == "alimeeting_eval_holdout":
                negatives_all = [row for row in negatives_all
                                 if row["domain"].endswith("far")]
            negatives = negatives_all[:len(positives)]
            for offset, (positive, negative) in enumerate(zip(positives, negatives)):
                for label, test in ((1, positive), (0, negative)):
                    trials.append({
                        "dataset": dataset, "base_scenario": "speaker_split_dev",
                        "trial_id": "{}:{}:{}:{}".format(dataset, speaker, offset, label),
                        "label": label,
                        "enrollment_utterance_id": enrollment["utterance_stable_id"],
                        "test_utterance_id": test["utterance_stable_id"],
                        "enrollment_speaker_id": speaker,
                        "test_speaker_id": test["speaker_stable_id"],
                    })
    return trials


def score_cached(trials, cache, baseline, candidate):
    rows = []
    for raw in trials:
        enroll = cache[raw["enrollment_utterance_id"]]
        test = cache[raw["test_utterance_id"]]
        enrollment = [l2(enroll[name]["full"]) for name in ("acoustic", "lm", "ptn")]
        for duration, cache_key, adapter_key in (
            ("full", "full", None), ("0p5", "0p5", "short05"),
            ("1p0", "1p0", "short10"),
        ):
            base_scores = []; candidate_scores = []
            for view in range(1 if duration == "full" else 2):
                if duration == "full":
                    base_test = [l2(test[name]["full"]) for name in ("acoustic", "lm", "ptn")]
                    candidate_test = base_test
                else:
                    base_test = [
                        l2(test["acoustic"][cache_key][view]),
                        adapt(test["lm"][cache_key][view], baseline["lm"][adapter_key]),
                        adapt(test["ptn"][cache_key][view], baseline["ptn"][adapter_key]),
                    ]
                    candidate_test = [
                        base_test[0],
                        adapt(test["lm"][cache_key][view], candidate["lm"][adapter_key]),
                        base_test[2],
                    ]
                base_scores.append(distillation_score(enrollment, base_test, duration))
                candidate_scores.append(distillation_score(enrollment, candidate_test, duration))
            rows.append({
                "dataset": raw["dataset"], "base_scenario": raw["base_scenario"],
                "duration": duration, "trial_id": raw["trial_id"],
                "label": int(raw["label"]),
                "enrollment_speaker_id": raw["enrollment_speaker_id"],
                "test_speaker_id": raw["test_speaker_id"],
                "baseline_score": float(np.mean(base_scores)),
                "candidate_score": float(np.mean(candidate_scores)),
            })
    return rows


def score_coverage(baseline, candidate):
    acoustic, acoustic_index, _ = load_acoustic()
    redim = {name: load_redim(MODEL_DIRS[name]) for name in ("lm", "ptn")}
    rows = []
    for raw in load_trials():
        enroll_id = raw["enrollment_utterance_id"]; test_id = raw["test_utterance_id"]
        enrollment = [l2(acoustic["full"][acoustic_index[enroll_id][0]]),
                      l2(redim["lm"][enroll_id]["full"]),
                      l2(redim["ptn"][enroll_id]["full"])]
        for duration, acoustic_key, redim_key, adapter_key in (
            ("full", "full", "full", None), ("0p5", "short05", "short05", "short05"),
            ("1p0", "short10", "short10", "short10"),
        ):
            base_scores = []; candidate_scores = []
            indices = [acoustic_index[test_id][0]] if duration == "full" else acoustic_index[test_id]
            for view, idx in enumerate(indices):
                if duration == "full":
                    base_test = [l2(acoustic["full"][idx]), l2(redim["lm"][test_id]["full"]),
                                 l2(redim["ptn"][test_id]["full"])]
                    candidate_test = base_test
                else:
                    base_test = [
                        l2(acoustic[acoustic_key][idx]),
                        adapt(redim["lm"][test_id][redim_key][view], baseline["lm"][adapter_key]),
                        adapt(redim["ptn"][test_id][redim_key][view], baseline["ptn"][adapter_key]),
                    ]
                    candidate_test = [base_test[0],
                        adapt(redim["lm"][test_id][redim_key][view], candidate["lm"][adapter_key]),
                        base_test[2]]
                base_scores.append(distillation_score(enrollment, base_test, duration))
                candidate_scores.append(distillation_score(enrollment, candidate_test, duration))
            rows.append({
                "dataset": raw["dataset"], "base_scenario": raw["base_scenario"],
                "duration": duration, "trial_id": raw["trial_id"], "label": int(raw["label"]),
                "enrollment_speaker_id": raw["enrollment_speaker_id"],
                "test_speaker_id": raw["test_speaker_id"],
                "baseline_score": float(np.mean(base_scores)),
                "candidate_score": float(np.mean(candidate_scores)),
            })
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, selected in sorted(groups.items()):
        labels = [row["label"] for row in selected]
        base = compute_eer_percent(labels, [row["baseline_score"] for row in selected])
        candidate = compute_eer_percent(labels, [row["candidate_score"] for row in selected])
        cells.append({"dataset": key[0], "base_scenario": key[1], "duration": key[2],
                      "rows": len(selected), "distillation_eer": base, "candidate_eer": candidate,
                      "improvement": base - candidate})
    short = [cell for cell in cells if cell["duration"] != "full"]
    datasets = {}
    for dataset in sorted({cell["dataset"] for cell in cells}):
        chosen = [cell for cell in cells if cell["dataset"] == dataset]
        chosen_short = [cell for cell in chosen if cell["duration"] != "full"]
        datasets[dataset] = {
            "macro": float(np.mean([cell["improvement"] for cell in chosen])),
            "short_macro": float(np.mean([cell["improvement"] for cell in chosen_short])),
            "worst": float(min(cell["improvement"] for cell in chosen)),
        }
    return {"macro": float(np.mean([cell["improvement"] for cell in cells])),
            "short_macro": float(np.mean([cell["improvement"] for cell in short])),
            "worst": float(min(cell["improvement"] for cell in cells)),
            "datasets": datasets, "cells": cells}


def candidate_from_extra(current, extra, mix):
    result = {"lm": {}, "ptn": {key: value.copy()
                                  for key, value in current["ptn"].items()}}
    for duration in DURATIONS:
        result["lm"][duration] = compose(current["lm"][duration], extra[duration], mix)
    return result


def main():
    torch.set_num_threads(4)
    current = load_distillation_adapters()
    sources, _official_trials, cache = load_suite(SUITES[0])
    train_speakers, dev_speakers, split_audit = split_speakers(sources)
    trials = dev_trials(sources, dev_speakers)
    selection = []; development_fits = {}
    for config in CONFIGS:
        print("DEVELOPMENT FIT {}".format(config["id"]), flush=True)
        extra, audit = fit(sources, cache, train_speakers, current, config)
        development_fits[config["id"]] = audit
        for mix in MIXES:
            candidate = candidate_from_extra(current, extra, mix)
            metrics = summarize(score_cached(trials, cache, current, candidate))
            checks = {
                "macro_positive": metrics["macro"] > 0.0,
                "short_positive": metrics["short_macro"] > 0.0,
                "worst_ge_minus_0p2": metrics["worst"] >= -0.2,
                "each_dataset_short_nonnegative": all(
                    item["short_macro"] >= 0.0 for item in metrics["datasets"].values()),
            }
            selection.append({"config": config, "mix": mix, "metrics": metrics,
                              "checks": checks, "eligible": all(checks.values())})
            print("{} mix={} macro={:.4f} short={:.4f} worst={:.4f} pass={}".format(
                config["id"], mix, metrics["macro"], metrics["short_macro"],
                metrics["worst"], all(checks.values())), flush=True)
    eligible = [row for row in selection if row["eligible"]]
    eligible.sort(key=lambda row: (row["metrics"]["short_macro"],
                                  row["metrics"]["macro"], -row["mix"]), reverse=True)
    selected = eligible[0] if eligible else None
    if selected is None:
        result = {"schema_version": 1, "status": "DEV_GATE_FAILED_DO_NOT_PACKAGE",
                  "split_audit": split_audit, "dev_trials": len(trials),
                  "selection": selection, "development_fits": development_fits}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True); atomic_json(RESULT_PATH, result)
        print(json.dumps({"status": result["status"], "result": str(RESULT_PATH)}, indent=2))
        return
    print("REFIT {}".format(selected["config"]["id"]), flush=True)
    all_speakers = train_speakers | dev_speakers
    extra, refit_audit = fit(sources, cache, all_speakers, current, selected["config"])
    candidate = candidate_from_extra(current, extra, selected["mix"])
    arrays = {"schema_version": np.asarray(1, dtype=np.int64),
              "seed_ids": np.asarray(SEEDS, dtype=np.int64)}
    for model in ("lm", "ptn"):
        for duration in DURATIONS:
            arrays[model + "_" + duration] = candidate[model][duration]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ADAPTER_PATH.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    validations = {}
    all_trace = []
    # AliMeeting Test and HI-MIA H1 are not used for selection or refit.
    for suite in (SUITES[1], SUITES[2]):
        _s, suite_trials, suite_cache = load_suite(suite)
        rows = score_cached(suite_trials, suite_cache, current, candidate)
        validations[suite["name"]] = summarize(rows); all_trace.extend(rows)
    coverage_rows = score_coverage(current, candidate)
    validations["coverage"] = summarize(coverage_rows); all_trace.extend(coverage_rows)
    checks = {
        "coverage_macro_positive": validations["coverage"]["macro"] > 0.0,
        "coverage_short_positive": validations["coverage"]["short_macro"] > 0.0,
        "coverage_worst_ge_minus_0p2": validations["coverage"]["worst"] >= -0.2,
        "alimeeting_test_macro_positive": validations["alimeeting_test"]["macro"] > 0.0,
        "alimeeting_test_short_positive": validations["alimeeting_test"]["short_macro"] > 0.0,
        "alimeeting_test_worst_ge_minus_0p2": validations["alimeeting_test"]["worst"] >= -0.2,
        "himia_not_regress": validations["himia_h1"]["worst"] >= -0.2,
    }
    status = "PASS_REQUIRES_RUNTIME" if all(checks.values()) else "FINAL_GATE_FAILED_DO_NOT_PACKAGE"
    atomic_jsonl(TRACE_PATH, all_trace)
    result = {
        "schema_version": 1,
        "artifact_kind": "joint_speaker_disjoint_in_domain_residual_adapter",
        "status": status, "distillation_online_anchor": 89.76798,
        "target_online_score": 91.0, "split_audit": split_audit,
        "selection": selection, "selected": selected,
        "development_fits": development_fits, "refit_audit": refit_audit,
        "validations": validations, "checks": checks,
        "adapter": {"path": str(ADAPTER_PATH.relative_to(ROOT)),
                    "sha256": sha256_file(ADAPTER_PATH),
                    "bytes": ADAPTER_PATH.stat().st_size},
        "trace": {"path": str(TRACE_PATH.relative_to(ROOT)),
                  "sha256": sha256_file(TRACE_PATH)},
        "limitations": [
            "AliMeeting Test and all coverage corpora were consumed in prior work.",
            "No public validation can guarantee a hidden score of 91.",
        ],
    }
    atomic_json(RESULT_PATH, result)
    print(json.dumps({"status": status, "selected": selected,
                      "validations": validations, "checks": checks,
                      "result": str(RESULT_PATH), "sha256": sha256_file(RESULT_PATH)},
                     ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
