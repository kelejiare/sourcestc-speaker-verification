#!/usr/bin/env python3
"""Search a joint-anchored low-rank cosine metric adapter for source one.

The acoustic models and the released joint strict embedding are frozen.  This
script learns only three near-identity transforms in the final 832-D space:
one for enrollment/full utterances and one for each official short duration.
Selection is speaker-disjoint inside the coverage corpus.  ChildMandarin dev
is opened only after the candidate/configuration has been selected.
"""

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

from tools.run_redim_output_kd_support import (
    MODEL_DIRS,
    load_redim,
    load_trials,
    load_acoustic,
)
from tools.score_lodo_coverage import compute_eer_percent
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.nap_lodo import adapt, sha256_file
from joint_kd_inference import (
    ENROLL_WEIGHTS,
    SHORT05_WEIGHTS,
    SHORT10_WEIGHTS,
    _load_joint_adapter,
)
from source_adapted_inference import (
    TARGET_ENROLL_WEIGHTS,
    TARGET_SHORT05_WEIGHTS,
    TARGET_SHORT10_WEIGHTS,
)
from source_kd_inference import (
    MOBILE_ENROLL_WEIGHTS,
    MOBILE_SHORT05_WEIGHTS,
    MOBILE_SHORT10_WEIGHTS,
    load_mobile_adapter,
)


OUTPUT_DIR = ROOT / "records/2026-09-10/078_short_metric_joint_anchored_hierarchical_cml"
RESULT_PATH = OUTPUT_DIR / "SEARCH_RESULT.json"
ADAPTER_PATH = OUTPUT_DIR / "SHORT_METRIC_CANDIDATE_ADAPTER.npz"
JOINT_ADAPTER_PATH = ROOT / "model/joint_kd_adapter.npz"
MOBILE_ADAPTER_PATH = ROOT / "model/mobile_kd_adapter.npz"

SOURCE1_DATASETS = {
    "3dspeaker",
    "childmandarin",
    "childmandarin_dev_holdout",
}
SEEDS = (826101, 826102, 826103)
ALPHAS = (0.10, 0.16, 0.22, 0.26)
CONFIGS = tuple(
    {
        "id": "r{}_i{}".format(rank, identity),
        "rank": rank,
        "identity": float(identity),
        "updates": 280,
        "lr": 0.002,
        "margin": 0.025,
        "scale": 24.0,
        "bad_direction": 0.75,
        "factor_l2": 1.0e-4,
    }
    for rank in (4, 8, 12)
    for identity in (4, 10)
)


def l2(value):
    array = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1.0e-8:
        raise ValueError("invalid embedding norm")
    return np.ascontiguousarray(array / norm, dtype=np.float32)


def fuse(blocks, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (3,) or np.any(weights < 0.0):
        raise ValueError("invalid block weights")
    if abs(float(weights.sum()) - 1.0) > 1.0e-10:
        raise ValueError("block weights do not sum to one")
    value = np.concatenate([
        math.sqrt(float(weight)) * l2(block)
        for block, weight in zip(blocks, weights)
    ]).astype(np.float32)
    if value.shape != (832,):
        raise ValueError("unexpected fused dimension")
    return l2(value)


def stable_fold(speaker, folds=3):
    digest = hashlib.sha256(str(speaker).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % int(folds)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
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


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def component_vectors(cache, duration, joint_matrices, mobile_matrices):
    """Return per-view final joint and rejected-mobile vectors for source one."""

    if duration == "full":
        blocks = [l2(cache[name]["full"]) for name in ("acoustic", "lm", "ptn")]
        return (
            [fuse(blocks, TARGET_ENROLL_WEIGHTS)],
            [fuse(blocks, MOBILE_ENROLL_WEIGHTS)],
        )

    cache_key = "0p5" if duration == "0p5" else "1p0"
    adapter_key = "short05" if duration == "0p5" else "short10"
    joint_weights = TARGET_SHORT05_WEIGHTS if duration == "0p5" else TARGET_SHORT10_WEIGHTS
    mobile_weights = MOBILE_SHORT05_WEIGHTS if duration == "0p5" else MOBILE_SHORT10_WEIGHTS
    joint_values = []
    mobile_values = []
    views = int(np.asarray(cache["acoustic"][cache_key]).shape[0])
    for view in range(views):
        acoustic = l2(cache["acoustic"][cache_key][view])
        lm_raw = l2(cache["lm"][cache_key][view])
        ptn_raw = l2(cache["ptn"][cache_key][view])
        if duration == "0p5":
            lm = adapt(lm_raw, mobile_matrices["lm_short05"])
            ptn = adapt(ptn_raw, mobile_matrices["ptn_short05"])
        else:
            lm = adapt(lm_raw, joint_matrices["lm_short10"])
            ptn = adapt(ptn_raw, joint_matrices["ptn_short10"])
        blocks = (acoustic, lm, ptn)
        joint_values.append(fuse(blocks, joint_weights))
        mobile_values.append(fuse(blocks, mobile_weights))
    return joint_values, mobile_values


def coverage_cache(required_utterances):
    acoustic, indices, metadata = load_acoustic()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in ("lm", "ptn")}
    output = {}
    for utterance in sorted(set(required_utterances)):
        if utterance not in indices or utterance not in redims["lm"] or utterance not in redims["ptn"]:
            raise RuntimeError("coverage component missing for {}".format(utterance))
        positions = indices[utterance]
        output[utterance] = {
            "acoustic": {
                "full": l2(acoustic["full"][positions[0]]),
                "0p5": np.stack([l2(acoustic["short05"][item]) for item in positions]),
                "1p0": np.stack([l2(acoustic["short10"][item]) for item in positions]),
            },
            "lm": {
                "full": redims["lm"][utterance]["full"],
                "0p5": redims["lm"][utterance]["short05"],
                "1p0": redims["lm"][utterance]["short10"],
            },
            "ptn": {
                "full": redims["ptn"][utterance]["full"],
                "0p5": redims["ptn"][utterance]["short05"],
                "1p0": redims["ptn"][utterance]["short10"],
            },
            "dataset": metadata[utterance]["dataset"],
            "speaker": metadata[utterance]["speaker"],
        }
    return output


def add_trial_rows(rows, suite, raw, cache, joint_matrices, mobile_matrices):
    dataset = str(raw["dataset"])
    if dataset not in SOURCE1_DATASETS:
        return
    enrollment_id = str(raw["enrollment_utterance_id"])
    test_id = str(raw["test_utterance_id"])
    enrollment, enrollment_mobile = component_vectors(
        cache[enrollment_id], "full", joint_matrices, mobile_matrices
    )
    enrollment = enrollment[0]
    enrollment_mobile = enrollment_mobile[0]
    for duration in ("full", "0p5", "1p0"):
        test, test_mobile = component_vectors(
            cache[test_id], duration, joint_matrices, mobile_matrices
        )
        for view, (test_vector, rejected_vector) in enumerate(zip(test, test_mobile)):
            rows.append({
                "suite": suite,
                "dataset": dataset,
                "base_scenario": str(raw["base_scenario"]),
                "duration": duration,
                "trial_id": str(raw["trial_id"]),
                "view": int(view),
                "label": int(raw["label"]),
                "enrollment_speaker_id": str(raw["enrollment_speaker_id"]),
                "test_speaker_id": str(raw["test_speaker_id"]),
                "enrollment": enrollment,
                "test": test_vector,
                "baseline_score": float(np.dot(enrollment, test_vector)),
                "rejected_mobile_score": float(np.dot(enrollment_mobile, rejected_vector)),
            })


def load_rows():
    joint_matrices, joint_receipt = _load_joint_adapter(JOINT_ADAPTER_PATH)
    mobile_matrices, mobile_receipt = load_mobile_adapter(MOBILE_ADAPTER_PATH)
    coverage_trials = load_trials()
    coverage_ids = set()
    for raw in coverage_trials:
        coverage_ids.add(str(raw["enrollment_utterance_id"]))
        coverage_ids.add(str(raw["test_utterance_id"]))
    coverage = coverage_cache(coverage_ids)
    development = []
    for raw in coverage_trials:
        add_trial_rows(
            development, "coverage_lodo", raw, coverage,
            joint_matrices, mobile_matrices,
        )

    suite = SUITES[0]
    _, trials, cache = load_suite(suite)
    audit = []
    for raw in trials:
        add_trial_rows(
            audit, suite["name"], raw, cache,
            joint_matrices, mobile_matrices,
        )
    audit = [row for row in audit if row["dataset"] == "childmandarin_dev_holdout"]
    if not development or not audit:
        raise RuntimeError("short_metric development/audit rows are empty")
    return development, audit, joint_receipt, mobile_receipt


def split_rows(rows, held_fold):
    train = []
    validation = []
    dropped = 0
    for row in rows:
        left = stable_fold(row["enrollment_speaker_id"])
        right = stable_fold(row["test_speaker_id"])
        if left == held_fold and right == held_fold:
            validation.append(row)
        elif left != held_fold and right != held_fold:
            train.append(row)
        else:
            dropped += 1
    return train, validation, dropped


def group_indices(rows):
    groups = defaultdict(lambda: {0: [], 1: []})
    for index, row in enumerate(rows):
        key = (row["dataset"], row["base_scenario"], row["duration"])
        groups[key][int(row["label"])].append(index)
    output = {}
    for key, labels in groups.items():
        if labels[0] and labels[1]:
            output[key] = {
                label: np.asarray(value, dtype=np.int64)
                for label, value in labels.items()
            }
    if not output:
        raise RuntimeError("no balanced training groups")
    return output


def row_arrays(rows):
    return {
        "enrollment": np.stack([row["enrollment"] for row in rows]).astype(np.float32),
        "test": np.stack([row["test"] for row in rows]).astype(np.float32),
        "duration": np.asarray([
            0 if row["duration"] == "full" else 1 if row["duration"] == "0p5" else 2
            for row in rows
        ], dtype=np.int64),
        "baseline": np.asarray([row["baseline_score"] for row in rows], dtype=np.float32),
        "bad_delta": np.asarray([
            row["rejected_mobile_score"] - row["baseline_score"] for row in rows
        ], dtype=np.float32),
    }


def train_one(rows, config, seed):
    arrays = row_arrays(rows)
    groups = group_indices(rows)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    torch.manual_seed(int(seed))
    rank = int(config["rank"])
    a_params = torch.nn.ParameterList()
    b_params = torch.nn.ParameterList()
    for _ in range(3):
        a_params.append(torch.nn.Parameter(
            torch.randn((rank, 832), generator=generator) * 0.004
        ))
        b_params.append(torch.nn.Parameter(torch.zeros((832, rank))))
    parameters = list(a_params.parameters()) + list(b_params.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(config["lr"]))
    enrollment_all = torch.from_numpy(arrays["enrollment"])
    test_all = torch.from_numpy(arrays["test"])
    duration_all = torch.from_numpy(arrays["duration"])
    baseline_all = torch.from_numpy(arrays["baseline"])
    bad_all = torch.from_numpy(arrays["bad_delta"])
    trace = []
    started = time.perf_counter()

    def transform(x, duration_index):
        a = a_params[int(duration_index)]
        b = b_params[int(duration_index)]
        residual = torch.matmul(torch.matmul(x, a.t()), b.t())
        return torch.nn.functional.normalize(x + residual, dim=1)

    for update in range(int(config["updates"])):
        sampled = []
        spans = []
        for key in sorted(groups):
            begin = len(sampled)
            for label in (1, 0):
                available = groups[key][label]
                count = min(32, max(8, len(available)))
                draw = torch.randint(
                    0, len(available), (count,), generator=generator
                ).numpy()
                sampled.extend(int(available[item]) for item in draw)
            spans.append((key, begin, len(sampled), count))
        indices = torch.tensor(sampled, dtype=torch.long)
        enrollment = enrollment_all.index_select(0, indices)
        test = test_all.index_select(0, indices)
        durations = duration_all.index_select(0, indices)
        baseline = baseline_all.index_select(0, indices)
        bad_delta = bad_all.index_select(0, indices)
        q_enrollment = transform(enrollment, 0)
        q_test = torch.empty_like(test)
        for duration_index in (0, 1, 2):
            mask = durations == duration_index
            if bool(mask.any()):
                q_test[mask] = transform(test[mask], duration_index)
        expert = torch.sum(q_enrollment * q_test, dim=1)
        anchored = 0.78 * baseline + 0.22 * expert
        ranking_terms = []
        for _, begin, end, count in spans:
            scores = anchored[begin:end]
            positive = scores[:count]
            negative = scores[count:]
            differences = positive[:, None] - negative[None, :]
            ranking_terms.append(torch.nn.functional.softplus(
                float(config["scale"]) * (float(config["margin"]) - differences)
            ).mean() / float(config["scale"]))
        ranking = torch.stack(ranking_terms).mean()
        identity = (
            (1.0 - torch.sum(q_enrollment * enrollment, dim=1)).mean()
            + (1.0 - torch.sum(q_test * test, dim=1)).mean()
        ) * 0.5
        candidate_delta = anchored - baseline
        centered_candidate = candidate_delta - candidate_delta.mean()
        centered_bad = bad_delta - bad_delta.mean()
        denominator = (
            torch.sqrt(torch.mean(centered_candidate * centered_candidate) + 1.0e-10)
            * torch.sqrt(torch.mean(centered_bad * centered_bad) + 1.0e-10)
        )
        correlation = torch.mean(centered_candidate * centered_bad) / denominator
        bad_penalty = torch.relu(correlation) ** 2
        factor_l2 = torch.stack([
            torch.mean(value * value) for value in parameters
        ]).mean()
        loss = (
            ranking
            + float(config["identity"]) * identity
            + float(config["bad_direction"]) * bad_penalty
            + float(config["factor_l2"]) * factor_l2
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite short_metric training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 2.0)
        optimizer.step()
        if update in (0, 39, 119, 199, int(config["updates"]) - 1):
            trace.append({
                "update": update + 1,
                "loss": float(loss.detach()),
                "ranking": float(ranking.detach()),
                "identity": float(identity.detach()),
                "bad_direction_correlation": float(correlation.detach()),
                "gradient_norm": float(gradient.detach()),
            })
    deltas = []
    for duration_index in (0, 1, 2):
        deltas.append(torch.matmul(
            b_params[duration_index], a_params[duration_index]
        ).detach().cpu().numpy().astype(np.float32))
    return deltas, {
        "seed": int(seed),
        "elapsed_seconds": time.perf_counter() - started,
        "trace": trace,
        "delta_frobenius": [float(np.linalg.norm(value)) for value in deltas],
    }


def transform_numpy(values, delta):
    values = np.asarray(values, dtype=np.float32)
    proposed = values + np.matmul(values, delta.T)
    norms = np.linalg.norm(proposed, axis=1, keepdims=True)
    if not np.isfinite(proposed).all() or float(norms.min()) <= 1.0e-8:
        raise RuntimeError("invalid short_metric transformed vectors")
    return np.ascontiguousarray(proposed / norms, dtype=np.float32)


def scored_rows(rows, deltas, alpha):
    arrays = row_arrays(rows)
    q_enrollment = transform_numpy(arrays["enrollment"], deltas[0])
    q_test = np.empty_like(arrays["test"])
    for duration_index in (0, 1, 2):
        mask = arrays["duration"] == duration_index
        q_test[mask] = transform_numpy(arrays["test"][mask], deltas[duration_index])
    expert = np.sum(q_enrollment * q_test, axis=1)
    scores = (1.0 - float(alpha)) * arrays["baseline"] + float(alpha) * expert
    output = []
    for row, score in zip(rows, scores):
        value = dict(row)
        for key in ("enrollment", "test"):
            value.pop(key, None)
        value["candidate_score"] = float(score)
        value["candidate_delta"] = float(score - row["baseline_score"])
        value["rejected_delta"] = float(
            row["rejected_mobile_score"] - row["baseline_score"]
        )
        output.append(value)
    return output


def aggregate_views(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["suite"], row["dataset"], row["base_scenario"],
            row["duration"], row["trial_id"],
        )
        grouped[key].append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        first = selected[0]
        output.append({
            "suite": key[0], "dataset": key[1], "base_scenario": key[2],
            "duration": key[3], "trial_id": key[4],
            "label": int(first["label"]),
            "enrollment_speaker_id": first["enrollment_speaker_id"],
            "test_speaker_id": first["test_speaker_id"],
            "baseline_score": float(np.mean([row["baseline_score"] for row in selected])),
            "candidate_score": float(np.mean([row["candidate_score"] for row in selected])),
            "candidate_delta": float(np.mean([row["candidate_delta"] for row in selected])),
            "rejected_delta": float(np.mean([row["rejected_delta"] for row in selected])),
        })
    return output


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or float(np.std(left)) <= 1.0e-12 or float(np.std(right)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def summarize(rows):
    aggregated = aggregate_views(rows)
    grouped = defaultdict(list)
    for row in aggregated:
        grouped[(
            row["suite"], row["dataset"], row["base_scenario"], row["duration"]
        )].append(row)
    cells = []
    for key, selected in sorted(grouped.items()):
        labels = [row["label"] for row in selected]
        baseline = compute_eer_percent(
            labels, [row["baseline_score"] for row in selected]
        )
        candidate = compute_eer_percent(
            labels, [row["candidate_score"] for row in selected]
        )
        cells.append({
            "suite": key[0], "dataset": key[1],
            "base_scenario": key[2], "duration": key[3],
            "trial_count": len(selected),
            "joint_eer_percent": baseline,
            "candidate_eer_percent": candidate,
            "improvement_points_vs_joint": baseline - candidate,
        })
    improvements = [cell["improvement_points_vs_joint"] for cell in cells]
    short = [
        cell["improvement_points_vs_joint"] for cell in cells
        if cell["duration"] != "full"
    ]
    by_dataset = {}
    for dataset in sorted({cell["dataset"] for cell in cells}):
        values = [
            cell["improvement_points_vs_joint"] for cell in cells
            if cell["dataset"] == dataset
        ]
        by_dataset[dataset] = float(np.mean(values))
    return {
        "macro": float(np.mean(improvements)),
        "short_macro": float(np.mean(short)),
        "worst": float(min(improvements)),
        "datasets": by_dataset,
        "rejected_mobile_delta_correlation": correlation(
            [row["candidate_delta"] for row in aggregated],
            [row["rejected_delta"] for row in aggregated],
        ),
        "cells": cells,
    }


def bootstrap(rows, repeats=1000, seed=826199):
    aggregated = aggregate_views(rows)
    speakers = sorted({
        row["enrollment_speaker_id"] for row in aggregated
    } | {
        row["test_speaker_id"] for row in aggregated
    })
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(repeats)):
        selected = set(rng.choice(speakers, size=len(speakers), replace=True).tolist())
        sample = [
            row for row in aggregated
            if row["enrollment_speaker_id"] in selected
            and row["test_speaker_id"] in selected
        ]
        groups = defaultdict(list)
        for row in sample:
            groups[(row["dataset"], row["base_scenario"], row["duration"])].append(row)
        cell_delta = []
        for selected_rows in groups.values():
            labels = [row["label"] for row in selected_rows]
            if len(set(labels)) != 2:
                continue
            base = compute_eer_percent(labels, [row["baseline_score"] for row in selected_rows])
            cand = compute_eer_percent(labels, [row["candidate_score"] for row in selected_rows])
            cell_delta.append(cand - base)
        if cell_delta:
            deltas.append(float(np.mean(cell_delta)))
    if len(deltas) < repeats // 2:
        raise RuntimeError("insufficient bootstrap replicates")
    return {
        "repeats_requested": int(repeats),
        "repeats_valid": len(deltas),
        "eer_delta_candidate_minus_joint_mean": float(np.mean(deltas)),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
    }


def main():
    torch.set_num_threads(4)
    development, audit, joint_receipt, mobile_receipt = load_rows()
    train, validation, dropped = split_rows(development, held_fold=0)
    print(
        "short_metric rows development={} train={} validation={} dropped={} audit={}".format(
            len(development), len(train), len(validation), dropped, len(audit)
        ),
        flush=True,
    )
    search = []
    fitted = {}
    for config_index, config in enumerate(CONFIGS, 1):
        print(
            "SHORT_METRIC CONFIG {}/{} {}".format(config_index, len(CONFIGS), config["id"]),
            flush=True,
        )
        seed_deltas = []
        audits = []
        for seed in SEEDS:
            deltas, fit_audit = train_one(train, config, seed)
            seed_deltas.append(deltas)
            audits.append(fit_audit)
            print(
                "  seed={} elapsed={:.2f}s fro={}".format(
                    seed, fit_audit["elapsed_seconds"],
                    [round(value, 4) for value in fit_audit["delta_frobenius"]],
                ),
                flush=True,
            )
        soup = [
            np.mean(np.stack([row[index] for row in seed_deltas]).astype(np.float64), axis=0).astype(np.float32)
            for index in range(3)
        ]
        fitted[config["id"]] = soup
        for alpha in ALPHAS:
            metrics = summarize(scored_rows(validation, soup, alpha))
            checks = {
                "macro_positive": metrics["macro"] > 0.0,
                "short_positive": metrics["short_macro"] > 0.0,
                "worst_ge_minus_0p2": metrics["worst"] >= -0.2,
                "rejected_direction_corr_le_0p15": metrics["rejected_mobile_delta_correlation"] <= 0.15,
            }
            search.append({
                "config": config,
                "alpha": float(alpha),
                "metrics": metrics,
                "checks": checks,
                "eligible": all(checks.values()),
                "fit_audits": audits,
            })
            print(
                "    alpha={:.2f} macro={:.6f} short={:.6f} worst={:.6f} corr={:.4f}".format(
                    alpha, metrics["macro"], metrics["short_macro"],
                    metrics["worst"], metrics["rejected_mobile_delta_correlation"],
                ),
                flush=True,
            )
    search.sort(key=lambda row: (
        row["eligible"], row["metrics"]["macro"],
        row["metrics"]["short_macro"], row["metrics"]["worst"],
        -row["alpha"],
    ), reverse=True)
    selected = next((row for row in search if row["eligible"]), search[0])
    selected_deltas = fitted[selected["config"]["id"]]

    # Open the frozen ChildMandarin dev audit only after selection is complete.
    audit_rows = scored_rows(audit, selected_deltas, selected["alpha"])
    audit_metrics = summarize(audit_rows)
    combined_bootstrap = bootstrap(
        scored_rows(validation, selected_deltas, selected["alpha"]) + audit_rows
    )
    release_checks = {
        "selection_macro_ge_1p2": selected["metrics"]["macro"] >= 1.2,
        "selection_short_ge_1p2": selected["metrics"]["short_macro"] >= 1.2,
        "selection_worst_ge_minus_0p15": selected["metrics"]["worst"] >= -0.15,
        "audit_macro_ge_0p8": audit_metrics["macro"] >= 0.8,
        "audit_short_ge_0p8": audit_metrics["short_macro"] >= 0.8,
        "audit_worst_ge_minus_0p15": audit_metrics["worst"] >= -0.15,
        "rejected_direction_corr_le_0p15": selected["metrics"]["rejected_mobile_delta_correlation"] <= 0.15,
        "bootstrap_ci95_upper_lt_zero": combined_bootstrap["ci95"][1] < 0.0,
    }
    status = (
        "RELEASE_GATE_PASS_REQUIRES_FULL_RETRAIN_RUNTIME"
        if all(release_checks.values())
        else "RELEASE_GATE_FAILED_RESEARCH_ONLY"
    )
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "status": np.asarray(status),
        "config_id": np.asarray(selected["config"]["id"]),
        "seed_ids": np.asarray(SEEDS, dtype=np.int64),
        "alpha": np.asarray(selected["alpha"], dtype=np.float64),
        "full_delta": np.ascontiguousarray(selected_deltas[0], dtype=np.float32),
        "short05_delta": np.ascontiguousarray(selected_deltas[1], dtype=np.float32),
        "short10_delta": np.ascontiguousarray(selected_deltas[2], dtype=np.float32),
    }
    atomic_npz(ADAPTER_PATH, arrays)
    payload = {
        "schema_version": 1,
        "artifact_kind": "short_metric_joint_anchored_hierarchical_cosine_metric_search",
        "status": status,
        "online_anchor": {"release": "joint-target-kd-strict", "score": 90.91262},
        "requested_online_target": 91.5,
        "hidden_score_guarantee": False,
        "joint_adapter_receipt": joint_receipt,
        "mobile_adapter_receipt": mobile_receipt,
        "speaker_split": {
            "method": "sha256 speaker id modulo 3",
            "held_fold": 0,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "cross_fold_rows_dropped": dropped,
        },
        "audit_policy": "ChildMandarin dev opened once after candidate selection",
        "selected": {
            "config": selected["config"],
            "alpha": selected["alpha"],
            "selection_metrics": selected["metrics"],
            "audit_metrics": audit_metrics,
            "bootstrap": combined_bootstrap,
            "adapter_frobenius": [float(np.linalg.norm(value)) for value in selected_deltas],
        },
        "release_checks": release_checks,
        "candidate_adapter": {
            "path": str(ADAPTER_PATH.relative_to(ROOT)),
            "sha256": sha256_file(ADAPTER_PATH),
        },
        "search": search,
        "limitations": [
            "The official hidden trials are unavailable, so no local result can guarantee 91.5.",
            "ChildMandarin dev and the coverage corpora were used in earlier project stages; this is a frozen post-selection audit, not a pristine external benchmark.",
            "No package may be described as release-qualified when status is RELEASE_GATE_FAILED_RESEARCH_ONLY.",
        ],
    }
    atomic_json(RESULT_PATH, payload)
    print(json.dumps({
        "status": status,
        "selected_config": selected["config"]["id"],
        "selected_alpha": selected["alpha"],
        "selection": {key: selected["metrics"][key] for key in (
            "macro", "short_macro", "worst", "rejected_mobile_delta_correlation"
        )},
        "audit": {key: audit_metrics[key] for key in ("macro", "short_macro", "worst")},
        "bootstrap": combined_bootstrap,
        "release_checks": release_checks,
        "result": str(RESULT_PATH.relative_to(ROOT)),
        "adapter": str(ADAPTER_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
