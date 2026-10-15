#!/usr/bin/env python3
"""LODO search for a distillation-anchored, duration-specific NAP auxiliary view.

This is an offline experiment.  It never edits the released distillation artifact and
never creates a submission package.  The nuisance bases are fitted only from
same-utterance short/full differences, with the currently scored dataset left
out.  The deployed candidate remains a concatenation of independently
unit-normalized blocks, so the platform's cosine scorer implements the tested
score fusion exactly.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_redim_output_kd_support import (
    MODEL_DIRS,
    l2,
    load_redim,
    load_trials,
    load_acoustic,
)
from tools.score_lodo_coverage import compute_eer_percent


OUTPUT_DIR = ROOT / "records/2026-09-05/033_joint_duration_nap_lodo"
RESULT_PATH = OUTPUT_DIR / "LODO_RESULT.json"
TRACE_PATH = OUTPUT_DIR / "SELECTED_SCORE_TRACE.jsonl"
ADAPTER_PATH = ROOT / "model/distillation_allsource_redim_kd_adapter.npz"
DISTILLATION_ENROLL = np.asarray([0.86, 0.133, 0.007], dtype=np.float64)
DISTILLATION_SHORT = np.asarray([0.20, 0.76, 0.04], dtype=np.float64)
HELD_DATASETS = ("3dspeaker", "aishell4", "childmandarin")
TRAIN_DATASETS = (
    "3dspeaker",
    "aishell4",
    "childmandarin",
    "seniortalk",
    "commonvoice_beijing",
    "magicdata_slr68",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2,
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


def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_distillation_adapters():
    output = {"lm": {}, "ptn": {}}
    with np.load(str(ADAPTER_PATH), allow_pickle=False) as source:
        for name in output:
            for duration in ("short05", "short10"):
                key = name + "_" + duration
                value = np.asarray(source[key], dtype=np.float32)
                if value.shape != (192, 192) or not np.isfinite(value).all():
                    raise ValueError("invalid distillation adapter {}".format(key))
                output[name][duration] = np.ascontiguousarray(value.copy())
    return output


def adapt(vector, delta):
    source = l2(vector)
    return l2(source + np.matmul(source, delta.T))


def distillation_score(enrollment, test, duration):
    test_weights = DISTILLATION_ENROLL if duration == "full" else DISTILLATION_SHORT
    return float(sum(
        math.sqrt(float(left * right)) * float(np.dot(a, b))
        for a, b, left, right in zip(enrollment, test, DISTILLATION_ENROLL, test_weights)
    ))


def fit_duration_nuisance(redim, delta, duration, held_out):
    """Fit dataset-balanced covariance of tangent short/full differences."""

    covariances = []
    audits = {}
    for dataset in TRAIN_DATASETS:
        if dataset == held_out:
            continue
        rows = []
        speakers = defaultdict(list)
        for utterance, row in redim.items():
            if row["dataset"] != dataset:
                continue
            speakers[row["speaker"]].append(utterance)
        for speaker in sorted(speakers):
            speaker_rows = []
            for utterance in speakers[speaker]:
                full = l2(redim[utterance]["full"])
                for short in redim[utterance][duration]:
                    proposed = adapt(short, delta)
                    difference = proposed - full
                    # Remove the local radial component; cosine scoring only
                    # uses tangent changes on the unit hypersphere.
                    difference = difference - float(np.dot(difference, full)) * full
                    norm = float(np.linalg.norm(difference))
                    if norm > 1.0e-8 and math.isfinite(norm):
                        speaker_rows.append(difference.astype(np.float64))
            if speaker_rows:
                rows.append(np.mean(np.stack(speaker_rows), axis=0))
        if len(rows) < 2:
            continue
        matrix = np.stack(rows).astype(np.float64)
        matrix -= np.mean(matrix, axis=0, keepdims=True)
        covariance = np.matmul(matrix.T, matrix) / float(max(1, len(matrix) - 1))
        trace = float(np.trace(covariance))
        if not math.isfinite(trace) or trace <= 1.0e-12:
            continue
        covariances.append(covariance / trace)
        audits[dataset] = {
            "utterances": sum(len(value) for value in speakers.values()),
            "speakers": len(speakers),
            "speaker_residual_rows": len(rows),
            "covariance_trace_before_normalization": trace,
        }
    if len(covariances) < 2:
        raise ValueError("insufficient non-held-out nuisance domains")
    covariance = np.mean(np.stack(covariances), axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return np.ascontiguousarray(eigenvectors, dtype=np.float64), {
        "held_out": held_out,
        "duration": duration,
        "training_datasets": sorted(audits),
        "datasets": audits,
        "top_eigenvalues": eigenvalues[:32].tolist(),
        "total_normalized_trace": float(np.sum(eigenvalues)),
    }


def project(vector, basis, rank, strength):
    source = l2(vector).astype(np.float64)
    directions = basis[:, :rank]
    output = source - float(strength) * np.matmul(
        directions, np.matmul(directions.T, source)
    )
    return l2(output.astype(np.float32))


def candidate_score(base, nap05, nap10, duration, enroll_extra, short_extra):
    if duration == "full":
        half = enroll_extra / 2.0
        return float((1.0 - enroll_extra) * base + half * nap05 + half * nap10)
    if duration == "0p5":
        return float(
            math.sqrt((1.0 - enroll_extra) * (1.0 - short_extra)) * base
            + math.sqrt((enroll_extra / 2.0) * short_extra) * nap05
        )
    if duration == "1p0":
        return float(
            math.sqrt((1.0 - enroll_extra) * (1.0 - short_extra)) * base
            + math.sqrt((enroll_extra / 2.0) * short_extra) * nap10
        )
    raise ValueError("unknown duration")


def component_trace(acoustic, acoustic_index, redims, adapters, trials, bases, rank, strength):
    rows = []
    for raw in trials:
        fold = str(raw["dataset"])
        enrollment_id = str(raw["enrollment_utterance_id"])
        test_id = str(raw["test_utterance_id"])
        enrollment_index = acoustic_index[enrollment_id][0]
        enrollment = [
            l2(acoustic["full"][enrollment_index]),
            redims["lm"][enrollment_id]["full"],
            redims["ptn"][enrollment_id]["full"],
        ]
        enrollment_nap = {
            key: project(enrollment[1], bases[fold][key], rank, strength)
            for key in ("short05", "short10")
        }
        for duration, acoustic_key, redim_key, nap_key in (
            ("full", "full", "full", None),
            ("0p5", "short05", "short05", "short05"),
            ("1p0", "short10", "short10", "short10"),
        ):
            indices = [acoustic_index[test_id][0]] if duration == "full" else acoustic_index[test_id]
            base_scores = []
            nap05_scores = []
            nap10_scores = []
            for view, index in enumerate(indices):
                if duration == "full":
                    lm = redims["lm"][test_id]["full"]
                    ptn = redims["ptn"][test_id]["full"]
                    lm05 = project(lm, bases[fold]["short05"], rank, strength)
                    lm10 = project(lm, bases[fold]["short10"], rank, strength)
                else:
                    lm = adapt(
                        redims["lm"][test_id][redim_key][view],
                        adapters["lm"][nap_key],
                    )
                    ptn = adapt(
                        redims["ptn"][test_id][redim_key][view],
                        adapters["ptn"][nap_key],
                    )
                    lm05 = project(lm, bases[fold]["short05"], rank, strength)
                    lm10 = project(lm, bases[fold]["short10"], rank, strength)
                test = [l2(acoustic[acoustic_key][index]), lm, ptn]
                base_scores.append(distillation_score(enrollment, test, duration))
                nap05_scores.append(float(np.dot(enrollment_nap["short05"], lm05)))
                nap10_scores.append(float(np.dot(enrollment_nap["short10"], lm10)))
            rows.append({
                "fold": fold,
                "base_scenario": str(raw["base_scenario"]),
                "duration": duration,
                "trial_id": str(raw.get("trial_id", "")),
                "label": int(raw["label"]),
                "enrollment_speaker_id": str(raw.get("enrollment_speaker_id", "")),
                "test_speaker_id": str(raw.get("test_speaker_id", "")),
                "distillation_score": float(np.mean(base_scores)),
                "nap05_score": float(np.mean(nap05_scores)),
                "nap10_score": float(np.mean(nap10_scores)),
            })
    return rows


def summarize(rows, enroll_extra, short05_extra, short10_extra, include_trace=False):
    scored = []
    for row in rows:
        short_extra = short05_extra if row["duration"] == "0p5" else short10_extra
        candidate = candidate_score(
            row["distillation_score"], row["nap05_score"], row["nap10_score"],
            row["duration"], enroll_extra, short_extra,
        )
        scored.append(dict(row, candidate_score=candidate))
    grouped = defaultdict(list)
    for row in scored:
        grouped[(row["fold"], row["base_scenario"], row["duration"])].append(row)
    cells = []
    for key, selected in sorted(grouped.items()):
        labels = [row["label"] for row in selected]
        baseline = compute_eer_percent(labels, [row["distillation_score"] for row in selected])
        candidate = compute_eer_percent(labels, [row["candidate_score"] for row in selected])
        cells.append({
            "fold": key[0], "base_scenario": key[1], "duration": key[2],
            "trial_count": len(selected), "distillation_eer_percent": baseline,
            "candidate_eer_percent": candidate,
            "improvement_points_vs_distillation": baseline - candidate,
        })
    shorts = [cell for cell in cells if cell["duration"] != "full"]
    folds = {
        fold: float(np.mean([
            cell["improvement_points_vs_distillation"] for cell in cells
            if cell["fold"] == fold
        ]))
        for fold in HELD_DATASETS
    }
    result = {
        "macro_improvement_points_vs_distillation": float(np.mean([
            cell["improvement_points_vs_distillation"] for cell in cells
        ])),
        "short_macro_improvement_points_vs_distillation": float(np.mean([
            cell["improvement_points_vs_distillation"] for cell in shorts
        ])),
        "worst_cell_improvement_points_vs_distillation": float(min(
            cell["improvement_points_vs_distillation"] for cell in cells
        )),
        "folds": folds,
        "cells": cells,
    }
    if include_trace:
        result["trace"] = scored
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", default="2,4,8,12,16,24")
    parser.add_argument("--strengths", default="0.25,0.5,0.75,1.0")
    args = parser.parse_args()
    ranks = tuple(int(value) for value in args.ranks.split(","))
    strengths = tuple(float(value) for value in args.strengths.split(","))
    if not ranks or min(ranks) < 1 or max(ranks) > 64:
        raise ValueError("rank grid is invalid")
    if not strengths or min(strengths) <= 0.0 or max(strengths) > 1.0:
        raise ValueError("strength grid is invalid")

    acoustic, acoustic_index, _metadata = load_acoustic()
    redims = {name: load_redim(MODEL_DIRS[name]) for name in ("lm", "ptn")}
    adapters = load_distillation_adapters()
    trials = load_trials()
    bases = {}
    audits = []
    for held_out in HELD_DATASETS:
        bases[held_out] = {}
        for duration in ("short05", "short10"):
            basis, audit = fit_duration_nuisance(
                redims["lm"], adapters["lm"][duration], duration, held_out
            )
            bases[held_out][duration] = basis
            audits.append(audit)

    grid = []
    component_cache = {}
    for rank in ranks:
        for strength in strengths:
            print("COMPONENTS rank={} strength={}".format(rank, strength), flush=True)
            components = component_trace(
                acoustic, acoustic_index, redims, adapters, trials, bases, rank, strength
            )
            component_cache[(rank, strength)] = components
            for enroll_extra in (0.01, 0.02, 0.04, 0.06, 0.08):
                for short05_extra in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
                    for short10_extra in (0.05, 0.10, 0.20, 0.30, 0.40):
                        metrics = summarize(
                            components, enroll_extra, short05_extra, short10_extra
                        )
                        config = {
                            "rank": rank, "strength": strength,
                            "enroll_extra_total": enroll_extra,
                            "short05_extra": short05_extra,
                            "short10_extra": short10_extra,
                        }
                        grid.append({"config": config, "metrics": metrics})

    ranked = sorted(grid, key=lambda row: (
        row["metrics"]["macro_improvement_points_vs_distillation"],
        row["metrics"]["short_macro_improvement_points_vs_distillation"],
        row["metrics"]["worst_cell_improvement_points_vs_distillation"],
        min(row["metrics"]["folds"].values()),
        -row["config"]["enroll_extra_total"],
        -row["config"]["short05_extra"],
        -row["config"]["short10_extra"],
    ), reverse=True)
    eligible = [row for row in ranked if (
        row["metrics"]["macro_improvement_points_vs_distillation"] >= 0.35
        and row["metrics"]["short_macro_improvement_points_vs_distillation"] >= 0.50
        and row["metrics"]["worst_cell_improvement_points_vs_distillation"] >= -0.20
        and all(value >= 0.0 for value in row["metrics"]["folds"].values())
    )]
    selected = eligible[0] if eligible else None
    selected_trace = []
    if selected is not None:
        config = selected["config"]
        selected_with_trace = summarize(
            component_cache[(config["rank"], config["strength"])],
            config["enroll_extra_total"], config["short05_extra"],
            config["short10_extra"], include_trace=True,
        )
        selected_trace = selected_with_trace.pop("trace")
        if selected_with_trace != selected["metrics"]:
            raise RuntimeError("selected metric reconstruction mismatch")
        atomic_jsonl(TRACE_PATH, selected_trace)
    report = {
        "schema_version": 1,
        "artifact_kind": "joint_duration_specific_nap_lodo_search",
        "status": "LODO_GATE_PASS" if selected else "LODO_GATE_FAILED_DO_NOT_PACKAGE",
        "research_only": True,
        "submission_created": False,
        "distillation_online_anchor": 89.76798,
        "target_online_score": 91.0,
        "method": {
            "fit": "dataset-balanced tangent covariance of same-utterance adapted-short minus full LM embeddings",
            "evaluation": "each coverage dataset uses nuisance bases fitted without that dataset",
            "deployment_geometry": "distillation block plus duration-specific projected LM blocks",
        },
        "inputs": {
            "adapter": {"path": str(ADAPTER_PATH.relative_to(ROOT)),
                        "sha256": sha256_file(ADAPTER_PATH)},
            "redim_cache_counts": {name: len(value) for name, value in redims.items()},
            "coverage_trial_count": len(trials),
        },
        "nuisance_fit_audits": audits,
        "grid": {
            "ranks": list(ranks), "strengths": list(strengths),
            "evaluated_points": len(grid),
            "enroll_extra_total": [0.01, 0.02, 0.04, 0.06, 0.08],
            "short05_extra": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            "short10_extra": [0.05, 0.10, 0.20, 0.30, 0.40],
        },
        "gate": {
            "macro_improvement_vs_distillation": 0.35,
            "short_macro_improvement_vs_distillation": 0.50,
            "worst_cell_improvement_vs_distillation": -0.20,
            "each_fold_improvement_vs_distillation": 0.0,
        },
        "eligible_count": len(eligible),
        "selected": selected,
        "top_20": ranked[:20],
        "limitations": [
            "All acoustic caches are previously consumed development data.",
            "LODO reduces but does not eliminate selection bias.",
            "No offline experiment can guarantee a hidden leaderboard score of 91.",
        ],
    }
    atomic_json(RESULT_PATH, report)
    print(json.dumps({
        "status": report["status"], "eligible_count": len(eligible),
        "best": ranked[0], "selected": selected,
        "result": str(RESULT_PATH), "result_sha256": sha256_file(RESULT_PATH),
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
