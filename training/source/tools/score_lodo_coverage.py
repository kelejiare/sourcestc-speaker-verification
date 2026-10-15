#!/usr/bin/env python3
"""Score frozen distillation LODO soups on held-out secondary coverage trials.

This is a diagnostic scorer, not the formal 15-scenario release evaluator.
For every held-out corpus, three independently trained seed matrices are
combined exactly as deployment does: float64 arithmetic mean followed by a
single float32 cast.  The held-out trials are never used to select or refit a
candidate.
"""

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_distillation_gates import EvaluationError, compute_eer_percent  # noqa: E402
from distillation_training.cache_schema import EMBEDDING_DIM, read_embedding_cache  # noqa: E402
from distillation_training.lodo_orchestrator import (  # noqa: E402
    DEFAULT_SEEDS,
    REQUIRED_DATASETS,
    _cosine,
    build_cache_view_index,
    load_fold_seed_artifacts,
    read_jsonl_objects,
    sha256_file,
)


SCHEMA_VERSION = 1
ARTIFACT_KIND = "distillation_lodo_secondary_coverage_soup_score"
DURATION_VARIANTS = (
    ("full", "full", None),
    ("0p5", "short05", "L05"),
    ("1p0", "short10", "L10"),
)


class LodoScoreError(ValueError):
    pass


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
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


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_deployment_soups(
    artifacts_by_holdout: Mapping[str, Mapping[int, Mapping[str, object]]]
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[Dict[str, object]]]:
    """Return per-fold float64-mean -> float32 deployment matrices."""

    soups = {}
    audits = []
    for heldout in REQUIRED_DATASETS:
        artifacts = artifacts_by_holdout[heldout]
        if tuple(sorted(artifacts)) != tuple(DEFAULT_SEEDS):
            raise LodoScoreError("fold {} seed set mismatch".format(heldout))
        matrices = {}
        for matrix_name in ("L05", "L10"):
            source = [np.asarray(artifacts[seed][matrix_name]) for seed in DEFAULT_SEEDS]
            if any(item.shape != (EMBEDDING_DIM, EMBEDDING_DIM) for item in source):
                raise LodoScoreError("{} {} matrix shape mismatch".format(heldout, matrix_name))
            if any(item.dtype != np.float32 for item in source):
                raise LodoScoreError("seed matrix must be float32")
            soup = np.mean(np.stack(source).astype(np.float64), axis=0).astype(np.float32)
            if not np.isfinite(soup).all():
                raise LodoScoreError("deployment soup contains non-finite values")
            matrices[matrix_name] = soup
        soups[heldout] = matrices
        audits.append(
            {
                "held_out_dataset": heldout,
                "seeds": list(DEFAULT_SEEDS),
                "combination_rule": "float64_arithmetic_mean_then_float32_once",
                "seed_artifact_sha256": {
                    str(seed): str(artifacts[seed]["sha256"]) for seed in DEFAULT_SEEDS
                },
                "L05_sha256": _array_sha256(matrices["L05"]),
                "L10_sha256": _array_sha256(matrices["L10"]),
            }
        )
    return soups, audits


def _validate_coverage_trials(
    trials: Sequence[Mapping[str, object]], cache_indices: Mapping[str, Mapping[str, object]]
) -> Dict[str, object]:
    if not trials:
        raise LodoScoreError("coverage trial input is empty")
    trial_ids = []
    bases = defaultdict(list)
    for line_number, row in enumerate(trials, 1):
        trial_id = str(row.get("trial_id", ""))
        heldout = str(row.get("held_out_dataset", ""))
        base = str(row.get("base_scenario", ""))
        enrollment = str(row.get("enrollment_utterance_id", ""))
        test = str(row.get("test_utterance_id", ""))
        if not trial_id or heldout not in REQUIRED_DATASETS or not base.startswith("coverage_"):
            raise LodoScoreError("invalid coverage trial at line {}".format(line_number))
        if row.get("evidence_scope") != "coverage_only":
            raise LodoScoreError("coverage evidence scope mismatch")
        if enrollment not in cache_indices[heldout] or test not in cache_indices[heldout]:
            raise LodoScoreError("trial endpoint not in its held-out cache")
        if cache_indices[heldout][enrollment]["speaker_id"] != row.get(
            "enrollment_speaker_id"
        ):
            raise LodoScoreError("enrollment speaker/cache mismatch")
        if cache_indices[heldout][test]["speaker_id"] != row.get("test_speaker_id"):
            raise LodoScoreError("test speaker/cache mismatch")
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise LodoScoreError("coverage label must be binary")
        if (
            str(row["enrollment_speaker_id"]) == str(row["test_speaker_id"])
        ) != bool(label):
            raise LodoScoreError("coverage label conflicts with speaker identity")
        trial_ids.append(trial_id)
        bases[(heldout, base)].append(label)
    if len(trial_ids) != len(set(trial_ids)):
        raise LodoScoreError("duplicate coverage trial_id")
    for key, labels in bases.items():
        if set(labels) != {0, 1}:
            raise LodoScoreError("{} lacks positive/negative trials".format(key))
    return {
        "trial_count": len(trials),
        "base_count": len(bases),
        "bases": [
            {
                "held_out_dataset": key[0],
                "base_scenario": key[1],
                "trial_count": len(labels),
                "positive_count": int(sum(labels)),
                "negative_count": int(len(labels) - sum(labels)),
            }
            for key, labels in sorted(bases.items())
        ],
    }


def score_trials(
    trials: Sequence[Mapping[str, object]],
    cache_indices: Mapping[str, Mapping[str, object]],
    cache_sha256: Mapping[str, str],
    soups: Mapping[str, Mapping[str, np.ndarray]],
    soup_audits: Sequence[Mapping[str, object]],
    trials_sha256: str,
) -> List[Dict[str, object]]:
    soup_hashes = {
        str(item["held_out_dataset"]): {
            "L05": str(item["L05_sha256"]), "L10": str(item["L10_sha256"])
        }
        for item in soup_audits
    }
    transformed = {}
    required_tests = defaultdict(set)
    for row in trials:
        required_tests[str(row["held_out_dataset"])].add(
            str(row["test_utterance_id"])
        )
    for heldout in REQUIRED_DATASETS:
        for _suffix, field, matrix_name in DURATION_VARIANTS[1:]:
            keys = []
            vectors = []
            for utterance_id in sorted(required_tests[heldout]):
                item = cache_indices[heldout][utterance_id]
                for view in item["views"]:
                    keys.append((heldout, str(view["embedding_id"]), field))
                    vectors.append(np.asarray(view[field], dtype=np.float32))
            if vectors:
                # One float64 BLAS call per fold/duration is substantially
                # faster than thousands of Python-dispatched matrix-vector
                # products and preserves the registered arithmetic precision.
                batch = np.stack(vectors).astype(np.float64)
                output_batch = np.matmul(
                    batch, soups[heldout][matrix_name].astype(np.float64).T
                )
                for key, output in zip(keys, output_batch):
                    transformed[key] = output

    trace = []
    for input_line, raw in enumerate(trials, 1):
        heldout = str(raw["held_out_dataset"])
        base = str(raw["base_scenario"])
        enrollment = cache_indices[heldout][str(raw["enrollment_utterance_id"])]
        test = cache_indices[heldout][str(raw["test_utterance_id"])]
        input_trial_sha = _json_sha256(raw)
        for suffix, field, matrix_name in DURATION_VARIANTS:
            scenario = "{}_{}".format(base, suffix)
            views = (
                [{"embedding_id": str(raw["test_utterance_id"]) + "@full", "full": test["full"]}]
                if matrix_name is None
                else test["views"]
            )
            for view in views:
                test_vector = np.asarray(view[field], dtype=np.float32)
                baseline_score = _cosine(enrollment["full"], test_vector)
                if matrix_name is None:
                    output = test_vector
                    candidate_score = baseline_score
                    matrix_sha = "identity_exact_acoustic"
                else:
                    output = transformed[(heldout, str(view["embedding_id"]), field)]
                    candidate_score = _cosine(enrollment["full"], output)
                    matrix_sha = soup_hashes[heldout][matrix_name]
                row = {
                    "trace_schema_version": SCHEMA_VERSION,
                    "fold": heldout,
                    "held_out_dataset": heldout,
                    "base_scenario": base,
                    "duration": suffix,
                    "scenario": scenario,
                    "trial_id": str(raw["trial_id"]),
                    "input_line": input_line,
                    "input_trial_sha256": input_trial_sha,
                    "input_trials_file_sha256": trials_sha256,
                    "label": int(raw["label"]),
                    "enrollment_speaker_id": str(raw["enrollment_speaker_id"]),
                    "test_speaker_id": str(raw["test_speaker_id"]),
                    "enrollment_utterance_id": str(raw["enrollment_utterance_id"]),
                    "test_utterance_id": str(raw["test_utterance_id"]),
                    "test_embedding_id": str(view["embedding_id"]),
                    "baseline_score": baseline_score,
                    "candidate_score": candidate_score,
                    "candidate_minus_baseline_score": candidate_score - baseline_score,
                    "held_out_cache_sha256": cache_sha256[heldout],
                    "deployment_matrix": matrix_name or "identity_exact_acoustic",
                    "deployment_matrix_sha256": matrix_sha,
                    "enrollment_embedding_sha256": _array_sha256(
                        np.asarray(enrollment["full"], dtype=np.float32)
                    ),
                    "test_input_embedding_sha256": _array_sha256(test_vector),
                    "candidate_output_embedding_sha256": _array_sha256(
                        np.asarray(output)
                    ),
                    "evidence_scope": "secondary_coverage_only",
                    "formal_gate_eligible": False,
                }
                row["trace_row_sha256"] = _json_sha256(row)
                trace.append(row)
    if not trace:
        raise LodoScoreError("scoring produced no trace")
    if any(
        row["duration"] == "full"
        and row["candidate_score"] != row["baseline_score"]
        for row in trace
    ):
        raise LodoScoreError("full-duration candidate is not exact acoustic identity")
    return trace


def _metric(
    rows: Sequence[Mapping[str, object]], weights: Optional[Sequence[float]] = None
) -> Dict[str, float]:
    labels = [int(row["label"]) for row in rows]
    baseline = compute_eer_percent(
        labels, [float(row["baseline_score"]) for row in rows], weights
    )
    candidate = compute_eer_percent(
        labels, [float(row["candidate_score"]) for row in rows], weights
    )
    return {
        "baseline_eer_percent": baseline,
        "candidate_eer_percent": candidate,
        "improvement_points": baseline - candidate,
        "delta_candidate_minus_baseline": candidate - baseline,
    }


def summarize_metrics(trace: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    cells = defaultdict(list)
    for row in trace:
        cells[(str(row["fold"]), str(row["base_scenario"]), str(row["duration"]))].append(row)
    cell_rows = []
    for (fold, base, duration), rows in sorted(cells.items()):
        metric = _metric(rows)
        metric.update(
            {
                "fold": fold,
                "base_scenario": base,
                "duration": duration,
                "trace_row_count": len(rows),
                "unique_trial_count": len({str(row["trial_id"]) for row in rows}),
            }
        )
        cell_rows.append(metric)

    fold_rows = []
    for fold in REQUIRED_DATASETS:
        selected = [row for row in cell_rows if row["fold"] == fold]
        short = [row for row in selected if row["duration"] != "full"]
        fold_rows.append(
            {
                "fold": fold,
                "cell_count": len(selected),
                "short_cell_count": len(short),
                "macro_baseline_eer_percent": float(
                    np.mean([float(row["baseline_eer_percent"]) for row in selected])
                ),
                "macro_candidate_eer_percent": float(
                    np.mean([float(row["candidate_eer_percent"]) for row in selected])
                ),
                "macro_improvement_points": float(
                    np.mean([float(row["improvement_points"]) for row in selected])
                ),
                "short_macro_improvement_points": float(
                    np.mean([float(row["improvement_points"]) for row in short])
                ),
            }
        )
    all_short = [row for row in cell_rows if row["duration"] != "full"]
    return {
        "cell_metrics": cell_rows,
        "fold_metrics": fold_rows,
        "coverage_macro": {
            "cell_count": len(cell_rows),
            "macro_improvement_points": float(
                np.mean([float(row["improvement_points"]) for row in cell_rows])
            ),
            "short_cell_count": len(all_short),
            "short_macro_improvement_points": float(
                np.mean([float(row["improvement_points"]) for row in all_short])
            ),
        },
    }


def _macro_delta(
    cells: Mapping[Tuple[str, str], Sequence[Mapping[str, object]]],
    weight_resolver,
) -> float:
    deltas = []
    for _key, rows in sorted(cells.items()):
        weights = [float(weight_resolver(row)) for row in rows]
        metric = _metric(rows, weights)
        deltas.append(metric["delta_candidate_minus_baseline"])
    return float(np.mean(deltas))


def _prepare_eer(labels: np.ndarray, scores: np.ndarray) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    starts = np.ones(len(order), dtype=np.bool_)
    if len(order) > 1:
        starts[1:] = ordered_scores[1:] != ordered_scores[:-1]
    groups = np.cumsum(starts, dtype=np.int64) - 1
    return {"order": order, "labels": ordered_labels, "groups": groups}


def _prepared_weighted_eer(
    prepared: Mapping[str, np.ndarray], weights: np.ndarray
) -> float:
    order = prepared["order"]
    labels = prepared["labels"]
    groups = prepared["groups"]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    group_count = int(groups[-1]) + 1
    positives = np.bincount(
        groups, weights=ordered_weights * (labels == 1), minlength=group_count
    )
    negatives = np.bincount(
        groups, weights=ordered_weights * (labels == 0), minlength=group_count
    )
    positive_total = float(positives.sum())
    negative_total = float(negatives.sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise EvaluationError("EER needs positive and negative weighted trials")
    fpr = np.cumsum(negatives) / negative_total
    fnr = (positive_total - np.cumsum(positives)) / positive_total
    delta = fnr - fpr
    crossings = np.flatnonzero(delta <= 0.0)
    if crossings.size == 0:
        return 100.0 * 0.5 * float(fpr[-1] + fnr[-1])
    index = int(crossings[0])
    if delta[index] == 0.0:
        return 100.0 * float(fpr[index])
    previous_fpr = 0.0 if index == 0 else float(fpr[index - 1])
    previous_fnr = 1.0 if index == 0 else float(fnr[index - 1])
    previous_delta = previous_fnr - previous_fpr
    fraction = previous_delta / (previous_delta - float(delta[index]))
    return 100.0 * float(
        previous_fpr + fraction * (float(fpr[index]) - previous_fpr)
    )


def _prepared_macro_delta(
    cells: Sequence[Mapping[str, object]], weights: np.ndarray
) -> float:
    deltas = []
    for cell in cells:
        indices = np.asarray(cell["indices"], dtype=np.int64)
        cell_weights = weights[indices]
        baseline = _prepared_weighted_eer(cell["baseline"], cell_weights)
        candidate = _prepared_weighted_eer(cell["candidate"], cell_weights)
        deltas.append(candidate - baseline)
    return float(np.mean(deltas))


def speaker_bootstrap(
    trace: Sequence[Mapping[str, object]],
    fold: str,
    repeats: int,
    seed: int,
    short_only: bool,
) -> Dict[str, object]:
    """Base-stratified paired enrollment/test/pigeonhole speaker bootstrap."""

    if repeats < 2000:
        raise LodoScoreError("LODO diagnostic requires at least 2000 bootstrap repeats")
    selected = [
        row
        for row in trace
        if row["fold"] == fold and (not short_only or row["duration"] != "full")
    ]
    cell_indices = defaultdict(list)
    for index, row in enumerate(selected):
        cell_indices[(str(row["base_scenario"]), str(row["duration"]))].append(index)
    cells = []
    for key, indices in sorted(cell_indices.items()):
        rows = [selected[index] for index in indices]
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
        cells.append(
            {
                "key": key,
                "indices": np.asarray(indices, dtype=np.int64),
                "baseline": _prepare_eer(
                    labels,
                    np.asarray([float(row["baseline_score"]) for row in rows]),
                ),
                "candidate": _prepare_eer(
                    labels,
                    np.asarray([float(row["candidate_score"]) for row in rows]),
                ),
            }
        )
    strata = sorted({str(row["base_scenario"]) for row in selected})
    enrollment = {}
    test = {}
    base_indices = {}
    enrollment_codes = {}
    test_codes = {}
    for base in strata:
        indices = np.asarray(
            [index for index, row in enumerate(selected) if row["base_scenario"] == base],
            dtype=np.int64,
        )
        base_indices[base] = indices
        enrollment[base], enrollment_codes[base] = np.unique(
            [str(selected[index]["enrollment_speaker_id"]) for index in indices],
            return_inverse=True,
        )
        test[base], test_codes[base] = np.unique(
            [str(selected[index]["test_speaker_id"]) for index in indices],
            return_inverse=True,
        )
    if any(len(enrollment[base]) < 2 or len(test[base]) < 2 for base in strata):
        raise LodoScoreError("each bootstrap stratum needs at least two speakers per endpoint")
    generator = np.random.RandomState(seed)
    values = {"enrollment_cluster": [], "test_cluster": [], "pigeonhole_two_way": []}
    attempts = 0
    maximum_attempts = repeats * 20
    while min(len(item) for item in values.values()) < repeats:
        attempts += 1
        if attempts > maximum_attempts:
            raise LodoScoreError(
                "could not obtain {} valid replicates within {} attempts".format(
                    repeats, maximum_attempts
                )
            )
        enrollment_weights = np.zeros(len(selected), dtype=np.float64)
        test_weights = np.zeros(len(selected), dtype=np.float64)
        for base in strata:
            ecount = np.bincount(
                generator.randint(0, len(enrollment[base]), size=len(enrollment[base])),
                minlength=len(enrollment[base]),
            )
            tcount = np.bincount(
                generator.randint(0, len(test[base]), size=len(test[base])),
                minlength=len(test[base]),
            )
            indices = base_indices[base]
            enrollment_weights[indices] = ecount[enrollment_codes[base]]
            test_weights[indices] = tcount[test_codes[base]]

        for name, weights in (
            ("enrollment_cluster", enrollment_weights),
            ("test_cluster", test_weights),
            ("pigeonhole_two_way", enrollment_weights * test_weights),
        ):
            if len(values[name]) >= repeats:
                continue
            try:
                values[name].append(_prepared_macro_delta(cells, weights))
            except EvaluationError:
                pass
    components = {}
    for name, deltas in values.items():
        if len(deltas) != repeats:
            raise LodoScoreError(
                "only {} valid {} bootstrap replicates; expected {}".format(
                    len(deltas), name, repeats
                )
            )
        array = np.asarray(deltas, dtype=np.float64)
        components[name] = {
            "valid_repeats": len(deltas),
            "delta_mean": float(np.mean(array)),
            "delta_ci95_lower": float(np.quantile(array, 0.025)),
            "delta_ci95_upper": float(np.quantile(array, 0.975)),
        }
    return {
        "fold": fold,
        "scope": "short_cells" if short_only else "all_cells_including_exact_acoustic_full",
        "strategy": "base_stratified_conservative_enrollment_test_pigeonhole_two_way_v1",
        "requested_repeats": repeats,
        "sampling_attempts": attempts,
        "invalid_draws_are_rejected_until_each_method_has_requested_repeats": True,
        "bootstrap_seed": seed,
        "component_intervals": components,
        "delta_ci95_lower": min(item["delta_ci95_lower"] for item in components.values()),
        "delta_ci95_upper": max(item["delta_ci95_upper"] for item in components.values()),
        "ci_upper_below_zero": bool(
            max(item["delta_ci95_upper"] for item in components.values()) < 0.0
        ),
        "seed_replicates_resampled_independently": False,
        "deployment_soup_scored_once": True,
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    folds_dir = args.folds_dir.resolve()
    training_dir = args.training_dir.resolve()
    trials_path = args.coverage_trials.resolve()
    output_dir = args.output_dir.resolve()
    cache_paths = {
        dataset: folds_dir / dataset / "cache_v2.npz" for dataset in REQUIRED_DATASETS
    }
    artifact_dirs = {dataset: training_dir / dataset for dataset in REQUIRED_DATASETS}
    artifacts, artifact_audits = load_fold_seed_artifacts(artifact_dirs, cache_paths)
    soups, soup_audits = build_deployment_soups(artifacts)
    cache_sha = {dataset: sha256_file(path) for dataset, path in cache_paths.items()}
    caches = {dataset: read_embedding_cache(path) for dataset, path in cache_paths.items()}
    indices = {dataset: build_cache_view_index(cache) for dataset, cache in caches.items()}
    trials = read_jsonl_objects(trials_path)
    trial_audit = _validate_coverage_trials(trials, indices)
    trace = score_trials(
        trials, indices, cache_sha, soups, soup_audits, sha256_file(trials_path)
    )
    trace_path = output_dir / "lodo_coverage_soup_scores.trace.jsonl"
    _atomic_jsonl(trace_path, trace)
    metrics = summarize_metrics(trace)
    bootstraps = []
    for index, fold in enumerate(REQUIRED_DATASETS):
        bootstraps.append(
            speaker_bootstrap(trace, fold, args.bootstrap_repeats, args.bootstrap_seed + index, False)
        )
        bootstraps.append(
            speaker_bootstrap(trace, fold, args.bootstrap_repeats, args.bootstrap_seed + index, True)
        )
    for item in metrics["fold_metrics"]:
        all_bootstrap = next(
            row
            for row in bootstraps
            if row["fold"] == item["fold"] and row["scope"].startswith("all_cells")
        )
        item["improvement_above_zero"] = bool(item["macro_improvement_points"] > 0.0)
        item["ci95_upper_below_zero"] = bool(all_bootstrap["ci_upper_below_zero"])
        item["diagnostic_fold_gate_pass"] = bool(
            item["improvement_above_zero"] and item["ci95_upper_below_zero"]
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": "SECONDARY_COVERAGE_DIAGNOSTIC_DO_NOT_SUBMIT",
        "candidate_selected_from_lodo_or_test": False,
        "deployment_rule": "per-fold three-seed float64 mean then float32 soup, scored once",
        "identity_baseline": "same held-out exact-acoustic cache; full candidate is bit-exact score assignment",
        "coverage_limitations": {
            "formal_15_scenario_gate_eligible": False,
            "official_3dspeaker_trials_are_cache_covered_subsets": True,
            "hidden_test_equivalence_claimed": False,
            "base_scenario_count": trial_audit["base_count"],
            "duration_cell_count": len(metrics["cell_metrics"]),
        },
        "inputs": {
            "coverage_trials": str(trials_path),
            "coverage_trials_sha256": sha256_file(trials_path),
            "fold_cache_sha256": cache_sha,
            "seed_artifacts": artifact_audits,
        },
        "trial_input_audit": trial_audit,
        "deployment_soups": soup_audits,
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "trace_row_count": len(trace),
        "metrics": metrics,
        "speaker_bootstrap": bootstraps,
        "all_folds_improve_and_ci_upper_below_zero": all(
            item["diagnostic_fold_gate_pass"] for item in metrics["fold_metrics"]
        ),
    }
    report_path = output_dir / "lodo_coverage_soup_score_report.json"
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    report["report_sha256"] = sha256_file(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-trials", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=8231101)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
