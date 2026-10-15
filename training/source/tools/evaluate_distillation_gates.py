#!/usr/bin/env python3
"""Strict, dependency-light release gates for distillation-domain-KD.

The evaluator consumes trial-level CSV or JSONL.  Required columns are::

    seed,scenario,speaker_id,dataset,domain,held_out_dataset,label,
    baseline_score,candidate_score,is_short,categories

``categories`` is a JSON list or a ``;``/``,``/``|`` separated string.  The
four guarded category names are ``child``, ``far``, ``same_gender`` and
``meeting``.  Every ``held_out_dataset`` value denotes one genuine LODO run;
the evaluator checks that its trials come only from that dataset.

EER, macro averages, and speaker-bootstrap confidence intervals are computed
with NumPy only (compatible with Python 3.8 / NumPy 1.24).  Synthetic or proxy
evidence can exercise the pipeline, but is never marked release-eligible.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for _path in (PROJECT_ROOT, TOOLS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from distillation_build_manifest import ManifestError, read_manifest, validate_manifest
from distillation_training.cache_schema import read_embedding_cache


TRIAL_FIELDS = (
    "seed",
    "scenario",
    "speaker_id",
    "dataset",
    "domain",
    "held_out_dataset",
    "label",
    "baseline_score",
    "candidate_score",
    "is_short",
    "categories",
)
CRITICAL_CATEGORIES = ("child", "far", "same_gender", "meeting")
BASELINE_SYSTEM = "acoustic-duration-adaptive-pt-campplus"
BASELINE_ONLINE_SCORE = 88.41131
GATE_SCHEMA_VERSION = 2
REAL_EVIDENCE_MANIFEST_KIND = "distillation_real_domain_matched_evidence_manifest"
REAL_SCORE_AUDIT_KIND = "distillation_real_domain_matched_lodo_score_audit"
REAL_EVIDENCE_SCOPE = "real_domain_matched"
DEPLOYMENT_SCORING_POLICY = (
    "three_seed_float64_matrix_mean_then_float32_scored_once_v1"
)
DEFAULT_RELEASE_SEEDS = (3407, 3408, 3409)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DEPLOYMENT_DIGEST_SEPARATOR = bytes.fromhex("67616d653832332d7631312d6c6f646f2d6465706c6f796d656e742d6d61747269782d763100")

FORMAL_TRIAL_PROVENANCE_FIELDS = (
    "trial_id",
    "base_scenario",
    "enrollment_speaker_id",
    "test_speaker_id",
    "enrollment_utterance_id",
    "test_embedding_id",
    "test_cache_field",
    "deployment_matrix_payload_sha256",
)


class EvaluationError(ValueError):
    """Raised when trial evidence is incomplete or statistically invalid."""


class VerifiedRealEvidence:
    """Opaque result of byte-level validation of a formal evidence bundle."""

    __slots__ = ("bindings", "metric_trials")

    _TOKEN = object()

    def __init__(self, token, bindings, metric_trials):
        if token is not self._TOKEN:
            raise EvaluationError(
                "VerifiedRealEvidence can only be created by evidence validation"
            )
        self.bindings = bindings
        self.metric_trials = metric_trials


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    text = str(value).strip().lower()
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise EvaluationError("{} must be 64 lowercase hex digits".format(label))
    return text


def _json_object(path: Path, label: str) -> Dict[str, object]:
    path = Path(path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise EvaluationError("{} must be a JSON object".format(label))
    return value


def _resolve_bound_path(owner: Path, value: object, label: str) -> Path:
    text = str(value).strip()
    if not text:
        raise EvaluationError("{} path must not be empty".format(label))
    path = Path(text)
    if not path.is_absolute():
        path = Path(owner).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise EvaluationError("{} file does not exist: {}".format(label, path))
    return path


def _validated_file_ref(owner: Path, value: object, label: str) -> Dict[str, object]:
    if (
        not isinstance(value, dict)
        or not {"path", "sha256"}.issubset(set(value))
        or not set(value).issubset({"path", "sha256", "bytes"})
    ):
        raise EvaluationError("{} must contain path/sha256 and optional bytes".format(label))
    path = _resolve_bound_path(owner, value["path"], label)
    declared = _require_sha256(value["sha256"], label + ".sha256")
    actual = sha256_file(path)
    if actual != declared:
        raise EvaluationError(
            "{} SHA-256 mismatch: declared={} actual={}".format(
                label, declared, actual
            )
        )
    size = path.stat().st_size
    if "bytes" in value:
        try:
            declared_size = int(value["bytes"])
        except (TypeError, ValueError):
            raise EvaluationError("{} bytes must be an integer".format(label))
        if declared_size != size:
            raise EvaluationError("{} byte count mismatch".format(label))
    return {"path": str(path), "sha256": actual, "bytes": size}


def _deployment_matrix_payload_sha256(
    held_out_dataset: str,
    seeds: Sequence[int],
    artifact_sha256s: Sequence[str],
    matrix_0p5: np.ndarray,
    matrix_1p0: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(_DEPLOYMENT_DIGEST_SEPARATOR)
    digest.update(str(held_out_dataset).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(np.ascontiguousarray(seeds, dtype="<i8").tobytes())
    for value in artifact_sha256s:
        digest.update(_require_sha256(value, "seed artifact SHA-256").encode("ascii"))
    digest.update(np.ascontiguousarray(matrix_0p5, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(matrix_1p0, dtype="<f4").tobytes())
    return digest.hexdigest()


def _parse_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    raise EvaluationError("{} must be a boolean, got {!r}".format(field, value))


def _parse_categories(value: object) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        text = str(value).strip()
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as error:
                raise EvaluationError("invalid categories JSON: {}".format(error))
            if not isinstance(loaded, list):
                raise EvaluationError("categories JSON must be a list")
            raw = [str(item) for item in loaded]
        else:
            normalized = text.replace("|", ";").replace(",", ";")
            raw = normalized.split(";")
    return tuple(sorted({item.strip().lower() for item in raw if item.strip()}))


def canonicalize_trial(record: Mapping[str, object]) -> Dict[str, object]:
    missing = [field for field in TRIAL_FIELDS if field not in record]
    if missing:
        raise EvaluationError("trial missing fields: {}".format(", ".join(missing)))
    result = {}  # type: Dict[str, object]
    for field in ("scenario", "speaker_id", "dataset", "domain", "held_out_dataset"):
        result[field] = str(record[field]).strip()
        if not result[field]:
            raise EvaluationError("trial field {!r} must not be empty".format(field))
    try:
        result["seed"] = int(record["seed"])
        result["label"] = int(record["label"])
        result["baseline_score"] = float(record["baseline_score"])
        result["candidate_score"] = float(record["candidate_score"])
    except (TypeError, ValueError) as error:
        raise EvaluationError("invalid numeric trial field: {}".format(error))
    if result["label"] not in (0, 1):
        raise EvaluationError("label must be 0 or 1")
    if not math.isfinite(result["baseline_score"]) or not math.isfinite(
        result["candidate_score"]
    ):
        raise EvaluationError("trial scores must be finite")
    result["is_short"] = _parse_bool(record["is_short"], "is_short")
    result["categories"] = _parse_categories(record["categories"])
    speaker_id = str(result["speaker_id"])
    dataset = str(result["dataset"])
    if not speaker_id.startswith(dataset + ":") or not speaker_id[len(dataset) + 1 :]:
        raise EvaluationError(
            "speaker_id {!r} must be namespaced as {!r}:<local-id>".format(
                speaker_id, dataset
            )
        )
    if result["held_out_dataset"] != dataset:
        raise EvaluationError(
            "LODO row dataset {!r} does not match held_out_dataset {!r}".format(
                dataset, result["held_out_dataset"]
            )
        )
    # These fields are mandatory only for formal real evidence.  Keeping them
    # here lets proxy/synthetic fixtures exercise the metric code while the
    # formal validator can still bind every score to both speaker endpoints,
    # the exact cache row, and the seed-averaged deployment matrix.
    for field in FORMAL_TRIAL_PROVENANCE_FIELDS:
        if field in record:
            value = str(record[field]).strip()
            if not value:
                raise EvaluationError("trial field {!r} must not be empty".format(field))
            result[field] = value
    return result


def read_trials(path: Path) -> List[Dict[str, object]]:
    path = Path(path)
    rows = []  # type: List[Mapping[str, object]]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise EvaluationError("trial CSV has no header")
            rows.extend(reader)
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise EvaluationError(
                        "invalid JSON at {}:{}: {}".format(path, line_number, error)
                    )
                if not isinstance(row, dict):
                    raise EvaluationError("trial JSONL rows must be objects")
                rows.append(row)
    else:
        raise EvaluationError("trials must end in .csv, .jsonl, or .ndjson")
    result = []
    for index, row in enumerate(rows, 2 if path.suffix.lower() == ".csv" else 1):
        try:
            result.append(canonicalize_trial(row))
        except EvaluationError as error:
            raise EvaluationError("{} row {}: {}".format(path, index, error))
    if not result:
        raise EvaluationError("trial file contains no rows")
    return result


def _canonical_seed_artifact_refs(
    owner: Path, values: object, expected_datasets: Sequence[str]
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[int, Dict[str, object]]]]:
    if not isinstance(values, list) or len(values) != 9:
        raise EvaluationError("formal evidence must bind exactly 3 folds x 3 seed artifacts")
    expected_cells = {
        (str(dataset), int(seed))
        for dataset in expected_datasets
        for seed in DEFAULT_RELEASE_SEEDS
    }
    if len(expected_datasets) != 3:
        raise EvaluationError("formal evidence requires exactly three LODO datasets")
    canonical = []
    artifacts = defaultdict(dict)
    seen = set()
    # Lazy import avoids making proxy-only evaluator use depend on the exporter.
    from distillation_training.export_adapter import load_seed_artifact

    for index, raw in enumerate(values):
        if not isinstance(raw, dict) or set(raw) != {
            "held_out_dataset",
            "seed",
            "path",
            "sha256",
        }:
            raise EvaluationError(
                "fold_seed_artifacts[{}] has an invalid schema".format(index)
            )
        held_out = str(raw["held_out_dataset"]).strip()
        try:
            seed = int(raw["seed"])
        except (TypeError, ValueError):
            raise EvaluationError("fold seed must be an integer")
        cell = (held_out, seed)
        if cell in seen:
            raise EvaluationError("duplicate fold/seed artifact {}".format(cell))
        seen.add(cell)
        ref = _validated_file_ref(
            owner,
            {"path": raw["path"], "sha256": raw["sha256"]},
            "fold_seed_artifacts[{}]".format(index),
        )
        try:
            loaded = load_seed_artifact(Path(ref["path"]))
        except (OSError, ValueError) as error:
            raise EvaluationError(
                "invalid fold/seed artifact {}: {}".format(cell, error)
            )
        if loaded["seed"] != seed or loaded["holdout_datasets"] != (held_out,):
            raise EvaluationError(
                "fold/seed artifact content does not match {}".format(cell)
            )
        item = {
            "held_out_dataset": held_out,
            "seed": seed,
            "path": ref["path"],
            "sha256": ref["sha256"],
            "bytes": ref["bytes"],
        }
        canonical.append(item)
        artifacts[held_out][seed] = loaded
    if seen != expected_cells:
        raise EvaluationError(
            "fold/seed artifact grid mismatch: expected={} actual={}".format(
                sorted(expected_cells), sorted(seen)
            )
        )
    candidate_signatures = {
        (
            str(artifact["candidate_id"]),
            str(artifact["candidate_family"]),
            int(artifact["candidate_rank"]),
            float(artifact["ridge_lambda"]),
            float(artifact["alpha"]),
            str(artifact["selection_protocol"]),
            str(artifact["selection_report_sha256"]),
            str(artifact["selection_cache_sha256"]),
        )
        for by_seed in artifacts.values()
        for artifact in by_seed.values()
    }
    if len(candidate_signatures) != 1:
        raise EvaluationError(
            "formal LODO artifacts did not freeze one global dev-selected candidate"
        )
    for held_out, by_seed in artifacts.items():
        for artifact in by_seed.values():
            if held_out in set(artifact["training_datasets"]):
                raise EvaluationError(
                    "fold seed artifact leaks held-out dataset into training_datasets"
                )
    canonical.sort(key=lambda item: (item["held_out_dataset"], item["seed"]))
    return canonical, dict(artifacts)


def _deployment_matrices(
    artifacts: Mapping[str, Mapping[int, Mapping[str, object]]]
) -> Dict[str, Dict[str, object]]:
    output = {}
    for held_out in sorted(artifacts):
        by_seed = artifacts[held_out]
        seeds = tuple(sorted(by_seed))
        if seeds != DEFAULT_RELEASE_SEEDS:
            raise EvaluationError("deployment matrix seed grid is incomplete")
        artifact_sha256s = [str(by_seed[seed]["sha256"]) for seed in seeds]
        matrix_0p5 = np.mean(
            np.stack([np.asarray(by_seed[seed]["L05"]) for seed in seeds]).astype(
                np.float64
            ),
            axis=0,
        ).astype(np.float32)
        matrix_1p0 = np.mean(
            np.stack([np.asarray(by_seed[seed]["L10"]) for seed in seeds]).astype(
                np.float64
            ),
            axis=0,
        ).astype(np.float32)
        if not np.isfinite(matrix_0p5).all() or not np.isfinite(matrix_1p0).all():
            raise EvaluationError("deployment matrix contains NaN/Inf")
        output[held_out] = {
            "seed_ids": list(seeds),
            "seed_artifact_sha256": artifact_sha256s,
            "payload_sha256": _deployment_matrix_payload_sha256(
                held_out, seeds, artifact_sha256s, matrix_0p5, matrix_1p0
            ),
            "L05": matrix_0p5,
            "L10": matrix_1p0,
        }
    return output


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not np.isfinite(denominator) or denominator < 1.0e-12:
        raise EvaluationError("cannot score zero-norm embedding")
    value = float(np.dot(left, right) / denominator)
    if not np.isfinite(value):
        raise EvaluationError("recomputed score is not finite")
    return value


def _verify_deployment_scores(
    trials: Sequence[Mapping[str, object]],
    cache: Mapping[str, np.ndarray],
    deployments: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Recompute scores using each fold's actual three-seed mean matrix."""

    by_embedding = {
        str(embedding_id): index
        for index, embedding_id in enumerate(cache["embedding_id"])
    }
    by_utterance = defaultdict(list)
    for index, utterance_id in enumerate(cache["utterance_id"]):
        by_utterance[str(utterance_id)].append(index)
    groups = defaultdict(list)
    score_tolerance = 1.0e-10
    for row in trials:
        missing = [field for field in FORMAL_TRIAL_PROVENANCE_FIELDS if field not in row]
        if missing:
            raise EvaluationError(
                "formal real trial is missing provenance fields: {}".format(
                    ", ".join(missing)
                )
            )
        held_out = str(row["held_out_dataset"])
        if held_out not in deployments:
            raise EvaluationError("trial references unknown LODO deployment {}".format(held_out))
        expected_payload = str(deployments[held_out]["payload_sha256"])
        if row["deployment_matrix_payload_sha256"] != expected_payload:
            raise EvaluationError("trial deployment matrix payload SHA-256 mismatch")
        base = str(row["base_scenario"])
        scenario = str(row["scenario"])
        suffix = scenario[len(base) + 1 :] if scenario.startswith(base + "_") else ""
        expected_fields = {"full": "full", "0p5": "short05", "1p0": "short10"}
        if suffix not in expected_fields or row["test_cache_field"] != expected_fields[suffix]:
            raise EvaluationError("scenario/base/cache-field provenance is inconsistent")
        enrollment_id = str(row["enrollment_utterance_id"])
        test_embedding_id = str(row["test_embedding_id"])
        if enrollment_id not in by_utterance or test_embedding_id not in by_embedding:
            raise EvaluationError("formal trial endpoint is absent from the combined cache")
        enrollment_indices = by_utterance[enrollment_id]
        enrollment_vectors = [cache["full"][index] for index in enrollment_indices]
        enrollment = enrollment_vectors[0]
        if any(not np.array_equal(enrollment, item) for item in enrollment_vectors[1:]):
            raise EvaluationError("enrollment full embedding differs across crop views")
        enrollment_speakers = {
            str(cache["speaker_id"][index]) for index in enrollment_indices
        }
        if enrollment_speakers != {str(row["enrollment_speaker_id"])}:
            raise EvaluationError("enrollment speaker/cache provenance mismatch")
        test_index = by_embedding[test_embedding_id]
        if str(cache["speaker_id"][test_index]) != str(row["test_speaker_id"]):
            raise EvaluationError("test speaker/cache provenance mismatch")
        if str(row["speaker_id"]) != str(row["test_speaker_id"]):
            raise EvaluationError("speaker_id must equal the test endpoint speaker")
        test_vector = cache[str(row["test_cache_field"])][test_index]
        baseline = _cosine(enrollment, test_vector)
        if suffix == "full":
            candidate = baseline
        else:
            matrix_name = "L05" if suffix == "0p5" else "L10"
            transformed = np.matmul(
                np.asarray(deployments[held_out][matrix_name], dtype=np.float64),
                np.asarray(test_vector, dtype=np.float64),
            )
            candidate = _cosine(enrollment, transformed)
        if abs(float(row["baseline_score"]) - baseline) > score_tolerance:
            raise EvaluationError("stored baseline score does not match exact cache recomputation")
        if abs(float(row["candidate_score"]) - candidate) > score_tolerance:
            raise EvaluationError(
                "stored candidate score was not produced by the three-seed mean matrix"
            )
        key = (
            held_out,
            str(row["trial_id"]),
            scenario,
            test_embedding_id,
        )
        groups[key].append(dict(row))

    collapsed = []
    for key, rows in sorted(groups.items()):
        seeds = sorted(int(row["seed"]) for row in rows)
        if tuple(seeds) != DEFAULT_RELEASE_SEEDS:
            raise EvaluationError(
                "deployment score {} must be repeated for exactly three seeds".format(key)
            )
        invariant_fields = tuple(TRIAL_FIELDS[1:]) + FORMAL_TRIAL_PROVENANCE_FIELDS
        first = rows[0]
        for row in rows[1:]:
            if any(row.get(field) != first.get(field) for field in invariant_fields):
                raise EvaluationError(
                    "seed rows differ even though deployment matrix is seed-averaged"
                )
        collapsed.append(first)
    if not collapsed:
        raise EvaluationError("formal deployment score set is empty")
    return collapsed


def validate_real_evidence_bundle(
    trials_path: Path,
    evidence_manifest_path: Path,
    score_audit_path: Path,
    trials: Sequence[Mapping[str, object]],
) -> VerifiedRealEvidence:
    """Authenticate a formal real-domain score bundle and recompute scores."""

    trials_path = Path(trials_path).resolve()
    evidence_manifest_path = Path(evidence_manifest_path).resolve()
    score_audit_path = Path(score_audit_path).resolve()
    manifest = _json_object(evidence_manifest_path, "evidence manifest")
    score_audit = _json_object(score_audit_path, "score audit")
    for payload, kind, label in (
        (manifest, REAL_EVIDENCE_MANIFEST_KIND, "evidence manifest"),
        (score_audit, REAL_SCORE_AUDIT_KIND, "score audit"),
    ):
        if payload.get("schema_version") != GATE_SCHEMA_VERSION:
            raise EvaluationError("{} schema_version must be {}".format(label, GATE_SCHEMA_VERSION))
        if payload.get("artifact_kind") != kind:
            raise EvaluationError("{} artifact_kind is not formal real evidence".format(label))
        if payload.get("evidence_scope") != REAL_EVIDENCE_SCOPE:
            raise EvaluationError("{} evidence_scope is proxy/coverage or unknown".format(label))
        if payload.get("formal_release_eligible") is not True:
            raise EvaluationError("{} is not explicitly formal-release eligible".format(label))

    manifest_sha = sha256_file(evidence_manifest_path)
    if _require_sha256(
        score_audit.get("evidence_manifest_sha256"),
        "score_audit.evidence_manifest_sha256",
    ) != manifest_sha:
        raise EvaluationError("score audit does not bind the exact evidence manifest")
    trial_ref = _validated_file_ref(
        evidence_manifest_path, manifest.get("trial_records"), "trial_records"
    )
    if Path(trial_ref["path"]) != trials_path:
        raise EvaluationError("evidence manifest binds a different trial file")
    if _require_sha256(
        score_audit.get("trial_records_sha256"), "score_audit.trial_records_sha256"
    ) != trial_ref["sha256"]:
        raise EvaluationError("score audit trial SHA-256 does not match the evidence manifest")
    cache_ref = _validated_file_ref(
        evidence_manifest_path, manifest.get("combined_cache"), "combined_cache"
    )
    if _require_sha256(
        score_audit.get("combined_cache_sha256"),
        "score_audit.combined_cache_sha256",
    ) != cache_ref["sha256"]:
        raise EvaluationError("score audit combined-cache SHA-256 mismatch")
    lodo_manifest_values = manifest.get("lodo_manifests")
    if not isinstance(lodo_manifest_values, list) or len(lodo_manifest_values) != 3:
        raise EvaluationError("evidence manifest must bind exactly three LODO manifests")
    lodo_manifest_refs = []
    lodo_manifest_datasets = set()
    for index, item in enumerate(lodo_manifest_values):
        if not isinstance(item, dict) or set(item) != {
            "held_out_dataset",
            "path",
            "sha256",
        }:
            raise EvaluationError("lodo_manifests[{}] schema is invalid".format(index))
        dataset = str(item["held_out_dataset"]).strip()
        if not dataset or dataset in lodo_manifest_datasets:
            raise EvaluationError("LODO manifest held-out dataset is empty/duplicate")
        lodo_manifest_datasets.add(dataset)
        ref = _validated_file_ref(
            evidence_manifest_path,
            {"path": item["path"], "sha256": item["sha256"]},
            "lodo_manifests[{}]".format(index),
        )
        lodo_manifest_refs.append(
            {"held_out_dataset": dataset, **ref}
        )
    if score_audit.get("deployment_scoring_policy") != DEPLOYMENT_SCORING_POLICY:
        raise EvaluationError("score audit did not use the formal deployment matrix policy")

    datasets = sorted({str(row["held_out_dataset"]) for row in trials})
    if lodo_manifest_datasets != set(datasets):
        raise EvaluationError("LODO manifest bindings do not cover trial datasets")
    refs, artifacts = _canonical_seed_artifact_refs(
        evidence_manifest_path, manifest.get("fold_seed_artifacts"), datasets
    )
    selection_cache_hashes = {
        str(artifact["selection_cache_sha256"])
        for by_seed in artifacts.values()
        for artifact in by_seed.values()
    }
    if selection_cache_hashes != {cache_ref["sha256"]}:
        raise EvaluationError(
            "fold seed artifacts were not selected on the bound combined cache"
        )
    audit_refs = score_audit.get("fold_seed_artifacts")
    if not isinstance(audit_refs, list):
        raise EvaluationError("score audit fold_seed_artifacts must be a list")
    normalized_audit_refs = []
    for item in audit_refs:
        if not isinstance(item, dict):
            raise EvaluationError("score audit seed artifact entry must be an object")
        try:
            seed = int(item.get("seed", -1))
        except (TypeError, ValueError):
            raise EvaluationError("score audit seed artifact seed is invalid")
        normalized_audit_refs.append(
            {
                "held_out_dataset": str(item.get("held_out_dataset", "")),
                "seed": seed,
                "sha256": _require_sha256(
                    item.get("sha256"), "score audit seed artifact SHA-256"
                ),
            }
        )
    normalized_audit_refs.sort(
        key=lambda item: (item["held_out_dataset"], item["seed"])
    )
    normalized_manifest_refs = [
        {
            "held_out_dataset": item["held_out_dataset"],
            "seed": item["seed"],
            "sha256": item["sha256"],
        }
        for item in refs
    ]
    if normalized_audit_refs != normalized_manifest_refs:
        raise EvaluationError("score audit does not bind the exact 3x3 seed artifact grid")

    deployments = _deployment_matrices(artifacts)
    declared_deployments = score_audit.get("deployment_matrices")
    if not isinstance(declared_deployments, dict) or set(declared_deployments) != set(datasets):
        raise EvaluationError("score audit deployment_matrices does not cover all folds")
    deployment_bindings = {}
    for dataset in datasets:
        declared = declared_deployments[dataset]
        if not isinstance(declared, dict) or set(declared) != {
            "seed_ids",
            "seed_artifact_sha256",
            "payload_sha256",
        }:
            raise EvaluationError("deployment matrix audit schema is invalid")
        actual = deployments[dataset]
        if declared["seed_ids"] != actual["seed_ids"]:
            raise EvaluationError("deployment matrix seed IDs mismatch")
        if declared["seed_artifact_sha256"] != actual["seed_artifact_sha256"]:
            raise EvaluationError("deployment matrix seed artifact SHA-256 mismatch")
        if _require_sha256(
            declared["payload_sha256"], "deployment matrix payload SHA-256"
        ) != actual["payload_sha256"]:
            raise EvaluationError("deployment matrix payload SHA-256 mismatch")
        deployment_bindings[dataset] = {
            "seed_ids": actual["seed_ids"],
            "seed_artifact_sha256": actual["seed_artifact_sha256"],
            "payload_sha256": actual["payload_sha256"],
        }

    try:
        cache = read_embedding_cache(
            Path(cache_ref["path"]), allow_v1_migration=False
        )
    except (OSError, ValueError) as error:
        raise EvaluationError("invalid formal combined cache: {}".format(error))
    metric_trials = _verify_deployment_scores(trials, cache, deployments)
    if int(score_audit.get("input_trial_row_count", -1)) != len(trials):
        raise EvaluationError("score audit input_trial_row_count mismatch")
    if int(score_audit.get("deployment_trial_count", -1)) != len(metric_trials):
        raise EvaluationError("score audit deployment_trial_count mismatch")
    bindings = {
        "validation_policy": "byte_bound_real_evidence_and_score_recomputation_v1",
        "evidence_manifest": {
            "path": str(evidence_manifest_path),
            "sha256": manifest_sha,
            "artifact_kind": REAL_EVIDENCE_MANIFEST_KIND,
        },
        "score_audit": {
            "path": str(score_audit_path),
            "sha256": sha256_file(score_audit_path),
            "artifact_kind": REAL_SCORE_AUDIT_KIND,
        },
        "trials": trial_ref,
        "combined_cache": cache_ref,
        "lodo_manifests": sorted(
            lodo_manifest_refs, key=lambda item: item["held_out_dataset"]
        ),
        "fold_seed_artifacts": refs,
        "deployment_matrices": deployment_bindings,
        "deployment_scoring_policy": DEPLOYMENT_SCORING_POLICY,
        "input_trial_row_count": len(trials),
        "deployment_trial_count": len(metric_trials),
    }
    return VerifiedRealEvidence(VerifiedRealEvidence._TOKEN, bindings, metric_trials)


def revalidate_gate_evidence_bindings(value: object) -> Dict[str, object]:
    """Re-hash every external byte source named by a schema-v2 gate."""

    if not isinstance(value, dict):
        raise EvaluationError("gate evidence_bindings must be an object")
    if value.get("validation_policy") != (
        "byte_bound_real_evidence_and_score_recomputation_v1"
    ):
        raise EvaluationError("gate evidence validation policy is not formal")
    if value.get("deployment_scoring_policy") != DEPLOYMENT_SCORING_POLICY:
        raise EvaluationError("gate deployment scoring policy is not formal")
    checked = {}
    for key, expected_kind in (
        ("evidence_manifest", REAL_EVIDENCE_MANIFEST_KIND),
        ("score_audit", REAL_SCORE_AUDIT_KIND),
    ):
        item = value.get(key)
        if not isinstance(item, dict) or item.get("artifact_kind") != expected_kind:
            raise EvaluationError("gate {} artifact kind is invalid".format(key))
        checked[key] = _validated_file_ref(
            Path(str(item.get("path", "."))),
            {"path": item.get("path"), "sha256": item.get("sha256")},
            "gate evidence_bindings.{}".format(key),
        )
        payload = _json_object(Path(checked[key]["path"]), key)
        if (
            payload.get("artifact_kind") != expected_kind
            or payload.get("evidence_scope") != REAL_EVIDENCE_SCOPE
            or payload.get("formal_release_eligible") is not True
        ):
            raise EvaluationError("bound {} no longer contains formal real evidence".format(key))
    for key in ("trials", "combined_cache"):
        item = value.get(key)
        checked[key] = _validated_file_ref(
            Path(str(item.get("path", "."))) if isinstance(item, dict) else Path("."),
            item,
            "gate evidence_bindings.{}".format(key),
        )
    bound_manifest_payload = _json_object(
        Path(checked["evidence_manifest"]["path"]), "evidence manifest"
    )
    bound_score_payload = _json_object(
        Path(checked["score_audit"]["path"]), "score audit"
    )
    if _require_sha256(
        bound_score_payload.get("evidence_manifest_sha256"),
        "bound score audit evidence_manifest_sha256",
    ) != checked["evidence_manifest"]["sha256"]:
        raise EvaluationError("bound score audit does not bind the evidence manifest")
    if _require_sha256(
        bound_score_payload.get("trial_records_sha256"),
        "bound score audit trial_records_sha256",
    ) != checked["trials"]["sha256"]:
        raise EvaluationError("bound score audit does not bind the trial bytes")
    if _require_sha256(
        bound_score_payload.get("combined_cache_sha256"),
        "bound score audit combined_cache_sha256",
    ) != checked["combined_cache"]["sha256"]:
        raise EvaluationError("bound score audit does not bind the cache bytes")
    for field, checked_key in (
        ("trial_records", "trials"),
        ("combined_cache", "combined_cache"),
    ):
        item = bound_manifest_payload.get(field)
        if not isinstance(item, dict) or _require_sha256(
            item.get("sha256"), "bound manifest {} SHA-256".format(field)
        ) != checked[checked_key]["sha256"]:
            raise EvaluationError(
                "bound evidence manifest does not bind {}".format(field)
            )
    lodo_refs = value.get("lodo_manifests")
    if not isinstance(lodo_refs, list) or len(lodo_refs) != 3:
        raise EvaluationError("gate must bind exactly three LODO manifests")
    checked_lodo = []
    for index, item in enumerate(lodo_refs):
        if not isinstance(item, dict):
            raise EvaluationError("gate LODO manifest binding is invalid")
        ref = _validated_file_ref(
            Path(str(item.get("path", "."))),
            {"path": item.get("path"), "sha256": item.get("sha256")},
            "gate lodo_manifests[{}]".format(index),
        )
        checked_lodo.append(
            {"held_out_dataset": str(item.get("held_out_dataset", "")), **ref}
        )
    if len({item["held_out_dataset"] for item in checked_lodo}) != 3:
        raise EvaluationError("gate LODO manifest held-out bindings are not unique")
    manifest_lodo = bound_manifest_payload.get("lodo_manifests")
    if not isinstance(manifest_lodo, list):
        raise EvaluationError("bound evidence manifest lacks LODO manifest bindings")
    expected_lodo = sorted(
        (item["held_out_dataset"], item["sha256"]) for item in checked_lodo
    )
    declared_lodo = sorted(
        (
            str(item.get("held_out_dataset", "")),
            _require_sha256(item.get("sha256"), "manifest LODO SHA-256"),
        )
        for item in manifest_lodo
        if isinstance(item, dict)
    )
    if declared_lodo != expected_lodo:
        raise EvaluationError("bound evidence manifest LODO bindings mismatch")
    refs = value.get("fold_seed_artifacts")
    if not isinstance(refs, list) or len(refs) != 9:
        raise EvaluationError("gate must bind exactly nine fold/seed artifacts")
    cells = set()
    checked_refs = []
    for index, item in enumerate(refs):
        if not isinstance(item, dict):
            raise EvaluationError("gate seed artifact binding is invalid")
        held_out = str(item.get("held_out_dataset", ""))
        try:
            seed = int(item.get("seed"))
        except (TypeError, ValueError):
            raise EvaluationError("gate seed artifact binding has invalid seed")
        cell = (held_out, seed)
        if cell in cells:
            raise EvaluationError("gate contains duplicate fold/seed binding")
        cells.add(cell)
        ref = _validated_file_ref(
            Path(str(item.get("path", "."))),
            {"path": item.get("path"), "sha256": item.get("sha256")},
            "gate fold_seed_artifacts[{}]".format(index),
        )
        checked_refs.append(
            {
                "held_out_dataset": held_out,
                "seed": seed,
                "path": ref["path"],
                "sha256": ref["sha256"],
                "bytes": ref["bytes"],
            }
        )
    datasets = sorted({cell[0] for cell in cells})
    expected = {
        (dataset, seed) for dataset in datasets for seed in DEFAULT_RELEASE_SEEDS
    }
    if len(datasets) != 3 or cells != expected:
        raise EvaluationError("gate fold/seed binding is not a complete 3x3 grid")
    expected_seed_refs = sorted([
        (item["held_out_dataset"], item["seed"], item["sha256"])
        for item in checked_refs
    ])
    for payload, label in (
        (bound_manifest_payload, "evidence manifest"),
        (bound_score_payload, "score audit"),
    ):
        raw_refs = payload.get("fold_seed_artifacts")
        if not isinstance(raw_refs, list):
            raise EvaluationError("bound {} lacks fold/seed bindings".format(label))
        declared_refs = sorted(
            (
                str(item.get("held_out_dataset", "")),
                int(item.get("seed", -1)),
                _require_sha256(
                    item.get("sha256"), "{} seed artifact SHA-256".format(label)
                ),
            )
            for item in raw_refs
            if isinstance(item, dict)
        )
        if declared_refs != expected_seed_refs:
            raise EvaluationError("bound {} fold/seed bindings mismatch".format(label))
    deployments = value.get("deployment_matrices")
    if not isinstance(deployments, dict) or set(deployments) != set(datasets):
        raise EvaluationError("gate deployment-matrix bindings are incomplete")
    for dataset, item in deployments.items():
        if (
            not isinstance(item, dict)
            or item.get("seed_ids") != list(DEFAULT_RELEASE_SEEDS)
            or len(item.get("seed_artifact_sha256", [])) != 3
        ):
            raise EvaluationError("gate deployment-matrix binding is invalid")
        _require_sha256(item.get("payload_sha256"), "deployment payload SHA-256")
        expected_shas = [
            ref["sha256"]
            for ref in checked_refs
            if ref["held_out_dataset"] == dataset
        ]
        if item["seed_artifact_sha256"] != expected_shas:
            raise EvaluationError("deployment binding seed-artifact SHA-256 mismatch")
    if bound_score_payload.get("deployment_scoring_policy") != DEPLOYMENT_SCORING_POLICY:
        raise EvaluationError("bound score audit deployment policy mismatch")
    if bound_score_payload.get("deployment_matrices") != deployments:
        raise EvaluationError("bound score audit deployment-matrix bindings mismatch")
    checked["fold_seed_artifacts"] = checked_refs
    checked["lodo_manifests"] = checked_lodo
    checked["deployment_matrices"] = deployments
    return checked


def compute_eer_percent(
    labels: Sequence[int], scores: Sequence[float], weights: Optional[Sequence[float]] = None
) -> float:
    """Compute linearly interpolated EER while grouping tied scores."""

    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise EvaluationError("labels and scores must be one-dimensional")
    if labels_array.size != scores_array.size or labels_array.size == 0:
        raise EvaluationError("labels and scores must have equal non-zero length")
    if not np.all(np.isfinite(scores_array)):
        raise EvaluationError("scores must be finite")
    if not np.all((labels_array == 0) | (labels_array == 1)):
        raise EvaluationError("labels must be binary")
    if weights is None:
        weight_array = np.ones(labels_array.size, dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != scores_array.shape:
            raise EvaluationError("weights must have the same shape as scores")
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
            raise EvaluationError("weights must be finite and non-negative")
    positive_total = float(weight_array[labels_array == 1].sum())
    negative_total = float(weight_array[labels_array == 0].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise EvaluationError("EER needs positive and negative weighted trials")

    order = np.argsort(-scores_array, kind="mergesort")
    ordered_scores = scores_array[order]
    ordered_labels = labels_array[order]
    ordered_weights = weight_array[order]
    false_positive = 0.0
    true_positive = 0.0
    previous_fpr = 0.0
    previous_fnr = 1.0
    index = 0
    while index < ordered_scores.size:
        end = index + 1
        while end < ordered_scores.size and ordered_scores[end] == ordered_scores[index]:
            end += 1
        group_labels = ordered_labels[index:end]
        group_weights = ordered_weights[index:end]
        true_positive += float(group_weights[group_labels == 1].sum())
        false_positive += float(group_weights[group_labels == 0].sum())
        fpr = false_positive / negative_total
        fnr = (positive_total - true_positive) / positive_total
        previous_delta = previous_fnr - previous_fpr
        delta = fnr - fpr
        if delta == 0.0:
            return 100.0 * fpr
        if previous_delta >= 0.0 and delta < 0.0:
            fraction = previous_delta / (previous_delta - delta)
            eer = previous_fpr + fraction * (fpr - previous_fpr)
            return 100.0 * float(eer)
        previous_fpr = fpr
        previous_fnr = fnr
        index = end
    return 100.0 * 0.5 * (previous_fpr + previous_fnr)


def _cell_key(row: Mapping[str, object]) -> Tuple[int, str, str]:
    return (int(row["seed"]), str(row["held_out_dataset"]), str(row["scenario"]))


def _base_scenario_for_row(row: Mapping[str, object]) -> str:
    scenario = str(row["scenario"])
    declared = str(row.get("base_scenario", "")).strip()
    if declared:
        if not scenario.startswith(declared + "_"):
            raise EvaluationError("scenario is not namespaced by base_scenario")
        suffix = scenario[len(declared) + 1 :]
        if suffix not in ("full", "0p5", "1p0"):
            raise EvaluationError("formal scenario suffix must be full, 0p5, or 1p0")
        return declared
    for suffix in ("_full", "_0p5", "_1p0"):
        if scenario.endswith(suffix) and len(scenario) > len(suffix):
            return scenario[: -len(suffix)]
    return scenario


def _selected_cells(
    trials: Sequence[Mapping[str, object]],
    scenario_filter: Optional[Set[str]] = None,
    held_out_filter: Optional[str] = None,
) -> Dict[Tuple[int, str, str], List[Mapping[str, object]]]:
    cells = defaultdict(list)  # type: Dict[Tuple[int, str, str], List[Mapping[str, object]]]
    for row in trials:
        if scenario_filter is not None and row["scenario"] not in scenario_filter:
            continue
        if held_out_filter is not None and row["held_out_dataset"] != held_out_filter:
            continue
        cells[_cell_key(row)].append(row)
    if not cells:
        raise EvaluationError("metric selection contains no trial cells")
    return dict(cells)


def _metric_from_cells(
    cells: Mapping[Tuple[int, str, str], Sequence[Mapping[str, object]]],
    row_weight=None,
) -> Dict[str, float]:
    baseline_values = []
    candidate_values = []
    for (seed, held_out, _scenario), rows in sorted(cells.items()):
        labels = [int(row["label"]) for row in rows]
        weights = None
        if row_weight is not None:
            weights = [float(row_weight(row)) for row in rows]
        baseline_values.append(
            compute_eer_percent(labels, [float(row["baseline_score"]) for row in rows], weights)
        )
        candidate_values.append(
            compute_eer_percent(labels, [float(row["candidate_score"]) for row in rows], weights)
        )
    baseline = float(np.mean(baseline_values))
    candidate = float(np.mean(candidate_values))
    return {
        "baseline_eer_percent": baseline,
        "candidate_eer_percent": candidate,
        "improvement_points": baseline - candidate,
        "delta_candidate_minus_baseline": candidate - baseline,
        "cell_count": len(cells),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    # ``method`` was introduced after the required NumPy 1.24 environment;
    # ``interpolation`` remains supported there and on newer NumPy versions.
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def speaker_bootstrap_delta(
    trials: Sequence[Mapping[str, object]],
    cells: Mapping[Tuple[int, str, str], Sequence[Mapping[str, object]]],
    repeats: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    """Conservative two-ended speaker-cluster/pigeonhole bootstrap.

    Training seeds are repeated measurements of the same held-out speakers,
    not independent observations.  Therefore one speaker resample is drawn per
    held-out dataset and reused across all seeds.  Enrollment and test speaker
    clusters are sampled independently.  Release uses the largest upper bound
    of enrollment-only, test-only and their pigeonhole product.
    """

    if repeats < 20:
        raise EvaluationError("speaker bootstrap needs at least 20 repeats")
    enrollment_speakers = defaultdict(set)  # type: Dict[str, Set[str]]
    test_speakers = defaultdict(set)  # type: Dict[str, Set[str]]
    selected_rows = [row for rows in cells.values() for row in rows]
    for row in selected_rows:
        held_out = str(row["held_out_dataset"])
        enrollment_speakers[held_out].add(
            str(row.get("enrollment_speaker_id", row["speaker_id"]))
        )
        test_speakers[held_out].add(
            str(row.get("test_speaker_id", row["speaker_id"]))
        )
    for endpoint, groups in (
        ("enrollment", enrollment_speakers),
        ("test", test_speakers),
    ):
        for key, values in groups.items():
            if len(values) < 2:
                raise EvaluationError(
                    "{} speaker bootstrap group {} needs at least two speakers".format(
                        endpoint, key
                    )
                )
    if set(enrollment_speakers) != set(test_speakers):
        raise EvaluationError("bootstrap endpoint held-out groups do not match")
    ordered_enrollment = {
        key: sorted(values) for key, values in enrollment_speakers.items()
    }
    ordered_test = {key: sorted(values) for key, values in test_speakers.items()}
    generator = np.random.RandomState(bootstrap_seed)
    method_deltas = {
        "enrollment_cluster": [],
        "test_cluster": [],
        "pigeonhole_two_way": [],
    }
    for _ in range(repeats):
        enrollment_counters = {}
        test_counters = {}
        for held_out in sorted(ordered_enrollment):
            enroll_values = ordered_enrollment[held_out]
            test_values = ordered_test[held_out]
            enrollment_counters[held_out] = Counter(
                str(value)
                for value in generator.choice(
                    enroll_values, size=len(enroll_values), replace=True
                )
            )
            test_counters[held_out] = Counter(
                str(value)
                for value in generator.choice(
                    test_values, size=len(test_values), replace=True
                )
            )

        def enrollment_weight(row):
            return enrollment_counters[str(row["held_out_dataset"])][
                str(row.get("enrollment_speaker_id", row["speaker_id"]))
            ]

        def test_weight(row):
            return test_counters[str(row["held_out_dataset"])][
                str(row.get("test_speaker_id", row["speaker_id"]))
            ]

        def pigeonhole_weight(row):
            return enrollment_weight(row) * test_weight(row)

        for method, resolver in (
            ("enrollment_cluster", enrollment_weight),
            ("test_cluster", test_weight),
            ("pigeonhole_two_way", pigeonhole_weight),
        ):
            try:
                metric = _metric_from_cells(cells, resolver)
            except EvaluationError:
                continue
            method_deltas[method].append(metric["delta_candidate_minus_baseline"])

    minimum_valid = max(10, int(math.ceil(0.8 * repeats)))
    component_intervals = {}
    for method, deltas in method_deltas.items():
        if len(deltas) < minimum_valid:
            raise EvaluationError(
                "only {} of {} {} bootstrap replicates were valid".format(
                    len(deltas), repeats, method
                )
            )
        component_intervals[method] = {
            "valid_repeats": len(deltas),
            "delta_mean": float(np.mean(deltas)),
            "delta_ci95_lower": _quantile(deltas, 0.025),
            "delta_ci95_upper": _quantile(deltas, 0.975),
        }
    conservative_lower = min(
        value["delta_ci95_lower"] for value in component_intervals.values()
    )
    conservative_upper = max(
        value["delta_ci95_upper"] for value in component_intervals.values()
    )
    return {
        "unit": "both enrollment and test speaker clusters, shared across seeds",
        "strategy": "conservative_enrollment_test_pigeonhole_two_way_v1",
        "requested_repeats": repeats,
        "valid_repeats": min(
            value["valid_repeats"] for value in component_intervals.values()
        ),
        "seed": bootstrap_seed,
        "delta_mean": max(
            value["delta_mean"] for value in component_intervals.values()
        ),
        "delta_ci95_lower": conservative_lower,
        "delta_ci95_upper": conservative_upper,
        "component_intervals": component_intervals,
        "seed_replicates_resampled_independently": False,
    }


def validate_trial_design(
    trials: Sequence[Mapping[str, object]], expected_scenarios: int, expected_short: int
) -> Dict[str, object]:
    seeds = sorted({int(row["seed"]) for row in trials})
    scenarios = sorted({str(row["scenario"]) for row in trials})
    held_out = sorted({str(row["held_out_dataset"]) for row in trials})
    datasets = sorted({str(row["dataset"]) for row in trials})
    domains = sorted({str(row["domain"]) for row in trials})
    if len(seeds) < 3:
        raise EvaluationError("at least three independent training seeds are required")
    if len(scenarios) != expected_scenarios:
        raise EvaluationError(
            "expected {} scenarios, got {}".format(expected_scenarios, len(scenarios))
        )
    metadata = {}
    for scenario in scenarios:
        rows = [row for row in trials if row["scenario"] == scenario]
        flags = {bool(row["is_short"]) for row in rows}
        category_sets = {tuple(row["categories"]) for row in rows}
        held_out_sets = {str(row["held_out_dataset"]) for row in rows}
        dataset_sets = {str(row["dataset"]) for row in rows}
        domain_sets = {str(row["domain"]) for row in rows}
        if (
            len(flags) != 1
            or len(category_sets) != 1
            or len(held_out_sets) != 1
            or len(dataset_sets) != 1
            or len(domain_sets) != 1
        ):
            raise EvaluationError(
                "scenario {!r} has inconsistent duration/category/domain provenance".format(
                    scenario
                )
            )
        metadata[scenario] = {
            "is_short": next(iter(flags)),
            "categories": list(next(iter(category_sets))),
            "held_out_dataset": next(iter(held_out_sets)),
            "dataset": next(iter(dataset_sets)),
            "domain": next(iter(domain_sets)),
            "base_scenario": _base_scenario_for_row(rows[0]),
        }
    short_scenarios = sorted(
        scenario for scenario in scenarios if metadata[scenario]["is_short"]
    )
    if len(short_scenarios) != expected_short:
        raise EvaluationError(
            "expected {} short scenarios, got {}".format(expected_short, len(short_scenarios))
        )
    if len(datasets) < 2 or len(domains) < 2:
        raise EvaluationError("at least two datasets and two domains are required")
    if set(held_out) != set(datasets):
        raise EvaluationError(
            "LODO folds {} do not cover datasets {}".format(held_out, datasets)
        )
    present_categories = set()
    for value in metadata.values():
        present_categories.update(value["categories"])
    missing_categories = sorted(set(CRITICAL_CATEGORIES).difference(present_categories))
    if missing_categories:
        raise EvaluationError(
            "15-scenario design is missing guarded categories: {}".format(
                ", ".join(missing_categories)
            )
        )

    # The contract is fifteen scenarios in total, not fifteen scenarios per
    # held-out corpus.  A scenario is bound to one real source domain (for
    # example ChildMandarin/child, 3D-Speaker/far or AISHELL-4/meeting), then
    # evaluated for every training seed in the corresponding LODO fold.  A
    # Cartesian fold x scenario requirement would either multiply the stated
    # contract or force false labels such as marking adult speech as child.
    expected_cells = {
        (seed, metadata[scenario]["held_out_dataset"], scenario)
        for seed in seeds
        for scenario in scenarios
    }
    actual_cells = {_cell_key(row) for row in trials}
    missing_cells = sorted(expected_cells.difference(actual_cells))
    extra_cells = sorted(actual_cells.difference(expected_cells))
    if missing_cells or extra_cells:
        raise EvaluationError(
            "incomplete seed/LODO/scenario grid: missing={}, extra={}".format(
                missing_cells[:3], extra_cells[:3]
            )
        )
    for key, rows in _selected_cells(trials).items():
        labels = {int(row["label"]) for row in rows}
        if labels != {0, 1}:
            raise EvaluationError("trial cell {} lacks positive or negative trials".format(key))
    return {
        "seeds": seeds,
        "seed_count": len(seeds),
        "scenarios": scenarios,
        "scenario_count": len(scenarios),
        "short_scenarios": short_scenarios,
        "short_scenario_count": len(short_scenarios),
        "datasets": datasets,
        "domains": domains,
        "lodo_folds": held_out,
        "speaker_count": len({str(row["speaker_id"]) for row in trials}),
        "trial_count": len(trials),
        "scenario_metadata": metadata,
        "base_scenarios": sorted(
            {value["base_scenario"] for value in metadata.values()}
        ),
    }


def _gate(
    name: str,
    passed: bool,
    observed: object,
    comparison: str,
    threshold: object,
    required: bool = True,
) -> Dict[str, object]:
    return {
        "name": name,
        "required": required,
        "passed": bool(passed),
        "observed": observed,
        "comparison": comparison,
        "threshold": threshold,
    }


def evaluate_gates(
    trials: Sequence[Mapping[str, object]],
    evidence_label: str,
    expected_scenarios: int = 15,
    expected_short: int = 10,
    bootstrap_repeats: int = 1000,
    bootstrap_seed: int = 82311,
    macro_min_improvement: float = 2.0,
    macro_target_improvement: float = 2.6,
    short_min_improvement: float = 3.0,
    max_category_regression: float = 0.2,
    manifest_audits: Optional[Sequence[Mapping[str, object]]] = None,
    verified_evidence: Optional[VerifiedRealEvidence] = None,
) -> Dict[str, object]:
    if evidence_label not in ("synthetic", "proxy", "real"):
        raise EvaluationError("evidence_label must be synthetic, proxy, or real")
    canonical = [canonicalize_trial(row) for row in trials]
    design = validate_trial_design(canonical, expected_scenarios, expected_short)
    if verified_evidence is not None and not isinstance(
        verified_evidence, VerifiedRealEvidence
    ):
        raise EvaluationError("verified_evidence has not been authenticated")
    if verified_evidence is not None and evidence_label != "real":
        raise EvaluationError("authenticated real evidence cannot be relabeled")
    metric_trials = (
        list(verified_evidence.metric_trials)
        if verified_evidence is not None
        else canonical
    )
    all_cells = _selected_cells(metric_trials)
    macro = _metric_from_cells(all_cells)
    macro_bootstrap = speaker_bootstrap_delta(
        metric_trials, all_cells, bootstrap_repeats, bootstrap_seed
    )

    short_set = set(design["short_scenarios"])
    short_cells = _selected_cells(metric_trials, scenario_filter=short_set)
    short = _metric_from_cells(short_cells)

    categories = {}
    category_bootstraps = {}
    for index, category in enumerate(CRITICAL_CATEGORIES):
        scenario_set = {
            name
            for name, metadata in design["scenario_metadata"].items()
            if category in metadata["categories"]
        }
        cells = _selected_cells(metric_trials, scenario_filter=scenario_set)
        categories[category] = _metric_from_cells(cells)
        category_bootstraps[category] = speaker_bootstrap_delta(
            metric_trials,
            cells,
            bootstrap_repeats,
            bootstrap_seed + index + 1,
        )

    by_seed = {}
    for seed in design["seeds"]:
        rows = [row for row in canonical if row["seed"] == seed]
        by_seed[str(seed)] = _metric_from_cells(_selected_cells(rows))
    lodo = {}
    for held_out in design["lodo_folds"]:
        cells = _selected_cells(metric_trials, held_out_filter=held_out)
        lodo[held_out] = _metric_from_cells(cells)

    base_scenarios = {}
    for base in design["base_scenarios"]:
        scenario_set = {
            name
            for name, metadata in design["scenario_metadata"].items()
            if metadata["base_scenario"] == base
        }
        base_scenarios[base] = _metric_from_cells(
            _selected_cells(metric_trials, scenario_filter=scenario_set)
        )
    formal_base_shape = len(base_scenarios) == 5 and all(
        {
            name[len(base) + 1 :]
            for name in design["scenarios"]
            if name.startswith(base + "_")
        }
        == {"full", "0p5", "1p0"}
        for base in base_scenarios
    )

    audits = list(manifest_audits or [])
    audited_holdouts = {
        str(audit.get("lodo_held_out_dataset"))
        for audit in audits
        if audit.get("lodo_held_out_dataset")
    }
    manifest_gate_passed = (
        len(audits) >= 2
        and audited_holdouts == set(design["datasets"])
        and all(bool(audit.get("speaker_disjoint")) for audit in audits)
        and all(bool(audit.get("domain_disjoint")) for audit in audits)
    )
    if verified_evidence is not None:
        bound_manifests = {
            (
                str(item["held_out_dataset"]),
                str(item["sha256"]),
            )
            for item in verified_evidence.bindings["lodo_manifests"]
        }
        audited_manifests = {
            (
                str(item.get("lodo_held_out_dataset", "")),
                str(item.get("sha256", "")),
            )
            for item in audits
        }
        manifest_gate_passed = manifest_gate_passed and (
            audited_manifests == bound_manifests
        )
    gates = [
        _gate(
            "formal_real_evidence_bundle_authenticated",
            verified_evidence is not None,
            None
            if verified_evidence is None
            else verified_evidence.bindings["validation_policy"],
            "==",
            "byte_bound_real_evidence_and_score_recomputation_v1",
            required=evidence_label == "real",
        ),
        _gate(
            "deployment_matrix_scored_after_three_seed_mean",
            verified_evidence is not None,
            None
            if verified_evidence is None
            else verified_evidence.bindings["deployment_scoring_policy"],
            "==",
            DEPLOYMENT_SCORING_POLICY,
            required=evidence_label == "real",
        ),
        _gate(
            "five_base_scenarios_each_full_0p5_1p0",
            formal_base_shape,
            {
                name: sorted(
                    scenario[len(name) + 1 :]
                    for scenario in design["scenarios"]
                    if scenario.startswith(name + "_")
                )
                for name in base_scenarios
            },
            "==",
            "5 base scenarios x {full,0p5,1p0}",
            required=evidence_label == "real",
        ),
        _gate(
            "speaker_domain_disjoint_lodo_manifests",
            manifest_gate_passed,
            {
                "audit_count": len(audits),
                "held_out_datasets": sorted(audited_holdouts),
                "expected_datasets": design["datasets"],
            },
            "==",
            "two or more audited folds covering every dataset",
        ),
        _gate("training_seed_count", design["seed_count"] >= 3, design["seed_count"], ">=", 3),
        _gate(
            "dataset_and_domain_count",
            len(design["datasets"]) >= 2 and len(design["domains"]) >= 2,
            {"datasets": len(design["datasets"]), "domains": len(design["domains"])},
            ">=",
            {"datasets": 2, "domains": 2},
        ),
        _gate(
            "macro_eer_improvement_points",
            macro["improvement_points"] >= macro_min_improvement,
            macro["improvement_points"],
            ">=",
            macro_min_improvement,
        ),
        _gate(
            "macro_eer_target_points",
            macro["improvement_points"] >= macro_target_improvement,
            macro["improvement_points"],
            ">=",
            macro_target_improvement,
            required=evidence_label == "real",
        ),
        _gate(
            "ten_short_scenarios_improvement_points",
            short["improvement_points"] >= short_min_improvement,
            short["improvement_points"],
            ">=",
            short_min_improvement,
        ),
        _gate(
            "speaker_bootstrap_delta_ci95_upper",
            macro_bootstrap["delta_ci95_upper"] < 0.0,
            macro_bootstrap["delta_ci95_upper"],
            "<",
            0.0,
        ),
        _gate(
            "all_lodo_folds_improve",
            all(value["improvement_points"] > 0.0 for value in lodo.values()),
            {name: value["improvement_points"] for name, value in lodo.items()},
            ">",
            0.0,
        ),
    ]
    for category in CRITICAL_CATEGORIES:
        metric = categories[category]
        bootstrap = category_bootstraps[category]
        # A regression is statistically significant only when the entire
        # candidate-minus-baseline CI is above zero.
        gates.append(
            _gate(
                "{}_no_significant_regression".format(category),
                bootstrap["delta_ci95_lower"] <= 0.0,
                bootstrap["delta_ci95_lower"],
                "<=",
                0.0,
            )
        )
        gates.append(
            _gate(
                "{}_point_regression_limit".format(category),
                metric["delta_candidate_minus_baseline"] <= max_category_regression,
                metric["delta_candidate_minus_baseline"],
                "<=",
                max_category_regression,
            )
        )

    for base, metric in sorted(base_scenarios.items()):
        gates.append(
            _gate(
                "base_scenario_{}_point_regression_limit".format(base),
                metric["delta_candidate_minus_baseline"]
                <= max_category_regression,
                metric["delta_candidate_minus_baseline"],
                "<=",
                max_category_regression,
            )
        )

    required_checks = {
        str(gate["name"]): bool(gate["passed"])
        for gate in gates
        if bool(gate["required"])
    }
    required_pass = bool(required_checks) and all(required_checks.values())
    release_eligible = (
        evidence_label == "real"
        and verified_evidence is not None
        and required_pass
    )
    if release_eligible:
        status = "PASS"
    elif required_pass:
        status = "METRICS_PASS_BUT_NONREAL_EVIDENCE"
    else:
        status = "FAIL"
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "system": "distillation-domain-KD",
        "baseline_system": BASELINE_SYSTEM,
        "baseline_online_score": BASELINE_ONLINE_SCORE,
        "status": status,
        "evidence_label": evidence_label,
        "overall_pass": release_eligible,
        "release_eligible": release_eligible,
        "all_required_gates_passed": required_pass,
        "checks": required_checks,
        "warning": (
            "Synthetic/proxy evidence is pipeline validation only and cannot release a submission."
            if evidence_label != "real"
            else "The real label was not backed by an authenticated evidence bundle."
            if verified_evidence is None
            else None
        ),
        "evidence_bindings": (
            None if verified_evidence is None else verified_evidence.bindings
        ),
        "metric_policy": (
            "actual_three_seed_mean_deployment_matrix_scores_once"
            if verified_evidence is not None
            else "nonrelease_seed_cell_diagnostic"
        ),
        "design": design,
        "manifest_audits": audits,
        "metrics": {
            "macro_15_scenarios": macro,
            "short_10_scenarios": short,
            "by_seed": by_seed,
            "by_lodo_held_out_dataset": lodo,
            "guarded_categories": categories,
            "by_base_scenario": base_scenarios,
            "max_base_scenario_regression_points": max(
                value["delta_candidate_minus_baseline"]
                for value in base_scenarios.values()
            ),
        },
        "bootstrap": {
            "macro_15_scenarios": macro_bootstrap,
            "guarded_categories": category_bootstraps,
        },
        "gates": gates,
    }


def _audit_manifests(paths: Sequence[Path]) -> List[Dict[str, object]]:
    audits = []
    held_out_names = set()
    for path in paths:
        path = Path(path).resolve()
        records = read_manifest(path, namespace_speakers=True)
        test_datasets = {row["dataset"] for row in records if row["split"] == "test"}
        train_datasets = {
            row["dataset"] for row in records if row["split"] in ("train", "dev")
        }
        if len(test_datasets) != 1:
            raise EvaluationError(
                "LODO manifest {} must have exactly one test dataset".format(path)
            )
        held_out = next(iter(test_datasets))
        if held_out in train_datasets:
            raise EvaluationError(
                "LODO manifest {} leaks held-out dataset into train/dev".format(path)
            )
        audit = validate_manifest(
            records,
            require_speaker_disjoint=True,
            require_domain_disjoint=True,
            lodo_held_out_dataset=held_out,
        )
        audit["path"] = str(path)
        audit["sha256"] = sha256_file(path)
        audits.append(audit)
        held_out_names.add(held_out)
    if paths and len(held_out_names) < 2:
        raise EvaluationError("at least two LODO manifests with different holdouts are required")
    return audits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--evidence-label", choices=("synthetic", "proxy", "real"), required=True
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        help="schema-v2 byte-bound real-domain evidence manifest (required for real)",
    )
    parser.add_argument(
        "--score-audit",
        type=Path,
        help="schema-v2 real-domain deployment-score audit (required for real)",
    )
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--expected-scenarios", type=int, default=15)
    parser.add_argument("--expected-short", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=82311)
    parser.add_argument("--macro-min-improvement", type=float, default=2.0)
    parser.add_argument("--macro-target-improvement", type=float, default=2.6)
    parser.add_argument("--short-min-improvement", type=float, default=3.0)
    parser.add_argument("--max-category-regression", type=float, default=0.2)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        audits = _audit_manifests(arguments.manifest)
        trials = read_trials(arguments.trials)
        verified_evidence = None
        if arguments.evidence_label == "real":
            if arguments.evidence_manifest is None or arguments.score_audit is None:
                raise EvaluationError(
                    "real evidence requires --evidence-manifest and --score-audit"
                )
            verified_evidence = validate_real_evidence_bundle(
                arguments.trials,
                arguments.evidence_manifest,
                arguments.score_audit,
                trials,
            )
        elif arguments.evidence_manifest is not None or arguments.score_audit is not None:
            raise EvaluationError(
                "proxy/synthetic runs must not attach formal real-evidence files"
            )
        result = evaluate_gates(
            trials,
            evidence_label=arguments.evidence_label,
            expected_scenarios=arguments.expected_scenarios,
            expected_short=arguments.expected_short,
            bootstrap_repeats=arguments.bootstrap_repeats,
            bootstrap_seed=arguments.bootstrap_seed,
            macro_min_improvement=arguments.macro_min_improvement,
            macro_target_improvement=arguments.macro_target_improvement,
            short_min_improvement=arguments.short_min_improvement,
            max_category_regression=arguments.max_category_regression,
            manifest_audits=audits,
            verified_evidence=verified_evidence,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "{} macro_improvement={:.6f} short_improvement={:.6f} ci95_upper={:.6f}".format(
                result["status"],
                result["metrics"]["macro_15_scenarios"]["improvement_points"],
                result["metrics"]["short_10_scenarios"]["improvement_points"],
                result["bootstrap"]["macro_15_scenarios"]["delta_ci95_upper"],
            )
        )
        if result["release_eligible"]:
            return 0
        # A proxy/coverage run that clears numerical thresholds is still not a
        # successful release command.  Distinct non-zero status prevents shell
        # pipelines from mistaking it for a formal gate pass.
        return 4 if arguments.evidence_label != "real" else 3
    except (EvaluationError, ManifestError, OSError) as error:
        print("distillation gate evaluation failed: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
