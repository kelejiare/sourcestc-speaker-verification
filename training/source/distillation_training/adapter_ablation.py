#!/usr/bin/env python3
"""Pre-registered, dev-only adapter ablations for distillation-domain-KD.

The acoustic front end and the 448-dimensional acoustic geometry stay frozen.  This
module compares a deliberately small family of linear maps for the 0.5 s and
1.0 s branches.  Every map is fitted on ``split=train`` using dataset-balanced
speaker bootstraps and is selected solely by speaker-verification macro EER on
``split=dev``.  ``split=test`` and LODO outcomes are never accepted as inputs
to model selection.

The command-line entry point writes a research-only matrix artifact under the
requested experiment directory.  It is intentionally incompatible with the
formal distillation adapter schema and must not be copied into ``model/`` or submitted.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distillation_training.cache_schema import EMBEDDING_DIM, read_embedding_cache
from distillation_training.train_ridge_kd import (
    DEFAULT_SEEDS,
    dataset_balanced_row_weights,
    identity_regularized_ridge,
    leave_one_utterance_full_prototypes,
    select_training_rows,
    sha256_file,
    speaker_bootstrap_indices,
)
from tools.evaluate_distillation_gates import compute_eer_percent


ABLATION_SCHEMA_VERSION = 1
RESEARCH_MATRIX_SCHEMA_VERSION = 1
SELECTION_PROTOCOL = "train_fit_dev_speaker_verification_macro_eer_v1"
TRAIN_WEIGHTING_POLICY = "speaker_bootstrap_then_each_dataset_total_weight_1_v1"
DEV_TRIAL_POLICY = "loo_positive_plus_sha256_cyclic_impostors_v1"
BLOCK_DIMS = (256, 192)
RIDGE_LAMBDAS = (0.001, 0.01, 0.1)
ALPHAS = (0.25, 0.5, 0.75)
LOW_RANKS = (8, 16, 32, 64)
IMPOSTORS_PER_PROBE = 5
FAMILY_ORDER = (
    "identity",
    "full_ridge",
    "block_ridge",
    "block_procrustes",
    "lowrank_ridge_residual",
)
DURATION_FIELDS = ("short05", "short10")


class AblationError(ValueError):
    """Raised when the frozen ablation protocol would be violated."""


def _stable_digest(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _format_grid_value(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values_64 = np.asarray(values, dtype=np.float64)
    if values_64.ndim != 2 or values_64.shape[0] == 0:
        raise AblationError("embedding 矩阵必须是非空二维数组")
    norms = np.linalg.norm(values_64, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1.0e-12):
        raise AblationError("embedding 包含非有限或零范数行")
    return values_64 / norms


def _validate_blocks(dimension: int, block_dims: Sequence[int]) -> Tuple[int, ...]:
    blocks = tuple(int(value) for value in block_dims)
    if not blocks or any(value <= 0 for value in blocks):
        raise AblationError("block_dims 必须是非空正整数序列")
    if sum(blocks) != int(dimension):
        raise AblationError(
            "block_dims 总和必须等于 embedding 维度：{} != {}".format(
                sum(blocks), dimension
            )
        )
    return blocks


def _validate_fixed_protocol(
    ridge_lambdas: Sequence[float],
    alphas: Sequence[float],
    low_ranks: Sequence[int],
    seeds: Sequence[int],
    dimension: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[int, ...]]:
    normalized_seeds = tuple(int(value) for value in seeds)
    if normalized_seeds != DEFAULT_SEEDS:
        raise AblationError(
            "adapter 消融必须严格使用预注册三种子 {}".format(DEFAULT_SEEDS)
        )
    lambdas = tuple(float(value) for value in ridge_lambdas)
    alpha_grid = tuple(float(value) for value in alphas)
    ranks = tuple(int(value) for value in low_ranks)
    if not lambdas or len(set(lambdas)) != len(lambdas):
        raise AblationError("ridge lambda 网格必须非空且无重复")
    if any(not np.isfinite(value) or value <= 0.0 for value in lambdas):
        raise AblationError("ridge lambda 必须为有限正数")
    if not alpha_grid or len(set(alpha_grid)) != len(alpha_grid):
        raise AblationError("alpha 网格必须非空且无重复")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in alpha_grid):
        raise AblationError("alpha 必须位于 [0,1]")
    if not ranks or len(set(ranks)) != len(ranks):
        raise AblationError("rank 网格必须非空且无重复")
    if any(value <= 0 or value > int(dimension) for value in ranks):
        raise AblationError("rank 必须位于 [1, embedding_dim]")
    return lambdas, alpha_grid, ranks


def weighted_orthogonal_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit an orthogonal source-to-target row-vector map with identity blend."""

    source_64 = np.asarray(source, dtype=np.float64)
    target_64 = np.asarray(target, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if source_64.ndim != 2 or target_64.shape != source_64.shape:
        raise AblationError("Procrustes source/target 必须是相同二维形状")
    if weights.shape != (source_64.shape[0],):
        raise AblationError("Procrustes sample_weights 行数错误")
    if (
        not np.isfinite(source_64).all()
        or not np.isfinite(target_64).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise AblationError("Procrustes 输入必须有限且权重为正")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise AblationError("Procrustes alpha 必须位于 [0,1]")
    cross = np.matmul((target_64 * weights[:, None]).T, source_64)
    left, _singular, right_t = np.linalg.svd(cross, full_matrices=False)
    orthogonal = np.matmul(left, right_t)
    identity = np.eye(source_64.shape[1], dtype=np.float64)
    matrix = (1.0 - alpha) * identity + alpha * orthogonal
    if not np.isfinite(matrix).all():
        raise RuntimeError("Procrustes 产生 NaN/Inf")
    return np.ascontiguousarray(matrix.astype(np.float32))


def block_diagonal_ridge(
    source: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    ridge_lambda: float,
    alpha: float,
    block_dims: Sequence[int],
) -> np.ndarray:
    """Fit independent identity-regularized ridge maps for PT and CAM++ blocks."""

    source_array = np.asarray(source)
    target_array = np.asarray(target)
    if source_array.ndim != 2 or target_array.shape != source_array.shape:
        raise AblationError("block ridge source/target 必须是相同二维形状")
    blocks = _validate_blocks(source_array.shape[1], block_dims)
    output = np.zeros((source_array.shape[1], source_array.shape[1]), dtype=np.float32)
    offset = 0
    for block in blocks:
        end = offset + block
        output[offset:end, offset:end] = identity_regularized_ridge(
            source_array[:, offset:end],
            target_array[:, offset:end],
            ridge_lambda=ridge_lambda,
            alpha=alpha,
            sample_weights=sample_weights,
        )
        offset = end
    return output


def block_diagonal_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    alpha: float,
    block_dims: Sequence[int],
) -> np.ndarray:
    """Fit independent orthogonal Procrustes maps for PT and CAM++ blocks."""

    source_array = np.asarray(source)
    target_array = np.asarray(target)
    if source_array.ndim != 2 or target_array.shape != source_array.shape:
        raise AblationError("block Procrustes source/target 必须是相同二维形状")
    blocks = _validate_blocks(source_array.shape[1], block_dims)
    output = np.zeros((source_array.shape[1], source_array.shape[1]), dtype=np.float32)
    offset = 0
    for block in blocks:
        end = offset + block
        output[offset:end, offset:end] = weighted_orthogonal_procrustes(
            source_array[:, offset:end],
            target_array[:, offset:end],
            sample_weights=sample_weights,
            alpha=alpha,
        )
        offset = end
    return output


def truncated_ridge_residual(
    full_ridge_matrix: np.ndarray, rank: int, alpha: float
) -> np.ndarray:
    """Keep only the leading singular directions of ``L_ridge - I``."""

    matrix = np.asarray(full_ridge_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AblationError("low-rank 输入必须是方阵")
    rank = int(rank)
    alpha = float(alpha)
    if rank <= 0 or rank > matrix.shape[0]:
        raise AblationError("low-rank rank 越界")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise AblationError("low-rank alpha 必须位于 [0,1]")
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    left, singular, right_t = np.linalg.svd(matrix - identity, full_matrices=False)
    residual = np.matmul(left[:, :rank] * singular[:rank], right_t[:rank])
    output = identity + alpha * residual
    if not np.isfinite(output).all():
        raise RuntimeError("low-rank ridge residual 产生 NaN/Inf")
    return np.ascontiguousarray(output.astype(np.float32))


def _speaker_prototypes(
    cache: Mapping[str, np.ndarray], indices: np.ndarray
) -> Dict[Tuple[str, str], np.ndarray]:
    full = np.asarray(cache["full"])
    datasets = np.asarray(cache["dataset"])
    speakers = np.asarray(cache["speaker_id"])
    utterances = np.asarray(cache["utterance_id"])
    utterance_rows = {}  # type: Dict[Tuple[str, str, str], List[int]]
    for raw_index in np.asarray(indices, dtype=np.int64):
        index = int(raw_index)
        key = (str(datasets[index]), str(speakers[index]), str(utterances[index]))
        utterance_rows.setdefault(key, []).append(index)
    by_speaker = {}  # type: Dict[Tuple[str, str], List[np.ndarray]]
    for (dataset, speaker, _utterance), row_indices in sorted(utterance_rows.items()):
        prototype = np.mean(full[row_indices].astype(np.float64), axis=0)
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(norm) or norm < 1.0e-12:
            raise AblationError("dev utterance prototype 非有限或零范数")
        by_speaker.setdefault((dataset, speaker), []).append(prototype / norm)
    output = {}
    for key, values in sorted(by_speaker.items()):
        prototype = np.mean(np.stack(values), axis=0)
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(norm) or norm < 1.0e-12:
            raise AblationError("dev speaker prototype 非有限或零范数")
        output[key] = prototype / norm
    return output


def build_dev_verification_design(
    cache: Mapping[str, np.ndarray],
    minimum_dev_datasets: int = 2,
    impostors_per_probe: int = IMPOSTORS_PER_PROBE,
) -> Dict[str, object]:
    """Freeze genuine/impostor enrollments using dev rows only.

    A genuine enrollment is the leave-one-utterance full prototype already
    used by the KD trainer.  Impostor enrollments are full prototypes of other
    dev speakers from the same dataset.  SHA-256 chooses a cyclic starting
    point in the sorted impostor list, making the design independent of Python
    hash randomization and identical for every candidate.
    """

    if int(impostors_per_probe) <= 0:
        raise AblationError("impostors_per_probe 必须为正数")
    dev_indices = select_training_rows(
        cache,
        train_split="dev",
        holdout_datasets=(),
        minimum_training_datasets=minimum_dev_datasets,
    )
    teacher_payload = leave_one_utterance_full_prototypes(cache, dev_indices)
    eligible = np.asarray(teacher_payload["indices"], dtype=np.int64)
    teachers = np.asarray(teacher_payload["teacher"], dtype=np.float64)
    prototypes = _speaker_prototypes(cache, eligible)
    datasets = np.asarray(cache["dataset"])
    speakers = np.asarray(cache["speaker_id"])
    utterances = np.asarray(cache["utterance_id"])
    embedding_ids = np.asarray(cache["embedding_id"])
    speakers_by_dataset = {}  # type: Dict[str, List[str]]
    for dataset, speaker in prototypes:
        speakers_by_dataset.setdefault(dataset, []).append(speaker)
    for dataset in speakers_by_dataset:
        speakers_by_dataset[dataset] = sorted(set(speakers_by_dataset[dataset]))
        if len(speakers_by_dataset[dataset]) < 2:
            raise AblationError(
                "dev speaker-verification 每个 dataset 至少需要2位 speaker：{}".format(
                    dataset
                )
            )

    rows = []  # type: List[Dict[str, object]]
    for relative, raw_index in enumerate(eligible):
        index = int(raw_index)
        dataset = str(datasets[index])
        speaker = str(speakers[index])
        other_speakers = [
            value for value in speakers_by_dataset[dataset] if value != speaker
        ]
        count = min(int(impostors_per_probe), len(other_speakers))
        start = int(
            _stable_digest(
                DEV_TRIAL_POLICY,
                dataset,
                speaker,
                str(utterances[index]),
                str(embedding_ids[index]),
            )[:16],
            16,
        ) % len(other_speakers)
        selected_impostors = [
            other_speakers[(start + offset) % len(other_speakers)]
            for offset in range(count)
        ]
        rows.append(
            {
                "cache_index": index,
                "dataset": dataset,
                "speaker_id": speaker,
                "genuine_enrollment": teachers[relative],
                "impostor_enrollments": np.stack(
                    [prototypes[(dataset, value)] for value in selected_impostors]
                ),
                "impostor_speaker_ids": tuple(selected_impostors),
            }
        )
    dataset_names = sorted(set(str(datasets[index]) for index in eligible))
    return {
        "policy": DEV_TRIAL_POLICY,
        "indices": eligible,
        "rows": rows,
        "datasets": dataset_names,
        "dataset_probe_counts": {
            dataset: sum(row["dataset"] == dataset for row in rows)
            for dataset in dataset_names
        },
        "positive_trial_count": len(rows),
        "negative_trial_count": sum(
            int(np.asarray(row["impostor_enrollments"]).shape[0]) for row in rows
        ),
        "teacher_audit": teacher_payload["audit"],
        "test_sample_count_used": 0,
        "lodo_metric_count_used": 0,
    }


def score_dev_verification(
    matrix: np.ndarray,
    cache: Mapping[str, np.ndarray],
    design: Mapping[str, object],
    duration_field: str,
) -> Dict[str, object]:
    if duration_field not in DURATION_FIELDS:
        raise AblationError("未知短时长字段：{}".format(duration_field))
    matrix_64 = np.asarray(matrix, dtype=np.float64)
    source = np.asarray(cache[duration_field], dtype=np.float64)
    if matrix_64.shape != (source.shape[1], source.shape[1]):
        raise AblationError("候选矩阵与 embedding 维度不一致")
    probe_indices = np.asarray(design["indices"], dtype=np.int64)
    transformed = _normalize_rows(np.matmul(source[probe_indices], matrix_64.T))
    by_dataset_rows = {}  # type: Dict[str, Dict[str, List[float]]]
    for relative, row in enumerate(design["rows"]):
        dataset = str(row["dataset"])
        bucket = by_dataset_rows.setdefault(dataset, {"labels": [], "scores": []})
        probe = transformed[relative]
        genuine = np.asarray(row["genuine_enrollment"], dtype=np.float64)
        bucket["labels"].append(1)
        bucket["scores"].append(float(np.dot(probe, genuine)))
        for enrollment in np.asarray(row["impostor_enrollments"], dtype=np.float64):
            bucket["labels"].append(0)
            bucket["scores"].append(float(np.dot(probe, enrollment)))
    by_dataset = {}
    for dataset in sorted(by_dataset_rows):
        bucket = by_dataset_rows[dataset]
        by_dataset[dataset] = compute_eer_percent(
            bucket["labels"], bucket["scores"]
        )
    if not by_dataset:
        raise AblationError("dev 评分没有 dataset")
    return {
        "macro_eer_percent": float(np.mean(list(by_dataset.values()))),
        "worst_dataset_eer_percent": float(max(by_dataset.values())),
        "by_dataset_eer_percent": by_dataset,
    }


def matrix_diagnostics(matrix: np.ndarray) -> Dict[str, object]:
    matrix_64 = np.asarray(matrix, dtype=np.float64)
    if matrix_64.ndim != 2 or matrix_64.shape[0] != matrix_64.shape[1]:
        raise AblationError("矩阵诊断只接受方阵")
    singular = np.linalg.svd(matrix_64, compute_uv=False)
    spectral_norm = float(singular[0])
    smallest = float(singular[-1])
    condition = None if smallest <= 0.0 else float(spectral_norm / smallest)
    if condition is not None and not np.isfinite(condition):
        condition = None
    identity_distance = float(
        np.linalg.norm(matrix_64 - np.eye(matrix_64.shape[0]), ord="fro")
        / np.sqrt(float(matrix_64.shape[0]))
    )
    return {
        "spectral_norm": spectral_norm,
        "smallest_singular_value": smallest,
        "condition_number": condition,
        "condition_number_finite": condition is not None,
        "identity_distance_normalized_frobenius": identity_distance,
    }


def _candidate_specs(
    lambdas: Sequence[float], alphas: Sequence[float], ranks: Sequence[int]
) -> List[Dict[str, object]]:
    specs = [
        {
            "candidate_index": 0,
            "candidate_id": "identity",
            "family": "identity",
            "ridge_lambda": None,
            "alpha": 0.0,
            "rank": None,
        }
    ]
    for family in ("full_ridge", "block_ridge"):
        for ridge_lambda in lambdas:
            for alpha in alphas:
                specs.append(
                    {
                        "candidate_index": len(specs),
                        "candidate_id": "{}__lambda_{}__alpha_{}".format(
                            family,
                            _format_grid_value(ridge_lambda),
                            _format_grid_value(alpha),
                        ),
                        "family": family,
                        "ridge_lambda": float(ridge_lambda),
                        "alpha": float(alpha),
                        "rank": None,
                    }
                )
    for alpha in alphas:
        specs.append(
            {
                "candidate_index": len(specs),
                "candidate_id": "block_procrustes__alpha_{}".format(
                    _format_grid_value(alpha)
                ),
                "family": "block_procrustes",
                "ridge_lambda": None,
                "alpha": float(alpha),
                "rank": None,
            }
        )
    for ridge_lambda in lambdas:
        for alpha in alphas:
            for rank in ranks:
                specs.append(
                    {
                        "candidate_index": len(specs),
                        "candidate_id": (
                            "lowrank_ridge_residual__lambda_{}__alpha_{}__rank_{}"
                        ).format(
                            _format_grid_value(ridge_lambda),
                            _format_grid_value(alpha),
                            rank,
                        ),
                        "family": "lowrank_ridge_residual",
                        "ridge_lambda": float(ridge_lambda),
                        "alpha": float(alpha),
                        "rank": int(rank),
                    }
                )
    return specs


def _seed_training_payloads(
    cache: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    seeds: Sequence[int],
) -> List[Dict[str, object]]:
    teacher_payload = leave_one_utterance_full_prototypes(cache, train_indices)
    eligible = np.asarray(teacher_payload["indices"], dtype=np.int64)
    teacher = np.asarray(teacher_payload["teacher"], dtype=np.float32)
    output = []
    for seed in seeds:
        relative = speaker_bootstrap_indices(
            np.asarray(cache["speaker_id"])[eligible],
            np.asarray(cache["dataset"])[eligible],
            int(seed),
        )
        indices = eligible[relative]
        targets = teacher[relative]
        weights, balance = dataset_balanced_row_weights(
            np.asarray(cache["dataset"])[indices]
        )
        output.append(
            {
                "seed": int(seed),
                "indices": indices,
                "target": targets,
                "weights": weights,
                "domain_balance": balance,
            }
        )
    return output


def _fit_candidate_seed_matrix(
    spec: Mapping[str, object],
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    block_dims: Sequence[int],
    fit_state: Dict[str, object],
) -> np.ndarray:
    dimension = int(source.shape[1])
    family = str(spec["family"])
    if family == "identity":
        return np.eye(dimension, dtype=np.float32)
    alpha = float(spec["alpha"])
    if family == "full_ridge":
        ridge_lambda = float(spec["ridge_lambda"])
        full_bases = fit_state.setdefault("full_ridge", {})
        if ridge_lambda not in full_bases:
            full_bases[ridge_lambda] = identity_regularized_ridge(
                source,
                target,
                ridge_lambda=ridge_lambda,
                alpha=1.0,
                sample_weights=weights,
            )
        identity = np.eye(dimension, dtype=np.float64)
        base = np.asarray(full_bases[ridge_lambda], dtype=np.float64)
        return np.ascontiguousarray(
            ((1.0 - alpha) * identity + alpha * base).astype(np.float32)
        )
    if family == "block_ridge":
        ridge_lambda = float(spec["ridge_lambda"])
        block_bases = fit_state.setdefault("block_ridge", {})
        if ridge_lambda not in block_bases:
            block_bases[ridge_lambda] = block_diagonal_ridge(
                source,
                target,
                weights,
                ridge_lambda=ridge_lambda,
                alpha=1.0,
                block_dims=block_dims,
            )
        identity = np.eye(dimension, dtype=np.float64)
        base = np.asarray(block_bases[ridge_lambda], dtype=np.float64)
        return np.ascontiguousarray(
            ((1.0 - alpha) * identity + alpha * base).astype(np.float32)
        )
    if family == "block_procrustes":
        if "block_procrustes" not in fit_state:
            fit_state["block_procrustes"] = block_diagonal_procrustes(
                source,
                target,
                weights,
                alpha=1.0,
                block_dims=block_dims,
            )
        identity = np.eye(dimension, dtype=np.float64)
        base = np.asarray(fit_state["block_procrustes"], dtype=np.float64)
        return np.ascontiguousarray(
            ((1.0 - alpha) * identity + alpha * base).astype(np.float32)
        )
    if family == "lowrank_ridge_residual":
        ridge_lambda = float(spec["ridge_lambda"])
        full_bases = fit_state.setdefault("full_ridge", {})
        if ridge_lambda not in full_bases:
            full_bases[ridge_lambda] = identity_regularized_ridge(
                source,
                target,
                ridge_lambda=ridge_lambda,
                alpha=1.0,
                sample_weights=weights,
            )
        decompositions = fit_state.setdefault("lowrank_svd", {})
        if ridge_lambda not in decompositions:
            base = np.asarray(full_bases[ridge_lambda], dtype=np.float64)
            decompositions[ridge_lambda] = np.linalg.svd(
                base - np.eye(dimension, dtype=np.float64),
                full_matrices=False,
            )
        left, singular, right_t = decompositions[ridge_lambda]
        rank = int(spec["rank"])
        residual = np.matmul(left[:, :rank] * singular[:rank], right_t[:rank])
        return np.ascontiguousarray(
            (np.eye(dimension, dtype=np.float64) + alpha * residual).astype(
                np.float32
            )
        )
    raise AblationError("未知 candidate family：{}".format(family))


def select_adapter_ablation_on_dev(
    cache: Mapping[str, np.ndarray],
    ridge_lambdas: Sequence[float] = RIDGE_LAMBDAS,
    alphas: Sequence[float] = ALPHAS,
    low_ranks: Sequence[int] = LOW_RANKS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    block_dims: Sequence[int] = BLOCK_DIMS,
    minimum_datasets: int = 2,
    impostors_per_probe: int = IMPOSTORS_PER_PROBE,
    enforce_registered_grid: bool = True,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Fit the registered families and select one using dev macro EER only."""

    required = (
        "embedding_id",
        "utterance_id",
        "speaker_id",
        "dataset",
        "split",
        "full",
        "short05",
        "short10",
    )
    missing = [key for key in required if key not in cache]
    if missing:
        raise AblationError("cache 缺少字段：{}".format(missing))
    full = np.asarray(cache["full"])
    if full.ndim != 2 or full.shape[0] == 0:
        raise AblationError("cache full 必须是非空二维数组")
    dimension = int(full.shape[1])
    blocks = _validate_blocks(dimension, block_dims)
    lambdas, alpha_grid, ranks = _validate_fixed_protocol(
        ridge_lambdas, alphas, low_ranks, seeds, dimension
    )
    if enforce_registered_grid:
        expected = (
            RIDGE_LAMBDAS,
            ALPHAS,
            LOW_RANKS,
            BLOCK_DIMS,
            IMPOSTORS_PER_PROBE,
            2,
        )
        actual = (
            lambdas,
            alpha_grid,
            ranks,
            blocks,
            int(impostors_per_probe),
            int(minimum_datasets),
        )
        if actual != expected:
            raise AblationError(
                "正式 dev 消融禁止覆盖预注册网格：expected={} actual={}".format(
                    expected, actual
                )
            )
    train_indices = select_training_rows(
        cache,
        train_split="train",
        holdout_datasets=(),
        minimum_training_datasets=minimum_datasets,
    )
    seed_payloads = _seed_training_payloads(cache, train_indices, seeds)
    design = build_dev_verification_design(
        cache,
        minimum_dev_datasets=minimum_datasets,
        impostors_per_probe=impostors_per_probe,
    )
    specs = _candidate_specs(lambdas, alpha_grid, ranks)
    candidates = []
    candidate_matrices = {}  # type: Dict[str, Dict[str, np.ndarray]]
    fit_states = {
        (duration, int(payload["seed"])): {}
        for duration in DURATION_FIELDS
        for payload in seed_payloads
    }  # type: Dict[Tuple[str, int], Dict[str, object]]
    for spec in specs:
        duration_matrices = {}
        duration_seed_matrices = {}
        for duration_field in DURATION_FIELDS:
            per_seed = []
            for payload in seed_payloads:
                indices = np.asarray(payload["indices"], dtype=np.int64)
                per_seed.append(
                    _fit_candidate_seed_matrix(
                        spec,
                        np.asarray(cache[duration_field])[indices],
                        np.asarray(payload["target"]),
                        np.asarray(payload["weights"]),
                        blocks,
                        fit_states[(duration_field, int(payload["seed"]))],
                    )
                )
            averaged = np.mean(
                np.stack(per_seed).astype(np.float64), axis=0
            ).astype(np.float32)
            duration_matrices[duration_field] = np.ascontiguousarray(averaged)
            duration_seed_matrices[duration_field] = per_seed
        candidate_id = str(spec["candidate_id"])
        candidate_matrices[candidate_id] = duration_matrices
        duration_scores = {
            duration_field: score_dev_verification(
                duration_matrices[duration_field], cache, design, duration_field
            )
            for duration_field in DURATION_FIELDS
        }
        dataset_names = list(design["datasets"])
        by_dataset_mean = {
            dataset: float(
                np.mean(
                    [
                        duration_scores[duration]["by_dataset_eer_percent"][dataset]
                        for duration in DURATION_FIELDS
                    ]
                )
            )
            for dataset in dataset_names
        }
        cell_values = {
            "{}:{}".format(dataset, duration): float(
                duration_scores[duration]["by_dataset_eer_percent"][dataset]
            )
            for dataset in dataset_names
            for duration in DURATION_FIELDS
        }
        diagnostics = {
            "L05": matrix_diagnostics(duration_matrices["short05"]),
            "L10": matrix_diagnostics(duration_matrices["short10"]),
        }
        dev_by_seed = []
        for seed_index, seed in enumerate(seeds):
            per_seed_scores = {
                duration: score_dev_verification(
                    duration_seed_matrices[duration][seed_index],
                    cache,
                    design,
                    duration,
                )
                for duration in DURATION_FIELDS
            }
            per_seed_cells = [
                float(
                    per_seed_scores[duration]["by_dataset_eer_percent"][dataset]
                )
                for dataset in dataset_names
                for duration in DURATION_FIELDS
            ]
            dev_by_seed.append(
                {
                    "seed": int(seed),
                    "mean_short_eer_percent": float(np.mean(per_seed_cells)),
                    "worst_dataset_duration_eer_percent": float(
                        max(per_seed_cells)
                    ),
                    "by_duration": per_seed_scores,
                }
            )
        worst_dataset = max(
            sorted(by_dataset_mean), key=lambda name: by_dataset_mean[name]
        )
        worst_cell = max(sorted(cell_values), key=lambda name: cell_values[name])
        record = dict(spec)
        record.update(
            {
                "dev_mean_short_eer_percent": float(
                    np.mean(list(cell_values.values()))
                ),
                "dev_worst_dataset": worst_dataset,
                "dev_worst_dataset_mean_short_eer_percent": by_dataset_mean[
                    worst_dataset
                ],
                "dev_worst_dataset_duration_cell": worst_cell,
                "dev_worst_dataset_duration_eer_percent": cell_values[worst_cell],
                "dev_by_duration": duration_scores,
                "dev_by_dataset_mean_short_eer_percent": by_dataset_mean,
                "matrix_diagnostics": diagnostics,
                "dev_by_seed": dev_by_seed,
                "max_spectral_norm": max(
                    float(item["spectral_norm"]) for item in diagnostics.values()
                ),
                "max_condition_number": (
                    max(
                        float(item["condition_number"])
                        for item in diagnostics.values()
                        if item["condition_number"] is not None
                    )
                    if all(
                        item["condition_number"] is not None
                        for item in diagnostics.values()
                    )
                    else None
                ),
                "max_identity_distance_normalized_frobenius": max(
                    float(item["identity_distance_normalized_frobenius"])
                    for item in diagnostics.values()
                ),
            }
        )
        candidates.append(record)

    selected = min(
        candidates,
        key=lambda item: (
            item["dev_mean_short_eer_percent"],
            item["dev_worst_dataset_mean_short_eer_percent"],
            item["candidate_index"],
        ),
    )
    selected_id = str(selected["candidate_id"])
    identity = next(item for item in candidates if item["family"] == "identity")
    report = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "artifact_kind": "distillation_dev_only_adapter_ablation_report",
        "selection_protocol": SELECTION_PROTOCOL,
        "release_status": "RESEARCH_ONLY_NOT_RELEASED_DO_NOT_SUBMIT",
        "fit_split": "train",
        "selection_split": "dev",
        "test_sample_count_used": 0,
        "lodo_metric_count_used": 0,
        "seeds": [int(value) for value in seeds],
        "seed_aggregation": "float64_mean_then_float32",
        "training_weighting_policy": TRAIN_WEIGHTING_POLICY,
        "dev_trial_design": {
            key: value
            for key, value in design.items()
            if key not in ("indices", "rows")
        },
        "fixed_grid": {
            "families": list(FAMILY_ORDER),
            "ridge_lambda": list(lambdas),
            "alpha": list(alpha_grid),
            "low_rank": list(ranks),
            "block_dims": list(blocks),
            "impostors_per_probe": int(impostors_per_probe),
            "candidate_count": len(specs),
            "candidate_order": (
                "identity; full ridge; block ridge; block Procrustes; "
                "low-rank ridge residual"
            ),
            "tie_break": (
                "mean short EER, worst-dataset mean EER, candidate index"
            ),
        },
        "train_sample_count_selected": int(train_indices.size),
        "train_datasets": sorted(
            set(str(value) for value in np.asarray(cache["dataset"])[train_indices])
        ),
        "training_domain_balance_by_seed": [
            {"seed": payload["seed"], **payload["domain_balance"]}
            for payload in seed_payloads
        ],
        "identity_baseline": {
            "candidate_id": identity["candidate_id"],
            "dev_mean_short_eer_percent": identity[
                "dev_mean_short_eer_percent"
            ],
            "dev_worst_dataset_mean_short_eer_percent": identity[
                "dev_worst_dataset_mean_short_eer_percent"
            ],
        },
        "selection_uses_only_train_and_dev": True,
        "pre_registered_defaults_enforced": bool(enforce_registered_grid),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": dict(selected),
    }
    matrices = {
        "L05": candidate_matrices[selected_id]["short05"].copy(),
        "L10": candidate_matrices[selected_id]["short10"].copy(),
    }
    return report, matrices


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_research_outputs(
    output_dir: Path,
    report: Mapping[str, object],
    matrices: Mapping[str, np.ndarray],
    cache_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Write a DO-NOT-SUBMIT artifact; dimensions are strictly 448 x 448."""

    output_dir = Path(output_dir)
    l05 = np.asarray(matrices.get("L05"))
    l10 = np.asarray(matrices.get("L10"))
    expected = (EMBEDDING_DIM, EMBEDDING_DIM)
    if l05.shape != expected or l10.shape != expected:
        raise AblationError(
            "research 输出矩阵必须都是 {}，实际 {} / {}".format(
                expected, l05.shape, l10.shape
            )
        )
    if l05.dtype != np.float32 or l10.dtype != np.float32:
        raise AblationError("research 输出矩阵必须为 float32")
    if not np.isfinite(l05).all() or not np.isfinite(l10).all():
        raise AblationError("research 输出矩阵包含 NaN/Inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "selected_research_matrices_DO_NOT_SUBMIT.npz"
    report_path = output_dir / "adapter_ablation_report.json"
    selected = dict(report["selected"])
    arrays = {
        "schema_version": np.asarray(
            RESEARCH_MATRIX_SCHEMA_VERSION, dtype=np.int64
        ),
        "artifact_kind": np.asarray("distillation_dev_only_research_matrices"),
        "release_status": np.asarray("DO_NOT_SUBMIT"),
        "selection_protocol": np.asarray(SELECTION_PROTOCOL),
        "candidate_id": np.asarray(str(selected["candidate_id"])),
        "L05": np.ascontiguousarray(l05),
        "L10": np.ascontiguousarray(l10),
    }
    _atomic_npz(matrix_path, arrays)
    report_payload = dict(report)
    report_payload["research_matrix_artifact"] = {
        "path": str(matrix_path.resolve()),
        "sha256": sha256_file(matrix_path),
        "formal_adapter_schema_compatible": False,
        "submission_allowed": False,
    }
    if cache_path is not None:
        report_payload["cache"] = {
            "path": str(Path(cache_path).resolve()),
            "sha256": sha256_file(Path(cache_path)),
        }
    _atomic_json(report_path, report_payload)
    return {
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": sha256_file(matrix_path),
        "release_status": "RESEARCH_ONLY_NOT_RELEASED_DO_NOT_SUBMIT",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    cache = read_embedding_cache(arguments.cache, allow_v1_migration=False)
    report, matrices = select_adapter_ablation_on_dev(cache)
    receipt = write_research_outputs(
        arguments.output_dir,
        report,
        matrices,
        cache_path=arguments.cache,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
