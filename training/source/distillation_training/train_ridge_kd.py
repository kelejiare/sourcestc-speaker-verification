#!/usr/bin/env python3
"""Train three deterministic identity-regularized ridge-KD seed maps.

For row-major observations X (short) and Y (full teacher), the implemented
closed-form map is exactly (W is optional dataset-balanced row weight)::

    L = (Y W X^T + lambda I) (X W X^T + lambda I)^-1

The inverse is evaluated with ``numpy.linalg.solve``.  The optional alpha
blend returns ``(1-alpha) I + alpha L``.  Acoustic backbones stay frozen.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distillation_training.cache_schema import EMBEDDING_DIM, read_embedding_cache


SEED_ARTIFACT_SCHEMA_VERSION = 4
SEED_ARTIFACT_KIND = "distillation_preregistered_lowrank_ridge_residual_seed"
HYPERPARAMETER_SELECTION_SCHEMA_VERSION = 2
DEFAULT_SEEDS = (3407, 3408, 3409)
TEACHER_POLICY = "same_speaker_leave_one_utterance_full_prototype_v1"
SELECTION_PROTOCOL = "train_fit_dev_only_fixed_grid_domain_macro_v2"
FORMAL_CANDIDATE_SELECTION_PROTOCOL = (
    "train_fit_dev_speaker_verification_macro_eer_v1"
)
FORMAL_CANDIDATE_FAMILY = "lowrank_ridge_residual"
DOMAIN_WEIGHTING_POLICY = "each_dataset_total_weight_1_v1"
DEV_SELECTION_POLICY = "equal_dataset_macro_then_equal_duration_v1"
SEED_ARTIFACT_KEYS = (
    "schema_version",
    "artifact_kind",
    "seed",
    "candidate_id",
    "candidate_family",
    "candidate_rank",
    "ridge_lambda",
    "alpha",
    "train_split",
    "holdout_datasets",
    "training_datasets",
    "cache_sha256",
    "sample_count",
    "speaker_count",
    "teacher_policy",
    "teacher_selected_sample_count",
    "teacher_eligible_sample_count",
    "teacher_excluded_sample_count",
    "teacher_selected_speaker_count",
    "teacher_eligible_speaker_count",
    "teacher_excluded_speaker_count",
    "teacher_excluded_speaker_keys",
    "teacher_invalid_utterance_keys",
    "teacher_zero_norm_prototype_embedding_ids",
    "teacher_self_overlap_count",
    "selection_protocol",
    "selection_report_sha256",
    "selection_report_json_utf8",
    "selection_cache_sha256",
    "domain_weighting_policy",
    "domain_total_weight_target",
    "domain_names",
    "domain_sample_counts",
    "domain_total_weights",
    "L05",
    "L10",
    "short05_identity_mse",
    "short05_trained_mse",
    "short10_identity_mse",
    "short10_trained_mse",
)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _reject_json_constant(value: str):
    raise ValueError("selection report 包含非标准 JSON 常量：{}".format(value))


def validate_candidate_selection_report_bytes(
    report_bytes: bytes,
    expected_selection_cache_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Authenticate the dev-only ablation decision without using its matrices.

    The report selects only a *candidate specification*.  Formal matrices are
    always refitted independently for each bootstrap seed.  Embedding the exact
    report bytes in every seed artifact makes ``selection_report_sha256`` a
    verifiable binding instead of an unauthenticated text field.
    """

    if not isinstance(report_bytes, bytes) or not report_bytes:
        raise ValueError("candidate selection report 必须是非空 bytes")
    if len(report_bytes) > 16 * 1024 * 1024:
        raise ValueError("candidate selection report 超过 16 MiB")
    try:
        report = json.loads(
            report_bytes.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("candidate selection report 不是严格 UTF-8 JSON：{}".format(error))
    if not isinstance(report, dict):
        raise ValueError("candidate selection report 必须是 JSON object")

    # Import lazily: adapter_ablation imports primitives from this module.
    from distillation_training import adapter_ablation as ablation

    required_exact = {
        "schema_version": ablation.ABLATION_SCHEMA_VERSION,
        "artifact_kind": "distillation_dev_only_adapter_ablation_report",
        "selection_protocol": FORMAL_CANDIDATE_SELECTION_PROTOCOL,
        "release_status": "RESEARCH_ONLY_NOT_RELEASED_DO_NOT_SUBMIT",
        "fit_split": "train",
        "selection_split": "dev",
        "test_sample_count_used": 0,
        "lodo_metric_count_used": 0,
        "seeds": list(DEFAULT_SEEDS),
        "seed_aggregation": "float64_mean_then_float32",
        "training_weighting_policy": ablation.TRAIN_WEIGHTING_POLICY,
        "selection_uses_only_train_and_dev": True,
        "pre_registered_defaults_enforced": True,
        "candidate_count": 58,
    }
    for key, expected in required_exact.items():
        if report.get(key) != expected:
            raise ValueError(
                "candidate selection report {} 错误：expected={!r} actual={!r}".format(
                    key, expected, report.get(key)
                )
            )

    fixed_grid = report.get("fixed_grid")
    if not isinstance(fixed_grid, dict):
        raise ValueError("candidate selection report 缺少 fixed_grid")
    fixed_expected = {
        "families": list(ablation.FAMILY_ORDER),
        "ridge_lambda": list(ablation.RIDGE_LAMBDAS),
        "alpha": list(ablation.ALPHAS),
        "low_rank": list(ablation.LOW_RANKS),
        "block_dims": list(ablation.BLOCK_DIMS),
        "impostors_per_probe": ablation.IMPOSTORS_PER_PROBE,
        "candidate_count": 58,
    }
    for key, expected in fixed_expected.items():
        if fixed_grid.get(key) != expected:
            raise ValueError("candidate selection report fixed_grid.{} 未预注册".format(key))

    registered_specs = ablation._candidate_specs(
        ablation.RIDGE_LAMBDAS, ablation.ALPHAS, ablation.LOW_RANKS
    )
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(registered_specs):
        raise ValueError("candidate selection report 候选数量错误")
    seen_ids = set()
    for registered, candidate in zip(registered_specs, candidates):
        if not isinstance(candidate, dict):
            raise ValueError("candidate selection report 候选必须是 object")
        for key in (
            "candidate_index",
            "candidate_id",
            "family",
            "ridge_lambda",
            "alpha",
            "rank",
        ):
            if candidate.get(key) != registered[key]:
                raise ValueError(
                    "candidate selection report 候选网格/顺序被修改：{}".format(key)
                )
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen_ids:
            raise ValueError("candidate selection report candidate_id 重复")
        seen_ids.add(candidate_id)
        for metric in (
            "dev_mean_short_eer_percent",
            "dev_worst_dataset_mean_short_eer_percent",
        ):
            value = candidate.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
            ):
                raise ValueError("candidate selection report {} 非有限".format(metric))

    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("candidate selection report 缺少 selected")
    selected_id = str(selected.get("candidate_id", ""))
    matching = [item for item in candidates if item["candidate_id"] == selected_id]
    if len(matching) != 1 or matching[0] != selected:
        raise ValueError("candidate selection report selected 不是候选表中的原始记录")
    recomputed = min(
        candidates,
        key=lambda item: (
            float(item["dev_mean_short_eer_percent"]),
            float(item["dev_worst_dataset_mean_short_eer_percent"]),
            int(item["candidate_index"]),
        ),
    )
    if recomputed["candidate_id"] != selected_id:
        raise ValueError("candidate selection report selected 不符合预注册 tie-break")
    if selected.get("family") != FORMAL_CANDIDATE_FAMILY:
        raise ValueError(
            "当前正式物化器只接受已选中的 {} 候选".format(
                FORMAL_CANDIDATE_FAMILY
            )
        )
    selected_spec = next(
        item for item in registered_specs if item["candidate_id"] == selected_id
    )
    if (
        selected_spec["ridge_lambda"] is None
        or selected_spec["rank"] is None
        or int(selected_spec["rank"]) <= 0
    ):
        raise ValueError("lowrank 候选缺少 ridge_lambda/rank")

    cache_ref = report.get("cache")
    if not isinstance(cache_ref, dict) or set(cache_ref) != {"path", "sha256"}:
        raise ValueError("candidate selection report cache binding 错误")
    selection_cache_sha256 = str(cache_ref.get("sha256", ""))
    if not _is_sha256(selection_cache_sha256):
        raise ValueError("candidate selection report cache SHA-256 错误")
    if (
        expected_selection_cache_sha256 is not None
        and selection_cache_sha256 != str(expected_selection_cache_sha256)
    ):
        raise ValueError("candidate selection report 未绑定指定 selection cache")
    train_datasets = report.get("train_datasets")
    if (
        not isinstance(train_datasets, list)
        or len(set(str(value) for value in train_datasets)) < 2
        or any(not str(value).strip() for value in train_datasets)
    ):
        raise ValueError("candidate selection report 至少需要两个训练域")

    return {
        "report": report,
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "selection_cache_sha256": selection_cache_sha256,
        "candidate_id": str(selected_spec["candidate_id"]),
        "candidate_family": str(selected_spec["family"]),
        "candidate_rank": int(selected_spec["rank"]),
        "ridge_lambda": float(selected_spec["ridge_lambda"]),
        "alpha": float(selected_spec["alpha"]),
        "selection_protocol": FORMAL_CANDIDATE_SELECTION_PROTOCOL,
        "selection_train_datasets": tuple(sorted(str(value) for value in train_datasets)),
    }


def load_candidate_selection_report(
    report_path: Path,
    selection_cache_path: Path,
    recompute_selection: bool = True,
) -> Tuple[bytes, Dict[str, object]]:
    report_path = Path(report_path)
    selection_cache_path = Path(selection_cache_path)
    if not report_path.is_file():
        raise FileNotFoundError("candidate selection report 不存在：{}".format(report_path))
    if not selection_cache_path.is_file():
        raise FileNotFoundError("selection cache 不存在：{}".format(selection_cache_path))
    report_bytes = report_path.read_bytes()
    selection_cache_sha256 = sha256_file(selection_cache_path)
    metadata = validate_candidate_selection_report_bytes(
        report_bytes,
        expected_selection_cache_sha256=selection_cache_sha256,
    )
    if recompute_selection:
        from distillation_training.adapter_ablation import select_adapter_ablation_on_dev

        selection_cache = read_embedding_cache(
            selection_cache_path, allow_v1_migration=False
        )
        recomputed_report, _unused_selected_matrices = select_adapter_ablation_on_dev(
            selection_cache
        )
        recomputed_selected = recomputed_report.get("selected", {})
        stored_selected = metadata["report"].get("selected", {})
        if recomputed_selected.get("candidate_id") != stored_selected.get(
            "candidate_id"
        ):
            raise ValueError(
                "candidate selection report 选择结果无法从绑定 cache 复现"
            )
        for metric in (
            "dev_mean_short_eer_percent",
            "dev_worst_dataset_mean_short_eer_percent",
        ):
            recomputed_value = float(recomputed_selected[metric])
            stored_value = float(stored_selected[metric])
            if not np.isclose(
                recomputed_value, stored_value, rtol=0.0, atol=1.0e-10
            ):
                raise ValueError(
                    "candidate selection report {} 无法从绑定 cache 复现".format(
                        metric
                    )
                )
        metadata["selection_recomputed_from_cache"] = True
        metadata["recomputed_selected_candidate_id"] = str(
            recomputed_selected["candidate_id"]
        )
    else:
        metadata["selection_recomputed_from_cache"] = False
    return report_bytes, metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_regularized_ridge(
    source: np.ndarray,
    target: np.ndarray,
    ridge_lambda: float,
    alpha: float = 1.0,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit a source-to-target linear map with an identity prior.

    ``sample_weights`` scales observation rows by ``sqrt(weight)`` before the
    sufficient statistics are formed.  The unscaled ``ridge_lambda * I`` term
    therefore has a stable, explicit meaning relative to the total supplied
    statistical weight.
    """

    source = np.asarray(source)
    target = np.asarray(target)
    if source.ndim != 2 or target.shape != source.shape:
        raise ValueError(
            "source/target 必须是相同二维形状，实际 {} / {}".format(
                source.shape, target.shape
            )
        )
    if source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("source/target 不能为空")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source/target 包含 NaN/Inf")
    ridge_lambda = float(ridge_lambda)
    alpha = float(alpha)
    if not np.isfinite(ridge_lambda) or ridge_lambda <= 0.0:
        raise ValueError("ridge_lambda 必须为有限正数")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")

    source_64 = np.asarray(source, dtype=np.float64)
    target_64 = np.asarray(target, dtype=np.float64)
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != (source.shape[0],):
            raise ValueError("sample_weights shape 必须为 ({},)".format(source.shape[0]))
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("sample_weights 必须为有限正数")
        scale = np.sqrt(weights)[:, None]
        source_64 = source_64 * scale
        target_64 = target_64 * scale
    x = source_64.T
    y = target_64.T
    dimension = int(x.shape[0])
    identity = np.eye(dimension, dtype=np.float64)
    numerator = np.matmul(y, x.T) + ridge_lambda * identity
    denominator = np.matmul(x, x.T) + ridge_lambda * identity
    # numerator @ inv(denominator), evaluated without forming the inverse.
    fitted = np.linalg.solve(denominator, numerator.T).T
    blended = (1.0 - alpha) * identity + alpha * fitted
    if not np.isfinite(blended).all():
        raise RuntimeError("ridge-KD 产生 NaN/Inf")
    return np.ascontiguousarray(blended.astype(np.float32))


def mean_squared_kd_loss(
    matrix: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    predicted = np.matmul(source, matrix.T)
    per_row = np.mean(np.square(predicted - target), axis=1)
    if sample_weights is None:
        return float(np.mean(per_row))
    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.shape != per_row.shape:
        raise ValueError("sample_weights 与 loss 行数不一致")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("sample_weights 必须为有限正数")
    return float(np.sum(per_row * weights) / np.sum(weights))


def mean_cosine_kd_loss(
    matrix: np.ndarray, source: np.ndarray, target: np.ndarray
) -> float:
    """Deployment-aligned mean cosine distance after applying one map."""

    matrix = np.asarray(matrix, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.shape != source.shape:
        raise ValueError("source/target 必须是相同二维形状")
    if matrix.shape != (source.shape[1], source.shape[1]):
        raise ValueError("matrix 与 embedding 维度不一致")
    predicted = np.matmul(source, matrix.T)
    predicted_norms = np.linalg.norm(predicted, axis=1, keepdims=True)
    target_norms = np.linalg.norm(target, axis=1, keepdims=True)
    if (
        np.any(~np.isfinite(predicted_norms))
        or np.any(~np.isfinite(target_norms))
        or np.any(predicted_norms < 1.0e-12)
        or np.any(target_norms < 1.0e-12)
    ):
        raise ValueError("cosine KD 遇到非有限或零范数向量")
    predicted /= predicted_norms
    normalized_target = target / target_norms
    similarities = np.sum(predicted * normalized_target, axis=1)
    similarities = np.clip(similarities, -1.0, 1.0)
    return float(np.mean(1.0 - similarities))


def dataset_balanced_row_weights(
    datasets: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Assign each dataset exactly one unit of total statistical weight."""

    datasets = np.asarray(datasets)
    if datasets.ndim != 1 or datasets.size == 0:
        raise ValueError("datasets 必须是非空一维数组")
    domain_names = sorted(set(str(value) for value in datasets))
    weights = np.empty(datasets.shape[0], dtype=np.float64)
    sample_counts = []
    total_weights = []
    for domain in domain_names:
        indices = np.flatnonzero(datasets == domain)
        if indices.size == 0:
            raise RuntimeError("dataset 计数异常：{}".format(domain))
        weights[indices] = 1.0 / float(indices.size)
        sample_counts.append(int(indices.size))
        total_weights.append(float(np.sum(weights[indices])))
    audit = {
        "policy": DOMAIN_WEIGHTING_POLICY,
        "domain_total_weight_target": 1.0,
        "domain_names": domain_names,
        "domain_sample_counts": sample_counts,
        "domain_total_weights": total_weights,
        "all_domains_equal_total_weight": bool(
            np.allclose(total_weights, np.ones(len(total_weights)), atol=1.0e-12)
        ),
        "total_statistical_weight": float(np.sum(weights)),
    }
    return weights, audit


def macro_dataset_cosine_kd_loss(
    matrix: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    datasets: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Average cosine KD loss within each dataset, then equally across datasets."""

    datasets = np.asarray(datasets)
    if datasets.ndim != 1 or datasets.shape[0] != np.asarray(source).shape[0]:
        raise ValueError("datasets 与 source 行数不一致")
    by_domain = {}
    for domain in sorted(set(str(value) for value in datasets)):
        indices = np.flatnonzero(datasets == domain)
        by_domain[domain] = mean_cosine_kd_loss(
            matrix,
            np.asarray(source)[indices],
            np.asarray(target)[indices],
        )
    if not by_domain:
        raise ValueError("dev macro 至少需要一个 dataset")
    return float(np.mean(list(by_domain.values()))), by_domain


def leave_one_utterance_full_prototypes(
    cache: Mapping[str, np.ndarray],
    selected_indices: np.ndarray,
    insufficient_speaker_policy: str = "exclude",
) -> Dict[str, object]:
    """Build same-speaker full-embedding teachers without self leakage.

    Speakers are scoped by ``(dataset, speaker_id)`` and views are grouped by
    ``utterance_id``.  For every view of utterance ``i`` the teacher is the
    L2-normalized mean of the per-utterance full prototypes for every *other*
    utterance from that speaker.  Therefore another crop/view of the current
    utterance can never leak into its own teacher.  Speakers with fewer than
    two distinct utterances are either rejected or explicitly excluded.
    """

    if insufficient_speaker_policy not in ("exclude", "reject"):
        raise ValueError("insufficient_speaker_policy 必须是 exclude 或 reject")
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if selected_indices.ndim != 1 or selected_indices.size == 0:
        raise ValueError("selected_indices 必须是非空一维数组")
    if len(set(int(value) for value in selected_indices)) != selected_indices.size:
        raise ValueError("selected_indices 不允许重复")

    full = np.asarray(cache["full"])
    datasets = np.asarray(cache["dataset"])
    speakers = np.asarray(cache["speaker_id"])
    embedding_ids = np.asarray(cache["embedding_id"])
    if "utterance_id" not in cache:
        raise ValueError("cache 缺少 schema v2 utterance_id")
    utterance_ids = np.asarray(cache["utterance_id"])
    if full.ndim != 2:
        raise ValueError("cache full 必须是二维")
    if (
        datasets.ndim != 1
        or speakers.shape != datasets.shape
        or embedding_ids.shape != datasets.shape
        or utterance_ids.shape != datasets.shape
        or full.shape[0] != datasets.shape[0]
    ):
        raise ValueError("cache 行数不一致")
    if np.any(selected_indices < 0) or np.any(selected_indices >= full.shape[0]):
        raise IndexError("selected_indices 越界")
    if not np.isfinite(full[selected_indices]).all():
        raise ValueError("selected full embedding 包含 NaN/Inf")

    groups = {}
    utterance_groups = {}
    for index in selected_indices:
        speaker_key = (str(datasets[index]), str(speakers[index]))
        utterance_key = speaker_key + (str(utterance_ids[index]),)
        groups.setdefault(speaker_key, []).append(int(index))
        utterance_groups.setdefault(utterance_key, []).append(int(index))

    utterance_prototypes = {}
    invalid_utterance_keys = []
    for utterance_key, indices in utterance_groups.items():
        prototype = np.mean(full[indices].astype(np.float64), axis=0)
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(norm) or norm < 1.0e-12:
            invalid_utterance_keys.append(utterance_key)
            continue
        utterance_prototypes[utterance_key] = prototype / norm

    speaker_utterance_keys = {}
    for utterance_key in utterance_prototypes:
        speaker_utterance_keys.setdefault(utterance_key[:2], []).append(
            utterance_key
        )
    singleton_keys = sorted(
        key
        for key in groups
        if len(speaker_utterance_keys.get(key, ())) < 2
    )
    if singleton_keys and insufficient_speaker_policy == "reject":
        raise ValueError(
            "LOO teacher 要求每位 speaker 至少2条 utterance：{}".format(
                ["{}:{}".format(*key) for key in singleton_keys[:10]]
            )
        )

    group_sums = {
        key: np.sum(
            np.stack(
                [utterance_prototypes[item] for item in utterance_keys]
            ),
            axis=0,
        )
        for key, utterance_keys in speaker_utterance_keys.items()
        if len(utterance_keys) >= 2
    }
    eligible_indices = []
    teachers = []
    zero_norm_ids = []
    for index in selected_indices:
        speaker_key = (str(datasets[index]), str(speakers[index]))
        utterance_key = speaker_key + (str(utterance_ids[index]),)
        own_utterance_prototype = utterance_prototypes.get(utterance_key)
        other_utterance_keys = speaker_utterance_keys.get(speaker_key, ())
        if own_utterance_prototype is None or len(other_utterance_keys) < 2:
            continue
        # Subtract the entire current utterance prototype, not just this row.
        # This excludes every crop/view sharing utterance_id.
        prototype = (
            group_sums[speaker_key] - own_utterance_prototype
        ) / float(
            len(other_utterance_keys) - 1
        )
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(norm) or norm < 1.0e-12:
            zero_norm_ids.append(str(embedding_ids[index]))
            continue
        eligible_indices.append(int(index))
        teachers.append((prototype / norm).astype(np.float32))

    if not eligible_indices:
        raise ValueError("LOO teacher 筛选后没有可训练 utterance")
    eligible_indices_array = np.asarray(eligible_indices, dtype=np.int64)
    teacher_array = np.ascontiguousarray(np.stack(teachers), dtype=np.float32)
    eligible_keys = {
        (str(datasets[index]), str(speakers[index]))
        for index in eligible_indices_array
    }
    excluded_keys = sorted(set(groups) - eligible_keys)
    excluded_sample_count = int(selected_indices.size - eligible_indices_array.size)
    audit = {
        "teacher_policy": TEACHER_POLICY,
        "selected_sample_count": int(selected_indices.size),
        "eligible_sample_count": int(eligible_indices_array.size),
        "excluded_sample_count": excluded_sample_count,
        "selected_speaker_count": len(groups),
        "eligible_speaker_count": len(eligible_keys),
        "excluded_speaker_count": len(excluded_keys),
        "excluded_speaker_keys": [
            "{}:{}".format(*key) for key in excluded_keys
        ],
        "invalid_utterance_keys": sorted(
            "{}:{}:{}".format(*key) for key in invalid_utterance_keys
        ),
        "zero_norm_prototype_embedding_ids": sorted(zero_norm_ids),
        "self_overlap_count": 0,
    }
    return {
        "indices": eligible_indices_array,
        "teacher": teacher_array,
        "audit": audit,
    }


def speaker_bootstrap_indices(
    speaker_ids: np.ndarray,
    datasets: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Resample speakers with replacement independently inside each domain."""

    speaker_ids = np.asarray(speaker_ids)
    datasets = np.asarray(datasets)
    if speaker_ids.ndim != 1 or datasets.shape != speaker_ids.shape:
        raise ValueError("speaker_ids/datasets 必须是同形状一维数组")
    if speaker_ids.size == 0:
        raise ValueError("bootstrap 输入不能为空")
    generator = np.random.default_rng(int(seed))
    output = []
    for dataset in sorted(set(str(value) for value in datasets)):
        domain_mask = datasets == dataset
        domain_speakers = sorted(
            set(str(value) for value in speaker_ids[domain_mask])
        )
        if not domain_speakers:
            raise RuntimeError("domain 没有 speaker：{}".format(dataset))
        selected = generator.choice(
            np.asarray(domain_speakers),
            size=len(domain_speakers),
            replace=True,
        )
        for speaker in selected:
            indices = np.flatnonzero(domain_mask & (speaker_ids == speaker))
            if indices.size == 0:
                raise RuntimeError("bootstrap speaker 丢失：{}".format(speaker))
            output.extend(int(index) for index in indices)
    return np.asarray(output, dtype=np.int64)


def select_training_rows(
    cache: Mapping[str, np.ndarray],
    train_split: str = "train",
    holdout_datasets: Sequence[str] = (),
    minimum_training_datasets: int = 2,
) -> np.ndarray:
    holdout = set(str(value) for value in holdout_datasets)
    mask = cache["split"] == str(train_split)
    if holdout:
        mask &= ~np.isin(cache["dataset"], np.asarray(sorted(holdout)))
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("训练筛选后没有样本")
    domains = sorted(set(str(value) for value in cache["dataset"][indices]))
    if len(domains) < int(minimum_training_datasets):
        raise ValueError(
            "训练域不足：要求至少 {}，实际 {}".format(
                minimum_training_datasets, domains
            )
        )
    return indices.astype(np.int64)


def train_one_seed(
    cache: Mapping[str, np.ndarray],
    selected_indices: np.ndarray,
    seed: int,
    ridge_lambda: float,
    alpha: float,
    insufficient_speaker_policy: str = "exclude",
) -> Dict[str, object]:
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    teacher_payload = leave_one_utterance_full_prototypes(
        cache,
        selected_indices,
        insufficient_speaker_policy=insufficient_speaker_policy,
    )
    eligible_indices = teacher_payload["indices"]
    eligible_teachers = teacher_payload["teacher"]
    bootstrap_relative = speaker_bootstrap_indices(
        cache["speaker_id"][eligible_indices],
        cache["dataset"][eligible_indices],
        seed,
    )
    indices = eligible_indices[bootstrap_relative]
    teacher = eligible_teachers[bootstrap_relative]
    short05 = cache["short05"][indices]
    short10 = cache["short10"][indices]
    sample_weights, domain_balance_audit = dataset_balanced_row_weights(
        cache["dataset"][indices]
    )
    identity = np.eye(teacher.shape[1], dtype=np.float32)
    matrix_0p5 = identity_regularized_ridge(
        short05,
        teacher,
        ridge_lambda,
        alpha,
        sample_weights=sample_weights,
    )
    matrix_1p0 = identity_regularized_ridge(
        short10,
        teacher,
        ridge_lambda,
        alpha,
        sample_weights=sample_weights,
    )
    return {
        "seed": int(seed),
        "indices": indices,
        "teacher_audit": teacher_payload["audit"],
        "domain_balance_audit": domain_balance_audit,
        "L05": matrix_0p5,
        "L10": matrix_1p0,
        "short05_identity_mse": mean_squared_kd_loss(
            identity, short05, teacher, sample_weights=sample_weights
        ),
        "short05_trained_mse": mean_squared_kd_loss(
            matrix_0p5, short05, teacher, sample_weights=sample_weights
        ),
        "short10_identity_mse": mean_squared_kd_loss(
            identity, short10, teacher, sample_weights=sample_weights
        ),
        "short10_trained_mse": mean_squared_kd_loss(
            matrix_1p0, short10, teacher, sample_weights=sample_weights
        ),
    }


def train_selected_lowrank_seed(
    cache: Mapping[str, np.ndarray],
    selected_indices: np.ndarray,
    seed: int,
    ridge_lambda: float,
    alpha: float,
    rank: int,
    insufficient_speaker_policy: str = "exclude",
) -> Dict[str, object]:
    """Refit one independently bootstrapped low-rank residual candidate.

    This intentionally does not accept or load the research matrix artifact.
    For each duration and seed it fits a fresh full ridge base with
    ``alpha=1``, computes the SVD of ``L_ridge - I``, truncates that residual,
    and only then applies the selected alpha.
    """

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    teacher_payload = leave_one_utterance_full_prototypes(
        cache,
        selected_indices,
        insufficient_speaker_policy=insufficient_speaker_policy,
    )
    eligible_indices = np.asarray(teacher_payload["indices"], dtype=np.int64)
    eligible_teachers = np.asarray(teacher_payload["teacher"], dtype=np.float32)
    bootstrap_relative = speaker_bootstrap_indices(
        np.asarray(cache["speaker_id"])[eligible_indices],
        np.asarray(cache["dataset"])[eligible_indices],
        int(seed),
    )
    indices = eligible_indices[bootstrap_relative]
    teacher = eligible_teachers[bootstrap_relative]
    sample_weights, domain_balance_audit = dataset_balanced_row_weights(
        np.asarray(cache["dataset"])[indices]
    )
    identity = np.eye(teacher.shape[1], dtype=np.float32)

    # Lazy import keeps adapter_ablation -> train_ridge_kd imports acyclic at
    # module initialization while using the exact registered truncation code.
    from distillation_training.adapter_ablation import truncated_ridge_residual

    matrices = {}
    sources = {}
    for field in ("short05", "short10"):
        source = np.asarray(cache[field])[indices]
        full_ridge = identity_regularized_ridge(
            source,
            teacher,
            ridge_lambda=float(ridge_lambda),
            alpha=1.0,
            sample_weights=sample_weights,
        )
        matrices[field] = truncated_ridge_residual(
            full_ridge,
            rank=int(rank),
            alpha=float(alpha),
        )
        sources[field] = source

    return {
        "seed": int(seed),
        "indices": indices,
        "teacher_audit": teacher_payload["audit"],
        "domain_balance_audit": domain_balance_audit,
        "L05": matrices["short05"],
        "L10": matrices["short10"],
        "short05_identity_mse": mean_squared_kd_loss(
            identity, sources["short05"], teacher, sample_weights=sample_weights
        ),
        "short05_trained_mse": mean_squared_kd_loss(
            matrices["short05"],
            sources["short05"],
            teacher,
            sample_weights=sample_weights,
        ),
        "short10_identity_mse": mean_squared_kd_loss(
            identity, sources["short10"], teacher, sample_weights=sample_weights
        ),
        "short10_trained_mse": mean_squared_kd_loss(
            matrices["short10"],
            sources["short10"],
            teacher,
            sample_weights=sample_weights,
        ),
    }


def _validated_fixed_grid(values: Sequence[float], name: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("{} 固定网格必须非空且无重复".format(name))
    if any(not np.isfinite(value) for value in result):
        raise ValueError("{} 固定网格包含 NaN/Inf".format(name))
    return result


def select_hyperparameters_on_dev(
    cache: Mapping[str, np.ndarray],
    ridge_lambdas: Sequence[float],
    alphas: Sequence[float],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    holdout_datasets: Sequence[str] = (),
    minimum_training_datasets: int = 2,
) -> Dict[str, object]:
    """Select a pre-registered grid using train-fit/dev-score only.

    ``test`` rows and rows from ``holdout_datasets`` are never indexed.  Every
    candidate is fit with the same three speaker-bootstrap seeds.  Its two
    seed-averaged maps are scored using deployment-aligned cosine distance to
    dev LOO teachers; test and LODO outcome metrics are not accepted as inputs.
    """

    ridge_grid = _validated_fixed_grid(ridge_lambdas, "ridge_lambda")
    alpha_grid = _validated_fixed_grid(alphas, "alpha")
    if any(value <= 0.0 for value in ridge_grid):
        raise ValueError("ridge_lambda 固定网格必须全为正数")
    if any(value < 0.0 or value > 1.0 for value in alpha_grid):
        raise ValueError("alpha 固定网格必须位于 [0,1]")
    normalized_seeds = tuple(int(value) for value in seeds)
    if normalized_seeds != DEFAULT_SEEDS:
        raise ValueError("dev 选参必须严格使用预注册三种子 {}".format(DEFAULT_SEEDS))

    train_indices = select_training_rows(
        cache,
        train_split="train",
        holdout_datasets=holdout_datasets,
        minimum_training_datasets=minimum_training_datasets,
    )
    dev_indices = select_training_rows(
        cache,
        train_split="dev",
        holdout_datasets=holdout_datasets,
        minimum_training_datasets=minimum_training_datasets,
    )
    train_teacher = leave_one_utterance_full_prototypes(cache, train_indices)
    dev_teacher = leave_one_utterance_full_prototypes(cache, dev_indices)
    train_eligible = train_teacher["indices"]
    dev_eligible = dev_teacher["indices"]
    train_domains = sorted(
        set(str(value) for value in cache["dataset"][train_eligible])
    )
    dev_domains = sorted(
        set(str(value) for value in cache["dataset"][dev_eligible])
    )
    if len(train_domains) < int(minimum_training_datasets):
        raise ValueError("LOO 排除后训练域不足：{}".format(train_domains))
    if len(dev_domains) < int(minimum_training_datasets):
        raise ValueError("LOO 排除后 dev 域不足：{}".format(dev_domains))
    if not set(dev_domains).issubset(set(train_domains)):
        raise ValueError("dev 包含 train 中不存在的 dataset")

    dev_target = dev_teacher["teacher"]
    dev_short05 = cache["short05"][dev_eligible]
    dev_short10 = cache["short10"][dev_eligible]
    dev_dataset_rows = cache["dataset"][dev_eligible]
    identity = np.eye(dev_target.shape[1], dtype=np.float32)
    baseline_0p5, baseline_0p5_by_domain = macro_dataset_cosine_kd_loss(
        identity, dev_short05, dev_target, dev_dataset_rows
    )
    baseline_1p0, baseline_1p0_by_domain = macro_dataset_cosine_kd_loss(
        identity, dev_short10, dev_target, dev_dataset_rows
    )
    training_domain_balance_by_seed = None
    candidates = []
    for candidate_index, ridge_lambda in enumerate(ridge_grid):
        for alpha_index, alpha in enumerate(alpha_grid):
            seed_results = [
                train_one_seed(
                    cache,
                    train_indices,
                    seed,
                    ridge_lambda,
                    alpha,
                )
                for seed in normalized_seeds
            ]
            if training_domain_balance_by_seed is None:
                training_domain_balance_by_seed = [
                    {
                        "seed": int(result["seed"]),
                        **result["domain_balance_audit"],
                    }
                    for result in seed_results
                ]
            matrix_0p5 = np.mean(
                np.stack([result["L05"] for result in seed_results]).astype(
                    np.float64
                ),
                axis=0,
            ).astype(np.float32)
            matrix_1p0 = np.mean(
                np.stack([result["L10"] for result in seed_results]).astype(
                    np.float64
                ),
                axis=0,
            ).astype(np.float32)
            loss_0p5, loss_0p5_by_domain = macro_dataset_cosine_kd_loss(
                matrix_0p5,
                dev_short05,
                dev_target,
                dev_dataset_rows,
            )
            loss_1p0, loss_1p0_by_domain = macro_dataset_cosine_kd_loss(
                matrix_1p0,
                dev_short10,
                dev_target,
                dev_dataset_rows,
            )
            candidates.append(
                {
                    "candidate_index": int(
                        candidate_index * len(alpha_grid) + alpha_index
                    ),
                    "ridge_lambda": ridge_lambda,
                    "alpha": alpha,
                    # Retain the original field names for reader compatibility;
                    # their semantics are now explicitly dataset-macro.
                    "dev_short05_cosine_kd_loss": loss_0p5,
                    "dev_short10_cosine_kd_loss": loss_1p0,
                    "dev_mean_cosine_kd_loss": 0.5 * (loss_0p5 + loss_1p0),
                    "dev_short05_by_dataset": loss_0p5_by_domain,
                    "dev_short10_by_dataset": loss_1p0_by_domain,
                }
            )
    selected = min(
        candidates,
        key=lambda item: (
            item["dev_mean_cosine_kd_loss"], item["candidate_index"]
        ),
    )
    holdout = sorted(set(str(value) for value in holdout_datasets))
    return {
        "schema_version": HYPERPARAMETER_SELECTION_SCHEMA_VERSION,
        "selection_protocol": SELECTION_PROTOCOL,
        "training_domain_weighting_policy": DOMAIN_WEIGHTING_POLICY,
        "dev_selection_policy": DEV_SELECTION_POLICY,
        "seeds": list(normalized_seeds),
        "fixed_grid": {
            "ridge_lambda": list(ridge_grid),
            "alpha": list(alpha_grid),
            "candidate_order": "ridge_outer_alpha_inner",
            "tie_break": "lowest_candidate_index",
        },
        "fit_split": "train",
        "selection_split": "dev",
        "test_sample_count_used": 0,
        "lodo_metric_count_used": 0,
        "holdout_datasets": holdout,
        "holdout_sample_count_used": 0,
        "training_datasets": train_domains,
        "dev_datasets": dev_domains,
        "train_teacher_audit": train_teacher["audit"],
        "dev_teacher_audit": dev_teacher["audit"],
        "training_domain_balance_by_seed": training_domain_balance_by_seed,
        "dev_domain_balance": {
            "policy": "each_dataset_equal_macro_weight_v1",
            "domain_names": dev_domains,
            "domain_sample_counts": [
                int(np.sum(dev_dataset_rows == domain)) for domain in dev_domains
            ],
            "domain_total_weights": [1.0 for _ in dev_domains],
        },
        "dev_identity_baseline": {
            "short05_cosine_kd_loss": baseline_0p5,
            "short10_cosine_kd_loss": baseline_1p0,
            "mean_cosine_kd_loss": 0.5 * (baseline_0p5 + baseline_1p0),
            "short05_by_dataset": baseline_0p5_by_domain,
            "short10_by_dataset": baseline_1p0_by_domain,
        },
        "candidates": candidates,
        "selected": dict(selected),
    }


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_seed_artifact(
    path: Path,
    result: Mapping[str, object],
    candidate_metadata: Mapping[str, object],
    selection_report_bytes: bytes,
    train_split: str,
    holdout_datasets: Sequence[str],
    training_datasets: Sequence[str],
    cache_sha256: str,
    speaker_count: int,
) -> None:
    teacher_audit = result.get("teacher_audit")
    if not isinstance(teacher_audit, dict):
        raise ValueError("result 缺少真实 teacher_audit，拒绝伪造 artifact")
    domain_balance_audit = result.get("domain_balance_audit")
    if not isinstance(domain_balance_audit, dict):
        raise ValueError("result 缺少真实 domain_balance_audit，拒绝伪造 artifact")
    selection = validate_candidate_selection_report_bytes(
        selection_report_bytes,
        expected_selection_cache_sha256=str(
            candidate_metadata.get("selection_cache_sha256", "")
        ),
    )
    for key in (
        "candidate_id",
        "candidate_family",
        "candidate_rank",
        "ridge_lambda",
        "alpha",
        "selection_protocol",
        "selection_cache_sha256",
    ):
        if candidate_metadata.get(key) != selection[key]:
            raise ValueError("candidate_metadata.{} 与 selection report 不一致".format(key))
    selection_report_sha256 = str(selection["report_sha256"])
    arrays = {
        "schema_version": np.asarray(
            SEED_ARTIFACT_SCHEMA_VERSION, dtype=np.int64
        ),
        "artifact_kind": np.asarray(SEED_ARTIFACT_KIND),
        "seed": np.asarray(result["seed"], dtype=np.int64),
        "candidate_id": np.asarray(selection["candidate_id"]),
        "candidate_family": np.asarray(selection["candidate_family"]),
        "candidate_rank": np.asarray(selection["candidate_rank"], dtype=np.int64),
        "ridge_lambda": np.asarray(selection["ridge_lambda"], dtype=np.float64),
        "alpha": np.asarray(selection["alpha"], dtype=np.float64),
        "train_split": np.asarray(train_split),
        # Explicit Unicode dtype keeps empty holdout lists pickle-free and
        # schema-stable instead of silently becoming float64.
        "holdout_datasets": np.asarray(sorted(holdout_datasets), dtype="U"),
        "training_datasets": np.asarray(sorted(training_datasets), dtype="U"),
        "cache_sha256": np.asarray(cache_sha256),
        "sample_count": np.asarray(len(result["indices"]), dtype=np.int64),
        "speaker_count": np.asarray(speaker_count, dtype=np.int64),
        "teacher_policy": np.asarray(teacher_audit["teacher_policy"]),
        "teacher_selected_sample_count": np.asarray(
            teacher_audit["selected_sample_count"], dtype=np.int64
        ),
        "teacher_eligible_sample_count": np.asarray(
            teacher_audit["eligible_sample_count"], dtype=np.int64
        ),
        "teacher_excluded_sample_count": np.asarray(
            teacher_audit["excluded_sample_count"], dtype=np.int64
        ),
        "teacher_selected_speaker_count": np.asarray(
            teacher_audit["selected_speaker_count"], dtype=np.int64
        ),
        "teacher_eligible_speaker_count": np.asarray(
            teacher_audit["eligible_speaker_count"], dtype=np.int64
        ),
        "teacher_excluded_speaker_count": np.asarray(
            teacher_audit["excluded_speaker_count"], dtype=np.int64
        ),
        "teacher_excluded_speaker_keys": np.asarray(
            teacher_audit["excluded_speaker_keys"], dtype="U"
        ),
        "teacher_invalid_utterance_keys": np.asarray(
            teacher_audit["invalid_utterance_keys"], dtype="U"
        ),
        "teacher_zero_norm_prototype_embedding_ids": np.asarray(
            teacher_audit["zero_norm_prototype_embedding_ids"], dtype="U"
        ),
        "teacher_self_overlap_count": np.asarray(
            teacher_audit["self_overlap_count"], dtype=np.int64
        ),
        "selection_protocol": np.asarray(selection["selection_protocol"]),
        "selection_report_sha256": np.asarray(selection_report_sha256),
        "selection_report_json_utf8": np.frombuffer(
            selection_report_bytes, dtype=np.uint8
        ).copy(),
        "selection_cache_sha256": np.asarray(
            selection["selection_cache_sha256"]
        ),
        "domain_weighting_policy": np.asarray(
            domain_balance_audit["policy"]
        ),
        "domain_total_weight_target": np.asarray(
            domain_balance_audit["domain_total_weight_target"],
            dtype=np.float64,
        ),
        "domain_names": np.asarray(
            domain_balance_audit["domain_names"], dtype="U"
        ),
        "domain_sample_counts": np.asarray(
            domain_balance_audit["domain_sample_counts"], dtype=np.int64
        ),
        "domain_total_weights": np.asarray(
            domain_balance_audit["domain_total_weights"], dtype=np.float64
        ),
        "L05": np.asarray(result["L05"], dtype=np.float32),
        "L10": np.asarray(result["L10"], dtype=np.float32),
        "short05_identity_mse": np.asarray(
            result["short05_identity_mse"], dtype=np.float64
        ),
        "short05_trained_mse": np.asarray(
            result["short05_trained_mse"], dtype=np.float64
        ),
        "short10_identity_mse": np.asarray(
            result["short10_identity_mse"], dtype=np.float64
        ),
        "short10_trained_mse": np.asarray(
            result["short10_trained_mse"], dtype=np.float64
        ),
    }
    _atomic_savez(path, arrays)


def load_config(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("训练 config schema_version 必须为 1")
    if tuple(payload.get("seeds", ())) != DEFAULT_SEEDS:
        raise ValueError("训练 config 必须预注册 seeds={}".format(DEFAULT_SEEDS))
    if payload.get("train_split") != "train":
        raise ValueError("训练 config train_split 必须为 train")
    grid = payload.get("hyperparameter_grid")
    if not isinstance(grid, dict):
        raise ValueError("训练 config 缺少 hyperparameter_grid")
    if grid.get("selection_protocol") != SELECTION_PROTOCOL:
        raise ValueError("训练 config selection_protocol 错误")
    _validated_fixed_grid(grid.get("ridge_lambda", ()), "ridge_lambda")
    _validated_fixed_grid(grid.get("alpha", ()), "alpha")
    weighting = payload.get("domain_weighting")
    if (
        not isinstance(weighting, dict)
        or weighting.get("policy") != DOMAIN_WEIGHTING_POLICY
    ):
        raise ValueError("训练 config domain_weighting policy 错误")
    if (
        payload.get("formal_candidate_selection_protocol")
        != FORMAL_CANDIDATE_SELECTION_PROTOCOL
    ):
        raise ValueError("训练 config formal_candidate_selection_protocol 错误")
    if payload.get("formal_materialization_family") != FORMAL_CANDIDATE_FAMILY:
        raise ValueError("训练 config formal_materialization_family 错误")
    if payload.get("formal_selection_report_required") is not True:
        raise ValueError("训练 config 必须要求冻结的候选选择报告")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--candidate-selection-report",
        required=True,
        type=Path,
        help="exact dev-only adapter_ablation_report.json used to freeze the candidate",
    )
    parser.add_argument(
        "--selection-cache",
        required=True,
        type=Path,
        help="exact combined cache bound by the candidate selection report",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-dataset", action="append", default=[])
    parser.add_argument(
        "--allow-v1-cache-migration",
        action="store_true",
        help="显式允许将旧 cache 的 utterance_id 解释为 embedding_id",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    cache = read_embedding_cache(
        arguments.cache,
        allow_v1_migration=arguments.allow_v1_cache_migration,
    )
    minimum_domains = int(config["minimum_training_datasets"])
    if arguments.holdout_dataset:
        # LODO with exactly two source corpora necessarily trains each fold on
        # one remaining corpus.  Final release training still requires >=2.
        minimum_domains = int(config["lodo_minimum_training_datasets"])
    selected = select_training_rows(
        cache,
        train_split=str(config["train_split"]),
        holdout_datasets=arguments.holdout_dataset,
        minimum_training_datasets=minimum_domains,
    )
    datasets = sorted(set(str(value) for value in cache["dataset"][selected]))
    speakers = sorted(
        set(
            (str(cache["dataset"][index]), str(cache["speaker_id"][index]))
            for index in selected
        )
    )
    cache_digest = sha256_file(arguments.cache)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    selection_report_bytes, candidate = load_candidate_selection_report(
        arguments.candidate_selection_report,
        arguments.selection_cache,
    )
    if not arguments.holdout_dataset and cache_digest != candidate["selection_cache_sha256"]:
        raise ValueError(
            "正式全域重训 cache 必须与 dev 候选选择 cache 完全相同"
        )
    if not set(datasets).issubset(set(candidate["selection_train_datasets"])):
        raise ValueError("训练 cache 包含 selection report 未见过的训练域")
    for held_out in arguments.holdout_dataset:
        if str(held_out) not in set(candidate["selection_train_datasets"]):
            raise ValueError("holdout dataset 不在 selection report 训练域：{}".format(held_out))
    selection_snapshot = arguments.output_dir / "candidate_selection_report.snapshot.json"
    _atomic_write_bytes(selection_snapshot, selection_report_bytes)
    if sha256_file(selection_snapshot) != candidate["report_sha256"]:
        raise RuntimeError("candidate selection report 快照回读 SHA-256 不一致")

    report = {
        "schema_version": 2,
        "method": "preregistered_lowrank_ridge_residual_kd",
        "cache": str(arguments.cache.resolve()),
        "cache_sha256": cache_digest,
        "selection_cache": str(arguments.selection_cache.resolve()),
        "selection_cache_sha256": candidate["selection_cache_sha256"],
        "train_split": str(config["train_split"]),
        "holdout_datasets": sorted(arguments.holdout_dataset),
        "minimum_training_datasets": minimum_domains,
        "training_datasets": datasets,
        "selected_sample_count": int(selected.size),
        "selected_speaker_count": len(speakers),
        "teacher_policy": TEACHER_POLICY,
        "domain_weighting_policy": DOMAIN_WEIGHTING_POLICY,
        "selection_protocol": candidate["selection_protocol"],
        "candidate_selection_report": str(selection_snapshot.resolve()),
        "candidate_selection_report_sha256": candidate["report_sha256"],
        "selected_candidate_id": candidate["candidate_id"],
        "selected_candidate_family": candidate["candidate_family"],
        "selected_candidate_rank": candidate["candidate_rank"],
        "selected_ridge_lambda": candidate["ridge_lambda"],
        "selected_alpha": candidate["alpha"],
        "research_matrix_artifact_read": False,
        "selection_recomputed_from_cache": candidate[
            "selection_recomputed_from_cache"
        ],
        "per_seed_refit": True,
        "seeds": [],
    }
    for seed in DEFAULT_SEEDS:
        result = train_selected_lowrank_seed(
            cache,
            selected,
            seed,
            candidate["ridge_lambda"],
            candidate["alpha"],
            candidate["candidate_rank"],
        )
        output_path = arguments.output_dir / "seed_{}.npz".format(seed)
        write_seed_artifact(
            output_path,
            result,
            candidate_metadata=candidate,
            selection_report_bytes=selection_report_bytes,
            train_split=str(config["train_split"]),
            holdout_datasets=arguments.holdout_dataset,
            training_datasets=datasets,
            cache_sha256=cache_digest,
            speaker_count=int(result["teacher_audit"]["eligible_speaker_count"]),
        )
        record = {
            "seed": seed,
            "candidate_id": candidate["candidate_id"],
            "candidate_family": candidate["candidate_family"],
            "candidate_rank": candidate["candidate_rank"],
            "artifact": str(output_path.resolve()),
            "artifact_sha256": sha256_file(output_path),
            "bootstrap_sample_count": int(len(result["indices"])),
            "teacher_audit": result["teacher_audit"],
            "domain_balance_audit": result["domain_balance_audit"],
            "short05_identity_mse": result["short05_identity_mse"],
            "short05_trained_mse": result["short05_trained_mse"],
            "short10_identity_mse": result["short10_identity_mse"],
            "short10_trained_mse": result["short10_trained_mse"],
        }
        report["seeds"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report_path = arguments.output_dir / "training_report.json"
    _atomic_write_json(report_path, report)
    print("training_report={}".format(report_path.resolve()), flush=True)


if __name__ == "__main__":
    main()
