#!/usr/bin/env python3
"""Strict, pickle-free cache format for frozen acoustic embeddings."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping

import numpy as np


CACHE_SCHEMA_VERSION = 2
EMBEDDING_DIM = 448
CACHE_KEYS = (
    "schema_version",
    "embedding_id",
    "utterance_id",
    "speaker_id",
    "dataset",
    "split",
    "full",
    "short05",
    "short10",
)
ALLOWED_SPLITS = ("train", "dev", "test")
V1_CACHE_KEYS = tuple(key for key in CACHE_KEYS if key != "utterance_id")


def _validate_text_vector(value: np.ndarray, name: str, count: int) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1 or value.shape[0] != count or value.dtype.kind != "U":
        raise ValueError(
            "{} 必须是 shape=({},) 的 Unicode 数组，实际 shape={} dtype={}".format(
                name, count, value.shape, value.dtype
            )
        )
    if any(not str(item).strip() for item in value):
        raise ValueError("{} 不允许空字符串".format(name))
    return np.ascontiguousarray(value).copy()


def _validate_embeddings(value: np.ndarray, name: str, count: int) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != (count, EMBEDDING_DIM):
        raise ValueError(
            "{} 形状错误：期望 ({}, {})，实际 {}".format(
                name, count, EMBEDDING_DIM, value.shape
            )
        )
    if value.dtype != np.dtype(np.float32):
        raise ValueError("{} 必须是 float32，实际 {}".format(name, value.dtype))
    if not np.isfinite(value).all():
        raise ValueError("{} 包含 NaN/Inf".format(name))
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms < 1.0e-12):
        raise ValueError("{} 包含零范数 embedding".format(name))
    # acoustic emits unit vectors.  A small tolerance catches wrong caches without
    # rejecting normal float32 accumulation error.
    if np.max(np.abs(norms - 1.0)) > 2.0e-4:
        raise ValueError("{} 未按行 L2 归一化".format(name))
    return np.ascontiguousarray(value).copy()


def _migrate_v1_payload(payload: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Migrate legacy one-row-per-utterance caches to schema v2.

    The only safe automatic interpretation is ``utterance_id=embedding_id``.
    Formal CLI training disables this migration unless the operator explicitly
    passes ``--allow-v1-cache-migration``.
    """

    schema = np.asarray(payload.get("schema_version"))
    if (
        schema.shape != ()
        or schema.dtype != np.dtype(np.int64)
        or int(schema.item()) != 1
        or tuple(sorted(payload.keys())) != tuple(sorted(V1_CACHE_KEYS))
    ):
        raise ValueError("不是可迁移的 schema v1 cache")
    migrated = {key: np.asarray(value) for key, value in payload.items()}
    migrated["schema_version"] = np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64)
    migrated["utterance_id"] = np.asarray(payload["embedding_id"]).copy()
    return migrated


def validate_embedding_cache(
    payload: Mapping[str, np.ndarray], allow_v1_migration: bool = True
) -> Dict[str, np.ndarray]:
    """Validate a cache payload and return owned NumPy arrays."""

    schema_probe = np.asarray(payload.get("schema_version"))
    if (
        allow_v1_migration
        and schema_probe.shape == ()
        and schema_probe.dtype == np.dtype(np.int64)
        and int(schema_probe.item()) == 1
    ):
        payload = _migrate_v1_payload(payload)

    actual_keys = tuple(sorted(payload.keys()))
    expected_keys = tuple(sorted(CACHE_KEYS))
    if actual_keys != expected_keys:
        raise ValueError(
            "cache schema 键错误：expected={} actual={}".format(
                expected_keys, actual_keys
            )
        )
    schema = np.asarray(payload["schema_version"])
    if schema.shape != () or schema.dtype != np.dtype(np.int64):
        raise ValueError("schema_version 必须是 int64 标量")
    if int(schema.item()) != CACHE_SCHEMA_VERSION:
        raise ValueError("不支持的 cache schema_version={}".format(schema.item()))

    embedding_ids_raw = np.asarray(payload["embedding_id"])
    if embedding_ids_raw.ndim != 1:
        raise ValueError("embedding_id 必须是一维")
    count = int(embedding_ids_raw.shape[0])
    if count == 0:
        raise ValueError("cache 不能为空")

    result = {
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
        "embedding_id": _validate_text_vector(
            embedding_ids_raw, "embedding_id", count
        ),
        "utterance_id": _validate_text_vector(
            payload["utterance_id"], "utterance_id", count
        ),
        "speaker_id": _validate_text_vector(
            payload["speaker_id"], "speaker_id", count
        ),
        "dataset": _validate_text_vector(payload["dataset"], "dataset", count),
        "split": _validate_text_vector(payload["split"], "split", count),
        "full": _validate_embeddings(payload["full"], "full", count),
        "short05": _validate_embeddings(payload["short05"], "short05", count),
        "short10": _validate_embeddings(payload["short10"], "short10", count),
    }
    if len(set(str(item) for item in result["embedding_id"])) != count:
        raise ValueError("embedding_id 必须全局唯一")
    unknown_splits = sorted(set(str(item) for item in result["split"]) - set(ALLOWED_SPLITS))
    if unknown_splits:
        raise ValueError("未知 split：{}".format(unknown_splits))
    # Speaker identifiers are dataset-scoped.  A speaker appearing in more
    # than one split inside the same corpus would leak identity information
    # from training into validation/test and invalidates the release gate.
    speaker_splits = {}
    for dataset, speaker, split in zip(
        result["dataset"], result["speaker_id"], result["split"]
    ):
        key = (str(dataset), str(speaker))
        speaker_splits.setdefault(key, set()).add(str(split))
    leaking = sorted(
        "{}:{}".format(dataset, speaker)
        for (dataset, speaker), splits in speaker_splits.items()
        if len(splits) > 1
    )
    if leaking:
        raise ValueError(
            "同一 dataset/speaker 不允许跨 split：{}".format(leaking[:10])
        )
    return result


def read_embedding_cache(
    path: Path, allow_v1_migration: bool = True
) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("embedding cache 不存在：{}".format(path))
    with np.load(str(path), allow_pickle=False) as payload:
        copied = {key: payload[key] for key in payload.files}
    return validate_embedding_cache(copied, allow_v1_migration=allow_v1_migration)


def write_embedding_cache(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    """Atomically write a validated cache."""

    path = Path(path)
    validated = validate_embedding_cache(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **validated)
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
