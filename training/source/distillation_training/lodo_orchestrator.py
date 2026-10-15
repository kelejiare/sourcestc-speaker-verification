#!/usr/bin/env python3
"""Assemble and score the pre-registered distillation real-data LODO protocol.

This module is deliberately an *orchestrator*, not an evidence generator.  It
combines already-built schema-v2 embedding caches, constructs deterministic
trials from source annotations, and projects traceable scores into the frozen
``tools/evaluate_distillation_gates.py`` schema.  It never fabricates model weights,
labels, scores, or a release gate.

The protocol has five base scenarios.  Each is bound to exactly one held-out
dataset and expands to full/0.5 s/1.0 s, giving 15 scenarios total.  The full
candidate score is assigned from the acoustic baseline score exactly; only the two
short branches may pass through the duration-specific KD matrices.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
for _path in (PROJECT_ROOT, TOOLS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from distillation_training.cache_schema import (
    CACHE_SCHEMA_VERSION,
    EMBEDDING_DIM,
    read_embedding_cache,
    validate_embedding_cache,
    write_embedding_cache,
)
from distillation_training.export_adapter import load_seed_artifact
from distillation_training.train_ridge_kd import (
    DEFAULT_SEEDS,
    load_candidate_selection_report,
    sha256_file,
)
from distillation_build_manifest import (
    MANIFEST_FIELDS,
    ManifestError,
    canonicalize_record,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from evaluate_distillation_gates import (
    TRIAL_FIELDS,
    canonicalize_trial,
    main as evaluate_gates_main,
    validate_trial_design,
)


ORCHESTRATOR_SCHEMA_VERSION = 1
SOURCE_CONFIG_SCHEMA_VERSION = 1
PROTOCOL_SCHEMA_VERSION = 1
REQUIRED_DATASETS = ("3dspeaker", "aishell4", "childmandarin")
POST_CV_AUXILIARY_DATASETS = (
    "commonvoice_beijing",
    "magicdata_slr68",
    "seniortalk",
)
DURATION_VARIANTS = (
    ("full", "full", False),
    ("0p5", "short05", True),
    ("1p0", "short10", True),
)
BASE_SCENARIOS = (
    {
        "name": "mobile_age_mix",
        "dataset": "childmandarin",
        "domain": "proxy/mobile_age_mix",
        "constructor": "explicit_age_labeled_mobile_pairs",
        "categories": ("child",),
        "ratio_groups": ("child", "young", "elder"),
        "ratio": (7, 6, 1),
        "balance_policy": "fixed_test_age_ratio",
    },
    {
        "name": "mobile_gender_mix",
        "dataset": "childmandarin",
        "domain": "proxy/mobile_gender_mix",
        "constructor": "explicit_gender_labeled_mobile_pairs",
        "categories": ("child", "same_gender"),
        "ratio_groups": ("male_male", "female_female"),
        "ratio": (4, 7),
        "balance_policy": "fixed_enroll_test_gender_ratio",
    },
    {
        "name": "near_to_far",
        "dataset": "3dspeaker",
        "domain": "proxy/near_to_far",
        "constructor": "explicit_paired_near_enroll_far_test",
        "categories": ("far",),
        "ratio_groups": (
            "child_male",
            "child_female",
            "young_male",
            "young_female",
            "elder_male",
            "elder_female",
        ),
        "ratio": (1, 1, 1, 1, 1, 1),
        "balance_policy": "equal_explicit_test_age_gender_cells",
    },
    {
        "name": "meeting_age_mix",
        "dataset": "aishell4",
        "domain": "proxy/meeting_age_mix",
        "constructor": "explicit_age_labeled_meeting_pairs",
        "categories": ("child", "meeting"),
        "ratio_groups": ("child", "young"),
        "ratio": (1, 6),
        "balance_policy": "fixed_test_age_ratio",
    },
    {
        "name": "meeting_gender_mix",
        "dataset": "aishell4",
        "domain": "proxy/meeting_gender_mix",
        "constructor": "explicit_gender_labeled_meeting_pairs",
        "categories": ("meeting",),
        "ratio_groups": ("male_male", "male_female", "female_female"),
        "ratio": (9, 2, 9),
        "balance_policy": "fixed_enroll_test_gender_ratio",
    },
)
BASE_SCENARIO_BY_NAME = {item["name"]: item for item in BASE_SCENARIOS}
COVERAGE_SCENARIOS = (
    {
        "name": "coverage_child_clean",
        "dataset": "childmandarin",
        "domain": "coverage/childmandarin_clean",
        "categories": ("child",),
    },
    {
        "name": "coverage_child_same_gender",
        "dataset": "childmandarin",
        "domain": "coverage/childmandarin_same_gender",
        "categories": ("child", "same_gender"),
    },
    {
        "name": "coverage_cross_device",
        "dataset": "3dspeaker",
        "domain": "coverage/3dspeaker_cross_device",
        "categories": (),
    },
    {
        "name": "coverage_cross_distance",
        "dataset": "3dspeaker",
        "domain": "coverage/3dspeaker_cross_distance",
        "categories": ("far",),
    },
    {
        "name": "coverage_meeting_same_session",
        "dataset": "aishell4",
        "domain": "coverage/aishell4_same_session",
        "categories": ("meeting",),
    },
)
COVERAGE_SCENARIO_BY_NAME = {item["name"]: item for item in COVERAGE_SCENARIOS}
ALL_SCENARIO_BY_NAME = dict(BASE_SCENARIO_BY_NAME)
ALL_SCENARIO_BY_NAME.update(COVERAGE_SCENARIO_BY_NAME)
_DEVICE_PATTERN = re.compile(r"(?:^|_)Device([^_]+)(?:_|$)", re.IGNORECASE)
_DISTANCE_PATTERN = re.compile(r"(?:^|_)Distance([^_]+)(?:_|$)", re.IGNORECASE)
OFFICIAL_PAIR_COMMON_FIELDS = (
    "base_scenario",
    "enrollment_utterance_id",
    "test_utterance_id",
    "label",
    "evidence_scope",
    "enrollment_context",
    "test_context",
    "annotation_source",
    "annotation_sha256",
)
OFFICIAL_PAIR_DEMOGRAPHIC_FIELDS = (
    "enrollment_age_group",
    "test_age_group",
    "enrollment_gender",
    "test_gender",
)
# Stable serialization/header order.  Presence is scenario-conditional; see
# OFFICIAL_PAIR_SCHEMA.  Optional fields are normalized to the literal
# ``unknown`` before a base-trial artifact is written.
OFFICIAL_PAIR_FIELDS = (
    OFFICIAL_PAIR_COMMON_FIELDS[:5]
    + OFFICIAL_PAIR_DEMOGRAPHIC_FIELDS
    + OFFICIAL_PAIR_COMMON_FIELDS[5:]
)
OFFICIAL_PAIR_SCHEMA_VERSION = 1
OFFICIAL_PAIR_SCHEMA = {
    "mobile_age_mix": {
        "required_demographics": ("enrollment_age_group", "test_age_group"),
        "optional_demographics": ("enrollment_gender", "test_gender"),
    },
    "meeting_age_mix": {
        "required_demographics": ("enrollment_age_group", "test_age_group"),
        "optional_demographics": ("enrollment_gender", "test_gender"),
    },
    "mobile_gender_mix": {
        "required_demographics": ("enrollment_gender", "test_gender"),
        "optional_demographics": ("enrollment_age_group", "test_age_group"),
    },
    "meeting_gender_mix": {
        "required_demographics": ("enrollment_gender", "test_gender"),
        "optional_demographics": ("enrollment_age_group", "test_age_group"),
    },
    "near_to_far": {
        "required_demographics": OFFICIAL_PAIR_DEMOGRAPHIC_FIELDS,
        "optional_demographics": (),
    },
}
CANONICAL_AGE_GROUPS = ("child", "young", "elder")
UNKNOWN_DEMOGRAPHIC_TOKENS = (
    "",
    "unknown",
    "unspecified",
    "n/a",
    "na",
    "none",
    "null",
)


class OrchestrationError(ValueError):
    """Raised when data provenance or the frozen protocol is violated."""


def _stable_sha256(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_bytes(path, content)


def atomic_write_csv(
    path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields))
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in fields})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def utterance_id_for_record(record: Mapping[str, str]) -> str:
    return "{}:{}".format(record["dataset"], Path(record["path"]).stem)


def protocol_payload() -> Dict[str, object]:
    scenarios = []
    for base in BASE_SCENARIOS:
        for suffix, cache_field, is_short in DURATION_VARIANTS:
            scenarios.append(
                {
                    "scenario": "{}_{}".format(base["name"], suffix),
                    "base_scenario": base["name"],
                    "held_out_dataset": base["dataset"],
                    "dataset": base["dataset"],
                    "domain": base["domain"],
                    "constructor": base["constructor"],
                    "cache_field": cache_field,
                    "is_short": is_short,
                    "categories": list(base["categories"]),
                    "ratio_groups": list(base["ratio_groups"]),
                    "ratio": list(base["ratio"]),
                    "balance_policy": base["balance_policy"],
                    "enrollment_cache_field": "full",
                }
            )
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "artifact_kind": "distillation_lodo_15_scenario_protocol",
        "official_pair_schema_version": OFFICIAL_PAIR_SCHEMA_VERSION,
        "official_pair_schema": {
            name: {
                "required_fields": list(
                    OFFICIAL_PAIR_COMMON_FIELDS
                    + tuple(specification["required_demographics"])
                ),
                "optional_demographics": list(
                    specification["optional_demographics"]
                ),
            }
            for name, specification in sorted(OFFICIAL_PAIR_SCHEMA.items())
        },
        "required_datasets": list(REQUIRED_DATASETS),
        "seeds": list(DEFAULT_SEEDS),
        "full_policy": "candidate_score_bit_exact_assignment_from_acoustic_baseline",
        "evidence_scope": "real_audio_proxy_protocol_not_hidden_test_equivalence",
        "unknown_age_policy": "reject_never_infer_young_or_elder_from_dataset",
        "cross_corpus_policy": "proxy_only_with_corpus_bias_sensitivity",
        "scenario_count": len(scenarios),
        "short_scenario_count": sum(bool(item["is_short"]) for item in scenarios),
        "scenarios": scenarios,
    }


def validate_protocol(payload: Mapping[str, object]) -> Dict[str, object]:
    expected = protocol_payload()
    if dict(payload) != expected:
        raise OrchestrationError(
            "LODO protocol 与代码中预注册的 15 场景定义不一致"
        )
    return expected


def _resolve_path(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = (Path(config_path).parent / path).resolve()
    return path


def load_source_config(
    path: Path, allow_training_auxiliary: bool = False
) -> Dict[str, Dict[str, object]]:
    """Load three LODO bindings plus explicitly scoped auxiliary sources.

    ``training_auxiliary`` is accepted only for the separate post-CV
    follow-up diagnostic. Keeping it behind an explicit caller flag prevents
    the original pre-CV/formal paths from silently changing their population.
    """

    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OrchestrationError("无法读取 source config：{}".format(error))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise OrchestrationError("source config schema_version 必须为 1")
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) < 3:
        raise OrchestrationError("source config 至少包含三个 LODO sources")
    result = {}
    for raw in sources:
        if not isinstance(raw, dict):
            raise OrchestrationError("每个 source 必须是 JSON object")
        dataset = str(raw.get("dataset", "")).strip()
        if not dataset or dataset in result:
            raise OrchestrationError("source dataset 为空或重复：{!r}".format(dataset))
        if "manifest" not in raw or "cache" not in raw:
            raise OrchestrationError("source {} 缺少 manifest/cache".format(dataset))
        trial_sources = raw.get("trial_sources", {})
        if not isinstance(trial_sources, dict):
            raise OrchestrationError("trial_sources 必须是 object")
        role = str(raw.get("role", "lodo")).strip().lower()
        if dataset in REQUIRED_DATASETS and role != "lodo":
            raise OrchestrationError("预注册 LODO dataset {} 的 role 必须为 lodo".format(dataset))
        if dataset not in REQUIRED_DATASETS:
            allowed_roles = {"trial_only_auxiliary"}
            if allow_training_auxiliary:
                allowed_roles.add("training_auxiliary")
            if role not in allowed_roles:
                raise OrchestrationError(
                    "额外 dataset {} role 必须属于 {}".format(
                        dataset, sorted(allowed_roles)
                    )
                )
        result[dataset] = {
            "dataset": dataset,
            "role": role,
            "manifest": _resolve_path(path, raw["manifest"]),
            "cache": _resolve_path(path, raw["cache"]),
            "trial_sources": {
                str(name): _resolve_path(path, value)
                for name, value in trial_sources.items()
            },
        }
    if not set(REQUIRED_DATASETS).issubset(set(result)):
        raise OrchestrationError(
            "source datasets 必须至少包含 {}，实际 {}".format(
                REQUIRED_DATASETS, tuple(sorted(result))
            )
        )
    required_3d = {"cross_device", "far_distance"}
    present_3d = set(result["3dspeaker"]["trial_sources"])
    if present_3d != required_3d:
        raise OrchestrationError(
            "3dspeaker trial_sources 必须严格包含 {}".format(sorted(required_3d))
        )
    return result


def validate_post_cv_auxiliary_source_specs(
    source_specs: Mapping[str, Mapping[str, object]]
) -> Tuple[str, ...]:
    """Freeze the exact auxiliary-domain population for the follow-up LODO."""

    expected = tuple(sorted(REQUIRED_DATASETS + POST_CV_AUXILIARY_DATASETS))
    if tuple(sorted(source_specs)) != expected:
        raise OrchestrationError(
            "post-CV auxiliary LODO 必须严格包含 primary={} 与 auxiliary={}".format(
                REQUIRED_DATASETS, POST_CV_AUXILIARY_DATASETS
            )
        )
    for dataset in REQUIRED_DATASETS:
        if str(source_specs[dataset].get("role")) != "lodo":
            raise OrchestrationError("primary dataset {} role 必须为 lodo".format(dataset))
    for dataset in POST_CV_AUXILIARY_DATASETS:
        if str(source_specs[dataset].get("role")) != "training_auxiliary":
            raise OrchestrationError(
                "auxiliary dataset {} role 必须为 training_auxiliary".format(dataset)
            )
    return POST_CV_AUXILIARY_DATASETS


def _manifest_index(
    records: Sequence[Mapping[str, str]], dataset: str
) -> Dict[str, Dict[str, str]]:
    result = {}
    for raw in records:
        record = canonicalize_record(raw, namespace_speakers=True)
        if record["dataset"] != dataset:
            raise OrchestrationError(
                "manifest dataset 不匹配：期望 {}，实际 {}".format(
                    dataset, record["dataset"]
                )
            )
        utterance_id = utterance_id_for_record(record)
        if utterance_id in result:
            raise OrchestrationError(
                "manifest path stem 不能唯一映射 utterance_id：{}".format(utterance_id)
            )
        result[utterance_id] = record
    return result


def validate_cache_manifest_alignment(
    records: Sequence[Mapping[str, str]],
    cache: Mapping[str, np.ndarray],
    dataset: str,
) -> Dict[str, object]:
    """Prove every cache row maps to one source-manifest utterance."""

    index = _manifest_index(records, dataset)
    cache_datasets = {str(value) for value in cache["dataset"]}
    if cache_datasets != {dataset}:
        raise OrchestrationError(
            "cache {} 包含错误 dataset：{}".format(dataset, sorted(cache_datasets))
        )
    covered = set()
    view_counts = Counter()
    for utterance, speaker, split in zip(
        cache["utterance_id"], cache["speaker_id"], cache["split"]
    ):
        utterance_id = str(utterance)
        if utterance_id not in index:
            raise OrchestrationError(
                "cache utterance 不在 manifest：{}".format(utterance_id)
            )
        record = index[utterance_id]
        if str(speaker) != record["speaker_id"] or str(split) != record["split"]:
            raise OrchestrationError(
                "cache/manifest speaker 或 split 不一致：{}".format(utterance_id)
            )
        covered.add(utterance_id)
        view_counts[utterance_id] += 1
    return {
        "dataset": dataset,
        "manifest_utterance_count": len(index),
        "cache_utterance_count": len(covered),
        "cache_row_count": int(np.asarray(cache["embedding_id"]).shape[0]),
        "minimum_views_per_utterance": min(view_counts.values()),
        "maximum_views_per_utterance": max(view_counts.values()),
        "all_cache_rows_mapped": True,
    }


def load_sources(
    source_specs: Mapping[str, Mapping[str, object]]
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Dict[str, np.ndarray]], List[Dict[str, object]]]:
    manifests = {}
    caches = {}
    audits = []
    for dataset in sorted(source_specs):
        spec = source_specs[dataset]
        manifest_path = Path(spec["manifest"])
        cache_path = Path(spec["cache"])
        records = read_manifest(manifest_path, namespace_speakers=True)
        # A source manifest retains its own speaker-disjoint train/dev/test.
        validate_manifest(records, require_speaker_disjoint=True)
        cache = read_embedding_cache(cache_path, allow_v1_migration=False)
        alignment = validate_cache_manifest_alignment(records, cache, dataset)
        alignment.update(
            {
                "role": str(spec.get("role", "lodo")),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "cache": str(cache_path.resolve()),
                "cache_sha256": sha256_file(cache_path),
            }
        )
        manifests[dataset] = records
        caches[dataset] = cache
        audits.append(alignment)
    return manifests, caches, audits


def _concatenate_cache_parts(
    parts: Sequence[Mapping[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    if not parts:
        raise OrchestrationError("没有可合并的 cache")
    output = {"schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64)}
    for key in (
        "embedding_id",
        "utterance_id",
        "speaker_id",
        "dataset",
        "split",
        "full",
        "short05",
        "short10",
    ):
        output[key] = np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
    return validate_embedding_cache(output, allow_v1_migration=False)


def assemble_lodo_fold(
    manifests: Mapping[str, Sequence[Mapping[str, str]]],
    caches: Mapping[str, Mapping[str, np.ndarray]],
    held_out_dataset: str,
) -> Tuple[List[Dict[str, str]], Dict[str, np.ndarray], Dict[str, object]]:
    """Build one true LODO fold while retaining source train/dev only."""

    held_out_dataset = str(held_out_dataset)
    if tuple(sorted(manifests)) != REQUIRED_DATASETS or tuple(sorted(caches)) != REQUIRED_DATASETS:
        raise OrchestrationError("LODO 组装必须收到恰好三套预注册数据")
    if held_out_dataset not in REQUIRED_DATASETS:
        raise OrchestrationError("未知 held-out dataset：{}".format(held_out_dataset))

    fold_records = []
    cache_parts = []
    dropped_manifest = Counter()
    dropped_cache = Counter()
    for dataset in REQUIRED_DATASETS:
        for raw in manifests[dataset]:
            record = canonicalize_record(raw, namespace_speakers=True)
            if dataset == held_out_dataset:
                record["split"] = "test"
                fold_records.append(record)
            elif record["split"] in ("train", "dev"):
                fold_records.append(record)
            else:
                dropped_manifest[dataset] += 1

        cache = caches[dataset]
        if dataset == held_out_dataset:
            selected = np.arange(len(cache["embedding_id"]), dtype=np.int64)
            splits = np.asarray(["test"] * len(selected))
        else:
            selected = np.flatnonzero(np.isin(cache["split"], ("train", "dev")))
            dropped_cache[dataset] += int(len(cache["embedding_id"]) - len(selected))
            splits = np.asarray(cache["split"])[selected].copy()
        if selected.size == 0:
            raise OrchestrationError("fold {} 的 {} cache 为空".format(held_out_dataset, dataset))
        part = {
            "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
            "split": splits,
        }
        for key in (
            "embedding_id",
            "utterance_id",
            "speaker_id",
            "dataset",
            "full",
            "short05",
            "short10",
        ):
            part[key] = np.asarray(cache[key])[selected].copy()
        cache_parts.append(part)

    fold_records = sorted(
        fold_records,
        key=lambda row: (row["dataset"], row["split"], row["speaker_id"], row["path"]),
    )
    audit = validate_manifest(
        fold_records,
        require_speaker_disjoint=True,
        require_domain_disjoint=True,
        lodo_held_out_dataset=held_out_dataset,
    )
    fold_cache = _concatenate_cache_parts(cache_parts)
    cache_split_datasets = {
        split: sorted(
            set(
                str(dataset)
                for dataset, row_split in zip(fold_cache["dataset"], fold_cache["split"])
                if str(row_split) == split
            )
        )
        for split in ("train", "dev", "test")
    }
    if cache_split_datasets["test"] != [held_out_dataset]:
        raise OrchestrationError("fold cache test 不是唯一 held-out dataset")
    if held_out_dataset in cache_split_datasets["train"] + cache_split_datasets["dev"]:
        raise OrchestrationError("fold cache 存在 held-out 泄漏")
    audit.update(
        {
            "artifact_kind": "distillation_lodo_fold",
            "held_out_dataset": held_out_dataset,
            "source_datasets": sorted(set(REQUIRED_DATASETS) - {held_out_dataset}),
            "dropped_source_test_manifest_rows": dict(sorted(dropped_manifest.items())),
            "dropped_source_test_cache_rows": dict(sorted(dropped_cache.items())),
            "cache_row_count": int(len(fold_cache["embedding_id"])),
            "cache_datasets_by_split": cache_split_datasets,
        }
    )
    return fold_records, fold_cache, audit


def assemble_post_cv_auxiliary_lodo_fold(
    manifests: Mapping[str, Sequence[Mapping[str, str]]],
    caches: Mapping[str, Mapping[str, np.ndarray]],
    held_out_dataset: str,
    auxiliary_datasets: Sequence[str] = POST_CV_AUXILIARY_DATASETS,
) -> Tuple[List[Dict[str, str]], Dict[str, np.ndarray], Dict[str, object]]:
    """Assemble one post-CV auxiliary-domain LODO diagnostic fold.

    The held-out primary corpus is test-only. The other primary corpora keep
    train/dev, while auxiliary corpora contribute train rows only. This is a
    separate follow-up diagnostic and is never formal-gate eligible.
    """

    held_out_dataset = str(held_out_dataset)
    auxiliary_datasets = tuple(sorted(str(value) for value in auxiliary_datasets))
    expected_auxiliary = tuple(sorted(POST_CV_AUXILIARY_DATASETS))
    if auxiliary_datasets != expected_auxiliary:
        raise OrchestrationError(
            "post-CV auxiliary dataset 集合必须为 {}".format(expected_auxiliary)
        )
    expected = set(REQUIRED_DATASETS).union(auxiliary_datasets)
    if set(manifests) != expected or set(caches) != expected:
        raise OrchestrationError(
            "post-CV auxiliary LODO source 集合错误：期望 {}".format(sorted(expected))
        )
    if held_out_dataset not in REQUIRED_DATASETS:
        raise OrchestrationError("未知 held-out dataset：{}".format(held_out_dataset))

    fold_records = []
    cache_parts = []
    dropped_manifest = Counter()
    dropped_cache = Counter()
    for dataset in sorted(expected):
        is_primary = dataset in REQUIRED_DATASETS
        for raw in manifests[dataset]:
            record = canonicalize_record(raw, namespace_speakers=True)
            if dataset == held_out_dataset:
                record["split"] = "test"
                fold_records.append(record)
            elif is_primary and record["split"] in ("train", "dev"):
                fold_records.append(record)
            elif not is_primary and record["split"] == "train":
                fold_records.append(record)
            else:
                dropped_manifest[dataset] += 1

        cache = caches[dataset]
        if dataset == held_out_dataset:
            selected = np.arange(len(cache["embedding_id"]), dtype=np.int64)
            splits = np.asarray(["test"] * len(selected))
        elif is_primary:
            selected = np.flatnonzero(np.isin(cache["split"], ("train", "dev")))
            splits = np.asarray(cache["split"])[selected].copy()
        else:
            selected = np.flatnonzero(np.asarray(cache["split"]) == "train")
            splits = np.asarray(["train"] * len(selected))
        dropped_cache[dataset] += int(len(cache["embedding_id"]) - len(selected))
        if selected.size == 0:
            raise OrchestrationError(
                "post-CV fold {} 的 {} cache 选择后为空".format(
                    held_out_dataset, dataset
                )
            )
        part = {
            "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
            "split": splits,
        }
        for key in (
            "embedding_id",
            "utterance_id",
            "speaker_id",
            "dataset",
            "full",
            "short05",
            "short10",
        ):
            part[key] = np.asarray(cache[key])[selected].copy()
        cache_parts.append(part)

    fold_records = sorted(
        fold_records,
        key=lambda row: (row["dataset"], row["split"], row["speaker_id"], row["path"]),
    )
    fold_cache = _concatenate_cache_parts(cache_parts)
    audit = validate_manifest(
        fold_records,
        require_speaker_disjoint=True,
        require_domain_disjoint=True,
        lodo_held_out_dataset=held_out_dataset,
    )
    datasets_by_split = {
        split: sorted(
            {
                str(dataset)
                for dataset, value in zip(fold_cache["dataset"], fold_cache["split"])
                if str(value) == split
            }
        )
        for split in ("train", "dev", "test")
    }
    expected_train = sorted(
        (set(REQUIRED_DATASETS) - {held_out_dataset}).union(auxiliary_datasets)
    )
    expected_dev = sorted(set(REQUIRED_DATASETS) - {held_out_dataset})
    if datasets_by_split["train"] != expected_train:
        raise OrchestrationError(
            "post-CV fold train dataset 错误：{}".format(datasets_by_split["train"])
        )
    if datasets_by_split["dev"] != expected_dev:
        raise OrchestrationError(
            "post-CV fold dev 必须只含非 held-out primary：{}".format(
                datasets_by_split["dev"]
            )
        )
    if datasets_by_split["test"] != [held_out_dataset]:
        raise OrchestrationError("post-CV fold test 不是唯一 held-out primary")

    embedding_ids = [str(value) for value in fold_cache["embedding_id"]]
    if len(embedding_ids) != len(set(embedding_ids)):
        raise OrchestrationError("post-CV fold 存在重复 embedding endpoint")
    utterance_ids = [utterance_id_for_record(row) for row in fold_records]
    if len(utterance_ids) != len(set(utterance_ids)):
        raise OrchestrationError("post-CV fold manifest 存在重复 utterance endpoint")

    audit.update(
        {
            "schema_version": 2,
            "artifact_kind": "distillation_post_cv_auxiliary_lodo_fold",
            "status": "FOLLOW_UP_DIAGNOSTIC_DO_NOT_SUBMIT",
            "held_out_dataset": held_out_dataset,
            "primary_datasets": list(REQUIRED_DATASETS),
            "auxiliary_training_datasets": list(auxiliary_datasets),
            "training_datasets": expected_train,
            "development_datasets": expected_dev,
            "dropped_manifest_rows_by_dataset": dict(sorted(dropped_manifest.items())),
            "dropped_cache_rows_by_dataset": dict(sorted(dropped_cache.items())),
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_row_count": int(len(fold_cache["embedding_id"])),
            "cache_datasets_by_split": datasets_by_split,
            "heldout_primary_forced_to_test_only": True,
            "auxiliary_train_rows_only": True,
            "cache_embedding_endpoint_ids_unique": True,
            "manifest_utterance_endpoint_ids_unique": True,
            "candidate_selected_before_this_followup_diagnostic": True,
            "formal_gate_eligible": False,
        }
    )
    return fold_records, fold_cache, audit


def write_post_cv_auxiliary_lodo_folds(
    source_specs: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    candidate_selection_report: Path,
    selection_cache: Path,
) -> Dict[str, object]:
    """Write three isolated schema-v2 post-CV auxiliary LODO folds."""

    auxiliary_datasets = validate_post_cv_auxiliary_source_specs(source_specs)
    manifests, caches, source_audits = load_sources(source_specs)
    output_dir = Path(output_dir).resolve()
    fold_receipts = []
    for heldout in REQUIRED_DATASETS:
        fold_records, fold_cache, audit = assemble_post_cv_auxiliary_lodo_fold(
            manifests, caches, heldout, auxiliary_datasets
        )
        fold_dir = output_dir / "folds" / heldout
        manifest_path = fold_dir / "manifest.csv"
        cache_path = fold_dir / "cache_v2.npz"
        audit_path = fold_dir / "audit.json"
        write_manifest(fold_records, manifest_path)
        write_embedding_cache(cache_path, fold_cache)
        audit.update(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "cache": str(cache_path),
                "cache_sha256": sha256_file(cache_path),
            }
        )
        atomic_write_json(audit_path, audit)
        fold_receipts.append(dict(audit))
    receipt = {
        "schema_version": 2,
        "artifact_kind": "distillation_post_cv_auxiliary_lodo_fold_receipt",
        "status": "FOLLOW_UP_DIAGNOSTIC_DO_NOT_SUBMIT",
        "source_audits": source_audits,
        "folds": fold_receipts,
        "auxiliary_training_datasets": list(auxiliary_datasets),
        "candidate_selected_before_this_followup_diagnostic": True,
        "formal_15_scenario_gate_eligible": False,
        "formal_release_adapter_written": False,
        "formal_submission_tar_written": False,
    }
    atomic_write_json(output_dir / "lodo_receipt.json", receipt)
    plan = write_training_plan(
        output_dir,
        fold_receipts,
        candidate_selection_report=candidate_selection_report,
        selection_cache=selection_cache,
    )
    plan.update(
        {
            "schema_version": 2,
            "artifact_kind": "distillation_post_cv_auxiliary_lodo_training_plan",
            "status": "FOLLOW_UP_DIAGNOSTIC_DO_NOT_SUBMIT",
            "auxiliary_training_datasets": list(auxiliary_datasets),
            "candidate_selected_before_this_followup_diagnostic": True,
            "formal_gate_eligible": False,
        }
    )
    atomic_write_json(output_dir / "training_plan.json", plan)
    return receipt


def write_lodo_folds(
    source_specs: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    candidate_selection_report: Optional[Path] = None,
    selection_cache: Optional[Path] = None,
) -> Dict[str, object]:
    manifests, caches, source_audits = load_sources(source_specs)
    primary_manifests = {name: manifests[name] for name in REQUIRED_DATASETS}
    primary_caches = {name: caches[name] for name in REQUIRED_DATASETS}
    output_dir = Path(output_dir).resolve()
    fold_receipts = []
    for heldout in REQUIRED_DATASETS:
        fold_records, fold_cache, audit = assemble_lodo_fold(
            primary_manifests, primary_caches, heldout
        )
        fold_dir = output_dir / "folds" / heldout
        manifest_path = fold_dir / "manifest.csv"
        cache_path = fold_dir / "cache_v2.npz"
        audit_path = fold_dir / "audit.json"
        # write_manifest is deterministic; write into the fold directory only.
        write_manifest(fold_records, manifest_path)
        write_embedding_cache(cache_path, fold_cache)
        audit.update(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "cache": str(cache_path),
                "cache_sha256": sha256_file(cache_path),
            }
        )
        atomic_write_json(audit_path, audit)
        fold_receipts.append(dict(audit))
    receipt = {
        "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "artifact_kind": "distillation_lodo_fold_receipt",
        "protocol": protocol_payload(),
        "source_audits": source_audits,
        "trial_only_auxiliary_datasets_excluded_from_training": sorted(
            set(source_specs) - set(REQUIRED_DATASETS)
        ),
        "folds": fold_receipts,
        "formal_release_adapter_written": False,
        "formal_submission_tar_written": False,
    }
    atomic_write_json(output_dir / "lodo_receipt.json", receipt)
    write_training_plan(
        output_dir,
        fold_receipts,
        candidate_selection_report=candidate_selection_report,
        selection_cache=selection_cache,
    )
    return receipt


def write_training_plan(
    output_dir: Path,
    fold_receipts: Sequence[Mapping[str, object]],
    candidate_selection_report: Optional[Path] = None,
    selection_cache: Optional[Path] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir).resolve()
    if (candidate_selection_report is None) != (selection_cache is None):
        raise OrchestrationError(
            "candidate_selection_report 与 selection_cache 必须同时提供"
        )
    selection = None
    if candidate_selection_report is not None:
        _report_bytes, selection = load_candidate_selection_report(
            Path(candidate_selection_report), Path(selection_cache)
        )
    commands = []
    for fold in fold_receipts:
        heldout = str(fold["held_out_dataset"])
        argv = []
        if selection is not None:
            argv = [
                sys.executable,
                str(PROJECT_ROOT / "distillation_training/train_ridge_kd.py"),
                "--cache",
                str(fold["cache"]),
                "--candidate-selection-report",
                str(Path(candidate_selection_report).resolve()),
                "--selection-cache",
                str(Path(selection_cache).resolve()),
                "--config",
                str(PROJECT_ROOT / "distillation_training/config.json"),
                "--holdout-dataset",
                heldout,
                "--output-dir",
                str(output_dir / "training" / heldout),
            ]
        commands.append(
            {
                "held_out_dataset": heldout,
                "status": "READY" if selection is not None else "BLOCKED",
                "blocked_reason": (
                    None
                    if selection is not None
                    else "missing frozen candidate-selection report/cache binding"
                ),
                "argv": argv,
            }
        )
    payload = {
        "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "artifact_kind": "distillation_lodo_training_plan",
        "selection_policy": "one global dev-only pre-registered candidate frozen across folds",
        "candidate_selection": (
            None
            if selection is None
            else {
                "candidate_id": selection["candidate_id"],
                "candidate_family": selection["candidate_family"],
                "candidate_rank": selection["candidate_rank"],
                "report": str(Path(candidate_selection_report).resolve()),
                "report_sha256": selection["report_sha256"],
                "selection_cache": str(Path(selection_cache).resolve()),
                "selection_cache_sha256": selection["selection_cache_sha256"],
            }
        ),
        "seeds": list(DEFAULT_SEEDS),
        "commands": commands,
        "executed": False,
        "note": "This plan is provenance only; run commands explicitly after reviewing folds.",
    }
    atomic_write_json(output_dir / "training_plan.json", payload)
    return payload


def _known_gender(value: object) -> Optional[str]:
    text = str(value).strip().lower()
    if text in ("m", "male", "man", "boy", "男"):
        return "male"
    if text in ("f", "female", "woman", "girl", "女"):
        return "female"
    return None


def _record_trial(
    base_scenario: str,
    enrollment: Mapping[str, str],
    test: Mapping[str, str],
    label: int,
    source_kind: str,
    source_file: str = "",
    source_sha256: str = "",
    source_line: int = 0,
    evidence_scope: str = "coverage_only",
) -> Dict[str, object]:
    if base_scenario not in ALL_SCENARIO_BY_NAME:
        raise OrchestrationError("未知场景：{}".format(base_scenario))
    spec = ALL_SCENARIO_BY_NAME[base_scenario]
    if enrollment["dataset"] != spec["dataset"]:
        raise OrchestrationError(
            "场景 {} 的 enrollment 必须属于绑定数据集 {}".format(
                base_scenario, spec["dataset"]
            )
        )
    if int(label) not in (0, 1):
        raise OrchestrationError("trial label 必须为 0/1")
    if (enrollment["speaker_id"] == test["speaker_id"]) != bool(label):
        raise OrchestrationError("trial label 与 speaker identity 不一致")
    enrollment_utterance = utterance_id_for_record(enrollment)
    test_utterance = utterance_id_for_record(test)
    if enrollment_utterance == test_utterance:
        raise OrchestrationError("trial 不允许 enrollment/test 为同一 utterance")
    identity = _stable_sha256(
        base_scenario,
        enrollment_utterance,
        test_utterance,
        int(label),
        source_kind,
        source_file,
        source_line,
    )
    return {
        "trial_id": "{}-{}".format(base_scenario, identity[:24]),
        "base_scenario": base_scenario,
        "dataset": spec["dataset"],
        "domain": spec["domain"],
        "held_out_dataset": spec["dataset"],
        "categories": list(spec["categories"]),
        "label": int(label),
        "speaker_id": enrollment["speaker_id"],
        "enrollment_speaker_id": enrollment["speaker_id"],
        "test_speaker_id": test["speaker_id"],
        "enrollment_utterance_id": enrollment_utterance,
        "test_utterance_id": test_utterance,
        "enrollment_path": enrollment["path"],
        "test_path": test["path"],
        "enrollment_source_domain": enrollment["domain"],
        "test_source_domain": test["domain"],
        "enrollment_dataset": enrollment["dataset"],
        "test_dataset": test["dataset"],
        "cross_corpus": enrollment["dataset"] != test["dataset"],
        "evidence_scope": evidence_scope,
        "source_kind": source_kind,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "source_line": int(source_line),
    }


def _rank_records(records: Sequence[Mapping[str, str]], *salt: object) -> List[Mapping[str, str]]:
    return sorted(
        records,
        key=lambda row: _stable_sha256(
            *salt, row["dataset"], row["speaker_id"], row["path"]
        ),
    )


def build_child_trials(
    records: Sequence[Mapping[str, str]],
    same_gender: bool,
    allowed_utterance_ids: Optional[Sequence[str]] = None,
    maximum_positive_pairs_per_speaker: int = 32,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Create score-independent ChildMandarin test trials.

    Every positive is cross-utterance.  Its paired negative reuses the same
    enrollment utterance and is selected by stable SHA-256 rank.  For the
    guarded scenario the negative speaker must have the same known gender.
    """

    if maximum_positive_pairs_per_speaker < 1:
        raise OrchestrationError("maximum_positive_pairs_per_speaker 必须为正数")
    base = (
        "coverage_child_same_gender" if same_gender else "coverage_child_clean"
    )
    allowed = None if allowed_utterance_ids is None else set(allowed_utterance_ids)
    candidates = []
    for raw in records:
        row = canonicalize_record(raw, namespace_speakers=True)
        utterance = utterance_id_for_record(row)
        if row["dataset"] != "childmandarin" or row["split"] != "test":
            continue
        if allowed is not None and utterance not in allowed:
            continue
        if same_gender and _known_gender(row["gender"]) is None:
            continue
        candidates.append(row)
    by_speaker = defaultdict(list)
    for row in candidates:
        by_speaker[row["speaker_id"]].append(row)
    positives = []
    for speaker in sorted(by_speaker):
        ranked = _rank_records(by_speaker[speaker], base, "positive", speaker)
        if len(ranked) < 2:
            continue
        count = min(maximum_positive_pairs_per_speaker, len(ranked) - 1)
        for index in range(count):
            positives.append((ranked[index], ranked[index + 1]))
    trials = []
    for enrollment, positive_test in positives:
        trials.append(
            _record_trial(base, enrollment, positive_test, 1, "deterministic_manifest")
        )
        gender = _known_gender(enrollment["gender"])
        eligible = [
            row
            for row in candidates
            if row["speaker_id"] != enrollment["speaker_id"]
            and (not same_gender or _known_gender(row["gender"]) == gender)
        ]
        if not eligible:
            raise OrchestrationError(
                "{} 无可用异说话人负例：{}".format(base, enrollment["speaker_id"])
            )
        negative_test = _rank_records(
            eligible,
            base,
            "negative",
            utterance_id_for_record(enrollment),
            utterance_id_for_record(positive_test),
        )[0]
        trials.append(
            _record_trial(base, enrollment, negative_test, 0, "deterministic_manifest")
        )
    audit = _validate_base_trials(trials, base)
    audit.update(
        {
            "same_gender_negative_required": bool(same_gender),
            "candidate_utterance_count": len(candidates),
            "candidate_speaker_count": len(by_speaker),
            "maximum_positive_pairs_per_speaker": maximum_positive_pairs_per_speaker,
        }
    )
    if same_gender:
        for trial in trials:
            if trial["label"] != 0:
                continue
            enroll_gender = _known_gender(
                next(row["gender"] for row in candidates if utterance_id_for_record(row) == trial["enrollment_utterance_id"])
            )
            test_gender = _known_gender(
                next(row["gender"] for row in candidates if utterance_id_for_record(row) == trial["test_utterance_id"])
            )
            if enroll_gender is None or enroll_gender != test_gender:
                raise OrchestrationError("Child 同性负例构造失败")
    return sorted(trials, key=lambda row: row["trial_id"]), audit


def _condition_token(utterance: str, kind: str) -> str:
    pattern = _DEVICE_PATTERN if kind == "device" else _DISTANCE_PATTERN
    match = pattern.search(utterance)
    if match is None:
        raise OrchestrationError(
            "3D-Speaker utterance 缺少 {} token：{}".format(kind, utterance)
        )
    return match.group(1).lower()


def _parse_official_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in ("target", "1", "true"):
        return 1
    if normalized in ("nontarget", "non-target", "0", "false"):
        return 0
    raise OrchestrationError("未知 official trial label：{}".format(value))


def build_3dspeaker_official_trials(
    records: Sequence[Mapping[str, str]],
    trial_file: Path,
    condition: str,
    allowed_utterance_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Parse official cross-device/distance trials without relabeling."""

    if condition not in ("device", "distance"):
        raise OrchestrationError("condition 必须为 device 或 distance")
    base = (
        "coverage_cross_device"
        if condition == "device"
        else "coverage_cross_distance"
    )
    trial_file = Path(trial_file).resolve()
    if not trial_file.is_file():
        raise FileNotFoundError("official trial file 不存在：{}".format(trial_file))
    index = _manifest_index(records, "3dspeaker")
    allowed = None if allowed_utterance_ids is None else set(allowed_utterance_ids)
    source_digest = sha256_file(trial_file)
    trials = []
    excluded_missing_cache = 0
    with trial_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 3:
                raise OrchestrationError(
                    "official trial {}:{} 必须恰有三列".format(trial_file, line_number)
                )
            enroll_token, test_token, label_text = parts
            enroll_id = "3dspeaker:{}".format(enroll_token)
            test_id = "3dspeaker:{}".format(test_token)
            if enroll_id not in index or test_id not in index:
                raise OrchestrationError(
                    "official trial 端点不在 manifest：{}:{}".format(trial_file, line_number)
                )
            if _condition_token(enroll_token, condition) == _condition_token(test_token, condition):
                raise OrchestrationError(
                    "official cross-{} trial 条件未变化：{}:{}".format(
                        condition, trial_file, line_number
                    )
                )
            if allowed is not None and (enroll_id not in allowed or test_id not in allowed):
                excluded_missing_cache += 1
                continue
            trials.append(
                _record_trial(
                    base,
                    index[enroll_id],
                    index[test_id],
                    _parse_official_label(label_text),
                    "3dspeaker_official_{}".format(condition),
                    str(trial_file),
                    source_digest,
                    line_number,
                )
            )
    audit = _validate_base_trials(trials, base)
    audit.update(
        {
            "condition": condition,
            "official_trial_file": str(trial_file),
            "official_trial_sha256": source_digest,
            "excluded_missing_cache_endpoint_count": excluded_missing_cache,
            "official_labels_preserved": True,
        }
    )
    return trials, audit


def build_aishell4_meeting_trials(
    records: Sequence[Mapping[str, str]],
    allowed_utterance_ids: Optional[Sequence[str]] = None,
    maximum_positive_pairs_per_speaker_session: int = 32,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Create same-session meeting positives and deterministic negatives."""

    allowed = None if allowed_utterance_ids is None else set(allowed_utterance_ids)
    candidates = []
    for raw in records:
        row = canonicalize_record(raw, namespace_speakers=True)
        utterance = utterance_id_for_record(row)
        if row["dataset"] != "aishell4" or row["split"] != "test":
            continue
        if allowed is not None and utterance not in allowed:
            continue
        candidates.append(row)
    by_session = defaultdict(list)
    by_speaker_session = defaultdict(list)
    for row in candidates:
        by_session[row["session"]].append(row)
        by_speaker_session[(row["session"], row["speaker_id"])].append(row)
    trials = []
    excluded_single_speaker_sessions = []
    for (session, speaker), speaker_rows in sorted(by_speaker_session.items()):
        ranked = _rank_records(speaker_rows, "meeting", session, speaker)
        if len(ranked) < 2:
            continue
        different_speaker = [
            row for row in by_session[session] if row["speaker_id"] != speaker
        ]
        if not different_speaker:
            excluded_single_speaker_sessions.append(session)
            continue
        count = min(maximum_positive_pairs_per_speaker_session, len(ranked) - 1)
        for index in range(count):
            enrollment, positive_test = ranked[index], ranked[index + 1]
            trials.append(
                _record_trial(
                    "coverage_meeting_same_session",
                    enrollment,
                    positive_test,
                    1,
                    "aishell4_official_segments",
                )
            )
            negative = _rank_records(
                different_speaker,
                "meeting",
                "negative",
                session,
                utterance_id_for_record(enrollment),
                utterance_id_for_record(positive_test),
            )[0]
            trials.append(
                _record_trial(
                    "coverage_meeting_same_session",
                    enrollment,
                    negative,
                    0,
                    "aishell4_official_segments",
                )
            )
    audit = _validate_base_trials(trials, "coverage_meeting_same_session")
    audit.update(
        {
            "candidate_utterance_count": len(candidates),
            "candidate_session_count": len(by_session),
            "same_session_required": True,
            "excluded_single_speaker_sessions": sorted(set(excluded_single_speaker_sessions)),
            "maximum_positive_pairs_per_speaker_session": maximum_positive_pairs_per_speaker_session,
        }
    )
    for trial in trials:
        enrollment = next(
            row for row in candidates if utterance_id_for_record(row) == trial["enrollment_utterance_id"]
        )
        test = next(
            row for row in candidates if utterance_id_for_record(row) == trial["test_utterance_id"]
        )
        if enrollment["session"] != test["session"]:
            raise OrchestrationError("AISHELL-4 meeting trial 跨 session")
    return sorted(trials, key=lambda row: row["trial_id"]), audit


def _validate_base_trials(
    trials: Sequence[Mapping[str, object]], base_scenario: str
) -> Dict[str, object]:
    if not trials:
        raise OrchestrationError("{} 没有可评分 trial".format(base_scenario))
    labels = Counter(int(row["label"]) for row in trials)
    if set(labels) != {0, 1}:
        raise OrchestrationError("{} 必须同时有正负 trial".format(base_scenario))
    ids = [str(row["trial_id"]) for row in trials]
    if len(ids) != len(set(ids)):
        raise OrchestrationError("{} trial_id 重复".format(base_scenario))
    return {
        "base_scenario": base_scenario,
        "trial_count": len(trials),
        "positive_count": labels[1],
        "negative_count": labels[0],
        "speaker_count": len({str(row["speaker_id"]) for row in trials}),
    }


def _canonical_age_group(value: object, field: str) -> str:
    """Accept only explicit official groups; never infer age from a corpus."""

    result = str(value).strip().lower()
    if result not in CANONICAL_AGE_GROUPS:
        raise OrchestrationError(
            "{} 必须显式为 child/young/elder，不能从数据集名称猜测，实际 {!r}".format(
                field, value
            )
        )
    return result


def _is_unknown_demographic(value: object) -> bool:
    return value is None or str(value).strip().lower() in UNKNOWN_DEMOGRAPHIC_TOKENS


def _canonical_optional_age_group(value: object, field: str) -> str:
    if _is_unknown_demographic(value):
        return "unknown"
    return _canonical_age_group(value, field)


def _canonical_optional_gender(value: object, field: str) -> str:
    if _is_unknown_demographic(value):
        return "unknown"
    result = _known_gender(value)
    if result is None:
        raise OrchestrationError(
            "{} 必须显式为 male/female 或 unknown/空，实际 {!r}".format(
                field, value
            )
        )
    return result


def _canonical_context(value: object, field: str) -> str:
    result = str(value).strip().lower()
    allowed = ("mobile", "near", "far", "meeting")
    if result not in allowed:
        raise OrchestrationError(
            "{} 必须显式为 {}，实际 {!r}".format(field, allowed, value)
        )
    return result


def _required_official_pair_fields(base_scenario: str) -> Tuple[str, ...]:
    if base_scenario not in OFFICIAL_PAIR_SCHEMA:
        raise OrchestrationError(
            "未知 official base scenario：{}".format(base_scenario)
        )
    specification = OFFICIAL_PAIR_SCHEMA[base_scenario]
    return OFFICIAL_PAIR_COMMON_FIELDS + tuple(
        specification["required_demographics"]
    )


def _canonical_official_pair_annotations(
    base_scenario: str, pair: Mapping[str, object]
) -> Dict[str, str]:
    """Normalize only demographics that a scenario can legitimately prove.

    Age-mix scenarios require age and allow gender to remain unknown;
    gender-mix scenarios do the reverse.  Near-to-far requires both.  An
    optional value, when supplied, is still canonicalized strictly rather
    than inferred from its corpus.
    """

    required = _required_official_pair_fields(base_scenario)
    missing = [field for field in required if field not in pair]
    if missing:
        raise OrchestrationError(
            "official pair 缺场景必需字段：{}".format(", ".join(missing))
        )
    required_demographics = set(
        OFFICIAL_PAIR_SCHEMA[base_scenario]["required_demographics"]
    )
    age_values = {}
    for field in ("enrollment_age_group", "test_age_group"):
        value = pair.get(field)
        if field in required_demographics:
            age_values[field] = _canonical_age_group(value, field)
        else:
            age_values[field] = _canonical_optional_age_group(value, field)
    gender_values = {}
    for field in ("enrollment_gender", "test_gender"):
        value = pair.get(field)
        if field in required_demographics:
            normalized = _canonical_optional_gender(value, field)
            if normalized == "unknown":
                raise OrchestrationError(
                    "{} 是该场景必需人口统计字段，不能为 unknown/空".format(field)
                )
            gender_values[field] = normalized
        else:
            gender_values[field] = _canonical_optional_gender(value, field)
    return {
        "enrollment_age_group": age_values["enrollment_age_group"],
        "test_age_group": age_values["test_age_group"],
        "enrollment_gender": gender_values["enrollment_gender"],
        "test_gender": gender_values["test_gender"],
        "enrollment_context": _canonical_context(
            pair["enrollment_context"], "enrollment_context"
        ),
        "test_context": _canonical_context(
            pair["test_context"], "test_context"
        ),
    }


def _pair_group(base_scenario: str, annotations: Mapping[str, str]) -> str:
    if base_scenario in ("mobile_age_mix", "meeting_age_mix"):
        return annotations["test_age_group"]
    if base_scenario in ("mobile_gender_mix", "meeting_gender_mix"):
        return "{}_{}".format(
            annotations["enrollment_gender"], annotations["test_gender"]
        )
    if base_scenario == "near_to_far":
        return "{}_{}".format(
            annotations["test_age_group"], annotations["test_gender"]
        )
    raise OrchestrationError("未知 official base scenario")


def _validate_official_pair_semantics(
    base_scenario: str,
    pair: Mapping[str, object],
    enrollment: Mapping[str, str],
    test: Mapping[str, str],
) -> Dict[str, str]:
    missing = [
        field for field in _required_official_pair_fields(base_scenario)
        if field not in pair
    ]
    if missing:
        raise OrchestrationError(
            "official pair 缺场景必需字段：{}".format(", ".join(missing))
        )
    if str(pair["base_scenario"]).strip() != base_scenario:
        raise OrchestrationError("official pair base_scenario 不一致")
    evidence_scope = str(pair["evidence_scope"]).strip().lower()
    if evidence_scope not in ("proxy", "real_domain_matched"):
        raise OrchestrationError("evidence_scope 必须为 proxy 或 real_domain_matched")
    annotation_source = str(pair["annotation_source"]).strip()
    if not annotation_source:
        raise OrchestrationError("缺少人口统计/场景 annotation_source，拒绝猜标签")
    annotation_sha256 = str(pair["annotation_sha256"]).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", annotation_sha256):
        raise OrchestrationError("annotation_sha256 必须为64位小写十六进制")
    annotations = {
        **_canonical_official_pair_annotations(base_scenario, pair),
        "evidence_scope": evidence_scope,
        "annotation_source": annotation_source,
        "annotation_sha256": annotation_sha256,
    }

    # When an optional or required gender was actually supplied and the source
    # manifest also knows it, the two independent annotations must agree.
    known_enrollment_gender = _known_gender(enrollment["gender"])
    known_test_gender = _known_gender(test["gender"])
    if (
        annotations["enrollment_gender"] != "unknown"
        and known_enrollment_gender is not None
        and known_enrollment_gender != annotations["enrollment_gender"]
    ):
        raise OrchestrationError("enrollment gender 与源 manifest 冲突")
    if (
        annotations["test_gender"] != "unknown"
        and known_test_gender is not None
        and known_test_gender != annotations["test_gender"]
    ):
        raise OrchestrationError("test gender 与源 manifest 冲突")

    if enrollment["dataset"] != test["dataset"] and evidence_scope != "proxy":
        raise OrchestrationError("跨语料 trial 只能标记 evidence_scope=proxy")
    if base_scenario in ("mobile_age_mix", "meeting_age_mix"):
        if annotations["enrollment_age_group"] != "child":
            raise OrchestrationError("age-mix 场景 enrollment 必须有显式 child 标注")
        if int(pair["label"]) == 1 and annotations["test_age_group"] != "child":
            raise OrchestrationError("young/elder test 不可伪造为同 speaker target")
    if base_scenario == "mobile_age_mix":
        if annotations["enrollment_context"] != "mobile" or annotations["test_context"] != "mobile":
            raise OrchestrationError("mobile_age_mix 两端必须有 mobile 标注")
    elif base_scenario == "mobile_gender_mix":
        if annotations["enrollment_context"] != "mobile" or annotations["test_context"] != "mobile":
            raise OrchestrationError("mobile_gender_mix 两端必须有 mobile 标注")
        if annotations["enrollment_gender"] != annotations["test_gender"]:
            raise OrchestrationError("mobile_gender_mix 只允许 M-M/F-F")
    elif base_scenario == "near_to_far":
        if annotations["enrollment_context"] != "near" or annotations["test_context"] != "far":
            raise OrchestrationError("near_to_far 必须近场注册、远场测试")
    elif base_scenario == "meeting_age_mix":
        if annotations["enrollment_context"] != "meeting" or annotations["test_context"] != "meeting":
            raise OrchestrationError("meeting_age_mix 两端必须有 meeting 标注")
        if annotations["test_age_group"] == "elder":
            raise OrchestrationError("meeting_age_mix 官方代理只允许 child/young")
    elif base_scenario == "meeting_gender_mix":
        if annotations["enrollment_context"] != "meeting" or annotations["test_context"] != "meeting":
            raise OrchestrationError("meeting_gender_mix 两端必须有 meeting 标注")
        if (
            annotations["enrollment_gender"], annotations["test_gender"]
        ) == ("female", "male"):
            raise OrchestrationError("meeting_gender_mix 未预注册 F-M cell")
    else:
        raise OrchestrationError("未知 official base scenario：{}".format(base_scenario))
    return annotations  # type: ignore


def build_official_proxy_trials(
    pair_rows: Sequence[Mapping[str, object]],
    manifests: Mapping[str, Sequence[Mapping[str, str]]],
    base_scenario: str,
    allowed_utterance_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Build one official-shaped proxy from explicit, sourced annotations.

    Required demographics are scenario-specific: age+context for age-mix,
    gender+context for gender-mix, and age+gender+near/far for near-to-far.
    Optional demographics remain ``unknown`` when unavailable.  In particular,
    this function never turns ChildMandarin into young/elder labels, nor does
    it infer 3D-Speaker age/gender from filename tokens.
    """

    if base_scenario not in BASE_SCENARIO_BY_NAME:
        raise OrchestrationError("未知 official base scenario：{}".format(base_scenario))
    global_index = {}
    for dataset, records in manifests.items():
        for utterance, record in _manifest_index(records, dataset).items():
            if utterance in global_index:
                raise OrchestrationError("跨 manifest utterance_id 重复：{}".format(utterance))
            global_index[utterance] = record
    allowed = None if allowed_utterance_ids is None else set(allowed_utterance_ids)
    candidates = []
    for row_number, pair in enumerate(pair_rows, 1):
        if str(pair.get("base_scenario", "")).strip() != base_scenario:
            continue
        enroll_id = str(pair.get("enrollment_utterance_id", "")).strip()
        test_id = str(pair.get("test_utterance_id", "")).strip()
        if enroll_id not in global_index or test_id not in global_index:
            raise OrchestrationError(
                "official pair row {} 端点不在 source manifests".format(row_number)
            )
        if allowed is not None and (enroll_id not in allowed or test_id not in allowed):
            continue
        enrollment = global_index[enroll_id]
        test = global_index[test_id]
        annotations = _validate_official_pair_semantics(
            base_scenario, pair, enrollment, test
        )
        trial = _record_trial(
            base_scenario,
            enrollment,
            test,
            int(pair["label"]),
            "explicit_sourced_official_shape_proxy",
            source_file=str(pair.get("pair_manifest_path", annotations["annotation_source"])),
            source_sha256=str(pair.get("pair_manifest_sha256", "")),
            source_line=int(pair.get("pair_manifest_line", 0)),
            evidence_scope=annotations["evidence_scope"],
        )
        trial.update(annotations)
        trial["pair_manifest_path"] = str(pair.get("pair_manifest_path", ""))
        trial["pair_manifest_sha256"] = str(pair.get("pair_manifest_sha256", ""))
        trial["pair_manifest_line"] = int(pair.get("pair_manifest_line", 0))
        trial["ratio_group"] = _pair_group(base_scenario, annotations)
        candidates.append(trial)
    candidate_ids = [str(row["trial_id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise OrchestrationError("official pair manifest 包含重复 trial")
    spec = BASE_SCENARIO_BY_NAME[base_scenario]
    groups = tuple(spec["ratio_groups"])
    ratios = tuple(int(value) for value in spec["ratio"])
    if len(groups) != len(ratios) or not groups or any(value <= 0 for value in ratios):
        raise OrchestrationError("official ratio 注册错误：{}".format(base_scenario))
    by_group = defaultdict(list)
    for trial in candidates:
        by_group[str(trial["ratio_group"])].append(trial)
    unknown_groups = sorted(set(by_group) - set(groups))
    if unknown_groups:
        raise OrchestrationError(
            "{} 出现未预注册 ratio groups：{}".format(base_scenario, unknown_groups)
        )
    missing_groups = [group for group in groups if not by_group[group]]
    if missing_groups:
        raise OrchestrationError(
            "{} 缺少 ratio groups：{}；不得用未知标签补齐".format(
                base_scenario, missing_groups
            )
        )
    scale = min(len(by_group[group]) // ratio for group, ratio in zip(groups, ratios))
    if scale < 1:
        raise OrchestrationError("{} 不足以满足预注册比例".format(base_scenario))
    selected = []
    for group, ratio in zip(groups, ratios):
        ranked = sorted(
            by_group[group], key=lambda row: _stable_sha256(base_scenario, group, row["trial_id"])
        )
        selected.extend(ranked[: scale * ratio])
    # Ratio balancing is label-blind, but EER needs both classes.  If the
    # deterministic prefix accidentally drops one class, swap within the same
    # ratio group so the pre-registered demographic count remains unchanged.
    selected_labels = {int(row["label"]) for row in selected}
    for missing_label in sorted({0, 1} - selected_labels):
        replacement_done = False
        selected_ids = {str(row["trial_id"]) for row in selected}
        for group in groups:
            candidate = next(
                (
                    row
                    for row in sorted(
                        by_group[group],
                        key=lambda item: _stable_sha256(
                            base_scenario, group, item["trial_id"]
                        ),
                    )
                    if int(row["label"]) == missing_label
                    and str(row["trial_id"]) not in selected_ids
                ),
                None,
            )
            if candidate is None:
                continue
            replace_index = next(
                (
                    index
                    for index, row in enumerate(selected)
                    if str(row["ratio_group"]) == group
                    and int(row["label"]) != missing_label
                ),
                None,
            )
            if replace_index is not None:
                selected[replace_index] = candidate
                replacement_done = True
                break
        if not replacement_done:
            raise OrchestrationError(
                "{} 比例平衡后无法同时保留正负类".format(base_scenario)
            )
    selected = sorted(selected, key=lambda row: row["trial_id"])
    audit = _validate_base_trials(selected, base_scenario)
    selected_counts = Counter(str(row["ratio_group"]) for row in selected)
    cross_corpus_negative_count = sum(
        bool(row["cross_corpus"]) and int(row["label"]) == 0 for row in selected
    )
    within_corpus_negative_count = sum(
        not bool(row["cross_corpus"]) and int(row["label"]) == 0 for row in selected
    )
    if cross_corpus_negative_count and not within_corpus_negative_count:
        raise OrchestrationError(
            "跨语料 impostor proxy 必须同时包含同语料 impostor 才能做 corpus-bias sensitivity"
        )
    audit.update(
        {
            "official_shape_proxy": True,
            "hidden_test_equivalence_claimed": False,
            "unknown_age_inference_count": 0,
            "ratio_groups": list(groups),
            "ratio": list(ratios),
            "ratio_scale": scale,
            "selected_group_counts": {
                group: selected_counts[group] for group in groups
            },
            "cross_corpus_trial_count": sum(bool(row["cross_corpus"]) for row in selected),
            "cross_corpus_negative_count": cross_corpus_negative_count,
            "within_corpus_negative_count": within_corpus_negative_count,
            "corpus_bias_sensitivity_possible": bool(
                not cross_corpus_negative_count or within_corpus_negative_count
            ),
            "annotation_sources": sorted(
                {str(row["annotation_source"]) for row in selected}
            ),
        }
    )
    return selected, audit


def validate_official_base_trials(
    trials: Sequence[Mapping[str, object]]
) -> Dict[str, object]:
    """Revalidate a serialized official-shaped base-trial file before score."""

    if not trials:
        raise OrchestrationError("official base trials 不能为空")
    ids = [str(row.get("trial_id", "")) for row in trials]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise OrchestrationError("official base trial_id 为空或重复")
    by_base = defaultdict(list)
    for row in trials:
        base = str(row.get("base_scenario", ""))
        if base not in BASE_SCENARIO_BY_NAME:
            raise OrchestrationError("official base trial 含未知场景：{}".format(base))
        spec = BASE_SCENARIO_BY_NAME[base]
        if row.get("dataset") != spec["dataset"] or row.get("held_out_dataset") != spec["dataset"]:
            raise OrchestrationError("official base trial dataset/heldout 绑定错误")
        if row.get("domain") != spec["domain"]:
            raise OrchestrationError("official base trial domain 绑定错误")
        if row.get("evidence_scope") not in ("proxy", "real_domain_matched"):
            raise OrchestrationError("official base trial evidence_scope 错误")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("annotation_sha256", ""))):
            raise OrchestrationError("official base trial 缺少可信 annotation_sha256")
        if not str(row.get("annotation_source", "")).strip():
            raise OrchestrationError("official base trial 缺 annotation_source")
        if row.get("source_kind") != "explicit_sourced_official_shape_proxy":
            raise OrchestrationError("official base trial source_kind 错误")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_sha256", ""))):
            raise OrchestrationError("official base trial 缺 pair-manifest SHA-256")
        if row.get("pair_manifest_sha256") != row.get("source_sha256"):
            raise OrchestrationError("official pair/source SHA-256 绑定错误")
        if row.get("speaker_id") != row.get("enrollment_speaker_id"):
            raise OrchestrationError("official evaluator speaker 不是 enrollment speaker")
        if row.get("enrollment_dataset") != spec["dataset"]:
            raise OrchestrationError("official enrollment dataset 与 heldout 绑定错误")
        enrollment_speaker = str(row.get("enrollment_speaker_id", ""))
        test_speaker = str(row.get("test_speaker_id", ""))
        if not enrollment_speaker.startswith(spec["dataset"] + ":"):
            raise OrchestrationError("official enrollment speaker namespace 错误")
        if (enrollment_speaker == test_speaker) != bool(int(row.get("label", -1))):
            raise OrchestrationError("serialized official label 与 speaker identity 冲突")
        annotations = _canonical_official_pair_annotations(base, row)
        if str(row.get("ratio_group", "")) != _pair_group(base, annotations):
            raise OrchestrationError("serialized official trial ratio_group 被篡改")
        if bool(row.get("cross_corpus", False)) and row.get("evidence_scope") != "proxy":
            raise OrchestrationError("跨语料 serialized trial 必须为 proxy")
        if int(row.get("label", -1)) not in (0, 1):
            raise OrchestrationError("serialized official trial label 错误")
        if base in ("mobile_age_mix", "meeting_age_mix"):
            if annotations["enrollment_age_group"] != "child":
                raise OrchestrationError("serialized age-mix enrollment 不是 child")
            if int(row["label"]) == 1 and annotations["test_age_group"] != "child":
                raise OrchestrationError("serialized young/elder target 不合法")
        if base in ("mobile_age_mix", "mobile_gender_mix"):
            if annotations["enrollment_context"] != "mobile" or annotations["test_context"] != "mobile":
                raise OrchestrationError("serialized mobile context 被篡改")
        elif base == "near_to_far":
            if annotations["enrollment_context"] != "near" or annotations["test_context"] != "far":
                raise OrchestrationError("serialized near/far context 被篡改")
        else:
            if annotations["enrollment_context"] != "meeting" or annotations["test_context"] != "meeting":
                raise OrchestrationError("serialized meeting context 被篡改")
        if base == "mobile_gender_mix" and annotations["enrollment_gender"] != annotations["test_gender"]:
            raise OrchestrationError("serialized mobile gender cell 被篡改")
        if base == "meeting_age_mix" and annotations["test_age_group"] == "elder":
            raise OrchestrationError("serialized meeting age cell 被篡改")
        if base == "meeting_gender_mix" and (
            annotations["enrollment_gender"], annotations["test_gender"]
        ) == ("female", "male"):
            raise OrchestrationError("serialized meeting F-M cell 未预注册")
        by_base[base].append(row)
    if set(by_base) != set(BASE_SCENARIO_BY_NAME):
        raise OrchestrationError("serialized official trials 未覆盖五个 base")
    audits = []
    for base, rows in sorted(by_base.items()):
        audit = _validate_base_trials(rows, base)
        spec = BASE_SCENARIO_BY_NAME[base]
        groups = tuple(spec["ratio_groups"])
        ratios = tuple(int(value) for value in spec["ratio"])
        counts = Counter(str(row["ratio_group"]) for row in rows)
        if set(counts) != set(groups):
            raise OrchestrationError("{} serialized ratio group 覆盖错误".format(base))
        scales = []
        for group, ratio in zip(groups, ratios):
            if counts[group] % ratio != 0:
                raise OrchestrationError("{} serialized ratio 非整数倍".format(base))
            scales.append(counts[group] // ratio)
        if not scales or len(set(scales)) != 1 or scales[0] < 1:
            raise OrchestrationError("{} serialized ratio 不符合预注册比例".format(base))
        cross_negatives = sum(
            bool(row.get("cross_corpus")) and int(row["label"]) == 0 for row in rows
        )
        within_negatives = sum(
            not bool(row.get("cross_corpus")) and int(row["label"]) == 0 for row in rows
        )
        if cross_negatives and not within_negatives:
            raise OrchestrationError("serialized proxy 缺少 corpus-bias 同语料负例对照")
        audit.update(
            {
                "ratio_groups": list(groups),
                "ratio": list(ratios),
                "ratio_scale": scales[0],
                "cross_corpus_negative_count": cross_negatives,
                "within_corpus_negative_count": within_negatives,
            }
        )
        audits.append(audit)
    return {
        "base_trial_count": len(trials),
        "scenario_audits": audits,
        "official_shape_proxy_validated": True,
    }


def build_all_official_proxy_trials(
    pair_rows: Sequence[Mapping[str, object]],
    manifests: Mapping[str, Sequence[Mapping[str, str]]],
    caches: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    allowed = sorted(
        {
            str(value)
            for dataset in sorted(caches)
            for value in caches[dataset]["utterance_id"]
        }
    )
    trials = []
    audits = []
    for spec in BASE_SCENARIOS:
        selected, audit = build_official_proxy_trials(
            pair_rows, manifests, spec["name"], allowed
        )
        trials.extend(selected)
        audits.append(audit)
    if {str(row["base_scenario"]) for row in trials} != set(BASE_SCENARIO_BY_NAME):
        raise OrchestrationError("official proxy 五个 base scenario 覆盖不完整")
    return sorted(trials, key=lambda row: (row["base_scenario"], row["trial_id"])), {
        "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "artifact_kind": "distillation_official_shape_proxy_trial_audit",
        "evidence_scope": "proxy",
        "hidden_test_equivalence_claimed": False,
        "protocol": protocol_payload(),
        "scenario_audits": audits,
        "base_trial_count": len(trials),
        "all_endpoints_cache_covered": True,
        "known_data_gaps": [
            "ChildMandarin only proves child age; it has no young/elder cohort.",
            "3D-Speaker manifest has no trustworthy age/gender fields.",
            "AISHELL-4 has no child cohort and only a small number of speakers.",
        ],
    }


def build_all_coverage_trials(
    manifests: Mapping[str, Sequence[Mapping[str, str]]],
    caches: Mapping[str, Mapping[str, np.ndarray]],
    source_specs: Mapping[str, Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    allowed = {
        dataset: sorted(set(str(value) for value in caches[dataset]["utterance_id"]))
        for dataset in REQUIRED_DATASETS
    }
    trials = []
    audits = []
    child_clean, audit = build_child_trials(
        manifests["childmandarin"], False, allowed["childmandarin"]
    )
    trials.extend(child_clean)
    audits.append(audit)
    child_gender, audit = build_child_trials(
        manifests["childmandarin"], True, allowed["childmandarin"]
    )
    trials.extend(child_gender)
    audits.append(audit)
    cross_device, audit = build_3dspeaker_official_trials(
        manifests["3dspeaker"],
        source_specs["3dspeaker"]["trial_sources"]["cross_device"],
        "device",
        allowed["3dspeaker"],
    )
    trials.extend(cross_device)
    audits.append(audit)
    far, audit = build_3dspeaker_official_trials(
        manifests["3dspeaker"],
        source_specs["3dspeaker"]["trial_sources"]["far_distance"],
        "distance",
        allowed["3dspeaker"],
    )
    trials.extend(far)
    audits.append(audit)
    meeting, audit = build_aishell4_meeting_trials(
        manifests["aishell4"], allowed["aishell4"]
    )
    trials.extend(meeting)
    audits.append(audit)
    if {str(row["base_scenario"]) for row in trials} != set(COVERAGE_SCENARIO_BY_NAME):
        raise OrchestrationError("coverage scenario 覆盖不完整")
    trials = sorted(trials, key=lambda row: (row["base_scenario"], row["trial_id"]))
    return trials, {
        "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "artifact_kind": "distillation_secondary_coverage_trial_audit",
        "evidence_scope": "coverage_only",
        "formal_15_cell_protocol": False,
        "scenario_audits": audits,
        "base_trial_count": len(trials),
        "all_endpoints_cache_covered": True,
    }


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise OrchestrationError("{} 必须是一维 finite embedding".format(name))
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise OrchestrationError("{} 是零范数 embedding".format(name))
    return vector / norm


def build_cache_view_index(
    cache: Mapping[str, np.ndarray]
) -> Dict[str, Dict[str, object]]:
    """Index every crop view; assert repeated full embeddings are exact acoustic."""

    validated = validate_embedding_cache(cache, allow_v1_migration=False)
    groups = defaultdict(list)
    for index, utterance in enumerate(validated["utterance_id"]):
        groups[str(utterance)].append(index)
    result = {}
    for utterance, indices in groups.items():
        first = indices[0]
        for index in indices[1:]:
            if not np.array_equal(validated["full"][first], validated["full"][index]):
                raise OrchestrationError(
                    "同一 utterance 的 full acoustic embedding 不逐元素一致：{}".format(utterance)
                )
        result[utterance] = {
            "speaker_id": str(validated["speaker_id"][first]),
            "dataset": str(validated["dataset"][first]),
            "full": validated["full"][first].copy(),
            "views": [
                {
                    "embedding_id": str(validated["embedding_id"][index]),
                    "short05": validated["short05"][index].copy(),
                    "short10": validated["short10"][index].copy(),
                }
                for index in sorted(indices, key=lambda item: str(validated["embedding_id"][item]))
            ],
        }
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_normalized = _normalize(left, "cosine left")
    right_normalized = _normalize(right, "cosine right")
    value = float(np.dot(left_normalized, right_normalized))
    if not np.isfinite(value):
        raise OrchestrationError("cosine score 非有限")
    return float(np.clip(value, -1.0, 1.0))


def corpus_bias_sensitivity(
    trace_rows: Sequence[Mapping[str, object]]
) -> Dict[str, object]:
    """Compare within- vs cross-corpus impostor scores without claiming EER."""

    buckets = defaultdict(list)
    for row in trace_rows:
        if int(row["label"]) != 0:
            continue
        relation = "cross_corpus" if bool(row.get("cross_corpus", False)) else "within_corpus"
        key = (str(row["scenario"]), int(row["seed"]), relation)
        buckets[key].append(row)
    results = []
    bases_with_cross = sorted(
        {
            str(row["base_scenario"])
            for row in trace_rows
            if int(row["label"]) == 0 and bool(row.get("cross_corpus", False))
        }
    )
    missing_comparators = []
    scenarios = sorted({str(row["scenario"]) for row in trace_rows})
    for scenario in scenarios:
        base = scenario.rsplit("_", 1)[0]
        if base not in bases_with_cross:
            continue
        for seed in DEFAULT_SEEDS:
            cross = buckets.get((scenario, seed, "cross_corpus"), [])
            within = buckets.get((scenario, seed, "within_corpus"), [])
            if not cross or not within:
                missing_comparators.append((scenario, seed))
                continue
            result = {
                "scenario": scenario,
                "seed": seed,
                "cross_corpus_count": len(cross),
                "within_corpus_count": len(within),
            }
            for system, field in (
                ("baseline", "baseline_score"), ("candidate", "candidate_score")
            ):
                cross_mean = float(np.mean([float(row[field]) for row in cross]))
                within_mean = float(np.mean([float(row[field]) for row in within]))
                result[system + "_cross_mean"] = cross_mean
                result[system + "_within_mean"] = within_mean
                result[system + "_cross_minus_within"] = cross_mean - within_mean
            results.append(result)
    if missing_comparators:
        raise OrchestrationError(
            "corpus-bias sensitivity 缺少同语料 impostor 对照：{}".format(
                missing_comparators[:3]
            )
        )
    return {
        "policy": "compare_impostor_score_means_within_vs_cross_corpus_v1",
        "bases_with_cross_corpus_trials": bases_with_cross,
        "comparison_count": len(results),
        "comparisons": results,
        "release_metric": False,
        "interpretation": "Large gaps indicate corpus classification may inflate proxy separation.",
    }


def load_fold_seed_artifacts(
    artifact_dirs: Mapping[str, Path], fold_cache_paths: Mapping[str, Path]
) -> Tuple[Dict[str, Dict[int, Dict[str, object]]], List[Dict[str, object]]]:
    if tuple(sorted(artifact_dirs)) != REQUIRED_DATASETS:
        raise OrchestrationError("必须提供三个 LODO fold artifact 目录")
    output = {}
    audits = []
    candidate_signatures = set()
    for heldout in REQUIRED_DATASETS:
        cache_path = Path(fold_cache_paths[heldout]).resolve()
        expected_cache_sha = sha256_file(cache_path)
        by_seed = {}
        for seed in DEFAULT_SEEDS:
            artifact_path = Path(artifact_dirs[heldout]) / "seed_{}.npz".format(seed)
            artifact = load_seed_artifact(artifact_path)
            if artifact["seed"] != seed:
                raise OrchestrationError("seed artifact 文件名/内容不一致")
            if artifact["holdout_datasets"] != (heldout,):
                raise OrchestrationError(
                    "seed artifact 不是对应 LODO fold：{}".format(heldout)
                )
            if artifact["cache_sha256"] != expected_cache_sha:
                raise OrchestrationError("seed artifact cache SHA-256 不匹配")
            if heldout in set(artifact["training_datasets"]):
                raise OrchestrationError("seed artifact training_datasets 泄漏 held-out dataset")
            candidate_signatures.add(
                (
                    artifact["candidate_id"],
                    artifact["candidate_family"],
                    artifact["candidate_rank"],
                    artifact["ridge_lambda"],
                    artifact["alpha"],
                    artifact["selection_protocol"],
                    artifact["selection_report_sha256"],
                    artifact["selection_cache_sha256"],
                )
            )
            by_seed[seed] = artifact
            audits.append(
                {
                    "held_out_dataset": heldout,
                    "seed": seed,
                    "artifact": artifact["path"],
                    "artifact_sha256": artifact["sha256"],
                    "fold_cache": str(cache_path),
                    "fold_cache_sha256": expected_cache_sha,
                    "candidate_id": artifact["candidate_id"],
                    "candidate_family": artifact["candidate_family"],
                    "candidate_rank": artifact["candidate_rank"],
                    "selection_report_sha256": artifact[
                        "selection_report_sha256"
                    ],
                    "selection_cache_sha256": artifact[
                        "selection_cache_sha256"
                    ],
                }
            )
        output[heldout] = by_seed
    if len(candidate_signatures) != 1:
        raise OrchestrationError("三个 LODO fold 未冻结同一个全局 dev 候选")
    return output, audits


def score_protocol_trials(
    base_trials: Sequence[Mapping[str, object]],
    combined_cache: Mapping[str, np.ndarray],
    artifacts_by_holdout: Mapping[str, Mapping[int, Mapping[str, object]]],
    cache_sha256: str,
    require_sourced_protocol: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Score all deterministic crop views and emit trace + evaluator rows."""

    protocol_audit = (
        validate_official_base_trials(base_trials)
        if require_sourced_protocol
        else {"official_shape_proxy_validated": False, "synthetic_test_bypass": True}
    )
    index = build_cache_view_index(combined_cache)
    trace_rows = []
    evaluator_rows = []
    full_score_pairs = []
    for raw in sorted(base_trials, key=lambda row: str(row["trial_id"])):
        base = str(raw["base_scenario"])
        if base not in BASE_SCENARIO_BY_NAME:
            raise OrchestrationError("未知 base scenario：{}".format(base))
        heldout = str(raw["held_out_dataset"])
        if heldout != BASE_SCENARIO_BY_NAME[base]["dataset"]:
            raise OrchestrationError("trial heldout 与 protocol 绑定不一致")
        enroll_id = str(raw["enrollment_utterance_id"])
        test_id = str(raw["test_utterance_id"])
        if enroll_id not in index or test_id not in index:
            raise OrchestrationError("trial endpoint 不在 combined cache")
        enrollment = index[enroll_id]
        test = index[test_id]
        if enrollment["speaker_id"] != raw["enrollment_speaker_id"]:
            raise OrchestrationError("enrollment speaker 与 cache 不一致")
        if test["speaker_id"] != raw["test_speaker_id"]:
            raise OrchestrationError("test speaker 与 cache 不一致")
        if heldout not in artifacts_by_holdout:
            raise OrchestrationError("缺少 heldout seed artifacts：{}".format(heldout))
        for seed in DEFAULT_SEEDS:
            if seed not in artifacts_by_holdout[heldout]:
                raise OrchestrationError("缺少 seed artifact：{} {}".format(heldout, seed))
            artifact = artifacts_by_holdout[heldout][seed]
            for suffix, cache_field, is_short in DURATION_VARIANTS:
                scenario = "{}_{}".format(base, suffix)
                test_views = (
                    [{"embedding_id": test_id + "@full", "full": test["full"]}]
                    if not is_short
                    else test["views"]
                )
                for view in test_views:
                    test_vector = view[cache_field]
                    baseline_score = _cosine(enrollment["full"], test_vector)
                    if not is_short:
                        # Do not recompute this value through any adapter path.
                        candidate_score = baseline_score
                        matrix_name = "identity_exact_acoustic_bypass"
                    else:
                        matrix_name = "L05" if cache_field == "short05" else "L10"
                        matrix = np.asarray(artifact[matrix_name])
                        if matrix.shape != (EMBEDDING_DIM, EMBEDDING_DIM):
                            raise OrchestrationError("adapter matrix shape 错误")
                        transformed = np.matmul(
                            matrix.astype(np.float64), test_vector.astype(np.float64)
                        )
                        candidate_score = _cosine(enrollment["full"], transformed)
                    score_id = _stable_sha256(
                        raw["trial_id"], scenario, seed, view["embedding_id"]
                    )
                    trace = dict(raw)
                    trace.update(
                        {
                            "score_id": score_id,
                            "scenario": scenario,
                            "seed": seed,
                            "is_short": is_short,
                            "test_cache_field": cache_field,
                            "test_embedding_id": view["embedding_id"],
                            "enrollment_cache_field": "full",
                            "baseline_score": baseline_score,
                            "candidate_score": candidate_score,
                            "matrix_name": matrix_name,
                            "cache_sha256": cache_sha256,
                            "seed_artifact_path": str(artifact.get("path", "")),
                            "seed_artifact_sha256": str(artifact.get("sha256", "")),
                            "full_candidate_exact_acoustic": bool(
                                not is_short and candidate_score == baseline_score
                            ),
                        }
                    )
                    trace_rows.append(trace)
                    evaluator = {
                        "seed": seed,
                        "scenario": scenario,
                        "speaker_id": raw["speaker_id"],
                        "dataset": raw["dataset"],
                        "domain": raw["domain"],
                        "held_out_dataset": heldout,
                        "label": raw["label"],
                        "baseline_score": baseline_score,
                        "candidate_score": candidate_score,
                        "is_short": is_short,
                        "categories": list(raw["categories"]),
                    }
                    evaluator_rows.append(canonicalize_trial(evaluator))
                    if not is_short:
                        full_score_pairs.append((baseline_score, candidate_score))
    design = validate_trial_design(evaluator_rows, 15, 10)
    if not full_score_pairs or any(left != right for left, right in full_score_pairs):
        raise OrchestrationError("full 分支未严格复用 acoustic score")
    trace_ids = [str(row["score_id"]) for row in trace_rows]
    if len(trace_ids) != len(set(trace_ids)):
        raise OrchestrationError("score_id 重复")
    audit = {
        "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "artifact_kind": "distillation_lodo_score_audit",
        "protocol": protocol_payload(),
        "combined_cache_sha256": cache_sha256,
        "base_trial_count": len(base_trials),
        "score_trace_count": len(trace_rows),
        "evaluator_trial_count": len(evaluator_rows),
        "full_score_count": len(full_score_pairs),
        "full_candidate_scores_exact_acoustic": True,
        "design": design,
        "base_trial_protocol_audit": protocol_audit,
        "corpus_bias_sensitivity": corpus_bias_sensitivity(trace_rows),
        "evidence_scope": "proxy",
        "hidden_test_equivalence_claimed": False,
    }
    return trace_rows, evaluator_rows, audit


def _combined_source_cache(caches: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return _concatenate_cache_parts([caches[name] for name in sorted(caches)])


def read_pair_manifest(path: Path) -> List[Dict[str, object]]:
    """Read explicit official-shaped pair annotations from CSV/JSONL."""

    path = Path(path).resolve()
    rows = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise OrchestrationError("pair CSV 无 header")
            rows.extend(dict(row) for row in reader)
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        rows = read_jsonl_objects(path)
    else:
        raise OrchestrationError("pair manifest 必须是 CSV/JSONL")
    if not rows:
        raise OrchestrationError("pair manifest 不能为空")
    digest = sha256_file(path)
    for line_number, row in enumerate(rows, 2 if path.suffix.lower() == ".csv" else 1):
        base_scenario = str(row.get("base_scenario", "")).strip()
        missing = [
            field for field in _required_official_pair_fields(base_scenario)
            if field not in row
        ]
        if missing:
            raise OrchestrationError(
                "{}:{} 缺场景必需字段 {}".format(path, line_number, missing)
            )
        # Keep a stable serialized schema even when the input legitimately
        # omits an irrelevant demographic axis.
        for field in OFFICIAL_PAIR_DEMOGRAPHIC_FIELDS:
            row.setdefault(field, "")
        row["pair_manifest_path"] = str(path)
        row["pair_manifest_sha256"] = digest
        row["pair_manifest_line"] = line_number
    return rows


def write_base_trials(
    source_specs: Mapping[str, Mapping[str, object]],
    pair_manifest_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    manifests, caches, source_audits = load_sources(source_specs)
    pair_rows = read_pair_manifest(pair_manifest_path)
    trials, audit = build_all_official_proxy_trials(pair_rows, manifests, caches)
    output_dir = Path(output_dir).resolve()
    trial_path = output_dir / "official_proxy_base_trials.jsonl"
    audit["source_audits"] = source_audits
    audit["pair_manifest"] = str(Path(pair_manifest_path).resolve())
    audit["pair_manifest_sha256"] = sha256_file(pair_manifest_path)
    atomic_write_jsonl(trial_path, trials)
    audit["base_trials"] = str(trial_path)
    audit["base_trials_sha256"] = sha256_file(trial_path)
    atomic_write_json(output_dir / "base_trials.audit.json", audit)
    return audit


def write_coverage_trials(
    source_specs: Mapping[str, Mapping[str, object]], output_dir: Path
) -> Dict[str, object]:
    """Write secondary coverage diagnostics; never the formal 15-cell file."""

    manifests, caches, source_audits = load_sources(source_specs)
    trials, audit = build_all_coverage_trials(manifests, caches, source_specs)
    output_dir = Path(output_dir).resolve()
    path = output_dir / "secondary_coverage_trials.jsonl"
    atomic_write_jsonl(path, trials)
    audit.update(
        {
            "source_audits": source_audits,
            "coverage_trials": str(path),
            "coverage_trials_sha256": sha256_file(path),
            "must_not_be_passed_to_evaluate_distillation_gates": True,
        }
    )
    atomic_write_json(output_dir / "secondary_coverage_trials.audit.json", audit)
    return audit


def read_jsonl_objects(path: Path) -> List[Dict[str, object]]:
    result = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise OrchestrationError(
                    "{}:{} JSON 错误：{}".format(path, line_number, error)
                )
            if not isinstance(value, dict):
                raise OrchestrationError("JSONL 行必须是 object")
            result.append(value)
    if not result:
        raise OrchestrationError("JSONL 不能为空：{}".format(path))
    return result


def write_scores(
    source_specs: Mapping[str, Mapping[str, object]],
    base_trials_path: Path,
    artifact_dirs: Mapping[str, Path],
    folds_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    _manifests, caches, source_audits = load_sources(source_specs)
    combined_cache = _combined_source_cache(caches)
    # Materialize one immutable combined cache so score provenance names bytes,
    # not merely an in-memory concatenation order.
    output_dir = Path(output_dir).resolve()
    combined_path = output_dir / "combined_source_cache_v2.npz"
    write_embedding_cache(combined_path, combined_cache)
    combined_sha = sha256_file(combined_path)
    fold_cache_paths = {
        dataset: Path(folds_dir) / dataset / "cache_v2.npz"
        for dataset in REQUIRED_DATASETS
    }
    artifacts, artifact_audits = load_fold_seed_artifacts(
        artifact_dirs, fold_cache_paths
    )
    base_trials = read_jsonl_objects(base_trials_path)
    trace, evaluator, audit = score_protocol_trials(
        base_trials, combined_cache, artifacts, combined_sha
    )
    trace_path = output_dir / "trial_scores.trace.jsonl"
    evaluator_jsonl = output_dir / "evaluator_trials.jsonl"
    evaluator_csv = output_dir / "evaluator_trials.csv"
    atomic_write_jsonl(trace_path, trace)
    atomic_write_jsonl(evaluator_jsonl, evaluator)
    csv_rows = []
    for row in evaluator:
        csv_row = dict(row)
        csv_row["categories"] = json.dumps(row["categories"], ensure_ascii=False)
        csv_rows.append(csv_row)
    atomic_write_csv(evaluator_csv, csv_rows, TRIAL_FIELDS)
    audit.update(
        {
            "source_audits": source_audits,
            "artifact_audits": artifact_audits,
            "base_trials": str(Path(base_trials_path).resolve()),
            "base_trials_sha256": sha256_file(base_trials_path),
            "combined_source_cache": str(combined_path),
            "trace": str(trace_path),
            "trace_sha256": sha256_file(trace_path),
            "evaluator_jsonl": str(evaluator_jsonl),
            "evaluator_jsonl_sha256": sha256_file(evaluator_jsonl),
            "evaluator_csv": str(evaluator_csv),
            "evaluator_csv_sha256": sha256_file(evaluator_csv),
            "release_gate_written": False,
            "formal_release_adapter_written": False,
            "formal_submission_tar_written": False,
        }
    )
    atomic_write_json(output_dir / "scores.audit.json", audit)
    return audit


def _parse_dataset_path(values: Sequence[str], label: str) -> Dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise OrchestrationError("{} 必须使用 dataset=/path 格式".format(label))
        dataset, path = value.split("=", 1)
        dataset = dataset.strip()
        if dataset in result or dataset not in REQUIRED_DATASETS or not path.strip():
            raise OrchestrationError("{} dataset 重复/未知/空路径：{}".format(label, value))
        result[dataset] = Path(path).resolve()
    if tuple(sorted(result)) != REQUIRED_DATASETS:
        raise OrchestrationError("{} 必须覆盖 {}".format(label, REQUIRED_DATASETS))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    protocol = subparsers.add_parser("write-protocol")
    protocol.add_argument("--output", type=Path, required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--source-config", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--candidate-selection-report", type=Path)
    assemble.add_argument("--selection-cache", type=Path)
    auxiliary = subparsers.add_parser(
        "assemble-post-cv-auxiliary",
        help="build follow-up LODO folds with fixed train-only auxiliary domains",
    )
    auxiliary.add_argument("--source-config", type=Path, required=True)
    auxiliary.add_argument("--output-dir", type=Path, required=True)
    auxiliary.add_argument("--candidate-selection-report", type=Path, required=True)
    auxiliary.add_argument("--selection-cache", type=Path, required=True)
    trials = subparsers.add_parser("build-trials")
    trials.add_argument("--source-config", type=Path, required=True)
    trials.add_argument(
        "--pair-manifest",
        type=Path,
        required=True,
        help="explicit sourced age/gender/context pair annotations",
    )
    trials.add_argument("--output-dir", type=Path, required=True)
    coverage = subparsers.add_parser("build-coverage-trials")
    coverage.add_argument("--source-config", type=Path, required=True)
    coverage.add_argument("--output-dir", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--source-config", type=Path, required=True)
    score.add_argument("--base-trials", type=Path, required=True)
    score.add_argument("--folds-dir", type=Path, required=True)
    score.add_argument(
        "--artifact-dir",
        action="append",
        required=True,
        help="repeat dataset=/absolute/path/to/fold_training",
    )
    score.add_argument("--output-dir", type=Path, required=True)
    evaluate = subparsers.add_parser(
        "evaluate-proxy",
        help="run the frozen evaluator with evidence_label=proxy (never release-eligible)",
    )
    evaluate.add_argument("--trials", type=Path, required=True)
    evaluate.add_argument("--folds-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "write-protocol":
        atomic_write_json(arguments.output, protocol_payload())
        return 0
    if arguments.command == "evaluate-proxy":
        evaluator_argv = [
            str(arguments.trials),
            "--output",
            str(arguments.output),
            "--evidence-label",
            "proxy",
            "--bootstrap-repeats",
            str(arguments.bootstrap_repeats),
        ]
        for dataset in REQUIRED_DATASETS:
            evaluator_argv.extend(
                [
                    "--manifest",
                    str(Path(arguments.folds_dir) / dataset / "manifest.csv"),
                ]
            )
        return int(evaluate_gates_main(evaluator_argv))
    specs = load_source_config(
        arguments.source_config,
        allow_training_auxiliary=(
            arguments.command == "assemble-post-cv-auxiliary"
        ),
    )
    if arguments.command == "assemble":
        write_lodo_folds(
            specs,
            arguments.output_dir,
            candidate_selection_report=arguments.candidate_selection_report,
            selection_cache=arguments.selection_cache,
        )
    elif arguments.command == "assemble-post-cv-auxiliary":
        write_post_cv_auxiliary_lodo_folds(
            specs,
            arguments.output_dir,
            candidate_selection_report=arguments.candidate_selection_report,
            selection_cache=arguments.selection_cache,
        )
    elif arguments.command == "build-trials":
        write_base_trials(specs, arguments.pair_manifest, arguments.output_dir)
    elif arguments.command == "build-coverage-trials":
        write_coverage_trials(specs, arguments.output_dir)
    elif arguments.command == "score":
        artifact_dirs = _parse_dataset_path(arguments.artifact_dir, "artifact-dir")
        write_scores(
            specs,
            arguments.base_trials,
            artifact_dirs,
            Path(arguments.folds_dir),
            arguments.output_dir,
        )
    else:
        raise RuntimeError("unreachable command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
