#!/usr/bin/env python3
"""Gate-locked export of the three-seed distillation matrix soup."""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distillation_training.cache_schema import EMBEDDING_DIM
from distillation_training.train_ridge_kd import (
    DEFAULT_SEEDS,
    DOMAIN_WEIGHTING_POLICY,
    FORMAL_CANDIDATE_FAMILY,
    FORMAL_CANDIDATE_SELECTION_PROTOCOL,
    SEED_ARTIFACT_KIND,
    SEED_ARTIFACT_KEYS,
    SEED_ARTIFACT_SCHEMA_VERSION,
    TEACHER_POLICY,
    sha256_file,
    validate_candidate_selection_report_bytes,
)
from domain_kd_inference import (
    ADAPTER_SCHEMA_VERSION,
    compute_adapter_payload_sha256,
)
from tools.evaluate_distillation_gates import (
    DEPLOYMENT_SCORING_POLICY,
    GATE_SCHEMA_VERSION,
    EvaluationError,
    revalidate_gate_evidence_bindings,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _scalar(array: np.ndarray, name: str, kind: str):
    array = np.asarray(array)
    if array.shape != () or array.dtype.kind not in kind:
        raise ValueError("{} 标量类型错误：{} {}".format(name, array.shape, array.dtype))
    return array.item()


def _text_vector(array: np.ndarray, name: str):
    array = np.asarray(array)
    if array.ndim != 1 or array.dtype.kind != "U":
        raise ValueError("{} 必须是一维 Unicode 数组".format(name))
    return tuple(str(value) for value in array)


def _int64_vector(array: np.ndarray, name: str):
    array = np.asarray(array)
    if array.ndim != 1 or array.dtype != np.dtype(np.int64):
        raise ValueError("{} 必须是一维 int64 数组".format(name))
    return tuple(int(value) for value in array)


def _float64_vector(array: np.ndarray, name: str):
    array = np.asarray(array)
    if array.ndim != 1 or array.dtype != np.dtype(np.float64):
        raise ValueError("{} 必须是一维 float64 数组".format(name))
    return tuple(float(value) for value in array)


def _uint8_bytes(array: np.ndarray, name: str) -> bytes:
    array = np.asarray(array)
    if array.ndim != 1 or array.dtype != np.dtype(np.uint8) or array.size == 0:
        raise ValueError("{} 必须是非空一维 uint8 数组".format(name))
    return np.ascontiguousarray(array).tobytes()


def load_seed_artifact(path: Path) -> Dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("seed artifact 不存在：{}".format(path))
    with np.load(str(path), allow_pickle=False) as payload:
        actual = tuple(sorted(payload.files))
        expected = tuple(sorted(SEED_ARTIFACT_KEYS))
        if actual != expected:
            raise ValueError(
                "seed artifact schema 键错误：expected={} actual={}".format(
                    expected, actual
                )
            )
        schema = int(_scalar(payload["schema_version"], "schema_version", "iu"))
        kind = str(_scalar(payload["artifact_kind"], "artifact_kind", "U"))
        seed = int(_scalar(payload["seed"], "seed", "iu"))
        matrix_0p5 = payload["L05"].copy()
        matrix_1p0 = payload["L10"].copy()
        selection_report_bytes = _uint8_bytes(
            payload["selection_report_json_utf8"],
            "selection_report_json_utf8",
        )
        result = {
            "schema_version": schema,
            "artifact_kind": kind,
            "seed": seed,
            "candidate_id": str(
                _scalar(payload["candidate_id"], "candidate_id", "U")
            ),
            "candidate_family": str(
                _scalar(payload["candidate_family"], "candidate_family", "U")
            ),
            "candidate_rank": int(
                _scalar(payload["candidate_rank"], "candidate_rank", "iu")
            ),
            "ridge_lambda": float(
                _scalar(payload["ridge_lambda"], "ridge_lambda", "f")
            ),
            "alpha": float(_scalar(payload["alpha"], "alpha", "f")),
            "train_split": str(
                _scalar(payload["train_split"], "train_split", "U")
            ),
            "cache_sha256": str(
                _scalar(payload["cache_sha256"], "cache_sha256", "U")
            ),
            "holdout_datasets": tuple(
                str(value) for value in payload["holdout_datasets"]
            ),
            "training_datasets": tuple(
                str(value) for value in payload["training_datasets"]
            ),
            "sample_count": int(
                _scalar(payload["sample_count"], "sample_count", "iu")
            ),
            "speaker_count": int(
                _scalar(payload["speaker_count"], "speaker_count", "iu")
            ),
            "teacher_policy": str(
                _scalar(payload["teacher_policy"], "teacher_policy", "U")
            ),
            "teacher_selected_sample_count": int(
                _scalar(
                    payload["teacher_selected_sample_count"],
                    "teacher_selected_sample_count",
                    "iu",
                )
            ),
            "teacher_eligible_sample_count": int(
                _scalar(
                    payload["teacher_eligible_sample_count"],
                    "teacher_eligible_sample_count",
                    "iu",
                )
            ),
            "teacher_excluded_sample_count": int(
                _scalar(
                    payload["teacher_excluded_sample_count"],
                    "teacher_excluded_sample_count",
                    "iu",
                )
            ),
            "teacher_selected_speaker_count": int(
                _scalar(
                    payload["teacher_selected_speaker_count"],
                    "teacher_selected_speaker_count",
                    "iu",
                )
            ),
            "teacher_eligible_speaker_count": int(
                _scalar(
                    payload["teacher_eligible_speaker_count"],
                    "teacher_eligible_speaker_count",
                    "iu",
                )
            ),
            "teacher_excluded_speaker_count": int(
                _scalar(
                    payload["teacher_excluded_speaker_count"],
                    "teacher_excluded_speaker_count",
                    "iu",
                )
            ),
            "teacher_excluded_speaker_keys": _text_vector(
                payload["teacher_excluded_speaker_keys"],
                "teacher_excluded_speaker_keys",
            ),
            "teacher_invalid_utterance_keys": _text_vector(
                payload["teacher_invalid_utterance_keys"],
                "teacher_invalid_utterance_keys",
            ),
            "teacher_zero_norm_prototype_embedding_ids": _text_vector(
                payload["teacher_zero_norm_prototype_embedding_ids"],
                "teacher_zero_norm_prototype_embedding_ids",
            ),
            "teacher_self_overlap_count": int(
                _scalar(
                    payload["teacher_self_overlap_count"],
                    "teacher_self_overlap_count",
                    "iu",
                )
            ),
            "selection_protocol": str(
                _scalar(payload["selection_protocol"], "selection_protocol", "U")
            ),
            "selection_report_sha256": str(
                _scalar(
                    payload["selection_report_sha256"],
                    "selection_report_sha256",
                    "U",
                )
            ),
            "selection_cache_sha256": str(
                _scalar(
                    payload["selection_cache_sha256"],
                    "selection_cache_sha256",
                    "U",
                )
            ),
            "domain_weighting_policy": str(
                _scalar(
                    payload["domain_weighting_policy"],
                    "domain_weighting_policy",
                    "U",
                )
            ),
            "domain_total_weight_target": float(
                _scalar(
                    payload["domain_total_weight_target"],
                    "domain_total_weight_target",
                    "f",
                )
            ),
            "domain_names": _text_vector(
                payload["domain_names"], "domain_names"
            ),
            "domain_sample_counts": _int64_vector(
                payload["domain_sample_counts"], "domain_sample_counts"
            ),
            "domain_total_weights": _float64_vector(
                payload["domain_total_weights"], "domain_total_weights"
            ),
            "L05": matrix_0p5,
            "L10": matrix_1p0,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    if schema != SEED_ARTIFACT_SCHEMA_VERSION or kind != SEED_ARTIFACT_KIND:
        raise ValueError("seed artifact 版本或类型错误")
    try:
        selection = validate_candidate_selection_report_bytes(
            selection_report_bytes,
            expected_selection_cache_sha256=result["selection_cache_sha256"],
        )
    except ValueError as error:
        raise ValueError("seed artifact 内嵌 selection report 无效：{}".format(error))
    for field in (
        "candidate_id",
        "candidate_family",
        "candidate_rank",
        "ridge_lambda",
        "alpha",
        "selection_protocol",
        "selection_cache_sha256",
    ):
        if result[field] != selection[field]:
            raise ValueError("seed artifact {} 与内嵌 selection report 不一致".format(field))
    if result["selection_report_sha256"] != selection["report_sha256"]:
        raise ValueError("seed artifact selection report SHA-256 与内嵌字节不一致")
    for name, matrix in (("L05", matrix_0p5), ("L10", matrix_1p0)):
        if matrix.shape != (EMBEDDING_DIM, EMBEDDING_DIM):
            raise ValueError("{} shape 错误：{}".format(name, matrix.shape))
        if matrix.dtype != np.dtype(np.float32) or not np.isfinite(matrix).all():
            raise ValueError("{} 必须为 finite float32".format(name))
    if (
        not np.isfinite(result["ridge_lambda"])
        or result["ridge_lambda"] <= 0.0
    ):
        raise ValueError("ridge_lambda 必须为有限正数")
    if not np.isfinite(result["alpha"]) or not 0.0 <= result["alpha"] <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")
    if result["train_split"] != "train":
        raise ValueError("正式 seed artifact train_split 必须为 train")
    if not _SHA256_PATTERN.fullmatch(result["cache_sha256"]):
        raise ValueError("cache_sha256 必须为 64 位小写十六进制")
    if not result["training_datasets"] or any(
        not value.strip() for value in result["training_datasets"]
    ):
        raise ValueError("training_datasets 不能为空")
    if result["teacher_policy"] != TEACHER_POLICY:
        raise ValueError("seed artifact teacher_policy 错误")
    if result["teacher_self_overlap_count"] != 0:
        raise ValueError("seed artifact 检测到 teacher 自身泄漏")
    if (
        result["teacher_selected_sample_count"]
        != result["teacher_eligible_sample_count"]
        + result["teacher_excluded_sample_count"]
    ):
        raise ValueError("teacher sample audit 计数不守恒")
    if (
        result["teacher_selected_speaker_count"]
        != result["teacher_eligible_speaker_count"]
        + result["teacher_excluded_speaker_count"]
    ):
        raise ValueError("teacher speaker audit 计数不守恒")
    if (
        result["teacher_excluded_speaker_count"]
        != len(result["teacher_excluded_speaker_keys"])
    ):
        raise ValueError("teacher excluded speaker 审计不一致")
    if result["candidate_family"] != FORMAL_CANDIDATE_FAMILY:
        raise ValueError("seed artifact candidate_family 不是正式低秩候选")
    if result["candidate_rank"] <= 0 or result["candidate_rank"] > EMBEDDING_DIM:
        raise ValueError("seed artifact candidate_rank 越界")
    if not result["candidate_id"].strip():
        raise ValueError("seed artifact candidate_id 不能为空")
    if result["selection_protocol"] != FORMAL_CANDIDATE_SELECTION_PROTOCOL:
        raise ValueError("seed artifact selection_protocol 错误")
    if not _SHA256_PATTERN.fullmatch(result["selection_report_sha256"]):
        raise ValueError("selection_report_sha256 必须为 64 位小写十六进制")
    if not _SHA256_PATTERN.fullmatch(result["selection_cache_sha256"]):
        raise ValueError("selection_cache_sha256 必须为 64 位小写十六进制")
    if result["domain_weighting_policy"] != DOMAIN_WEIGHTING_POLICY:
        raise ValueError("seed artifact domain_weighting_policy 错误")
    domain_count = len(result["domain_names"])
    if (
        domain_count == 0
        or len(result["domain_sample_counts"]) != domain_count
        or len(result["domain_total_weights"]) != domain_count
        or len(set(result["domain_names"])) != domain_count
        or any(count <= 0 for count in result["domain_sample_counts"])
    ):
        raise ValueError("seed artifact domain balance 向量错误")
    if tuple(sorted(result["domain_names"])) != result["domain_names"]:
        raise ValueError("seed artifact domain_names 必须排序")
    if (
        not np.isfinite(result["domain_total_weight_target"])
        or result["domain_total_weight_target"] != 1.0
    ):
        raise ValueError("domain_total_weight_target 必须严格为 1.0")
    if sum(result["domain_sample_counts"]) != result["sample_count"]:
        raise ValueError("domain_sample_counts 与 sample_count 不守恒")
    if not np.allclose(
        np.asarray(result["domain_total_weights"], dtype=np.float64),
        result["domain_total_weight_target"],
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError("每个 dataset 总权重必须严格相等")
    if result["domain_names"] != tuple(sorted(result["training_datasets"])):
        raise ValueError("domain balance 域名与 training_datasets 不一致")
    if result["sample_count"] <= 0 or result["speaker_count"] <= 0:
        raise ValueError("seed artifact 样本/说话人数必须为正")
    if result["speaker_count"] != result["teacher_eligible_speaker_count"]:
        raise ValueError("speaker_count 与 teacher eligible speaker 审计不一致")
    return result


def _nested(payload: Mapping[str, object], path: Sequence[str]):
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("gate 缺少字段 {}".format(".".join(path)))
        value = value[key]
    return value


def validate_release_gate(gate: Mapping[str, object]) -> Dict[str, object]:
    if not isinstance(gate, dict) or gate.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ValueError("gate schema_version 必须为 {}".format(GATE_SCHEMA_VERSION))
    if gate.get("system") != "distillation-domain-KD" or gate.get("status") != "PASS":
        raise ValueError("gate system/status 不是正式 PASS")
    if gate.get("evidence_label") != "real":
        raise ValueError("gate evidence_label 必须为 real")
    if gate.get("metric_policy") != "actual_three_seed_mean_deployment_matrix_scores_once":
        raise ValueError("gate 未评估三 seed 均值后的实际部署矩阵")
    if gate.get("released_adapter") is not None:
        raise ValueError("gate 已绑定 adapter；拒绝覆盖或重复导出")
    try:
        verified_bindings = revalidate_gate_evidence_bindings(
            gate.get("evidence_bindings")
        )
    except EvaluationError as error:
        raise ValueError("gate evidence binding 校验失败：{}".format(error))
    for name in (
        "overall_pass",
        "release_eligible",
        "all_required_gates_passed",
    ):
        if gate.get(name) is not True:
            raise ValueError("gate {} 必须严格为 true".format(name))
    checks = gate.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("gate checks 必须是非空对象")
    invalid_checks = [
        name for name, passed in checks.items() if passed is not True
    ]
    if invalid_checks:
        raise ValueError("gate checks 未全部通过：{}".format(invalid_checks))

    macro_improvement = float(
        _nested(gate, ("metrics", "macro_15_scenarios", "improvement_points"))
    )
    short_improvement = float(
        _nested(gate, ("metrics", "short_10_scenarios", "improvement_points"))
    )
    by_base = _nested(gate, ("metrics", "by_base_scenario"))
    bootstrap_upper = float(
        _nested(
            gate,
            ("bootstrap", "macro_15_scenarios", "delta_ci95_upper"),
        )
    )
    if not np.isfinite(macro_improvement) or macro_improvement < 2.6:
        raise ValueError("正式发布的15场景宏 EER 改善必须 >=2.6（2.0仅为最低诊断线）")
    if not np.isfinite(short_improvement) or short_improvement < 3.0:
        raise ValueError("10个短场景宏 EER 改善必须 >=3.0")
    if not isinstance(by_base, dict) or len(by_base) != 5:
        raise ValueError("必须逐项提供五个 base scenario 指标")
    failed_base = []
    for base, metrics in by_base.items():
        try:
            regression = float(metrics["delta_candidate_minus_baseline"])
        except (KeyError, TypeError, ValueError):
            failed_base.append(base)
            continue
        if not np.isfinite(regression) or regression > 0.2:
            failed_base.append(base)
    if failed_base:
        raise ValueError("base scenario 回退超过 0.2：{}".format(failed_base))
    if not np.isfinite(bootstrap_upper) or bootstrap_upper >= 0.0:
        raise ValueError("speaker bootstrap delta CI95 upper 必须 <0")
    bootstrap = _nested(gate, ("bootstrap", "macro_15_scenarios"))
    if (
        bootstrap.get("strategy")
        != "conservative_enrollment_test_pigeonhole_two_way_v1"
        or bootstrap.get("seed_replicates_resampled_independently") is not False
    ):
        raise ValueError("bootstrap 必须使用双端 speaker cluster/pigeonhole 保守策略")
    lodo = _nested(gate, ("metrics", "by_lodo_held_out_dataset"))
    if not isinstance(lodo, dict) or len(lodo) < 2:
        raise ValueError("LODO 至少需要两个 held-out dataset")
    failed_lodo = []
    for dataset, metrics in lodo.items():
        if not isinstance(metrics, dict):
            failed_lodo.append(dataset)
            continue
        improvement = metrics.get("improvement_points")
        if (
            isinstance(improvement, bool)
            or not isinstance(improvement, (int, float))
            or not np.isfinite(float(improvement))
            or float(improvement) <= 0.0
        ):
            failed_lodo.append(dataset)
    if failed_lodo:
        raise ValueError("LODO 未全部改善：{}".format(failed_lodo))
    return verified_bindings


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


def export_released_adapter(
    seed_paths: Sequence[Path], gate_path: Path, output_path: Path
) -> Dict[str, object]:
    if len(seed_paths) != 3:
        raise ValueError("正式导出必须恰好提供三个 seed artifact")
    gate_path = Path(gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    verified_bindings = validate_release_gate(gate)

    artifacts = [load_seed_artifact(path) for path in seed_paths]
    actual_seeds = tuple(sorted(item["seed"] for item in artifacts))
    if actual_seeds != DEFAULT_SEEDS:
        raise ValueError(
            "seed 必须严格为 {}，实际 {}".format(DEFAULT_SEEDS, actual_seeds)
        )
    if len(set(item["cache_sha256"] for item in artifacts)) != 1:
        raise ValueError("三个 seed artifact 必须来自同一 cache")
    combined_cache_sha256 = verified_bindings["combined_cache"]["sha256"]
    if artifacts[0]["cache_sha256"] != combined_cache_sha256:
        raise ValueError("正式三 seed artifact 未绑定 gate 的 combined cache SHA-256")
    if artifacts[0]["selection_cache_sha256"] != combined_cache_sha256:
        raise ValueError("正式候选选择报告未绑定 gate 的 combined cache SHA-256")
    if len(set(item["training_datasets"] for item in artifacts)) != 1:
        raise ValueError("三个 seed artifact 的 training_datasets 不一致")
    if len(artifacts[0]["training_datasets"]) < 2:
        raise ValueError("正式 adapter 必须由至少两个训练域产生")
    if artifacts[0]["holdout_datasets"]:
        raise ValueError("正式 adapter 必须用全部训练域重训，不能导出 LODO fold")
    for field in (
        "candidate_id",
        "candidate_family",
        "candidate_rank",
        "ridge_lambda",
        "alpha",
        "train_split",
        "holdout_datasets",
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
        "selection_cache_sha256",
        "domain_weighting_policy",
        "domain_total_weight_target",
        "domain_names",
        "domain_total_weights",
    ):
        if len(set(item[field] for item in artifacts)) != 1:
            raise ValueError("三个 seed artifact 的 {} 不一致".format(field))

    matrix_0p5 = np.mean(
        np.stack([item["L05"] for item in artifacts]).astype(np.float64),
        axis=0,
    ).astype(np.float32)
    matrix_1p0 = np.mean(
        np.stack([item["L10"] for item in artifacts]).astype(np.float64),
        axis=0,
    ).astype(np.float32)
    if not np.isfinite(matrix_0p5).all() or not np.isfinite(matrix_1p0).all():
        raise RuntimeError("三 seed 矩阵均值包含 NaN/Inf")
    identity = np.eye(EMBEDDING_DIM, dtype=np.float32)
    if np.array_equal(matrix_0p5, identity) and np.array_equal(matrix_1p0, identity):
        raise ValueError("拒绝把纯 identity adapter 标记为正式发布")

    seed_ids = np.asarray(DEFAULT_SEEDS, dtype=np.int64)
    release_status = "passed"
    payload_sha256 = compute_adapter_payload_sha256(
        ADAPTER_SCHEMA_VERSION,
        release_status,
        seed_ids,
        matrix_0p5,
        matrix_1p0,
    )
    arrays = {
        "schema_version": np.asarray(ADAPTER_SCHEMA_VERSION, dtype=np.int64),
        "release_status": np.asarray(release_status),
        "seed_ids": seed_ids,
        "L05": matrix_0p5,
        "L10": matrix_1p0,
        "payload_sha256": np.asarray(payload_sha256),
    }
    output_path = Path(output_path)
    _atomic_savez(output_path, arrays)
    output_sha256 = sha256_file(output_path)
    released_adapter = {
        "schema_version": 1,
        "binding_kind": "distillation_released_adapter_binding",
        "path": str(output_path.resolve()),
        "sha256": output_sha256,
        "payload_sha256": payload_sha256,
        "seed_ids": list(DEFAULT_SEEDS),
        "combined_cache_sha256": combined_cache_sha256,
        "full_training_seed_artifacts": [
            {
                "seed": item["seed"],
                "path": item["path"],
                "sha256": item["sha256"],
            }
            for item in sorted(artifacts, key=lambda value: value["seed"])
        ],
    }
    # The evaluator cannot know the final full-domain adapter before export.
    # Write the payload/file binding back into the same gate atomically so the
    # package builder can cross-check the actual bytes rather than trusting an
    # unbound export receipt.
    gate["released_adapter"] = released_adapter
    _atomic_write_json(gate_path, gate)
    receipt = {
        "status": "EXPORTED",
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": output_sha256,
        "payload_sha256": payload_sha256,
        "gate": str(gate_path.resolve()),
        "gate_sha256": sha256_file(gate_path),
        "released_adapter": released_adapter,
        "seed_ids": list(DEFAULT_SEEDS),
        "teacher_policy": TEACHER_POLICY,
        "selection_protocol": FORMAL_CANDIDATE_SELECTION_PROTOCOL,
        "selection_report_sha256": artifacts[0]["selection_report_sha256"],
        "selection_cache_sha256": artifacts[0]["selection_cache_sha256"],
        "candidate": {
            "id": artifacts[0]["candidate_id"],
            "family": artifacts[0]["candidate_family"],
            "rank": artifacts[0]["candidate_rank"],
            "ridge_lambda": artifacts[0]["ridge_lambda"],
            "alpha": artifacts[0]["alpha"],
        },
        "domain_weighting_policy": DOMAIN_WEIGHTING_POLICY,
        "domain_balance_by_seed": [
            {
                "seed": item["seed"],
                "domain_names": list(item["domain_names"]),
                "domain_sample_counts": list(item["domain_sample_counts"]),
                "domain_total_weights": list(item["domain_total_weights"]),
            }
            for item in artifacts
        ],
        "teacher_audit": {
            "selected_sample_count": artifacts[0]["teacher_selected_sample_count"],
            "eligible_sample_count": artifacts[0]["teacher_eligible_sample_count"],
            "excluded_sample_count": artifacts[0]["teacher_excluded_sample_count"],
            "selected_speaker_count": artifacts[0]["teacher_selected_speaker_count"],
            "eligible_speaker_count": artifacts[0]["teacher_eligible_speaker_count"],
            "excluded_speaker_count": artifacts[0]["teacher_excluded_speaker_count"],
            "excluded_speaker_keys": list(
                artifacts[0]["teacher_excluded_speaker_keys"]
            ),
        },
        "seed_artifacts": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in artifacts
        ],
        "training_datasets": list(artifacts[0]["training_datasets"]),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-artifact", action="append", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    receipt = export_released_adapter(
        arguments.seed_artifact, arguments.gate, arguments.output
    )
    if arguments.receipt is not None:
        arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.receipt.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
