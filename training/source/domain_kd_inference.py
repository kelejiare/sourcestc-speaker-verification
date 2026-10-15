#!/usr/bin/env python3
"""Safe distillation duration-specific metric adaptation on top of the acoustic embedding.

The expensive acoustic path is deliberately identical to acoustic.  A released
distillation checkpoint contains two 448 x 448 linear maps, obtained by averaging the
materialized maps from three independently trained domain-KD seeds.  Full
utterances bypass the maps exactly; only the official 0.5 s and 1.0 s duration
windows are adapted.

The checkpoint is optional at runtime.  Missing, unreleased, malformed, or
corrupt checkpoints enable a global identity fallback.  A numerical problem
while transforming an individual file also falls back to that file's original
acoustic vector.  Consequently an adapter problem cannot change the output shape or
turn an otherwise valid acoustic inference into a failed submission.
"""

import hashlib
import hmac
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ensemble_inference import DurationAdaptiveEnsembleExtractor
from speaker_inference import l2_normalize


ADAPTER_SCHEMA_VERSION = 1
BASE_EMBEDDING_DIM = 448
OUTPUT_DIM = 448
RELEASE_STATUS_PASSED = "passed"
CHECKPOINT_KEYS = (
    "schema_version",
    "release_status",
    "seed_ids",
    "L05",
    "L10",
    "payload_sha256",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_DOMAIN_SEPARATOR = bytes.fromhex("67616d653832332d7631312d646f6d61696e2d6b642d6c696e6561722d736f75702d763100")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a checkpoint without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_int64_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array, dtype="<i8").tobytes(order="C")


def _canonical_float32_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array, dtype="<f4").tobytes(order="C")


def compute_adapter_payload_sha256(
    schema_version: int,
    release_status: str,
    seed_ids: np.ndarray,
    matrix_0p5: np.ndarray,
    matrix_1p0: np.ndarray,
) -> str:
    """Hash the semantic payload using a deterministic byte representation.

    Export tools should call this helper before writing ``payload_sha256``.
    The self-contained payload digest complements the optional whole-file hash
    supplied by ``expected_adapter_sha256``.
    """

    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN_SEPARATOR)
    digest.update(_canonical_int64_bytes(np.asarray(schema_version)))
    digest.update(release_status.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(_canonical_int64_bytes(seed_ids))
    digest.update(_canonical_float32_bytes(matrix_0p5))
    digest.update(_canonical_float32_bytes(matrix_1p0))
    return digest.hexdigest()


def _strict_scalar_integer(value: np.ndarray, name: str) -> int:
    if value.shape != () or value.dtype != np.dtype(np.int64):
        raise ValueError(
            "{} 必须是 int64 标量，实际 shape={} dtype={}".format(
                name, value.shape, value.dtype
            )
        )
    return int(value.item())


def _strict_scalar_text(value: np.ndarray, name: str) -> str:
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(
            "{} 必须是 Unicode 标量，实际 shape={} dtype={}".format(
                name, value.shape, value.dtype
            )
        )
    return str(value.item())


def _strict_matrix(value: np.ndarray, name: str) -> np.ndarray:
    if value.shape != (BASE_EMBEDDING_DIM, BASE_EMBEDDING_DIM):
        raise ValueError(
            "{} 形状错误：期望 ({}, {})，实际 {}".format(
                name,
                BASE_EMBEDDING_DIM,
                BASE_EMBEDDING_DIM,
                value.shape,
            )
        )
    if value.dtype != np.dtype(np.float32):
        raise ValueError("{} 必须是 float32，实际 {}".format(name, value.dtype))
    if not np.isfinite(value).all():
        raise ValueError("{} 包含 NaN/Inf".format(name))
    matrix = np.ascontiguousarray(value).copy()
    matrix.setflags(write=False)
    return matrix


def load_domain_kd_adapter(
    checkpoint_path: Path,
    expected_adapter_sha256: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Strictly load and authenticate one released distillation adapter checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("distillation adapter 不存在：{}".format(checkpoint_path))

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if expected_adapter_sha256 is not None:
        normalized_expected = expected_adapter_sha256.strip().lower()
        if not _SHA256_PATTERN.fullmatch(normalized_expected):
            raise ValueError("expected_adapter_sha256 必须是 64 位十六进制字符串")
        if not hmac.compare_digest(checkpoint_sha256, normalized_expected):
            raise ValueError(
                "distillation adapter 文件 SHA-256 不匹配：expected={} actual={}".format(
                    normalized_expected, checkpoint_sha256
                )
            )

    with np.load(str(checkpoint_path), allow_pickle=False) as payload:
        actual_keys = tuple(sorted(payload.files))
        expected_keys = tuple(sorted(CHECKPOINT_KEYS))
        if actual_keys != expected_keys:
            raise ValueError(
                "distillation adapter schema 键错误：expected={} actual={}".format(
                    expected_keys, actual_keys
                )
            )
        schema_version = _strict_scalar_integer(
            payload["schema_version"], "schema_version"
        )
        release_status = _strict_scalar_text(
            payload["release_status"], "release_status"
        )
        seed_ids = payload["seed_ids"]
        if seed_ids.dtype != np.dtype(np.int64) or seed_ids.shape != (3,):
            raise ValueError(
                "seed_ids 必须是 shape=(3,) 的 int64，实际 shape={} dtype={}".format(
                    seed_ids.shape, seed_ids.dtype
                )
            )
        seed_ids = np.ascontiguousarray(seed_ids).copy()
        matrix_0p5 = _strict_matrix(payload["L05"], "L05")
        matrix_1p0 = _strict_matrix(payload["L10"], "L10")
        declared_payload_sha256 = _strict_scalar_text(
            payload["payload_sha256"], "payload_sha256"
        ).lower()

    if schema_version != ADAPTER_SCHEMA_VERSION:
        raise ValueError(
            "distillation adapter schema_version 不支持：{}".format(schema_version)
        )
    if release_status != RELEASE_STATUS_PASSED:
        raise ValueError(
            "distillation adapter 未通过发布门禁：release_status={}".format(
                release_status
            )
        )
    if len(set(int(value) for value in seed_ids)) != 3:
        raise ValueError("seed_ids 必须包含三个不同训练种子")
    if not _SHA256_PATTERN.fullmatch(declared_payload_sha256):
        raise ValueError("payload_sha256 必须是 64 位小写十六进制字符串")

    actual_payload_sha256 = compute_adapter_payload_sha256(
        schema_version,
        release_status,
        seed_ids,
        matrix_0p5,
        matrix_1p0,
    )
    if not hmac.compare_digest(
        actual_payload_sha256, declared_payload_sha256
    ):
        raise ValueError(
            "distillation adapter payload SHA-256 不匹配：declared={} actual={}".format(
                declared_payload_sha256, actual_payload_sha256
            )
        )

    matrices = {
        "short_0p5": matrix_0p5,
        "short_1p0": matrix_1p0,
    }
    metadata = {
        "available": True,
        "schema_version": schema_version,
        "release_status": release_status,
        "seed_ids": tuple(int(value) for value in seed_ids),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "payload_sha256": actual_payload_sha256,
    }
    return matrices, metadata


class DomainKDExtractor:
    """Apply released duration-specific metric maps to exact acoustic embeddings."""

    output_dim = OUTPUT_DIM

    def __init__(
        self,
        pt_model_path: Optional[Path] = None,
        campplus_model_path: Optional[Path] = None,
        adapter_path: Optional[Path] = None,
        expected_adapter_sha256: Optional[str] = None,
        device_name: str = "auto",
        max_chunk_seconds: float = 60.0,
        chunk_hop_seconds: float = 30.0,
        base_extractor: Optional[DurationAdaptiveEnsembleExtractor] = None,
    ):
        if base_extractor is None:
            if pt_model_path is None or campplus_model_path is None:
                raise ValueError(
                    "未注入 base_extractor 时必须提供 PT 和 CAM++ 权重"
                )
            base_extractor = DurationAdaptiveEnsembleExtractor(
                pt_model_path=Path(pt_model_path),
                campplus_model_path=Path(campplus_model_path),
                device_name=device_name,
                max_chunk_seconds=max_chunk_seconds,
                chunk_hop_seconds=chunk_hop_seconds,
            )
        if int(base_extractor.output_dim) != BASE_EMBEDDING_DIM:
            raise ValueError(
                "distillation 只接受 448D acoustic base，实际为 {}".format(
                    base_extractor.output_dim
                )
            )

        self.acoustic_extractor = base_extractor
        self.device = base_extractor.device
        self.pt_metadata = getattr(base_extractor, "pt_metadata", {})
        self.campplus_metadata = getattr(
            base_extractor, "campplus_metadata", {}
        )
        self._adapter_matrices = {}
        self.adapter_metadata = {
            "available": False,
            "schema_version": None,
            "release_status": "identity_fallback",
            "seed_ids": (),
            "checkpoint_path": (
                str(Path(adapter_path).resolve())
                if adapter_path is not None
                else None
            ),
            "checkpoint_sha256": None,
            "payload_sha256": None,
            "load_error": "adapter_path 未配置",
        }
        if adapter_path is not None:
            try:
                matrices, metadata = load_domain_kd_adapter(
                    Path(adapter_path), expected_adapter_sha256
                )
                self._adapter_matrices = matrices
                self.adapter_metadata = metadata
            except Exception as exc:
                self.adapter_metadata["load_error"] = "{}: {}".format(
                    type(exc).__name__, exc
                )

        self.release_status = self.adapter_metadata["release_status"]
        self.checkpoint_sha256 = self.adapter_metadata["checkpoint_sha256"]
        self.seed_ids = self.adapter_metadata["seed_ids"]

    @property
    def adapter_available(self) -> bool:
        return bool(self.adapter_metadata.get("available", False))

    @staticmethod
    def _validate_base_embedding(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding)
        if embedding.shape != (BASE_EMBEDDING_DIM,):
            raise ValueError(
                "acoustic base embedding 形状错误：{}".format(embedding.shape)
            )
        if embedding.dtype != np.dtype(np.float32):
            embedding = embedding.astype(np.float32)
        if not np.isfinite(embedding).all():
            raise ValueError("acoustic base embedding 包含 NaN/Inf")
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm < 1.0e-12:
            raise ValueError("acoustic base embedding 为零范数")
        # Do not renormalize here.  The acoustic extractor already returns a unit
        # float32 vector, and the full/global-fallback paths must preserve its
        # values exactly rather than introduce a second rounding step.
        return np.ascontiguousarray(embedding)

    def _adapt_short(
        self, embedding: np.ndarray, duration_policy: str
    ) -> np.ndarray:
        matrix = self._adapter_matrices[duration_policy]
        transformed = np.matmul(matrix, embedding)
        if transformed.shape != (OUTPUT_DIM,):
            raise ValueError(
                "distillation adapter 输出形状错误：{}".format(transformed.shape)
            )
        if transformed.dtype != np.dtype(np.float32):
            transformed = transformed.astype(np.float32)
        return l2_normalize(transformed, "distillation domain-KD")

    def extract_file(self, wav_path: Path):
        base_embedding, base_metadata = self.acoustic_extractor.extract_file(wav_path)
        base_embedding = self._validate_base_embedding(base_embedding)
        metadata = dict(base_metadata)
        duration_policy = metadata.get("duration_policy")
        if duration_policy not in ("full", "short_0p5", "short_1p0"):
            duration_seconds = float(metadata.get("duration_seconds", -1.0))
            duration_policy, _ = self.acoustic_extractor.duration_policy(
                duration_seconds
            )

        fallback_mode = "none"
        fallback_reason = ""
        if duration_policy == "full":
            embedding = base_embedding
            adapter_policy = "identity_full"
        elif not self.adapter_available:
            embedding = base_embedding
            adapter_policy = "identity_global_fallback"
            fallback_mode = "identity"
            fallback_reason = str(
                self.adapter_metadata.get("load_error", "adapter unavailable")
            )
        else:
            try:
                embedding = self._adapt_short(base_embedding, duration_policy)
                adapter_policy = duration_policy
            except Exception as exc:
                embedding = base_embedding
                adapter_policy = "identity_file_fallback"
                fallback_mode = "identity"
                fallback_reason = "{}: {}".format(type(exc).__name__, exc)

        embedding = self._validate_base_embedding(embedding)
        metadata.update(
            {
                "duration_policy": duration_policy,
                "adapter_policy": adapter_policy,
                "adapter_available": self.adapter_available,
                "adapter_release_status": self.release_status,
                "adapter_checkpoint_sha256": self.checkpoint_sha256,
                "adapter_seed_ids": self.seed_ids,
                "fallback_mode": fallback_mode,
                "fallback_reason": fallback_reason,
                "output_dim": self.output_dim,
            }
        )
        return embedding, metadata
