#!/usr/bin/env python3
"""joint: distillation-compatible dual-ReDim KD with audited duration weights."""

import hashlib
import hmac
from pathlib import Path

import numpy as np

from audio_features import FbankConfig, load_audio
from ensemble_inference import DurationAdaptiveEnsembleExtractor
from multidomain_kd_inference import (
    MODEL_SPECS,
    OUTPUT_DIM,
    REDIM_DIM,
    ACOUSTIC_DIM,
    MultidomainReDimKDExtractor,
    _extract_acoustic_waveform,
    _load_redim,
    sha256_file,
)


ADAPTER_SHA256 = "a221c0ffcdafd6d56b8d15c2c39fb084f6f88025bce6843beee4f230396fad38"
ADAPTER_PAYLOAD_SHA256 = "d716dce1fe9b6cb694d545f3b783e24334eecbc8ba5f07531cb4440a42d08ea8"
ADAPTER_DOMAIN = bytes.fromhex("67616d653832332d7631322d6a6f696e742d726564696d2d6f75747075742d6b642d763100")
ADAPTER_CONFIG_ID = "d26_h29_c18_a08"
ADAPTER_RELEASE_STATUS = "audited_posthoc_robust_candidate_runtime_pending"
ADAPTER_KEYS = (
    "schema_version", "release_status", "adapter_config_id", "seed_ids",
    "lm_short05", "lm_short10", "ptn_short05", "ptn_short10",
    "payload_sha256",
)
MATRIX_KEYS = ("lm_short05", "lm_short10", "ptn_short05", "ptn_short10")
SEED_IDS = (825001, 825002, 825003)

# These are cosine-score weights because every block is unit-normalized and
# concatenated after multiplication by sqrt(weight).
ENROLL_WEIGHTS = np.asarray((0.72, 0.252, 0.028), dtype=np.float64)
SHORT05_WEIGHTS = np.asarray((0.32, 0.578, 0.102), dtype=np.float64)
SHORT10_WEIGHTS = np.asarray((0.26, 0.703, 0.037), dtype=np.float64)


def _payload_sha(schema_version, release_status, config_id, seeds, matrices):
    digest = hashlib.sha256()
    digest.update(ADAPTER_DOMAIN)
    digest.update(np.asarray(schema_version, dtype="<i8").tobytes())
    digest.update(release_status.encode("utf-8") + b"\x00")
    digest.update(config_id.encode("ascii") + b"\x00")
    digest.update(np.ascontiguousarray(seeds, dtype="<i8").tobytes())
    for key in MATRIX_KEYS:
        digest.update(key.encode("ascii") + b"\x00")
        digest.update(np.ascontiguousarray(matrices[key], dtype="<f4").tobytes())
    return digest.hexdigest()


def _load_joint_adapter(path):
    path = Path(path)
    observed = sha256_file(path)
    if not hmac.compare_digest(observed, ADAPTER_SHA256):
        raise RuntimeError("joint deployment adapter SHA-256 mismatch")
    with np.load(str(path), allow_pickle=False) as payload:
        if tuple(sorted(payload.files)) != tuple(sorted(ADAPTER_KEYS)):
            raise RuntimeError("joint deployment adapter key schema mismatch")
        schema_version = np.asarray(payload["schema_version"])
        release_status = np.asarray(payload["release_status"])
        config_id = np.asarray(payload["adapter_config_id"])
        seeds = np.asarray(payload["seed_ids"])
        declared = np.asarray(payload["payload_sha256"])
        if schema_version.shape != () or schema_version.dtype != np.int64 or int(schema_version) != 1:
            raise RuntimeError("joint adapter schema_version mismatch")
        if release_status.shape != () or release_status.dtype.kind != "U":
            raise RuntimeError("joint adapter release_status type mismatch")
        release_status = str(release_status.item())
        if release_status != ADAPTER_RELEASE_STATUS:
            raise RuntimeError("joint adapter release_status mismatch")
        if config_id.shape != () or config_id.dtype.kind != "U":
            raise RuntimeError("joint adapter config_id type mismatch")
        config_id = str(config_id.item())
        if config_id != ADAPTER_CONFIG_ID:
            raise RuntimeError("joint adapter config_id mismatch")
        if seeds.shape != (3,) or seeds.dtype != np.int64:
            raise RuntimeError("joint adapter seed schema mismatch")
        if tuple(int(item) for item in seeds) != SEED_IDS:
            raise RuntimeError("joint adapter seed contract mismatch")
        matrices = {}
        for key in MATRIX_KEYS:
            value = np.asarray(payload[key])
            if value.shape != (REDIM_DIM, REDIM_DIM) or value.dtype != np.float32:
                raise RuntimeError("{} adapter matrix shape/dtype mismatch".format(key))
            if not np.isfinite(value).all():
                raise RuntimeError("{} adapter matrix contains NaN/Inf".format(key))
            matrices[key] = np.ascontiguousarray(value).copy()
        if declared.shape != () or declared.dtype.kind != "U":
            raise RuntimeError("joint adapter payload digest type mismatch")
        declared = str(declared.item())
    actual_payload = _payload_sha(
        1, release_status, config_id, seeds, matrices
    )
    if not hmac.compare_digest(actual_payload, declared):
        raise RuntimeError("joint adapter declared semantic digest mismatch")
    if not hmac.compare_digest(actual_payload, ADAPTER_PAYLOAD_SHA256):
        raise RuntimeError("joint adapter frozen semantic digest mismatch")
    return matrices, {
        "sha256": observed,
        "payload_sha256": actual_payload,
        "config_id": config_id,
        "seed_ids": tuple(int(item) for item in seeds),
        "release_status": release_status,
    }


class JointKDReweightExtractor(MultidomainReDimKDExtractor):
    """Emit one finite, unit-normalized 832-D joint embedding."""

    output_dim = OUTPUT_DIM

    def __init__(self, pt_model_path, campplus_model_path, lm_model_path,
                 ptn_model_path, adapter_path, device_name="auto",
                 max_chunk_seconds=60.0, chunk_hop_seconds=30.0,
                 base_extractor=None):
        self.acoustic_extractor = base_extractor
        if self.acoustic_extractor is None:
            self.acoustic_extractor = DurationAdaptiveEnsembleExtractor(
                Path(pt_model_path), Path(campplus_model_path),
                device_name=device_name,
                max_chunk_seconds=max_chunk_seconds,
                chunk_hop_seconds=chunk_hop_seconds,
            )
        if int(self.acoustic_extractor.output_dim) != ACOUSTIC_DIM:
            raise RuntimeError("joint base extractor must be exact 448-D acoustic")
        self.device = self.acoustic_extractor.device
        self.fbank_config = FbankConfig()
        self.redim_models = {}
        self.redim_metadata = {}
        for name, path in (("lm", lm_model_path), ("ptn", ptn_model_path)):
            self.redim_models[name], self.redim_metadata[name] = _load_redim(
                Path(path), name, self.device
            )
        self.adapters, self.adapter_metadata = _load_joint_adapter(Path(adapter_path))
        self.pt_metadata = self.acoustic_extractor.pt_metadata
        self.campplus_metadata = self.acoustic_extractor.campplus_metadata

    def extract_file(self, wav_path):
        waveform = load_audio(
            Path(wav_path), self.fbank_config.sample_rate, remove_dc=False
        )
        duration_seconds = (
            float(waveform.size) / float(self.fbank_config.sample_rate)
        )
        policy, pt_weight = self.acoustic_extractor.duration_policy(duration_seconds)
        acoustic, pt_frames, camp_frames = _extract_acoustic_waveform(
            self.acoustic_extractor, waveform, pt_weight
        )
        try:
            segments = self.acoustic_extractor.pt_extractor._segments(waveform)
            lm, lm_samples = self._redim_embedding(
                self.redim_models["lm"], segments, "LM ReDimNet2"
            )
            ptn, ptn_samples = self._redim_embedding(
                self.redim_models["ptn"], segments, "PTN ReDimNet2"
            )
            if policy == "short_0p5":
                lm = self._adapt(lm, "lm", "short05")
                ptn = self._adapt(ptn, "ptn", "short05")
                weights = SHORT05_WEIGHTS
                adapter_policy = "short05"
            elif policy == "short_1p0":
                lm = self._adapt(lm, "lm", "short10")
                ptn = self._adapt(ptn, "ptn", "short10")
                weights = SHORT10_WEIGHTS
                adapter_policy = "short10"
            else:
                weights = ENROLL_WEIGHTS
                adapter_policy = "full_bypass"
            embedding = self._fuse(acoustic, lm, ptn, weights)
            fallback_mode = "none"
            fallback_reason = ""
        except Exception as exc:
            embedding = self.padded_acoustic(acoustic)
            weights = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
            adapter_policy = "padded_acoustic_file_fallback"
            lm_samples = []
            ptn_samples = []
            fallback_mode = "acoustic_padded"
            fallback_reason = "{}: {}".format(type(exc).__name__, exc)
        return embedding, {
            "duration_seconds": duration_seconds,
            "duration_policy": policy,
            "adapter_policy": adapter_policy,
            "branch_weights": tuple(float(item) for item in weights),
            "pt_frame_counts": pt_frames,
            "campplus_frame_counts": camp_frames,
            "lm_sample_counts": lm_samples,
            "ptn_sample_counts": ptn_samples,
            "fallback_mode": fallback_mode,
            "fallback_reason": fallback_reason,
            "output_dim": self.output_dim,
            "frontend_mode": "acoustic_exact_pure_torch+joint_dual_redimnet2_kd",
        }


class JointPaddedAcousticFallbackExtractor:
    """Numerically exact acoustic score in the fixed 832-D output space."""

    output_dim = OUTPUT_DIM

    def __init__(self, acoustic_extractor):
        self.acoustic_extractor = acoustic_extractor

    def extract_file(self, wav_path):
        acoustic, metadata = self.acoustic_extractor.extract_file(Path(wav_path))
        output = JointKDReweightExtractor.padded_acoustic(acoustic)
        metadata = dict(metadata)
        metadata.update({
            "adapter_policy": "padded_acoustic_fallback",
            "fallback_mode": "acoustic_padded",
            "fallback_reason": "joint-only resources unavailable",
            "output_dim": self.output_dim,
        })
        return output, metadata


def assert_frozen_contract():
    for name, value in (
        ("enroll", ENROLL_WEIGHTS),
        ("short05", SHORT05_WEIGHTS),
        ("short10", SHORT10_WEIGHTS),
    ):
        if value.shape != (3,) or np.any(value < 0.0):
            raise RuntimeError("{} weights are invalid".format(name))
        if abs(float(value.sum()) - 1.0) > 1.0e-12:
            raise RuntimeError("{} weights do not sum to one".format(name))
    if set(MODEL_SPECS) != {"lm", "ptn"}:
        raise RuntimeError("joint ReDimNet2 model set changed")


assert_frozen_contract()
