#!/usr/bin/env python3
"""mobile: official-source-aware mobile KD on top of the joint safe baseline."""

import hashlib
import hmac
from pathlib import Path

import numpy as np

from audio_features import load_audio
from speaker_inference import l2_normalize
from multidomain_kd_inference import OUTPUT_DIM, REDIM_DIM, _extract_acoustic_waveform, sha256_file
from joint_kd_inference import JointKDReweightExtractor


MOBILE_ADAPTER_SHA256 = "ca8626992836054cda80673eedb95d66604b9a2f5b2430bc9543110e6251c859"
MOBILE_PAYLOAD_SHA256 = "19d983b2f91b6ee606a7930182c7466bc3d8807ce48903723ee4bcc8d113c1a1"
MOBILE_DOMAIN = bytes.fromhex("67616d653832332d7631332d6d6f62696c652d726564696d2d6f75747075742d6b642d763100")
MOBILE_RELEASE_STATUS = "frozen_proxy_gate_pass_runtime_pending"
MOBILE_CONFIG_ID = "mobile_d42_c42_short05_only"
MOBILE_SEEDS = (825001, 825002, 825003)
MOBILE_MATRIX_KEYS = ("lm_short05", "lm_short10", "ptn_short05", "ptn_short10")
MOBILE_ADAPTER_KEYS = (
    "schema_version", "release_status", "adapter_config_id", "seed_ids",
    "mix05", "mix10", "lm_short05", "lm_short10", "ptn_short05",
    "ptn_short10", "payload_sha256",
)

# Cosine-score weights.  Each block is independently unit-normalized and is
# multiplied by sqrt(weight) before final concatenation/normalization.
MOBILE_ENROLL_WEIGHTS = np.asarray((0.50, 0.35, 0.15), dtype=np.float64)
MOBILE_SHORT05_WEIGHTS = np.asarray((0.15, 0.595, 0.255), dtype=np.float64)
MOBILE_SHORT10_WEIGHTS = np.asarray((0.32, 0.578, 0.102), dtype=np.float64)


def _mobile_payload_sha(schema_version, release_status, config_id, seeds,
                        mix05, mix10, matrices):
    digest = hashlib.sha256()
    digest.update(MOBILE_DOMAIN)
    digest.update(np.asarray(schema_version, dtype="<i8").tobytes())
    digest.update(release_status.encode("utf-8") + b"\x00")
    digest.update(config_id.encode("ascii") + b"\x00")
    digest.update(np.ascontiguousarray(seeds, dtype="<i8").tobytes())
    digest.update(np.asarray(mix05, dtype="<f8").tobytes())
    digest.update(np.asarray(mix10, dtype="<f8").tobytes())
    for key in MOBILE_MATRIX_KEYS:
        digest.update(key.encode("ascii") + b"\x00")
        digest.update(np.ascontiguousarray(matrices[key], dtype="<f4").tobytes())
    return digest.hexdigest()


def load_mobile_adapter(path):
    path = Path(path)
    observed = sha256_file(path)
    if not hmac.compare_digest(observed, MOBILE_ADAPTER_SHA256):
        raise RuntimeError("mobile mobile adapter SHA-256 mismatch")
    with np.load(str(path), allow_pickle=False) as payload:
        if tuple(sorted(payload.files)) != tuple(sorted(MOBILE_ADAPTER_KEYS)):
            raise RuntimeError("mobile mobile adapter key schema mismatch")
        schema = np.asarray(payload["schema_version"])
        status = np.asarray(payload["release_status"])
        config = np.asarray(payload["adapter_config_id"])
        seeds = np.asarray(payload["seed_ids"])
        mix05 = np.asarray(payload["mix05"])
        mix10 = np.asarray(payload["mix10"])
        declared = np.asarray(payload["payload_sha256"])
        if schema.shape != () or schema.dtype != np.int64 or int(schema) != 1:
            raise RuntimeError("mobile mobile adapter schema mismatch")
        if status.shape != () or status.dtype.kind != "U" or str(status.item()) != MOBILE_RELEASE_STATUS:
            raise RuntimeError("mobile mobile adapter release status mismatch")
        if config.shape != () or config.dtype.kind != "U" or str(config.item()) != MOBILE_CONFIG_ID:
            raise RuntimeError("mobile mobile adapter config mismatch")
        if seeds.shape != (3,) or seeds.dtype != np.int64 or tuple(int(item) for item in seeds) != MOBILE_SEEDS:
            raise RuntimeError("mobile mobile adapter seed contract mismatch")
        if mix05.shape != () or mix05.dtype != np.float64 or float(mix05) != 1.0:
            raise RuntimeError("mobile mobile short05 mix mismatch")
        if mix10.shape != () or mix10.dtype != np.float64 or float(mix10) != 0.0:
            raise RuntimeError("mobile mobile short10 mix mismatch")
        matrices = {}
        for key in MOBILE_MATRIX_KEYS:
            value = np.asarray(payload[key])
            if value.shape != (REDIM_DIM, REDIM_DIM) or value.dtype != np.float32:
                raise RuntimeError("{} matrix contract mismatch".format(key))
            if not np.isfinite(value).all():
                raise RuntimeError("{} contains NaN/Inf".format(key))
            matrices[key] = np.ascontiguousarray(value).copy()
        if declared.shape != () or declared.dtype.kind != "U":
            raise RuntimeError("mobile mobile adapter semantic digest type mismatch")
        declared = str(declared.item())
    actual = _mobile_payload_sha(
        1, MOBILE_RELEASE_STATUS, MOBILE_CONFIG_ID, seeds, 1.0, 0.0, matrices
    )
    if not hmac.compare_digest(actual, declared) or not hmac.compare_digest(actual, MOBILE_PAYLOAD_SHA256):
        raise RuntimeError("mobile mobile adapter semantic digest mismatch")
    return matrices, {
        "sha256": observed, "payload_sha256": actual,
        "config_id": MOBILE_CONFIG_ID, "seed_ids": MOBILE_SEEDS,
        "mix05": 1.0, "mix10": 0.0,
    }


def source_policy_for_path(wav_path):
    """Map only the official first source to mobile; all else stays joint."""

    parts = tuple(part.lower() for part in Path(wav_path).parts)
    if "data_scenery1" in parts:
        return "data_scenery1_mobile"
    if "data_scenery2" in parts:
        return "data_scenery2_joint"
    return "unknown_joint"


class SourceKDExtractor:
    """Share all acoustic models with joint and alter only official source one."""

    output_dim = OUTPUT_DIM

    def __init__(self, joint_extractor, mobile_adapter_path):
        if not isinstance(joint_extractor, JointKDReweightExtractor):
            raise TypeError("mobile requires an initialized strict joint extractor")
        if int(joint_extractor.output_dim) != OUTPUT_DIM:
            raise RuntimeError("mobile/joint output dimension mismatch")
        self.joint = joint_extractor
        self.mobile_adapters, self.mobile_adapter_metadata = load_mobile_adapter(
            mobile_adapter_path
        )

    def _mobile_adapt(self, vector, name, duration_key):
        source = l2_normalize(vector, "mobile {} source".format(name))
        delta = self.mobile_adapters[name + "_" + duration_key]
        proposed = source + np.matmul(delta, source)
        return l2_normalize(proposed.astype(np.float32), "mobile {} adapted".format(name))

    def extract_file(self, wav_path):
        source_policy = source_policy_for_path(wav_path)
        if source_policy != "data_scenery1_mobile":
            embedding, metadata = self.joint.extract_file(wav_path)
            metadata = dict(metadata)
            metadata.update({
                "source_policy": source_policy,
                "release_policy": "exact_joint",
            })
            return embedding, metadata

        waveform = load_audio(
            Path(wav_path), self.joint.fbank_config.sample_rate, remove_dc=False
        )
        duration_seconds = float(waveform.size) / float(self.joint.fbank_config.sample_rate)
        policy, pt_weight = self.joint.acoustic_extractor.duration_policy(duration_seconds)
        acoustic, pt_frames, camp_frames = _extract_acoustic_waveform(
            self.joint.acoustic_extractor, waveform, pt_weight
        )
        try:
            segments = self.joint.acoustic_extractor.pt_extractor._segments(waveform)
            lm, lm_samples = self.joint._redim_embedding(
                self.joint.redim_models["lm"], segments, "LM ReDimNet2"
            )
            ptn, ptn_samples = self.joint._redim_embedding(
                self.joint.redim_models["ptn"], segments, "PTN ReDimNet2"
            )
            if policy == "short_0p5":
                lm = self._mobile_adapt(lm, "lm", "short05")
                ptn = self._mobile_adapt(ptn, "ptn", "short05")
                weights = MOBILE_SHORT05_WEIGHTS
                adapter_policy = "mobile_short05"
            elif policy == "short_1p0":
                lm = self.joint._adapt(lm, "lm", "short10")
                ptn = self.joint._adapt(ptn, "ptn", "short10")
                weights = MOBILE_SHORT10_WEIGHTS
                adapter_policy = "joint_short10_mobile_weights"
            else:
                weights = MOBILE_ENROLL_WEIGHTS
                adapter_policy = "full_bypass_mobile_weights"
            embedding = self.joint._fuse(acoustic, lm, ptn, weights)
            fallback_mode = "none"
            fallback_reason = ""
        except Exception as exc:
            embedding = self.joint.padded_acoustic(acoustic)
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
            "source_policy": source_policy,
            "release_policy": "mobile_mobile",
            "branch_weights": tuple(float(item) for item in weights),
            "pt_frame_counts": pt_frames,
            "campplus_frame_counts": camp_frames,
            "lm_sample_counts": lm_samples,
            "ptn_sample_counts": ptn_samples,
            "fallback_mode": fallback_mode,
            "fallback_reason": fallback_reason,
            "output_dim": self.output_dim,
            "frontend_mode": "acoustic_exact_pure_torch+mobile_source_conditional_dual_redimnet2_kd",
        }


def assert_frozen_contract():
    for name, value in (
        ("mobile_enroll", MOBILE_ENROLL_WEIGHTS),
        ("mobile_short05", MOBILE_SHORT05_WEIGHTS),
        ("mobile_short10", MOBILE_SHORT10_WEIGHTS),
    ):
        if value.shape != (3,) or np.any(value < 0.0):
            raise RuntimeError("{} weights invalid".format(name))
        if abs(float(value.sum()) - 1.0) > 1.0e-12:
            raise RuntimeError("{} weights do not sum to one".format(name))


assert_frozen_contract()
