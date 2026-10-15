#!/usr/bin/env python3
"""short_metric: exact-joint score anchor plus source-one 0.5 s cosine metric experts."""

import hashlib
import hmac
import math
from pathlib import Path

import numpy as np

from speaker_inference import l2_normalize
from source_kd_inference import source_policy_for_path


JOINT_DIM = 832
OUTPUT_DIM = 3328
SEED_COUNT = 3
RANK = 8
ALPHA = 0.16
CONFIG_ID = "short05_r8_i0p2"
SEED_IDS = (826101, 826102, 826103)
RELEASE_STATUS = "frozen_release"
ADAPTER_SHA256 = "fbf893e6d77f1bc95ea3ffe6c9d8401fd4744a9eefdae860017b279ac3704340"
PAYLOAD_SHA256 = "ca7ccc1107b4d05ded799490e27af28921d980d9b7cf4f75180f87cf7e8a5e06"
DOMAIN = bytes.fromhex("67616d653832332d7631372d7631322d616e63686f7265642d73686f727430352d736565642d636d6c2d763100")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha(arrays):
    digest = hashlib.sha256()
    digest.update(DOMAIN)
    for key in (
        "schema_version", "release_status", "config_id", "seed_ids",
        "rank", "alpha",
    ):
        value = np.asarray(arrays[key])
        digest.update(key.encode("ascii") + b"\x00")
        if value.dtype.kind == "U":
            digest.update(str(value.item()).encode("utf-8") + b"\x00")
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
    for seed_index in range(SEED_COUNT):
        for suffix in ("a", "b"):
            key = "seed{}_short05_{}".format(seed_index, suffix)
            digest.update(key.encode("ascii") + b"\x00")
            digest.update(np.ascontiguousarray(arrays[key], dtype="<f4").tobytes())
    return digest.hexdigest()


def load_short_metric_adapter(path):
    path = Path(path)
    observed = sha256_file(path)
    if not hmac.compare_digest(observed, ADAPTER_SHA256):
        raise RuntimeError("short_metric adapter SHA-256 mismatch")
    expected_keys = {
        "schema_version", "release_status", "config_id", "seed_ids",
        "rank", "alpha", "payload_sha256",
    }
    for seed_index in range(SEED_COUNT):
        expected_keys.add("seed{}_short05_a".format(seed_index))
        expected_keys.add("seed{}_short05_b".format(seed_index))
    arrays = {}
    with np.load(str(path), allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise RuntimeError("short_metric adapter key schema mismatch")
        for key in payload.files:
            arrays[key] = np.asarray(payload[key]).copy()
    scalar_contract = (
        arrays["schema_version"].shape == ()
        and arrays["schema_version"].dtype == np.int64
        and int(arrays["schema_version"]) == 1
        and arrays["release_status"].shape == ()
        and arrays["release_status"].dtype.kind == "U"
        and str(arrays["release_status"].item()) == RELEASE_STATUS
        and arrays["config_id"].shape == ()
        and arrays["config_id"].dtype.kind == "U"
        and str(arrays["config_id"].item()) == CONFIG_ID
        and arrays["rank"].shape == ()
        and arrays["rank"].dtype == np.int64
        and int(arrays["rank"]) == RANK
        and arrays["alpha"].shape == ()
        and arrays["alpha"].dtype == np.float64
        and float(arrays["alpha"]) == ALPHA
    )
    if not scalar_contract:
        raise RuntimeError("short_metric adapter scalar contract mismatch")
    seeds = arrays["seed_ids"]
    if seeds.shape != (SEED_COUNT,) or seeds.dtype != np.int64:
        raise RuntimeError("short_metric seed schema mismatch")
    if tuple(int(item) for item in seeds) != SEED_IDS:
        raise RuntimeError("short_metric seed contract mismatch")
    factors = []
    for seed_index in range(SEED_COUNT):
        a = arrays["seed{}_short05_a".format(seed_index)]
        b = arrays["seed{}_short05_b".format(seed_index)]
        if a.shape != (RANK, JOINT_DIM) or b.shape != (JOINT_DIM, RANK):
            raise RuntimeError("short_metric factor shape mismatch")
        if a.dtype != np.float32 or b.dtype != np.float32:
            raise RuntimeError("short_metric factor dtype mismatch")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError("short_metric factor contains NaN/Inf")
        factors.append((np.ascontiguousarray(a), np.ascontiguousarray(b)))
    declared = arrays["payload_sha256"]
    if declared.shape != () or declared.dtype.kind != "U":
        raise RuntimeError("short_metric payload digest schema mismatch")
    actual = payload_sha(arrays)
    if not hmac.compare_digest(actual, str(declared.item())):
        raise RuntimeError("short_metric declared payload digest mismatch")
    if not hmac.compare_digest(actual, PAYLOAD_SHA256):
        raise RuntimeError("short_metric frozen payload digest mismatch")
    return factors, {
        "sha256": observed,
        "payload_sha256": actual,
        "config_id": CONFIG_ID,
        "seed_ids": SEED_IDS,
        "rank": RANK,
        "alpha": ALPHA,
    }


class ShortDurationMetricExtractor:
    """Emit a finite unit 3328-D vector with an explicit joint score anchor."""

    output_dim = OUTPUT_DIM

    def __init__(self, joint_target_extractor, adapter_path):
        if int(getattr(joint_target_extractor, "output_dim", -1)) != JOINT_DIM:
            raise RuntimeError("short_metric requires the exact 832-D joint extractor")
        self.joint = joint_target_extractor
        self.factors, self.adapter_metadata = load_short_metric_adapter(adapter_path)
        self.base_scale = math.sqrt(1.0 - ALPHA)
        self.expert_scale = math.sqrt(ALPHA / float(SEED_COUNT))

    @staticmethod
    def _expert(vector, factors):
        a, b = factors
        proposed = vector + np.matmul(b, np.matmul(a, vector))
        return l2_normalize(
            np.asarray(proposed, dtype=np.float32), "short_metric short05 metric expert"
        )

    def extract_file(self, wav_path):
        vector, metadata = self.joint.extract_file(wav_path)
        vector = l2_normalize(
            np.asarray(vector, dtype=np.float32), "short_metric joint anchor"
        )
        if vector.shape != (JOINT_DIM,):
            raise RuntimeError("short_metric joint anchor dimension mismatch")
        metadata = dict(metadata)
        source_policy = source_policy_for_path(wav_path)
        duration_policy = str(metadata.get("duration_policy", ""))
        changed = (
            source_policy == "data_scenery1_mobile"
            and duration_policy == "short_0p5"
        )
        experts = (
            [self._expert(vector, factors) for factors in self.factors]
            if changed
            else [vector, vector, vector]
        )
        output = np.concatenate(
            [self.base_scale * vector]
            + [self.expert_scale * expert for expert in experts]
        ).astype(np.float32)
        output = l2_normalize(output, "short_metric anchored output")
        if output.shape != (OUTPUT_DIM,):
            raise RuntimeError("short_metric output dimension mismatch")
        metadata.update({
            "source_policy": source_policy,
            "release_policy": (
                "short_metric_source1_short05_cml" if changed else "short_metric_exact_joint_score"
            ),
            "short_metric_changed_region": bool(changed),
            "short_metric_alpha": ALPHA if changed else 0.0,
            "short_metric_seed_count": SEED_COUNT,
            "short_metric_adapter_config": CONFIG_ID,
            "fallback_mode": "none",
            "fallback_reason": "",
            "output_dim": OUTPUT_DIM,
            "frontend_mode": "joint_strict_r3+short05_test_side_cml",
        })
        return output, metadata


if not (0.0 < ALPHA < 1.0):
    raise RuntimeError("short_metric alpha must be in (0, 1)")
if OUTPUT_DIM != JOINT_DIM * (1 + SEED_COUNT) or OUTPUT_DIM >= 4096:
    raise RuntimeError("short_metric output dimension contract mismatch")
