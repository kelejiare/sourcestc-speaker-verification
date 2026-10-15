#!/usr/bin/env python3
"""duration_metric: short_pt short-PT subspace adaptation composed before online short_metric CML.

Only source-one exact 0.5-second test files are changed.  Enrollment, source-two,
1.0-second and full-duration files retain the short_metric graph.  The composition keeps
the short_metric 3328-D output contract and fails closed on every adapter mismatch.
"""

import hashlib
import hmac
import math
from pathlib import Path

import numpy as np

from speaker_inference import l2_normalize
from source_kd_inference import source_policy_for_path
from short_duration_metric import (
    ALPHA,
    OUTPUT_DIM,
    load_short_metric_adapter,
)


JOINT_DIM = 832
PT_DIM = 256
SHORT_PT_ADAPTER_SHA256 = "b6376ee48daf7b41204e020bf2a2113dc952d02bf3dafa40d82fafa74d5496ae"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_short_pt_pt_adapter(path):
    path = Path(path)
    observed = sha256_file(path)
    if not hmac.compare_digest(observed, SHORT_PT_ADAPTER_SHA256):
        raise RuntimeError("duration_metric short_pt adapter SHA-256 mismatch")
    with np.load(str(path), allow_pickle=False) as payload:
        if set(payload.files) != {"schema_version", "weight", "bias", "seed"}:
            raise RuntimeError("duration_metric short_pt adapter schema mismatch")
        schema = np.asarray(payload["schema_version"])
        weight = np.asarray(payload["weight"])
        bias = np.asarray(payload["bias"])
        seed = np.asarray(payload["seed"])
    if schema.shape != () or schema.dtype != np.int64 or int(schema) != 1:
        raise RuntimeError("duration_metric short_pt schema version mismatch")
    if weight.shape != (PT_DIM, PT_DIM) or weight.dtype != np.float32:
        raise RuntimeError("duration_metric short_pt weight contract mismatch")
    if bias.shape != (PT_DIM,) or bias.dtype != np.float32:
        raise RuntimeError("duration_metric short_pt bias contract mismatch")
    if seed.shape != () or seed.dtype != np.int64 or int(seed) != -1:
        raise RuntimeError("duration_metric short_pt ensemble marker mismatch")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise RuntimeError("duration_metric short_pt adapter contains NaN/Inf")
    return np.ascontiguousarray(weight.copy()), np.ascontiguousarray(bias.copy()), observed


class DurationMetricExtractor:
    output_dim = OUTPUT_DIM

    def __init__(self, joint_target_extractor, short_metric_adapter_path, short_pt_adapter_path):
        if int(getattr(joint_target_extractor, "output_dim", -1)) != JOINT_DIM:
            raise RuntimeError("duration_metric requires the exact joint target extractor")
        self.joint = joint_target_extractor
        self.factors, self.short_metric_metadata = load_short_metric_adapter(short_metric_adapter_path)
        self.pt_weight, self.pt_bias, self.short_pt_sha256 = load_short_pt_pt_adapter(short_pt_adapter_path)
        self.base_scale = math.sqrt(1.0 - ALPHA)
        self.expert_scale = math.sqrt(ALPHA / 3.0)

    @staticmethod
    def _unit(value, label):
        return l2_normalize(np.asarray(value, dtype=np.float32), label)

    def _adapt_pt(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (JOINT_DIM,) or not np.isfinite(vector).all():
            raise RuntimeError("duration_metric joint anchor contract mismatch")
        output = np.array(vector, dtype=np.float32, copy=True)
        scale = float(np.linalg.norm(output[:PT_DIM]))
        if not math.isfinite(scale) or scale <= 1.0e-12:
            raise RuntimeError("duration_metric PT sub-block norm invalid")
        unit = output[:PT_DIM] / scale
        proposed = np.einsum("j,ij->i", unit, self.pt_weight, optimize=False) + self.pt_bias
        output[:PT_DIM] = np.float32(scale) * self._unit(proposed, "duration_metric adapted PT")
        return self._unit(output, "duration_metric short_pt anchor")

    def _expert(self, vector, factors):
        proposed = vector + np.matmul(factors[1], np.matmul(factors[0], vector))
        return self._unit(proposed, "duration_metric CML expert")

    def _cml(self, vector):
        vector = self._unit(vector, "duration_metric joint anchor")
        experts = [self._expert(vector, factors) for factors in self.factors]
        output = np.concatenate(
            [self.base_scale * vector]
            + [self.expert_scale * expert for expert in experts]
        ).astype(np.float32)
        output = self._unit(output, "duration_metric output")
        if output.shape != (OUTPUT_DIM,):
            raise RuntimeError("duration_metric output dimension mismatch")
        return output

    def extract_file(self, wav_path):
        vector, metadata = self.joint.extract_file(wav_path)
        metadata = dict(metadata)
        changed = (
            source_policy_for_path(wav_path) == "data_scenery1_mobile"
            and str(metadata.get("duration_policy", "")) == "short_0p5"
        )
        if changed:
            vector = self._adapt_pt(vector)
        output = self._cml(vector)
        metadata.update({
            "source_policy": source_policy_for_path(wav_path),
            "release_policy": "duration_metric_short_pt_pt_then_short_metric_cml" if changed else "duration_metric_exact_short_metric_graph",
            "duration_metric_changed_region": bool(changed),
            "duration_metric_short_pt_adapter_sha256": self.short_pt_sha256,
            "duration_metric_short_metric_adapter_sha256": self.short_metric_metadata["sha256"],
            "fallback_mode": "none",
            "fallback_reason": "",
            "output_dim": OUTPUT_DIM,
            "frontend_mode": "joint_target+short_pt_short05_pt_subspace+short_metric_cml",
        })
        return output, metadata
