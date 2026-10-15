#!/usr/bin/env python3
"""short_pt: exact-joint anchor with a frozen 0.5 s PT subspace adapter.

Only the normalized 256-D VoxBlink2 PT sub-block is transformed for files
that joint classifies as the official 0.5 s condition.  Its original block
norm is restored after the affine map, so all joint fusion weights and every
other block remain unchanged.  Other durations return the joint vector
without any numerical operation.
"""

import hashlib
import hmac
from pathlib import Path

import numpy as np

from speaker_inference import l2_normalize


JOINT_DIM = 832
PT_DIM = 256
ADAPTER_SHA256 = "b6376ee48daf7b41204e020bf2a2113dc952d02bf3dafa40d82fafa74d5496ae"
ADAPTER_KEYS = {"schema_version", "weight", "bias", "seed"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_short_pt_adapter(path):
    path = Path(path)
    observed = _sha256(path)
    if not hmac.compare_digest(observed, ADAPTER_SHA256):
        raise RuntimeError("short_pt adapter SHA-256 mismatch")
    with np.load(str(path), allow_pickle=False) as payload:
        if set(payload.files) != ADAPTER_KEYS:
            raise RuntimeError("short_pt adapter key schema mismatch")
        schema = np.asarray(payload["schema_version"])
        weight = np.asarray(payload["weight"])
        bias = np.asarray(payload["bias"])
        seed = np.asarray(payload["seed"])
    if schema.shape != () or schema.dtype != np.int64 or int(schema) != 1:
        raise RuntimeError("short_pt adapter schema version mismatch")
    if weight.shape != (PT_DIM, PT_DIM) or weight.dtype != np.float32:
        raise RuntimeError("short_pt adapter weight contract mismatch")
    if bias.shape != (PT_DIM,) or bias.dtype != np.float32:
        raise RuntimeError("short_pt adapter bias contract mismatch")
    if seed.shape != () or seed.dtype != np.int64 or int(seed) != -1:
        raise RuntimeError("short_pt ensemble marker mismatch")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise RuntimeError("short_pt adapter contains NaN/Inf")
    return (
        np.ascontiguousarray(weight.copy()),
        np.ascontiguousarray(bias.copy()),
        {"sha256": observed, "ensemble_seeds": (20260994, 20260995, 20260996)},
    )


class ShortPTExtractor:
    """Emit one finite unit 832-D vector; fail closed on contract errors."""

    output_dim = JOINT_DIM

    def __init__(self, joint_target_extractor, adapter_path):
        if int(getattr(joint_target_extractor, "output_dim", -1)) != JOINT_DIM:
            raise RuntimeError("short_pt requires the exact 832-D joint target extractor")
        self.joint = joint_target_extractor
        self.weight, self.bias, self.adapter_metadata = load_short_pt_adapter(
            adapter_path
        )

    def _adapt_pt_subblock(self, vector):
        vector = np.asarray(vector)
        if vector.shape != (JOINT_DIM,) or vector.dtype != np.float32:
            raise RuntimeError("short_pt joint anchor shape/dtype mismatch")
        if not np.isfinite(vector).all():
            raise RuntimeError("short_pt joint anchor contains NaN/Inf")
        pt_block = np.asarray(vector[:PT_DIM], dtype=np.float32)
        pt_scale = float(np.linalg.norm(pt_block))
        if not np.isfinite(pt_scale) or pt_scale <= 1.0e-12:
            raise RuntimeError("short_pt PT sub-block has invalid norm")
        pt_unit = pt_block / pt_scale
        adapted = np.einsum(
            "j,ij->i", pt_unit, self.weight, optimize=False
        ) + self.bias
        adapted = l2_normalize(
            np.asarray(adapted, dtype=np.float32), "short_pt adapted PT sub-block"
        )
        output = np.array(vector, dtype=np.float32, copy=True)
        output[:PT_DIM] = np.float32(pt_scale) * adapted
        return l2_normalize(output, "short_pt final embedding")

    def extract_file(self, wav_path):
        vector, metadata = self.joint.extract_file(wav_path)
        if not isinstance(metadata, dict):
            raise TypeError("short_pt requires dict metadata from joint")
        metadata = dict(metadata)
        changed = str(metadata.get("duration_policy", "")) == "short_0p5"
        if changed:
            vector = self._adapt_pt_subblock(vector)
        else:
            # Deliberately preserve joint bit-for-bit outside the 0.5 s route.
            vector = np.asarray(vector)
            if vector.shape != (JOINT_DIM,) or vector.dtype != np.float32:
                raise RuntimeError("short_pt unchanged joint vector contract mismatch")
        metadata.update(
            {
                "release_policy": (
                    "short_pt_short05_pt_subspace" if changed else "short_pt_exact_joint"
                ),
                "short_pt_changed_region": bool(changed),
                "short_pt_adapter_sha256": self.adapter_metadata["sha256"],
                "fallback_mode": "none",
                "fallback_reason": "",
                "output_dim": JOINT_DIM,
                "frontend_mode": "joint_strict_r3+duration_conditional_pt_subspace",
            }
        )
        return vector, metadata

