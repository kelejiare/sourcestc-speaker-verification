#!/usr/bin/env python3
"""plus: exact duration_metric anchor plus a route-guarded ReDimNet2+ robustness block."""

import hashlib
import hmac
import math
from pathlib import Path

import numpy as np
import torch

from audio_features import load_audio
from redimnet2.redimnet2 import ReDimNet2Wrap
from speaker_inference import l2_normalize
from source_kd_inference import source_policy_for_path


DURATION_METRIC_DIM = 3328
PLUS_DIM = 192
OUTPUT_DIM = DURATION_METRIC_DIM + PLUS_DIM
PLUS_SHA256 = "61fdc3835b71a46ee0440d8a75f61039adafd97aef2f85146aed21f65991a369"

# Deployment vector coefficients.  The induced cross-duration cosine score is
# exactly sqrt((1-a_enroll)(1-a_query))*s_duration_metric + sqrt(a_enroll*a_query)*s_plus.
ALPHA_FULL = 0.03
ALPHA_SHORT05 = 0.01
ALPHA_SHORT10 = 0.05
MIN_PLUS_SAMPLES = 6400

B6_STAGES = [
    ((1, 1), 3, 3, [(3, 3)], 64),
    ((2, 1), 4, 2, [(3, 3)], 64),
    ((1, 2), 5, 2, [(3, 3)], 48),
    ((2, 1), 5, 1, [(3, 3)], 48),
    ((1, 2), 4, 0.75, [(3, 3)], 32),
    ((2, 1), 3, 0.5, [(3, 3)], 24),
]
MODEL_CONFIG = {
    "F": 72,
    "C": 64,
    "out_channels": 224,
    "return_2d_output": True,
    "stages_setup": B6_STAGES,
    "causal": "none",
    "feat_type": "tf",
}
CHECKPOINT_CONFIG = dict(MODEL_CONFIG, embed_dim=PLUS_DIM)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_redimnet2plus(path, device):
    path = Path(path)
    observed = sha256_file(path)
    if not hmac.compare_digest(observed, PLUS_SHA256):
        raise RuntimeError("plus ReDimNet2+ checkpoint SHA-256 mismatch")
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"model_config", "state_dict"}:
        raise RuntimeError("plus ReDimNet2+ checkpoint schema mismatch")
    if payload["model_config"] != CHECKPOINT_CONFIG:
        raise RuntimeError("plus ReDimNet2+ model config mismatch")
    model = ReDimNet2Wrap(**MODEL_CONFIG)
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("plus ReDimNet2+ strict-load mismatch")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "sha256": observed,
        "tensor_count": len(payload["state_dict"]),
        "parameter_count": sum(
            int(parameter.numel()) for parameter in model.parameters()
        ),
    }


class PlusFeatureFusionExtractor:
    output_dim = OUTPUT_DIM

    def __init__(self, duration_metric_extractor, plus_model_path):
        if int(getattr(duration_metric_extractor, "output_dim", -1)) != DURATION_METRIC_DIM:
            raise RuntimeError("plus requires the exact 3328-D duration_metric extractor")
        self.duration_metric = duration_metric_extractor
        self.device = duration_metric_extractor.joint.joint.device
        self.plus, self.plus_metadata = load_redimnet2plus(
            plus_model_path, self.device
        )

    @staticmethod
    def _alpha(metadata):
        policy = str(metadata.get("duration_policy", ""))
        if policy == "short_0p5":
            return ALPHA_SHORT05
        if policy == "short_1p0":
            return ALPHA_SHORT10
        return ALPHA_FULL

    def _plus_embedding(self, waveform):
        tensor = torch.from_numpy(
            np.ascontiguousarray(waveform, dtype=np.float32)
        ).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            value = self.plus(tensor).float().cpu().numpy()[0]
        return l2_normalize(value, "plus ReDimNet2+ embedding")

    def extract_file(self, wav_path):
        anchor, metadata = self.duration_metric.extract_file(wav_path)
        anchor = l2_normalize(
            np.asarray(anchor, dtype=np.float32), "plus duration_metric anchor"
        )
        metadata = dict(metadata)
        source_policy = source_policy_for_path(wav_path)
        alpha = self._alpha(metadata)
        plus = np.zeros((PLUS_DIM,), dtype=np.float32)
        auxiliary_mode = "route_bypass"
        auxiliary_reason = ""

        if source_policy == "data_scenery2_joint" and alpha > 0.0:
            try:
                waveform = load_audio(Path(wav_path), 16000, remove_dc=False)
                if waveform.size >= MIN_PLUS_SAMPLES:
                    plus = self._plus_embedding(waveform)
                    auxiliary_mode = "redimnet2plus"
                else:
                    alpha = 0.0
                    auxiliary_mode = "ultrashort_bypass"
                    auxiliary_reason = "waveform shorter than {} samples".format(
                        MIN_PLUS_SAMPLES
                    )
            except Exception as exc:
                # The online-proven duration_metric anchor remains a valid complete speaker
                # embedding.  A local failure in the auxiliary block degrades to
                # the anchor for this file instead of invalidating submit.zip.
                alpha = 0.0
                plus.fill(0.0)
                auxiliary_mode = "file_safe_bypass"
                auxiliary_reason = "{}: {}".format(type(exc).__name__, exc)

        output = np.concatenate(
            [math.sqrt(1.0 - alpha) * l2_normalize(anchor, "plus duration_metric anchor"),
             math.sqrt(alpha) * plus]
        ).astype(np.float32)
        output = l2_normalize(output, "plus output")
        if output.shape != (OUTPUT_DIM,) or not np.isfinite(output).all():
            raise RuntimeError("plus output contract mismatch")
        metadata.update({
            "release_policy": "plus_duration_metric_anchored_redimnet2plus",
            "source_policy": source_policy,
            "plus_alpha": float(alpha),
            "plus_plus_sha256": self.plus_metadata["sha256"],
            "auxiliary_mode": auxiliary_mode,
            "auxiliary_reason": auxiliary_reason,
            "output_dim": OUTPUT_DIM,
            "frontend_mode": "duration_metric_exact+route_guarded_redimnet2plus_pure_torch",
        })
        return output, metadata
