#!/usr/bin/env python3
"""Duration-aware w2v-BERT features fused with an acoustic embedding.

The active source policy supplies the fusion coefficients. The merged LoRA
parameters are loaded once; CUDA failures retry the same teacher on CPU."""

import collections
import gc
import math
from pathlib import Path

import numpy as np
import torch

from audio_features import load_audio
from speaker_inference import l2_normalize
from source_kd_inference import source_policy_for_path
from teacher_template_inference import ANCHOR_DIM
from native_w2vbert2 import NativeFeatures, load_native_teacher
from lora_duration_model import apply_merged_patch


TEACHER_DIM = 256
OUTPUT_DIM = ANCHOR_DIM + TEACHER_DIM
PATCH_SHA256 = "5c3dcb32bd6c2f3066d72e6aa9daf6666b0d583b7456a8cfcbeb3b4782733a78"

# These are cosine-score masses because each block is independently unit
# normalized before concatenation.  They were fixed after the alpha sweep on
# dev/near-far proxies and are deliberately below the unstable high-alpha
# region.  The short condition receives the largest student mass.
FULL_ALPHA = 0.15
SHORT05_ALPHA = 0.25
SHORT10_ALPHA = 0.20


class LoRATeacherFusionExtractor:
    output_dim = OUTPUT_DIM

    def __init__(self, anchor, checkpoint_path, config_dir, patch_path,
                 device="cpu", expected_patch_sha256=PATCH_SHA256):
        if int(getattr(anchor, "output_dim", -1)) != ANCHOR_DIM:
            raise RuntimeError("teacher_fusion requires the exact 3520-D plus anchor")
        self.anchor = anchor
        self.device = "cpu"
        self.feature_extractor = NativeFeatures()
        self.teacher = load_native_teacher(
            Path(checkpoint_path), Path(config_dir) / "config.json"
        )
        # Load the parameter patch expected by this model.
        if not isinstance(expected_patch_sha256, str) or len(expected_patch_sha256) != 64:
            raise RuntimeError("teacher_fusion expected patch checksum is invalid")
        self.expected_patch_sha256 = expected_patch_sha256
        self.patch = apply_merged_patch(
            self.teacher, Path(patch_path), self.expected_patch_sha256
        )
        self.counts = collections.Counter()
        self.retries = []
        self.patch_path = str(Path(patch_path).resolve())
        requested = str(device)
        if requested.startswith("cuda"):
            try:
                self.teacher.to(requested)
                self.device = requested
            except RuntimeError as exc:
                self._recover_cpu("teacher initialization", exc)
        self.teacher.eval()
        # Check the output shape after loading the parameter patch.
        probe = np.zeros(16000, dtype=np.float32)
        value = self._forward([probe])[0]
        if value.shape != (TEACHER_DIM,) or not np.isfinite(value).all():
            raise RuntimeError("teacher_fusion patched teacher cold-start contract failed")
        self.counts["cold_start_probe"] += 1

    def _recover_cpu(self, stage, exc):
        self.teacher.cpu()
        self.device = "cpu"
        gc.collect()
        reason = "{}: {}: {}".format(stage, type(exc).__name__, exc)
        self.retries.append(reason)
        self.counts["same_model_cpu_retries"] += 1

    def _forward_once(self, waveforms):
        features = self.feature_extractor(waveforms)
        with torch.no_grad():
            values = self.teacher(features.to(self.device)).detach().float().cpu().numpy()
        if values.shape != (len(waveforms), TEACHER_DIM):
            raise RuntimeError("teacher_fusion teacher shape mismatch: {}".format(values.shape))
        if not np.isfinite(values).all():
            raise RuntimeError("teacher_fusion teacher produced NaN/Inf")
        return values

    def _forward(self, waveforms):
        try:
            return self._forward_once(waveforms)
        except RuntimeError as exc:
            if self.device == "cpu":
                raise
            self._recover_cpu("teacher forward", exc)
            return self._forward_once(waveforms)

    @staticmethod
    def _center_crop(waveform, samples):
        if waveform.size <= samples:
            return np.ascontiguousarray(waveform, dtype=np.float32)
        start = (waveform.size - samples) // 2
        return np.ascontiguousarray(waveform[start:start + samples], dtype=np.float32)

    def _student_embedding(self, waveform):
        values = self._forward([waveform])
        return l2_normalize(values[0].astype(np.float32), "teacher_fusion patched teacher")

    @staticmethod
    def _padded_anchor(anchor):
        output = np.concatenate((
            l2_normalize(anchor, "teacher_fusion plus anchor"),
            np.zeros(TEACHER_DIM, dtype=np.float32),
        )).astype(np.float32)
        return l2_normalize(output, "teacher_fusion padded anchor")

    @staticmethod
    def _compose(anchor, student, alpha):
        if not (0.0 < alpha < 1.0):
            raise RuntimeError("teacher_fusion invalid alpha")
        output = np.concatenate((
            np.float32(math.sqrt(1.0 - alpha)) *
            l2_normalize(anchor, "teacher_fusion anchor block"),
            np.float32(math.sqrt(alpha)) *
            l2_normalize(student, "teacher_fusion student block"),
        )).astype(np.float32)
        output = l2_normalize(output, "teacher_fusion output")
        if output.shape != (OUTPUT_DIM,) or not np.isfinite(output).all():
            raise RuntimeError("teacher_fusion output contract mismatch")
        return output

    def extract_file(self, wav_path):
        anchor, metadata = self.anchor.extract_file(wav_path)
        metadata = dict(metadata)
        route = source_policy_for_path(wav_path)
        duration_policy = str(metadata.get("duration_policy", ""))
        self.counts["files"] += 1

        # The plus anchor is still the safe contract for an unknown route.  The
        # official two-source tree is routed through the same patched teacher;
        # this avoids a source-name heuristic affecting vector dimensions.
        if route not in ("data_scenery1_mobile", "data_scenery2_joint"):
            output = self._padded_anchor(anchor)
            policy = "anchor_only_unknown_route"
            alpha = 0.0
            active = False
        else:
            waveform = load_audio(Path(wav_path), 16000, remove_dc=False)
            active = True
            if duration_policy == "short_0p5":
                student_wave = self._center_crop(waveform, 8000)
                alpha = SHORT05_ALPHA
                policy = "{}+short05".format(route)
            elif duration_policy == "short_1p0":
                student_wave = self._center_crop(waveform, 16000)
                alpha = SHORT10_ALPHA
                policy = "{}+short10".format(route)
            elif waveform.size >= 16000:
                # Three seconds matches the lora training/cache contract while
                # bounding quadratic attention memory on long recordings.
                student_wave = self._center_crop(waveform, 48000)
                alpha = FULL_ALPHA
                policy = "{}+full3".format(route)
            else:
                # The platform's 0.18-second smoke case has too few reliable
                # spectral frames for the teacher.  Keep the exact plus block
                # and publish a valid zero-padded vector instead of failing
                # the whole ZIP.  This is an explicit duration policy, not a
                # teacher-error substitution.
                output = self._padded_anchor(anchor)
                alpha = 0.0
                policy = "{}+subsecond_anchor".format(route)
                active = False
            if active:
                student = self._student_embedding(student_wave)
                output = self._compose(anchor, student, alpha)
                self.counts["teacher_active_files"] += 1

        metadata.update({
            "release_policy": "teacher_fusion_plus_anchored_lora_duration",
            "teacher_fusion_route": route,
            "teacher_fusion_policy": policy,
            "teacher_fusion_alpha": float(alpha),
            "teacher_fusion_teacher_active": active,
            "teacher_fusion_patch_sha256": self.expected_patch_sha256,
            "teacher_fusion_teacher_device": self.device,
            "teacher_fusion_cpu_retries": len(self.retries),
            "fallback_mode": "none",
            "output_dim": OUTPUT_DIM,
        })
        return output, metadata

    def audit_summary(self):
        return {
            "counts": dict(self.counts),
            "teacher_device": self.device,
            "same_model_cpu_retries": list(self.retries),
            "patch_sha256": self.expected_patch_sha256,
            "output_dim": OUTPUT_DIM,
            "alphas": {"full3": FULL_ALPHA, "short05": SHORT05_ALPHA,
                        "short10": SHORT10_ALPHA},
            "unknown_route_policy": "exact_plus_zero_padded",
            "per_file_teacher_failure_substitution": False,
        }


if OUTPUT_DIM > 4096:
    raise RuntimeError("teacher_fusion output dimension exceeds platform limit")
