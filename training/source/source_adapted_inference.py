#!/usr/bin/env python3
"""Source-aware joint acoustic adaptation for mobile and meeting audio."""

from pathlib import Path

import numpy as np

from audio_features import load_audio
from multidomain_kd_inference import _extract_acoustic_waveform
from source_kd_inference import (
    SourceKDExtractor,
    source_policy_for_path,
)


# Independent runner-up from the frozen mobile weight audit.  It sacrifices a
# small amount of child-heavy proxy gain for more LM weight and better
# cross-device robustness.  The mobile adapter itself is unchanged.
TARGET_ENROLL_WEIGHTS = np.asarray((0.50, 0.40, 0.10), dtype=np.float64)
TARGET_SHORT05_WEIGHTS = np.asarray((0.20, 0.56, 0.24), dtype=np.float64)
TARGET_SHORT10_WEIGHTS = np.asarray((0.32, 0.578, 0.102), dtype=np.float64)


class SourceAdaptedExtractor(SourceKDExtractor):
    """Distinct target-line candidate; never performs an explicit fallback."""

    def extract_file(self, wav_path):
        source_policy = source_policy_for_path(wav_path)
        if source_policy != "data_scenery1_mobile":
            embedding, metadata = self.joint.extract_file(wav_path)
            metadata = dict(metadata)
            metadata.update(
                {
                    "source_policy": source_policy,
                    "release_policy": "exact_joint_meeting_policy",
                }
            )
            return embedding, metadata

        waveform = load_audio(
            Path(wav_path), self.joint.fbank_config.sample_rate, remove_dc=False
        )
        duration_seconds = float(waveform.size) / float(
            self.joint.fbank_config.sample_rate
        )
        policy, pt_weight = self.joint.acoustic_extractor.duration_policy(duration_seconds)
        acoustic, pt_frames, camp_frames = _extract_acoustic_waveform(
            self.joint.acoustic_extractor, waveform, pt_weight
        )
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
            weights = TARGET_SHORT05_WEIGHTS
            adapter_policy = "target_mobile_short05"
        elif policy == "short_1p0":
            lm = self.joint._adapt(lm, "lm", "short10")
            ptn = self.joint._adapt(ptn, "ptn", "short10")
            weights = TARGET_SHORT10_WEIGHTS
            adapter_policy = "joint_short10_target_weights"
        else:
            weights = TARGET_ENROLL_WEIGHTS
            adapter_policy = "full_bypass_target_weights"
        embedding = self.joint._fuse(acoustic, lm, ptn, weights)
        return embedding, {
            "duration_seconds": duration_seconds,
            "duration_policy": policy,
            "adapter_policy": adapter_policy,
            "source_policy": source_policy,
            "release_policy": "joint_target_strong_mobile",
            "branch_weights": tuple(float(item) for item in weights),
            "pt_frame_counts": pt_frames,
            "campplus_frame_counts": camp_frames,
            "lm_sample_counts": lm_samples,
            "ptn_sample_counts": ptn_samples,
            "fallback_mode": "none",
            "fallback_reason": "",
            "output_dim": self.output_dim,
            "frontend_mode": "acoustic_exact_pure_torch+joint_target_mobile_dual_redimnet2_kd",
        }


def assert_target_contract():
    for name, weights in (
        ("enroll", TARGET_ENROLL_WEIGHTS),
        ("short05", TARGET_SHORT05_WEIGHTS),
        ("short10", TARGET_SHORT10_WEIGHTS),
    ):
        if weights.shape != (3,) or np.any(weights < 0.0):
            raise RuntimeError("{} target weights invalid".format(name))
        if abs(float(weights.sum()) - 1.0) > 1.0e-12:
            raise RuntimeError("{} target weights do not sum to one".format(name))


assert_target_contract()
