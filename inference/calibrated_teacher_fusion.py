#!/usr/bin/env python3
"""Calibrated teacher coefficients for mobile and meeting audio.

The meeting short-utterance coefficient accounts for the square-root weights
on both sides of cosine matching."""

import source_duration_policy as _source_policy


SOURCE1_POLICY = dict(_source_policy.SOURCE1_POLICY)
SOURCE2_POLICY = {
    "full3": 0.60,
    "short05": 0.1090909090909091,
    "short10": 0.70,
}


class CalibratedTeacherFusionExtractor(
        _source_policy.SourceDurationPolicyExtractor):
    """source_policy with a single, auditable source2 short-duration change."""

    def extract_file(self, wav_path):
        old_source1 = _source_policy.SOURCE1_POLICY
        old_source2 = _source_policy.SOURCE2_POLICY
        _source_policy.SOURCE1_POLICY = SOURCE1_POLICY
        _source_policy.SOURCE2_POLICY = SOURCE2_POLICY
        try:
            output, metadata = super().extract_file(wav_path)
        finally:
            _source_policy.SOURCE1_POLICY = old_source1
            _source_policy.SOURCE2_POLICY = old_source2
        metadata = dict(metadata)
        metadata.update({
            "release_policy": "calibrated_fusion_source_policy_anchored_source2_short05_calibrated",
            "calibrated_fusion_source1_policy": dict(SOURCE1_POLICY),
            "calibrated_fusion_source2_policy": dict(SOURCE2_POLICY),
            "calibrated_fusion_change_scope": "source2 short05 coefficient only",
        })
        return output, metadata

    def audit_summary(self):
        result = super().audit_summary()
        result.update({
            "release": "calibrated_fusion-source_policy-anchored-source2-short05-calibrated",
            "source1_policy": dict(SOURCE1_POLICY),
            "source2_policy": dict(SOURCE2_POLICY),
            "change_scope": "one coefficient; all source_policy models and fallback paths retained",
            "calibration_status": "exploratory_proxy_only",
            "hidden_score_guarantee": False,
        })
        return result


if CalibratedTeacherFusionExtractor.output_dim > 4096:
    raise RuntimeError("calibrated_fusion output dimension exceeds platform limit")
