#!/usr/bin/env python3
"""Source-specific duration policies for acoustic and teacher feature fusion.

The wrapper selects coefficients for mobile and meeting audio, then restores
the module settings after each extraction call."""

import lora_teacher_fusion as _teacher_fusion
from source_kd_inference import source_policy_for_path


# Frozen policy.  These values are deliberately explicit so the package can be
# audited without hidden environment variables or score-time tuning.
SOURCE1_POLICY = {"full3": 0.45, "short05": 0.32, "short10": 0.25}
SOURCE2_POLICY = {"full3": 0.60, "short05": 0.30, "short10": 0.70}


_teacher_fusion.FULL_ALPHA = SOURCE1_POLICY["full3"]
_teacher_fusion.SHORT05_ALPHA = SOURCE1_POLICY["short05"]
_teacher_fusion.SHORT10_ALPHA = SOURCE1_POLICY["short10"]


class SourceDurationPolicyExtractor(
        _teacher_fusion.LoRATeacherFusionExtractor):
    """teacher_fusion runtime with the frozen source-aware duration policy."""

    def extract_file(self, wav_path):
        route = source_policy_for_path(wav_path)
        selected = SOURCE2_POLICY if route == "data_scenery2_joint" else SOURCE1_POLICY
        old = (_teacher_fusion.FULL_ALPHA, _teacher_fusion.SHORT05_ALPHA, _teacher_fusion.SHORT10_ALPHA)
        _teacher_fusion.FULL_ALPHA = selected["full3"]
        _teacher_fusion.SHORT05_ALPHA = selected["short05"]
        _teacher_fusion.SHORT10_ALPHA = selected["short10"]
        try:
            output, metadata = super().extract_file(wav_path)
        finally:
            _teacher_fusion.FULL_ALPHA, _teacher_fusion.SHORT05_ALPHA, _teacher_fusion.SHORT10_ALPHA = old
        metadata = dict(metadata)
        metadata.update({
            "release_policy": "source_policy_source_optimized_duration",
            "source_policy_source_policy": route,
            "source_policy_selected_alphas": dict(selected),
        })
        return output, metadata

    def audit_summary(self):
        result = super().audit_summary()
        result.update({
            "release": "source_policy-policy_tuning-source-optimized",
            "policy_selection": "frozen after dev+nearfar source-aware proxy audit; confirmation posthoc",
            "source1_policy": dict(SOURCE1_POLICY),
            "source2_policy": dict(SOURCE2_POLICY),
        })
        return result


if SourceDurationPolicyExtractor.output_dim > 4096:
    raise RuntimeError("source_policy output dimension exceeds platform limit")
