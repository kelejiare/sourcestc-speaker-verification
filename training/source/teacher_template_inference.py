#!/usr/bin/env python3
"""teacher release candidate: exact plus plus two duration-matched teacher blocks."""

import math
from pathlib import Path

import numpy as np

from audio_features import load_audio
from speaker_inference import l2_normalize
from source_kd_inference import source_policy_for_path
from w2vbert2_multidomain_model import (
    build_feature_extractor,
    extract_waveforms,
    load_teacher,
)


ANCHOR_DIM = 3520
TEACHER_DIM = 256
OUTPUT_DIM = 4032

# Development ratios were frozen before the independent ChildMandarin-test
# teacher cache was generated.  The masses below exactly realize the additive
# cosine scores documented in POLICY_PRECOMMIT.json.
FULL_MASSES = {
    "anchor": 5.0 / 6.0,
    "template05": 1.0 / 24.0,
    "template10": 1.0 / 8.0,
}
SHORT05_MASSES = {
    "anchor": 5.0 / 9.0,
    "template05": 4.0 / 9.0,
    "template10": 0.0,
}
SHORT10_MASSES = {
    "anchor": 5.0 / 8.0,
    "template05": 0.0,
    "template10": 3.0 / 8.0,
}


def _crop_start(sample_count, crop_count, view_index, view_count=2):
    if crop_count <= 0 or sample_count < crop_count:
        raise ValueError("audio is shorter than the requested template")
    maximum = sample_count - crop_count
    if maximum == 0 or view_count == 1:
        return maximum // 2
    return int(round(float(view_index) * float(maximum) / float(view_count - 1)))


class TeacherTemplateExtractor:
    output_dim = OUTPUT_DIM

    def __init__(self, plus_extractor, checkpoint_path, config_dir, device="cpu"):
        if int(getattr(plus_extractor, "output_dim", -1)) != ANCHOR_DIM:
            raise RuntimeError("teacher requires the exact 3520-D plus extractor")
        self.anchor = plus_extractor
        self.device = str(device)
        self.teacher, self.teacher_metadata = load_teacher(
            Path(checkpoint_path), Path(config_dir) / "config.json", self.device
        )
        self.feature_extractor = build_feature_extractor(Path(config_dir))

    @staticmethod
    def _unit(value, label):
        return l2_normalize(np.asarray(value, dtype=np.float32), label)

    @staticmethod
    def _template_views(waveform, samples):
        output = []
        for view in (0, 1):
            start = _crop_start(waveform.size, samples, view)
            output.append(np.ascontiguousarray(
                waveform[start:start + samples], dtype=np.float32
            ))
        return output

    def _teacher_batch(self, waveforms):
        values = extract_waveforms(
            self.teacher, self.feature_extractor, waveforms, self.device
        )
        return [self._unit(value, "teacher teacher") for value in values]

    def _template(self, waveform, samples):
        values = self._teacher_batch(self._template_views(waveform, samples))
        return self._unit(
            np.mean(np.stack(values, axis=0).astype(np.float64), axis=0),
            "teacher teacher template",
        )

    @staticmethod
    def _padded_anchor(anchor):
        output = np.concatenate((
            l2_normalize(anchor, "teacher plus fallback"),
            np.zeros(2 * TEACHER_DIM, dtype=np.float32),
        )).astype(np.float32)
        return l2_normalize(output, "teacher exact-plus padded output")

    @staticmethod
    def _compose(anchor, masses, template05=None, template10=None):
        zero = np.zeros(TEACHER_DIM, dtype=np.float32)
        block05 = (
            zero if masses["template05"] == 0.0
            else math.sqrt(masses["template05"]) * l2_normalize(
                template05, "teacher teacher 05"
            )
        )
        block10 = (
            zero if masses["template10"] == 0.0
            else math.sqrt(masses["template10"]) * l2_normalize(
                template10, "teacher teacher 10"
            )
        )
        output = np.concatenate((
            math.sqrt(masses["anchor"]) * l2_normalize(
                anchor, "teacher plus anchor"
            ),
            block05,
            block10,
        )).astype(np.float32)
        output = l2_normalize(output, "teacher 4032-D output")
        if output.shape != (OUTPUT_DIM,) or not np.isfinite(output).all():
            raise RuntimeError("teacher output contract mismatch")
        return output

    def extract_file(self, wav_path):
        anchor, metadata = self.anchor.extract_file(wav_path)
        metadata = dict(metadata)
        source_policy = source_policy_for_path(wav_path)
        duration_policy = str(metadata.get("duration_policy", ""))
        if source_policy != "data_scenery1_mobile":
            output = self._padded_anchor(anchor)
            metadata.update({
                "teacher_policy": "exact_plus_zero_padded",
                "teacher_teacher_fallback": False,
                "output_dim": OUTPUT_DIM,
            })
            return output, metadata

        try:
            waveform = load_audio(Path(wav_path), 16000, remove_dc=False)
            if duration_policy == "short_0p5":
                teacher05 = self._teacher_batch([waveform])[0]
                output = self._compose(
                    anchor, SHORT05_MASSES, template05=teacher05
                )
            elif duration_policy == "short_1p0":
                teacher10 = self._teacher_batch([waveform])[0]
                output = self._compose(
                    anchor, SHORT10_MASSES, template10=teacher10
                )
            elif waveform.size >= 16000:
                teacher05 = self._template(waveform, 8000)
                teacher10 = self._template(waveform, 16000)
                output = self._compose(
                    anchor, FULL_MASSES,
                    template05=teacher05, template10=teacher10,
                )
                duration_policy = "full"
            else:
                raise RuntimeError("non-official sub-one-second duration")
            fallback = False
            fallback_reason = ""
        except Exception as exc:
            output = self._padded_anchor(anchor)
            fallback = True
            fallback_reason = "{}: {}".format(type(exc).__name__, exc)

        metadata.update({
            "teacher_policy": (
                "source1_" + duration_policy
                if not fallback else "exact_plus_file_fallback"
            ),
            "teacher_teacher_fallback": fallback,
            "teacher_teacher_fallback_reason": fallback_reason,
            "teacher_teacher_checkpoint_sha256": self.teacher_metadata[
                "checkpoint_sha256"
            ],
            "output_dim": OUTPUT_DIM,
        })
        return output, metadata


class PaddedPlusFallbackExtractor:
    """Keep a format-valid exact-plus result if the teacher cannot start."""

    output_dim = OUTPUT_DIM

    def __init__(self, plus_extractor, reason):
        self.anchor = plus_extractor
        self.reason = str(reason)

    def extract_file(self, wav_path):
        vector, metadata = self.anchor.extract_file(wav_path)
        output = TeacherTemplateExtractor._padded_anchor(vector)
        metadata = dict(metadata)
        metadata.update({
            "teacher_policy": "exact_plus_startup_fallback",
            "teacher_teacher_fallback": True,
            "teacher_teacher_fallback_reason": self.reason,
            "output_dim": OUTPUT_DIM,
        })
        return output, metadata


for _masses in (FULL_MASSES, SHORT05_MASSES, SHORT10_MASSES):
    if abs(sum(_masses.values()) - 1.0) > 1.0e-12:
        raise RuntimeError("teacher energy masses do not sum to one")
if OUTPUT_DIM != ANCHOR_DIM + 2 * TEACHER_DIM or OUTPUT_DIM > 4096:
    raise RuntimeError("teacher output dimension contract mismatch")
