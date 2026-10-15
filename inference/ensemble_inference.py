#!/usr/bin/env python3
"""Hermetic duration-adaptive VoxBlink2 PT + CAM++ embedding inference."""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from audio_features import FbankConfig, extract_kaldi_fbank_torch, load_audio
from campplus_model import load_campplus_model
from speaker_inference import SpeakerEmbeddingExtractor, l2_normalize


class DurationAdaptiveEnsembleExtractor:
    """Emit fixed 448-D embeddings with challenge-aware block scaling.

    The platform compares embeddings with cosine similarity. Concatenating
    unit branch vectors after sqrt(weight) scaling makes their cosine equal a
    weighted branch-score fusion. Enrollment/full utterances, 0.5 s crops,
    and 1.0 s crops use weights selected on the archived 15-condition proxy.
    """

    output_dim = 448
    pt_dim = 256
    campplus_dim = 192
    long_pt_weight = 0.78
    short_0p5_pt_weight = 0.5376047904191618
    short_1p0_pt_weight = 1.0

    def __init__(
        self,
        pt_model_path: Path,
        campplus_model_path: Path,
        device_name: str = "auto",
        max_chunk_seconds: float = 60.0,
        chunk_hop_seconds: float = 30.0,
    ):
        self.pt_extractor = SpeakerEmbeddingExtractor(
            model_path=pt_model_path,
            device_name=device_name,
            fbank_backend="auto",
            embedding_mode="single_full",
            frontend_mode="voxblink2_official",
            max_chunk_seconds=max_chunk_seconds,
            chunk_hop_seconds=chunk_hop_seconds,
        )
        self.device = self.pt_extractor.device
        self.fbank_config = FbankConfig()
        self.campplus_model, self.campplus_metadata = load_campplus_model(
            campplus_model_path, self.device
        )
        self.pt_metadata = self.pt_extractor.model_metadata
        self.max_chunk_seconds = max_chunk_seconds

    @classmethod
    def duration_policy(cls, duration_seconds: float) -> Tuple[str, float]:
        # Official random crops are exactly 0.5 s and 1.0 s. Narrow windows
        # avoid reclassifying naturally short full-scene speech.
        if 0.35 <= duration_seconds <= 0.65:
            return "short_0p5", cls.short_0p5_pt_weight
        if 0.85 <= duration_seconds <= 1.15:
            return "short_1p0", cls.short_1p0_pt_weight
        return "full", cls.long_pt_weight

    def _campplus_segments(self, waveform: np.ndarray):
        maximum = max(
            1,
            int(round(self.max_chunk_seconds * self.fbank_config.sample_rate)),
        )
        if waveform.size <= maximum:
            return [waveform]
        return [
            waveform[start : start + maximum]
            for start in range(0, waveform.size, maximum)
            if waveform[start : start + maximum].size > 0
        ]

    def _campplus_embedding(self, waveform: np.ndarray) -> Tuple[np.ndarray, list]:
        vectors = []
        frame_counts = []
        with torch.inference_mode():
            for segment in self._campplus_segments(waveform):
                features, _ = extract_kaldi_fbank_torch(
                    segment, self.fbank_config
                )
                frame_counts.append(int(features.shape[0]))
                tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
                vector = (
                    self.campplus_model(tensor)
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )
                vectors.append(l2_normalize(vector, "CAM++ 分支"))
        aggregate = np.mean(np.stack(vectors), axis=0)
        return l2_normalize(aggregate, "CAM++ 聚合"), frame_counts

    @staticmethod
    def _fuse(
        pt_embedding: np.ndarray,
        campplus_embedding: np.ndarray,
        pt_weight: float,
    ) -> np.ndarray:
        pt_embedding = l2_normalize(pt_embedding, "VoxBlink2 PT 分支")
        campplus_embedding = l2_normalize(
            campplus_embedding, "CAM++ 分支"
        )
        fused = np.concatenate(
            (
                math.sqrt(pt_weight) * pt_embedding,
                math.sqrt(1.0 - pt_weight) * campplus_embedding,
            )
        ).astype(np.float32)
        return l2_normalize(fused, "PT+CAM++ 融合")

    def extract_file(self, wav_path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
        waveform = load_audio(
            wav_path,
            self.fbank_config.sample_rate,
            remove_dc=False,
        )
        duration_seconds = float(
            waveform.size / self.fbank_config.sample_rate
        )
        policy, pt_weight = self.duration_policy(duration_seconds)
        pt_embedding, pt_frames = self.pt_extractor._embed_segments(
            self.pt_extractor._segments(waveform), "VoxBlink2 PT 分支"
        )

        campplus_frames = []
        if pt_weight >= 1.0:
            campplus_embedding = np.zeros(
                self.campplus_dim, dtype=np.float32
            )
            embedding = np.concatenate(
                (pt_embedding, campplus_embedding)
            ).astype(np.float32)
        else:
            campplus_embedding, campplus_frames = self._campplus_embedding(
                waveform
            )
            embedding = self._fuse(
                pt_embedding, campplus_embedding, pt_weight
            )
        embedding = l2_normalize(embedding, "448 维 embedding")
        metadata = {
            "duration_seconds": duration_seconds,
            "duration_policy": policy,
            "pt_weight": pt_weight,
            "pt_frame_counts": pt_frames,
            "campplus_frame_counts": campplus_frames,
            "frontend_mode": "voxblink2_official+kaldi_fbank_torch_exact",
            "fbank_backend": "pure_torch",
            "output_dim": self.output_dim,
        }
        return embedding, metadata
