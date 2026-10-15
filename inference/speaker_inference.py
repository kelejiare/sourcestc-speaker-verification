#!/usr/bin/env python3
"""High-level speaker embedding extraction and controlled legacy ablations."""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from audio_features import (
    FbankConfig,
    ensure_minimum_duration,
    extract_fbank,
    extract_voxblink2_logmel,
    load_audio,
)
from speaker_model import choose_device, load_model


EMBEDDING_MODES = ("robust_multiview", "single_full")
FRONTEND_MODES = ("voxblink2_official", "kaldi_fbank")


def l2_normalize(vector: np.ndarray, description: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise RuntimeError("{} embedding 非法或为零范数".format(description))
    return (vector / norm).astype(np.float32)


def fixed_duration_views(
    waveform: np.ndarray, target_samples: int
) -> Tuple[List[np.ndarray], List[int]]:
    """Return one repeated short view or start/centre/end fixed-size views."""

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform 必须是一维且非空")
    if target_samples <= 0:
        raise ValueError("target_samples 必须大于 0")
    if waveform.size < target_samples:
        repeat_count = int(math.ceil(target_samples / waveform.size))
        repeated = np.tile(waveform, repeat_count)[:target_samples]
        return [repeated.astype(np.float32, copy=False)], [0]
    if waveform.size == target_samples:
        return [waveform], [0]

    candidate_starts = (
        0,
        (waveform.size - target_samples) // 2,
        waveform.size - target_samples,
    )
    starts = list(dict.fromkeys(candidate_starts))
    views = [
        waveform[start : start + target_samples].astype(np.float32, copy=False)
        for start in starts
    ]
    return views, starts


def weighted_branch_concatenation(
    full_embedding: np.ndarray,
    duration_embedding: np.ndarray,
    full_weight: float,
) -> np.ndarray:
    """Fuse unit branch vectors while preserving an interpretable cosine mix."""

    if not 0.0 < full_weight < 1.0:
        raise ValueError("full_weight 必须位于 (0, 1)")
    full_embedding = l2_normalize(full_embedding, "整段分支")
    duration_embedding = l2_normalize(duration_embedding, "时长分支")
    if full_embedding.shape != duration_embedding.shape:
        raise ValueError("两个分支的 embedding 形状必须一致")
    duration_weight = 1.0 - full_weight
    fused = np.concatenate(
        (
            math.sqrt(full_weight) * full_embedding,
            math.sqrt(duration_weight) * duration_embedding,
        )
    )
    return l2_normalize(fused, "融合")


class SpeakerEmbeddingExtractor:
    def __init__(
        self,
        model_path: Path,
        device_name: str = "auto",
        fbank_backend: str = "auto",
        max_chunk_seconds: float = 60.0,
        chunk_hop_seconds: float = 30.0,
        embedding_mode: str = "robust_multiview",
        frontend_mode: str = "kaldi_fbank",
        duration_view_seconds: float = 2.0,
        full_view_weight: float = 0.75,
    ):
        if embedding_mode not in EMBEDDING_MODES:
            raise ValueError(
                "embedding_mode 必须是 {}，实际为 {}".format(
                    EMBEDDING_MODES, embedding_mode
                )
            )
        if frontend_mode not in FRONTEND_MODES:
            raise ValueError(
                "frontend_mode 必须是 {}，实际为 {}".format(
                    FRONTEND_MODES, frontend_mode
                )
            )
        if duration_view_seconds <= 0.0:
            raise ValueError("duration_view_seconds 必须大于 0")
        if not 0.0 < full_view_weight < 1.0:
            raise ValueError("full_view_weight 必须位于 (0, 1)")
        self.device = choose_device(device_name)
        self.fbank_backend = fbank_backend
        self.max_chunk_seconds = max_chunk_seconds
        self.chunk_hop_seconds = chunk_hop_seconds
        self.embedding_mode = embedding_mode
        self.frontend_mode = frontend_mode
        self.duration_view_seconds = duration_view_seconds
        self.full_view_weight = full_view_weight
        self.fbank_config = FbankConfig()
        self.model, self.model_metadata = load_model(model_path, self.device)
        self.last_backend = ""

    @property
    def output_dim(self) -> int:
        branch_dim = int(self.model_metadata["embedding_dim"])
        if self.embedding_mode == "robust_multiview":
            return 2 * branch_dim
        return branch_dim

    def _segments(self, waveform: np.ndarray) -> List[np.ndarray]:
        sample_rate = self.fbank_config.sample_rate
        waveform = ensure_minimum_duration(waveform, sample_rate)
        maximum = max(1, int(round(self.max_chunk_seconds * sample_rate)))
        hop = max(1, int(round(self.chunk_hop_seconds * sample_rate)))
        if waveform.size <= maximum:
            return [waveform]
        starts = list(range(0, waveform.size - maximum + 1, hop))
        final_start = waveform.size - maximum
        if starts[-1] != final_start:
            starts.append(final_start)
        return [waveform[start : start + maximum] for start in starts]

    def _embed_segments(
        self, segments: List[np.ndarray], description: str
    ) -> Tuple[np.ndarray, List[int]]:
        embeddings = []
        frame_counts = []
        with torch.inference_mode():
            for segment in segments:
                if self.frontend_mode == "voxblink2_official":
                    features, backend = extract_voxblink2_logmel(
                        segment, self.fbank_config
                    )
                else:
                    features, backend = extract_fbank(
                        segment, self.fbank_config, self.fbank_backend
                    )
                self.last_backend = backend
                frame_counts.append(int(features.shape[0]))
                tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
                embedding = self.model(tensor).squeeze(0).float().cpu().numpy()
                embeddings.append(l2_normalize(embedding, description))

        aggregate = np.mean(np.stack(embeddings, axis=0), axis=0)
        return l2_normalize(aggregate, description + "聚合"), frame_counts

    def extract_file(self, wav_path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
        waveform = load_audio(
            wav_path,
            self.fbank_config.sample_rate,
            remove_dc=(self.frontend_mode != "voxblink2_official"),
        )
        full_segments = self._segments(waveform)
        full_embedding, full_frame_counts = self._embed_segments(
            full_segments, "整段分支"
        )

        duration_frame_counts = []
        duration_view_starts = []
        if self.embedding_mode == "robust_multiview":
            target_samples = int(
                round(self.duration_view_seconds * self.fbank_config.sample_rate)
            )
            duration_views, duration_view_starts = fixed_duration_views(
                waveform, target_samples
            )
            duration_embedding, duration_frame_counts = self._embed_segments(
                duration_views, "2 秒分支"
            )
            aggregate = weighted_branch_concatenation(
                full_embedding, duration_embedding, self.full_view_weight
            )
        else:
            aggregate = full_embedding

        metadata = {
            "duration_seconds": float(waveform.size / self.fbank_config.sample_rate),
            "embedding_mode": self.embedding_mode,
            "frontend_mode": self.frontend_mode,
            "output_dim": self.output_dim,
            "full_segment_count": len(full_segments),
            "full_frame_counts": full_frame_counts,
            "duration_view_count": len(duration_frame_counts),
            "duration_frame_counts": duration_frame_counts,
            "duration_view_starts": duration_view_starts,
            "full_view_weight": self.full_view_weight,
            "fbank_backend": self.last_backend,
        }
        return aggregate, metadata
