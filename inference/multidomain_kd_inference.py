#!/usr/bin/env python3
"""Released distillation all-source output-KD speaker embedding extractor."""

import hashlib
import hmac
import math
from pathlib import Path

import numpy as np
import torch

from audio_features import FbankConfig, load_audio
from ensemble_inference import DurationAdaptiveEnsembleExtractor
from redimnet2.redimnet2 import ReDimNet2Wrap
from speaker_inference import l2_normalize


OUTPUT_DIM = 832
ACOUSTIC_DIM = 448
REDIM_DIM = 192
MODEL_SPECS = {
    "lm": {
        "sha256": "287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868",
        "tensor_count": 821, "parameter_count": 12459296,
    },
    "ptn": {
        "sha256": "295b968b39e846d8f4d4126bca51b1d409f765b90cc4478fdc5266ab5417a206",
        "tensor_count": 821, "parameter_count": 12459296,
    },
}
ADAPTER_SHA256 = "f4410766ed519228243656c16cd3a0612a8dae52fa7056859716a7c215ddb9e9"
ADAPTER_PAYLOAD_SHA256 = "494feed0bd329b612e8ba903a676bfaaf5d7d938dab97a8f3164181b2ee8aca5"
ADAPTER_DOMAIN = bytes.fromhex("67616d653832332d7631312d616c6c736f757263652d726564696d2d6f75747075742d6b642d763100")
ADAPTER_KEYS = (
    "schema_version", "release_status", "seed_ids", "lm_short05",
    "lm_short10", "ptn_short05", "ptn_short10", "payload_sha256",
)
ENROLL_WEIGHTS = np.asarray([0.86, 0.133, 0.007], dtype=np.float64)
SHORT_WEIGHTS = np.asarray([0.20, 0.76, 0.04], dtype=np.float64)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_redim(path, name, device):
    path = Path(path)
    specification = MODEL_SPECS[name]
    observed = sha256_file(path)
    if not hmac.compare_digest(observed, specification["sha256"]):
        raise RuntimeError("{} ReDimNet2 SHA-256 mismatch".format(name))
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"model_config", "state_dict"}:
        raise RuntimeError("{} ReDimNet2 payload schema mismatch".format(name))
    if len(payload["state_dict"]) != specification["tensor_count"]:
        raise RuntimeError("{} ReDimNet2 tensor count mismatch".format(name))
    if int(payload["model_config"].get("embed_dim", -1)) != REDIM_DIM:
        raise RuntimeError("{} ReDimNet2 embedding dimension mismatch".format(name))
    model = ReDimNet2Wrap(**payload["model_config"])
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("{} ReDimNet2 strict load failed".format(name))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != specification["parameter_count"]:
        raise RuntimeError("{} ReDimNet2 parameter count mismatch".format(name))
    model.to(device); model.eval()
    return model, {
        "sha256": observed, "state_tensor_count": len(payload["state_dict"]),
        "parameter_count": parameters, "embedding_dim": REDIM_DIM,
    }


def _payload_sha(schema_version, release_status, seeds, matrices):
    digest = hashlib.sha256()
    digest.update(ADAPTER_DOMAIN)
    digest.update(np.asarray(schema_version, dtype="<i8").tobytes())
    digest.update(release_status.encode("utf-8") + b"\x00")
    digest.update(np.ascontiguousarray(seeds, dtype="<i8").tobytes())
    for key in ("lm_short05", "lm_short10", "ptn_short05", "ptn_short10"):
        digest.update(key.encode("ascii") + b"\x00")
        digest.update(np.ascontiguousarray(matrices[key], dtype="<f4").tobytes())
    return digest.hexdigest()


def _load_adapter(path):
    path = Path(path)
    observed = sha256_file(path)
    if not hmac.compare_digest(observed, ADAPTER_SHA256):
        raise RuntimeError("distillation deployment adapter SHA-256 mismatch")
    with np.load(str(path), allow_pickle=False) as payload:
        if tuple(sorted(payload.files)) != tuple(sorted(ADAPTER_KEYS)):
            raise RuntimeError("distillation deployment adapter key schema mismatch")
        schema_version = np.asarray(payload["schema_version"])
        release_status = np.asarray(payload["release_status"])
        seeds = np.asarray(payload["seed_ids"])
        declared = np.asarray(payload["payload_sha256"])
        if schema_version.shape != () or schema_version.dtype != np.int64 or int(schema_version) != 1:
            raise RuntimeError("distillation adapter schema_version mismatch")
        if release_status.shape != () or release_status.dtype.kind != "U":
            raise RuntimeError("distillation adapter release_status type mismatch")
        release_status = str(release_status.item())
        if release_status != "passed_test_ali_final_confirmation":
            raise RuntimeError("distillation adapter was not final-confirmation released")
        if seeds.shape != (3,) or seeds.dtype != np.int64 or tuple(int(item) for item in seeds) != (825001, 825002, 825003):
            raise RuntimeError("distillation adapter seed contract mismatch")
        matrices = {}
        for key in ("lm_short05", "lm_short10", "ptn_short05", "ptn_short10"):
            value = np.asarray(payload[key])
            if value.shape != (REDIM_DIM, REDIM_DIM) or value.dtype != np.float32:
                raise RuntimeError("{} adapter matrix shape/dtype mismatch".format(key))
            if not np.isfinite(value).all():
                raise RuntimeError("{} adapter matrix contains NaN/Inf".format(key))
            matrices[key] = np.ascontiguousarray(value).copy()
        if declared.shape != () or declared.dtype.kind != "U":
            raise RuntimeError("distillation adapter payload digest type mismatch")
        declared = str(declared.item())
    actual_payload = _payload_sha(1, release_status, seeds, matrices)
    if not hmac.compare_digest(actual_payload, declared) or not hmac.compare_digest(actual_payload, ADAPTER_PAYLOAD_SHA256):
        raise RuntimeError("distillation adapter semantic payload digest mismatch")
    return matrices, {
        "sha256": observed, "payload_sha256": actual_payload,
        "seed_ids": tuple(int(item) for item in seeds),
        "release_status": release_status,
    }


def _extract_acoustic_waveform(extractor, waveform, pt_weight):
    pt, pt_frames = extractor.pt_extractor._embed_segments(
        extractor.pt_extractor._segments(waveform), "VoxBlink2 PT branch"
    )
    camp_frames = []
    if pt_weight >= 1.0:
        result = np.concatenate((pt, np.zeros(192, dtype=np.float32))).astype(np.float32)
    else:
        camp, camp_frames = extractor._campplus_embedding(waveform)
        result = extractor._fuse(pt, camp, pt_weight)
    return l2_normalize(result, "distillation exact acoustic block"), pt_frames, camp_frames


class MultidomainReDimKDExtractor:
    """Emit one finite unit-normalized 832-D distillation competition embedding."""

    output_dim = OUTPUT_DIM

    def __init__(self, pt_model_path, campplus_model_path, lm_model_path,
                 ptn_model_path, adapter_path, device_name="auto",
                 max_chunk_seconds=60.0, chunk_hop_seconds=30.0,
                 base_extractor=None):
        self.acoustic_extractor = base_extractor
        if self.acoustic_extractor is None:
            self.acoustic_extractor = DurationAdaptiveEnsembleExtractor(
                Path(pt_model_path), Path(campplus_model_path), device_name=device_name,
                max_chunk_seconds=max_chunk_seconds, chunk_hop_seconds=chunk_hop_seconds,
            )
        if int(self.acoustic_extractor.output_dim) != ACOUSTIC_DIM:
            raise RuntimeError("distillation base extractor must be exact 448-D acoustic")
        self.device = self.acoustic_extractor.device
        self.fbank_config = FbankConfig()
        self.redim_models = {}
        self.redim_metadata = {}
        for name, path in (("lm", lm_model_path), ("ptn", ptn_model_path)):
            self.redim_models[name], self.redim_metadata[name] = _load_redim(
                Path(path), name, self.device
            )
        self.adapters, self.adapter_metadata = _load_adapter(Path(adapter_path))
        self.pt_metadata = self.acoustic_extractor.pt_metadata
        self.campplus_metadata = self.acoustic_extractor.campplus_metadata

    def _redim_embedding(self, model, segments, label):
        vectors = []
        sample_counts = []
        with torch.inference_mode():
            for segment in segments:
                segment = np.ascontiguousarray(segment, dtype=np.float32)
                sample_counts.append(int(segment.size))
                tensor = torch.from_numpy(segment).unsqueeze(0).to(self.device)
                value = model(tensor).squeeze(0).float().cpu().numpy()
                vectors.append(l2_normalize(value, label))
        aggregate = np.mean(np.stack(vectors).astype(np.float64), axis=0).astype(np.float32)
        return l2_normalize(aggregate, label + " aggregate"), sample_counts

    def _adapt(self, vector, name, duration_key):
        source = l2_normalize(vector, name + " source")
        delta = self.adapters[name + "_" + duration_key]
        proposed = source + np.matmul(delta, source)
        return l2_normalize(proposed.astype(np.float32), name + " adapted")

    @staticmethod
    def _fuse(acoustic, lm, ptn, weights):
        output = np.concatenate((
            math.sqrt(float(weights[0])) * l2_normalize(acoustic, "acoustic block"),
            math.sqrt(float(weights[1])) * l2_normalize(lm, "LM block"),
            math.sqrt(float(weights[2])) * l2_normalize(ptn, "PTN block"),
        )).astype(np.float32)
        return l2_normalize(output, "distillation final 832-D embedding")

    @staticmethod
    def padded_acoustic(acoustic):
        output = np.concatenate((
            l2_normalize(acoustic, "fallback acoustic block"),
            np.zeros(2 * REDIM_DIM, dtype=np.float32),
        )).astype(np.float32)
        return l2_normalize(output, "fallback padded acoustic")

    def extract_file(self, wav_path):
        waveform = load_audio(Path(wav_path), self.fbank_config.sample_rate, remove_dc=False)
        duration_seconds = float(waveform.size) / float(self.fbank_config.sample_rate)
        policy, pt_weight = self.acoustic_extractor.duration_policy(duration_seconds)
        acoustic, pt_frames, camp_frames = _extract_acoustic_waveform(
            self.acoustic_extractor, waveform, pt_weight
        )
        try:
            segments = self.acoustic_extractor.pt_extractor._segments(waveform)
            lm, lm_samples = self._redim_embedding(self.redim_models["lm"], segments, "LM ReDimNet2")
            ptn, ptn_samples = self._redim_embedding(self.redim_models["ptn"], segments, "PTN ReDimNet2")
            if policy == "short_0p5":
                lm = self._adapt(lm, "lm", "short05")
                ptn = self._adapt(ptn, "ptn", "short05")
                weights = SHORT_WEIGHTS
                adapter_policy = "short05"
            elif policy == "short_1p0":
                lm = self._adapt(lm, "lm", "short10")
                ptn = self._adapt(ptn, "ptn", "short10")
                weights = SHORT_WEIGHTS
                adapter_policy = "short10"
            else:
                weights = ENROLL_WEIGHTS
                adapter_policy = "full_bypass"
            embedding = self._fuse(acoustic, lm, ptn, weights)
            fallback_mode = "none"
            fallback_reason = ""
        except Exception as exc:
            embedding = self.padded_acoustic(acoustic)
            weights = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            adapter_policy = "padded_acoustic_file_fallback"
            lm_samples = []; ptn_samples = []
            fallback_mode = "acoustic_padded"
            fallback_reason = "{}: {}".format(type(exc).__name__, exc)
        return embedding, {
            "duration_seconds": duration_seconds, "duration_policy": policy,
            "adapter_policy": adapter_policy, "branch_weights": tuple(float(item) for item in weights),
            "pt_frame_counts": pt_frames, "campplus_frame_counts": camp_frames,
            "lm_sample_counts": lm_samples, "ptn_sample_counts": ptn_samples,
            "fallback_mode": fallback_mode, "fallback_reason": fallback_reason,
            "output_dim": self.output_dim,
            "frontend_mode": "acoustic_exact_pure_torch+dual_redimnet2_waveform",
        }


class PaddedAcousticFallbackExtractor:
    """Numerically exact acoustic score in the fixed 832-D output space."""

    output_dim = OUTPUT_DIM

    def __init__(self, acoustic_extractor):
        self.acoustic_extractor = acoustic_extractor

    def extract_file(self, wav_path):
        acoustic, metadata = self.acoustic_extractor.extract_file(Path(wav_path))
        output = MultidomainReDimKDExtractor.padded_acoustic(acoustic)
        metadata = dict(metadata)
        metadata.update({
            "adapter_policy": "padded_acoustic_fallback",
            "fallback_mode": "acoustic_padded",
            "fallback_reason": "distillation-only resources unavailable",
            "output_dim": self.output_dim,
        })
        return output, metadata
