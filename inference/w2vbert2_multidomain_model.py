#!/usr/bin/env python3
"""Minimal inference-only loader for the multilingual w2v-BERT 2.0 SV teacher."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


EXPECTED_TEACHER_TENSORS = 933
EXPECTED_TEACHER_PARAMETERS = 586652993
EXPECTED_OUTPUT_DIMENSION = 256
EXPECTED_CHECKPOINT_SHA256 = "041ebf0fd6ebb22bd259eb315c082817ebcfd911f5d1722e103dfcb8555fd8db"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim=3200, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        weights = self.attention(x.transpose(1, 2)).transpose(1, 2)
        mean = torch.sum(x * weights, dim=1)
        std = torch.sqrt(
            (torch.sum((x ** 2) * weights, dim=1) - mean ** 2).clamp(min=1.0e-5)
        )
        return torch.cat((mean, std), dim=1)


class MultiDomainW2VBert2Speaker(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        try:
            from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
        except ImportError as exc:
            raise RuntimeError("Transformers with Wav2Vec2BertModel is required") from exc
        config = Wav2Vec2BertConfig.from_json_file(str(config_path))
        config.output_hidden_states = True
        config.apply_spec_augment = False
        self.encoder = Wav2Vec2BertModel(config)
        if hasattr(self.encoder, "masked_spec_embed"):
            delattr(self.encoder, "masked_spec_embed")
        self.adapter_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(1024, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
            )
            for _ in range(25)
        )
        self.pooling = AttentiveStatisticsPooling(3200, 128)
        self.bottleneck = nn.Linear(6400, EXPECTED_OUTPUT_DIMENSION)

    def forward(self, input_features, attention_mask=None):
        # Reproduce the model author's forward_w2v_bert exactly.  The released
        # speaker checkpoint was trained by applying feature_projection and
        # then calling each encoder layer directly; the generic Transformers
        # Wav2Vec2BertModel.forward adds encoder wrapper operations that are not
        # part of the published speaker inference graph.
        hidden = self.encoder.feature_projection(input_features)[0]
        hidden_states = [hidden]
        for layer in self.encoder.encoder.layers:
            hidden = layer(hidden)[0]
            hidden_states.append(hidden)
        if len(hidden_states) != len(self.adapter_layers):
            raise RuntimeError("unexpected w2v-BERT hidden-state contract")
        adapted = [layer(value) for layer, value in zip(self.adapter_layers, hidden_states)]
        embedding = self.bottleneck(self.pooling(torch.cat(adapted, dim=-1)))
        return torch.nn.functional.normalize(embedding, p=2.0, dim=1)


def load_teacher(checkpoint_path, config_path, device="cpu"):
    observed_sha256 = sha256_file(checkpoint_path)
    if observed_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("teacher checkpoint SHA-256 mismatch")
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "artifact_kind", "source_sha256", "source_epoch_idx",
        "embedding_dimension", "state_dict"
    }:
        raise RuntimeError("teacher checkpoint schema mismatch")
    if int(payload["schema_version"]) != 1 or int(payload["embedding_dimension"]) != 256:
        raise RuntimeError("teacher checkpoint metadata mismatch")
    source_state = payload["state_dict"]
    if len(source_state) != EXPECTED_TEACHER_TENSORS:
        raise RuntimeError("teacher tensor-count mismatch")
    parameter_count = sum(int(value.numel()) for value in source_state.values())
    if parameter_count != EXPECTED_TEACHER_PARAMETERS:
        raise RuntimeError("teacher parameter-count mismatch")

    remapped = {}
    for key, value in source_state.items():
        if key.startswith("front.encoder."):
            target = "encoder." + key[len("front.encoder."):]
        else:
            target = key
        remapped[target] = value
    model = MultiDomainW2VBert2Speaker(config_path)
    incompatible = model.load_state_dict(remapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("teacher strict-load mismatch")
    model.to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "checkpoint_sha256": observed_sha256,
        "tensor_count": len(source_state),
        "parameter_count": parameter_count,
        "storage_dtype": "float16",
        "compute_dtype": "float32",
        "output_dimension": EXPECTED_OUTPUT_DIMENSION,
    }


def build_feature_extractor(config_dir):
    try:
        from transformers import SeamlessM4TFeatureExtractor
    except ImportError as exc:
        raise RuntimeError("Transformers with SeamlessM4TFeatureExtractor is required") from exc
    return SeamlessM4TFeatureExtractor.from_pretrained(
        str(config_dir), local_files_only=True
    )


def extract_waveforms(model, feature_extractor, waveforms, device="cpu"):
    arrays = [np.ascontiguousarray(value, dtype=np.float32) for value in waveforms]
    if not arrays or len({int(value.size) for value in arrays}) != 1:
        raise ValueError(
            "author-exact w2v-BERT batches must be non-empty and equal length"
        )
    # Match forward_impl.py exactly.  SeamlessM4TFeatureExtractor's
    # ``padding=True`` adds a feature frame even for a one-item batch, which is
    # not what the released speaker checkpoint used during inference.
    features = feature_extractor(
        np.stack(arrays, axis=0),
        sampling_rate=16000,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_tensors="pt",
    )
    input_features = features["input_features"].to(torch.device(device))
    with torch.no_grad():
        output = model(input_features, attention_mask=None)
    result = output.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (len(arrays), EXPECTED_OUTPUT_DIMENSION):
        raise RuntimeError("teacher output shape mismatch")
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite teacher output")
    return result
