#!/usr/bin/env python3
"""Inference-only w2v-BERT speaker graph, without Transformers imports.

Conformer equations adapted from Hugging Face Transformers v4.46.3,
Copyright 2024 The Seamless Authors and HuggingFace Inc., Apache-2.0.
Only the released checkpoint's relative_key/24-layer/eval graph is supported.
The speaker head and author-exact encoder invocation match native_teacher.
"""

import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hf_audio_utils import mel_filter_bank, spectrogram, window_function
from w2vbert2_multidomain_model import (
    AttentiveStatisticsPooling, EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TEACHER_PARAMETERS, EXPECTED_TEACHER_TENSORS, sha256_file,
)


class FeatureProjection(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layer_norm = nn.LayerNorm(160, eps=c.layer_norm_eps)
        self.projection = nn.Linear(160, c.hidden_size)

    def forward(self, x):
        return self.projection(self.layer_norm(x))


class FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.intermediate_dense = nn.Linear(c.hidden_size, c.intermediate_size)
        self.output_dense = nn.Linear(c.intermediate_size, c.hidden_size)

    def forward(self, x):
        return self.output_dense(F.silu(self.intermediate_dense(x)))


class Convolution(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c.hidden_size
        self.layer_norm = nn.LayerNorm(h, eps=c.layer_norm_eps)
        self.pointwise_conv1 = nn.Conv1d(h, 2 * h, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(h, h, c.conv_depthwise_kernel_size,
                                        groups=h, bias=False)
        self.depthwise_layer_norm = nn.LayerNorm(h, eps=c.layer_norm_eps)
        self.pointwise_conv2 = nn.Conv1d(h, h, 1, bias=False)

    def forward(self, x):
        x = F.glu(self.pointwise_conv1(self.layer_norm(x).transpose(1, 2)), dim=1)
        x = self.depthwise_conv(F.pad(x, (self.depthwise_conv.kernel_size[0] - 1, 0)))
        x = self.depthwise_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.pointwise_conv2(F.silu(x)).transpose(1, 2)


class RelativeKeyAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.num_heads = c.num_attention_heads
        self.head_size = c.hidden_size // self.num_heads
        self.left = c.left_max_position_embeddings
        self.right = c.right_max_position_embeddings
        self.linear_q = nn.Linear(c.hidden_size, c.hidden_size)
        self.linear_k = nn.Linear(c.hidden_size, c.hidden_size)
        self.linear_v = nn.Linear(c.hidden_size, c.hidden_size)
        self.linear_out = nn.Linear(c.hidden_size, c.hidden_size)
        self.distance_embedding = nn.Embedding(self.left + self.right + 1, self.head_size)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.linear_k(x).view(b, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.linear_v(x).view(b, -1, self.num_heads, self.head_size).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        indices = torch.arange(t, dtype=torch.long, device=x.device)
        distances = (indices.view(1, -1) - indices.view(-1, 1)).clamp(-self.left, self.right)
        positions = self.distance_embedding(distances + self.left).to(dtype=q.dtype)
        scores = scores + torch.einsum('bhld,lrd->bhlr', q, positions) / math.sqrt(self.head_size)
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.linear_out(out.transpose(1, 2).reshape(b, -1, self.num_heads * self.head_size))


class EncoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        for name in ('ffn1_layer_norm', 'self_attn_layer_norm',
                     'ffn2_layer_norm', 'final_layer_norm'):
            setattr(self, name, nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps))
        self.ffn1 = FeedForward(c)
        self.ffn2 = FeedForward(c)
        self.self_attn = RelativeKeyAttention(c)
        self.conv_module = Convolution(c)

    def forward(self, x):
        x = self.ffn1(self.ffn1_layer_norm(x)) * 0.5 + x
        x = self.self_attn(self.self_attn_layer_norm(x)) + x
        x = x + self.conv_module(x)
        x = self.ffn2(self.ffn2_layer_norm(x)) * 0.5 + x
        return self.final_layer_norm(x)


class NativeSpeaker(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        raw = json.loads(Path(config_path).read_text())
        required = {'hidden_size': 1024, 'num_hidden_layers': 24,
                    'num_attention_heads': 16, 'intermediate_size': 4096,
                    'feature_projection_input_dim': 160, 'hidden_act': 'swish',
                    'position_embeddings_type': 'relative_key', 'add_adapter': False,
                    'use_intermediate_ffn_before_adapter': False}
        for key, expected in required.items():
            if raw.get(key) != expected:
                raise RuntimeError('unsupported native config: ' + key)
        c = SimpleNamespace(**raw)
        self.encoder = nn.Module()
        self.encoder.feature_projection = FeatureProjection(c)
        self.encoder.encoder = nn.Module()
        self.encoder.encoder.layers = nn.ModuleList([EncoderLayer(c) for _ in range(24)])
        self.adapter_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(1024, 128), nn.LayerNorm(128),
                          nn.ReLU(inplace=True), nn.Linear(128, 128)) for _ in range(25)
        ])
        self.pooling = AttentiveStatisticsPooling(3200, 128)
        self.bottleneck = nn.Linear(6400, 256)

    def forward(self, input_features, attention_mask=None):
        x = self.encoder.feature_projection(input_features)
        adapted = [self.adapter_layers[0](x)]
        for i, layer in enumerate(self.encoder.encoder.layers):
            x = layer(x)
            adapted.append(self.adapter_layers[i + 1](x))
        return F.normalize(self.bottleneck(self.pooling(torch.cat(adapted, dim=-1))), dim=1)


class NativeFeatures:
    """Exact SeamlessM4T padding=False, per-mel unbiased CMVN, stride 2."""
    def __init__(self):
        self.mel_filters = np.pad(mel_filter_bank(
            num_frequency_bins=256, num_mel_filters=80, min_frequency=20,
            max_frequency=8000, sampling_rate=16000, norm=None,
            mel_scale='kaldi', triangularize_in_mel_space=True), ((0, 1), (0, 0)))
        self.window = window_function(400, 'povey', periodic=False)

    def __call__(self, waveforms):
        result = []
        for waveform in waveforms:
            waveform = np.ascontiguousarray(waveform, dtype=np.float32).squeeze() * (2 ** 15)
            features = spectrogram(
                waveform, self.window, frame_length=400, hop_length=160,
                fft_length=512, power=2.0, center=False, preemphasis=0.97,
                mel_filters=self.mel_filters, log_mel='log',
                mel_floor=1.192092955078125e-07, remove_dc_offset=True).T
            features = (features - np.expand_dims(features.mean(0), 0)) / np.sqrt(
                np.expand_dims(features.var(0, ddof=1), 0) + 1e-7)
            features = np.asarray(features, dtype=np.float32)
            count = len(features) // 2 * 2
            if count < 2:
                raise ValueError('teacher audio has fewer than two feature frames')
            result.append(features[:count].reshape(count // 2, 160))
        if not result or len({v.shape for v in result}) != 1:
            raise ValueError('teacher batches require equal feature lengths')
        output = torch.from_numpy(np.stack(result))
        if not torch.isfinite(output).all():
            raise ValueError('non-finite native features')
        return output


def load_native_teacher(checkpoint_path, config_path,
                        expected_sha256=EXPECTED_CHECKPOINT_SHA256):
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError('native teacher expected SHA-256 is invalid')
    if sha256_file(checkpoint_path) != expected_sha256:
        raise RuntimeError('native teacher SHA-256 mismatch')
    payload = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
    if payload.get('schema_version') != 1 or payload.get('embedding_dimension') != 256:
        raise RuntimeError('native teacher metadata mismatch')
    state = payload['state_dict']
    if len(state) != EXPECTED_TEACHER_TENSORS or sum(v.numel() for v in state.values()) != EXPECTED_TEACHER_PARAMETERS:
        raise RuntimeError('native teacher tensor/parameter mismatch')
    remapped = {('encoder.' + k[len('front.encoder.'):] if k.startswith('front.encoder.') else k): v
                for k, v in state.items()}
    model = NativeSpeaker(config_path)
    model.load_state_dict(remapped, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del payload, state, remapped
    gc.collect()
    return model
