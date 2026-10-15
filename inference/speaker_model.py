#!/usr/bin/env python3
"""Checkpoint-compatible ResNet speaker embedding model.

The checkpoint contains only ``front.*`` and ``bottleneck.*`` tensors.  The
parameter-free temporal statistics pooling is reconstructed from the tensor
shapes and the public WeSpeaker ResNet design.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(features)
        output = F.relu(self.bn1(self.conv1(features)), inplace=False)
        output = F.relu(self.bn2(self.conv2(output)), inplace=False)
        output = self.bn3(self.conv3(output))
        return F.relu(output + residual, inplace=False)


class ResNetFront(nn.Module):
    def __init__(self, block_counts: Iterable[int] = (6, 16, 24, 3)):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        counts = tuple(block_counts)
        if counts != (6, 16, 24, 3):
            raise ValueError("比赛 checkpoint 要求 block_counts=(6, 16, 24, 3)")
        self.layer1 = self._make_layer(64, counts[0], stride=1)
        self.layer2 = self._make_layer(128, counts[1], stride=2)
        self.layer3 = self._make_layer(256, counts[2], stride=2)
        self.layer4 = self._make_layer(512, counts[3], stride=2)

    def _make_layer(self, planes: int, count: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (count - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(BottleneckBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes * BottleneckBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(features)), inplace=False)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.layer4(output)


class TemporalStatisticsPooling(nn.Module):
    """Concatenate temporal mean and standard deviation.

    Input shape is ``(batch, channels, frequency, frames)``.  The final axis is
    time.  With 80-bin FBank input, the ResNet frequency axis becomes 10 and
    the output is ``2048 * 10 * 2 = 40960`` elements.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] < 2:
            raise ValueError("统计池化至少需要 2 个输出时间帧")
        mean = features.mean(dim=-1).flatten(start_dim=1)
        std = torch.sqrt(torch.var(features, dim=-1) + 1.0e-7).flatten(start_dim=1)
        return torch.cat((mean, std), dim=1)


class SpeakerResNet(nn.Module):
    feature_dim = 80
    embedding_dim = 256
    pooled_dim = 40960

    def __init__(self):
        super().__init__()
        self.front = ResNetFront()
        self.pool = TemporalStatisticsPooling()
        self.bottleneck = nn.Linear(self.pooled_dim, self.embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "输入特征必须是 (batch, frames, 80)，实际为 {}".format(
                    tuple(features.shape)
                )
            )
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                "输入 FBank 必须为 {} 维，实际为 {}".format(
                    self.feature_dim, features.shape[-1]
                )
            )
        output = features.permute(0, 2, 1).unsqueeze(1)
        output = self.front(output)
        pooled = self.pool(output)
        if pooled.shape[1] != self.pooled_dim:
            raise RuntimeError(
                "统计池化维度错误：期望 {}，实际 {}".format(
                    self.pooled_dim, pooled.shape[1]
                )
            )
        return self.bottleneck(pooled)


def _unwrap_state_dict(checkpoint: object) -> OrderedDict:
    if isinstance(checkpoint, (dict, OrderedDict)):
        for key in ("state_dict", "model", "model_state_dict", "network"):
            value = checkpoint.get(key) if isinstance(checkpoint, dict) else None
            if isinstance(value, (dict, OrderedDict)):
                checkpoint = value
                break
    if not isinstance(checkpoint, (dict, OrderedDict)):
        raise TypeError("checkpoint 不是 state_dict：{}".format(type(checkpoint).__name__))
    normalized = OrderedDict()
    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith("module.") else key
        normalized[clean_key] = value
    return normalized


def load_model(model_path: Path, device: torch.device) -> Tuple[SpeakerResNet, Dict[str, int]]:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError("模型权重不存在：{}".format(model_path))
    try:
        checkpoint = torch.load(
            str(model_path), map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    model = SpeakerResNet()
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "权重键不匹配：missing={} unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    model.to(device)
    model.eval()
    metadata = {
        "state_tensor_count": len(state_dict),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_dim": model.embedding_dim,
    }
    return model, metadata


def choose_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前环境不可用")
    return device
