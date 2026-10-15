#!/usr/bin/env python3
"""Dependency-free ERes2NetV2 speaker embedding model.

This inference-only adaptation preserves the module names and tensor shapes of
the Apache-2.0 licensed 3D-Speaker implementation so the official
``iic/speech_eres2netv2_sv_zh-cn_16k-common`` checkpoint loads strictly.

Upstream:
https://github.com/modelscope/3D-Speaker/tree/main/speakerlab/models/eres2net
"""

import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ReLU(nn.Hardtanh):
    def __init__(self, inplace: bool = False):
        super().__init__(0, 20, inplace)


class AFF(nn.Module):
    def __init__(self, channels: int = 64, r: int = 4):
        super().__init__()
        inter_channels = channels // r
        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, downsampled: torch.Tensor) -> torch.Tensor:
        attention = self.local_att(torch.cat((x, downsampled), dim=1))
        attention = 1.0 + torch.tanh(attention)
        return x * attention + downsampled * (2.0 - attention)


class TSTP(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=-1).flatten(start_dim=1)
        std = torch.sqrt(torch.var(features, dim=-1) + 1.0e-8).flatten(
            start_dim=1
        )
        return torch.cat((mean, std), dim=1)


class BasicBlockERes2NetV2(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        baseWidth: int = 26,
        scale: int = 2,
        expansion: int = 2,
    ):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(
            in_planes, width * scale, 1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion
        self.convs = nn.ModuleList(
            [nn.Conv2d(width, width, 3, padding=1, bias=False) for _ in range(scale)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale)])
        self.relu = ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            width * scale, planes * expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    expansion * planes,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(expansion * planes),
            )
        self.width = width

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(features)
        output = self.relu(self.bn1(self.conv1(features)))
        split = torch.split(output, self.width, dim=1)
        merged = None
        branch = None
        for index in range(self.nums):
            branch = split[index] if index == 0 else branch + split[index]
            branch = self.relu(self.bns[index](self.convs[index](branch)))
            merged = branch if merged is None else torch.cat((merged, branch), dim=1)
        output = self.bn3(self.conv3(merged))
        return self.relu(output + residual)


class BasicBlockERes2NetV2AFF(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        baseWidth: int = 26,
        scale: int = 2,
        expansion: int = 2,
    ):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(
            in_planes, width * scale, 1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion
        self.convs = nn.ModuleList(
            [nn.Conv2d(width, width, 3, padding=1, bias=False) for _ in range(scale)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale)])
        self.fuse_models = nn.ModuleList(
            [AFF(channels=width, r=4) for _ in range(scale - 1)]
        )
        self.relu = ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            width * scale, planes * expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    expansion * planes,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(expansion * planes),
            )
        self.width = width

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(features)
        output = self.relu(self.bn1(self.conv1(features)))
        split = torch.split(output, self.width, dim=1)
        merged = None
        branch = None
        for index in range(self.nums):
            branch = (
                split[index]
                if index == 0
                else self.fuse_models[index - 1](branch, split[index])
            )
            branch = self.relu(self.bns[index](self.convs[index](branch)))
            merged = branch if merged is None else torch.cat((merged, branch), dim=1)
        output = self.bn3(self.conv3(merged))
        return self.relu(output + residual)


class ERes2NetV2(nn.Module):
    def __init__(
        self,
        num_blocks=(3, 4, 6, 3),
        m_channels: int = 64,
        feat_dim: int = 80,
        embedding_size: int = 192,
        baseWidth: int = 26,
        scale: int = 2,
        expansion: int = 2,
    ):
        super().__init__()
        self.in_planes = m_channels
        self.expansion = expansion
        self.conv1 = nn.Conv2d(1, m_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(
            BasicBlockERes2NetV2,
            m_channels,
            num_blocks[0],
            1,
            baseWidth,
            scale,
        )
        self.layer2 = self._make_layer(
            BasicBlockERes2NetV2,
            m_channels * 2,
            num_blocks[1],
            2,
            baseWidth,
            scale,
        )
        self.layer3 = self._make_layer(
            BasicBlockERes2NetV2AFF,
            m_channels * 4,
            num_blocks[2],
            2,
            baseWidth,
            scale,
        )
        self.layer4 = self._make_layer(
            BasicBlockERes2NetV2AFF,
            m_channels * 8,
            num_blocks[3],
            2,
            baseWidth,
            scale,
        )
        final_channels = m_channels * 8 * expansion
        self.layer3_ds = nn.Conv2d(
            m_channels * 4 * expansion,
            final_channels,
            3,
            padding=1,
            stride=2,
            bias=False,
        )
        self.fuse34 = AFF(channels=final_channels, r=4)
        stats_dim = (feat_dim // 8) * final_channels
        self.pool = TSTP(in_dim=stats_dim)
        self.seg_1 = nn.Linear(stats_dim * 2, embedding_size)
        self.seg_bn_1 = nn.Identity()
        self.seg_2 = nn.Identity()

    def _make_layer(
        self,
        block,
        planes: int,
        count: int,
        stride: int,
        base_width: int,
        scale: int,
    ) -> nn.Sequential:
        layers = []
        for layer_stride in [stride] + [1] * (count - 1):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    layer_stride,
                    baseWidth=base_width,
                    scale=scale,
                    expansion=self.expansion,
                )
            )
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = features.permute(0, 2, 1).unsqueeze(1)
        output = F.relu(self.bn1(self.conv1(output)))
        output1 = self.layer1(output)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output4 = self.layer4(output3)
        fused = self.fuse34(output4, self.layer3_ds(output3))
        return self.seg_1(self.pool(fused))


def load_eres2netv2_model(
    model_path: Path, device: torch.device
) -> Tuple[ERes2NetV2, Dict[str, object]]:
    payload = torch.load(str(model_path), map_location="cpu")
    if not isinstance(payload, (dict, OrderedDict)):
        raise RuntimeError("ERes2NetV2 checkpoint 格式错误")
    model = ERes2NetV2(feat_dim=80, embedding_size=192)
    incompatible = model.load_state_dict(payload, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("ERes2NetV2 checkpoint 参数键不匹配")
    model.to(device)
    model.eval()
    return model, {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_dim": 192,
        "checkpoint_path": str(model_path),
        "checkpoint_tensor_count": len(payload),
    }
