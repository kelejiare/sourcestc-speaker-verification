#!/usr/bin/env python3
"""Inference-only WeSpeaker ResNet34/TSTP implementation.

The module names and tensor shapes match the official CN-Celeb ResNet34-LM
checkpoint.  Training-only ArcMargin projection parameters are validated and
discarded before a strict speaker-model load.
"""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class TSTP(nn.Module):
    def forward(self, features):
        flattened = features.reshape(features.shape[0], -1, features.shape[-1])
        mean = flattened.mean(dim=-1)
        std = torch.sqrt(torch.var(flattened, dim=-1) + 1.0e-7)
        return torch.cat((mean, std), dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, features):
        output = F.relu(self.bn1(self.conv1(features)))
        output = self.bn2(self.conv2(output))
        output = output + self.shortcut(features)
        return F.relu(output)


class ResNet34(nn.Module):
    embedding_dim = 256

    def __init__(self, feat_dim=80, embed_dim=256):
        super().__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 3, 1)
        self.layer2 = self._make_layer(64, 4, 2)
        self.layer3 = self._make_layer(128, 6, 2)
        self.layer4 = self._make_layer(256, 3, 2)
        self.pool = TSTP()
        self.seg_1 = nn.Linear((feat_dim // 8) * 256 * 2, embed_dim)
        self.seg_bn_1 = nn.Identity()
        self.seg_2 = nn.Identity()

    def _make_layer(self, planes, count, stride):
        layers = []
        for item_stride in [stride] + [1] * (count - 1):
            layers.append(BasicBlock(self.in_planes, planes, item_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, features):
        output = features.permute(0, 2, 1).unsqueeze(1)
        output = F.relu(self.bn1(self.conv1(output)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return self.seg_1(self.pool(output))


def load_resnet34_lm(model_path: Path, device: torch.device):
    try:
        payload = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(model_path), map_location="cpu")
    if not isinstance(payload, (dict, OrderedDict)):
        raise RuntimeError("CN-Celeb ResNet34-LM checkpoint schema mismatch")
    payload = OrderedDict(payload)
    projection = payload.pop("projection.weight", None)
    if projection is None or tuple(projection.shape) != (8379, 256):
        raise RuntimeError("CN-Celeb training projection contract mismatch")
    model = ResNet34(feat_dim=80, embed_dim=256)
    incompatible = model.load_state_dict(payload, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("CN-Celeb ResNet34-LM strict load failed")
    model.to(device)
    model.eval()
    return model, {
        "checkpoint_tensor_count": len(payload) + 1,
        "inference_tensor_count": len(payload),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_dim": model.embedding_dim,
        "training_class_count": int(projection.shape[0]),
    }

