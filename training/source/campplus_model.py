#!/usr/bin/env python3
"""Dependency-free CAM++ speaker embedding model.

This file is a minimal inference-only adaptation of the Apache-2.0 licensed
3D-Speaker CAM++ implementation.  Module names and tensor shapes intentionally
match ``iic/speech_campplus_sv_zh_en_16k-common_advanced`` so its public
checkpoint can be loaded strictly without ModelScope or torchaudio.

Upstream:
https://github.com/modelscope/3D-Speaker/tree/main/speakerlab/models/campplus
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module(
                "batchnorm", nn.BatchNorm1d(channels, affine=False)
            )
        else:
            raise ValueError("未知非线性层：{}".format(name))
    return nonlinear


class StatsPool(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=-1)
        std = features.std(dim=-1, unbiased=True)
        return torch.cat((mean, std), dim=-1)


class TDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            if kernel_size % 2 != 1:
                raise ValueError("负 padding 只支持奇数 kernel_size")
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(features))


class CAMLayer(nn.Module):
    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ):
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(
            bn_channels, bn_channels // reduction, kernel_size=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(
            bn_channels // reduction, out_channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def seg_pooling(
        features: torch.Tensor, segment_length: int = 100
    ) -> torch.Tensor:
        segments = F.avg_pool1d(
            features,
            kernel_size=segment_length,
            stride=segment_length,
            ceil_mode=True,
        )
        shape = segments.shape
        expanded = (
            segments.unsqueeze(-1)
            .expand(*shape, segment_length)
            .reshape(*shape[:-1], -1)
        )
        return expanded[..., : features.shape[-1]]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        local = self.linear_local(features)
        context = features.mean(-1, keepdim=True) + self.seg_pooling(features)
        mask = self.sigmoid(self.linear2(self.relu(self.linear1(context))))
        return local * mask


class CAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("CAMDenseTDNNLayer 要求奇数 kernel_size")
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
        )

    def bn_function(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear1(self.nonlinear1(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # The public inference path runs in eval mode, so activation checkpointing
        # from the training implementation is intentionally unnecessary here.
        bottleneck = self.bn_function(features)
        return self.cam_layer(self.nonlinear2(bottleneck))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
    ):
        super().__init__()
        for index in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + index * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd{}".format(index + 1), layer)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = features
        for layer in self:
            output = torch.cat((output, layer(output)), dim=1)
        return output


class TransitLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(features))


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            output = self.linear(features.unsqueeze(-1)).squeeze(-1)
        else:
            output = self.linear(features)
        return self.nonlinear(output)


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=(stride, 1),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(features)))
        output = self.bn2(self.conv2(output))
        output = output + self.shortcut(features)
        return F.relu(output)


class FCM(nn.Module):
    def __init__(
        self,
        block=BasicResBlock,
        num_blocks=(2, 2),
        m_channels: int = 32,
        feat_dim: int = 80,
    ):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(
            block, m_channels, num_blocks[0], stride=2
        )
        self.layer2 = self._make_layer(
            block, m_channels, num_blocks[1], stride=2
        )
        self.conv2 = nn.Conv2d(
            m_channels,
            m_channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, count, stride):
        strides = [stride] + [1] * (count - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = features.unsqueeze(1)
        output = F.relu(self.bn1(self.conv1(output)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = F.relu(self.bn2(self.conv2(output)))
        shape = output.shape
        return output.reshape(shape[0], shape[1] * shape[2], shape[3])


class CAMPPlus(nn.Module):
    embedding_dim = 192

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_size
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    )
                ]
            )
        )
        channels = init_channels
        for index, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block{}".format(index + 1), block)
            channels += num_layers * growth_rate
            self.xvector.add_module(
                "transit{}".format(index + 1),
                TransitLayer(
                    channels,
                    channels // 2,
                    bias=False,
                    config_str=config_str,
                ),
            )
            channels //= 2
        self.xvector.add_module(
            "out_nonlinear", get_nonlinear(config_str, channels)
        )
        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module(
            "dense",
            DenseLayer(
                channels * 2,
                embedding_size,
                config_str="batchnorm_",
            ),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 80:
            raise ValueError(
                "CAM++ 输入必须为 (batch, frames, 80)，实际为 {}".format(
                    tuple(features.shape)
                )
            )
        output = features.permute(0, 2, 1)
        return self.xvector(self.head(output))


def load_campplus_model(
    model_path: Path, device: torch.device
) -> Tuple[CAMPPlus, Dict[str, int]]:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError("CAM++ 权重不存在：{}".format(model_path))
    try:
        state_dict = torch.load(
            str(model_path), map_location="cpu", weights_only=True
        )
    except TypeError:
        state_dict = torch.load(str(model_path), map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("CAM++ checkpoint 必须是 state_dict")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "CAM++ 权重键不匹配：missing={} unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    model.to(device)
    model.eval()
    return model, {
        "state_tensor_count": len(state_dict),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "embedding_dim": model.embedding_dim,
    }
