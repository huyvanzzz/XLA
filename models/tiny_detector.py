from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TinyDetector(nn.Module):
    """Small YOLO-style detector implemented from basic PyTorch layers."""

    def __init__(self, num_classes: int = 5, num_anchors: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.pred_dim = 5 + num_classes

        self.backbone = nn.Sequential(
            ConvBlock(3, 32, stride=2),      # 416 -> 208
            ConvBlock(32, 64, stride=2),     # 208 -> 104
            ResidualBlock(64),
            ConvBlock(64, 128, stride=2),    # 104 -> 52
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 256, stride=2),   # 52 -> 26
            ResidualBlock(256),
            ResidualBlock(256),
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, num_anchors * self.pred_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(self.backbone(x))
        n, _, h, w = y.shape
        y = y.view(n, self.num_anchors, self.pred_dim, h, w)
        return y.permute(0, 3, 4, 1, 2).contiguous()
