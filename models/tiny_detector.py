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


class SPPBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([x, self.pool5(x), self.pool9(x), self.pool13(x)], dim=1))


def make_residual_stack(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])


class TinyDetector(nn.Module):
    """Small YOLO-style detector implemented from basic PyTorch layers."""

    def __init__(
        self,
        num_classes: int = 5,
        num_anchors: int = 3,
        base_channels: int = 48,
        head_channels: int = 384,
        residual_blocks: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.pred_dim = 5 + num_classes
        residual_blocks = residual_blocks or {"stage2": 1, "stage3": 3, "stage4": 4}
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.backbone = nn.Sequential(
            ConvBlock(3, c1, stride=2),              # 416 -> 208
            ConvBlock(c1, c2, stride=2),             # 208 -> 104
            make_residual_stack(c2, residual_blocks.get("stage2", 1)),
            ConvBlock(c2, c3, stride=2),             # 104 -> 52
            make_residual_stack(c3, residual_blocks.get("stage3", 3)),
            ConvBlock(c3, c4, stride=2),             # 52 -> 26
            make_residual_stack(c4, residual_blocks.get("stage4", 4)),
            SPPBlock(c4),
        )
        self.head = nn.Sequential(
            nn.Conv2d(c4, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, num_anchors * self.pred_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(self.backbone(x))
        n, _, h, w = y.shape
        y = y.view(n, self.num_anchors, self.pred_dim, h, w)
        return y.permute(0, 3, 4, 1, 2).contiguous()
