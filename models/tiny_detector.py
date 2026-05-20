from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RepConv(nn.Module):
    """Train-time multi-branch conv inspired by YOLOv7 planned re-parameterization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.identity = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv3(x) + self.conv1(x) + self.identity(x))


class ELANBlock(nn.Module):
    """Small E-ELAN-like aggregation block built from basic layers."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 2) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.short = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.long = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([RepConv(hidden) for _ in range(depth)])
        self.fuse = ConvBNAct(hidden * (depth + 2), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self.short(x)]
        y = self.long(x)
        outputs.append(y)
        for block in self.blocks:
            y = block(y)
            outputs.append(y)
        return self.fuse(torch.cat(outputs, dim=1))


class SPPBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.fuse = ConvBNAct(channels * 4, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([x, self.pool5(x), self.pool9(x), self.pool13(x)], dim=1))


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, head_channels: int, num_anchors: int, pred_dim: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_channels, head_channels, kernel_size=3),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            RepConv(head_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(head_channels, num_anchors * pred_dim, kernel_size=1),
        )
        self.num_anchors = num_anchors
        self.pred_dim = pred_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(x)
        n, _, h, w = y.shape
        y = y.view(n, self.num_anchors, self.pred_dim, h, w)
        return y.permute(0, 3, 4, 1, 2).contiguous()


class TinyDetector(nn.Module):
    """YOLO-style detector with configurable backbone, RepConv heads, aux heads, and 3 scales."""

    def __init__(
        self,
        num_classes: int = 5,
        num_anchors: int | list[int] = 3,
        base_channels: int = 40,
        head_channels: int = 256,
        dropout: float = 0.05,
        elan_depth: int = 2,
        aux_head: bool = True,
        backbone: str = "resnet50",
        pretrained: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pred_dim = 5 + num_classes
        if isinstance(num_anchors, int):
            self.num_anchors = [num_anchors, num_anchors, num_anchors]
        else:
            self.num_anchors = num_anchors

        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = resnet50(weights=weights)
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = nn.Sequential(resnet.layer4, SPPBlock(2048))
            feature_channels = [512, 1024, 2048]
        elif backbone == "eelan":
            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4
            c4 = base_channels * 8
            c5 = base_channels * 12
            self.stem = nn.Sequential(
                ConvBNAct(3, c1, stride=2),      # 416 -> 208
                ConvBNAct(c1, c2, stride=2),     # 208 -> 104
                ELANBlock(c2, c2, depth=elan_depth),
            )
            self.down3 = ConvBNAct(c2, c3, stride=2)  # 104 -> 52
            self.elan3 = ELANBlock(c3, c3, depth=elan_depth)
            self.down4 = ConvBNAct(c3, c4, stride=2)  # 52 -> 26
            self.elan4 = ELANBlock(c4, c4, depth=elan_depth + 1)
            self.down5 = ConvBNAct(c4, c5, stride=2)  # 26 -> 13
            self.elan5 = nn.Sequential(ELANBlock(c5, c5, depth=elan_depth + 1), SPPBlock(c5))
            feature_channels = [c3, c4, c5]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone

        self.main_heads = nn.ModuleList(
            [
                DetectionHead(feature_channels[0], head_channels, self.num_anchors[0], self.pred_dim, dropout),
                DetectionHead(feature_channels[1], head_channels, self.num_anchors[1], self.pred_dim, dropout),
                DetectionHead(feature_channels[2], head_channels, self.num_anchors[2], self.pred_dim, dropout),
            ]
        )
        self.aux_head_enabled = aux_head
        self.aux_heads = nn.ModuleList(
            [
                DetectionHead(feature_channels[0], head_channels // 2, self.num_anchors[0], self.pred_dim, dropout),
                DetectionHead(feature_channels[1], head_channels // 2, self.num_anchors[1], self.pred_dim, dropout),
                DetectionHead(feature_channels[2], head_channels // 2, self.num_anchors[2], self.pred_dim, dropout),
            ]
        )

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        if self.backbone_name == "resnet50":
            x = self.layer1(self.stem(x))
            p3 = self.layer2(x)
            p4 = self.layer3(p3)
            p5 = self.layer4(p4)
        else:
            x = self.stem(x)
            p3 = self.elan3(self.down3(x))
            p4 = self.elan4(self.down4(p3))
            p5 = self.elan5(self.down5(p4))
        features = [p3, p4, p5]
        main = [head(feature) for head, feature in zip(self.main_heads, features)]
        if self.training and self.aux_head_enabled:
            aux = [head(feature) for head, feature in zip(self.aux_heads, features)]
        else:
            aux = []
        return {"main": main, "aux": aux}
