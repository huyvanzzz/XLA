from __future__ import annotations

import torch
from torch import nn


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


class LargeKernelBlock(nn.Module):
    """Depthwise large-kernel context block for deeper detector features."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PartialSelfAttention(nn.Module):
    """Apply self-attention to a channel subset, following the PSA idea used in recent YOLOs."""

    def __init__(self, channels: int, heads: int = 4, ratio: float = 0.5) -> None:
        super().__init__()
        attn_channels = max(heads, int(channels * ratio))
        attn_channels = max(heads, (attn_channels // heads) * heads)
        attn_channels = min(attn_channels, channels)
        self.attn_channels = attn_channels
        self.norm = nn.LayerNorm(attn_channels)
        self.attn = nn.MultiheadAttention(attn_channels, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(attn_channels),
            nn.Linear(attn_channels, attn_channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(attn_channels * 2, attn_channels),
        )
        self.fuse = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_channels <= 0:
            return x
        xa, xb = x[:, : self.attn_channels], x[:, self.attn_channels :]
        b, c, h, w = xa.shape
        tokens = xa.flatten(2).transpose(1, 2)
        attn_in = self.norm(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(tokens)
        xa = tokens.transpose(1, 2).reshape(b, c, h, w)
        return self.fuse(torch.cat([xa, xb], dim=1))


class FPNPANNeck(nn.Module):
    """Top-down FPN plus bottom-up PAN feature fusion for small-object recall."""

    def __init__(self, in_channels: list[int], out_channels: int = 256, attention_heads: int = 4) -> None:
        super().__init__()
        self.lateral3 = ConvBNAct(in_channels[0], out_channels, kernel_size=1)
        self.lateral4 = ConvBNAct(in_channels[1], out_channels, kernel_size=1)
        self.lateral5 = ConvBNAct(in_channels[2], out_channels, kernel_size=1)
        self.deep_context = nn.Sequential(
            SPPBlock(out_channels),
            LargeKernelBlock(out_channels, kernel_size=7),
            PartialSelfAttention(out_channels, heads=attention_heads, ratio=0.5),
        )
        self.fuse4 = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, kernel_size=1),
            ELANBlock(out_channels, out_channels, depth=2),
        )
        self.fuse3 = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, kernel_size=1),
            ELANBlock(out_channels, out_channels, depth=2),
        )
        self.down3 = ConvBNAct(out_channels, out_channels, stride=2)
        self.pan4 = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, kernel_size=1),
            ELANBlock(out_channels, out_channels, depth=2),
        )
        self.down4 = ConvBNAct(out_channels, out_channels, stride=2)
        self.pan5 = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, kernel_size=1),
            ELANBlock(out_channels, out_channels, depth=2),
            LargeKernelBlock(out_channels, kernel_size=7),
        )

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = features
        p3 = self.lateral3(p3)
        p4 = self.lateral4(p4)
        p5 = self.deep_context(self.lateral5(p5))

        p5_up = torch.nn.functional.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4_td = self.fuse4(torch.cat([p4, p5_up], dim=1))
        p4_up = torch.nn.functional.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")
        p3_out = self.fuse3(torch.cat([p3, p4_up], dim=1))

        p3_down = self.down3(p3_out)
        p4_out = self.pan4(torch.cat([p4_td, p3_down], dim=1))
        p4_down = self.down4(p4_out)
        p5_out = self.pan5(torch.cat([p5, p4_down], dim=1))
        return [p3_out, p4_out, p5_out]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50Backbone(nn.Module):
    WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        if pretrained:
            state = torch.hub.load_state_dict_from_url(self.WEIGHTS_URL, progress=True, map_location="cpu")
            self.load_state_dict({k: v for k, v in state.items() if not k.startswith("fc.")}, strict=False)

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [Bottleneck(self.inplanes, channels, stride=stride)]
        self.inplanes = channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        return p3, p4, p5


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
        neck_channels: int = 256,
        attention_heads: int = 4,
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
            self.resnet = ResNet50Backbone(pretrained=pretrained)
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
        self.neck = FPNPANNeck(feature_channels, out_channels=neck_channels, attention_heads=attention_heads)
        feature_channels = [neck_channels, neck_channels, neck_channels]

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
            p3, p4, p5 = self.resnet(x)
        else:
            x = self.stem(x)
            p3 = self.elan3(self.down3(x))
            p4 = self.elan4(self.down4(p3))
            p5 = self.elan5(self.down5(p4))
        features = self.neck((p3, p4, p5))
        main = [head(feature) for head, feature in zip(self.main_heads, features)]
        if self.training and self.aux_head_enabled:
            aux = [head(feature) for head, feature in zip(self.aux_heads, features)]
        else:
            aux = []
        return {"main": main, "aux": aux}
