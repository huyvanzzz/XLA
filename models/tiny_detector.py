from __future__ import annotations

import math

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


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block[0](x)
        y = y.permute(0, 2, 3, 1)
        y = self.block[1](y)
        y = self.block[2](y)
        y = self.block[3](y)
        y = self.block[4](y)
        y = y.permute(0, 3, 1, 2)
        return x + self.layer_scale * y


class ConvNeXtBackbone(nn.Module):
    WEIGHTS = {
        "convnext_small": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
        "convnext_base": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
    }
    CONFIGS = {
        "convnext_small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
        "convnext_base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
    }

    def __init__(self, variant: str = "convnext_small", pretrained: bool = True) -> None:
        super().__init__()
        if variant not in self.CONFIGS:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")
        config = self.CONFIGS[variant]
        dims = config["dims"]
        depths = config["depths"]

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(depths[0])])
        self.down1 = nn.Sequential(LayerNorm2d(dims[0], eps=1e-6), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(dims[1]) for _ in range(depths[1])])
        self.down2 = nn.Sequential(LayerNorm2d(dims[1], eps=1e-6), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(dims[2]) for _ in range(depths[2])])
        self.down3 = nn.Sequential(LayerNorm2d(dims[2], eps=1e-6), nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2))
        self.stage4 = nn.Sequential(*[ConvNeXtBlock(dims[3]) for _ in range(depths[3])])

        if pretrained:
            state = torch.hub.load_state_dict_from_url(self.WEIGHTS[variant], progress=True, map_location="cpu")
            self.load_state_dict(self._convert_torchvision_state(state), strict=False)

    def _convert_torchvision_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        converted: dict[str, torch.Tensor] = {}
        prefix_map = {
            "features.0.0.": "stem.0.",
            "features.0.1.": "stem.1.",
            "features.2.0.": "down1.0.",
            "features.2.1.": "down1.1.",
            "features.4.0.": "down2.0.",
            "features.4.1.": "down2.1.",
            "features.6.0.": "down3.0.",
            "features.6.1.": "down3.1.",
        }
        stage_map = {
            "features.1.": "stage1.",
            "features.3.": "stage2.",
            "features.5.": "stage3.",
            "features.7.": "stage4.",
        }
        for key, value in state.items():
            if key.startswith("classifier."):
                continue
            new_key = None
            for old, new in prefix_map.items():
                if key.startswith(old):
                    new_key = new + key[len(old) :]
                    break
            if new_key is None:
                for old, new in stage_map.items():
                    if key.startswith(old):
                        new_key = new + key[len(old) :]
                        break
            if new_key is None:
                continue
            new_key = new_key.replace(".block.2.", ".block.1.")
            new_key = new_key.replace(".block.3.", ".block.2.")
            new_key = new_key.replace(".block.5.", ".block.4.")
            converted[new_key] = value
        return converted

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stage1(self.stem(x))
        p3 = self.stage2(self.down1(x))
        p4 = self.stage3(self.down2(p3))
        p5 = self.stage4(self.down3(p4))
        return p3, p4, p5


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


class SPPCSPCBlock(nn.Module):
    """CSP-style SPP block inspired by YOLOv7, implemented locally for the detector neck."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = out_channels
        self.cv1 = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.cv2 = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.cv3 = ConvBNAct(hidden, hidden, kernel_size=3)
        self.cv4 = ConvBNAct(hidden, hidden, kernel_size=1)
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.cv5 = ConvBNAct(hidden * 4, hidden, kernel_size=1)
        self.cv6 = ConvBNAct(hidden, hidden, kernel_size=3)
        self.cv7 = ConvBNAct(hidden * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1, self.pool5(x1), self.pool9(x1), self.pool13(x1)], dim=1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat([y1, y2], dim=1))


class YoloV7ELANFusion(nn.Module):
    """ELAN-style multi-branch fusion used in YOLOv7 heads, scaled for this small detector."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 32)
        self.short = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.long = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([ConvBNAct(hidden, hidden, kernel_size=3) for _ in range(4)])
        self.fuse = ConvBNAct(hidden * 6, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self.short(x)]
        y = self.long(x)
        outputs.append(y)
        for block in self.blocks:
            y = block(y)
            outputs.append(y)
        return self.fuse(torch.cat(outputs, dim=1))


class YoloV7Downsample(nn.Module):
    """Two-branch downsample: maxpool route plus stride-2 conv route."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        half = max(channels // 2, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_proj = ConvBNAct(channels, half, kernel_size=1)
        self.conv_proj = ConvBNAct(channels, half, kernel_size=1)
        self.conv_down = ConvBNAct(half, half, kernel_size=3, stride=2)
        self.out = ConvBNAct(half * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_proj(self.pool(x))
        conved = self.conv_down(self.conv_proj(x))
        return self.out(torch.cat([conved, pooled], dim=1))


class YoloV7PANNeck(nn.Module):
    """YOLOv7-style SPPCSPC + ELAN FPN/PAN neck adapted to pretrained backbones."""

    def __init__(self, in_channels: list[int], out_channels: int = 192) -> None:
        super().__init__()
        self.lateral3 = ConvBNAct(in_channels[0], out_channels, kernel_size=1)
        self.lateral4 = ConvBNAct(in_channels[1], out_channels, kernel_size=1)
        self.lateral5 = ConvBNAct(in_channels[2], out_channels, kernel_size=1)
        self.spp = SPPCSPCBlock(out_channels, out_channels)
        self.fuse4 = YoloV7ELANFusion(out_channels * 2, out_channels)
        self.fuse3 = YoloV7ELANFusion(out_channels * 2, out_channels)
        self.down3 = YoloV7Downsample(out_channels)
        self.pan4 = YoloV7ELANFusion(out_channels * 2, out_channels)
        self.down4 = YoloV7Downsample(out_channels)
        self.pan5 = YoloV7ELANFusion(out_channels * 2, out_channels)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = features
        p3 = self.lateral3(p3)
        p4 = self.lateral4(p4)
        p5 = self.spp(self.lateral5(p5))

        p5_up = torch.nn.functional.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4_td = self.fuse4(torch.cat([p4, p5_up], dim=1))
        p4_up = torch.nn.functional.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")
        p3_out = self.fuse3(torch.cat([p3, p4_up], dim=1))

        p3_down = self.down3(p3_out)
        p4_out = self.pan4(torch.cat([p4_td, p3_down], dim=1))
        p4_down = self.down4(p4_out)
        p5_out = self.pan5(torch.cat([p5, p4_down], dim=1))
        return [p3_out, p4_out, p5_out]


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


class ConvNeXtFusionBlock(nn.Module):
    """ConvNeXt-style feature fusion block for normalized pretrained ConvNeXt features."""

    def __init__(self, channels: int, expansion: int = 2, layer_scale: float = 1e-6) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels, eps=1e-6)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.norm(y)
        y = self.pw2(self.act(self.pw1(y)))
        return x + self.layer_scale * y


class ConvNeXtProject(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm2d(in_channels, eps=1e-6),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNeXtDownsample(nn.Module):
    """Decouple channel projection and spatial downsampling to keep PAN efficient."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm2d(channels, eps=1e-6),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNeXtPANNeck(nn.Module):
    """FPN/PAN neck using ConvNeXt-style normalization and depthwise large-kernel fusion."""

    def __init__(self, in_channels: list[int], out_channels: int = 192, depth: int = 2) -> None:
        super().__init__()
        self.lateral3 = ConvNeXtProject(in_channels[0], out_channels)
        self.lateral4 = ConvNeXtProject(in_channels[1], out_channels)
        self.lateral5 = ConvNeXtProject(in_channels[2], out_channels)
        self.deep_context = nn.Sequential(*[ConvNeXtFusionBlock(out_channels) for _ in range(max(1, depth))])
        self.fuse4 = nn.Sequential(
            ConvNeXtProject(out_channels * 2, out_channels),
            *[ConvNeXtFusionBlock(out_channels) for _ in range(max(1, depth))],
        )
        self.fuse3 = nn.Sequential(
            ConvNeXtProject(out_channels * 2, out_channels),
            *[ConvNeXtFusionBlock(out_channels) for _ in range(max(1, depth))],
        )
        self.down3 = ConvNeXtDownsample(out_channels)
        self.pan4 = nn.Sequential(
            ConvNeXtProject(out_channels * 2, out_channels),
            *[ConvNeXtFusionBlock(out_channels) for _ in range(max(1, depth))],
        )
        self.down4 = ConvNeXtDownsample(out_channels)
        self.pan5 = nn.Sequential(
            ConvNeXtProject(out_channels * 2, out_channels),
            *[ConvNeXtFusionBlock(out_channels) for _ in range(max(1, depth))],
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
        context_layers: list[nn.Module] = [SPPBlock(out_channels), LargeKernelBlock(out_channels, kernel_size=7)]
        if attention_heads > 0:
            context_layers.append(PartialSelfAttention(out_channels, heads=attention_heads, ratio=0.5))
        self.deep_context = nn.Sequential(*context_layers)
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

    def initialize_biases(self, objectness_prior: float = 0.01) -> None:
        final_conv = self.head[-1]
        if not isinstance(final_conv, nn.Conv2d) or final_conv.bias is None:
            return
        prior = min(max(float(objectness_prior), 1e-4), 1.0 - 1e-4)
        bias = final_conv.bias.view(self.num_anchors, self.pred_dim)
        with torch.no_grad():
            bias[:, 4].fill_(math.log(prior / (1.0 - prior)))

    def initialize_class_biases(self, class_priors: torch.Tensor) -> None:
        final_conv = self.head[-1]
        if not isinstance(final_conv, nn.Conv2d) or final_conv.bias is None:
            return
        log_priors = class_priors.to(final_conv.bias.device, final_conv.bias.dtype).clamp(min=1e-6).log()
        bias = final_conv.bias.view(self.num_anchors, self.pred_dim)
        with torch.no_grad():
            bias[:, 5 : 5 + log_priors.numel()].copy_(log_priors)


class ConvNeXtDetectionHead(nn.Module):
    """Depthwise ConvNeXt-style prediction head with the same YOLO output layout."""

    def __init__(self, in_channels: int, head_channels: int, num_anchors: int, pred_dim: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvNeXtProject(in_channels, head_channels),
            ConvNeXtFusionBlock(head_channels),
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

    def initialize_biases(self, objectness_prior: float = 0.01) -> None:
        final_conv = self.head[-1]
        if not isinstance(final_conv, nn.Conv2d) or final_conv.bias is None:
            return
        prior = min(max(float(objectness_prior), 1e-4), 1.0 - 1e-4)
        bias = final_conv.bias.view(self.num_anchors, self.pred_dim)
        with torch.no_grad():
            bias[:, 4].fill_(math.log(prior / (1.0 - prior)))

    def initialize_class_biases(self, class_priors: torch.Tensor) -> None:
        final_conv = self.head[-1]
        if not isinstance(final_conv, nn.Conv2d) or final_conv.bias is None:
            return
        log_priors = class_priors.to(final_conv.bias.device, final_conv.bias.dtype).clamp(min=1e-6).log()
        bias = final_conv.bias.view(self.num_anchors, self.pred_dim)
        with torch.no_grad():
            bias[:, 5 : 5 + log_priors.numel()].copy_(log_priors)


class DecoupledDetectionHead(nn.Module):
    """Separate regression/objectness and classification towers for task-specific features."""

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        cls_channels: int,
        num_anchors: int,
        pred_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.reg_tower = nn.Sequential(
            ConvBNAct(in_channels, head_channels, kernel_size=3),
            RepConv(head_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(head_channels, num_anchors * 5, kernel_size=1),
        )
        self.cls_tower = nn.Sequential(
            ConvBNAct(in_channels, cls_channels, kernel_size=3),
            RepConv(cls_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cls_channels, num_anchors * num_classes, kernel_size=1),
        )
        self.num_anchors = num_anchors
        self.pred_dim = pred_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reg_obj = self.reg_tower(x)
        cls = self.cls_tower(x)
        n, _, h, w = reg_obj.shape
        reg_obj = reg_obj.view(n, self.num_anchors, 5, h, w)
        cls = cls.view(n, self.num_anchors, self.num_classes, h, w)
        y = torch.cat([reg_obj, cls], dim=2)
        return y.permute(0, 3, 4, 1, 2).contiguous()

    def initialize_biases(self, objectness_prior: float = 0.01) -> None:
        final_reg = self.reg_tower[-1]
        final_cls = self.cls_tower[-1]
        if isinstance(final_reg, nn.Conv2d) and final_reg.bias is not None:
            prior = min(max(float(objectness_prior), 1e-4), 1.0 - 1e-4)
            bias = final_reg.bias.view(self.num_anchors, 5)
            with torch.no_grad():
                bias[:, 4].fill_(math.log(prior / (1.0 - prior)))
        if isinstance(final_cls, nn.Conv2d) and final_cls.bias is not None:
            with torch.no_grad():
                final_cls.bias.zero_()

    def initialize_class_biases(self, class_priors: torch.Tensor) -> None:
        final_cls = self.cls_tower[-1]
        if not isinstance(final_cls, nn.Conv2d) or final_cls.bias is None:
            return
        log_priors = class_priors.to(final_cls.bias.device, final_cls.bias.dtype).clamp(min=1e-6).log()
        bias = final_cls.bias.view(self.num_anchors, self.num_classes)
        with torch.no_grad():
            bias[:, : log_priors.numel()].copy_(log_priors)


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
        objectness_prior: float = 0.01,
        dropout: float = 0.05,
        elan_depth: int = 2,
        aux_head: bool = True,
        decoupled_head: bool = False,
        cls_head_channels: int | None = None,
        neck_type: str = "fpnpan",
        head_type: str = "standard",
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
        elif backbone in {"convnext_small", "convnext_base"}:
            self.convnext = ConvNeXtBackbone(variant=backbone, pretrained=pretrained)
            feature_channels = ConvNeXtBackbone.CONFIGS[backbone]["dims"][1:]
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
        if neck_type == "yolov7_pan":
            self.neck = YoloV7PANNeck(feature_channels, out_channels=neck_channels)
        elif neck_type == "convnext_pan":
            self.neck = ConvNeXtPANNeck(feature_channels, out_channels=neck_channels, depth=max(1, elan_depth))
        else:
            self.neck = FPNPANNeck(feature_channels, out_channels=neck_channels, attention_heads=attention_heads)
        feature_channels = [neck_channels, neck_channels, neck_channels]
        cls_channels = int(cls_head_channels or max(head_channels // 2, 64))
        if decoupled_head:
            head_cls = DecoupledDetectionHead
        elif head_type == "convnext":
            head_cls = ConvNeXtDetectionHead
        else:
            head_cls = DetectionHead

        if decoupled_head:
            self.main_heads = nn.ModuleList(
                [
                    head_cls(feature_channels[0], head_channels, cls_channels, self.num_anchors[0], self.pred_dim, num_classes, dropout),
                    head_cls(feature_channels[1], head_channels, cls_channels, self.num_anchors[1], self.pred_dim, num_classes, dropout),
                    head_cls(feature_channels[2], head_channels, cls_channels, self.num_anchors[2], self.pred_dim, num_classes, dropout),
                ]
            )
        else:
            self.main_heads = nn.ModuleList(
                [
                    head_cls(feature_channels[0], head_channels, self.num_anchors[0], self.pred_dim, dropout),
                    head_cls(feature_channels[1], head_channels, self.num_anchors[1], self.pred_dim, dropout),
                    head_cls(feature_channels[2], head_channels, self.num_anchors[2], self.pred_dim, dropout),
                ]
            )
        self.aux_head_enabled = aux_head
        if decoupled_head:
            aux_cls_channels = max(cls_channels // 2, 32)
            self.aux_heads = nn.ModuleList(
                [
                    DecoupledDetectionHead(feature_channels[0], head_channels // 2, aux_cls_channels, self.num_anchors[0], self.pred_dim, num_classes, dropout),
                    DecoupledDetectionHead(feature_channels[1], head_channels // 2, aux_cls_channels, self.num_anchors[1], self.pred_dim, num_classes, dropout),
                    DecoupledDetectionHead(feature_channels[2], head_channels // 2, aux_cls_channels, self.num_anchors[2], self.pred_dim, num_classes, dropout),
                ]
            )
        else:
            self.aux_heads = nn.ModuleList(
                [
                    DetectionHead(feature_channels[0], head_channels // 2, self.num_anchors[0], self.pred_dim, dropout),
                    DetectionHead(feature_channels[1], head_channels // 2, self.num_anchors[1], self.pred_dim, dropout),
                    DetectionHead(feature_channels[2], head_channels // 2, self.num_anchors[2], self.pred_dim, dropout),
                ]
            )
        for head in list(self.main_heads) + list(self.aux_heads):
            head.initialize_biases(objectness_prior=objectness_prior)

    def initialize_class_biases(self, class_priors: list[float] | torch.Tensor) -> None:
        priors = torch.as_tensor(class_priors, dtype=torch.float32)
        priors = priors / priors.sum().clamp(min=1e-6)
        for head in list(self.main_heads) + list(self.aux_heads):
            if hasattr(head, "initialize_class_biases"):
                head.initialize_class_biases(priors)

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        if self.backbone_name == "resnet50":
            p3, p4, p5 = self.resnet(x)
        elif self.backbone_name.startswith("convnext_"):
            p3, p4, p5 = self.convnext(x)
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
