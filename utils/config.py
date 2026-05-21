from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "image_size": 416,
    "multi_scale": {"enabled": True, "sizes": [352, 384, 416, 448, 480]},
    "epochs": 40,
    "batch_size": 16,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "seed": 42,
    "amp": True,
    "warmup_epochs": 3,
    "label_smoothing": 0.03,
    "freeze_backbone_epochs": 1,
    "early_stopping_patience": 15,
    "augmentation": {
        "random_crop_prob": 0.35,
        "min_crop_scale": 0.75,
        "random_scale_prob": 0.35,
        "min_scale": 0.75,
        "max_scale": 1.25,
    },
    "model": {
        "backbone": "resnet50",
        "pretrained": True,
        "base_channels": 40,
        "head_channels": 256,
        "dropout": 0.05,
        "elan_depth": 2,
        "aux_head": True,
    },
    "anchors": {
        "auto": True,
        "kmeans_iters": 40,
        "per_scale": 3,
        "values": [
            [(10.0, 13.0), (16.0, 24.0), (32.0, 32.0)],
            [(32.0, 48.0), (64.0, 96.0), (96.0, 128.0)],
            [(128.0, 160.0), (220.0, 260.0), (320.0, 320.0)],
        ],
    },
    "class_weights": {"enabled": True},
    "loss_weights": {
        "box_weight": 5.0,
        "obj_weight": 1.0,
        "noobj_weight": 0.5,
        "cls_weight": 1.0,
        "iou_weight": 1.5,
        "aux_weight": 0.4,
        "objectness_focal_gamma": 1.5,
        "iou_aware_objectness": True,
    },
    "inference": {
        "conf_threshold": 0.05,
        "nms_threshold": 0.5,
        "nms_type": "soft",
        "max_detections": 100,
    },
    "validation_metric": {
        "enabled": False,
        "conf_threshold": 0.05,
        "nms_threshold": 0.5,
        "nms_type": "soft",
        "tune": False,
        "tune_every": 10,
        "conf_thresholds": [0.03, 0.05],
        "nms_thresholds": [0.5],
    },
    "ema": {
        "enabled": True,
        "decay": 0.999,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        return config

    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required for YAML config files. Run: pip install -r requirements.txt") from exc
            user_config = yaml.safe_load(f)
        else:
            user_config = json.load(f)
    if user_config is None:
        user_config = {}
    return deep_update(config, user_config)


def get_anchors(config: dict[str, Any]) -> list[list[tuple[float, float]]]:
    anchors = config["anchors"].get("values", config["anchors"]) if isinstance(config["anchors"], dict) else config["anchors"]
    if anchors and anchors[0] and isinstance(anchors[0][0], (int, float)):
        return [[(float(w), float(h)) for w, h in anchors]]
    return [[(float(w), float(h)) for w, h in scale] for scale in anchors]
