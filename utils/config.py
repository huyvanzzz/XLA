from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "image_size": 512,
    "preserve_aspect": False,
    "multi_scale": {"enabled": False, "sizes": [416, 448, 480, 512, 544]},
    "epochs": 80,
    "batch_size": 24,
    "val_batch_size": 64,
    "lr": 1e-4,
    "lr_final_factor": 0.05,
    "backbone_lr_mult": 0.2,
    "weight_decay": 5e-4,
    "num_workers": 4,
    "channels_last": True,
    "cudnn_benchmark": True,
    "seed": 42,
    "amp": True,
    "warmup_epochs": 3,
    "label_smoothing": 0.05,
    "freeze_backbone_epochs": 2,
    "backbone_trainable": "layer4",
    "backbone_freeze_bn": True,
    "early_stopping_patience": 10,
    "validation_loss": {"enabled": False},
    "augmentation": {
        "mosaic_prob": 0.0,
        "close_mosaic_epoch": 55,
        "random_crop_prob": 0.2,
        "min_crop_scale": 0.8,
        "random_scale_prob": 0.45,
        "min_scale": 0.75,
        "max_scale": 1.25,
        "random_erasing_prob": 0.25,
        "random_erasing_min_area": 0.02,
        "random_erasing_max_area": 0.12,
        "close_strong_aug_epoch": 55,
    },
    "model": {
        "backbone": "resnet50",
        "pretrained": True,
        "base_channels": 40,
        "head_channels": 192,
        "neck_channels": 192,
        "attention_heads": 0,
        "objectness_prior": 0.01,
        "dropout": 0.1,
        "elan_depth": 1,
        "neck_type": "yolov7_pan",
        "head_type": "standard",
        "aux_head": True,
        "aux_head_close_epoch": 30,
        "decoupled_head": False,
        "cls_head_channels": 128,
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
    "class_weights": {"enabled": True, "overrides": {"chair": 1.25}},
    "class_prior_bias": {"enabled": True, "smoothing": 1.0},
    "balanced_sampling": {"enabled": True, "power": 0.5, "empty_weight": 0.35},
    "loss_weights": {
        "box_weight": 0.0,
        "obj_weight": 1.0,
        "noobj_weight": 0.25,
        "cls_weight": 1.0,
        "iou_weight": 6.0,
        "aux_weight": 0.4,
        "objectness_focal_gamma": 1.5,
        "iou_aware_objectness": True,
        "assignment_strategy": "legacy",
        "positive_anchor_topk": 3,
        "ignore_anchor_iou": 0.5,
        "objectness_iou_mix": 1.0,
        "decode_style": "yolov7",
        "target_offsets": True,
        "target_offset_bias": 0.5,
        "scale_obj_balance": [4.0, 1.0, 0.4],
        "task_aligned_alpha": 0.5,
        "task_aligned_beta": 6.0,
        "task_aligned_center_radius": 2.5,
        "task_aligned_min_iou": 0.05,
        "noobj_hard_negative_ratio": 0.0,
        "noobj_hard_negative_min": 256,
        "classification_loss": "bce",
    },
    "inference": {
        "conf_threshold": 0.08,
        "class_conf_thresholds": {},
        "tta_hflip": False,
        "nms_threshold": 0.5,
        "nms_type": "diou",
        "merge_nms": True,
        "max_detections": 100,
        "pre_nms_topk": 300,
        "class_pre_nms_topk": 100,
        "decode_style": "yolov7",
        "class_activation": "sigmoid",
    },
    "validation_metric": {
        "enabled": True,
        "start_epoch": 30,
        "every": 1,
        "conf_threshold": 0.08,
        "class_conf_thresholds": {},
        "tta_hflip": False,
        "nms_threshold": 0.5,
        "nms_type": "diou",
        "merge_nms": True,
        "pre_nms_topk": 300,
        "class_pre_nms_topk": 100,
        "decode_style": "yolov7",
        "class_activation": "sigmoid",
        "tune": False,
        "tune_every": 5,
        "conf_thresholds": [0.08, 0.1, 0.12, 0.15],
        "nms_thresholds": [0.45, 0.5, 0.55],
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
