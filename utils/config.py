from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "image_size": 448,
    "preserve_aspect": False,
    "multi_scale": {"enabled": False, "sizes": [416, 448, 480, 512, 544]},
    "epochs": 80,
    "batch_size": 24,
    "val_batch_size": 64,
    "lr": 1.4e-4,
    "lr_final_factor": 0.05,
    "fine_tune_lr_factor": 0.5,
    "backbone_lr_mult": 0.1,
    "weight_decay": 1e-2,
    "num_workers": 4,
    "channels_last": True,
    "cudnn_benchmark": True,
    "seed": 42,
    "amp": True,
    "warmup_epochs": 3,
    "label_smoothing": 0.02,
    "freeze_backbone_epochs": 2,
    "backbone_trainable": "layer4",
    "late_backbone": {
        "enabled": True,
        "start_epoch": 12,
        "trainable": "stage3_tail_layer4",
    },
    "backbone_freeze_bn": True,
    "early_stopping_patience": 20,
    "validation_loss": {"enabled": False},
    "augmentation": {
        "mosaic_prob": 0.08,
        "close_mosaic_epoch": 28,
        "random_crop_prob": 0.12,
        "min_crop_scale": 0.85,
        "random_scale_prob": 0.3,
        "min_scale": 0.85,
        "max_scale": 1.15,
        "random_erasing_prob": 0.0,
        "random_erasing_min_area": 0.02,
        "random_erasing_max_area": 0.1,
        "color_jitter_prob": 0.3,
        "color_jitter_min": 0.8,
        "color_jitter_max": 1.2,
        "close_strong_aug_epoch": 28,
    },
    "model": {
        "architecture": "anchor_free",
        "backbone": "convnextv2_nano",
        "pretrained": True,
        "base_channels": 40,
        "head_channels": 160,
        "neck_channels": 128,
        "attention_heads": 0,
        "objectness_prior": 0.01,
        "dropout": 0.08,
        "elan_depth": 1,
        "neck_type": "convnext_fast_pan",
        "neck_attention": "eca",
        "head_attention": "eca",
        "head_coordconv": False,
        "head_refine": True,
        "head_type": "efficient_decoupled",
        "aux_head": False,
        "aux_head_close_epoch": 30,
        "decoupled_head": False,
        "cls_head_channels": 96,
        "quality_head": False,
        "reg_initial_distance": 4.0,
        "reg_max": 8,
        "localization_quality": False,
        "drop_path_rate": 0.1,
    },
    "anchors": {
        "auto": False,
        "kmeans_iters": 40,
        "evolve_generations": 0,
        "anchor_threshold": 4.0,
        "per_scale": 3,
        "values": [
            [(10.0, 13.0), (16.0, 24.0), (32.0, 32.0)],
            [(32.0, 48.0), (64.0, 96.0), (96.0, 128.0)],
            [(128.0, 160.0), (220.0, 260.0), (320.0, 320.0)],
        ],
    },
    "class_weights": {"enabled": True, "overrides": {"chair": 1.25}},
    "class_prior_bias": {"enabled": True, "smoothing": 1.0},
    "objectness_bias": {"enabled": False, "nominal_objects": 8.0},
    "balanced_sampling": {"enabled": False, "power": 0.5, "empty_weight": 0.35},
    "loss_weights": {
        "box_loss_type": "siou",
        "box_weight": 0.0,
        "obj_weight": 0.0,
        "noobj_weight": 0.0,
        "cls_weight": 1.0,
        "iou_weight": 5.0,
        "aux_weight": 0.0,
        "objectness_focal_gamma": 1.5,
        "iou_aware_objectness": True,
        "assignment_strategy": "task_aligned",
        "positive_anchor_topk": 13,
        "ignore_anchor_iou": 0.5,
        "objectness_iou_mix": 0.7,
        "decode_style": "anchor_free",
        "target_offsets": True,
        "target_offset_bias": 0.5,
        "scale_obj_balance": [4.0, 1.0, 0.4],
        "task_aligned_alpha": 1.0,
        "task_aligned_beta": 6.0,
        "task_aligned_center_radius": 2.5,
        "task_aligned_min_iou": 0.02,
        "noobj_hard_negative_ratio": 0.0,
        "noobj_hard_negative_min": 256,
        "classification_loss": "bce",
        "classification_quality_mix": 0.15,
        "classification_focal_gamma": 0.3,
        "quality_weight": 0.0,
        "quality_focal_loss": "qfl",
        "quality_focal_beta": 2.0,
        "dfl_weight": 0.5,
        "localization_quality_weight": 0.0,
        "varifocal_alpha": 0.75,
        "varifocal_gamma": 2.0,
        "assignment_warmup_epochs": 5,
    },
    "inference": {
        "conf_threshold": 0.01,
        "class_conf_thresholds": {},
        "tta_hflip": True,
        "tta_fusion": "wbf",
        "tta_iou_threshold": 0.55,
        "nms_threshold": 0.5,
        "nms_type": "hard",
        "merge_nms": False,
        "max_detections": 100,
        "pre_nms_topk": 500,
        "class_pre_nms_topk": 120,
        "decode_style": "anchor_free",
        "class_activation": "sigmoid",
        "quality_score_power": 0.0,
    },
    "validation_metric": {
        "enabled": True,
        "start_epoch": 1,
        "every": 1,
        "conf_threshold": 0.01,
        "class_conf_thresholds": {},
        "tta_hflip": True,
        "tta_fusion": "wbf",
        "tta_iou_threshold": 0.55,
        "nms_threshold": 0.5,
        "nms_type": "hard",
        "merge_nms": False,
        "pre_nms_topk": 500,
        "class_pre_nms_topk": 120,
        "decode_style": "anchor_free",
        "class_activation": "sigmoid",
        "quality_score_power": 0.0,
        "tune": False,
        "tune_every": 5,
        "conf_thresholds": [0.08, 0.1, 0.12, 0.15],
        "nms_thresholds": [0.45, 0.5, 0.55],
    },
    "ema": {
        "enabled": True,
        "decay": 0.999,
        "tau": 2000,
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
