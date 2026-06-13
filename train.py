from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from contextlib import nullcontext
from collections import Counter
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from models.tiny_detector import TinyDetector
from utils.config import get_anchors, load_config
from utils.dataset import DetectionDataset, collate_fn, load_classes
from utils.inference import decode_predictions
from utils.loss import AnchorFreeLoss, YoloLoss
from utils.box_ops import wh_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch tiny YOLO detector.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--classes", default="public/classes.json")
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--backbone_lr_mult", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--box_weight", type=float)
    parser.add_argument("--obj_weight", type=float)
    parser.add_argument("--noobj_weight", type=float)
    parser.add_argument("--cls_weight", type=float)
    parser.add_argument("--iou_weight", type=float)
    parser.add_argument("--aux_weight", type=float)
    parser.add_argument("--objectness_focal_gamma", type=float)
    parser.add_argument("--positive_anchor_topk", type=int)
    parser.add_argument("--ignore_anchor_iou", type=float)
    parser.add_argument("--objectness_iou_mix", type=float)
    parser.add_argument("--noobj_hard_negative_ratio", type=float)
    parser.add_argument("--noobj_hard_negative_min", type=int)
    parser.add_argument("--iou_aware_objectness", action="store_true")
    parser.add_argument("--no_iou_aware_objectness", action="store_true")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> tuple[argparse.Namespace, list[list[tuple[float, float]]], dict, dict]:
    config = load_config(args.config)
    loss_weights = config["loss_weights"]
    for name in [
        "image_size",
        "epochs",
        "batch_size",
        "val_batch_size",
        "lr",
        "backbone_lr_mult",
        "weight_decay",
        "num_workers",
        "seed",
        "warmup_epochs",
        "label_smoothing",
    ]:
        if getattr(args, name) is None:
            setattr(args, name, config[name])
    for name in [
        "box_weight",
        "obj_weight",
        "noobj_weight",
        "cls_weight",
        "iou_weight",
        "aux_weight",
        "objectness_focal_gamma",
        "positive_anchor_topk",
        "ignore_anchor_iou",
        "objectness_iou_mix",
        "noobj_hard_negative_ratio",
        "noobj_hard_negative_min",
        "iou_aware_objectness",
    ]:
        if getattr(args, name) is None:
            setattr(args, name, loss_weights[name])
    if args.no_amp:
        args.amp = False
    elif not args.amp:
        args.amp = bool(config.get("amp", True))
    if args.no_iou_aware_objectness:
        args.iou_aware_objectness = False
    elif not args.iou_aware_objectness:
        args.iou_aware_objectness = bool(loss_weights.get("iou_aware_objectness", True))
    return args, get_anchors(config), config["model"], config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(annotation_path: str, classes: list[str]) -> list[float]:
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    counts = Counter(ann["class"] for ann in data["annotations"])
    values = torch.tensor([float(counts.get(cls, 0) + 1) for cls in classes])
    weights = 1.0 / torch.sqrt(values)
    weights = weights / weights.mean().clamp(min=1e-6)
    return [float(v) for v in weights.tolist()]


def compute_class_priors(annotation_path: str, classes: list[str], smoothing: float = 1.0) -> list[float]:
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    counts = Counter(ann["class"] for ann in data["annotations"])
    values = torch.tensor([float(counts.get(cls, 0)) + float(smoothing) for cls in classes])
    values = values / values.sum().clamp(min=1e-6)
    return [float(v) for v in values.tolist()]


def apply_class_weight_overrides(weights: list[float], classes: list[str], overrides: dict[str, float] | None) -> list[float]:
    if not overrides:
        return weights
    updated = list(weights)
    for class_name, value in overrides.items():
        if class_name in classes:
            updated[classes.index(class_name)] = float(value)
    values = torch.tensor(updated, dtype=torch.float32)
    values = values / values.mean().clamp(min=1e-6)
    return [float(v) for v in values.tolist()]


def build_balanced_sampler(dataset: DetectionDataset, classes: list[str], config: dict) -> WeightedRandomSampler | None:
    if not bool(config.get("enabled", False)):
        return None
    counts = Counter()
    for item in dataset.images:
        counts.update({ann["class"] for ann in item["annotations"]})
    power = float(config.get("power", 0.5))
    empty_weight = float(config.get("empty_weight", 0.35))
    class_weight = {
        class_name: (1.0 / max(float(counts.get(class_name, 1)), 1.0)) ** power
        for class_name in classes
    }
    weights = []
    for item in dataset.images:
        labels = {ann["class"] for ann in item["annotations"]}
        if labels:
            weights.append(max(class_weight[label] for label in labels))
        else:
            weights.append(empty_weight * min(class_weight.values()))
    sample_weights = torch.tensor(weights, dtype=torch.double)
    sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-12)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def fit_auto_anchors(
    annotation_path: str,
    image_size: int,
    per_scale: int = 3,
    iters: int = 40,
    evolve_generations: int = 0,
    anchor_threshold: float = 4.0,
    preserve_aspect: bool = True,
) -> list[list[tuple[float, float]]]:
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    image_info = {item["id"]: item for item in data["images"]}
    wh_list = []
    for ann in data["annotations"]:
        image = image_info[ann["image_id"]]
        x1, y1, x2, y2 = [float(v) for v in ann["bbox"]]
        if preserve_aspect:
            scale = min(image_size / float(image["width"]), image_size / float(image["height"]))
            w = max(x2 - x1, 1.0) * scale
            h = max(y2 - y1, 1.0) * scale
        else:
            w = max(x2 - x1, 1.0) * image_size / float(image["width"])
            h = max(y2 - y1, 1.0) * image_size / float(image["height"])
        wh_list.append([w, h])
    wh = torch.tensor(wh_list, dtype=torch.float32)
    k = per_scale * 3
    order = torch.randperm(wh.shape[0])[:k]
    anchors = wh[order].clone()
    for _ in range(iters):
        assignment = wh_iou(anchors, wh).argmax(dim=0)
        for idx in range(k):
            selected = wh[assignment == idx]
            if selected.numel() > 0:
                anchors[idx] = selected.median(dim=0).values
    if evolve_generations > 0:
        anchors = evolve_anchors(anchors, wh, generations=evolve_generations, threshold=anchor_threshold)
    anchors = anchors[anchors.prod(dim=1).argsort()]
    return [
        [(float(w), float(h)) for w, h in anchors[i * per_scale : (i + 1) * per_scale].tolist()]
        for i in range(3)
    ]


def evolve_anchors(anchors: torch.Tensor, wh: torch.Tensor, generations: int = 150, threshold: float = 4.0) -> torch.Tensor:
    thr = 1.0 / max(float(threshold), 1e-6)

    def fitness(candidate: torch.Tensor) -> torch.Tensor:
        ratio = wh[:, None, :] / candidate[None, :, :].clamp(min=1e-6)
        match = torch.min(ratio, 1.0 / ratio).min(dim=2).values
        best = match.max(dim=1).values
        return (best * (best > thr).float()).mean()

    best = anchors.clone().clamp(min=2.0)
    best_fitness = fitness(best)
    for _ in range(max(0, int(generations))):
        mutation = (torch.randn_like(best) * 0.10 + 1.0).clamp(0.3, 3.0)
        mask = (torch.rand_like(best) < 0.90).float()
        candidate = (best * (mask * mutation + (1.0 - mask))).clamp(min=2.0)
        candidate_fitness = fitness(candidate)
        if candidate_fitness > best_fitness:
            best, best_fitness = candidate, candidate_fitness
    return best


def bbox_iou_list(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_predictions_map(
    ground_truth_path: str,
    predictions: list[dict[str, object]],
    classes: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    with Path(ground_truth_path).open("r", encoding="utf-8") as f:
        gt = json.load(f)
    gt_by_class = {name: {} for name in classes}
    for ann in gt["annotations"]:
        class_name = ann["class"]
        image_id = ann["image_id"]
        gt_by_class[class_name].setdefault(image_id, []).append(
            {"bbox": [float(v) for v in ann["bbox"]], "matched": False}
        )

    pred_by_class = {name: [] for name in classes}
    for item in predictions:
        image_id = str(item["image_id"])
        for box in item["boxes"]:  # type: ignore[index]
            pred_by_class[box["class"]].append(
                {
                    "image_id": image_id,
                    "confidence": float(box["confidence"]),
                    "bbox": [float(v) for v in box["bbox"]],
                }
            )

    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0
    per_class: dict[str, dict[str, float | int]] = {}
    for class_name in classes:
        class_gt = gt_by_class[class_name]
        num_gt = sum(len(v) for v in class_gt.values())
        class_preds = sorted(pred_by_class[class_name], key=lambda x: x["confidence"], reverse=True)
        tp_flags = []
        fp_flags = []
        for pred in class_preds:
            candidates = class_gt.get(pred["image_id"], [])
            best_iou = 0.0
            best_idx = -1
            for idx, candidate in enumerate(candidates):
                if candidate["matched"]:
                    continue
                iou = bbox_iou_list(pred["bbox"], candidate["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                candidates[best_idx]["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        tp_sum = 0
        fp_sum = 0
        recalls = []
        precisions = []
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp
            fp_sum += fp
            recalls.append(tp_sum / num_gt if num_gt else 0.0)
            precisions.append(tp_sum / max(tp_sum + fp_sum, 1))
        ap = compute_ap(recalls, precisions) if num_gt else 0.0
        if num_gt:
            aps.append(ap)
        total_tp += tp_sum
        total_fp += fp_sum
        total_gt += num_gt
        per_class[class_name] = {
            "ap": ap,
            "num_ground_truth": num_gt,
            "num_predictions": len(class_preds),
            "true_positives": tp_sum,
            "false_positives": fp_sum,
            "precision": tp_sum / max(tp_sum + fp_sum, 1),
            "recall": tp_sum / num_gt if num_gt else 0.0,
        }

    return {
        "map50": sum(aps) / len(aps) if aps else 0.0,
        "precision": total_tp / max(total_tp + total_fp, 1),
        "recall": total_tp / total_gt if total_gt else 0.0,
        "per_class": per_class,
    }


def print_per_class_map(title: str, metric_logs: dict[str, object] | None) -> None:
    if not metric_logs or "per_class" not in metric_logs:
        return
    print(title)
    per_class = metric_logs["per_class"]
    if not isinstance(per_class, dict):
        return
    for class_name, values in per_class.items():
        if not isinstance(values, dict):
            continue
        ap = float(values.get("ap", 0.0))
        precision = float(values.get("precision", 0.0))
        recall = float(values.get("recall", 0.0))
        num_gt = int(values.get("num_ground_truth", 0))
        num_pred = int(values.get("num_predictions", 0))
        print(
            f"  {class_name}: AP@0.5={ap:.4f} "
            f"precision={precision:.4f} recall={recall:.4f} "
            f"gt={num_gt} pred={num_pred}"
        )


class ModelEMA:
    def __init__(self, model: TinyDetector, decay: float = 0.999, tau: int = 2000) -> None:
        self.module = deepcopy(model).eval()
        self.decay = decay
        self.tau = int(tau)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: TinyDetector) -> None:
        self.updates += 1
        decay = self.decay if self.tau <= 0 else self.decay * (1.0 - math.exp(-self.updates / self.tau))
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


def _is_backbone_parameter(name: str) -> bool:
    return name.startswith(("resnet", "convnext", "stem", "down3", "down4", "down5", "elan3", "elan4", "elan5"))


def _is_trainable_backbone_parameter(name: str, mode: str) -> bool:
    if not _is_backbone_parameter(name):
        return True
    if mode == "none":
        return False
    if mode == "all":
        return True
    if name.startswith("resnet."):
        if mode == "layer4":
            return name.startswith("resnet.layer4")
        if mode in {"layer3_layer4", "layer34"}:
            return name.startswith(("resnet.layer3", "resnet.layer4"))
        if mode in {"layer2_layer3_layer4", "layer234"}:
            return name.startswith(("resnet.layer2", "resnet.layer3", "resnet.layer4"))
        return False
    if name.startswith("convnext."):
        if mode == "layer4":
            return name.startswith("convnext.stage4")
        if mode in {"layer3_layer4", "layer34"}:
            return name.startswith(("convnext.stage3", "convnext.stage4"))
        if mode in {"layer2_layer3_layer4", "layer234"}:
            return name.startswith(("convnext.stage2", "convnext.stage3", "convnext.stage4"))
        return False
    return mode == "all"


def set_backbone_trainable(model: TinyDetector, warmup_frozen: bool, trainable_mode: str) -> None:
    for name, parameter in model.named_parameters():
        if _is_backbone_parameter(name):
            parameter.requires_grad_(False if warmup_frozen else _is_trainable_backbone_parameter(name, trainable_mode))
        else:
            parameter.requires_grad_(True)


def set_frozen_feature_extractor_eval(model: TinyDetector) -> None:
    for name, module in model.named_children():
        if name in {"resnet", "convnext", "stem", "down3", "down4", "down5", "elan3", "elan4", "elan5"}:
            module.eval()


def set_backbone_batchnorm_eval(model: TinyDetector) -> None:
    for name, module in model.named_modules():
        if _is_backbone_parameter(name) and isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def run_epoch(
    model: TinyDetector,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    epoch: int = 0,
    total_epochs: int = 0,
    ema: ModelEMA | None = None,
    freeze_backbone: bool = False,
    freeze_backbone_bn: bool = False,
    channels_last: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    if hasattr(criterion, "current_epoch"):
        criterion.current_epoch = int(epoch)
    model.train(training)
    if training and freeze_backbone:
        set_frozen_feature_extractor_eval(model)
    if training and freeze_backbone_bn:
        set_backbone_batchnorm_eval(model)
    totals: dict[str, float] = {}
    steps = 0
    phase = "train" if training else "val"
    progress = tqdm(
        loader,
        desc=f"{phase} {epoch}/{total_epochs}" if total_epochs else phase,
        leave=False,
        dynamic_ncols=True,
    )

    for images, targets in progress:
        images = images.to(device, memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            autocast_enabled = use_amp and device.type == "cuda"
            amp_context = torch.amp.autocast("cuda", enabled=True) if autocast_enabled else nullcontext()
            with amp_context:
                pred = model(images)
                loss, logs = criterion(pred, targets)
            if training:
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                if ema is not None:
                    ema.update(model)

        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        steps += 1
        postfix = dict(
            loss=f"{logs['loss']:.3f}",
            box=f"{logs['box_loss']:.3f}",
            iou=f"{logs['iou_loss']:.3f}",
            obj=f"{logs['obj_loss']:.3f}",
            cls=f"{logs['cls_loss']:.3f}",
        )
        if "num_pos" in logs:
            postfix["pos"] = f"{logs['num_pos']:.0f}"
        if "mean_quality" in logs:
            postfix["quality"] = f"{logs['mean_quality']:.3f}"
        progress.set_postfix(**postfix)

    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_map(
    model: TinyDetector,
    loader: DataLoader,
    ground_truth_path: str,
    classes: list[str],
    anchors: list[list[tuple[float, float]]],
    image_size: int,
    device: torch.device,
    conf_threshold: float,
    class_conf_thresholds: dict[str, float] | None,
    nms_threshold: float,
    nms_type: str,
    merge_nms: bool,
    pre_nms_topk: int,
    class_pre_nms_topk: int,
    preserve_aspect: bool,
    channels_last: bool,
    decode_style: str,
    class_activation: str,
    quality_score_power: float,
) -> dict[str, object]:
    model.eval()
    predictions = []
    progress = tqdm(loader, desc="mAP eval", leave=False, dynamic_ncols=True)
    for images, targets in progress:
        images = images.to(device, memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        outputs = model(images)
        for idx, target in enumerate(targets):
            boxes = decode_predictions(
                {"main": [scale[idx : idx + 1] for scale in outputs["main"]]},
                classes=classes,
                anchors=anchors,
                image_size=image_size,
                orig_width=int(target["orig_width"]),
                orig_height=int(target["orig_height"]),
                conf_threshold=conf_threshold,
                class_conf_thresholds=class_conf_thresholds,
                nms_threshold=nms_threshold,
                nms_type=nms_type,
                merge_nms=merge_nms,
                pre_nms_topk=pre_nms_topk,
                class_pre_nms_topk=class_pre_nms_topk,
                preserve_aspect=preserve_aspect,
                decode_style=decode_style,
                class_activation=class_activation,
                quality_score_power=quality_score_power,
            )
            predictions.append({"image_id": str(target["image_id"]), "boxes": boxes})

    result = evaluate_predictions_map(ground_truth_path, predictions, classes, iou_threshold=0.5)
    result["conf_threshold"] = conf_threshold
    result["nms_threshold"] = nms_threshold
    return result


def evaluate_map_with_optional_tuning(
    model: TinyDetector,
    loader: DataLoader,
    ground_truth_path: str,
    classes: list[str],
    anchors: list[list[tuple[float, float]]],
    image_size: int,
    device: torch.device,
    metric_config: dict,
    epoch: int,
) -> dict[str, object]:
    should_tune = bool(metric_config.get("tune", False)) and epoch % int(metric_config.get("tune_every", 1)) == 0
    conf_values = metric_config["conf_thresholds"] if should_tune else [metric_config["conf_threshold"]]
    nms_values = metric_config["nms_thresholds"] if should_tune else [metric_config["nms_threshold"]]
    best = None
    for conf_threshold in conf_values:
        for nms_threshold in nms_values:
            result = evaluate_map(
                model,
                loader,
                ground_truth_path,
                classes,
                anchors,
                image_size,
                device,
                conf_threshold=float(conf_threshold),
                class_conf_thresholds=metric_config.get("class_conf_thresholds", {}),
                nms_threshold=float(nms_threshold),
                nms_type=str(metric_config.get("nms_type", "soft")),
                merge_nms=bool(metric_config.get("merge_nms", False)),
                pre_nms_topk=int(metric_config.get("pre_nms_topk", 1000)),
                class_pre_nms_topk=int(metric_config.get("class_pre_nms_topk", 100)),
                preserve_aspect=bool(metric_config.get("preserve_aspect", True)),
                channels_last=bool(metric_config.get("channels_last", False)),
                decode_style=str(metric_config.get("decode_style", "standard")),
                class_activation=str(metric_config.get("class_activation", "softmax")),
                quality_score_power=float(metric_config.get("quality_score_power", 0.0)),
            )
            if best is None or result["map50"] > best["map50"]:
                best = result
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    args, anchors, model_config, config = apply_config(args)
    seed_everything(args.seed)
    preserve_aspect = bool(config.get("preserve_aspect", True))
    config["validation_metric"]["preserve_aspect"] = preserve_aspect
    if str(model_config.get("architecture", "yolo")) == "anchor_free":
        config["inference"]["decode_style"] = "anchor_free"
        config["validation_metric"]["decode_style"] = "anchor_free"
        loss_weights = config["loss_weights"]
        loss_weights["decode_style"] = "anchor_free"
    channels_last = bool(config.get("channels_last", True)) and torch.cuda.is_available()
    config["validation_metric"]["channels_last"] = channels_last
    if bool(config.get("cudnn_benchmark", True)) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    classes = load_classes(args.classes)
    if isinstance(config["anchors"], dict) and config["anchors"].get("auto", False):
        anchors = fit_auto_anchors(
            args.train_data,
            image_size=args.image_size,
            per_scale=int(config["anchors"].get("per_scale", 3)),
            iters=int(config["anchors"].get("kmeans_iters", 40)),
            evolve_generations=int(config["anchors"].get("evolve_generations", 0)),
            anchor_threshold=float(config["anchors"].get("anchor_threshold", 4.0)),
            preserve_aspect=preserve_aspect,
        )
        if str(model_config.get("architecture", "yolo")) == "anchor_free":
            anchors = [[(1.0, 1.0)], [(1.0, 1.0)], [(1.0, 1.0)]]
        else:
            print(f"auto anchors: {anchors}")
    class_weights = None
    if config["class_weights"]["enabled"]:
        class_weights = compute_class_weights(args.train_data, classes)
        class_weights = apply_class_weight_overrides(class_weights, classes, config["class_weights"].get("overrides"))
        print(f"class weights: {class_weights}")
    loss_weights = config["loss_weights"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_set = DetectionDataset(
        args.train_data,
        args.image_dir,
        classes,
        args.image_size,
        augment=True,
        augment_config=config["augmentation"],
        preserve_aspect=preserve_aspect,
    )
    val_set = DetectionDataset(
        args.val_data,
        args.val_image_dir,
        classes,
        args.image_size,
        augment=False,
        preserve_aspect=preserve_aspect,
    )
    train_sampler = build_balanced_sampler(train_set, classes, config.get("balanced_sampling", {}))
    if train_sampler is not None:
        print("balanced sampling: enabled")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = TinyDetector(num_classes=len(classes), num_anchors=[len(scale) for scale in anchors], **model_config).to(device)
    if bool(config.get("class_prior_bias", {}).get("enabled", False)):
        class_priors = compute_class_priors(
            args.train_data,
            classes,
            smoothing=float(config.get("class_prior_bias", {}).get("smoothing", 1.0)),
        )
        model.initialize_class_biases(class_priors)
        print(f"class prior bias: {class_priors}")
    if bool(config.get("objectness_bias", {}).get("enabled", False)):
        objectness_logits = model.initialize_scale_objectness_biases(
            image_size=args.image_size,
            nominal_objects=float(config.get("objectness_bias", {}).get("nominal_objects", 8.0)),
        )
        print(f"scale objectness bias logits: {[round(v, 4) for v in objectness_logits]}")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if str(model_config.get("architecture", "yolo")) == "anchor_free":
        criterion = AnchorFreeLoss(
            image_size=args.image_size,
            num_classes=len(classes),
            cls_weight=args.cls_weight,
            iou_weight=args.iou_weight,
            task_aligned_alpha=float(loss_weights.get("task_aligned_alpha", 0.5)),
            task_aligned_beta=float(loss_weights.get("task_aligned_beta", 6.0)),
            task_aligned_center_radius=float(loss_weights.get("task_aligned_center_radius", 2.5)),
            positive_anchor_topk=int(loss_weights.get("positive_anchor_topk", 10)),
            class_weights=class_weights,
            quality_focal_loss=str(loss_weights.get("quality_focal_loss", "qfl")),
            quality_focal_beta=float(loss_weights.get("quality_focal_beta", 2.0)),
            varifocal_alpha=float(loss_weights.get("varifocal_alpha", 0.75)),
            varifocal_gamma=float(loss_weights.get("varifocal_gamma", 2.0)),
            assignment_warmup_epochs=int(loss_weights.get("assignment_warmup_epochs", 5)),
        ).to(device)
    else:
        criterion = YoloLoss(
            anchors,
            image_size=args.image_size,
            num_classes=len(classes),
            box_weight=args.box_weight,
            obj_weight=args.obj_weight,
            noobj_weight=args.noobj_weight,
            cls_weight=args.cls_weight,
            iou_weight=args.iou_weight,
            aux_weight=args.aux_weight,
            objectness_focal_gamma=args.objectness_focal_gamma,
            iou_aware_objectness=args.iou_aware_objectness,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            positive_anchor_topk=args.positive_anchor_topk,
            ignore_anchor_iou=args.ignore_anchor_iou,
            objectness_iou_mix=args.objectness_iou_mix,
            noobj_hard_negative_ratio=args.noobj_hard_negative_ratio,
            noobj_hard_negative_min=args.noobj_hard_negative_min,
            assignment_strategy=str(loss_weights.get("assignment_strategy", "legacy")),
            assignment_warmup_epochs=int(loss_weights.get("assignment_warmup_epochs", 0)),
            task_aligned_alpha=float(loss_weights.get("task_aligned_alpha", 0.5)),
            task_aligned_beta=float(loss_weights.get("task_aligned_beta", 6.0)),
            task_aligned_center_radius=float(loss_weights.get("task_aligned_center_radius", 2.5)),
            task_aligned_min_iou=float(loss_weights.get("task_aligned_min_iou", 0.05)),
            decode_style=str(loss_weights.get("decode_style", "standard")),
            target_offsets=bool(loss_weights.get("target_offsets", False)),
            target_offset_bias=float(loss_weights.get("target_offset_bias", 0.5)),
            scale_obj_balance=loss_weights.get("scale_obj_balance"),
            classification_loss=str(loss_weights.get("classification_loss", "ce")),
            classification_quality_mix=float(loss_weights.get("classification_quality_mix", 0.0)),
            classification_focal_gamma=float(loss_weights.get("classification_focal_gamma", 0.0)),
            quality_weight=float(loss_weights.get("quality_weight", 0.0)),
        ).to(device)
    decay_backbone = []
    decay_detector = []
    no_decay_backbone = []
    no_decay_detector = []
    for module_name, module in model.named_modules():
        for param_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            is_backbone = full_name.startswith(("resnet", "convnext", "stem", "down3", "down4", "down5", "elan3", "elan4", "elan5"))
            is_no_decay = param_name.endswith("bias") or isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm))
            if is_backbone and is_no_decay:
                no_decay_backbone.append(parameter)
            elif is_backbone:
                decay_backbone.append(parameter)
            elif is_no_decay:
                no_decay_detector.append(parameter)
            else:
                decay_detector.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_backbone, "lr": args.lr * args.backbone_lr_mult, "weight_decay": args.weight_decay},
            {"params": no_decay_backbone, "lr": args.lr * args.backbone_lr_mult, "weight_decay": 0.0},
            {"params": decay_detector, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": no_decay_detector, "lr": args.lr, "weight_decay": 0.0},
        ],
    )
    def lr_lambda(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        final_factor = float(config.get("lr_final_factor", 0.0))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        return final_factor + (1.0 - final_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = (
        ModelEMA(model, decay=float(config["ema"]["decay"]), tau=int(config["ema"].get("tau", 2000)))
        if config["ema"]["enabled"]
        else None
    )
    freeze_backbone_epochs = int(config["freeze_backbone_epochs"])
    backbone_trainable = str(config.get("backbone_trainable", "layer4"))
    late_backbone_config = config.get("late_backbone", {})
    late_backbone_enabled = bool(late_backbone_config.get("enabled", False))
    late_backbone_start_epoch = int(late_backbone_config.get("start_epoch", 0))
    late_backbone_trainable = str(late_backbone_config.get("trainable", backbone_trainable))
    freeze_backbone_bn = bool(config.get("backbone_freeze_bn", True))
    early_stopping_patience = int(config["early_stopping_patience"])

    use_map_for_best = bool(config["validation_metric"]["enabled"])
    use_val_loss = bool(config.get("validation_loss", {}).get("enabled", not use_map_for_best))
    if not use_map_for_best and not use_val_loss:
        raise ValueError("Enable validation_metric or validation_loss so best checkpoint can be selected.")
    best_score = float("-inf") if use_map_for_best else float("inf")
    best_metric_logs = None
    best_epoch = 0
    epochs_without_improvement = 0
    mosaic_closed = False
    strong_aug_closed = False
    aux_head_closed = False
    for epoch in range(1, args.epochs + 1):
        close_mosaic_epoch = int(config["augmentation"].get("close_mosaic_epoch", 0))
        if close_mosaic_epoch > 0 and epoch >= close_mosaic_epoch and not mosaic_closed:
            train_set.augment_config["mosaic_prob"] = 0.0
            mosaic_closed = True
            print(f"closing mosaic augmentation at epoch {epoch}")
        close_strong_aug_epoch = int(config["augmentation"].get("close_strong_aug_epoch", 0))
        if close_strong_aug_epoch > 0 and epoch >= close_strong_aug_epoch and not strong_aug_closed:
            for key in ["random_crop_prob", "random_scale_prob", "random_erasing_prob"]:
                train_set.augment_config[key] = 0.0
            strong_aug_closed = True
            print(f"closing strong augmentation at epoch {epoch}")

        if config["multi_scale"]["enabled"]:
            train_size = int(random.choice(config["multi_scale"]["sizes"]))
            train_set.image_size = train_size
            criterion.image_size = train_size
            print(f"multi-scale train image_size={train_size}")
        else:
            criterion.image_size = args.image_size

        freeze_backbone = epoch <= freeze_backbone_epochs
        active_backbone_trainable = backbone_trainable
        if late_backbone_enabled and late_backbone_start_epoch > 0 and epoch >= late_backbone_start_epoch:
            active_backbone_trainable = late_backbone_trainable
            if epoch == late_backbone_start_epoch:
                print(f"late unfreezing backbone mode={active_backbone_trainable} at epoch {epoch}")
        set_backbone_trainable(model, freeze_backbone, active_backbone_trainable)
        if epoch == 1 and freeze_backbone:
            print(f"freezing backbone for first {freeze_backbone_epochs} epoch(s)")
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
            print(f"unfreezing backbone mode={active_backbone_trainable}")
        aux_head_close_epoch = int(model_config.get("aux_head_close_epoch", 0))
        if bool(model_config.get("aux_head", True)) and aux_head_close_epoch > 0:
            model.aux_head_enabled = epoch < aux_head_close_epoch
            if epoch >= aux_head_close_epoch and not aux_head_closed:
                aux_head_closed = True
                print(f"closing auxiliary detection heads at epoch {epoch}")

        train_start = time.perf_counter()
        train_logs = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            args.amp,
            epoch=epoch,
            total_epochs=args.epochs,
            ema=ema,
            freeze_backbone=freeze_backbone,
            freeze_backbone_bn=freeze_backbone_bn,
            channels_last=channels_last,
        )
        train_time = time.perf_counter() - train_start
        eval_model = ema.module if ema is not None else model
        criterion.image_size = args.image_size
        val_logs = None
        if use_val_loss:
            with torch.no_grad():
                val_logs = run_epoch(
                    eval_model,
                    val_loader,
                    criterion,
                    device,
                    use_amp=args.amp,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    channels_last=channels_last,
                )
        scheduler.step()

        log_line = f"epoch {epoch:03d}/{args.epochs} train_loss={train_logs['loss']:.4f} train_time={train_time:.1f}s"
        if val_logs is not None:
            log_line += (
                f" val_loss={val_logs['loss']:.4f} "
                f"box={val_logs['box_loss']:.4f} "
                f"iou={val_logs['iou_loss']:.4f} "
                f"obj={val_logs['obj_loss']:.4f} "
                f"noobj={val_logs['noobj_loss']:.4f} "
                f"cls={val_logs['cls_loss']:.4f}"
            )
        print(log_line)

        metric_logs = None
        metric_every = max(1, int(config["validation_metric"].get("every", 1)))
        metric_start_epoch = max(1, int(config["validation_metric"].get("start_epoch", 1)))
        metric_started = epoch >= metric_start_epoch
        should_eval_map = (
            bool(config["validation_metric"]["enabled"])
            and metric_started
            and (epoch % metric_every == 0 or epoch == args.epochs)
        )
        if should_eval_map:
            metric_logs = evaluate_map_with_optional_tuning(
                eval_model,
                val_loader,
                args.val_data,
                classes,
                anchors,
                args.image_size,
                device,
                config["validation_metric"],
                epoch,
            )
            print(
                f"val_mAP@0.5={metric_logs['map50']:.4f} "
                f"precision={metric_logs['precision']:.4f} "
                f"recall={metric_logs['recall']:.4f} "
                f"conf={metric_logs['conf_threshold']:.2f} "
                f"nms={metric_logs['nms_threshold']:.2f}"
            )
            print_per_class_map("per-class AP@0.5:", metric_logs)

        state = {
            "model": model.state_dict(),
            "raw_model": model.state_dict(),
            "classes": classes,
            "anchors": anchors,
            "image_size": args.image_size,
            "preserve_aspect": preserve_aspect,
            "model_config": model_config,
            "loss_weights": {
                "box_weight": args.box_weight,
                "obj_weight": args.obj_weight,
                "noobj_weight": args.noobj_weight,
                "cls_weight": args.cls_weight,
                "iou_weight": args.iou_weight,
                "aux_weight": args.aux_weight,
                "objectness_focal_gamma": args.objectness_focal_gamma,
                "iou_aware_objectness": args.iou_aware_objectness,
                "positive_anchor_topk": args.positive_anchor_topk,
                "ignore_anchor_iou": args.ignore_anchor_iou,
                "objectness_iou_mix": args.objectness_iou_mix,
                "noobj_hard_negative_ratio": args.noobj_hard_negative_ratio,
                "noobj_hard_negative_min": args.noobj_hard_negative_min,
                "assignment_strategy": str(loss_weights.get("assignment_strategy", "legacy")),
                "assignment_warmup_epochs": int(loss_weights.get("assignment_warmup_epochs", 0)),
                "task_aligned_alpha": float(loss_weights.get("task_aligned_alpha", 0.5)),
                "task_aligned_beta": float(loss_weights.get("task_aligned_beta", 6.0)),
                "task_aligned_center_radius": float(loss_weights.get("task_aligned_center_radius", 2.5)),
                "task_aligned_min_iou": float(loss_weights.get("task_aligned_min_iou", 0.05)),
                "decode_style": str(loss_weights.get("decode_style", "standard")),
                "target_offsets": bool(loss_weights.get("target_offsets", False)),
                "target_offset_bias": float(loss_weights.get("target_offset_bias", 0.5)),
                "scale_obj_balance": loss_weights.get("scale_obj_balance"),
                "classification_loss": str(loss_weights.get("classification_loss", "ce")),
                "classification_quality_mix": float(loss_weights.get("classification_quality_mix", 0.0)),
                "classification_focal_gamma": float(loss_weights.get("classification_focal_gamma", 0.0)),
                "quality_weight": float(loss_weights.get("quality_weight", 0.0)),
                "quality_focal_loss": str(loss_weights.get("quality_focal_loss", "qfl")),
                "quality_focal_beta": float(loss_weights.get("quality_focal_beta", 2.0)),
            },
            "lr": args.lr,
            "backbone_lr_mult": args.backbone_lr_mult,
            "amp": args.amp,
            "label_smoothing": args.label_smoothing,
            "ema_enabled": ema is not None,
            "freeze_backbone_epochs": freeze_backbone_epochs,
            "backbone_trainable": active_backbone_trainable,
            "base_backbone_trainable": backbone_trainable,
            "late_backbone": late_backbone_config,
            "backbone_freeze_bn": freeze_backbone_bn,
            "aux_head_close_epoch": int(model_config.get("aux_head_close_epoch", 0)),
            "epoch": epoch,
            "val_loss": val_logs["loss"] if val_logs is not None else None,
            "val_map50": float(metric_logs["map50"]) if metric_logs is not None else None,
            "per_class_map50": metric_logs.get("per_class") if metric_logs is not None else None,
            "best_conf_threshold": metric_logs["conf_threshold"] if metric_logs is not None else config["inference"]["conf_threshold"],
            "best_nms_threshold": metric_logs["nms_threshold"] if metric_logs is not None else config["inference"]["nms_threshold"],
            "nms_type": config["validation_metric"].get("nms_type", config["inference"].get("nms_type", "soft")),
            "merge_nms": config["validation_metric"].get("merge_nms", config["inference"].get("merge_nms", False)),
            "pre_nms_topk": config["validation_metric"].get("pre_nms_topk", config["inference"].get("pre_nms_topk", 1000)),
            "class_pre_nms_topk": config["validation_metric"].get("class_pre_nms_topk", config["inference"].get("class_pre_nms_topk", 100)),
            "decode_style": config["validation_metric"].get("decode_style", config["inference"].get("decode_style", "standard")),
            "class_activation": config["validation_metric"].get("class_activation", config["inference"].get("class_activation", "softmax")),
            "quality_score_power": config["validation_metric"].get("quality_score_power", config["inference"].get("quality_score_power", 0.0)),
        }
        if ema is not None:
            state["model"] = ema.module.state_dict()

        if use_map_for_best and metric_logs is None:
            continue
        torch.save(state, checkpoint_dir / "last.pth")

        current_score = float(metric_logs["map50"]) if use_map_for_best else float(val_logs["loss"])
        improved = current_score > best_score if use_map_for_best else current_score < best_score
        if improved:
            best_score = current_score
            best_metric_logs = metric_logs
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(state, checkpoint_dir / "best.pth")
            metric_name = "mAP@0.5" if use_map_for_best else "val_loss"
            print(f"saved best checkpoint: {checkpoint_dir / 'best.pth'} ({metric_name}={current_score:.4f})")
            if use_map_for_best:
                print_per_class_map("best per-class AP@0.5:", best_metric_logs)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                metric_name = "mAP@0.5" if use_map_for_best else "val_loss"
                print(f"early stopping after {early_stopping_patience} epoch(s) without {metric_name} improvement")
                break

    if use_map_for_best and best_metric_logs is not None:
        print(f"best validation summary: epoch={best_epoch} mAP@0.5={best_score:.4f}")
        print_per_class_map("best per-class AP@0.5:", best_metric_logs)


if __name__ == "__main__":
    main()
