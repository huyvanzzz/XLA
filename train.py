from __future__ import annotations

import argparse
import json
from copy import deepcopy
from contextlib import nullcontext
from collections import Counter
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.tiny_detector import TinyDetector
from utils.config import get_anchors, load_config
from utils.dataset import DetectionDataset, collate_fn, load_classes
from utils.inference import decode_predictions
from utils.loss import YoloLoss
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
    parser.add_argument("--lr", type=float)
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
        "lr",
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


def fit_auto_anchors(
    annotation_path: str,
    image_size: int,
    per_scale: int = 3,
    iters: int = 40,
) -> list[list[tuple[float, float]]]:
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    image_info = {item["id"]: item for item in data["images"]}
    wh_list = []
    for ann in data["annotations"]:
        image = image_info[ann["image_id"]]
        x1, y1, x2, y2 = [float(v) for v in ann["bbox"]]
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
    anchors = anchors[anchors.prod(dim=1).argsort()]
    return [
        [(float(w), float(h)) for w, h in anchors[i * per_scale : (i + 1) * per_scale].tolist()]
        for i in range(3)
    ]


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
) -> dict[str, float]:
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
        if num_gt:
            aps.append(compute_ap(recalls, precisions))
        total_tp += tp_sum
        total_fp += fp_sum
        total_gt += num_gt

    return {
        "map50": sum(aps) / len(aps) if aps else 0.0,
        "precision": total_tp / max(total_tp + total_fp, 1),
        "recall": total_tp / total_gt if total_gt else 0.0,
    }


class ModelEMA:
    def __init__(self, model: TinyDetector, decay: float = 0.999) -> None:
        self.module = deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: TinyDetector) -> None:
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def set_backbone_frozen(model: TinyDetector, frozen: bool) -> None:
    head_prefixes = ("main_heads", "aux_heads")
    for name, parameter in model.named_parameters():
        is_head = name.startswith(head_prefixes)
        parameter.requires_grad_(is_head or not frozen)


def set_frozen_feature_extractor_eval(model: TinyDetector) -> None:
    for name, module in model.named_children():
        if name not in {"main_heads", "aux_heads"}:
            module.eval()


def run_epoch(
    model: TinyDetector,
    loader: DataLoader,
    criterion: YoloLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    epoch: int = 0,
    total_epochs: int = 0,
    ema: ModelEMA | None = None,
    freeze_backbone: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        set_frozen_feature_extractor_eval(model)
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
        images = images.to(device)
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
        progress.set_postfix(
            loss=f"{logs['loss']:.3f}",
            box=f"{logs['box_loss']:.3f}",
            iou=f"{logs['iou_loss']:.3f}",
            obj=f"{logs['obj_loss']:.3f}",
            cls=f"{logs['cls_loss']:.3f}",
        )

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
    nms_threshold: float,
    nms_type: str,
) -> dict[str, float]:
    model.eval()
    predictions = []
    progress = tqdm(loader, desc="mAP eval", leave=False, dynamic_ncols=True)
    for images, targets in progress:
        images = images.to(device)
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
                nms_threshold=nms_threshold,
                nms_type=nms_type,
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
) -> dict[str, float]:
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
                nms_threshold=float(nms_threshold),
                nms_type=str(metric_config.get("nms_type", "soft")),
            )
            if best is None or result["map50"] > best["map50"]:
                best = result
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    args, anchors, model_config, config = apply_config(args)
    seed_everything(args.seed)

    classes = load_classes(args.classes)
    if isinstance(config["anchors"], dict) and config["anchors"].get("auto", False):
        anchors = fit_auto_anchors(
            args.train_data,
            image_size=args.image_size,
            per_scale=int(config["anchors"].get("per_scale", 3)),
            iters=int(config["anchors"].get("kmeans_iters", 40)),
        )
        print(f"auto anchors: {anchors}")
    class_weights = None
    if config["class_weights"]["enabled"]:
        class_weights = compute_class_weights(args.train_data, classes)
        print(f"class weights: {class_weights}")
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
    )
    val_set = DetectionDataset(args.val_data, args.val_image_dir, classes, args.image_size, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = TinyDetector(num_classes=len(classes), num_anchors=[len(scale) for scale in anchors], **model_config).to(device)
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
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_lambda(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = ModelEMA(model, decay=float(config["ema"]["decay"])) if config["ema"]["enabled"] else None
    freeze_backbone_epochs = int(config["freeze_backbone_epochs"])
    early_stopping_patience = int(config["early_stopping_patience"])

    best_val = float("inf")
    best_map = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        if config["multi_scale"]["enabled"]:
            train_size = int(random.choice(config["multi_scale"]["sizes"]))
            train_set.image_size = train_size
            criterion.image_size = train_size
            print(f"multi-scale train image_size={train_size}")
        else:
            criterion.image_size = args.image_size

        freeze_backbone = epoch <= freeze_backbone_epochs
        set_backbone_frozen(model, freeze_backbone)
        if epoch == 1 and freeze_backbone:
            print(f"freezing backbone for first {freeze_backbone_epochs} epoch(s)")
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
            print("unfreezing backbone")

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
        )
        eval_model = ema.module if ema is not None else model
        criterion.image_size = args.image_size
        with torch.no_grad():
            val_logs = run_epoch(
                eval_model,
                val_loader,
                criterion,
                device,
                use_amp=args.amp,
                epoch=epoch,
                total_epochs=args.epochs,
            )
        scheduler.step()

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_logs['loss']:.4f} "
            f"val_loss={val_logs['loss']:.4f} "
            f"box={val_logs['box_loss']:.4f} "
            f"iou={val_logs['iou_loss']:.4f} "
            f"obj={val_logs['obj_loss']:.4f} "
            f"noobj={val_logs['noobj_loss']:.4f} "
            f"cls={val_logs['cls_loss']:.4f}"
        )

        metric_logs = None
        if config["validation_metric"]["enabled"]:
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

        state = {
            "model": model.state_dict(),
            "raw_model": model.state_dict(),
            "classes": classes,
            "anchors": anchors,
            "image_size": args.image_size,
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
            },
            "amp": args.amp,
            "label_smoothing": args.label_smoothing,
            "ema_enabled": ema is not None,
            "freeze_backbone_epochs": freeze_backbone_epochs,
            "epoch": epoch,
            "val_loss": val_logs["loss"],
            "val_map50": metric_logs["map50"] if metric_logs is not None else None,
            "best_conf_threshold": metric_logs["conf_threshold"] if metric_logs is not None else config["inference"]["conf_threshold"],
            "best_nms_threshold": metric_logs["nms_threshold"] if metric_logs is not None else config["inference"]["nms_threshold"],
            "nms_type": config["validation_metric"].get("nms_type", config["inference"].get("nms_type", "soft")),
        }
        if ema is not None:
            state["model"] = ema.module.state_dict()
        torch.save(state, checkpoint_dir / "last.pth")
        improved = False
        if metric_logs is not None:
            improved = metric_logs["map50"] > best_map
        else:
            improved = val_logs["loss"] < best_val

        if improved:
            best_val = val_logs["loss"]
            if metric_logs is not None:
                best_map = metric_logs["map50"]
            epochs_without_improvement = 0
            torch.save(state, checkpoint_dir / "best.pth")
            print(f"saved best checkpoint: {checkpoint_dir / 'best.pth'}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"early stopping after {early_stopping_patience} epoch(s) without validation improvement")
                break


if __name__ == "__main__":
    main()
