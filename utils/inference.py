from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from utils.box_ops import clip_boxes, nms, soft_nms
from utils.dataset import DetectionDataset


DEFAULT_ANCHORS = [
    [(10.0, 13.0), (16.0, 24.0), (32.0, 32.0)],
    [(32.0, 48.0), (64.0, 96.0), (96.0, 128.0)],
    [(128.0, 160.0), (220.0, 260.0), (320.0, 320.0)],
]


def load_image_for_inference(path: str | Path, image_size: int, preserve_aspect: bool = True) -> tuple[torch.Tensor, int, int]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if preserve_aspect:
        scale = min(image_size / width, image_size / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        pad_x = (image_size - new_w) // 2
        pad_y = (image_size - new_h) // 2
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
        canvas.paste(resized, (pad_x, pad_y))
        image = canvas
    else:
        image = image.resize((image_size, image_size), Image.BILINEAR)
    tensor = DetectionDataset._to_tensor(image)
    return tensor, width, height


@torch.no_grad()
def decode_predictions(
    pred: dict[str, list[torch.Tensor]] | list[torch.Tensor] | torch.Tensor,
    classes: list[str],
    anchors: list[list[tuple[float, float]]],
    image_size: int,
    orig_width: int,
    orig_height: int,
    conf_threshold: float = 0.25,
    class_conf_thresholds: dict[str, float] | None = None,
    nms_threshold: float = 0.5,
    nms_type: str = "hard",
    max_detections: int = 100,
    pre_nms_topk: int = 1000,
    class_pre_nms_topk: int = 100,
    preserve_aspect: bool = True,
) -> list[dict[str, object]]:
    if isinstance(pred, dict):
        preds = pred["main"]
    elif isinstance(pred, list):
        preds = pred
    else:
        preds = [pred]

    if anchors and anchors[0] and isinstance(anchors[0][0], (int, float)):
        anchors = [anchors]  # type: ignore[list-item]

    all_boxes = []
    all_scores = []
    all_labels = []
    for scale_pred, scale_anchors in zip(preds, anchors):
        if scale_pred.dim() == 5:
            scale_pred = scale_pred[0]
        boxes, scores, labels = _decode_scale(
            scale_pred,
            classes,
            scale_anchors,
            image_size,
            orig_width,
            orig_height,
            preserve_aspect=preserve_aspect,
        )
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    if not all_boxes:
        return []
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if class_conf_thresholds:
        thresholds = torch.full_like(scores, float(conf_threshold))
        for label_idx, class_name in enumerate(classes):
            if class_name in class_conf_thresholds:
                thresholds[labels == label_idx] = float(class_conf_thresholds[class_name])
        keep = scores >= thresholds
    else:
        keep = scores >= conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if boxes.numel() == 0:
        return []
    if pre_nms_topk > 0 and scores.numel() > pre_nms_topk:
        top_scores, top_idx = scores.topk(pre_nms_topk)
        boxes = boxes[top_idx]
        scores = top_scores
        labels = labels[top_idx]

    results: list[dict[str, object]] = []
    for label_idx, class_name in enumerate(classes):
        class_mask = labels == label_idx
        if not class_mask.any():
            continue
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        if class_pre_nms_topk > 0 and class_scores.numel() > class_pre_nms_topk:
            class_scores, class_top_idx = class_scores.topk(class_pre_nms_topk)
            class_boxes = class_boxes[class_top_idx]
        if nms_type == "soft":
            kept = soft_nms(class_boxes, class_scores, nms_threshold)
        else:
            kept = nms(class_boxes, class_scores, nms_threshold)
        for idx in kept:
            box = class_boxes[idx]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            results.append(
                {
                    "class": class_name,
                    "confidence": round(float(class_scores[idx].cpu()), 6),
                    "bbox": [round(float(v), 2) for v in box.cpu().tolist()],
                }
            )

    results.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return results[:max_detections]


def flip_detections_horizontally(detections: list[dict[str, object]], width: int) -> list[dict[str, object]]:
    flipped = []
    for item in detections:
        x1, y1, x2, y2 = [float(v) for v in item["bbox"]]  # type: ignore[index]
        flipped.append(
            {
                "class": item["class"],
                "confidence": item["confidence"],
                "bbox": [round(width - x2, 2), y1, round(width - x1, 2), y2],
            }
        )
    return flipped


def merge_detections(
    detections: list[dict[str, object]],
    classes: list[str],
    nms_threshold: float,
    nms_type: str = "hard",
    max_detections: int = 100,
) -> list[dict[str, object]]:
    if not detections:
        return []
    results: list[dict[str, object]] = []
    device = torch.device("cpu")
    for class_name in classes:
        class_items = [item for item in detections if item["class"] == class_name]
        if not class_items:
            continue
        boxes = torch.tensor([item["bbox"] for item in class_items], dtype=torch.float32, device=device)
        scores = torch.tensor([float(item["confidence"]) for item in class_items], dtype=torch.float32, device=device)
        kept = soft_nms(boxes, scores, nms_threshold) if nms_type == "soft" else nms(boxes, scores, nms_threshold)
        for idx in kept:
            box = boxes[idx]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            results.append(
                {
                    "class": class_name,
                    "confidence": round(float(scores[idx]), 6),
                    "bbox": [round(float(v), 2) for v in box.tolist()],
                }
            )
    results.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return results[:max_detections]


def _decode_scale(
    pred: torch.Tensor,
    classes: list[str],
    anchors: list[tuple[float, float]],
    image_size: int,
    orig_width: int,
    orig_height: int,
    preserve_aspect: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pred.device
    grid_h, grid_w, num_anchors, _ = pred.shape
    stride_x = image_size / grid_w
    stride_y = image_size / grid_h
    anchor_t = torch.tensor(anchors, dtype=torch.float32, device=device)

    yy, xx = torch.meshgrid(
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
        indexing="ij",
    )
    xx = xx[..., None].float()
    yy = yy[..., None].float()

    center_x = (pred[..., 0].sigmoid() + xx) * stride_x
    center_y = (pred[..., 1].sigmoid() + yy) * stride_y
    box_w = pred[..., 2].clamp(min=-6, max=6).exp() * anchor_t[:, 0]
    box_h = pred[..., 3].clamp(min=-6, max=6).exp() * anchor_t[:, 1]

    boxes = torch.stack(
        [
            center_x - box_w * 0.5,
            center_y - box_h * 0.5,
            center_x + box_w * 0.5,
            center_y + box_h * 0.5,
        ],
        dim=-1,
    ).reshape(-1, 4)

    object_scores = pred[..., 4].sigmoid().reshape(-1)
    class_scores, labels = pred[..., 5:].softmax(dim=-1).reshape(-1, len(classes)).max(dim=1)
    scores = object_scores * class_scores

    if preserve_aspect:
        scale = min(image_size / orig_width, image_size / orig_height)
        new_w = round(orig_width * scale)
        new_h = round(orig_height * scale)
        pad_x = (image_size - new_w) / 2.0
        pad_y = (image_size - new_h) / 2.0
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    else:
        boxes[:, [0, 2]] *= orig_width / image_size
        boxes[:, [1, 3]] *= orig_height / image_size
    boxes = clip_boxes(boxes, orig_width, orig_height)
    return boxes, scores, labels
