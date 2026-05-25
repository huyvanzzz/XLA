from __future__ import annotations

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def wh_iou(anchor_wh: torch.Tensor, gt_wh: torch.Tensor) -> torch.Tensor:
    anchor_wh = anchor_wh[:, None, :]
    gt_wh = gt_wh[None, :, :]
    inter = torch.min(anchor_wh, gt_wh).prod(dim=2)
    union = anchor_wh.prod(dim=2) + gt_wh.prod(dim=2) - inter
    return inter / union.clamp(min=1e-6)


def bbox_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Complete IoU for aligned xyxy box tensors."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))

    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=1e-6)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=1e-6)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=1e-6)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=1e-6)
    area1 = w1 * h1
    area2 = w2 * h2
    iou = inter / (area1 + area2 - inter).clamp(min=1e-6)

    cx1 = (boxes1[:, 0] + boxes1[:, 2]) * 0.5
    cy1 = (boxes1[:, 1] + boxes1[:, 3]) * 0.5
    cx2 = (boxes2[:, 0] + boxes2[:, 2]) * 0.5
    cy2 = (boxes2[:, 1] + boxes2[:, 3]) * 0.5
    center_dist = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)

    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_diag = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2)

    v = (4.0 / (torch.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v).clamp(min=1e-6)
    return iou - center_dist / enc_diag.clamp(min=1e-6) - alpha * v


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep).long()


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    sigma: float = 0.5,
    score_threshold: float = 1e-3,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    boxes_work = boxes.clone()
    scores_work = scores.clone()
    indices = torch.arange(scores.shape[0], device=boxes.device)
    keep = []

    while scores_work.numel() > 0:
        max_pos = torch.argmax(scores_work)
        keep.append(indices[max_pos])

        current_box = boxes_work[max_pos].unsqueeze(0)
        boxes_work = torch.cat([boxes_work[:max_pos], boxes_work[max_pos + 1 :]], dim=0)
        remaining_scores = torch.cat([scores_work[:max_pos], scores_work[max_pos + 1 :]], dim=0)
        indices = torch.cat([indices[:max_pos], indices[max_pos + 1 :]], dim=0)
        if boxes_work.numel() == 0:
            break

        ious = box_iou(current_box, boxes_work).squeeze(0)
        decay = torch.exp(-((ious * ious) / sigma))
        decay = torch.where(ious > iou_threshold, decay, torch.ones_like(decay))
        scores_work = remaining_scores * decay
        valid = scores_work >= score_threshold
        boxes_work = boxes_work[valid]
        scores_work = scores_work[valid]
        indices = indices[valid]

    return torch.stack(keep).long() if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


def clip_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return boxes
