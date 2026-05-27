from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from utils.box_ops import bbox_ciou, box_iou, wh_iou


class YoloLoss(nn.Module):
    """Multi-scale YOLO loss with explicit components and auxiliary-head support."""

    def __init__(
        self,
        anchors: list[list[tuple[float, float]]],
        image_size: int,
        num_classes: int,
        box_weight: float = 7.5,
        obj_weight: float = 1.0,
        noobj_weight: float = 0.35,
        cls_weight: float = 1.0,
        iou_weight: float = 1.5,
        aux_weight: float = 0.4,
        objectness_focal_gamma: float = 1.5,
        iou_aware_objectness: bool = True,
        class_weights: list[float] | None = None,
        label_smoothing: float = 0.03,
        positive_anchor_topk: int = 3,
        ignore_anchor_iou: float = 0.5,
        objectness_iou_mix: float = 0.25,
        noobj_hard_negative_ratio: float = 0.1,
        noobj_hard_negative_min: int = 256,
    ) -> None:
        super().__init__()
        self.anchors = [[(float(w), float(h)) for w, h in scale] for scale in anchors]
        self.image_size = image_size
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight
        self.aux_weight = aux_weight
        self.objectness_focal_gamma = objectness_focal_gamma
        self.iou_aware_objectness = iou_aware_objectness
        self.label_smoothing = label_smoothing
        self.positive_anchor_topk = max(1, int(positive_anchor_topk))
        self.ignore_anchor_iou = float(ignore_anchor_iou)
        self.objectness_iou_mix = min(max(float(objectness_iou_mix), 0.0), 1.0)
        self.noobj_hard_negative_ratio = min(max(float(noobj_hard_negative_ratio), 0.0), 1.0)
        self.noobj_hard_negative_min = max(1, int(noobj_hard_negative_min))
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(
        self,
        pred: dict[str, list[torch.Tensor]] | list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(pred, dict):
            main_preds = pred["main"]
            aux_preds = pred.get("aux", [])
        else:
            main_preds = pred
            aux_preds = []

        main_loss, main_logs = self._loss_for_predictions(main_preds, targets)
        if aux_preds:
            aux_loss, aux_logs = self._loss_for_predictions(aux_preds, targets)
            total = main_loss + self.aux_weight * aux_loss
            main_logs["loss"] = float(total.detach().cpu())
            main_logs["aux_loss"] = float(aux_loss.detach().cpu())
            main_logs["aux_box_loss"] = aux_logs["box_loss"]
            main_logs["aux_iou_loss"] = aux_logs["iou_loss"]
            return total, main_logs
        main_logs["aux_loss"] = 0.0
        return main_loss, main_logs

    def _loss_for_predictions(
        self,
        preds: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = preds[0].device
        anchors_by_scale = [torch.tensor(scale, dtype=torch.float32, device=device) for scale in self.anchors]

        obj_targets: list[torch.Tensor] = []
        ignore_masks: list[torch.Tensor] = []
        box_targets: list[torch.Tensor] = []
        xyxy_targets: list[torch.Tensor] = []
        cls_targets: list[torch.Tensor] = []
        for pred in preds:
            b, h, w, a, _ = pred.shape
            obj_targets.append(torch.zeros((b, h, w, a), device=device))
            ignore_masks.append(torch.zeros((b, h, w, a), dtype=torch.bool, device=device))
            box_targets.append(torch.zeros((b, h, w, a, 4), device=device))
            xyxy_targets.append(torch.zeros((b, h, w, a, 4), device=device))
            cls_targets.append(torch.zeros((b, h, w, a), dtype=torch.long, device=device))

        flat_anchors = torch.cat(anchors_by_scale, dim=0)
        scale_offsets = []
        offset = 0
        for anchors in anchors_by_scale:
            scale_offsets.append(offset)
            offset += anchors.shape[0]

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)
            if boxes.numel() == 0:
                continue
            wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1.0)
            centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            anchor_scores = wh_iou(flat_anchors, wh)
            topk = min(self.positive_anchor_topk, flat_anchors.shape[0])
            best_flat_anchors = anchor_scores.topk(topk, dim=0).indices
            for obj_idx in range(boxes.shape[0]):
                ignored_flat_anchors = (anchor_scores[:, obj_idx] >= self.ignore_anchor_iou).nonzero(as_tuple=False).flatten()
                for flat_anchor_idx_t in best_flat_anchors[:, obj_idx]:
                    flat_anchor_idx = int(flat_anchor_idx_t.item())
                    scale_idx = max(i for i, start in enumerate(scale_offsets) if flat_anchor_idx >= start)
                    anchor_idx = flat_anchor_idx - scale_offsets[scale_idx]
                    pred = preds[scale_idx]
                    _, grid_h, grid_w, _, _ = pred.shape
                    stride_x = self.image_size / grid_w
                    stride_y = self.image_size / grid_h
                    cell_x = (centers[obj_idx, 0] / stride_x).clamp(0, grid_w - 1e-4)
                    cell_y = (centers[obj_idx, 1] / stride_y).clamp(0, grid_h - 1e-4)
                    gx = cell_x.floor().long()
                    gy = cell_y.floor().long()
                    anchor = anchors_by_scale[scale_idx][anchor_idx]

                    obj_targets[scale_idx][batch_idx, gy, gx, anchor_idx] = 1.0
                    box_targets[scale_idx][batch_idx, gy, gx, anchor_idx, 0] = cell_x - gx.float()
                    box_targets[scale_idx][batch_idx, gy, gx, anchor_idx, 1] = cell_y - gy.float()
                    box_targets[scale_idx][batch_idx, gy, gx, anchor_idx, 2] = torch.log(wh[obj_idx, 0] / anchor[0].clamp(min=1e-6))
                    box_targets[scale_idx][batch_idx, gy, gx, anchor_idx, 3] = torch.log(wh[obj_idx, 1] / anchor[1].clamp(min=1e-6))
                    xyxy_targets[scale_idx][batch_idx, gy, gx, anchor_idx] = boxes[obj_idx]
                    cls_targets[scale_idx][batch_idx, gy, gx, anchor_idx] = labels[obj_idx]
                for flat_anchor_idx_t in ignored_flat_anchors:
                    flat_anchor_idx = int(flat_anchor_idx_t.item())
                    scale_idx = max(i for i, start in enumerate(scale_offsets) if flat_anchor_idx >= start)
                    anchor_idx = flat_anchor_idx - scale_offsets[scale_idx]
                    pred = preds[scale_idx]
                    _, grid_h, grid_w, _, _ = pred.shape
                    stride_x = self.image_size / grid_w
                    stride_y = self.image_size / grid_h
                    cell_x = (centers[obj_idx, 0] / stride_x).clamp(0, grid_w - 1e-4)
                    cell_y = (centers[obj_idx, 1] / stride_y).clamp(0, grid_h - 1e-4)
                    gx = cell_x.floor().long()
                    gy = cell_y.floor().long()
                    ignore_masks[scale_idx][batch_idx, gy, gx, anchor_idx] = True

        box_loss = preds[0].sum() * 0.0
        iou_loss = preds[0].sum() * 0.0
        cls_loss = preds[0].sum() * 0.0
        obj_loss = preds[0].sum() * 0.0
        noobj_loss = preds[0].sum() * 0.0
        total_pos = torch.tensor(0.0, device=device)

        for scale_idx, pred in enumerate(preds):
            anchors = anchors_by_scale[scale_idx]
            obj_target = obj_targets[scale_idx]
            pos_mask = obj_target == 1
            neg_mask = (obj_target == 0) & (~ignore_masks[scale_idx])
            num_pos = pos_mask.sum().clamp(min=1).float()
            total_pos += pos_mask.sum().float()

            pred_xy = pred[..., 0:2].sigmoid()
            pred_wh = pred[..., 2:4]
            pred_obj_logit = pred[..., 4]
            pred_cls_logit = pred[..., 5:]

            if pos_mask.any():
                box_loss = box_loss + F.smooth_l1_loss(
                    pred_xy[pos_mask],
                    box_targets[scale_idx][..., 0:2][pos_mask],
                    reduction="sum",
                ) / num_pos
                box_loss = box_loss + F.smooth_l1_loss(
                    pred_wh[pos_mask],
                    box_targets[scale_idx][..., 2:4][pos_mask],
                    reduction="sum",
                ) / num_pos
                decoded_boxes = self._decode_boxes(pred, anchors)
                pred_pos_boxes = decoded_boxes[pos_mask]
                target_pos_boxes = xyxy_targets[scale_idx][pos_mask]
                ious = box_iou(pred_pos_boxes, target_pos_boxes).diag()
                if self.iou_aware_objectness:
                    quality = ious.detach().clamp(0.0, 1.0)
                    obj_target[pos_mask] = (1.0 - self.objectness_iou_mix) + self.objectness_iou_mix * quality
                ciou = bbox_ciou(pred_pos_boxes, target_pos_boxes)
                iou_loss = iou_loss + (1.0 - ciou).sum() / num_pos
                cls_loss = cls_loss + F.cross_entropy(
                    pred_cls_logit[pos_mask],
                    cls_targets[scale_idx][pos_mask],
                    reduction="sum",
                    weight=self.class_weights,
                    label_smoothing=self.label_smoothing,
                ) / num_pos

            obj_loss = obj_loss + self._objectness_loss(
                pred_obj_logit[pos_mask],
                obj_target[pos_mask],
                normalizer=num_pos,
            )
            noobj_loss = noobj_loss + self._objectness_loss(
                pred_obj_logit[neg_mask],
                obj_target[neg_mask],
                normalizer=neg_mask.sum().clamp(min=1).float(),
                hard_fraction=self.noobj_hard_negative_ratio,
                hard_min=self.noobj_hard_negative_min,
            )

        num_scales = max(len(preds), 1)
        box_loss = box_loss / num_scales
        iou_loss = iou_loss / num_scales
        cls_loss = cls_loss / num_scales
        obj_loss = obj_loss / num_scales
        noobj_loss = noobj_loss / num_scales
        total = (
            self.box_weight * box_loss
            + self.iou_weight * iou_loss
            + self.obj_weight * obj_loss
            + self.noobj_weight * noobj_loss
            + self.cls_weight * cls_loss
        )
        logs = {
            "loss": float(total.detach().cpu()),
            "box_loss": float(box_loss.detach().cpu()),
            "iou_loss": float(iou_loss.detach().cpu()),
            "obj_loss": float(obj_loss.detach().cpu()),
            "noobj_loss": float(noobj_loss.detach().cpu()),
            "cls_loss": float(cls_loss.detach().cpu()),
            "num_pos": float(total_pos.detach().cpu()),
        }
        return total, logs

    def _objectness_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        normalizer: torch.Tensor,
        hard_fraction: float = 0.0,
        hard_min: int = 1,
    ) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.objectness_focal_gamma <= 0:
            loss_vec = bce
        else:
            probs = torch.sigmoid(logits)
            pt = probs * targets + (1.0 - probs) * (1.0 - targets)
            focal = (1.0 - pt).pow(self.objectness_focal_gamma)
            loss_vec = focal * bce
        if hard_fraction > 0.0 and loss_vec.numel() > hard_min:
            k = min(loss_vec.numel(), max(hard_min, int(loss_vec.numel() * hard_fraction)))
            loss_vec = loss_vec.topk(k).values
            normalizer = loss_vec.new_tensor(float(k))
        return loss_vec.sum() / normalizer

    def _decode_boxes(self, pred: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        _, grid_h, grid_w, _, _ = pred.shape
        device = pred.device
        stride_x = self.image_size / grid_w
        stride_y = self.image_size / grid_h
        yy, xx = torch.meshgrid(
            torch.arange(grid_h, device=device),
            torch.arange(grid_w, device=device),
            indexing="ij",
        )
        xx = xx[None, :, :, None].float()
        yy = yy[None, :, :, None].float()
        cx = (pred[..., 0].sigmoid() + xx) * stride_x
        cy = (pred[..., 1].sigmoid() + yy) * stride_y
        bw = pred[..., 2].clamp(min=-6, max=6).exp() * anchors[:, 0]
        bh = pred[..., 3].clamp(min=-6, max=6).exp() * anchors[:, 1]
        return torch.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], dim=-1)
