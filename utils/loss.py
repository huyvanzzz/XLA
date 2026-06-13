from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from utils.box_ops import bbox_ciou, box_iou, wh_iou


class AnchorFreeLoss(nn.Module):
    """Quality-aware anchor-free loss inspired by FCOS/GFL/VFL style detectors."""

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        cls_weight: float = 1.0,
        iou_weight: float = 5.0,
        task_aligned_alpha: float = 0.5,
        task_aligned_beta: float = 6.0,
        task_aligned_center_radius: float = 2.5,
        positive_anchor_topk: int = 10,
        class_weights: list[float] | None = None,
        varifocal_alpha: float = 0.75,
        varifocal_gamma: float = 2.0,
        assignment_warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.cls_weight = float(cls_weight)
        self.iou_weight = float(iou_weight)
        self.task_aligned_alpha = float(task_aligned_alpha)
        self.task_aligned_beta = float(task_aligned_beta)
        self.task_aligned_center_radius = float(task_aligned_center_radius)
        self.positive_anchor_topk = max(1, int(positive_anchor_topk))
        self.varifocal_alpha = float(varifocal_alpha)
        self.varifocal_gamma = float(varifocal_gamma)
        self.assignment_warmup_epochs = max(0, int(assignment_warmup_epochs))
        self.current_epoch = 0
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(
        self,
        pred: dict[str, list[torch.Tensor]] | list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        preds = pred["main"] if isinstance(pred, dict) else pred
        boxes, cls_logits, centers, strides = self._flatten_predictions(preds)
        batch_size, num_candidates, _ = boxes.shape
        device = boxes.device

        cls_targets = cls_logits.new_zeros((batch_size, num_candidates, self.num_classes))
        box_targets = boxes.new_zeros((batch_size, num_candidates, 4))
        positive = torch.zeros((batch_size, num_candidates), dtype=torch.bool, device=device)
        assigned_quality = boxes.new_zeros((batch_size, num_candidates))

        for batch_idx, target in enumerate(targets):
            gt_boxes = target["boxes"].to(device=device, dtype=torch.float32)
            gt_labels = target["labels"].to(device)
            if gt_boxes.numel() == 0:
                continue
            pred_boxes = boxes[batch_idx].float()
            pred_scores = cls_logits[batch_idx].float().sigmoid()
            centers_f = centers.float()
            strides_f = strides.float()
            ious = box_iou(pred_boxes, gt_boxes).clamp(min=0.0)
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
            in_box = (
                (centers_f[:, None, 0] >= gt_boxes[None, :, 0])
                & (centers_f[:, None, 0] <= gt_boxes[None, :, 2])
                & (centers_f[:, None, 1] >= gt_boxes[None, :, 1])
                & (centers_f[:, None, 1] <= gt_boxes[None, :, 3])
            )
            center_delta = (centers_f[:, None, :] - gt_centers[None, :, :]).abs()
            center_limit = strides_f[:, None, :] * self.task_aligned_center_radius
            in_center = (center_delta[..., 0] <= center_limit[..., 0]) & (center_delta[..., 1] <= center_limit[..., 1])
            candidate_mask = in_box | in_center
            matched_scores = pred_scores[:, gt_labels]
            if self.current_epoch <= self.assignment_warmup_epochs:
                gt_wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=1.0)
                normalized_delta = center_delta / (gt_wh[None, :, :] * 0.5).clamp(min=1.0)
                center_metric = torch.exp(-2.0 * normalized_delta.square().sum(dim=-1))
                metric = center_metric + 0.01 * ious
            else:
                metric = matched_scores.clamp(min=1e-6).pow(self.task_aligned_alpha) * ious.clamp(min=1e-4).pow(
                    self.task_aligned_beta
                )
            metric = torch.where(candidate_mask, metric, metric.new_full(metric.shape, -1.0))

            assigned_gt = torch.full((num_candidates,), -1, dtype=torch.long, device=device)
            assigned_metric = torch.full((num_candidates,), -1.0, device=device)
            assigned_iou = torch.zeros((num_candidates,), device=device)
            topk = min(self.positive_anchor_topk, num_candidates)
            for obj_idx in range(gt_boxes.shape[0]):
                values, indices = metric[:, obj_idx].topk(topk)
                keep = values > 0
                if not keep.any():
                    continue
                indices = indices[keep]
                values = values[keep]
                replace = values > assigned_metric[indices]
                if replace.any():
                    chosen = indices[replace]
                    assigned_gt[chosen] = obj_idx
                    assigned_metric[chosen] = values[replace]
                    assigned_iou[chosen] = ious[chosen, obj_idx]

            pos_idx = (assigned_gt >= 0).nonzero(as_tuple=False).flatten()
            if pos_idx.numel() == 0:
                continue
            positive[batch_idx, pos_idx] = True
            gt_idx = assigned_gt[pos_idx]
            if self.current_epoch <= self.assignment_warmup_epochs:
                quality = assigned_metric[pos_idx].detach().clamp(0.25, 1.0)
            else:
                quality = assigned_iou[pos_idx].detach().clamp(0.05, 1.0)
            labels = gt_labels[gt_idx]
            cls_targets[batch_idx, pos_idx, labels] = quality.to(cls_targets.dtype)
            box_targets[batch_idx, pos_idx] = gt_boxes[gt_idx].to(box_targets.dtype)
            assigned_quality[batch_idx, pos_idx] = quality.to(assigned_quality.dtype)

        normalizer = positive.sum().clamp(min=1).to(cls_logits.dtype)
        cls_loss = self._varifocal_loss(cls_logits, cls_targets).sum() / normalizer
        if positive.any():
            pred_pos = boxes[positive].float()
            target_pos = box_targets[positive].float()
            ciou = bbox_ciou(pred_pos, target_pos).clamp(min=-1.0, max=1.0)
            weight = assigned_quality[positive].clamp(min=0.05)
            iou_loss = ((1.0 - ciou) * weight).sum() / weight.sum().clamp(min=1e-6)
            box_loss = iou_loss
        else:
            iou_loss = boxes.sum() * 0.0
            box_loss = iou_loss
        loss = self.cls_weight * cls_loss + self.iou_weight * iou_loss
        return loss, {
            "loss": float(loss.detach().cpu()),
            "box_loss": float(box_loss.detach().cpu()),
            "iou_loss": float(iou_loss.detach().cpu()),
            "obj_loss": 0.0,
            "noobj_loss": 0.0,
            "cls_loss": float(cls_loss.detach().cpu()),
            "quality_loss": 0.0,
            "num_pos": float(positive.sum().detach().cpu()),
        }

    def _flatten_predictions(self, preds: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_boxes = []
        all_logits = []
        all_centers = []
        all_strides = []
        for pred in preds:
            b, h, w, _, d = pred.shape
            device = pred.device
            stride_x = self.image_size / w
            stride_y = self.image_size / h
            yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            centers = torch.stack([(xx.float() + 0.5) * stride_x, (yy.float() + 0.5) * stride_y], dim=-1).reshape(-1, 2)
            stride = torch.empty_like(centers)
            stride[:, 0] = stride_x
            stride[:, 1] = stride_y
            raw = pred[..., 0, :]
            distances = F.softplus(raw[..., :4])
            scale = torch.tensor([stride_x, stride_y, stride_x, stride_y], dtype=pred.dtype, device=device)
            distances = distances * scale
            center_b = centers.view(1, h, w, 2)
            boxes = torch.stack(
                [
                    center_b[..., 0] - distances[..., 0],
                    center_b[..., 1] - distances[..., 1],
                    center_b[..., 0] + distances[..., 2],
                    center_b[..., 1] + distances[..., 3],
                ],
                dim=-1,
            ).reshape(b, -1, 4)
            boxes[..., 0::2].clamp_(0, self.image_size)
            boxes[..., 1::2].clamp_(0, self.image_size)
            all_boxes.append(boxes)
            all_logits.append(raw[..., 4 : 4 + self.num_classes].reshape(b, -1, self.num_classes))
            all_centers.append(centers)
            all_strides.append(stride)
        return torch.cat(all_boxes, dim=1), torch.cat(all_logits, dim=1), torch.cat(all_centers, dim=0), torch.cat(all_strides, dim=0)

    def _varifocal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_score = logits.sigmoid()
        focal_weight = self.varifocal_alpha * pred_score.pow(self.varifocal_gamma) * (targets <= 0).to(logits.dtype) + targets
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * focal_weight
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(logits.device, logits.dtype).view(1, 1, -1)
        return loss


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
        assignment_strategy: str = "legacy",
        assignment_warmup_epochs: int = 0,
        task_aligned_alpha: float = 0.5,
        task_aligned_beta: float = 6.0,
        task_aligned_center_radius: float = 2.5,
        task_aligned_min_iou: float = 0.05,
        decode_style: str = "standard",
        target_offsets: bool = False,
        target_offset_bias: float = 0.5,
        scale_obj_balance: list[float] | None = None,
        classification_loss: str = "ce",
        classification_quality_mix: float = 0.0,
        classification_focal_gamma: float = 0.0,
        quality_weight: float = 0.0,
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
        self.assignment_strategy = str(assignment_strategy)
        self.assignment_warmup_epochs = max(0, int(assignment_warmup_epochs))
        self.current_epoch = 0
        self.task_aligned_alpha = float(task_aligned_alpha)
        self.task_aligned_beta = float(task_aligned_beta)
        self.task_aligned_center_radius = float(task_aligned_center_radius)
        self.task_aligned_min_iou = float(task_aligned_min_iou)
        self.decode_style = str(decode_style)
        self.target_offsets = bool(target_offsets)
        self.target_offset_bias = float(target_offset_bias)
        self.scale_obj_balance = [float(v) for v in scale_obj_balance] if scale_obj_balance else []
        self.classification_loss = str(classification_loss).lower()
        self.classification_quality_mix = min(max(float(classification_quality_mix), 0.0), 1.0)
        self.classification_focal_gamma = max(float(classification_focal_gamma), 0.0)
        self.quality_weight = max(float(quality_weight), 0.0)
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

        effective_assignment_strategy = self.assignment_strategy
        if (
            self.assignment_strategy == "task_aligned"
            and self.assignment_warmup_epochs > 0
            and 0 < self.current_epoch <= self.assignment_warmup_epochs
        ):
            effective_assignment_strategy = "legacy"

        if effective_assignment_strategy == "task_aligned":
            self._assign_task_aligned(
                preds,
                targets,
                anchors_by_scale,
                obj_targets,
                ignore_masks,
                box_targets,
                xyxy_targets,
                cls_targets,
            )
        else:
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
                        anchor = anchors_by_scale[scale_idx][anchor_idx]
                        for gx, gy in self._target_cells(cell_x, cell_y, grid_w, grid_h):
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
        quality_loss = preds[0].sum() * 0.0
        obj_loss = preds[0].sum() * 0.0
        noobj_loss = preds[0].sum() * 0.0
        total_pos = torch.tensor(0.0, device=device)

        for scale_idx, pred in enumerate(preds):
            anchors = anchors_by_scale[scale_idx]
            obj_target = obj_targets[scale_idx]
            pos_mask = obj_target > 0
            neg_mask = (obj_target == 0) & (~ignore_masks[scale_idx])
            num_pos = pos_mask.sum().clamp(min=1).float()
            total_pos += pos_mask.sum().float()

            if self.decode_style == "yolov7":
                pred_xy = pred[..., 0:2].sigmoid() * 2.0 - 0.5
                pred_wh = ((pred[..., 2:4].sigmoid() * 2.0).clamp(min=1e-6).pow(2)).log()
            else:
                pred_xy = pred[..., 0:2].sigmoid()
                pred_wh = pred[..., 2:4]
            pred_obj_logit = pred[..., 4]
            pred_cls_logit = pred[..., 5 : 5 + self.num_classes]
            pred_quality_logit = pred[..., 5 + self.num_classes] if pred.shape[-1] > 5 + self.num_classes else None
            balance = self.scale_obj_balance[scale_idx] if scale_idx < len(self.scale_obj_balance) else 1.0

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
                cls_loss = cls_loss + self._classification_loss(
                    pred_cls_logit[pos_mask],
                    cls_targets[scale_idx][pos_mask],
                    quality=ious.detach().clamp(0.0, 1.0),
                    normalizer=num_pos,
                )
                if pred_quality_logit is not None and self.quality_weight > 0.0:
                    quality_loss = quality_loss + F.binary_cross_entropy_with_logits(
                        pred_quality_logit[pos_mask],
                        ious.detach().clamp(0.0, 1.0),
                        reduction="sum",
                    ) / num_pos

            obj_loss = obj_loss + balance * self._objectness_loss(
                pred_obj_logit[pos_mask],
                obj_target[pos_mask],
                normalizer=num_pos,
            )
            noobj_loss = noobj_loss + balance * self._objectness_loss(
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
        quality_loss = quality_loss / num_scales
        obj_loss = obj_loss / num_scales
        noobj_loss = noobj_loss / num_scales
        total = (
            self.box_weight * box_loss
            + self.iou_weight * iou_loss
            + self.obj_weight * obj_loss
            + self.noobj_weight * noobj_loss
            + self.cls_weight * cls_loss
            + self.quality_weight * quality_loss
        )
        logs = {
            "loss": float(total.detach().cpu()),
            "box_loss": float(box_loss.detach().cpu()),
            "iou_loss": float(iou_loss.detach().cpu()),
            "obj_loss": float(obj_loss.detach().cpu()),
            "noobj_loss": float(noobj_loss.detach().cpu()),
            "cls_loss": float(cls_loss.detach().cpu()),
            "quality_loss": float(quality_loss.detach().cpu()),
            "num_pos": float(total_pos.detach().cpu()),
        }
        return total, logs

    def _target_cells(
        self,
        cell_x: torch.Tensor,
        cell_y: torch.Tensor,
        grid_w: int,
        grid_h: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        gx0 = cell_x.floor().long()
        gy0 = cell_y.floor().long()
        if not self.target_offsets:
            return [(gx0, gy0)]

        cells = [(gx0, gy0)]
        frac_x = cell_x - gx0.float()
        frac_y = cell_y - gy0.float()
        bias = self.target_offset_bias
        if frac_x < bias and gx0 > 0:
            cells.append((gx0 - 1, gy0))
        if frac_y < bias and gy0 > 0:
            cells.append((gx0, gy0 - 1))
        if frac_x > 1.0 - bias and gx0 < grid_w - 1:
            cells.append((gx0 + 1, gy0))
        if frac_y > 1.0 - bias and gy0 < grid_h - 1:
            cells.append((gx0, gy0 + 1))
        return cells

    @torch.no_grad()
    def _assign_task_aligned(
        self,
        preds: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        anchors_by_scale: list[torch.Tensor],
        obj_targets: list[torch.Tensor],
        ignore_masks: list[torch.Tensor],
        box_targets: list[torch.Tensor],
        xyxy_targets: list[torch.Tensor],
        cls_targets: list[torch.Tensor],
    ) -> None:
        device = preds[0].device
        flat_boxes = []
        flat_scores = []
        flat_centers = []
        flat_strides = []
        flat_scale_idx = []
        flat_grid_y = []
        flat_grid_x = []
        flat_anchor_idx = []
        for scale_idx, pred in enumerate(preds):
            batch_size, grid_h, grid_w, num_anchors, _ = pred.shape
            stride_x = self.image_size / grid_w
            stride_y = self.image_size / grid_h
            yy, xx = torch.meshgrid(
                torch.arange(grid_h, device=device),
                torch.arange(grid_w, device=device),
                indexing="ij",
            )
            grid_y = yy[:, :, None].expand(grid_h, grid_w, num_anchors)
            grid_x = xx[:, :, None].expand(grid_h, grid_w, num_anchors)
            anchor_idx = torch.arange(num_anchors, device=device)[None, None, :].expand(grid_h, grid_w, num_anchors)

            centers = torch.stack(
                [
                    (grid_x.float() + 0.5) * stride_x,
                    (grid_y.float() + 0.5) * stride_y,
                ],
                dim=-1,
            ).reshape(-1, 2)
            strides = torch.full((centers.shape[0], 2), 0.0, device=device)
            strides[:, 0] = stride_x
            strides[:, 1] = stride_y

            decoded = self._decode_boxes(pred.float(), anchors_by_scale[scale_idx].float()).reshape(batch_size, -1, 4)
            object_scores = pred[..., 4].float().sigmoid()
            class_logits = pred[..., 5 : 5 + self.num_classes].float()
            if self.classification_loss == "bce":
                class_scores = class_logits.sigmoid()
            else:
                class_scores = class_logits.softmax(dim=-1)
            scores = (object_scores[..., None] * class_scores).reshape(batch_size, -1, self.num_classes)

            flat_boxes.append(decoded)
            flat_scores.append(scores)
            flat_centers.append(centers)
            flat_strides.append(strides)
            flat_scale_idx.append(torch.full((centers.shape[0],), scale_idx, dtype=torch.long, device=device))
            flat_grid_y.append(grid_y.reshape(-1).long())
            flat_grid_x.append(grid_x.reshape(-1).long())
            flat_anchor_idx.append(anchor_idx.reshape(-1).long())

        all_boxes = torch.cat(flat_boxes, dim=1)
        all_scores = torch.cat(flat_scores, dim=1)
        all_centers = torch.cat(flat_centers, dim=0)
        all_strides = torch.cat(flat_strides, dim=0)
        all_scale_idx = torch.cat(flat_scale_idx, dim=0)
        all_grid_y = torch.cat(flat_grid_y, dim=0)
        all_grid_x = torch.cat(flat_grid_x, dim=0)
        all_anchor_idx = torch.cat(flat_anchor_idx, dim=0)
        num_candidates = all_centers.shape[0]

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device=device, dtype=torch.float32)
            labels = target["labels"].to(device)
            if boxes.numel() == 0:
                continue

            ious = box_iou(all_boxes[batch_idx], boxes).clamp(min=0.0)
            gt_centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            in_box = (
                (all_centers[:, None, 0] >= boxes[None, :, 0])
                & (all_centers[:, None, 0] <= boxes[None, :, 2])
                & (all_centers[:, None, 1] >= boxes[None, :, 1])
                & (all_centers[:, None, 1] <= boxes[None, :, 3])
            )
            center_delta = (all_centers[:, None, :] - gt_centers[None, :, :]).abs()
            center_limit = all_strides[:, None, :] * self.task_aligned_center_radius
            in_center = (center_delta[..., 0] <= center_limit[..., 0]) & (center_delta[..., 1] <= center_limit[..., 1])
            candidate_mask = (in_box | in_center) & (ious >= self.task_aligned_min_iou)

            matched_scores = all_scores[batch_idx][:, labels]
            metric = matched_scores.clamp(min=1e-9).pow(self.task_aligned_alpha) * ious.pow(self.task_aligned_beta)
            metric = torch.where(candidate_mask, metric, metric.new_full(metric.shape, -1.0))

            assigned_gt = torch.full((num_candidates,), -1, dtype=torch.long, device=device)
            assigned_metric = torch.full((num_candidates,), -1.0, device=device)
            assigned_iou = torch.zeros((num_candidates,), device=device)
            topk = min(self.positive_anchor_topk, num_candidates)
            for obj_idx in range(boxes.shape[0]):
                values, indices = metric[:, obj_idx].topk(topk)
                keep = values > 0
                if not keep.any():
                    continue
                indices = indices[keep]
                values = values[keep]
                replace = values > assigned_metric[indices]
                if replace.any():
                    selected = indices[replace]
                    assigned_gt[selected] = obj_idx
                    assigned_metric[selected] = values[replace]
                    assigned_iou[selected] = ious[selected, obj_idx]

            ignore = ious.max(dim=1).values >= self.ignore_anchor_iou
            for flat_idx_t in ignore.nonzero(as_tuple=False).flatten():
                flat_idx = int(flat_idx_t.item())
                scale_idx = int(all_scale_idx[flat_idx].item())
                gy = all_grid_y[flat_idx]
                gx = all_grid_x[flat_idx]
                anchor_idx = all_anchor_idx[flat_idx]
                ignore_masks[scale_idx][batch_idx, gy, gx, anchor_idx] = True

            positive_indices = (assigned_gt >= 0).nonzero(as_tuple=False).flatten()
            for flat_idx_t in positive_indices:
                flat_idx = int(flat_idx_t.item())
                obj_idx = int(assigned_gt[flat_idx].item())
                scale_idx = int(all_scale_idx[flat_idx].item())
                gy = all_grid_y[flat_idx]
                gx = all_grid_x[flat_idx]
                anchor_idx = all_anchor_idx[flat_idx]
                pred = preds[scale_idx]
                _, grid_h, grid_w, _, _ = pred.shape
                stride_x = self.image_size / grid_w
                stride_y = self.image_size / grid_h
                gt_wh = (boxes[obj_idx, 2:] - boxes[obj_idx, :2]).clamp(min=1.0)
                gt_center = (boxes[obj_idx, :2] + boxes[obj_idx, 2:]) * 0.5
                cell_x = (gt_center[0] / stride_x).clamp(0, grid_w - 1e-4)
                cell_y = (gt_center[1] / stride_y).clamp(0, grid_h - 1e-4)
                anchor = anchors_by_scale[scale_idx][anchor_idx]
                quality = assigned_iou[flat_idx].clamp(0.0, 1.0)
                obj_target = (1.0 - self.objectness_iou_mix) + self.objectness_iou_mix * quality

                obj_targets[scale_idx][batch_idx, gy, gx, anchor_idx] = obj_target
                ignore_masks[scale_idx][batch_idx, gy, gx, anchor_idx] = False
                box_targets[scale_idx][batch_idx, gy, gx, anchor_idx, 0] = cell_x - gx.float()
                box_targets[scale_idx][batch_idx, gy, gx, anchor_idx, 1] = cell_y - gy.float()
                box_targets[scale_idx][batch_idx, gy, gx, anchor_idx, 2] = torch.log(gt_wh[0] / anchor[0].clamp(min=1e-6))
                box_targets[scale_idx][batch_idx, gy, gx, anchor_idx, 3] = torch.log(gt_wh[1] / anchor[1].clamp(min=1e-6))
                xyxy_targets[scale_idx][batch_idx, gy, gx, anchor_idx] = boxes[obj_idx]
                cls_targets[scale_idx][batch_idx, gy, gx, anchor_idx] = labels[obj_idx]

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

    def _classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        quality: torch.Tensor,
        normalizer: torch.Tensor,
    ) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0
        if self.classification_loss == "bce":
            target = logits.new_full(logits.shape, self.label_smoothing / max(self.num_classes - 1, 1))
            positive = logits.new_full((logits.shape[0], 1), 1.0 - self.label_smoothing)
            if self.classification_quality_mix > 0.0:
                quality_target = (1.0 - self.classification_quality_mix) + self.classification_quality_mix * quality[:, None]
                positive = positive * quality_target.to(logits.dtype)
            target.scatter_(1, labels[:, None], positive)
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            if self.classification_focal_gamma > 0.0:
                probs = torch.sigmoid(logits)
                pt = probs * target + (1.0 - probs) * (1.0 - target)
                loss = loss * (1.0 - pt).pow(self.classification_focal_gamma)
            if self.class_weights is not None:
                loss = loss * self.class_weights.to(logits.device, logits.dtype).view(1, -1)
            return loss.sum() / normalizer
        return F.cross_entropy(
            logits,
            labels,
            reduction="sum",
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        ) / normalizer

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
        if self.decode_style == "yolov7":
            cx = (pred[..., 0].sigmoid() * 2.0 - 0.5 + xx) * stride_x
            cy = (pred[..., 1].sigmoid() * 2.0 - 0.5 + yy) * stride_y
            bw = (pred[..., 2].sigmoid() * 2.0).pow(2) * anchors[:, 0]
            bh = (pred[..., 3].sigmoid() * 2.0).pow(2) * anchors[:, 1]
        else:
            cx = (pred[..., 0].sigmoid() + xx) * stride_x
            cy = (pred[..., 1].sigmoid() + yy) * stride_y
            bw = pred[..., 2].clamp(min=-6, max=6).exp() * anchors[:, 0]
            bh = pred[..., 3].clamp(min=-6, max=6).exp() * anchors[:, 1]
        return torch.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], dim=-1)
