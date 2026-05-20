from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from utils.box_ops import box_iou
from utils.box_ops import wh_iou


class YoloLoss(nn.Module):
    """YOLO-style loss written as explicit, inspectable components.

    Prediction format per anchor:
        [tx, ty, tw, th, object_logit, class_logits...]

    Loss components:
        box_loss: SmoothL1 on center offsets and log width/height for positive anchors.
        iou_loss: 1 - IoU on decoded positive boxes.
        obj_loss: BCEWithLogits for anchors assigned to a ground-truth object.
        noobj_loss: BCEWithLogits for background anchors.
        cls_loss: CrossEntropy over classes for positive anchors.
    """

    def __init__(
        self,
        anchors: list[tuple[float, float]],
        image_size: int,
        num_classes: int,
        box_weight: float = 5.0,
        obj_weight: float = 1.0,
        noobj_weight: float = 0.5,
        cls_weight: float = 1.0,
        iou_weight: float = 1.5,
        label_smoothing: float = 0.03,
    ) -> None:
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.image_size = image_size
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight
        self.label_smoothing = label_smoothing

    def forward(self, pred: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, dict[str, float]]:
        device = pred.device
        batch_size, grid_h, grid_w, num_anchors, _ = pred.shape
        stride_x = self.image_size / grid_w
        stride_y = self.image_size / grid_h

        obj_target = torch.zeros((batch_size, grid_h, grid_w, num_anchors), device=device)
        box_target = torch.zeros((batch_size, grid_h, grid_w, num_anchors, 4), device=device)
        xyxy_target = torch.zeros((batch_size, grid_h, grid_w, num_anchors, 4), device=device)
        cls_target = torch.zeros((batch_size, grid_h, grid_w, num_anchors), dtype=torch.long, device=device)

        anchors = self.anchors.to(device)
        for b, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)
            if boxes.numel() == 0:
                continue

            centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1.0)
            cell_x = (centers[:, 0] / stride_x).clamp(0, grid_w - 1e-4)
            cell_y = (centers[:, 1] / stride_y).clamp(0, grid_h - 1e-4)
            grid_x = cell_x.floor().long()
            grid_y = cell_y.floor().long()

            best_anchor = wh_iou(anchors, wh).argmax(dim=0)
            for i in range(boxes.shape[0]):
                gy = grid_y[i]
                gx = grid_x[i]
                a = best_anchor[i]
                obj_target[b, gy, gx, a] = 1.0
                box_target[b, gy, gx, a, 0] = cell_x[i] - gx.float()
                box_target[b, gy, gx, a, 1] = cell_y[i] - gy.float()
                box_target[b, gy, gx, a, 2] = torch.log(wh[i, 0] / anchors[a, 0].clamp(min=1e-6))
                box_target[b, gy, gx, a, 3] = torch.log(wh[i, 1] / anchors[a, 1].clamp(min=1e-6))
                xyxy_target[b, gy, gx, a] = boxes[i]
                cls_target[b, gy, gx, a] = labels[i]

        pred_xy = pred[..., 0:2].sigmoid()
        pred_wh = pred[..., 2:4]
        pred_obj_logit = pred[..., 4]
        pred_cls_logit = pred[..., 5:]

        pos_mask = obj_target == 1
        neg_mask = obj_target == 0
        num_pos = pos_mask.sum().clamp(min=1).float()

        if pos_mask.any():
            box_loss_xy = F.smooth_l1_loss(pred_xy[pos_mask], box_target[..., 0:2][pos_mask], reduction="sum") / num_pos
            box_loss_wh = F.smooth_l1_loss(pred_wh[pos_mask], box_target[..., 2:4][pos_mask], reduction="sum") / num_pos
            box_loss = box_loss_xy + box_loss_wh
            decoded_boxes = self._decode_boxes(pred, anchors, stride_x, stride_y)
            ious = box_iou(decoded_boxes[pos_mask], xyxy_target[pos_mask]).diag()
            iou_loss = (1.0 - ious).sum() / num_pos
            cls_loss = F.cross_entropy(
                pred_cls_logit[pos_mask],
                cls_target[pos_mask],
                reduction="sum",
                label_smoothing=self.label_smoothing,
            ) / num_pos
        else:
            box_loss = pred.sum() * 0.0
            iou_loss = pred.sum() * 0.0
            cls_loss = pred.sum() * 0.0

        obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj_logit[pos_mask],
            obj_target[pos_mask],
            reduction="sum",
        ) / num_pos

        noobj_count = neg_mask.sum().clamp(min=1).float()
        noobj_loss = F.binary_cross_entropy_with_logits(
            pred_obj_logit[neg_mask],
            obj_target[neg_mask],
            reduction="sum",
        ) / noobj_count

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
            "num_pos": float(num_pos.detach().cpu()),
        }
        return total, logs

    @staticmethod
    def _decode_boxes(pred: torch.Tensor, anchors: torch.Tensor, stride_x: float, stride_y: float) -> torch.Tensor:
        _, grid_h, grid_w, _, _ = pred.shape
        device = pred.device
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
