# Session Summary

Muc dich file nay: dung de mo chat moi ma van tiep tuc dung huong. Doc file nay truoc, sau do doc `EXPERIMENTS.md`, `configs/default.yaml`, `train.py`, `utils/loss.py`, `models/tiny_detector.py` neu can sua tiep.

## Bai toan

- Object detection 5 class: `person`, `car`, `dog`, `cat`, `chair`.
- Metric chinh: `mAP@0.5` theo public evaluator.
- Backbone van dung pretrained.
- Yeu cau "from scratch": khong dung complete detector co san nhu YOLO/Detectron/MMDetection/torchvision FasterRCNN/SSD; khong dung `torchvision.ops.nms`.
- NMS hien tai tu cai trong `utils/box_ops.py`.

## Huong dang dung

- Resize truc tiep anh ve `512x512`, khong letterbox.
- `preserve_aspect: false`.
- ResNet50 pretrained backbone.
- Sau warmup chi fine-tune `resnet.layer4`; cac layer thap hon freeze.
- Freeze BatchNorm stats cua backbone pretrained.
- Optimizer: AdamW voi param groups, backbone LR nho hon detector LR.
- Validation chon `best.pth` theo `mAP@0.5`, khong theo val loss.
- `validation_loss.enabled: false`.
- Bat dau tinh mAP va save best tu epoch 30 de tiet kiem thoi gian.
- Trong train chi dung mot bo threshold: `conf_threshold: 0.08`, `nms_threshold: 0.5`.
- Khong dung threshold grid/tuning trong train.

## Config hien tai

File chinh: `configs/default.yaml`.

Diem quan trong:
- `image_size: 512`
- `batch_size: 24`
- `val_batch_size: 64`
- `amp: true`
- `channels_last: true`
- `cudnn_benchmark: true`
- `freeze_backbone_epochs: 2`
- `backbone_trainable: layer4`
- `backbone_freeze_bn: true`
- `early_stopping_patience: 10`
- `augmentation.mosaic_prob: 0.0`
- `inference.tta_hflip: false`
- `validation_metric.start_epoch: 30`
- `validation_metric.tune: false`

Model hien tai:
- `backbone: resnet50`
- `pretrained: true`
- `neck_channels: 192`
- `head_channels: 192`
- `attention_heads: 0`
- `aux_head: true`
- `aux_head_close_epoch: 30`
- `decoupled_head: true`
- `cls_head_channels: 128`

Loss/assignment hien tai:
- `assignment_strategy: task_aligned`
- `positive_anchor_topk: 8`
- `task_aligned_alpha: 0.5`
- `task_aligned_beta: 6.0`
- `task_aligned_center_radius: 2.5`
- `task_aligned_min_iou: 0.05`
- `iou_aware_objectness: true`
- `objectness_iou_mix: 0.4`
- `noobj_hard_negative_ratio: 0.0`

## Version da ghi

Xem chi tiet trong `EXPERIMENTS.md`.

- v1: baseline YOLO-style ResNet50 pretrained, mAP public val `0.604047`.
- v2/v3: direct resize 512, train nhanh hon; user bao v3 len khoang `0.615`, train/epoch khoang `4'30`, best gan epoch 60.
- v4: hard-negative + mosaic + TTA, ket qua rat te; da reset/tat.
- v5 hien tai: Task-Aligned Assignment + Decoupled Detection Head + tat aux head tu epoch 30. Chua co ket qua train day du luc viet summary.

## Ly do v5

V3 bi gioi han quanh `0.615` mAP va precision thap. Thay vi them augment nang, v5 doi hai diem co tac dong truc tiep vao detection:

- Task-Aligned Assignment: positive sample duoc chon theo ca classification confidence va IoU hien tai, khong chi theo anchor shape.
- Decoupled Head: bbox/objectness va classification co tower rieng, giam xung dot giua localization va classification.

Inference output van giu format cu `[x, y, w, h, obj, class...]`, nen `predict.py` va NMS khong can doi.

## Dieu can tranh

- Khong quay lai letterbox neu user khong yeu cau.
- Khong bat lai mosaic/hard-negative/TTA mac dinh vi v4 da te.
- Khong sua notebook neu co the sua config/code.
- Khong dung torchvision NMS/detector co san.
- Khong danh gia thanh cong bang train loss thap hon neu mAP khong tang.
- Khong tune nhieu conf/NMS trong moi epoch train vi lam mAP validation cham.

## Kiem tra da chay

- `python -m py_compile train.py predict.py utils\config.py utils\dataset.py utils\loss.py utils\inference.py models\tiny_detector.py`
- `python -m unittest test_predict_cli.py`
- Forward test voi `TinyDetector(... decoupled_head=True ...)` output dung shape:
  - main: `[B,64,64,3,10]`, `[B,32,32,3,10]`, `[B,16,16,3,10]` voi input `512x512`
  - aux tuong tu khi dang train
- Loss/backward test gia voi `assignment_strategy='task_aligned'` OK.

## Neu tiep tuc cai tien

Thu tu nen lam:

1. Train/evaluate v5 truoc, ghi mAP/precision/recall/time vao `EXPERIMENTS.md`.
2. Neu v5 tot hon v3: fine-tune nhe quanh assignment/head, vi huong nay co tin hieu.
3. Neu v5 te hon v3: revert `decoupled_head` hoac `task_aligned` rieng le de biet thanh phan nao gay te.
4. Neu precision van qua thap: xem per-class predictions, dac biet `chair`; can nhac post-hoc class threshold tuning ngoai train, khong bat grid trong train.
