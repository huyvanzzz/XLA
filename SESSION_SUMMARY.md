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
- Post-processing active: DIoU-NMS tu cai, khong dung torchvision.
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
- `inference.nms_type: diou`
- `validation_metric.nms_type: diou`

Model hien tai:
- `backbone: convnext_small`
- `pretrained: true`
- `neck_channels: 192`
- `head_channels: 192`
- `attention_heads: 0`
- `neck_type: convnext_pan`
- `head_type: convnext`
- `aux_head: true`
- `aux_head_close_epoch: 30`
- `decoupled_head: false`
- `cls_head_channels: 128`

Loss/assignment hien tai da reset ve v3:
- `assignment_strategy: legacy`
- `positive_anchor_topk: 3`
- `task_aligned_alpha: 0.5`
- `task_aligned_beta: 6.0`
- `task_aligned_center_radius: 2.5`
- `task_aligned_min_iou: 0.05`
- `iou_aware_objectness: true`
- `objectness_iou_mix: 0.25`
- `noobj_hard_negative_ratio: 0.0`

## Version da ghi

Xem chi tiet trong `EXPERIMENTS.md`.

- v1: baseline YOLO-style ResNet50 pretrained, mAP public val `0.604047`.
- v2/v3: direct resize 512, train nhanh hon; user bao v3 len khoang `0.615`, train/epoch khoang `4'30`, best gan epoch 60.
- v4: hard-negative + mosaic + TTA, ket qua rat te; da reset/tat.
- v5: Task-Aligned Assignment + Decoupled Detection Head + tat aux head tu epoch 30. Bi reset vi train lau va khong on.
- Hien tai active: v8 dung ConvNeXt-Small pretrained tu cai lam backbone, kem ConvNeXt-style PAN neck/head. Khong dung task-aligned, khong dung decoupled head. Cai tien nhe la tat aux head tu epoch 30 va dung DIoU-NMS trong post-processing.

## Ly do reset ve v3

V5 lam train cham do task-aligned phai decode nhieu candidate va tinh IoU de assign positive; decoupled head cung them compute. Vi ket qua khong on, active config da reset ve v3 de giu toc do train.

Sau do da them v7: thay ResNet50 pretrained bang ConvNeXt-Small pretrained. ConvNeXt-Small manh hon Tiny va ResNet50 ve feature, nhung co the ton VRAM/thoi gian hon. Neu Kaggle OOM, giam `batch_size` tu 24 xuong 16.

V8 sua tiep kien truc cho hop ConvNeXt:
- `ConvNeXtPANNeck`: FPN/PAN nhung fusion bang LayerNorm2d + depthwise 7x7 + GELU + pointwise.
- `ConvNeXtDetectionHead`: head depthwise ConvNeXt-style, output format giu nguyen.
- Ly do: paper YOLOv10/YOLOv12 nhan manh depthwise/lightweight/large-kernel va feature fusion hieu qua; ConvNeXt backbone cung dung LN/GELU/depthwise large-kernel nen neck/head nen cung he.

Neu tiep tuc cai tien, uu tien cac huong khong tang thoi gian train:
- post-processing / per-class threshold sau train;
- DIoU-NMS dang duoc bat mac dinh vi khong tang train time;
- checkpoint/resume cho Kaggle;
- ablation nhe trong config, khong them assignment dong nang;
- giam validation/predict overhead, khong them TTA/grid tuning trong train.

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
- Forward/loss v5 tung test OK, nhung khong con la active config.

## Neu tiep tuc cai tien

Thu tu nen lam:

1. Train/evaluate active v3-reset truoc, ghi mAP/precision/recall/time vao `EXPERIMENTS.md`.
2. Neu can cai tien ma khong tang train time: lam post-hoc threshold/NMS per class sau train, hoac sua predict/evaluate pipeline.
3. Neu can sua model/loss: chi them cai co chi phi gan nhu bang 0; khong quay lai task-aligned/decoupled neu user khong chap nhan train cham.
4. Neu precision van qua thap: xem per-class predictions, dac biet `chair`; can nhac post-hoc class threshold tuning ngoai train, khong bat grid trong train.
