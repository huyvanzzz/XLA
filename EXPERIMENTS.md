# Experiment Versions

Chỉ ghi các version được đem đi train/evaluate thật sự. Các chỉ số để trống để điền sau khi chạy.

Mục tiêu: kết quả mAP@0.5 trên validation là 0.8.

## v1_initial_train

Ngày:

Mô tả:
- YOLO-style detector với ResNet50 pretrained backbone.
- Resize ảnh trực tiếp về hình vuông 512*512.
- FPN/PAN neck bản 256 channels.
- Có partial self-attention.
- Top-k anchor assignment, hard NMS, chọn `best.pth` theo `mAP@0.5`.

Kết quả:
- Train_loss: 0.85
- mAP@0.5: 0.604047
- Precision: 0.060136
- Recall: 0.873330
- Thời gian train/epoch: 5'30s
- Thời gian mAP validation: 55s
- Tổng epoch train: 45. (Tính tại epoch tốt nhất)
- Ghi chú: Thời gian train chậm, kết quả cũng không quá cao.

## v2_current_direct_resize_fast_finetune

Ngày:

Mô tả:
- Giữ ResNet50 pretrained backbone.
- Quay lại resize trực tiếp về hình vuông như v1, không dùng letterbox.
- Auto anchors fit theo direct resize.
- Dùng CIoU loss cho bbox decoded.
- Objectness bias init với prior 0.01.
- Optimizer AdamW chia param groups: backbone LR thấp hơn, không decay bias/BatchNorm/LayerNorm.
- Sau warmup chỉ unfreeze `resnet.layer4`, không fine-tune toàn bộ backbone.
- Freeze BatchNorm stats của backbone pretrained khi train để fine-tune ổn định hơn.
- `val_batch_size` riêng, channels-last và CuDNN benchmark khi có CUDA.
- Thêm random erasing/cutout augmentation sau resize để giảm overfit và tăng robustness.
- Balanced image sampling để tăng tần suất ảnh chứa class khó/ít hơn.
- Boost class weight cho `chair` vì kết quả trước có AP chair thấp nhất.
- Objectness quality-aware nhẹ: target objectness = `0.75 + 0.25 * IoU`, giúp ranking confidence tốt hơn nhưng không làm positive target quá thấp đầu train.
- Tăng global inference/validation `conf_threshold` lên 0.10 để giảm false positives confidence thấp.
- Thêm `class_conf_thresholds` trong config để sau này tune threshold riêng theo class mà không train lại.

Config chính:
- `image_size: 512`
- `preserve_aspect: false`
- `batch_size: 24`
- `val_batch_size: 64`
- `num_workers: 4`
- `lr: 0.0001`
- `backbone_lr_mult: 0.2`
- `freeze_backbone_epochs: 2`
- `backbone_trainable: layer4`
- `backbone_freeze_bn: true`
- `random_erasing_prob: 0.25`
- `neck_channels: 192`
- `head_channels: 192`
- `attention_heads: 0`
- `balanced_sampling.enabled: true`
- `class_weights.overrides.chair: 1.25`
- `iou_aware_objectness: true`
- `objectness_iou_mix: 0.25`
- `conf_threshold: 0.1`

Kết quả:
- mAP@0.5:
- Precision:
- Recall:
- Thời gian train/epoch:
- Thời gian mAP validation:
- Tổng epoch train:
- Epoch tốt nhất:
- Ghi chú:
