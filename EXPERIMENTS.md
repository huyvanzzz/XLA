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
- Dùng global inference/validation `conf_threshold` 0.08 để giữ recall khi chọn best theo mAP.
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
- `conf_threshold: 0.08`

Kết quả:
- mAP@0.5: 0.6
- Precision:
- Recall:
- Thời gian train/epoch: 4'30
- Thời gian mAP validation: 55
- Tổng epoch train: 60
- Epoch tốt nhất:
- Ghi chú:

## v3_direct_resize_regularized_finetune

Ngày:

Mô tả:
- Kế thừa direct resize `512x512`, không dùng letterbox.
- Giữ ResNet50 pretrained backbone.
- Chỉ fine-tune `resnet.layer4` sau warmup, các block backbone thấp hơn vẫn freeze.
- Freeze BatchNorm stats của backbone pretrained khi train.
- Thêm random erasing/cutout augmentation để giảm overfit.
- Giữ CIoU loss, top-k anchor assignment, ignore mask.
- Giữ quality-aware objectness nhẹ: target objectness = `0.75 + 0.25 * IoU`.
- Giữ balanced image sampling và boost class weight cho `chair`.
- Giữ optimizer AdamW param groups, channels-last, CuDNN benchmark.

Config chính:
- `image_size: 512`
- `preserve_aspect: false`
- `batch_size: 24`
- `val_batch_size: 64`
- `num_workers: 4`
- `backbone_trainable: layer4`
- `backbone_freeze_bn: true`
- `random_erasing_prob: 0.25`
- `head_channels: 192`
- `neck_channels: 192`
- `attention_heads: 0`
- `balanced_sampling.enabled: true`
- `class_weights.overrides.chair: 1.25`
- `conf_threshold: 0.1`

Kết quả:
- mAP@0.5: 61.5
- Precision:
- Recall:
- Thời gian train/epoch: 4'30
- Thời gian mAP validation: 55s
- Tổng epoch train: 70
- Epoch tốt nhất: 60
- Ghi chú:

## v4_hard_negative_tta Rất tệ

Ngày:

Mô tả:
- Kế thừa v3.
- Thêm hard-negative mining cho no-objectness loss: chỉ tập trung phần loss background vào các negative anchors khó nhất.
- Mục tiêu là giảm false positives, vì các version trước có số predictions rất cao và precision thấp.
- Thêm horizontal flip TTA trong `predict.py`: chạy thêm ảnh lật ngang, flip bbox về ảnh gốc, rồi merge bằng NMS tự cài.
- TTA chỉ bật ở inference mặc định, không bật trong validation train để không làm chậm epoch.
- Thêm mosaic augmentation tự cài: mỗi sample có thể ghép 4 ảnh thành một ảnh train `512x512`.
- Tắt mosaic từ epoch 55 để các epoch cuối fine-tune lại trên phân phối ảnh thật.
- Trong lúc train chỉ dùng một ngưỡng cố định để tiết kiệm thời gian; threshold tuning nên làm sau bằng `predict.py`/config nếu cần.
- Vẫn không dùng detector/NMS có sẵn; NMS và merge đều tự cài.

Config chính:
- `noobj_hard_negative_ratio: 0.1` (đã tắt lại trong config hiện tại)
- `noobj_hard_negative_min: 256`
- `inference.tta_hflip: true` (đã tắt lại trong config hiện tại)
- `validation_metric.tta_hflip: false`
- `augmentation.mosaic_prob: 0.35` (đã tắt lại trong config hiện tại)
- `augmentation.close_mosaic_epoch: 55`
- `validation_metric.tune: false`
- `validation_metric.tune_every: 5`
- `validation_metric.conf_threshold: 0.08`
- `validation_metric.nms_threshold: 0.5`

Kết quả:
- mAP@0.5:
- Precision:
- Recall:
- Thời gian train/epoch:
- Thời gian mAP validation:
- Thời gian predict val:
- Tổng epoch train:
- Epoch tốt nhất:
- Ghi chú:

Trạng thái:
- Version này bị loại vì kết quả tệ.
- Config hiện tại đã reset về hướng v3: không mosaic, không hard-negative mining, không hflip TTA.

## v5_task_aligned_assignment

Ngày:

Mô tả:
- Kế thừa v3: direct resize `512x512`, ResNet50 pretrained backbone, chỉ fine-tune `resnet.layer4`, không letterbox, không mosaic, không TTA.
- Hướng cải tiến chính là thay label assignment cũ bằng Task-Aligned Assignment, dựa trên ý tưởng YOLOv6/TOOD: positive sample được chọn theo cả classification confidence và IoU của box hiện tại.
- Vẫn giữ anchor-based head from scratch, nhưng target không còn chỉ dựa vào anchor width/height tại một cell nữa.
- Positive target objectness được gắn theo chất lượng box: `0.6 + 0.4 * IoU`, giúp confidence ranking gần hơn với bbox quality.
- Đổi detection head sang decoupled head: bbox/objectness tower và classification tower riêng, giúp hai nhiệm vụ không tranh cùng feature cuối.
- Tắt auxiliary detection heads từ epoch 30 để giảm compute giai đoạn sau và tránh aux loss kéo model quá lâu.
- Mục tiêu: giảm positive kém chất lượng, tăng chất lượng classification/localization, giảm FP sau NMS, tăng AP mà không làm predict/validation chậm hơn vì output inference không đổi; train sau epoch 30 cũng nhẹ hơn.

Config chính:
- `assignment_strategy: task_aligned`
- `positive_anchor_topk: 8`
- `task_aligned_alpha: 0.5`
- `task_aligned_beta: 6.0`
- `task_aligned_center_radius: 2.5`
- `task_aligned_min_iou: 0.05`
- `objectness_iou_mix: 0.4`
- `model.decoupled_head: true`
- `model.cls_head_channels: 128`
- `model.aux_head_close_epoch: 30`
- `conf_threshold: 0.08`
- `nms_threshold: 0.5`

Kết quả:
- mAP@0.5:
- Precision:
- Recall:
- Thời gian train/epoch:
- Thời gian mAP validation:
- Tổng epoch train:
- Epoch tốt nhất:
- Ghi chú:
