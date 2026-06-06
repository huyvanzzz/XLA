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

Trạng thái:
- Version này bị loại/reset vì train lâu và kết quả không ổn.
- Config hiện tại đã quay về v3: `assignment_strategy: legacy`, `decoupled_head: false`, `objectness_iou_mix: 0.25`; chỉ giữ cải tiến nhẹ là tắt aux head từ epoch 30.

## v6_v3_reset_lightweight

Ngày:

Mô tả:
- Reset active config về hướng v3 để giữ tốc độ train.
- Không dùng Task-Aligned Assignment vì phần này làm train chậm đáng kể.
- Không dùng Decoupled Detection Head vì tăng compute và chưa chứng minh tốt hơn.
- Giữ direct resize `512x512`, không letterbox.
- Giữ ResNet50 pretrained backbone, chỉ fine-tune `resnet.layer4`.
- Giữ CIoU loss, legacy top-k anchor assignment, quality-aware objectness nhẹ.
- Tắt auxiliary head từ epoch 30 để train giai đoạn sau nhẹ hơn, không tăng thời gian train.
- Đổi post-processing từ hard NMS sang DIoU-NMS tự cài để xử lý các box trùng/lệch tâm tốt hơn mà không làm train chậm.
- Giữ mAP làm tiêu chí chọn `best.pth`, bắt đầu tính từ epoch 30.
- Hướng cải tiến tiếp theo nên nằm ngoài train hoặc chi phí rất thấp: post-hoc threshold/NMS per class, resume checkpoint, hoặc tối ưu validation/predict.

Config chính:
- `image_size: 512`
- `preserve_aspect: false`
- `batch_size: 24`
- `val_batch_size: 64`
- `assignment_strategy: legacy`
- `positive_anchor_topk: 3`
- `objectness_iou_mix: 0.25`
- `decoupled_head: false`
- `aux_head_close_epoch: 30`
- `conf_threshold: 0.08`
- `nms_threshold: 0.5`
- `nms_type: diou`

Kết quả:
- mAP@0.5:
- Precision:
- Recall:
- Thời gian train/epoch:
- Thời gian mAP validation:
- Tổng epoch train:
- Epoch tốt nhất:
- Ghi chú:

## v7_convnext_small_pretrained

Ngày:

Mô tả:
- Kế thừa pipeline nhanh của v6: legacy anchor assignment, head cũ, không task-aligned, không decoupled head.
- Đổi backbone từ ResNet50 pretrained sang ConvNeXt-Small pretrained ImageNet.
- ConvNeXt được tự cài trong `models/tiny_detector.py`, chỉ load pretrained weights classification; detector head/loss/NMS vẫn tự cài.
- Dùng 3 feature scale của ConvNeXt: stride 8/16/32 đưa vào FPN/PAN hiện tại.
- Vẫn giữ direct resize `512x512`, không letterbox, không mosaic, không TTA.
- DIoU-NMS vẫn bật trong post-processing.

Config chính:
- `model.backbone: convnext_small`
- `model.pretrained: true`
- `assignment_strategy: legacy`
- `decoupled_head: false`
- `nms_type: diou`
- `batch_size: 24` nếu không OOM; nếu OOM giảm xuống `16`.

Kết quả:
- mAP@0.5:
- Precision:
- Recall:
- Thời gian train/epoch:
- Thời gian mAP validation:
- Tổng epoch train:
- Epoch tốt nhất:
- Ghi chú:

## v8_convnext_aligned_neck_head

Ngày:

Mô tả:
- Kế thừa v7: ConvNeXt-Small pretrained backbone tự cài, direct resize `512x512`, legacy anchor assignment.
- Thiết kế lại neck/head để hợp ConvNeXt hơn thay vì dùng neck/head kiểu ResNet/YOLO cũ.
- Neck mới `ConvNeXtPANNeck`: vẫn giữ FPN/PAN multi-scale, nhưng fusion dùng `LayerNorm2d + depthwise 7x7 + GELU + pointwise`.
- Downsample trong PAN tách channel projection và spatial downsample, theo tinh thần spatial-channel decoupled downsampling của YOLOv10.
- Head mới `ConvNeXtDetectionHead`: dùng projection + ConvNeXt-style depthwise block, giữ nguyên output layout `[x, y, w, h, obj, class...]`.
- Không dùng task-aligned assignment, không dùng decoupled head nặng, không dùng TTA/mosaic.
- Mục tiêu: feature fusion hợp phân phối ConvNeXt hơn, giảm lệch giữa pretrained backbone và detector neck/head.

Config chính:
- `model.backbone: convnext_small`
- `model.neck_type: convnext_pan`
- `model.head_type: convnext`
- `model.pretrained: true`
- `assignment_strategy: legacy`
- `decoupled_head: false`
- `nms_type: diou`

Kết quả:
- mAP@0.5:
- Precision:
- Recall:
- Thời gian train/epoch:
- Thời gian mAP validation:
- Tổng epoch train:
- Epoch tốt nhất:
- Ghi chú:

## v9_convnext_pan_lite

Ngày:

Mô tả:
- Kế thừa v8 nhưng giảm phần làm train chậm.
- Giữ ConvNeXt-Small pretrained backbone.
- Giữ `ConvNeXtPANNeck`, nhưng giảm fusion depth từ 2 xuống 1 qua `elan_depth: 1`.
- Đổi head về `standard` để tránh LayerNorm/depthwise ConvNeXt block ở cả main head và aux head.
- Vẫn không dùng task-aligned, không decoupled head, không mosaic/TTA.
- Mục tiêu: giữ neck hợp ConvNeXt hơn v7, nhưng train nhẹ hơn v8.

Config chính:
- `model.backbone: convnext_small`
- `model.neck_type: convnext_pan`
- `model.head_type: standard`
- `model.elan_depth: 1`
- `assignment_strategy: legacy`
- `nms_type: diou`

Kết quả:
- mAP@0.5:
- Precision:
- Recall:
- Thời gian train/epoch:
- Thời gian mAP validation:
- Tổng epoch train:
- Epoch tốt nhất:
- Ghi chú:

## v10_resnet50_yolov7_pan

Ngày:

Mô tả:
- Reset active backbone về `resnet50` pretrained theo yêu cầu.
- Đọc YOLOv7 repo và thiết kế lại neck theo logic chính, không copy detector hoàn chỉnh.
- Thêm `SPPCSPCBlock` ở feature sâu P5.
- Thêm `YoloV7ELANFusion` cho các bước concat/fusion trong FPN/PAN.
- Thêm `YoloV7Downsample` hai nhánh: maxpool route + stride-2 conv route.
- Giữ head YOLO-style tự cài hiện tại với RepConv, không dùng YOLOv7 Detect/IDetect.
- Giữ loss, assignment, NMS, predict pipeline tự cài.
- Không dùng task-aligned, không decoupled head, không mosaic/TTA.

Config chính:
- `model.backbone: resnet50`
- `model.neck_type: yolov7_pan`
- `model.head_type: standard`
- `model.pretrained: true`
- `assignment_strategy: legacy`
- `nms_type: diou`

Kết quả:
- mAP@0.5:
- Precision:
- Recall:
- Thời gian train/epoch:
- Thời gian mAP validation:
- Tổng epoch train:
- Epoch tốt nhất:
- Ghi chú:

## v11_resnet50_yolov7_loss_decode

Ngày:

Mô tả:
- Kế thừa v10 ResNet50 + YOLOv7-style PAN neck.
- Chỉnh loss/decode theo các ý tưởng chính của YOLOv7, nhưng vẫn tự cài trong `utils/loss.py` và `utils/inference.py`.
- Decode bbox dùng công thức YOLOv7: center `sigmoid * 2 - 0.5`, width/height `(sigmoid * 2)^2 * anchor`.
- Assignment legacy thêm positive ở các cell lân cận khi tâm object gần biên cell, giống offset target của YOLOv7.
- Objectness target dùng IoU ratio mạnh hơn: `objectness_iou_mix: 1.0`.
- Objectness balance theo scale `[4.0, 1.0, 0.4]` để ưu tiên P3/small-object như YOLOv7.
- Bỏ SmoothL1 bbox khỏi tổng loss (`box_weight: 0.0`), dựa chính vào CIoU loss.
- Không dùng OTA vì nặng và phụ thuộc YOLOv7 Detect head; vẫn giữ assignment tự cài nhẹ hơn.

Config chính:
- `box_weight: 0.0`
- `iou_weight: 6.0`
- `decode_style: yolov7`
- `target_offsets: true`
- `target_offset_bias: 0.5`
- `scale_obj_balance: [4.0, 1.0, 0.4]`
- `objectness_iou_mix: 1.0`

Kết quả:
- mAP@0.5: 72,8
- Precision:
- Recall:
- Thời gian train/epoch: 8'
- Thời gian mAP validation: 55s
- Tổng epoch train: 55
- Epoch tốt nhất:
- Ghi chú:

## v12_convnext_small_frozen_fast

Ngay:

Mo ta:
- Huong ConvNeXt moi de hop backbone manh hon nhung khong lam train-time tang qua nhieu.
- Dung `convnext_small` pretrained, khong dung Tiny.
- Freeze backbone trong toan bo 80 epoch bang `freeze_backbone_epochs: 80` va `backbone_trainable: none`.
- Ly do freeze: van tan dung pretrained feature extractor manh, nhung tranh backward qua ConvNeXt-Small, phan thuong lam train cham nhat.
- Khong dung `convnext_pan`/ConvNeXt head nang cua v8-v9 vi LayerNorm/depthwise/permute lam train cham.
- Dung lai `fpnpan` + `standard` head, giam `neck_channels/head_channels` xuong 160 de detector adapter nhanh hon.
- Giu direct resize `512x512`, khong letterbox.
- Giu YOLOv7-style decode/loss cua v11: decode `sigmoid*2-0.5`, `(sigmoid*2)^2*anchor`, target offsets, scale obj balance.
- Giu DIoU-NMS tu cai, khong dung torchvision NMS.
- Giu validation mAP bat dau tu epoch 30 va chon `best.pth` theo `mAP@0.5`.

Config chinh:
- `model.backbone: convnext_small`
- `model.pretrained: true`
- `model.neck_type: fpnpan`
- `model.head_type: standard`
- `model.neck_channels: 160`
- `model.head_channels: 160`
- `freeze_backbone_epochs: 80`
- `backbone_trainable: none`
- `batch_size: 24`
- `conf_threshold: 0.08`
- `decode_style: yolov7`
- `nms_type: diou`

Ket qua:
- mAP@0.5:
- Precision:
- Recall:
- Thoi gian train/epoch:
- Thoi gian mAP validation:
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:

## v13_resnet50_v11_merge_nms

Ngay:

Mo ta:
- Reset active config ve nen v11 vi ConvNeXt khong ngon bang ky vong.
- Dung lai `resnet50` pretrained + `yolov7_pan`.
- Dung lai `head_channels: 192`, `neck_channels: 192`.
- Dung lai fine-tune nhe `resnet.layer4` sau 2 epoch warmup.
- Giu YOLOv7-style loss/decode cua v11: target offsets, objectness IoU mix 1.0, scale obj balance `[4.0, 1.0, 0.4]`, CIoU la bbox loss chinh.
- Cai tien moi: bat `merge_nms: true` trong validation/inference.
- Merge-NMS fuse toa do box sau NMS bang trung binh co trong so theo score voi cac box cung class co IoU cao.
- Them class-prior bias init: khoi tao class logits theo phan phoi class trong train annotation, khong them compute luc train.
- Muc tieu: cai thien localization/ranking mAP@0.5 ma khong tang thoi gian train/backward. Chi co validation/predict them mot buoc post-process nho.

Config chinh:
- `model.backbone: resnet50`
- `model.neck_type: yolov7_pan`
- `model.head_type: standard`
- `model.neck_channels: 192`
- `model.head_channels: 192`
- `freeze_backbone_epochs: 2`
- `backbone_trainable: layer4`
- `decode_style: yolov7`
- `nms_type: diou`
- `merge_nms: true`
- `class_prior_bias.enabled: true`

Ket qua:
- mAP@0.5:
- Precision:
- Recall:
- Thoi gian train/epoch:
- Thoi gian mAP validation:
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:

## v14_resnet50_yolov7_bce_late_finetune

Ngay:

Mo ta:
- Cai tien rong hon tren nen v11/v13, van giu ResNet50 pretrained + YOLOv7 PAN/decode/loss core.
- Doi classification loss tu CE/softmax sang BCE/sigmoid theo logic YOLO-style head: moi class logit doc lap, inference lay `obj * max(sigmoid(class))`.
- Giu CIoU/objectness/target-offset cua v11, khong dung OTA/task-aligned vi qua nang.
- Giu merge-NMS va class-prior bias init cua v13.
- Them late clean fine-tune: tu epoch 55 tat random crop, random scale, random erasing de cac epoch cuoi hoc tren phan phoi anh that hon.
- Them `lr_final_factor: 0.05` de cosine LR khong roi sat 0 qua som, giu kha nang tinh chinh sau epoch 55.
- Muc tieu: cai thien ranking/class confidence va on dinh box cuoi train, trong khi train/epoch gan nhu giu bang v11/v13.

Config chinh:
- `model.backbone: resnet50`
- `model.neck_type: yolov7_pan`
- `loss_weights.classification_loss: bce`
- `inference.class_activation: sigmoid`
- `validation_metric.class_activation: sigmoid`
- `augmentation.close_strong_aug_epoch: 55`
- `lr_final_factor: 0.05`
- `merge_nms: true`
- `class_prior_bias.enabled: true`

Ket qua:
- mAP@0.5: 73.5
- Precision:
- Recall:
- Thoi gian train/epoch: 6'
- Thoi gian mAP validation: 55s
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:

## v15_yolov7_freebies_quality_anchor_ema

Ngay:

Mo ta:
- Cai tien tiep tren nen v14 da dat mAP@0.5 = 73.5.
- Van giu ResNet50 pretrained + YOLOv7 PAN + YOLOv7 decode/loss core.
- Khong them backbone/head/assignment nang, nen thoi gian train moi epoch khong tang dang ke.
- Them quality-aware BCE classification target: positive class target duoc tron nhe voi IoU quality (`classification_quality_mix: 0.25`) de confidence ranking gan AP hon.
- Them scale-aware objectness bias init theo stride, dua tren cong thuc bias cua YOLOv7 (`nominal_objects: 8.0`).
- Them EMA ramp theo YOLOv7: EMA decay tang dan theo update, tranh EMA qua i o dau train.
- Cai tien auto anchors: sau kmeans co genetic/evolution nhe theo ratio fitness nhu YOLO autoanchor. Phan nay chi chay truoc train, khong tang time/epoch.
- Giu late clean fine-tune, BCE/sigmoid class path, merge-NMS, class-prior bias cua v14.

Config chinh:
- `loss_weights.classification_loss: bce`
- `loss_weights.classification_quality_mix: 0.25`
- `objectness_bias.enabled: true`
- `objectness_bias.nominal_objects: 8.0`
- `anchors.evolve_generations: 150`
- `anchors.anchor_threshold: 4.0`
- `ema.decay: 0.999`
- `ema.tau: 2000`
- `inference.class_activation: sigmoid`
- `merge_nms: true`

Ket qua:
- mAP@0.5: 72.06
- Precision:
- Recall:
- Thoi gian train/epoch:
- Thoi gian mAP validation:
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:

Trang thai:
- User bao phien ban hien tai khong tot, khong nen tiep tuc huong v15.
- Cac phan rui ro can reset: `classification_quality_mix`, objectness bias, anchor evolution, EMA ramp.

## v16_quality_prediction_head

Ngay:

Mo ta:
- Doi huong rong hon sau khi v15 khong tot.
- Reset cac phan rui ro cua v15: `classification_quality_mix: 0.0`, `objectness_bias.enabled: false`, `anchors.evolve_generations: 0`, `ema.tau: 0`.
- Giu nen da tot cua v14: ResNet50 pretrained, YOLOv7 PAN, YOLOv7 decode/loss core, BCE/sigmoid class path, late clean fine-tune, merge-NMS, class-prior bias.
- Them quality prediction head: moi anchor them 1 logit du doan bbox quality/IoU.
- Loss moi `quality_weight: 0.35`: quality logit hoc target la IoU cua positive bbox.
- Inference dung score moi: `score = objectness * class_score * quality^0.5`.
- Muc tieu: cai thien ranking cua prediction theo chat luong bbox, giam truong hop confidence cao nhung box kem, ma chi tang 1 output channel/anchor nen train time gan nhu khong tang.
- Huong nay dua tren logic quality-aware detection cua YOLOv6/GFL/QFL, nhung tu cai cuc bo va khong dung framework/model co san.

Config chinh:
- `model.quality_head: true`
- `loss_weights.quality_weight: 0.35`
- `inference.quality_score_power: 0.5`
- `validation_metric.quality_score_power: 0.5`
- `loss_weights.classification_quality_mix: 0.0`
- `objectness_bias.enabled: false`
- `anchors.evolve_generations: 0`
- `ema.tau: 0`

Ket qua:
- mAP@0.5: 73.17
- Precision:
- Recall:
- Thoi gian train/epoch:
- Thoi gian mAP validation:
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:

Trang thai:
- User bao v16 van thua v14, chi duoc 73.17.
- Khong tiep tuc quality head cho active config.

## v17_v14_eca_neck_tta

Ngay:

Mo ta:
- Doi huong sau khi v16 thua v14.
- Reset active config ve gan v14: tat `quality_head`, `quality_weight`, `quality_score_power`.
- Giu ResNet50 pretrained + YOLOv7 PAN + BCE/sigmoid + late clean fine-tune + merge-NMS + class-prior bias.
- Them ECA channel attention rat nhe vao 3 output scale cua YOLOv7 PAN neck.
- Y tuong dua theo huong YOLOv12/attention-centric nhung dung attention sieu nhe, khong dung transformer/PSA nang.
- Bat hflip TTA chi trong `predict.py`/inference (`inference.tta_hflip: true`), khong bat trong train validation (`validation_metric.tta_hflip: false`) de khong lam train map cham.
- Muc tieu: tang kha nang chon kenh feature va tang final mAP bang test-time augmentation, trong khi train/epoch chi tang rat nhe do ECA va validation train khong TTA.

Config chinh:
- `model.quality_head: false`
- `model.neck_attention: eca`
- `loss_weights.quality_weight: 0.0`
- `inference.quality_score_power: 0.0`
- `inference.tta_hflip: true`
- `validation_metric.tta_hflip: false`
- `merge_nms: true`

Ket qua:
- mAP@0.5:
- Precision:
- Recall:
- Thoi gian train/epoch:
- Thoi gian mAP validation:
- Thoi gian predict/evaluate final:
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:

## v18_v17_slight_capacity_aug

Ngay:

Mo ta:
- Giu nguyen y tuong v17: ResNet50 pretrained, YOLOv7 PAN, ECA neck attention, BCE/sigmoid, merge-NMS, DIoU-NMS, hflip TTA chi cho final inference.
- Tang tham so nhe o detector neck/head, khong doi backbone: `neck_channels/head_channels` tu 192 len 208, `cls_head_channels` tu 128 len 144.
- Tang dropout tu 0.10 len 0.12 de bu lai rui ro overfit khi head/neck lon hon.
- Augmentation tang nhe nhung van tranh mosaic/letterbox: crop/scale xac suat cao hon mot chut, random erasing nhe hon, color jitter configurable va rong hon.
- Muc tieu: tang suc bieu dien cua phan detector ma khong lam train cham manh nhu doi backbone, task-aligned assignment, decoupled head, hay TTA trong validation.

Config chinh:
- `model.neck_channels: 208`
- `model.head_channels: 208`
- `model.cls_head_channels: 144`
- `model.dropout: 0.12`
- `augmentation.random_crop_prob: 0.25`
- `augmentation.random_scale_prob: 0.5`
- `augmentation.random_erasing_prob: 0.2`
- `augmentation.random_erasing_max_area: 0.1`
- `augmentation.color_jitter_prob: 0.35`
- `augmentation.color_jitter_min/max: 0.7/1.3`

Ket qua:
- mAP@0.5: 74.1
- Precision:
- Recall:
- Thoi gian train/epoch: 6'
- Thoi gian mAP validation:
- Thoi gian predict/evaluate final:
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:

## v19_v18_head_eca

Ngay:

Mo ta:
- Giu nguyen nen v18 dang tot, user bao dat khoang 74.1 mAP.
- Khong tang tiep channels de tranh train cham/overfit sau khi config thuc te da len `neck/head_channels: 224` va `cls_head_channels: 160`.
- Them `head_attention: eca` cho main DetectionHead sau RepConv va truoc conv du doan cuoi.
- Aux head van giu khong attention de khong lam 30 epoch dau nang them nhieu.
- Y tuong: neck da co ECA de chon kenh feature fuse, head them ECA nhe de cai thien channel ranking ngay truoc object/class/box logits.

Config chinh:
- `model.head_attention: eca`
- Giu `model.neck_attention: eca`
- Giu `model.neck_channels: 224`
- Giu `model.head_channels: 224`
- Giu `model.cls_head_channels: 160`

Ket qua:
- mAP@0.5:
- Precision:
- Recall:
- Thoi gian train/epoch:
- Thoi gian mAP validation:
- Thoi gian predict/evaluate final:
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:

## v20_convnext_small_fast_pan_eff_head

Ngay:

Mo ta:
- Doi huong rong hon theo yeu cau: dung backbone manh hon ResNet50 la `ConvNeXt-Small` pretrained.
- Khong chi thay backbone: them neck moi `convnext_fast_pan` de hop ConvNeXt nhung tranh loi cham cua v8/v9.
- Neck moi dung FPN/PAN, projection 1x1, large-kernel depthwise 7x7, BN/GELU, residual layer-scale va ECA, tat ca giu NCHW de tranh LayerNorm/permute trong adapter.
- Them `EfficientDecoupledHead`: reg/objectness tower manh hon, classification tower nhe bang depthwise conv, theo tinh than YOLOv6/YOLOv10 rang cls head nen tiet kiem compute hon reg head.
- Giu YOLOv7-style decode/loss/target offsets/scale objectness balance, DIoU-NMS va merge-NMS tu cai.
- De kiem soat thoi gian: chi fine-tune `convnext.stage4`, backbone LR giam 0.1, batch size ve 16, neck/head 192 thay vi 224.

Config chinh:
- `model.backbone: convnext_small`
- `model.neck_type: convnext_fast_pan`
- `model.head_type: efficient_decoupled`
- `model.neck_channels: 192`
- `model.head_channels: 192`
- `model.cls_head_channels: 96`
- `model.neck_attention: eca`
- `model.head_attention: eca`
- `batch_size: 16`
- `val_batch_size: 48`
- `backbone_trainable: layer4`
- `backbone_lr_mult: 0.1`
- `freeze_backbone_epochs: 3`

Ket qua:
- mAP@0.5: 70
- Precision:
- Recall:
- Thoi gian train/epoch: 10'
- Thoi gian mAP validation:
- Thoi gian predict/evaluate final:
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:

## v21_resnet50_channel_spatial_attention

Ngay:

Mo ta:
- Reset active config ve nhanh ResNet50 tot nhat sau khi huong ConvNeXt hien tai khong on.
- Giu nen user bao tot khoang 74.1 mAP: ResNet50 pretrained, YOLOv7 PAN, direct resize 512, BCE/sigmoid, DIoU/merge-NMS, augmentation v18, capacity detector 224/224/160.
- Cai tien rong nhe tren nen do: mo rong attention tu ECA channel-only sang `eca_spatial`, gom ECA chon kenh + spatial attention 7x7 tu avg/max map de tap trung vung object.
- Ap dung `eca_spatial` cho output cua YOLOv7 PAN neck va main DetectionHead; aux head van giu nhe.
- Huong nay dua theo logic YOLOv11/YOLOv12 ve spatial/attention module, nhung dung ban cuc re, khong transformer/window attention, khong tang assignment/NMS.
- Tat duong ConvNeXt trong config active; code ConvNeXt van con de tham khao nhung khong duoc dung khi train voi config hien tai.

Config chinh:
- `model.backbone: resnet50`
- `model.neck_type: yolov7_pan`
- `model.head_type: standard`
- `model.neck_channels: 224`
- `model.head_channels: 224`
- `model.cls_head_channels: 160`
- `model.neck_attention: eca_spatial`
- `model.head_attention: eca_spatial`
- `batch_size: 24`
- `val_batch_size: 64`
- `freeze_backbone_epochs: 2`
- `backbone_lr_mult: 0.2`

Ket qua:
- mAP@0.5:
- Precision:
- Recall:
- Thoi gian train/epoch:
- Thoi gian mAP validation:
- Thoi gian predict/evaluate final:
- Tong epoch train:
- Epoch tot nhat:
- Ghi chu:
