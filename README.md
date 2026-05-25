# From-Scratch Tiny YOLO Object Detector

Mô hình này là detector kiểu YOLO nhỏ tự cài bằng PyTorch. Code không dùng YOLOv5/v8, Detectron2, MMDetection hay Faster R-CNN/SSD có sẵn.

## Cài đặt

```bash
pip install -r requirements.txt
```

Nếu máy có GPU, cài bản PyTorch phù hợp CUDA theo hướng dẫn chính thức của PyTorch.

## Huấn luyện

Tham số chính nằm trong:

```text
configs/default.yaml
```

Bạn có thể sửa trực tiếp file YAML này để đổi `image_size`, `epochs`, `batch_size`, `lr`, `anchors`, `loss_weights`, `conf_threshold`, `nms_threshold`.
Mục `model` điều chỉnh backbone/neck/head. Mặc định dùng ResNet50 pretrained ImageNet làm backbone, còn FPN/PAN neck, attention/context blocks, YOLO heads, auxiliary heads, loss, decode và NMS vẫn tự triển khai.
Config cũng có các cơ chế chống overfit: random crop/scale augmentation, dropout trong neck/head, freeze backbone vài epoch đầu, early stopping theo mAP và EMA weights.
Các cơ chế tối ưu mAP gồm auto anchors từ train annotations, one-to-many top-k anchor assignment, focal objectness, IoU-aware objectness, class weights cho dữ liệu lệch lớp, multi-scale training, hard NMS tự cài theo lớp với pre-NMS top-k và chọn `best.pth` theo `mAP@0.5`.

Lệnh bắt buộc của đề:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Checkpoint tốt nhất được lưu tại:

```text
models/best.pth
```

Có thể override tạm thời bằng command line:

```bash
python train.py ... --epochs 80 --batch_size 16 --image_size 416 --lr 0.0002
```

Các trọng số loss cũng chỉnh trực tiếp được:

```bash
python train.py ... --box_weight 5.0 --obj_weight 1.0 --noobj_weight 0.5 --cls_weight 1.0
```

Nếu muốn dùng file config khác:

```bash
python train.py ... --config configs/experiment_01.yaml
```

## Suy luận

Lệnh bắt buộc của đề:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Mặc định `predict.py` đọc trọng số từ `models/best.pth`. Nếu muốn dùng checkpoint khác:

```bash
python predict.py --image_dir ./public/val/images --output val_predictions.json --checkpoint ./models/best.pth
```

## Kiểm tra trên validation

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

## Kiến trúc

- Backbone mặc định là ResNet50 pretrained ImageNet. Kiến trúc ResNet50 được cài trong repo và tải weight bằng `torch.hub`, không phụ thuộc `torchvision`.
- Có thể đổi `model.backbone: eelan` và `pretrained: false` để dùng E-ELAN-like CNN tự viết.
- Neck FPN/PAN fuse 3 scale stride 8/16/32 từ backbone, có SPP, large-kernel depthwise context và partial self-attention ở tầng sâu.
- Detection head dự đoán trên 3 scale stride 8/16/32, tức ảnh `512x512` cho feature maps `64x64`, `32x32`, `16x16`.
- Có auxiliary heads dùng khi train theo tinh thần trainable bag-of-freebies; inference chỉ dùng main heads.
- Mỗi scale có 3 anchors, cấu hình trong `configs/default.yaml`.
- Mỗi anchor dự đoán:
  - `tx, ty`: offset tâm box trong ô lưới.
  - `tw, th`: log-scale chiều rộng/cao so với anchor.
  - `object_logit`: điểm có object.
  - `class_logits`: logits cho 5 lớp.

## Loss

Loss nằm trong `utils/loss.py`, được tách rõ thành các phần:

- `box_loss`: Smooth L1 cho bbox của positive anchors.
- `iou_loss`: `1 - IoU` trên bbox đã decode của positive anchors.
- `obj_loss`: BCEWithLogits cho anchor được gán object.
- `noobj_loss`: BCEWithLogits cho background anchors.
- `cls_loss`: Cross Entropy cho class của positive anchors.

Công thức tổng:

```text
total_loss =
  7.5 * box_loss
  + 1.5 * iou_loss
  + 1.0 * obj_loss
  + 0.35 * noobj_loss
  + 1.0 * cls_loss
  + 0.4 * aux_loss
```

Ground-truth box được gán vào ô lưới chứa tâm box. Trong ô đó, top-k anchors có IoU theo width/height cao nhất sẽ là positive anchors để tăng tín hiệu học và recall.

## Inference

`predict.py` thực hiện:

- Decode output YOLO về bbox trên ảnh resize.
- Scale bbox về kích thước ảnh gốc.
- Lọc theo confidence threshold.
- NMS riêng từng lớp.
- Xuất JSON đúng format của đề, kể cả ảnh không có object vẫn có `"boxes": []`.
