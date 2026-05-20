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
Mục `model` điều chỉnh backbone/head. Mặc định dùng `resnet34` pretrained ImageNet làm backbone, còn detection head, loss, decode và NMS vẫn tự triển khai.

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

- Backbone mặc định là ResNet34 pretrained ImageNet từ `torchvision.models`, lấy feature tới `layer3` để giữ stride 16.
- Có thể đổi `model.backbone: custom` để dùng CNN tự viết bằng `Conv2d`, `BatchNorm2d`, `SiLU`, residual blocks và SPP.
- Detection head dự đoán trên lưới stride 16, tức ảnh `416x416` cho feature map `26x26`.
- Mỗi ô lưới có 5 anchor, cấu hình trong `configs/default.yaml`.
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
```

Ground-truth box được gán vào ô lưới chứa tâm box. Trong ô đó, anchor có IoU theo width/height cao nhất sẽ là positive anchor.

## Inference

`predict.py` thực hiện:

- Decode output YOLO về bbox trên ảnh resize.
- Scale bbox về kích thước ảnh gốc.
- Lọc theo confidence threshold.
- NMS riêng từng lớp.
- Xuất JSON đúng format của đề, kể cả ảnh không có object vẫn có `"boxes": []`.
