# From-Scratch Anchor-Free Object Detector

This project implements a small from-scratch object detector for the five required classes: `person`, `car`, `dog`, `cat`, and `chair`. It does not use YOLOv5/v8, Detectron2, MMDetection, torchvision Faster R-CNN/SSD, or a complete detector framework. The pretrained ConvNeXtV2 backbone is used only as a feature extractor; the detector neck, head, assignment, loss, decode, NMS, training loop, and prediction JSON export are implemented in this repo.

## Install

```bash
pip install -r requirements.txt
```

Install the CUDA build of PyTorch that matches the machine if training on GPU.

## Train

Main hyperparameters are in:

```text
configs/default.yaml
```

The default design follows the research plan:

- `ConvNeXtV2-Nano` pretrained backbone, with stride 8/16/32 feature maps.
- Fixed `448x448` input for stable P100 throughput.
- Lightweight PAN neck with depthwise convolution and ECA attention.
- Decoupled anchor-free head. Each location predicts distributed `l,t,r,b` distances, five class-quality logits, and a localization-quality score.
- No separate objectness branch in the default anchor-free path.
- Center-prior plus task-aligned top-k assignment.
- Quality Focal Loss, CIoU, Distribution Focal Loss (`reg_max=8`), and localization-quality reranking.
- AMP, `cudnn.benchmark`, pinned memory, persistent workers, EMA, gradual backbone unfreezing, and train-time logging.
- Validation after each epoch prints overall `mAP@0.5`, per-class `AP@0.5`, precision, recall, prediction count, and GT count.

Required training command:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

The best checkpoint is selected by validation `mAP@0.5` and saved to:

```text
models/best.pth
```

Useful overrides:

```bash
python train.py ... --epochs 80 --batch_size 24 --image_size 448 --lr 0.00014
```

If P100 train time is above the 3 minute per-epoch target, switch only the backbone to Pico first:

```yaml
model:
  backbone: convnextv2_pico
```

## Predict

Required inference command:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

By default, `predict.py` reads `models/best.pth`. To use another checkpoint:

```bash
python predict.py --image_dir ./public/val/images --output val_predictions.json --checkpoint ./models/best.pth
```

The output is a JSON array in the required format. Images with no detections are still emitted with `"boxes": []`.

## Validate Predictions

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

## Implementation Notes

- `models/tiny_detector.py` contains the ConvNeXtV2 backbone, necks, and anchor-free head.
- `utils/loss.py` contains `AnchorFreeLoss`, including task-aligned assignment and Quality Focal Loss.
- `utils/inference.py` decodes anchor-free boxes and runs per-class NMS without `torchvision.ops.nms`.
- `train.py` logs train epoch time separately from validation time, so the 3 minute budget can be checked on Kaggle P100.
- Validation mAP is computed in-process and the best checkpoint stores the chosen metric/decode settings.

## Research Basis

The default choices are based on ConvNeXtV2 for small pretrained ConvNet features, RTMDet/PP-YOLOE-style efficient neck and task-aligned assignment, and VarifocalNet/GFL-style quality-aware class scores for AP ranking.
