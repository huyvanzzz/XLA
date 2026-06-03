---
name: xla-object-detection
description: Continue, review, debug, or improve the D:\XLA from-scratch object detection assignment repo. Use when the user asks about the XLA detector, mAP@0.5, train.py, predict.py, configs/default.yaml, YOLOv7-inspired changes, ResNet50/ConvNeXt backbone experiments, validation speed, NMS, loss, anchors, Kaggle training, EXPERIMENTS.md, SESSION_SUMMARY.md, or any continuation of this object-detection project.
---

# XLA Object Detection

## Quick Start

When this skill triggers inside the `D:\XLA` repo:

1. Read `SESSION_SUMMARY.md`, `EXPERIMENTS.md`, and `configs/default.yaml` first.
2. Inspect the relevant code before editing:
   - model/neck/head: `models/tiny_detector.py`
   - loss/assignment: `utils/loss.py`
   - inference/NMS/decode: `utils/inference.py`, `utils/box_ops.py`, `predict.py`
   - train loop/checkpoint/metric: `train.py`
3. Keep the assignment constraints in mind:
   - from-scratch detector pipeline;
   - pretrained feature extractor is allowed if teacher permits;
   - no complete detector frameworks/models;
   - no `torchvision.ops.nms`;
   - mAP@0.5 is the main metric.
4. Preserve user preferences unless explicitly changed:
   - direct resize `512x512`, not letterbox;
   - optimize for mAP but avoid train-time explosions;
   - do not edit notebooks when config/code can solve it;
   - record only actually trained/evaluated versions in `EXPERIMENTS.md`.

## Current Active Direction

The active direction is **v16**:

- ResNet50 pretrained backbone.
- YOLOv7-style PAN neck implemented locally.
- YOLOv7-style bbox decode and light target-offset assignment.
- BCE/sigmoid classification path is active.
- Separate quality prediction head is active.
- Merge-NMS enabled for validation/inference box fusion.
- Class-prior bias initialization enabled from train annotation distribution.
- Legacy anchor assignment, not OTA/task-aligned.
- DIoU-NMS implemented locally.

For exact current state, read [project-state.md](references/project-state.md).

## Workflows

### Continue Optimization

Use this order:

1. Ask whether the user has new mAP/time results if they did not provide them.
2. If results exist, update `EXPERIMENTS.md` before changing architecture.
3. Compare the requested change against known bad paths in [engineering-guide.md](references/engineering-guide.md).
4. Prefer one meaningful change per version so the result is attributable.
5. Run at least:
   - `python -m py_compile train.py predict.py utils\config.py utils\dataset.py utils\loss.py utils\inference.py utils\box_ops.py models\tiny_detector.py`
   - `python -m unittest test_predict_cli.py`
6. If touching model/loss/decode, run a small forward/loss sanity check.

### Handle "Use YOLOv7" Requests

Do not copy YOLOv7 as a complete model. Instead:

- read the upstream code if needed;
- extract architectural logic only;
- implement small local blocks compatible with this repo;
- keep `train.py`, `predict.py`, loss, NMS, JSON output self-contained.

Current YOLOv7-inspired parts are local reimplementations:

- `SPPCSPCBlock`
- `YoloV7ELANFusion`
- `YoloV7Downsample`
- `YoloV7PANNeck`
- YOLOv7-style bbox decode in loss/inference.

### Handle "Train Is Too Slow"

First identify whether slowness comes from:

- model forward/backward: backbone/neck/head;
- assignment/loss: task-aligned or many positives;
- validation mAP/predict loop;
- NMS candidate count;
- Kaggle/runtime constraints.

Avoid reintroducing known slow paths unless the user accepts the cost:

- task-aligned assignment;
- ConvNeXt-Small/Base;
- heavy LayerNorm/depthwise neck/head;
- TTA during validation;
- threshold/NMS grid search each epoch.

## References

- [project-state.md](references/project-state.md): current config, version history, known results.
- [engineering-guide.md](references/engineering-guide.md): constraints, safe edit patterns, validation commands, known good/bad directions.
