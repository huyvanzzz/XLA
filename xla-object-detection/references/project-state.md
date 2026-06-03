# Project State

## Assignment

- Repo: `D:\XLA`.
- Task: from-scratch object detection for 5 classes: `person`, `car`, `dog`, `cat`, `chair`.
- Main metric: `mAP@0.5` using `public/tools/evaluate_predictions.py`.
- Submission must provide `train.py`, `predict.py`, `models/`, `utils/`, `README.md`, `requirements.txt`.
- `predict.py` must output JSON with image-level `boxes`.

## Hard Constraints

- Do not use complete object detectors such as YOLOv5/v8, Detectron2, MMDetection, torchvision Faster R-CNN/SSD.
- Do not use `torchvision.ops.nms`.
- It is acceptable to use pretrained feature extractors if allowed by instructor.
- Keep NMS, loss, assignment, train loop, predict pipeline self-contained in this repo.

## User Preferences

- Answer in Vietnamese.
- Optimize for mAP, but do not make train much slower without warning.
- Prefer config/code changes over notebook edits.
- Direct resize to `512x512`; user rejected letterbox.
- Do not enable multi-threshold grid tuning during train.
- Use one validation/predict threshold by default, currently `conf_threshold: 0.08`.
- Use mAP@0.5 for `best.pth`; `val_loss` is disabled.
- mAP validation and best checkpoint start at epoch 30.
- Only log versions in `EXPERIMENTS.md` when they are intended to be trained/evaluated.

## Active v16 Summary

Config file: `configs/default.yaml`.

Core:

- `image_size: 512`
- `preserve_aspect: false`
- `batch_size: 24`
- `val_batch_size: 64`
- `amp: true`
- `channels_last: true`
- `validation_metric.start_epoch: 30`
- `validation_metric.tune: false`

Model:

- `backbone: resnet50`
- `pretrained: true`
- `neck_type: yolov7_pan`
- `head_type: standard`
- `neck_channels: 192`
- `head_channels: 192`
- `freeze_backbone_epochs: 2`
- `backbone_trainable: layer4`
- `aux_head: true`
- `aux_head_close_epoch: 30`
- `decoupled_head: false`
- `quality_head: true`

Loss/assignment:

- `assignment_strategy: legacy`
- `positive_anchor_topk: 3`
- `decode_style: yolov7`
- `target_offsets: true`
- `classification_loss: bce`
- `classification_quality_mix: 0.0`
- `quality_weight: 0.35`
- `target_offset_bias: 0.5`
- `scale_obj_balance: [4.0, 1.0, 0.4]`
- `objectness_iou_mix: 1.0`
- `box_weight: 0.0`
- `iou_weight: 6.0`
- `noobj_hard_negative_ratio: 0.0`

Inference:

- `conf_threshold: 0.08`
- `nms_threshold: 0.5`
- `nms_type: diou`
- `merge_nms: true`
- `decode_style: yolov7`
- `class_prior_bias.enabled: true`
- `class_activation: sigmoid`
- `quality_score_power: 0.5`
- `objectness_bias.enabled: false`
- `ema.tau: 0`
- `anchors.evolve_generations: 0`
- `pre_nms_topk: 300`
- `class_pre_nms_topk: 100`
- `tta_hflip: false`

## Version History

- **v1**: baseline YOLO-style ResNet50 pretrained. Public val mAP `0.604047`; precision `0.060136`; recall `0.87333`; train about `5'30/epoch`.
- **v2/v3**: direct resize 512, faster fine-tuning. User reported v3 around `0.615` mAP, `4'30/epoch`, best near epoch 60.
- **v4**: mosaic + hard-negative + TTA. User reported very bad; disabled/reset.
- **v5**: task-aligned assignment + decoupled head. Too slow/unstable; disabled/reset.
- **v6**: v3 reset lightweight + aux head close epoch 30 + DIoU-NMS.
- **v7**: ConvNeXt-Small pretrained backbone. Stronger but heavier.
- **v8**: ConvNeXt-specific PAN/head. More aligned but slower due LayerNorm/depthwise/permute.
- **v9**: ConvNeXt PAN lite. Still heavier than ResNet path.
- **v10**: reset to ResNet50, add YOLOv7-style PAN neck.
- **v11**: add YOLOv7-style bbox decode, target offsets, scale objectness balance, CIoU-focused bbox loss.
- **v12**: ConvNeXt-Small pretrained, fully frozen backbone, fast FPN/PAN adapter and standard head at 160 channels.
- **v13**: reset to v11 ResNet50 path, add merge-NMS box fusion and class-prior bias init without increasing train/backward time.
- **v14**: add YOLO-style BCE/sigmoid classification, late clean fine-tune, and nonzero final cosine LR on top of v13.
- **v15**: add quality-aware class target, scale-aware objectness bias, EMA ramp, and lightweight auto-anchor evolution on top of v14.
- **v16**: reset risky v15 pieces and add a separate bbox-quality prediction head for score ranking.

## Important Files

- `models/tiny_detector.py`: ResNet50, ConvNeXt, necks, heads.
- `utils/loss.py`: anchor assignment, YOLO loss, YOLOv7 decode style.
- `utils/inference.py`: decode predictions, postprocessing.
- `utils/box_ops.py`: IoU, CIoU, hard NMS, soft NMS, DIoU-NMS.
- `train.py`: config loading, auto anchors, class weights, balanced sampler, EMA, mAP validation, checkpointing.
- `predict.py`: final inference CLI.
- `EXPERIMENTS.md`: version log.
- `SESSION_SUMMARY.md`: current project summary.
