from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from models.tiny_detector import TinyDetector
from utils.config import get_anchors, load_config
from utils.dataset import load_classes
from utils.inference import decode_predictions, flip_detections_horizontally, load_image_for_inference, merge_detections


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the from-scratch tiny YOLO detector.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="models/best.pth")
    parser.add_argument("--classes", default="public/classes.json")
    parser.add_argument("--conf_threshold", type=float)
    parser.add_argument("--nms_threshold", type=float)
    parser.add_argument("--nms_type", choices=["hard", "soft", "diou"])
    parser.add_argument("--merge_nms", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_detections", type=int)
    parser.add_argument("--pre_nms_topk", type=int)
    parser.add_argument("--class_pre_nms_topk", type=int)
    parser.add_argument("--decode_style", choices=["standard", "yolov7", "anchor_free"])
    parser.add_argument("--class_activation", choices=["softmax", "sigmoid"])
    parser.add_argument("--quality_score_power", type=float)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)
    inference = config["inference"]
    for name in ["conf_threshold", "nms_threshold", "nms_type", "merge_nms", "max_detections", "pre_nms_topk", "class_pre_nms_topk", "decode_style", "class_activation", "quality_score_power"]:
        if getattr(args, name) is None:
            setattr(args, name, inference[name])
    return args


def main() -> None:
    args = parse_args()
    cli_conf_threshold = args.conf_threshold
    cli_nms_threshold = args.nms_threshold
    cli_merge_nms = args.merge_nms
    cli_decode_style = args.decode_style
    cli_class_activation = args.class_activation
    cli_quality_score_power = args.quality_score_power
    args = apply_config(args)
    image_dir = Path(args.image_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train first or pass --checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = load_config(args.config)
    classes = checkpoint.get("classes") or load_classes(args.classes)
    anchors = checkpoint.get("anchors") or get_anchors(config)
    image_size = int(checkpoint.get("image_size", config["image_size"]))
    preserve_aspect = bool(checkpoint.get("preserve_aspect", config.get("preserve_aspect", True)))
    model_config = dict(checkpoint.get("model_config", config["model"]))
    model_config["pretrained"] = False
    channels_last = bool(config.get("channels_last", True)) and torch.cuda.is_available()
    if cli_conf_threshold is None:
        args.conf_threshold = checkpoint.get("best_conf_threshold", args.conf_threshold)
    if cli_nms_threshold is None:
        args.nms_threshold = checkpoint.get("best_nms_threshold", args.nms_threshold)
    if cli_merge_nms is None:
        args.merge_nms = checkpoint.get("merge_nms", args.merge_nms)
    if cli_decode_style is None:
        args.decode_style = checkpoint.get("decode_style", checkpoint.get("loss_weights", {}).get("decode_style", args.decode_style))
    if cli_class_activation is None:
        args.class_activation = checkpoint.get("class_activation", args.class_activation)
    if cli_quality_score_power is None:
        args.quality_score_power = checkpoint.get("quality_score_power", args.quality_score_power)
    tta_hflip = bool(config["inference"].get("tta_hflip", False))

    model = TinyDetector(num_classes=len(classes), num_anchors=[len(scale) for scale in anchors], **model_config).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    predictions = []
    batch_size = max(1, int(args.batch_size))
    progress = tqdm(total=len(image_paths), desc="predict", unit="img", dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad(), progress:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_images = []
            image_sizes = []
            for image_path in batch_paths:
                image_t, orig_w, orig_h = load_image_for_inference(image_path, image_size, preserve_aspect=preserve_aspect)
                batch_images.append(image_t)
                image_sizes.append((orig_w, orig_h))

            batch_tensor = torch.stack(batch_images, dim=0).to(
                device,
                memory_format=torch.channels_last if channels_last else torch.contiguous_format,
            )
            pred = model(batch_tensor)
            pred_flip = None
            if tta_hflip:
                pred_flip = model(torch.flip(batch_tensor, dims=[3]))
            for idx, (image_path, (orig_w, orig_h)) in enumerate(zip(batch_paths, image_sizes)):
                boxes = decode_predictions(
                    {"main": [scale[idx : idx + 1] for scale in pred["main"]]},
                    classes=classes,
                    anchors=anchors,
                    image_size=image_size,
                    orig_width=orig_w,
                    orig_height=orig_h,
                    conf_threshold=args.conf_threshold,
                    class_conf_thresholds=config["inference"].get("class_conf_thresholds", {}),
                    nms_threshold=args.nms_threshold,
                    nms_type=args.nms_type,
                    merge_nms=bool(args.merge_nms),
                    max_detections=args.max_detections,
                    pre_nms_topk=args.pre_nms_topk,
                    class_pre_nms_topk=args.class_pre_nms_topk,
                    preserve_aspect=preserve_aspect,
                    decode_style=args.decode_style,
                    class_activation=args.class_activation,
                    quality_score_power=float(args.quality_score_power),
                )
                if pred_flip is not None:
                    flip_boxes = decode_predictions(
                        {"main": [scale[idx : idx + 1] for scale in pred_flip["main"]]},
                        classes=classes,
                        anchors=anchors,
                        image_size=image_size,
                        orig_width=orig_w,
                        orig_height=orig_h,
                        conf_threshold=args.conf_threshold,
                        class_conf_thresholds=config["inference"].get("class_conf_thresholds", {}),
                        nms_threshold=args.nms_threshold,
                        nms_type=args.nms_type,
                        merge_nms=bool(args.merge_nms),
                        max_detections=args.max_detections,
                        pre_nms_topk=args.pre_nms_topk,
                        class_pre_nms_topk=args.class_pre_nms_topk,
                        preserve_aspect=preserve_aspect,
                        decode_style=args.decode_style,
                        class_activation=args.class_activation,
                        quality_score_power=float(args.quality_score_power),
                    )
                    boxes = merge_detections(
                        boxes + flip_detections_horizontally(flip_boxes, orig_w),
                        classes=classes,
                        nms_threshold=args.nms_threshold,
                        nms_type=args.nms_type,
                        merge_nms=bool(args.merge_nms),
                        max_detections=args.max_detections,
                    )
                predictions.append({"image_id": image_path.name, "boxes": boxes})
            progress.update(len(batch_paths))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path("") else None
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
