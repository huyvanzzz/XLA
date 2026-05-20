from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.tiny_detector import TinyDetector
from utils.config import get_anchors, load_config
from utils.dataset import load_classes
from utils.inference import decode_predictions, load_image_for_inference


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
    parser.add_argument("--max_detections", type=int)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)
    inference = config["inference"]
    for name in ["conf_threshold", "nms_threshold", "max_detections"]:
        if getattr(args, name) is None:
            setattr(args, name, inference[name])
    return args


def main() -> None:
    args = parse_args()
    args = apply_config(args)
    image_dir = Path(args.image_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train first or pass --checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint.get("classes") or load_classes(args.classes)
    anchors = checkpoint.get("anchors") or get_anchors(load_config(args.config))
    image_size = int(checkpoint.get("image_size", load_config(args.config)["image_size"]))

    model = TinyDetector(num_classes=len(classes), num_anchors=len(anchors)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    predictions = []
    with torch.no_grad():
        for image_path in image_paths:
            image_t, orig_w, orig_h = load_image_for_inference(image_path, image_size)
            pred = model(image_t.unsqueeze(0).to(device))
            boxes = decode_predictions(
                pred,
                classes=classes,
                anchors=anchors,
                image_size=image_size,
                orig_width=orig_w,
                orig_height=orig_h,
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold,
                max_detections=args.max_detections,
            )
            predictions.append({"image_id": image_path.name, "boxes": boxes})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path("") else None
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
