from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


def load_classes(path: str | Path = "public/classes.json") -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class DetectionDataset(Dataset):
    def __init__(
        self,
        annotation_path: str | Path,
        image_dir: str | Path,
        classes: list[str],
        image_size: int = 416,
        augment: bool = False,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.image_size = image_size
        self.augment = augment

        with self.annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ann in data["annotations"]:
            grouped[ann["image_id"]].append(ann)

        self.images = []
        for item in data["images"]:
            image_id = item["id"]
            self.images.append(
                {
                    "id": image_id,
                    "file_name": Path(item["file_name"]).name,
                    "width": int(item["width"]),
                    "height": int(item["height"]),
                    "annotations": grouped.get(image_id, []),
                }
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor | str | int]]:
        info = self.images[idx]
        image_path = self.image_dir / str(info["file_name"])
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        boxes = []
        labels = []
        for ann in info["annotations"]:
            x1, y1, x2, y2 = [float(v) for v in ann["bbox"]]
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_to_idx[ann["class"]])

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        if self.augment:
            image, boxes_t = self._augment(image, boxes_t)

        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        if boxes_t.numel() > 0:
            boxes_t[:, [0, 2]] *= scale_x
            boxes_t[:, [1, 3]] *= scale_y
            boxes_t[:, 0::2].clamp_(0, self.image_size)
            boxes_t[:, 1::2].clamp_(0, self.image_size)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_t = self._to_tensor(image)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": str(info["id"]),
            "orig_width": int(orig_w),
            "orig_height": int(orig_h),
        }
        return image_t, target

    def _augment(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        width, _ = image.size
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes.numel() > 0:
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1

        if random.random() < 0.3:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.3:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.3:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
        return image, boxes

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        data = data.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (data - mean) / std


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
