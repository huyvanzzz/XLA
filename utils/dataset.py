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
        augment_config: dict[str, float] | None = None,
        preserve_aspect: bool = True,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.image_size = image_size
        self.augment = augment
        self.augment_config = augment_config or {}
        self.preserve_aspect = preserve_aspect

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
            image, boxes_t, labels_t = self._augment(image, boxes_t, labels_t)

        if self.preserve_aspect:
            image, boxes_t = self._letterbox(image, boxes_t)
        else:
            aug_w, aug_h = image.size
            scale_x = self.image_size / aug_w
            scale_y = self.image_size / aug_h
            if boxes_t.numel() > 0:
                boxes_t[:, [0, 2]] *= scale_x
                boxes_t[:, [1, 3]] *= scale_y
                boxes_t[:, 0::2].clamp_(0, self.image_size)
                boxes_t[:, 1::2].clamp_(0, self.image_size)
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.augment:
            image = self._random_erasing(image)
        image_t = self._to_tensor(image)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": str(info["id"]),
            "orig_width": int(orig_w),
            "orig_height": int(orig_h),
        }
        return image_t, target

    def _augment(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        width, _ = image.size
        image, boxes = self._random_scale(image, boxes)
        image, boxes, labels = self._random_crop(image, boxes, labels)
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
        return image, boxes, labels

    def _random_scale(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        prob = float(self.augment_config.get("random_scale_prob", 0.0))
        if random.random() >= prob:
            return image, boxes

        scale = random.uniform(
            float(self.augment_config.get("min_scale", 0.75)),
            float(self.augment_config.get("max_scale", 1.25)),
        )
        width, height = image.size
        new_w = max(16, int(width * scale))
        new_h = max(16, int(height * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= new_w / width
            boxes[:, [1, 3]] *= new_h / height
        return image, boxes

    def _random_crop(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        prob = float(self.augment_config.get("random_crop_prob", 0.0))
        if boxes.numel() == 0 or random.random() >= prob:
            return image, boxes, labels

        width, height = image.size
        min_scale = float(self.augment_config.get("min_crop_scale", 0.75))
        crop_w = random.randint(max(16, int(width * min_scale)), width)
        crop_h = random.randint(max(16, int(height * min_scale)), height)
        left = random.randint(0, max(width - crop_w, 0))
        top = random.randint(0, max(height - crop_h, 0))
        right = left + crop_w
        bottom = top + crop_h

        cropped = boxes.clone()
        cropped[:, [0, 2]] -= left
        cropped[:, [1, 3]] -= top
        cropped[:, 0::2].clamp_(0, crop_w)
        cropped[:, 1::2].clamp_(0, crop_h)
        wh = cropped[:, 2:] - cropped[:, :2]
        keep = (wh[:, 0] >= 4) & (wh[:, 1] >= 4)
        if not keep.any():
            return image, boxes, labels

        image = image.crop((left, top, right, bottom))
        return image, cropped[keep], labels[keep]

    def _letterbox(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        width, height = image.size
        scale = min(self.image_size / width, self.image_size / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        pad_x = (self.image_size - new_w) // 2
        pad_y = (self.image_size - new_h) // 2

        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        canvas.paste(resized, (pad_x, pad_y))
        if boxes.numel() > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
            boxes[:, 0::2].clamp_(0, self.image_size)
            boxes[:, 1::2].clamp_(0, self.image_size)
        return canvas, boxes

    def _random_erasing(self, image: Image.Image) -> Image.Image:
        prob = float(self.augment_config.get("random_erasing_prob", 0.0))
        if random.random() >= prob:
            return image
        width, height = image.size
        area = width * height
        min_area = float(self.augment_config.get("random_erasing_min_area", 0.02))
        max_area = float(self.augment_config.get("random_erasing_max_area", 0.12))
        erase_area = random.uniform(min_area, max_area) * area
        aspect = random.uniform(0.3, 3.3)
        erase_w = int(round((erase_area * aspect) ** 0.5))
        erase_h = int(round((erase_area / aspect) ** 0.5))
        if erase_w <= 0 or erase_h <= 0 or erase_w >= width or erase_h >= height:
            return image
        left = random.randint(0, width - erase_w)
        top = random.randint(0, height - erase_h)
        image = image.copy()
        image.paste((114, 114, 114), (left, top, left + erase_w, top + erase_h))
        return image

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        data = data.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (data - mean) / std


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
