from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import predict
import torch

from utils.box_ops import bbox_siou
from utils.inference import weighted_box_fusion


class PredictCliTest(unittest.TestCase):
    def test_predict_accepts_batch_size_argument(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "predict.py",
                "--image_dir",
                "images",
                "--output",
                "predictions.json",
                "--batch_size",
                "8",
                "--distribution_quality_power",
                "0.25",
            ],
        ):
            args = predict.parse_args()

        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.distribution_quality_power, 0.25)


class DetectionMathTest(unittest.TestCase):
    def test_siou_is_finite_for_identical_boxes(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
        score = bbox_siou(boxes, boxes.detach())
        (1.0 - score).sum().backward()

        self.assertAlmostEqual(float(score.item()), 1.0, places=6)
        self.assertTrue(torch.isfinite(boxes.grad).all())

    def test_weighted_box_fusion_combines_augmented_views(self) -> None:
        fused = weighted_box_fusion(
            [
                [{"class": "chair", "confidence": 0.8, "bbox": [0.0, 0.0, 10.0, 10.0]}],
                [{"class": "chair", "confidence": 0.9, "bbox": [1.0, 0.0, 11.0, 10.0]}],
            ],
            ["chair"],
            iou_threshold=0.5,
        )

        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(float(fused[0]["bbox"][0]), 0.53, places=2)
        self.assertEqual(fused[0]["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
