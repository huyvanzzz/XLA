from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import predict


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
            ],
        ):
            args = predict.parse_args()

        self.assertEqual(args.batch_size, 8)


if __name__ == "__main__":
    unittest.main()
