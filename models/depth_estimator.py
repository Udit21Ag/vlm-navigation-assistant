"""Depth Estimation Module

Estimates monocular depth (distance) for every pixel using MiDaS DPT-LeViT.
Transforms RGB images into normalized depth maps [0, 1].

Features:
- Single-image depth estimation (no stereo required)
- Relative depth information
- Normalized output [0, 1] for easy threshold-based classification
- Lightweight architecture (~51M parameters)
"""

import torch
import cv2
import numpy as np
import sys

sys.path.append("MiDaS")

from midas.model_loader import load_model


class DepthEstimator:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.set_grad_enabled(False)

        model_type = "dpt_levit_224"
        model_path = "MiDaS/weights/dpt_levit_224.pt"

        self.model, self.transform, _, _ = load_model(
            device=self.device,
            model_type=model_type,
            model_path=model_path,
            optimize=True   # FIXED
        )

        self.model.eval()
        self.prev_depth = None
        self._running_low = None
        self._running_high = None

    def estimate_depth(self, image):

        orig_h, orig_w = image.shape[:2]

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (384, 384))   # faster inference

        input_batch = self.transform({"image": img})["image"]

        if isinstance(input_batch, np.ndarray):
            input_batch = torch.from_numpy(input_batch)

        input_batch = input_batch.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(orig_h, orig_w),   # restore original resolution
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        low = float(np.percentile(depth, 5))
        high = float(np.percentile(depth, 95))

        if self._running_low is None:
            self._running_low = low
            self._running_high = high
        else:
            self._running_low = 0.9 * self._running_low + 0.1 * low
            self._running_high = 0.9 * self._running_high + 0.1 * high

        if self._running_high - self._running_low > 1e-6:
            depth = (depth - self._running_low) / (self._running_high - self._running_low)

        depth = np.clip(depth, 0.0, 1.0)

        # Temporal smoothing (optional but useful)
        if self.prev_depth is not None:
            depth = 0.7 * depth + 0.3 * self.prev_depth

        self.prev_depth = depth

        return depth

    def estimate_depth_roi(self, image, bbox):
        depth_map = self.estimate_depth(image)
        h, w = depth_map.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return float(depth_map[min(max((y1 + y2) // 2, 0), h - 1), min(max((x1 + x2) // 2, 0), w - 1)])

        return float(np.median(region))