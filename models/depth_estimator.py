import torch
import cv2
import numpy as np
import sys

sys.path.append("MiDaS")

from midas.model_loader import load_model


class DepthEstimator:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_type = "dpt_levit_224"
        model_path = "MiDaS/weights/dpt_levit_224.pt"

        self.model, self.transform, net_w, net_h = load_model(
            device=self.device,
            model_type=model_type,
            model_path=model_path,
            optimize=False
        )

        self.model.eval()

    def estimate_depth(self, image):

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_batch = self.transform({"image": img})["image"]

        if isinstance(input_batch, np.ndarray):
            input_batch = torch.from_numpy(input_batch)

        input_batch = input_batch.unsqueeze(0)
        input_batch = input_batch.contiguous().to(self.device)

        with torch.no_grad():

            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        return depth