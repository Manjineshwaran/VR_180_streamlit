import cv2
import torch
import numpy as np
import os
from enum import Enum

class ModelType(Enum):
    DPT_LARGE = "DPT_Large"
    DPT_Hybrid = "DPT_Hybrid"
    MIDAS_SMALL = "MiDaS_small"

class Midas:
    def __init__(self, modelType:ModelType=ModelType.MIDAS_SMALL, device=None):
        self.modelType = modelType
        self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.modelType.value in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(inp)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depthMap = prediction.cpu().numpy()
        depthMap = cv2.normalize(depthMap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depthMap = cv2.cvtColor(depthMap, cv2.COLOR_GRAY2BGR)
        return depthMap

    def predict_batch(self, frame_paths, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for p in frame_paths:
            frame = cv2.imread(p)
            d = self.predict_frame(frame)
            outp = os.path.join(out_dir, os.path.basename(p))
            cv2.imwrite(outp, d)
            results.append(outp)
        return results
