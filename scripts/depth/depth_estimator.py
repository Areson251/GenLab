from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from os.path import isfile, join
from tqdm import tqdm
import time
import argparse

from scripts.depth.metrics import Metrics


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
DEPTH_MODELS = {
    "Depth_Anything": "LiheYoung/depth-anything-small-hf",
    "Depth_Anything_v2": "depth-anything/Depth-Anything-V2-Small-hf",
    "Intel": "Intel/dpt-large",
}

class DepthEstimator():
    def __init__(self, args):
        self.images_dir = args.images_dir
        self.output_path = args.output_path
        self.model_name = args.model
        self.device = args.device

        self.pipe = pipeline(task="depth-estimation", model=self.model_name, device=self.device)

    def calculate_depth(self, image):
        with torch.no_grad():
            depth = self.pipe(image)["depth"]
        return depth
        
    def depth_estimation(self):
        if self.count_metrics:
                metrics = Metrics()

        images_paths = [join(self.images_dir, f) for f in os.listdir(self.images_dir) if join(self.images_dir, f).endswith(IMAGE_EXTENSIONS)]
        assert len(images_paths) != 0

        if self.output_path:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

        avg_time = 0
        for img_path in tqdm(images_paths):
            filename = img_path.split("/")[-1].split(".")[0]

            image = Image.open(img_path)
            start_time = time.time()
            depth = self.calculate_depth(image)
            avg_time += time.time() - start_time  

            image = np.asarray(image)
            depth *= 255.0
            depth = depth.astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            if self.output_path:
                cv2.imwrite(join(self.output_path, filename + "_depth.png"), depth_color)

            if self.count_metrics:
                gt_file_name = "depth"+filename.split("left")[-1]
                gt_path = join('/'.join(self.images_dir.split("/")[:-1]),
                               "depth", 
                               gt_file_name+".png")
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

                metrics.RMSE_transit(depth, gt)
                metrics.AbsRel_transit(depth, gt)

        print(f"MODEL NAME: {self.model_name}")
        print("AVERAGE TIME IS: ", avg_time/len(images_paths))

        if self.count_metrics:
            print("RMSE: ", metrics.RMSE_total())
            print("AbsRel: ", metrics.AbsRel_total())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=DEPTH_MODELS.keys())
    parser.add_argument("--calc_metrics", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    estimator = DepthEstimator(args)
    estimator.depth_estimation()