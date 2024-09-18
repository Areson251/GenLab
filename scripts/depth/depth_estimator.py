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


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
MODELS = {
    "Depth_Anything": "LiheYoung/depth-anything-small-hf",
    "Depth_Anything_v2": "depth-anything/Depth-Anything-V2-Small-hf",
}

class DepthEstimator():
    def __init__(self, images_dir, output_path, model):
        self.images_dir = images_dir
        self.output_path = output_path
        self.model = model
        
    def depth_estimation(self):
        images_paths = [join(self.images_dir, f) for f in os.listdir(self.images_dir) if join(self.images_dir, f).endswith(IMAGE_EXTENSIONS)]
        assert len(images_paths) != 0

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        pipe = pipeline(task="depth-estimation", model=self.model)
        avg_time = 0
        for img_path in tqdm(images_paths):
            filename = img_path.split("/")[-1].split(".")[0]

            image = Image.open(img_path)
            start_time = time.time()
            with torch.no_grad():
                depth = pipe(image)["depth"]
            avg_time += time.time() - start_time  

            image = np.asarray(image)
            depth = np.asarray(depth)

            h, w = image.shape[:2]    
            # depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    
            depth = depth.astype(np.uint8)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            cv2.imwrite(join(self.output_path, filename + "_depth.png"), depth)

        print("AVERAGE TIME IS: ", avg_time/len(images_paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=MODELS.keys())
    args = parser.parse_args()

    estimator = DepthEstimator(args.images_dir, args.output_path, MODELS[args.model])
    estimator.depth_estimation()