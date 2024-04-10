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


images_dir = "images/output_imgs/exp6"
images_paths = [join(images_dir, f) for f in os.listdir(images_dir) if join(images_dir, f).endswith(".png")]

# images_paths = [
#     "images/output_imgs/exp2/KAND_log.png",
#     "images/output_imgs/exp2/SDXL_log.png",
#     "images/output_imgs/exp3/SD2_log.png",
#     "images/output_imgs/exp4/SD1-5_log.png",
# ]

output_path = "images/output_imgs/exp6_depth"
if not os.path.exists(output_path):
    os.makedirs(output_path)

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
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
    cv2.imwrite(join(output_path, filename + "_depth.png"), depth)

print("AVERAGE TIME IS: ", avg_time/len(images_paths))