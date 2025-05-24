import os 
import sys
import random
import pandas as pd

from tqdm import tqdm

import ultralytics
from ultralytics import YOLO


SEED = 42
random.seed(SEED)

tqdm.pandas()

print(sys.executable)
ultralytics.checks()


DATA_PATH = '/home/docker_diffdepth/diff_depth_new/datasets/augmented/YCOR_taomr_loss-umasked-1500'
# DATA_PATH = '/home/docker_diffdepth/diff_depth_new/datasets/original/TAOMR'


model = YOLO("yolov8m.pt")
model.info()

results = model.train(
    # data=os.path.join(DATA_PATH, "data_ycor_leaked.yaml"), 
    # data=os.path.join(DATA_PATH, "data_ycor.yaml"), 
    data=os.path.join(DATA_PATH, "data.yaml"), 
    # data=os.path.join(DATA_PATH, "data_ycor_2objs_custom.yaml"), 
    epochs=100, 
    imgsz=640,
    device="cuda:0",  
    # device="cuda:1",  
    verbose=True,
)

# model_path = "runs/detect/taomr_clear/"
# # model_path = "runs/detect/taomr_ycor_4objs/"
# model = YOLO(model_path+"weights/best.pt")

# csv_path = model_path+"results.csv"  
# df = pd.read_csv(csv_path)
# best_epoch = df["metrics/mAP50(B)"].idxmax()

# print(f"Best epoch: {best_epoch}")


# metrics = model.val(
#     data=os.path.join(DATA_PATH, "data.yaml"), 
#     # data=os.path.join(DATA_PATH, "data_ycor_4objs.yaml"), 
#     split="test",    
#     device="cuda",  
#     save_json=True
#     )

# python yolo_train.py


# print(f"mAP@50:      {metrics.box.map50:.4f}")  
# print(f"mAP@75:      {metrics.box.map75:.4f}")  
# print(f"mAP@50-95:   {metrics.box.map:.4f}")   