import diffusers
import argparse
import os
from PIL import Image
import tqdm
import numpy as np
import cv2
import time
from os.path import join

from scripts.depth.metrics import Metrics

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
MODELS = {
    "depth": "prs-eth/marigold-depth-lcm-v1-0",
    "normals": "prs-eth/marigold-normals-lcm-v0-1",
}

class Estimator():
    def __init__(self, model_type, device="cuda") -> None:
        self.model_type = model_type
        self.device = device

        self.load_pipe()
    
    def load_pipe(self):
        if self.model_type == "depth":
            self.pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                MODELS[self.model_type],
                ).to(self.device)
        if self.model_type == "normals":
            self.pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
                MODELS[self.model_type],
                ).to(self.device)
        print(f"Model {MODELS[self.model_type]} loaded")

    def estimate(self, image, dimension="depth"):            
        output = self.pipe(image)    
        visualisation_name = f"visualize_{self.model_type}"
        if dimension == "depth":
            vis = getattr(self.pipe.image_processor, visualisation_name)(output.prediction, color_map="binary")[0]
        else:  # assuming 'normals' is the other dimension
            vis = getattr(self.pipe.image_processor, visualisation_name)(output.prediction)[0]
        return vis
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", type=str, required=True)
    parser.add_argument("--dimension", type=str, choices=["depth", "normals"], 
                        default="depth", required=True)
    parser.add_argument("--calc_metrics", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=False)
    args = parser.parse_args()

    if args.output:
        output_folder = os.path.join(args.dimension, args.images_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    if args.calc_metrics:
        metrics = Metrics()

    estimator = Estimator(args.dimension, args.device)

    avg_time = 0
    images_paths = [f for f in os.listdir(args.images_folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
    for image_name in tqdm.tqdm(images_paths):
        image = Image.open(os.path.join(args.images_folder, image_name))

        start_time = time.time()
        ouput = estimator.estimate(image, args.dimension)
        avg_time += time.time() - start_time  

        image_name = image_name.split(".")[0] + ".png"

        if args.output:
            ouput.save(os.path.join(output_folder, image_name))

        ouput.save("test.png")

        if args.calc_metrics:
            gt_file_name = "depth"+image_name.split("left")[-1]
            gt_path = join('/'.join(args.images_folder.split("/")[:-1]),
                            "depth", 
                            gt_file_name)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            ouput = np.array(ouput.convert("L"))
            metrics.RMSE_transit(ouput, gt)
            metrics.AbsRel_transit(ouput, gt)

    print(f"MODEL NAME: prs-eth/marigold-depth-lcm-v1-0")
    print("AVERAGE TIME IS: ", avg_time/len(images_paths))

    if args.calc_metrics:
        print("RMSE: ", metrics.RMSE_total())
        print("AbsRel: ", metrics.AbsRel_total())
        

