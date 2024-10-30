import diffusers
import argparse
import os
from PIL import Image
import tqdm
import numpy as np

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
MODELS = {
    "depth": "prs-eth/marigold-depth-lcm-v1-0",
    "normals": "prs-eth/marigold-normals-lcm-v0-1",
}

class Estimator:
    model_type = None
    pipe = None
    
    def load_pipe():
        if Estimator.model_type == "depth":
            Estimator.pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                MODELS[Estimator.model_type]
                )
        if Estimator.model_type == "normals":
            Estimator.pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
                MODELS[Estimator.model_type]
                )
        print(f"Model {MODELS[Estimator.model_type]} loaded")

    @classmethod
    def estimate(cls, image, dimension="depth"):
        if not cls.pipe:
            cls.model_type = dimension
            cls.load_pipe()
            
        output = cls.pipe(image)    
        visualisation_name = f"visualize_{cls.model_type}"
        if dimension == "depth":
            vis = getattr(cls.pipe.image_processor, visualisation_name)(output.prediction, color_map="binary")[0]
        else:  # assuming 'normals' is the other dimension
            vis = getattr(cls.pipe.image_processor, visualisation_name)(output.prediction)[0]
        return vis
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", type=str, required=True)
    parser.add_argument("--dimension", type=str, choices=["depth", "normals"], 
                        default="depth", required=True)
    args = parser.parse_args()

    output_folder = os.path.join(args.dimension, args.images_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images_paths = [f for f in os.listdir(args.images_folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
    for image_name in tqdm.tqdm(images_paths):
        image = Image.open(os.path.join(args.images_folder, image_name))
        ouput = Estimator.estimate(image, args.dimension)
        image_name = image_name.split(".")[0] + ".png"
        ouput.save(os.path.join(output_folder, image_name))
        

