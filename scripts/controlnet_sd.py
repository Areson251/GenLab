import os
import argparse
from PIL import Image
import tqdm
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler


from depth.marigold import Estimator


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

class AugmentationPipe():
    def __init__(self):
        self.generator = torch.manual_seed(0)

        self.controlnet = ControlNetModel.from_pretrained(
        # "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", 
        controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16,
        ).to("cuda")

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
    
    # CONTROL_CONDITIONING_SCALE
    def __call__(self, prompt, image, num_inference_steps=20):
        return self.pipe(prompt, image, num_inference_steps=20).images[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", type=str, required=True)
    parser.add_argument("--dimension", type=str, choices=["depth", "normals"], 
                            default="depth", required=True)
    args = parser.parse_args()

    output_folder = os.path.join(args.dimension+"_exps", args.images_folder)
    dimension_path = os.path.join(output_folder, args.dimension)
    augmentation_path = os.path.join(output_folder, "augmented")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(dimension_path):
        os.makedirs(dimension_path)
    if not os.path.exists(augmentation_path):
        os.makedirs(augmentation_path)

    augmentation = AugmentationPipe()

    images_paths = [f for f in os.listdir(args.images_folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
    for image_name in tqdm.tqdm(images_paths):
        image = Image.open(os.path.join(args.images_folder, image_name))
        estimator_ouput = Estimator.estimate(image, args.dimension)

        image_name = image_name.split(".")[0] + ".png"
        estimator_ouput.save(os.path.join(dimension_path, image_name))

        prompt = f"a photo of a snowy street scene"
        # obj = image_name.split(".")[0]
        # prompt = f"a photo of a {obj}"
        ouput = augmentation(prompt, estimator_ouput)
        ouput.save(os.path.join(augmentation_path, image_name))