import os
import argparse
from PIL import Image
import tqdm
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from depth.marigold import Estimator


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
MODELS = {
    "depth": "lllyasviel/sd-controlnet-depth",
    "normals": "fusing/stable-diffusion-v1-5-controlnet-normal",
}

class AugmentationPipe():
    def __init__(self, dimension="depth"):
        self.generator = torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.controlnet = ControlNetModel.from_pretrained(
        MODELS[dimension], 
        torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", 
        controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16,
        ).to(self.device)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
    
    # CONTROL_CONDITIONING_SCALE
    def __call__(self, prompt, image, num_inference_steps=20,
                 guidance_scale=7.5,
                 controlnet_conditioning_scale=1.0,):

        return self.pipe(prompt, image, 
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale,
                         controlnet_conditioning_scale=controlnet_conditioning_scale,).images[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", type=str, required=True)
    parser.add_argument("--dimension", type=str, choices=["depth", "normals"], 
                            default="depth", required=True)
    parser.add_argument("--prompts_path", type=str, default="prompts/test.txt")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    
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
    
    with open(args.prompts_path, "r") as f:
        prompts = f.readlines()
    images_paths = sorted([f for f in os.listdir(args.images_folder) if f.lower().endswith(IMAGE_EXTENSIONS)])
    
    for idx, image_name in enumerate(tqdm.tqdm(images_paths)):
        image = Image.open(os.path.join(args.images_folder, image_name))
        estimator_ouput = Estimator.estimate(image, args.dimension)

        image_name = image_name.split(".")[0] + ".png"
        estimator_ouput.save(os.path.join(dimension_path, image_name))

        prompt = prompts[idx].split("\n")[0]
        ouput = augmentation(prompt, estimator_ouput, 
                             args.num_steps,
                            args.guidance_scale,
                            args.controlnet_conditioning_scale,)
        ouput.save(os.path.join(augmentation_path, image_name))