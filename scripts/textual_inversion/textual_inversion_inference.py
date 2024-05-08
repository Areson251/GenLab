from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionInpaintPipeline
import torch
import safetensors
import os

MODEL_CHECKPOINT = "model_output/ti_pothole-inpainting/checkpoint-6000"
IMAGE_PATH = "images/test_pothole-ti/"
IMAGE_NAME = "pothole-ti"
PROMPTS = [
    "<pothole>"
    ]

pipeline = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16).to("cuda")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

# for idx, prompt in enumerate(PROMPTS):
#     image = pipeline(prompt, num_inference_steps=50).images[0]
#     image.save(IMAGE_PATH+IMAGE_NAME+f"_SD2_{idx}.png")

print(f"load textual inversion from {MODEL_CHECKPOINT}")
pipeline.load_textual_inversion(MODEL_CHECKPOINT)

for idx, prompt in enumerate(PROMPTS):
    image = pipeline(prompt, num_inference_steps=50).images[0]
    image.save(IMAGE_PATH+IMAGE_NAME+f"_SD2_tuned_{idx}.png")

