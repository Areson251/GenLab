from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import safetensors
import os

MODEL_CHECKPOINT = "model_output/exp3_cat-avocado/checkpoint-6000"
IMAGE_PATH = "images/test_avocado2/"
IMAGE_NAME = "cat-avocado"
PROMPTS = [
    "A <cat-avocado>",
    "A a road with <cat-avocado> leading off into the distance and surrounded by a clearing",
    "<cat-avocado> on paper",
    "<cat-avocado> on the crossroad",
    "<cat-avocado> in pink room",
    ]

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16).to("cuda")
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

