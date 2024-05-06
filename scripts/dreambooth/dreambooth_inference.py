from diffusers import DiffusionPipeline, UNet2DConditionModel, StableDiffusionInpaintPipeline, DDIMScheduler
from transformers import CLIPTextModel
import torch

PROMPTS = [
    "A ghe cat-avocado",
    "A a road with ghe cat-avocado leading off into the distance and surrounded by a clearing",
    "ghe cat-avocado on paper",
    "ghe cat-avocado on the crossroads road",
    ]

unet = UNet2DConditionModel.from_pretrained("model_output/exp2_cat-avocado/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("model_output/exp2_cat-avocado/text_encoder")

# pipeline = DiffusionPipeline.from_pretrained(
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "model_output/exp2_cat-avocado/checkpoint-6000", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
    # "stabilityai/stable-diffusion-2-inpainting", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
).to("cuda")
pipeline.scheduler = DDIMScheduler.from_config("model_output/exp2_cat-avocado/scheduler")

for idx, prompt in enumerate(PROMPTS):
    print(prompt)
    # image = pipeline("A photo of sks pothole in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
    image = pipeline(prompt=prompt, num_inference_steps=20)
    image = image.images[0]
    image.save(f"images/dreambooth_experiments/exp2_cat-avocado_{idx}.png")