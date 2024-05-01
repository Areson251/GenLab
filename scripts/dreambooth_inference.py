from diffusers import DiffusionPipeline, UNet2DConditionModel, StableDiffusionInpaintPipeline, DDIMScheduler
from transformers import CLIPTextModel
import torch

PROMPTS = [
    "A sks pothole",
    "A a road with sks pothole leading off into the distance and surrounded by a clearing",
    "sks pothole on paper",
    "sks pothole on the crossroads road",
    ]

unet = UNet2DConditionModel.from_pretrained("model_output/db_inp_exp2/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("model_output/db_inp_exp2/text_encoder")

# pipeline = DiffusionPipeline.from_pretrained(
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
).to("cuda")
pipeline.scheduler = DDIMScheduler.from_config("model_output/db_inp_exp2/scheduler")

for idx, prompt in enumerate(PROMPTS):
    # image = pipeline("A photo of sks pothole in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
    image = pipeline("A photo of sks pothole in a bucket", num_inference_steps=20).images[0]
    image.save(f"images/dreambooth_experiments/pothole_{idx}.png")