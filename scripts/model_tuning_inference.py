from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import safetensors

PROMPT = "A pothole"

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16).to("mps")
pipeline.load_textual_inversion("model_output/checkpoint-1000")

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

image = pipeline(PROMPT, num_inference_steps=50).images[0]
image.save("images/experiments/pothole_SD2.png")
