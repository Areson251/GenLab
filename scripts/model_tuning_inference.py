from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import safetensors

MODEL_CHECKPOINT = "model_output/exp3/checkpoint-1000"
PROMPTS = [
    "A <pothole>",
    "A a road with <pothole> leading off into the distance and surrounded by a clearing",
    "<pothole> on paper",
    "<pothole> on the crossroads road",
    ]

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16).to("mps")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

for idx, prompt in enumerate(PROMPTS):
    image = pipeline(prompt, num_inference_steps=50).images[0]
    image.save(f"images/otchet/pothole_SD2_{idx}.png")

print(f"load textual inversion from {MODEL_CHECKPOINT}")
pipeline.load_textual_inversion(MODEL_CHECKPOINT)

for idx, prompt in enumerate(PROMPTS):
    image = pipeline(prompt, num_inference_steps=50).images[0]
    image.save(f"images/otchet/pothole_SD2_tuned_{idx}.png")

