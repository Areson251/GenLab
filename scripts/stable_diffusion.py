import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from transformers import CLIPTextModel


class StableDiffusionModel():
    def __init__(self, 
                 pretrained="stabilityai/stable-diffusion-2-inpainting", device="cuda") -> None:
        
        self.pretrained = pretrained
        self.textual_inversion_checkpoint = None

        self.device = torch.device(device)
        print("DEVICE FOR SD: ", self.device)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path=self.pretrained,
            requires_safety_checker=False,
            safety_checker=None,
            variant='fp16',
            torch_dtype=torch.float32,
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def load_lora(self, lora_checkpoint):
        self.pipe.load_lora_weights(lora_checkpoint, weight_name="pytorch_lora_weights.safetensors")
        print(f"Loaded lora weights from {lora_checkpoint}")

    def load_textual_inversion(self, textual_inversion_checkpoint):
        self.textual_inversion_checkpoint = textual_inversion_checkpoint
        self.pipe.load_textual_inversion(self.textual_inversion_checkpoint)

    def unload_textual_inversion(self):
        self.textual_inversion_checkpoint = None
        self.pipe.unload_textual_inversion()

    def __call__(self, image, mask, 
                          positive_prompt, negative_prompt, 
                          w_orig, h_orig, 
                          iter_number, guidance_scale):
        
        inpaint_image = self.pipe(
            num_inference_steps=iter_number,
            prompt=positive_prompt,
            # negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            strength=1.0
        ).images[0]

        return inpaint_image
    