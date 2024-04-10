import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline


class StableDiffusionModel():
    def __init__(self, 
                 pretrained="stabilityai/stable-diffusion-2-inpainting", 
                 textual_inversion_checkpoint="model_output/exp1/checkpoint-1000") -> None:
        
        self.pretrained = pretrained
        self.textual_inversion_checkpoint = textual_inversion_checkpoint

        self.device = torch.device("mps")
        print("DEVICE FOR SD: ", self.device)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            # "runwayml/stable-diffusion-v1-5",
            # "stabilityai/stable-diffusion-2-inpainting",
            # "model_output/checkpoint-1000",
            pretrained_model_name_or_path=self.pretrained,
            requires_safety_checker=False,
            safety_checker=None,
            variant='fp16',
            torch_dtype=torch.float32,
        # )
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def load_textual_inversion(self):
        self.pipe.load_textual_inversion(self.textual_inversion_checkpoint)

    def diffusion_inpaint(self, image, mask, 
                          positive_prompt, negative_prompt, 
                          w_orig, h_orig, 
                          iter_number, guidance_scale):
        
        inpaint_images = self.pipe(
            num_inference_steps=iter_number,
            prompt=positive_prompt,
            # negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            strength=1.0
        ).images

        print("GENERATED IMAGE COUNT: ", len(inpaint_images))
        inpaint_image = inpaint_images[0]

        inpaint_image = inpaint_image.resize((w_orig, h_orig))
        return np.array(inpaint_image)
    

# stable_diffusion = StableDiffusionModel()