import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from transformers import CLIPTextModel


class StableDiffusionModel():
    def __init__(self, 
                 pretrained="stabilityai/stable-diffusion-2-inpainting", 
                 textual_inversion_checkpoint="model_output/exp1/checkpoint-1000",
                 dreambooth_checkpoint="model_output/db_pothole") -> None:
        
        self.pretrained = pretrained
        self.textual_inversion_checkpoint = textual_inversion_checkpoint
        self.dreambooth_checkpoint = dreambooth_checkpoint

        self.device = torch.device("cuda")
        print("DEVICE FOR SD_DB: ", self.device)

        unet = UNet2DConditionModel.from_pretrained(self.dreambooth_checkpoint+"/unet")

        # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
        text_encoder = CLIPTextModel.from_pretrained(self.dreambooth_checkpoint+"/text_encoder")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.dreambooth_checkpoint, unet=unet, text_encoder=text_encoder
        ).to("cuda")
        self.pipe.scheduler = DDIMScheduler.from_config(self.dreambooth_checkpoint)

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
        inpaint_image = inpaint_images[0]

        return inpaint_image

