import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from transformers import CLIPTextModel


class StableDiffusionModel():
    def __init__(self, 
                 pretrained="stabilityai/stable-diffusion-2-inpainting", 
                 textual_inversion_checkpoint="model_output/exp1/checkpoint-1000") -> None:
        
        self.pretrained = pretrained
        self.textual_inversion_checkpoint = textual_inversion_checkpoint

        self.device = torch.device("cuda")
        print("DEVICE FOR SD: ", self.device)

        unet = UNet2DConditionModel.from_pretrained("model_output/db_pothole/unet")
        # unet = UNet2DConditionModel.from_pretrained("model_output/db_inp_exp2/unet")

        # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
        text_encoder = CLIPTextModel.from_pretrained("model_output/db_pothole/text_encoder")
        # text_encoder = CLIPTextModel.from_pretrained("model_output/db_inp_exp2/text_encoder")

        # pipeline = DiffusionPipeline.from_pretrained(
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "model_output/db_pothole/checkpoint-1000", unet=unet, text_encoder=text_encoder
            # "model_output/db_inp_exp2/checkpoint-1000", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
            # "stabilityai/stable-diffusion-2-inpainting", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
        ).to("cuda")
        self.pipe.scheduler = DDIMScheduler.from_config("model_output/db_pothole/checkpoint-1000")
        # self.pipe.scheduler = DDIMScheduler.from_config("model_output/db_inp_exp2/scheduler")

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

        # inpaint_image = inpaint_image.resize((w_orig, h_orig))
        return inpaint_image
        # return np.array(inpaint_image)
    

# stable_diffusion = StableDiffusionModel()