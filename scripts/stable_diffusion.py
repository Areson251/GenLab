import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from transformers import CLIPTextModel


class StableDiffusionModel():
    def __init__(self, 
                 pretrained="stabilityai/stable-diffusion-2-inpainting") -> None:
        
        self.pretrained = pretrained
        self.textual_inversion_checkpoint = None

        self.device = torch.device("cuda")
        print("DEVICE FOR SD: ", self.device)

       # dreambooth
        # unet = UNet2DConditionModel.from_pretrained(self.pretrained+"/unet")
        # # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
        # text_encoder = CLIPTextModel.from_pretrained(self.pretrained+"/text_encoder")
       
        # unet = UNet2DConditionModel.from_pretrained(self.pretrained)
        # text_encoder = CLIPTextModel.from_pretrained(self.pretrained)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path=self.pretrained,
            # unet=unet, 
            # text_encoder=text_encoder,
            requires_safety_checker=False,
            safety_checker=None,
            variant='fp16',
            torch_dtype=torch.float32,
        # )
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def load_textual_inversion(self, textual_inversion_checkpoint):
        self.textual_inversion_checkpoint = textual_inversion_checkpoint
        self.pipe.load_textual_inversion(self.textual_inversion_checkpoint)

    def unload_textual_inversion(self):
        self.textual_inversion_checkpoint = None
        self.pipe.unload_textual_inversion()

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