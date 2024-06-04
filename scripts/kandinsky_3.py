import torch
import numpy as np
from scripts.kandinsky3 import get_inpainting_pipeline


class Kandinsky3Model():
    def __init__(self) -> None:
        self.device = torch.device('cuda:1')
        print("DEVICE FOR KANDINSKY: ", self.device)

        self.dtype_map = {
            'unet': torch.float16,
            'text_encoder': torch.float16,
            'movq': torch.float32,
        }

        self.pipe = get_inpainting_pipeline(
            self.device, self.dtype_map,
        )

    def diffusion_inpaint(self, image, mask, 
                          positive_prompt, negative_prompt, 
                          w_orig, h_orig, 
                          iter_number, guidance_scale):
        
        inpaint_images = self.pipe(
            steps=iter_number,
            text=positive_prompt, 
            image=image, 
            mask=mask,
            guidance_weight_text=guidance_scale,
            )
        inpaint_image = inpaint_images[0]
        inpaint_image = inpaint_image.resize((w_orig, h_orig))

        return np.array(inpaint_image)
