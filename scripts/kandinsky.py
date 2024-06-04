import torch
import numpy as np
from diffusers import AutoPipelineForInpainting, KandinskyInpaintPipeline, KandinskyPriorPipeline


class KandinskyModel():
    def __init__(self) -> None:
        self.device = torch.device("cuda")
        # print("DEVICE FOR KANDINSKY: ", self.device)

        self.pipe = AutoPipelineForInpainting.from_pretrained(
        # self.pipe = KandinskyInpaintPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint", 
            # "kandinsky-community/kandinsky-2-1-inpaint",
            # torch_dtype=torch.float16
        # )
        ).to(self.device)
        # self.pipe.enable_model_cpu_offload()
        
        # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
        # self.pipe.enable_xformers_memory_efficient_attention()

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
            ).images
        
        print("GENERATED IMAGE COUNT: ", len(inpaint_images))
        inpaint_image = inpaint_images[0]
        
        inpaint_image = inpaint_image.resize((w_orig, h_orig))
        return np.array(inpaint_image)



# kandinsky = KandinskyModel()