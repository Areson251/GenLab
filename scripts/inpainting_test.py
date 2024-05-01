from kandinsky3 import get_inpainting_pipeline
import torch

device_map = torch.device('cuda:0')
dtype_map = {
    'unet': torch.float16,
    'text_encoder': torch.float16,
    'movq': torch.float32,
}

pipe = get_inpainting_pipeline(
    device_map, dtype_map,
)

image = ... # PIL Image
mask = ... # Numpy array (HxW). Set 1 where image should be masked
image = pipe( "A cute corgi lives in a house made out of sushi.", image, mask)