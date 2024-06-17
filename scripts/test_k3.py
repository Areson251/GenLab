import requests
from io import BytesIO

import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from scripts.kandinsky3 import get_inpainting_pipeline
from scripts.kandinsky3.utils import prepare_mask

device_map = torch.device('cuda')
dtype_map = {
    'unet': torch.float16,
    'text_encoder': torch.float16,
    'movq': torch.float32,
}

pipe = get_inpainting_pipeline(
    device_map, dtype_map,
)

w, h = 768, 768
shape = [(200, 100), (500, 700)] 
  
# creating new Image object 
mask_image = Image.new("L", (w, h)) 
  
# create rectangle image 
img1 = ImageDraw.Draw(mask_image)   
img1.rectangle(shape, fill ="#ffffff")

mask = prepare_mask(mask_image)



def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"

image = download_image(img_url).resize((768, 768))

res = pipe("cheburashka sitting on a bench", image, mask)

plt.imshow(res[0])
plt.show()
