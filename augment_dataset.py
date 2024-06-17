import argparse
import logging
import os
import time
import random

import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm
from os.path import join
from PIL import Image

from scripts.sd_inpaint_dreambooth import StableDiffusionModel

class AugmentDataset():
    pass
def find_object_dimensions(image):
    # image.save("2.jpg")
    image_array = np.array(image)
    
    # Find coordinates of the object (pixels with value 255)
    object_mask = (image_array > 200)
    
    print("object_mask: ", np.unique(object_mask))
    assert object_mask.size == []
    
    object_y, object_x = np.where(object_mask)
    
    # Debug: Print some object coordinates
    print(f"Sample object coordinates: {(object_y[:10], object_x[:10])}")
    
    # Find the bounding box
    min_x, max_x = object_x.min(), object_x.max()
    min_y, max_y = object_y.min(), object_y.max()
    
    # Calculate width and height
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    return width, height


def main(args):
    logger = logging.getLogger(__name__)

    # load pipe
    # pipe = StableDiffusionModel(dreambooth_checkpoint=args.dreambooth_chkpt)
    logger.info("Load SD pipeline")

    # load prompts 
    with open(args.prompts_path, "r") as file:
        prompts = [line.rstrip() for line in file]

    logger.info("Load prompts"+args.prompts_path)

    # load mask (crutch) 
    mask = Image.open(args.mask_path).convert('L')
    object_height, object_width = find_object_dimensions(mask)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # for image in images
    logger.info("Start generating")
    avg_total_time = 0
    avg_generation_time = 0

    images_paths = sorted([join(args.dataset_path, f) for f in os.listdir(args.dataset_path+"/images")])
    background_masks_paths = sorted([join(args.dataset_path, f) for f in os.listdir(args.dataset_path+"/masks")])
    for idx, image_pth in tqdm(enumerate(images_paths)):
        start_time = time.time()
    
        image = Image.open(image_pth)

        # load background mask for current image (NOW USE ONLY ONE MASK)
        background_mask = Image.open(background_masks_paths[idx])

        filename_img = image_pth.split("/")[-1].split(".")[0]
        filename_background_mask = background_masks_paths[idx].split("/")[-1].split(".")[0]
        assert filename_img != filename_background_mask

        # get pseudo mask

        # random coords for mask 
        # Convert the mask image to a binary numpy array
        mask_array = np.array(mask) // 255

        # Create a kernel of the object's size
        kernel = np.ones((object_height, object_width), dtype=np.uint8)

        # Convolve the mask with the kernel to find fit areas
        conv_result = convolve(mask_array, kernel, mode='constant', cval=0)

        # Find all valid top-left coordinates where the object can fit
        valid_coordinates = np.argwhere(conv_result == object_width * object_height)

        # If there are no valid coordinates, return None
        assert valid_coordinates.size == 0

        # Randomly select one of the valid coordinates
        random_index = random.choice(valid_coordinates)
        random_coordinate = list(random_index)

        print(random_coordinate)

        # resize mask for y coord (maybe Depth Anything) 

        # get box with mask (log to file [x, y, w, h])

        # inpainting
        start_generating_time = time.time()

        avg_generation_time += time.time() - start_generating_time  

        # save image
        avg_total_time += time.time() - start_time  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dreambooth_chkpt", type=str, required=True)
    args = parser.parse_args()
    
    main(args)