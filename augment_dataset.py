import argparse
import logging
import os
import time
import random
import pycocotools.coco
import torch
import cv2
import pycocotools

import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm
from os.path import join
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline

from scripts.sd_inpaint_dreambooth import StableDiffusionModel

class AugmentDataset():
    def __init__(self, args) -> None:
        self.prompts = args.prompts_path
        self.dataset_path = args.dataset_path
        self.output_path = args.output_path
        self.dreambooth_checkpoint = args.dreambooth_chkpt
        self.masks_path = args.masks_path
        self.padding = args.padding
        self.iter_number = args.iter_number
        self.guidance_scale = args.guidance_scale

        self.diffusion_pipe = None
        self.depth_pipe = None
        self.annotation = None

        self.logger = logging.getLogger(__name__)
        path = os.path.join(self.output_path, "log.log")
        logging.basicConfig(filename=path, level=logging.INFO)

        self.setup()

    def setup(self):
        self.init_annotation()

        self.load_pipes(self.dreambooth_checkpoint)
        self.load_prompts(self.prompts)

        self.make_dirs(self.output_path)
        self.make_dirs(f"{self.output_path}/images")
        self.make_dirs(f"{self.output_path}/masks")

    def init_annotation(self):
        self.annotation = pycocotools.coco.COCO()
        i=0

    def load_mask(self, masks_path):
        mask = Image.open(masks_path).convert('L')
        object_width, object_height, cropped_object = self.get_masked_object(mask)
        self.logger.info(f"Find object dimensions: {object_width}x{object_height}")

        return mask, object_width, object_height, cropped_object

    def load_pipes(self, dreambooth_checkpoint):
        self.diffusion_pipe = StableDiffusionModel(dreambooth_checkpoint=dreambooth_checkpoint)
        self.logger.info("Load SD pipeline")

        self.depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
        self.logger.info("Load DEPTH pipeline")

    def load_prompts(self, prompts_path):
        with open(prompts_path, "r") as file:
            self.prompts = [line.rstrip() for line in file]

        self.logger.info("Load prompts"+prompts_path)

    def make_dirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_masked_object(self, image):
        image_array = np.array(image)

        object_mask = (image_array > 200)
        assert object_mask.shape != ()
        object_y, object_x = np.where(object_mask)

        min_x, max_x = object_x.min(), object_x.max()
        min_y, max_y = object_y.min(), object_y.max()
        width = max_x - min_x + self.padding
        height = max_y - min_y + self.padding

        cropped_image = image.crop((min_x, min_y, max_x, max_y))
        cropped_image = ImageOps.expand(cropped_image, border=self.padding, fill='black')

        return width, height, cropped_image
    
    def get_filename_without_ext(self, image_pth):
        base_name = os.path.basename(image_pth)  # Get the filename
        name_without_ext = os.path.splitext(base_name)[0]  # Remove the last extension
        return name_without_ext

    def get_valid_coordinates(self, background_mask, object_height, object_width, depth_map):
        background_mask = np.array(background_mask)
        background_object = np.where(background_mask == 255)
        background_object_coords = list(zip(background_object[1], background_object[0]))

        valid_coordinates = []
        for coords in background_object_coords:
            x, y = coords
            resize_coeff = np.mean(depth_map[y:y+object_height, x:x+object_width])
            target_height = int(object_height * resize_coeff)
            target_width = int(object_width * resize_coeff)

            if y+target_height < background_mask.shape[0] and x+target_width < background_mask.shape[1]:
                if np.all(background_mask[y:y+target_height, x:x+target_width] == 255):
                        valid_coordinates.append((x, y, resize_coeff))


        assert len(valid_coordinates) != 0, "No valid coordinates found"
        return valid_coordinates, background_object_coords

    def get_depth_map(self, image):
        with torch.no_grad():
            depth = self.depth_pipe(image)["depth"]

        depth = np.asarray(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        return depth

    # DEBUG
    def draw_point_and_save(self, mask, random_coordinates, save_path):
        # Convert the mask to an RGB image
        image = mask.convert('RGB')

        # Create a draw object
        draw = ImageDraw.Draw(image)

        # Calculate the bounding box of the circle
        left = random_coordinates[0] - 5
        top = random_coordinates[1] - 5
        right = random_coordinates[0] + 5
        bottom = random_coordinates[1] + 5

        # Draw a red circle at the random coordinates
        draw.ellipse([(left, top), (right, bottom)], outline='red')

        # Save the image
        image.save(save_path)

    def draw_valid_coordinates_and_save(self, mask, valid_coordinates, save_path):
        # Convert the mask to an RGB image
        image = mask.convert('RGB')

        # Create a draw object
        draw = ImageDraw.Draw(image)

        # Draw a red point at each valid coordinate
        for coord in valid_coordinates:
            draw.point([coord[0], coord[1]], fill='red')

        # Save the image
        image.save(save_path)

    def save_depth(self, depth, save_path):
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(save_path, depth)

    # MAIN FUNCTION
    def augment_dataset(self):
        self.logger.info("Start generating")
        avg_total_time = 0
        avg_generation_time = 0

        images_paths = sorted([join(self.dataset_path, "images", f) for f in os.listdir(self.dataset_path+"/images")])
        background_masks_paths = sorted([join(self.dataset_path, "masks", f) for f in os.listdir(self.dataset_path+"/masks")])
        target_masks = sorted([join(self.masks_path, f) for f in os.listdir(self.masks_path)])

        for idx, image_pth in tqdm(enumerate(images_paths)):
            start_time = time.time()

            # load image and its background mask 
            image = Image.open(image_pth)
            background_mask = Image.open(background_masks_paths[idx])

            self.logger.info(f"Load image: {image_pth}")
            self.logger.info(f"Load background mask: {background_masks_paths[idx]}")

            filename_img = self.get_filename_without_ext(image_pth)
            filename_background_mask = self.get_filename_without_ext(background_masks_paths[idx])
            assert filename_img.split(".")[0] == filename_background_mask.split(".")[0]

            # get random pseudo mask (NOW USE ONLY ONE MASK)
            random_mask_path = random.choice(target_masks)
            self.logger.info(f"Load random mask: {random_mask_path}")
            target_mask, object_width, object_height, cropped_object = self.load_mask(random_mask_path)

            depth_map = self.get_depth_map(image)
            valid_coordinates, background_object_coords = self.get_valid_coordinates(background_mask, object_height, object_width, depth_map)

            # Randomly select one of the valid coordinates
            random_coordinates = random.choice(valid_coordinates)
            random_x, random_y, resize_coeff = random_coordinates
            self.logger.info(f"Random coordinates: {random_coordinates}")

            # self.draw_point_and_save(background_mask, random_coordinates, f"{self.output_path}/{filename_img}_background_mask.png")
            # self.draw_valid_coordinates_and_save(background_mask, valid_coordinates, f"{self.output_path}/{filename_img}_valid_coordinates.png")
            # self.draw_valid_coordinates_and_save(background_mask, background_object_coords, f"{self.output_path}/{filename_img}_background_object_coords.png")
            # self.save_depth(depth_map, f"{self.output_path}/{filename_img}_depth.png")

            # resize mask due to depth map
            resized_width = int(object_width * resize_coeff)
            resized_height = int(object_height * resize_coeff)
            cropped_resized_object = cropped_object.resize((resized_width, resized_height))
            
            # get box with mask (log to file [x, y, w, h])
            box = [random_x, random_y, resized_width, resized_height]
            self.logger.info(f"Box: {box}")
            cropped_image = image.crop((random_x, random_y, random_x+resized_width, random_y+resized_height))
            
            # inpainting
            for prompt in self.prompts:
                start_generating_time = time.time()
                generated_image = self.diffusion_pipe.diffusion_inpaint(
                    cropped_image, cropped_resized_object, 
                    prompt, None, image.size[0], image.size[1],
                    self.iter_number, self.guidance_scale
                )
                generation_time = time.time() - start_generating_time
                avg_generation_time += generation_time
                
                self.logger.info(f"Prompt: {prompt}")
                self.logger.info(f"Generation time: {generation_time}")

                resized_generated_image = generated_image.resize((resized_width, resized_height))

                augmented_image = image.copy()
                augmented_image.paste(resized_generated_image, (random_x, random_y))

                augmented_mask = background_mask.copy()
                augmented_mask.paste(cropped_resized_object, (random_x, random_y))  

                augmented_image.save(f"{self.output_path}/images/{filename_img}_{prompt}_augmented.png")
                augmented_mask.save(f"{self.output_path}/masks/{filename_img}_{prompt}_augmented.png")

            avg_total_time += time.time() - start_time
            self.logger.info(f"Average generation time: {avg_generation_time/(idx+1)}")

            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--masks_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dreambooth_chkpt", type=str, required=True)
    parser.add_argument("--padding", type=int, required=False, default="0")
    parser.add_argument("--iter_number", type=int, required=False, default="20")
    parser.add_argument("--guidance_scale", type=float, required=False, default="0.7")
    args = parser.parse_args()
    
    dataset_augmentator = AugmentDataset(args)
    dataset_augmentator.augment_dataset()