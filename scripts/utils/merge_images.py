from PIL import Image
import argparse
import os
import tqdm
from typing import List
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

class ImageMerger():
    def __init__(self, max_images):
        self.images = []
        self.max_images = max_images

    def add_image(self, image):
        self.images.append(image)

    def read_images(self, folder) -> List:
        images_paths = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
        images_paths = sorted(images_paths)
        return images_paths
    
    def merge_row(self):
        widths, heights = zip(*(i.size for i in self.images))
        min_height = min(heights)
        self.images, widths, heights = self.resize_images("height", min_height, widths, heights)

        total_width = sum(widths)
        max_height = max(heights)
        new_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for image in tqdm.tqdm(self.images):
            resize_koef = min_height / image.size[1]
            resized_image = image.resize((int(image.size[0] * resize_koef), 
                                          int(image.size[1] * resize_koef)))
            new_image.paste(resized_image, (x_offset, 0))
            x_offset += image.size[0]
        return new_image
    
    def merge_column(self) -> List: 
        all_images = []
        widths, heights = zip(*(i.size for i in self.images))
        min_width = min(widths)
        self.images, widths, heights = self.resize_images("width", min_width, widths, heights)
        images_count = len(self.images)

        if self.max_images:
            for i in range(0, images_count, self.max_images):
                cur_heights = heights[i:i+self.max_images]
                cur_widths = widths[i:i+self.max_images]

                total_height = sum(cur_heights)
                max_width = max(cur_widths)
                new_image = Image.new('RGB', (max_width, total_height))
                y_offset = 0
                for image in tqdm.tqdm(self.images[i:i+self.max_images]):
                    new_image.paste(image, (0, y_offset))
                    y_offset += image.size[1]
                
                all_images.append(new_image)
        
        else:
            total_height = sum(heights)
            max_width = max(widths)
            new_image = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for image in tqdm.tqdm(self.images):
                new_image.paste(image, (0, y_offset))
                y_offset += image.size[1]
            
            all_images.append(new_image)
        
        return all_images
    
    def resize_images(self, side_by, min_side, widths, heights):
        new_images = []
        new_widths = []
        new_heights = []
        for image, width, height in zip(self.images, widths, heights):
            resize_koef = min_side / locals()[side_by]
            new_widths.append(int(width * resize_koef))
            new_heights.append(int(height * resize_koef))
            new_images.append(image.resize((int(width * resize_koef), 
                                          int(height * resize_koef))))
        return new_images, new_widths, new_heights
    
    def clear_images(self):
        self.images = []

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="merged_image.png")
    parser.add_argument("--images_folders", nargs='+', 
                        required=True, help="Provide one or more images folders")
    parser.add_argument("--max_images", type=int, 
                        default=None, help="Count of images in one column")
    args = parser.parse_args()
    
    merger = ImageMerger(args.max_images)
    images_folders = args.images_folders

    merged_first = []
    original = True
    for folder in args.images_folders:
        if original:
            images_paths = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
        else:
            images_paths = [f for f in os.listdir(folder)]
            images_paths.remove('annotation.json')
            images_paths.remove('log.log')

        images_paths = sorted(images_paths)

        merger.clear_images()

        print(f"load {len(images_paths)} images from {folder}")
        for image_path in images_paths:
            if original:
                image = Image.open(os.path.join(folder, image_path))
            else:
                # file_name = os.path.join(folder, image_path, image_path)+"_pothole.png"
                file_name = os.path.join(folder, image_path, image_path)+"_pit.png"
                # file_name = os.path.join(folder, image_path, image_path)+"_0.jpg"
                image = Image.open(file_name)
            merger.add_image(image) 

        print("merge images")
        merged_first.append(merger.merge_column()) 
        original = False

    merger.clear_images()
    for idx, col in enumerate(zip(*merged_first)):
        for image in col:
            merger.add_image(image)

        merged_image = merger.merge_row()
        output_filename = "".join(args.output_path.split('.')[:-1]) + f"_{idx}.png"
        
        path = "/".join(output_filename.split('/')[:-1])
        if not os.path.exists(path):
            os.makedirs(path)

        merged_image.save(output_filename)
        merger.clear_images()
