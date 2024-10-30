from PIL import Image
import argparse
import os
import tqdm

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

class ImageMerger():
    def __init__(self):
        self.images = []

    def add_image(self, image):
        self.images.append(image)
    
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
    
    def merge_column(self):
        widths, heights = zip(*(i.size for i in self.images))
        min_width = min(widths)
        self.images, widths, heights = self.resize_images("width", min_width, widths, heights)

        total_height = sum(heights)
        max_width = max(widths)
        new_image = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for image in tqdm.tqdm(self.images):
            new_image.paste(image, (0, y_offset))
            y_offset += image.size[1]
        return new_image
    
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
    args = parser.parse_args()
    
    merged_rows = []
    for folder in args.images_folders:
        images_paths = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
        images_paths = sorted(images_paths)
        merger = ImageMerger()

        print(f"load {len(images_paths)} images from {folder}")
        for image_path in images_paths:
            image = Image.open(os.path.join(folder, image_path))
            merger.add_image(image) 

        print("merge images")
        merged_rows.append(merger.merge_column()) 
        # merged_rows.append(merger.merge_row()) 

    merger.clear_images()
    for image in merged_rows:
        merger.add_image(image)

    merged_image = merger.merge_row()
    # merged_image = merger.merge_column()
    if not args.output_path.lower().endswith(IMAGE_EXTENSIONS):
        args.output_path += ".png"
    merged_image.save(args.output_path)
