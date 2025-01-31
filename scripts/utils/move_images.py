from PIL import Image
import argparse
import os
import tqdm

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

class ImageMerger():
    def __init__(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="merged_image.png")
    parser.add_argument("--images_folders", nargs='+', 
                        required=True, help="Provide one or more images folders")
    args = parser.parse_args()