import os
import shutil
import argparse
from pycocotools.coco import COCO
from PIL import Image

def filter_and_copy_images(images_path, annotation_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    coco = COCO(annotation_path)
    image_ids = coco.getImgIds()
    images_info = coco.loadImgs(image_ids)
    images_names = {img["file_name"] for img in images_info}

    for filename in os.listdir(images_path):
        if filename in images_names:
            src_path = os.path.join(images_path, filename)
            dst_path = os.path.join(output_path, filename)
            
            try:
                with Image.open(src_path) as img:
                    img.verify()
                shutil.copy2(src_path, dst_path)
                
            except Exception as e:
                print(f"ERROR {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str)
    parser.add_argument("--annotation_path", type=str)
    parser.add_argument("--output_path", type=str)
    
    args = parser.parse_args()
    filter_and_copy_images(args.images_path, args.annotation_path, args.output_path)
