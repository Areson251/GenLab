import os
import argparse
from PIL import Image
import numpy as np

def process_images(input_folder: str, label: int, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            image = Image.open(input_path).convert("L")
            
            img_array = np.array(image)
            processed_array = np.where(img_array == label, 255, 0).astype(np.uint8)
            
            processed_image = Image.fromarray(processed_array)
            processed_image.save(output_path)
            
        except Exception as e:
            print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images by replacing specific pixel values.")
    parser.add_argument("--input_folder", type=str, help="Path to the input folder containing images.")
    parser.add_argument("--label", type=int, help="Pixel value to be replaced with 255.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder where processed images will be saved.")
    
    args = parser.parse_args()
    
    process_images(args.input_folder, args.label, args.output_folder)
