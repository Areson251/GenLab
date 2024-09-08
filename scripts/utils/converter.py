from PIL import Image
import argparse
import os
from os.path import isfile, join
import tqdm


class Converter():
    @staticmethod
    def webp2png(dataset_path, output_path):
        files_names = [f for f in os.listdir(dataset_path) if isfile(join(dataset_path, f))]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        converted_files = 0
        for file_name in tqdm.tqdm(files_names):
            file_path = join(dataset_path, file_name)
            if file_name.endswith('.webp'):
                image = Image.open(file_path).convert("RGB")
                image.save(join(output_path, file_name.replace('.webp', '.png')), 'PNG')
                converted_files += 1
        print(f"Converted {converted_files} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    Converter.webp2png(args.dataset_path, args.output_path)
