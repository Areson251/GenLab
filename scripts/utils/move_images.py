from PIL import Image
import argparse
import os
import tqdm
import shutil

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

class Mover():
    def __init__(self, images_folder, output_path):
        self.images_folder = images_folder
        self.output_path = output_path

    def make_dirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def move(self):
        self.make_dirs(self.output_path)

        # move annotation
        annotation_path = os.path.join(self.images_folder, "annotation.json")
        if os.path.exists(annotation_path):
            shutil.copy(annotation_path, os.path.join(self.output_path, "annotation.json"))

        # move images
        folders = sorted([os.path.join(self.images_folder, folder) for folder in next(os.walk(self.images_folder))[1]])
        for folder in folders:
            img_name = os.path.join(folder, folder.split("/")[-1]+"_0.jpg")
            if os.path.exists(img_name):
                img = Image.open(img_name)
                img.save(os.path.join(self.output_path, f"{folder.split('/')[-1]}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="moved")
    parser.add_argument("--images_folder", required=True, help="Provide one images folder")
    args = parser.parse_args()

    mover = Mover(args.images_folder, args.output_path)
    mover.move()
