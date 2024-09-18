import subprocess

arguments = [
    # ["--images_folder", "custom_datasets/images/objects/pothole", "--dimension", "depth"],
    # ["--images_folder", "custom_datasets/images/objects/pothole_far", "--dimension", "depth"],
    # ["--images_folder", "custom_datasets/images/objects/random_objects", "--dimension", "depth"],
    ["--images_folder", "custom_datasets/images/scenes/INSTSnowRoadDetection", "--dimension", "depth"],
    # ["--images_folder", "custom_datasets/images/scenes/random_scenes", "--dimension", "depth"],

    # ["--images_folder", "custom_datasets/images/objects/pothole", "--dimension", "normals"],
    # ["--images_folder", "custom_datasets/images/objects/pothole_far", "--dimension", "normals"],
    # ["--images_folder", "custom_datasets/images/objects/random_objects", "--dimension", "normals"],
    ["--images_folder", "custom_datasets/images/scenes/INSTSnowRoadDetection", "--dimension", "normals"],
    # ["--images_folder", "custom_datasets/images/scenes/random_scenes", "--dimension", "normals"],
]

for args in arguments:
    subprocess.run(["python", "scripts/controlnet_sd.py"] + args)