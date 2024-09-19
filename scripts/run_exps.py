import subprocess

# arguments = [
#     # ["--images_folder", "custom_datasets/images/objects/pothole", "--dimension", "depth"],
#     # ["--images_folder", "custom_datasets/images/objects/pothole_far", "--dimension", "depth"],
#     # ["--images_folder", "custom_datasets/images/objects/random_objects", "--dimension", "depth"],
#     # ["--images_folder", "custom_datasets/images/scenes/INSTSnowRoadDetection", "--dimension", "depth"],
#     ["--images_folder", "custom_datasets/images/scenes/random_scenes", "--dimension", "depth"],

#     # ["--images_folder", "custom_datasets/images/objects/pothole", "--dimension", "normals"],
#     # ["--images_folder", "custom_datasets/images/objects/pothole_far", "--dimension", "normals"],
#     # ["--images_folder", "custom_datasets/images/objects/random_objects", "--dimension", "normals"],
#     # ["--images_folder", "custom_datasets/images/scenes/INSTSnowRoadDetection", "--dimension", "normals"],
#     # ["--images_folder", "custom_datasets/images/scenes/random_scenes", "--dimension", "normals"],
# ]

# for args in arguments:
#     subprocess.run(["python", "scripts/controlnet_sd.py"] + args)

arguments = [
    ["--output_path", "results/pothole_augmented_depth.png", "--images_folders", "custom_datasets/images/objects/pothole", "depth_exps/custom_datasets/images/objects/pothole/depth", "depth_exps/custom_datasets/images/objects/pothole/augmented"],
    ["--output_path", "results/pothole_far_augmented_depth.png", "--images_folders", "custom_datasets/images/objects/pothole_far", "depth_exps/custom_datasets/images/objects/pothole_far/depth", "depth_exps/custom_datasets/images/objects/pothole_far/augmented"],
    ["--output_path", "results/random_objects_augmented_depth.png", "--images_folders", "custom_datasets/images/objects/random_objects", "depth_exps/custom_datasets/images/objects/random_objects/depth", "depth_exps/custom_datasets/images/objects/random_objects/augmented"],
    ["--output_path", "results/INSTSnowRoadDetection_augmented_depth.png", "--images_folders", "custom_datasets/images/scenes/INSTSnowRoadDetection", "depth_exps/custom_datasets/images/scenes/INSTSnowRoadDetection/depth", "depth_exps/custom_datasets/images/scenes/INSTSnowRoadDetection/augmented"],
    ["--output_path", "results/random_scenes_augmented_depth.png", "--images_folders", "custom_datasets/images/scenes/random_scenes", "depth_exps/custom_datasets/images/scenes/random_scenes/depth", "depth_exps/custom_datasets/images/scenes/random_scenes/augmented"],
    
    ["--output_path", "results/pothole_augmented_normals.png", "--images_folders", "custom_datasets/images/objects/pothole", "normals_exps/custom_datasets/images/objects/pothole/normals", "normals_exps/custom_datasets/images/objects/pothole/augmented"],
    ["--output_path", "results/pothole_far_augmented_normals.png", "--images_folders", "custom_datasets/images/objects/pothole_far", "normals_exps/custom_datasets/images/objects/pothole_far/normals", "normals_exps/custom_datasets/images/objects/pothole_far/augmented"],
    ["--output_path", "results/random_objects_augmented_normals.png", "--images_folders", "custom_datasets/images/objects/random_objects", "normals_exps/custom_datasets/images/objects/random_objects/normals", "normals_exps/custom_datasets/images/objects/random_objects/augmented"],
    ["--output_path", "results/INSTSnowRoadDetection_augmented_normals.png", "--images_folders", "custom_datasets/images/scenes/INSTSnowRoadDetection", "normals_exps/custom_datasets/images/scenes/INSTSnowRoadDetection/normals", "normals_exps/custom_datasets/images/scenes/INSTSnowRoadDetection/augmented"],
    ["--output_path", "results/random_scenes_augmented_normals.png", "--images_folders", "custom_datasets/images/scenes/random_scenes", "normals_exps/custom_datasets/images/scenes/random_scenes/normals", "normals_exps/custom_datasets/images/scenes/random_scenes/augmented"],
]

for args in arguments:
    subprocess.run(["python", "scripts/utils/merge_images.py"] + args)