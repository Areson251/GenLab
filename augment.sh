#!/bin/bash

# python3 scripts/generate_images_annots_entity.py \
#     --json_path="custom_datasets/scenes/pothole_scenes.json" \
#     --hq=False \
#     --output_json="custom_datasets/scenes/pothole_scenes_edit.json" \
#     --task="text-guided"

# echo "CONVERT ANNOTATIONS"

python augment_dataset.py \
            --images_path="datasets/original/YCOR/val/images" \
            --annotation_path="datasets/original/YCOR/val/masks_road" \
            --masks_path="custom_datasets/target_masks" \
            --prompts_path="prompts/pothole.txt" \
            --scene_prompts_path="prompts/YCOR.txt" \
            --output_path="tuning_exps/sd2_boxes/YCOR.augmented_val_gs-1_ckpt-6" \
            --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
            --lora_chkpt="model_output/lora_pothole-full2_sd2/checkpoint-6000" \
            --padding=20 \
            --guidance_scale=1 \
            --seed=0 
 
# declare -a gs_arr=(0.7 1 2 3)
# for gs in "${gs_arr[@]}"; do 
#     echo "START GS ${gs} GENERATION"
#     python augment_dataset.py \
#             --images_path="datasets/original/INSTSnowRoadDetection/test" \
#             --annotation_path="datasets/original/INSTSnowRoadDetection/annotation_test.json" \
#             --masks_path="custom_datasets/target_masks" \
#             --prompts_path="prompts/pothole.txt" \
#             --output_path="tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_gs-${gs}_ckpt-6" \
#             --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
#             --lora_chkpt="model_output/lora_pothole-full2_sd2/checkpoint-6000" \
#             --padding=20 \
#             --guidance_scale=${gs} \
#             --seed=0 
#     done
# echo "IMAGES GENERATED"

# declare -a gs_arr=(0.7 1 2 3)
# for gs in "${gs_arr[@]}"; do 
#     python3 scripts/utils/merge_images.py \
#         --output_path="results/sd2_boxes/synth_pothole_sd2_${gs}.png" \
#         --images_folders \
#         "custom_datasets/scenes/pothole_scenes" \
#         "tuning_exps/sd2_masks/full-2000-${gs}/" \
#         "tuning_exps/sd2_masks/full-4000-${gs}/" \
#         "tuning_exps/sd2_masks/full-6000-${gs}/" \
#         "tuning_exps/sd2_masks/full-8000-${gs}/" \
#         "tuning_exps/sd2_masks/full-10000-${gs}/" \
#         "tuning_exps/sd2_masks/full-12000-${gs}/" \
#         "tuning_exps/sd2_masks/full-14000-${gs}/" \
#         "tuning_exps/sd2_masks/full-${gs}/"
# done
# echo "DONE RESULT IMAGE"


# python3 scripts/utils/merge_images.py \
#     --output_path="results/sd2_boxes/INSTSnowRoadDetection.augmented_ckpt-6.png" \ 
#     --images_folders \  
#             "datasets/original/INSTSnowRoadDetection/test" \
#             "tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_gs-0.7_ckpt-6" \
#             "tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_gs-1_ckpt-6" \
#             "tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_gs-2_ckpt-6" \
#             "tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_gs-3_ckpt-6" \
#     --max_images="5"
# done
# echo "DONE RESULT IMAGE"



# echo "MOVING IMAGES"
# python3 scripts/utils/move_images.py \
#     --output_path="datasets/augmented/INSTSnowRoadDetection/test" \ 
#     --images_folder="tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_gs-1_ckpt-6"
# done

