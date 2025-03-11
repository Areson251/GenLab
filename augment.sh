#!/bin/bash

# python3 scripts/generate_images_annots_entity.py \
#     --json_path="custom_datasets/scenes/pothole_scenes.json" \
#     --hq=False \
#     --output_json="custom_datasets/scenes/pothole_scenes_edit.json" \
#     --task="text-guided"

# echo "CONVERT ANNOTATIONS"

# echo "CONVERT MASKS"
# python scripts/utils/convert_masks.py \
#             --input_folder=datasets/original/YCOR/train/masks \
#             --label=3 \
#             --output_folder=datasets/original/YCOR/train/masks_road

# echo "ALLOWDED_CATEGORIES"
# python scripts/utils/annotations_tools.py \
#             --annotation_path="datasets/original/TAOMR/val.json" \
#             --new_annotation_path="datasets/original/TAOMR/val_objects.json"

# echo "MOVE IMAGES"
# python scripts/utils/filter_images.py \
#             --images_path="datasets/original/TAOMR/val" \
#             --annotation_path="datasets/original/TAOMR/val_objects.json" \
#             --output_path="datasets/original/TAOMR/val_objects" 


 
declare -a ckpt_arr=(2 4 6 8 10)
for ckpt in "${ckpt_arr[@]}"; do
    echo "START ckpt ${ckpt} GENERATION"

    declare -a gs_arr=(0.7 1 2 3)
    for gs in "${gs_arr[@]}"; do  
        echo "START GS ${gs} GENERATION"
        python augment_dataset.py \
                --images_path="custom_datasets/scenes/pothole_scenes" \
                --annotation_path="custom_datasets/scenes/pothole_scenes_road.json" \
                --masks_path="custom_datasets/target_masks" \
                --prompts_path="prompts/pothole.txt" \
                --output_path="tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_${ckpt}" \
                --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
                --lora_chkpt="model_output/lora_taomr_sd2/checkpoint-$((ckpt*1000))" \
                --padding=20 \
                --guidance_scale=${gs} \
                --device="cuda:1" \
                --seed=0 
        done
    done
echo "IMAGES GENERATED"

declare -a gs_arr=(0.7 1 2 3)
for gs in "${gs_arr[@]}"; do 
    python3 scripts/utils/merge_images.py \
        --output_path="results/sd2_boxes/pothole_scenes_taomr_small_test_sd2_${gs}.png" \
        --images_folders \
        "custom_datasets/scenes/pothole_scenes" \
        "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_2" \
        "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_4" \
        "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_6" \
        "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_8" \
        "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_10" 
    done
echo "DONE RESULT IMAGE"


# echo "MOVING IMAGES"
# python3 scripts/utils/move_images.py \
#     --output_path="datasets/augmented/INSTSnowRoadDetection/test" \ 
#     --images_folder="tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_gs-1_ckpt-6"
# done

