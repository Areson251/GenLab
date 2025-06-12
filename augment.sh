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


# python augment_dataset.py \
#         --images_path="datasets/original/YCOR/val/images" \
#         --annotation_path="datasets/original/YCOR/val/masks_road" \
#         --masks_path="custom_datasets/target_masks" \
#         --prompts_path="prompts/taomr_objects_augm.txt" \
#         --output_path="tuning_exps/sd2_boxes/YCOR.augmented_gs-5_ckpt-taomr_9obj_loss-umasked_1500" \
#         --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
#         --lora_chkpt="model_output/lora_loss-umasked_taomr_9_objs/checkpoint-1500" \
#         --padding=20 \
#         --guidance_scale=5 \
#         --device="cuda:1" \
#         --use_crops=True \
#         --seed=0 

python augment_dataset.py \
        --images_path="datasets/original/YCOR/val/images" \
        --annotation_path="datasets/original/YCOR/val/masks_road" \
        --masks_path="custom_datasets/target_masks" \
        --prompts_path="prompts/taomr_objects.txt" \
        --output_path="tuning_exps/sd2_boxes/YCOR_val.augmented_gs-5_ckpt-taomr_9obj_mse_4000" \
        --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
        --lora_chkpt="model_output/lora_mse_taomr_9_objs/checkpoint-4000" \
        --padding=20 \
        --guidance_scale=5 \
        --device="cuda:0" \
        --use_crops=True \
        --seed=0 



python scripts/utils/count_gen_metrics.py \
        --dt_images_path="tuning_exps/sd2_boxes/taomr.augmented_gs-5_ckpt-taomr_9obj_loss-mse_4000" \
        --gt_images_path="datasets/original/TAOMR/val_objects" \
        --gt_annotation_path="datasets/original/TAOMR/val_9_objs_copy.json" \
        --exp_name="MSE" 

python scripts/utils/count_gen_metrics.py \
        --dt_images_path="tuning_exps/sd2_boxes/taomr.augmented_gs-5_ckpt-taomr_9obj_loss-umasked_4000" \
        --gt_images_path="datasets/original/TAOMR/val_objects" \
        --gt_annotation_path="datasets/original/TAOMR/val_9_objs_copy.json" \
        --exp_name="LOSS-UMASKED" 


        
 
# declare -a ckpt_arr=(8 10)
# declare -a ckpt_arr=(5 10 15 20 25 30 35 40)
# for ckpt in "${ckpt_arr[@]}"; do
#     echo "START ckpt ${ckpt} GENERATION"

#     declare -a gs_arr=(0.7 3 5)
#     for gs in "${gs_arr[@]}"; do  
#         echo "START GS ${gs} GENERATION"
#         python augment_dataset.py \
#                 --images_path="custom_datasets/scenes/pothole_scenes" \
#                 --annotation_path="custom_datasets/scenes/pothole_scenes_road.json" \
#                 --masks_path="custom_datasets/target_masks" \
#                 --prompts_path="prompts/taomr_objects.txt" \
#                 --output_path="tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_9obj_loss2_notcrops_${ckpt}" \
#                 --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
#                 --lora_chkpt="model_output/lora_loss2_taomr_9_objs/checkpoint-$((ckpt*100))" \
#                 --padding=20 \
#                 --guidance_scale=${gs} \
#                 --device="cuda:1" \
#                 --use_crops=True \
#                 --use_bboxes=False \
#                 --seed=0 
#         done
#     done
# echo "IMAGES GENERATED"


# declare -a ckpt_arr=(5 10 15 20 25 30 35 40)
# for ckpt in "${ckpt_arr[@]}"; do
#     echo "START ckpt ${ckpt} GENERATION"

#     declare -a gs_arr=(0.7 3 5)
#     for gs in "${gs_arr[@]}"; do  
#         echo "START GS ${gs} GENERATION"
#         python augment_dataset.py \
#                 --images_path="custom_datasets/scenes/pothole_scenes" \
#                 --annotation_path="custom_datasets/scenes/pothole_scenes_road.json" \
#                 --masks_path="custom_datasets/target_masks" \
#                 --prompts_path="prompts/taomr_objects.txt" \
#                 --output_path="tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_9obj_custom_notcrops_${ckpt}" \
#                 --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
#                 --lora_chkpt="model_output/lora_custom_taomr_9_objs_fixval/checkpoint-$((ckpt*100))" \
#                 --padding=0 \
#                 --guidance_scale=${gs} \
#                 --device="cuda:1" \
#                 --seed=0 
#         done
#     done
# echo "IMAGES GENERATED"


# declare -a ckpt_arr=(5 10 15 20 25 30 35 40)
# for ckpt in "${ckpt_arr[@]}"; do
#     echo "START ckpt ${ckpt} GENERATION"

#     declare -a gs_arr=(0.7 3 5)
#     for gs in "${gs_arr[@]}"; do  
#         echo "START GS ${gs} GENERATION"
#         python augment_dataset.py \
#                 --images_path="custom_datasets/scenes/pothole_scenes" \
#                 --annotation_path="custom_datasets/scenes/pothole_scenes_road.json" \
#                 --masks_path="custom_datasets/target_masks" \
#                 --prompts_path="prompts/taomr_objects.txt" \
#                 --output_path="tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_9obj_mse_${ckpt}" \
#                 --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
#                 --lora_chkpt="model_output/lora_mse_taomr_9_objs_fixval/checkpoint-$((ckpt*100))" \
#                 --padding=20 \
#                 --guidance_scale=${gs} \
#                 --device="cuda:1" \
#                 --seed=0 
#         done
#     done
# echo "IMAGES GENERATED"

# declare -a gs_arr=(0.7 1 2 3)
# for gs in "${gs_arr[@]}"; do 
#     python3 scripts/utils/merge_images.py \
#         --output_path="results/sd2_boxes/pothole_scenes_taomr_small_custom_${gs}.png" \
#         --images_folders \
#         "custom_datasets/scenes/pothole_scenes" \
#         "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_custom_2" \
#         "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_custom_4" \
#         "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_custom_6" \
#         "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_custom_8" \
#         "tuning_exps/sd2_boxes/pothole_scenes.augmented_gs-${gs}_ckpt-taomr_small_custom_10" 
#     done
# echo "DONE RESULT IMAGE"


# echo "MOVING IMAGES"
# python3 scripts/utils/move_images.py \
#     --output_path="datasets/augmented/INSTSnowRoadDetection/test" \ 
#     --images_folder="tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_gs-1_ckpt-6"
# done

