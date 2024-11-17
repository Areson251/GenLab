#!/bin/bash

# python3 scripts/generate_images_annots_entity.py \
#     --json_path="custom_datasets/scenes/pothole_scenes.json" \
#     --hq=False \
#     --output_json="custom_datasets/scenes/pothole_scenes_edit.json" \
#     --task="text-guided"

# echo "CONVERT ANNOTATIONS"

python augment_dataset.py \
            --images_path="otchet/imgs" \
            --annotation_path="otchet/anns" \
            --masks_path="custom_datasets/target_masks" \
            --prompts_path="prompts/pothole.txt" \
            --output_path="tuning_exps/test" \
            --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
            --lora_chkpt="model_output/lora_pothole-full2_sd2/checkpoint-2000" \
            --padding=20 \
            --guidance_scale=0.7 \
            --seed=0

# declare -a gs_arr=(0.7 2 5 7 10 13 15)
# for gs in "${gs_arr[@]}"; do 
#     for ((ckpt_iter=2000; ckpt_iter<=14000; ckpt_iter+=2000)); do 
#         python augment_dataset.py \
#             --images_path="custom_datasets/scenes/pothole_scenes" \
#             --annotation_path="custom_datasets/scenes/pothole_scenes_road.json" \
#             --masks_path="custom_datasets/target_masks" \
#             --prompts_path="prompts/pothole.txt" \
#             --output_path="tuning_exps/sd2_masks/full-${ckpt_iter}-${gs}" \
#             --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
#             --lora_chkpt="model_output/lora_pothole-full2_sd2/checkpoint-${ckpt_iter}" \
#             --padding=20 \
#             --guidance_scale=${gs}
#     done

#     python augment_dataset.py \
#         --images_path="custom_datasets/scenes/pothole_scenes" \
#         --annotation_path="custom_datasets/scenes/pothole_scenes_road.json" \
#         --masks_path="custom_datasets/target_masks" \
#         --prompts_path="prompts/pothole.txt" \
#         --output_path="tuning_exps/sd2_masks/full-${gs}" \
#         --sd_chkpt="stabilityai/stable-diffusion-2-inpainting" \
#         --lora_chkpt="model_output/lora_pothole-full2_sd2" \
#         --padding=20 \
#         --guidance_scale=${gs}
# done

# echo "IMAGES GENERATED"

# declare -a gs_arr=(0.7 2 5 7 10 13 15)
# for gs in "${gs_arr[@]}"; do 
#     python3 scripts/utils/merge_images.py \
#         --output_path="results/sd2_masks/pothole_full_sd2_${gs}.png" \
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

# gs=2
# accelerate launch --main_process_port=12360  --num_processes=1 scripts/power_paint_accelerate_entity.py \
#         --images_path="custom_datasets/scenes/pothole_scenes" \
#         --json_path="custom_datasets/scenes/pothole_scenes_edit.json" \
#         --output_path="datasets/augmented/lora_pp_pothole_full/pothole_scenes_edit/full-2000-${gs}" \
#         --lora_chkpt="model_output/lora_pothole-full_pp/checkpoint-2000/pytorch_lora_chkpt.safetensors" \
#         --guidance_scale=${gs}
