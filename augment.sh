#!/bin/bash

python3 scripts/generate_images_annots_entity.py \
    --json_path="custom_datasets/scenes/pothole_scenes.json" \
    --hq=False \
    --output_json="custom_datasets/scenes/pothole_scenes_edit.json" \
    --task="text-guided"

echo "CONVERT ANNOTATIONS"

declare -a gs_arr=(0.7 2 5)
for gs in "${gs_arr[@]}"; do 
    for ((ckpt_iter=2000; ckpt_iter<=10000; ckpt_iter+=2000)); do 
        accelerate launch --main_process_port=12363  --num_processes=1 scripts/power_paint_accelerate_entity.py \
        --images_path="custom_datasets/scenes/pothole_scenes" \
        --json_path="custom_datasets/scenes/pothole_scenes_edit.json" \
        --output_path="datasets/augmented/lora_pp_pothole_full/pothole_scenes_edit/full-${ckpt_iter}-${gs}" \
        --lora_weights="model_output/lora_pothole-full_pp/checkpoint-${ckpt_iter}/pytorch_lora_weights.safetensors" \
        --guidance_scale=${gs}
    done
done

echo "IMAGES GENERATED"

declare -a gs_arr=(0.7 2 5)
for gs in "${gs_arr[@]}"; do 
    python3 scripts/utils/merge_images.py \
        --output_path="results/pp_boxes/pothole_full_pp_${gs}.png" \
        --images_folders \
        "custom_datasets/scenes/pothole_scenes" \
        "tuning_exps/pp_boxes/full-2000-${gs}/" \
        "tuning_exps/pp_boxes/full-4000-${gs}/" \
        "tuning_exps/pp_boxes/full-6000-${gs}/" \
        "tuning_exps/pp_boxes/full-8000-${gs}/" \
        "tuning_exps/pp_boxes/full-10000-${gs}/"
done

echo "DONE RESULT IMAGE"

# gs=2
# accelerate launch --main_process_port=12360  --num_processes=1 scripts/power_paint_accelerate_entity.py \
#         --images_path="custom_datasets/scenes/pothole_scenes" \
#         --json_path="custom_datasets/scenes/pothole_scenes_edit.json" \
#         --output_path="datasets/augmented/lora_pp_pothole_full/pothole_scenes_edit/full-2000-${gs}" \
#         --lora_weights="model_output/lora_pothole-full_pp/checkpoint-2000/pytorch_lora_weights.safetensors" \
#         --guidance_scale=${gs}
