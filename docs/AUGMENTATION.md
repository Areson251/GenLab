### DreamBooth-inpainting augmentation
To run augmentation use folowing sommand:
```
python augment_dataset.py \
--dataset_path="datasets/original/INSTSnowRoadDetection" \
--prompts_path="prompts/pothole.txt" \
--masks_path="custom_datasets/target_masks" \
--output_path="datasets/augmented/INSTSnowRoadDetection_augmented-pothole-15-10k" \
--dreambooth_chkpt="model_output/db_inp_snowy_pothole_sd2/checkpoint-10000" \
--padding="20" \
--guidance_scale="15"
```

#### DreamBooth preprocessing
To prepare dreambooth output for using in script below use folowing sommand:
```
python scripts/utils/prepare_weights_dreambooth.py \
--model_path="model_output/db_inp_snowy_pothole_sd2" \
--ckpt="checkpoint-2000" 
```

### PowerPaint augmentation
To run augmentation use folowing sommand:
```
accelerate launch --main_process_port=12547  --num_processes=1 scripts/lora/power_paint_accelerate_entity.py \
--images_path="datasets/original/entity_01_11580/images" \
--json_path="datasets/original/entity_01_11580/train_01_edit.json" \
--output_path="datasets/augmented/lora_inp_tuning/b64_gas4_lr1e-5/scale2/entity_500_water" \
--lora_weights="model_output/lora_b64_gas4_lr1e-5/500_steps/pytorch_lora_weights.safetensors"
```

--main_process_port=12547 
