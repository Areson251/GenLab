<!-- ### PowerPaint tuning via LoRA-inpainting  -->
### StableDiffusion tuning via LoRA-inpainting 
This script use COCO annotation format. *--instance_data_dir* is folder with your images and *--annotation_path* is path to annotation in COCO format.

To run training use folowing sommand:
```
accelerate launch scripts/lora-inpainting/train_inpainting_lora_pp.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir=<DATASET FOLDER> \
  --annotation_path=<COCO ANNOTATION> \
  --dataloader_num_workers=1 \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=32 \
  --max_train_steps=15000 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=<LORA OUTPUT PATH> \
  --checkpointing_steps=2000 \
  --report_to="wandb" \
  --seed=1337
```

Example:
```
accelerate launch --main_process_port=12550 scripts/lora/train_inpainting_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
  --instance_data_dir="datasets/original/TAOMR/train_objects" \
  --annotation_path="datasets/original/TAOMR/train_objects.json" \
  --dataloader_num_workers=1 \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --output_dir="model_output/lora_taomr_sd2" \
  --max_train_steps=10000 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=2000 \
  --report_to="wandb" \
  --loss="custom" \
  --seed=1337
```

### StableDiffusion tuning via CUSTOM LOSS 
```
accelerate launch --main_process_port=12553 scripts/lora/train_inpainting_lora_loss.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
  --instance_data_dir="datasets/original/TAOMR/train_objects" \
  --annotation_path="datasets/original/TAOMR/train_objects.json" \
  --validation_data_dir="datasets/original/TAOMR/val_objects" \
  --annotation_val_path="datasets/original/TAOMR/val_objects.json" \
  --num_validation_images="1" \
  --validation_epochs="1" \
  --resolution="512" \
  --train_batch_size="4" \
  --gradient_accumulation_steps="4" \
  --output_dir="model_output/lora_custom_taomr" \
  --max_train_steps="10000" \
  --learning_rate="1e-06" \
  --max_grad_norm="1" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps="0" \
  --checkpointing_steps="2000" \
  --report_to="wandb" \
  --loss="custom" \
  --seed="1337" 
```






accelerate launch --main_process_port=12560 scripts/lora/train_inpainting_lora_loss.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
  --instance_data_dir="datasets/original/TAOMR/train" \
  --annotation_path="datasets/original/TAOMR/train_9_objs.json" \
  --validation_data_dir="datasets/original/TAOMR/val" \
  --annotation_val_path="datasets/original/TAOMR/val_9_objs.json" \
  --num_validation_images="1" \
  --validation_epochs="1" \
  --resolution="512" \
  --train_batch_size="1" \
  --gradient_accumulation_steps="4" \
  --output_dir="model_output/lora_custom_taomr_debug" \
  --max_train_steps="10" \
  --learning_rate="1e-06" \
  --max_grad_norm="1" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps="0" \
  --checkpointing_steps="2" \
  --report_to="wandb" \
  --loss="custom" \
  --loss_alpha="0.7" \
  --loss_beta="0.3" \
  --seed="1337" 











  


##### TODO: add experiments names in wandb 
  --scale_lr=True \
  --report_to="wandb" \


#### Prepare dataset
If you have VOC dataset format use folowwing command:

```
python scripts/utils/voc2coco.py --root_dir="datasets/original/VOC2012" 
```
