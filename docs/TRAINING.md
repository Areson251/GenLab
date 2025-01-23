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
accelerate launch --main_process_port=12549 scripts/lora/train_inpainting_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
  --instance_data_dir="datasets/original/pothole_full/images" \
  --annotation_path="datasets/original/pothole_full/pothole_full.json" \
  --dataloader_num_workers=1 \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --output_dir="model_output/lora_pothole-full2_sd2" \
  --max_train_steps=15000 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=2000 \
  --report_to="wandb" \
  --seed=1337
```


accelerate launch scripts/lora/train_inpainting_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
  --instance_data_dir="custom_datasets/objects/pothole_small" \
  --annotation_path="custom_datasets/objects/pothole_small.json" \
  --dataloader_num_workers=1 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --output_dir="model_output/lora_pothole-full2_sd2" \
  --max_train_steps=15000 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=2000 \
  --loss="CUSTOM" \
  --seed=1337
  --report_to="wandb" \

#### Prepare dataset
If you have VOC dataset format use folowwing command:

```
python scripts/utils/voc2coco.py --root_dir="datasets/original/VOC2012" 
```
