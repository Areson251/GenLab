### PowerPaint tuning via LoRA-inpainting 
To run use folowing sommand:
```
accelerate launch --mixed_precision="fp16"  scripts/lora/train_inpainting_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="custom_datasets/snowy_pothole" \
  --dataloader_num_workers=1 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="model_output/lora_snowy-pothole_sd15" \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="snowy pothole." \
  --seed=1337
```

accelerate launch --mixed_precision="fp16"  scripts/lora/train_inpainting_lora.py \
accelerate launch scripts/lora/train_inpainting_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="custom_datasets/snowy_pothole" \
  --annotation_path="custom_datasets/snowy_pothole.json" \
  --dataloader_num_workers=1 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="model_output/lora_snowy-pothole_sd15" \
  --checkpointing_steps=1 \
  --validation_prompt="snowy_pothole" \
  --seed=1337

  --report_to=wandb \