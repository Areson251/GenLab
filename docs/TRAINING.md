### PowerPaint tuning via LoRA-inpainting 
To run use folowing sommand:
```
accelerate launch --main_process_port=12547 scripts/lora/train_inpainting_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="datasets/original/COCO2014/images/train" \
  --annotation_path="datasets/original/COCO2014/annotations/instances_train2014.json" \
  --dataloader_num_workers=1 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --max_train_steps=15000 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="model_output/lora_COCO_1e-06" \
  --checkpointing_steps=2000 \
  --report_to="wandb" \
  --seed=1337
```

#### Prepare dataset
If you have VOC dataset format use folowwing command:

```
python scripts/utils/voc2coco.py --root_dir="datasets/original/VOC2012" 
```
