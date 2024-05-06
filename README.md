# GUI FOR DIFFUSION MODELS
Support Stable Diffusion v2, Stable Diffusion XL, Kandinsky 2.2

## Installation
python verison is 3.12
```
pip install -r requirements.txt
```

## Training
Also this instrument support training diffusion odels on custom datasets via **Textual inversion** and **DreamBooth-inpainting** using about ~3-5 images of targeted object

### Textual inversion
To run trainig use folowing sommand:
```
accelerate launch scripts/textual_inversion.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
--train_data_dir="custom_datasets/cat-avocado" \
--output_dir="model_output/exp_cat-avocado" \
--placeholder_token="<cat-avocado>" \
--initializer_token="toys" \
--report_to="wandb" \
--train_batch_size=1 \
--max_train_steps=6000 \
--validation_prompt="A <cat-avocado> on the beach" \
--num_validation_images=2 \
--checkpointing_steps=2000 \
--validation_steps=3000 

# you may add
--mixed_precision="bfp16" \
```

### DreamBooth-inpainting
To run trainig use folowing sommand:
```
accelerate launch scripts/train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting"  \
  --instance_data_dir="custom_datasets/pothole" \
  --output_dir="model_output/db_inp_exp2" \
  --instance_prompt="a photo of sks pothole" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=6000 \
  --checkpointing_steps=2000 \
  --report_to="wandb" 
  ```
