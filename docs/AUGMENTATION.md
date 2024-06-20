### DreamBooth-inpainting augmentation
To run augmentation use folowing sommand:
```
python augment_dataset.py \
--dataset_path="custom_datasets/INSTSnowRoadDetection" \
--prompts_path="prompts/pothole.txt" \
--masks_path="custom_datasets/target_masks" \
--output_path="augmented_dataset-pothole" \
--dreambooth_chkpt="model_output/db_pothole/checkpoint-1000" \
--padding="20"
```

### DreamBooth preprocessing
To prepare dreambooth output for using in script below use folowing sommand:
```
python scripts/utils/prepare_weights_dreambooth.py \
--model_path="model_output/db_inp_snowy_pothole_sd2" \
--ckpt="checkpoint-1000" 
```