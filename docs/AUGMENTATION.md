### DreamBooth-inpainting augmentation
To run augmentation use folowing sommand:
```
python augment_dataset.py \
--dataset_path="custom_datasets/INSTSnowRoadDetection" \
--prompts_path="prompts/pothole.txt" \
--masks_path="custom_datasets/INSTSnowRoadDetection/target_masks" \
--output_path="augmented_dataset" \
--dreambooth_chkpt="model_output/db_pothole/checkpoint-1000" \
--padding="20"
```