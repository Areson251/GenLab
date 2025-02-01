## Moving images
```
python scripts/utils/move_images.py \
--output_path="datasets/augmented/INSTSnowRoadDetection/validation" \
--images_folder="tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_val_gs-1_ckpt-6" 
```

## Show dataset statistics
```
python scripts/utils/annotations_tools.py \
--annotation_path="datasets/augmented/INSTSnowRoadDetection/test/annotation.json" 
```