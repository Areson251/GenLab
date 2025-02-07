## Moving images
```
python scripts/utils/move_images.py \
--output_path="datasets/augmented/INSTSnowRoadDetection/train" \
--images_folder="tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_train_gs-1_ckpt-6" 
```

## Show dataset statistics
```
python scripts/utils/annotations_tools.py \
--annotation_path="datasets/augmented/INSTSnowRoadDetection/train/annotation.json" 
```