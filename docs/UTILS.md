## Moving images
```
python scripts/utils/move_images.py \
--output_path="datasets/augmented/INSTSnowRoadDetection/train" \
--images_folder="tuning_exps/sd2_boxes/INSTSnowRoadDetection.augmented_train_gs-1_ckpt-6" 
```

## Convert masks to [0, 255]
If you use masks in images format it should be only [0, 255] unique pixels. If the placement targeting mask has a label not equal to 255, use this script to convert it: 
```
python scripts/utils/convert_masks.py --input_folder=datasets/original/YCOR/val/masks --label=3 --output_folder=datasets/original/YCOR/val/masks_road
```

## Show dataset statistics
```
python scripts/utils/annotations_tools.py \
--annotation_path="datasets/augmented/INSTSnowRoadDetection/train/annotation.json" 
```

## Extract certain cats
Change ALLOWDED_CATEGORIES that you want to see in your new annotation:
```
python scripts/utils/annotations_tools.py \
--annotation_path="datasets/original/TAOMR/train.json" \
--new_annotation_path="datasets/original/TAOMR/train_objects.json"
```

## Extract certain images to folder
Move selected images due to annotation:
```
python scripts/utils/filter_images.py \
--images_path="datasets/original/TAOMR/train" \
--annotation_path="datasets/original/TAOMR/train_objects.json" \
--output_path="datasets/original/TAOMR/train_objects" 
```