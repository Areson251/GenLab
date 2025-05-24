#!/bin/bash

SOURCE_DIR="tuning_exps/sd2_boxes/YCOR.augmented_gs-5_ckpt-taomr_9obj_loss-umasked_1500"
TARGET_DIR="datasets/augmented/YCOR_taomr_loss-umasked-1500/images/train_objects_copy"

mkdir -p "$TARGET_DIR"

find "$SOURCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" \) -exec cp {} "$TARGET_DIR" \;

echo "Все изображения скопированы в $TARGET_DIR"


# IMAGES_DIR="datasets/augmented/YCOR_taomr/images/train_objects_copy"
# LABELS_DIR="datasets/augmented/YCOR_taomr/labels/train_objects_copy"

# if [ ! -d "$IMAGES_DIR" ]; then
#   echo "Папка $IMAGES_DIR не существует!"
#   exit 1
# fi

# mkdir -p "$LABELS_DIR"

# for img in "$IMAGES_DIR"/*.{jpg,jpeg,png,gif,bmp}; do
#   if [ -f "$img" ]; then
#     filename=$(basename -- "$img")
#     name_noext="${filename%.*}"
#     touch "$LABELS_DIR/${name_noext}.txt"
#   fi
# done

# echo "Готово! Пустые текстовые файлы созданы в $LABELS_DIR."