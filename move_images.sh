#!/bin/bash

SOURCE_DIR="tuning_exps/sd2_boxes/YCOR.augmented_gs-5_ckpt-taomr_9obj_mse_3000"
TARGET_DIR="datasets/augmented/YCOR_taomr_mse/images/val"

mkdir -p "$TARGET_DIR"

find "$SOURCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" \) -exec cp {} "$TARGET_DIR" \;

echo "Все изображения скопированы в $TARGET_DIR"