#!/bin/bash

cd /home/docker_diffdepth/diff_depth_new/datasets/augmented/YCOR_taomr_loss-umasked-1500/labels/train_objects_copy

for file in t_YCOR_*; do
    if [ -f "$file" ]; then
        if [[ $file != *_manhole.txt && $file != *_pothole.txt ]]; then
        # if [[ $file != *_manhole.png && $file != *_pothole.png ]]; then
            echo "Удаляю файл: $file"
            rm "$file"
        fi
    fi
done

echo "Готово! Оставлены только файлы, оканчивающиеся на manhole.png и pothole.png"