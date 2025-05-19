from pycocotools.coco import COCO
import os

data_path = "datasets/original/TAOMR/val_objects_copy"
ann = COCO("datasets/original/TAOMR/val_9_objs_copy.json")

# Получаем все изображения в папке (только с поддерживаемыми расширениями)
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
image_files = set(os.listdir(data_path))
# print(image_files[0])

# Получаем все изображения, которые есть в аннотации
annotated_images = {img_info['file_name'] for img_info in ann.dataset['images']}  # COCO хранит ID как int, поэтому преобразуем в str

# Находим изображения, которых нет в аннотации
missing_in_ann = image_files - annotated_images

print(f"Всего изображений в папке: {len(image_files)}")
print(f"Изображений в аннотации: {len(annotated_images)}")
print(f"Изображений без аннотации: {len(missing_in_ann)}")

if missing_in_ann:
    print("\nФайлы без аннотации:")
    for img_name in missing_in_ann:
        print(img_name)
else:
    print("\nВсе изображения имеют аннотации.")