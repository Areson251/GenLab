import os
import json
import cv2
from tqdm import tqdm
from PIL import Image


def coco_to_yolo(images_dir, labels_json, output_dir, class_names_file=None, resize_to=None):
    """
    Конвертирует аннотации из формата COCO в формат YOLO.
    
    Args:
        images_dir (str): Путь к директории с изображениями.
        labels_json (str): Путь к JSON файлу с аннотациями COCO.
        output_dir (str): Путь к директории для сохранения YOLO аннотаций.
        class_names_file (str, optional): Путь к файлу с именами классов. Если None, имена берутся из COCO.
        resize_to (tuple, optional): Размер (width, height) для ресайза изображений. Если None, ресайз не выполняется.
    """

    os.makedirs(output_dir, exist_ok=True)

    with open(labels_json, 'r') as f:
        coco_data = json.load(f)
    
    if class_names_file:
        with open(class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        category_id_to_class_idx = {}
        for category in coco_data['categories']:
            category_name = category['name']
            if category_name in class_names:
                category_id_to_class_idx[category['id']] = class_names.index(category_name)
            else:
                print(f"Warning: Category '{category_name}' from COCO not found in class_names file")
    else:
        class_names = [category['name'] for category in sorted(coco_data['categories'], key=lambda x: x['id'])]
        category_id_to_class_idx = {category['id']: idx for idx, category in enumerate(sorted(coco_data['categories'], key=lambda x: x['id']))}

    image_id_to_info = {image['id']: image for image in coco_data['images']}
    
    image_id_to_annotations = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)

    for image_id, annotations in tqdm(image_id_to_annotations.items()):
        image_info = image_id_to_info[image_id]
        image_file = image_info['file_name']
        image_path = os.path.join(images_dir, image_file)
        
        if resize_to:
            with Image.open(image_path) as img:
                orig_width, orig_height = img.size
                new_width, new_height = resize_to
        else:
            orig_width, orig_height = image_info['width'], image_info['height']
            new_width, new_height = orig_width, orig_height
        
        txt_file = os.path.splitext(image_file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_file)
        
        with open(txt_path, 'w') as f:
            for ann in annotations:
                if ann['category_id'] not in category_id_to_class_idx:
                    continue
                
                class_idx = category_id_to_class_idx[ann['category_id']]

                x_min, y_min, width, height = ann['bbox']
                
                x_min = max(0, min(x_min, orig_width - 1))
                y_min = max(0, min(y_min, orig_height - 1))
                width = max(1, min(width, orig_width - x_min))
                height = max(1, min(height, orig_height - y_min))
                
                x_center = (x_min + width / 2) / orig_width
                y_center = (y_min + height / 2) / orig_height
                norm_width = width / orig_width
                norm_height = height / orig_height
                
                if resize_to:
                    scale_x = new_width / orig_width
                    scale_y = new_height / orig_height
                    x_center *= scale_x
                    y_center *= scale_y
                    norm_width *= scale_x
                    norm_height *= scale_y

                    x_center /= new_width
                    y_center /= new_height
                    norm_width /= new_width
                    norm_height /= new_height
                
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"Конвертация завершена. Аннотации сохранены в {output_dir}")
    print(f"Всего классов: {len(class_names)}")
    print("Имена классов:", class_names)


# images_dir = '/home/docker_diffdepth/diff_depth_new/datasets/augmented/YCOR_taomr_mse/images/val'
# labels_json = '/home/docker_diffdepth/diff_depth_new/tuning_exps/sd2_boxes/YCOR.augmented_gs-5_ckpt-taomr_9obj_mse_3000/annotation.json'
# # class_names_file = '/home/nlmk/Izmesteva/vesovye_dataset/obj.names'  # optional
# output_dir = '/home/docker_diffdepth/diff_depth_new/datasets/augmented/YCOR_taomr_mse/labels/val'


images_dir = '/home/docker_diffdepth/diff_depth_new/datasets/augmented/YCOR_taomr_mse/images/val'
labels_json = '/home/docker_diffdepth/diff_depth_new/tuning_exps/sd2_boxes/YCOR.augmented_gs-5_ckpt-taomr_9obj_mse_3000/annotation.json'
class_names_file = '/home/docker_diffdepth/diff_depth_new/datasets/original/TAOMR/obj.names'  # optional
output_dir = '/home/docker_diffdepth/diff_depth_new/datasets/augmented/YCOR_taomr_mse/labels/val'


coco_to_yolo(
    images_dir=images_dir,
    labels_json=labels_json,
    output_dir=output_dir,
    class_names_file=class_names_file,
    # resize_to=(224, 224),
)


