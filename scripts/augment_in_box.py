import argparse
import os
import json
from glob import glob
from PIL import Image
from pycocotools.coco import COCO

import numpy as np
import cv2
import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import resnet50

from stable_diffusion import StableDiffusionModel as SDPipe



def load_coco_annotations(annotation_path):
    data = COCO(annotation_path)
    dataset = data.dataset

    image_id_to_anns = {}
    for ann in dataset['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_anns:
            image_id_to_anns[image_id] = []
        image_id_to_anns[image_id].append(ann)

    id_to_fname = {img['id']: img['file_name'] for img in dataset['images']}
    categories = {cat['id']: cat['name'] for cat in dataset['categories']}
    
    return id_to_fname, image_id_to_anns, categories


def extract_bboxes_and_prompts(image_id_to_anns, categories):
    """
    Возвращает список кортежей (image_id, bbox, prompt)
    """
    bboxes_prompts = []
    for image_id, anns in image_id_to_anns.items():
        for ann in anns:
            ann_id = ann['id']
            bbox = ann['bbox']
            category_id = ann['category_id']
            prompt = categories[category_id]
            bboxes_prompts.append((ann_id, image_id, bbox, prompt))
    return bboxes_prompts
    

def augment_images(args):
    id_to_fname, image_id_to_anns, categories = load_coco_annotations(args.gt_annotation_path)
    bboxes_prompts = extract_bboxes_and_prompts(image_id_to_anns, categories)

    diffusion_pipe = SDPipe(pretrained=args.sd_chkpt, device=args.device)
    diffusion_pipe.load_lora(args.lora_chkpt)

    os.makedirs(args.output_path, exist_ok=True)

    for ann_id, image_id, bbox, prompt in bboxes_prompts:
        gt_fname = id_to_fname.get(image_id)
        if not gt_fname:
            continue

        gt_path = os.path.join(args.gt_images_path, gt_fname)

        try:
            image = Image.open(gt_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {gt_path}: {e}")
            continue

        # Create mask
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)

        x_min, y_min, box_width, box_height = map(int, bbox)
        x_max = x_min + box_width
        y_max = y_min + box_height

        # Clip bounding box to image dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color=255, thickness=-1)
        masked_image = Image.fromarray(mask)

        detailed_prompt = f"A highly detailed and realistic depiction of {prompt}, featuring sharp details, natural lighting, and a clean, well-balanced composition."

        # inpainting
        augmented_image = diffusion_pipe(
            image, masked_image, 
            detailed_prompt, None, image.size[0], image.size[1],
            args.iter_number, args.guidance_scale
        )

        # Generate output filename
        base_name = os.path.splitext(gt_fname)[0]
        output_fname = f"{base_name}_{image_id}_{ann_id}.png"
        output_path = os.path.join(args.output_path, output_fname)

        # output_fname_input = f"{base_name}_{image_id}_{ann_id}_input.png"
        # output_fname_masked = f"{base_name}_{image_id}_{ann_id}_masked.png"
        # output_path_input = os.path.join(args.output_path, output_fname_input)
        # output_path_masked = os.path.join(args.output_path, output_fname_masked)

        # Save the result
        augmented_image.save(output_path)
        # image.save(output_path_input)
        # masked_image.save(output_path_masked)
        # print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--gt_images_path", type=str, required=True)
    parser.add_argument("--gt_annotation_path", type=str, required=True)
    parser.add_argument("--sd_chkpt", type=str, required=True)
    parser.add_argument("--lora_chkpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iter_number", type=int, default="20")
    parser.add_argument("--guidance_scale", type=float, default="0.7")
    parser.add_argument("--seed", type=float, default="0")

    args = parser.parse_args()

    augment_images(args)

print("ем а што проесходет")