import argparse
import os
import json
from tqdm import tqdm
from glob import glob
from PIL import Image
from pycocotools.coco import COCO

import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import resnet50
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import clip


clip_model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")


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


def crop_image_from_bbox(image, bbox):
    x, y, w, h = map(int, bbox)
    cropped = image.crop((x, y, x + w, y + h))
    return cropped


def calculate_clip_score(images, prompts):
    images = torch.stack([preprocess(img).to(next(clip_model.parameters()).device) for img in images])
    prompts = clip.tokenize(prompts).to(next(clip_model.parameters()).device)

    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        text_features = clip_model.encode_text(prompts)

    scores = (image_features @ text_features.T).squeeze().cpu().numpy()
    return float(np.mean(scores))


def calculate_ssim(gt_images, dt_images):
    scores = []
    skipped = 0
    for gt, dt in zip(gt_images, dt_images):
        gt = np.array(gt.convert('L'))
        dt = np.array(dt.convert('L'))

        min_size = min(gt.shape)
        win_size = min(11, min_size if min_size % 2 == 1 else min_size - 1)

        if win_size < 3:
            skipped+=1
            continue

        score = ssim(gt, dt, win_size=win_size, data_range=dt.max() - dt.min() or 1)
        scores.append(score)
    
    print(f"SSIM: skip {skipped} objects")
    return float(np.mean(scores))


def calculate_psnr(gt_images, dt_images):
    scores = []
    for gt, dt in zip(gt_images, dt_images):
        gt = np.array(gt)
        dt = np.array(dt)
        score = psnr(gt, dt, data_range=255)
        scores.append(score)
    return float(np.mean(scores))


def calculate_fid(gt_images, dt_images):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(normalize=True).to(device)

    def to_tensor(images):
        tensors = [preprocess(img).unsqueeze(0) for img in images]
        return torch.cat(tensors)

    with torch.no_grad():
        fid.update(to_tensor(gt_images).to(device), real=True)
        fid.update(to_tensor(dt_images).to(device), real=False)
    return float(fid.compute().item())


def calculate_lpips(gt_images, dt_images):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    scores = []
    for gt, dt in zip(gt_images, dt_images):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),                  # [0, 1]
            torchvision.transforms.Resize((224, 224)),           # или 256 если нужно
            torchvision.transforms.Normalize(mean=0.5, std=0.5), # [0, 1] -> [-1, 1]
        ])

        try:
            gt_tensor = transform(gt).unsqueeze(0).to(next(lpips.parameters()).device)
            dt_tensor = transform(dt).unsqueeze(0).to(next(lpips.parameters()).device)

            score = lpips(gt_tensor, dt_tensor).item()
            scores.append(score)
        except Exception as e:
            print(f"LPIPS error: {e}")
            continue

    return float(np.mean(scores)) if scores else 0.0


def calculate_metrics(args):
    id_to_fname, image_id_to_anns, categories = load_coco_annotations(args.gt_annotation_path)
    bboxes_prompts = extract_bboxes_and_prompts(image_id_to_anns, categories)

    gt_images = []
    dt_images = []
    prompts = []
    results = {}

    print("Preparing images...")
    for ann_id, image_id, bbox, prompt in tqdm(bboxes_prompts):
        gt_fname = id_to_fname.get(image_id)
        if not gt_fname:
            continue

        gt_path = os.path.join(args.gt_images_path, gt_fname)

        base_name = os.path.splitext(gt_fname)[0]
        fname = f"{base_name}_{image_id}_{ann_id}.png"
        dt_path = os.path.join(args.dt_images_path, fname)

        if not os.path.exists(gt_path):
            print(f"Missing files: {gt_path}")
            continue

        if not os.path.exists(dt_path):
            print(f"Missing files: {dt_path}")
            continue

        gt_image = Image.open(gt_path).convert('RGB')
        dt_image = Image.open(dt_path).convert('RGB')

        dt_image = dt_image.resize(gt_image.size, resample=Image.BILINEAR)
        
        gt_cropped = crop_image_from_bbox(gt_image, bbox)
        dt_cropped = crop_image_from_bbox(dt_image, bbox)

        gt_images.append(gt_cropped)
        dt_images.append(dt_cropped)
        prompts.append(prompt)

    print(f"Found {len(gt_images)} valid pairs for evaluation.")

    results["CLIP Score"] = calculate_clip_score(dt_images, prompts)
    results["SSIM"] = calculate_ssim(gt_images, dt_images)
    results["PSNR"] = calculate_psnr(gt_images, dt_images)
    results["FID"] = calculate_fid(gt_images, dt_images)
    results["LPIPS"] = calculate_lpips(gt_images, dt_images)

    print(f"\nMETRICS FOR {args.exp_name}")
    for metric, value in results.items():
        if metric in ["CLIP Score", "SSIM", "PSNR"]:
            arrow = "🔼"
        elif metric in ["FID", "LPIPS"]:
            arrow = "🔽"
        else:
            arrow = ""
        print(f"{arrow+metric+':':<10} {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt_images_path", type=str, required=True)
    parser.add_argument("--gt_images_path", type=str, required=True)
    parser.add_argument("--gt_annotation_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str)

    args = parser.parse_args()

    calculate_metrics(args)

print("ЭЩКЕРЕЕЕ")