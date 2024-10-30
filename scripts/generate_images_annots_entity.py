import json
import argparse
import numpy as np

from collections import defaultdict
from pycocotools import mask as maskUtils


def main(args):
    to_json = {}

    with open(args.json_path, 'r') as jsonFile:
        images = json.load(jsonFile)

    image2annots = defaultdict(list)

    for annot in images['annotations']:
        image2annots[annot['image_id']].append(annot)

    for image in images['images']:
        image_id = image['id']
        h, w = image['height'], image['width']
        if True:
        # if h > 1000 and w > 1000 or not args.hq:
            file_name = image['file_name']

            image_annots = []
            for annot in image2annots[image_id]:
                if True:
                # if annot['area'] * 25 >= h * w and annot['area'] * 2 <= h * w:
                    # use bbox as segmentation

                    x, y, width, height = annot['bbox']  
                    x, y, width, height = int(x), int(y), int(width), int(height)
                    annot['bbox'] = [x, y, width, height]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    mask[int(y):int(y+height), int(x):int(x+width)] = 1
                    rle = maskUtils.encode(np.asfortranarray(mask)) 
                    annot['segmentation'] = {
                        'size': [h, w], 
                        'counts': rle['counts'].decode('ascii') 
                    }
                    
                    # rle = annot['segmentation']
                    # if isinstance(rle['counts'], list):
                    #     rle['counts'] = maskUtils.frPyObjects(rle, rle['size'][0], rle['size'][1])['counts']
                    # rle_decoded = maskUtils.decode(rle) 
                    # rle_encoded = maskUtils.encode(rle_decoded)
                    # annot['segmentation'] = {
                    #     'size': rle_encoded['size'], 
                    #     'counts': rle_encoded['counts'].decode('ascii') 
                    # }

                    image_annots.append({
                        'segmentation': annot['segmentation'],
                        'id': annot['id'],
                        'bbox': annot['bbox'],
                        'category_id': annot['category_id'],
                        'area': annot['area']
                    })

            for annot in image_annots:
                for category in images['categories']:
                    if annot['category_id'] == category['id']:
                        annot['category_name'] = category['name']
                        annot['task'] = args.task
                        annot['new_object'] = category['name']

            if not image_annots:
                continue

            to_json[image_id] = {
                'height': h,
                'width': w,
                'file_name': file_name,
                'annots': image_annots
            }

    with open(args.output_json, 'w') as jsonFile:
        json.dump(to_json, jsonFile)        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--hq", type=bool, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
