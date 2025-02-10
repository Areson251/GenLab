import os
import sys
import tqdm
import json
import random 
import argparse
import numpy as np
from PIL import Image
from itertools import groupby
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


THRESHOLD = 1000
ALLOWDED_CATEGORIES = {
    8: "bump",
    15: "curb",
    17: "rail_track",
    19: "manhole",
    7: "pit",
    6: "puddle",
    1: "firehose",
    2: "hose",
    3: "wire",
    20: "catch_basin",
    5: "poop",
    }

CATS_PRIORITY = {
    'firehose': 1,
    'hose': 2,
    'wire': 3,
    'rope': 4,
    'poop': 5,
    'manhole': 6,
    'catch_basin': 7,
    'pit': 8,
    'bump': 9,
    'curb': 10,
    'puddle': 11,
    'vegetation': 12,
    'sand': 13, 
    'terrain': 14,
    'sidewalk': 15,
    'pedestrian_area': 16,
    'rail_track': 17,
    'grass': 18,
    'bike_lane': 19,
    'road': 20,
}

CATS_IDS_ORIG = {
    'firehose': 1,
    'hose': 2,
    'wire': 3,
    'rope': 4,
    'poop': 5,
    'puddle': 6,
    'pit': 7,
    'bump': 8,
    'curb': 9,
    'rail_track': 10,
    'sand': 11, 
    'manhole': 12,
    'catch_basin': 13,
    'sidewalk': 14,
    'pedestrian_area': 15,
    'bike_lane': 16,
    'grass': 17,
    'terrain': 18,
    'vegetation': 19,
    'road': 20,
}

IMG_IDS = [
    [(1, 107)],
    [(1, 300)],
]

class Cleaner():
    def __init__(self, annotation_pth, new_annotation_path=None) -> None:
        self.annotation_pth = annotation_pth

        self.annotations_data = None
        self.new_annotations_data = []
        self.new_cats_data = []
        self.new_imgs_data = []
        self.coco = None
        self.data = None
        self.threshold = THRESHOLD
        self.new_annotation_path = new_annotation_path

        self.new_data = {
                'licenses': [{'name': '', 'id': 0, 'url': ''}], 
                'info': {'contributor': '', 'date_created': '', 
                        'description': '', 'url': '', 'version': '', 'year': ''}, 
                'categories': [],
                'images': [],
                'annotations': []
            }

        self.load_annotation()
        self.load_annotation_COCO()

    def load_annotation(self):
        with open(self.annotation_pth, 'r') as file:
            self.data = json.load(file)
        self.annotations_data = self.data['annotations']

    def load_annotation_COCO(self):
        self.coco = COCO(self.annotation_pth)

    def clean(self):
        for annotation in tqdm.tqdm(self.annotations_data):
            if annotation['area'] < self.threshold and annotation['category_id'] in ALLOWDED_CATEGORIES:
                continue
            self.new_annotations_data.append(annotation)
        
        self.set_up_data()

        print('NEW ANNOTATIONS COUNT: ', len(self.new_data['annotations']))
        self.save_annotations()

        self.data = self.new_data
        self.new_data = {}
        self.new_annotations_data = []
        self.new_cats_data = []

    def extract_certain_cats(self):
        for annotation in tqdm.tqdm(self.annotations_data):
            if annotation['category_id'] not in ALLOWDED_CATEGORIES:
                continue
            category_info = self.coco.loadCats(ids=[annotation['category_id']])[0]
            img_info = self.coco.loadImgs(ids=[annotation['image_id']])[0]

            if category_info not in self.new_cats_data:
                self.new_cats_data.append(category_info)

            if img_info not in self.new_imgs_data:
                self.new_imgs_data.append(img_info)

            self.new_annotations_data.append(annotation)
        
        self.set_up_data()

        print('NEW ANNOTATIONS COUNT: ', len(self.new_data['annotations']))
        self.save_annotations()

        self.new_data = {}
        self.new_annotations_data = []

    def concatenate_masks(self):
        print('ANNOTATIONS COUNT: ', len(self.data['annotations']))
        self.img_ids = [info['id'] for info in self.coco.dataset['images']]
        for img_id in tqdm.tqdm(self.img_ids):
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            
            if len(anns) == 0:
                continue

            height, width = anns[0]['segmentation']['size']

            changed_cats, concatenated_masks, new_anns_ids = {}, [], []
            for idx, x in enumerate(ALLOWDED_CATEGORIES):
                changed_cats[x] = 0 
                new_mask = np.zeros((height, width), dtype='uint8')
                concatenated_masks.append(new_mask)
                new_anns_ids.append(0)

            for ann in anns:
                if ann['category_id'] not in ALLOWDED_CATEGORIES:
                    self.new_annotations_data.append(ann)
                    continue

                changed_cats[ann['category_id']] += 1
                cat_idx = list(ALLOWDED_CATEGORIES.keys()).index(ann['category_id'])

                if not new_anns_ids[cat_idx]:
                    new_anns_ids[cat_idx] = ann['id']

                old_mask = self.rle2mask(ann['segmentation']['counts'], (width, height)) 
                new_mask = np.where(old_mask, old_mask, concatenated_masks[cat_idx])
                concatenated_masks[cat_idx] = new_mask

            for changed in changed_cats:
                if not changed_cats[changed]:
                    continue

                cat_idx = list(ALLOWDED_CATEGORIES.keys()).index(changed)
                vals, cnts = np.unique(concatenated_masks[cat_idx], return_counts=True)

                if vals.size == 2:
                    new_area = cnts[1]
                else:
                    new_area = cnts[0]

                new_rle = self.mask2rle(concatenated_masks[cat_idx])['counts']
                new_bbox = self.calculate_bbox(concatenated_masks[cat_idx])
              
                new_ann = {
                    'id': int(new_anns_ids[cat_idx]),
                    'image_id': int(img_id),
                    'category_id': int(changed),
                    'segmentation': {
                        'size': [int(height), int(width)],
                        'counts': new_rle,
                    },
                    'area': int(new_area),
                    'bbox': new_bbox,
                    'iscrowd': int(0)
                }

                self.new_annotations_data.append(new_ann)

        for idx, ann in enumerate(self.new_annotations_data):
            ann['id'] = idx + 1 

        self.set_up_data()

        print('NEW ANNOTATIONS COUNT: ', len(self.new_data['annotations']))
        self.save_annotations()
        
        self.new_data = {}
        self.new_annotations_data = []

    def fix_ids(self):
        self.second2new_imgs_ids = {}
        self.second2new_cats_ids = {}

        for idx, cat in enumerate(tqdm.tqdm(self.data['categories'])):
            cat_id = cat['id']
            self.second2new_cats_ids[cat_id] = idx + 1
            self.data['categories'][idx]['id'] = idx + 1

            new_cat = cat.copy()
            new_cat['id'] = idx + 1
            self.new_cats_data.append(new_cat)

        for idx, img in enumerate(tqdm.tqdm(self.data['images'])):
            img_id = img['id']
            self.second2new_imgs_ids[img_id] = idx + 1
            self.data['images'][idx]['id'] = idx + 1

            new_img = img.copy()
            new_img['id'] = idx + 1
            self.new_imgs_data.append(new_img)

        for idx, ann in enumerate(tqdm.tqdm(self.data['annotations'])):
            new_img_id = self.second2new_imgs_ids[ann['image_id']]
            new_cat_id = self.second2new_cats_ids[ann['category_id']]

            ann['image_id'] = new_img_id
            ann['category_id'] = new_cat_id
            ann['id'] = idx + 1

            self.new_annotations_data.append(ann)
        
        self.set_up_data()
        self.save_annotations()
        self.set_default()

    def join_annotations(self):
        self.set_up_data()
        img_ids = [info['id'] for info in self.coco.dataset['images']]
        cat_ids = {info['id']: info['name'] for info in self.coco.dataset['categories']}
        for img_id in tqdm.tqdm(img_ids):
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            
            if len(anns) == 0:
                continue

            sorted_anns = sorted(anns, key=lambda d: CATS_PRIORITY[cat_ids[d['category_id']]]) 
            height, width = anns[0]['segmentation']['size']
            filled_mask = np.zeros((height, width), dtype='uint8')

            for ann in sorted_anns:
                orig_mask = self.rle2mask(ann['segmentation']['counts'], (width, height))
                new_mask = np.where(filled_mask, 0, orig_mask)
                filled_mask = np.where(orig_mask, orig_mask, filled_mask)

                vals, cnts = np.unique(new_mask, return_counts=True)
                if vals.size == 2:
                    new_area = cnts[1]
                else:
                    if vals[0]:
                        new_area = cnts[0]
                    else:
                        continue

                if new_area == 0:
                    print("NULL AREA")

                new_rle = self.mask2rle(new_mask)['counts']
                new_bbox = self.calculate_bbox(new_mask)
                
                if self.new_data['annotations']:
                    new_id = self.new_data['annotations'][-1]['id'] + 1
                else:
                    new_id = 1
              
                new_ann = {
                    'id': int(new_id),
                    'image_id': int(ann['image_id']),
                    'category_id': int(ann['category_id']),
                    'segmentation': {
                        'size': [int(height), int(width)],
                        'counts': new_rle,
                    },
                    'area': int(new_area),
                    'bbox': new_bbox,
                    'iscrowd': int(0)
                }

                self.new_data['annotations'].append(new_ann)

        self.new_annotation_path = self.new_annotation_path
        print('NEW ANNOTATIONS COUNT: ', len(self.new_data['annotations']))
        self.save_annotations()

        self.set_default()

    def show_statics(self):
        obj_counts = {}
        imgs_with_obj_counts = {}
        cat_ids = {info['id']: info['name'] for info in self.coco.dataset['categories']}
        cats = self.coco.loadCats(ids=cat_ids)
        for cat in cats:
            cat_id = cat['id']
            cat_name = cat['name']
            anns_ids = self.coco.getAnnIds(catIds=[cat_id])
            anns = self.coco.loadAnns(ids=anns_ids)
            imgs_ids = set(map(lambda x: x['image_id'] , anns))

            obj_counts[(cat_id, cat_name)] = len(anns_ids)
            imgs_with_obj_counts[(cat_id, cat_name)] = len(imgs_ids)

        print('ANNOTATIONS COUNT: ', len(self.coco.dataset['annotations']))
        print("TOTAL COUNT OF EACH CATEGORY:\n", obj_counts)
        print("COUNT OF IMAGES TO CATEGORY:\n", imgs_with_obj_counts)
        
    def split_train_test(self, test_size=0, folder_pth=None):
        self.train_data = {
                'licenses': [{'name': '', 'id': 0, 'url': ''}], 
                'info': {'contributor': '', 'date_created': '', 
                        'description': '', 'url': '', 'version': '', 'year': ''}, 
                'categories': [],
                'images': [],
                'annotations': []
            }

        self.test_data = {
                'licenses': [{'name': '', 'id': 0, 'url': ''}], 
                'info': {'contributor': '', 'date_created': '', 
                        'description': '', 'url': '', 'version': '', 'year': ''}, 
                'categories': [],
                'images': [],
                'annotations': []
            }
        
        for cat in CATS_IDS_ORIG:
            self.train_data['categories'].append({
                'id': CATS_IDS_ORIG[cat], 
                'name': cat,
                'supercategory': ''
                })

            self.test_data['categories'].append({
                'id': CATS_IDS_ORIG[cat], 
                'name': cat,
                'supercategory': ''
                })
            
        test_imgIds = [x for x in range(1, 40)]
        train_imgIds = self.coco.getImgIds()
        _ = [train_imgIds.remove(x) for x in test_imgIds]

        for idx in range(test_size):
            elem = random.choice(train_imgIds)
            test_imgIds.append(elem)
            train_imgIds.remove(elem)

        train_imgIds = sorted(train_imgIds)
        test_imgIds = sorted(test_imgIds)

        self.train_data['images'] = self.coco.loadImgs(train_imgIds)
        self.test_data['images'] = self.coco.loadImgs(test_imgIds)

        train_annsIds = self.coco.getAnnIds(train_imgIds)
        test_annsIds = self.coco.getAnnIds(test_imgIds)

        self.train_data['annotations'] = self.coco.loadAnns(train_annsIds)
        self.test_data['annotations'] = self.coco.loadAnns(test_annsIds)

        for idx, ann in enumerate(self.train_data['annotations']):
            ann['id'] = idx+1

        for idx, ann in enumerate(self.test_data['annotations']):
            ann['id'] = idx+1

        images_pth = 'datasets/nkb_v1.0_full/train'

        new_images_pth_train = folder_pth + '/train'
        new_images_pth_test = folder_pth + '/test'
        
        if not os.path.exists(new_images_pth_train):
            os.makedirs(new_images_pth_train)

        if not os.path.exists(new_images_pth_test):
            os.makedirs(new_images_pth_test)
        
        with open(new_images_pth_train + '/train.json', 'w') as outfile:
            json.dump(self.train_data, outfile)
        print('saved annotation to ', new_images_pth_train + '/train.json')

        with open(new_images_pth_test + '/test.json', 'w') as outfile:
            json.dump(self.test_data, outfile)
        print('saved annotation to ', new_images_pth_test + '/test.json')

        for img in self.train_data['images']:
            image_name = img['file_name']

            image = Image.open(images_pth + "/" + image_name)
            image.save(os.path.join(new_images_pth_train + "/", image_name))

        for img in self.test_data['images']:
            image_name = img['file_name']

            image = Image.open(images_pth + "/" + image_name)
            image.save(os.path.join(new_images_pth_test + "/", image_name))

    def calculate_bbox(self, binary_mask):
        x = np.where(binary_mask == 255)[1]
        y = np.where(binary_mask == 255)[0]

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        return [int(xmin), int(ymax), int(ymax-ymin), int(xmax-xmin)]
    
    def mask2rle(self, binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 255:
                counts.append(0)
            cnt = len(list(elements))
            counts.append(cnt)
        return rle

    def rle2mask(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (width,height) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        '''
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        i, counter = 0, 0
        for j in mask_rle:
            counter+=1
            if not counter % 2: pixel = 255
            else: pixel = 0
            img[i:i+j] = pixel
            i = i+j
        return img.reshape(shape).T

    def set_up_data(self):
        self.new_data['licenses'] = self.data['licenses']
        self.new_data['info'] = self.data['info']
        self.new_data['categories'] = self.new_cats_data
        self.new_data['images'] = self.new_imgs_data
        self.new_data['annotations'] = self.new_annotations_data

    def save_annotations(self):
        with open(self.new_annotation_path, 'w') as outfile:
            json.dump(self.new_data, outfile)
        print('saved annotation to ', self.new_annotation_path)

    def set_default(self):
            self.new_annotations_data = []
            self.second2new_cats_ids = {}
            self.second2new_imgs_ids = {}
            self.first_categories = None
            self.second_categories = None
            self.first_anns = None
            self.second_anns = None
            self.first_imgs = []
            self.second_imgs = []
                    
            self.new_data = {
                'licenses': [{'name': '', 'id': 0, 'url': ''}], 
                'info': {'contributor': '', 'date_created': '', 
                        'description': '', 'url': '', 'version': '', 'year': ''}, 
                'categories': [],
                'images': [],
                'annotations': []
            }


class JoinAnns():
    def __init__(self) -> None:
        self.new_annotations_data = []
        self.new_annotation_path = None
        self.first_imgs = []
        self.second_imgs = []

        self.new_data = {
            'licenses': [{'name': '', 'id': 0, 'url': ''}], 
            'info': {'contributor': '', 'date_created': '', 
                     'description': '', 'url': '', 'version': '', 'year': ''}, 
            'categories': [],
            'images': [],
            'annotations': []
        }

    def add_annotations(self, first_annotation, second_annotation, new_annotation_path):
        self.add_new_cats(first_annotation, second_annotation, custom_cats=True)
        # self.add_new_cats(first_annotation, second_annotation)
        self.add_new_imgs(first_annotation, second_annotation)
        self.add_new_anns(first_annotation, second_annotation, custom_cats=True)
        # self.add_new_anns(first_annotation, second_annotation)

        self.new_annotation_path = new_annotation_path
        print('NEW ANNOTATIONS COUNT: ', len(self.new_data['annotations']))
        self.save_annotations()
        
        self.set_default()

    def add_new_cats(self, first_annotation, second_annotation, custom_cats=False):
        print("PREPARE CATEGORIES")

        self.first_categories = list(first_annotation.coco.cats.values())
        self.second_categories = list(second_annotation.coco.cats.values())

        if custom_cats:
            self.new_data['categories'] = []
            for cat in CATS_IDS_ORIG:
                self.new_data['categories'].append({
                    'id': CATS_IDS_ORIG[cat], 
                    'name': cat,
                    'supercategory': ''
                    })

        else:
            self.second2new_cats_ids = {}
            self.new_data['categories'] = self.first_categories.copy()
            cats_names = {cat['name']: cat['id'] for cat in self.first_categories}
            for cat in self.second_categories:
                if cat['name'] not in cats_names:
                    new_cat = cat.copy()
                    new_id = self.new_data['categories'][-1]['id'] + 1
                    new_cat['id'] = new_id
                    self.new_data['categories'].append(new_cat)
                    self.second2new_cats_ids[cat['id']] = new_id
                else:
                    cat_name = cat['name']
                    new_id = cats_names[cat_name]
                    self.second2new_cats_ids[cat['id']] = new_id
        

    def add_new_imgs(self, first_annotation, second_annotation):
        print("PREPARE IMAGES")

        for ids_tuple in IMG_IDS[0]:
            self.first_imgs += self.add_imgs_data(first_annotation, ids_tuple)  # add img data for first annotation
        self.new_data['images'] = self.first_imgs.copy()

        for ids_tuple in IMG_IDS[1]:
            self.second_imgs += self.add_imgs_data(second_annotation, ids_tuple)  # add img data for second annotation

        self.second2new_imgs_ids = {}
        imgs_names = {img['file_name']: img['id'] for img in self.first_imgs}
        for idx, img_ann in enumerate(self.second_imgs):
            if img_ann['file_name'] not in imgs_names:
                new_id = self.new_data['images'][-1]['id'] + 1
                img_data = self.second_imgs[idx].copy()
                img_data['id'] = new_id
                self.new_data['images'].append(img_data)
                self.second2new_imgs_ids[img_ann['id']] = new_id
            else:
                img_name = img_ann['file_name']
                new_id = imgs_names[img_name]
                self.second2new_imgs_ids[img_ann['id']] = new_id

    def add_new_anns(self, first_annotation, second_annotation, custom_cats=False):
        print("PREPARE ANNOTATIONS")

        self.first_anns = self.get_anns(self.first_imgs, first_annotation)
        self.second_anns = self.get_anns(self.second_imgs, second_annotation)
        self.new_data['annotations'] = self.first_anns.copy()
        start_id = 1

        if custom_cats:
            for idx, annotation in enumerate(self.new_data['annotations']):
                annotation['id'] = idx + 1
                old_id = annotation['category_id']
                cat_name = [x['name'] for x in self.first_categories if x['id']==old_id][0]
                new_id = CATS_IDS_ORIG[cat_name]
                annotation['category_id'] = new_id

            for annotation in self.second_anns:
                ann = annotation.copy()
                img_id = ann['image_id']
                ann['image_id'] = self.second2new_imgs_ids[img_id]

                old_id = ann['category_id']
                cat_name = [x['name'] for x in self.second_categories if x['id']==old_id][0]
                new_id = CATS_IDS_ORIG[cat_name]
                ann['category_id'] = new_id

                ann['id'] = self.new_data['annotations'][-1]['id'] + 1

                self.new_data['annotations'].append(ann)

        else:
            for annotation in self.second_anns:
                ann = annotation.copy()
                img_id = ann['image_id']
                ann['image_id'] = self.second2new_imgs_ids[img_id]
                cat_id = ann['category_id']
                ann['category_id'] = self.second2new_cats_ids[cat_id]
                ann['id'] = self.new_data['annotations'][-1]['id'] + 1

                self.new_data['annotations'].append(ann)

    def add_imgs_data(self, data, id_range):
        imgIds = [x for x in range(id_range[0], id_range[1]+1)]
        return data.coco.loadImgs(ids=imgIds) 

    def get_anns(self, img_data, data):
        imgs_ids = [x['id'] for x in img_data]
        anns_ids = data.coco.getAnnIds(imgIds=imgs_ids, iscrowd=None)
        return data.coco.loadAnns(anns_ids)

    def save_annotations(self):
        with open(self.new_annotation_path, 'w') as outfile:
            json.dump(self.new_data, outfile)
        print('saved annotation to ', self.new_annotation_path)

    def set_default(self):
        self.new_annotations_data = []
        self.second2new_cats_ids = {}
        self.second2new_imgs_ids = {}
        self.first_categories = None
        self.second_categories = None
        self.first_anns = None
        self.second_anns = None
        self.first_imgs = []
        self.second_imgs = []
                
        self.new_data = {
            'licenses': [{'name': '', 'id': 0, 'url': ''}], 
            'info': {'contributor': '', 'date_created': '', 
                     'description': '', 'url': '', 'version': '', 'year': ''}, 
            'categories': [],
            'images': [],
            'annotations': []
        }


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', help='Path to the user annotation of images', default='images/')
    parser.add_argument('--second_annotation_path', help='Path to the user annotation of images', default='images/')
    parser.add_argument('--new_annotation_path', help='Path to the user annotation of images', default='images/')
    args = parser.parse_args()
    annotation_pth = args.annotation_path
    new_annotation_path = args.new_annotation_path

    first_annotation = Cleaner(annotation_pth, new_annotation_path)
    first_annotation.show_statics()
    first_annotation.extract_certain_cats()

    first_annotation = Cleaner(new_annotation_path, new_annotation_path)
    first_annotation.fix_ids()

    first_annotation = Cleaner(new_annotation_path)
    first_annotation.show_statics()


if __name__ == '__main__':
    main()
    sys.exit(0)