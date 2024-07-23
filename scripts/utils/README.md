### Download dataset
This script use url.json format of dataset EntitySeg. To download images use folowwing command:
```
python scripts/utils/download_dataset.py \
--url_json="datasets/original/entity_01_11580/url.json" \
--output="datasets/original/entity_01_11580/images" \
--annotation_json="datasets/original/entity_01_11580/train_01_edit.json"
```