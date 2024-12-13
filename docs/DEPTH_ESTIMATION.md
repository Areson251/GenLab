### DepthAnything
Please **don't rename** python file cause it has dependences in another files for augmentation
To run depthAnything v1 or v2 use folowing sommand:
```
python scripts/depth/depth_estimator.py \
        --images_dir="datasets/original/NPO/images" \
        --model="Depth_Anything" \
        --calc_metrics=True \
        --device="cuda:0" \
        --output_path="depth_exps" \
```

python scripts/depth/depth_estimator.py \
        --images_dir="datasets/original/NPO/images" \
        --model="Intel" \
        --calc_metrics=True \
        --device="cuda:1"

### MaryGold
To run dept or normals estimation use folowing sommand:
```
python scripts/depth/marigold.py \
        --images_folder="datasets/original/NPO/images" \
        --dimension="depth" \
        --calc_metrics=True \
        --device="cuda:0" \
        --output=True \
        --dimension="normals" \
```
