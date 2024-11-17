### DepthAnything
Please **don't rename** python file cause it has dependences in another files for augmentation
To run depthAnything v1 or v2 use folowing sommand:
```
python scripts/depth/depth_estimator.py \
        --images_dir="atest/pothole/images" \
        --output_path="atest" \
        --model="Depth_Anything" \
        --calc_metrics=True 
```