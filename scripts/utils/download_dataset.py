import argparse
import os
import zipfile
import urllib.request
import json

def main():
    args = parser.parse_args()
    url_json = args.url_json
    annotation_json = args.annotation_json
    output = args.output

    with open(url_json, "r") as f:
        urls = json.load(f)

    with open(annotation_json, "r") as f:
        annotations = json.load(f)

    os.makedirs(output, exist_ok=True)
    for id in annotations:
        filename = annotations[id]['file_name']
        url = next(url for url in urls if url['img_name'].split("/")[-1] == filename)["url_image"]
        output_path = os.path.join(output, filename)
        if not os.path.exists(output_path):
            print(f"Downloading {url} to {output_path}")
            urllib.request.urlretrieve(url, output_path)
            if output_path.endswith(".zip"):
                with zipfile.ZipFile(output_path, "r") as zip_ref:
                    zip_ref.extractall(output)
        else:
            print(f"{output_path} already exists, skipping")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_json", type=str, required=True)
    parser.add_argument("--annotation_json", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    main()