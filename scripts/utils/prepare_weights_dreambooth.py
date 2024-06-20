import argparse
import shutil
import os

from safetensors2bin import Safetensors2Bin


class PrepareWeights():
    def __init__(self) -> None:
        self.s2b = Safetensors2Bin()

    def prepare(self, model_path, ckpt):
        output_path = os.path.join(model_path, ckpt)

        model_index = os.path.join(model_path, "model_index.json")
        shutil.copy2(model_index, output_path)
        print(f"model_index.json copied to {output_path}")

        preprocessor_config = os.path.join(model_path, "feature_extractor/preprocessor_config.json")
        shutil.copy2(preprocessor_config, output_path)
        print(f"preprocessor_config.json copied to {output_path}")

        scheduler_config = os.path.join(model_path, "scheduler/scheduler_config.json")
        shutil.copy2(scheduler_config, output_path)
        print(f"scheduler_config.json copied to {output_path}")

        vae = os.path.join(model_path, "vae/config.json")
        shutil.copy2(vae, output_path)
        print(f"vae config.json copied to {output_path}")

        diffusion_pytorch_model = os.path.join(model_path, "vae/diffusion_pytorch_model.safetensors")
        diffusion_pytorch_model_bin = os.path.join(output_path, "diffusion_pytorch_model.bin")
        self.s2b.convert(diffusion_pytorch_model, diffusion_pytorch_model_bin)

        self.make_dirs(os.path.join(output_path, "text_encoder"))
        self.make_dirs(os.path.join(output_path, "tokenizer"))
        self.make_dirs(os.path.join(output_path, "unet"))

        text_encoder = os.path.join(model_path, "text_encoder")
        self.copytree(text_encoder, os.path.join(output_path, "text_encoder"))
        print(f"text_encoder copied to {output_path}")

        tokenizer = os.path.join(model_path, "tokenizer")
        self.copytree(tokenizer, os.path.join(output_path, "tokenizer"))
        print(f"tokenizer copied to {output_path}")

        unet = os.path.join(model_path, "unet")
        self.copytree(unet, os.path.join(output_path, "unet"))
        print(f"unet copied to {output_path}")

        print(f"weights prepared at {output_path}")

    def copytree(self, src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    def make_dirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    
    prep = PrepareWeights()
    prep.prepare(args.model_path, args.ckpt)