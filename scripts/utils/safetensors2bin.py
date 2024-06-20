import safetensors.torch
import torch
import safetensors
import argparse


class Safetensors2Bin():
    def __init__(self) -> None:
        pass
        # SAFETENSORS_PATH = "model_output/db_inp_stone_sd2/vae/diffusion_pytorch_model.safetensors"
        # BIN_PATH = "model_output/db_inp_stone_sd2/checkpoint-1000/diffusion_pytorch_model.bin"

    def convert(self, SAFETENSORS_PATH, BIN_PATH):
        pt_state_dict = safetensors.torch.load_file(
            SAFETENSORS_PATH
            # SAFETENSORS_PATH, device="mps"
        )
        torch.save(pt_state_dict, BIN_PATH)
        print(f"saved to {BIN_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors_path", type=str, required=True)
    parser.add_argument("--bin_path", type=str, required=True)
    args = parser.parse_args()
    
    s2b = Safetensors2Bin()
    s2b.convert(args.safetensors_path, args.bin_path)