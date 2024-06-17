import safetensors.torch
import torch
import safetensors
# from safetensors.torch import 

SAFETENSORS_PATH = "model_output/db_inp_stone_sd2/vae/diffusion_pytorch_model.safetensors"
BIN_PATH = "model_output/db_inp_stone_sd2/checkpoint-1000/diffusion_pytorch_model.bin"

pt_state_dict = safetensors.torch.load_file(
    SAFETENSORS_PATH
    # SAFETENSORS_PATH, device="mps"
)
torch.save(pt_state_dict, BIN_PATH)
print("model saved!")