import safetensors.torch
import torch
import safetensors
# from safetensors.torch import 

SAFETENSORS_PATH = "model_output/exp_cat-avocado/learned_embeds-steps-1000.safetensors"
BIN_PATH = "model_output/exp_cat-avocado/checkpoint-1000/learned_embeds.bin"

pt_state_dict = safetensors.torch.load_file(
    SAFETENSORS_PATH, device="mps"
)
torch.save(pt_state_dict, BIN_PATH)