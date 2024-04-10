import torch
import safetensors

SAFETENSORS_PATH = "model_output/learned_embeds-steps-500.safetensors"
BIN_PATH = "model_output/checkpoint-500/learned_embeds.bin"

pt_state_dict = safetensors.torch.load_file(
    SAFETENSORS_PATH, device="mps"
)
torch.save(pt_state_dict, BIN_PATH)