import functools
import json
import operator
import os
import re
import shutil
from glob import glob
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.checkpoint_io import GeneralCheckpointIO
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets.utils import download_url
from copy import deepcopy

from .misc import get_logger

hf_endpoint = os.environ.get("HF_ENDPOINT")
if hf_endpoint is None:
    hf_endpoint = "https://huggingface.co"

pretrained_models = {
    "DiT-XL-2-512x512.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "Latte-XL-2-256x256-ucf101.pt": hf_endpoint + "/maxin-cn/Latte/resolve/main/ucf101.pt",
    "PixArt-XL-2-256x256.pth": hf_endpoint + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256.pth",
    "PixArt-XL-2-SAM-256x256.pth": hf_endpoint + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-SAM-256x256.pth",
    "PixArt-XL-2-512x512.pth": hf_endpoint + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth",
    "PixArt-XL-2-1024-MS.pth": hf_endpoint + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth",
    "OpenSora-v1-16x256x256.pth": hf_endpoint + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-16x256x256.pth",
    "OpenSora-v1-HQ-16x256x256.pth": hf_endpoint + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth",
    "OpenSora-v1-HQ-16x512x512.pth": hf_endpoint + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth",
    "PixArt-Sigma-XL-2-256x256.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-256x256.pth",
    "PixArt-Sigma-XL-2-512-MS.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-512-MS.pth",
    "PixArt-Sigma-XL-2-1024-MS.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-1024-MS.pth",
    "PixArt-Sigma-XL-2-2K-MS.pth": hf_endpoint + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-2K-MS.pth",
}


def reparameter(ckpt, name=None, model=None):
    model_name = name
    name = os.path.basename(name)
    if not dist.is_initialized() or dist.get_rank() == 0:
        get_logger().info("loading pretrained model: %s", model_name)
    if name in ["DiT-XL-2-512x512.pt", "DiT-XL-2-256x256.pt"]:
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    if name in ["Latte-XL-2-256x256-ucf101.pt"]:
        ckpt = ckpt["ema"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
        del ckpt["temp_embed"]
    if name in [
        "PixArt-XL-2-256x256.pth",
        "PixArt-XL-2-SAM-256x256.pth",
        "PixArt-XL-2-512x512.pth",
        "PixArt-XL-2-1024-MS.pth",
        "PixArt-Sigma-XL-2-256x256.pth",
        "PixArt-Sigma-XL-2-512-MS.pth",
        "PixArt-Sigma-XL-2-1024-MS.pth",
        "PixArt-Sigma-XL-2-2K-MS.pth",
    ]:
        ckpt = ckpt["state_dict"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        if "pos_embed" in ckpt:
            del ckpt["pos_embed"]

    if name in [
        "PixArt-1B-2.pth",
    ]:
        ckpt = ckpt["state_dict"]
        if "pos_embed" in ckpt:
            del ckpt["pos_embed"]

    # no need pos_embed
    if "pos_embed_temporal" in ckpt:
        del ckpt["pos_embed_temporal"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    # different text length
    if "y_embedder.y_embedding" in ckpt:
        if ckpt["y_embedder.y_embedding"].shape[0] < model.y_embedder.y_embedding.shape[0]:
            get_logger().info(
                "Extend y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            additional_length = model.y_embedder.y_embedding.shape[0] - ckpt["y_embedder.y_embedding"].shape[0]
            new_y_embedding = torch.zeros(additional_length, model.y_embedder.y_embedding.shape[1])
            new_y_embedding[:] = ckpt["y_embedder.y_embedding"][-1]
            ckpt["y_embedder.y_embedding"] = torch.cat([ckpt["y_embedder.y_embedding"], new_y_embedding], dim=0)
        elif ckpt["y_embedder.y_embedding"].shape[0] > model.y_embedder.y_embedding.shape[0]:
            get_logger().info(
                "Shrink y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            ckpt["y_embedder.y_embedding"] = ckpt["y_embedder.y_embedding"][: model.y_embedder.y_embedding.shape[0]]
    # stdit3 special case
    if type(model).__name__ == "STDiT3" and "PixArt" in model_name:
        ckpt_keys = list(ckpt.keys())
        for key in ckpt_keys:
            if "blocks." in key:
                ckpt[key.replace("blocks.", "spatial_blocks.")] = ckpt[key]
                del ckpt[key]

    return ckpt


def find_model(model_name, model=None):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        model_ckpt = download_model(model_name)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
        model_ckpt = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    return model_ckpt


def download_model(model_name=None, local_path=None, url=None):
    """
    Downloads a pre-trained DiT model from the web.
    """
    if model_name is not None:
        assert model_name in pretrained_models
        local_path = f"pretrained_models/{model_name}"
        web_path = pretrained_models[model_name]
    else:
        assert local_path is not None
        assert url is not None
        web_path = url
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        dir_name = os.path.dirname(local_path)
        file_name = os.path.basename(local_path)
        download_url(web_path, dir_name, file_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def load_from_sharded_state_dict(model, ckpt_path, model_name="model", strict=False):
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.load_model(model, os.path.join(ckpt_path, model_name), strict=strict)


def model_sharding(model: torch.nn.Module, device=None):
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        if device is None:
            device = param.device
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params.to(device)


def model_gathering(model: torch.nn.Module, model_shape_dict: dict, device=None):
    global_rank = dist.get_rank()
    global_size = dist.get_world_size()
    for name, param in model.named_parameters():
        if device is None:
            device = param.device
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        if int(global_rank) == 0:
            all_params = torch.cat(all_params)
            param.data = remove_padding(all_params, model_shape_dict[name]).view(model_shape_dict[name]).to(device)
    dist.barrier()


def remove_padding(tensor: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
    return tensor[: functools.reduce(operator.mul, original_shape)]


def record_model_param_shape(model: torch.nn.Module) -> dict:
    param_shape = {}
    for name, param in model.named_parameters():
        param_shape[name] = param.shape
    return param_shape


def adapt_16ch_vae(state_dict):
    if "x_embedder.proj.weight" in state_dict:
        #     state_dict.pop("x_embedder.proj.weight")
        #     state_dict.pop("final_layer.linear.bias")
        #     state_dict.pop("final_layer.linear.weight")

        state_dict["x_embedder.proj.weight"] = state_dict["x_embedder.proj.weight"].repeat(1, 4, 1, 1, 1) / 4
        state_dict["x_embedder.proj.bias"] = state_dict["x_embedder.proj.bias"]
        state_dict["final_layer.linear.weight"] = (
            state_dict["final_layer.linear.weight"].reshape(4, 2, 4, -1).repeat(1, 1, 4, 1).reshape(128, -1)
        )
        state_dict["final_layer.linear.bias"] = (
            state_dict["final_layer.linear.bias"].reshape(4, 2, 4).repeat(1, 1, 4).reshape(128)
        )
    return state_dict


# def adapt_v2v(state_dict, first_adapter = False):
#     if first_adapter:
#         ori_dim = state_dict['final_layer.linear.weight'].shape[0]
#         state_dict['final_layer.linear.weight'] = state_dict['final_layer.linear.weight'][:ori_dim//2,:]
#         state_dict['final_layer.linear.bias'] = state_dict['final_layer.linear.bias'][:ori_dim//2]
#     else:
#         pass
#     return state_dict


def load_from_hf_hub(repo_path, cache_dir):
    repo_id = "/".join(repo_path.split("/")[:-1])
    repo_file = repo_path.split("/")[-1]
    try:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=repo_file, cache_dir=cache_dir)
        return ckpt_path
    except Exception as e:
        print(f"Error downloading from Hugging Face Hub: {e}")
        return None


def load_checkpoint(
    model,
    ckpt_path,
    save_as_pt=False,
    model_name="model",
    strict=False,
    adapt_16ch=False,
    cache_dir=None,
    device: torch.device | str = "cpu",
):
    if not os.path.exists(ckpt_path):
        get_logger().info(f"Checkpoint not found at {ckpt_path} trying to download from Hugging Face Hub")
        ckpt_path = load_from_hf_hub(ckpt_path, cache_dir)

    get_logger().info(f"Loading checkpoint from {ckpt_path}")

    if ckpt_path.endswith(".safetensors"):
        ckpt = load_file(ckpt_path, device=str(device))
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=strict)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)

    elif ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        # state_dict = find_model(ckpt_path, model=model)
        state_dict = torch.load(ckpt_path, map_location=device)
        if adapt_16ch:
            state_dict = adapt_16ch_vae(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    else:
        assert os.path.isdir(ckpt_path), f"Invalid checkpoint path: {ckpt_path}"
        load_from_sharded_state_dict(model, ckpt_path, model_name, strict=strict)
        get_logger().info("Model checkpoint loaded from %s", ckpt_path)
        if save_as_pt:
            save_path = os.path.join(ckpt_path, model_name + "_ckpt.pt")
            torch.save(model.state_dict(), save_path)
            get_logger().info("Model checkpoint saved to %s", save_path)

    return model

def load_checkpoint_exclude_layers(
    model,
    ckpt_path,
    save_as_pt=False,
    model_name="model",
    strict=False,
    adapt_16ch=False,
    cache_dir=None,
    device: torch.device | str = "cuda",
    freeze_other=True,  # Added this argument to control freezing behavior
    init_cross_attn=True,
):
    if not os.path.exists(ckpt_path):
        get_logger().info(f"Checkpoint not found at {ckpt_path}, trying to download from Hugging Face Hub")
        ckpt_path = load_from_hf_hub(ckpt_path, cache_dir)

    get_logger().info(f"Loading checkpoint from {ckpt_path}")

    if ckpt_path.endswith(".safetensors"):
        ckpt = load_file(ckpt_path, device=str(device))
    elif ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        ckpt = torch.load(ckpt_path, map_location=device)
        if adapt_16ch:
            ckpt = adapt_16ch_vae(ckpt)
    else:
        assert os.path.isdir(ckpt_path), f"Invalid checkpoint path: {ckpt_path}"
        load_from_sharded_state_dict(model, ckpt_path, model_name, strict=strict)
        get_logger().info("Model checkpoint loaded from %s", ckpt_path)
        if save_as_pt:
            save_path = os.path.join(ckpt_path, model_name + "_ckpt.pt")
            torch.save(model.state_dict(), save_path)
            get_logger().info("Model checkpoint saved to %s", save_path)
        return model

    # Filter out cross_attn and y_embedder
    filtered_ckpt = None
    if init_cross_attn:
        print("Initialize cross attention...")
        filtered_ckpt = {k: v for k, v in ckpt.items() if "cross_attn" not in k and "y_embedder" not in k}
    
    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_ckpt, strict=False)
    get_logger().info("Missing keys: %s", missing_keys)
    get_logger().info("Unexpected keys: %s", unexpected_keys)

    # Initialize cross_attn and y_embedder weights to zero
    if init_cross_attn or freeze_other:
        if freeze_other:
            print("Frezze layers except cross attention layes")
        for name, param in model.named_parameters():
            if init_cross_attn and ("cross_attn" in name or "y_embedder" in name):
                if "weight" in name:
                    # Xavier/Glorot uniform initialization (good for most attention mechanisms)
                    nn.init.xavier_uniform_(param.data, gain=0.02)
                elif "bias" in name:
                    # Initialize biases to zero
                    nn.init.zeros_(param.data)
                # Special case for projection layers
                if ".proj.weight" in name:
                    # Small initialization for projection layers (helps stability)
                    nn.init.normal_(param.data, std=0.01)
            elif freeze_other:
                param.requires_grad = False  # Freeze other layers if specified

    return model

def load_checkpoint_with_scaled_mapping(
    model,
    ckpt_path,
    save_as_pt=False,
    model_name="model",
    strict=False,
    adapt_16ch=False,
    cache_dir=None,
    device: torch.device | str = "cuda",
    orig_mapping_size=256,
    new_mapping_size=512,
    adaptation_method="linear",  # Options: "linear", "truncate", "copy" 
):
    """
    Load checkpoint with different mapping_size for text encoder.
    This function handles shape mismatches in y_embedder and cross_attn components.
    
    Args:
        model: Model to load the checkpoint into
        ckpt_path: Path to the checkpoint
        save_as_pt: Whether to save the loaded checkpoint as .pt file
        model_name: Name of the model in the checkpoint
        strict: Whether to strictly enforce that the keys match
        adapt_16ch: Whether to adapt the checkpoint for 16 channel VAE
        cache_dir: Cache directory for downloading checkpoints
        device: Device to load the checkpoint to
        orig_mapping_size: Original mapping size used in the checkpoint
        new_mapping_size: New mapping size to use
        adaptation_method: Method to use for adapting weights ("linear", "truncate", "copy")
    """
    if not os.path.exists(ckpt_path):
        get_logger().info(f"Checkpoint not found at {ckpt_path}, trying to download from Hugging Face Hub")
        ckpt_path = load_from_hf_hub(ckpt_path, cache_dir)

    get_logger().info(f"Loading checkpoint from {ckpt_path} with scaled mapping size ({orig_mapping_size} → {new_mapping_size})")
    
    # Function for adapting tensor dimensions based on the selected method
    def adapt_tensor_dim(tensor, target_shape, dim=0):
        if tensor.shape[dim] == target_shape[dim]:
            return tensor
            
        get_logger().info(f"Adapting tensor from shape {tensor.shape} to {target_shape}")
        
        if adaptation_method == "truncate":
            # Simple truncation or zero-padding
            result = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)
            # Calculate how much of the original tensor we can copy
            slice_size = min(tensor.shape[dim], target_shape[dim])
            if dim == 0:
                result[:slice_size] = tensor[:slice_size]
            elif dim == 1:
                result[:, :slice_size] = tensor[:, :slice_size]
            return result
            
        elif adaptation_method == "linear":
            # Linear projection/interpolation
            if dim == 0:
                # For first dimension (e.g., token length)
                result = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)
                slice_size = min(tensor.shape[0], target_shape[0])
                result[:slice_size] = tensor[:slice_size]
                # For any new positions, initialize with mean of existing tokens
                if target_shape[0] > tensor.shape[0]:
                    mean_emb = tensor.mean(dim=0, keepdim=True)
                    result[slice_size:] = mean_emb
            elif dim == 1:
                # For second dimension (e.g., embedding dimension)
                if target_shape[1] > tensor.shape[1]:
                    # Projection to larger dimension
                    projection = nn.Linear(tensor.shape[1], target_shape[1], bias=False).to(device)
                    nn.init.xavier_uniform_(projection.weight, gain=1.0)
                    with torch.no_grad():
                        result = projection(tensor)
                else:
                    # Projection to smaller dimension
                    projection = nn.Linear(tensor.shape[1], target_shape[1], bias=False).to(device)
                    nn.init.xavier_uniform_(projection.weight, gain=1.0)
                    with torch.no_grad():
                        result = projection(tensor)
            return result
            
        else:  # "copy" or fallback
            # Simple copy with zeros for new dimensions
            result = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)
            min_shape = [min(s1, s2) for s1, s2 in zip(tensor.shape, target_shape)]
            if len(min_shape) == 1:
                result[:min_shape[0]] = tensor[:min_shape[0]]
            elif len(min_shape) == 2:
                result[:min_shape[0], :min_shape[1]] = tensor[:min_shape[0], :min_shape[1]]
            elif len(min_shape) == 3:
                result[:min_shape[0], :min_shape[1], :min_shape[2]] = tensor[:min_shape[0], :min_shape[1], :min_shape[2]]
            elif len(min_shape) == 4:
                result[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]] = \
                    tensor[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
            return result

    # Special function to handle projection layers with their specific dimensions
    def adapt_projection_layer(key, param, checkpoint_param):
        if "fc1.weight" in key:
            # For fc1.weight, input dimension (dim=1) is the embedding size that changed
            get_logger().info(f"Adapting projection layer: {key}")
            if param.shape[1] != checkpoint_param.shape[1]:
                # Create a new weight matrix with the right dimensions
                adapted = torch.zeros(param.shape, device=checkpoint_param.device, dtype=checkpoint_param.dtype)
                
                # Copy the existing weights
                min_dim = min(param.shape[1], checkpoint_param.shape[1])
                adapted[:, :min_dim] = checkpoint_param[:, :min_dim]
                
                # Initialize the new part of the weights if expanding
                if param.shape[1] > checkpoint_param.shape[1]:
                    # Initialize new weights with small random values
                    std_dev = checkpoint_param.std() * 0.1
                    new_weights = torch.randn(param.shape[0], param.shape[1] - min_dim, 
                                            device=checkpoint_param.device, 
                                            dtype=checkpoint_param.dtype) * std_dev
                    adapted[:, min_dim:] = new_weights
                
                return adapted
            return checkpoint_param
            
        elif "fc2.weight" in key:
            # For fc2.weight, output dimension (dim=0) hasn't changed
            # But input dimension (dim=1) is the hidden dimension and might need adjustment
            get_logger().info(f"Adapting projection layer: {key}")
            if param.shape[1] != checkpoint_param.shape[1]:
                # Typically this is the hidden dimension which is usually a multiple of the input dim
                scale_factor = param.shape[1] / checkpoint_param.shape[1]
                if scale_factor.is_integer():
                    scale_factor = int(scale_factor)
                    # Repeat the columns to match the new dimension
                    adapted = checkpoint_param.repeat(1, scale_factor)
                    adapted = adapted / scale_factor  # Scale down to maintain output magnitude
                else:
                    # If not a clean multiple, create a new matrix and initialize
                    adapted = torch.zeros(param.shape, device=checkpoint_param.device, dtype=checkpoint_param.dtype)
                    # Copy as much as we can
                    min_dim = min(param.shape[1], checkpoint_param.shape[1])
                    adapted[:, :min_dim] = checkpoint_param[:, :min_dim]
                    # Initialize the rest with small random values if expanding
                    if param.shape[1] > min_dim:
                        std_dev = checkpoint_param.std() * 0.1
                        new_weights = torch.randn(param.shape[0], param.shape[1] - min_dim,
                                                device=checkpoint_param.device,
                                                dtype=checkpoint_param.dtype) * std_dev
                        adapted[:, min_dim:] = new_weights
                
                return adapted
            return checkpoint_param
            
        elif "fc1.bias" in key or "fc2.bias" in key:
            # Biases typically don't change dimension with mapping size changes
            # But we'll handle just in case
            if param.shape != checkpoint_param.shape:
                get_logger().info(f"Adapting bias: {key}")
                adapted = torch.zeros(param.shape, device=checkpoint_param.device, dtype=checkpoint_param.dtype)
                min_dim = min(param.shape[0], checkpoint_param.shape[0])
                adapted[:min_dim] = checkpoint_param[:min_dim]
                return adapted
            return checkpoint_param
            
        # Default fallback
        return adapt_tensor_dim(checkpoint_param, param.shape)
    
    # Function to adapt cross attention weights
    def adapt_cross_attention(key, param, checkpoint_param):
        # Handle different cases of cross-attention components
        if "q_linear.weight" in key:
            # Query dimension remains the same, but embedding dimension changes
            if param.shape[1] != checkpoint_param.shape[1]:
                get_logger().info(f"Adapting cross-attention query: {key}")
                return adapt_tensor_dim(checkpoint_param, param.shape, dim=1)
            return checkpoint_param
                
        elif "kv_linear.weight" in key:
            # KV linear weight handles the embedding dimension change
            if param.shape[0] != checkpoint_param.shape[0] or param.shape[1] != checkpoint_param.shape[1]:
                get_logger().info(f"Adapting cross-attention kv: {key}")
                # First adjust output dimension (0) if needed
                adapted = checkpoint_param
                if param.shape[0] != checkpoint_param.shape[0]:
                    adapted = adapt_tensor_dim(adapted, (param.shape[0], adapted.shape[1]), dim=0)
                # Then adjust input dimension (1) if needed
                if param.shape[1] != checkpoint_param.shape[1]:
                    adapted = adapt_tensor_dim(adapted, param.shape, dim=1)
                return adapted
            return checkpoint_param
                
        elif "proj.weight" in key:
            # Output projection needs to handle output dimension properly
            if param.shape[0] != checkpoint_param.shape[0] or param.shape[1] != checkpoint_param.shape[1]:
                get_logger().info(f"Adapting cross-attention projection: {key}")
                # First adapt input dimension (1)
                adapted = checkpoint_param
                if param.shape[1] != checkpoint_param.shape[1]:
                    adapted = adapt_tensor_dim(adapted, (adapted.shape[0], param.shape[1]), dim=1)
                # Then adapt output dimension (0)
                if param.shape[0] != adapted.shape[0]:
                    adapted = adapt_tensor_dim(adapted, param.shape, dim=0)
                return adapted
            return checkpoint_param
                
        elif "bias" in key:
            if param.shape != checkpoint_param.shape:
                get_logger().info(f"Adapting cross-attention bias: {key}")
                return adapt_tensor_dim(checkpoint_param, param.shape, dim=0)
            return checkpoint_param
                
        # Default case - use standard adaptation
        return adapt_tensor_dim(checkpoint_param, param.shape)

    # Load checkpoint
    state_dict = None
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path, device=str(device))
    elif ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = torch.load(ckpt_path, map_location=device)
        if adapt_16ch:
            state_dict = adapt_16ch_vae(state_dict)
    
    # Directory-based checkpoint handling
    if state_dict is None:
        assert os.path.isdir(ckpt_path), f"Invalid checkpoint path: {ckpt_path}"
        
        # Create a temporary model to load the original weights
        temp_model = deepcopy(model)
        load_from_sharded_state_dict(temp_model, ckpt_path, model_name, strict=False)
        get_logger().info(f"Model checkpoint loaded from {ckpt_path}")
        
        # Extract the state dictionary
        state_dict = temp_model.state_dict()
        
        # Free up memory
        del temp_model
        torch.cuda.empty_cache()
        
    # Identify parameters that need shape adjustment
    model_state_dict = model.state_dict()
    adapted_ckpt = {}
    
    # Find which parameters have shape mismatches and handle them
    for key, param in model_state_dict.items():
        if key in state_dict:
            checkpoint_param = state_dict[key]
            
            # If shapes match, use checkpoint parameter directly
            if param.shape == checkpoint_param.shape:
                adapted_ckpt[key] = checkpoint_param
            else:
                # Special handling for shape mismatches due to mapping_size changes
                if "y_embedder.y_embedding" in key:
                    # For y_embedding, handle both token length and embedding dimension changes
                    get_logger().info(f"Adapting y_embedding: {key}, from shape {checkpoint_param.shape} to {param.shape}")
                    
                    # If token dimension changed
                    if param.shape[0] != checkpoint_param.shape[0]:
                        adapted_param = adapt_tensor_dim(checkpoint_param, param.shape, dim=0)
                    else:
                        adapted_param = checkpoint_param
                    
                    # If embedding dimension changed
                    if param.shape[1] != adapted_param.shape[1]:
                        adapted_param = adapt_tensor_dim(adapted_param, param.shape, dim=1)
                    
                    adapted_ckpt[key] = adapted_param
                
                elif "y_embedder.y_proj.fc" in key:
                    # Special handling for MLP layers in y_proj
                    adapted_ckpt[key] = adapt_projection_layer(key, param, checkpoint_param)
                
                elif "cross_attn" in key:
                    # For cross-attention components, use specialized adaptation
                    adapted_ckpt[key] = adapt_cross_attention(key, param, checkpoint_param)
                
                else:
                    # For other parameters with shape mismatches, log and try basic adaptation
                    get_logger().info(f"Shape mismatch for {key}: model {param.shape} vs checkpoint {checkpoint_param.shape}")
                    adapted_ckpt[key] = adapt_tensor_dim(checkpoint_param, param.shape)
        else:
            # For parameters not in checkpoint, initialize them
            get_logger().info(f"Parameter {key} not found in checkpoint, initializing")
            if "weight" in key:
                if ".proj." in key:
                    # Use smaller initialization for projection layers
                    adapted_ckpt[key] = nn.init.normal_(torch.zeros_like(param), std=0.01)
                else:
                    # Use Xavier/Glorot uniform for other weights
                    adapted_ckpt[key] = nn.init.xavier_uniform_(torch.zeros_like(param), gain=0.02)
            elif "bias" in key:
                # Initialize biases to zero
                adapted_ckpt[key] = torch.zeros_like(param)
            else:
                # For other parameters, just use zeros
                adapted_ckpt[key] = torch.zeros_like(param)
                
    # Load the adapted checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(adapted_ckpt, strict=False)
    
    # Log missing and unexpected keys
    get_logger().info("Missing keys: %s", missing_keys)
    get_logger().info("Unexpected keys: %s", unexpected_keys)
    
    # Save the adapted checkpoint if requested
    if save_as_pt:
        save_path = os.path.join(os.path.dirname(ckpt_path), model_name + "_adapted_ckpt.pt")
        torch.save(model.state_dict(), save_path)
        get_logger().info("Adapted model checkpoint saved to %s", save_path)
    
    return model

def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


# save and load for training


def save(
    booster: Booster,
    save_dir: str,
    model: nn.Module = None,
    ema: nn.Module = None,
    optimizer: Optimizer = None,
    lr_scheduler: _LRScheduler = None,
    sampler=None,
    epoch: int = None,
    step: int = None,
    global_step: int = None,
    batch_size: int = None,
    is_lora_train: bool = False,
    is_save_both_lora_and_main: bool = False,
):
    save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{global_step}")
    if is_lora_train:
        os.makedirs(os.path.join(save_dir, "lora"), exist_ok=True)
    else:
        os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    if model is not None:
        if not is_lora_train:
            booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
        else:
            if is_save_both_lora_and_main:
                booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
                booster.save_lora_as_pretrained(model, os.path.join(save_dir, "lora"))
            else:
                booster.save_lora_as_pretrained(model, os.path.join(save_dir, "lora"))

    if optimizer is not None:
        booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096)
    if lr_scheduler is not None:
        booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    if dist.get_rank() == 0:
        running_states = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "batch_size": batch_size,
        }
        save_json(running_states, os.path.join(save_dir, "running_states.json"))

        if ema is not None:
            torch.save(ema.state_dict(), os.path.join(save_dir, "ema.pt"))

        if sampler is not None:
            # only for VariableVideoBatchSampler
            torch.save(sampler.state_dict(step), os.path.join(save_dir, "sampler"))
    dist.barrier()
    return save_dir


def load(
    booster: Booster,
    load_dir: str,
    model: nn.Module = None,
    ema: nn.Module = None,
    optimizer: Optimizer = None,
    lr_scheduler: _LRScheduler = None,
    sampler=None,
    is_lora_train: bool = False,
    is_load_both_lora_and_main: bool = False,
) -> Tuple[int, int, int]:
    assert os.path.exists(load_dir), f"Checkpoint directory {load_dir} does not exist"
    assert os.path.exists(os.path.join(load_dir, "running_states.json")), "running_states.json does not exist"
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    if model is not None:
        if is_lora_train:
            if is_load_both_lora_and_main:
                booster.load_model(model, os.path.join(load_dir, "lora", "adapter_model.bin"), strict=False)
                booster.load_model(model, os.path.join(load_dir, "model"), strict=False)
            else:
                booster.load_model(model, os.path.join(load_dir, "lora", "adapter_model.bin"), strict=False)
        else:
            booster.load_model(model, os.path.join(load_dir, "model"), strict=False)
    if ema is not None:
        # ema is not boosted, so we don't use booster.load_model
        ema.load_state_dict(
            torch.load(os.path.join(load_dir, "ema.pt"), map_location=torch.device("cpu")),
            strict=False,
        )
    if optimizer is not None:
        booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
    if lr_scheduler is not None:
        booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    if sampler is not None:
        sampler.load_state_dict(torch.load(os.path.join(load_dir, "sampler")))
    dist.barrier()

    return (
        running_states["epoch"],
        running_states["step"],
    )


def rm_checkpoints(
    save_dir: str,
    keep_n_latest: int = 0,
):
    if keep_n_latest <= 0 or dist.get_rank() != 0:
        return
    files = glob(os.path.join(save_dir, "epoch*-global_step*"))
    files = sorted(
        files, key=lambda s: tuple(map(int, re.search(r"epoch(\d+)-global_step(\d+)", s).groups())), reverse=True
    )
    to_remove = files[keep_n_latest:]
    for f in to_remove:
        shutil.rmtree(f)
