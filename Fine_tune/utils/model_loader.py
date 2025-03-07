import os
import torch
from huggingface_hub import hf_hub_download
from urllib.parse import urlparse
import requests
from tqdm import tqdm
import safetensors.torch

def download_file(url, destination):
    """
    Download a file from URL with progress bar
    
    Args:
        url: URL to download from
        destination: Local path to save the file
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Get file size if available
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8KB
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    
    # Download with progress bar
    with open(destination, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    
    return destination

def load_pretrained_vae(model_class, pretrained_path, micro_frame_size=None, normalization=None, save_to=None):
    """
    Load a pretrained VAE model from either a local path, a direct URL, or a HuggingFace repo.
    
    Args:
        model_class: The VAE model class to instantiate
        pretrained_path: Local path, direct URL, or HuggingFace repo ID
        micro_frame_size: Optional frame size for the model
        normalization: Optional normalization mode for the model
        save_to: Optional path to save the downloaded model
    
    Returns:
        The loaded model instance
    """
    # First initialize the model with the given parameters
    model_kwargs = {}
    if micro_frame_size is not None:
        model_kwargs['micro_frame_size'] = micro_frame_size
    if normalization is not None:
        model_kwargs['normalization'] = normalization
    
    model = model_class(**model_kwargs)
    
    # Handle direct URLs
    if pretrained_path and (pretrained_path.startswith("http://") or pretrained_path.startswith("https://")):
        print(f"Downloading model from URL: {pretrained_path}")
        
        # Determine the file extension and download location
        file_name = os.path.basename(urlparse(pretrained_path).path)
        download_path = save_to if save_to else os.path.join("./tmp_models", file_name)
        os.makedirs(os.path.dirname(os.path.abspath(download_path)), exist_ok=True)
        
        try:
            # Download the file
            download_file(pretrained_path, download_path)
            print(f"Downloaded model to {download_path}")
            
            # Load based on file extension
            if download_path.endswith('.safetensors') or download_path.endswith('.safetensor'):
                # Load safetensor format
                state_dict = safetensors.torch.load_file(download_path)
                model.load_state_dict(state_dict)
            else:
                # Assume PyTorch format
                model.load_state_dict(torch.load(download_path))
                
            return model
        except Exception as e:
            raise ValueError(f"Error downloading or loading model from URL {pretrained_path}: {e}")
    
    # Handle HuggingFace repo IDs (not URLs)
    elif pretrained_path and not os.path.exists(pretrained_path) and "/" in pretrained_path:
        print(f"Downloading model from HuggingFace repo: {pretrained_path}")
        try:
            # Try to find and download the model file
            repo_id = pretrained_path
            try:
                # First try safetensor format (without 's')
                filename = "model.safetensor"
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                state_dict = safetensors.torch.load_file(model_path)
                model.load_state_dict(state_dict)
            except:
                try:
                    # Then try with 's' at the end
                    filename = "model.safetensors"
                    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    state_dict = safetensors.torch.load_file(model_path)
                    model.load_state_dict(state_dict)
                except:
                    # Fall back to PyTorch format
                    filename = "pytorch_model.bin"
                    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    model.load_state_dict(torch.load(model_path))
            
            # Save the model locally if requested
            if save_to:
                if model_path.endswith('.safetensors') or model_path.endswith('.safetensor'):
                    import shutil
                    shutil.copy(model_path, save_to)
                else:
                    torch.save(model.state_dict(), save_to)
                print(f"Model saved to {save_to}")
                
            return model
        except Exception as e:
            raise ValueError(f"Error loading model from HuggingFace repo {pretrained_path}: {e}")
    
    # Handle local files
    elif os.path.isfile(pretrained_path):
        print(f"Loading model from local file: {pretrained_path}")
        try:
            # Load based on file extension
            if pretrained_path.endswith('.safetensors') or pretrained_path.endswith('.safetensor'):
                # Load safetensor format
                state_dict = safetensors.torch.load_file(pretrained_path)
                model.load_state_dict(state_dict)
            else:
                # Assume PyTorch format
                model.load_state_dict(torch.load(pretrained_path))
            
            # Save the model locally if requested and different from source
            if save_to and save_to != pretrained_path:
                if save_to.endswith('.safetensors') or save_to.endswith('.safetensor'):
                    safetensors.torch.save_file(model.state_dict(), save_to)
                else:
                    torch.save(model.state_dict(), save_to)
                print(f"Model saved to {save_to}")
                
            return model
        except Exception as e:
            raise ValueError(f"Error loading model from file {pretrained_path}: {e}")
    
    else:
        raise ValueError(f"Model path {pretrained_path} not found or invalid")
