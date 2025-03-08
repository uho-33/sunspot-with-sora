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

def load_pretrained_vae(model_class, pretrained_path, micro_frame_size=None, normalization=None, save_to=None, strict=True, verbose=True):
    """
    Load a pretrained VAE model from either a local path, a direct URL, or a HuggingFace repo.
    
    Args:
        model_class: The VAE model class to instantiate
        pretrained_path: Local path, direct URL, or HuggingFace repo ID
        micro_frame_size: Optional frame size for the model
        normalization: Optional normalization mode for the model
        save_to: Optional path to save the downloaded model
        strict: Whether to strictly enforce that the keys in state_dict match the keys in model
        verbose: Print detailed information about the loading process
    
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
    
    if not pretrained_path:
        if verbose:
            print("No pretrained path provided, using randomly initialized model.")
        return model
    
    # Helper function to handle actual loading with better error messages
    def load_state_dict_with_error_handling(model, state_dict, strict=True):
        try:
            # Try loading the state dict
            incompatible_keys = model.load_state_dict(state_dict, strict=strict)
            
            if verbose and not strict and incompatible_keys:
                missing = len(incompatible_keys.missing_keys)
                unexpected = len(incompatible_keys.unexpected_keys)
                print(f"Model loaded with {missing} missing keys and {unexpected} unexpected keys.")
                if missing > 0 and verbose:
                    print(f"First few missing keys: {incompatible_keys.missing_keys[:5]}")
                if unexpected > 0 and verbose:
                    print(f"First few unexpected keys: {incompatible_keys.unexpected_keys[:5]}")
                
            return True
        except RuntimeError as e:
            if strict:
                print(f"Error loading state dict with strict=True: {e}")
                print("Attempting to load with strict=False...")
                return load_state_dict_with_error_handling(model, state_dict, strict=False)
            else:
                # This means we already tried with strict=False and still failed
                print(f"Failed to load state dict even with strict=False: {e}")
                return False
    
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
                if not load_state_dict_with_error_handling(model, state_dict, strict=strict):
                    raise ValueError(f"Failed to load model weights from {download_path}")
            else:
                # For PyTorch format, try different loading approaches
                try:
                    # Try loading directly
                    checkpoint = torch.load(download_path)
                    
                    # Check if checkpoint is already a state dict or if it's wrapped
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict'] 
                    elif isinstance(checkpoint, dict) and all(k.startswith('module.') for k in checkpoint.keys()):
                        # Handle DataParallel wrapped state dict
                        state_dict = {k[7:]: v for k, v in checkpoint.items()}
                    elif not any(k.startswith('module.') for k in checkpoint.keys()):
                        # Direct state dict
                        state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    if not load_state_dict_with_error_handling(model, state_dict, strict=strict):
                        raise ValueError(f"Failed to load model weights from {download_path}")
                except Exception as e:
                    print(f"Error loading model in first attempt: {e}")
                    # If it fails, try directly loading the model state
                    try:
                        model.load_state_dict(torch.load(download_path), strict=False)
                        print("Model loaded with strict=False")
                    except Exception as e2:
                        raise ValueError(f"Failed to load model weights: {e2}")
                
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
                if not load_state_dict_with_error_handling(model, state_dict, strict=strict):
                    raise ValueError(f"Failed to load model weights from {model_path}")
            except:
                try:
                    # Then try with 's' at the end
                    filename = "model.safetensors"
                    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    state_dict = safetensors.torch.load_file(model_path)
                    if not load_state_dict_with_error_handling(model, state_dict, strict=strict):
                        raise ValueError(f"Failed to load model weights from {model_path}")
                except:
                    # Fall back to PyTorch format
                    try:
                        filename = "pytorch_model.bin"
                        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                        checkpoint = torch.load(model_path)
                        
                        # Check if it's a state dict directly or wrapped in a larger dict
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint
                            
                        if not load_state_dict_with_error_handling(model, state_dict, strict=strict):
                            raise ValueError(f"Failed to load model weights from {model_path}")
                    except Exception as e:
                        print(f"Failed to load from standard filenames: {e}")
                        # Try finding any .bin or .pt file
                        import glob
                        try:
                            model_files = hf_hub_download(repo_id=repo_id, filename="*", local_dir="./tmp_hf")
                            model_files = glob.glob(os.path.join("./tmp_hf", "*.bin")) + glob.glob(os.path.join("./tmp_hf", "*.pt"))
                            if model_files:
                                model_path = model_files[0]
                                print(f"Trying to load from found file: {model_path}")
                                checkpoint = torch.load(model_path)
                                
                                # Check if it's a state dict directly or wrapped in a larger dict
                                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                                    state_dict = checkpoint['model_state_dict']
                                else:
                                    state_dict = checkpoint
                                    
                                if not load_state_dict_with_error_handling(model, state_dict, strict=strict):
                                    raise ValueError(f"Failed to load model weights from {model_path}")
                            else:
                                raise ValueError(f"No model file found in the HuggingFace repo {repo_id}")
                        except Exception as inner_e:
                            raise ValueError(f"Failed to load any model from HuggingFace repo {repo_id}: {inner_e}")
            
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
                if not load_state_dict_with_error_handling(model, state_dict, strict=strict):
                    raise ValueError(f"Failed to load model weights from {pretrained_path}")
            else:
                # For PyTorch format, try different loading approaches
                try:
                    checkpoint = torch.load(pretrained_path)
                    
                    # Check if checkpoint is already a state dict or if it's wrapped
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        if verbose:
                            print("Found model_state_dict in checkpoint")
                    elif isinstance(checkpoint, dict) and all(k.startswith('module.') for k in checkpoint.keys()):
                        # Handle DataParallel wrapped state dict
                        state_dict = {k[7:]: v for k, v in checkpoint.items()}
                        if verbose:
                            print("Found DataParallel wrapped state dict")
                    else:
                        # Try to load as a direct state dict
                        state_dict = checkpoint
                        if verbose:
                            print("Treating checkpoint as direct state dict")
                    
                    if not load_state_dict_with_error_handling(model, state_dict, strict=strict):
                        raise ValueError(f"Failed to load model weights from {pretrained_path}")
                except Exception as e:
                    print(f"Error in primary loading attempt: {e}")
                    print("Falling back to non-strict loading...")
                    
                    try:
                        # If direct loading fails, try loading the checkpoint and inspecting it
                        checkpoint = torch.load(pretrained_path)
                        
                        if verbose:
                            if isinstance(checkpoint, dict):
                                print(f"Checkpoint keys: {list(checkpoint.keys())}")
                                
                                # If it contains a state_dict key, try that
                                if 'state_dict' in checkpoint:
                                    state_dict = checkpoint['state_dict']
                                    print("Found 'state_dict' key in checkpoint")
                                elif any(k == 'model' or k.endswith('.model') for k in checkpoint.keys()):
                                    # Look for model or *.model keys
                                    for k in checkpoint.keys():
                                        if k == 'model' or k.endswith('.model'):
                                            state_dict = checkpoint[k]
                                            print(f"Using '{k}' from checkpoint")
                                            break
                                else:
                                    # Just use the whole thing as state_dict if nothing else worked
                                    state_dict = checkpoint
                            else:
                                # If checkpoint isn't a dict, it might be the model itself
                                if hasattr(checkpoint, 'state_dict'):
                                    state_dict = checkpoint.state_dict()
                                    print("Using state_dict() method from checkpoint object")
                                else:
                                    raise ValueError("Checkpoint is not a dictionary and doesn't have state_dict method")
                        else:
                            # Non-verbose path - just try common options
                            if isinstance(checkpoint, dict):
                                if 'state_dict' in checkpoint:
                                    state_dict = checkpoint['state_dict']
                                elif 'model_state_dict' in checkpoint:
                                    state_dict = checkpoint['model_state_dict']
                                else:
                                    state_dict = checkpoint
                            else:
                                if hasattr(checkpoint, 'state_dict'):
                                    state_dict = checkpoint.state_dict()
                                else:
                                    raise ValueError("Cannot extract state dict from checkpoint")
                        
                        # Try loading with the extracted state_dict
                        if not load_state_dict_with_error_handling(model, state_dict, strict=False):
                            raise ValueError(f"Failed to load model weights from {pretrained_path}")
                    except Exception as e2:
                        raise ValueError(f"All loading attempts failed for {pretrained_path}: {e2}")
            
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
