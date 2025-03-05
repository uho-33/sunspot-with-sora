import os
import torch
from huggingface_hub import hf_hub_download
from urllib.parse import urlparse

def load_pretrained_vae(model_class, pretrained_path, **model_kwargs):
    """
    Load a pretrained VAE model from either a local path or a HuggingFace repo URL.
    
    Args:
        model_class: The VAE model class to instantiate
        pretrained_path: Local path or HuggingFace URL (e.g., "https://huggingface.co/org/model")
        **model_kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        The loaded model instance
    """
    # Check if the path is a URL
    if pretrained_path and (pretrained_path.startswith("http://") or pretrained_path.startswith("https://")):
        # Extract repository info from URL
        parsed_url = urlparse(pretrained_path)
        path_parts = parsed_url.path.strip("/").split("/")
        
        if len(path_parts) >= 2 and "huggingface.co" in parsed_url.netloc:
            # HuggingFace URL format: huggingface.co/org_name/model_name
            repo_id = f"{path_parts[0]}/{path_parts[1]}"
            
            # Create a temporary directory for downloading
            os.makedirs("./tmp_models", exist_ok=True)
            
            # Try to download the config and model weights
            try:
                # Download the model files (assuming pytorch_model.bin is the main file)
                model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
                print(f"Downloaded model from {repo_id}")
                
                # Create and return model with downloaded weights
                model = model_class(**model_kwargs)
                model.load_state_dict(torch.load(model_path))
                return model
            except Exception as e:
                # If the direct download approach fails, try using from_pretrained if available
                print(f"Direct download failed: {e}")
                print("Trying to use the model's from_pretrained method...")
                
                if hasattr(model_class, "from_pretrained"):
                    return model_class.from_pretrained(repo_id, **model_kwargs)
                else:
                    raise ValueError(f"Could not load model from {pretrained_path}: {e}")
        else:
            raise ValueError(f"Invalid HuggingFace URL format: {pretrained_path}")
    else:
        # Load from local path or use from_pretrained with model identifier
        try:
            # First try to use from_pretrained if it's available
            if hasattr(model_class, "from_pretrained"):
                return model_class.from_pretrained(pretrained_path, **model_kwargs)
            # Otherwise, try loading state dict directly
            elif os.path.isfile(pretrained_path):
                model = model_class(**model_kwargs)
                model.load_state_dict(torch.load(pretrained_path))
                return model
            else:
                raise ValueError(f"Model path {pretrained_path} not found and model class doesn't support from_pretrained")
        except Exception as e:
            raise ValueError(f"Error loading model from {pretrained_path}: {e}")
