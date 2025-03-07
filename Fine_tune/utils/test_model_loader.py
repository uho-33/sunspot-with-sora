import os
import sys
import torch
import time
from pathlib import Path

# Add necessary paths to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
opensora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'origin_opensora'))
sys.path.append(opensora_path)

from opensora.models.vae_v1_3.vae import OpenSoraVAE_V1_3
from Fine_tune.utils.model_loader import load_pretrained_vae

def test_direct_url_loading():
    """Test loading model directly from URL"""
    print("\n=== Testing Direct URL Loading ===\n")
    
    # Create a directory for saving models if it doesn't exist
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    
    # URL to the OpenSora VAE model
    url = "https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.3/resolve/main/model.safetensors"
    save_path = os.path.join(save_dir, "opensora_vae_v1.3.safetensors")
    
    # Time the loading process
    start_time = time.time()
    
    try:
        model = load_pretrained_vae(
            model_class=OpenSoraVAE_V1_3,
            pretrained_path=url,
            micro_frame_size=None,
            normalization="video",
            save_to=save_path
        )
        
        download_time = time.time() - start_time
        print(f"Model loaded successfully from URL in {download_time:.2f} seconds")
        print(f"Model saved to: {save_path}")
        
        # Create a dummy input and test the model
        test_model_forward_pass(model)
        
        return True, save_path
    except Exception as e:
        print(f"Error loading model from URL: {e}")
        return False, None

def test_local_loading(filepath):
    """Test loading model from local file"""
    print("\n=== Testing Local File Loading ===\n")
    
    if not filepath or not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return False
    
    # Time the loading process
    start_time = time.time()
    
    try:
        model = load_pretrained_vae(
            model_class=OpenSoraVAE_V1_3,
            pretrained_path=filepath,
            micro_frame_size=None,
            normalization="video"
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully from local file in {load_time:.2f} seconds")
        
        # Create a dummy input and test the model
        test_model_forward_pass(model)
        
        return True
    except Exception as e:
        print(f"Error loading model from local file: {e}")
        return False

def test_huggingface_loading():
    """Test loading model from HuggingFace"""
    print("\n=== Testing HuggingFace Loading ===\n")
    
    # Create a directory for saving models
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "opensora_vae_hf.safetensors")
    
    # Time the loading process
    start_time = time.time()
    
    try:
        model = load_pretrained_vae(
            model_class=OpenSoraVAE_V1_3,
            pretrained_path="hpcai-tech/OpenSora-VAE-v1.3",
            micro_frame_size=None,
            normalization="video",
            save_to=save_path
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully from HuggingFace in {load_time:.2f} seconds")
        print(f"Model saved to: {save_path}")
        
        # Create a dummy input and test the model
        test_model_forward_pass(model)
        
        return True
    except Exception as e:
        print(f"Error loading model from HuggingFace: {e}")
        return False

def test_model_forward_pass(model):
    """Test the model with a dummy input"""
    print("\n=== Testing Model Forward Pass ===\n")
    
    try:
        # Detect model dtype
        model_dtype = None
        for param in model.parameters():
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                model_dtype = param.dtype
                break
        if model_dtype is None:
            model_dtype = torch.float32
        
        print(f"Model using dtype: {model_dtype}")
        
        # Move model to CPU for testing
        model = model.to("cpu")
        model.eval()
        
        # Create a dummy input (B, C, T, H, W) = (1, 3, 8, 256, 256)
        # Make sure to use the same dtype as the model
        dummy_input = torch.randn(1, 3, 8, 256, 256, dtype=model_dtype)
        
        # Record time for forward pass
        with torch.no_grad():
            start_time = time.time()
            result = model(dummy_input)
            forward_time = time.time() - start_time
        
        # Check output format
        if isinstance(result, tuple) and len(result) > 1:
            latent, decoded = result[0], result[1]
            print(f"Latent shape: {latent.shape}")
            print(f"Decoded shape: {decoded.shape}")
            # Verify that output dimensions match input dimensions
            assert decoded.shape == dummy_input.shape, "Output shape doesn't match input shape"
            print(f"Forward pass successful in {forward_time:.2f} seconds")
            return True
        else:
            print("Unexpected output format")
            return False
    
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Starting model loader tests...")
    
    # Test direct URL loading
    url_success, saved_path = test_direct_url_loading()
    
    # Test local loading if URL loading succeeded
    if url_success and saved_path:
        local_success = test_local_loading(saved_path)
        if local_success:
            print("\nLocal loading test successful!")
        else:
            print("\nLocal loading test failed!")
    
    # Test HuggingFace loading
    hf_success = test_huggingface_loading()
    if hf_success:
        print("\nHuggingFace loading test successful!")
    else:
        print("\nHuggingFace loading test failed!")
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Direct URL Loading: {'Success' if url_success else 'Failed'}")
    print(f"Local File Loading: {'Success' if url_success and local_success else 'Failed' if url_success else 'Not Tested'}")
    print(f"HuggingFace Loading: {'Success' if hf_success else 'Failed'}")

if __name__ == "__main__":
    main()
