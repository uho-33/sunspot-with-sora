import os
import pytest
import torch
import numpy as np
import tempfile
import shutil
from PIL import Image
from pathlib import Path
from datetime import datetime, timedelta

import sys
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Get the absolute path of the "opensora" folder
opensora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'origin_opensora'))
# Add it to the system path
sys.path.insert(0, opensora_path)

# Import the dataset class
from backup.finetune_vae import SunObservationDataset

class TestSunObservationDataset:
    
    @pytest.fixture
    def sample_image_dir(self):
        """Create a temporary directory with sample test images"""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create sample images with timestamps in filenames to ensure chronological order
        width, height = 64, 64
        base_time = datetime.now()
        
        # Create 10 test images with different timestamps in their filenames
        for i in range(10):
            # Create timestamp for filename (ensures chronological order)
            timestamp = base_time + timedelta(minutes=i)
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
            
            # Create a test image with a unique pattern
            # Each image has a different color intensity to easily check sequence order
            color = int(25.5 * i)  # 0 to 255 range across 10 images
            img_array = np.full((height, width, 3), color, dtype=np.uint8)
            
            # Add an identifier number to the center of the image
            font_size = 20
            img = Image.fromarray(img_array)
            # Save the image
            img.save(os.path.join(temp_dir, filename))
        
        yield temp_dir
        
        # Cleanup after test
        shutil.rmtree(temp_dir)
    
    def test_dataset_loading(self, sample_image_dir):
        """Test that the dataset loads files in correct order"""
        # Create dataset with sequence length 1 (just to check order)
        dataset = SunObservationDataset(sample_image_dir, sequence_length=1)
        
        # Verify the dataset has the correct length
        assert len(dataset) == 10, f"Expected 10 items, got {len(dataset)}"
        
        # Verify files were sorted correctly by checking image content
        for i in range(len(dataset)):
            img_tensor = dataset[i]
            
            # Check tensor shape [C, T, H, W]
            assert img_tensor.shape == torch.Size([3, 1, 64, 64]), f"Wrong shape: {img_tensor.shape}"
            
            # Convert tensor to numpy and check mean color value
            # The color should increase with index (0-9 correspond to colors 0-229)
            img_np = img_tensor.numpy()
            avg_color = np.mean(img_np)
            expected_color_normalized = (25.5 * i) / 255.0  # Normalized to [0,1]
            
            # Allow small tolerance due to image compression
            assert abs(avg_color - expected_color_normalized) < 0.1, \
                f"Image {i} has wrong color value: {avg_color} vs expected {expected_color_normalized}"
    
    def test_sequence_creation(self, sample_image_dir):
        """Test that the dataset creates correct sequences of images"""
        seq_length = 3
        dataset = SunObservationDataset(sample_image_dir, sequence_length=seq_length)
        
        # Verify dataset length (10 images - sequence_length + 1)
        assert len(dataset) == 8, f"Expected 8 items for sequence length 3, got {len(dataset)}"
        
        # Check a sequence from the middle
        seq_idx = 2
        sequence = dataset[seq_idx]
        
        # Check shape [C, T, H, W]
        assert sequence.shape == torch.Size([3, seq_length, 64, 64]), f"Wrong sequence shape: {sequence.shape}"
        
        # Verify the sequence contains the correct consecutive frames
        # For each frame in the sequence, check its average color value
        for i in range(seq_length):
            frame_np = sequence[:, i, :, :].numpy()
            avg_color = np.mean(frame_np)
            expected_color_normalized = (25.5 * (seq_idx + i)) / 255.0
            
            # Allow small tolerance due to image compression
            assert abs(avg_color - expected_color_normalized) < 0.1, \
                f"Frame {i} in sequence {seq_idx} has wrong color: {avg_color} vs expected {expected_color_normalized}"
    
    def test_transform_application(self, sample_image_dir):
        """Test that transforms are correctly applied to the images"""
        from torchvision import transforms
        
        # Create a simple transform that resizes images
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        
        # Create dataset with the transform
        dataset = SunObservationDataset(sample_image_dir, sequence_length=2, transform=transform)
        
        # Get a sample from the dataset
        sample = dataset[0]
        
        # Check that the transform was applied (images should be 32x32)
        assert sample.shape == torch.Size([3, 2, 32, 32]), f"Transform not applied correctly: {sample.shape}"
        
        # Check that values were normalized to [0,1] by the ToTensor transform
        assert torch.max(sample) <= 1.0, f"Values not normalized to [0,1], max: {torch.max(sample)}"
        assert torch.min(sample) >= 0.0, f"Values not normalized to [0,1], min: {torch.min(sample)}"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
