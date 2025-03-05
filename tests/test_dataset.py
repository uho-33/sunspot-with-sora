import os
import unittest
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Import the dataset class
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Fine_tune.vae.finetune_vae import SunObservationDataset

class TestSunObservationDataset(unittest.TestCase):
    def setUp(self):
        """Set up the test with path to real data."""
        # Use the real dataset path
        self.base_dir = "/content/drive/MyDrive/projects/sunspot-with-sora"
        self.test_dir = os.path.join(self.base_dir, "dataset/training/time-series/L16-S8")
        self.sequence_length = 5
        
        # Verify that the directory exists
        if not os.path.exists(self.test_dir):
            self.skipTest(f"Test directory {self.test_dir} does not exist")
            
        # Get a specific sequence directory to use for testing
        sequence_dirs = [d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))]
        if not sequence_dirs:
            self.skipTest(f"No sequence directories found in {self.test_dir}")
            
        # Sort directories to get a stable test sequence
        sequence_dirs.sort()
        self.test_sequence_dir = os.path.join(self.test_dir, sequence_dirs[0])
        
        # Verify we have enough images in the sequence
        image_files = [f for f in os.listdir(self.test_sequence_dir) if f.endswith('.png')]
        if len(image_files) < self.sequence_length:
            self.skipTest(f"Not enough images in test sequence: found {len(image_files)}, need {self.sequence_length}")
            
        print(f"Using test sequence directory: {self.test_sequence_dir}")
        print(f"Found {len(image_files)} images for testing")

    def test_dataset_loading(self):
        """Test that the dataset loads correctly."""
        dataset = SunObservationDataset(self.test_dir, sequence_length=self.sequence_length)
        
        # Check that the dataset has entries
        self.assertGreater(len(dataset), 0, "Dataset should have at least one entry")
        
        # Check that the image paths are populated
        self.assertGreater(len(dataset.image_paths), 0, "Dataset should have found image paths")
        
        print(f"Dataset has {len(dataset)} sequences")
        print(f"Dataset found {len(dataset.image_paths)} images")

    def test_sequence_shape(self):
        """Test that sequences have the correct shape."""
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        dataset = SunObservationDataset(self.test_dir, sequence_length=self.sequence_length, transform=transform)
        
        # Get a sequence
        sequence = dataset[0]
        
        # Check shape: should be [C, T, H, W]
        self.assertEqual(len(sequence.shape), 4, "Sequence should have 4 dimensions")
        self.assertEqual(sequence.shape[0], 3, "First dimension should be channels (3)")
        self.assertEqual(sequence.shape[1], self.sequence_length, f"Second dimension should be sequence length ({self.sequence_length})")
        
        print(f"Sequence shape: {sequence.shape}")

    def test_sequence_content(self):
        """Test that sequences contain valid image data."""
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        dataset = SunObservationDataset(self.test_dir, sequence_length=self.sequence_length, transform=transform)
        
        # Get a sequence
        sequence = dataset[0]
        
        # Check that tensor values are in valid range for images [0, 1]
        self.assertGreaterEqual(sequence.min().item(), 0.0, "Minimum tensor value should be >= 0")
        self.assertLessEqual(sequence.max().item(), 1.0, "Maximum tensor value should be <= 1")
        
        print(f"Tensor value range: [{sequence.min().item():.4f}, {sequence.max().item():.4f}]")

    def test_sliding_window(self):
        """Test that sequences from sliding window are related."""
        # Skip test if dataset is too small for multiple sequences
        dataset = SunObservationDataset(self.test_dir, sequence_length=self.sequence_length)
        if len(dataset) < 2:
            self.skipTest("Dataset too small to test sliding window")
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        dataset = SunObservationDataset(self.test_dir, sequence_length=self.sequence_length, transform=transform)
        
        # Get first two sequences
        seq1 = dataset[0]
        seq2 = dataset[1]
        
        # The first frame of seq2 should be similar to the second frame of seq1
        # (allowing for some differences in how the dataset is constructed)
        self.assertTrue(torch.allclose(seq1[:, 1], seq2[:, 0], rtol=1e-3, atol=1e-3) or 
                        torch.mean(torch.abs(seq1[:, 1] - seq2[:, 0])) < 0.1,
                        "Sliding window sequences should overlap")
        
        print("Sliding window test passed")

    def test_different_sequence_lengths(self):
        """Test that the dataset works with different sequence lengths."""
        # Test with different sequence lengths
        for seq_len in [3, 5, 8]:
            try:
                dataset = SunObservationDataset(self.test_dir, sequence_length=seq_len)
                print(f"Dataset with sequence_length={seq_len} has {len(dataset)} sequences")
                
                # Just verify we can get a sequence without error
                if len(dataset) > 0:
                    sequence = dataset[0]
            except Exception as e:
                self.fail(f"Dataset creation failed with sequence_length={seq_len}: {e}")

if __name__ == "__main__":
    unittest.main()
