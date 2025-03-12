import os
import sys
import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the opensora directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'origin_opensora')))

# Import the modules to test
from opensora.models.text_encoder.fourier import FourierFeatureEncoder
from opensora.datasets.datasets import SunObservationDataset

class TestSunObservationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up sample data structure for testing"""
        # Create a temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Create the structure needed for the dataset
        cls.time_series_dir = os.path.join(cls.test_dir, "training", "time-series", "360p", "L16-S8")
        cls.brightness_dir = os.path.join(cls.test_dir, "training", "brightness", "L16-S8")
        
        # Create the directories
        os.makedirs(cls.time_series_dir, exist_ok=True)
        os.makedirs(cls.brightness_dir, exist_ok=True)
        
        # Create test sequences
        cls.num_sequences = 3
        cls.num_frames = 16
        
        # Create 3 test sequences
        for seq_idx in range(cls.num_sequences):
            seq_name = f"sequence_{seq_idx:03d}"
            seq_dir = os.path.join(cls.time_series_dir, seq_name)
            os.makedirs(seq_dir, exist_ok=True)
            
            # Create images for this sequence
            for frame_idx in range(cls.num_frames):
                # Create a simple test image with gradient
                intensity = int(255 * frame_idx / cls.num_frames)
                color = (intensity, intensity, intensity)
                img = Image.new('RGB', (64, 64), color=color)
                img.save(os.path.join(seq_dir, f"{frame_idx:03d}_image.png"))
            
            # Create a brightness data file for this sequence
            brightness_data = np.linspace(0.1, 1.0, cls.num_frames).astype(np.float32)
            np.savez(
                os.path.join(cls.brightness_dir, f"{seq_name}.npz"),
                data=brightness_data
            )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test data structure"""
        shutil.rmtree(cls.test_dir)
    
    def test_dataset_initialization(self):
        """Test dataset initialization and length"""
        dataset = SunObservationDataset(
            time_series_dir=self.time_series_dir,
            brightness_dir=self.brightness_dir,
            num_frames=16
        )
        
        # Check that we found all sequences
        self.assertEqual(len(dataset), self.num_sequences)
    
    def test_getitem(self):
        """Test retrieving an item from the dataset"""
        dataset = SunObservationDataset(
            time_series_dir=self.time_series_dir,
            brightness_dir=self.brightness_dir,
            num_frames=16,
            image_size=(64, 64)
        )
        
        # Get an item
        item = dataset[0]
        
        # Check the structure of the returned item
        self.assertIn('video', item)
        self.assertIn('num_frames', item)
        self.assertIn('height', item)
        self.assertIn('width', item)
        self.assertIn('text', item)
        
        # Check the shapes
        self.assertEqual(item['video'].shape[0], 3)  # 3 channels (RGB)
        self.assertEqual(item['video'].shape[1], 16)  # 16 frames
        self.assertEqual(item['video'].shape[2], 64)  # Height 64
        self.assertEqual(item['video'].shape[3], 64)  # Width 64
        
        # Check the brightness data
        self.assertEqual(item['text'].shape, (16,))  # 16 brightness values
    
    def test_with_fourier_encoder(self):
        """Test integration with the Fourier Feature Encoder"""
        # Initialize the dataset
        dataset = SunObservationDataset(
            time_series_dir=self.time_series_dir,
            brightness_dir=self.brightness_dir,
            num_frames=16,
            image_size=(64, 64)
        )
        
        # Initialize the encoder
        encoder = FourierFeatureEncoder(
            input_dim=1,
            mapping_size=128,
            model_max_length=32,
            device="cuda"
        )
        
        # Set the tokenize function in the dataset
        dataset.tokenize_fn = encoder.tokenize_fn
        
        # Get an item
        item = dataset[0]
        item['input_ids'] = item['input_ids'].unsqueeze(0)
        item['attention_mask'] = item['attention_mask'].unsqueeze(0)
        
        # Check that the tokenization worked
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        
        # Check that we can encode the tokenized data
        encoded = encoder.encode(item['input_ids'], item['attention_mask'])
        
        # Check the encoded output
        self.assertIn('y', encoded)
        self.assertIn('mask', encoded)
        self.assertEqual(encoded['y'].shape[0], 1)  # batch size
        self.assertEqual(encoded['y'].shape[2], 32)  # sequence length
        self.assertEqual(encoded['y'].shape[3], 128)  # embedding dimension


class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Check if real data exists
        self.data_exists = False
        base_dir = Path(os.getcwd())
        
        # Find the dataset directory
        dataset_dir = Path("/content/dataset")
        
        
        if dataset_dir is not None:
            time_series_dir = dataset_dir / "training" / "time-series" / "360p" / "L16-S8"
            brightness_dir = dataset_dir / "training" / "brightness" / "L16-S8"
            
            if time_series_dir.exists() and brightness_dir.exists():
                self.data_exists = True
                self.time_series_dir = str(time_series_dir)
                self.brightness_dir = str(brightness_dir)
    
    def test_real_data(self):
        """Test with real data if it exists"""
        if not self.data_exists:
            self.skipTest("Real data not found in the expected directories")
        
        # Initialize the dataset with real data
        dataset = SunObservationDataset(
            time_series_dir=self.time_series_dir,
            brightness_dir=self.brightness_dir,
            num_frames=16,
            image_size=(256, 256)
        )
        
        # Initialize the encoder
        encoder = FourierFeatureEncoder(
            input_dim=1,
            mapping_size=256,
            model_max_length=64,
            device="cuda"
        )
        
        # Set the tokenize function in the dataset
        dataset.tokenize_fn = encoder.tokenize_fn
        
        # Print basic information about the dataset
        print(f"Found {len(dataset)} sequences in the dataset")
        
        if len(dataset) > 0:
            # Get the first item
            item = dataset
            
            # Check the basic structure
            self.assertIn('video', item)
            self.assertIn('text', item)
            
            # Check the shapes
            print(f"Video shape: {item['video'].shape}")
            print(f"Text shape: {type(item['text'])} - {item['text'].shape if hasattr(item['text'], 'shape') else 'no shape'}")
            
            if 'input_ids' in item:
                # Encode the tokenized data
                encoded = encoder.encode(item['input_ids'], item['attention_mask'])
                print(f"Encoded shape: {encoded['y'].shape}")


if __name__ == '__main__':
    unittest.main()
