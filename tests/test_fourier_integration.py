import os
import sys
import unittest
import torch
import numpy as np
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the opensora directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'origin_opensora')))

# Import the modules to test
from opensora.models.text_encoder.fourier import FourierFeatureEncoder
from opensora.datasets.datasets import SunObservationDataset
from opensora.registry import MODELS

class TestIntegrationWithOpenSora(unittest.TestCase):
    def setUp(self):
        # Check if real data exists
        
        base_dir = Path(os.getcwd())
        
        self.dataset_dir = Path("/content/dataset")
        self.data_exists = self.dataset_dir.exists()
                
        if self.data_exists:
            self.time_series_dir = self.dataset_dir / "training" / "figure" / "360p" / "L16-S8"
            self.brightness_dir = self.dataset_dir / "training" / "brightness" / "L16-S8"
            
            a = self.time_series_dir.exists()
            b = self.brightness_dir.exists()
            if self.time_series_dir.exists() and self.brightness_dir.exists():
                self.data_exists = True

    def test_registry(self):
        """Test that the FourierFeatureEncoder is properly registered in OpenSora's MODELS registry."""
        self.assertIn("fourier", MODELS.module_dict)
        encoder_class = MODELS.get("fourier")
        self.assertEqual(encoder_class, FourierFeatureEncoder)
    
    def test_dataloader_integration(self):
        """Test integration with PyTorch DataLoader."""
        if not self.data_exists:
            self.skipTest("Real data not found in the expected directories")
        
        # Initialize the encoder
        encoder = FourierFeatureEncoder(
            input_dim=1,
            mapping_size=256,
            device="cuda"
        )
        
        # Initialize the dataset
        dataset = SunObservationDataset(
            time_series_dir=str(self.time_series_dir),
            brightness_dir=str(self.brightness_dir),
            num_frames=16,
            tokenize_fn=encoder.tokenize_fn
        )
        
        # Check if there's data in the dataset
        if len(dataset) == 0:
            self.skipTest("No data found in the dataset directories")
            
        # Create a DataLoader
        from torch.utils.data import DataLoader
        batch_size = min(2, len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Check batch structure
        self.assertIn('video', batch)
        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)
        
        # Check batch shapes
        self.assertEqual(batch['video'].shape[0], batch_size)  # batch dimension
        self.assertEqual(batch['input_ids'].shape[0], batch_size)  # batch dimension
        self.assertEqual(batch['attention_mask'].shape[0], batch_size)  # batch dimension
        
        # Test encoding
        encoded = encoder.encode(batch['input_ids'], batch['attention_mask'])
        self.assertIn('y', encoded)
        self.assertIn('mask', encoded)
        self.assertEqual(encoded['y'].shape[0], batch_size)  # batch size

    def test_with_stdit_input_format(self):
        """Test that the encoder produces output compatible with STDiT model input requirements."""
        if not self.data_exists:
            self.skipTest("Real data not found in the expected directories")
        
        # Initialize the encoder
        encoder = FourierFeatureEncoder(
            input_dim=1,
            mapping_size=1024,  # Typical dimension for text encoders in STDiT
            device="cuda"
        )
        
        # Initialize the dataset
        dataset = SunObservationDataset(
            time_series_dir=str(self.time_series_dir),
            brightness_dir=str(self.brightness_dir),
            num_frames=16,
            tokenize_fn=encoder.tokenize_fn
        )
        
        # Check if there's data in the dataset
        if len(dataset) == 0:
            self.skipTest("No data found in the dataset directories")
            
        # Get a sample
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        batch = next(iter(dataloader))
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Encode the sample
        encoded = encoder.encode(input_ids, attention_mask)
        
        # Check that the output has the expected format for STDiT
        # STDiT expects 'y' to be [B, 1, N_token, C]
        self.assertEqual(len(encoded['y'].shape), 4)  # 4 dimensions
        self.assertEqual(encoded['y'].shape[1], 1)    # second dimension should be 1
        

if __name__ == '__main__':
    unittest.main()
