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

class TestFourierFeatureEncoder(unittest.TestCase):
    def setUp(self):
        # Set up the encoder for testing
        self.encoder = FourierFeatureEncoder(
            input_dim=1,
            mapping_size=128,
            max_period=10000,
            model_max_length=32,
            device="cuda",  # Use cuda for testing
            dtype=torch.float32
        )
        
        # Create sample brightness data
        self.sample_data_single = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.sample_data_batch = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([0.6, 0.7, 0.8, 0.9, 1.0]),
            np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        ]

    def test_initialization(self):
        """Test that the encoder initializes correctly with different parameters"""
        # Test with default parameters
        encoder = FourierFeatureEncoder()
        self.assertEqual(encoder.output_dim, 256)
        self.assertEqual(encoder.model_max_length, 64)
        
        # Test with custom parameters
        encoder = FourierFeatureEncoder(
            input_dim=2, 
            mapping_size=64, 
            model_max_length=16,
            dtype="float16"
        )
        self.assertEqual(encoder.output_dim, 64)
        self.assertEqual(encoder.model_max_length, 16)
        self.assertEqual(encoder.dtype, torch.float16)

    def test_tokenize_fn(self):
        """Test the tokenize_fn method with different input formats"""
        # Test with single array
        tokens = self.encoder.tokenize_fn(self.sample_data_single)
        self.assertIsInstance(tokens, dict)
        self.assertIn('input_ids', tokens)
        self.assertIn('attention_mask', tokens)
        self.assertEqual(tokens['input_ids'].shape, (1, 32))  # batch_size=1, seq_len=32
        self.assertEqual(tokens['attention_mask'].shape, (1, 32))
        
        # Check that our data is correctly placed in the tensor
        self.assertTrue(torch.allclose(
            tokens['input_ids'][0, :5],
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
        ))
        
        # Test with batch of arrays
        tokens = self.encoder.tokenize_fn(self.sample_data_batch)
        self.assertEqual(tokens['input_ids'].shape, (3, 32))  # batch_size=3, seq_len=32
        self.assertEqual(tokens['attention_mask'].shape, (3, 32))
        
        # Check that attention mask has 1s where we have input and 0s for padding
        self.assertTrue((tokens['attention_mask'][0, :5] == 1).all())
        self.assertTrue((tokens['attention_mask'][0, 5:] == 0).all())

    def test_encode(self):
        """Test the encode method"""
        # First tokenize the data
        tokens = self.encoder.tokenize_fn(self.sample_data_single)
        
        # Then encode it
        encoded = self.encoder.encode(
            tokens['input_ids'], 
            attention_mask=tokens['attention_mask']
        )
        
        # Check the output format
        self.assertIsInstance(encoded, dict)
        self.assertIn('y', encoded)
        self.assertIn('mask', encoded)
        
        # Check the shapes
        self.assertEqual(encoded['y'].shape, (1, 1, 32, 128))  # [batch, 1, seq_len, dim]
        self.assertEqual(encoded['mask'].shape, (1, 32))

    def test_null(self):
        """Test the null method for unconditional generation"""
        # Generate null embeddings for batch size 3
        null_embeddings = self.encoder.null(3)
        
        # Check the shape
        self.assertEqual(null_embeddings.shape, (3, 1, 128))
        
        # Check that the values are zeros (null embeddings)
        self.assertTrue(torch.allclose(null_embeddings, torch.zeros_like(null_embeddings)))


if __name__ == '__main__':
    unittest.main()
