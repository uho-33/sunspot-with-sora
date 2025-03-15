import torch
import torch.nn as nn
import numpy as np
import math
from opensora.registry import MODELS


class FourierFeatureEmbedder(nn.Module):
    """
    Embedder for numerical data using Fourier feature mapping.
    
    Applies a Fourier feature transformation to numerical inputs using
    deterministic sinusoidal encodings at different frequencies.
    """
    def __init__(
        self,
        input_dim=1,
        mapping_size=256,
        max_period=20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.output_dim = mapping_size
        self.max_period = max_period
        
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=20):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
        
    def forward(self, x):
        """
        Apply Fourier feature mapping to input numerical data.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len] or [batch_size, seq_len, input_dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, output_dim]
        """
        # Ensure input has the right shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            
        batch_size, seq_len, in_dim = x.shape
        assert in_dim == self.input_dim, f"Expected input dimension {self.input_dim}, got {in_dim}"
        
        # Reshape for embedding calculation
        x_flat = x.reshape(-1)  # Flatten to apply embedding to each value
        
        # Apply deterministic Fourier embedding
        fourier_features = self.timestep_embedding(x_flat, self.mapping_size, self.max_period)
        
        # Reshape back to original dimensions
        fourier_features = fourier_features.reshape(batch_size, seq_len, in_dim * self.mapping_size)
        
        # Project to output dimension
        return fourier_features


class FourierFeatureEncoderInner:
    """Inner class that handles the actual encoding operations."""
    
    def __init__(
        self,
        input_dim=1,
        mapping_size=256,
        max_period=20,
        model_max_length=64,
        device="cuda",
        dtype=torch.float,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_max_length = model_max_length
        self.output_dim = mapping_size
        
        self.embedder = FourierFeatureEmbedder(
            input_dim=input_dim,
            mapping_size=mapping_size,
            max_period=max_period,
        ).to(device=self.device, dtype=self.dtype)
        
        # Create a null embedding buffer (similar to T5)
        self.register_buffer("y_embedding", torch.zeros(mapping_size, dtype=dtype))
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def tokenize(self, text):
        """
        Convert brightness data (numerical sequences) to tensor format suitable for the model.
        
        Args:
            text: Numpy array or list of brightness values
            
        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        if isinstance(text, (list, tuple)):
            # Handle batch input (list of arrays)
            batch_size = len(text)
            input_ids = torch.zeros((batch_size, self.model_max_length), dtype=self.dtype)
            attention_mask = torch.zeros((batch_size, self.model_max_length), dtype=torch.bool)
            
            for i, seq in enumerate(text):
                # Convert to numpy if needed
                if isinstance(seq, torch.Tensor):
                    seq = seq.detach().cpu().numpy()
                
                # Handle scalar input
                if np.isscalar(seq) or (hasattr(seq, 'size') and seq.size == 1):
                    seq = np.array([seq])
                
                seq_len = min(len(seq), self.model_max_length)
                input_ids[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=self.dtype)
                attention_mask[i, :seq_len] = 1
        else:
            # Handle single sequence input
            if isinstance(text, torch.Tensor):
                text = text.detach().cpu().numpy()
            
            # Handle scalar input
            if np.isscalar(text) or (hasattr(text, 'size') and text.size == 1):
                text = np.array([text])
                
            seq_len = min(len(text), self.model_max_length)
            input_ids = torch.zeros((1, self.model_max_length), dtype=self.dtype)
            attention_mask = torch.zeros((1, self.model_max_length), dtype=torch.bool)
            
            input_ids[0, :seq_len] = torch.tensor(text[:seq_len], dtype=self.dtype)
            attention_mask[0, :seq_len] = 1
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def get_text_embeddings(self, input_ids, attention_mask=None):
        """
        Get embeddings for the input numerical data
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
            attention_mask: Tensor of shape [batch_size, seq_len]
            
        Returns:
            Tuple of (embeddings, mask)
        """
        input_ids = input_ids.to(device=self.device, dtype=self.dtype)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
        else:
            # If no attention mask is provided, assume all tokens are valid
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
        
        # Apply Fourier embedding
        with torch.no_grad():
            embeddings = self.embedder(input_ids).detach()
            
        return embeddings, attention_mask


@MODELS.register_module("fourier")
class FourierFeatureEncoder:
    """
    Encoder for numerical brightness data using Fourier feature mapping.
    
    This encoder transforms numerical brightness data into high-dimensional
    embeddings using the Fourier feature mapping technique, making it suitable
    for the T2V model to process brightness data as "text".
    """
    def __init__(
        self,
        input_dim=1,
        mapping_size=256,
        max_period=10000,
        model_max_length=64,
        device="cuda",
        dtype=torch.float,
    ):
        # Use the specific pytorch dtype object
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            
        self.fourier = FourierFeatureEncoderInner(
            input_dim=input_dim,
            mapping_size=mapping_size,
            max_period=max_period,
            model_max_length=model_max_length,
            device=device,
            dtype=dtype,
        )
        
        self.model_max_length = model_max_length
        self.output_dim = mapping_size
        self.dtype = dtype
        
        # For compatibility with the null method in T5
        self.y_embedder = self.fourier
        
    @property
    def tokenize_fn(self):
        """Returns the tokenize function to match T5Encoder API"""
        return self.fourier.tokenize
    
    def encode(self, input_ids, attention_mask=None):
        """
        Encode numerical data using Fourier features.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
            attention_mask: Tensor of shape [batch_size, seq_len]
            
        Returns:
            Dict with 'y' and 'mask' keys
        """
        caption_embs, emb_masks = self.fourier.get_text_embeddings(input_ids, attention_mask)
        caption_embs = caption_embs[:, None]  # Add dimension to match T5 format
        return dict(y=caption_embs, mask=emb_masks)
    
    def null(self, n):
        """
        Create null/empty embeddings for unconditional generation.
        
        Args:
            n: Batch size
            
        Returns:
            Tensor of shape [n, 1, output_dim]
        """
        null_y = self.y_embedder.y_embedding[None].repeat(n, 1)[:, None]
        return null_y
