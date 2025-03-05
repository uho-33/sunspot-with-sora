import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import argparse
from pathlib import Path
import wandb
from datetime import datetime
from torchvision import transforms
import sys
import re

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Get the absolute path of the "opensora" folder and add it to Python path
opensora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'origin_opensora'))
sys.path.append(opensora_path)  # Add opensora_path to sys.path


# Update import path to use correct package structure
from opensora.models.vae_v1_3.vae import OpenSoraVAE_V1_3
from Fine_tune.utils.model_loader import load_pretrained_vae

class SunObservationDataset(Dataset):
    """
    Dataset for Sun observation sequences.
    
    Each sequence is a time series of sun images stored in a subdirectory.
    Directory structure should be:
    root_dir/
        sequence_dir_1/
            000_image1.png
            001_image2.png
            ...
        sequence_dir_2/
            000_image1.png
            ...
        ...
    """
    
    def __init__(self, root_dir, sequence_length=16, transform=None):
        """
        Args:
            root_dir (string): Directory with all the sequence folders
            sequence_length (int): Number of frames to include in each sequence
            transform (callable, optional): Transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Find all sequence directories
        self.sequence_dirs = []
        for path in self.root_dir.iterdir():
            if path.is_dir():
                # Check if directory has enough images
                image_files = sorted(list(path.glob('*.png')))
                if len(image_files) >= sequence_length:
                    self.sequence_dirs.append(path)
        
        # For debugging
        print(f"Found {len(self.sequence_dirs)} valid sequence directories")
        
        # Get all image paths for quick access in __getitem__
        self.image_paths = []
        for seq_dir in self.sequence_dirs:
            # Get PNG files from this sequence directory
            seq_images = sorted(list(seq_dir.glob('*.png')))
            
            # Use numeric sorting to ensure correct order
            def get_index(filename):
                match = re.search(r'^(\d+)_', filename.name)
                if match:
                    return int(match.group(1))
                return 0  # Default index if pattern not found
            
            seq_images.sort(key=get_index)
            
            # Only keep sequences with enough images
            if len(seq_images) >= sequence_length:
                self.image_paths.append(seq_images[:sequence_length])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns a sequence of images as tensor with shape [C, T, H, W]
        C: channels (3 for RGB)
        T: time/sequence length
        H: height
        W: width
        """
        sequence_paths = self.image_paths[idx]
        frames = []
        
        for img_path in sequence_paths:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        # If transforms were applied, frames should already be tensors
        if self.transform:
            # Stack along new dimension to get [T, C, H, W]
            sequence = torch.stack(frames)
            # Permute to get [C, T, H, W]
            sequence = sequence.permute(1, 0, 2, 3)
        else:
            # If no transforms, frames are PIL images
            # Convert to numpy arrays, stack, and then to tensor
            frames_np = [np.array(frame) for frame in frames]
            sequence_np = np.stack(frames_np, axis=0)  # [T, H, W, C]
            sequence_np = sequence_np.transpose(3, 0, 1, 2)  # [C, T, H, W]
            sequence = torch.from_numpy(sequence_np).float() / 255.0
        
        return sequence

def train_vae(
    model,
    train_loader,
    optimizer,
    device,
    epochs,
    save_path,
    beta=0.01,  # More standard beta value for VAE
    save_interval=10,
    log_interval=10
):
    """Train the VAE model"""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        pbar = tqdm(train_loader)
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = batch.to(device)  # [B, C, T, H, W]
            
            optimizer.zero_grad()
            
            # Forward pass - matching OpenSoraVAE_V1_3 interface
            result = model(batch)
            
            # Extract needed values based on model's actual return format
            # For OpenSoraVAE_V1_3, the model typically returns decoded image and posterior
            decoded = result[1] if isinstance(result, tuple) and len(result) > 1 else result
            
            # Compute reconstruction loss
            recon_loss = F.mse_loss(decoded, batch)
            
            # Compute KL divergence - extract it from the model or results if available
            # If not directly available, use a default or zero value
            kl_loss = 0.0
            if hasattr(model, "kl_loss"):
                kl_loss = model.kl_loss
            elif isinstance(result, tuple) and len(result) > 2:
                posterior = result[2]
                if hasattr(posterior, "kl"):
                    kl_loss = posterior.kl().mean()
            
            # Total loss with configurable beta
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            
            pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.6f}, Recon: {recon_loss.item():.6f}, KL: {kl_loss:.6f}")
            
            # Log to wandb periodically
            if batch_idx % log_interval == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_recon_loss": recon_loss.item(),
                    "batch_kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
                    "batch": epoch * len(train_loader) + batch_idx
                })
                
                # Log image samples periodically
                if batch_idx % (log_interval * 5) == 0:
                    # Get a sample frame from the batch and reconstruction
                    sample_idx = 0
                    frame_idx = 0
                    
                    # Convert from [-1,1] to [0,1] range for visualization
                    orig_img = (batch[sample_idx, :, frame_idx].cpu().permute(1, 2, 0) + 1) / 2
                    recon_img = (decoded[sample_idx, :, frame_idx].detach().cpu().permute(1, 2, 0) + 1) / 2
                    
                    wandb.log({
                        "original": wandb.Image(orig_img.numpy()),
                        "reconstruction": wandb.Image(recon_img.numpy())
                    })
        
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_recon_loss": avg_recon_loss,
            "avg_kl_loss": avg_kl_loss
        })
        
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(save_path, f"vae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            # Log model to wandb
            wandb.save(ckpt_path)

def main():
    parser = argparse.ArgumentParser(description='Fine-tune VAE on sun observation images')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing sun images')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained VAE model or HuggingFace URL')
    parser.add_argument('--output_dir', type=str, default='./vae_finetuned', help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='Weight for KL divergence loss')
    parser.add_argument('--sequence_length', type=int, default=5, help='Number of frames per sequence')
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images to (square)')
    parser.add_argument('--run_name', type=str, default='', help='Name for this run in wandb')
    parser.add_argument('--wandb_project', type=str, default='sunspot-vae', help='WandB project name')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize weights and biases
    run_name = args.run_name if args.run_name else f"vae_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # Initialize transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset and dataloader
    dataset = SunObservationDataset(
        image_dir=args.data_dir,
        sequence_length=args.sequence_length,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # Log dataset information
    wandb.log({"dataset_size": len(dataset)})
    
    # Load model using the utility function
    print(f"Loading model from {args.pretrained_path}")
    model = load_pretrained_vae(
        model_class=OpenSoraVAE_V1_3,
        pretrained_path=args.pretrained_path,
        micro_frame_size=None,  # Let the model handle all frames at once if possible
        normalization="video"
    )
    model = model.to(device)
    
    # Log model architecture summary
    wandb.watch(model, log_freq=100)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Train
    train_vae(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        save_path=args.output_dir,
        beta=args.beta
    )
    
    # Finish wandb run
    wandb.finish()
    
    print(f"Fine-tuning complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()