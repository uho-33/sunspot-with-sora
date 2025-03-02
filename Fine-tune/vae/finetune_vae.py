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

from opensora.opensora.models.vae_v1_3.vae import OpenSoraVAE_V1_3

class SunObservationDataset(Dataset):
    def __init__(self, image_dir, sequence_length=5, transform=None):
        """
        Dataset for sun observation images.
        Args:
            image_dir: Directory containing sun observation images
            sequence_length: Number of consecutive frames to use as a sequence
            transform: Optional transforms to apply to images
        """
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                                  glob.glob(os.path.join(image_dir, "*.png")))
        self.sequence_length = sequence_length
        self.transform = transform
        
    def __len__(self):
        return max(0, len(self.image_paths) - self.sequence_length + 1)
    
    def __getitem__(self, idx):
        # Create a sequence of images
        sequence = []
        for i in range(self.sequence_length):
            img = Image.open(self.image_paths[idx + i]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            sequence.append(img)
        
        # Stack images to create a video-like tensor [C, T, H, W]
        sequence = torch.stack(sequence, dim=1)  # Shape becomes [C, T, H, W]
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
            
            # Forward pass
            z, decoded, posterior = model(batch, is_training=True)
            
            # Compute reconstruction loss
            recon_loss = F.mse_loss(decoded, batch)
            
            # Compute KL divergence
            kl_loss = posterior.kl().mean()
            
            # Total loss with configurable beta
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.6f}, Recon: {recon_loss.item():.6f}, KL: {kl_loss.item():.6f}")
            
            # Log to wandb periodically
            if batch_idx % log_interval == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_recon_loss": recon_loss.item(),
                    "batch_kl_loss": kl_loss.item(),
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
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained VAE model')
    parser.add_argument('--output_dir', type=str, default='./vae_finetuned', help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='Weight for KL divergence loss')
    parser.add_argument('--sequence_length', type=int, default=5, help='Number of frames per sequence')
    parser.add_argument('--img_size', type=int, default=256, help='Size to resize images to (square)')
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
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
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
    
    # Load model
    model = OpenSoraVAE_V1_3(
        from_pretrained=args.pretrained_path,
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
