import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import argparse
from pathlib import Path
import wandb
from datetime import datetime, timedelta
import time
from torchvision import transforms
import sys
import re
import gc


# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Get the absolute path of the "opensora" folder and add it to Python path
opensora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'origin_opensora'))
sys.path.append(opensora_path)  # Add opensora_path to sys.path


# Update import path to use correct package structure
from opensora.models.vae_v1_3.vae import OpenSoraVAE_V1_3
from Fine_tune.utils.model_loader import load_pretrained_vae
from opensora.utils.misc import to_torch_dtype

# Set environment variables to avoid fragmentation in CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        try:
            for path in self.root_dir.iterdir():
                if path.is_dir():
                    # Check if directory has enough images
                    image_files = sorted([f for f in path.glob('*.png') if f.is_file()])
                    if len(image_files) >= sequence_length:
                        self.sequence_dirs.append(path)
        except Exception as e:
            print(f"Error reading directory {root_dir}: {e}")
            self.sequence_dirs = []
        
        # For debugging
        print(f"Found {len(self.sequence_dirs)} valid sequence directories")
        
        # Get all image paths for quick access in __getitem__
        self.image_paths = []
        for seq_dir in self.sequence_dirs:
            # Get PNG files from this sequence directory
            seq_images = sorted([f for f in seq_dir.glob('*.png') if f.is_file()])
            
            # Use numeric sorting to ensure correct order
            def get_index(filename):
                match = re.search(r'^(\d+)_', filename.name)
                if match:
                    return int(match.group(1))
                return 0  # Default index if pattern not found
            
            seq_images.sort(key=get_index)
            
            # Only keep sequences with enough images
            if len(seq_images) >= sequence_length:
                # Make sure to take exactly sequence_length frames
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
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.image_paths)} items")
            
        sequence_paths = self.image_paths[idx]
        frames = []
        
        try:
            # Verify we have exactly sequence_length paths
            assert len(sequence_paths) == self.sequence_length, \
                f"Expected {self.sequence_length} frames but got {len(sequence_paths)}"
                
            for img_path in sequence_paths:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            
            # Verify we have loaded the exact number of frames
            assert len(frames) == self.sequence_length, \
                f"Expected {self.sequence_length} frames but loaded {len(frames)}"
            
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
                
            # Final verification of tensor shape
            expected_shape = (3, self.sequence_length, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]) \
                if self.transform and hasattr(self.transform, 'transforms') else (3, self.sequence_length, None, None)
            assert sequence.shape[0] == expected_shape[0] and sequence.shape[1] == expected_shape[1], \
                f"Expected tensor shape starting with {expected_shape[:2]} but got {tuple(sequence.shape[:2])}"
                
            return sequence
        except Exception as e:
            print(f"Error loading sequence {idx} from {sequence_paths[0].parent}: {e}")
            # Return a placeholder tensor in case of error
            if self.transform:
                # Create placeholder with the expected shape based on transform
                placeholder = torch.zeros(3, self.sequence_length, 256, 256)
                if hasattr(self.transform, 'transforms'):
                    for t in self.transform.transforms:
                        if hasattr(t, 'size'):
                            placeholder = torch.zeros(3, self.sequence_length, t.size[0], t.size[1])
                return placeholder
            else:
                # Default placeholder
                return torch.zeros(3, self.sequence_length, 256, 256)


def compute_kl_loss(model, result):
    """Helper function to compute KL divergence loss from model and result"""
    kl_loss = 0.0
    if hasattr(model, "kl_loss"):
        kl_loss = model.kl_loss
    elif isinstance(result, tuple) and len(result) > 2:
        posterior = result[2]
        if hasattr(posterior, "kl"):
            kl_loss = posterior.kl().mean()
    
    return kl_loss

def get_model_dtype(model):
    """Detect the dtype used by the model parameters"""
    model_dtype = None
    for param in model.parameters():
        if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            model_dtype = param.dtype
            break
    if model_dtype is None:
        model_dtype = torch.float32
    return model_dtype

def validate_vae(model, val_loader, device, beta=0.01, amp_dtype=None):
    """Evaluate the model on a validation set"""
    model.eval()
    total_val_loss = 0
    total_val_recon_loss = 0
    total_val_kl_loss = 0
    
    # Get model dtype for type conversion
    model_dtype = get_model_dtype(model)
    
    print(f"Starting validation on {len(val_loader)} batches")
    
    with torch.no_grad():  # No gradient computation during validation
        for batch_idx, batch in enumerate(val_loader):
            # Clear cache before loading new batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Log progress periodically instead of using tqdm
            if batch_idx % 10 == 0:
                print(f"Validating batch {batch_idx}/{len(val_loader)}")
                
            # Move batch to CPU first, then to device and convert to model's dtype
            batch = batch.to(device).to(model_dtype)
            
            # Use automatic mixed precision for validation if specified
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    # Forward pass - explicitly set is_training=False for validation
                    result = model(batch, is_training=False)
                    
                    # Extract needed values
                    decoded = result[1] if isinstance(result, tuple) and len(result) > 1 else result
                    
                    # Compute reconstruction loss
                    recon_loss = F.mse_loss(decoded, batch)
                    
                    # Compute KL divergence
                    kl_loss = compute_kl_loss(model, result)
                    
                    # Total loss
                    loss = recon_loss + beta * kl_loss
            else:
                # Standard precision
                # Forward pass - explicitly set is_training=False for validation
                result = model(batch, is_training=False)
                
                # Extract needed values
                decoded = result[1] if isinstance(result, tuple) and len(result) > 1 else result
                
                # Compute reconstruction loss
                recon_loss = F.mse_loss(decoded, batch)
                
                # Compute KL divergence
                kl_loss = compute_kl_loss(model, result)
                
                # Total loss
                loss = recon_loss + beta * kl_loss
            
            # Track losses
            total_val_loss += loss.item()
            total_val_recon_loss += recon_loss.item()
            total_val_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            
            # Free up memory
            del batch, result, decoded, loss, recon_loss, kl_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate average losses
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_recon_loss = total_val_recon_loss / len(val_loader)
    avg_val_kl_loss = total_val_kl_loss / len(val_loader)
    
    model.train()  # Switch back to training mode
    
    return {
        "val_loss": avg_val_loss,
        "val_recon_loss": avg_val_recon_loss,
        "val_kl_loss": avg_val_kl_loss
    }

def train_vae(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    save_path,
    optimizer,
    beta=0.01,
    save_interval=10,
    log_interval=10,
    validate_interval=1,
    patience=10,
    scheduler=None,
    gradient_accumulation_steps=1,
    amp_dtype=None,
    max_grad_norm=1.0,
    
):
    """Train the VAE model with validation and memory optimization"""
    model.train()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Get model dtype for type conversion
    model_dtype = get_model_dtype(model)
    print(f"Training with model dtype: {model_dtype}")
    
    # Create directories for saving models if they don't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize scaler for mixed precision training if using PyTorch AMP
    scaler = torch.cuda.amp.GradScaler() if amp_dtype is not None and device.type == 'cuda' else None
    
    # Track consecutive OOM errors
    oom_errors = 0
    max_oom_errors = 3  # Stop training after this many consecutive OOM errors
    
    training_start_time = time.time()
    
    for epoch in range(epochs):
        
        
        # Training loop
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Reset optimizer gradients at the beginning of each epoch
        optimizer.zero_grad()
        
        batch_start_time = time.time()
        epoch_batch_times = []
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Clear cache before loading new batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Debug output for batch shape
                orig_shape = batch.shape
                if batch.shape[1] != 3 or batch.shape[2] != model.micro_frame_size * (batch.shape[2] // model.micro_frame_size):
                    print(f"Warning: Input batch shape {batch.shape} may not be compatible with model's micro_frame_size {model.micro_frame_size}")
                    # Ensure sequence length is a multiple of micro_frame_size by truncating
                    new_seq_len = (batch.shape[2] // model.micro_frame_size) * model.micro_frame_size
                    if new_seq_len < batch.shape[2]:
                        print(f"Truncating sequence length from {batch.shape[2]} to {new_seq_len}")
                        batch = batch[:, :, :new_seq_len]
                
                # Move batch to device and convert to model's dtype
                batch = batch.to(device).to(model_dtype)
                
                # Use automatic mixed precision if specified
                if amp_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        # Forward pass - explicitly set is_training=True for training
                        result = model(batch, is_training=True)
                        
                        # Extract needed values based on model's actual return format
                        decoded = result[1] if isinstance(result, tuple) and len(result) > 1 else result
                        
                        # Verify decoded shape matches input shape
                        if decoded.shape != batch.shape:
                            print(f"Shape mismatch: input={batch.shape}, output={decoded.shape}")
                            # If only time dimension differs, this is likely due to padding in the VAE
                            # Try to crop the longer one to match the shorter one
                            min_time = min(decoded.shape[2], batch.shape[2])
                            decoded = decoded[:, :, :min_time]
                            batch = batch[:, :, :min_time]
                            print(f"Adjusted shapes: input={batch.shape}, output={decoded.shape}")
                        
                        # Compute reconstruction loss
                        recon_loss = F.mse_loss(decoded, batch)
                        
                        # Compute KL divergence using helper function
                        kl_loss = compute_kl_loss(model, result)
                        
                        # Total loss with configurable beta
                        loss = recon_loss + beta * kl_loss
                        
                        # Scale loss by gradient accumulation steps
                        loss = loss / gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Step optimizer only after accumulating enough gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients to prevent exploding gradients
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        # Update parameters with gradient scaling
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # Standard precision training
                    # Forward pass - explicitly set is_training=True for training
                    result = model(batch, is_training=True)
                    
                    # Extract needed values based on model's actual return format
                    decoded = result[1] if isinstance(result, tuple) and len(result) > 1 else result
                    
                    # Verify decoded shape matches input shape
                    if decoded.shape != batch.shape:
                        print(f"Shape mismatch: input={batch.shape}, output={decoded.shape}")
                        # If only time dimension differs, this is likely due to padding in the VAE
                        # Try to crop the longer one to match the shorter one
                        min_time = min(decoded.shape[2], batch.shape[2])
                        decoded = decoded[:, :, :min_time]
                        batch = batch[:, :, :min_time]
                        print(f"Adjusted shapes: input={batch.shape}, output={decoded.shape}")
                    
                    # Compute reconstruction loss
                    recon_loss = F.mse_loss(decoded, batch)
                    
                    # Compute KL divergence using helper function
                    kl_loss = compute_kl_loss(model, result)
                    
                    # Total loss with configurable beta
                    loss = recon_loss + beta * kl_loss
                    
                    # Scale loss by gradient accumulation steps
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Step optimizer only after accumulating enough gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        # Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Reset OOM error counter on successful batch
                oom_errors = 0
                
                # Track losses (multiply by gradient_accumulation_steps to get true loss)
                actual_loss = loss.item() * gradient_accumulation_steps
                total_loss += actual_loss
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                
                kl_value = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                
                # Record batch time
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                epoch_batch_times.append(batch_duration)
                
                
                # Update progress bar with time info
                pbar.set_description(f"Epoch {epoch+1}, Loss: {actual_loss:.6f}, Time: {batch_duration:.2f}s")
                
                # Log to wandb periodically
                if batch_idx % log_interval == 0:
                    wandb.log({
                        "batch_loss": actual_loss,
                        "batch_recon_loss": recon_loss.item(),
                        "batch_kl_loss": kl_value,
                        "batch": epoch * len(train_loader) + batch_idx,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "batch_time": batch_duration
                    })
                    
                    # Log image samples periodically
                    if batch_idx % (log_interval * 5) == 0:
                        # Get a sample frame from the batch and reconstruction
                        sample_idx = 0
                        frame_idx = 0
                        
                        # Convert from [-1,1] to [0,1] range for visualization
                        orig_img = (batch[sample_idx, :, frame_idx].cpu().to(torch.float32).permute(1, 2, 0) + 1) / 2
                        recon_img = (decoded[sample_idx, :, frame_idx].detach().cpu().to(torch.float32).permute(1, 2, 0) + 1) / 2
                        
                        wandb.log({
                            "original": wandb.Image(orig_img.numpy()),
                            "reconstruction": wandb.Image(recon_img.numpy())
                        })
                
                # Free up memory
                del batch, result, decoded, loss, recon_loss, kl_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Start timing next batch
                batch_start_time = time.time()
            
            except torch.cuda.OutOfMemoryError as e:
                # Handle CUDA OOM errors
                print(f"CUDA Out of Memory Error ({oom_errors+1}/{max_oom_errors}): {e}")
                oom_errors += 1
                
                # Free up memory and skip this batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Stop training if too many consecutive OOM errors
                if oom_errors >= max_oom_errors:
                    print(f"Training stopped due to {max_oom_errors} consecutive CUDA Out of Memory errors.")
                    # Save the model before exiting
                    save_path_interrupted = os.path.join(save_path, "vae_interrupted.pt")
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, save_path_interrupted)
                        print(f"Interrupted model state saved to {save_path_interrupted}")
                    except Exception as save_err:
                        print(f"Could not save interrupted model: {save_err}")
                    
                    return  # Exit the training function
                
                # Reset the optimizer to clear any gradients
                optimizer.zero_grad()
            
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                # Make sure to zero out gradients in case of error to avoid accumulation
                optimizer.zero_grad()
                # Try to reclaim CUDA memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # Manual garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Calculate training metrics
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        
        # Calculate average batch time for this epoch
        avg_epoch_batch_time = sum(epoch_batch_times) / len(epoch_batch_times) if epoch_batch_times else 0
        
        # Run validation if it's time
        validation_start_time = time.time()
        validation_duration = 0
        
        if (epoch + 1) % validate_interval == 0 and val_loader is not None:
            try:
                val_metrics = validate_vae(model, val_loader, device, beta, amp_dtype)
                
                wandb.log({
                    "epoch": epoch + 1,
                    "val_loss": val_metrics["val_loss"],
                    "val_recon_loss": val_metrics["val_recon_loss"], 
                    "val_kl_loss": val_metrics["val_kl_loss"],
                    "validation_time": validation_duration
                })
                
                print(f"Validation Loss: {val_metrics['val_loss']:.6f}, " 
                      f"Recon: {val_metrics['val_recon_loss']:.6f}, "
                      f"KL: {val_metrics['val_kl_loss']:.6f}, "
                      f"Time: {validation_duration:.2f}s")
                
                # Update learning rate scheduler
                if scheduler is not None:
                    scheduler.step(val_metrics["val_loss"])
                
                # Check if this is the best model
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    epochs_without_improvement = 0
                    
                    # Save best model
                    best_model_path = os.path.join(save_path, "vae_best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    }, best_model_path)
                    print(f"New best model saved to {best_model_path}")
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement} epochs")
                    
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            except Exception as e:
                print(f"Error during validation: {e}")
                # Continue with training even if validation fails
        
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_recon_loss": avg_recon_loss,
            "train_kl_loss": avg_kl_loss,
        })
        
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(save_path, f"vae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            # Log model to wandb
            wandb.save(ckpt_path)
    
    # Calculate and log final timing statistics
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # Format for human readability
    total_time_formatted = str(timedelta(seconds=int(total_training_time)))
    
    # Log final timing info
    print(f"\nTraining completed in {total_time_formatted}")
    
    
    # Log final timing to wandb
    wandb.run.summary["total_training_time"] = total_training_time

def main():
    parser = argparse.ArgumentParser(description='Fine-tune VAE on sun observation images')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing sun images')
    parser.add_argument('--val_dir', type=str, default='dataset/validation/time-series/360p/L16-S8', 
                       help='Directory containing validation data')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained VAE model or HuggingFace URL')
    parser.add_argument('--output_dir', type=str, default='./vae_finetuned', help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Validation batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='Weight for KL divergence loss')
    parser.add_argument('--sequence_length', type=int, default=16, help='Number of frames per sequence')
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images to (square)')
    parser.add_argument('--run_name', type=str, default='', help='Name for this run in wandb')
    parser.add_argument('--wandb_project', type=str, default='sunspot-vae', help='WandB project name')
    parser.add_argument('--save_interval', type=int, default=10, help='save checkpoint every N epochs')
    parser.add_argument('--validate_interval', type=int, default=None, help='Run validation every N epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='Log metrics every N batches')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience (epochs)')
    parser.add_argument('--lr_patience', type=int, default=3, help='Learning rate scheduler patience (epochs)')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Learning rate reduction factor')
    parser.add_argument('--micro_frame_size', type=int, default=2, 
                       help='Micro frame size for VAE model (default is 2 for OpenSoraVAE)')
    parser.add_argument('--dtype', type=str, default=None, help='Model data type (float32, float16, bfloat16)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, 
                       help='Number of update steps to accumulate gradients for')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', 
                       help='Data type for AMP (float16 or bfloat16)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                       help='Maximum gradient norm for gradient clipping')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Use pinned memory for dataloaders')
    
    args = parser.parse_args()
    
    # Set default validate_interval if not provided
    if args.validate_interval is None:
        args.validate_interval = args.save_interval
        
    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize weights and biases
    run_name = args.run_name if args.run_name else f"vae_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # Configure mixed precision training
    amp_dtype = None
    if args.use_amp:
        if args.amp_dtype == 'float16':
            amp_dtype = torch.float16
        elif args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
        else:
            print(f"Warning: Unknown AMP dtype '{args.amp_dtype}'. Using default bfloat16.")
            amp_dtype = torch.bfloat16
    
    # Load model using the utility function
    try:
        print(f"Loading model from {args.pretrained_path}")
        
        # Force CPU loading initially to save GPU memory 
        cpu_device = torch.device('cpu')
        model = load_pretrained_vae(
            model_class=OpenSoraVAE_V1_3,
            pretrained_path=args.pretrained_path,
            micro_frame_size=args.micro_frame_size,
            normalization="video"
        )
        
        # Get model's dtype
        model_dtype = get_model_dtype(model)
        print(f"Model is using dtype: {model_dtype}")
        
        # Override model dtype if specified
        if args.dtype:
            try:
                target_dtype = to_torch_dtype(args.dtype)
                if target_dtype != model_dtype:
                    print(f"Converting model from {model_dtype} to {target_dtype}")
                    model = model.to(target_dtype)
                    model_dtype = target_dtype
            except (ValueError, TypeError) as e:
                print(f"Error converting model to {args.dtype}: {e}")
                print("Continuing with model's original dtype")
        
        if hasattr(model, "micro_frame_size"):
            print(f"Model micro_frame_size: {model.micro_frame_size}")
            # Ensure sequence length is a multiple of micro_frame_size
            if args.sequence_length % model.micro_frame_size != 0:
                new_sequence_length = (args.sequence_length // model.micro_frame_size) * model.micro_frame_size
                print(f"Warning: sequence_length ({args.sequence_length}) is not a multiple of micro_frame_size ({model.micro_frame_size})")
                print(f"Adjusting sequence_length from {args.sequence_length} to {new_sequence_length}")
                args.sequence_length = new_sequence_length
        
        # Move to device after we know the dtype
        model = model.to(device)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Log model architecture summary
    wandb.watch(model, log_freq=100)
    
    # Initialize transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create training dataset and dataloader
    try:
        train_dataset = SunObservationDataset(
            root_dir=args.data_dir,
            sequence_length=args.sequence_length,
            transform=transform
        )
        if len(train_dataset) == 0:
            print(f"Error: No valid sequences found in {args.data_dir}")
            return
            
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,  # Reduced number of workers
            drop_last=True,
            pin_memory=args.pin_memory
        )
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        return
    
    # Create validation dataset and dataloader if validation directory exists
    val_dataloader = None
    if os.path.exists(args.val_dir):
        try:
            val_dataset = SunObservationDataset(
                root_dir=args.val_dir,
                sequence_length=args.sequence_length,
                transform=transform
            )
            if len(val_dataset) > 0:
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,  # Reduced number of workers
                    drop_last=False,
                    pin_memory=args.pin_memory
                )
                print(f"Validation set loaded with {len(val_dataset)} sequences")
            else:
                print(f"Warning: No valid sequences found in validation directory {args.val_dir}")
        except Exception as e:
            print(f"Error loading validation dataset: {e}")
            val_dataloader = None
    else:
        print(f"Warning: Validation directory {args.val_dir} not found. Skipping validation.")
    
    # Log dataset information
    wandb.log({"dataset_size": len(train_dataset)})
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=args.lr_factor, 
        patience=args.lr_patience, 
        verbose=True
    )
    
    # Train
    try:
        train_vae(
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            save_path=args.output_dir,
            beta=args.beta,
            save_interval=args.save_interval,
            validate_interval=args.validate_interval,
            log_interval=args.log_interval,
            patience=args.early_stopping,
            scheduler=scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            amp_dtype=amp_dtype,
            max_grad_norm=args.max_grad_norm
        )
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Finish wandb run
        wandb.finish()
    
    print(f"Fine-tuning complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
    
    from google.colab import runtime
    runtime.unassign()