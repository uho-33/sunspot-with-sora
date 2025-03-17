import os
import sys
import re
import glob
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import logging

import colossalai
import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import wandb  # Import wandb for logging

# Add paths to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../origin_opensora')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, DATASETS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype

# Create a basic logger that doesn't depend on distributed status
def setup_basic_logger():
    logger = logging.getLogger("checkpoint_eval")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    return logger

# # Rest of functions remain unchanged
# def load_checkpoint(model, checkpoint_path, device):
#     """Load checkpoint (ema.pt) for a model."""
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     # Handle both regular checkpoint and EMA checkpoint
#     if "model" in checkpoint:
#         model.load_state_dict(checkpoint["model"])
#     else:
#         model.load_state_dict(checkpoint)
#     return model

def extract_checkpoint_info(checkpoint_path):
    """Extract epoch and step information from checkpoint path."""
    match = re.search(r'epoch(\d+)-global_step(\d+)', checkpoint_path)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        return epoch, step
    return None, None

def calculate_metrics(ground_truth, prediction):
    """Calculate multiple image quality metrics."""
    # Ensure tensors are on CPU and convert to float32
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().to(torch.float)
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().to(torch.float)
    
    # Handle frame count mismatch
    if ground_truth.shape[1] != prediction.shape[1]:
        # Get minimum number of frames from both videos
        min_frames = min(ground_truth.shape[1], prediction.shape[1])
        # Use only the first min_frames for both videos
        ground_truth = ground_truth[:, :min_frames]
        prediction = prediction[:, :min_frames]
    
    # Calculate Mean Absolute Error (MAE)
    mae = F.l1_loss(prediction, ground_truth).item()
    
    # Calculate Mean Squared Error (MSE)
    mse = F.mse_loss(prediction, ground_truth).item()
    
    # Convert to numpy for skimage metrics
    gt_np = ground_truth.numpy()
    pred_np = prediction.numpy()
    
    # Calculate PSNR - handle frame by frame
    psnr_values = []
    ssim_values = []
    
    # For video data [C, T, H, W], calculate metrics per frame
    for t in range(gt_np.shape[1]):
        # Get individual frames - [C, H, W]
        gt_frame = gt_np[:, t]
        pred_frame = pred_np[:, t]
        
        # Convert to format expected by skimage metrics
        # Normalize to [0, 1] range if needed
        gt_frame = np.transpose(gt_frame, (1, 2, 0))  # [H, W, C]
        pred_frame = np.transpose(pred_frame, (1, 2, 0))  # [H, W, C]
        
        # Calculate PSNR
        psnr_val = psnr(gt_frame, pred_frame, data_range=1.0)
        psnr_values.append(psnr_val)
        
        # Calculate SSIM
        ssim_val = ssim(gt_frame, pred_frame, data_range=1.0, channel_axis=2, multichannel=True)
        ssim_values.append(ssim_val)
    
    return {
        'mae': mae,
        'mse': mse,
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values)
    }

def evaluate_checkpoint(cfg, checkpoint_path, validation_dataloader, device, dtype, logger, is_main=True, results_dir=None):
    """Evaluate a single checkpoint on the validation data."""
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Build models
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    # Prepare video size
    image_size = cfg.get("image_size", (240, 240))
    num_frames = get_num_frames(cfg.get("num_frames", 16))
    
    # Build diffusion model
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    enable_sequence_parallelism = False
    if dist.is_initialized():
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    
    cfg.model["from_pretrained"] = checkpoint_path
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
    ).to(device, dtype).eval()
    
    # Load checkpoint
    text_encoder.y_embedder = model.y_embedder
    
    # Patch for y_embedder null method in fourier text encoder
    if hasattr(text_encoder, 'null'):
        original_null = text_encoder.null
        # Create a patched null method
        def patched_null(n):
            try:
                # Try the original method first
                return original_null(n)
            except RuntimeError as e:
                logger.warning(f"Original null method failed: {e}")
                # Debug y_embedder structure
                y_embedding = text_encoder.y_embedder.y_embedding
                logger.info(f"y_embedding shape: {y_embedding.shape}, dtype: {y_embedding.dtype}")
                
                # Ensure we return a 4D tensor with shape [n, 1, seq_len, embed_dim]
                if y_embedding.dim() == 1:
                    # For 1D embeddings (embed_dim,), expand to 4D
                    embed_dim = y_embedding.shape[0]
                    null_y = y_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(n, 1, 1, 1)
                    logger.info(f"Created null_y with shape: {null_y.shape} from 1D embedding")
                    return null_y
                elif y_embedding.dim() == 2:
                    # For 2D embeddings [seq_len, embed_dim], add batch and channel dimensions
                    seq_len, embed_dim = y_embedding.shape
                    # Convert to [n, 1, seq_len, embed_dim]
                    null_y = y_embedding.unsqueeze(0).unsqueeze(1).repeat(n, 1, 1, 1)
                    logger.info(f"Created null_y with shape: {null_y.shape} from 2D embedding")
                    return null_y
                else:
                    # For other cases, create a zero tensor with expected shape
                    logger.warning(f"Creating fallback zero tensor for null embedding")
                    # Try to infer the expected shape from encoder output_dim
                    embed_dim = getattr(text_encoder, "output_dim", 4096)
                    seq_len = getattr(text_encoder, "model_max_length", 300)
                    # Create 4D tensor [n, 1, seq_len, embed_dim]
                    return torch.zeros(n, 1, seq_len, embed_dim, device=device, dtype=dtype)
                
        # Replace the null method
        text_encoder.null = patched_null
    
    # Build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # Evaluation
    all_metrics = defaultdict(list)
    batch_size = cfg.get("batch_size", 1)
    total_samples = 0
    
    # Extract checkpoint information for logging
    epoch, step = extract_checkpoint_info(checkpoint_path)
    
    with torch.no_grad():
        for batch in tqdm(validation_dataloader, desc=f"Evaluating epoch {epoch}, step {step}"):
            # Extract ground_truth and brightness data
            ground_truth = batch["video"].to(device, dtype)
            brightness_data = batch["text"]
            
            # Generate samples from brightness data
            encoded = vae.encode(ground_truth)
            
            # Add noise (using the same approach as in training)
            torch.manual_seed(0)  # For reproducibility
            z = torch.randn_like(encoded)
            
            # Process brightness data to ensure it's in the expected format for the text encoder
            if isinstance(brightness_data, torch.Tensor):
                # Convert brightness tensors to lists of numeric values
                try:
                    # Check if we're dealing with numeric brightness values
                    if brightness_data.dim() > 1:
                        # Convert each tensor row to a list of float values
                        formatted_prompts = []
                        for b in range(brightness_data.size(0)):
                            # Convert to list of float values (not strings)
                            values = brightness_data[b].cpu().tolist()
                            formatted_prompts.append(values)  # Pass as a list, not a string
                        brightness_data = formatted_prompts
                    else:
                        # For 1D tensor, convert to list directly
                        brightness_data = [brightness_data.cpu().tolist()]
                except Exception as e:
                    logger.error(f"Error formatting brightness data: {e}")
                    # Fall back to simple float representation if the above fails
                    try:
                        brightness_data = [[float(x) for x in b] for b in brightness_data]
                    except:
                        logger.error("Failed to convert to float lists, using numeric constants")
                        # Last resort: use simple numeric values
                        brightness_data = [0.5] * len(brightness_data)  # Default value as fallback
            elif isinstance(brightness_data, list):
                # Check if the list contains tensors
                if brightness_data and isinstance(brightness_data[0], torch.Tensor):
                    brightness_data = [b.cpu().tolist() if isinstance(b, torch.Tensor) else b for b in brightness_data]
            
            try:
                # Get tokenized data for the text encoder
                tokens = text_encoder.tokenize_fn(brightness_data)
                encoded_data = text_encoder.encode(**tokens) if tokens else None
                
                # Extract video dimensions from the batch
                batch_size = ground_truth.shape[0]
                height = batch["height"].to(device) if "height" in batch else torch.ones(batch_size, device=device) * ground_truth.shape[2]
                width = batch["width"].to(device) if "width" in batch else torch.ones(batch_size, device=device) * ground_truth.shape[3]
                num_frames = batch["num_frames"].to(device) if "num_frames" in batch else torch.ones(batch_size, device=device) * ground_truth.shape[1]
                
                # Create model_kwargs dictionary required by scheduler
                model_kwargs = {
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "fps": batch["fps"].to(device) if "fps" in batch else torch.ones(batch_size, device=device) * 16.0,  # Add fps parameter
                }
                
                # Sample using scheduler with properly formatted brightness data
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=brightness_data,
                    device=device,
                    progress=False,
                    additional_args=model_kwargs,  # This parameter name changed from model_kwargs to additional_args
                )
                
                # Decode samples to pixel space
                generated_videos = vae.decode(samples.to(dtype), num_frames=num_frames[0].item())
                
                # Calculate metrics between ground truth and generated videos
                for i in range(len(ground_truth)):
                    metrics = calculate_metrics(ground_truth[i], generated_videos[i])
                    for metric_name, value in metrics.items():
                        all_metrics[metric_name].append(value)
            except Exception as e:
                logger.error(f"Error during sampling: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error("Skipping this batch due to error")
                continue  # Skip to the next batch
            
            total_samples += len(ground_truth)
            
            # Log all frames from first video to wandb for visualization (first batch only)
            if is_main and total_samples <= batch_size and cfg.get("wandb", False):
                # Get all frames from the first video
                gt_video = ground_truth[0].detach().cpu()  # [C, T, H, W]
                pred_video = generated_videos[0].detach().cpu()  # [C, T, H, W]
                
                # Get number of frames (use the minimum between ground truth and prediction)
                num_video_frames = min(gt_video.shape[1], pred_video.shape[1])
                
                # Create directory to save frames
                frames_dir = os.path.join(results_dir, f"frames_epoch{epoch}_step{step}")
                os.makedirs(frames_dir, exist_ok=True)
                
                # Process each frame
                all_comparisons = []
                for frame_idx in range(num_video_frames):
                    # Get individual frames
                    gt_frame = gt_video[:, frame_idx]
                    pred_frame = pred_video[:, frame_idx]
                    
                    # Normalize if needed (if data isn't already in [0, 1])
                    if gt_frame.min() < 0 or gt_frame.max() > 1:
                        gt_frame = (gt_frame - gt_frame.min()) / (gt_frame.max() - gt_frame.min() + 1e-8)
                    
                    if pred_frame.min() < 0 or pred_frame.max() > 1:
                        pred_frame = (pred_frame - pred_frame.min()) / (pred_frame.max() - pred_frame.min() + 1e-8)
                    
                    # Convert to numpy
                    gt_frame = gt_frame.to(torch.float).numpy()
                    pred_frame = pred_frame.to(torch.float).numpy()
                    
                    # Convert from CHW to HWC for visualization
                    gt_frame = np.transpose(gt_frame, (1, 2, 0))
                    pred_frame = np.transpose(pred_frame, (1, 2, 0))
                    
                    # Ensure values are in [0, 1] range
                    gt_frame = np.clip(gt_frame, 0, 1)
                    pred_frame = np.clip(pred_frame, 0, 1)
                    
                    # Create a side-by-side comparison
                    comparison = np.concatenate([gt_frame, pred_frame], axis=1)
                    all_comparisons.append(comparison)
                    
                    # Save individual frame
                    plt.figure(figsize=(10, 5))
                    plt.imshow(comparison)
                    plt.title(f"Frame {frame_idx} - Left: Ground Truth, Right: Generated")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(frames_dir, f"frame_{frame_idx:03d}.png"))
                    plt.close()
                
                # Log video to wandb
                if len(all_comparisons) > 0:
                    try:
                        # Make sure dimensions are correct for wandb.Video: [T, H, W, C]
                        # Convert to uint8 for video with values 0-255
                        video_array = np.stack([
                            (np.clip(comp, 0, 1) * 255).astype(np.uint8) 
                            for comp in all_comparisons
                        ])
                        
                        logger.info(f"Video array shape: {video_array.shape}, dtype: {video_array.dtype}")
                        logger.info(f"Video array min: {video_array.min()}, max: {video_array.max()}")
                        
                        # Try saving the video as a temporary file first
                        import imageio
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmpfile:
                            writer = imageio.get_writer(tmpfile.name, fps=4.0, format='FFMPEG', quality=8)
                            for frame in video_array:
                                writer.append_data(frame)
                            writer.close()
                            
                            # Upload the video file
                            wandb.log({
                                f"checkpoint_{epoch}_{step}/video_comparison": wandb.Video(
                                    tmpfile.name,
                                    fps=4.0,
                                    format="mp4",
                                    caption=f"Video Comparison (Epoch {epoch}, Step {step})"
                                )
                            })
                            logger.info(f"Successfully logged video from file: {tmpfile.name}")
                    except Exception as e:
                        logger.warning(f"Failed to create video for wandb: {e}")
                        # Create a grid of frames as a fallback
                        try:
                            # Create a grid of the first few frames (at most 9)
                            num_frames_to_log = min(9, len(all_comparisons))
                            grid_size = int(np.ceil(np.sqrt(num_frames_to_log)))
                            
                            # Create a large combined figure with multiple frames
                            plt.figure(figsize=(15, 15))
                            for i in range(num_frames_to_log):
                                plt.subplot(grid_size, grid_size, i+1)
                                plt.imshow(all_comparisons[i])
                                plt.title(f"Frame {i}")
                                plt.axis('off')
                            plt.tight_layout()
                            
                            # Save as a file
                            grid_path = os.path.join(frames_dir, f"frames_grid.png") 
                            plt.savefig(grid_path)
                            plt.close()
                            
                            # Log the grid image
                            wandb.log({
                                f"checkpoint_{epoch}_{step}/frames_grid": wandb.Image(
                                    grid_path,
                                    caption=f"Frames Grid (Epoch {epoch}, Step {step})"
                                )
                            })
                            
                            logger.info("Logged frames grid as fallback")
                        except Exception as grid_error:
                            logger.error(f"Failed to create frames grid: {grid_error}")
                    
                    # Always log first frame for quick reference
                    wandb.log({
                        f"checkpoint_{epoch}_{step}/first_frame": wandb.Image(
                            all_comparisons[0], 
                            caption=f"First Frame (Epoch {epoch}, Step {step})"
                        )
                    })
    
    # Calculate average metrics (only if we have collected some metrics)
    if all_metrics:
        avg_metrics = {metric: sum(values) / len(values) for metric, values in all_metrics.items()}
        
        # Add checkpoint info
        avg_metrics["epoch"] = epoch
        avg_metrics["step"] = step
        
        # Log metrics to wandb if enabled
        if is_main and cfg.get("wandb", False):
            # Log to wandb with step as x-axis
            wandb_metrics = {
                "mae": avg_metrics["mae"],
                "mse": avg_metrics["mse"],
                "psnr": avg_metrics["psnr"],
                "ssim": avg_metrics["ssim"],
            }
            wandb.log(wandb_metrics, step=step)
            
            # Also log as a separate entry with checkpoint info in the name
            checkpoint_metrics = {f"checkpoint_{epoch}_{step}/{k}": v for k, v in wandb_metrics.items()}
            wandb.log(checkpoint_metrics)
        
        logger.info(f"Finished evaluation of checkpoint (Epoch {epoch}, Step {step})")
        logger.info(f"Metrics: MAE={avg_metrics['mae']:.4f}, MSE={avg_metrics['mse']:.4f}, "
                    f"PSNR={avg_metrics['psnr']:.2f}, SSIM={avg_metrics['ssim']:.4f}")
    else:
        # If no metrics were collected, return a default structure with just epoch and step
        logger.warning("No metrics were collected for this checkpoint (all batches failed)")
        avg_metrics = {
            "epoch": epoch,
            "step": step,
            "mae": 0.0,
            "mse": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "error": "All batches failed"
        }
    
    return avg_metrics

def find_checkpoints(checkpoints_dir, checkpoint_name="ema.pt"):
    """Find all checkpoint files (default: ema.pt) in the given directory and its subdirectories."""
    checkpoint_pattern = os.path.join(checkpoints_dir, "epoch*-global_step*", checkpoint_name)
    return sorted(glob.glob(checkpoint_pattern))

def plot_metrics(metrics_list, save_path):
    """Plot metrics over training steps."""
    # Extract data
    epochs = [m['epoch'] for m in metrics_list]
    steps = [m['step'] for m in metrics_list]
    mae = [m['mae'] for m in metrics_list]
    mse = [m['mse'] for m in metrics_list]
    psnr = [m['psnr'] for m in metrics_list]
    ssim = [m['ssim'] for m in metrics_list]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE
    axes[0, 0].plot(steps, mae, 'b-o')
    axes[0, 0].set_title('Mean Absolute Error (MAE)')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].grid(True)
    
    # MSE
    axes[0, 1].plot(steps, mse, 'r-o')
    axes[0, 1].set_title('Mean Squared Error (MSE)')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].grid(True)
    
    # PSNR
    axes[1, 0].plot(steps, psnr, 'g-o')
    axes[1, 0].set_title('Peak Signal-to-Noise Ratio (PSNR)')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].grid(True)
    
    # SSIM
    axes[1, 1].plot(steps, ssim, 'm-o')
    axes[1, 1].set_title('Structural Similarity Index (SSIM)')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('SSIM')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # First, set up a basic logger that doesn't depend on distributed status
    logger = setup_basic_logger()
    
    parser = argparse.ArgumentParser(description='Evaluate T2V model performance across checkpoints')
    parser.add_argument('--config', type=str, default='Fine_tune/configs/train/fine_tune_stage1.py',
                        help='Path to the model configuration file')
    parser.add_argument('--checkpoints_dir', type=str, default='outputs/0002-Sunspot_STDiT3-XL-2',
                        help='Directory containing checkpoint directories')
    parser.add_argument('--validation_data_dir', type=str, default='/content/dataset/validation',
                        help='Directory containing validation data')
    parser.add_argument('--results_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for validation')
    parser.add_argument('--specific_checkpoint', type=str, default=None,
                        help='Evaluate only this specific checkpoint path')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable logging to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='sun-reconstruction-eval',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name')
    args = parser.parse_args()
    
    # Log initial information
    logger.info("Starting evaluation script")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up environment
    torch.set_grad_enabled(False)
    
    # Parse configs
    sys.argv = ['', args.config]
    cfg = parse_configs(training=False)
    
    # Override configuration settings for validation
    cfg.dataset = dict(
        type="SunObservationDataset",
        transform_name="center",
        time_series_dir=os.path.join(args.validation_data_dir, "figure/360p/L16-S8/"),
        brightness_dir=os.path.join(args.validation_data_dir, "brightness/L16-S8/"),
    )
    cfg.batch_size = args.batch_size
    
    # Set wandb flag in config
    if args.use_wandb:
        cfg.wandb = True
    
    # Device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed_run = False
    is_main = True  # Default to True for non-distributed case
    
    logger.info(f"Environment setup: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    
    try:
        if world_size > 1:
            # Initialize PyTorch distributed directly
            torch.cuda.set_device(local_rank)
            logger.info(f"Initializing process group with {world_size} processes")
            dist.init_process_group(backend="nccl")
            is_distributed_run = dist.is_initialized()
            is_main = dist.get_rank() == 0
            logger.info(f"Process group initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
            
            # Initialize colossalai if needed
            if is_distributed_run:
                try:
                    logger.info("Initializing ColossalAI from torch")
                    colossalai.launch_from_torch({})
                    logger.info("ColossalAI initialized successfully")
                except Exception as e:
                    logger.warning(f"Could not initialize colossalai: {e}. Continuing without it.")
    except Exception as e:
        logger.error(f"Failed to initialize distributed environment: {e}")
        # Continue with non-distributed mode
        is_distributed_run = False
        is_main = True
    
    logger.info(f"Distributed setup complete: is_distributed={is_distributed_run}, is_main={is_main}")
    
    set_random_seed(seed=cfg.get("seed", 1024))
    
    # Use is_main variable instead of calling is_main_process()
    if is_main and (cfg.get("wandb", False) or args.use_wandb):
        model_name = os.path.basename(args.checkpoints_dir)
        wandb_run_name = args.wandb_run_name or f"evaluation_{model_name}"
        
        logger.info(f"Initializing wandb with project={args.wandb_project}, name={wandb_run_name}")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config={
                "model_dir": args.checkpoints_dir,
                "validation_data_dir": args.validation_data_dir,
                "batch_size": args.batch_size,
                "config_file": args.config,
                "specific_checkpoint": args.specific_checkpoint
            }
        )
        
        # Create a table for metrics summary
        metrics_table = wandb.Table(columns=["Epoch", "Step", "MAE", "MSE", "PSNR", "SSIM"])
        logger.info("Wandb initialized successfully")
    
    logger.info("Evaluation configuration loaded")
    
    try:
        # Build dataset
        logger.info("Building validation dataset...")
        dataset = build_module(cfg.dataset, DATASETS)
        logger.info(f"Validation dataset contains {len(dataset)} samples.")
        
        # Build dataloader - handle distributed vs non-distributed case differently
        logger.info("Creating validation dataloader...")
        
        # Base dataloader arguments that are common to both cases
        dataloader_args = dict(
            dataset=dataset,
            batch_size=cfg.get("batch_size", 1),
            num_workers=cfg.get("num_workers", 4),
            seed=cfg.get("seed", 1024),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=cfg.get("prefetch_factor", None),
        )
        
        # Handle bucket config args
        bucket_args = {}
        if cfg.get("bucket_config") is not None:
            bucket_args["bucket_config"] = cfg.get("bucket_config")
            bucket_args["num_bucket_build_workers"] = cfg.get("num_bucket_build_workers", 1)
        
        if is_distributed_run:
            # Get data parallel group for distributed training
            logger.info("Using distributed dataloader setup")
            process_group = get_data_parallel_group()
            dataloader_args["process_group"] = process_group
            
            validation_dataloader, _ = prepare_dataloader(
                **dataloader_args,
                **bucket_args
            )
        else:
            # Create a simple PyTorch dataloader for non-distributed case
            logger.info("Using non-distributed dataloader setup")
            from torch.utils.data import DataLoader
            
            # If OpenSora's dataloader can work without process_group, use it
            try:
                logger.info("Attempting to use OpenSora dataloader without process group")
                # Try using OpenSora's dataloader without process_group
                validation_dataloader, _ = prepare_dataloader(
                    **dataloader_args,
                    **bucket_args
                )
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Could not use OpenSora dataloader: {e}")
                logger.info("Falling back to standard PyTorch DataLoader")
                # Fall back to standard PyTorch DataLoader
                validation_dataloader = DataLoader(
                    dataset,
                    batch_size=dataloader_args["batch_size"],
                    num_workers=dataloader_args["num_workers"],
                    shuffle=dataloader_args["shuffle"],
                    drop_last=dataloader_args["drop_last"],
                    pin_memory=dataloader_args["pin_memory"],
                )
        
        logger.info("Dataloader created successfully")
        
        # Find checkpoints or use specific checkpoint if provided
        if args.specific_checkpoint:
            checkpoints = [args.specific_checkpoint]
        else:
            logger.info(f"Finding checkpoints in {args.checkpoints_dir}")
            checkpoints = find_checkpoints(args.checkpoints_dir, args.checkpoint_name)
            # Sort checkpoints by step number for proper progression in wandb
            checkpoints.sort(key=lambda x: extract_checkpoint_info(x)[1] or 0)
        
        if not checkpoints:
            logger.error(f"No checkpoints found in {args.checkpoints_dir}")
            if is_main and (cfg.get("wandb", False) or args.use_wandb):
                wandb.finish()
            return
        
        logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")
        
        # Evaluate each checkpoint
        all_metrics = []
        for checkpoint_path in checkpoints:
            metrics = evaluate_checkpoint(
                cfg, 
                checkpoint_path, 
                validation_dataloader, 
                device, 
                dtype, 
                logger,
                is_main=is_main,
                results_dir=args.results_dir  # Pass results_dir here
            )
            all_metrics.append(metrics)
            
            # Add to wandb metrics table
            if is_main and (cfg.get("wandb", False) or args.use_wandb):
                metrics_table.add_data(
                    metrics["epoch"],
                    metrics["step"],
                    round(metrics["mae"], 4),
                    round(metrics["mse"], 4),
                    round(metrics["psnr"], 2),
                    round(metrics["ssim"], 4)
                )
        
        # Save results
        if is_main:
            logger.info(f"Saving results to {args.results_dir}")
            results_path = os.path.join(args.results_dir, 'metrics.json')
            with open(results_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            # Plot metrics
            plot_path = os.path.join(args.results_dir, 'metrics_plot.png')
            plot_metrics(all_metrics, plot_path)
            
            # Log final summary to wandb
            if cfg.get("wandb", False) or args.use_wandb:
                # Log the summary metrics table
                wandb.log({"metrics_summary": metrics_table})
                
                # Log the metrics plot
                wandb.log({"metrics_plot": wandb.Image(plot_path)})
                
                # Create performance comparison charts
                epochs = [m["epoch"] for m in all_metrics]
                steps = [m["step"] for m in all_metrics]
                
                # Log line charts
                wandb.log({
                    "performance/mae_vs_step": wandb.plot.line(
                        table=wandb.Table(data=[[s, m["mae"]] for s, m in zip(steps, all_metrics)],
                                        columns=["step", "mae"]),
                        x="step",
                        y="mae",
                        title="MAE vs Training Step"
                    ),
                    "performance/mse_vs_step": wandb.plot.line(
                        table=wandb.Table(data=[[s, m["mse"]] for s, m in zip(steps, all_metrics)],
                                        columns=["step", "mse"]),
                        x="step",
                        y="mse",
                        title="MSE vs Training Step"
                    ),
                    "performance/psnr_vs_step": wandb.plot.line(
                        table=wandb.Table(data=[[s, m["psnr"]] for s, m in zip(steps, all_metrics)],
                                        columns=["step", "psnr"]),
                        x="step",
                        y="psnr",
                        title="PSNR vs Training Step"
                    ),
                    "performance/ssim_vs_step": wandb.plot.line(
                        table=wandb.Table(data=[[s, m["ssim"]] for s, m in zip(steps, all_metrics)],
                                        columns=["step", "ssim"]),
                        x="step",
                        y="ssim",
                        title="SSIM vs Training Step"
                    ),
                })
                
                # Find best checkpoints for each metric
                best_mae_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i]["mae"])
                best_mse_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i]["mse"])
                best_psnr_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]["psnr"])
                best_ssim_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]["ssim"])
                
                # Log best metrics summary
                wandb.run.summary.update({
                    "best_mae": all_metrics[best_mae_idx]["mae"],
                    "best_mae_epoch": all_metrics[best_mae_idx]["epoch"],
                    "best_mae_step": all_metrics[best_mae_idx]["step"],
                    "best_mse": all_metrics[best_mse_idx]["mse"],
                    "best_mse_epoch": all_metrics[best_mse_idx]["epoch"],
                    "best_mse_step": all_metrics[best_mse_idx]["step"],
                    "best_psnr": all_metrics[best_psnr_idx]["psnr"],
                    "best_psnr_epoch": all_metrics[best_psnr_idx]["epoch"],
                    "best_psnr_step": all_metrics[best_psnr_idx]["step"],
                    "best_ssim": all_metrics[best_ssim_idx]["ssim"],
                    "best_ssim_epoch": all_metrics[best_ssim_idx]["epoch"],
                    "best_ssim_step": all_metrics[best_ssim_idx]["step"],
                })
                
            
            logger.info(f"Evaluation complete. Results saved to {args.results_dir}")
            
            # Print summary
            logger.info("Summary of results:")
            for metrics in all_metrics:
                logger.info(f"Epoch {metrics['epoch']}, Step {metrics['step']}: "
                            f"MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}, "
                            f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
    finally:
        # Cleanup
        if is_distributed_run:
            logger.info("Cleaning up distributed environment")
            dist.destroy_process_group()
        
        # Finish wandb if it was initialized
        if is_main and (cfg.get("wandb", False) or args.use_wandb):
            logger.info("Finishing wandb session")
            wandb.finish()
        
        logger.info("Script completed")

if __name__ == "__main__":
    main()
